//! Explicit SIMD building blocks shared by the CPU kernels: an 8-lane f32 vector, broadcasts,
//! sine/cosine, and joint transforms composed per lane.
//!
//! Explicit because LLVM did not auto-vectorize the `[f32; 8]` lane loops these kernels
//! started with (see SPIKE.md, "What the checker went through").

use crate::layout::GpuLink;
use crate::model::JointType;

/// Lanes per vector: 8 (one AVX register on x86, two NEON registers on ARM, where 8 lanes
/// measured faster end to end than 4).
pub const LANES: usize = 8;
pub type V = wide::f32x8;

/// Broadcast `x` to every lane, through the architecture intrinsic: `wide`'s `splat` builds
/// an array in memory, which LLVM lowers to a `memset_pattern16` call on macOS.
#[inline(always)]
pub fn v(x: f32) -> V {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always present on aarch64; f32x8 is laid out as two float32x4_t.
        unsafe {
            let h = std::arch::aarch64::vdupq_n_f32(x);
            std::mem::transmute::<[std::arch::aarch64::float32x4_t; 2], V>([h, h])
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        // SAFETY: AVX is enabled at compile time; f32x8 has the layout of __m256.
        unsafe {
            std::mem::transmute::<std::arch::x86_64::__m256, V>(std::arch::x86_64::_mm256_set1_ps(
                x,
            ))
        }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    {
        // SAFETY: SSE2 is part of x86_64; f32x8 without AVX has the layout of two __m128.
        unsafe {
            let h = std::arch::x86_64::_mm_set1_ps(x);
            std::mem::transmute::<[std::arch::x86_64::__m128; 2], V>([h, h])
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        V::splat(x)
    }
}

/// Sine and cosine per lane: reduction by pi/2 in two parts, then the Cephes
/// single-precision polynomials on [-pi/4, pi/4], quadrant chosen with masks. (wide's own
/// `sin_cos` rebuilds its constants on the stack on every call on NEON.) Within 2e-7 of std
/// for joint angles.
#[inline(always)]
pub fn sin_cos(x: V) -> (V, V) {
    let k = (x * v(std::f32::consts::FRAC_2_PI)).round();
    let r = (-k).mul_add(v(-4.371_139e-8), (-k).mul_add(v(1.570_796_4), x));
    let z = r * r;
    let sin_r = (z * r).mul_add(
        z.mul_add(
            z.mul_add(v(-1.951_529_6e-4), v(8.332_161e-3)),
            v(-1.666_665_5e-1),
        ),
        r,
    );
    let cos_r = (z * z).mul_add(
        z.mul_add(
            z.mul_add(v(2.443_315_7e-5), v(-1.388_731_6e-3)),
            v(4.166_664_6e-2),
        ),
        z.mul_add(v(-0.5), v(1.0)),
    );
    // Quadrant q = k mod 4: (sin, cos) = (s, c), (c, -s), (-s, -c), (-c, s).
    let q = k - (k * v(0.25)).floor() * v(4.0);
    let odd = (q - (q * v(0.5)).floor() * v(2.0)).simd_eq(v(1.0));
    let s = odd.select(cos_r, sin_r);
    let c = odd.select(sin_r, cos_r);
    let neg_s = q.simd_ge(v(2.0));
    let neg_c = q.simd_eq(v(1.0)) | q.simd_eq(v(2.0));
    (neg_s.select(-s, s), neg_c.select(-c, c))
}

/// 3x4 row-major transform per lane.
pub type M = [V; 12];

#[inline(always)]
pub fn mul34(a: &M, b: &M) -> M {
    let mut m = [v(0.0); 12];
    for r in 0..3 {
        for c in 0..4 {
            let mut x = a[r * 4].mul_add(
                b[c],
                a[r * 4 + 1].mul_add(b[4 + c], a[r * 4 + 2] * b[8 + c]),
            );
            if c == 3 {
                x += a[r * 4 + 3];
            }
            m[r * 4 + c] = x;
        }
    }
    m
}

/// Joint transform per lane, with a short path for rotations about the local z axis.
#[inline(always)]
pub fn local_transform(link: &GpuLink, q: V) -> M {
    let o = &link.origin;
    match JointType::from_u32(link.joint_type) {
        JointType::Fixed => o.map(v),
        JointType::Prismatic => {
            let [x, y, z] = link.axis;
            let mut m = o.map(v);
            for r in 0..3 {
                let d = o[r * 4] * x + o[r * 4 + 1] * y + o[r * 4 + 2] * z;
                m[r * 4 + 3] = q.mul_add(v(d), m[r * 4 + 3]);
            }
            m
        }
        JointType::Revolute => {
            let (s, c) = sin_cos(q);
            if link.axis == [0.0, 0.0, 1.0] {
                // origin * Rz(q): columns 0 and 1 rotate, column 2 and the translation stay.
                let mut m = [v(0.0); 12];
                for r in 0..3 {
                    let (o0, o1) = (v(o[r * 4]), v(o[r * 4 + 1]));
                    m[r * 4] = o0.mul_add(c, o1 * s);
                    m[r * 4 + 1] = o1.mul_add(c, -(o0 * s));
                    m[r * 4 + 2] = v(o[r * 4 + 2]);
                    m[r * 4 + 3] = v(o[r * 4 + 3]);
                }
                return m;
            }
            let [x, y, z] = link.axis.map(v);
            let t = v(1.0) - c;
            let zero = v(0.0);
            let rot = [
                t * x * x + c,
                t * x * y - s * z,
                t * x * z + s * y,
                zero,
                t * x * y + s * z,
                t * y * y + c,
                t * y * z - s * x,
                zero,
                t * x * z - s * y,
                t * y * z + s * x,
                t * z * z + c,
                zero,
            ];
            mul34(&o.map(v), &rot)
        }
    }
}

/// Parent frame (entries from `pe`) times the joint transform of `link` at `q`, built
/// without storing constant matrices: on NEON a stored broadcast `f32x8` (two identical
/// halves) becomes a `memset_pattern16` call.
#[inline(always)]
pub fn compose(pe: impl Fn(usize) -> V, link: &GpuLink, q: V) -> M {
    let o = &link.origin;
    // Row `r` of the parent rotation times column `c` of the joint origin (+ translation).
    let dot = |r: usize, c: usize| -> V {
        let x = pe(r * 4).mul_add(
            v(o[c]),
            pe(r * 4 + 1).mul_add(v(o[4 + c]), pe(r * 4 + 2) * v(o[8 + c])),
        );
        if c == 3 {
            x + pe(r * 4 + 3)
        } else {
            x
        }
    };
    match JointType::from_u32(link.joint_type) {
        JointType::Fixed => std::array::from_fn(|k| dot(k / 4, k % 4)),
        JointType::Revolute if link.axis == [0.0, 0.0, 1.0] => {
            // P * O * Rz(q): columns 0 and 1 of P * O rotate, columns 2 and 3 stay.
            let (s, c) = sin_cos(q);
            let a: [V; 3] = std::array::from_fn(|r| dot(r, 0));
            let b: [V; 3] = std::array::from_fn(|r| dot(r, 1));
            std::array::from_fn(|k| {
                let r = k / 4;
                match k % 4 {
                    0 => a[r].mul_add(c, b[r] * s),
                    1 => b[r].mul_add(c, -(a[r] * s)),
                    col => dot(r, col),
                }
            })
        }
        _ => {
            let l = local_transform(link, q);
            std::array::from_fn(|k| {
                let (r, c) = (k / 4, k % 4);
                let x = pe(r * 4).mul_add(
                    l[c],
                    pe(r * 4 + 1).mul_add(l[4 + c], pe(r * 4 + 2) * l[8 + c]),
                );
                if c == 3 {
                    x + pe(r * 4 + 3)
                } else {
                    x
                }
            })
        }
    }
}

/// World position of a frame-local point, per lane.
#[inline(always)]
pub fn to_world(m: &M, c: [f32; 3]) -> [V; 3] {
    let (x, y, z) = (v(c[0]), v(c[1]), v(c[2]));
    [0, 1, 2].map(|r| {
        m[r * 4].mul_add(
            x,
            m[r * 4 + 1].mul_add(y, m[r * 4 + 2].mul_add(z, m[r * 4 + 3])),
        )
    })
}

#[inline(always)]
pub fn dist_sq(a: &[V; 3], b: [V; 3]) -> V {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx.mul_add(dx, dy.mul_add(dy, dz * dz))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_sin_cos_matches_std() {
        let mut worst = 0.0f32;
        for i in -20_000..=20_000 {
            let x = i as f32 * 4e-4; // [-8, 8] rad, beyond any joint range
            let (s, c) = sin_cos(v(x));
            let (s, c) = (s.to_array()[0], c.to_array()[0]);
            worst = worst.max((s - x.sin()).abs()).max((c - x.cos()).abs());
        }
        assert!(worst < 3e-7, "{worst:e}");
    }
}

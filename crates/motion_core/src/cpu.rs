//! CPU f32 kernels over the shared layout: the "Rust CPU" compute backend.
//!
//! Two variants of the same fused operation (batched FK -> link spheres ->
//! sphere/cuboid distance + gradient):
//!
//! * [`query_scalar`]: one configuration at a time, line-for-line the same
//!   arithmetic as the WGSL kernel. This is the oracle for the GPU path.
//! * [`query_lanes`]: `LANES` configurations at a time, every value held as
//!   `[f32; LANES]`, so the inner loops are straight-line lane loops that LLVM
//!   turns into SIMD (NEON on Apple Silicon, AVX on x86) without intrinsics.
//!
//! Both parallelize over configurations with rayon.

#![allow(clippy::needless_range_loop)] // lane loops index several arrays in lock step on purpose

use rayon::prelude::*;

use crate::layout::{GpuCuboid, GpuLink, GpuSphere, ModelBuffers, MAX_LINKS, MAX_SPHERES};
use crate::model::JointType;

pub const LANES: usize = 8;

/// Outputs of a distance query for a batch, in the shared batch layout.
pub struct BatchOutput {
    pub batch: usize,
    pub sphere_dist: Vec<f32>,
    /// xyz + unused w, `[s * B + b]`
    pub sphere_grad: Vec<[f32; 4]>,
    pub config_min: Vec<f32>,
}

impl BatchOutput {
    pub fn new(batch: usize, n_spheres: usize) -> Self {
        Self {
            batch,
            sphere_dist: vec![0.0; batch * n_spheres],
            sphere_grad: vec![[0.0; 4]; batch * n_spheres],
            config_min: vec![0.0; batch],
        }
    }
}

/// Raw output pointers shared across rayon tasks; every task writes disjoint batch indices.
#[derive(Clone, Copy)]
struct OutPtr {
    dist: *mut f32,
    grad: *mut [f32; 4],
    min: *mut f32,
}
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    fn new(out: &mut BatchOutput) -> Self {
        Self {
            dist: out.sphere_dist.as_mut_ptr(),
            grad: out.sphere_grad.as_mut_ptr(),
            min: out.config_min.as_mut_ptr(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar

#[inline(always)]
fn mul34(a: &[f32; 12], b: &[f32; 12]) -> [f32; 12] {
    let mut m = [0.0f32; 12];
    for r in 0..3 {
        for c in 0..4 {
            let mut v = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
            if c == 3 {
                v += a[r * 4 + 3];
            }
            m[r * 4 + c] = v;
        }
    }
    m
}

#[inline(always)]
fn local_transform(link: &GpuLink, q: f32) -> [f32; 12] {
    let o = &link.origin;
    match JointType::from_u32(link.joint_type) {
        JointType::Fixed => *o,
        JointType::Prismatic => {
            let [x, y, z] = link.axis;
            let mut m = *o;
            m[3] += o[0] * x * q + o[1] * y * q + o[2] * z * q;
            m[7] += o[4] * x * q + o[5] * y * q + o[6] * z * q;
            m[11] += o[8] * x * q + o[9] * y * q + o[10] * z * q;
            m
        }
        JointType::Revolute => {
            let (s, c) = q.sin_cos();
            let t = 1.0 - c;
            let [x, y, z] = link.axis;
            let r = [
                t * x * x + c,
                t * x * y - s * z,
                t * x * z + s * y,
                0.0,
                t * x * y + s * z,
                t * y * y + c,
                t * y * z - s * x,
                0.0,
                t * x * z - s * y,
                t * y * z + s * x,
                t * z * z + c,
                0.0,
            ];
            mul34(o, &r)
        }
    }
}

#[inline(always)]
fn sphere_box(p: [f32; 3], b: &GpuCuboid) -> (f32, [f32; 3]) {
    let m = &b.inv;
    let l = [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    ];
    let h = b.half_extents;
    let q = [l[0].abs() - h[0], l[1].abs() - h[1], l[2].abs() - h[2]];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let outside = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    let sx = if l[0] < 0.0 { -1.0 } else { 1.0 };
    let sy = if l[1] < 0.0 { -1.0 } else { 1.0 };
    let sz = if l[2] < 0.0 { -1.0 } else { 1.0 };
    let g = if outside > 0.0 {
        let inv_len = 1.0 / outside;
        [
            sx * o[0] * inv_len,
            sy * o[1] * inv_len,
            sz * o[2] * inv_len,
        ]
    } else if q[0] >= q[1] && q[0] >= q[2] {
        [sx, 0.0, 0.0]
    } else if q[1] >= q[2] {
        [0.0, sy, 0.0]
    } else {
        [0.0, 0.0, sz]
    };
    // World gradient = R_box * g = inv_rot^T * g.
    let gw = [
        m[0] * g[0] + m[4] * g[1] + m[8] * g[2],
        m[1] * g[0] + m[5] * g[1] + m[9] * g[2],
        m[2] * g[0] + m[6] * g[1] + m[10] * g[2],
    ];
    (outside + inside, gw)
}

/// Fused query for one configuration. `q(j)` returns joint `j`.
#[inline(always)]
fn query_one(
    model: &ModelBuffers,
    q: impl Fn(usize) -> f32,
    mut write: impl FnMut(usize, f32, [f32; 3]),
) -> f32 {
    let mut frames = [[0.0f32; 12]; MAX_LINKS];
    for (i, link) in model.links.iter().enumerate() {
        let value = if link.q_index >= 0 {
            q(link.q_index as usize)
        } else {
            0.0
        };
        let local = local_transform(link, value);
        frames[i] = if link.parent < 0 {
            local
        } else {
            mul34(&frames[link.parent as usize], &local)
        };
    }
    let mut min = f32::INFINITY;
    for (s, sphere) in model.spheres.iter().enumerate() {
        let m = &frames[sphere.link as usize];
        let c = sphere.center;
        let p = [
            m[0] * c[0] + m[1] * c[1] + m[2] * c[2] + m[3],
            m[4] * c[0] + m[5] * c[1] + m[6] * c[2] + m[7],
            m[8] * c[0] + m[9] * c[1] + m[10] * c[2] + m[11],
        ];
        let mut best = f32::INFINITY;
        let mut best_g = [0.0f32; 3];
        for (k, b) in model.cuboids.iter().enumerate() {
            if b.touch_mask & (1 << sphere.link) != 0
                || (k < 32 && sphere.touch_boxes & (1 << k) != 0)
            {
                continue;
            }
            let (d, g) = sphere_box(p, b);
            if d < best {
                best = d;
                best_g = g;
            }
        }
        let d = best - sphere.radius;
        min = min.min(d);
        write(s, d, best_g);
    }
    min
}

/// Fused batched query, one configuration per iteration. `q` is SoA: `q[j * B + b]`.
pub fn query_scalar(model: &ModelBuffers, q: &[f32], out: &mut BatchOutput) {
    let batch = out.batch;
    assert_eq!(q.len(), model.dof * batch);
    let ptr = OutPtr::new(out);
    (0..batch).into_par_iter().with_min_len(256).for_each(|b| {
        // Rebind: 2021 closures capture fields, which would bypass OutPtr's Sync impl.
        #[allow(clippy::redundant_locals)]
        let ptr = ptr;
        let min = query_one(
            model,
            |j| q[j * batch + b],
            |s, d, g| unsafe {
                *ptr.dist.add(s * batch + b) = d;
                *ptr.grad.add(s * batch + b) = [g[0], g[1], g[2], 0.0];
            },
        );
        unsafe { *ptr.min.add(b) = min };
    });
}

// ---------------------------------------------------------------------------
// Lanes: LANES configurations at once

type L = [f32; LANES];

#[inline(always)]
fn splat(v: f32) -> L {
    [v; LANES]
}

#[inline(always)]
fn mul34_lanes(a: &[L; 12], b: &[L; 12]) -> [L; 12] {
    let mut m = [splat(0.0); 12];
    for r in 0..3 {
        for c in 0..4 {
            for l in 0..LANES {
                let mut v = a[r * 4][l] * b[c][l]
                    + a[r * 4 + 1][l] * b[4 + c][l]
                    + a[r * 4 + 2][l] * b[8 + c][l];
                if c == 3 {
                    v += a[r * 4 + 3][l];
                }
                m[r * 4 + c][l] = v;
            }
        }
    }
    m
}

#[inline(always)]
fn local_transform_lanes(link: &GpuLink, q: &L) -> [L; 12] {
    let o = link.origin.map(splat);
    match JointType::from_u32(link.joint_type) {
        JointType::Fixed => o,
        JointType::Prismatic => {
            let [x, y, z] = link.axis;
            let mut m = o;
            for l in 0..LANES {
                let v = q[l];
                m[3][l] += (o[0][l] * x + o[1][l] * y + o[2][l] * z) * v;
                m[7][l] += (o[4][l] * x + o[5][l] * y + o[6][l] * z) * v;
                m[11][l] += (o[8][l] * x + o[9][l] * y + o[10][l] * z) * v;
            }
            m
        }
        JointType::Revolute => {
            let [x, y, z] = link.axis;
            let mut r = [splat(0.0); 12];
            for l in 0..LANES {
                let (s, c) = q[l].sin_cos();
                let t = 1.0 - c;
                r[0][l] = t * x * x + c;
                r[1][l] = t * x * y - s * z;
                r[2][l] = t * x * z + s * y;
                r[4][l] = t * x * y + s * z;
                r[5][l] = t * y * y + c;
                r[6][l] = t * y * z - s * x;
                r[8][l] = t * x * z - s * y;
                r[9][l] = t * y * z + s * x;
                r[10][l] = t * z * z + c;
            }
            mul34_lanes(&o, &r)
        }
    }
}

#[inline(always)]
fn spheres_lanes(
    model: &ModelBuffers,
    frames: &[[L; 12]],
    sphere: &GpuSphere,
) -> ([L; 3], L, [L; 3]) {
    let m = &frames[sphere.link as usize];
    let c = sphere.center;
    let mut p = [splat(0.0); 3];
    for l in 0..LANES {
        p[0][l] = m[0][l] * c[0] + m[1][l] * c[1] + m[2][l] * c[2] + m[3][l];
        p[1][l] = m[4][l] * c[0] + m[5][l] * c[1] + m[6][l] * c[2] + m[7][l];
        p[2][l] = m[8][l] * c[0] + m[9][l] * c[1] + m[10][l] * c[2] + m[11][l];
    }
    let mut best = splat(f32::INFINITY);
    let mut best_g = [splat(0.0); 3];
    for (k, b) in model.cuboids.iter().enumerate() {
        if b.touch_mask & (1 << sphere.link) != 0 || (k < 32 && sphere.touch_boxes & (1 << k) != 0)
        {
            continue;
        }
        let m = &b.inv;
        let h = b.half_extents;
        for l in 0..LANES {
            let lx = m[0] * p[0][l] + m[1] * p[1][l] + m[2] * p[2][l] + m[3];
            let ly = m[4] * p[0][l] + m[5] * p[1][l] + m[6] * p[2][l] + m[7];
            let lz = m[8] * p[0][l] + m[9] * p[1][l] + m[10] * p[2][l] + m[11];
            let qx = lx.abs() - h[0];
            let qy = ly.abs() - h[1];
            let qz = lz.abs() - h[2];
            let ox = qx.max(0.0);
            let oy = qy.max(0.0);
            let oz = qz.max(0.0);
            let outside = (ox * ox + oy * oy + oz * oz).sqrt();
            let inside = qx.max(qy).max(qz).min(0.0);
            let d = outside + inside;
            let sx = if lx < 0.0 { -1.0 } else { 1.0 };
            let sy = if ly < 0.0 { -1.0 } else { 1.0 };
            let sz = if lz < 0.0 { -1.0 } else { 1.0 };
            // Branch-free gradient selection so the lane loop stays vectorizable.
            let is_out = outside > 0.0;
            let inv_len = if is_out { 1.0 / outside } else { 0.0 };
            let ax = qx >= qy && qx >= qz;
            let ay = !ax && qy >= qz;
            let az = !ax && !ay;
            let gx = if is_out {
                sx * ox * inv_len
            } else if ax {
                sx
            } else {
                0.0
            };
            let gy = if is_out {
                sy * oy * inv_len
            } else if ay {
                sy
            } else {
                0.0
            };
            let gz = if is_out {
                sz * oz * inv_len
            } else if az {
                sz
            } else {
                0.0
            };
            let wx = m[0] * gx + m[4] * gy + m[8] * gz;
            let wy = m[1] * gx + m[5] * gy + m[9] * gz;
            let wz = m[2] * gx + m[6] * gy + m[10] * gz;
            let closer = d < best[l];
            best[l] = if closer { d } else { best[l] };
            best_g[0][l] = if closer { wx } else { best_g[0][l] };
            best_g[1][l] = if closer { wy } else { best_g[1][l] };
            best_g[2][l] = if closer { wz } else { best_g[2][l] };
        }
    }
    (p, best, best_g)
}

/// Fused batched query, `LANES` configurations per iteration. Batch must be a multiple of `LANES`.
pub fn query_lanes(model: &ModelBuffers, q: &[f32], out: &mut BatchOutput) {
    let ptr = OutPtr::new(out);
    lanes_impl::<true, false>(model, q, out.batch, ptr);
}

/// Collision check against the scene and the robot itself: per-configuration minimum of
/// sphere-obstacle and sphere-sphere (self) clearance. Negative = in collision.
pub fn collision_min_lanes(model: &ModelBuffers, q: &[f32], config_min: &mut [f32]) {
    let ptr = OutPtr {
        dist: std::ptr::null_mut(),
        grad: std::ptr::null_mut(),
        min: config_min.as_mut_ptr(),
    };
    lanes_impl::<false, true>(model, q, config_min.len(), ptr);
}

/// Collision check only: per-configuration minimum, no per-sphere outputs (`config_min.len()` = batch).
pub fn query_lanes_min(model: &ModelBuffers, q: &[f32], config_min: &mut [f32]) {
    let ptr = OutPtr {
        dist: std::ptr::null_mut(),
        grad: std::ptr::null_mut(),
        min: config_min.as_mut_ptr(),
    };
    lanes_impl::<false, false>(model, q, config_min.len(), ptr);
}

fn lanes_impl<const PER_SPHERE: bool, const SELF: bool>(
    model: &ModelBuffers,
    q: &[f32],
    batch: usize,
    ptr: OutPtr,
) {
    assert_eq!(q.len(), model.dof * batch);
    assert_eq!(batch % LANES, 0, "batch must be a multiple of {LANES}");
    (0..batch / LANES)
        .into_par_iter()
        .with_min_len(32)
        .for_each(|chunk| {
            // Rebind: 2021 closures capture fields, which would bypass OutPtr's Sync impl.
            #[allow(clippy::redundant_locals)]
            let ptr = ptr;
            let b0 = chunk * LANES;
            let mut frames = [[splat(0.0); 12]; MAX_LINKS];
            for (i, link) in model.links.iter().enumerate() {
                let mut value = splat(0.0);
                if link.q_index >= 0 {
                    let base = link.q_index as usize * batch + b0;
                    value.copy_from_slice(&q[base..base + LANES]);
                }
                let local = local_transform_lanes(link, &value);
                frames[i] = if link.parent < 0 {
                    local
                } else {
                    mul34_lanes(&frames[link.parent as usize], &local)
                };
            }
            let mut min = splat(f32::INFINITY);
            for (s, sphere) in model.spheres.iter().enumerate() {
                let (_, best, g) = spheres_lanes(model, &frames[..model.links.len()], sphere);
                for l in 0..LANES {
                    let d = best[l] - sphere.radius;
                    min[l] = min[l].min(d);
                    if PER_SPHERE {
                        unsafe {
                            *ptr.dist.add(s * batch + b0 + l) = d;
                            *ptr.grad.add(s * batch + b0 + l) = [g[0][l], g[1][l], g[2][l], 0.0];
                        }
                    }
                }
            }
            if SELF {
                let mut world = [[splat(0.0); 3]; MAX_SPHERES];
                for (s, sphere) in model.self_spheres.iter().enumerate() {
                    let m = &frames[sphere.link as usize];
                    let c = sphere.center;
                    for l in 0..LANES {
                        world[s][0][l] = m[0][l] * c[0] + m[1][l] * c[1] + m[2][l] * c[2] + m[3][l];
                        world[s][1][l] = m[4][l] * c[0] + m[5][l] * c[1] + m[6][l] * c[2] + m[7][l];
                        world[s][2][l] =
                            m[8][l] * c[0] + m[9][l] * c[1] + m[10][l] * c[2] + m[11][l];
                    }
                }
                for &[i, j] in &model.self_pairs {
                    let (a, b) = (&world[i as usize], &world[j as usize]);
                    let r = model.self_spheres[i as usize].radius
                        + model.self_spheres[j as usize].radius;
                    for l in 0..LANES {
                        let dx = a[0][l] - b[0][l];
                        let dy = a[1][l] - b[1][l];
                        let dz = a[2][l] - b[2][l];
                        min[l] = min[l].min((dx * dx + dy * dy + dz * dz).sqrt() - r);
                    }
                }
            }
            for l in 0..LANES {
                unsafe { *ptr.min.add(b0 + l) = min[l] };
            }
        });
}

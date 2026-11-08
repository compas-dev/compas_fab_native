//! CPU f32 kernels over the shared layout: the "Rust CPU" compute backend.
//!
//! Two variants of the same fused operation (batched FK -> link spheres ->
//! sphere/cuboid distance + gradient):
//!
//! * [`query_scalar`]: one configuration at a time, line-for-line the same
//!   arithmetic as the WGSL kernel. This is the oracle for the GPU path.
//! * [`query_lanes`]: `LANES` configurations at a time in explicit SIMD (`crate::simd`;
//!   an earlier `[f32; LANES]` version relied on auto-vectorization, which LLVM did not do).
//!
//! Both parallelize over configurations with rayon.

#![allow(clippy::needless_range_loop)] // lane loops index several arrays in lock step on purpose

use rayon::prelude::*;

use std::mem::MaybeUninit;

use crate::layout::{GpuCuboid, GpuLink, GpuSphere, ModelBuffers, MAX_LINKS, MAX_SPHERES};
use crate::model::JointType;
pub use crate::simd::LANES;
use crate::simd::{compose, dist_sq, to_world, v, M, V};

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
        if let Some(esdf) = &model.esdf {
            let (d, g) = esdf.sample(p);
            if d - model.esdf_margin < best {
                best = d - model.esdf_margin;
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
// Lanes: LANES configurations at once, in explicit SIMD (`crate::simd`)

/// Distance from a sphere center (world, per lane) to the nearest box it may not touch, and
/// the world gradient of that distance; then the ESDF, if any.
#[inline(always)]
fn env_distance(model: &ModelBuffers, sphere: &GpuSphere, p: &[V; 3]) -> (V, [V; 3]) {
    let mut best = v(f32::INFINITY);
    let mut best_g = [v(0.0); 3];
    let (zero, one) = (v(0.0), v(1.0));
    for (k, b) in model.cuboids.iter().enumerate() {
        if b.touch_mask & (1 << sphere.link) != 0 || (k < 32 && sphere.touch_boxes & (1 << k) != 0)
        {
            continue;
        }
        let m = &b.inv;
        let h = b.half_extents;
        let l: [V; 3] = std::array::from_fn(|r| {
            v(m[r * 4]).mul_add(
                p[0],
                v(m[r * 4 + 1]).mul_add(p[1], v(m[r * 4 + 2]).mul_add(p[2], v(m[r * 4 + 3]))),
            )
        });
        let q: [V; 3] = std::array::from_fn(|k| l[k].abs() - v(h[k]));
        let o: [V; 3] = q.map(|x| x.max(zero));
        let outside = o[0].mul_add(o[0], o[1].mul_add(o[1], o[2] * o[2])).sqrt();
        let inside = q[0].max(q[1]).max(q[2]).min(zero);
        let d = outside + inside;
        let sign: [V; 3] = l.map(|x| x.simd_lt(zero).select(-one, one));
        // Outside: along the offset to the box; inside: along the axis of the nearest face.
        let is_out = outside.simd_gt(zero);
        let inv_len = is_out.select(one / outside, zero);
        let ax = q[0].simd_ge(q[1]) & q[0].simd_ge(q[2]);
        let ay = q[1].simd_ge(q[2]);
        let face = [
            ax.select(sign[0], zero),
            ax.select(zero, ay.select(sign[1], zero)),
            ax.select(zero, ay.select(zero, sign[2])),
        ];
        let g: [V; 3] = std::array::from_fn(|k| is_out.select(sign[k] * o[k] * inv_len, face[k]));
        // World gradient = R_box * g = inv_rot^T * g.
        let gw: [V; 3] = std::array::from_fn(|c| {
            v(m[c]).mul_add(g[0], v(m[4 + c]).mul_add(g[1], v(m[8 + c]) * g[2]))
        });
        let closer = d.simd_lt(best);
        best = closer.select(d, best);
        for c in 0..3 {
            best_g[c] = closer.select(gw[c], best_g[c]);
        }
    }
    if let Some(esdf) = &model.esdf {
        let (x, y, z) = (p[0].to_array(), p[1].to_array(), p[2].to_array());
        let (mut b, mut g) = (best.to_array(), best_g.map(|c| c.to_array()));
        for l in 0..LANES {
            let (d, grad) = esdf.sample([x[l], y[l], z[l]]);
            let d = d - model.esdf_margin;
            if d < b[l] {
                b[l] = d;
                for c in 0..3 {
                    g[c][l] = grad[c];
                }
            }
        }
        best = V::new(b);
        best_g = g.map(V::new);
    }
    (best, best_g)
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
            let mut frames = [const { MaybeUninit::<M>::uninit() }; MAX_LINKS];
            for (i, link) in model.links.iter().enumerate() {
                let value = if link.q_index >= 0 {
                    let base = link.q_index as usize * batch + b0;
                    V::new(q[base..base + LANES].try_into().unwrap())
                } else {
                    v(0.0)
                };
                let frame = if link.parent < 0 {
                    crate::simd::local_transform(link, value)
                } else {
                    // Parents come before children, so the parent frame is initialized.
                    let p = unsafe { frames[link.parent as usize].assume_init_ref() };
                    compose(|k| p[k], link, value)
                };
                frames[i].write(frame);
            }
            // Every link frame was written above.
            let frame = |l: u32| unsafe { frames[l as usize].assume_init_ref() };
            let mut min = v(f32::INFINITY);
            for (s, sphere) in model.spheres.iter().enumerate() {
                let p = to_world(frame(sphere.link), sphere.center);
                let (best, g) = env_distance(model, sphere, &p);
                let d = best - v(sphere.radius);
                min = min.min(d);
                if PER_SPHERE {
                    let (d, g) = (d.to_array(), g.map(|c| c.to_array()));
                    for l in 0..LANES {
                        unsafe {
                            *ptr.dist.add(s * batch + b0 + l) = d[l];
                            *ptr.grad.add(s * batch + b0 + l) = [g[0][l], g[1][l], g[2][l], 0.0];
                        }
                    }
                }
            }
            if SELF {
                let mut world = [const { MaybeUninit::<[V; 3]>::uninit() }; MAX_SPHERES];
                for (w, sphere) in world.iter_mut().zip(&model.self_spheres) {
                    w.write(to_world(frame(sphere.link), sphere.center));
                }
                for &[i, j] in &model.self_pairs {
                    // Every self sphere's position was written above.
                    let (a, b) = unsafe {
                        (
                            world[i as usize].assume_init_ref(),
                            world[j as usize].assume_init_ref(),
                        )
                    };
                    let r = model.self_spheres[i as usize].radius
                        + model.self_spheres[j as usize].radius;
                    min = min.min(dist_sq(a, *b).sqrt() - v(r));
                }
            }
            let min = min.to_array();
            for l in 0..LANES {
                unsafe { *ptr.min.add(b0 + l) = min[l] };
            }
        });
}

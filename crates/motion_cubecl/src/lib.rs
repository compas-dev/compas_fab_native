//! CubeCL implementation of the batched sphere distance query, for the Phase 0
//! "CubeCL vs wgpu" comparison. Same operation, same batch layout and same
//! arithmetic as `motion_core::cpu::query_scalar` and `motion_gpu`'s WGSL.
//!
//! Model records are passed as flat arrays (CubeCL kernels take slices of
//! scalars), built from the shared `ModelBuffers`:
//!
//! * `link_f`: 16 f32 per link: origin (12), axis (3), pad
//! * `link_i`: 4 i32 per link: parent, joint type, q index, pad
//! * `sphere_f`: 4 f32 per sphere: centre (3), radius; `sphere_link`: u32 per sphere
//! * `cuboid_f`: 16 f32 per cuboid: inverse pose (12), half extents (3), pad
//!
//! Two kernel flavours:
//! * runtime loops over links / spheres / cuboids, like the WGSL kernel;
//! * `specialized`: counts are compile-time constants, so CubeCL JIT-compiles one
//!   kernel per robot + scene shape with every loop unrolled.

use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::wgpu::{AutoCompiler, WgpuDevice, WgpuRuntime};
use cubecl_runtime::runtime::Runtime as _;
use motion_core::cpu::BatchOutput;
use motion_core::layout::{ModelBuffers, MAX_LINKS};

const FRAME: usize = 12;
const WG: u32 = 64;
const MAX_WG_X: u32 = 32768;

// ---------------------------------------------------------------------------
// Device functions

#[cube]
fn local_transform(
    link_f: &[f32],
    link_i: &[i32],
    i: usize,
    value: f32,
    out: &mut Array<f32>,
    base: usize,
) {
    let lf = i * 16;
    let kind = link_i[i * 4 + 1];
    let ax = link_f[lf + 12];
    let ay = link_f[lf + 13];
    let az = link_f[lf + 14];
    if kind == 1 {
        let s = value.sin();
        let c = value.cos();
        let t = 1.0 - c;
        let r00 = t * ax * ax + c;
        let r01 = t * ax * ay - s * az;
        let r02 = t * ax * az + s * ay;
        let r10 = t * ax * ay + s * az;
        let r11 = t * ay * ay + c;
        let r12 = t * ay * az - s * ax;
        let r20 = t * ax * az - s * ay;
        let r21 = t * ay * az + s * ax;
        let r22 = t * az * az + c;
        // out = origin * [R | 0]
        #[unroll]
        for r in 0..3usize {
            let o0 = link_f[lf + r * 4];
            let o1 = link_f[lf + r * 4 + 1];
            let o2 = link_f[lf + r * 4 + 2];
            out[base + r * 4] = o0 * r00 + o1 * r10 + o2 * r20;
            out[base + r * 4 + 1] = o0 * r01 + o1 * r11 + o2 * r21;
            out[base + r * 4 + 2] = o0 * r02 + o1 * r12 + o2 * r22;
            out[base + r * 4 + 3] = link_f[lf + r * 4 + 3];
        }
    } else {
        #[unroll]
        for k in 0..12usize {
            out[base + k] = link_f[lf + k];
        }
        if kind == 2 {
            #[unroll]
            for r in 0..3usize {
                let d = link_f[lf + r * 4] * ax
                    + link_f[lf + r * 4 + 1] * ay
                    + link_f[lf + r * 4 + 2] * az;
                out[base + r * 4 + 3] += d * value;
            }
        }
    }
}

/// frames[dst] = frames[a] * local (local stored at `tmp`).
#[cube]
fn compose(frames: &mut Array<f32>, a: usize, tmp: usize, dst: usize) {
    #[unroll]
    for r in 0..3usize {
        let a0 = frames[a + r * 4];
        let a1 = frames[a + r * 4 + 1];
        let a2 = frames[a + r * 4 + 2];
        let a3 = frames[a + r * 4 + 3];
        let c0 = a0 * frames[tmp] + a1 * frames[tmp + 4] + a2 * frames[tmp + 8];
        let c1 = a0 * frames[tmp + 1] + a1 * frames[tmp + 5] + a2 * frames[tmp + 9];
        let c2 = a0 * frames[tmp + 2] + a1 * frames[tmp + 6] + a2 * frames[tmp + 10];
        let c3 = a0 * frames[tmp + 3] + a1 * frames[tmp + 7] + a2 * frames[tmp + 11] + a3;
        frames[dst + r * 4] = c0;
        frames[dst + r * 4 + 1] = c1;
        frames[dst + r * 4 + 2] = c2;
        frames[dst + r * 4 + 3] = c3;
    }
}

/// Signed distance of point p to cuboid k; returns (dist, gx, gy, gz) packed in a Line-free way.
#[cube]
fn sphere_box(cuboid_f: &[f32], k: usize, px: f32, py: f32, pz: f32, out: &mut Array<f32>) {
    let b = k * 16;
    let lx = cuboid_f[b] * px + cuboid_f[b + 1] * py + cuboid_f[b + 2] * pz + cuboid_f[b + 3];
    let ly = cuboid_f[b + 4] * px + cuboid_f[b + 5] * py + cuboid_f[b + 6] * pz + cuboid_f[b + 7];
    let lz = cuboid_f[b + 8] * px + cuboid_f[b + 9] * py + cuboid_f[b + 10] * pz + cuboid_f[b + 11];
    let qx = lx.abs() - cuboid_f[b + 12];
    let qy = ly.abs() - cuboid_f[b + 13];
    let qz = lz.abs() - cuboid_f[b + 14];
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    let oz = qz.max(0.0);
    let outside = (ox * ox + oy * oy + oz * oz).sqrt();
    let inside = qx.max(qy).max(qz).min(0.0);
    let sx = select(lx < 0.0, -1.0f32, 1.0f32);
    let sy = select(ly < 0.0, -1.0f32, 1.0f32);
    let sz = select(lz < 0.0, -1.0f32, 1.0f32);
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut gz = 0.0f32;
    if outside > 0.0 {
        let inv = 1.0 / outside;
        gx = sx * ox * inv;
        gy = sy * oy * inv;
        gz = sz * oz * inv;
    } else if qx >= qy && qx >= qz {
        gx = sx;
    } else if qy >= qz {
        gy = sy;
    } else {
        gz = sz;
    }
    out[0] = outside + inside;
    out[1] = cuboid_f[b] * gx + cuboid_f[b + 4] * gy + cuboid_f[b + 8] * gz;
    out[2] = cuboid_f[b + 1] * gx + cuboid_f[b + 5] * gy + cuboid_f[b + 9] * gz;
    out[3] = cuboid_f[b + 2] * gx + cuboid_f[b + 6] * gy + cuboid_f[b + 10] * gz;
}

#[cube]
fn fk(
    link_f: &[f32],
    link_i: &[i32],
    q: &[f32],
    b: usize,
    batch: usize,
    n_links: usize,
    frames: &mut Array<f32>,
) {
    // Slot MAX_LINKS holds the scratch local transform.
    let tmp = MAX_LINKS * FRAME;
    for i in 0..n_links {
        let qi = link_i[i * 4 + 2];
        let mut value = 0.0f32;
        if qi >= 0 {
            value = q[usize::cast_from(qi) * batch + b];
        }
        let parent = link_i[i * 4];
        if parent < 0 {
            local_transform(link_f, link_i, i, value, frames, i * FRAME);
        } else {
            local_transform(link_f, link_i, i, value, frames, tmp);
            compose(frames, usize::cast_from(parent) * FRAME, tmp, i * FRAME);
        }
    }
}

#[cube]
fn sphere_query(
    sphere_f: &[f32],
    sphere_link: &[u32],
    cuboid_f: &[f32],
    frames: &Array<f32>,
    s: usize,
    n_cuboids: usize,
    best: &mut Array<f32>,
    scratch: &mut Array<f32>,
) {
    let f = usize::cast_from(sphere_link[s]) * FRAME;
    let cx = sphere_f[s * 4];
    let cy = sphere_f[s * 4 + 1];
    let cz = sphere_f[s * 4 + 2];
    let px = frames[f] * cx + frames[f + 1] * cy + frames[f + 2] * cz + frames[f + 3];
    let py = frames[f + 4] * cx + frames[f + 5] * cy + frames[f + 6] * cz + frames[f + 7];
    let pz = frames[f + 8] * cx + frames[f + 9] * cy + frames[f + 10] * cz + frames[f + 11];
    best[0] = 3.0e38;
    for k in 0..n_cuboids {
        sphere_box(cuboid_f, k, px, py, pz, scratch);
        if scratch[0] < best[0] {
            best[0] = scratch[0];
            best[1] = scratch[1];
            best[2] = scratch[2];
            best[3] = scratch[3];
        }
    }
    best[0] -= sphere_f[s * 4 + 3];
}

// ---------------------------------------------------------------------------
// Kernels

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fused_full(
    link_f: &[f32],
    link_i: &[i32],
    sphere_f: &[f32],
    sphere_link: &[u32],
    cuboid_f: &[f32],
    q: &[f32],
    sphere_dist: &mut [f32],
    sphere_grad: &mut [f32],
    config_min: &mut [f32],
    batch: usize,
    n_links: usize,
    n_spheres: usize,
    n_cuboids: usize,
) {
    let b = ABSOLUTE_POS;
    if b < batch {
        let mut frames = Array::<f32>::new((MAX_LINKS + 1) * FRAME);
        let mut best = Array::<f32>::new(4usize);
        let mut scratch = Array::<f32>::new(4usize);
        fk(link_f, link_i, q, b, batch, n_links, &mut frames);
        let mut m = 3.0e38f32;
        for s in 0..n_spheres {
            sphere_query(
                sphere_f,
                sphere_link,
                cuboid_f,
                &frames,
                s,
                n_cuboids,
                &mut best,
                &mut scratch,
            );
            m = m.min(best[0]);
            let i = s * batch + b;
            sphere_dist[i] = best[0];
            sphere_grad[i * 4] = best[1];
            sphere_grad[i * 4 + 1] = best[2];
            sphere_grad[i * 4 + 2] = best[3];
            sphere_grad[i * 4 + 3] = 0.0;
        }
        config_min[b] = m;
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fused_min(
    link_f: &[f32],
    link_i: &[i32],
    sphere_f: &[f32],
    sphere_link: &[u32],
    cuboid_f: &[f32],
    q: &[f32],
    config_min: &mut [f32],
    batch: usize,
    n_links: usize,
    n_spheres: usize,
    n_cuboids: usize,
) {
    let b = ABSOLUTE_POS;
    if b < batch {
        let mut frames = Array::<f32>::new((MAX_LINKS + 1) * FRAME);
        let mut best = Array::<f32>::new(4usize);
        let mut scratch = Array::<f32>::new(4usize);
        fk(link_f, link_i, q, b, batch, n_links, &mut frames);
        let mut m = 3.0e38f32;
        for s in 0..n_spheres {
            sphere_query(
                sphere_f,
                sphere_link,
                cuboid_f,
                &frames,
                s,
                n_cuboids,
                &mut best,
                &mut scratch,
            );
            m = m.min(best[0]);
        }
        config_min[b] = m;
    }
}

/// Same as `fused_min`, with the model's shape baked in at JIT time: one kernel per
/// (n_links, n_spheres, n_cuboids), every loop unrolled.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fused_min_specialized(
    link_f: &[f32],
    link_i: &[i32],
    sphere_f: &[f32],
    sphere_link: &[u32],
    cuboid_f: &[f32],
    q: &[f32],
    config_min: &mut [f32],
    batch: usize,
    #[comptime] n_links: usize,
    #[comptime] n_spheres: usize,
    #[comptime] n_cuboids: usize,
) {
    let b = ABSOLUTE_POS;
    if b < batch {
        let mut frames = Array::<f32>::new((MAX_LINKS + 1) * FRAME);
        let mut best = Array::<f32>::new(4usize);
        let mut scratch = Array::<f32>::new(4usize);
        let tmp = MAX_LINKS * FRAME;
        #[unroll]
        for i in 0..n_links {
            let qi = link_i[i * 4 + 2];
            let mut value = 0.0f32;
            if qi >= 0 {
                value = q[usize::cast_from(qi) * batch + b];
            }
            let parent = link_i[i * 4];
            if parent < 0 {
                local_transform(link_f, link_i, i, value, &mut frames, i * FRAME);
            } else {
                local_transform(link_f, link_i, i, value, &mut frames, tmp);
                compose(
                    &mut frames,
                    usize::cast_from(parent) * FRAME,
                    tmp,
                    i * FRAME,
                );
            }
        }
        let mut m = 3.0e38f32;
        #[unroll]
        for s in 0..n_spheres {
            let f = usize::cast_from(sphere_link[s]) * FRAME;
            let cx = sphere_f[s * 4];
            let cy = sphere_f[s * 4 + 1];
            let cz = sphere_f[s * 4 + 2];
            let px = frames[f] * cx + frames[f + 1] * cy + frames[f + 2] * cz + frames[f + 3];
            let py = frames[f + 4] * cx + frames[f + 5] * cy + frames[f + 6] * cz + frames[f + 7];
            let pz = frames[f + 8] * cx + frames[f + 9] * cy + frames[f + 10] * cz + frames[f + 11];
            best[0] = 3.0e38;
            #[unroll]
            for k in 0..n_cuboids {
                sphere_box(cuboid_f, k, px, py, pz, &mut scratch);
                best[0] = best[0].min(scratch[0]);
            }
            m = m.min(best[0] - sphere_f[s * 4 + 3]);
        }
        config_min[b] = m;
    }
}

// ---------------------------------------------------------------------------
// Host side

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    FusedFull,
    FusedMin,
    /// `FusedMin` with the robot/scene shape compiled into the kernel.
    FusedMinSpecialized,
}

pub struct CubeclQuery {
    client: cubecl::client::Client,
    link_f: cubecl::server::Handle,
    link_i: cubecl::server::Handle,
    sphere_f: cubecl::server::Handle,
    sphere_link: cubecl::server::Handle,
    cuboid_f: cubecl::server::Handle,
    q: cubecl::server::Handle,
    sphere_dist: cubecl::server::Handle,
    sphere_grad: cubecl::server::Handle,
    config_min: cubecl::server::Handle,
    n_links: usize,
    n_cuboids: usize,
    pub batch: usize,
    pub n_spheres: usize,
    pub dof: usize,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

impl CubeclQuery {
    pub fn new(model: &ModelBuffers, batch: usize) -> Self {
        let client = WgpuRuntime::<AutoCompiler>::client(&WgpuDevice::default());
        let mut link_f = Vec::new();
        let mut link_i = Vec::new();
        for l in &model.links {
            link_f.extend_from_slice(&l.origin);
            link_f.extend_from_slice(&l.axis);
            link_f.push(0.0);
            link_i.extend_from_slice(&[l.parent, l.joint_type as i32, l.q_index, 0]);
        }
        let sphere_f: Vec<f32> = model
            .spheres
            .iter()
            .flat_map(|s| [s.center[0], s.center[1], s.center[2], s.radius])
            .collect();
        let sphere_link: Vec<u32> = model.spheres.iter().map(|s| s.link).collect();
        let cuboid_f: Vec<f32> = model
            .cuboids
            .iter()
            .flat_map(|c| {
                c.inv
                    .iter()
                    .copied()
                    .chain(c.half_extents)
                    .chain([0.0])
                    .collect::<Vec<_>>()
            })
            .collect();
        let n_spheres = model.spheres.len();
        Self {
            link_f: client.create_from_slice(f32_bytes(&link_f)),
            link_i: client.create_from_slice(bytemuck::cast_slice(&link_i)),
            sphere_f: client.create_from_slice(f32_bytes(&sphere_f)),
            sphere_link: client.create_from_slice(bytemuck::cast_slice(&sphere_link)),
            cuboid_f: client.create_from_slice(f32_bytes(&cuboid_f)),
            q: client.empty(model.dof * batch * 4),
            sphere_dist: client.empty(batch * n_spheres * 4),
            sphere_grad: client.empty(batch * n_spheres * 16),
            config_min: client.empty(batch * 4),
            client,
            n_links: model.links.len(),
            n_cuboids: model.cuboids.len(),
            batch,
            n_spheres,
            dof: model.dof,
        }
    }

    pub fn upload_q(&mut self, q: &[f32]) {
        assert_eq!(q.len(), self.dof * self.batch);
        self.q = self.client.create_from_slice(f32_bytes(q));
    }

    fn arg(h: &cubecl::server::Handle, len: usize) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(h.clone(), len) }
    }

    fn launch(&self, kernel: Kernel) {
        let groups = (self.batch as u32).div_ceil(WG);
        let x = groups.min(MAX_WG_X);
        let count = CubeCount::Static(x, groups.div_ceil(x), 1);
        let dim = CubeDim::new_1d(WG);
        let n = self.n_links;
        let (ns, nc, b) = (self.n_spheres, self.n_cuboids, self.batch);
        let a = |h: &cubecl::server::Handle, len: usize| Self::arg(h, len);
        unsafe {
            match kernel {
                Kernel::FusedFull => fused_full::launch_unchecked(
                    &self.client,
                    count,
                    dim,
                    a(&self.link_f, n * 16),
                    a(&self.link_i, n * 4),
                    a(&self.sphere_f, ns * 4),
                    a(&self.sphere_link, ns),
                    a(&self.cuboid_f, nc * 16),
                    a(&self.q, self.dof * b),
                    a(&self.sphere_dist, ns * b),
                    a(&self.sphere_grad, ns * b * 4),
                    a(&self.config_min, b),
                    b,
                    n,
                    ns,
                    nc,
                ),
                Kernel::FusedMin => fused_min::launch_unchecked(
                    &self.client,
                    count,
                    dim,
                    a(&self.link_f, n * 16),
                    a(&self.link_i, n * 4),
                    a(&self.sphere_f, ns * 4),
                    a(&self.sphere_link, ns),
                    a(&self.cuboid_f, nc * 16),
                    a(&self.q, self.dof * b),
                    a(&self.config_min, b),
                    b,
                    n,
                    ns,
                    nc,
                ),
                Kernel::FusedMinSpecialized => fused_min_specialized::launch_unchecked(
                    &self.client,
                    count,
                    dim,
                    a(&self.link_f, n * 16),
                    a(&self.link_i, n * 4),
                    a(&self.sphere_f, ns * 4),
                    a(&self.sphere_link, ns),
                    a(&self.cuboid_f, nc * 16),
                    a(&self.q, self.dof * b),
                    a(&self.config_min, b),
                    b,
                    n,
                    ns,
                    nc,
                ),
            }
        }
    }

    fn sync(&self) {
        cubecl::future::block_on(self.client.sync()).expect("sync");
    }

    /// `reps` launches, then one sync; time per launch.
    pub fn time_throughput(&self, kernel: Kernel, reps: u32) -> Duration {
        self.sync();
        let t = Instant::now();
        for _ in 0..reps {
            self.launch(kernel);
        }
        self.sync();
        t.elapsed() / reps
    }

    pub fn time_latency(&self, kernel: Kernel) -> Duration {
        self.sync();
        let t = Instant::now();
        self.launch(kernel);
        self.sync();
        t.elapsed()
    }

    pub fn round_trip_min(&mut self, q: &[f32], kernel: Kernel) -> (Vec<f32>, Duration) {
        self.sync();
        let t = Instant::now();
        self.upload_q(q);
        self.launch(kernel);
        let bytes = self.client.read_one(self.config_min.clone()).expect("read");
        let out = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        (out, t.elapsed())
    }

    pub fn run_full(&mut self, q: &[f32]) -> BatchOutput {
        self.upload_q(q);
        self.launch(Kernel::FusedFull);
        let read = |h: &cubecl::server::Handle| -> Vec<f32> {
            bytemuck::cast_slice::<u8, f32>(&self.client.read_one(h.clone()).expect("read"))
                .to_vec()
        };
        let grad = read(&self.sphere_grad);
        BatchOutput {
            batch: self.batch,
            sphere_dist: read(&self.sphere_dist),
            sphere_grad: grad
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
            config_min: read(&self.config_min),
        }
    }
}

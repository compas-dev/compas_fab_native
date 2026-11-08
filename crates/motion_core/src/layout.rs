//! Buffer layouts shared by every compute backend.
//!
//! Rules (from the PRD): f32 only, structure-of-arrays for anything indexed by
//! batch, fixed-size records with explicit padding, no vec3 in storage (WGSL
//! aligns vec3 to 16 bytes, which silently breaks a Rust `[f32; 3]` mirror).
//! Every struct here has a byte-identical WGSL twin in `motion_gpu`.
//!
//! # Model buffers (uploaded once per robot / scene)
//!
//! | buffer    | element       | count       |
//! |-----------|---------------|-------------|
//! | `links`   | [`GpuLink`]   | `n_links`   |
//! | `spheres` | [`GpuSphere`] | `n_spheres` |
//! | `cuboids` | [`GpuCuboid`] | `n_cuboids` |
//!
//! # Batch buffers (per query, `B` = batch size)
//!
//! Batch index is always the fastest-moving index, so neighbouring GPU threads
//! (and neighbouring SIMD lanes on the CPU) touch neighbouring memory.
//!
//! | buffer          | element | index         | meaning |
//! |-----------------|---------|---------------|---------|
//! | `q`             | f32     | `j * B + b`   | joint `j` of configuration `b` |
//! | `spheres_world` | vec4    | `s * B + b`   | xyz world centre, w radius |
//! | `sphere_dist`   | f32     | `s * B + b`   | signed distance to the nearest obstacle minus radius (< 0: collision) |
//! | `sphere_grad`   | vec4    | `s * B + b`   | xyz gradient of `sphere_dist` w.r.t. the sphere centre, w unused |
//! | `config_min`    | f32     | `b`           | min of `sphere_dist` over the configuration's spheres |
//!
//! # Voxel blocks (sensed geometry, Phase 1 offline ESDF / Phase 3 live TSDF)
//!
//! Not used by the kernels in this spike; fixed here so the distance query and
//! the planning buffers are designed around them from the start.
//!
//! * Space is split into blocks of `VOXEL_BLOCK_DIM^3` voxels (8^3 = 512).
//! * Allocated blocks live in a pool; block `k` owns voxels `k * 512 .. (k+1) * 512`,
//!   voxel order `x + 8 * (y + 8 * z)` inside the block.
//! * `block_hash`: open-addressing table of [`GpuBlockSlot`], keyed by integer block
//!   coordinates, linear probing; `pool_index == u32::MAX` marks an empty slot.
//! * `tsdf_pool`: one `u32` per voxel, two f16 packed with WGSL's core
//!   `pack2x16float` (tsdf, weight). Keeps WebGPU's optional `shader-f16` out of it.
//! * `esdf_pool`: one `f32` per voxel, metres, positive outside.
//! * A sphere query against the ESDF is a hash lookup + trilinear sample + the
//!   same `dist - radius` contract as the CAD geometry, so solvers cannot tell
//!   sensed from known geometry.

use bytemuck::{Pod, Zeroable};

use crate::model::{Cuboid, JointType, Mat34, RobotModel, Scene};

/// Upper bound on links per robot in a GPU kernel (private-memory transform stack).
pub const MAX_LINKS: usize = 32;
pub const VOXEL_BLOCK_DIM: u32 = 8;
pub const VOXELS_PER_BLOCK: u32 = VOXEL_BLOCK_DIM * VOXEL_BLOCK_DIM * VOXEL_BLOCK_DIM;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuLink {
    pub origin: [f32; 12],
    pub axis: [f32; 3],
    pub parent: i32,
    pub joint_type: u32,
    pub q_index: i32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub link: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCuboid {
    /// Inverse pose: world -> box coordinates.
    pub inv: [f32; 12],
    pub half_extents: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBlockSlot {
    pub coord: [i32; 3],
    pub pool_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub batch: u32,
    pub dof: u32,
    pub n_links: u32,
    pub n_spheres: u32,
    pub n_cuboids: u32,
    pub _pad: [u32; 3],
}

/// The model and scene, flattened into the shared layout.
#[derive(Clone, Debug)]
pub struct ModelBuffers {
    pub dof: usize,
    pub links: Vec<GpuLink>,
    pub spheres: Vec<GpuSphere>,
    pub cuboids: Vec<GpuCuboid>,
}

impl ModelBuffers {
    pub fn new(model: &RobotModel, scene: &Scene) -> Self {
        assert!(
            model.links.len() <= MAX_LINKS,
            "more than {MAX_LINKS} links"
        );
        let links = model
            .links
            .iter()
            .map(|l| GpuLink {
                origin: l.origin.map(|v| v as f32),
                axis: l.axis.map(|v| v as f32),
                parent: l.parent,
                joint_type: l.joint_type as u32,
                q_index: l.q_index,
                _pad: [0; 2],
            })
            .collect();
        // Sorted by link, so spheres of one link are contiguous and the link
        // transform stays in registers while they are processed.
        let mut spheres: Vec<GpuSphere> = model
            .spheres
            .iter()
            .map(|s| GpuSphere {
                center: s.center.map(|v| v as f32),
                radius: s.radius as f32,
                link: s.link,
                _pad: [0; 3],
            })
            .collect();
        spheres.sort_by_key(|s| s.link);
        let cuboids = scene.cuboids.iter().map(cuboid_record).collect();
        Self {
            dof: model.dof(),
            links,
            spheres,
            cuboids,
        }
    }

    pub fn params(&self, batch: usize) -> Params {
        Params {
            batch: batch as u32,
            dof: self.dof as u32,
            n_links: self.links.len() as u32,
            n_spheres: self.spheres.len() as u32,
            n_cuboids: self.cuboids.len() as u32,
            _pad: [0; 3],
        }
    }
}

fn cuboid_record(c: &Cuboid) -> GpuCuboid {
    GpuCuboid {
        inv: invert_rigid(&c.frame).map(|v| v as f32),
        half_extents: c.half_extents.map(|v| v as f32),
        _pad: 0.0,
    }
}

pub fn invert_rigid(m: &Mat34) -> Mat34 {
    // [R t]^-1 = [R^T, -R^T t]
    let (t0, t1, t2) = (m[3], m[7], m[11]);
    [
        m[0],
        m[4],
        m[8],
        -(m[0] * t0 + m[4] * t1 + m[8] * t2),
        m[1],
        m[5],
        m[9],
        -(m[1] * t0 + m[5] * t1 + m[9] * t2),
        m[2],
        m[6],
        m[10],
        -(m[2] * t0 + m[6] * t1 + m[10] * t2),
    ]
}

impl JointType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => JointType::Revolute,
            2 => JointType::Prismatic,
            _ => JointType::Fixed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_sizes_match_wgsl() {
        assert_eq!(std::mem::size_of::<GpuLink>(), 80);
        assert_eq!(std::mem::size_of::<GpuSphere>(), 32);
        assert_eq!(std::mem::size_of::<GpuCuboid>(), 64);
        assert_eq!(std::mem::size_of::<GpuBlockSlot>(), 16);
        assert_eq!(std::mem::size_of::<Params>(), 32);
    }
}

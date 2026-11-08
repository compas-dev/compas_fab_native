//! Sparse voxel ESDF in the block layout of `layout.rs` (8^3 voxels per block, a hash from
//! block coordinates to a slot in the pool, one f32 distance per voxel).
//!
//! Distances are stored at grid points `g * voxel` (g integer). A query interpolates the
//! eight surrounding grid points trilinearly; grid points in blocks that were never
//! allocated are "far" (the truncation distance). Built offline from meshes (see
//! `motion_exact::build_esdf`) or, later, from sensed TSDF data: the query is the same.

use std::collections::HashMap;

use crate::layout::{VOXELS_PER_BLOCK, VOXEL_BLOCK_DIM};

const DIM: i32 = VOXEL_BLOCK_DIM as i32;

#[derive(Clone, Debug)]
pub struct VoxelEsdf {
    /// Grid spacing, metres.
    pub voxel: f32,
    /// Distances at or beyond this are not stored; missing blocks read as this value.
    pub truncation: f32,
    /// Block coordinates -> pool slot.
    pub slots: HashMap<[i32; 3], u32>,
    /// Block `k` owns `pool[k * 512 .. (k + 1) * 512]`, voxel order `x + 8 * (y + 8 * z)`.
    pub pool: Vec<f32>,
}

impl VoxelEsdf {
    pub fn new(voxel: f32, truncation: f32) -> Self {
        Self {
            voxel,
            truncation,
            slots: HashMap::new(),
            pool: Vec::new(),
        }
    }

    /// Store a block of 512 distances (grid points of block `coord`).
    pub fn insert_block(&mut self, coord: [i32; 3], distances: &[f32]) {
        assert_eq!(distances.len(), VOXELS_PER_BLOCK as usize);
        let slot = self.slots.len() as u32;
        self.slots.insert(coord, slot);
        self.pool.extend_from_slice(distances);
    }

    pub fn n_blocks(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn grid(&self, g: [i32; 3]) -> f32 {
        let b = [
            g[0].div_euclid(DIM),
            g[1].div_euclid(DIM),
            g[2].div_euclid(DIM),
        ];
        match self.slots.get(&b) {
            Some(&slot) => {
                let l = [g[0] - b[0] * DIM, g[1] - b[1] * DIM, g[2] - b[2] * DIM];
                self.pool[slot as usize * VOXELS_PER_BLOCK as usize
                    + (l[0] + DIM * (l[1] + DIM * l[2])) as usize]
            }
            None => self.truncation,
        }
    }

    /// Interpolated signed distance and its gradient at `p`.
    pub fn sample(&self, p: [f32; 3]) -> (f32, [f32; 3]) {
        let s = [p[0] / self.voxel, p[1] / self.voxel, p[2] / self.voxel];
        let g = [
            s[0].floor() as i32,
            s[1].floor() as i32,
            s[2].floor() as i32,
        ];
        let f = [s[0] - g[0] as f32, s[1] - g[1] as f32, s[2] - g[2] as f32];
        let mut c = [0.0f32; 8];
        for (k, v) in c.iter_mut().enumerate() {
            let o = [(k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32];
            *v = self.grid([g[0] + o[0], g[1] + o[1], g[2] + o[2]]);
        }
        // Trilinear interpolation and its derivative.
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x00 = lerp(c[0], c[1], f[0]);
        let x10 = lerp(c[2], c[3], f[0]);
        let x01 = lerp(c[4], c[5], f[0]);
        let x11 = lerp(c[6], c[7], f[0]);
        let y0 = lerp(x00, x10, f[1]);
        let y1 = lerp(x01, x11, f[1]);
        let d = lerp(y0, y1, f[2]);
        let dx = lerp(
            lerp(c[1] - c[0], c[3] - c[2], f[1]),
            lerp(c[5] - c[4], c[7] - c[6], f[1]),
            f[2],
        );
        let dy = lerp(x10 - x00, x11 - x01, f[2]);
        let dz = y1 - y0;
        let inv = 1.0 / self.voxel;
        (d, [dx * inv, dy * inv, dz * inv])
    }
}

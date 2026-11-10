//! Free-space (point-to-point) motion planning in joint space.
//!
//! 1. The straight motion from the start to each goal is tried first.
//! 2. RRT-Connect (Kuffner and LaValle, 2000), balanced and with dynamic domain, on the
//!    yes/no checker of `check.rs`, as VAMP runs it (Thomason, Kingston and Kavraki,
//!    "Motions in Microseconds via Vectorized Sampling-Based Planning", ICRA 2024).
//! 3. Simplification: shortcutting, then B-spline smoothing, repeated while they help.
//!
//! The default settings are VAMP's (`RRTCSettings`, `SimplifySettings`), so benchmarks
//! against it compare implementations, not parameters.
//!
//! Attribution: the planner and the simplification follow VAMP's `planning/rrtc.hh`,
//! `planning/simplify.hh` and `random/halton.hh` (https://github.com/KavrakiLab/vamp,
//! Apache-2.0, Kingston, Thomason, Ramsey and contributors), rewritten in Rust and modified
//! (see `NOTICE`; license in `licenses/Apache-2.0.txt`).

use std::time::{Duration, Instant};

use crate::check::Checker;
use crate::simd::{v, LANES, V};

#[derive(Clone, Debug)]
pub struct FreeSpaceSettings {
    /// Longest tree extension, in joint-space (L2) distance.
    pub range: f32,
    /// Collision checks per unit of joint-space distance along a motion.
    pub resolution: f32,
    pub dynamic_domain: bool,
    pub dd_radius: f32,
    pub dd_alpha: f32,
    pub dd_min_radius: f32,
    /// Grow the smaller tree.
    pub balance: bool,
    pub tree_ratio: f32,
    pub max_iterations: usize,
    pub max_samples: usize,
    pub start_tree_first: bool,
    /// Rounds of (shortcut, B-spline) simplification; 0 disables it.
    pub simplify_rounds: usize,
    pub bspline_steps: usize,
    pub bspline_min_change: f32,
    pub bspline_midpoint: f32,
    /// Halton sequence offset (different offsets give different, reproducible runs).
    pub seed: u64,
}

impl Default for FreeSpaceSettings {
    fn default() -> Self {
        Self {
            range: 1.0,
            resolution: 32.0,
            dynamic_domain: true,
            dd_radius: 4.0,
            dd_alpha: 1e-4,
            dd_min_radius: 1.0,
            balance: true,
            tree_ratio: 1.0,
            max_iterations: 1_000_000,
            max_samples: 1_000_000,
            start_tree_first: true,
            simplify_rounds: 5,
            bspline_steps: 1,
            bspline_min_change: 0.1,
            bspline_midpoint: 0.5,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FreeSpaceStats {
    pub iterations: usize,
    pub tree_sizes: [usize; 2],
    pub planning: Duration,
    pub simplification: Duration,
    pub initial_cost: f32,
    pub cost: f32,
}

#[derive(Clone, Debug)]
pub struct FreeSpacePlan {
    pub path: Vec<Vec<f32>>,
    pub stats: FreeSpaceStats,
}

#[derive(Clone, Debug)]
pub enum FreeSpaceError {
    StartInCollision,
    AllGoalsInCollision,
    NoPath(Box<FreeSpaceStats>),
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

pub fn path_cost(path: &[Vec<f32>]) -> f32 {
    path.windows(2).map(|w| distance(&w[0], &w[1])).sum()
}

/// Halton sequence in the joint limits (bases 3, 5, 7, ... as in VAMP).
struct Halton {
    index: u64,
    lower: Vec<f32>,
    span: Vec<f32>,
}

/// Most joints a planned robot may have.
pub const MAX_DOF: usize = 16;

const PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];

impl Halton {
    fn new(checker: &Checker, seed: u64) -> Self {
        Self {
            index: seed,
            lower: checker.lower.clone(),
            span: checker
                .upper
                .iter()
                .zip(&checker.lower)
                .map(|(u, l)| u - l)
                .collect(),
        }
    }

    fn next(&mut self, out: &mut [f32]) {
        self.index += 1;
        for (j, o) in out.iter_mut().enumerate() {
            let base = PRIMES[j];
            let (mut i, mut f, mut r) = (self.index, 1.0f64, 0.0f64);
            while i > 0 {
                f /= base as f64;
                r += f * (i % base) as f64;
                i /= base;
            }
            *o = self.lower[j] + self.span[j] * r as f32;
        }
    }
}

/// One tree's nodes for nearest-neighbour queries: blocks of `LANES` nodes stored per
/// dimension (`block * dof * LANES + j * LANES + lane`), scanned `LANES` nodes at a time.
/// Unused lanes of the last block hold infinity.
struct Tree {
    dof: usize,
    blocks: Vec<f32>,
    ids: Vec<u32>,
}

impl Tree {
    fn new(dof: usize) -> Self {
        Self {
            dof,
            blocks: Vec::new(),
            ids: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn insert(&mut self, id: u32, q: &[f32]) {
        let (block, lane) = (self.ids.len() / LANES, self.ids.len() % LANES);
        if lane == 0 {
            self.blocks
                .resize((block + 1) * self.dof * LANES, f32::INFINITY);
        }
        for (j, &x) in q.iter().enumerate() {
            self.blocks[(block * self.dof + j) * LANES + lane] = x;
        }
        self.ids.push(id);
    }

    /// Nearest node (id, distance).
    fn nearest(&self, q: &[f32]) -> (u32, f32) {
        let dof = self.dof;
        let mut qv = [v(0.0); MAX_DOF];
        for (j, &x) in q.iter().enumerate() {
            qv[j] = v(x);
        }
        let mut best_d = v(f32::INFINITY);
        let mut best_b = v(0.0);
        for (b, block) in self.blocks.chunks_exact(dof * LANES).enumerate() {
            let mut d = v(0.0);
            for j in 0..dof {
                let x: [f32; LANES] = block[j * LANES..(j + 1) * LANES].try_into().unwrap();
                let e = V::new(x) - qv[j];
                d = e.mul_add(e, d);
            }
            let closer = d.simd_lt(best_d);
            best_d = closer.select(d, best_d);
            best_b = closer.select(v(b as f32), best_b);
        }
        let (d, b) = (best_d.to_array(), best_b.to_array());
        let mut lane = 0;
        for l in 1..LANES {
            // Ties go to the earlier node, as in a plain linear scan.
            let (k, kl) = (b[l] as usize * LANES + l, b[lane] as usize * LANES + lane);
            if d[l] < d[lane] || (d[l] == d[lane] && k < kl) {
                lane = l;
            }
        }
        (self.ids[b[lane] as usize * LANES + lane], d[lane].sqrt())
    }
}

/// Nodes of both trees: coordinates, parent (self for roots) and dynamic-domain radius.
struct Nodes {
    dof: usize,
    coords: Vec<f32>,
    parents: Vec<u32>,
    radii: Vec<f32>,
}

impl Nodes {
    fn add(&mut self, q: &[f32], parent: Option<u32>) -> u32 {
        let id = self.parents.len() as u32;
        self.coords.extend_from_slice(q);
        self.parents.push(parent.unwrap_or(id));
        self.radii.push(f32::INFINITY);
        id
    }

    fn get(&self, id: u32) -> &[f32] {
        &self.coords[id as usize * self.dof..(id as usize + 1) * self.dof]
    }

    fn len(&self) -> usize {
        self.parents.len()
    }
}

/// Plan a collision-free path from `start` to any of `goals`.
pub fn plan(
    checker: &Checker,
    start: &[f32],
    goals: &[Vec<f32>],
    settings: &FreeSpaceSettings,
) -> Result<FreeSpacePlan, FreeSpaceError> {
    if checker.collides(start) {
        return Err(FreeSpaceError::StartInCollision);
    }
    let goals: Vec<&Vec<f32>> = goals.iter().filter(|g| !checker.collides(g)).collect();
    if goals.is_empty() {
        return Err(FreeSpaceError::AllGoalsInCollision);
    }
    let t0 = Instant::now();
    let (path, mut stats) = rrt_connect(checker, start, &goals, settings)?;
    stats.planning = t0.elapsed();
    stats.initial_cost = path_cost(&path);
    let t1 = Instant::now();
    let path = simplify(checker, path, settings);
    stats.simplification = t1.elapsed();
    stats.cost = path_cost(&path);
    Ok(FreeSpacePlan { path, stats })
}

fn rrt_connect(
    checker: &Checker,
    start: &[f32],
    goals: &[&Vec<f32>],
    s: &FreeSpaceSettings,
) -> Result<(Vec<Vec<f32>>, FreeSpaceStats), FreeSpaceError> {
    let dof = checker.dof;
    let mut stats = FreeSpaceStats::default();
    for goal in goals {
        if checker.motion_free(start, goal, s.resolution) {
            stats.tree_sizes = [1, 1];
            return Ok((vec![start.to_vec(), goal.to_vec()], stats));
        }
    }

    let mut nodes = Nodes {
        dof,
        coords: Vec::with_capacity(1024 * dof),
        parents: Vec::with_capacity(1024),
        radii: Vec::with_capacity(1024),
    };
    let mut start_tree = Tree::new(dof);
    let mut goal_tree = Tree::new(dof);
    let id = nodes.add(start, None);
    start_tree.insert(id, start);
    for goal in goals {
        let id = nodes.add(goal, None);
        goal_tree.insert(id, goal);
    }
    // `a` is the tree that grows towards the sample; the trees swap each iteration.
    let (mut a, mut b) = if s.start_tree_first {
        (&mut goal_tree, &mut start_tree)
    } else {
        (&mut start_tree, &mut goal_tree)
    };
    let mut a_is_start = !s.start_tree_first;

    let mut halton = Halton::new(checker, s.seed);
    let mut sample = vec![0.0f32; dof];
    let mut new = vec![0.0f32; dof];
    let mut prior = vec![0.0f32; dof];
    let mut step = vec![0.0f32; dof];

    while stats.iterations < s.max_iterations && nodes.len() < s.max_samples {
        stats.iterations += 1;
        let ratio = (a.len() as f32 - b.len() as f32).abs() / a.len() as f32;
        if !s.balance || ratio < s.tree_ratio {
            std::mem::swap(&mut a, &mut b);
            a_is_start = !a_is_start;
        }
        halton.next(&mut sample);
        let (near, near_dist) = a.nearest(&sample);
        let near_radius = nodes.radii[near as usize];
        if s.dynamic_domain && near_radius < near_dist {
            continue;
        }
        let mut near_q = [0.0f32; MAX_DOF];
        near_q[..dof].copy_from_slice(nodes.get(near));
        let near_q = &near_q[..dof];
        let reach = near_dist < s.range;
        let scale = if reach { 1.0 } else { s.range / near_dist };
        for j in 0..dof {
            new[j] = near_q[j] + (sample[j] - near_q[j]) * scale;
        }
        if !checker.motion_free(near_q, &new, s.resolution) {
            if s.dynamic_domain {
                let r = &mut nodes.radii[near as usize];
                *r = if r.is_infinite() {
                    s.dd_radius
                } else {
                    (*r * (1.0 - s.dd_alpha)).max(s.dd_min_radius)
                };
            }
            continue;
        }
        let new_id = nodes.add(&new, Some(near));
        a.insert(new_id, &new);
        if s.dynamic_domain && near_radius.is_finite() {
            nodes.radii[near as usize] *= 1.0 + s.dd_alpha;
        }

        // Connect: extend from the new node towards the other tree's nearest node.
        let (other, other_dist) = b.nearest(&new);
        let mut other_q = [0.0f32; MAX_DOF];
        other_q[..dof].copy_from_slice(nodes.get(other));
        let n_ext = (other_dist / s.range).ceil().max(1.0) as usize;
        for j in 0..dof {
            step[j] = (other_q[j] - new[j]) / n_ext as f32;
        }
        prior.copy_from_slice(&new);
        let mut last = new_id;
        let mut connected = true;
        for _ in 0..n_ext {
            if nodes.len() >= s.max_samples {
                connected = false;
                break;
            }
            let mut next = [0.0f32; MAX_DOF];
            for j in 0..dof {
                next[j] = prior[j] + step[j];
            }
            let next = &next[..dof];
            if !checker.motion_free(&prior, next, s.resolution) {
                connected = false;
                break;
            }
            last = nodes.add(next, Some(last));
            a.insert(last, next);
            prior.copy_from_slice(next);
        }
        if !connected {
            continue;
        }
        // Path: root of `a` .. last, then `other` .. root of `b`.
        let node = |id: u32| nodes.get(id).to_vec();
        let parents = &nodes.parents;
        let mut half_a = Vec::new();
        let mut id = last;
        loop {
            half_a.push(node(id));
            if parents[id as usize] == id {
                break;
            }
            id = parents[id as usize];
        }
        half_a.reverse();
        // `last` sits on `other` (same configuration), so start from its parent.
        let mut half_b = Vec::new();
        let mut id = other;
        while parents[id as usize] != id {
            id = parents[id as usize];
            half_b.push(node(id));
        }
        let mut path = half_a;
        path.extend(half_b);
        if !a_is_start {
            path.reverse();
        }
        stats.tree_sizes = [start_tree.len(), goal_tree.len()];
        return Ok((path, stats));
    }
    stats.tree_sizes = [start_tree.len(), goal_tree.len()];
    Err(FreeSpaceError::NoPath(Box::new(stats)))
}

fn interpolate(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect()
}

/// Shortcutting and B-spline smoothing (VAMP's default simplification).
pub fn simplify(
    checker: &Checker,
    mut path: Vec<Vec<f32>>,
    s: &FreeSpaceSettings,
) -> Vec<Vec<f32>> {
    let free = |a: &[f32], b: &[f32]| checker.motion_free(a, b, s.resolution);
    if path.len() <= 2 {
        return path;
    }
    if free(&path[0], &path[path.len() - 1]) {
        return vec![path[0].clone(), path[path.len() - 1].clone()];
    }
    for _ in 0..s.simplify_rounds {
        let mut changed = false;
        // Shortcut: from each vertex, jump to the farthest vertex reachable in a straight line.
        let mut i = 0;
        while path.len() >= 3 && i < path.len() - 2 {
            let mut j = path.len() - 1;
            while j > i + 1 {
                if free(&path[i], &path[j]) {
                    path.drain(i + 1..j);
                    changed = true;
                    break;
                }
                j -= 1;
            }
            i += 1;
        }
        // B-spline: subdivide, then pull every other vertex towards its neighbours' midpoint.
        if path.len() >= 3 {
            for _ in 0..s.bspline_steps {
                let mut sub = Vec::with_capacity(path.len() * 2);
                for w in path.windows(2) {
                    sub.push(w[0].clone());
                    sub.push(interpolate(&w[0], &w[1], 0.5));
                }
                sub.push(path[path.len() - 1].clone());
                path = sub;
                let mut updated = false;
                let mut k = 2;
                while k + 1 < path.len() {
                    let t1 = interpolate(&path[k], &path[k - 1], s.bspline_midpoint);
                    let t2 = interpolate(&path[k], &path[k + 1], s.bspline_midpoint);
                    let mid = interpolate(&t1, &t2, 0.5);
                    if distance(&path[k], &mid) > s.bspline_min_change
                        && free(&path[k - 1], &mid)
                        && free(&mid, &path[k + 1])
                    {
                        path[k] = mid;
                        updated = true;
                        changed = true;
                    }
                    k += 2;
                }
                if !updated {
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    path
}

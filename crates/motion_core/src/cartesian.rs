//! Cartesian path planner: layered graph + dynamic programming (PRD, Solvers 4).
//!
//! One layer per waypoint; nodes are collision-free IK solutions of the waypoint, sampled
//! over a free rotation about the tool axis; edges connect consecutive layers with a
//! joint-space cost. The cheapest path is found by dynamic programming. Edge motions are
//! collision-checked lazily: only edges on the current best path are interpolated and
//! checked (in one batch), invalid edges are removed and the search is repeated.

use std::collections::HashSet;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

use crate::cpu::{collision_min_lanes, LANES};
use crate::layout::ModelBuffers;
use crate::model::{Mat34, RobotModel};
use crate::reference::mul;

pub type Config = [f64; 6];

#[derive(Clone, Debug)]
pub struct CartesianRequest {
    /// Target frames of the tool centre point (TCF), in robot coordinates.
    pub waypoints: Vec<Mat34>,
    /// TCF relative to the end-effector link (`tool0`); identity without a tool.
    pub tcf: Mat34,
    /// Rotations about the TCF z axis sampled per waypoint (1 = the frame as given).
    pub free_axis_samples: usize,
    /// Half-range of the free rotation, radians (π = any rotation about the tool axis).
    pub free_axis_range: f64,
    /// Start configuration; the path begins with it when given.
    pub start: Option<Vec<f64>>,
    /// Largest joint change allowed between consecutive waypoints, radians.
    pub max_joint_step: f64,
    /// Interpolation resolution for edge collision checks, radians.
    pub edge_check_step: f64,
    /// Minimum clearance to obstacles and between links, metres.
    pub margin: f32,
}

impl Default for CartesianRequest {
    fn default() -> Self {
        Self {
            waypoints: Vec::new(),
            tcf: crate::model::IDENTITY,
            free_axis_samples: 1,
            free_axis_range: PI,
            start: None,
            max_joint_step: 0.5,
            edge_check_step: 0.05,
            margin: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CartesianStats {
    pub ik_nodes: usize,
    pub free_nodes: usize,
    pub edges_evaluated: usize,
    pub edge_configs_checked: usize,
    pub search_rounds: usize,
    pub t_ik: Duration,
    pub t_node_checks: Duration,
    pub t_search: Duration,
    pub t_edge_checks: Duration,
    pub t_total: Duration,
}

#[derive(Clone, Debug)]
pub struct CartesianResult {
    /// One configuration per waypoint (plus the start, when given), continuous in joint space.
    pub path: Vec<Config>,
    pub cost: f64,
    pub stats: CartesianStats,
}

#[derive(Debug)]
pub enum CartesianError {
    NoIk { waypoint: usize },
    NoCollisionFreeIk { waypoint: usize },
    NoPath { stats: CartesianStats },
}

fn rot_z(theta: f64) -> Mat34 {
    let (s, c) = theta.sin_cos();
    [c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Value of joint `j` equivalent to `v` (mod 2π, within limits) closest to `prev`.
fn closest_equivalent(v: f64, prev: f64, [lo, hi]: [f64; 2]) -> f64 {
    let k = ((prev - v) / (2.0 * PI)).round();
    let mut best = v;
    let mut best_d = f64::INFINITY;
    for dk in [-1.0, 0.0, 1.0] {
        let c = v + (k + dk) * 2.0 * PI;
        if c >= lo - 1e-12 && c <= hi + 1e-12 && (c - prev).abs() < best_d {
            best = c;
            best_d = (c - prev).abs();
        }
    }
    best
}

fn step(a: &Config, b: &Config) -> f64 {
    (0..6).map(|j| (a[j] - b[j]).abs()).fold(0.0, f64::max)
}

/// Collision-check configurations (f64, row-major) on the CPU lanes kernel; returns clearances.
pub fn check_configs(buffers: &ModelBuffers, configs: &[Config]) -> Vec<f32> {
    let n = configs.len();
    if n == 0 {
        return Vec::new();
    }
    let batch = n.div_ceil(LANES) * LANES;
    let mut q = vec![0.0f32; 6 * batch];
    for b in 0..batch {
        let c = &configs[b.min(n - 1)];
        for j in 0..6 {
            q[j * batch + b] = c[j] as f32;
        }
    }
    let mut out = vec![0.0f32; batch];
    collision_min_lanes(buffers, &q, &mut out);
    out.truncate(n);
    out
}

pub fn plan_cartesian(
    model: &RobotModel,
    buffers: &ModelBuffers,
    req: &CartesianRequest,
) -> Result<CartesianResult, CartesianError> {
    let t0 = Instant::now();
    let mut stats = CartesianStats::default();
    let ik = model
        .analytic_ik
        .as_ref()
        .expect("Cartesian planner needs an analytic IK solver");
    let limits = &model.joint_limits;
    let tcf_inv = crate::layout::invert_rigid(&req.tcf);

    // --- Layers of IK solutions -------------------------------------------------------
    let samples = req.free_axis_samples.max(1);
    let angles: Vec<f64> = if samples == 1 {
        vec![0.0]
    } else {
        (0..samples)
            .map(|k| -req.free_axis_range + 2.0 * req.free_axis_range * k as f64 / samples as f64)
            .collect()
    };
    let mut layers: Vec<Vec<Config>> = Vec::with_capacity(req.waypoints.len() + 1);
    if let Some(start) = &req.start {
        layers.push(vec![std::array::from_fn(|j| start[j])]);
    }
    for (w, frame) in req.waypoints.iter().enumerate() {
        let mut nodes = Vec::new();
        for &a in &angles {
            let target = mul(&mul(frame, &rot_z(a)), &tcf_inv);
            nodes.extend(ik.solve(&target, limits));
        }
        if nodes.is_empty() {
            return Err(CartesianError::NoIk { waypoint: w });
        }
        layers.push(nodes);
    }
    stats.ik_nodes = layers.iter().map(Vec::len).sum();
    stats.t_ik = t0.elapsed();

    // --- Node collision checks: every node of every layer in one batch ------------------
    let t = Instant::now();
    let all: Vec<Config> = layers.iter().flatten().copied().collect();
    let clearance = check_configs(buffers, &all);
    let mut offset = 0;
    for (i, layer) in layers.iter_mut().enumerate() {
        let n = layer.len();
        let keep: Vec<Config> = layer
            .iter()
            .zip(&clearance[offset..offset + n])
            .filter(|(_, &c)| c > req.margin)
            .map(|(q, _)| *q)
            .collect();
        offset += n;
        if keep.is_empty() {
            let waypoint = if req.start.is_some() {
                i.saturating_sub(1)
            } else {
                i
            };
            return Err(CartesianError::NoCollisionFreeIk { waypoint });
        }
        *layer = keep;
    }
    stats.free_nodes = layers.iter().map(Vec::len).sum();
    stats.t_node_checks = t.elapsed();

    // --- Lazy DP ----------------------------------------------------------------------
    // Edge cost: sum of absolute joint changes, using the closest 2π-equivalent per joint.
    let edge = |a: &Config, b: &Config| -> Option<(f64, Config)> {
        let mut moved = *b;
        let mut cost = 0.0;
        for j in 0..6 {
            moved[j] = closest_equivalent(b[j], a[j], limits[j]);
            let d = (moved[j] - a[j]).abs();
            if d > req.max_joint_step {
                return None;
            }
            cost += d;
        }
        Some((cost, moved))
    };
    let mut blocked: HashSet<(usize, usize, usize)> = HashSet::new();
    loop {
        stats.search_rounds += 1;
        let ts = Instant::now();
        // best[i][n] = (cost, predecessor, configuration as reached)
        let mut best: Vec<Vec<(f64, usize, Config)>> =
            vec![layers[0].iter().map(|q| (0.0, usize::MAX, *q)).collect()];
        for i in 1..layers.len() {
            let prev = &best[i - 1];
            let mut cur = Vec::with_capacity(layers[i].len());
            for (n, node) in layers[i].iter().enumerate() {
                let mut entry = (f64::INFINITY, usize::MAX, *node);
                for (m, (pc, _, pq)) in prev.iter().enumerate() {
                    if !pc.is_finite() || blocked.contains(&(i, m, n)) {
                        continue;
                    }
                    stats.edges_evaluated += 1;
                    if let Some((c, reached)) = edge(pq, node) {
                        if pc + c < entry.0 {
                            entry = (pc + c, m, reached);
                        }
                    }
                }
                cur.push(entry);
            }
            best.push(cur);
        }
        stats.t_search += ts.elapsed();
        let last = best.last().unwrap();
        let (end, &(cost, _, _)) = match last
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0.is_finite())
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        {
            Some(x) => x,
            None => {
                stats.t_total = t0.elapsed();
                return Err(CartesianError::NoPath { stats });
            }
        };
        // Reconstruct.
        let mut idx = vec![0usize; layers.len()];
        idx[layers.len() - 1] = end;
        for i in (1..layers.len()).rev() {
            idx[i - 1] = best[i][idx[i]].1;
        }
        let path: Vec<Config> = (0..layers.len()).map(|i| best[i][idx[i]].2).collect();

        // Check the motion along every edge of the path, all interpolated configurations in one batch.
        let te = Instant::now();
        let mut interp = Vec::new();
        let mut owner = Vec::new();
        for i in 1..path.len() {
            let steps = (step(&path[i - 1], &path[i]) / req.edge_check_step)
                .ceil()
                .max(1.0) as usize;
            for s in 1..steps {
                let f = s as f64 / steps as f64;
                interp.push(std::array::from_fn(|j| {
                    path[i - 1][j] + f * (path[i][j] - path[i - 1][j])
                }));
                owner.push(i);
            }
        }
        stats.edge_configs_checked += interp.len();
        let clear = check_configs(buffers, &interp);
        let mut bad: Vec<usize> = owner
            .iter()
            .zip(&clear)
            .filter(|(_, &c)| c <= req.margin)
            .map(|(&i, _)| i)
            .collect();
        bad.dedup();
        stats.t_edge_checks += te.elapsed();
        if bad.is_empty() {
            stats.t_total = t0.elapsed();
            return Ok(CartesianResult { path, cost, stats });
        }
        for i in bad {
            blocked.insert((i, idx[i - 1], idx[i]));
        }
    }
}

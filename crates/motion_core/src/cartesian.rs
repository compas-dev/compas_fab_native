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
    pub exact_configs_checked: usize,
    pub exact_rejections: usize,
    pub t_ik: Duration,
    pub t_node_checks: Duration,
    pub t_search: Duration,
    pub t_edge_checks: Duration,
    pub t_exact: Duration,
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
    NoPath { stats: Box<CartesianStats> },
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

/// Collision-check configurations (f64, `buffers.dof` values each) on the CPU lanes kernel;
/// returns clearances.
pub fn check_configs<C: AsRef<[f64]>>(buffers: &ModelBuffers, configs: &[C]) -> Vec<f32> {
    let n = configs.len();
    if n == 0 {
        return Vec::new();
    }
    let dof = buffers.dof;
    let batch = n.div_ceil(LANES) * LANES;
    let mut q = vec![0.0f32; dof * batch];
    for b in 0..batch {
        let c = configs[b.min(n - 1)].as_ref();
        for j in 0..dof {
            q[j * batch + b] = c[j] as f32;
        }
    }
    let mut out = vec![0.0f32; batch];
    collision_min_lanes(buffers, &q, &mut out);
    out.truncate(n);
    out
}

/// Exact collision check used as the final pass: one flag (true = collision) per configuration.
pub type ExactValidator<'a> = &'a (dyn Fn(&[Config]) -> Vec<bool> + Sync);

pub fn plan_cartesian(
    model: &RobotModel,
    buffers: &ModelBuffers,
    req: &CartesianRequest,
) -> Result<CartesianResult, CartesianError> {
    plan_cartesian_with(model, buffers, req, None)
}

/// As [`plan_cartesian`], with an exact validator: once a path passes the sphere checks, its
/// nodes and interpolated edges are checked exactly; failing nodes and edges are blocked and
/// the search is repeated, so the result is collision-free under the exact model too.
pub fn plan_cartesian_with(
    model: &RobotModel,
    buffers: &ModelBuffers,
    req: &CartesianRequest,
    exact: Option<ExactValidator>,
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
    let mut blocked_nodes: HashSet<(usize, usize)> = HashSet::new();
    loop {
        stats.search_rounds += 1;
        let ts = Instant::now();
        // best[i][n] = (cost, predecessor, configuration as reached)
        let mut best: Vec<Vec<(f64, usize, Config)>> = vec![layers[0]
            .iter()
            .enumerate()
            .map(|(n, q)| {
                let cost = if blocked_nodes.contains(&(0, n)) {
                    f64::INFINITY
                } else {
                    0.0
                };
                (cost, usize::MAX, *q)
            })
            .collect()];
        for i in 1..layers.len() {
            let prev = &best[i - 1];
            let mut cur = Vec::with_capacity(layers[i].len());
            for (n, node) in layers[i].iter().enumerate() {
                let mut entry = (f64::INFINITY, usize::MAX, *node);
                if blocked_nodes.contains(&(i, n)) {
                    cur.push(entry);
                    continue;
                }
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
                return Err(CartesianError::NoPath {
                    stats: Box::new(stats),
                });
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
            let Some(exact) = exact else {
                stats.t_total = t0.elapsed();
                return Ok(CartesianResult { path, cost, stats });
            };
            // Final pass on the exact model: path nodes, then the same interpolated edges.
            let tx = Instant::now();
            let mut configs = path.clone();
            configs.extend_from_slice(&interp);
            let hits = exact(&configs);
            stats.exact_configs_checked += configs.len();
            stats.t_exact += tx.elapsed();
            let mut rejected = false;
            for (i, &hit) in hits[..path.len()].iter().enumerate() {
                if hit {
                    blocked_nodes.insert((i, idx[i]));
                    rejected = true;
                }
            }
            for (k, &hit) in hits[path.len()..].iter().enumerate() {
                if hit {
                    let i = owner[k];
                    blocked.insert((i, idx[i - 1], idx[i]));
                    rejected = true;
                }
            }
            if !rejected {
                stats.t_total = t0.elapsed();
                return Ok(CartesianResult { path, cost, stats });
            }
            stats.exact_rejections += 1;
            continue;
        }
        for i in bad {
            blocked.insert((i, idx[i - 1], idx[i]));
        }
    }
}

// ---------------------------------------------------------------------------------------
// Robots without closed-form IK (redundant arms, external axes): follow the path

/// One configuration per waypoint, any number of joints.
#[derive(Clone, Debug)]
pub struct TrackedResult {
    pub path: Vec<Vec<f64>>,
    pub cost: f64,
    pub stats: CartesianStats,
}

/// Exact collision check for configurations of any length: true = collision.
pub type ExactValidatorN<'a> = &'a (dyn Fn(&[Vec<f64>]) -> Vec<bool> + Sync);

/// Configurations strictly between `a` and `b`, no joint moving more than `step` between two.
fn between(a: &[f64], b: &[f64], step: f64) -> Vec<Vec<f64>> {
    let jump = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let n = (jump / step).ceil().max(1.0) as usize;
    (1..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
        })
        .collect()
}

/// Largest absolute joint difference.
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Sum of absolute joint differences (the path cost, as for the closed-form planner).
fn l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Cartesian path by numerical IK, for robots without a closed form: a beam search over
/// the waypoints.
///
/// Every waypoint gets a layer of up to `tracks` configurations. The first layer is the
/// start configuration, or distinct collision-free solutions of the first waypoint
/// (solved from `seeds` first). Each next layer is solved from every configuration of the
/// previous one (which is also the rest posture of that solve, so the joints move as
/// little as possible); with `free_axis` the rotation about the tool axis is left to the
/// solver. When fewer than `tracks` distinct configurations survive, the layer branches:
/// each survivor's parent is perturbed a little and solved again, which finds other
/// postures of a redundant arm (and gets around a joint limit or an obstacle that ended
/// a track). Configurations that collide (sphere model), or are further than
/// `max_joint_step` from every configuration of the previous layer, are dropped; the
/// layer keeps the `tracks` cheapest.
///
/// The path is then the cheapest through the layers (sum of joint changes), found by
/// dynamic programming between any two configurations of consecutive layers within
/// `max_joint_step`. As in the closed-form planner, the motion along the chosen edges is
/// checked lazily (sphere model, then the exact validator), and failing edges and nodes
/// are blocked before searching again.
#[allow(clippy::too_many_arguments)]
pub fn plan_cartesian_tracked(
    model: &RobotModel,
    solver: &crate::numik::Solver,
    buffers: &ModelBuffers,
    req: &CartesianRequest,
    free_axis: bool,
    tracks: usize,
    seeds: &[Vec<f64>],
    ik: &crate::numik::IkSettings,
    exact: Option<ExactValidatorN>,
) -> Result<TrackedResult, CartesianError> {
    use crate::numik::IkGoal;
    use crate::sampling::Rng;
    use rayon::prelude::*;

    let t0 = Instant::now();
    let mut stats = CartesianStats::default();
    if req.waypoints.is_empty() {
        return Err(CartesianError::NoPath {
            stats: Box::new(stats),
        });
    }
    let beam = tracks.max(1);
    let goal = |w: usize| IkGoal {
        target: req.waypoints[w],
        tcf: req.tcf,
        free_axis,
    };
    let clear = |q: &[f64]| check_configs(buffers, &[q])[0] > req.margin;

    // --- First layer ---------------------------------------------------------------------
    let (starts, first) = match &req.start {
        Some(start) => (vec![start.clone()], 0),
        None => {
            let settings = crate::numik::IkSettings {
                max_results: beam,
                ..ik.clone()
            };
            let solutions = solver.solve(&goal(0), seeds, &settings, &clear);
            if solutions.is_empty() {
                let any = crate::numik::IkSettings {
                    max_results: 1,
                    collision_weight: 0.0,
                    ..ik.clone()
                };
                return Err(if solver.solve(&goal(0), &[], &any, &|_| true).is_empty() {
                    CartesianError::NoIk { waypoint: 0 }
                } else {
                    CartesianError::NoCollisionFreeIk { waypoint: 0 }
                });
            }
            (solutions, 1)
        }
    };
    stats.t_ik = t0.elapsed();

    // --- Layers ----------------------------------------------------------------------------
    let tracking = ik.clone();
    let dof = model.dof();
    // Perturbation for branching: 0.15 rad for revolute joints, 2% of the travel for
    // prismatic ones.
    let sigma: Vec<f64> = (0..dof)
        .map(|j| {
            let prismatic = model.links.iter().any(|l| {
                l.q_index == j as i32 && l.joint_type == crate::model::JointType::Prismatic
            });
            let [lo, hi] = model.joint_limits[j];
            if prismatic {
                0.02 * (hi - lo)
            } else {
                0.15
            }
        })
        .collect();
    let mut layers: Vec<Vec<Vec<f64>>> = vec![starts];
    let mut costs: Vec<Vec<f64>> = vec![vec![0.0; layers[0].len()]];
    let ts = Instant::now();
    for w in first..req.waypoints.len() {
        let prev = layers.last().unwrap();
        let prev_cost = costs.last().unwrap();
        let g = goal(w);
        // Children: each parent tracked to this waypoint.
        let solve_all = |seeds: Vec<(Vec<f64>, Vec<f64>)>| -> Vec<Vec<f64>> {
            seeds
                .par_iter()
                .filter_map(|(seed, rest)| solver.solve_from(seed, rest, &g, &tracking))
                .collect()
        };
        let distinct =
            |set: &[Vec<f64>], q: &[f64]| set.iter().all(|f| max_diff(f, q) > ik.distinct);
        // Converged children that are reachable from the previous layer and collision-free.
        let keep = |candidates: Vec<Vec<f64>>, unique: &mut Vec<Vec<f64>>| {
            let fresh: Vec<Vec<f64>> = candidates
                .into_iter()
                .filter(|q| prev.iter().any(|p| max_diff(p, q) <= req.max_joint_step))
                .collect();
            let clearance = check_configs(buffers, &fresh);
            for (q, c) in fresh.into_iter().zip(clearance) {
                if c > req.margin && distinct(unique, &q) {
                    unique.push(q);
                }
            }
        };
        let mut unique: Vec<Vec<f64>> = Vec::new();
        keep(
            solve_all(prev.iter().map(|p| (p.clone(), p.clone())).collect()),
            &mut unique,
        );
        stats.ik_nodes += prev.len();
        // Branch when tracks die and the beam runs short (tracks merging onto the same
        // posture is not a reason to); harder when nothing survived at all.
        if unique.len() < (beam / 4).max(1) {
            let per_parent = if unique.is_empty() { 8 } else { 3 };
            let mut rng = Rng::new(ik.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (w as u64 + 1));
            let mut normal = || {
                let (u, v) = (rng.next_f64().max(1e-12), rng.next_f64());
                (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
            };
            let mut branches = Vec::new();
            for p in prev {
                for _ in 0..per_parent {
                    let seed: Vec<f64> = (0..dof).map(|j| p[j] + sigma[j] * normal()).collect();
                    branches.push((seed.clone(), seed));
                }
            }
            stats.ik_nodes += branches.len();
            keep(solve_all(branches), &mut unique);
        }
        let mut layer: Vec<(f64, Vec<f64>)> = unique
            .into_iter()
            .map(|q| {
                let cost = prev
                    .iter()
                    .zip(prev_cost)
                    .filter(|(p, _)| max_diff(p, &q) <= req.max_joint_step)
                    .map(|(p, c)| c + l1(p, &q))
                    .fold(f64::INFINITY, f64::min);
                (cost, q)
            })
            .collect();
        if layer.is_empty() {
            stats.t_search = ts.elapsed();
            stats.t_total = t0.elapsed();
            // Lost the path here: say whether the waypoint is reachable at all.
            let any = crate::numik::IkSettings {
                max_results: 1,
                collision_weight: 0.0,
                ..ik.clone()
            };
            return Err(if solver.solve(&g, &[], &any, &|_| true).is_empty() {
                CartesianError::NoIk { waypoint: w }
            } else {
                CartesianError::NoPath {
                    stats: Box::new(stats),
                }
            });
        }
        layer.sort_by(|a, b| a.0.total_cmp(&b.0));
        layer.truncate(beam);
        costs.push(layer.iter().map(|(c, _)| *c).collect());
        layers.push(layer.into_iter().map(|(_, q)| q).collect());
    }
    stats.t_search = ts.elapsed();
    stats.free_nodes = layers.iter().map(Vec::len).sum();

    // --- Cheapest path, edges checked lazily ---------------------------------------------
    let mut blocked: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut blocked_nodes: HashSet<(usize, usize)> = HashSet::new();
    loop {
        stats.search_rounds += 1;
        let mut best: Vec<Vec<(f64, usize)>> = vec![layers[0]
            .iter()
            .enumerate()
            .map(|(n, _)| {
                (
                    if blocked_nodes.contains(&(0, n)) {
                        f64::INFINITY
                    } else {
                        0.0
                    },
                    usize::MAX,
                )
            })
            .collect()];
        for i in 1..layers.len() {
            let mut cur = Vec::with_capacity(layers[i].len());
            for (n, q) in layers[i].iter().enumerate() {
                let mut entry = (f64::INFINITY, usize::MAX);
                if !blocked_nodes.contains(&(i, n)) {
                    for (m, p) in layers[i - 1].iter().enumerate() {
                        let pc = best[i - 1][m].0;
                        if !pc.is_finite()
                            || blocked.contains(&(i, m, n))
                            || max_diff(p, q) > req.max_joint_step
                        {
                            continue;
                        }
                        stats.edges_evaluated += 1;
                        let c = pc + l1(p, q);
                        if c < entry.0 {
                            entry = (c, m);
                        }
                    }
                }
                cur.push(entry);
            }
            best.push(cur);
        }
        let last = best.last().unwrap();
        let Some((end, &(cost, _))) = last
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0.is_finite())
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        else {
            stats.t_total = t0.elapsed();
            return Err(CartesianError::NoPath {
                stats: Box::new(stats),
            });
        };
        let mut idx = vec![0usize; layers.len()];
        idx[layers.len() - 1] = end;
        for i in (1..layers.len()).rev() {
            idx[i - 1] = best[i][idx[i]].1;
        }
        let path: Vec<Vec<f64>> = (0..layers.len())
            .map(|i| layers[i][idx[i]].clone())
            .collect();

        // Motion along the edges, sphere model, in one batch.
        let te = Instant::now();
        let mut interp = Vec::new();
        let mut owner = Vec::new();
        for i in 1..path.len() {
            for q in between(&path[i - 1], &path[i], req.edge_check_step) {
                interp.push(q);
                owner.push(i);
            }
        }
        stats.edge_configs_checked += interp.len();
        let clear_edges = check_configs(buffers, &interp);
        let mut bad: Vec<usize> = owner
            .iter()
            .zip(&clear_edges)
            .filter(|(_, &c)| c <= req.margin)
            .map(|(&i, _)| i)
            .collect();
        bad.dedup();
        stats.t_edge_checks += te.elapsed();
        if !bad.is_empty() {
            for i in bad {
                blocked.insert((i, idx[i - 1], idx[i]));
            }
            continue;
        }
        if let Some(exact) = exact {
            let tx = Instant::now();
            let mut configs = path.clone();
            configs.extend(interp.iter().cloned());
            let hits = exact(&configs);
            stats.exact_configs_checked += configs.len();
            stats.t_exact += tx.elapsed();
            let mut rejected = false;
            for (i, &hit) in hits[..path.len()].iter().enumerate() {
                if hit {
                    blocked_nodes.insert((i, idx[i]));
                    rejected = true;
                }
            }
            for (k, &hit) in hits[path.len()..].iter().enumerate() {
                if hit {
                    let i = owner[k];
                    blocked.insert((i, idx[i - 1], idx[i]));
                    rejected = true;
                }
            }
            if rejected {
                stats.exact_rejections += 1;
                continue;
            }
        }
        stats.t_total = t0.elapsed();
        return Ok(TrackedResult { path, cost, stats });
    }
}

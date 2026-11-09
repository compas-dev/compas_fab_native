//! Numerical inverse kinematics for any chain (redundant arms, external axes), as weighted
//! nonlinear least squares solved by Levenberg–Marquardt, in the manner of PyRoki's IK
//! (pose, joint-limit, rest-posture and collision costs in one solve). Unlike PyRoki, the
//! Jacobians are written out rather than taken by automatic differentiation: the geometric
//! Jacobian of the chain, and the distance gradients the sphere model already computes.
//!
//! Residuals, stacked:
//!
//! * **pose** of the tool centre point (TCF): position, and either the full rotation
//!   (log map) or, with a free axis, only the direction of the TCF z axis;
//! * **rest posture**: the distance from the seed, weak, normalised per joint so that
//!   external axes on long rails move freely. Dropped near the target, so it never biases
//!   the final pose;
//! * **collision**: obstacle spheres within `margin` of an obstacle, and self-collision
//!   sphere pairs within `margin` of each other (the same pairs, touch exemptions and ESDF
//!   as the checker). The margin drops to zero near the target.
//!
//! Joint limits are kept by projection (each step is clamped into the limits). Solutions
//! come from several seeds (the caller's first, then random ones within the limits, run in
//! parallel), are deduplicated, and are accepted by a caller predicate (typically the
//! collision checker), so every returned configuration is reachable and acceptable.

#![allow(clippy::needless_range_loop)] // residual rows index several arrays in lock step

use rayon::prelude::*;

use crate::model::{JointType, Mat34, RobotModel, Scene};
use crate::reference::{forward_kinematics, mul, point_box_distance, transform_point};
use crate::sampling::Rng;

type V3 = [f64; 3];
/// Self-collision sphere pairs `(i, j)` between links `a` and `b`: `(a, b, pairs)`.
type SelfGroup = (usize, usize, Vec<(usize, usize)>);

#[derive(Clone, Debug)]
pub struct IkGoal {
    /// Target of the tool centre point (TCF), in robot coordinates.
    pub target: Mat34,
    /// TCF relative to the end-effector link (identity: the link frame itself).
    pub tcf: Mat34,
    /// Leave the rotation about the TCF z axis free (point-axis targets).
    pub free_axis: bool,
}

#[derive(Clone, Debug)]
pub struct IkSettings {
    /// Stop after this many distinct, accepted solutions.
    pub max_results: usize,
    /// Seeds tried at most (the caller's seeds included).
    pub max_seeds: usize,
    /// Offset of the random seed sequence (reproducible results).
    pub seed: u64,
    pub max_iterations: usize,
    /// Convergence: TCF position error, metres.
    pub position_tolerance: f64,
    /// Convergence: TCF rotation error (or z-axis angle with a free axis), radians.
    pub orientation_tolerance: f64,
    /// Metres of position error worth one radian of rotation error.
    pub rotation_weight: f64,
    /// Weight of the rest-posture residual (0 = none).
    pub rest_weight: f64,
    /// Weight of the collision residuals (0 = ignore collisions while solving).
    pub collision_weight: f64,
    /// Clearance the collision residuals keep while the TCF is still far from the target.
    pub collision_margin: f64,
    /// Two solutions are the same if no joint differs by more than this.
    pub distinct: f64,
}

impl Default for IkSettings {
    fn default() -> Self {
        Self {
            max_results: 8,
            max_seeds: 64,
            seed: 0,
            max_iterations: 150,
            position_tolerance: 1e-6,
            orientation_tolerance: 1e-6,
            rotation_weight: 0.5,
            rest_weight: 1e-4,
            collision_weight: 1.0,
            collision_margin: 0.01,
            distinct: 0.05,
        }
    }
}

/// Caller's verdict on a converged configuration (e.g. collision-free on the checker).
pub type Accept<'a> = &'a (dyn Fn(&[f64]) -> bool + Sync);

/// One driven joint on the way from the root to a link.
#[derive(Clone, Copy)]
struct Driver {
    link: usize,
    q: usize,
    prismatic: bool,
}

/// What a solve needs from the model and the scene, precomputed once (keep it while the
/// model and the scene stay the same; see `Solver::with_data`).
pub struct SolverData {
    /// Per link, the driven joints between the root and it.
    drivers: Vec<Vec<Driver>>,
    /// Per obstacle sphere, the boxes it is exempt from (touch links and auto exemptions).
    exempt: Vec<Vec<bool>>,
    /// Self-collision sphere pairs that are checked, grouped by link pair
    /// `(link a, link b, pairs)`, so far-apart links are skipped as a whole.
    self_groups: Vec<SelfGroup>,
    /// Per link: bounding sphere of its self spheres (centre in link coordinates, radius).
    self_bounds: Vec<Option<(V3, f64)>>,
    /// Per box: centre (world) and the radius of its bounding sphere.
    box_bounds: Vec<(V3, f64)>,
    /// Per joint: scale of the rest residual (1 rad for revolute, the travel for prismatic).
    rest_scale: Vec<f64>,
    /// Per joint: prismatic (an external axis or a linear stage).
    prismatic: Vec<bool>,
}

/// Numerical IK on a model and a scene.
pub struct Solver<'a> {
    model: &'a RobotModel,
    scene: &'a Scene,
    data: std::sync::Arc<SolverData>,
}

impl std::ops::Deref for Solver<'_> {
    type Target = SolverData;
    fn deref(&self) -> &SolverData {
        &self.data
    }
}

/// Normal equations of the stacked residuals at one configuration.
struct Eval {
    cost: f64,
    position_error: f64,
    orientation_error: f64,
    /// Collision rows active at margin zero (actual contacts).
    contacts: usize,
    h: Vec<f64>,
    g: Vec<f64>,
}

/// Which residuals are on, decided from the current pose error.
#[derive(Clone, Copy)]
struct Terms {
    rest: f64,
    collision: bool,
    margin: f64,
    /// The carry stage: position only, and only prismatic joints move.
    carry: bool,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn column(m: &Mat34, c: usize) -> V3 {
    [m[c], m[4 + c], m[8 + c]]
}
fn origin(m: &Mat34) -> V3 {
    [m[3], m[7], m[11]]
}

/// Rotation vector (axis * angle) of the rotation taking frame `t`'s axes to frame `c`'s,
/// in world coordinates: log(R_c R_tᵀ).
fn rotation_error(c: &Mat34, t: &Mat34) -> V3 {
    // R = R_c R_tᵀ, row-major 3x3.
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| c[i * 4 + k] * t[j * 4 + k]).sum();
        }
    }
    let cos = ((r[0] + r[4] + r[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    let skew = [r[7] - r[5], r[2] - r[6], r[3] - r[1]];
    if angle < 1e-6 {
        return [skew[0] * 0.5, skew[1] * 0.5, skew[2] * 0.5];
    }
    if std::f64::consts::PI - angle > 1e-4 {
        let s = angle / (2.0 * angle.sin());
        return [skew[0] * s, skew[1] * s, skew[2] * s];
    }
    // Near π: the axis from the diagonal of R = 2 a aᵀ - I, signed by the skew part.
    let d = [r[0], r[4], r[8]];
    let k = (0..3).max_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap();
    let mut a = [0.0; 3];
    a[k] = ((d[k] + 1.0) * 0.5).max(0.0).sqrt();
    for i in 0..3 {
        if i != k {
            a[i] = (r[k * 3 + i] + r[i * 3 + k]) / (4.0 * a[k]);
        }
    }
    let n = norm(a);
    let sign = if dot(a, skew) < 0.0 { -1.0 } else { 1.0 };
    [
        sign * a[0] / n * angle,
        sign * a[1] / n * angle,
        sign * a[2] / n * angle,
    ]
}

/// Solve the dense symmetric system `a x = b` (n x n, row-major) by Gaussian elimination
/// with partial pivoting.
fn solve_dense(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&x, &y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(pivot * n + c, col * n + c);
            }
            b.swap(pivot, col);
        }
        for r in col + 1..n {
            let f = a[r * n + col] / a[col * n + col];
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r * n + c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r * n + r];
    }
    Some(x)
}

impl SolverData {
    pub fn new(model: &RobotModel, scene: &Scene) -> Self {
        let mut drivers: Vec<Vec<Driver>> = Vec::with_capacity(model.links.len());
        for (i, l) in model.links.iter().enumerate() {
            let mut d = if l.parent >= 0 {
                drivers[l.parent as usize].clone()
            } else {
                Vec::new()
            };
            if l.q_index >= 0 && l.joint_type != JointType::Fixed {
                d.push(Driver {
                    link: i,
                    q: l.q_index as usize,
                    prismatic: l.joint_type == JointType::Prismatic,
                });
            }
            drivers.push(d);
        }
        // Per box, the links that may touch it (by index, not by name per query).
        let n_links = model.links.len();
        let touch: Vec<Vec<bool>> = scene
            .cuboids
            .iter()
            .map(|b| {
                model
                    .links
                    .iter()
                    .map(|l| b.touch_links.contains(&l.name))
                    .collect()
            })
            .collect();
        let exempt = model
            .spheres
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (0..scene.cuboids.len())
                    .map(|k| {
                        (k < 32 && scene.sphere_touch.get(i).is_some_and(|m| m & (1 << k) != 0))
                            || touch[k][s.link as usize]
                    })
                    .collect()
            })
            .collect();
        let mut disabled = vec![false; n_links * n_links];
        for &[a, b] in &model.disabled_collisions {
            disabled[a * n_links + b] = true;
            disabled[b * n_links + a] = true;
        }
        // Links moved by the same joints keep their distance: no gradient, skip the pair.
        let rigid = |a: usize, b: usize| {
            drivers[a].len() == drivers[b].len()
                && drivers[a].iter().zip(&drivers[b]).all(|(x, y)| x.q == y.q)
        };
        let ss = &model.self_spheres;
        let mut by_pair: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
            Default::default();
        for i in 0..ss.len() {
            for j in i + 1..ss.len() {
                let (a, b) = (ss[i].link as usize, ss[j].link as usize);
                if a != b && !disabled[a * n_links + b] && !rigid(a, b) {
                    by_pair
                        .entry((a.min(b), a.max(b)))
                        .or_default()
                        .push((i, j));
                }
            }
        }
        let self_groups = by_pair.into_iter().map(|((a, b), p)| (a, b, p)).collect();
        // Bounding sphere per link: centroid of its sphere centres, radius to cover all.
        let self_bounds = (0..n_links)
            .map(|l| {
                let mine: Vec<_> = ss.iter().filter(|s| s.link as usize == l).collect();
                if mine.is_empty() {
                    return None;
                }
                let n = mine.len() as f64;
                let c = mine.iter().fold([0.0; 3], |a, s| {
                    [
                        a[0] + s.center[0] / n,
                        a[1] + s.center[1] / n,
                        a[2] + s.center[2] / n,
                    ]
                });
                let r = mine
                    .iter()
                    .map(|s| norm(sub(s.center, c)) + s.radius)
                    .fold(0.0, f64::max);
                Some((c, r))
            })
            .collect();
        let box_bounds = scene
            .cuboids
            .iter()
            .map(|b| (origin(&b.frame), norm(b.half_extents)))
            .collect();
        let mut rest_scale = vec![1.0; model.dof()];
        let mut prismatic = vec![false; model.dof()];
        for l in &model.links {
            if l.q_index >= 0 && l.joint_type == JointType::Prismatic {
                let j = l.q_index as usize;
                let [lo, hi] = model.joint_limits[j];
                rest_scale[j] = (hi - lo).max(0.1);
                prismatic[j] = true;
            }
        }
        Self {
            drivers,
            exempt,
            self_groups,
            self_bounds,
            box_bounds,
            rest_scale,
            prismatic,
        }
    }
}

impl<'a> Solver<'a> {
    pub fn new(model: &'a RobotModel, scene: &'a Scene) -> Self {
        Self::with_data(
            model,
            scene,
            std::sync::Arc::new(SolverData::new(model, scene)),
        )
    }

    /// A solver on precomputed data, which must have been built from this model and scene.
    pub fn with_data(
        model: &'a RobotModel,
        scene: &'a Scene,
        data: std::sync::Arc<SolverData>,
    ) -> Self {
        Self { model, scene, data }
    }

    fn clamp(&self, q: &mut [f64]) {
        for (v, [lo, hi]) in q.iter_mut().zip(&self.model.joint_limits) {
            *v = v.clamp(*lo, *hi);
        }
    }

    /// World axis of a driven joint (the axis is fixed in its child link's frame).
    fn joint_axis(&self, frames: &[Mat34], d: &Driver) -> V3 {
        let f = &frames[d.link];
        let a = self.model.links[d.link].axis;
        [
            f[0] * a[0] + f[1] * a[1] + f[2] * a[2],
            f[4] * a[0] + f[5] * a[1] + f[6] * a[2],
            f[8] * a[0] + f[9] * a[1] + f[10] * a[2],
        ]
    }

    /// Adds `s * dir · d(p)/dq` to `row`, for the point `p` (world) fixed to `link`.
    fn point_jacobian(
        &self,
        frames: &[Mat34],
        link: usize,
        p: V3,
        dir: V3,
        s: f64,
        row: &mut [f64],
    ) {
        for d in &self.drivers[link] {
            let axis = self.joint_axis(frames, d);
            let v = if d.prismatic {
                axis
            } else {
                cross(axis, sub(p, origin(&frames[d.link])))
            };
            row[d.q] += s * dot(dir, v);
        }
    }

    fn evaluate(
        &self,
        q: &[f64],
        goal: &IkGoal,
        rest: &[f64],
        terms: Terms,
        settings: &IkSettings,
    ) -> Eval {
        let n = q.len();
        let frames = forward_kinematics(self.model, q);
        let ee = self.model.end_effector_link;
        let tcf = mul(&frames[ee], &goal.tcf);
        let mut acc = Normal {
            n,
            h: vec![0.0; n * n],
            g: vec![0.0; n],
            cost: 0.0,
            row: vec![0.0; n],
            mask: (0..n).map(|j| !terms.carry || self.prismatic[j]).collect(),
        };

        // Pose: the TCF origin.
        let p = origin(&tcf);
        let dp = sub(p, origin(&goal.target));
        for c in 0..3 {
            let mut e = [0.0; 3];
            e[c] = 1.0;
            self.point_jacobian(&frames, ee, p, e, 1.0, &mut acc.row);
            acc.add(dp[c]);
        }
        let position_error = norm(dp);

        // Pose: rotation (skipped while carrying).
        let w = settings.rotation_weight;
        let orientation_error = if goal.free_axis {
            let zc = column(&tcf, 2);
            let zt = column(&goal.target, 2);
            if !terms.carry {
                // r = w (z_c - z_t); dz_c/dq = axis × z_c for revolute joints.
                let dz = sub(zc, zt);
                for c in 0..3 {
                    for d in &self.drivers[ee] {
                        if !d.prismatic {
                            acc.row[d.q] += w * cross(self.joint_axis(&frames, d), zc)[c];
                        }
                    }
                    acc.add(w * dz[c]);
                }
            }
            dot(zc, zt).clamp(-1.0, 1.0).acos()
        } else {
            let e = rotation_error(&tcf, &goal.target);
            if !terms.carry {
                for c in 0..3 {
                    for d in &self.drivers[ee] {
                        if !d.prismatic {
                            acc.row[d.q] += w * self.joint_axis(&frames, d)[c];
                        }
                    }
                    acc.add(w * e[c]);
                }
            }
            norm(e)
        };

        // Rest posture.
        if terms.rest > 0.0 {
            let s = terms.rest.sqrt();
            for j in 0..n {
                acc.row[j] = s / self.rest_scale[j];
                acc.add(s * (q[j] - rest[j]) / self.rest_scale[j]);
            }
        }

        // Collisions: r = wc (margin - d) for every distance d below the margin.
        let mut contacts = 0;
        if terms.collision && settings.collision_weight > 0.0 {
            let wc = settings.collision_weight.sqrt();
            for (i, sph) in self.model.spheres.iter().enumerate() {
                let link = sph.link as usize;
                if self.drivers[link].is_empty() {
                    continue; // nothing moves it
                }
                let c = transform_point(&frames[link], sph.center);
                let mut best = (f64::INFINITY, [0.0; 3]);
                for (k, b) in self.scene.cuboids.iter().enumerate() {
                    // Broad phase: a box this sphere cannot come within the margin of.
                    let (bc, br) = self.box_bounds[k];
                    if norm(sub(c, bc)) - br - sph.radius >= terms.margin {
                        continue;
                    }
                    if !self.exempt[i][k] {
                        let (d, grad) = point_box_distance(c, &b.frame, b.half_extents);
                        if d < best.0 {
                            best = (d, grad);
                        }
                    }
                }
                if let Some(esdf) = &self.scene.esdf {
                    let (d, grad) = esdf.sample([c[0] as f32, c[1] as f32, c[2] as f32]);
                    let d = (d - self.scene.esdf_margin) as f64;
                    if d < best.0 {
                        best = (d, grad.map(|v| v as f64));
                    }
                }
                let d = best.0 - sph.radius;
                if d < terms.margin {
                    contacts += (d < 0.0) as usize;
                    self.point_jacobian(&frames, link, c, best.1, -wc, &mut acc.row);
                    acc.add(wc * (terms.margin - d));
                }
            }
            let ss = &self.model.self_spheres;
            let world: Vec<V3> = ss
                .iter()
                .map(|s| transform_point(&frames[s.link as usize], s.center))
                .collect();
            let pairs = self.self_groups.iter().filter(|(a, b, _)| {
                // Broad phase: links whose bounding spheres stay apart by the margin.
                match (self.self_bounds[*a], self.self_bounds[*b]) {
                    (Some((ca, ra)), Some((cb, rb))) => {
                        let d = norm(sub(
                            transform_point(&frames[*a], ca),
                            transform_point(&frames[*b], cb),
                        ));
                        d - ra - rb < terms.margin
                    }
                    _ => false,
                }
            });
            for &(i, j) in pairs.flat_map(|(_, _, p)| p) {
                let v = sub(world[i], world[j]);
                let dist = norm(v);
                let d = dist - ss[i].radius - ss[j].radius;
                if d >= terms.margin {
                    continue;
                }
                contacts += (d < 0.0) as usize;
                let u = if dist > 1e-12 {
                    [v[0] / dist, v[1] / dist, v[2] / dist]
                } else {
                    [0.0, 0.0, 1.0]
                };
                // dd/dq = u · (dc_i/dq - dc_j/dq)
                self.point_jacobian(&frames, ss[i].link as usize, world[i], u, -wc, &mut acc.row);
                self.point_jacobian(&frames, ss[j].link as usize, world[j], u, wc, &mut acc.row);
                acc.add(wc * (terms.margin - d));
            }
        }
        Eval {
            cost: acc.cost,
            position_error,
            orientation_error,
            contacts,
            h: acc.h,
            g: acc.g,
        }
    }

    fn converged(&self, e: &Eval, s: &IkSettings) -> bool {
        e.position_error <= s.position_tolerance && e.orientation_error <= s.orientation_tolerance
    }

    /// Residual terms for the current state of a solve.
    fn terms(&self, e: Option<&Eval>, s: &IkSettings) -> Terms {
        // Near the target: no rest pull, and collisions only where there is contact.
        let near = e.is_some_and(|e| e.position_error < 1e-3 && e.orientation_error < 1e-2);
        // Far from the target, collision terms would only slow the solve down.
        let close = e.is_some_and(|e| e.position_error < 0.05 && e.orientation_error < 0.3);
        // The rest pull only steers the redundant directions, where the pose cost is flat;
        // close to the target it would hold the pose off it.
        let settled = e.is_some_and(|e| e.position_error < 0.01 && e.orientation_error < 0.05);
        Terms {
            rest: if settled { 0.0 } else { s.rest_weight },
            collision: close || near,
            margin: if near { 0.0 } else { s.collision_margin },
            carry: false,
        }
    }

    /// Levenberg–Marquardt on the stacked residuals from `seed`, pulled towards `rest`.
    /// Returns the configuration if the pose converged without contacts.
    pub fn solve_from(
        &self,
        seed: &[f64],
        rest: &[f64],
        goal: &IkGoal,
        s: &IkSettings,
    ) -> Option<Vec<f64>> {
        let n = seed.len();
        let mut q = seed.to_vec();
        self.clamp(&mut q);
        if self.prismatic.iter().any(|&p| p) {
            // Carry stage: move only the external axes, so the TCF reaches the target
            // position with the seed's arm posture. A random seed on a long rail is
            // otherwise metres away and the arm stretches for it.
            let terms = Terms {
                rest: 0.0,
                collision: false,
                margin: 0.0,
                carry: true,
            };
            for _ in 0..4 {
                let e = self.evaluate(&q, goal, rest, terms, s);
                if e.position_error < 1e-3 {
                    break;
                }
                let mut h = e.h.clone();
                for j in 0..n {
                    h[j * n + j] += if self.prismatic[j] { 1e-9 } else { 1.0 };
                }
                let Some(dq) = solve_dense(h, e.g.iter().map(|v| -v).collect(), n) else {
                    break;
                };
                for j in 0..n {
                    if self.prismatic[j] {
                        q[j] += dq[j];
                    }
                }
                self.clamp(&mut q);
            }
        }
        let mut lambda = 1e-3;
        let mut terms = self.terms(None, s);
        let mut cur = self.evaluate(&q, goal, rest, terms, s);
        for _ in 0..s.max_iterations {
            let next_terms = self.terms(Some(&cur), s);
            if next_terms.rest != terms.rest
                || next_terms.collision != terms.collision
                || next_terms.margin != terms.margin
            {
                terms = next_terms;
                cur = self.evaluate(&q, goal, rest, terms, s);
            }
            if self.converged(&cur, s) && cur.contacts == 0 {
                return Some(q);
            }
            let mut h = cur.h.clone();
            for j in 0..n {
                h[j * n + j] += lambda * (1.0 + h[j * n + j]);
            }
            let Some(dq) = solve_dense(h, cur.g.iter().map(|v| -v).collect(), n) else {
                lambda *= 10.0;
                continue;
            };
            let mut trial: Vec<f64> = q.iter().zip(&dq).map(|(a, b)| a + b).collect();
            self.clamp(&mut trial);
            let next = self.evaluate(&trial, goal, rest, terms, s);
            if next.cost < cur.cost {
                q = trial;
                cur = next;
                lambda = (lambda * 0.3).max(1e-10);
            } else {
                lambda *= 10.0;
                if lambda > 1e10 {
                    break;
                }
            }
        }
        (self.converged(&cur, s) && cur.contacts == 0).then_some(q)
    }

    /// Distinct accepted solutions: from `seeds` first (each also the rest posture of its
    /// own solve), then random seeds within the limits, in parallel batches.
    pub fn solve(
        &self,
        goal: &IkGoal,
        seeds: &[Vec<f64>],
        s: &IkSettings,
        accept: Accept,
    ) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(s.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED);
        let mut candidates: Vec<Vec<f64>> = seeds.iter().take(s.max_seeds).cloned().collect();
        while candidates.len() < s.max_seeds {
            candidates.push(
                self.model
                    .joint_limits
                    .iter()
                    .map(|[lo, hi]| lo + (hi - lo) * rng.next_f64())
                    .collect(),
            );
        }
        // The caller's seeds form batches of their own: when they are enough (tracking a
        // path from the previous configuration, say), no random seed is solved at all.
        let batch = rayon::current_num_threads().clamp(4, 16);
        let given = seeds.len().min(s.max_seeds);
        let chunks = candidates[..given]
            .chunks(batch)
            .chain(candidates[given..].chunks(batch));
        let mut found: Vec<Vec<f64>> = Vec::new();
        for chunk in chunks {
            let solved: Vec<Option<Vec<f64>>> = chunk
                .par_iter()
                .map(|seed| self.solve_from(seed, seed, goal, s).filter(|q| accept(q)))
                .collect();
            for q in solved.into_iter().flatten() {
                let distinct = found.iter().all(|f| {
                    f.iter()
                        .zip(&q)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                        > s.distinct
                });
                if distinct {
                    found.push(q);
                    if found.len() >= s.max_results {
                        return found;
                    }
                }
            }
        }
        found
    }
}

/// Accumulates JᵀJ, Jᵀr and Σr² one residual row at a time.
struct Normal {
    n: usize,
    h: Vec<f64>,
    g: Vec<f64>,
    cost: f64,
    /// The row being built (dr/dq), cleared by `add`.
    row: Vec<f64>,
    /// Joints that may move; the others get no Jacobian entries.
    mask: Vec<bool>,
}

impl Normal {
    fn add(&mut self, r: f64) {
        let n = self.n;
        for i in 0..n {
            if !self.mask[i] {
                self.row[i] = 0.0;
            }
        }
        for i in 0..n {
            let a = self.row[i];
            if a == 0.0 {
                continue;
            }
            self.g[i] += a * r;
            for j in 0..n {
                self.h[i * n + j] += a * self.row[j];
            }
        }
        self.cost += r * r;
        self.row.iter_mut().for_each(|v| *v = 0.0);
    }
}

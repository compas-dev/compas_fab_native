//! Time parameterization of joint-space paths under joint velocity and acceleration limits.
//!
//! 1. The path: straight segments between the vertices, with a circular blend at each
//!    interior vertex that deviates at most `blend_tolerance` from it (after Kunz and
//!    Stilman, "Time-Optimal Trajectory Generation for Path Following with Bounded
//!    Acceleration and Velocity", RSS 2012). Where a blend is not allowed (zero tolerance,
//!    a reversal, or the caller's collision check rejects it) the robot stops at the vertex.
//! 2. The timing: TOPP-RA (Pham and Pham, "A New Approach to Time-Optimal Path
//!    Parameterization Based on Reachability Analysis", IEEE T-RO 2018) on a grid along the
//!    path: a backward pass computes the controllable path-velocity intervals, a forward pass
//!    takes the largest admissible path acceleration at each step. Start and end at rest.
//!
//! Implemented from the papers (see `native/NOTICE`). Acceleration is piecewise constant
//! along the grid: jerk is not limited.

/// Limits and settings. `max_velocity` and `max_acceleration` are per joint (rad/s, rad/s²
/// or m/s, m/s² for prismatic joints).
#[derive(Clone, Debug)]
pub struct TimingSettings {
    pub max_velocity: Vec<f64>,
    pub max_acceleration: Vec<f64>,
    /// Largest joint-space distance between a blend and the vertex it replaces.
    pub blend_tolerance: f64,
    /// Path parameter step of the TOPP-RA grid (joint-space distance).
    pub grid_step: f64,
    /// Sampling period of the output.
    pub time_step: f64,
}

/// A timed trajectory sampled every `time_step` (and at the end).
#[derive(Clone, Debug, Default)]
pub struct Timed {
    pub times: Vec<f64>,
    pub positions: Vec<Vec<f64>>,
    pub velocities: Vec<Vec<f64>>,
    pub accelerations: Vec<Vec<f64>>,
    pub duration: f64,
    /// Blend tolerance used at each interior vertex (0 = the robot stops there).
    pub blends: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TimingError {
    /// Fewer than two vertices, or a limit that is not positive.
    InvalidInput(String),
    /// No admissible velocity profile (should not happen with positive limits).
    Infeasible(usize),
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One piece of the path, parameterized by arc length `s` in [0, length].
#[derive(Clone, Debug)]
enum Piece {
    Line {
        start: Vec<f64>,
        dir: Vec<f64>,
        length: f64,
    },
    Arc {
        center: Vec<f64>,
        x: Vec<f64>,
        y: Vec<f64>,
        radius: f64,
        angle: f64,
        /// Interior vertex the blend replaces (index into the vertices).
        vertex: usize,
    },
}

impl Piece {
    fn length(&self) -> f64 {
        match self {
            Piece::Line { length, .. } => *length,
            Piece::Arc { radius, angle, .. } => radius * angle,
        }
    }

    /// q(s), q'(s), q''(s).
    fn eval(&self, s: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        match self {
            Piece::Line { start, dir, .. } => (
                start.iter().zip(dir).map(|(p, d)| p + d * s).collect(),
                dir.clone(),
                vec![0.0; dir.len()],
            ),
            Piece::Arc {
                center,
                x,
                y,
                radius,
                ..
            } => {
                // Rotation from x (towards y) by s / radius.
                let (st, ct) = (s / radius).sin_cos();
                let n = center.len();
                (
                    (0..n)
                        .map(|j| center[j] + radius * (ct * x[j] + st * y[j]))
                        .collect(),
                    (0..n).map(|j| -st * x[j] + ct * y[j]).collect(),
                    (0..n).map(|j| -(ct * x[j] + st * y[j]) / radius).collect(),
                )
            }
        }
    }
}

/// The blended path: pieces, and whether the robot must stop at the start of each piece.
struct Path {
    pieces: Vec<Piece>,
    stop_before: Vec<bool>,
}

/// Lines between vertices with circular blends of at most `tolerances[i]` deviation at
/// interior vertex `i + 1`.
fn build_path(vertices: &[Vec<f64>], tolerances: &[f64]) -> Path {
    let n = vertices.len();
    let seg: Vec<Vec<f64>> = (0..n - 1)
        .map(|i| sub(&vertices[i + 1], &vertices[i]))
        .collect();
    let len: Vec<f64> = seg.iter().map(|d| norm(d)).collect();
    let dir: Vec<Vec<f64>> = seg
        .iter()
        .zip(&len)
        .map(|(d, l)| d.iter().map(|x| x / l).collect())
        .collect();
    // Blend length (distance from the vertex where the arc starts) per interior vertex.
    let mut blend = vec![0.0; n];
    let mut arcs: Vec<Option<Piece>> = vec![None; n];
    for i in 1..n - 1 {
        let (a, b) = (&dir[i - 1], &dir[i]);
        let cos = dot(a, b).clamp(-1.0, 1.0);
        let angle = cos.acos(); // turning angle
        let tol = tolerances[i - 1];
        if angle < 1e-9 || tol <= 0.0 || angle > std::f64::consts::PI - 1e-6 {
            continue;
        }
        let half = angle / 2.0;
        // Kunz & Stilman: deviation delta gives blend length l = delta sin(a/2) / (1 - cos(a/2)).
        let from_tol = tol * half.sin() / (1.0 - half.cos());
        let l = from_tol.min(len[i - 1] / 2.0).min(len[i] / 2.0);
        if l < 1e-9 {
            continue;
        }
        let radius = l / half.tan();
        // Arc from vertex - l*a to vertex + l*b; center along the bisector.
        let x: Vec<f64> = (0..a.len()).map(|j| -b[j] + a[j]).collect();
        let xn = norm(&x);
        let bis: Vec<f64> = x.iter().map(|v| v / xn).collect(); // points from vertex away from center
        let center: Vec<f64> = (0..a.len())
            .map(|j| vertices[i][j] - bis[j] * radius / half.cos())
            .collect();
        let start: Vec<f64> = (0..a.len()).map(|j| vertices[i][j] - l * a[j]).collect();
        let ux: Vec<f64> = (0..a.len())
            .map(|j| (start[j] - center[j]) / radius)
            .collect();
        // y: the direction of motion at the arc start (= a).
        arcs[i] = Some(Piece::Arc {
            center,
            x: ux,
            y: a.clone(),
            radius,
            angle,
            vertex: i,
        });
        blend[i] = l;
    }
    let mut pieces = Vec::new();
    let mut stop_before = Vec::new();
    for i in 0..n - 1 {
        let start: Vec<f64> = (0..dir[i].len())
            .map(|j| vertices[i][j] + blend[i] * dir[i][j])
            .collect();
        let length = len[i] - blend[i] - blend[i + 1];
        // A corner without a blend (or the path start) is a stop.
        let stop = i == 0 || blend[i] == 0.0;
        if length > 1e-12 {
            pieces.push(Piece::Line {
                start,
                dir: dir[i].clone(),
                length,
            });
            stop_before.push(stop);
        }
        if let Some(arc) = arcs[i + 1].take() {
            pieces.push(arc);
            stop_before.push(false);
        }
    }
    Path {
        pieces,
        stop_before,
    }
}

/// Grid point: path parameter, piece index, local parameter, and q', q'' there.
struct Node {
    s: f64,
    piece: usize,
    local: f64,
    /// q' and q'' at the end of the step that starts here, on this node's piece (the path's
    /// curvature jumps where a blend meets a line, so the next node's values differ).
    dq_end: Vec<f64>,
    ddq_end: Vec<f64>,
    dq: Vec<f64>,
    ddq: Vec<f64>,
    /// Upper bound on x = sdot² from velocity limits (and 0 at stops).
    x_max: f64,
}

/// Constraints of the grid step starting at `node` (length `ds`), a x + b u <= c in
/// (x, u): the acceleration limits at the step's start, and at its end with
/// x_next = x + 2 ds u (TOPP-RA's "interpolation" discretization, tighter than enforcing them
/// at the start only), plus 0 <= x <= x_max.
fn step_constraints(node: &Node, ds: f64, s: &TimingSettings) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(4 * node.dq.len() + 2);
    for j in 0..node.dq.len() {
        let a = s.max_acceleration[j];
        // -a <= q''(s_i) x + q'(s_i) u <= a
        let (d1, d2) = (node.dq[j], node.ddq[j]);
        out.push([d2, d1, a]);
        out.push([-d2, -d1, a]);
        // -a <= q''(s_i+1) (x + 2 ds u) + q'(s_i+1) u <= a
        let (n1, n2) = (node.dq_end[j], node.ddq_end[j]);
        out.push([n2, 2.0 * ds * n2 + n1, a]);
        out.push([-n2, -(2.0 * ds * n2 + n1), a]);
    }
    out.push([-1.0, 0.0, 0.0]); // x >= 0
    out.push([1.0, 0.0, node.x_max]); // x <= x_max
    out
}

/// Extreme x (max if `maximize`) over {(x, u) : rows}, a 2-variable LP solved by checking
/// every vertex of the feasible polygon. None if empty.
fn lp_x(rows: &[[f64; 3]], maximize: bool) -> Option<f64> {
    let eps = 1e-9;
    let feasible = |x: f64, u: f64| {
        rows.iter()
            .all(|r| r[0] * x + r[1] * u <= r[2] + eps * (1.0 + r[2].abs()))
    };
    let mut best: Option<f64> = None;
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            let (a, b) = (rows[i], rows[j]);
            let det = a[0] * b[1] - a[1] * b[0];
            if det.abs() < 1e-14 {
                continue;
            }
            let x = (a[2] * b[1] - a[1] * b[2]) / det;
            let u = (a[0] * b[2] - a[2] * b[0]) / det;
            if feasible(x, u) {
                best = Some(match best {
                    None => x,
                    Some(v) if maximize => v.max(x),
                    Some(v) => v.min(x),
                });
            }
        }
    }
    best
}

/// Time-parameterize the path through `vertices`. `blend_free(a, b)` must say whether the
/// straight joint-space motion between two nearby configurations is collision-free; it is
/// used to check blends, which leave the original path (halving a blend's tolerance until it
/// is free, down to a stop).
pub fn time_parameterize(
    vertices: &[Vec<f64>],
    settings: &TimingSettings,
    blend_free: impl Fn(&[f64], &[f64]) -> bool,
) -> Result<Timed, TimingError> {
    if vertices.len() < 2 {
        return Err(TimingError::InvalidInput("fewer than two vertices".into()));
    }
    let dof = vertices[0].len();
    if settings.max_velocity.len() != dof
        || settings.max_acceleration.len() != dof
        || settings
            .max_velocity
            .iter()
            .chain(&settings.max_acceleration)
            .any(|&v| v <= 0.0 || !v.is_finite())
    {
        return Err(TimingError::InvalidInput(
            "limits must be positive and one per joint".into(),
        ));
    }
    // Drop repeated vertices.
    let mut verts: Vec<Vec<f64>> = vec![vertices[0].clone()];
    for v in &vertices[1..] {
        if norm(&sub(v, verts.last().unwrap())) > 1e-12 {
            verts.push(v.clone());
        }
    }
    if verts.len() < 2 {
        return Ok(Timed {
            times: vec![0.0],
            positions: vec![verts[0].clone()],
            velocities: vec![vec![0.0; dof]],
            accelerations: vec![vec![0.0; dof]],
            duration: 0.0,
            blends: Vec::new(),
        });
    }

    // Blends, each checked along its arc; halve a rejected blend's tolerance (5 times, then 0).
    let mut tolerances = vec![settings.blend_tolerance; verts.len() - 2];
    let path = loop {
        let path = build_path(&verts, &tolerances);
        let rejected = path.pieces.iter().find_map(|piece| match piece {
            Piece::Arc { vertex, .. } => {
                let n = (piece.length() / 0.01).ceil().max(2.0) as usize;
                let pts: Vec<Vec<f64>> = (0..=n)
                    .map(|k| piece.eval(piece.length() * k as f64 / n as f64).0)
                    .collect();
                pts.windows(2)
                    .any(|w| !blend_free(&w[0], &w[1]))
                    .then_some(vertex - 1)
            }
            Piece::Line { .. } => None,
        });
        match rejected {
            None => break path,
            Some(c) => {
                tolerances[c] = if tolerances[c] < settings.blend_tolerance / 16.0 {
                    0.0
                } else {
                    tolerances[c] / 2.0
                };
            }
        }
    };

    // Grid along the path: every piece boundary plus steps of at most `grid_step`.
    let mut nodes: Vec<Node> = Vec::new();
    let mut s0 = 0.0;
    // Velocity limits: |q'_j| sdot <= v_j. (|q'| = 1 along the path, so this is finite.)
    let x_cap = |dq: &[f64]| -> f64 {
        (0..dof)
            .filter(|&j| dq[j].abs() > 1e-12)
            .map(|j| (settings.max_velocity[j] / dq[j].abs()).powi(2))
            .fold(f64::INFINITY, f64::min)
    };
    for (k, piece) in path.pieces.iter().enumerate() {
        let length = piece.length();
        // On a blend, also at most 0.02 rad of turn per step: the limits are enforced at the
        // grid points, and the acceleration direction rotates between them.
        let step = match piece {
            Piece::Arc { radius, .. } => settings.grid_step.min(0.02 * radius),
            Piece::Line { .. } => settings.grid_step,
        };
        let n = (length / step).ceil().max(1.0) as usize;
        for i in 0..n {
            let local = length * i as f64 / n as f64;
            let (_, dq, ddq) = piece.eval(local);
            let (_, dq_end, ddq_end) = piece.eval(length * (i + 1) as f64 / n as f64);
            let mut x_max = x_cap(&dq);
            if i == 0 && path.stop_before[k] {
                x_max = 0.0;
            }
            nodes.push(Node {
                s: s0 + local,
                piece: k,
                local,
                dq_end,
                ddq_end,
                dq,
                ddq,
                x_max,
            });
        }
        s0 += length;
    }
    let last = path.pieces.last().unwrap();
    let (_, dq, ddq) = last.eval(last.length());
    nodes.push(Node {
        s: s0,
        piece: path.pieces.len() - 1,
        local: last.length(),
        dq_end: dq.clone(),
        ddq_end: ddq.clone(),
        dq,
        ddq,
        x_max: 0.0, // end at rest
    });

    // Backward pass: controllable intervals K_i = [lo, hi] of x = sdot².
    let m = nodes.len();
    let mut k_hi = vec![0.0f64; m];
    let mut k_lo = vec![0.0f64; m];
    for i in (0..m - 1).rev() {
        let ds = nodes[i + 1].s - nodes[i].s;
        let mut rows = step_constraints(&nodes[i], ds, settings);
        // x + 2 ds u in K_{i+1}
        rows.push([1.0, 2.0 * ds, k_hi[i + 1]]);
        rows.push([-1.0, -2.0 * ds, -k_lo[i + 1]]);
        let hi = lp_x(&rows, true).ok_or(TimingError::Infeasible(i))?;
        let lo = lp_x(&rows, false).ok_or(TimingError::Infeasible(i))?;
        k_hi[i] = hi.max(0.0);
        k_lo[i] = lo.max(0.0);
    }

    // Forward pass: from rest, the largest admissible u at each step.
    let mut x = vec![0.0f64; m];
    let mut u = vec![0.0f64; m - 1];
    for i in 0..m - 1 {
        let ds = nodes[i + 1].s - nodes[i].s;
        let xi = x[i].min(k_hi[i]);
        // Each row a x + b u <= c bounds u at x = xi.
        let (mut u_lo, mut u_hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for r in step_constraints(&nodes[i], ds, settings) {
            let rest = r[2] - r[0] * xi;
            if r[1] > 1e-12 {
                u_hi = u_hi.min(rest / r[1]);
            } else if r[1] < -1e-12 {
                u_lo = u_lo.max(rest / r[1]);
            }
        }
        u_hi = u_hi.min((k_hi[i + 1] - xi) / (2.0 * ds));
        u_lo = u_lo.max((k_lo[i + 1] - xi) / (2.0 * ds));
        // Rounding can leave a sliver of infeasibility; prefer staying controllable.
        let ui = if u_hi >= u_lo { u_hi } else { u_lo };
        u[i] = ui;
        x[i + 1] = (xi + 2.0 * ds * ui).max(0.0);
        x[i] = xi;
    }

    // Times at the grid points.
    let mut t = vec![0.0; m];
    for i in 0..m - 1 {
        let ds = nodes[i + 1].s - nodes[i].s;
        let v = x[i].sqrt() + x[i + 1].sqrt();
        t[i + 1] = t[i] + if v > 0.0 { 2.0 * ds / v } else { 0.0 };
    }
    let duration = t[m - 1];

    // Sample: within a grid step, sdot changes linearly in time (constant u).
    let mut out = Timed {
        duration,
        blends: tolerances,
        ..Default::default()
    };
    let n_samples = (duration / settings.time_step).ceil() as usize;
    let mut i = 0;
    for k in 0..=n_samples {
        let time = (k as f64 * settings.time_step).min(duration);
        while i < m - 2 && t[i + 1] < time {
            i += 1;
        }
        let tau = (time - t[i]).max(0.0);
        let sd0 = x[i].sqrt();
        let ds = nodes[i + 1].s - nodes[i].s;
        let local = (sd0 * tau + 0.5 * u[i] * tau * tau).clamp(0.0, ds);
        let sdot = (sd0 + u[i] * tau).max(0.0);
        // The exact path point (grid nodes never straddle two pieces).
        let (q, dq, ddq) = path.pieces[nodes[i].piece].eval(nodes[i].local + local);
        out.times.push(time);
        out.velocities.push(dq.iter().map(|d| d * sdot).collect());
        out.accelerations.push(
            (0..dof)
                .map(|j| ddq[j] * sdot * sdot + dq[j] * u[i])
                .collect(),
        );
        out.positions.push(q);
        if time >= duration {
            break;
        }
    }
    // Exact endpoints, at rest.
    let last = out.positions.len() - 1;
    out.positions[0] = verts[0].clone();
    out.positions[last] = verts[verts.len() - 1].clone();
    out.velocities[0] = vec![0.0; dof];
    out.velocities[last] = vec![0.0; dof];
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_join_in_position_and_direction() {
        let verts = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, -0.5, 0.3],
            vec![1.2, 0.4, -0.6],
            vec![0.2, 0.9, -0.2],
        ];
        let path = build_path(&verts, &[0.1, 0.1]);
        for w in path.pieces.windows(2) {
            let (q0, d0, _) = w[0].eval(w[0].length());
            let (q1, d1, _) = w[1].eval(0.0);
            let gap = norm(&sub(&q0, &q1));
            let turn = norm(&sub(&d0, &d1));
            assert!(gap < 1e-9 && turn < 1e-9, "gap {gap}, turn {turn}");
        }
    }
}

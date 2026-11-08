//! Closed-form inverse kinematics, all solutions per target.
//!
//! Solvers work in the robot model's own terms: the target is the world frame of the
//! end-effector link (e.g. `tool0`) and the returned joint values are model joint values,
//! wrapped into the joint limits. Every solution is checked by the model's FK in tests.
//!
//! `offset_wrist_ik` and `offset_wrist_fk` are a port of `ur_kin.cpp` from ROS-Industrial's
//! `ur_kinematics` (https://github.com/ros-industrial/universal_robot), whose license
//! requires this notice:
//!
//!   Copyright (c) 2013, Georgia Institute of Technology. All rights reserved.
//!   Author: Kelsey Hawkins. Software License Agreement (BSD License): redistribution and
//!   use in source and binary forms, with or without modification, are permitted provided
//!   that the conditions in `native/licenses/ur_kinematics-BSD-3-Clause.txt` are met.
//!   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//!   EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED (see that file for the full text).

#![allow(clippy::needless_range_loop)] // the closed form indexes several arrays in lock step

use std::f64::consts::PI;

use crate::layout::invert_rigid;
use crate::model::Mat34;
use crate::reference::mul;

/// Description of a closed-form solver, as exported with the robot fixture.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // one solver per robot, never in bulk
pub enum AnalyticIk {
    /// Offset-wrist 6R arms (Universal Robots). Port of ROS `ur_kinematics/ur_kin.cpp`, via
    /// compas_fab's `offset_wrist.py`. `params` = [d1, a2, a3, d4, d5, d6].
    OffsetWrist {
        params: [f64; 6],
        flange: Mat34,
        flange_inv: Mat34,
    },
    /// 6R arms whose last three axes intersect (ABB, KUKA, Stäubli, ...). Derived from the
    /// robot model itself (product of exponentials + Paden–Kahan subproblems), so no
    /// per-robot parameters or joint conventions are needed.
    SphericalWrist(Box<SphericalWrist>),
}

/// Zero-configuration geometry of a spherical-wrist arm (robot coordinates).
#[derive(Clone, Debug)]
pub struct SphericalWrist {
    /// Point on and unit direction of each joint axis at q = 0.
    p: [[f64; 3]; 6],
    w: [[f64; 3]; 6],
    /// End-effector frame at q = 0.
    ee0: Mat34,
    /// Wrist centre at q = 0, and in end-effector coordinates (constant).
    c0: [f64; 3],
    c_ee: [f64; 3],
}

pub const MAX_SOLUTIONS: usize = 8;
const ZERO_THRESH: f64 = 1e-8;

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl AnalyticIk {
    pub fn offset_wrist(params: [f64; 6], flange: Mat34) -> Self {
        AnalyticIk::OffsetWrist {
            params,
            flange,
            flange_inv: invert_rigid(&flange),
        }
    }

    /// All solutions for the end-effector link at `target` (world = robot root),
    /// wrapped into `limits`. Solutions outside the limits are dropped.
    pub fn solve(&self, target: &Mat34, limits: &[[f64; 2]]) -> Vec<[f64; 6]> {
        let mut out = Vec::with_capacity(MAX_SOLUTIONS);
        match self {
            AnalyticIk::OffsetWrist {
                params, flange_inv, ..
            } => {
                let t = mul(target, flange_inv);
                offset_wrist_ik(&t, params, |q| out.push(q));
            }
            AnalyticIk::SphericalWrist(sw) => sw.solve(target, |q| out.push(q)),
        }
        // Near singularities the closed form loses precision; polish with Newton steps on the
        // chain FK and drop what does not converge.
        out.into_iter()
            .filter_map(|q| self.refine(q, target))
            .filter_map(|q| wrap_into_limits(q, limits))
            .collect()
    }

    /// Levenberg–Marquardt polish towards `target` (adaptive damping, so it also converges at
    /// elbow and wrist singularities); `None` if the residual stays above 1e-8.
    pub fn refine(&self, mut q: [f64; 6], target: &Mat34) -> Option<[f64; 6]> {
        const TOL: f64 = 1e-10;
        let max_abs = |e: &[f64; 6]| e.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let mut e = pose_error(&self.forward(&q), target);
        let mut lambda = 1e-9;
        for _ in 0..30 {
            if max_abs(&e) < TOL {
                return Some(q);
            }
            let jac = self.numeric_jacobian(&q);
            let dq = damped_solve(&jac, &e, lambda)?;
            let trial: [f64; 6] = std::array::from_fn(|j| q[j] + dq[j]);
            let e_trial = pose_error(&self.forward(&trial), target);
            if max_abs(&e_trial) < max_abs(&e) {
                q = trial;
                e = e_trial;
                lambda = (lambda * 0.1).max(1e-15);
            } else {
                lambda *= 10.0;
                if lambda > 1e3 {
                    break;
                }
            }
        }
        (max_abs(&e) < 1e-8).then_some(q)
    }

    fn numeric_jacobian(&self, q: &[f64; 6]) -> [[f64; 6]; 6] {
        let h = 1e-7;
        let base = self.forward(q);
        let mut jac = [[0.0; 6]; 6];
        for j in 0..6 {
            let mut qh = *q;
            qh[j] += h;
            let e = pose_error(&base, &self.forward(&qh));
            for r in 0..6 {
                jac[r][j] = e[r] / h;
            }
        }
        jac
    }

    /// Forward kinematics of the closed-form chain, in model terms (for tests).
    pub fn forward(&self, q: &[f64; 6]) -> Mat34 {
        match self {
            AnalyticIk::OffsetWrist { params, flange, .. } => {
                mul(&offset_wrist_fk(q, params), flange)
            }
            AnalyticIk::SphericalWrist(sw) => sw.forward(q),
        }
    }
}

/// Bring each joint value to the 2π-equivalent closest to zero that lies within its limits.
pub fn wrap_into_limits(q: [f64; 6], limits: &[[f64; 2]]) -> Option<[f64; 6]> {
    let mut out = q;
    for j in 0..6 {
        let [lo, hi] = limits[j];
        let mut v = q[j].rem_euclid(2.0 * PI);
        if v > PI {
            v -= 2.0 * PI;
        }
        if v < lo {
            v += 2.0 * PI * ((lo - v) / (2.0 * PI)).ceil();
        } else if v > hi {
            v -= 2.0 * PI * ((v - hi) / (2.0 * PI)).ceil();
        }
        if v < lo - 1e-12 || v > hi + 1e-12 {
            return None;
        }
        out[j] = v;
    }
    Some(out)
}

pub fn offset_wrist_fk(q: &[f64; 6], p: &[f64; 6]) -> Mat34 {
    let [d1, a2, a3, d4, d5, d6] = *p;
    let (s1, c1) = q[0].sin_cos();
    let (s2, c2) = q[1].sin_cos();
    let (s4, c4) = q[3].sin_cos();
    let (s5, c5) = q[4].sin_cos();
    let (s6, c6) = q[5].sin_cos();
    let q23 = q[1] + q[2];
    let q234 = q23 + q[3];
    let (s23, c23) = q23.sin_cos();
    let (s234, c234) = q234.sin_cos();
    [
        c234 * c1 * s5 - c5 * s1,
        c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6,
        -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6,
        d6 * c234 * c1 * s5
            - a3 * c23 * c1
            - a2 * c1 * c2
            - d6 * c5 * s1
            - d5 * s234 * c1
            - d4 * s1,
        c1 * c5 + c234 * s1 * s5,
        -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6,
        s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1,
        d6 * (c1 * c5 + c234 * s1 * s5) + d4 * c1 - a3 * c23 * s1 - a2 * c2 * s1 - d5 * s234 * s1,
        -s234 * s5,
        -c234 * s6 - s234 * c5 * c6,
        s234 * c5 * s6 - c234 * c6,
        d1 + a3 * s23 + a2 * s2 - d5 * (c23 * c4 - s23 * s4) - d6 * s5 * (c23 * s4 + s23 * c4),
    ]
}

/// The ur_kin.cpp closed form. `t` is the flange (DH end) frame. Calls `emit` per solution.
pub fn offset_wrist_ik(t: &Mat34, p: &[f64; 6], mut emit: impl FnMut([f64; 6])) {
    let [d1, a2, a3, d4, d5, d6] = *p;
    // ur_kin reads the frame with axes permuted: x <- yaxis, y <- zaxis, z <- xaxis (see offset_wrist.py).
    let (t00, t10, t20) = (t[1], t[5], t[9]);
    let (t01, t11, t21) = (t[2], t[6], t[10]);
    let (mut t02, mut t12, t22) = (t[0], t[4], t[8]);
    let (mut t03, mut t13, t23) = (t[3], t[7], t[11]);
    let t20 = -t20;
    let t21 = -t21;
    t02 = -t02;
    t03 = -t03;
    t12 = -t12;
    t13 = -t13;

    // Shoulder (q1)
    let mut q1 = [0.0; 2];
    let a = d6 * t12 - t13;
    let b = d6 * t02 - t03;
    let r = a * a + b * b;
    if a.abs() < ZERO_THRESH {
        let div = if (d4.abs() - b.abs()).abs() < ZERO_THRESH {
            -sign(d4) * sign(b)
        } else {
            -d4 / b
        };
        let mut arcsin = div.asin();
        if arcsin.abs() < ZERO_THRESH {
            arcsin = 0.0;
        }
        q1[0] = if arcsin < 0.0 {
            arcsin + 2.0 * PI
        } else {
            arcsin
        };
        q1[1] = PI - arcsin;
    } else if b.abs() < ZERO_THRESH {
        let div = if (d4.abs() - a.abs()).abs() < ZERO_THRESH {
            sign(d4) * sign(a)
        } else {
            d4 / a
        };
        let arccos = div.acos();
        q1[0] = arccos;
        q1[1] = 2.0 * PI - arccos;
    } else if d4 * d4 > r {
        return;
    } else {
        let arccos = (d4 / r.sqrt()).acos();
        let arctan = (-b).atan2(a);
        let mut pos = arccos + arctan;
        let mut neg = -arccos + arctan;
        if pos.abs() < ZERO_THRESH {
            pos = 0.0;
        }
        if neg.abs() < ZERO_THRESH {
            neg = 0.0;
        }
        q1[0] = if pos >= 0.0 { pos } else { 2.0 * PI + pos };
        q1[1] = if neg >= 0.0 { neg } else { 2.0 * PI + neg };
    }

    // Wrist 2 (q5)
    let mut q5 = [[0.0; 2]; 2];
    for i in 0..2 {
        let numer = t03 * q1[i].sin() - t13 * q1[i].cos() - d4;
        let div = if (numer.abs() - d6.abs()).abs() < ZERO_THRESH {
            sign(numer) * sign(d6)
        } else {
            numer / d6
        };
        if div.abs() > 1.0 {
            continue; // unreachable orientation for this shoulder solution
        }
        let arccos = div.acos();
        q5[i] = [arccos, 2.0 * PI - arccos];
    }

    for i in 0..2 {
        let (s1, c1) = q1[i].sin_cos();
        for j in 0..2 {
            let (s5, c5) = q5[i][j].sin_cos();
            // Wrist 3 (q6)
            let mut q6 = if s5.abs() < ZERO_THRESH {
                0.0
            } else {
                (sign(s5) * -(t01 * s1 - t11 * c1)).atan2(sign(s5) * (t00 * s1 - t10 * c1))
            };
            if q6.abs() < ZERO_THRESH {
                q6 = 0.0;
            }
            if q6 < 0.0 {
                q6 += 2.0 * PI;
            }
            // RRR (q2, q3, q4)
            let (s6, c6) = q6.sin_cos();
            let x04x = -s5 * (t02 * c1 + t12 * s1)
                - c5 * (s6 * (t01 * c1 + t11 * s1) - c6 * (t00 * c1 + t10 * s1));
            let x04y = c5 * (t20 * c6 - t21 * s6) - t22 * s5;
            let p13x = d5 * (s6 * (t00 * c1 + t10 * s1) + c6 * (t01 * c1 + t11 * s1))
                - d6 * (t02 * c1 + t12 * s1)
                + t03 * c1
                + t13 * s1;
            let p13y = t23 - d1 - d6 * t22 + d5 * (t21 * c6 + t20 * s6);
            let mut c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3);
            if (c3.abs() - 1.0).abs() < ZERO_THRESH {
                c3 = sign(c3);
            } else if c3.abs() > 1.0 {
                continue;
            }
            let arccos = c3.acos();
            let q3 = [arccos, 2.0 * PI - arccos];
            let denom = a2 * a2 + a3 * a3 + 2.0 * a2 * a3 * c3;
            let s3 = arccos.sin();
            let aa = a2 + a3 * c3;
            let bb = a3 * s3;
            let q2 = [
                ((aa * p13y - bb * p13x) / denom).atan2((aa * p13x + bb * p13y) / denom),
                ((aa * p13y + bb * p13x) / denom).atan2((aa * p13x - bb * p13y) / denom),
            ];
            for k in 0..2 {
                let (s23, c23) = (q2[k] + q3[k]).sin_cos();
                let q4 = (c23 * x04y - s23 * x04x).atan2(x04x * c23 + x04y * s23);
                emit([q1[i], q2[k], q3[k], q4, q5[i][j], q6]);
            }
        }
    }
}

/// Pose error from `a` to `t`: translation (3) and small-angle rotation vector (3), world frame.
pub fn pose_error(a: &Mat34, t: &Mat34) -> [f64; 6] {
    let mut e = [t[3] - a[3], t[7] - a[7], t[11] - a[11], 0.0, 0.0, 0.0];
    for c in 0..3 {
        let u = [a[c], a[4 + c], a[8 + c]];
        let v = [t[c], t[4 + c], t[8 + c]];
        e[3] += 0.5 * (u[1] * v[2] - u[2] * v[1]);
        e[4] += 0.5 * (u[2] * v[0] - u[0] * v[2]);
        e[5] += 0.5 * (u[0] * v[1] - u[1] * v[0]);
    }
    e
}

/// Solve (JᵀJ + λI) x = Jᵀe by Gaussian elimination with partial pivoting.
pub fn damped_solve(jac: &[[f64; 6]; 6], e: &[f64; 6], lambda: f64) -> Option<[f64; 6]> {
    let mut m = [[0.0; 7]; 6];
    for r in 0..6 {
        for c in 0..6 {
            m[r][c] = (0..6).map(|k| jac[k][r] * jac[k][c]).sum::<f64>()
                + if r == c { lambda } else { 0.0 };
        }
        m[r][6] = (0..6).map(|k| jac[k][r] * e[k]).sum();
    }
    for col in 0..6 {
        let pivot = (col..6).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        for r in 0..6 {
            if r != col {
                let f = m[r][col] / m[col][col];
                for c in col..7 {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    Some(std::array::from_fn(|r| m[r][6] / m[r][r]))
}

// ---------------------------------------------------------------------------
// Spherical wrist, from the model (product of exponentials)

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
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

/// Rotation matrix (3x3, row-major) about unit axis `w` by `t`.
fn rot(w: V3, t: f64) -> [f64; 9] {
    let (s, c) = t.sin_cos();
    let k = 1.0 - c;
    let [x, y, z] = w;
    [
        k * x * x + c,
        k * x * y - s * z,
        k * x * z + s * y,
        k * x * y + s * z,
        k * y * y + c,
        k * y * z - s * x,
        k * x * z - s * y,
        k * y * z + s * x,
        k * z * z + c,
    ]
}
fn rmul(a: &[f64; 9], v: V3) -> V3 {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}
fn rmm(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut m = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            m[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    m
}
fn rt(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}
fn rot_of(m: &Mat34) -> [f64; 9] {
    [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
}

/// Paden–Kahan subproblem 1: angle rotating `u` onto `v` about unit axis `w` (through the origin).
fn pk1(w: V3, u: V3, v: V3) -> f64 {
    let up = sub(u, scale(w, dot(w, u)));
    let vp = sub(v, scale(w, dot(w, v)));
    dot(w, cross(up, vp)).atan2(dot(up, vp))
}

/// Subproblem 3: angles θ with |R(w, θ) u - v| = d (axis through the origin).
fn pk3(w: V3, u: V3, v: V3, d: f64) -> Vec<f64> {
    let up = sub(u, scale(w, dot(w, u)));
    let vp = sub(v, scale(w, dot(w, v)));
    let dz = dot(w, sub(u, v));
    let d2 = (d * d - dz * dz).max(0.0);
    let (nu, nv) = (norm(up), norm(vp));
    if nu < 1e-12 || nv < 1e-12 {
        return Vec::new();
    }
    let theta0 = pk1(w, up, vp);
    let c = (nu * nu + nv * nv - d2) / (2.0 * nu * nv);
    // Allow a little slack: model geometry is only approximately ideal (e.g. wrist axes
    // missing each other by microns), and the Newton polish removes the residual.
    if c.abs() > 1.0 + 1e-5 {
        return Vec::new();
    }
    let a = c.clamp(-1.0, 1.0).acos();
    vec![theta0 + a, theta0 - a]
}

/// Subproblem 2: (θ1, θ2) with R(w1, θ1) R(w2, θ2) p = q, unit axes through the origin.
fn pk2(w1: V3, w2: V3, p: V3, q: V3) -> Vec<(f64, f64)> {
    let c12 = dot(w1, w2);
    let den = c12 * c12 - 1.0;
    if den.abs() < 1e-12 {
        return Vec::new();
    }
    let alpha = (c12 * dot(w2, p) - dot(w1, q)) / den;
    let beta = (c12 * dot(w1, q) - dot(w2, p)) / den;
    let n = cross(w1, w2);
    let g2 = (dot(p, p) - alpha * alpha - beta * beta - 2.0 * alpha * beta * c12) / dot(n, n);
    if g2 < -1e-9 {
        return Vec::new();
    }
    let g = g2.max(0.0).sqrt();
    let mut out = Vec::with_capacity(2);
    for sgn in [1.0, -1.0] {
        let z = add(add(scale(w1, alpha), scale(w2, beta)), scale(n, sgn * g));
        // R(w2, θ2) p = z and R(w1, θ1) z = q.
        out.push((pk1(w1, z, q), pk1(w2, p, z)));
    }
    out
}

impl SphericalWrist {
    /// Build from the model if its 6 configurable joints are revolute, joints 2 and 3 are
    /// parallel and perpendicular to joint 1, and axes 4-6 meet in a point.
    pub fn from_model(model: &crate::model::RobotModel) -> Result<Self, String> {
        use crate::model::JointType;
        if model.dof() != 6 {
            return Err("spherical-wrist IK needs 6 joints".into());
        }
        let frames = crate::reference::forward_kinematics(model, &[0.0; 6]);
        let mut p = [[0.0; 3]; 6];
        let mut w = [[0.0; 3]; 6];
        for (i, link) in model.links.iter().enumerate() {
            if link.q_index < 0 {
                continue;
            }
            if link.joint_type != JointType::Revolute {
                return Err("spherical-wrist IK needs revolute joints".into());
            }
            let j = link.q_index as usize;
            let f = &frames[i];
            p[j] = [f[3], f[7], f[11]];
            let a = link.axis;
            let d = [
                f[0] * a[0] + f[1] * a[1] + f[2] * a[2],
                f[4] * a[0] + f[5] * a[1] + f[6] * a[2],
                f[8] * a[0] + f[9] * a[1] + f[10] * a[2],
            ];
            w[j] = scale(d, 1.0 / norm(d));
        }
        if norm(cross(w[1], w[2])) > 1e-9 {
            return Err("joints 2 and 3 are not parallel".into());
        }
        if dot(w[0], w[1]).abs() > 1e-9 {
            return Err("joint 2 is not perpendicular to joint 1".into());
        }
        // Wrist centre: least-squares intersection of axes 4, 5, 6.
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for k in 3..6 {
            for r in 0..3 {
                for c in 0..3 {
                    let m = if r == c { 1.0 } else { 0.0 } - w[k][r] * w[k][c];
                    a[r][c] += m;
                    b[r] += m * p[k][c];
                }
            }
        }
        let c0 = solve3(a, b).ok_or("axes 4-6 are parallel")?;
        for k in 3..6 {
            let d = sub(c0, p[k]);
            let off = norm(sub(d, scale(w[k], dot(w[k], d))));
            if off > 1e-4 {
                return Err(format!(
                    "axes 4-6 do not intersect (axis {} misses by {off:.2e} m)",
                    k + 1
                ));
            }
        }
        let ee0 = frames[model.end_effector_link];
        let inv = invert_rigid(&ee0);
        let c_ee = [
            inv[0] * c0[0] + inv[1] * c0[1] + inv[2] * c0[2] + inv[3],
            inv[4] * c0[0] + inv[5] * c0[1] + inv[6] * c0[2] + inv[7],
            inv[8] * c0[0] + inv[9] * c0[1] + inv[10] * c0[2] + inv[11],
        ];
        Ok(Self {
            p,
            w,
            ee0,
            c0,
            c_ee,
        })
    }

    /// FK by product of exponentials (agrees with the model's FK).
    pub fn forward(&self, q: &[f64; 6]) -> Mat34 {
        let mut t = crate::model::IDENTITY;
        for j in 0..6 {
            let r = rot(self.w[j], q[j]);
            let pj = self.p[j];
            let tr = sub(pj, rmul(&r, pj));
            let e = [
                r[0], r[1], r[2], tr[0], r[3], r[4], r[5], tr[1], r[6], r[7], r[8], tr[2],
            ];
            t = mul(&t, &e);
        }
        mul(&t, &self.ee0)
    }

    /// Raw closed-form solutions (before polishing and limit wrapping).
    pub fn solve_raw(&self, target: &Mat34, emit: impl FnMut([f64; 6])) {
        self.solve(target, emit)
    }

    fn solve(&self, target: &Mat34, mut emit: impl FnMut([f64; 6])) {
        let (p, w) = (&self.p, &self.w);
        let ce = self.c_ee;
        let c = [
            target[0] * ce[0] + target[1] * ce[1] + target[2] * ce[2] + target[3],
            target[4] * ce[0] + target[5] * ce[1] + target[6] * ce[2] + target[7],
            target[8] * ce[0] + target[9] * ce[1] + target[10] * ce[2] + target[11],
        ];
        // q1: rotating joint 2's axis about joint 1 must keep the wrist centre at its
        // zero-configuration offset along that axis (the arm plane).
        let k0 = dot(w[1], sub(self.c0, p[1]));
        let v = sub(c, p[0]);
        let big_k = k0 + dot(w[1], sub(p[1], p[0]));
        let a = dot(w[1], v);
        let b = dot(cross(w[0], w[1]), v);
        let r = (a * a + b * b).sqrt();
        if r < 1e-12 || big_k.abs() > r {
            return;
        }
        let phi = b.atan2(a);
        let delta = (big_k / r).clamp(-1.0, 1.0).acos();
        let r_target = rot_of(target);
        let r_ee0_t = rt(&rot_of(&self.ee0));
        for q1 in [phi + delta, phi - delta] {
            // Wrist centre with joint 1 undone.
            let r1 = rot(w[0], q1);
            let c1 = add(rmul(&rt(&r1), sub(c, p[0])), p[0]);
            // q3: distance from joint 2's axis point to the wrist centre depends on q3 only.
            let d = norm(sub(c1, p[1]));
            for q3 in pk3(w[2], sub(self.c0, p[2]), sub(p[1], p[2]), d) {
                let r3 = rot(w[2], q3);
                let c3 = add(rmul(&r3, sub(self.c0, p[2])), p[2]);
                let q2 = pk1(w[1], sub(c3, p[1]), sub(c1, p[1]));
                // Wrist: R4 R5 R6 = (R1 R2 R3)^T R_target R_ee0^T.
                let r123 = rmm(&rmm(&r1, &rot(w[1], q2)), &r3);
                let r_wrist = rmm(&rmm(&rt(&r123), &r_target), &r_ee0_t);
                for (q4, q5) in pk2(w[3], w[4], w[5], rmul(&r_wrist, w[5])) {
                    let r45 = rmm(&rot(w[3], q4), &rot(w[4], q5));
                    // q6 from any vector perpendicular to axis 6.
                    let probe = if w[5][0].abs() < 0.9 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    let u = sub(probe, scale(w[5], dot(w[5], probe)));
                    let q6 = pk1(w[5], u, rmul(&rt(&r45), rmul(&r_wrist, u)));
                    emit([q1, q2, q3, q4, q5, q6]);
                }
            }
        }
    }
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut x = [0.0; 3];
    for i in 0..3 {
        let mut m = a;
        for r in 0..3 {
            m[r][i] = b[r];
        }
        x[i] = det(m) / d;
    }
    Some(x)
}

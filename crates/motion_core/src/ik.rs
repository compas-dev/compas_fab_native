//! Closed-form inverse kinematics, all solutions per target.
//!
//! Solvers work in the robot model's own terms: the target is the world frame of the
//! end-effector link (e.g. `tool0`) and the returned joint values are model joint values,
//! wrapped into the joint limits. Every solution is checked by the model's FK in tests.

#![allow(clippy::needless_range_loop)] // the closed form indexes several arrays in lock step

use std::f64::consts::PI;

use crate::layout::invert_rigid;
use crate::model::Mat34;
use crate::reference::mul;

/// Description of a closed-form solver, as exported with the robot fixture.
#[derive(Clone, Debug)]
pub enum AnalyticIk {
    /// Offset-wrist 6R arms (Universal Robots). Port of ROS `ur_kinematics/ur_kin.cpp`, via
    /// compas_fab's `offset_wrist.py`. `params` = [d1, a2, a3, d4, d5, d6].
    OffsetWrist {
        params: [f64; 6],
        flange: Mat34,
        flange_inv: Mat34,
    },
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
        }
        // Near singularities the closed form loses precision; polish with Newton steps on the
        // chain FK and drop what does not converge.
        out.into_iter()
            .filter_map(|q| self.refine(q, target))
            .filter_map(|q| wrap_into_limits(q, limits))
            .collect()
    }

    /// Damped Newton refinement towards `target`; `None` if the residual stays above tolerance.
    pub fn refine(&self, mut q: [f64; 6], target: &Mat34) -> Option<[f64; 6]> {
        const TOL: f64 = 1e-10;
        for _ in 0..4 {
            let e = pose_error(&self.forward(&q), target);
            if e.iter().all(|v| v.abs() < TOL) {
                return Some(q);
            }
            let jac = self.numeric_jacobian(&q);
            let dq = damped_solve(&jac, &e, 1e-12)?;
            for j in 0..6 {
                q[j] += dq[j];
            }
        }
        let e = pose_error(&self.forward(&q), target);
        e.iter().all(|v| v.abs() < 1e-8).then_some(q)
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

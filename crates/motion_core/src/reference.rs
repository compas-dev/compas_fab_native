//! f64 reference implementation: straightforward, one configuration at a time.
//! Checked against compas_robots; every f32 kernel is checked against this.

use crate::layout::invert_rigid;
use crate::model::{JointType, Mat34, RobotModel, Scene};

pub fn mul(a: &Mat34, b: &Mat34) -> Mat34 {
    let mut m = [0.0; 12];
    for r in 0..3 {
        for c in 0..4 {
            let mut v = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
            if c == 3 {
                v += a[r * 4 + 3];
            }
            m[r * 4 + c] = v;
        }
    }
    m
}

pub fn transform_point(m: &Mat34, p: [f64; 3]) -> [f64; 3] {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    ]
}

/// Motion of a joint at value `q`, in the joint's own frame.
pub fn joint_motion(kind: JointType, axis: [f64; 3], q: f64) -> Mat34 {
    match kind {
        JointType::Fixed => crate::model::IDENTITY,
        JointType::Prismatic => [
            1.0,
            0.0,
            0.0,
            axis[0] * q,
            0.0,
            1.0,
            0.0,
            axis[1] * q,
            0.0,
            0.0,
            1.0,
            axis[2] * q,
        ],
        JointType::Revolute => {
            let (s, c) = q.sin_cos();
            let t = 1.0 - c;
            let [x, y, z] = axis;
            [
                t * x * x + c,
                t * x * y - s * z,
                t * x * z + s * y,
                0.0,
                t * x * y + s * z,
                t * y * y + c,
                t * y * z - s * x,
                0.0,
                t * x * z - s * y,
                t * y * z + s * x,
                t * z * z + c,
                0.0,
            ]
        }
    }
}

/// World transform of every link (robot root = world).
pub fn forward_kinematics(model: &RobotModel, q: &[f64]) -> Vec<Mat34> {
    let mut frames: Vec<Mat34> = Vec::with_capacity(model.links.len());
    for link in &model.links {
        let value = if link.q_index >= 0 {
            q[link.q_index as usize]
        } else {
            0.0
        };
        let local = mul(
            &link.origin,
            &joint_motion(link.joint_type, link.axis, value),
        );
        let world = if link.parent < 0 {
            local
        } else {
            mul(&frames[link.parent as usize], &local)
        };
        frames.push(world);
    }
    frames
}

/// Signed distance from a point to a box, and its gradient, in world coordinates.
pub fn point_box_distance(p: [f64; 3], frame: &Mat34, half: [f64; 3]) -> (f64, [f64; 3]) {
    let inv = invert_rigid(frame);
    let l = transform_point(&inv, p);
    let q = [
        l[0].abs() - half[0],
        l[1].abs() - half[1],
        l[2].abs() - half[2],
    ];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let outside = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    let sign = [l[0].signum(), l[1].signum(), l[2].signum()];
    let g_local = if outside > 0.0 {
        [
            sign[0] * o[0] / outside,
            sign[1] * o[1] / outside,
            sign[2] * o[2] / outside,
        ]
    } else {
        let axis = if q[0] >= q[1] && q[0] >= q[2] {
            0
        } else if q[1] >= q[2] {
            1
        } else {
            2
        };
        let mut g = [0.0; 3];
        g[axis] = sign[axis];
        g
    };
    // Rotate the gradient back to world: R * g.
    let g = [
        frame[0] * g_local[0] + frame[1] * g_local[1] + frame[2] * g_local[2],
        frame[4] * g_local[0] + frame[5] * g_local[1] + frame[6] * g_local[2],
        frame[8] * g_local[0] + frame[9] * g_local[1] + frame[10] * g_local[2],
    ];
    (outside + inside, g)
}

/// Per-sphere distance (minus radius) to the nearest cuboid, its gradient, and the configuration minimum.
pub struct Query {
    pub dist: Vec<f64>,
    pub grad: Vec<[f64; 3]>,
    pub min: f64,
}

pub fn distance_query(model: &RobotModel, scene: &Scene, q: &[f64]) -> Query {
    let frames = forward_kinematics(model, q);
    let mut dist = Vec::with_capacity(model.spheres.len());
    let mut grad = Vec::with_capacity(model.spheres.len());
    for (i, s) in model.spheres.iter().enumerate() {
        let c = transform_point(&frames[s.link as usize], s.center);
        let mut best = (f64::INFINITY, [0.0; 3]);
        for (k, b) in scene.cuboids.iter().enumerate() {
            let exempt = k < 32 && scene.sphere_touch.get(i).is_some_and(|m| m & (1 << k) != 0);
            if exempt || b.touch_links.contains(&model.links[s.link as usize].name) {
                continue;
            }
            let (d, g) = point_box_distance(c, &b.frame, b.half_extents);
            if d < best.0 {
                best = (d, g);
            }
        }
        if let Some(esdf) = &scene.esdf {
            let (d, g) = esdf.sample([c[0] as f32, c[1] as f32, c[2] as f32]);
            let d = (d - scene.esdf_margin) as f64;
            if d < best.0 {
                best = (d, g.map(|v| v as f64));
            }
        }
        dist.push(best.0 - s.radius);
        grad.push(best.1);
    }
    let min = dist.iter().copied().fold(f64::INFINITY, f64::min);
    Query { dist, grad, min }
}

/// Minimum clearance over all checked sphere pairs (self-collision), f64.
pub fn self_collision_min(model: &RobotModel, q: &[f64]) -> f64 {
    let frames = forward_kinematics(model, q);
    let world: Vec<[f64; 3]> = model
        .self_spheres
        .iter()
        .map(|s| transform_point(&frames[s.link as usize], s.center))
        .collect();
    let disabled = |a: u32, b: u32| {
        model
            .disabled_collisions
            .iter()
            .any(|&[x, y]| (x as u32 == a && y as u32 == b) || (x as u32 == b && y as u32 == a))
    };
    let mut min = f64::INFINITY;
    for i in 0..model.self_spheres.len() {
        for j in i + 1..model.self_spheres.len() {
            let (si, sj) = (&model.self_spheres[i], &model.self_spheres[j]);
            if si.link == sj.link || disabled(si.link, sj.link) {
                continue;
            }
            let d = ((world[i][0] - world[j][0]).powi(2)
                + (world[i][1] - world[j][1]).powi(2)
                + (world[i][2] - world[j][2]).powi(2))
            .sqrt();
            min = min.min(d - si.radius - sj.radius);
        }
    }
    min
}

//! MotionBenchMaker problems as shipped by VAMP (`resources/<robot>/problems.json`, written
//! by VAMP's `problem_tar_to_pkl_json.py`), turned into obstacles with VAMP's own rules:
//! cylinders become capsules, except in the `box` scenario, where they become boxes.
//!
//! The problems are MotionBenchMaker's (Chamzas et al., RA-L 2022, BSD-3-Clause, Rice
//! University), read from a local VAMP checkout (Apache-2.0); not included. See `native/NOTICE`.

use motion_core::check::Obstacles;
use motion_core::{load_robot, RobotModel};
use serde_json::Value;

pub struct Problem {
    pub scenario: String,
    pub index: usize,
    pub start: Vec<f32>,
    pub goals: Vec<Vec<f32>>,
    pub valid: bool,
    pub obstacles: Obstacles,
}

fn vec3(v: &Value) -> [f64; 3] {
    std::array::from_fn(|k| v[k].as_f64().unwrap())
}

fn config(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// 3x4 frame from a center and Euler XYZ angles, R = Rz(z) Ry(y) Rx(x) (as VAMP and URDF).
pub fn frame(center: [f64; 3], euler: [f64; 3]) -> [f64; 12] {
    let (sr, cr) = euler[0].sin_cos();
    let (sp, cp) = euler[1].sin_cos();
    let (sy, cy) = euler[2].sin_cos();
    [
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        center[0],
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        center[1],
        -sp,
        cp * sr,
        cp * cr,
        center[2],
    ]
}

pub fn obstacles(p: &Value) -> Obstacles {
    let mut obs = Obstacles::default();
    let scenario = p["problem"].as_str().unwrap();
    for s in p["sphere"].as_array().unwrap() {
        obs.add_sphere(vec3(&s["position"]), s["radius"].as_f64().unwrap());
    }
    for c in p["cylinder"].as_array().unwrap() {
        let f = frame(vec3(&c["position"]), vec3(&c["orientation_euler_xyz"]));
        let (r, len) = (c["radius"].as_f64().unwrap(), c["length"].as_f64().unwrap());
        if scenario == "box" {
            obs.add_box(&f, [r, r, len / 2.0]);
        } else {
            let z = [f[2], f[6], f[10]];
            let at =
                |s: f64| std::array::from_fn(|k| [f[3], f[7], f[11]][k] + s * len / 2.0 * z[k]);
            obs.add_capsule(at(-1.0), at(1.0), r);
        }
    }
    for b in p["box"].as_array().unwrap() {
        let f = frame(vec3(&b["position"]), vec3(&b["orientation_euler_xyz"]));
        obs.add_box(&f, vec3(&b["half_extents"]));
    }
    obs
}

/// All problems of a `problems.json`, in file order (scenario by scenario).
pub fn load(path: &str) -> Vec<Problem> {
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut out = Vec::new();
    for (scenario, list) in data["problems"].as_object().unwrap() {
        for p in list.as_array().unwrap() {
            out.push(Problem {
                scenario: scenario.clone(),
                index: p["index"].as_u64().unwrap() as usize,
                start: config(&p["start"]),
                goals: p["goals"].as_array().unwrap().iter().map(config).collect(),
                valid: p["valid"].as_bool().unwrap_or(true),
                obstacles: obstacles(p),
            });
        }
    }
    out
}

/// A VAMP fixture, with link pairs that overlap in every configuration disabled (MoveIt's
/// "always in collision" rule; e.g. the UR5's wrist and force-torque sensor spheres, which
/// VAMP does not check either).
pub fn robot(path: &str) -> RobotModel {
    let (mut model, _) = load_robot(&std::fs::read_to_string(path).unwrap()).unwrap();
    let disabled = model.disable_always_colliding(2000, 0.999, 7);
    for [a, b] in disabled {
        eprintln!(
            "{}: disabled always-colliding pair {} / {}",
            model.name, model.links[a].name, model.links[b].name
        );
    }
    model
}

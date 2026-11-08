//! Compare sphere-model collision verdicts with PyBullet's exact-mesh verdicts
//! (from `scripts/baseline_python.py`) on the same configurations.
//!
//! Usage: compare_verdicts <baseline_python.json> [--robot fixtures/ur5.json] [--scene fixtures/scene_boxes.json]

use motion_core::reference::{distance_query, self_collision_min};
use motion_core::{load_robot, load_scene};

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let mut args = std::env::args().skip(1);
    let baseline_path = args.next().expect("baseline json");
    let (mut robot, mut scene) = (format!("{dir}/ur5.json"), format!("{dir}/scene_boxes.json"));
    while let Some(flag) = args.next() {
        let v = args.next().unwrap();
        match flag.as_str() {
            "--robot" => robot = v,
            "--scene" => scene = v,
            f => panic!("unknown flag {f}"),
        }
    }
    let (mut model, _) = load_robot(&std::fs::read_to_string(robot).unwrap()).unwrap();
    let added = if std::env::var_os("AUTO_DISABLE").is_some() {
        model.disable_always_colliding(2000, 0.95, 99)
    } else {
        Vec::new()
    };
    let names: Vec<String> = added
        .iter()
        .map(|&[a, b]| format!("{}-{}", model.links[a].name, model.links[b].name))
        .collect();
    println!("sphere model: disabled always-colliding link pairs {names:?}");
    let scene = load_scene(&std::fs::read_to_string(scene).unwrap()).unwrap();
    let baseline: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(baseline_path).unwrap()).unwrap();
    let configs = baseline["configs"].as_array().unwrap();
    for (check, with_self) in [("robot_vs_obstacles", false), ("full", true)] {
        let verdicts = baseline["pybullet"][check]["verdicts"].as_array().unwrap();
        let (mut both, mut spheres_only, mut mesh_only, mut neither) = (0, 0, 0, 0);
        let mut disputed = Vec::new();
        for (idx, (c, v)) in configs.iter().zip(verdicts).enumerate() {
            let q: Vec<f64> = c
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let mut d = distance_query(&model, &scene, &q).min;
            if with_self {
                d = d.min(self_collision_min(&model, &q));
            }
            match (d < 0.0, v.as_bool().unwrap()) {
                (true, true) => both += 1,
                (true, false) => {
                    spheres_only += 1;
                    disputed.push(serde_json::json!({"index": idx, "spheres": true}));
                }
                (false, true) => {
                    mesh_only += 1;
                    disputed.push(serde_json::json!({"index": idx, "spheres": false}));
                }
                (false, false) => neither += 1,
            }
        }
        println!(
            "{check:<19} both {both:4}  free {neither:4}  false alarms {spheres_only:4} ({:.1}% of mesh-free)  MISSED {mesh_only}",
            100.0 * spheres_only as f64 / (spheres_only + neither).max(1) as f64
        );
        if let Ok(dir) = std::env::var("DISPUTED_DIR") {
            let path = format!("{dir}/disputed_{check}.json");
            std::fs::write(&path, serde_json::to_string(&disputed).unwrap()).unwrap();
        }
    }
}

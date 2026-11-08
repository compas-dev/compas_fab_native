//! Compare sphere-model collision verdicts with PyBullet's exact-mesh verdicts
//! (from `scripts/baseline_python.py`) on the same configurations.
//!
//! Usage: compare_verdicts <baseline_python.json> [--robot fixtures/ur5.json] [--scene fixtures/scene_boxes.json]

use motion_core::reference::distance_query;
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
    let (model, _) = load_robot(&std::fs::read_to_string(robot).unwrap()).unwrap();
    let scene = load_scene(&std::fs::read_to_string(scene).unwrap()).unwrap();
    let baseline: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(baseline_path).unwrap()).unwrap();
    let configs = baseline["configs"].as_array().unwrap();
    let verdicts = baseline["pybullet"]["robot_vs_obstacles"]["verdicts"]
        .as_array()
        .unwrap();

    let (mut both, mut spheres_only, mut mesh_only, mut neither) = (0, 0, 0, 0);
    let mut miss_depth = 0.0f64;
    for (c, v) in configs.iter().zip(verdicts) {
        let q: Vec<f64> = c
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let d = distance_query(&model, &scene, &q).min;
        match (d < 0.0, v.as_bool().unwrap()) {
            (true, true) => both += 1,
            (true, false) => spheres_only += 1,
            (false, true) => {
                mesh_only += 1;
                miss_depth = miss_depth.max(d);
            }
            (false, false) => neither += 1,
        }
    }
    let n = configs.len();
    println!("{n} configurations, robot vs obstacles (no self-collision):");
    println!("  collision in both          {both}");
    println!("  free in both               {neither}");
    println!(
        "  spheres only (false alarm) {spheres_only}  ({:.1}% of mesh-free configs)",
        100.0 * spheres_only as f64 / (spheres_only + neither) as f64
    );
    println!("  mesh only (MISSED)         {mesh_only}  (largest sphere clearance among misses: {miss_depth:.4} m)");
}

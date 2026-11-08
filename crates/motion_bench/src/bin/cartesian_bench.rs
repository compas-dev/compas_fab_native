//! Native Cartesian planner on the waypoints written by `scripts/baseline_cartesian.py`.
//!
//! Usage: cartesian_bench <baseline_cartesian.json> [--robot fixtures/ur5.json] [--scene fixtures/scene_boxes.json]

use std::time::{Duration, Instant};

use motion_core::cartesian::{plan_cartesian, CartesianRequest};
use motion_core::{load_robot, load_scene, ModelBuffers};

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let mut args = std::env::args().skip(1);
    let baseline_path = args.next().expect("baseline_cartesian.json");
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
    let buffers = ModelBuffers::new(&model, &scene);
    let baseline: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&baseline_path).unwrap()).unwrap();
    let waypoints: Vec<[f64; 12]> = baseline["waypoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| std::array::from_fn(|k| w[k].as_f64().unwrap()))
        .collect();
    let start: Vec<f64> = baseline["start"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    println!(
        "{} waypoints, {} obstacle spheres, {} self pairs",
        waypoints.len(),
        buffers.spheres.len(),
        buffers.self_pairs.len()
    );
    if let Some(results) = baseline["results"].as_object() {
        for (name, r) in results {
            println!(
                "  compas_fab {name:<28} {:>8.1} ms  {}",
                r["seconds"].as_f64().unwrap_or(f64::NAN) * 1e3,
                r["error"]
                    .as_str()
                    .map(|e| e.lines().next().unwrap_or("").to_string())
                    .unwrap_or_default()
            );
        }
    }
    for (label, samples, with_start) in [
        ("fixed axis, from start", 1, true),
        ("fixed axis, free start", 1, false),
        ("12 axis samples, free start", 12, false),
        ("36 axis samples, free start", 36, false),
    ] {
        let req = CartesianRequest {
            waypoints: waypoints.clone(),
            free_axis_samples: samples,
            start: with_start.then(|| start.clone()),
            max_joint_step: 0.3,
            ..Default::default()
        };
        let mut times = Vec::new();
        let mut last = None;
        let t_budget = Instant::now();
        while times.len() < 5
            || (t_budget.elapsed() < Duration::from_millis(500) && times.len() < 200)
        {
            let t = Instant::now();
            last = Some(plan_cartesian(&model, &buffers, &req));
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];
        match last.unwrap() {
            Ok(res) => {
                let s = res.stats;
                println!(
                    "  native {label:<28} {:>8.2} ms  ({} IK nodes, {} free, {} edges, {} edge configs, {} rounds)",
                    median.as_secs_f64() * 1e3,
                    s.ik_nodes,
                    s.free_nodes,
                    s.edges_evaluated,
                    s.edge_configs_checked,
                    s.search_rounds
                );
            }
            Err(e) => println!(
                "  native {label:<28} {:>8.2} ms  FAILED {e:?}",
                median.as_secs_f64() * 1e3
            ),
        }
    }
}

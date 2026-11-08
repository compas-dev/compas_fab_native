//! Exact (parry3d) collision verdicts on the configurations of `scripts/baseline_python.py`,
//! compared with PyBullet and, where available, with an independent Python exact check.
//!
//! Usage: exact_verdicts <baseline_python.json> [<python_exact.json>...]

use std::time::Instant;

use motion_core::{load_robot, load_scene};
use motion_exact::ExactChecker;

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let mut args = std::env::args().skip(1);
    let baseline: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args.next().unwrap()).unwrap()).unwrap();
    let (model, _) =
        load_robot(&std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap()).unwrap();
    let scene =
        load_scene(&std::fs::read_to_string(format!("{dir}/scene_boxes.json")).unwrap()).unwrap();
    let mut checker = ExactChecker::new(
        model,
        &std::fs::read_to_string(format!("{dir}/ur5_meshes.json")).unwrap(),
    )
    .unwrap();
    checker.set_obstacle_boxes(&scene);

    let configs: Vec<Vec<f64>> = baseline["configs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let t = Instant::now();
    let verdicts: Vec<bool> = configs.iter().map(|q| checker.check(q).is_some()).collect();
    let dt = t.elapsed();
    println!(
        "exact (parry3d): {} / {} in collision, {:.1} µs per configuration",
        verdicts.iter().filter(|&&v| v).count(),
        configs.len(),
        dt.as_secs_f64() * 1e6 / configs.len() as f64
    );
    let pyb = baseline["pybullet"]["full"]["verdicts"].as_array().unwrap();
    let agree = verdicts
        .iter()
        .zip(pyb)
        .filter(|(a, b)| **a == b.as_bool().unwrap())
        .count();
    println!(
        "agreement with PyBullet (convex hulls): {agree} / {}",
        configs.len()
    );
    for path in args {
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (mut same, mut total) = (0, 0);
        let mut differ = Vec::new();
        for r in reference.as_array().unwrap() {
            let i = r["index"].as_u64().unwrap() as usize;
            let exact = r["exact"].as_bool().unwrap();
            total += 1;
            if verdicts[i] == exact {
                same += 1;
            } else {
                differ.push((
                    i,
                    exact,
                    checker
                        .check(&configs[i])
                        .map(|c| format!("{}-{}", c.a, c.b)),
                    r["pairs"].clone(),
                ));
            }
        }
        println!("agreement with Python exact check ({path}): {same} / {total}");
        for d in differ.iter().take(8) {
            println!(
                "  config {}: python exact {} (pairs {}), parry {:?}",
                d.0, d.1, d.3, d.2
            );
        }
    }
}

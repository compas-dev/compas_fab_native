//! CPU benchmarks for CI: a fixed set of single-threaded measurements, written as JSON in the
//! `customSmallerIsBetter` format of github-action-benchmark (`[{name, unit, value}]`) and as
//! a Markdown table on stdout.
//!
//! Single-threaded so that numbers from shared CI runners are comparable between runs; the
//! GPU is not measured (hosted runners have none).
//!
//! Usage: ci_bench [out.json]

use std::time::{Duration, Instant};

use motion_core::cartesian::{plan_cartesian_with, CartesianRequest, Config};
use motion_core::cpu::collision_min_lanes;
use motion_core::reference::forward_kinematics;
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, load_scene, ModelBuffers, RobotModel};
use motion_exact::ExactChecker;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

struct Bench {
    rows: Vec<(String, String, f64)>,
}

impl Bench {
    /// Median wall time of `f` over at least 5 runs and at most `budget`.
    fn time(&mut self, name: &str, unit: &str, scale: f64, budget: Duration, mut f: impl FnMut()) {
        f(); // warm-up
        let mut times = Vec::new();
        let start = Instant::now();
        while times.len() < 5 || (start.elapsed() < budget && times.len() < 1000) {
            let t = Instant::now();
            f();
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        let value = times[times.len() / 2] * scale;
        eprintln!("{name:<44} {value:>10.3} {unit}  ({} runs)", times.len());
        self.rows.push((name.into(), unit.into(), value));
    }
}

fn robot(name: &str) -> RobotModel {
    let (mut model, _) =
        load_robot(&std::fs::read_to_string(format!("{DIR}/fixtures/{name}.json")).unwrap())
            .unwrap();
    // As NativePlanner does by default.
    model.disable_always_colliding(1000, 0.95, 7);
    model
}

fn main() {
    let out_path = std::env::args().nth(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut bench = Bench { rows: Vec::new() };
    let budget = Duration::from_millis(1500);

    pool.install(|| {
        // ---- Collision queries (spheres vs boxes + self), per configuration -------
        let ur5 = robot("ur5");
        let scene = load_scene(
            &std::fs::read_to_string(format!("{DIR}/fixtures/scene_boxes.json")).unwrap(),
        )
        .unwrap();
        let buffers = ModelBuffers::new(&ur5, &scene);
        let batch = 4096;
        let q: Vec<f32> = random_configurations(&ur5, batch, 1)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let mut out = vec![0.0f32; batch];
        bench.time(
            "collision check, ur5 + boxes (per config)",
            "ns",
            1e9 / batch as f64,
            budget,
            || collision_min_lanes(&buffers, &q, &mut out),
        );

        // ---- Forward and inverse kinematics, per call -----------------------------
        for name in ["ur5", "abb_irb4600_40_255", "staubli_tx2_60l"] {
            let model = robot(name);
            let n = 256;
            let q = random_configurations(&model, n, 12);
            let configs: Vec<Vec<f64>> = (0..n)
                .map(|b| (0..6).map(|j| q[j * n + b]).collect())
                .collect();
            let targets: Vec<_> = configs
                .iter()
                .map(|c| forward_kinematics(&model, c)[model.end_effector_link])
                .collect();
            bench.time(
                &format!("forward kinematics, {name} (per call)"),
                "us",
                1e6 / n as f64,
                budget,
                || {
                    for c in &configs {
                        std::hint::black_box(forward_kinematics(&model, c));
                    }
                },
            );
            let ik = model.analytic_ik.as_ref().unwrap();
            bench.time(
                &format!("inverse kinematics, {name} (per target)"),
                "us",
                1e6 / n as f64,
                budget,
                || {
                    for t in &targets {
                        std::hint::black_box(ik.solve(t, &model.joint_limits));
                    }
                },
            );
        }

        // ---- Cartesian planning: 100 waypoints, UR5 among boxes, free start --------------
        let baseline: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{DIR}/results/baseline_cartesian.json")).unwrap(),
        )
        .unwrap();
        let waypoints: Vec<[f64; 12]> = baseline["waypoints"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| std::array::from_fn(|k| w[k].as_f64().unwrap()))
            .collect();
        let mut checker = ExactChecker::new(
            ur5.clone(),
            &std::fs::read_to_string(format!("{DIR}/fixtures/ur5_meshes.json")).unwrap(),
        )
        .unwrap();
        checker.set_obstacle_boxes(&scene);
        let exact = |configs: &[Config]| -> Vec<bool> {
            let qs: Vec<Vec<f64>> = configs.iter().map(|q| q.to_vec()).collect();
            checker
                .check_many(&qs)
                .into_iter()
                .map(|r| r.is_some())
                .collect()
        };
        for (label, samples, with_exact) in [
            ("cartesian 100 waypoints, fixed axis", 1, false),
            ("cartesian 100 waypoints, fixed axis + exact", 1, true),
            ("cartesian 100 waypoints, 12 axis samples", 12, false),
        ] {
            let req = CartesianRequest {
                waypoints: waypoints.clone(),
                free_axis_samples: samples,
                max_joint_step: 0.3,
                ..Default::default()
            };
            let validator =
                with_exact.then_some(&exact as &(dyn Fn(&[Config]) -> Vec<bool> + Sync));
            plan_cartesian_with(&ur5, &buffers, &req, validator)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            bench.time(label, "ms", 1e3, budget, || {
                std::hint::black_box(plan_cartesian_with(&ur5, &buffers, &req, validator).ok());
            });
        }

        // ---- Exact mesh check, per configuration -------------------------------------
        let n = 64;
        let q = random_configurations(&ur5, n, 3);
        let configs: Vec<Vec<f64>> = (0..n)
            .map(|b| (0..6).map(|j| q[j * n + b]).collect())
            .collect();
        bench.time(
            "exact mesh check, ur5 + boxes (per config)",
            "us",
            1e6 / n as f64,
            budget,
            || {
                std::hint::black_box(checker.check_many(&configs));
            },
        );
    });

    println!("| benchmark | value | unit |\n|---|---:|---|");
    for (name, unit, value) in &bench.rows {
        println!("| {name} | {value:.3} | {unit} |");
    }
    if let Some(path) = out_path {
        let json: Vec<_> = bench
            .rows
            .iter()
            .map(|(name, unit, value)| serde_json::json!({"name": name, "unit": unit, "value": value}))
            .collect();
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}

//! Phase 0 spike benchmark: batched FK + sphere distance query on every backend.
//!
//! 1. Checks each GPU kernel against the CPU scalar oracle.
//! 2. Times every backend over a range of batch sizes.
//!
//! Usage: motion_bench [--robot fixtures/ur5.json] [--scene fixtures/scene_boxes.json] [--out results/run.json]

use std::time::{Duration, Instant};

use motion_core::cpu::{query_lanes, query_lanes_min, query_scalar, BatchOutput, LANES};
use motion_core::reference::distance_query;
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, load_scene, ModelBuffers};
use motion_gpu::{GpuContext, GpuQuery, Kernel};

const BATCHES: &[usize] = &[1, 8, 64, 512, 4096, 32768, 131072, 524288];

struct Args {
    robot: String,
    scene: String,
    out: Option<String>,
}

fn args() -> Args {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let mut a = Args {
        robot: format!("{dir}/ur5.json"),
        scene: format!("{dir}/scene_boxes.json"),
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().expect("flag value");
        match flag.as_str() {
            "--robot" => a.robot = value,
            "--scene" => a.scene = value,
            "--out" => a.out = Some(value),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

/// Median of repeated runs, at least `min_reps` and at least ~`budget` of total time.
fn median_time(mut f: impl FnMut() -> Duration, min_reps: usize, budget: Duration) -> Duration {
    f(); // warm-up
    let mut samples = Vec::new();
    let start = Instant::now();
    while samples.len() < min_reps || (start.elapsed() < budget && samples.len() < 1000) {
        samples.push(f());
    }
    samples.sort();
    samples[samples.len() / 2]
}

fn timed(mut f: impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}

struct Row {
    backend: String,
    batch: usize,
    time: Duration,
}

impl Row {
    fn per_second(&self) -> f64 {
        self.batch as f64 / self.time.as_secs_f64()
    }
}

fn fmt_time(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 {
        format!("{us:.1} µs")
    } else {
        format!("{:.2} ms", us / 1000.0)
    }
}

fn fmt_rate(r: f64) -> String {
    if r >= 1e6 {
        format!("{:.1} M/s", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.1} k/s", r / 1e3)
    } else {
        format!("{r:.0} /s")
    }
}

fn compare(label: &str, oracle: &BatchOutput, other: &BatchOutput) -> (f32, f32, usize) {
    let mut dist = 0.0f32;
    let mut grad = 0.0f32;
    let mut disagree = 0;
    for i in 0..oracle.sphere_dist.len() {
        dist = dist.max((oracle.sphere_dist[i] - other.sphere_dist[i]).abs());
        // Ignore gradient flips at equidistant ties (where the distance itself agrees).
        let g = (0..3)
            .map(|k| (oracle.sphere_grad[i][k] - other.sphere_grad[i][k]).abs())
            .fold(0.0, f32::max);
        if g < 0.5 {
            grad = grad.max(g);
        }
    }
    for b in 0..oracle.batch {
        dist = dist.max((oracle.config_min[b] - other.config_min[b]).abs());
        disagree += ((oracle.config_min[b] < 0.0) != (other.config_min[b] < 0.0)) as usize;
    }
    println!("  {label:<22} max |Δdist| {dist:.2e} m   max |Δgrad| {grad:.2e}   collision verdicts differing: {disagree}/{}", oracle.batch);
    (dist, grad, disagree)
}

fn main() {
    let a = args();
    let (model, _) = load_robot(&std::fs::read_to_string(&a.robot).unwrap()).unwrap();
    let scene = load_scene(&std::fs::read_to_string(&a.scene).unwrap()).unwrap();
    let buffers = ModelBuffers::new(&model, &scene);
    let n_spheres = buffers.spheres.len();
    let threads = rayon::current_num_threads();
    println!(
        "robot {} ({} dof, {} links, {} spheres), scene {} cuboids, {} CPU threads",
        model.name,
        model.dof(),
        buffers.links.len(),
        n_spheres,
        buffers.cuboids.len(),
        threads
    );

    let gpu = GpuContext::new();
    match &gpu {
        Ok(ctx) => println!(
            "GPU: {} ({:?}, {:?})",
            ctx.info.name, ctx.info.backend, ctx.info.device_type
        ),
        Err(e) => println!("GPU: unavailable ({e})"),
    }

    // ---- Correctness ------------------------------------------------------------
    let check_batch = 4096;
    let q: Vec<f32> = random_configurations(&model, check_batch, 1)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let mut oracle = BatchOutput::new(check_batch, n_spheres);
    query_scalar(&buffers, &q, &mut oracle);
    let collisions = oracle.config_min.iter().filter(|&&d| d < 0.0).count();
    println!("\nCorrectness against the CPU scalar oracle ({check_batch} configs, {collisions} in collision):");
    let mut lanes = BatchOutput::new(check_batch, n_spheres);
    query_lanes(&buffers, &q, &mut lanes);
    let mut checks = vec![("cpu_lanes", compare("CPU lanes", &oracle, &lanes))];
    let mut lanes_min = BatchOutput::new(check_batch, n_spheres);
    lanes_min.sphere_dist.clone_from(&oracle.sphere_dist);
    lanes_min.sphere_grad.clone_from(&oracle.sphere_grad);
    query_lanes_min(&buffers, &q, &mut lanes_min.config_min);
    checks.push((
        "cpu_lanes_min",
        compare("CPU lanes min", &oracle, &lanes_min),
    ));
    if let Ok(ctx) = &gpu {
        let gq = GpuQuery::new(ctx, &buffers, check_batch).unwrap();
        for (name, kernel) in [
            ("gpu_fused_full", Kernel::FusedFull),
            ("gpu_split", Kernel::Split),
        ] {
            checks.push((name, compare(name, &oracle, &gq.run_full(&q, kernel))));
        }
        let (min, _) = gq.round_trip_min(&q, Kernel::FusedMin);
        let mut min_only = BatchOutput::new(check_batch, n_spheres);
        min_only.sphere_dist.clone_from(&oracle.sphere_dist);
        min_only.sphere_grad.clone_from(&oracle.sphere_grad);
        min_only.config_min = min;
        checks.push((
            "gpu_fused_min",
            compare("gpu_fused_min", &oracle, &min_only),
        ));
    }
    // f64 reference vs f32 oracle, as a sanity bound on f32 itself.
    let mut f64_err = 0.0f64;
    for b in 0..256 {
        let qb: Vec<f64> = (0..model.dof())
            .map(|j| q[j * check_batch + b] as f64)
            .collect();
        f64_err = f64_err
            .max((distance_query(&model, &scene, &qb).min - oracle.config_min[b] as f64).abs());
    }
    println!(
        "  {:<22} max |Δmin| {f64_err:.2e} m (f32 oracle vs f64 reference, 256 configs)",
        "f64 reference"
    );

    // ---- Timing -----------------------------------------------------------------
    let mut rows: Vec<Row> = Vec::new();
    let budget = Duration::from_millis(300);
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for &batch in BATCHES {
        let q64 = random_configurations(&model, batch, 2);
        let q: Vec<f32> = q64.iter().map(|&v| v as f32).collect();
        let mut out = BatchOutput::new(batch, n_spheres);

        if batch <= 4096 {
            let t = median_time(
                || {
                    timed(|| {
                        for b in 0..batch {
                            let qb: Vec<f64> =
                                (0..model.dof()).map(|j| q64[j * batch + b]).collect();
                            std::hint::black_box(distance_query(&model, &scene, &qb));
                        }
                    })
                },
                3,
                budget,
            );
            rows.push(Row {
                backend: "cpu_f64_reference_1t".into(),
                batch,
                time: t,
            });
        }
        let t = median_time(|| timed(|| query_scalar(&buffers, &q, &mut out)), 5, budget);
        rows.push(Row {
            backend: format!("cpu_scalar_{threads}t"),
            batch,
            time: t,
        });
        let t = single
            .install(|| median_time(|| timed(|| query_scalar(&buffers, &q, &mut out)), 3, budget));
        rows.push(Row {
            backend: "cpu_scalar_1t".into(),
            batch,
            time: t,
        });
        if batch % LANES == 0 {
            let t = median_time(|| timed(|| query_lanes(&buffers, &q, &mut out)), 5, budget);
            rows.push(Row {
                backend: format!("cpu_lanes_{threads}t"),
                batch,
                time: t,
            });
            let t = single.install(|| {
                median_time(|| timed(|| query_lanes(&buffers, &q, &mut out)), 3, budget)
            });
            rows.push(Row {
                backend: "cpu_lanes_1t".into(),
                batch,
                time: t,
            });
            let mut mins = vec![0.0f32; batch];
            let t = median_time(
                || timed(|| query_lanes_min(&buffers, &q, &mut mins)),
                5,
                budget,
            );
            rows.push(Row {
                backend: format!("cpu_lanes_min_{threads}t"),
                batch,
                time: t,
            });
            let t = single.install(|| {
                median_time(
                    || timed(|| query_lanes_min(&buffers, &q, &mut mins)),
                    3,
                    budget,
                )
            });
            rows.push(Row {
                backend: "cpu_lanes_min_1t".into(),
                batch,
                time: t,
            });
        }

        if let Ok(ctx) = &gpu {
            match GpuQuery::new(ctx, &buffers, batch) {
                Ok(gq) => {
                    gq.upload_q(&q);
                    for (name, kernel) in [
                        ("full", Kernel::FusedFull),
                        ("min", Kernel::FusedMin),
                        ("split", Kernel::Split),
                    ] {
                        let reps = (2_000_000 / batch).clamp(4, 200) as u32;
                        let t = median_time(|| gq.time_throughput(kernel, reps), 3, budget);
                        rows.push(Row {
                            backend: format!("gpu_fused_{name}_throughput")
                                .replace("fused_split", "split"),
                            batch,
                            time: t,
                        });
                    }
                    let t = median_time(|| gq.time_latency(Kernel::FusedMin), 5, budget);
                    rows.push(Row {
                        backend: "gpu_fused_min_latency".into(),
                        batch,
                        time: t,
                    });
                    let t = median_time(|| gq.round_trip_min(&q, Kernel::FusedMin).1, 5, budget);
                    rows.push(Row {
                        backend: "gpu_fused_min_round_trip".into(),
                        batch,
                        time: t,
                    });
                }
                Err(e) => println!("  skipping GPU at batch {batch}: {e}"),
            }
        }
    }

    // ---- Report -----------------------------------------------------------------
    let mut backends: Vec<String> = Vec::new();
    for r in &rows {
        if !backends.contains(&r.backend) {
            backends.push(r.backend.clone());
        }
    }
    println!("\nThroughput (configurations per second; time per batch in parentheses):\n");
    print!("| backend |");
    for b in BATCHES {
        print!(" {b} |");
    }
    println!();
    print!("|---|");
    for _ in BATCHES {
        print!("---|");
    }
    println!();
    for name in &backends {
        print!("| {name} |");
        for &b in BATCHES {
            match rows.iter().find(|r| &r.backend == name && r.batch == b) {
                Some(r) => print!(" {} ({}) |", fmt_rate(r.per_second()), fmt_time(r.time)),
                None => print!(" – |"),
            }
        }
        println!();
    }

    if let Some(path) = a.out {
        let json = serde_json::json!({
            "robot": model.name,
            "dof": model.dof(),
            "links": buffers.links.len(),
            "spheres": n_spheres,
            "cuboids": buffers.cuboids.len(),
            "cpu_threads": threads,
            "gpu": gpu.as_ref().ok().map(|c| format!("{} ({:?})", c.info.name, c.info.backend)),
            "checks": checks.iter().map(|(n, (d, g, x))| serde_json::json!({"backend": n, "max_dist_err_m": d, "max_grad_err": g, "verdicts_differing": x})).collect::<Vec<_>>(),
            "results": rows.iter().map(|r| serde_json::json!({"backend": r.backend, "batch": r.batch, "seconds": r.time.as_secs_f64(), "configs_per_second": r.per_second()})).collect::<Vec<_>>(),
        });
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("\nwrote {path}");
    }
}

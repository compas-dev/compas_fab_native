//! Free-space planning on the MotionBenchMaker problems shipped with VAMP, reported like
//! VAMP's `scripts/evaluate_mbm.py` (same statistics, same problems, same robot spheres).
//!
//! Usage: mbm_bench <fixture.json> <problems.json> [--range 1.0] [--out paths.json] [--trials 1]
//!
//! Single-threaded. With `--out`, writes every path for `scripts/vamp_check_paths.py`.

use std::time::{Duration, Instant};

use motion_bench::mbm;
use motion_core::check::Checker;
use motion_core::freespace::{plan, FreeSpaceSettings};

fn percentiles(label: &str, unit: &str, mut v: Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return format!("| {label} | – | – | – | – | – |");
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let p = |q: f64| v[((n - 1) as f64 * q).round() as usize];
    format!(
        "| {label} ({unit}) | {mean:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
        p(0.25),
        p(0.5),
        p(0.95),
        v[n - 1]
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = mbm::robot(&args[0]);
    let problems = mbm::load(&args[1]);
    let mut settings = FreeSpaceSettings::default();
    let mut out_path = None;
    let mut trials = 1;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--range" => settings.range = args[i + 1].parse().unwrap(),
            "--out" => out_path = Some(args[i + 1].clone()),
            "--trials" => trials = args[i + 1].parse().unwrap(),
            "--no-simplify" => {
                settings.simplify_rounds = 0;
                i -= 1;
            }
            f => panic!("unknown flag {f}"),
        }
        i += 2;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let (mut planning, mut simplification, mut total, mut iterations) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut initial_cost, mut cost, mut setup) = (Vec::new(), Vec::new(), Vec::new());
    let (mut solved, mut valid) = (0, 0);
    let mut paths = Vec::new();
    let mut failures = Vec::new();
    let wall = Instant::now();
    pool.install(|| {
        for p in &problems {
            if !p.valid {
                continue;
            }
            valid += 1;
            for trial in 0..trials {
                let t = Instant::now();
                let checker = Checker::new(&model, p.obstacles.clone());
                setup.push(t.elapsed().as_secs_f64() * 1e6);
                settings.seed = trial as u64 * 100_003;
                match plan(&checker, &p.start, &p.goals, &settings) {
                    Ok(result) => {
                        solved += 1;
                        let s = &result.stats;
                        planning.push(s.planning.as_secs_f64() * 1e6);
                        simplification.push(s.simplification.as_secs_f64() * 1e6);
                        total.push((s.planning + s.simplification).as_secs_f64() * 1e6);
                        iterations.push(s.iterations as f64);
                        initial_cost.push(s.initial_cost as f64);
                        cost.push(s.cost as f64);
                        if trial == 0 {
                            paths.push(serde_json::json!({
                                "scenario": p.scenario, "index": p.index, "path": result.path,
                            }));
                        }
                    }
                    Err(e) => failures.push(format!("{} #{}: {e:?}", p.scenario, p.index)),
                }
            }
        }
    });
    let wall = wall.elapsed();

    println!("{}: {solved} solved / {valid} valid problems x {trials} trials (range {}, resolution {}), single thread",
        model.name, settings.range, settings.resolution);
    println!("\n| | mean | 25% | 50% | 95% | max |\n|---|---:|---:|---:|---:|---:|");
    println!("{}", percentiles("Planning", "µs", planning));
    println!("{}", percentiles("Simplification", "µs", simplification));
    println!("{}", percentiles("Total", "µs", total));
    println!("{}", percentiles("Planning iterations", "#", iterations));
    println!("{}", percentiles("Checker setup", "µs", setup));
    println!("{}", percentiles("Initial cost", "L2", initial_cost));
    println!("{}", percentiles("Simplified cost", "L2", cost));
    println!(
        "\nwall time for all problems: {:.1} ms",
        wall.as_secs_f64() * 1e3
    );
    for f in failures.iter().take(10) {
        println!("  failed: {f}");
    }
    if let Some(path) = out_path {
        std::fs::write(path, serde_json::to_string(&paths).unwrap()).unwrap();
    }
    let _ = Duration::ZERO;
}

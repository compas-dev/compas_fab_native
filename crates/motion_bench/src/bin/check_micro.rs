//! Microbenchmark of the yes/no checker: time per configuration for collides_lanes
//! (FK + checks) on one MotionBenchMaker scene.
//!
//! Usage: check_micro <fixture.json> <problems.json>

use std::time::Instant;

use motion_bench::mbm;
use motion_core::check::{Checker, LANES};
use motion_core::sampling::random_configurations;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = mbm::robot(&args[0]);
    let problems = mbm::load(&args[1]);
    let dof = model.dof();
    let n = 8 * 4096;
    let q = random_configurations(&model, n, 3);
    {
        let checker = Checker::new(&model, Default::default());
        let blocks: Vec<Vec<[f32; LANES]>> = (0..n / LANES)
            .map(|k| {
                (0..dof)
                    .map(|j| std::array::from_fn(|l| q[j * n + k * LANES + l] as f32))
                    .collect()
            })
            .collect();
        let mut best = f64::INFINITY;
        let mut acc = 0.0;
        for _ in 0..7 {
            let t = Instant::now();
            for b in &blocks {
                acc += checker.fk_only_lanes(b);
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{:<14} {:.1} ns per configuration ({:.0} ns per LANES-lane FK) [{acc:.0}]",
            "FK only",
            best / n as f64 * 1e9,
            best / (n / LANES) as f64 * 1e9
        );
    }
    for (label, obstacles) in [
        ("no obstacles", Default::default()),
        ("scene", problems[0].obstacles.clone()),
    ] {
        let checker = Checker::new(&model, obstacles);
        let blocks: Vec<Vec<[f32; LANES]>> = (0..n / LANES)
            .map(|k| {
                (0..dof)
                    .map(|j| std::array::from_fn(|l| q[j * n + k * LANES + l] as f32))
                    .collect()
            })
            .collect();
        let mut best = f64::INFINITY;
        let mut hits = 0;
        for _ in 0..7 {
            let t = Instant::now();
            hits = 0;
            for b in &blocks {
                hits += checker.collides_lanes(b) as usize;
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{label:<14} {:.1} ns per configuration ({:.0} ns per LANES-lane check), {hits}/{} blocks collide",
            best / n as f64 * 1e9,
            best / (n / LANES) as f64 * 1e9,
            n / LANES
        );
    }
}

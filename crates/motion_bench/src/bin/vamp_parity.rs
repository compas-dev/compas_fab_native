//! Writes our collision verdicts and sphere positions for random configurations on the
//! MotionBenchMaker scenes, for `scripts/vamp_parity.py` to compare with VAMP's own.
//!
//! Usage: vamp_parity <fixture.json> <problems.json> <out.json>

use motion_bench::mbm;
use motion_core::check::Checker;
use motion_core::sampling::random_configurations;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = mbm::robot(&args[0]);
    let problems = mbm::load(&args[1]);
    let n = 500;
    let q = random_configurations(&model, n, 7);
    let dof = model.dof();
    let configs: Vec<Vec<f32>> = (0..n)
        .map(|b| (0..dof).map(|j| q[j * n + b] as f32).collect())
        .collect();
    let mut cases = Vec::new();
    let mut seen = std::collections::HashMap::new();
    for p in &problems {
        let count = seen.entry(p.scenario.clone()).or_insert(0);
        *count += 1;
        if *count > 3 {
            continue;
        }
        let checker = Checker::new(&model, p.obstacles.clone());
        let mut probe = configs.clone();
        probe.push(p.start.clone());
        probe.extend(p.goals.iter().cloned());
        let verdicts: Vec<bool> = probe.iter().map(|c| checker.collides(c)).collect();
        cases.push(serde_json::json!({
            "scenario": p.scenario, "index": p.index, "configs": probe, "collides": verdicts,
        }));
    }
    let empty = Checker::new(&model, Default::default());
    let spheres: Vec<_> = configs[..20]
        .iter()
        .map(|c| empty.sphere_positions(c))
        .collect();
    let out =
        serde_json::json!({"cases": cases, "sphere_configs": &configs[..20], "spheres": spheres});
    std::fs::write(&args[2], serde_json::to_string(&out).unwrap()).unwrap();
    println!("{} scenes x {} configurations", cases.len(), n + 2);
}

use motion_core::ik::MAX_SOLUTIONS;
use motion_core::reference::forward_kinematics;
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, RobotModel};

/// UR5 (offset wrist, ported closed form) and two spherical-wrist arms (closed form derived
/// from the model).
const ROBOTS: [&str; 3] = ["ur5", "abb_irb4600_40_255", "staubli_tx2_60l"];

fn robot(name: &str) -> RobotModel {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    load_robot(&std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap())
        .unwrap()
        .0
}

fn max_diff(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0 * std::f64::consts::PI);
    d.min(2.0 * std::f64::consts::PI - d)
}

#[test]
fn closed_form_chain_matches_model_fk() {
    for name in ROBOTS {
        let model = robot(name);
        let ik = model.analytic_ik.as_ref().unwrap();
        let n = 500;
        let q = random_configurations(&model, n, 11);
        let mut worst = 0.0f64;
        for b in 0..n {
            let qb: [f64; 6] = std::array::from_fn(|j| q[j * n + b]);
            let model_frame = forward_kinematics(&model, &qb)[model.end_effector_link];
            worst = worst.max(max_diff(&ik.forward(&qb), &model_frame));
        }
        assert!(worst < 1e-9, "{name}: analytic FK vs model FK: {worst:e}");
    }
}

#[test]
fn every_solution_reaches_the_target_and_the_source_config_is_found() {
    for name in ROBOTS {
        let model = robot(name);
        let ik = model.analytic_ik.as_ref().unwrap();
        let n = 2000;
        let q = random_configurations(&model, n, 12);
        let (mut worst, mut total, mut found) = (0.0f64, 0usize, 0usize);
        let mut missed = Vec::new();
        for b in 0..n {
            let qb: [f64; 6] = std::array::from_fn(|j| q[j * n + b]);
            let target = forward_kinematics(&model, &qb)[model.end_effector_link];
            let solutions = ik.solve(&target, &model.joint_limits);
            assert!(solutions.len() <= MAX_SOLUTIONS);
            total += solutions.len();
            for s in &solutions {
                for (&v, &[lo, hi]) in s.iter().zip(&model.joint_limits) {
                    assert!(v >= lo - 1e-12 && v <= hi + 1e-12);
                }
                worst = worst.max(max_diff(
                    &forward_kinematics(&model, s)[model.end_effector_link],
                    &target,
                ));
            }
            // At a wrist singularity (sin q5 ~ 0) only q4 + q6 is determined: infinitely many
            // configurations reach the target, and IK returns valid ones, not necessarily the source.
            // Near an elbow singularity (arm stretched) neighbouring configurations reach the same
            // pose to 1e-10, so a solution within a few mrad of the source counts as recovering it.
            let singular = qb[4].sin().abs() < 1e-2;
            let tol = 1e-5;
            let near = |s: &[f64; 6], t: f64| (0..6).all(|j| angle_diff(s[j], qb[j]) < t);
            if singular
                || solutions.iter().any(|s| near(s, tol))
                || solutions.iter().any(|s| near(s, 1e-2))
            {
                found += 1;
            } else {
                let closest = solutions
                    .iter()
                    .map(|s| (0..6).map(|j| angle_diff(s[j], qb[j])).fold(0.0, f64::max))
                    .fold(f64::INFINITY, f64::min);
                missed.push((qb, closest));
            }
        }
        println!("{name}: {n} targets, {total} solutions, worst FK residual {worst:e}, source config recovered for {found}/{n}");
        for (m, closest) in missed.iter().take(3) {
            println!("  missed {m:?}, closest solution {closest:.2e} rad away");
        }
        assert!(worst < 1e-8, "{name}: IK solution FK residual {worst:e}");
        assert_eq!(
            found, n,
            "{name}: source configuration not recovered (away from wrist singularities)"
        );
    }
}

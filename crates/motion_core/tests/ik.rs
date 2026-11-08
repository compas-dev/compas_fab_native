use motion_core::ik::MAX_SOLUTIONS;
use motion_core::reference::forward_kinematics;
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, RobotModel};

fn ur5() -> RobotModel {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    load_robot(&std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap())
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
    let model = ur5();
    let ik = model.analytic_ik.as_ref().unwrap();
    let n = 500;
    let q = random_configurations(&model, n, 11);
    let mut worst = 0.0f64;
    for b in 0..n {
        let qb: [f64; 6] = std::array::from_fn(|j| q[j * n + b]);
        let model_frame = forward_kinematics(&model, &qb)[model.end_effector_link];
        worst = worst.max(max_diff(&ik.forward(&qb), &model_frame));
    }
    assert!(worst < 1e-9, "analytic FK vs model FK: {worst:e}");
}

#[test]
fn every_solution_reaches_the_target_and_the_source_config_is_found() {
    let model = ur5();
    let ik = model.analytic_ik.as_ref().unwrap();
    let n = 2000;
    let q = random_configurations(&model, n, 12);
    let (mut worst, mut total, mut found) = (0.0f64, 0usize, 0usize);
    for b in 0..n {
        let qb: [f64; 6] = std::array::from_fn(|j| q[j * n + b]);
        let target = forward_kinematics(&model, &qb)[model.end_effector_link];
        let solutions = ik.solve(&target, &model.joint_limits);
        assert!(solutions.len() <= MAX_SOLUTIONS);
        total += solutions.len();
        for s in &solutions {
            for j in 0..6 {
                let [lo, hi] = model.joint_limits[j];
                assert!(s[j] >= lo && s[j] <= hi);
            }
            worst = worst.max(max_diff(
                &forward_kinematics(&model, s)[model.end_effector_link],
                &target,
            ));
        }
        // At a wrist singularity (sin q5 ~ 0) only q4 + q6 is determined: infinitely many
        // configurations reach the target, and IK returns valid ones, not necessarily the source.
        let singular = qb[4].sin().abs() < 1e-2;
        if singular
            || solutions
                .iter()
                .any(|s| (0..6).all(|j| angle_diff(s[j], qb[j]) < 1e-5))
        {
            found += 1;
        }
    }
    println!("{n} targets, {total} solutions, worst FK residual {worst:e}, source config recovered for {found}/{n}");
    assert!(worst < 1e-8, "IK solution FK residual {worst:e}");
    assert_eq!(
        found, n,
        "source configuration not recovered (away from wrist singularities)"
    );
}

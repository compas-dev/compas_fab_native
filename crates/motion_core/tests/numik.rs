//! Numerical IK: every solution reaches its target (checked by the model's FK), stays in
//! the joint limits, and solutions are distinct.

use motion_core::model::{Mat34, Scene};
use motion_core::numik::{IkGoal, IkSettings, Solver};
use motion_core::reference::forward_kinematics;
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, RobotModel};

fn robot(name: &str) -> RobotModel {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    load_robot(&std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap())
        .unwrap()
        .0
}

const IDENTITY: Mat34 = motion_core::model::IDENTITY;

fn pose_gap(a: &Mat34, b: &Mat34) -> (f64, f64) {
    let p = ((a[3] - b[3]).powi(2) + (a[7] - b[7]).powi(2) + (a[11] - b[11]).powi(2)).sqrt();
    let r = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        .iter()
        .map(|&k| (a[k] - b[k]).abs())
        .fold(0.0, f64::max);
    (p, r)
}

fn within_limits(model: &RobotModel, q: &[f64]) -> bool {
    q.iter()
        .zip(&model.joint_limits)
        .all(|(v, [lo, hi])| *v >= lo - 1e-12 && *v <= hi + 1e-12)
}

#[test]
fn seven_axis_panda_reaches_random_poses() {
    let model = robot("vamp_panda");
    let scene = Scene::default();
    let solver = Solver::new(&model, &scene);
    let n = 40;
    let q = random_configurations(&model, n, 3);
    let settings = IkSettings {
        collision_weight: 0.0,
        ..Default::default()
    };
    let mut solved = 0;
    let mut total = 0;
    for b in 0..n {
        let qb: Vec<f64> = (0..model.dof()).map(|j| q[j * n + b]).collect();
        let target = forward_kinematics(&model, &qb)[model.end_effector_link];
        let goal = IkGoal {
            target,
            tcf: IDENTITY,
            free_axis: false,
        };
        let solutions = solver.solve(&goal, &[], &settings, &|_| true);
        if !solutions.is_empty() {
            solved += 1;
        }
        total += solutions.len();
        for s in &solutions {
            let (p, r) = pose_gap(
                &forward_kinematics(&model, s)[model.end_effector_link],
                &target,
            );
            assert!(p < 1e-6 && r < 1e-5, "pose gap {p} m, {r}");
            assert!(within_limits(&model, s));
        }
        for (i, a) in solutions.iter().enumerate() {
            for c in &solutions[i + 1..] {
                let d = a
                    .iter()
                    .zip(c)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                assert!(d > settings.distinct);
            }
        }
    }
    // Every target is reachable by construction.
    assert_eq!(solved, n, "solved {solved} of {n}");
    assert!(total >= n * 4, "only {total} solutions for {n} targets");
}

#[test]
fn free_axis_keeps_only_the_tool_axis() {
    let model = robot("vamp_panda");
    let scene = Scene::default();
    let solver = Solver::new(&model, &scene);
    let n = 20;
    let q = random_configurations(&model, n, 5);
    let settings = IkSettings {
        collision_weight: 0.0,
        max_results: 4,
        ..Default::default()
    };
    for b in 0..n {
        let qb: Vec<f64> = (0..model.dof()).map(|j| q[j * n + b]).collect();
        let mut target = forward_kinematics(&model, &qb)[model.end_effector_link];
        // Rotate the target about its own z: the free-axis solve must not care.
        let (s, c) = 1.1f64.sin_cos();
        let x = [target[0], target[4], target[8]];
        let y = [target[1], target[5], target[9]];
        for k in 0..3 {
            target[k * 4] = c * x[k] + s * y[k];
            target[k * 4 + 1] = -s * x[k] + c * y[k];
        }
        let goal = IkGoal {
            target,
            tcf: IDENTITY,
            free_axis: true,
        };
        let solutions = solver.solve(&goal, std::slice::from_ref(&qb), &settings, &|_| true);
        assert!(!solutions.is_empty());
        for s in &solutions {
            let f = forward_kinematics(&model, s)[model.end_effector_link];
            let (p, _) = pose_gap(&f, &target);
            let z = f[2] * target[2] + f[6] * target[6] + f[10] * target[10];
            assert!(p < 1e-6 && z > 1.0 - 1e-9, "gap {p} m, z·z {z}");
        }
        // Seeded at an exact solution, the first answer is that seed (it already converged).
        let d = solutions[0]
            .iter()
            .zip(&qb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(d < 1e-6);
    }
}

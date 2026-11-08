use motion_core::check::{Checker, Obstacles, LANES};
use motion_core::cpu::collision_min_lanes;
use motion_core::freespace::{plan, FreeSpaceSettings};
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, load_scene, ModelBuffers, RobotModel};

fn fixture(name: &str) -> String {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    std::fs::read_to_string(format!("{dir}/{name}")).unwrap()
}

fn robot(name: &str) -> RobotModel {
    load_robot(&fixture(name)).unwrap().0
}

/// The yes/no checker and the distance kernel agree on every configuration whose clearance
/// is not within rounding of zero.
#[test]
fn checker_agrees_with_the_distance_kernel() {
    let mut model = robot("ur5.json");
    model.disable_always_colliding(500, 0.95, 7);
    let scene = load_scene(&fixture("scene_boxes.json")).unwrap();
    let buffers = ModelBuffers::new(&model, &scene);
    let mut obstacles = Obstacles::default();
    for c in &scene.cuboids {
        obstacles.add_box(&c.frame, c.half_extents);
    }
    let checker = Checker::new(&model, obstacles);
    let n = 4096;
    let q64 = random_configurations(&model, n, 5);
    let q: Vec<f32> = q64.iter().map(|&v| v as f32).collect();
    let mut min = vec![0.0f32; n];
    collision_min_lanes(&buffers, &q, &mut min);
    let (mut compared, mut colliding) = (0, 0);
    for b in 0..n {
        if min[b].abs() < 1e-5 {
            continue;
        }
        let config: Vec<f32> = (0..model.dof()).map(|j| q[j * n + b]).collect();
        assert_eq!(
            checker.collides(&config),
            min[b] < 0.0,
            "configuration {b}, clearance {}",
            min[b]
        );
        compared += 1;
        colliding += (min[b] < 0.0) as usize;
    }
    assert!(compared > n - 10 && colliding > 100 && colliding < compared - 100);
}

/// Lanes of one call agree with single-configuration calls.
#[test]
fn lanes_agree_with_single_configurations() {
    let model = robot("vamp_panda.json");
    let mut obstacles = Obstacles::default();
    obstacles.add_sphere([0.4, 0.0, 0.5], 0.15);
    obstacles.add_capsule([0.0, 0.4, 0.0], [0.0, 0.4, 1.0], 0.05);
    let checker = Checker::new(&model, obstacles);
    let n = LANES * 256;
    let q = random_configurations(&model, n, 9);
    let dof = model.dof();
    for block in 0..n / LANES {
        let lanes: Vec<[f32; LANES]> = (0..dof)
            .map(|j| std::array::from_fn(|l| q[j * n + block * LANES + l] as f32))
            .collect();
        let any = (0..LANES).any(|l| {
            let c: Vec<f32> = (0..dof).map(|j| lanes[j][l]).collect();
            checker.collides(&c)
        });
        assert_eq!(checker.collides_lanes(&lanes), any, "block {block}");
    }
}

/// A Panda reaching from one side of a wall to the other: the planner finds a path, and the
/// path is free when checked 16 times as densely as it was planned.
#[test]
fn panda_plans_around_a_wall() {
    let model = robot("vamp_panda.json");
    let mut obstacles = Obstacles::default();
    // A wall in front of the robot and a table under the workspace.
    let frame = |x: f64, y: f64, z: f64| [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z];
    obstacles.add_box(&frame(0.5, 0.0, 0.45), [0.02, 0.2, 0.45]);
    obstacles.add_box(&frame(0.5, 0.0, -0.05), [0.4, 0.6, 0.05]);
    let checker = Checker::new(&model, obstacles);
    let start = [-1.0, 0.6, 0.0, -1.6, 0.0, 2.2, 0.8];
    let goal = vec![1.0, 0.6, 0.0, -1.6, 0.0, 2.2, 0.8];
    assert!(!checker.collides(&start) && !checker.collides(&goal));
    assert!(
        !checker.motion_free(&start, &goal, 32.0),
        "the straight motion must hit the wall"
    );
    let result = plan(
        &checker,
        &start,
        std::slice::from_ref(&goal),
        &FreeSpaceSettings::default(),
    )
    .unwrap();
    let path = &result.path;
    assert_eq!(path[0], start.to_vec());
    assert_eq!(path[path.len() - 1], goal);
    for w in path.windows(2) {
        assert!(checker.motion_free(&w[0], &w[1], 512.0));
    }
    assert!(result.stats.cost <= result.stats.initial_cost + 1e-5);
}

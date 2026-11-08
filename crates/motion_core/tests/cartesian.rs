use motion_core::cartesian::{check_configs, plan_cartesian, CartesianRequest};
use motion_core::reference::forward_kinematics;
use motion_core::{load_robot, load_scene, ModelBuffers};

/// Flange pointing down (z = -Z world), x along world X.
fn down(x: f64, y: f64, z: f64) -> [f64; 12] {
    [1.0, 0.0, 0.0, x, 0.0, -1.0, 0.0, y, 0.0, 0.0, -1.0, z]
}

pub fn line_over_table(n: usize) -> Vec<[f64; 12]> {
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            down(0.45, -0.3 + 0.6 * t, 0.50)
        })
        .collect()
}

#[test]
fn line_over_table_is_planned_continuous_and_collision_free() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let (mut model, _) =
        load_robot(&std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap()).unwrap();
    let scene =
        load_scene(&std::fs::read_to_string(format!("{dir}/scene_boxes.json")).unwrap()).unwrap();
    model.disabled_collisions.sort();
    let buffers = ModelBuffers::new(&model, &scene);
    let req = CartesianRequest {
        waypoints: line_over_table(100),
        free_axis_samples: 12,
        max_joint_step: 0.3,
        ..Default::default()
    };
    let res = plan_cartesian(&model, &buffers, &req).expect("path");
    let s = &res.stats;
    println!(
        "{} waypoints: {} IK nodes, {} free, {} edges, {} edge configs, {} rounds; ik {:?} nodes {:?} search {:?} edges {:?} total {:?}",
        req.waypoints.len(), s.ik_nodes, s.free_nodes, s.edges_evaluated, s.edge_configs_checked, s.search_rounds,
        s.t_ik, s.t_node_checks, s.t_search, s.t_edge_checks, s.t_total
    );
    assert_eq!(res.path.len(), req.waypoints.len());
    for (i, (q, wp)) in res.path.iter().zip(&req.waypoints).enumerate() {
        let f = forward_kinematics(&model, q)[model.end_effector_link];
        // Position and tool axis must match; rotation about the tool axis is free.
        let pos =
            ((f[3] - wp[3]).powi(2) + (f[7] - wp[7]).powi(2) + (f[11] - wp[11]).powi(2)).sqrt();
        let z_dot = f[2] * wp[2] + f[6] * wp[6] + f[10] * wp[10];
        assert!(
            pos < 1e-8 && z_dot > 1.0 - 1e-9,
            "waypoint {i}: pos err {pos:e}, z·z {z_dot}"
        );
        if i > 0 {
            let step = (0..6)
                .map(|j| (q[j] - res.path[i - 1][j]).abs())
                .fold(0.0, f64::max);
            assert!(step <= req.max_joint_step + 1e-12, "jump {step} at {i}");
        }
    }
    // Dense re-check of the whole path.
    let mut dense: Vec<[f64; 6]> = Vec::new();
    for w in res.path.windows(2) {
        for k in 0..20 {
            let f = k as f64 / 20.0;
            dense.push(std::array::from_fn(|j| w[0][j] + f * (w[1][j] - w[0][j])));
        }
    }
    let clear = check_configs(&buffers, &dense);
    let worst = clear.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "path cost {:.3} rad, min clearance along the path {worst:.4} m",
        res.cost
    );
    assert!(worst > 0.0);
}

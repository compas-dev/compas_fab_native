use motion_core::cpu::{query_lanes, query_scalar, BatchOutput};
use motion_core::reference::{distance_query, forward_kinematics};
use motion_core::sampling::random_configurations;
use motion_core::{load_robot, load_scene, ModelBuffers};

fn fixtures() -> (
    motion_core::RobotModel,
    motion_core::model::ReferenceFk,
    motion_core::Scene,
) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let robot = std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap();
    let scene = std::fs::read_to_string(format!("{dir}/scene_boxes.json")).unwrap();
    let (model, reference) = load_robot(&robot).unwrap();
    (model, reference, load_scene(&scene).unwrap())
}

#[test]
fn reference_fk_matches_compas_robots() {
    let (model, reference, _) = fixtures();
    let mut worst = 0.0f64;
    for (q, frames) in reference.configs.iter().zip(&reference.link_frames) {
        let ours = forward_kinematics(&model, q);
        for (a, b) in ours.iter().zip(frames) {
            for k in 0..12 {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
    }
    assert!(worst < 1e-9, "max deviation from compas_robots: {worst:e}");
}

#[test]
fn f32_kernels_match_f64_reference() {
    let (model, _, scene) = fixtures();
    let buffers = ModelBuffers::new(&model, &scene);
    // Layout sorts spheres by link; the exporter already emits them in that order.
    assert!(buffers
        .spheres
        .iter()
        .zip(&model.spheres)
        .all(|(a, b)| a.link == b.link && (a.radius - b.radius as f32).abs() < 1e-7));

    let batch = 1024;
    let q64 = random_configurations(&model, batch, 7);
    let q32: Vec<f32> = q64.iter().map(|&v| v as f32).collect();
    let n = buffers.spheres.len();
    let mut scalar = BatchOutput::new(batch, n);
    let mut lanes = BatchOutput::new(batch, n);
    query_scalar(&buffers, &q32, &mut scalar);
    query_lanes(&buffers, &q32, &mut lanes);

    let (mut dist_err, mut grad_err, mut lane_err) = (0.0f64, 0.0f64, 0.0f32);
    let mut collisions = 0;
    for b in 0..batch {
        // Reference on the f32-rounded configuration, so only kernel arithmetic differs.
        let q: Vec<f64> = (0..model.dof())
            .map(|j| q32[j * batch + b] as f64)
            .collect();
        let r = distance_query(&model, &scene, &q);
        collisions += (r.min < 0.0) as usize;
        dist_err = dist_err.max((r.min - scalar.config_min[b] as f64).abs());
        for s in 0..n {
            let i = s * batch + b;
            dist_err = dist_err.max((r.dist[s] - scalar.sphere_dist[i] as f64).abs());
            // Gradients are discontinuous where two boxes (or two box faces) are equidistant; skip near-ties.
            let g = scalar.sphere_grad[i];
            let e = (0..3)
                .map(|k| (r.grad[s][k] - g[k] as f64).abs())
                .fold(0.0, f64::max);
            if e > 1e-3 {
                let alt = (0..3).map(|k| g[k] as f64 * r.grad[s][k]).sum::<f64>();
                assert!(
                    alt > 0.5 || r.dist[s].abs() < 1e-3,
                    "gradient mismatch at config {b} sphere {s}: {e}"
                );
            } else {
                grad_err = grad_err.max(e);
            }
            lane_err = lane_err.max((scalar.sphere_dist[i] - lanes.sphere_dist[i]).abs());
        }
        lane_err = lane_err.max((scalar.config_min[b] - lanes.config_min[b]).abs());
    }
    println!("collisions: {collisions}/{batch}, dist err {dist_err:e} m, grad err {grad_err:e}, lanes vs scalar {lane_err:e}");
    assert!(dist_err < 1e-5, "f32 distance error {dist_err:e} m");
    assert!(lane_err < 1e-5, "lanes vs scalar {lane_err:e}");
    assert!(
        collisions > batch / 10 && collisions < batch * 9 / 10,
        "scene should split configs, got {collisions}/{batch}"
    );
}

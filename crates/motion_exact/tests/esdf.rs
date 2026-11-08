use std::time::Instant;

use motion_core::reference::point_box_distance;
use motion_core::sampling::Rng;
use motion_core::{load_scene, Scene};
use motion_exact::build_esdf;

/// The scene's boxes as closed, outward-oriented meshes (obstacle mesh JSON).
pub fn boxes_as_meshes(scene: &Scene) -> String {
    let mut obstacles = Vec::new();
    for c in &scene.cuboids {
        let h = c.half_extents;
        let m = &c.frame;
        let mut vertices = Vec::new();
        for k in 0..8 {
            let l = [
                if k & 1 == 0 { -h[0] } else { h[0] },
                if k & 2 == 0 { -h[1] } else { h[1] },
                if k & 4 == 0 { -h[2] } else { h[2] },
            ];
            for r in 0..3 {
                vertices.push(
                    m[r * 4] * l[0] + m[r * 4 + 1] * l[1] + m[r * 4 + 2] * l[2] + m[r * 4 + 3],
                );
            }
        }
        // Outward-facing triangles of a box with corners indexed by (x, y, z) bits.
        let faces: Vec<u32> = vec![
            0, 2, 1, 1, 2, 3, // -z
            4, 5, 6, 5, 7, 6, // +z
            0, 1, 4, 1, 5, 4, // -y
            2, 6, 3, 3, 6, 7, // +y
            0, 4, 2, 2, 4, 6, // -x
            1, 3, 5, 3, 7, 5, // +x
        ];
        obstacles.push(serde_json::json!({"name": c.name, "touch_links": [], "vertices": vertices, "faces": faces}));
    }
    serde_json::json!({ "obstacles": obstacles }).to_string()
}

#[test]
fn esdf_matches_exact_box_distances() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let scene =
        load_scene(&std::fs::read_to_string(format!("{dir}/scene_boxes.json")).unwrap()).unwrap();
    let (voxel, truncation) = (0.02f32, 0.3f32);
    let t = Instant::now();
    let esdf = build_esdf(&boxes_as_meshes(&scene), voxel, truncation).unwrap();
    println!(
        "ESDF: {} blocks ({:.1} MB), built in {:?}",
        esdf.n_blocks(),
        esdf.pool.len() as f64 * 4.0 / 1e6,
        t.elapsed()
    );

    let mut rng = Rng::new(3);
    let mut errors = Vec::new();
    for _ in 0..200_000 {
        let p = [
            rng.next_f64() * 2.4 - 1.0,
            rng.next_f64() * 2.4 - 1.2,
            rng.next_f64() * 1.6 - 0.1,
        ];
        let exact = scene
            .cuboids
            .iter()
            .map(|c| point_box_distance(p, &c.frame, c.half_extents).0)
            .fold(f64::INFINITY, f64::min);
        if exact.abs() > (truncation - 2.0 * voxel) as f64 {
            continue;
        }
        let (d, _) = esdf.sample([p[0] as f32, p[1] as f32, p[2] as f32]);
        errors.push((d as f64 - exact).abs());
    }
    errors.sort_by(f64::total_cmp);
    let pct = |q: f64| errors[((errors.len() - 1) as f64 * q) as usize];
    println!(
        "{} samples in the band: median error {:.4} m, p99 {:.4} m, max {:.4} m (voxel {voxel} m, half diagonal {:.4} m)",
        errors.len(),
        pct(0.5),
        pct(0.99),
        pct(1.0),
        voxel * 3f32.sqrt() / 2.0
    );
    assert!(pct(1.0) <= (voxel * 3f32.sqrt() / 2.0) as f64 + 1e-4);
}

#[test]
fn esdf_obstacles_give_the_same_verdicts_as_boxes() {
    use motion_core::cpu::collision_min_lanes;
    use motion_core::sampling::random_configurations;
    use motion_core::{load_robot, ModelBuffers};
    use std::sync::Arc;

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let (model, _) =
        load_robot(&std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap()).unwrap();
    let boxes =
        load_scene(&std::fs::read_to_string(format!("{dir}/scene_boxes.json")).unwrap()).unwrap();
    let voxel = 0.02f32;
    let esdf = Arc::new(build_esdf(&boxes_as_meshes(&boxes), voxel, 0.3).unwrap());

    let batch = 4096;
    let q: Vec<f32> = random_configurations(&model, batch, 5)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let run = |scene: &Scene| {
        let buffers = ModelBuffers::new(&model, scene);
        let mut out = vec![0.0f32; batch];
        let t = Instant::now();
        for _ in 0..10 {
            collision_min_lanes(&buffers, &q, &mut out);
        }
        (out, t.elapsed() / 10)
    };
    let (reference, t_boxes) = run(&boxes);
    for margin in [0.0, voxel * 3f32.sqrt() / 2.0] {
        let scene = Scene {
            cuboids: Vec::new(),
            sphere_touch: Vec::new(),
            esdf: Some(esdf.clone()),
            esdf_margin: margin,
        };
        let (out, t_esdf) = run(&scene);
        let missed = reference
            .iter()
            .zip(&out)
            .filter(|(r, e)| **r <= 0.0 && **e > 0.0)
            .count();
        let extra = reference
            .iter()
            .zip(&out)
            .filter(|(r, e)| **r > 0.0 && **e <= 0.0)
            .count();
        let collide = reference.iter().filter(|r| **r <= 0.0).count();
        println!(
            "margin {margin:.4}: {collide} colliding with boxes; ESDF misses {missed}, adds {extra}; batch time boxes {t_boxes:?}, ESDF {t_esdf:?}"
        );
        if margin > 0.0 {
            assert_eq!(
                missed, 0,
                "with the half-diagonal margin the ESDF must not miss box collisions"
            );
        }
    }
}

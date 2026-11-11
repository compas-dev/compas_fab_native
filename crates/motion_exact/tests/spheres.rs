//! Sphere fitting on shapes with known answers: boxes (inside test, inscribed spheres,
//! coverage), meshes with mixed winding, open meshes and stray slivers.

use motion_core::sampling::Rng;
use motion_exact::spheres::{
    covering_spheres, fit_sphere_sets, surface_points, FitOptions, ObstacleSpheres, Solid, Triangle,
};

/// An axis-aligned box as 12 outward triangles.
fn cuboid(center: [f64; 3], size: [f64; 3]) -> Vec<Triangle> {
    let corner = |i: usize| -> [f64; 3] {
        std::array::from_fn(|k| center[k] + size[k] * (((i >> k) & 1) as f64 - 0.5))
    };
    // Each face as a quad of corner indices, counter-clockwise seen from outside.
    let quads = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    quads
        .iter()
        .flat_map(|q| {
            [
                [corner(q[0]), corner(q[1]), corner(q[2])],
                [corner(q[0]), corner(q[2]), corner(q[3])],
            ]
        })
        .collect()
}

/// Signed distance to an axis-aligned box from inside (positive), exact.
fn depth_in_box(p: &[f64; 3], center: [f64; 3], size: [f64; 3]) -> f64 {
    (0..3)
        .map(|k| size[k] / 2.0 - (p[k] - center[k]).abs())
        .fold(f64::INFINITY, f64::min)
}

const CENTER: [f64; 3] = [0.1, -0.2, 0.3];
const SIZE: [f64; 3] = [0.3, 0.2, 0.12];

#[test]
fn signed_distance_of_a_box() {
    let solid = Solid::from_triangles(&cuboid(CENTER, SIZE)).expect("a box is a solid");
    let mut rng = Rng::new(1);
    for _ in 0..500 {
        let p: [f64; 3] = std::array::from_fn(|k| CENTER[k] + 0.5 * (rng.next_f64() - 0.5));
        let expected = depth_in_box(&p, CENTER, SIZE);
        let got = solid.signed_distance(&p);
        assert_eq!(
            got > 0.0,
            expected > 0.0,
            "inside test at {p:?}: {got} vs {expected}"
        );
        if expected > 0.0 {
            assert!(
                (got - expected).abs() < 1e-5,
                "depth at {p:?}: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn inscribed_spheres_stay_inside_and_cover_the_surface() {
    let tris = cuboid(CENTER, SIZE);
    let options = FitOptions::default();
    let sets = fit_sphere_sets(&tris, 7, &options);
    assert!(!sets.self_spheres.is_empty());
    assert!(sets.self_spheres.len() <= options.max_self_spheres);
    for (c, r) in &sets.self_spheres {
        assert!(
            depth_in_box(c, CENTER, SIZE) >= r - 1e-5,
            "sphere {c:?} r={r} pokes out"
        );
    }
    // Obstacle spheres are the self spheres grown by the padding.
    assert_eq!(sets.obstacle.len(), sets.self_spheres.len());
    for ((c0, r0), (c1, r1)) in sets.obstacle.iter().zip(&sets.self_spheres) {
        assert_eq!(c0, c1);
        assert!((r0 - r1 - options.obstacle_padding).abs() < 1e-12);
    }
    // 16 spheres cannot reach into a box's corners, but most of the surface is covered.
    assert!(sets.miss < 0.05, "miss {}", sets.miss);
}

#[test]
fn the_same_seed_gives_the_same_spheres() {
    let tris = cuboid(CENTER, SIZE);
    let options = FitOptions::default();
    assert_eq!(
        fit_sphere_sets(&tris, 3, &options),
        fit_sphere_sets(&tris, 3, &options)
    );
}

#[test]
fn mixed_winding_is_turned_outward() {
    let mut tris = cuboid(CENTER, SIZE);
    for t in tris.iter_mut().step_by(3) {
        t.swap(1, 2);
    }
    let solid = Solid::from_triangles(&tris).expect("still a closed, orientable mesh");
    assert!(solid.signed_distance(&CENTER) > 0.05);
    assert!(solid.signed_distance(&[1.0, 1.0, 1.0]) < 0.0);
    // All faces inverted: inside is still inside.
    let inverted: Vec<Triangle> = cuboid(CENTER, SIZE)
        .into_iter()
        .map(|[a, b, c]| [a, c, b])
        .collect();
    let solid = Solid::from_triangles(&inverted).unwrap();
    assert!(solid.signed_distance(&CENTER) > 0.05);
}

#[test]
fn separate_parts_and_slivers() {
    let other = [0.6, 0.0, 0.0];
    let mut tris = cuboid(CENTER, SIZE);
    tris.extend(cuboid(other, [0.1, 0.1, 0.1]));
    // A stray triangle, and one that collapses to a line.
    tris.push([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0]]);
    tris.push([[3.0, 0.0, 0.0], [3.1, 0.0, 0.0], [3.0, 0.0, 0.0]]);
    let solid = Solid::from_triangles(&tris).expect("two boxes, the strays ignored");
    assert!(solid.signed_distance(&CENTER) > 0.0);
    assert!(solid.signed_distance(&other) > 0.0);
    assert!(solid.signed_distance(&[0.35, 0.0, 0.0]) < 0.0); // between them
}

#[test]
fn closed_parts_touching_along_an_edge_stay_separate() {
    // Two cubes sharing a vertical edge: four faces on it. Each cube is closed on its own.
    let (a, b) = ([0.0, 0.0, 0.0], [0.1, 0.1, 0.0]);
    let mut tris = cuboid(a, [0.1, 0.1, 0.1]);
    tris.extend(cuboid(b, [0.1, 0.1, 0.1]));
    let solid = Solid::from_triangles(&tris).expect("two closed parts");
    assert!((solid.signed_distance(&a) - 0.05).abs() < 1e-5);
    assert!((solid.signed_distance(&b) - 0.05).abs() < 1e-5);
    assert!(solid.signed_distance(&[0.1, 0.0, 0.0]) < 0.0);
    let sets = fit_sphere_sets(&tris, 1, &FitOptions::default());
    for (c, r) in &sets.self_spheres {
        let depth = depth_in_box(c, a, [0.1; 3]).max(depth_in_box(c, b, [0.1; 3]));
        assert!(depth >= r - 1e-5, "sphere {c:?} r={r} pokes out");
    }
}

#[test]
fn open_and_flat_meshes_fall_back_to_covering_spheres() {
    let mut open = cuboid(CENTER, SIZE);
    open.pop();
    assert!(Solid::from_triangles(&open).is_none());
    let flat = vec![[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]];
    assert!(Solid::from_triangles(&flat).is_none());

    let sets = fit_sphere_sets(&open, 5, &FitOptions::default());
    assert!(!sets.self_spheres.is_empty());
    assert_eq!(sets.obstacle, sets.self_spheres);
    // Covering spheres enclose every vertex.
    for t in &open {
        for p in t {
            let inside = sets.obstacle.iter().any(|(c, r)| {
                let d: f64 = (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>().sqrt();
                d <= r + 1e-12
            });
            assert!(inside, "vertex {p:?} outside every covering sphere");
        }
    }
    assert_eq!(sets.miss, 0.0);
}

#[test]
fn covering_spheres_on_request() {
    let tris = cuboid(CENTER, SIZE);
    let options = FitOptions {
        obstacle_spheres: ObstacleSpheres::Covering,
        max_spheres: 4,
        ..FitOptions::default()
    };
    let sets = fit_sphere_sets(&tris, 2, &options);
    assert!(sets.obstacle.len() <= 4);
    assert_ne!(sets.obstacle, sets.self_spheres);
    let mut rng = Rng::new(4);
    for p in surface_points(&tris, 500, &mut rng) {
        let d = sets
            .obstacle
            .iter()
            .map(|(c, r)| (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>().sqrt() - r)
            .fold(f64::INFINITY, f64::min);
        // Clusters enclose their samples; points between samples stay close.
        assert!(d < 0.01, "surface point {p:?} is {d} outside");
    }
}

#[test]
fn covering_spheres_count_follows_the_extent() {
    let mut rng = Rng::new(9);
    let points: Vec<[f64; 3]> = (0..400).map(|i| [i as f64 * 0.001, 0.0, 0.0]).collect();
    // 0.4 m long, spheres of about 0.05 m radius: 4 clusters.
    assert_eq!(covering_spheres(&points, 0.05, 16, &mut rng).len(), 4);
    assert_eq!(covering_spheres(&points, 0.05, 2, &mut rng).len(), 2);
}

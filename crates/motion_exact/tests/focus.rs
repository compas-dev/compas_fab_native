//! The convex-hull check of `check_focus` contains the mesh check of `check`: hulls contain
//! their meshes (up to parry's hull tolerance, which `HULL_MARGIN` covers), so every mesh
//! contact is a hull contact.

use motion_core::load_robot;
use motion_core::sampling::random_configurations;
use motion_exact::ExactChecker;

#[test]
fn hull_check_contains_the_mesh_check() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
    let (model, _) =
        load_robot(&std::fs::read_to_string(format!("{dir}/ur5.json")).unwrap()).unwrap();
    let meshes = std::fs::read_to_string(format!("{dir}/ur5_meshes.json")).unwrap();
    let mut checker = ExactChecker::new(model.clone(), &meshes).unwrap();
    let all: Vec<usize> = (0..model.links.len()).collect();
    checker.set_focus(&all);
    let n = 3000;
    let q = random_configurations(&model, n, 9);
    let (mut contacts, mut missed) = (0, Vec::new());
    for b in 0..n {
        let qb: Vec<f64> = (0..model.dof()).map(|j| q[j * n + b]).collect();
        if let Some(c) = checker.check(&qb) {
            contacts += 1;
            if !checker.check_focus(&qb) {
                missed.push((c, qb));
            }
        }
    }
    assert!(contacts > 100, "only {contacts} contacts");
    assert!(
        missed.is_empty(),
        "{} of {contacts} mesh contacts not seen by the hulls, e.g. {:?}",
        missed.len(),
        missed.first()
    );
}

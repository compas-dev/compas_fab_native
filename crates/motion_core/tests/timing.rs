use motion_core::timing::{time_parameterize, TimingSettings};

fn settings(dof: usize, v: f64, a: f64, blend: f64) -> TimingSettings {
    TimingSettings {
        max_velocity: vec![v; dof],
        max_acceleration: vec![a; dof],
        blend_tolerance: blend,
        grid_step: 0.005,
        time_step: 0.004,
    }
}

fn free(_: &[f64], _: &[f64]) -> bool {
    true
}

/// One joint, one segment: the textbook trapezoid (and triangle) durations.
#[test]
fn single_joint_profiles_are_time_optimal() {
    let s = settings(1, 1.0, 2.0, 0.0);
    // Long move: accelerate to v, cruise, brake: T = d / v + v / a.
    let t = time_parameterize(&[vec![0.0], vec![2.0]], &s, free).unwrap();
    assert!((t.duration - 2.5).abs() < 0.01, "{}", t.duration);
    // Short move, never reaching v: T = 2 sqrt(d / a).
    let t = time_parameterize(&[vec![0.0], vec![0.1]], &s, free).unwrap();
    let expected = 2.0 * (0.1f64 / 2.0).sqrt();
    assert!(
        (t.duration - expected).abs() / expected < 0.02,
        "{}",
        t.duration
    );
}

fn path6() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, -0.5, 0.3, 0.0, 0.2, 0.0],
        vec![1.2, 0.4, -0.6, 0.5, 0.2, -0.3],
        vec![0.2, 0.9, -0.2, 0.5, -0.4, 0.4],
    ]
}

/// A blended 6-joint path: limits hold, the motion starts and ends at rest at the path's
/// ends, positions are continuous, and blends stay within the tolerance of their vertex.
#[test]
fn limits_continuity_and_blend_tolerance() {
    let v = vec![1.0, 1.2, 1.5, 2.0, 2.0, 2.5];
    let a = vec![2.0, 2.0, 3.0, 4.0, 4.0, 5.0];
    let s = TimingSettings {
        max_velocity: v.clone(),
        max_acceleration: a.clone(),
        blend_tolerance: 0.1,
        grid_step: 0.005,
        time_step: 0.004,
    };
    let path = path6();
    let t = time_parameterize(&path, &s, free).unwrap();
    assert_eq!(t.positions[0], path[0]);
    assert_eq!(t.positions[t.positions.len() - 1], path[path.len() - 1]);
    assert!(t.velocities[0]
        .iter()
        .chain(&t.velocities[t.velocities.len() - 1])
        .all(|&x| x == 0.0));
    for (vel, acc) in t.velocities.iter().zip(&t.accelerations) {
        for j in 0..6 {
            assert!(
                vel[j].abs() <= v[j] * 1.001,
                "velocity {} > {}",
                vel[j],
                v[j]
            );
            assert!(
                acc[j].abs() <= a[j] * 1.02,
                "acceleration {} > {}",
                acc[j],
                a[j]
            );
        }
    }
    for w in t.positions.windows(2) {
        for j in 0..6 {
            assert!((w[1][j] - w[0][j]).abs() <= v[j] * s.time_step * 1.01);
        }
    }
    // Blends: the closest sample to each interior vertex is within the tolerance (+ sampling).
    for vertex in &path[1..path.len() - 1] {
        let d = t
            .positions
            .iter()
            .map(|p| {
                p.iter()
                    .zip(vertex)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(d <= 0.1 + 0.01, "blend deviates {d}");
        assert!(d > 1e-3, "no blend at the vertex");
    }
    assert!(t.blends.iter().all(|&b| b == 0.1));
}

/// Blends that the collision check rejects shrink and finally become stops: the robot then
/// passes exactly through the vertex, at rest.
#[test]
fn rejected_blends_become_stops() {
    let s = settings(6, 1.0, 2.0, 0.1);
    let path = path6();
    let t = time_parameterize(&path, &s, |_, _| false).unwrap();
    assert!(t.blends.iter().all(|&b| b == 0.0));
    assert!(t.stops.iter().all(|&s| s));
    for vertex in &path[1..path.len() - 1] {
        let (k, d) = t
            .positions
            .iter()
            .enumerate()
            .map(|(k, p)| {
                (
                    k,
                    p.iter()
                        .zip(vertex)
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                )
            })
            .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
        assert!(d < 1e-3, "vertex missed by {d}");
        let speed = t.velocities[k].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(speed < 0.05, "passes the vertex at {speed}");
    }
    // Blending saves time.
    let blended = time_parameterize(&path, &s, free).unwrap();
    assert!(blended.duration < t.duration);
}

/// Collinear interior vertices (path simplification leaves midpoints like these) are passed
/// straight through: no stop, and no degenerate blend (the turning angle of two equal unit
/// directions used to come out a few 1e-8 rad, giving a NaN arc).
#[test]
fn collinear_vertices_are_passed_through() {
    let a = [-1.32124, -0.68326, 0.44585, -0.56866, 1.03846, 0.33684];
    let b = [-2.539, -2.71, 1.114, -3.095, 1.573, -1.492];
    let mid: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
    let third: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + (y - x) / 3.0).collect();
    let straight = vec![a.to_vec(), b.to_vec()];
    let split = vec![a.to_vec(), third, mid, b.to_vec()];
    for blend in [0.05, 0.0] {
        let s = settings(6, 1.0, 2.0, blend);
        for blend_free in [true, false] {
            let one = time_parameterize(&straight, &s, |_, _| blend_free).unwrap();
            let many = time_parameterize(&split, &s, |_, _| blend_free).unwrap();
            assert!(
                (one.duration - many.duration).abs() < 1e-3,
                "{} vs {}",
                one.duration,
                many.duration
            );
            assert!(many.positions.iter().flatten().all(|x| x.is_finite()));
            assert!(many.stops.iter().all(|&s| !s));
        }
    }
}

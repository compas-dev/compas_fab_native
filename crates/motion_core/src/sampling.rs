//! Deterministic configuration sampling (no `rand` dependency, reproducible across runs and platforms).

use crate::model::RobotModel;

/// SplitMix64: tiny, fast, good enough for uniform test configurations.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `batch` random configurations within the joint limits, in SoA layout (`q[j * batch + b]`), f64.
pub fn random_configurations(model: &RobotModel, batch: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let dof = model.dof();
    let mut q = vec![0.0; dof * batch];
    for b in 0..batch {
        for (j, [lo, hi]) in model.joint_limits.iter().enumerate() {
            q[j * batch + b] = lo + (hi - lo) * rng.next_f64();
        }
    }
    q
}

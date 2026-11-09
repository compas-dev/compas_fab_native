//! Python bindings for the native motion core (spike).
//!
//! A thin layer: the robot model and scene come in as JSON (built on the Python side from
//! a compas_fab RobotCell, see `compas_fab.backends.native`), queries run in Rust with the
//! GIL released. Transforms cross as flat lists of 12 floats (3x4, row-major).

use std::collections::HashMap;
use std::f64::consts::PI;

use motion_core::cartesian::{
    check_configs, plan_cartesian_tracked, plan_cartesian_with, CartesianError, CartesianRequest,
    CartesianStats, Config,
};
use motion_core::check::{Checker, ExtraCheck, Obstacles};
use motion_core::freespace::{self, FreeSpaceError, FreeSpaceSettings};
use motion_core::model::Mat34;
use motion_core::numik::{IkGoal, IkSettings, Solver, SolverData};
use motion_core::reference::forward_kinematics;
use motion_core::timing::{time_parameterize, TimingSettings};
use motion_core::{load_robot, load_scene, ModelBuffers, RobotModel, Scene};
use motion_exact::ExactChecker;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// (path, cost, stats) returned by `plan_cartesian`.
type CartesianOutput = (Vec<Vec<f64>>, f64, HashMap<String, f64>);
/// (path, stats) returned by `plan_motion`.
type MotionOutput = (Vec<Vec<f64>>, HashMap<String, f64>);

fn mat34(v: &[f64]) -> PyResult<Mat34> {
    v.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected 12 values (3x4 transform), got {}",
            v.len()
        ))
    })
}

fn check_dof(model: &RobotModel, v: &[f64]) -> PyResult<()> {
    if v.len() == model.dof() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "expected {} joint values, got {}",
            model.dof(),
            v.len()
        )))
    }
}

/// Upper bound on how far any robot sphere can be from the root: the chain length to its
/// link plus its offset and radius. Used to pick a motion-check resolution in metres.
fn reach(model: &RobotModel) -> f64 {
    let mut chain = vec![0.0f64; model.links.len()];
    for (i, l) in model.links.iter().enumerate() {
        if l.parent >= 0 {
            let t = (l.origin[3].powi(2) + l.origin[7].powi(2) + l.origin[11].powi(2)).sqrt();
            chain[i] = chain[l.parent as usize] + t;
        }
    }
    model
        .spheres
        .iter()
        .map(|s| {
            chain[s.link as usize]
                + (s.center[0].powi(2) + s.center[1].powi(2) + s.center[2].powi(2)).sqrt()
                + s.radius
        })
        .fold(0.0, f64::max)
}

#[pyclass(module = "compas_fab_native")]
struct NativeCore {
    model: RobotModel,
    scene: Scene,
    buffers: ModelBuffers,
    auto_disable_samples: usize,
    auto_touch: Vec<(String, String, usize)>,
    exact: Option<std::sync::Arc<ExactChecker>>,
    /// Links the sphere model approximates poorly (`exact_links` of the robot JSON): with
    /// exact meshes set, every collision check also tests them on their meshes.
    exact_links: Vec<usize>,
    esdf: Option<(std::sync::Arc<motion_core::esdf::VoxelEsdf>, f32)>,
    /// Yes/no checker for free-space planning, rebuilt when the scene changes.
    checker: Option<Checker>,
    /// Links as loaded, before passive joint values are baked in.
    rest_links: Vec<motion_core::model::Link>,
    passive: HashMap<String, f64>,
    /// Numerical IK precomputation, rebuilt when the model or the scene changes.
    ik_data: Option<std::sync::Arc<SolverData>>,
}

/// Configurations along a joint-space path, no joint moving more than `step` between two.
fn densify(path: &[Vec<f64>], step: f64) -> Vec<Vec<f64>> {
    let mut out = vec![path[0].clone()];
    for w in path.windows(2) {
        let jump = w[0]
            .iter()
            .zip(&w[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let n = (jump / step).ceil().max(1.0) as usize;
        for k in 1..=n {
            let t = k as f64 / n as f64;
            out.push(
                w[0].iter()
                    .zip(&w[1])
                    .map(|(a, b)| a + t * (b - a))
                    .collect(),
            );
        }
    }
    out
}

/// The model with every sphere (obstacle and self) grown by `margin` metres.
fn inflate_spheres(model: &RobotModel, margin: f64) -> RobotModel {
    let mut m = model.clone();
    for s in m.spheres.iter_mut().chain(m.self_spheres.iter_mut()) {
        s.radius += margin;
    }
    m
}

impl NativeCore {
    /// Numerical IK data for the current model and scene (built on first use).
    fn ik_data(&mut self) -> std::sync::Arc<SolverData> {
        if self.ik_data.is_none() {
            self.ik_data = Some(std::sync::Arc::new(SolverData::new(
                &self.model,
                &self.scene,
            )));
        }
        self.ik_data.clone().unwrap()
    }

    fn checker(&mut self) -> &Checker {
        if self.checker.is_none() {
            let checker =
                Checker::new(&self.model, Obstacles::from_scene(&self.model, &self.scene));
            self.checker = Some(match self.focus_check() {
                Some(extra) => checker.with_extra(extra),
                None => checker,
            });
        }
        self.checker.as_ref().unwrap()
    }

    /// The exact test of the poorly approximated links, for `Checker::with_extra`.
    fn focus_check(&self) -> Option<ExtraCheck> {
        let exact = self.exact.as_ref().filter(|e| e.has_focus())?.clone();
        Some(std::sync::Arc::new(move |q: &[f32]| {
            exact.check_focus(&q.iter().map(|&x| x as f64).collect::<Vec<f64>>())
        }))
    }

    /// Clearances of the sphere model, set to 0 (contact) where a poorly approximated
    /// link collides on its exact mesh.
    fn clearances(&self, configs: &[Vec<f64>]) -> Vec<f32> {
        let mut clearance = check_configs(&self.buffers, configs);
        if let Some(exact) = self.exact.as_ref().filter(|e| e.has_focus()) {
            use rayon::prelude::*;
            let hits: Vec<bool> = configs
                .par_iter()
                .zip(&clearance)
                .map(|(q, &c)| c > 0.0 && exact.check_focus(q))
                .collect();
            for (c, hit) in clearance.iter_mut().zip(hits) {
                if hit {
                    *c = 0.0;
                }
            }
        }
        clearance
    }

    /// The exact checker, for changing it: the checker hooks that share it are dropped.
    fn exact_mut(&mut self) -> Option<&mut ExactChecker> {
        self.checker = None;
        self.exact.as_mut().and_then(std::sync::Arc::get_mut)
    }

    fn load_scene(&mut self, scene_json: &str) -> PyResult<()> {
        let mut scene = load_scene(scene_json).map_err(PyValueError::new_err)?;
        if let Some((esdf, margin)) = &self.esdf {
            scene.esdf = Some(esdf.clone());
            scene.esdf_margin = *margin;
        }
        self.auto_touch = if self.auto_disable_samples > 0 {
            scene.auto_touch_spheres(&self.model, self.auto_disable_samples, 0.95, 8)
        } else {
            Vec::new()
        };
        self.buffers = ModelBuffers::new(&self.model, &scene);
        self.scene = scene;
        self.checker = None;
        self.ik_data = None;
        let scene = self.scene.clone();
        if let Some(exact) = self.exact_mut() {
            exact.set_focus_obstacles(&scene);
        }
        Ok(())
    }
}

#[pymethods]
impl NativeCore {
    /// `auto_disable_samples` > 0 disables self-collision checks between link pairs whose
    /// self spheres overlap in at least 95% of that many random configurations (MoveIt's
    /// "always in collision"), e.g. an attached tool against the wrist it is mounted on.
    #[new]
    #[pyo3(signature = (robot_json, scene_json, auto_disable_samples = 1000))]
    fn new(robot_json: &str, scene_json: &str, auto_disable_samples: usize) -> PyResult<Self> {
        let (mut model, _) = load_robot(robot_json).map_err(PyValueError::new_err)?;
        let exact_links: Vec<usize> = serde_json::from_str::<serde_json::Value>(robot_json)
            .ok()
            .and_then(|v| v.get("exact_links").cloned())
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        if auto_disable_samples > 0 {
            model.disable_always_colliding(auto_disable_samples, 0.95, 7);
        }
        let buffers = ModelBuffers::new(&model, &Scene::default());
        let mut core = Self {
            model,
            scene: Scene::default(),
            buffers,
            auto_disable_samples,
            auto_touch: Vec::new(),
            exact: None,
            exact_links,
            esdf: None,
            checker: None,
            rest_links: Vec::new(),
            passive: HashMap::new(),
            ik_data: None,
        };
        core.rest_links = core.model.links.clone();
        core.load_scene(scene_json)?;
        Ok(core)
    }

    /// (box, link, sphere) exemptions: obstacle spheres overlapping a box in (almost) every
    /// configuration, treated as touching it.
    #[getter]
    fn auto_touch_spheres(&self) -> Vec<(String, String, usize)> {
        self.auto_touch.clone()
    }

    /// Link pairs excluded from self-collision checks (SRDF + automatically disabled).
    #[getter]
    fn disabled_collisions(&self) -> Vec<(String, String)> {
        let name = |i: usize| self.model.links[i].name.clone();
        self.model
            .disabled_collisions
            .iter()
            .map(|&[a, b]| (name(a), name(b)))
            .collect()
    }

    /// Replace the obstacles (same JSON format as the scene fixtures).
    fn set_scene(&mut self, scene_json: &str) -> PyResult<()> {
        self.load_scene(scene_json)
    }

    #[getter]
    fn dof(&self) -> usize {
        self.model.dof()
    }

    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.model.joint_names.clone()
    }

    #[getter]
    fn link_names(&self) -> Vec<String> {
        self.model.links.iter().map(|l| l.name.clone()).collect()
    }

    #[getter]
    fn end_effector_link(&self) -> usize {
        self.model.end_effector_link
    }

    #[getter]
    fn sphere_counts(&self) -> (usize, usize, usize) {
        (
            self.buffers.spheres.len(),
            self.buffers.self_spheres.len(),
            self.buffers.cuboids.len(),
        )
    }

    /// World frames (robot coordinates) of every link.
    fn link_frames(&self, q: Vec<f64>) -> Vec<Vec<f64>> {
        forward_kinematics(&self.model, &q)
            .iter()
            .map(|m| m.to_vec())
            .collect()
    }

    /// All closed-form IK solutions for the end-effector link at `frame` (robot coordinates).
    fn inverse_kinematics(&self, frame: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let ik = self
            .model
            .analytic_ik
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no closed-form IK for this robot"))?;
        let target = mat34(&frame)?;
        Ok(ik
            .solve(&target, &self.model.joint_limits)
            .iter()
            .map(|q| q.to_vec())
            .collect())
    }

    /// True if the robot has a closed-form IK solver (`inverse_kinematics`).
    #[getter]
    fn has_analytic_ik(&self) -> bool {
        self.model.analytic_ik.is_some()
    }

    /// Numerical IK (any chain): distinct solutions for the TCF at `frame` (robot
    /// coordinates), `tcf` relative to the end-effector link. With `free_axis`, any
    /// rotation about the TCF z axis is a solution. `seeds` are tried first (e.g. the
    /// current configuration), then random ones. With `check_collision`, the solve steers
    /// clear of collisions and only configurations with clearance above `margin` are
    /// returned.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        frame,
        tcf = None,
        free_axis = false,
        seeds = Vec::new(),
        max_results = 8,
        max_seeds = 64,
        seed = 0,
        check_collision = true,
        margin = 0.0,
    ))]
    fn numeric_inverse_kinematics(
        &mut self,
        py: Python<'_>,
        frame: Vec<f64>,
        tcf: Option<Vec<f64>>,
        free_axis: bool,
        seeds: Vec<Vec<f64>>,
        max_results: usize,
        max_seeds: usize,
        seed: u64,
        check_collision: bool,
        margin: f32,
    ) -> PyResult<Vec<Vec<f64>>> {
        for q in &seeds {
            check_dof(&self.model, q)?;
        }
        let goal = IkGoal {
            target: mat34(&frame)?,
            tcf: match tcf {
                Some(t) => mat34(&t)?,
                None => motion_core::model::IDENTITY,
            },
            free_axis,
        };
        let settings = IkSettings {
            max_results,
            max_seeds: max_seeds.max(seeds.len()),
            seed,
            collision_weight: if check_collision { 1.0 } else { 0.0 },
            ..Default::default()
        };
        let data = self.ik_data();
        let buffers = &self.buffers;
        let focus = self.exact.clone().filter(|e| e.has_focus());
        let solver = Solver::with_data(&self.model, &self.scene, data);
        Ok(py.detach(|| {
            let accept = |q: &[f64]| {
                !check_collision
                    || (check_configs(buffers, &[q])[0] > margin
                        && !focus.as_ref().is_some_and(|e| e.check_focus(q)))
            };
            solver.solve(&goal, &seeds, &settings, &accept)
        }))
    }

    /// Enable the exact final pass: collision meshes per link (link coordinates) and obstacle
    /// meshes (robot coordinates), as produced by `compas_fab.backends.native.model`.
    fn set_exact_meshes(
        &mut self,
        robot_meshes_json: &str,
        obstacle_meshes_json: &str,
    ) -> PyResult<()> {
        let mut checker = ExactChecker::new(self.model.clone(), robot_meshes_json)
            .map_err(PyValueError::new_err)?;
        checker
            .set_obstacle_meshes(obstacle_meshes_json)
            .map_err(PyValueError::new_err)?;
        checker.set_focus(&self.exact_links);
        checker.set_focus_obstacles(&self.scene);
        self.exact = Some(std::sync::Arc::new(checker));
        self.checker = None;
        Ok(())
    }

    /// Obstacle meshes (robot coordinates) as a signed distance field for the sphere search,
    /// in addition to the boxes of the scene. Returns (blocks, seconds to build).
    #[pyo3(signature = (obstacle_meshes_json, voxel = 0.02, truncation = 0.3, margin = 0.01))]
    fn set_esdf_obstacles(
        &mut self,
        py: Python<'_>,
        obstacle_meshes_json: &str,
        voxel: f32,
        truncation: f32,
        margin: f32,
    ) -> PyResult<(usize, f64)> {
        let t = std::time::Instant::now();
        let esdf = py
            .detach(|| motion_exact::build_esdf(obstacle_meshes_json, voxel, truncation))
            .map_err(PyValueError::new_err)?;
        let blocks = esdf.n_blocks();
        self.esdf = Some((std::sync::Arc::new(esdf), margin));
        self.scene.esdf = self.esdf.as_ref().map(|(e, _)| e.clone());
        self.scene.esdf_margin = margin;
        self.buffers = ModelBuffers::new(&self.model, &self.scene);
        self.checker = None;
        self.ik_data = None;
        Ok((blocks, t.elapsed().as_secs_f64()))
    }

    /// Replace the obstacle meshes of the exact checker.
    fn set_exact_obstacles(&mut self, obstacle_meshes_json: &str) -> PyResult<()> {
        match self.exact_mut() {
            Some(c) => c
                .set_obstacle_meshes(obstacle_meshes_json)
                .map_err(PyValueError::new_err),
            None => Err(PyRuntimeError::new_err("exact meshes not set")),
        }
    }

    #[getter]
    fn has_exact(&self) -> bool {
        self.exact.is_some()
    }

    /// Exact collision of each configuration: None, or the colliding pair (names).
    fn exact_check(
        &self,
        py: Python<'_>,
        configs: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Option<(String, String)>>> {
        let checker = self
            .exact
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("exact meshes not set"))?;
        Ok(py.detach(|| {
            checker
                .check_many(&configs)
                .into_iter()
                .map(|c| c.map(|c| (c.a, c.b)))
                .collect()
        }))
    }

    /// Clearance (metres; negative = collision) of each configuration against the scene and itself.
    fn clearance(&self, py: Python<'_>, configs: Vec<Vec<f64>>) -> PyResult<Vec<f32>> {
        for c in &configs {
            check_dof(&self.model, c)?;
        }
        Ok(py.detach(|| self.clearances(&configs)))
    }

    /// Values (by joint name) of joints outside the planning group, e.g. gripper fingers;
    /// missing ones are held at zero. Returns True if anything changed.
    fn set_passive_joints(&mut self, values: HashMap<String, f64>) -> bool {
        if values == self.passive {
            return false;
        }
        self.model.set_passive_joints(&self.rest_links, &values);
        self.passive = values;
        self.buffers = ModelBuffers::new(&self.model, &self.scene);
        self.checker = None;
        self.ik_data = None;
        let links = self.model.links.clone();
        if let Some(exact) = self.exact_mut() {
            exact.set_links(links);
        }
        true
    }

    /// Time-parameterize a joint-space path under per-joint velocity and acceleration
    /// limits: straight segments with circular blends of at most `blend_tolerance` at the
    /// corners (blends that collide on the sphere model shrink, down to a stop), timed with
    /// TOPP-RA, starting and ending at rest. Returns a dict with `times`, `positions`,
    /// `velocities`, `accelerations` (sampled every `time_step`), `duration`, `blends`
    /// (tolerance used per interior vertex; 0 = no blend) and `stops` (whether the robot
    /// stops at each interior vertex; straight-through vertices need no stop).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, max_velocity, max_acceleration, blend_tolerance = 0.05, time_step = 0.01, grid_step = 0.005, resolution = None))]
    fn time_parameterize<'py>(
        &mut self,
        py: Python<'py>,
        path: Vec<Vec<f64>>,
        max_velocity: Vec<f64>,
        max_acceleration: Vec<f64>,
        blend_tolerance: f64,
        time_step: f64,
        grid_step: f64,
        resolution: Option<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        for q in &path {
            check_dof(&self.model, q)?;
        }
        let resolution = resolution.unwrap_or_else(|| self.default_resolution()) as f32;
        let settings = TimingSettings {
            max_velocity,
            max_acceleration,
            blend_tolerance,
            grid_step,
            time_step,
        };
        let checker = self.checker();
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let timed = py
            .detach(|| {
                time_parameterize(&path, &settings, |a, b| {
                    checker.motion_free(&f32s(a), &f32s(b), resolution)
                })
            })
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("times", timed.times)?;
        d.set_item("positions", timed.positions)?;
        d.set_item("velocities", timed.velocities)?;
        d.set_item("accelerations", timed.accelerations)?;
        d.set_item("duration", timed.duration)?;
        d.set_item("blends", timed.blends)?;
        d.set_item("stops", timed.stops)?;
        Ok(d)
    }

    /// Suggested motion-check resolution (checks per unit of joint-space distance): at least
    /// VAMP's 32, and enough that no robot sphere moves more than about 2 cm between checks
    /// of a single-joint motion.
    #[getter]
    fn default_resolution(&self) -> f64 {
        (reach(&self.model) / 0.02).ceil().max(32.0)
    }

    /// True if the straight joint-space motion from `a` to `b` is collision-free on the
    /// sphere model, checked at `resolution` points per unit of joint-space distance.
    #[pyo3(signature = (a, b, resolution = None))]
    fn motion_free(&mut self, a: Vec<f64>, b: Vec<f64>, resolution: Option<f64>) -> PyResult<bool> {
        check_dof(&self.model, &a)?;
        check_dof(&self.model, &b)?;
        let resolution = resolution.unwrap_or_else(|| self.default_resolution()) as f32;
        let checker = self.checker();
        let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        Ok(checker.motion_free(&f(&a), &f(&b), resolution))
    }

    /// Plan a collision-free joint-space path from `start` to any of `goals`: the straight
    /// motion first, then RRT-Connect and shortcut + B-spline simplification. With `exact`
    /// (and exact meshes set), the path is checked on the meshes every `exact_step` radians
    /// and re-planned with another seed (up to `attempts` times) if it touches.
    ///
    /// Returns (path, stats). Raises RuntimeError("start_in_collision" | "goals_in_collision"
    /// | "no_path" | "exact_collision ...").
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        start,
        goals,
        range = 1.0,
        resolution = None,
        max_iterations = 100_000,
        seed = 0,
        simplify = true,
        exact = true,
        exact_step = 0.02,
        attempts = 5,
    ))]
    fn plan_motion(
        &mut self,
        py: Python<'_>,
        start: Vec<f64>,
        goals: Vec<Vec<f64>>,
        range: f64,
        resolution: Option<f64>,
        max_iterations: usize,
        seed: u64,
        simplify: bool,
        exact: bool,
        exact_step: f64,
        attempts: usize,
    ) -> PyResult<MotionOutput> {
        check_dof(&self.model, &start)?;
        for g in &goals {
            check_dof(&self.model, g)?;
        }
        if goals.is_empty() {
            return Err(PyValueError::new_err("no goals"));
        }
        let resolution = resolution.unwrap_or_else(|| self.default_resolution());
        self.checker();
        let checker = self.checker.as_ref().unwrap();
        let exact_checker = if exact { self.exact.as_deref() } else { None };
        let focus = self.focus_check();
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let (start32, goals32): (Vec<f32>, Vec<Vec<f32>>) =
            (f32s(&start), goals.iter().map(|g| f32s(g)).collect());
        py.detach(|| {
            let mut stats = HashMap::new();
            let mut exact_rejections = 0usize;
            for attempt in 0..attempts.max(1) {
                let settings = FreeSpaceSettings {
                    range: range as f32,
                    resolution: resolution as f32,
                    max_iterations,
                    max_samples: max_iterations,
                    simplify_rounds: if simplify { 5 } else { 0 },
                    seed: seed + attempt as u64 * 100_003,
                    ..Default::default()
                };
                // After an exact rejection, plan on inflated spheres: the sphere model passed
                // a path that grazes the meshes, and re-seeding alone tends to find the same
                // path (always so when the straight motion is free). If the inflated spheres
                // put the start or every goal in collision, fall back to the plain ones.
                let inflated = (exact_rejections > 0).then(|| {
                    let margin = [0.005, 0.01, 0.02, 0.03][(exact_rejections - 1).min(3)];
                    let model = inflate_spheres(&self.model, margin);
                    let c = Checker::new(&model, Obstacles::from_scene(&model, &self.scene));
                    match &focus {
                        Some(f) => c.with_extra(f.clone()),
                        None => c,
                    }
                });
                let mut result = match &inflated {
                    Some(c) => freespace::plan(c, &start32, &goals32, &settings),
                    None => freespace::plan(checker, &start32, &goals32, &settings),
                };
                if inflated.is_some()
                    && matches!(
                        result,
                        Err(FreeSpaceError::StartInCollision | FreeSpaceError::AllGoalsInCollision)
                    )
                {
                    result = freespace::plan(checker, &start32, &goals32, &settings);
                }
                stats.insert(
                    "inflation".to_string(),
                    if inflated.is_some() {
                        [0.005, 0.01, 0.02, 0.03][(exact_rejections - 1).min(3)]
                    } else {
                        0.0
                    },
                );
                let plan = match result {
                    Ok(plan) => plan,
                    Err(FreeSpaceError::StartInCollision) => {
                        return Err(PyRuntimeError::new_err("start_in_collision"))
                    }
                    Err(FreeSpaceError::AllGoalsInCollision) => {
                        return Err(PyRuntimeError::new_err("goals_in_collision"))
                    }
                    Err(FreeSpaceError::NoPath(s)) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "no_path after {} iterations",
                            s.iterations
                        )))
                    }
                };
                // Back to f64; the endpoints are the caller's own values, not f32 copies.
                let mut path: Vec<Vec<f64>> = plan
                    .path
                    .iter()
                    .map(|q| q.iter().map(|&x| x as f64).collect())
                    .collect();
                path[0] = start.clone();
                let last = path.len() - 1;
                let goal = goals
                    .iter()
                    .min_by(|a, b| {
                        let d = |g: &Vec<f64>| {
                            g.iter()
                                .zip(&path[last])
                                .map(|(x, y)| (x - y).abs())
                                .fold(0.0, f64::max)
                        };
                        d(a).total_cmp(&d(b))
                    })
                    .unwrap();
                path[last] = goal.clone();
                let s = &plan.stats;
                stats.insert("iterations".to_string(), s.iterations as f64);
                stats.insert("planning_us".to_string(), s.planning.as_secs_f64() * 1e6);
                stats.insert(
                    "simplification_us".to_string(),
                    s.simplification.as_secs_f64() * 1e6,
                );
                stats.insert("initial_cost".to_string(), s.initial_cost as f64);
                stats.insert("cost".to_string(), s.cost as f64);
                stats.insert("attempts".to_string(), (attempt + 1) as f64);
                if let Some(ex) = exact_checker {
                    let t = std::time::Instant::now();
                    let dense = densify(&path, exact_step);
                    let contact = ex.check_many(&dense).into_iter().position(|c| c.is_some());
                    stats.insert("exact_configs_checked".to_string(), dense.len() as f64);
                    stats.insert("exact_us".to_string(), t.elapsed().as_secs_f64() * 1e6);
                    if contact.is_some() {
                        exact_rejections += 1;
                        stats.insert("exact_rejections".to_string(), exact_rejections as f64);
                        continue;
                    }
                }
                stats.insert("exact_rejections".to_string(), exact_rejections as f64);
                return Ok((path, stats));
            }
            Err(PyRuntimeError::new_err(format!(
                "exact_collision: every one of {attempts} paths touched the exact meshes"
            )))
        })
    }

    /// Plan a Cartesian path through TCF `waypoints` (robot coordinates; `tcf` relative to
    /// the end-effector link). With the closed-form IK: layered graph search over IK
    /// branches and `free_axis_samples` rotations about the tool axis. With numerical IK
    /// (`numeric`, default: when there is no closed form or with `free_axis`): `tracks`
    /// solutions of the first waypoint (from `seeds` first) are followed along the path,
    /// the rotation about the tool axis left free with `free_axis`.
    /// Returns (path, cost, stats).
    #[pyo3(signature = (
        waypoints,
        tcf = None,
        free_axis_samples = 1,
        free_axis_range = PI,
        start = None,
        max_joint_step = 0.5,
        edge_check_step = 0.05,
        margin = 0.0,
        exact = true,
        free_axis = false,
        numeric = None,
        tracks = 8,
        seed = 0,
        seeds = Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn plan_cartesian(
        &mut self,
        py: Python<'_>,
        waypoints: Vec<Vec<f64>>,
        tcf: Option<Vec<f64>>,
        free_axis_samples: usize,
        free_axis_range: f64,
        start: Option<Vec<f64>>,
        max_joint_step: f64,
        edge_check_step: f64,
        margin: f32,
        exact: bool,
        free_axis: bool,
        numeric: Option<bool>,
        tracks: usize,
        seed: u64,
        seeds: Vec<Vec<f64>>,
    ) -> PyResult<CartesianOutput> {
        for q in &seeds {
            check_dof(&self.model, q)?;
        }
        let req = CartesianRequest {
            waypoints: waypoints
                .iter()
                .map(|w| mat34(w))
                .collect::<PyResult<_>>()?,
            tcf: match tcf {
                Some(t) => mat34(&t)?,
                None => motion_core::model::IDENTITY,
            },
            free_axis_samples,
            free_axis_range,
            start,
            max_joint_step,
            edge_check_step,
            margin,
        };
        if let Some(start) = &req.start {
            check_dof(&self.model, start)?;
        }
        // Numerical IK when asked, or when there is no closed form; also for a continuous
        // free axis, which the closed form can only sample.
        let numeric = numeric.unwrap_or(self.model.analytic_ik.is_none() || free_axis);
        if !numeric && self.model.analytic_ik.is_none() {
            return Err(PyRuntimeError::new_err("no closed-form IK for this robot"));
        }
        if !numeric && self.model.dof() != 6 {
            return Err(PyRuntimeError::new_err(
                "the closed-form Cartesian planner needs 6 joints",
            ));
        }
        let ik_data = if numeric { Some(self.ik_data()) } else { None };
        let checker = if exact { self.exact.as_deref() } else { None };
        let result: Result<(Vec<Vec<f64>>, f64, CartesianStats), CartesianError> = if numeric {
            let ik = IkSettings {
                seed,
                ..Default::default()
            };
            let validator = checker.map(|c| {
                move |configs: &[Vec<f64>]| -> Vec<bool> {
                    c.check_many(configs)
                        .into_iter()
                        .map(|r| r.is_some())
                        .collect()
                }
            });
            py.detach(|| {
                let v = validator
                    .as_ref()
                    .map(|f| f as &(dyn Fn(&[Vec<f64>]) -> Vec<bool> + Sync));
                let solver = Solver::with_data(&self.model, &self.scene, ik_data.clone().unwrap());
                plan_cartesian_tracked(
                    &self.model,
                    &solver,
                    &self.buffers,
                    &req,
                    free_axis,
                    tracks,
                    &seeds,
                    &ik,
                    v,
                )
                .map(|r| (r.path, r.cost, r.stats))
            })
        } else {
            let validator = checker.map(|c| {
                move |configs: &[Config]| -> Vec<bool> {
                    let qs: Vec<Vec<f64>> = configs.iter().map(|q| q.to_vec()).collect();
                    c.check_many(&qs).into_iter().map(|r| r.is_some()).collect()
                }
            });
            py.detach(|| {
                let v = validator
                    .as_ref()
                    .map(|f| f as &(dyn Fn(&[Config]) -> Vec<bool> + Sync));
                plan_cartesian_with(&self.model, &self.buffers, &req, v)
                    .map(|r| (r.path.iter().map(|q| q.to_vec()).collect(), r.cost, r.stats))
            })
        };
        match result {
            Ok((path, cost, s)) => {
                let stats = HashMap::from([
                    ("ik_nodes".to_string(), s.ik_nodes as f64),
                    ("free_nodes".to_string(), s.free_nodes as f64),
                    ("edges_evaluated".to_string(), s.edges_evaluated as f64),
                    ("edge_configs_checked".to_string(), s.edge_configs_checked as f64),
                    ("search_rounds".to_string(), s.search_rounds as f64),
                    ("seconds_ik".to_string(), s.t_ik.as_secs_f64()),
                    ("seconds_node_checks".to_string(), s.t_node_checks.as_secs_f64()),
                    ("seconds_search".to_string(), s.t_search.as_secs_f64()),
                    ("seconds_edge_checks".to_string(), s.t_edge_checks.as_secs_f64()),
                    ("exact_configs_checked".to_string(), s.exact_configs_checked as f64),
                    ("exact_rejections".to_string(), s.exact_rejections as f64),
                    ("seconds_exact".to_string(), s.t_exact.as_secs_f64()),
                    ("seconds_total".to_string(), s.t_total.as_secs_f64()),
                ]);
                Ok((path, cost, stats))
            }
            Err(CartesianError::NoIk { waypoint }) => {
                Err(PyRuntimeError::new_err(format!("no IK solution for waypoint {waypoint}")))
            }
            Err(CartesianError::NoCollisionFreeIk { waypoint }) => {
                Err(PyRuntimeError::new_err(format!("no collision-free IK solution for waypoint {waypoint}")))
            }
            Err(CartesianError::NoPath { .. }) => Err(PyRuntimeError::new_err(
                "no continuous collision-free path through the waypoints (joint step limit or collisions)",
            )),
        }
    }
}

#[pymodule]
fn compas_fab_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCore>()?;
    Ok(())
}

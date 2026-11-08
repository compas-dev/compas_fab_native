//! Python bindings for the native motion core (spike).
//!
//! A thin layer: the robot model and scene come in as JSON (built on the Python side from
//! a compas_fab RobotCell, see `compas_fab.backends.native`), queries run in Rust with the
//! GIL released. Transforms cross as flat lists of 12 floats (3x4, row-major).

use std::collections::HashMap;
use std::f64::consts::PI;

use motion_core::cartesian::{
    check_configs, plan_cartesian_with, CartesianError, CartesianRequest, Config,
};
use motion_core::model::Mat34;
use motion_core::reference::forward_kinematics;
use motion_core::{load_robot, load_scene, ModelBuffers, RobotModel, Scene};
use motion_exact::ExactChecker;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// (path, cost, stats) returned by `plan_cartesian`.
type CartesianOutput = (Vec<Vec<f64>>, f64, HashMap<String, f64>);

fn mat34(v: &[f64]) -> PyResult<Mat34> {
    v.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected 12 values (3x4 transform), got {}",
            v.len()
        ))
    })
}

fn config(v: &[f64]) -> PyResult<[f64; 6]> {
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("expected 6 joint values, got {}", v.len())))
}

#[pyclass(module = "compas_fab_native")]
struct NativeCore {
    model: RobotModel,
    scene: Scene,
    buffers: ModelBuffers,
    auto_disable_samples: usize,
    auto_touch: Vec<(String, String, usize)>,
    exact: Option<ExactChecker>,
}

impl NativeCore {
    fn load_scene(&mut self, scene_json: &str) -> PyResult<()> {
        let mut scene = load_scene(scene_json).map_err(PyValueError::new_err)?;
        self.auto_touch = if self.auto_disable_samples > 0 {
            scene.auto_touch_spheres(&self.model, self.auto_disable_samples, 0.95, 8)
        } else {
            Vec::new()
        };
        self.buffers = ModelBuffers::new(&self.model, &scene);
        self.scene = scene;
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
        };
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
        self.exact = Some(checker);
        Ok(())
    }

    /// Replace the obstacle meshes of the exact checker.
    fn set_exact_obstacles(&mut self, obstacle_meshes_json: &str) -> PyResult<()> {
        match self.exact.as_mut() {
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
        let configs: Vec<[f64; 6]> = configs.iter().map(|c| config(c)).collect::<PyResult<_>>()?;
        Ok(py.detach(|| check_configs(&self.buffers, &configs)))
    }

    /// Plan a Cartesian path. Returns (path, cost, stats).
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
    ))]
    #[allow(clippy::too_many_arguments)]
    fn plan_cartesian(
        &self,
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
    ) -> PyResult<CartesianOutput> {
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
        if self.model.analytic_ik.is_none() {
            return Err(PyRuntimeError::new_err("no closed-form IK for this robot"));
        }
        let checker = if exact { self.exact.as_ref() } else { None };
        let validator = checker.map(|c| {
            move |configs: &[Config]| -> Vec<bool> {
                let qs: Vec<Vec<f64>> = configs.iter().map(|q| q.to_vec()).collect();
                c.check_many(&qs).into_iter().map(|r| r.is_some()).collect()
            }
        });
        let result = py.detach(|| {
            let v = validator
                .as_ref()
                .map(|f| f as &(dyn Fn(&[Config]) -> Vec<bool> + Sync));
            plan_cartesian_with(&self.model, &self.buffers, &req, v)
        });
        match result {
            Ok(r) => {
                let s = r.stats;
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
                Ok((r.path.iter().map(|q| q.to_vec()).collect(), r.cost, stats))
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

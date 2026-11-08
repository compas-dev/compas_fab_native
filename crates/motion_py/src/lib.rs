//! Python bindings for the native motion core (spike).
//!
//! A thin layer: the robot model and scene come in as JSON (built on the Python side from
//! a compas_fab RobotCell, see `compas_fab.backends.native`), queries run in Rust with the
//! GIL released. Transforms cross as flat lists of 12 floats (3x4, row-major).

use std::collections::HashMap;
use std::f64::consts::PI;

use motion_core::cartesian::{check_configs, plan_cartesian, CartesianError, CartesianRequest};
use motion_core::model::Mat34;
use motion_core::reference::forward_kinematics;
use motion_core::{load_robot, load_scene, ModelBuffers, RobotModel, Scene};
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
        let result = py.detach(|| plan_cartesian(&self.model, &self.buffers, &req));
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

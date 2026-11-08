"""Native motion core as a compas_fab planner (spike).

``NativePlanner`` runs forward and inverse kinematics, collision checks and Cartesian
planning in the Rust core (``compas_fab_native``). User code switches to it by changing
the planner:

    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(robot_cell, robot_cell_state)
    trajectory = planner.plan_cartesian_motion(waypoints, start_state)

Limits of the spike: closed-form IK only (offset-wrist arms), static obstacles as
oriented bounding boxes, attached tools as spheres on the end-effector link, attached
workpieces ignored.
"""

import json
from typing import Optional

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation
from compas_robots import Configuration

from compas_fab.backends.exceptions import CollisionCheckError
from compas_fab.backends.exceptions import InverseKinematicsError
from compas_fab.backends.interfaces import ClientInterface
from compas_fab.backends.interfaces import PlannerInterface
from compas_fab.backends.interfaces.backend_features import CheckCollision
from compas_fab.backends.interfaces.backend_features import ForwardKinematics
from compas_fab.backends.interfaces.backend_features import InverseKinematics
from compas_fab.backends.interfaces.backend_features import PlanCartesianMotion
from compas_fab.backends.interfaces.backend_features import SetRobotCell
from compas_fab.backends.interfaces.backend_features import SetRobotCellState
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.robots import FrameTarget
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import JointTrajectory
from compas_fab.robots import JointTrajectoryPoint
from compas_fab.robots import TargetMode

from .model import add_attached_tools
from .model import exact_obstacle_meshes
from .model import exact_robot_meshes
from .model import frame_to_3x4
from .model import matrix_from_3x4
from .model import obstacle_boxes
from .model import robot_description

__all__ = ["NativeClient", "NativePlanner"]


def _native():
    try:
        import compas_fab_native
    except ImportError as e:
        raise ImportError("The native backend needs the compas_fab_native extension (native/crates/motion_py, built with cargo).") from e
    return compas_fab_native


def _frame_from_3x4(values) -> Frame:
    return Frame.from_transformation(Transformation.from_matrix(matrix_from_3x4(values).tolist()))


class NativeClient(ClientInterface):
    """Holds the robot cell, its state and the Rust core built from them."""

    def __init__(self):
        super(NativeClient, self).__init__()
        self.core = None
        self.group = None
        self._tool_key = None
        self._scene_key = None


class NativePlanner(
    SetRobotCell,
    SetRobotCellState,
    ForwardKinematics,
    InverseKinematics,
    CheckCollision,
    PlanCartesianMotion,
    PlannerInterface,
):
    """Planner backed by the native (Rust) motion core.

    Parameters
    ----------
    kinematics_solver
        compas_fab analytical solver of the robot (e.g. ``UR5Kinematics()``); its parameters
        are handed to the core's closed-form IK.
    sphere_options
        Sphere fitting options (see ``model.fit_sphere_sets``).
    exact
        Keep the exact collision meshes in the core. Planned Cartesian paths are then
        validated on the meshes (parry3d) after the sphere search, and the search continues
        until the path is collision-free under both models. Default True.
    """

    def __init__(self, kinematics_solver=None, sphere_options: Optional[dict] = None, exact: bool = True):
        super(NativePlanner, self).__init__()
        self._client = NativeClient()
        self.kinematics_solver = kinematics_solver
        self.sphere_options = sphere_options or {}
        self.exact = exact
        self._robot_description = None

    # -- SetRobotCell / SetRobotCellState ---------------------------------------------

    def set_robot_cell(self, robot_cell, robot_cell_state=None, options: Optional[dict] = None):
        options = options or {}
        client: NativeClient = self.client
        client._robot_cell = robot_cell
        client.group = options.get("group", robot_cell.main_group_name)
        self._robot_description = robot_description(robot_cell, client.group, self.kinematics_solver, sphere_options=self.sphere_options)
        client.core = None
        client._tool_key = None
        self.set_robot_cell_state(robot_cell_state or robot_cell.default_cell_state())

    def set_robot_cell_state(self, robot_cell_state, options: Optional[dict] = None):
        client: NativeClient = self.client
        client._robot_cell_state = robot_cell_state
        # Tools change the robot's own spheres: rebuild the core when the attached tools change.
        tool_key = tuple(sorted((tid, str(ts.attached_to_group), bool(ts.is_hidden), str(ts.attachment_frame)) for tid, ts in robot_cell_state.tool_states.items()))
        scene_key = (
            str(robot_cell_state.robot_base_frame),
            tuple(sorted((bid, str(bs.frame), str(bs.attached_to_link), str(bs.attached_to_tool), bool(bs.is_hidden)) for bid, bs in robot_cell_state.rigid_body_states.items())),
        )
        if client.core is None or tool_key != client._tool_key:
            description = json.loads(json.dumps(self._robot_description))
            add_attached_tools(description, client.robot_cell, robot_cell_state, client.group, sphere_options=self.sphere_options)
            scene = json.dumps(obstacle_boxes(client.robot_cell, robot_cell_state))
            client.core = _native().NativeCore(json.dumps(description), scene)
            if self.exact:
                client.core.set_exact_meshes(
                    json.dumps(exact_robot_meshes(client.robot_cell, robot_cell_state, client.group)),
                    json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state)),
                )
            client._tool_key, client._scene_key = tool_key, scene_key
        elif scene_key != client._scene_key:
            client.core.set_scene(json.dumps(obstacle_boxes(client.robot_cell, robot_cell_state)))
            if self.exact:
                client.core.set_exact_obstacles(json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state)))
            client._scene_key = scene_key

    def _sync_state(self, robot_cell_state):
        """Bring obstacles and tools up to date with ``robot_cell_state`` (cheap if unchanged)."""
        if robot_cell_state is not None:
            self.set_robot_cell_state(robot_cell_state)

    # -- Frames ---------------------------------------------------------------------------

    def _t_wcf_rcf(self, robot_cell_state) -> Transformation:
        base = robot_cell_state.robot_base_frame or Frame.worldXY()
        return Transformation.from_frame(base)

    def _group_values(self, configuration: Configuration):
        names = self.client.core.joint_names
        values = dict(zip(configuration.joint_names, configuration.joint_values)) if configuration.joint_names else None
        if values is None:
            return list(configuration.joint_values)
        return [values[n] for n in names]

    def _configuration(self, values) -> Configuration:
        config = self.client.robot_cell.zero_configuration(self.client.group)
        config.joint_values = list(values)
        return config

    def _pcf_rcf_to_3x4(self, frame_wcf, robot_cell_state, t_rcf_wcf=None):
        if t_rcf_wcf is None:
            t_rcf_wcf = self._t_rcf_wcf(robot_cell_state)
        m = t_rcf_wcf @ np.array(Transformation.from_frame(frame_wcf).matrix)
        return m[:3, :].reshape(-1).tolist()

    def _t_rcf_wcf(self, robot_cell_state):
        """World -> robot coordinates as a numpy 4x4 (rigid inverse, computed once)."""
        t = np.array(self._t_wcf_rcf(robot_cell_state).matrix)
        inv = np.eye(4)
        inv[:3, :3] = t[:3, :3].T
        inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
        return inv

    # -- ForwardKinematics ------------------------------------------------------------------

    def forward_kinematics(self, robot_cell_state, target_mode=TargetMode.ROBOT, group=None, native_scale=None, options=None):
        client: NativeClient = self.client
        self._sync_state(robot_cell_state)
        q = self._group_values(robot_cell_state.robot_configuration)
        pcf_rcf = client.core.link_frames(q)[client.core.end_effector_link]
        t = self._t_wcf_rcf(robot_cell_state) * Transformation.from_matrix(matrix_from_3x4(pcf_rcf).tolist())
        pcf_wcf = Frame.from_transformation(t)
        frame = client.robot_cell.pcf_to_target_frames(robot_cell_state, pcf_wcf, target_mode, group or client.group)
        if native_scale:
            frame.scale(1.0 / native_scale)
        return frame

    # -- InverseKinematics ------------------------------------------------------------------

    def iter_inverse_kinematics(self, target, robot_cell_state=None, group=None, options=None):
        """All closed-form solutions, collision-free unless ``options["check_collision"]`` is False."""
        options = options or {}
        client: NativeClient = self.client
        robot_cell_state = robot_cell_state or client.robot_cell_state
        self._sync_state(robot_cell_state)
        if not isinstance(target, FrameTarget):
            raise InverseKinematicsError("NativePlanner supports FrameTarget only.")
        target = target.normalized_to_meters()
        pcf = client.robot_cell.target_frames_to_pcf(robot_cell_state, target.target_frame, target.target_mode, group or client.group)
        solutions = client.core.inverse_kinematics(self._pcf_rcf_to_3x4(pcf, robot_cell_state))
        if options.get("check_collision", True) and solutions:
            clearance = client.core.clearance(solutions)
            solutions = [q for q, c in zip(solutions, clearance) if c > 0]
        if not solutions:
            raise InverseKinematicsError("No (collision-free) IK solution for the target.")
        return (self._configuration(q) for q in solutions)

    # -- CheckCollision -----------------------------------------------------------------------

    def check_collision(self, robot_cell_state, options=None):
        """Collision check on the sphere model, or on the exact meshes with ``options["exact"]``."""
        options = options or {}
        client: NativeClient = self.client
        self._sync_state(robot_cell_state)
        q = self._group_values(robot_cell_state.robot_configuration)
        if options.get("exact", False):
            contact = client.core.exact_check([q])[0]
            if contact is not None:
                raise CollisionCheckError("Collision (exact meshes) between {} and {}".format(*contact))
            return
        clearance = client.core.clearance([q])[0]
        if clearance <= 0:
            raise CollisionCheckError("Collision (sphere model): penetration {:.4f} m".format(-clearance))

    # -- PlanCartesianMotion ------------------------------------------------------------------

    def plan_cartesian_motion(self, waypoints, start_state, group=None, options=None):
        """Cartesian path through frame waypoints.

        Options
        -------
        free_axis_samples : int
            Rotations about the tool axis (TCF z) sampled per waypoint. Default 1 (frames as given).
        free_axis_range : float
            Half-range of that rotation in radians. Default pi.
        max_joint_step : float
            Largest joint change between consecutive waypoints (rad). Default 0.5.
        start_from_state : bool
            Begin the path at ``start_state``'s configuration. Default False (any IK branch).
        exact_check : bool
            Validate the path on the exact meshes (needs ``NativePlanner(exact=True)``). Default True.
        """
        options = options or {}
        client: NativeClient = self.client
        group = group or client.group
        self._sync_state(start_state)
        if not isinstance(waypoints, FrameWaypoints):
            raise CartesianMotionError("NativePlanner supports FrameWaypoints only.")
        if waypoints.native_scale not in (None, 1.0):
            waypoints = waypoints.normalized_to_meters()  # deep copy; only when scaling
        robot_cell = client.robot_cell
        t_rcf_wcf = self._t_rcf_wcf(start_state)
        tcf = None
        frames = waypoints.target_frames
        if waypoints.target_mode == TargetMode.TOOL:
            # Plan TCF frames directly, so the free rotation happens about the tool axis.
            tool_id = start_state.get_attached_tool_id(group)
            t_pcf_tcf = robot_cell.t_pcf_tcf(start_state, tool_id)
            tcf = frame_to_3x4(t_pcf_tcf)
            targets = [self._pcf_rcf_to_3x4(f, start_state, t_rcf_wcf) for f in frames]
        else:
            pcf = robot_cell.target_frames_to_pcf(start_state, frames, waypoints.target_mode, group)
            targets = [self._pcf_rcf_to_3x4(f, start_state, t_rcf_wcf) for f in pcf]
        start = self._group_values(start_state.robot_configuration) if options.get("start_from_state") else None
        try:
            path, _, stats = client.core.plan_cartesian(
                targets,
                tcf=tcf,
                free_axis_samples=int(options.get("free_axis_samples", 1)),
                free_axis_range=float(options.get("free_axis_range", 3.141592653589793)),
                start=start,
                max_joint_step=float(options.get("max_joint_step", 0.5)),
                edge_check_step=float(options.get("edge_check_step", 0.05)),
                exact=bool(options.get("exact_check", True)) and client.core.has_exact,
            )
        except RuntimeError as e:
            error = CartesianMotionError()
            error.args = (str(e),)
            raise error
        joint_names = client.core.joint_names
        joint_types = robot_cell.get_configurable_joint_types(group)
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        trajectory.points = [JointTrajectoryPoint(q, joint_types, joint_names=joint_names) for q in path]
        trajectory.fraction = 1.0
        trajectory.start_configuration = robot_cell.zero_full_configuration().merged(self._configuration(path[0]))
        trajectory.start_state = start_state
        trajectory.attributes["native_stats"] = stats
        return trajectory

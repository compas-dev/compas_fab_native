"""Native motion core as a compas_fab planner (spike).

``NativePlanner`` runs forward and inverse kinematics, collision checks, Cartesian and
free-space planning in the Rust core (``compas_fab_native``). User code switches to it by
changing the planner:

    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(robot_cell, robot_cell_state)
    trajectory = planner.plan_cartesian_motion(waypoints, start_state)
    trajectory = planner.plan_motion(target, start_state)

Limits of the spike: closed-form IK only (UR arms, spherical-wrist arms), so frame targets
and Cartesian paths need a 6-axis arm; ``plan_motion`` to a ConfigurationTarget works for
any serial robot. Obstacles are boxes or distance fields, attached tools are supported,
attached workpieces are ignored.
"""

import json
from typing import Optional

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation
from compas_robots import Configuration

from compas_fab.backends.exceptions import CollisionCheckError
from compas_fab.backends.exceptions import InverseKinematicsError
from compas_fab.backends.exceptions import MPNoIKSolutionError
from compas_fab.backends.exceptions import MPNoPlanFoundError
from compas_fab.backends.exceptions import MPStartStateInCollisionError
from compas_fab.backends.exceptions import MPTargetInCollisionError
from compas_fab.backends.interfaces import ClientInterface
from compas_fab.backends.interfaces import PlannerInterface
from compas_fab.backends.interfaces.backend_features import CheckCollision
from compas_fab.backends.interfaces.backend_features import ForwardKinematics
from compas_fab.backends.interfaces.backend_features import InverseKinematics
from compas_fab.backends.interfaces.backend_features import PlanCartesianMotion
from compas_fab.backends.interfaces.backend_features import PlanMotion
from compas_fab.backends.interfaces.backend_features import SetRobotCell
from compas_fab.backends.interfaces.backend_features import SetRobotCellState
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.robots import ConfigurationTarget
from compas_fab.robots import Duration
from compas_fab.robots import FrameTarget
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import JointTrajectory
from compas_fab.robots import JointTrajectoryPoint
from compas_fab.robots import PointAxisTarget
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
        raise ImportError("The native backend needs the compas_fab_native extension (`pip install ./native`).") from e
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
    PlanMotion,
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
    obstacles
        How the sphere search sees static rigid bodies: ``"boxes"`` (oriented bounding boxes,
        default) or ``"esdf"`` (a signed distance field of their meshes; bodies with touch
        links, like a floor, stay boxes).
    esdf_options
        ``voxel`` (0.02 m), ``truncation`` (0.3 m) and ``margin`` (0.01 m) of the ESDF.
    """

    def __init__(
        self,
        kinematics_solver=None,
        sphere_options: Optional[dict] = None,
        exact: bool = True,
        obstacles: str = "boxes",
        esdf_options: Optional[dict] = None,
    ):
        super(NativePlanner, self).__init__()
        self._client = NativeClient()
        self.kinematics_solver = kinematics_solver
        self.sphere_options = sphere_options or {}
        self.exact = exact
        if obstacles not in ("boxes", "esdf"):
            raise ValueError("obstacles must be 'boxes' or 'esdf'")
        self.obstacles = obstacles
        self.esdf_options = esdf_options or {}
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
            client.core = _native().NativeCore(json.dumps(description), self._scene_json(robot_cell_state))
            self._update_esdf(robot_cell_state)
            if self.exact:
                client.core.set_exact_meshes(
                    json.dumps(exact_robot_meshes(client.robot_cell, robot_cell_state, client.group)),
                    json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state)),
                )
            client._tool_key, client._scene_key = tool_key, scene_key
        elif scene_key != client._scene_key:
            client.core.set_scene(self._scene_json(robot_cell_state))
            self._update_esdf(robot_cell_state)
            if self.exact:
                client.core.set_exact_obstacles(json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state)))
            client._scene_key = scene_key
        client.core.set_passive_joints(self._passive_values(robot_cell_state))

    def _passive_values(self, robot_cell_state) -> dict:
        """Values of the joints outside the planning group (e.g. gripper fingers), from the
        state's configuration; mimic joints follow their leader."""
        client: NativeClient = self.client
        model = client.robot_cell.robot_model
        group_joints = set(client.core.joint_names)
        config = robot_cell_state.robot_configuration
        known = dict(zip(config.joint_names, config.joint_values)) if config and config.joint_names else {}
        values = {}
        for joint in model.joints:
            if joint.name in group_joints or not joint.is_configurable() and not joint.mimic:
                continue
            if joint.mimic:
                leader = known.get(joint.mimic.joint, 0.0)
                values[joint.name] = joint.mimic.multiplier * leader + joint.mimic.offset
            elif joint.name in known:
                values[joint.name] = known[joint.name]
        return values

    def _scene_json(self, robot_cell_state):
        boxes = obstacle_boxes(self.client.robot_cell, robot_cell_state, only_touching=self.obstacles == "esdf")
        return json.dumps(boxes)

    def _update_esdf(self, robot_cell_state):
        if self.obstacles != "esdf":
            return
        meshes = exact_obstacle_meshes(self.client.robot_cell, robot_cell_state, without_touching=True)
        self.esdf_stats = self.client.core.set_esdf_obstacles(json.dumps(meshes), **self.esdf_options)

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
        start_values = path[0]
        if start is not None:
            # The core returns the start configuration followed by one per waypoint.
            path = path[1:]
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        trajectory.points = [JointTrajectoryPoint(q, joint_types, joint_names=joint_names) for q in path]
        trajectory.fraction = 1.0
        trajectory.start_configuration = robot_cell.zero_full_configuration().merged(self._configuration(start_values))
        trajectory.start_state = start_state
        trajectory.attributes["native_stats"] = stats
        return trajectory

    # -- PlanMotion ---------------------------------------------------------------------------

    def plan_motion(self, target, start_state, group=None, options=None):
        """Collision-free joint-space motion from ``start_state`` to ``target``.

        The straight joint-space motion is tried first, then RRT-Connect from the start and
        from every goal at once, then shortcutting and B-spline smoothing. The result is
        checked on the exact meshes and re-planned with another seed if it touches them.

        Targets
        -------
        ConfigurationTarget
            One goal.
        FrameTarget
            Every IK solution is a goal; the planner picks whichever it reaches.
        PointAxisTarget
            IK solutions of ``axis_samples`` rotations about the axis.

        Options
        -------
        range : float
            Longest tree extension, in joint-space distance. Default 1.0.
        resolution : float
            Collision checks per unit of joint-space distance. Default: enough that no robot
            sphere moves more than about 2 cm between checks (``core.default_resolution``).
        max_iterations : int
            Default 100000.
        seed : int
            Sampling sequence offset, for different (reproducible) paths. Default 0.
        simplify : bool
            Shortcut and smooth the path. Default True.
        exact_check : bool
            Validate the path on the exact meshes. Default True.
        exact_step : float
            Largest joint change between exact checks (rad). Default 0.02.
        axis_samples : int
            Rotations sampled about a PointAxisTarget's axis. Default 16.
        time_parameterization : bool
            Time the path under joint velocity and acceleration limits (TOPP-RA), with
            blended corners, starting and ending at rest. Points then carry
            ``time_from_start``, velocities and accelerations. Default True.
        velocity_scaling, acceleration_scaling : float
            Fractions of the limits to use. Default 1.0.
        max_acceleration : float or list
            Joint acceleration limits (URDFs carry none). Default: twice each joint's
            velocity limit, i.e. full speed in half a second.
        blend_tolerance : float
            Largest joint-space deviation of a blended corner from the planned path.
            Blends that collide shrink, down to a stop at that corner. Default 0.05.
        time_step : float
            Sampling period of the timed trajectory in seconds. Default 0.01.
        max_joint_step : float
            Without time parameterization: interpolate the path so no joint moves more than
            this between points. Default: the path's vertices only.
        """
        options = options or {}
        client: NativeClient = self.client
        group = group or client.group
        self._sync_state(start_state)
        if not start_state.robot_configuration:
            raise ValueError("start_state.robot_configuration must be provided.")
        start = self._group_values(start_state.robot_configuration)
        goals = self._motion_goals(target, start_state, group, options)
        try:
            path, stats = client.core.plan_motion(
                start,
                goals,
                range=float(options.get("range", 1.0)),
                resolution=options.get("resolution"),
                max_iterations=int(options.get("max_iterations", 100000)),
                seed=int(options.get("seed", 0)),
                simplify=bool(options.get("simplify", True)),
                exact=bool(options.get("exact_check", True)) and client.core.has_exact,
                exact_step=float(options.get("exact_step", 0.02)),
            )
        except RuntimeError as e:
            message = str(e)
            if message.startswith("start_in_collision"):
                raise MPStartStateInCollisionError("The start configuration is in collision.", start_state=start_state)
            if message.startswith("goals_in_collision"):
                raise MPTargetInCollisionError("Every goal configuration of the target is in collision.", target=target)
            raise MPNoPlanFoundError("No path found: {}".format(message))
        joint_names = client.core.joint_names
        joint_types = client.robot_cell.get_configurable_joint_types(group)
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        # The start configuration is the trajectory's start, not its first point.
        if options.get("time_parameterization", True):
            timed = self._time_parameterize(path, group, options)
            stats["duration"] = timed["duration"]
            stats["blends_stopped"] = float(sum(timed["stops"]))
            trajectory.points = [
                JointTrajectoryPoint(q, joint_types, velocities=v, accelerations=a, time_from_start=Duration(t, 0), joint_names=joint_names)
                for t, q, v, a in list(zip(timed["times"], timed["positions"], timed["velocities"], timed["accelerations"]))[1:]
            ]
        else:
            step = options.get("max_joint_step")
            if step:
                path = _densify(path, float(step))
            trajectory.points = [JointTrajectoryPoint(q, joint_types, joint_names=joint_names) for q in path[1:]]
        trajectory.fraction = 1.0
        trajectory.start_configuration = client.robot_cell.zero_full_configuration().merged(self._configuration(path[0]))
        trajectory.start_state = start_state
        trajectory.attributes["native_stats"] = stats
        return trajectory

    def _joint_limits(self, group, options):
        """Velocity and acceleration limits of the group's joints, scaled by the options."""
        client: NativeClient = self.client
        joints = {j.name: j for j in client.robot_cell.robot_model.get_configurable_joints()}
        velocity = []
        for name in client.core.joint_names:
            limit = joints[name].limit if name in joints else None
            v = limit.velocity if limit is not None and limit.velocity else 1.0
            velocity.append(float(v))
        acceleration = options.get("max_acceleration")
        if acceleration is None:
            acceleration = [2.0 * v for v in velocity]
        elif not isinstance(acceleration, (list, tuple)):
            acceleration = [float(acceleration)] * len(velocity)
        vs, as_ = float(options.get("velocity_scaling", 1.0)), float(options.get("acceleration_scaling", 1.0))
        return [v * vs for v in velocity], [a * as_ for a in acceleration]

    def _time_parameterize(self, path, group, options):
        """Timed samples of a collision-free path. Blends leave the path, so the timed motion
        is checked on the exact meshes too; if it touches, the corners become stops (the
        robot then follows the checked path exactly)."""
        client: NativeClient = self.client
        velocity, acceleration = self._joint_limits(group, options)
        kwargs = dict(time_step=float(options.get("time_step", 0.01)), resolution=options.get("resolution"))
        timed = client.core.time_parameterize(path, velocity, acceleration, blend_tolerance=float(options.get("blend_tolerance", 0.05)), **kwargs)
        if client.core.has_exact and options.get("exact_check", True) and any(b > 0 for b in timed["blends"]):
            dense = _densify(timed["positions"], float(options.get("exact_step", 0.02)))
            if any(c is not None for c in client.core.exact_check(dense)):
                timed = client.core.time_parameterize(path, velocity, acceleration, blend_tolerance=0.0, **kwargs)
        return timed

    def _motion_goals(self, target, start_state, group, options):
        """Goal configurations (group joint values) of a motion target."""
        if isinstance(target, ConfigurationTarget):
            return [self._group_values(target.target_configuration)]
        if isinstance(target, FrameTarget):
            frame_targets = [target]
        elif isinstance(target, PointAxisTarget):
            target = target.normalized_to_meters()
            z = target.target_z_axis
            # Any x axis perpendicular to z, then rotated about z.
            helper = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
            x0 = np.cross(z, helper)
            x0 /= np.linalg.norm(x0)
            y0 = np.cross(z, x0)
            n = int(options.get("axis_samples", 16))
            frame_targets = []
            for k in range(n):
                a = 2 * np.pi * k / n
                x = np.cos(a) * x0 + np.sin(a) * y0
                y = np.cross(z, x)
                frame_targets.append(FrameTarget(Frame(target.target_point, x.tolist(), y.tolist()), target.target_mode))
        else:
            raise NotImplementedError("NativePlanner.plan_motion supports ConfigurationTarget, FrameTarget and PointAxisTarget.")
        goals = []
        for frame_target in frame_targets:
            try:
                configs = self.iter_inverse_kinematics(frame_target, start_state, group, {"check_collision": False})
                goals.extend(self._group_values(c) for c in configs)
            except InverseKinematicsError:
                continue
        if not goals:
            raise MPNoIKSolutionError("No IK solution for the target.", target=target)
        return goals


def _densify(path, step):
    out = [list(path[0])]
    for a, b in zip(path, path[1:]):
        n = max(1, int(np.ceil(max(abs(x - y) for x, y in zip(a, b)) / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append([x + t * (y - x) for x, y in zip(a, b)])
    return out

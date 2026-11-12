"""Native motion core as a compas_fab planner.

``NativePlanner`` runs forward and inverse kinematics, collision checks, Cartesian and
free-space planning in the Rust core (``compas_fab_native``). User code switches to it by
changing the planner:

    planner = NativePlanner()
    planner.set_robot_cell(robot_cell, robot_cell_state)
    trajectory = planner.plan_cartesian_motion(waypoints, start_state)
    trajectory = planner.plan_motion(target, start_state)

IK is closed-form where the robot has one and numerical otherwise. Spherical-wrist arms
get a closed form built from the robot model; UR arms get one when their compas_fab solver
is passed (``NativePlanner(UR5Kinematics())``). Obstacles are boxes or distance fields. What the robot carries (attached tools,
the workpieces they hold, bodies attached to its links) collides like its links do.
"""

import json
from typing import Optional

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation
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
from compas_fab.robots import PointAxisWaypoints
from compas_fab.robots import TargetMode
from compas_robots import Configuration

from ._core import NativeCore
from .model import add_carried_objects
from .model import exact_obstacle_meshes
from .model import exact_robot_meshes
from .model import frame_to_3x4
from .model import matrix_from_3x4
from .model import moving_link_names
from .model import obstacle_boxes
from .model import planning_links
from .model import robot_description
from .model import static_obstacles

__all__ = ["NativeClient", "NativePlanner"]


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
        are handed to the core's closed-form IK. Optional: spherical-wrist arms get a closed
        form without it, other robots get numerical IK (see ``has_closed_form_ik``).
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
    exact_links
        Links to check on their exact geometry inside planning too, not only in the final
        exact check: ``None`` (default), ``"auto"`` (the links spheres represent poorly,
        see ``poorly_fitted_links``) or a list of link names. They are tested as convex
        hulls (conservative) against each other and the scene's boxes, on every
        configuration the spheres find free. Useful when the final check keeps rejecting
        paths near large flat links (a gantry carriage); it slows planning down, and the
        hulls take free space away. Needs ``exact=True``.
    """

    def __init__(
        self,
        kinematics_solver=None,
        sphere_options: Optional[dict] = None,
        exact: bool = True,
        obstacles: str = "boxes",
        esdf_options: Optional[dict] = None,
        exact_links=None,
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
        if exact_links not in (None, "auto") and not isinstance(exact_links, (list, tuple)):
            raise ValueError("exact_links must be None, 'auto' or a list of link names")
        self.exact_links = exact_links
        self._robot_description = None
        self._has_static_links = False
        self._static_cache = None
        self._moving = None

    @property
    def has_closed_form_ik(self) -> bool:
        """True if IK for the current robot cell is solved in closed form, False if numerically.

        Known once the robot cell is set. Closed form: the ``kinematics_solver`` given, or a
        spherical-wrist arm recognised from the robot model.
        """
        if self.client.core is None:
            raise ValueError("Set a robot cell first.")
        return self.client.core.has_analytic_ik

    # -- SetRobotCell / SetRobotCellState ---------------------------------------------

    def set_robot_cell(self, robot_cell, robot_cell_state=None, options: Optional[dict] = None):
        options = options or {}
        client: NativeClient = self.client
        client._robot_cell = robot_cell
        client.group = options.get("group", robot_cell.main_group_name)
        self._robot_description = robot_description(robot_cell, client.group, self.kinematics_solver, sphere_options=self.sphere_options)
        self._has_static_links = bool(planning_links(robot_cell, client.group)[1])
        self._static_cache = None
        self._moving = None
        client.core = None
        client._tool_key = None
        self.set_robot_cell_state(robot_cell_state or robot_cell.default_cell_state())

    def set_robot_cell_state(self, robot_cell_state, options: Optional[dict] = None):
        client: NativeClient = self.client
        client._robot_cell_state = robot_cell_state
        # What the robot carries changes its own spheres: rebuild the core when any
        # attachment changes.
        tool_key = self._attachment_key(robot_cell_state)
        scene_key = (
            str(robot_cell_state.robot_base_frame),
            tuple(
                sorted(
                    (bid, str(bs.frame), str(bs.attached_to_link), str(bs.attached_to_tool), bool(bs.is_hidden), tuple(bs.touch_links or ()))
                    for bid, bs in robot_cell_state.rigid_body_states.items()
                )
            ),
            self._static_key(robot_cell_state),
        )
        if client.core is None or tool_key != client._tool_key:
            description = json.loads(json.dumps(self._robot_description))
            add_carried_objects(description, client.robot_cell, robot_cell_state, client.group, sphere_options=self.sphere_options)
            description["exact_links"] = self._exact_link_indices(description)
            client.core = NativeCore(json.dumps(description), self._scene_json(robot_cell_state))
            self._update_esdf(robot_cell_state)
            if self.exact:
                client.core.set_exact_meshes(
                    json.dumps(exact_robot_meshes(client.robot_cell, robot_cell_state, client.group)),
                    json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state, group=client.group, static=self._static(robot_cell_state))),
                )
            client._tool_key, client._scene_key = tool_key, scene_key
        elif scene_key != client._scene_key:
            client.core.set_scene(self._scene_json(robot_cell_state))
            self._update_esdf(robot_cell_state)
            if self.exact:
                client.core.set_exact_obstacles(json.dumps(exact_obstacle_meshes(client.robot_cell, robot_cell_state, group=client.group, static=self._static(robot_cell_state))))
            client._scene_key = scene_key
        client.core.set_passive_joints(self._passive_values(robot_cell_state))

    @property
    def poorly_fitted_links(self):
        """Links whose collision spheres miss their mesh by more than
        ``model.EXACT_LINK_MISS`` (95th percentile): the spheres cannot represent them (a
        gantry carriage, a tall base), so collisions with them are only caught by the
        exact check. See the ``exact_links`` option."""
        if self._robot_description is None:
            return []
        names = [link["name"] for link in self._robot_description["links"]]
        return [names[i] for i in self._robot_description.get("exact_links", [])]

    def _exact_link_indices(self, description):
        names = [link["name"] for link in description["links"]]
        if self.exact_links is None:
            return []
        if self.exact_links == "auto":
            return list(description.get("exact_links", []))
        unknown = [n for n in self.exact_links if n not in names]
        if unknown:
            raise ValueError("exact_links: unknown or static links {}".format(unknown))
        return [names.index(n) for n in self.exact_links]

    def _attachment_key(self, robot_cell_state):
        """Tools and rigid bodies: what they are attached to, where, and whether shown."""
        tools = tuple(
            sorted(
                (
                    tid,
                    str(ts.attached_to_group),
                    bool(ts.is_hidden),
                    str(ts.attachment_frame),
                    "" if ts.attached_to_group else str(ts.frame),
                    tuple(ts.touch_links or ()),
                    str(ts.configuration),
                )
                for tid, ts in robot_cell_state.tool_states.items()
            )
        )
        bodies = tuple(
            sorted(
                (bid, str(bs.attached_to_link), str(bs.attached_to_tool), str(bs.attachment_frame), bool(bs.is_hidden), tuple(bs.touch_links or ()))
                for bid, bs in robot_cell_state.rigid_body_states.items()
                if bs.attached_to_link or bs.attached_to_tool
            )
        )
        return tools, bodies

    def _mounted_elsewhere(self, robot_cell_state):
        """True if a tool or body sits on a part of the robot the group does not move (so
        its pose depends on joints outside the group)."""
        group = self.client.group
        tool_groups = {tid: ts.attached_to_group for tid, ts in robot_cell_state.tool_states.items()}
        if any(g and g != group for g in tool_groups.values()):
            return True
        bodies = [bs for bs in robot_cell_state.rigid_body_states.values() if bs.attached_to_link or bs.attached_to_tool]
        if not bodies:
            return False
        moving = self._moving_links()
        return any((bs.attached_to_link and bs.attached_to_link not in moving) or (bs.attached_to_tool and tool_groups.get(bs.attached_to_tool) != group) for bs in bodies)

    def _moving_links(self):
        if self._moving is None:
            self._moving = moving_link_names(self.client.robot_cell, self.client.group)
        return self._moving

    def _static_key(self, robot_cell_state):
        """What poses the static obstacles: the attachments, and the joint values if any
        static obstacle hangs on the robot (links the group cannot move, other groups'
        tools). Constant for a single robot with its own tool, so moving it never
        rebuilds the scene."""
        attachments = self._attachment_key(robot_cell_state)
        if not self._has_static_links and not self._mounted_elsewhere(robot_cell_state):
            return attachments, None
        config = robot_cell_state.robot_configuration
        values = tuple(zip(config.joint_names, (round(v, 9) for v in config.joint_values))) if config and config.joint_names else None
        return attachments, values

    def _static(self, robot_cell_state):
        """``static_obstacles``, recomputed only when what poses them changes."""
        key = self._static_key(robot_cell_state)
        if self._static_cache is None or self._static_cache[0] != key:
            self._static_cache = (key, static_obstacles(self.client.robot_cell, robot_cell_state, self.client.group))
        return self._static_cache[1]

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
        boxes = obstacle_boxes(self.client.robot_cell, robot_cell_state, only_touching=self.obstacles == "esdf", group=self.client.group, static=self._static(robot_cell_state))
        return json.dumps(boxes)

    def _update_esdf(self, robot_cell_state):
        if self.obstacles != "esdf":
            return
        meshes = exact_obstacle_meshes(self.client.robot_cell, robot_cell_state, without_touching=True, group=self.client.group, static=self._static(robot_cell_state))
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
        """IK solutions of a ``FrameTarget`` or ``PointAxisTarget``, collision-free unless
        ``options["check_collision"]`` is False.

        Robots with a closed form (``kinematics_solver``, or a spherical wrist the core
        derives one for) get all of its solutions; for a ``PointAxisTarget``, those of
        ``axis_samples`` rotations about the axis. Other robots (redundant arms, external
        axes) use the core's numerical IK, which treats the rotation about a point-axis
        target's axis as free.

        Options
        -------
        check_collision : bool
            Only collision-free solutions (sphere model). Default True.
        ik : str
            ``"auto"`` (default: closed form if there is one), ``"numeric"`` or ``"analytic"``.
        max_results : int
            Numerical IK: distinct solutions at most. Default 8.
        seed : int
            Numerical IK: offset of the random seeds. Default 0. The state's configuration
            is always tried first.
        axis_samples : int
            Closed form, point-axis targets: rotations sampled about the axis. Default 16.
        """
        options = options or {}
        client: NativeClient = self.client
        robot_cell_state = robot_cell_state or client.robot_cell_state
        self._sync_state(robot_cell_state)
        solutions = self._ik_solutions(target, robot_cell_state, group or client.group, options, bool(options.get("check_collision", True)))
        if not solutions:
            raise InverseKinematicsError("No (collision-free) IK solution for the target.")
        return (self._configuration(q) for q in solutions)

    def _numeric(self, options) -> bool:
        mode = options.get("ik", "auto")
        if mode not in ("auto", "numeric", "analytic"):
            raise ValueError("options['ik'] must be 'auto', 'numeric' or 'analytic'")
        if mode == "analytic" and not self.client.core.has_analytic_ik:
            raise InverseKinematicsError("This robot has no closed-form IK.")
        return mode == "numeric" or (mode == "auto" and not self.client.core.has_analytic_ik)

    def _reference(self, robot_cell_state, target_mode, group):
        """Transformation from the planner frame (end-effector link) to the frame a target
        of ``target_mode`` refers to: the tool's TCF, the workpiece's OCF, or none."""
        robot_cell = self.client.robot_cell
        if target_mode == TargetMode.TOOL:
            return robot_cell.t_pcf_tcf(robot_cell_state, robot_cell_state.get_attached_tool_id(group))
        if target_mode == TargetMode.WORKPIECE:
            workpiece_ids = robot_cell_state.get_attached_workpiece_ids(group)
            if len(workpiece_ids) != 1:
                raise ValueError("Only one workpiece should be attached to the robot in group '{}'.".format(group))
            return robot_cell.t_pcf_ocf(robot_cell_state, workpiece_ids[0])
        if target_mode in (TargetMode.ROBOT, None):
            return Transformation()
        raise ValueError("Unsupported target mode: '{}'.".format(target_mode))

    def _current_values(self, robot_cell_state):
        config = robot_cell_state.robot_configuration if robot_cell_state else None
        if not config or not config.joint_values:
            return []
        try:
            return [self._group_values(config)]
        except KeyError:
            return []

    def _ik_solutions(self, target, robot_cell_state, group, options, check_collision):
        """Group joint values solving ``target``."""
        client: NativeClient = self.client
        if isinstance(target, FrameTarget):
            target = target.normalized_to_meters()
            frame, free = target.target_frame, False
        elif isinstance(target, PointAxisTarget):
            target = target.normalized_to_meters()
            frame, free = _point_axis_frames([(target.target_point, target.target_z_axis)])[0], True
        else:
            raise InverseKinematicsError("NativePlanner supports FrameTarget and PointAxisTarget.")
        t_rcf_wcf = self._t_rcf_wcf(robot_cell_state)
        if self._numeric(options):
            # Solve for the reference frame itself, so a free rotation is about its z axis.
            reference = self._reference(robot_cell_state, target.target_mode, group)
            return client.core.numeric_inverse_kinematics(
                self._pcf_rcf_to_3x4(frame, robot_cell_state, t_rcf_wcf),
                tcf=frame_to_3x4(reference),
                free_axis=free,
                seeds=self._current_values(robot_cell_state),
                max_results=int(options.get("max_results", 8)),
                seed=int(options.get("seed", 0)),
                check_collision=check_collision,
            )
        frames = [frame]
        if free:
            n = int(options.get("axis_samples", 16))
            frames = [_rotated_about_z(frame, 2 * np.pi * k / n) for k in range(n)]
        solutions = []
        for f in frames:
            pcf = client.robot_cell.target_frames_to_pcf(robot_cell_state, f, target.target_mode, group)
            solutions += client.core.inverse_kinematics(self._pcf_rcf_to_3x4(pcf, robot_cell_state, t_rcf_wcf))
        if check_collision and solutions:
            clearance = client.core.clearance(solutions)
            solutions = [q for q, c in zip(solutions, clearance) if c > 0]
        return solutions

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
        """Cartesian path through ``FrameWaypoints`` or ``PointAxisWaypoints``.

        The tool moves in straight lines between waypoints: frames are interpolated every
        ``max_step`` metres and ``max_step_angle`` radians (like MoveIt and PyBullet), and the
        trajectory has one point per interpolated frame, passing through every waypoint.

        Robots with a closed form plan over all IK branches at once (layered graph search).
        Other robots (redundant arms, external axes) follow the path with numerical IK from
        several starting solutions and keep the cheapest that stays collision-free.
        Point-axis waypoints leave the rotation about the axis free: continuously with
        numerical IK, sampled (``free_axis_samples``) with the closed form.

        Options
        -------
        max_step : float
            Largest distance between interpolated frames, metres. Default 0.01. 0 or None:
            the waypoints only. ``max_step_distance`` is accepted as an alias.
        max_step_angle : float
            Largest rotation between interpolated frames, radians. Default 0.1.
        free_axis_samples : int
            Closed form: rotations about the tool axis sampled per waypoint. Default 1 for
            frame waypoints (the frames as given), 16 for point-axis waypoints.
        free_axis_range : float
            Half-range of that rotation in radians. Default pi.
        ik : str
            ``"auto"`` (default: closed form if there is one), ``"numeric"`` or ``"analytic"``.
        tracks : int
            Numerical IK: starting solutions followed along the path. Default 8.
        seed : int
            Numerical IK: offset of the random seeds. Default 0.
        max_joint_step : float
            Largest joint change between consecutive frames (rad, or m for prismatic
            joints). Default 0.5.
        start_from_state : bool
            Begin the path at ``start_state``'s configuration. Default False (any IK branch).
        exact_check : bool
            Validate the path on the exact meshes (needs ``NativePlanner(exact=True)``). Default True.
        """
        options = options or {}
        client: NativeClient = self.client
        group = group or client.group
        self._sync_state(start_state)
        if isinstance(waypoints, FrameWaypoints):
            if waypoints.native_scale not in (None, 1.0):
                waypoints = waypoints.normalized_to_meters()  # deep copy; only when scaling
            frames, free = list(waypoints.target_frames), False
        elif isinstance(waypoints, PointAxisWaypoints):
            if waypoints.native_scale not in (None, 1.0):
                waypoints = waypoints.normalized_to_meters()
            frames, free = _point_axis_frames(waypoints.target_points_and_axes), True
        else:
            raise CartesianMotionError("NativePlanner supports FrameWaypoints and PointAxisWaypoints.")
        if not frames:
            raise CartesianMotionError("No waypoints.")
        max_step = options.get("max_step", options.get("max_step_distance", 0.01))
        if max_step:
            frames = _interpolate_frames(frames, float(max_step), float(options.get("max_step_angle", 0.1)))
        robot_cell = client.robot_cell
        t_rcf_wcf = self._t_rcf_wcf(start_state)
        # Plan the reference frame (TCF, OCF or flange) directly, so a free rotation is
        # about its own z axis.
        tcf = frame_to_3x4(self._reference(start_state, waypoints.target_mode, group))
        targets = [self._pcf_rcf_to_3x4(f, start_state, t_rcf_wcf) for f in frames]
        start = self._group_values(start_state.robot_configuration) if options.get("start_from_state") else None
        numeric = self._numeric(options)
        try:
            path, _, stats = client.core.plan_cartesian(
                targets,
                tcf=tcf,
                free_axis_samples=int(options.get("free_axis_samples", 16 if free else 1)),
                free_axis_range=float(options.get("free_axis_range", 3.141592653589793)),
                start=start,
                max_joint_step=float(options.get("max_joint_step", 0.5)),
                edge_check_step=float(options.get("edge_check_step", 0.05)),
                exact=bool(options.get("exact_check", True)) and client.core.has_exact,
                free_axis=free and numeric,
                numeric=numeric,
                tracks=int(options.get("tracks", 8)),
                seed=int(options.get("seed", 0)),
                seeds=self._current_values(start_state),
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
        if bool(options.get("exact_check", True)) and client.core.has_exact:
            # The sphere model can miss thin contacts (a forearm grazing a wall). A goal like
            # that makes every path to it fail the exact check, so drop it up front.
            goals = [g for g, contact in zip(goals, client.core.exact_check(goals)) if contact is None]
            if not goals:
                raise MPTargetInCollisionError("Every goal configuration of the target is in collision (exact meshes).", target=target)
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
        if not isinstance(target, (FrameTarget, PointAxisTarget)):
            raise NotImplementedError("NativePlanner.plan_motion supports ConfigurationTarget, FrameTarget and PointAxisTarget.")
        # Closed form: every solution, collisions are the planner's business. Numerical IK:
        # the solve itself steers clear of collisions, so its goals are collision-free.
        numeric = self._numeric(options)
        try:
            goals = self._ik_solutions(target, start_state, group, options, check_collision=numeric)
        except InverseKinematicsError:
            goals = []
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


def _rotated_about_z(frame, angle):
    """``frame`` rotated by ``angle`` about its own z axis."""
    c, s = np.cos(angle), np.sin(angle)
    x, y = np.array(frame.xaxis), np.array(frame.yaxis)
    return Frame(frame.point, (c * x + s * y).tolist(), (-s * x + c * y).tolist())


def _point_axis_frames(points_and_axes):
    """Frames with z along each axis. The x axis is arbitrary for the first one and carried
    over to the next (projected onto its plane), so the frames turn as little as possible."""
    frames = []
    x_prev = None
    for point, axis in points_and_axes:
        z = np.asarray(list(axis), dtype=float)
        z /= np.linalg.norm(z)
        x = None
        if x_prev is not None:
            x = x_prev - np.dot(x_prev, z) * z
            if np.linalg.norm(x) < 1e-6:
                x = None
        if x is None:
            helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(z, helper)
        x /= np.linalg.norm(x)
        frames.append(Frame(list(point), x.tolist(), np.cross(z, x).tolist()))
        x_prev = x
    return frames


def _axis_angle(r):
    """Axis and angle of a 3x3 rotation matrix."""
    angle = float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # angle ~ pi: axis from the symmetric part
        k = int(np.argmax(np.diag(r)))
        axis = (r[:, k] + np.eye(3)[k]) / np.sqrt(2.0 * (1.0 + r[k, k]))
    return axis / np.linalg.norm(axis), angle


def _rotation(axis, angle):
    x, y, z = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _interpolate_frames(frames, max_step, max_angle):
    """Frames between consecutive ``frames`` (linear position, shortest rotation), at most
    ``max_step`` metres and ``max_angle`` radians apart; the given frames are kept."""
    out = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        ra = np.array([list(a.xaxis), list(a.yaxis), list(a.zaxis)]).T
        rb = np.array([list(b.xaxis), list(b.yaxis), list(b.zaxis)]).T
        axis, angle = _axis_angle(ra.T @ rb)
        pa, pb = np.array(list(a.point)), np.array(list(b.point))
        n = max(1, int(np.ceil(np.linalg.norm(pb - pa) / max_step)), int(np.ceil(angle / max_angle)) if max_angle else 1)
        for k in range(1, n):
            t = k / n
            r = ra @ _rotation(axis, t * angle)
            out.append(Frame((pa + t * (pb - pa)).tolist(), r[:, 0].tolist(), r[:, 1].tolist()))
        out.append(b)
    return out

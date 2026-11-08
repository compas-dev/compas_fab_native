"""The five live scenarios. Each owns a robot cell, a NativePlanner and the bodies the
browser shows; the session streams whatever motion they plan."""

import time

import numpy as np
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Torus
from kin import Arm
from scene import compas_mesh_tris
from scene import frame_from_matrix
from scene import frame_rows
from scene import matrix_from_frame
from scene import mesh_json
from scene import ring_axis_tris
from scene import robot_parts
from scene import spiral_points
from scene import target_marker_tris
from scipy.spatial.transform import Rotation

import compas_fab
from compas_fab.backends import UR5Kinematics
from compas_fab.backends.exceptions import BackendError
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.backends.native import NativePlanner
from compas_fab.robots import ConfigurationTarget
from compas_fab.robots import FrameTarget
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import PointAxisTarget
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

DT = 1 / 60
ACCENT = "#f2b705"
BRICK = "#b4573e"
STEEL = "#8a96a3"
BAD = "#d13b3b"


class PlanFailed(Exception):
    pass


class Body:
    """Something the browser draws: a rigid body of the cell, a target marker, a guide."""

    def __init__(self, guid, name, tris, color, frame, draggable=False, body_id=None, opacity=1.0, flat=False, clickable=False):
        self.guid = guid
        self.name = name
        self.tris = tris
        self.color = color
        self.frame = frame
        self.home = frame.copy()
        self.draggable = draggable
        self.body_id = body_id
        self.opacity = opacity
        self.flat = flat
        self.clickable = clickable

    def json(self):
        return {
            "guid": self.guid,
            "name": self.name,
            "mesh": mesh_json(self.tris),
            "color": self.color,
            "pose": frame_rows(self.frame),
            "draggable": self.draggable,
            "clickable": self.clickable,
            "opacity": self.opacity,
            "flat": self.flat,
        }


def timed_from_trajectory(start, trajectory):
    pts = trajectory.points
    return {
        "times": [0.0] + [p.time_from_start.seconds for p in pts],
        "positions": [list(start)] + [list(p.joint_values) for p in pts],
        "velocities": [[0.0] * len(start)] + [list(p.velocities) for p in pts],
    }


class Scenario:
    id = ""
    title = ""
    hint = ""
    palette = "ur5"
    controls = {}
    camera = {"position": [0.55, -1.75, 0.95], "target": [0.3, 0.0, 0.4]}
    scale = 1.0  # of the scene, for the browser (path thickness)
    done = "Done. Drag something to plan again."
    ready = "Ready."
    group = None

    def __init__(self):
        self.bodies = {}
        self.q = None
        self.updates = []  # body messages for the browser, sent after the next plan
        self.setup()
        self._parts = robot_parts(self.cell, self.core, self.palette, self.state)
        self.reset()

    # -- set up by subclasses -------------------------------------------------------------

    def setup(self):
        raise NotImplementedError

    def reset(self):
        """Back to the initial configuration and body placements."""
        self.q = np.array(self.q0, dtype=float)
        for body in self.bodies.values():
            body.frame = body.home.copy()
            self._sync_body(body)
        self.planner.set_robot_cell_state(self.state)

    # -- shared -----------------------------------------------------------------------

    @property
    def core(self):
        return self.planner.client.core

    def joints(self):
        model = {j.name: j for j in self.cell.robot_model.get_configurable_joints()}
        names = self.core.joint_names
        return {
            "names": names,
            "lower": [model[n].limit.lower for n in names],
            "upper": [model[n].limit.upper for n in names],
            "vmax": self.limits()[0],
            "prismatic": [int(model[n].type) == 2 for n in names],
        }

    def solve_ik(self, target, seeds=(), max_results=8, free_axis=False, check_collision=True, margin=0.0, max_seeds=64):
        """The backend's numerical IK (``core.numeric_inverse_kinematics``) for the TCP
        at ``target`` (4x4, robot coordinates), seeded with ``seeds`` first."""
        self._ik_calls = getattr(self, "_ik_calls", 0) + 1
        return [
            np.asarray(q)
            for q in self.core.numeric_inverse_kinematics(
                [float(x) for x in np.asarray(target)[:3, :].ravel()],
                tcf=[float(x) for x in self.arm.tcf[:3, :].ravel()],
                free_axis=free_axis,
                seeds=[[float(x) for x in q] for q in seeds],
                max_results=max_results,
                max_seeds=max_seeds,
                seed=self._ik_calls,
                check_collision=check_collision,
                margin=margin,
            )
        ]

    def make_arm(self, tcf=None):
        j = self.joints()
        return Arm(self.core, j["lower"], j["upper"], j["vmax"], tcf=tcf)

    def add_body(self, body, rigid=None, touch_links=None):
        """Register a browser body; with ``rigid`` (a compas Mesh, local coordinates) also
        a rigid body of the cell that the planner avoids."""
        self.bodies[body.guid] = body
        if rigid is not None:
            body.body_id = body.guid
            self.cell.rigid_body_models[body.guid] = RigidBody.from_mesh(rigid)
            self.state.rigid_body_states[body.guid] = RigidBodyState(body.frame.copy())
            if touch_links:
                self.state.rigid_body_states[body.guid].touch_links = list(touch_links)

    def _sync_body(self, body):
        if body.body_id is not None:
            self.state.rigid_body_states[body.body_id].frame = body.frame.copy()

    def scene_json(self):
        return {
            "type": "scene",
            "id": self.id,
            "title": self.title,
            "hint": self.hint,
            "controls": self.controls,
            "camera": self.camera,
            "scale": self.scale,
            "joints": self.joints(),
            "parts": self._parts,
            "bodies": [b.json() for b in self.bodies.values()],
        }

    def set_q(self, q):
        """Put ``q`` in the cell state's configuration (what the planner starts from).
        ``self.q``, where the robot is, is left alone."""
        config = self.state.robot_configuration
        values = dict(zip(config.joint_names, config.joint_values))
        values.update(zip(self.core.joint_names, (float(x) for x in q)))
        config.joint_values = [values[n] for n in config.joint_names]

    def move_body(self, guid, matrix):
        body = self.bodies[guid]
        body.frame = frame_from_matrix(matrix)
        self._sync_body(body)
        self.planner.set_robot_cell_state(self.state)
        return body

    def limits(self, scale=1.0):
        model = {j.name: j for j in self.cell.robot_model.get_configurable_joints()}
        v = [model[n].limit.velocity * scale for n in self.core.joint_names]
        return v, [2.0 * x for x in v]

    def take_updates(self):
        updates, self.updates = self.updates, []
        return updates

    def time(self, path, scale=1.0, blend=0.02):
        v, a = self.limits(scale)
        return self.core.time_parameterize([list(map(float, q)) for q in path], v, a, blend_tolerance=blend, time_step=DT)

    def tcp_points(self, segments, arm):
        pts = []
        for seg in segments:
            for q in seg.get("positions", []):
                pts.extend(round(x, 4) for x in arm.fk(q)[:3, 3])
        return pts

    # -- events (overridden) ------------------------------------------------------------

    def plan(self):
        """Plan from ``self.q``; returns (segments, metrics). Raises PlanFailed."""
        return [], {}

    def on_moved(self, body):
        """Re-plan after the user moved ``body``; returns plan() or None to stay put."""
        return True

    def on_picked(self, body):
        """The user clicked a ``clickable`` body; returns whether to re-plan."""
        return False


# ---------------------------------------------------------------------------------------
# a) UR5 around a wall
# ---------------------------------------------------------------------------------------


class WallUR5(Scenario):
    id = "wall"
    title = "UR5 around a wall"
    hint = (
        "Drag the yellow pointer (or the wall). On release the planner finds every IK solution for the pointer, plans around the wall to whichever it reaches "
        "first, and times the motion."
    )
    palette = "ur5"
    q0 = [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0]

    def setup(self):
        self.cell, _ = RobotCellLibrary.ur5()
        wall = Box(0.05, 0.8, 0.8).to_mesh(triangulated=True)
        self.state = self.cell.default_cell_state()
        self.add_body(Body("wall", "Wall", compas_mesh_tris(wall), BRICK, Frame([0.45, 0.0, 0.4], [1, 0, 0], [0, 1, 0]), draggable=True, flat=True), rigid=wall)
        self.planner = NativePlanner(UR5Kinematics())
        self.planner.set_robot_cell(self.cell, self.state)
        self.arm = self.make_arm()
        goal = self.arm.fk([1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
        self.add_body(Body("target", "Target", target_marker_tris(), ACCENT, frame_from_matrix(goal), draggable=True))

    def plan(self):
        self.set_q(self.q)
        target = FrameTarget(self.bodies["target"].frame, TargetMode.ROBOT)
        t0 = time.perf_counter()
        try:
            traj = self.planner.plan_motion(target, self.state, options={"time_step": DT})
        except BackendError as e:
            raise PlanFailed(_reason(e))
        ms = (time.perf_counter() - t0) * 1e3
        stats = traj.attributes["native_stats"]
        timed = timed_from_trajectory(self.q, traj)
        timed["label"] = "Free motion"
        return [timed], {"plan_ms": ms, "duration": stats["duration"], "iterations": stats.get("iterations")}


# ---------------------------------------------------------------------------------------
# b) Panda around a post
# ---------------------------------------------------------------------------------------


class PandaMixin:
    palette = "panda"

    def panda_cell(self):
        self.cell, _ = RobotCellLibrary.panda()
        self.state = self.cell.default_cell_state()
        config = self.state.robot_configuration
        config.joint_values = [0.03 if "finger" in n else v for n, v in zip(config.joint_names, config.joint_values)]

    def plan_to_frame(self, target, seeds_around=()):
        """Numerical IK (several collision-free solutions) then RRT-Connect to whichever
        goal it reaches."""
        t0 = time.perf_counter()
        goals = [list(map(float, q)) for q in self.solve_ik(target, [self.q, *seeds_around])]
        if not goals:
            raise PlanFailed("No collision-free IK solution for the target.")
        ik_ms = (time.perf_counter() - t0) * 1e3
        try:
            path, stats = self.core.plan_motion(list(map(float, self.q)), goals, exact=self.core.has_exact)
        except RuntimeError as e:
            raise PlanFailed(_core_reason(str(e)))
        timed = self.planner._time_parameterize(path, self.planner.client.group, {"time_step": DT})
        ms = (time.perf_counter() - t0) * 1e3
        timed["label"] = "Free motion"
        return [timed], {"plan_ms": ms, "ik_ms": ik_ms, "goals": len(goals), "duration": timed["duration"], "iterations": stats.get("iterations")}


class PostPanda(PandaMixin, Scenario):
    id = "post"
    title = "Panda around a post"
    hint = (
        "Drag the yellow pointer (or the post). Seven axes and no closed-form IK: damped least squares finds several solutions, then the planner goes around "
        "the post to one of them."
    )
    q0 = [-1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]
    camera = {"position": [0.6, -1.85, 1.1], "target": [0.3, 0.0, 0.45]}

    def setup(self):
        self.panda_cell()
        post = Box(0.1, 0.1, 1.2).to_mesh(triangulated=True)
        self.add_body(Body("post", "Post", compas_mesh_tris(post), BRICK, Frame([0.45, 0.0, 0.6], [1, 0, 0], [0, 1, 0]), draggable=True, flat=True), rigid=post)
        self.planner = NativePlanner()
        self.planner.set_robot_cell(self.cell, self.state)
        self.arm = self.make_arm()
        goal = self.arm.fk([1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8])
        self.add_body(Body("target", "Target", target_marker_tris(), ACCENT, frame_from_matrix(goal), draggable=True))

    def plan(self):
        self.set_q(self.q)
        return self.plan_to_frame(matrix_from_frame(self.bodies["target"].frame))


# ---------------------------------------------------------------------------------------
# c) Servo a Panda
# ---------------------------------------------------------------------------------------


class ServoPanda(PandaMixin, Scenario):
    id = "servo"
    title = "Servo a Panda"
    hint = (
        "Jog the gripper with the keys or the pad. Every step is a damped-least-squares move checked for collisions before it is sent; bump the crate and the "
        "arm stops. Home plans a collision-free way back."
    )
    q0 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    controls = {"servo": True}
    camera = {"position": [1.55, -1.25, 1.0], "target": [0.35, 0.0, 0.3]}
    linear_speed = 0.18
    angular_speed = 0.8

    def setup(self):
        self.panda_cell()
        floor = RigidBody.from_mesh(_floor_mesh())
        self.cell.rigid_body_models["floor"] = floor
        self.state.rigid_body_states["floor"] = RigidBodyState(Frame.worldXY())
        self.state.rigid_body_states["floor"].touch_links = ["panda_link0"]
        crate = Box(0.22, 0.32, 0.26).to_mesh(triangulated=True)
        self.add_body(Body("crate", "Crate", compas_mesh_tris(crate), BRICK, Frame([0.55, 0.22, 0.13], [1, 0, 0], [0, 1, 0]), draggable=True, flat=True), rigid=crate)
        pole = Box(0.07, 0.07, 0.9).to_mesh(triangulated=True)
        self.add_body(Body("pole", "Pole", compas_mesh_tris(pole), STEEL, Frame([0.35, -0.42, 0.45], [1, 0, 0], [0, 1, 0]), draggable=True, flat=True), rigid=pole)
        self.planner = NativePlanner()
        self.planner.set_robot_cell(self.cell, self.state)
        self.arm = self.make_arm()
        self.command = None
        self.blocked = False

    def plan(self):
        """Home: a planned (collision-free) motion back to the start configuration."""
        self.set_q(self.q)
        t0 = time.perf_counter()
        try:
            path, stats = self.core.plan_motion(list(map(float, self.q)), [list(self.q0)], exact=self.core.has_exact)
        except RuntimeError as e:
            raise PlanFailed(_core_reason(str(e)))
        timed = self.planner._time_parameterize(path, self.planner.client.group, {"time_step": DT})
        timed["label"] = "Home"
        return [timed], {"plan_ms": (time.perf_counter() - t0) * 1e3, "duration": timed["duration"], "iterations": stats.get("iterations")}

    def on_moved(self, body):
        return False

    def servo(self, twist, dt):
        """One servo period. Returns (moved, blocked_reason)."""
        twist = np.clip(np.asarray(twist, dtype=float), -1, 1)
        if not np.any(twist):
            self.command = None
            return False, None
        current = self.arm.fk(self.q)
        if self.command is None or np.linalg.norm(self.command[:3, 3] - current[:3, 3]) > 0.03:
            self.command = current
        command = self.command.copy()
        command[:3, 3] += twist[:3] * self.linear_speed * dt
        command[:3, :3] = Rotation.from_rotvec(twist[3:] * self.angular_speed * dt).as_matrix() @ command[:3, :3]
        q_new = self.arm.servo_step(self.q, command, dt, home=self.q0)
        if not self.core.motion_free(list(map(float, self.q)), list(map(float, q_new)), None):
            self.command = current
            return False, "Contact ahead: the step would collide, so it was not sent."
        self.command = command
        self.q = q_new
        return True, None


# ---------------------------------------------------------------------------------------
# d) Milling a spiral
# ---------------------------------------------------------------------------------------


class MillUR5(Scenario):
    id = "mill"
    title = "Milling a spiral"
    hint = (
        "Drag or tilt the stock. The planner reaches the start of a coplanar spiral, then follows 5 turns with the cone's axis normal to the face. The tool may"
        " touch the stock; the arm may not."
    )
    palette = "ur5"
    q0 = [-0.2, -1.6, 1.6, -1.6, -1.57, 0.0]
    camera = {"position": [1.3, -1.0, 0.85], "target": [0.4, 0.08, 0.12]}
    turns = 5
    feed_scale = 0.25

    def setup(self):
        self.cell, self.state = RobotCellLibrary.ur5_cone_tool()
        stock = Box(0.28, 0.28, 0.1).to_mesh(triangulated=True)
        # The cone may touch the stock (it is cutting it); everything else must clear it.
        self.add_body(
            Body("stock", "Stock", compas_mesh_tris(stock), "#c7a97a", Frame([0.45, 0.12, 0.05], [1, 0, 0], [0, 1, 0]), draggable=True, flat=True),
            rigid=stock,
            touch_links=["tool0"],
        )
        self.planner = NativePlanner(UR5Kinematics())
        self.planner.set_robot_cell(self.cell, self.state)
        tcf = np.array(self.cell.t_pcf_tcf(self.state, "cone").matrix)
        self.arm = self.make_arm(tcf=tcf)

    def toolpath(self):
        stock = matrix_from_frame(self.bodies["stock"].frame)
        top = 0.05
        pts = [stock @ np.array([x, y, top, 1.0]) for x, y in spiral_points(0.11, 0.012, self.turns, 0.006)]
        return [p[:3] for p in pts], stock[:3, 2]

    def plan(self):
        self.set_q(self.q)
        t0 = time.perf_counter()
        points, normal = self.toolpath()
        lift = 0.06
        approach = points[0] + normal * lift
        try:
            traj = self.planner.plan_motion(PointAxisTarget(approach.tolist(), (-normal).tolist(), TargetMode.TOOL), self.state, options={"time_step": DT, "axis_samples": 24})
        except BackendError as e:
            raise PlanFailed("Approach: " + _reason(e))
        free = timed_from_trajectory(self.q, traj)
        free["label"] = "Approach"
        q1 = free["positions"][-1]
        rot = self.arm.fk(q1)[:3, :3]
        lead_in = [approach + (points[0] - approach) * k / 8 for k in range(1, 9)]
        lead_out = [points[-1] + normal * lift * k / 8 for k in range(1, 9)]
        waypoints = [approach, *lead_in, *points[1:], *lead_out]
        frames = [Frame(p.tolist(), rot[:, 0].tolist(), rot[:, 1].tolist()) for p in waypoints]
        self.set_q(q1)
        cut = None
        for samples in (1, 12):
            try:
                traj = self.planner.plan_cartesian_motion(
                    FrameWaypoints(frames, TargetMode.TOOL),
                    self.state,
                    options={"start_from_state": True, "free_axis_samples": samples, "free_axis_range": 0.6, "max_joint_step": 0.3},
                )
                cut = [list(q1)] + [list(p.joint_values) for p in traj.points]
                break
            except (BackendError, CartesianMotionError) as e:
                reason = _reason(e)
        self.set_q(self.q)
        if cut is None:
            raise PlanFailed("Spiral: " + reason)
        timed = self.time(cut, self.feed_scale, blend=0.005)
        timed["label"] = "Cutting"
        ms = (time.perf_counter() - t0) * 1e3
        return [free, timed], {"plan_ms": ms, "waypoints": len(waypoints), "duration": free["times"][-1] + timed["duration"]}


# ---------------------------------------------------------------------------------------
# e) Plunge through a ring
# ---------------------------------------------------------------------------------------


class RingUR5(Scenario):
    id = "ring"
    title = "Plunge through a ring"
    hint = (
        "Drag or tilt the ring. The robot finds its axis and plunges the wrist through the hole. With the bounding-box model the ring's box fills the hole and "
        "there is no plan; with the distance field there is."
    )
    palette = "ur5"
    q0 = [-2.2, -1.5, 1.5, -1.57, -1.57, 0.0]
    controls = {"obstacles": ["esdf", "boxes"]}
    camera = {"position": [1.45, -1.25, 1.15], "target": [0.4, 0.0, 0.35]}

    def setup(self):
        self.cell, _ = RobotCellLibrary.ur5()
        self.state = self.cell.default_cell_state()
        ring = Torus(0.26, 0.025).to_mesh(u=64, v=12, triangulated=True)
        frame = Frame([0.45, 0.0, 0.42], [1, 0, 0], [0, 1, 0])
        self.add_body(Body("ring", "Ring", compas_mesh_tris(ring), BRICK, frame, draggable=True), rigid=ring)
        self.add_body(Body("axis", "Ring axis", ring_axis_tris(), ACCENT, frame.copy(), opacity=0.6, flat=True))
        # A 1.5 cm field with a 1.5 cm margin: the ring's tube is only 5 cm thick, and a
        # coarser field lets the sphere search graze it (the exact check then rejects).
        esdf = NativePlanner(UR5Kinematics(), obstacles="esdf", esdf_options={"voxel": 0.015, "margin": 0.015})
        self.planners = {"esdf": esdf, "boxes": NativePlanner(UR5Kinematics())}
        for p in self.planners.values():
            p.set_robot_cell(self.cell, self.state)
        self.mode = "esdf"
        self.planner = self.planners["esdf"]
        self.arm = self.make_arm()

    def reset(self):
        super().reset()
        self.bodies["axis"].frame = self.bodies["ring"].frame.copy()
        self.set_mode("esdf")

    def scene_json(self):
        scene = super().scene_json()
        scene["controls"] = {"obstacles": ["esdf", "boxes"], "mode": self.mode}
        return scene

    def set_mode(self, mode):
        self.mode = mode
        self.planner = self.planners[mode]
        self.planner.set_robot_cell_state(self.state)
        self.arm = self.make_arm()

    def move_body(self, guid, matrix):
        body = super().move_body(guid, matrix)
        self.bodies["axis"].frame = body.frame.copy()
        return body

    def plan(self):
        self.set_q(self.q)
        t0 = time.perf_counter()
        ring = matrix_from_frame(self.bodies["ring"].frame)
        center, axis = ring[:3, 3], ring[:3, 2]
        towards_base = -center / max(np.linalg.norm(center), 1e-9)
        sides = sorted([axis, -axis], key=lambda n: -(n[2] + 0.5 * float(n @ towards_base)))
        reason = None
        for n in sides:
            try:
                segments = self._plunge(center, n)
                ms = (time.perf_counter() - t0) * 1e3
                return segments, {"plan_ms": ms, "duration": sum(s["times"][-1] for s in segments if "times" in s) + 0.4, "model": self.mode}
            except PlanFailed as e:
                reason = str(e)
        raise PlanFailed(reason)

    def _plunge(self, center, n):
        """The plunge first (any IK branch, free rotation about the axis), then a free
        motion to its first configuration, so the approach can't strand the arm."""
        above, below = center + n * 0.2, center - n * 0.05
        z = -n
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        frames = [Frame((above + (below - above) * k / 30).tolist(), x.tolist(), y.tolist()) for k in range(31)]
        try:
            traj = self.planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), self.state, options={"free_axis_samples": 12})
        except (BackendError, CartesianMotionError) as e:
            raise PlanFailed("Plunge: " + _reason(e))
        path = [list(p.joint_values) for p in traj.points]
        # Keep the rotation about the axis that worked at the first waypoint for the whole
        # plunge: a straight line with the wrist still, rather than one that wanders.
        rot = self.arm.fk(path[0])[:3, :3]
        steady = [Frame(f.point, rot[:, 0].tolist(), rot[:, 1].tolist()) for f in frames]
        try:
            traj = self.planner.plan_cartesian_motion(FrameWaypoints(steady, TargetMode.ROBOT), self.state)
            path = [list(p.joint_values) for p in traj.points]
        except (BackendError, CartesianMotionError):
            pass
        try:
            goal = self.cell.zero_configuration()
            goal.joint_values = path[0]
            traj = self.planner.plan_motion(ConfigurationTarget(goal), self.state, options={"time_step": DT})
        except BackendError as e:
            raise PlanFailed("Approach: " + _reason(e))
        free = timed_from_trajectory(self.q, traj)
        free["label"] = "Approach"
        plunge = self.time(path, 0.35, blend=0.01)
        plunge["label"] = "Plunge"
        retract = self.time(path[::-1], 0.6, blend=0.01)
        retract["label"] = "Retract"
        return [free, plunge, {"dwell": 0.4, "label": "Through"}, retract]


# ---------------------------------------------------------------------------------------
# f) RFL: touch a cube
# ---------------------------------------------------------------------------------------


class CubesRFL(Scenario):
    """Robot 11 of the RFL on its gantry (X bridge, Y carriage, Z column, then six axes) as
    one 9-axis group. Robot 12 rides on the same bridge, so it moves with X; the second
    bridge with robots 21 and 22, the rails and the hall stand still and are obstacles."""

    id = "rfl"
    title = "RFL: touch a cube"
    hint = (
        "Click a cube. Robot 11 drives its gantry and arm as one 9-axis group, plans around the other cubes and the hall, and touches the face turned most "
        "towards it. Click the same cube again for its next face."
    )
    palette = "rfl"
    group = "robot11_eaXYZ"
    camera = {"position": [5.2, -6.6, 4.6], "target": [5.0, 3.0, 1.6]}
    scale = 6.0
    q0 = [2.0, -1.2, -3.3, 0.0, 0.2, 0.5, 0.0, 0.85, 0.0]
    done = "On the face. Click another cube, or this one again for its next face."
    ready = "Click a cube."
    n_cubes = 22
    side = 0.32
    standoff = 0.15
    contact = 0.008
    gantry_speed = 0.8  # m/s, slower than the URDF's limit, for a hall-sized robot
    arm_scale = 0.6

    def setup(self):
        from scene import static_link_bodies

        self.cell, self.state = RobotCellLibrary.rfl()
        rng = np.random.default_rng(7)
        cube = Box(self.side, self.side, self.side).to_mesh(triangulated=True)
        centres = []
        while len(centres) < self.n_cubes:
            c = rng.uniform([1.4, 0.4, 0.9], [8.6, 5.4, 2.7])
            if all(np.linalg.norm(c - o) > 0.95 for o in centres):
                centres.append(c)
        for i, c in enumerate(centres):
            r = Rotation.from_rotvec(rng.normal(size=3) * 0.6).as_matrix()
            frame = Frame(c.tolist(), r[:, 0].tolist(), r[:, 1].tolist())
            color = CUBE_COLORS[i % len(CUBE_COLORS)]
            self.add_body(Body("cube{}".format(i), "Cube {}".format(i + 1), compas_mesh_tris(cube), color, frame, clickable=True, flat=True), rigid=cube)
        self.planner = NativePlanner()
        self.planner.set_robot_cell(self.cell, self.state, options={"group": self.group})
        self.arm = self.make_arm()
        plate = Box(self.side * 0.8, self.side * 0.8, 0.012).to_mesh(triangulated=True)
        self.add_body(Body("face", "Face", compas_mesh_tris(plate), ACCENT, Frame([0, 0, -40], [1, 0, 0], [0, 1, 0]), flat=True))
        for body in static_link_bodies(self.cell, self.state, self.core, self.palette):
            self.add_body(body)
        self.picked = None  # (cube guid, rank of the face last tried)
        self.touching = None  # (cube guid, outward normal) while the flange is on a face

    def reset(self):
        super().reset()
        self.picked = None
        self.touching = None
        self._seeds_drawn = 0

    def limits(self, scale=1.0):
        v, _ = super().limits(scale)
        prismatic = [int(j.type) == 2 for j in map(self.cell.robot_model.get_joint_by_name, self.core.joint_names)]
        v = [min(x, self.gantry_speed * scale) if p else x * self.arm_scale for x, p in zip(v, prismatic)]
        return v, [1.5 * x for x in v]

    def on_moved(self, body):
        return False

    def on_picked(self, body):
        previous = self.picked
        rank = previous[1] + 1 if previous and previous[0] == body.guid else 0
        if previous and previous[0] != body.guid:
            self.updates.append({"type": "body", "guid": previous[0], "color": self.bodies[previous[0]].color})
        self.updates.append({"type": "body", "guid": body.guid, "color": PICKED})
        self.picked = (body.guid, rank)
        return True

    # -- planning -----------------------------------------------------------------------

    def faces(self, guid):
        """The cube's six faces as (centre, outward normal), the face turned most towards
        the flange (and up, as the robot hangs from the gantry) first."""
        m = matrix_from_frame(self.bodies[guid].frame)
        centre = m[:3, 3]
        flange = self.arm.fk(self.q)[:3, 3]
        towards = flange - centre
        towards /= max(np.linalg.norm(towards), 1e-9)
        normals = [m[:3, k] * sign for k in range(3) for sign in (1, -1)]
        normals.sort(key=lambda n: -(float(n @ towards) + 0.6 * n[2]))
        return [(centre + n * self.side / 2, n) for n in normals]

    def seeds(self, point, n=12):
        """Gantry over the point, the arm in its usual range: first seeds for the IK. Random
        seeds work too, but can park robot 11's carriage against robot 12's, which the
        sphere model under-reports (see README)."""
        rng = np.random.default_rng(self._seeds_drawn)
        self._seeds_drawn += 1
        seeds = []
        for _ in range(n):
            q = rng.uniform(self.arm.lower, self.arm.upper)
            q[0] = point[0] + rng.uniform(-1.2, 1.2)
            q[1] = -(point[1] + rng.uniform(-1.2, 1.2))
            q[2] = -(point[2] + rng.uniform(0.8, 2.0))
            q[3:] = rng.uniform([-np.pi, -0.6, -1.2, -np.pi, -2.0, -np.pi], [np.pi, 1.2, 1.0, np.pi, 2.0, np.pi])
            seeds.append(self.arm.clip(q))
        return seeds

    def line(self, q, end, steps=10):
        """A straight TCP line from where ``q`` puts it to ``end``, orientation held."""
        start = self.arm.fk(q)
        path = [np.asarray(q, dtype=float)]
        for k in range(1, steps + 1):
            target = start.copy()
            target[:3, 3] = start[:3, 3] + (end - start[:3, 3]) * k / steps
            found = self.solve_ik(target, [path[-1]], max_results=1, check_collision=False, max_seeds=1)
            if not found or np.max(np.abs(found[0] - path[-1])) > 0.3:
                return None
            path.append(found[0])
        return path

    def free(self, path):
        return all(self.core.motion_free(list(map(float, a)), list(map(float, b)), None) for a, b in zip(path, path[1:]))

    def touch(self, guid, on):
        """Let the flange touch ``guid`` (the face it is pressed against) or not."""
        self.state.rigid_body_states[guid].touch_links = ["robot11_link_6"] if on else []
        self.planner.set_robot_cell_state(self.state)

    def plan(self):
        if self.picked is None:
            return [], {}
        guid, rank = self.picked
        t0 = time.perf_counter()
        self.set_q(self.q)
        start = np.asarray(self.q, dtype=float)
        touching = self.touching
        try:
            return self._plan_from(start, guid, rank, t0)
        except PlanFailed:
            self.touching = touching
            raise

    def _plan_from(self, start, guid, rank, t0):
        segments = []
        if self.touching is not None and self.core.clearance([list(map(float, start))])[0] > 0.005:
            self.touching = None  # stopped before it got there
        if self.touching is not None:
            # On the last face (or nearly): back off it first.
            last, normal = self.touching
            self.touch(last, True)
            back = self.line(start, self.arm.fk(start)[:3, 3] + normal * self.standoff)
            ok = back is not None and self.free(back)
            self.touch(last, False)
            if not ok:
                raise PlanFailed("Stuck on the last face: no straight way back.")
            retreat = self.time(back, 0.5, blend=0.01)
            retreat["label"] = "Back off"
            segments.append(retreat)
            start = back[-1]
            self.touching = None
        faces = self.faces(guid)
        reason = "No face of this cube can be reached."
        for k in range(6):
            index = (rank + k) % 6
            centre, normal = faces[index]
            try:
                plunge = self._touch_face(guid, start, centre, normal)
            except PlanFailed as e:
                reason = str(e)
                continue
            self.picked = (guid, index)
            m = np.eye(4)
            m[:3, :3] = _frame_with_z(normal)
            m[:3, 3] = centre + normal * 0.007
            self.updates.append({"type": "body", "guid": "face", "pose": frame_rows(frame_from_matrix(m))})
            self.bodies["face"].frame = frame_from_matrix(m)
            segments += plunge["segments"]
            metrics = plunge["metrics"]
            metrics["plan_ms"] = (time.perf_counter() - t0) * 1e3
            metrics["face"] = index + 1
            metrics["duration"] = sum(s["times"][-1] for s in segments if "times" in s) + 0.5
            self.touching = (guid, normal)
            self.q = start  # the session streams from here
            return segments, metrics
        self.bodies["face"].frame = Frame([0, 0, -40], [1, 0, 0], [0, 1, 0])
        self.updates.append({"type": "body", "guid": "face", "pose": frame_rows(self.bodies["face"].frame)})
        raise PlanFailed(reason)

    def _touch_face(self, guid, start, centre, normal):
        """IK for the flange over the face (axis into it, free about it), a free motion
        there, then straight in until it touches."""
        t0 = time.perf_counter()
        target = np.eye(4)
        target[:3, :3] = _frame_with_z(-normal)
        target[:3, 3] = centre + normal * self.standoff
        # Flange axis into the face, free about it.
        goals = self.solve_ik(target, [start, *self.seeds(target[:3, 3])], max_results=6, free_axis=True, margin=0.01)
        if not goals:
            raise PlanFailed("No collision-free IK solution over this face.")
        ik_ms = (time.perf_counter() - t0) * 1e3
        # The straight move in, from each goal, before committing to a free motion there.
        self.touch(guid, True)
        try:
            ins = [(g, self.line(g, centre + normal * self.contact)) for g in goals]
            ins = [(g, p) for g, p in ins if p is not None and self.free(p)]
        finally:
            self.touch(guid, False)
        if not ins:
            raise PlanFailed("No straight way onto a face.")
        try:
            path, stats = self.core.plan_motion(list(map(float, start)), [list(map(float, g)) for g, _ in ins], exact=self.core.has_exact)
        except RuntimeError as e:
            raise PlanFailed(_core_reason(str(e)))
        end = np.asarray(path[-1], dtype=float)
        approach = min(ins, key=lambda gp: np.max(np.abs(gp[0] - end)))[1]
        free = self.time(path)
        free["label"] = "Free motion"
        press = self.time(approach, 0.25, blend=0.005)
        press["label"] = "Touch"
        segments = [free, press, {"dwell": 0.5, "label": "Touching"}]
        return {"segments": segments, "metrics": {"ik_ms": ik_ms, "goals": len(ins), "iterations": stats.get("iterations")}}


CUBE_COLORS = ["#c7a97a", "#b98f5c", "#d4bf98", "#a8865e"]
PICKED = "#e0672b"


def _frame_with_z(z):
    """A rotation matrix whose z axis is ``z`` (x and y arbitrary but stable)."""
    z = np.asarray(z, dtype=float) / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _floor_mesh():
    from compas.datastructures import Mesh

    return Mesh.from_stl(compas_fab.get("planning_scene/floor.stl"))


def _reason(error):
    name = type(error).__name__
    text = str(error.args[0]) if error.args else ""
    return {
        "MPNoIKSolutionError": "Out of reach: no IK solution for the target.",
        "MPTargetInCollisionError": "Every IK solution for the target is in collision.",
        "MPStartStateInCollisionError": "The robot is in collision where it stands.",
        "MPNoPlanFoundError": "No collision-free path found.",
        "CartesianMotionError": "No collision-free Cartesian path: " + text if text else "No collision-free Cartesian path.",
    }.get(name, text or name)


def _core_reason(message):
    if message.startswith("start_in_collision"):
        return "The robot is in collision where it stands."
    if message.startswith("goals_in_collision"):
        return "Every IK solution for the target is in collision."
    return "No collision-free path found."


SCENARIOS = [WallUR5, PostPanda, ServoPanda, MillUR5, RingUR5, CubesRFL]

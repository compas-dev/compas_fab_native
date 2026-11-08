"""The five live scenarios. Each owns a robot cell, a NativePlanner and the bodies the
browser shows; the session streams whatever motion they plan."""

import time

import numpy as np
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Torus
from kin import Arm
from kin import random_seeds
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

    def __init__(self, guid, name, tris, color, frame, draggable=False, body_id=None, opacity=1.0, flat=False):
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

    def json(self):
        return {
            "guid": self.guid,
            "name": self.name,
            "mesh": mesh_json(self.tris),
            "color": self.color,
            "pose": frame_rows(self.frame),
            "draggable": self.draggable,
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

    def __init__(self):
        self.bodies = {}
        self.q = None
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
            "vmax": [model[n].limit.velocity for n in names],
        }

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
        v = [x * scale for x in self.joints()["vmax"]]
        return v, [2.0 * x for x in v]

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
        """Numeric IK (several solutions) then RRT-Connect to whichever goal it reaches."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(len(self._ik_seeds_used))
        self._ik_seeds_used.append(1)
        seeds = random_seeds(self.arm, 24, rng, around=[self.q, *seeds_around])
        solutions = self.arm.ik(target, seeds, max_solutions=8, max_iterations=60)
        if not solutions:
            raise PlanFailed("Out of reach: no IK solution for the target.")
        clearance = self.core.clearance([list(map(float, s)) for s in solutions])
        goals = [list(map(float, s)) for s, c in zip(solutions, clearance) if c > 0]
        if not goals:
            raise PlanFailed("Every IK solution for the target is in collision.")
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
        self._ik_seeds_used = []
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
        self._ik_seeds_used = []
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


SCENARIOS = [WallUR5, PostPanda, ServoPanda, MillUR5, RingUR5]

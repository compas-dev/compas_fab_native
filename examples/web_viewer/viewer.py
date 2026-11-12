"""A web viewer for the native planner: pick a robot, drag a target, plan.

    pip install compas_fab_native compas_threejs
    python examples/web_viewer/viewer.py

Then open the URL it prints. Your own robot instead of the library ones:

    python examples/web_viewer/viewer.py --urdf robot.urdf --srdf robot.srdf --meshes path/to/packages

See README.md next to this file for what to try.
"""

import argparse
import json
import queue
import time

import compas_pb
import numpy as np
from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Transformation
from compas_fab import backends
from compas_fab.backends import CollisionCheckError
from compas_fab.backends import InverseKinematicsError
from compas_fab.backends import UR5Kinematics
from compas_fab.backends import UR10eKinematics
from compas_fab.robots import FrameTarget
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import PointAxisTarget
from compas_fab.robots import PointAxisWaypoints
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCell
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import RobotSemantics
from compas_fab.robots import TargetMode
from compas_robots import RobotModel
from compas_robots.model import Joint
from compas_robots.resources import LocalPackageMeshLoader
from compas_threejs.materials import Material
from compas_threejs.metadata import Metadata
from compas_threejs.tag import TextTag
from compas_threejs.ui import Button
from compas_threejs.ui import Checkbox
from compas_threejs.ui import Selection
from compas_threejs.ui import Slider
from compas_threejs.viewer import App

from compas_fab_native import NativePlanner
from compas_fab_native.model import collision_items
from compas_fab_native.model import joint_limits
from compas_fab_native.model import meshes_to_triangles
from compas_fab_native.model import rigid_body_triangles
from compas_fab_native.model import tool_triangles

# Robots from the compas_fab library, with the closed-form solver URs need (others: None).
LIBRARY = {
    "UR5": (RobotCellLibrary.ur5, UR5Kinematics),
    "UR5 with a cone tool": (RobotCellLibrary.ur5_cone_tool, UR5Kinematics),
    "UR10e carrying a beam": (RobotCellLibrary.ur10e_gripper_one_beam, UR10eKinematics),
    "ABB IRB 4600 with a printing tool": (RobotCellLibrary.abb_irb4600_40_255_printing_tool, None),
    "Staubli TX2-60L": (RobotCellLibrary.staubli_tx2_60l, None),
    "Panda (7 axes)": (RobotCellLibrary.panda, None),
}

LINK_COLOR = Color.from_hex("#c9ced6")
TOOL_COLOR = Color.from_hex("#f2b705")
WORKPIECE_COLOR = Color.from_hex("#c08a4b")
STATIC_COLOR = Color.from_hex("#9d9a93")
OBSTACLE_COLOR = Color.from_hex("#5b8fd1")
TARGET_COLOR = Color.from_hex("#2bb673")
FAILED_COLOR = Color.from_hex("#d13b3b")
JOINT_SPEED = 1.0  # rad/s for Cartesian paths (the planner does not time them) at full speed
JOG_STEP = np.radians(10)


# ---------------------------------------------------------------------------
# Loading robots


def load_urdf(urdf, srdf=None, meshes=None):
    """A robot cell from a URDF, and an SRDF if there is one. Without an SRDF, the planning
    group is the chain from the root to the last movable joint in the URDF (and the fixed
    links after it, up to a branch). Pass an SRDF when that guess is wrong, for example
    when the last movable joint is a gripper finger."""
    model = RobotModel.from_urdf_file(urdf)
    if meshes:
        model.load_geometry(LocalPackageMeshLoader(meshes, ""))
    if srdf:
        semantics = RobotSemantics.from_srdf_file(srdf, model)
    else:
        end = model.get_configurable_joints()[-1].child.link
        # Carry on through fixed joints (flange, tool0) while the chain does not branch.
        children = model.find_children_joints(model.get_link_by_name(end))
        while len(children) == 1 and children[0].type == Joint.FIXED:
            end = children[0].child.link
            children = model.find_children_joints(model.get_link_by_name(end))
        links = [link.name for link in model.iter_link_chain(model.root.name, end)]
        joints = [joint.name for joint in model.iter_joint_chain(model.root.name, end)]
        semantics = RobotSemantics(groups={"arm": {"links": links, "joints": joints}}, main_group_name="arm", end_effectors=[end])
    cell = RobotCell(model, semantics)
    return cell, cell.default_cell_state()


def tris_to_mesh(tris):
    """A welded mesh from a (T, 3, 3) triangle soup."""
    v = np.round(np.asarray(tris, dtype=float).reshape(-1, 3), 6)
    vertices, faces = np.unique(v, axis=0, return_inverse=True)
    return Mesh.from_vertices_and_faces(vertices.tolist(), faces.reshape(-1, 3).tolist())


def to_matrix(frame):
    return np.array(Transformation.from_frame(frame or Frame.worldXY()).matrix)


class Part:
    """One mesh in the viewer. Moving parts follow a link of the core (`link` is its index)."""

    def __init__(self, mesh, link=None, joint=None):
        self.mesh = mesh
        self.link = link
        self.joint = joint  # the joint a click on this part jogs
        self.shown = np.eye(4)  # the pose the browser shows it at

    @property
    def guid(self):
        return str(self.mesh.guid)


# ---------------------------------------------------------------------------
# The viewer


class RobotViewer(App):
    def __init__(self, robots, port):
        super().__init__(websocket_port=port)
        self.robots = robots
        self.commands = queue.Queue()  # filled by the browser, emptied by the main loop
        self.parts = {}
        self.boxes = {}
        self.playback = None
        self.free_axis = False
        self.exact = False
        self.speed = 0.5
        self.obstacles = "boxes"
        self.loop = self.tick
        self.loop_interval = 1 / 60

        self.world_axis = False
        names = list(robots)
        self.add_ui_element(Selection(names, names[0], self.queue("robot"), label="Robot"))
        self.add_ui_element(Button("Free motion to target", self.queue("free"), label="Plan"))
        self.add_ui_element(Button("Straight line to target", self.queue("line")))
        self.add_ui_element(Checkbox("Free rotation about the tool axis", self.queue("free_axis")))
        self.add_ui_element(Slider(0.1, 1.0, 0.05, self.speed, self.queue("speed"), label="Speed"))
        self.add_ui_element(Button("Next IK solution", self.queue("next_ik"), label="Target"))
        self.add_ui_element(Button("Target to robot", self.queue("target_to_robot")))
        self.add_ui_element(Button("Random configuration", self.queue("random"), label="Robot pose"))
        self.add_ui_element(Button("Add a box", self.queue("add_box"), label="Obstacles"))
        self.add_ui_element(Button("Remove boxes", self.queue("remove_boxes")))
        self.add_ui_element(Selection(["boxes", "distance field"], "boxes", self.queue("obstacles"), label="Obstacle model"))
        self.add_ui_element(Checkbox("Check on exact meshes", self.queue("exact"), label="Collisions"))
        self.jog_buttons = [Button("+10°", lambda obj, _: self.commands.put(("jog", (obj, 1)))), Button("−10°", lambda obj, _: self.commands.put(("jog", (obj, -1))))]
        self.status = TextTag(Point(0, 0, 0), "")
        self.target_tag = TextTag(Point(0, 0, 0), "")
        self.add_tag(self.status)
        self.add_tag(self.target_tag)
        self.load(names[0])

    def queue(self, name):
        return lambda value=None: self.commands.put((name, value))

    def on_message(self, message, workspace_id="main"):
        # Dragging with the gizmo moves the Python object too (see compas_threejs' Inbox);
        # the planner reacts in the main loop.
        super().on_message(message, workspace_id)
        data = json.loads(message.decode())
        if data.get("dispatch") == "object_transform":
            self.commands.put(("moved", (data["guid"], np.array(data["matrix"], dtype=float))))

    # -- main loop --------------------------------------------------------------------------

    def tick(self, _):
        while not self.commands.empty():
            name, value = self.commands.get()
            try:
                getattr(self, "do_" + name)(value)
            except Exception as e:  # show planning errors in the browser, keep running
                self.say("{}: {}".format(type(e).__name__, e))
        self.play()

    def say(self, text):
        self.status.text = text
        self.update_tag(self.status)

    # -- robot ------------------------------------------------------------------------------

    def load(self, name):
        self.start_spinner("Loading {}".format(name))
        try:
            for part in list(self.parts.values()):
                self.remove_object(part.mesh)
                self.outbox.forget(("pose", part.guid))
            for box in list(self.boxes.values()):
                self.remove_object(box)
            self.parts, self.boxes, self.playback = {}, {}, None
            if hasattr(self, "target"):
                self.remove_object(self.target)

            cell, state, solver = self.robots[name]()
            self.cell, self.state = cell, state
            self.planner = NativePlanner(solver, obstacles="esdf" if self.obstacles == "distance field" else "boxes")
            self.planner.set_robot_cell(cell, state)
            self.core = self.planner.client.core
            self.base = to_matrix(state.robot_base_frame)
            config = state.robot_configuration
            values = dict(zip(config.joint_names, config.joint_values))
            self.q = np.array([values.get(j, 0.0) for j in self.core.joint_names])
            tool = next((t for t, s in state.tool_states.items() if s.attached_to_group), None)
            self.mode = TargetMode.TOOL if tool else TargetMode.ROBOT
            self.add_robot_parts(tool)
            self.solutions, self.solution = [], 0

            # Start with the target in front of the robot, pointing down, and the robot at it.
            base = Point(*self.base[:3, 3])
            self.reach = reach = base.distance_to_point(self.planner.forward_kinematics(self.current_state(), self.mode).point)
            self.target_frame = Frame(base + [0.5 * reach, 0, 0.4 * reach], [1, 0, 0], [0, -1, 0])
            self.target = tris_to_mesh(target_marker(max(1.0, reach)))
            self.target.transform(Transformation.from_frame(self.target_frame))
            self.target_material = Material(color=TARGET_COLOR)
            self.add_geometry(self.target, self.target_material)
            self.set_view(base + [1.6 * reach, -2.0 * reach, 1.3 * reach], base + [0.3 * reach, 0, 0.4 * reach])
            self.status.point = base + [0, 0, 1.3 * reach]
            self.pose_robot(self.q)
            self.solve_ik()
            ik = "closed-form" if self.planner.has_closed_form_ik else "numerical"
            self.say("{}: {} IK. Click the green pointer and drag it (W: move, E: rotate).".format(name, ik))
        finally:
            self.stop_spinner()

    def add_robot_parts(self, tool):
        cell, state, core = self.cell, self.state, self.core
        model = cell.robot_model
        # Links the planning group moves: in link coordinates, posed by the core.
        for index, name in enumerate(core.link_names):
            link = model.get_link_by_name(name)
            elements = link.visual or link.collision
            tris = meshes_to_triangles(collision_items(elements)) if elements else []
            if len(tris):
                joint = self.jogged_joint(model, link)
                metadata = Metadata(link=name, joint=core.joint_names[joint] if joint is not None else "-")
                self.add_part(Part(tris_to_mesh(tris), index, joint), LINK_COLOR, metadata)
        # What the robot carries: the tool and the workpieces it holds, on the end-effector link.
        if tool:
            t = to_matrix(state.tool_states[tool].attachment_frame)
            tris = tool_triangles(cell.tool_models[tool], state.tool_states[tool].configuration)
            self.add_part(Part(tris_to_mesh(tris @ t[:3, :3].T + t[:3, 3]), core.end_effector_link), TOOL_COLOR, Metadata(tool=tool))
        for body, body_state in state.rigid_body_states.items():
            tris = rigid_body_triangles(cell, body)
            if not len(tris):
                continue
            if body_state.attached_to_tool:
                t = np.array(cell.t_pcf_ocf(state, body).matrix)
                self.add_part(Part(tris_to_mesh(tris @ t[:3, :3].T + t[:3, 3]), core.end_effector_link), WORKPIECE_COLOR, Metadata(workpiece=body))
            elif not body_state.attached_to_link:
                t = to_matrix(body_state.frame)
                self.add_part(Part(tris_to_mesh(tris @ t[:3, :3].T + t[:3, 3])), STATIC_COLOR, Metadata(obstacle=body))
        # Links the group cannot move (a multi-robot cell, a track): drawn once, where they are.
        moving = set(core.link_names)
        config = state.robot_configuration
        for link in model.links:
            elements = link.visual or link.collision
            if link.name in moving or not elements:
                continue
            t = self.base @ to_matrix(model.forward_kinematics(config, link.name))
            tris = meshes_to_triangles([(m, t @ tm, s) for m, tm, s in collision_items(elements)])
            if len(tris):
                self.add_part(Part(tris_to_mesh(tris)), STATIC_COLOR, Metadata(link=link.name))

    def jogged_joint(self, model, link):
        """Index of the nearest planned joint at or above a link (what clicking it jogs)."""
        names = self.core.joint_names
        while link is not None and link.parent_joint is not None:
            if link.parent_joint.name in names:
                return names.index(link.parent_joint.name)
            link = model.get_link_by_name(link.parent_joint.parent.link)
        return None

    def add_part(self, part, color, metadata):
        actions = self.jog_buttons if part.joint is not None else None
        self.add_geometry(part.mesh, Material(color=color), metadata=metadata, actions=actions)
        self.parts[part.guid] = part

    def current_state(self):
        config = self.state.robot_configuration
        values = dict(zip(self.core.joint_names, self.q.tolist()))
        config.joint_values = [values.get(name, value) for name, value in zip(config.joint_names, config.joint_values)]
        return self.state

    # -- showing poses ----------------------------------------------------------------------

    def pose_robot(self, q):
        """Pose the moving parts for joint values `q`. Sends each part's change of pose
        (small messages, no meshes), and keeps its absolute pose for browsers that connect
        later."""
        frames = self.core.link_frames([float(x) for x in q])
        for part in self.parts.values():
            if part.link is None:
                continue
            m = np.eye(4)
            m[:3, :] = np.array(frames[part.link]).reshape(3, 4)
            pose = self.base @ m
            delta = pose @ np.linalg.inv(part.shown)
            if np.allclose(delta, np.eye(4), atol=1e-9):
                continue
            self.send_transform(part.guid, delta, pose)
            part.shown = pose

    def send_transform(self, guid, delta, pose):
        message = {"dispatch": "handle_geometry", "type": "apply_transform", "guid": guid, "matrix": delta.tolist()}
        self.outbox.send_bytes(compas_pb.pb_dump_bts(message), persist=False)
        message["matrix"] = pose.tolist()
        self.outbox.send_bytes(compas_pb.pb_dump_bts(message), ("pose", guid), broadcast=False)

    def play(self):
        if self.playback is None:
            return
        times, path, t0 = self.playback
        t = time.monotonic() - t0
        if t >= times[-1]:
            self.q = path[-1].copy()
            self.pose_robot(self.q)
            self.playback = None
            self.check()
            return
        i = int(np.searchsorted(times, t, side="right")) - 1
        u = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
        self.pose_robot(path[i] + (path[i + 1] - path[i]) * u)

    def start_playback(self, trajectory):
        if not trajectory.points:
            return
        names = trajectory.joint_names or self.core.joint_names
        order = [names.index(j) for j in self.core.joint_names]
        path = np.array([[p.joint_values[k] for k in order] for p in trajectory.points])
        path = np.vstack([self.q, path])
        times = [0.0] + [p.time_from_start.seconds for p in trajectory.points]
        if times[-1] <= 0:  # untimed (Cartesian): a constant joint speed
            steps = np.abs(np.diff(path, axis=0)).max(axis=1) / (JOINT_SPEED * self.speed)
            times = np.concatenate([[0.0], np.cumsum(np.maximum(steps, 1e-3))])
        self.playback = (np.asarray(times), path, time.monotonic())

    # -- checks and IK ----------------------------------------------------------------------

    def check(self):
        """Collision check and clearance of the current configuration."""
        state = self.current_state()
        clearance = self.core.clearance([self.q.tolist()])[0]
        try:
            self.planner.check_collision(state, options={"exact": self.exact})
            if np.isfinite(clearance):
                self.say("Collision-free. Clearance {:.1f} cm (sphere model).".format(clearance * 100))
            else:
                self.say("Collision-free.")
        except CollisionCheckError as e:
            self.say("In collision: {}".format(e))

    def tag_target(self, text):
        self.target_tag.text = text
        self.target_tag.point = Point(*self.target_frame.point) + [0, 0, 0.15 * self.reach]
        self.update_tag(self.target_tag)

    def make_target(self):
        if self.free_axis:
            return PointAxisTarget(self.target_frame.point, self.target_frame.zaxis, self.mode)
        return FrameTarget(self.target_frame, self.mode)

    def solve_ik(self):
        self.playback = None
        try:
            solutions = list(self.planner.iter_inverse_kinematics(self.make_target(), self.current_state(), options={"exact": self.exact}))
        except InverseKinematicsError:
            solutions = []
        # Closest to where the robot is first.
        values = [np.array([dict(zip(s.joint_names, s.joint_values))[j] for j in self.core.joint_names]) for s in solutions]
        self.solutions = sorted(values, key=lambda q: np.abs(q - self.q).sum())
        self.solution = 0
        ok = bool(self.solutions)
        self.target_material.color = TARGET_COLOR if ok else FAILED_COLOR
        self.update_material(self.target_material)
        self.tag_target("{} IK solutions".format(len(self.solutions)) if ok else "no collision-free IK solution")
        if ok:
            self.q = self.solutions[0].copy()
            self.pose_robot(self.q)
            self.check()

    # -- commands from the browser ----------------------------------------------------------

    def do_robot(self, name):
        self.load(name)

    def do_moved(self, value):
        guid, matrix = value
        delta = Transformation.from_matrix(matrix.tolist())
        if guid == str(self.target.guid):
            self.target_frame.transform(delta)
            self.solve_ik()
        elif guid in self.boxes:
            self.update_box(guid)
        elif guid in self.parts:
            # Robot parts follow the robot: put this one back.
            part = self.parts[guid]
            part.mesh.transform(delta.inverse())
            self.update_geometry(part.mesh)
            if part.link is not None:
                self.send_transform(guid, part.shown, part.shown)

    def do_next_ik(self, _):
        if not self.solutions:
            return self.solve_ik()
        self.solution = (self.solution + 1) % len(self.solutions)
        self.q = self.solutions[self.solution].copy()
        self.pose_robot(self.q)
        self.tag_target("IK solution {} of {}".format(self.solution + 1, len(self.solutions)))
        self.check()

    def do_target_to_robot(self, _):
        tcp = self.planner.forward_kinematics(self.current_state(), self.mode)
        move = Transformation.from_frame(tcp) * Transformation.from_frame(self.target_frame).inverse()
        self.transform_geometry(self.target, move)
        self.target_frame = tcp
        self.tag_target("")

    def do_random(self, _):
        self.playback = None
        config = self.cell.random_configuration(self.cell.main_group_name)
        values = dict(zip(config.joint_names, config.joint_values))
        self.q = np.array([values[j] for j in self.core.joint_names])
        self.pose_robot(self.q)
        self.check()

    def do_jog(self, value):
        obj, sign = value
        part = self.parts.get(str(obj.guid))
        if part is None or part.joint is None:
            return
        self.playback = None
        joint = self.cell.robot_model.get_joint_by_name(self.core.joint_names[part.joint])
        low, high = joint_limits(joint)
        step = 0.05 if joint.type == Joint.PRISMATIC else JOG_STEP  # 5 cm or 10°
        self.q[part.joint] = min(max(self.q[part.joint] + sign * step, low), high)
        self.pose_robot(self.q)
        self.check()

    def do_free(self, _):
        self.say("Planning a free motion…")
        trajectory = self.planner.plan_motion(self.make_target(), self.current_state(), options={"velocity_scaling": self.speed})
        stats = trajectory.attributes["native_stats"]
        self.start_playback(trajectory)
        self.say("Free motion: {:.1f} s, {:.0f} iterations.".format(stats["duration"], stats["iterations"]))

    def do_line(self, _):
        self.say("Planning a straight line…")
        state = self.current_state()
        start = self.planner.forward_kinematics(state, self.mode)
        if self.free_axis:
            waypoints = PointAxisWaypoints([(start.point, start.zaxis), (self.target_frame.point, self.target_frame.zaxis)], self.mode)
        else:
            waypoints = FrameWaypoints([start, self.target_frame], self.mode)
        trajectory = self.planner.plan_cartesian_motion(waypoints, state, options={"start_from_state": True})
        self.start_playback(trajectory)
        self.say("Straight line: {} points.".format(len(trajectory.points)))

    def do_free_axis(self, value):
        self.free_axis = bool(value)

    def do_exact(self, value):
        self.exact = bool(value)
        self.check()

    def do_speed(self, value):
        self.speed = float(value)

    def do_obstacles(self, value):
        self.obstacles = value
        self.planner.obstacles = "esdf" if value == "distance field" else "boxes"
        self.planner.set_robot_cell(self.cell, self.current_state())
        self.core = self.planner.client.core
        self.check()

    # -- obstacles --------------------------------------------------------------------------

    def do_add_box(self, _):
        # Beside the target, so moving the target across means going around it.
        size = 0.25 * self.reach
        box = Box(size, size, size, Frame(self.target_frame.point + [0, 0.35 * self.reach, 0], [1, 0, 0], [0, 1, 0]))
        self.add_geometry(box, Material(color=OBSTACLE_COLOR, transparent=True, opacity=0.8), metadata=Metadata(obstacle="box"))
        self.boxes[str(box.guid)] = box
        self.update_box(str(box.guid))
        self.say("Box added. Click it to move (W), rotate (E) or scale (R) it.")

    def do_remove_boxes(self, _):
        for guid, box in self.boxes.items():
            self.remove_object(box)
            del self.cell.rigid_body_models[guid]
        self.boxes = {}
        self.reset_cell()

    def update_box(self, guid):
        """Boxes are rigid bodies of the robot cell: its mesh, placed at the box's frame."""
        box = self.boxes[guid]
        local = Box(box.xsize, box.ysize, box.zsize).to_mesh(triangulated=True)
        self.cell.rigid_body_models[guid] = RigidBody.from_mesh(local)
        self.reset_cell()
        self.check()

    def reset_cell(self):
        """The cell's bodies changed: rebuild the state (keeping poses) and resend the cell."""
        old = self.state
        state = self.cell.default_cell_state()
        state.robot_configuration = old.robot_configuration
        state.robot_base_frame = old.robot_base_frame
        for key, value in old.tool_states.items():
            state.tool_states[key] = value
        for key, value in old.rigid_body_states.items():
            if key in state.rigid_body_states:
                state.rigid_body_states[key] = value
        for guid, box in self.boxes.items():
            state.rigid_body_states[guid] = RigidBodyState(box.frame.copy())
        self.state = state
        self.planner.set_robot_cell(self.cell, state)
        self.core = self.planner.client.core


def target_marker(scale=1.0):
    """A pointer for a frame target, about 13 cm long at scale 1: tip at the origin, shaft
    along +z (beyond the target, so the robot does not hide it), and a fin along +x so
    rotations about z show."""
    size = 0.09
    shaft = Box(0.012, 0.012, size, Frame([0, 0, size / 2 + 0.02], [1, 0, 0], [0, 1, 0]))
    fin = Box(0.05, 0.004, 0.03, Frame([0.025, 0, 0.07], [1, 0, 0], [0, 1, 0]))
    tris = [box_tris(shaft), box_tris(fin)]
    r, h = 0.022, 0.035
    base = [(r, r, h), (-r, r, h), (-r, -r, h), (r, -r, h)]
    head = [[base[(i + 1) % 4], base[i], (0, 0, 0)] for i in range(4)] + [[base[0], base[1], base[2]], [base[0], base[2], base[3]]]
    tris.append(np.array(head, dtype=float))
    return np.concatenate(tris) * scale


def box_tris(box):
    v, f = box.to_mesh(triangulated=True).to_vertices_and_faces(triangulated=True)
    return np.asarray(v, dtype=float)[np.asarray(f, dtype=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", help="your robot's URDF (instead of the library robots)")
    parser.add_argument("--srdf", help="its SRDF (optional: without it, the planning group is the joint chain)")
    parser.add_argument("--meshes", help="folder the URDF's mesh paths are relative to (package:// URLs)")
    parser.add_argument("--solver", help="compas_fab closed-form solver for it, e.g. UR5Kinematics (optional)")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    robots = {}
    if args.urdf:

        def custom():
            cell, state = load_urdf(args.urdf, args.srdf, args.meshes)
            return cell, state, getattr(backends, args.solver)() if args.solver else None

        robots[cell_name(args.urdf)] = custom
    for name, (factory, solver) in LIBRARY.items():
        robots[name] = lambda f=factory, s=solver: (*f(), s() if s else None)

    RobotViewer(robots, args.port).start(show=True)


def cell_name(path):
    return "Your robot ({})".format(path.replace("\\", "/").rsplit("/", 1)[-1])


if __name__ == "__main__":
    main()

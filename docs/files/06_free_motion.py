"""06 - Free motion around obstacles.

A Cartesian path says where the tool goes. Free motion only says where the robot ends up,
and the planner finds a way around obstacles. It tries the straight joint motion first,
then searches (RRT-Connect), then shortens and smooths the path.
"""

from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Vector
from compas_fab.backends import UR5Kinematics
from compas_fab.robots import ConfigurationTarget
from compas_fab.robots import FrameTarget
from compas_fab.robots import PointAxisTarget
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# A UR5 with a wall in front of it.
robot_cell, _ = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["wall"] = RigidBody.from_mesh(Box(0.05, 0.8, 0.8).to_mesh(triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["wall"] = RigidBodyState(Frame([0.45, 0.0, 0.4], [1, 0, 0], [0, 1, 0]))
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# From left of the wall to right of it. Moving straight would hit the wall.
state.robot_configuration.joint_values = [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0]
goal = robot_cell.zero_configuration()
goal.joint_values = [1.2, -1.2, 1.4, -1.8, -1.57, 0.0]
trajectory = planner.plan_motion(ConfigurationTarget(goal), state)
stats = trajectory.attributes["native_stats"]
print("around the wall: {:.0f} iterations".format(stats["iterations"]))
print("joint-space length: {:.2f} before smoothing, {:.2f} after".format(stats["initial_cost"], stats["cost"]))

# To a frame. The planner aims at all IK solutions of the frame at once.
target = FrameTarget(Frame([0.3, 0.35, 0.3], [1, 0, 0], [0, -1, 0]), TargetMode.ROBOT)
trajectory = planner.plan_motion(target, state)
print("to a frame: {:.2f} s".format(trajectory.attributes["native_stats"]["duration"]))

# To a point and an axis, like a drill: any rotation about the axis is fine.
target = PointAxisTarget(Point(0.3, -0.35, 0.3), Vector(0, 0, -1), TargetMode.ROBOT)
trajectory = planner.plan_motion(target, state, options={"axis_samples": 16})
print("to a point and axis: {:.2f} s".format(trajectory.attributes["native_stats"]["duration"]))

# Trajectories are timed: within the joint speed and acceleration limits, starting and
# ending at rest. Each point has its time, velocities and accelerations.
print("timed: {} points over {:.2f} s".format(len(trajectory.points), trajectory.attributes["native_stats"]["duration"]))
slow = planner.plan_motion(ConfigurationTarget(goal), state, options={"velocity_scaling": 0.25, "max_acceleration": 1.0})
print("at a quarter of the speed: {:.2f} s".format(slow.attributes["native_stats"]["duration"]))
middle = trajectory.points[len(trajectory.points) // 2]
print("halfway: t = {:.2f} s, joint speeds {}".format(middle.time_from_start.seconds, ["{:+.2f}".format(v) for v in middle.velocities]))
untimed = planner.plan_motion(ConfigurationTarget(goal), state, options={"time_parameterization": False})
print("untimed, corners only: {} points".format(len(untimed.points)))

# Free motion works for any robot, like this 7-axis Panda (no solver given).
panda_cell, _ = RobotCellLibrary.panda()
panda_cell.rigid_body_models["post"] = RigidBody.from_mesh(Box(0.1, 0.1, 1.2).to_mesh(triangulated=True))
panda_state = panda_cell.default_cell_state()
panda_state.rigid_body_states["post"] = RigidBodyState(Frame([0.45, 0.0, 0.6], [1, 0, 0], [0, 1, 0]))
panda = NativePlanner()
panda.set_robot_cell(panda_cell, panda_state)
start = panda_cell.zero_configuration()
start.joint_values = [-1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]
panda_state.robot_configuration = panda_state.robot_configuration.merged(start)
goal = panda_cell.zero_configuration()
goal.joint_values = [1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]
trajectory = panda.plan_motion(ConfigurationTarget(goal), panda_state)
print("panda around a post: {:.2f} s".format(trajectory.attributes["native_stats"]["duration"]))

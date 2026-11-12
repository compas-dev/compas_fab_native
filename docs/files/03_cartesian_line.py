"""03 - A Cartesian path: a straight line of frames.

The planner solves IK for every frame, keeps all solutions, and picks the smoothest
collision-free sequence through them. So it does not get stuck on one IK branch.
"""

from compas.geometry import Frame
from compas_fab.backends import UR5Kinematics
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

robot_cell, state = RobotCellLibrary.ur5()
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# 100 frames on a 60 cm line, flange pointing down.
frames = [Frame([0.45, -0.3 + 0.6 * i / 99, 0.4], [1, 0, 0], [0, -1, 0]) for i in range(100)]
waypoints = FrameWaypoints(frames, TargetMode.ROBOT)

# Plan. The result is also checked on the exact meshes.
trajectory = planner.plan_cartesian_motion(waypoints, state)
print("{} points".format(len(trajectory.points)))

# What the planner did.
stats = trajectory.attributes["native_stats"]
print("IK solutions considered: {:.0f}, collision-free: {:.0f}".format(stats["ik_nodes"], stats["free_nodes"]))

# The path is smooth: the largest joint change between two points.
points = trajectory.points
largest = max(max(abs(a - b) for a, b in zip(p.joint_values, q.joint_values)) for p, q in zip(points, points[1:]))
print("largest joint step: {:.3f} rad".format(largest))

# By default the path may start on any IK branch. `start_from_state` starts it where the
# robot is.
state.robot_configuration = trajectory.points[0].copy()
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"start_from_state": True})
print("from the current configuration: {} points".format(len(trajectory.points)))

"""03 - A Cartesian path: a straight line of frames.

`plan_cartesian_motion` finds joint configurations for every frame such that consecutive
configurations are close (no sudden flips). The native planner solves IK for every frame
in one go, keeps all the solutions, and picks the smoothest collision-free sequence through
them, rather than following one IK branch and hoping it stays valid.
"""

import time

from compas.geometry import Frame

from compas_fab.backends import UR5Kinematics
from compas_fab_native import NativePlanner
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

robot_cell, state = RobotCellLibrary.ur5()
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# 1. 100 frames on a 60 cm line in front of the robot, flange pointing down.
frames = [Frame([0.45, -0.3 + 0.6 * i / 99, 0.4], [1, 0, 0], [0, -1, 0]) for i in range(100)]
waypoints = FrameWaypoints(frames, TargetMode.ROBOT)

# 2. Plan. By default the result is also validated on the exact meshes.
t0 = time.perf_counter()
trajectory = planner.plan_cartesian_motion(waypoints, state)
print("{} points in {:.1f} ms".format(len(trajectory.points), (time.perf_counter() - t0) * 1e3))

# 3. What the planner did, from the statistics it attaches to the trajectory.
stats = trajectory.attributes["native_stats"]
print("IK solutions considered: {:.0f}, of which collision-free: {:.0f}".format(stats["ik_nodes"], stats["free_nodes"]))
print("configurations checked on the exact meshes: {:.0f}".format(stats["exact_configs_checked"]))

# 4. The path is smooth: the largest joint change between two consecutive points.
points = trajectory.points
largest = max(max(abs(a - b) for a, b in zip(p.joint_values, q.joint_values)) for p, q in zip(points, points[1:]))
print("largest joint step: {:.3f} rad".format(largest))

# 5. Starting from where the robot is. By default the planner may start on any IK branch;
#    `start_from_state` forces the path to begin at the state's configuration.
state.robot_configuration = trajectory.points[0].copy()
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"start_from_state": True})
print("from the current configuration: {} points".format(len(trajectory.points)))

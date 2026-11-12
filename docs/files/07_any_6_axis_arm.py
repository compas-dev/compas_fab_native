"""07 - Any 6-axis arm, no solver: 3D printing with an ABB IRB 4600.

Most industrial arms (ABB, KUKA, Staubli, Fanuc, Yaskawa) have a spherical wrist: the last
three axes meet in one point. The planner recognises this from the robot model and solves
IK in closed form. No solver needed. Other robots get numerical IK (example 08).

Your own robot loads like the library ones:

    robot_cell = RobotCell.from_urdf_and_srdf("robot.urdf", "robot.srdf", "path/to/meshes")
"""

import math

from compas.geometry import Frame
from compas_fab.robots import FrameTarget
from compas_fab.robots import PointAxisWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# An ABB IRB 4600 with a printing nozzle, on a floor. No solver given.
robot_cell, state = RobotCellLibrary.abb_irb4600_40_255_printing_tool()
planner = NativePlanner()
planner.set_robot_cell(robot_cell, state)
print("closed-form IK:", planner.has_closed_form_ik)

# IK for the nozzle tip (TargetMode.TOOL), pointing down. Up to 8 solutions in theory;
# this arm's joint limits leave 2 (wrist flipped or not).
target = FrameTarget(Frame([1.6, 0.0, 0.4], [1, 0, 0], [0, -1, 0]), TargetMode.TOOL)
solutions = list(planner.iter_inverse_kinematics(target, state))
print("{} collision-free IK solutions".format(len(solutions)))

# A vase: 10 layers of 2 cm, printed as one spiral. A nozzle only needs a point and a
# direction, so the waypoints are points with an axis, not frames. The planner chooses
# how the nozzle turns about its axis.
center, radius, layer_height = [1.6, 0.0, 0.1], 0.15, 0.02
points = []
for i in range(72 * 10 + 1):
    t = 2 * math.pi * i / 72
    points.append([center[0] + radius * math.cos(t), center[1] + radius * math.sin(t), center[2] + layer_height * i / 72])
waypoints = PointAxisWaypoints([(point, [0, 0, -1]) for point in points], TargetMode.TOOL)

trajectory = planner.plan_cartesian_motion(waypoints, state)
print("vase: {} waypoints, {} trajectory points".format(len(points), len(trajectory.points)))

# The same path with numerical IK, as a robot without a closed form would plan it.
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"ik": "numeric"})
print("vase, numerical IK: {} trajectory points".format(len(trajectory.points)))

"""04 - A tool that may turn about its own axis.

For milling, gluing, drilling or printing, how the tool is turned about its axis does not
matter. Fixing it anyway can make a path impossible. `free_axis_samples` lets the planner
choose that rotation at every waypoint.
"""

import math

from compas.geometry import Frame
from compas_fab.backends import UR5Kinematics
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# A UR5 with a cone-shaped tool. The tool collides like a link.
robot_cell, state = RobotCellLibrary.ur5_cone_tool()
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# A spiral of 2.5 turns, going down. Each frame's x-axis follows the path and z points
# down, so the tool would spin 2.5 times: more than the last joint can turn.
frames = []
for i in range(181):
    t = 2.5 * 2 * math.pi * i / 180
    point = [0.4 + 0.12 * math.cos(t), 0.12 * math.sin(t), 0.3 - 0.1 * i / 180]
    tangent = [-math.sin(t), math.cos(t), 0]
    frames.append(Frame(point, tangent, [math.cos(t), math.sin(t), 0]))
waypoints = FrameWaypoints(frames, TargetMode.TOOL)  # frames for the tool tip

# Rotation fixed as in the frames: no path.
try:
    planner.plan_cartesian_motion(waypoints, state)
    print("fixed rotation: found a path")
except CartesianMotionError:
    print("fixed rotation: no path")

# 12 rotations to choose from at every waypoint: the tool no longer winds up.
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"free_axis_samples": 12})
print("free rotation: {} points".format(len(trajectory.points)))


# The tip still follows the path: compare it with the nearest waypoint segment.
# (There are more points than waypoints: the planner adds one every 1 cm, `max_step`.)
def distance_to_path(p):
    best = float("inf")
    for a, b in zip(frames, frames[1:]):
        ab = [b.point[k] - a.point[k] for k in range(3)]
        ap = [p[k] - a.point[k] for k in range(3)]
        t = max(0.0, min(1.0, sum(x * y for x, y in zip(ab, ap)) / sum(x * x for x in ab)))
        best = min(best, math.dist(p, [a.point[k] + t * ab[k] for k in range(3)]))
    return best


check = state.copy()
worst = 0.0
for point in trajectory.points:
    check.robot_configuration = robot_cell.zero_configuration().merged(point)
    worst = max(worst, distance_to_path(planner.forward_kinematics(check, TargetMode.TOOL).point))
print("largest tip distance from the path: {:.1e} m".format(worst))

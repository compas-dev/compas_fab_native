"""04 - A tool, and letting it turn about its own axis.

For many processes (milling, gluing, drilling, 3D printing) the rotation of the tool about
its own axis does not matter. Fixing it anyway throws away solutions: the robot may be
unable to follow a path with the tool turned one way, but fine with it turned another.
`free_axis_samples` lets the planner choose that rotation at every waypoint.
"""

import math
import time

from compas.geometry import Frame
from compas_fab.backends import UR5Kinematics
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# 1. A UR5 with a cone-shaped tool. The tool is part of the collision model.
robot_cell, state = RobotCellLibrary.ur5_cone_tool()
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# 2. A spiral traced by the tool tip, 2.5 turns going down, like a print or a milling pass.
#    Each frame's x-axis points along the path (TargetMode.TOOL: frames are for the tool's TCF),
#    so taken literally the tool must spin 2.5 times about its axis. The last wrist joint
#    cannot turn that far.
frames = []
for i in range(181):
    t = 2.5 * 2 * math.pi * i / 180
    point = [0.4 + 0.12 * math.cos(t), 0.12 * math.sin(t), 0.3 - 0.1 * i / 180]
    tangent = [-math.sin(t), math.cos(t), 0]
    # x along the path, z (the tool axis) pointing down.
    frames.append(Frame(point, tangent, [math.cos(t), math.sin(t), 0]))
waypoints = FrameWaypoints(frames, TargetMode.TOOL)

# 3. With the tool rotation fixed as given in the frames.
try:
    planner.plan_cartesian_motion(waypoints, state)
    print("fixed rotation: found a path")
except CartesianMotionError:
    print("fixed rotation: no path")

# 4. With 12 rotations about the tool axis to choose from at every waypoint, the planner
#    can keep the tool from winding up.
t0 = time.perf_counter()
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"free_axis_samples": 12})
print("free rotation: {} points in {:.1f} ms".format(len(trajectory.points), (time.perf_counter() - t0) * 1e3))


# 5. The tip follows the path exactly; only the rotation about the tool axis changed.
#    (The planner adds frames between waypoints more than 1 cm apart, the `max_step`
#    option, so there are more points than waypoints: each lies on the straight segment
#    between two waypoints.)
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
    tcf = planner.forward_kinematics(check, TargetMode.TOOL)
    worst = max(worst, distance_to_path(tcf.point))
print("largest tip distance from the path: {:.1e} m".format(worst))

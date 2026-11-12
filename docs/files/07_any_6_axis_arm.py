"""07 - Any 6-axis arm, no solver: 3D printing with an ABB IRB 4600.

Examples 01 to 06 hand the planner a UR5 solver, because the UR's offset wrist is the one
geometry the core cannot recognise on its own. Every other robot needs nothing:

- most industrial 6-axis arms (ABB, KUKA, Staubli, Fanuc, Yaskawa...) have a spherical
  wrist: the last three axes meet in one point. The core recognises that from the robot
  model and solves IK in closed form, with every solution, at no setup cost.
- every other robot (7-axis arms, robots on tracks or gantries, unusual geometry) gets
  numerical IK, which also works for spherical-wrist arms. Example 08 compares the two.

Your own robot loads the same way as the library ones, from its URDF and SRDF:

    robot_cell = RobotCell.from_urdf_and_srdf("my_robot.urdf", "my_robot.srdf", "path/to/meshes")
"""

import math
import time

from compas.geometry import Frame
from compas_fab.robots import FrameTarget
from compas_fab.robots import PointAxisWaypoints
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# 1. An ABB IRB 4600 with a printing nozzle, standing on a floor. No kinematics solver.
robot_cell, state = RobotCellLibrary.abb_irb4600_40_255_printing_tool()
planner = NativePlanner()
planner.set_robot_cell(robot_cell, state)
print("closed-form IK:", planner.client.core.has_analytic_ik)

# 2. Every way the robot can put the nozzle at one point, pointing down (TargetMode.TOOL:
#    the frame is for the nozzle tip). A 6-axis arm has up to 8; the IRB 4600's joint limits
#    leave 2 here (wrist flipped or not). Collision checking would drop postures that hit
#    the floor or the robot itself.
nozzle_down = Frame([1.6, 0.0, 0.4], [1, 0, 0], [0, -1, 0])
everything = list(planner.iter_inverse_kinematics(FrameTarget(nozzle_down, TargetMode.TOOL), state, options={"check_collision": False}))
collision_free = list(planner.iter_inverse_kinematics(FrameTarget(nozzle_down, TargetMode.TOOL), state))
print("IK: {} solutions, {} collision-free".format(len(everything), len(collision_free)))

# 3. A vase: 10 layers of 2 cm, printed as one continuous spiral (no seam). A nozzle only
#    has a point and a direction; how it is turned about its own axis does not matter. So
#    the waypoints are points with an axis (straight down), not frames, and the planner
#    chooses that rotation. Fixing it would wind the wrist through 10 turns, which no
#    joint can do (see example 04).
center, radius, layers, layer_height = [1.6, 0.0, 0.1], 0.15, 10, 0.02
points = []
for i in range(72 * layers + 1):
    t = 2 * math.pi * i / 72
    points.append([center[0] + radius * math.cos(t), center[1] + radius * math.sin(t), center[2] + layer_height * i / 72])
waypoints = PointAxisWaypoints([(point, [0, 0, -1]) for point in points], TargetMode.TOOL)

t0 = time.perf_counter()
trajectory = planner.plan_cartesian_motion(waypoints, state)
print("vase: {} waypoints, {} trajectory points in {:.0f} ms".format(len(points), len(trajectory.points), (time.perf_counter() - t0) * 1e3))

# 4. The same path with numerical IK, as a robot without a closed form would plan it. The
#    closed form samples rotations about the axis; numerical IK leaves it free.
t0 = time.perf_counter()
trajectory = planner.plan_cartesian_motion(waypoints, state, options={"ik": "numeric"})
print("vase, numerical IK: {} trajectory points in {:.0f} ms".format(len(trajectory.points), (time.perf_counter() - t0) * 1e3))

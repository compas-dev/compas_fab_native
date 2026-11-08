"""01 - Forward and inverse kinematics with the native planner.

The native planner is a compas_fab backend like any other: you give it a RobotCell, then
ask it for kinematics, collision checks and paths. The difference is underneath: the work
happens in a compiled (Rust) core instead of Python, PyBullet or ROS.

Install the core once with ``pip install ./native`` from a compas_fab checkout.
"""

import time

from compas.geometry import Frame

from compas_fab.backends import UR5Kinematics
from compas_fab.backends.native import NativePlanner
from compas_fab.robots import FrameTarget
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

# 1. A robot cell: a UR5 on its own. `state` holds everything that moves: the robot's joint
#    values, its base frame, tools and obstacles.
robot_cell, state = RobotCellLibrary.ur5()

# 2. The planner. A UR has an offset wrist, so we tell it which closed-form solver to use.
#    (Spherical-wrist arms like ABB or Staubli need nothing: the solver is derived from the
#    robot model.) `set_robot_cell` sends the robot to the core, fitting collision spheres
#    to every link mesh. That takes a moment, once.
planner = NativePlanner(UR5Kinematics())
t0 = time.perf_counter()
planner.set_robot_cell(robot_cell, state)
print("set_robot_cell: {:.0f} ms (once per cell)".format((time.perf_counter() - t0) * 1e3))

# 3. Forward kinematics: where is the flange (the robot coordinate frame, RCF) for these joints?
state.robot_configuration.joint_values = [0.0, -1.2, 1.4, -1.8, -1.57, 0.0]
flange = planner.forward_kinematics(state, TargetMode.ROBOT)
print("flange at", flange.point)

# 4. Inverse kinematics: every way the robot can reach a frame. A 6-axis arm has up to 8
#    (shoulder left/right x elbow up/down x wrist flip). Collision checking is on by default,
#    so solutions where the robot hits itself or an obstacle are dropped.
target = FrameTarget(Frame([0.4, 0.2, 0.5], [1, 0, 0], [0, -1, 0]), TargetMode.ROBOT)
solutions = list(planner.iter_inverse_kinematics(target, state))
print("{} collision-free IK solutions".format(len(solutions)))
for configuration in solutions:
    print("  ", ["{:+.2f}".format(v) for v in configuration.joint_values])

# 5. How fast? Time many IK calls, including the collision check of every solution.
n = 200
t0 = time.perf_counter()
for _ in range(n):
    list(planner.iter_inverse_kinematics(target, state))
print("IK + collision check: {:.0f} us per target (mostly Python overhead)".format((time.perf_counter() - t0) / n * 1e6))

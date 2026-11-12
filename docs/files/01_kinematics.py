"""01 - Forward and inverse kinematics.

The native planner is a compas_fab backend like any other: give it a robot cell, then ask
for kinematics, collision checks and paths. The work happens in a compiled (Rust) core.

Install it with ``pip install compas_fab_native``.
"""

from compas.geometry import Frame
from compas_fab.backends import UR5Kinematics
from compas_fab.robots import FrameTarget
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# A UR5. `state` holds what moves: joint values, base frame, tools, obstacles.
robot_cell, state = RobotCellLibrary.ur5()

# URs need their solver for closed-form IK (example 08 explains why).
# `set_robot_cell` sends the robot to the core.
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)
print("closed-form IK:", planner.has_closed_form_ik)

# Forward kinematics: where is the flange for these joint values?
state.robot_configuration.joint_values = [0.0, -1.2, 1.4, -1.8, -1.57, 0.0]
flange = planner.forward_kinematics(state, TargetMode.ROBOT)
print("flange at", flange.point)

# Inverse kinematics: every way to reach a frame, up to 8 for a 6-axis arm.
# Solutions that collide are dropped.
target = FrameTarget(Frame([0.4, 0.2, 0.5], [1, 0, 0], [0, -1, 0]), TargetMode.ROBOT)
solutions = list(planner.iter_inverse_kinematics(target, state))
print("{} collision-free IK solutions".format(len(solutions)))
for configuration in solutions:
    print("  ", ["{:+.2f}".format(v) for v in configuration.joint_values])

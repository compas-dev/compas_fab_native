"""08 - Which IK each robot gets.

The planner picks the IK when the robot cell is set:

- a compas_fab solver given, like ``UR5Kinematics()``: its closed form.
- a 6-axis arm with a spherical wrist: a closed form built from the robot model. The
  planner checks that joints 2 and 3 are parallel and perpendicular to joint 1, and that
  axes 4 to 6 meet in one point. URs fail this check, so they need their solver.
- anything else: numerical IK.

Closed form returns every solution within the joint limits. Numerical IK returns the
solutions it found from random starting points (up to `max_results`, default 8).
"""

from compas_fab.backends import UR5Kinematics
from compas_fab.robots import FrameTarget
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

robots = [
    ("ABB IRB 4600", RobotCellLibrary.abb_irb4600_40_255, None),
    ("Staubli TX2-60L", RobotCellLibrary.staubli_tx2_60l, None),
    ("UR5 with UR5Kinematics()", RobotCellLibrary.ur5, UR5Kinematics()),
    ("UR5 without a solver", RobotCellLibrary.ur5, None),
    ("Panda (7 axes)", RobotCellLibrary.panda, None),
]

for name, load, solver in robots:
    robot_cell, state = load()
    planner = NativePlanner(solver)
    planner.set_robot_cell(robot_cell, state)
    kind = "closed form" if planner.has_closed_form_ik else "numerical"

    # A pose the robot can reach: the forward kinematics of a random configuration.
    check = state.copy()
    check.robot_configuration = robot_cell.random_configuration(robot_cell.main_group_name)
    target = FrameTarget(planner.forward_kinematics(check, TargetMode.ROBOT), TargetMode.ROBOT)
    solutions = list(planner.iter_inverse_kinematics(target, state, options={"check_collision": False}))
    print("{:26} {:12} {} solutions".format(name, kind, len(solutions)))

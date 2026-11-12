"""08 - Which IK each robot gets.

The planner chooses the IK when the robot cell is set:

- a compas_fab solver given (``NativePlanner(UR5Kinematics())``): its closed form.
- no solver, 6 revolute joints, joints 2 and 3 parallel and perpendicular to joint 1, and
  axes 4 to 6 meeting in one point (a spherical wrist): a closed form built from the
  robot model.
- anything else: numerical IK.

This runs the same task on five robots: reach a pose the robot can reach (the forward
kinematics of a random configuration), and time it.
"""

import time

from compas_fab.backends import UR5Kinematics
from compas_fab.robots import FrameTarget
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

robots = [
    ("ABB IRB 4600 (spherical wrist)", RobotCellLibrary.abb_irb4600_40_255, None),
    ("Staubli TX2-60L (spherical wrist)", RobotCellLibrary.staubli_tx2_60l, None),
    ("UR5 with UR5Kinematics()", RobotCellLibrary.ur5, UR5Kinematics()),
    ("UR5 without a solver", RobotCellLibrary.ur5, None),
    ("Panda (7 axes)", RobotCellLibrary.panda, None),
]

for name, load, solver in robots:
    robot_cell, state = load()
    planner = NativePlanner(solver)
    planner.set_robot_cell(robot_cell, state)
    kind = "closed form" if planner.client.core.has_analytic_ik else "numerical"

    # Poses the robot can reach: forward kinematics of random configurations.
    targets = []
    check = state.copy()
    for _ in range(20):
        check.robot_configuration = robot_cell.random_configuration(robot_cell.main_group_name)
        targets.append(FrameTarget(planner.forward_kinematics(check, TargetMode.ROBOT), TargetMode.ROBOT))

    t0 = time.perf_counter()
    counts = [len(list(planner.iter_inverse_kinematics(target, state, options={"check_collision": False}))) for target in targets]
    per_target = (time.perf_counter() - t0) / len(targets)
    print(
        "{:34} {:12} {:4.1f} solutions on average, {:5.2f} ms per target".format(name, kind, sum(counts) / len(counts), per_target * 1e3),
    )

# Closed form returns every solution within the joint limits (up to 8 for a 6-axis arm).
# Numerical IK returns the distinct solutions it found from random seeds (`max_results`,
# default 8), so a count below that does not prove there are no more.

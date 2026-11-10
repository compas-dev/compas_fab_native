"""06 - Free motion: from here to there, around obstacles.

Cartesian paths (examples 03 and 04) prescribe where the tool goes. Free motion only
prescribes where the robot ends up: the planner chooses the way, avoiding obstacles. It
first tries the straight joint-space motion, then grows trees from the start and from the
goal until they meet (RRT-Connect), then shortens and smooths the path.
"""

import time

from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Vector

from compas_fab.backends import UR5Kinematics
from compas_fab_native import NativePlanner
from compas_fab.robots import ConfigurationTarget
from compas_fab.robots import FrameTarget
from compas_fab.robots import PointAxisTarget
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

# 1. A UR5 with a wall in front of it.
robot_cell, _ = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["wall"] = RigidBody.from_mesh(Box(0.05, 0.8, 0.8).to_mesh(triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["wall"] = RigidBodyState(Frame([0.45, 0.0, 0.4], [1, 0, 0], [0, 1, 0]))
planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# 2. From the left of the wall to the right. The straight joint-space motion would sweep the
#    arm through the wall, so the planner has to find a way around.
state.robot_configuration.joint_values = [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0]
goal = robot_cell.zero_configuration()
goal.joint_values = [1.2, -1.2, 1.4, -1.8, -1.57, 0.0]
t0 = time.perf_counter()
trajectory = planner.plan_motion(ConfigurationTarget(goal), state)
stats = trajectory.attributes["native_stats"]
print("around the wall in {:.1f} ms".format((time.perf_counter() - t0) * 1e3))
print(
    "  planning {:.0f} us ({:.0f} iterations), smoothing {:.0f} us, exact-mesh check {:.0f} us".format(
        stats["planning_us"], stats["iterations"], stats["simplification_us"], stats["exact_us"]
    )
)
print("  joint-space length: {:.2f} before smoothing, {:.2f} after".format(stats["initial_cost"], stats["cost"]))

# 3. To a frame instead of a configuration. A 6-axis arm reaches a frame in up to 8 ways;
#    the planner grows its goal tree from all of them at once and takes whichever it
#    reaches first.
target = FrameTarget(Frame([0.3, 0.35, 0.3], [1, 0, 0], [0, -1, 0]), TargetMode.ROBOT)
trajectory = planner.plan_motion(target, state)
print("to a frame: {:.2f} s".format(trajectory.attributes["native_stats"]["duration"]))

# 4. To a point and an axis, e.g. a drill that may spin about its own axis: every rotation
#    about the axis is acceptable, so even more goals.
target = PointAxisTarget(Point(0.3, -0.35, 0.3), Vector(0, 0, -1), TargetMode.ROBOT)
trajectory = planner.plan_motion(target, state, options={"axis_samples": 16})
print("to a point and axis: {:.2f} s".format(trajectory.attributes["native_stats"]["duration"]))

# 5. Timing. The trajectory is timed under the joints' velocity limits (from the URDF) and
#    acceleration limits, starting and ending at rest. Corners are blended (rounded off by
#    at most 0.05 rad, if the rounded corner is collision-free) so the robot does not stop
#    at every waypoint. Each point carries its time, velocities and accelerations.
print("timed: {} points over {:.2f} s".format(len(trajectory.points), stats["duration"]))
slow = planner.plan_motion(ConfigurationTarget(goal), state, options={"velocity_scaling": 0.25, "max_acceleration": 1.0})
print("at a quarter of the speed: {:.2f} s".format(slow.attributes["native_stats"]["duration"]))
middle = trajectory.points[len(trajectory.points) // 2]
print("halfway: t = {:.2f} s, joint speeds {}".format(middle.time_from_start.seconds, ["{:+.2f}".format(v) for v in middle.velocities]))
untimed = planner.plan_motion(ConfigurationTarget(goal), state, options={"time_parameterization": False})
print("untimed, the path's corners only: {} points".format(len(untimed.points)))

# 6. Free motion does not need closed-form IK, so any serial robot works, like this 7-axis
#    Panda (no kinematics solver given).
panda_cell, panda_state = RobotCellLibrary.panda()
panda_cell.rigid_body_models["post"] = RigidBody.from_mesh(Box(0.1, 0.1, 1.2).to_mesh(triangulated=True))
panda_state = panda_cell.default_cell_state()
panda_state.rigid_body_states["post"] = RigidBodyState(Frame([0.45, 0.0, 0.6], [1, 0, 0], [0, 1, 0]))
panda = NativePlanner()
panda.set_robot_cell(panda_cell, panda_state)
start = panda_cell.zero_configuration()
start.joint_values = [-1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]
panda_state.robot_configuration = panda_state.robot_configuration.merged(start)
goal = panda_cell.zero_configuration()
goal.joint_values = [1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]
t0 = time.perf_counter()
trajectory = panda.plan_motion(ConfigurationTarget(goal), panda_state)
print("panda around a post in {:.1f} ms, {:.2f} s long".format((time.perf_counter() - t0) * 1e3, trajectory.attributes["native_stats"]["duration"]))

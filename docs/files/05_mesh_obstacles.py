"""05 - Obstacles with holes: distance fields instead of boxes.

By default each obstacle becomes its bounding box. That is fine for tables and walls, but
wrong for a ring: the box fills the hole. With ``obstacles="esdf"`` the core uses a
distance field of the obstacle meshes instead.
"""

from compas.geometry import Frame
from compas.geometry import Torus
from compas_fab.backends import UR5Kinematics
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

from compas_fab_native import NativePlanner

# A ring, 50 cm across, lying flat in front of the robot.
robot_cell, _ = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["ring"] = RigidBody.from_mesh(Torus(0.25, 0.025).to_mesh(u=48, v=12, triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["ring"] = RigidBodyState(Frame([0.45, 0.0, 0.45], [1, 0, 0], [0, 1, 0]))

# The task: move the flange straight down through the ring.
frames = [Frame([0.45, 0.0, 0.65 - 0.23 * i / 29], [1, 0, 0], [0, -1, 0]) for i in range(30)]
waypoints = FrameWaypoints(frames, TargetMode.ROBOT)
options = {"free_axis_samples": 12}

# With boxes: the box covers the hole, so there is no path.
boxes = NativePlanner(UR5Kinematics(), obstacles="boxes")
boxes.set_robot_cell(robot_cell, state)
try:
    boxes.plan_cartesian_motion(waypoints, state, options=options)
    print("boxes: found a path")
except CartesianMotionError:
    print("boxes: no path")

# With a distance field (2 cm grid): the hole is free.
esdf = NativePlanner(UR5Kinematics(), obstacles="esdf", esdf_options={"voxel": 0.02, "truncation": 0.3})
esdf.set_robot_cell(robot_cell, state)
trajectory = esdf.plan_cartesian_motion(waypoints, state, options=options)
print("distance field: {} points".format(len(trajectory.points)))

# Double-check the path on the exact meshes.
contacts = esdf.client.core.exact_check([p.joint_values for p in trajectory.points])
print("collisions on the exact meshes:", sum(c is not None for c in contacts))

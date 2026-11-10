"""05 - Obstacles with holes: distance fields instead of boxes.

By default every obstacle is reduced to its bounding box: fast, safe, and fine for tables,
walls and pallets. It is wrong for anything with a hole or a concave shape, where the box
fills the very space the robot needs. With ``obstacles="esdf"`` the core builds a distance
field from the obstacle meshes (a grid of distances to the surface, stored only near the
surface) and checks the robot spheres against it.
"""

import time

from compas.geometry import Frame
from compas.geometry import Torus

from compas_fab.backends import UR5Kinematics
from compas_fab.backends.kinematics.exceptions import CartesianMotionError
from compas_fab_native import NativePlanner
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode

# 1. A ring (50 cm across, tube 5 cm thick) lying flat in front of the robot. (Much
#    narrower and the UR5's wrist only just fits: whether it passes then depends on details
#    of the sphere fit.)
robot_cell, _ = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["ring"] = RigidBody.from_mesh(Torus(0.25, 0.025).to_mesh(u=48, v=12, triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["ring"] = RigidBodyState(Frame([0.45, 0.0, 0.45], [1, 0, 0], [0, 1, 0]))

# 2. The task: plunge the flange straight down through the middle of the ring.
frames = [Frame([0.45, 0.0, 0.65 - 0.23 * i / 29], [1, 0, 0], [0, -1, 0]) for i in range(30)]
waypoints = FrameWaypoints(frames, TargetMode.ROBOT)
options = {"free_axis_samples": 12}

# 3. With boxes: the ring's bounding box covers its hole, so this cannot work.
boxes = NativePlanner(UR5Kinematics(), obstacles="boxes")
boxes.set_robot_cell(robot_cell, state)
try:
    boxes.plan_cartesian_motion(waypoints, state, options=options)
    print("boxes: found a path")
except CartesianMotionError:
    print("boxes: no path (the bounding box fills the hole)")

# 4. With a distance field: 2 cm grid, distances stored up to 30 cm from the surface.
esdf = NativePlanner(UR5Kinematics(), obstacles="esdf", esdf_options={"voxel": 0.02, "truncation": 0.3})
t0 = time.perf_counter()
esdf.set_robot_cell(robot_cell, state)
print("distance field built in {:.0f} ms (again only when obstacles change)".format((time.perf_counter() - t0) * 1e3))
t0 = time.perf_counter()
trajectory = esdf.plan_cartesian_motion(waypoints, state, options=options)
print("distance field: {} points in {:.1f} ms".format(len(trajectory.points), (time.perf_counter() - t0) * 1e3))

# 5. And the final word, on the exact meshes: is the path really free?
contacts = esdf.client.core.exact_check([p.joint_values for p in trajectory.points])
print("collisions on the exact meshes:", sum(c is not None for c in contacts))

"""02 - Collision checking: spheres for speed, meshes to be sure.

The core fills each link with spheres fitted to its mesh. Sphere checks are cheap, so
thousands of configurations can be checked at once. Spheres are an approximation, so the
core also keeps the exact meshes.
"""

import random

from compas.geometry import Box
from compas.geometry import Frame
from compas_fab.backends import CollisionCheckError
from compas_fab.backends import UR5Kinematics
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary

from compas_fab_native import NativePlanner

# A UR5 next to a table.
robot_cell, state = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["table"] = RigidBody.from_mesh(Box(0.6, 0.6, 0.3).to_mesh(triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["table"] = RigidBodyState(Frame([0.5, 0.0, 0.15], [1, 0, 0], [0, 1, 0]))

planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# What the core sees: spheres for the robot, boxes for obstacles. Neighbouring links that
# always touch are never checked against each other.
core = planner.client.core
n_spheres, n_self_spheres, n_boxes = core.sphere_counts
print("{} spheres for obstacles, {} for self-collision, {} obstacle boxes".format(n_spheres, n_self_spheres, n_boxes))
print("link pairs never checked:", core.disabled_collisions)

# One configuration: the arm reaching down into the table.
state.robot_configuration.joint_values = [0.0, -0.6, 1.6, -1.0, -1.57, 0.0]
try:
    planner.check_collision(state)
    print("down: free")
except CollisionCheckError as e:
    print("down:", e)

# Many configurations at once. `clearance` gives the smallest distance to anything, per
# configuration. Negative means collision.
random.seed(0)
configurations = [robot_cell.random_configuration(robot_cell.main_group_name).joint_values for _ in range(10000)]
clearances = core.clearance(configurations)
print("{} of 10,000 random configurations collide".format(sum(c <= 0 for c in clearances)))

# The exact check on the meshes is slower but exact. Compare both on 300 configurations.
sample = configurations[:300]
exact = core.exact_check(sample)
spheres_say = [c <= 0 for c in clearances[:300]]
meshes_say = [contact is not None for contact in exact]
missed = [contact for s, m, contact in zip(spheres_say, meshes_say, exact) if m and not s]
false_alarms = sum(s and not m for s, m in zip(spheres_say, meshes_say))
print("spheres say collision, meshes say free: {}".format(false_alarms))
print("spheres say free, meshes collide: {}, for example:".format(len(missed)))
for contact in missed[:3]:
    print("  {} touches {}".format(*contact))

# The misses are near-touches between neighbouring links, where spheres leave gaps. That is
# why planned paths are checked on the meshes too (example 03), and why
# `check_collision(state, {"exact": True})` exists.

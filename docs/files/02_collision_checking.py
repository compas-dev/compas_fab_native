"""02 - Collision checking: spheres for speed, meshes for the final word.

The core checks collisions on a sphere model of the robot: each link is filled with spheres
fitted to its mesh. Sphere checks are cheap enough to test thousands of configurations in
a few milliseconds. Spheres are an approximation, so the core also keeps the exact meshes
and can check any configuration on them.
"""

import random
import time

from compas.geometry import Box
from compas.geometry import Frame

from compas_fab.backends import CollisionCheckError
from compas_fab.backends import UR5Kinematics
from compas_fab_native import NativePlanner
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary

# 1. A UR5 next to a table-sized box.
robot_cell, state = RobotCellLibrary.ur5()
robot_cell.rigid_body_models["table"] = RigidBody.from_mesh(Box(0.6, 0.6, 0.3).to_mesh(triangulated=True))
state = robot_cell.default_cell_state()
state.rigid_body_states["table"] = RigidBodyState(Frame([0.5, 0.0, 0.15], [1, 0, 0], [0, 1, 0]))

planner = NativePlanner(UR5Kinematics())
planner.set_robot_cell(robot_cell, state)

# 2. What the core sees. Obstacles are boxes (their bounding boxes, by default). The robot is
#    spheres, and some link pairs are never checked: neighbours that touch in every
#    configuration (found by sampling, like MoveIt's setup assistant).
core = planner.client.core
n_spheres, n_self_spheres, n_boxes = core.sphere_counts
print("{} spheres for obstacles, {} for self-collision, {} obstacle boxes".format(n_spheres, n_self_spheres, n_boxes))
print("link pairs never checked:", core.disabled_collisions)

# 3. One configuration: arm reaching down into the table.
state.robot_configuration.joint_values = [0.0, -0.6, 1.6, -1.0, -1.57, 0.0]
try:
    planner.check_collision(state)
    print("down: free")
except CollisionCheckError as e:
    print("down:", e)

# 4. Many configurations at once. `clearance` returns, per configuration, the smallest
#    distance between the robot spheres and anything else (negative = penetration).
random.seed(0)
configurations = [robot_cell.random_configuration(robot_cell.main_group_name).joint_values for _ in range(10000)]
t0 = time.perf_counter()
clearances = core.clearance(configurations)
elapsed = time.perf_counter() - t0
colliding = sum(c <= 0 for c in clearances)
print("10,000 random configurations in {:.1f} ms: {} in collision".format(elapsed * 1e3, colliding))

# 5. The exact check, on the link meshes. Slower (about 0.2 ms per configuration) but exact.
#    Here we compare both on the first 300 configurations.
sample = configurations[:300]
exact = core.exact_check(sample)
spheres_say = [c <= 0 for c in clearances[:300]]
meshes_say = [contact is not None for contact in exact]
missed = [(q, c) for q, s, m, c in zip(sample, spheres_say, meshes_say, exact) if m and not s]
false_alarms = [q for q, s, m in zip(sample, spheres_say, meshes_say) if s and not m]
print("spheres and meshes disagree on {}/300 configurations:".format(len(missed) + len(false_alarms)))
print("  {} false alarms (spheres say collision, meshes say free)".format(len(false_alarms)))
print("  {} missed contacts (spheres say free, meshes collide), for example:".format(len(missed)))
for q, contact in missed[:3]:
    print("    {} touches {}".format(*contact))

# The misses are near-contacts between neighbouring links, mostly the forearm and the wrist,
# where the inscribed spheres leave the mesh surface uncovered. This is why the planner
# validates its final paths on the exact meshes (see 03_cartesian_line.py), and why
# `check_collision(state, {"exact": True})` exists.

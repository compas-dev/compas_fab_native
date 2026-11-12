# Native (experimental)

A planning backend whose work happens in a compiled core (Rust) inside the Python process:
no Docker, no PyBullet, no ROS. It is experimental: the API is the usual compas_fab
backend API, but what it supports is limited (see below) and may change.

## When to use

- Cartesian paths (lines, spirals, toolpaths), fast: frame or point-axis waypoints, for
  6-axis arms, redundant arms and robots on external axes.
- Free motion between configurations, frames or point-axis targets, around obstacles.
- Inverse kinematics for any serial robot: closed form where there is one, numerical
  otherwise.
- Checking many configurations for collisions at once.
- Obstacles with holes or concave shapes, through distance fields.
- Trying it out and telling us what breaks.

## Trade-offs

| What you get | What you give up |
|---|---|
| Cartesian paths in milliseconds, through the compas_fab API | Robots without a closed form search a beam of postures along the path (8 by default, `tracks`): wide, but not exhaustive |
| Free motion in milliseconds, for any serial robot, timed under velocity and acceleration limits | No jerk limits yet |
| Closed-form IK for UR (offset wrist) and spherical-wrist arms; numerical IK (collision-aware) for every other robot | Numerical IK finds solutions from seeds: "no solution" means none was found, not that none exists |
| Sphere-model collision checks for thousands of configurations per millisecond, validated on exact meshes, including tools and workpieces the robot carries | Large flat links (gantry carriages, tall bases) are approximated poorly by spheres (`planner.poorly_fitted_links`); the exact check catches what the spheres miss, at the cost of slower plans (`exact_links` checks them during planning too, slower still) |
| Mesh obstacles with holes (distance fields) | Experimental: the API may change between minor versions |

## Setup

```bash
pip install compas_fab_native
```

Wheels are available for Linux (x86_64), macOS (Intel and Apple Silicon) and Windows (x64),
for Python 3.9 and later. Elsewhere, pip builds the extension from source, which needs a
[Rust toolchain](https://rustup.rs). From a checkout of the repository, `pip install -e .`
builds it in place.

## First example

Forward and inverse kinematics on a UR5:

```python
--8<-- "docs/files/01_kinematics.py"
```

## More examples

Each example builds on the previous one and explains what the planner does as it goes.

- [`02_collision_checking.py`](files/02_collision_checking.py){: download="02_collision_checking.py" }: the sphere model, checking 10,000 configurations at once, and how spheres compare with exact meshes
- [`03_cartesian_line.py`](files/03_cartesian_line.py){: download="03_cartesian_line.py" }: a 100-waypoint line, what the planner does, and starting from the current configuration
- [`04_tool_and_free_axis.py`](files/04_tool_and_free_axis.py){: download="04_tool_and_free_axis.py" }: a spiral toolpath that only works when the tool may turn about its own axis
- [`05_mesh_obstacles.py`](files/05_mesh_obstacles.py){: download="05_mesh_obstacles.py" }: plunging through a ring, impossible with bounding boxes, possible with a distance field
- [`06_free_motion.py`](files/06_free_motion.py){: download="06_free_motion.py" }: free motion around obstacles, to configurations, frames and point-axis targets, timed under the joint limits, on a UR5 and a 7-axis Panda
- [`07_any_6_axis_arm.py`](files/07_any_6_axis_arm.py){: download="07_any_6_axis_arm.py" }: an ABB IRB 4600 printing a spiral vase with no kinematics solver given, in closed form and with numerical IK
- [`08_which_ik.py`](files/08_which_ik.py){: download="08_which_ik.py" }: which IK five robots get (ABB, Stäubli, UR5 with and without a solver, Panda), and how many solutions each finds

To try it interactively, the
[web viewer](https://github.com/compas-dev/compas_fab_native/tree/main/examples/web_viewer)
runs in your browser: pick a library robot or load your own URDF, drag a target, and plan
around boxes you place.

## How it works, in short

The full explanation, with diagrams, is in [How it works](how-it-works.md).


1. `set_robot_cell` fits spheres inside every link mesh and sends the robot, its meshes and
   the obstacles to the core. Sphere fits are cached (in memory and in
   `~/.cache/compas_fab/native_spheres`; `COMPAS_FAB_NATIVE_CACHE=0` turns that off), so
   setting up the same cell again takes milliseconds. Tools and workpieces the planning
   group carries get spheres on the link that carries them; everything that stands still
   while it plans (other robots of the cell, their tools, parked tools) becomes an
   obstacle.
2. IK is closed-form where possible: ported for UR arms, derived from the model for
   spherical-wrist arms. Other robots (a 7-axis arm, a robot on a gantry) get numerical IK:
   a least-squares solve that also keeps the robot clear of collisions. `options={"ik":
   "numeric"}` uses it for any robot.
3. A Cartesian request first fills in frames between the waypoints (every `max_step`, 1 cm
   by default, and `max_step_angle`, 0.1 rad), so the tool moves in straight lines. With
   closed-form IK it solves every frame (and every sampled rotation about the tool axis),
   checks all solutions for collisions in one batch, and searches for the smoothest
   sequence through them. With numerical IK it carries a beam of postures along the path,
   branching into new ones when they die out, and picks the smoothest sequence. Point-axis waypoints leave the rotation about the axis
   free.
4. The chosen path is validated on the exact meshes; if a configuration collides there,
   the search continues without it.
5. Free motion tries the straight joint-space motion first, then RRT-Connect from the start
   and from every goal at once, then shortcutting and smoothing, and checks the result on
   the exact meshes (if it touches them, it plans again with a safety margin). Its
   collision checks run 8 configurations at a time in SIMD and stop at the first contact.
6. The path is then timed: corners are rounded off within a tolerance (if the rounded
   corner is collision-free), and the speed along the path is the fastest the joint
   velocity and acceleration limits allow, starting and ending at rest.

Design notes, measurements and open issues are in
[`SPIKE.md`](https://github.com/compas-dev/compas_fab_native/blob/main/SPIKE.md).

## API reference

See the [API reference](api.md).

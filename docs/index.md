# Native (experimental)

A planning backend whose work happens in a compiled core (Rust) inside the Python process:
no Docker, no PyBullet, no ROS. It is an early spike: the API is the usual compas_fab
backend API, but what it supports is limited (see below) and may change.

## When to use

- Cartesian paths (lines, spirals, toolpaths) for 6-axis arms, fast.
- Free motion between configurations, frames or point-axis targets, around obstacles.
- Checking many configurations for collisions at once.
- Obstacles with holes or concave shapes, through distance fields.
- Trying it out and telling us what breaks.

## Trade-offs

| What you get | What you give up |
|---|---|
| Cartesian paths in milliseconds, through the compas_fab API | Frame targets and Cartesian paths need a 6-axis arm: UR (offset wrist) or spherical wrist |
| Free motion in milliseconds, for any serial robot, timed under velocity and acceleration limits | No jerk limits yet |
| Closed-form IK derived from the robot model for spherical-wrist arms | |
| Sphere-model collision checks for thousands of configurations per millisecond, validated on exact meshes | Attached workpieces are ignored (tools are supported) |
| Mesh obstacles with holes (distance fields) | Experimental: not on PyPI, built from source |

## Setup

From a compas_fab checkout, with a [Rust toolchain](https://rustup.rs) installed:

```bash
pip install ./native
```

This builds the `compas_fab_native` extension with maturin. For development,
`maturin develop --release`, run inside `native/`, rebuilds it in place.

## First example

Forward and inverse kinematics on a UR5:

```python
--8<-- "docs/backends/native/files/01_kinematics.py"
```

## More examples

Each example builds on the previous one and explains what the planner does as it goes.

- [`02_collision_checking.py`](native/files/02_collision_checking.py){: download="02_collision_checking.py" }: the sphere model, checking 10,000 configurations at once, and how spheres compare with exact meshes
- [`03_cartesian_line.py`](native/files/03_cartesian_line.py){: download="03_cartesian_line.py" }: a 100-waypoint line, what the planner does, and starting from the current configuration
- [`04_tool_and_free_axis.py`](native/files/04_tool_and_free_axis.py){: download="04_tool_and_free_axis.py" }: a spiral toolpath that only works when the tool may turn about its own axis
- [`05_mesh_obstacles.py`](native/files/05_mesh_obstacles.py){: download="05_mesh_obstacles.py" }: plunging through a ring, impossible with bounding boxes, possible with a distance field
- [`06_free_motion.py`](native/files/06_free_motion.py){: download="06_free_motion.py" }: free motion around obstacles, to configurations, frames and point-axis targets, timed under the joint limits, on a UR5 and a 7-axis Panda

## How it works, in short

1. `set_robot_cell` fits spheres inside every link mesh and sends the robot, its meshes and
   the obstacles to the core.
2. IK is closed-form: ported for UR arms, derived from the model for spherical-wrist arms.
3. A Cartesian request solves IK for every waypoint (and every sampled rotation about the
   tool axis), checks all solutions for collisions in one batch, and searches for the
   smoothest sequence through them.
4. The chosen path is validated on the exact meshes; if a configuration collides there,
   the search continues without it.
5. Free motion tries the straight joint-space motion first, then RRT-Connect from the start
   and from every goal at once, then shortcutting and smoothing, and checks the result on
   the exact meshes. Its collision checks run 8 configurations at a time in SIMD and stop
   at the first contact.
6. The path is then timed: corners are rounded off within a tolerance (if the rounded
   corner is collision-free), and the speed along the path is the fastest the joint
   velocity and acceleration limits allow, starting and ending at rest.

Design notes, measurements and open issues are in
[`native/SPIKE.md`](https://github.com/compas-dev/compas_fab/blob/spike/native-core/native/SPIKE.md).

## API reference

- `compas_fab.backends.native.NativePlanner`

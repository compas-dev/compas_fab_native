# Native (experimental)

A planning backend whose work happens in a compiled core (Rust) inside the Python process:
no Docker, no PyBullet, no ROS. It is an early spike: the API is the usual compas_fab
backend API, but what it supports is limited (see below) and may change.

## When to use

- Cartesian paths (lines, spirals, toolpaths) for 6-axis arms, fast.
- Checking many configurations for collisions at once.
- Obstacles with holes or concave shapes, through distance fields.
- Trying it out and telling us what breaks.

## Trade-offs

| What you get | What you give up |
|---|---|
| Cartesian paths in milliseconds, through the compas_fab API | 6-axis arms only: UR (offset wrist) and spherical-wrist arms |
| Closed-form IK derived from the robot model for spherical-wrist arms | No point-to-point (free-space) motion planning yet |
| Sphere-model collision checks for thousands of configurations per millisecond, validated on exact meshes | Attached workpieces are ignored (tools are supported) |
| Mesh obstacles with holes (distance fields) | Experimental: not on PyPI, built from source |

## Setup

From a compas_fab checkout, with a [Rust toolchain](https://rustup.rs) installed:

```bash
pip install ./native
```

This builds the `compas_fab_native` extension with maturin. For development,
`maturin develop --release -m native/pyproject.toml` rebuilds it in place.

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

## How it works, in short

1. `set_robot_cell` fits spheres inside every link mesh and sends the robot, its meshes and
   the obstacles to the core.
2. IK is closed-form: ported for UR arms, derived from the model for spherical-wrist arms.
3. A Cartesian request solves IK for every waypoint (and every sampled rotation about the
   tool axis), checks all solutions for collisions in one batch, and searches for the
   smoothest sequence through them.
4. The chosen path is validated on the exact meshes; if a configuration collides there,
   the search continues without it.

Design notes, measurements and open issues are in
[`native/SPIKE.md`](https://github.com/compas-dev/compas_fab/blob/spike/native-core/native/SPIKE.md).

## API reference

- `compas_fab.backends.native.NativePlanner`

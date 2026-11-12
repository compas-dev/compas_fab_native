# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

### Changed

### Removed


## [0.1.0] 2026-11-12

### Added

* `NativePlanner` and `NativeClient`: a compas_fab planning backend on a compiled (Rust) core, moved here from the `compas_fab` spike branch with its history.
* Forward kinematics, and inverse kinematics for frame and point-axis targets: closed form for UR arms and spherical-wrist arms (derived from the robot model), numerical and collision-aware for any other serial robot.
* `check_collision` on the sphere model or on the exact meshes, including tools and workpieces the robot carries.
* `plan_cartesian_motion` for frame and point-axis waypoints in robot, tool and workpiece mode, with free rotation about the tool axis.
* `plan_motion` to configuration, frame and point-axis targets, timed under joint velocity and acceleration limits.
* Mesh obstacles as signed distance fields (`NativePlanner(obstacles="esdf")`), which keep holes and concave shapes open.
* Release workflow on `compas-dev/compas-actions`: wheels for Linux (x86_64), macOS (universal2) and Windows (x64), each tested on Python 3.9 and 3.13, a source distribution, and publishing to PyPI.
* License texts of the Rust crates compiled into the extension, in `licenses/THIRD_PARTY_RUST.txt`.
* Documentation site.

### Changed

* Collision spheres are fitted in the core instead of in Python with trimesh: about 20 times faster (0.08 s instead of 1.7 s to set up a UR5), and trimesh and rtree are no longer dependencies.

### Removed

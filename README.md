# compas_fab_native

[![Made with COMPAS](https://compas.dev/badge.svg)](https://compas.dev/mission-control/#compas_fab_native)
[![CI](https://github.com/compas-dev/compas_fab_native/actions/workflows/ci.yml/badge.svg)](https://github.com/compas-dev/compas_fab_native/actions/workflows/ci.yml)
[![Wheels](https://github.com/compas-dev/compas_fab_native/actions/workflows/wheels.yml/badge.svg)](https://github.com/compas-dev/compas_fab_native/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/compas_fab_native.svg)](https://pypi.org/project/compas_fab_native/)
[![Python versions](https://img.shields.io/pypi/pyversions/compas_fab_native.svg)](https://pypi.org/project/compas_fab_native/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/compas-dev/compas_fab_native/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://compas.dev/compas_fab_native)

A planning backend for [compas_fab](https://github.com/compas-dev/compas_fab) whose work
happens in a compiled Rust core inside the Python process: no Docker, no PyBullet, no ROS.

It is experimental. The API is the usual compas_fab backend API, but what it supports is
limited and may change. The [documentation](https://compas.dev/compas_fab_native) covers
what it does, its trade-offs and how it works; [`SPIKE.md`](SPIKE.md) has the design notes,
measurements and open issues.

## Install

```bash
pip install compas_fab_native
```

Wheels are available for Linux (x86_64), macOS (Intel and Apple Silicon) and Windows (x64),
for Python 3.9 and later. Elsewhere, pip builds from source, which needs a
[Rust toolchain](https://rustup.rs).

## Use

It works with any serial robot in a compas_fab `RobotCell`:

```python
from compas_fab.robots import RobotCell
from compas_fab.robots import RobotCellLibrary
from compas_fab_native import NativePlanner

robot_cell, state = RobotCellLibrary.abb_irb4600_40_255()
# or your own robot:
# robot_cell = RobotCell.from_urdf_and_srdf("robot.urdf", "robot.srdf", "path/to/meshes")
# state = robot_cell.default_cell_state()

planner = NativePlanner()
planner.set_robot_cell(robot_cell, state)
solutions = list(planner.iter_inverse_kinematics(target, state))
trajectory = planner.plan_cartesian_motion(waypoints, state)
```

The planner picks the inverse kinematics when the cell is set:

- 6-axis arms with a spherical wrist (most ABB, KUKA, Stäubli, Fanuc and Yaskawa arms) get
  a closed form, built from the robot model.
- UR arms get a closed form when you pass their compas_fab solver, e.g.
  `NativePlanner(UR5Kinematics())`.
- All other robots (7-axis arms, robots on tracks or gantries) get numerical IK.

More examples are in [`docs/files/`](docs/files), among them
[`07_any_6_axis_arm.py`](docs/files/07_any_6_axis_arm.py) (an ABB printing a vase) and
[`08_which_ik.py`](docs/files/08_which_ik.py) (which IK five robots get).

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Rust workspace: the core (`motion_core`), exact checks and sphere fitting (`motion_exact`), GPU kernels (`motion_gpu`), benchmarks (`motion_bench`) and Python bindings (`motion_py`) |
| `src/compas_fab_native/` | The compas_fab backend: `NativePlanner` and the robot model translation |
| `tests/` | Python tests of the backend; the Rust tests live in each crate |
| `docs/` | Documentation, examples and diagrams |
| `demo/live_viewer/` | Live browser demo |
| `scripts/`, `fixtures/`, `results/` | Benchmark tooling, robot and scene data, and recorded results |

## Development

With a [Rust toolchain](https://rustup.rs) installed:

```bash
pip install -e ".[dev]"     # builds the extension; rerun after changing Rust code
python -m pytest
cargo test --workspace --exclude motion_py --release
ruff check && ruff format --check
```

Documentation: `pip install -e ".[docs]"`, then `mkdocs serve`.

## Releasing

1. Run the **prepare release** workflow (Actions tab) with the kind of bump. It opens a
   release pull request that bumps the version and dates the changelog.
2. Review and merge it. The **release** workflow then builds and tests the wheels and the
   source distribution, publishes them to PyPI, creates the GitHub release and deploys the
   documentation.

## License

MIT, see [`LICENSE`](LICENSE). Third-party code, data and algorithms used here are credited
in [`NOTICE`](NOTICE), with their licenses in [`licenses/`](licenses), including those of
the Rust crates compiled into the extension.

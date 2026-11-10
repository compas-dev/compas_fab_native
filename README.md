# compas_fab_native

A planning backend for [compas_fab](https://github.com/compas-dev/compas_fab) whose work
happens in a compiled Rust core inside the Python process: no Docker, no PyBullet, no ROS.

It is an early spike. The API is the usual compas_fab backend API, but what it supports is
limited and may change. See the [overview](docs/index.md) for what it does and its
trade-offs, [how it works](docs/how-it-works.md) for the internals, and
[`SPIKE.md`](SPIKE.md) for the measurements and open issues.

## Install

Not on PyPI yet. From a checkout of this repository, with a
[Rust toolchain](https://rustup.rs) installed:

```bash
pip install .
```

## Use

```python
from compas_fab.backends import UR5Kinematics
from compas_fab_native import NativePlanner

planner = NativePlanner(UR5Kinematics())
```

More examples are in [`docs/files/`](docs/files).

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Rust workspace: the core (`motion_core`), exact checks (`motion_exact`), GPU kernels (`motion_gpu`), benchmarks (`motion_bench`) and Python bindings (`motion_py`) |
| `src/compas_fab_native/` | The compas_fab backend: `NativePlanner` and the robot model translation |
| `tests/` | Python tests of the backend; the Rust tests live in each crate |
| `docs/` | Documentation, examples and diagrams |
| `demo/live_viewer/` | Live browser demo |
| `scripts/`, `fixtures/`, `results/` | Benchmark tooling, robot and scene data, and recorded results |

## Development

```bash
pip install maturin pytest
maturin develop --release   # builds the extension in place
python -m pytest tests
cargo test --workspace --exclude motion_py --release
```

## License

MIT, see [`LICENSE`](LICENSE). Third-party code, data and algorithms used here are credited
in [`NOTICE`](NOTICE), with their licenses in [`licenses/`](licenses).

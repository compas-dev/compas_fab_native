# compas_fab_native

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

```python
from compas_fab.backends import UR5Kinematics
from compas_fab_native import NativePlanner

planner = NativePlanner(UR5Kinematics())
```

More examples are in [`docs/files/`](docs/files).

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

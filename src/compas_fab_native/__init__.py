"""Native (Rust) motion core backend: spike, see ``native/SPIKE.md``.

The planner needs the ``compas_fab_native`` extension module (``pip install ./native``,
built with maturin from ``native/crates/motion_py``); importing this package does not.
"""

from .planner import NativeClient
from .planner import NativePlanner

__all__ = ["NativeClient", "NativePlanner"]

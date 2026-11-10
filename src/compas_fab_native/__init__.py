"""Native (Rust) motion core backend for compas_fab: spike, see ``SPIKE.md``.

``NativeCore`` comes from the compiled extension ``compas_fab_native._core``, built with
maturin from ``crates/motion_py``.
"""

from ._core import NativeCore
from .planner import NativeClient
from .planner import NativePlanner

__all__ = ["NativeClient", "NativeCore", "NativePlanner"]

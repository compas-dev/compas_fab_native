"""Native (Rust) planning backend for compas_fab.

``NativePlanner`` is the backend; ``NativeCore`` is the compiled core it drives, from the
extension module ``compas_fab_native._core``.
"""

from ._core import NativeCore
from .planner import NativeClient
from .planner import NativePlanner

__version__ = "0.1.0"

__all__ = ["NativeClient", "NativeCore", "NativePlanner"]

"""halfmesh — fast, compact half-edge triangle mesh processing.

Python bindings over the C++20 halfmesh library. All array-taking functions
accept float32 [N,3] vertices and uint32 [M,3] faces (numpy) and return NEW
arrays: halfmesh's half-edge construction auto-repairs non-manifold input, so
vertex/face indices are never guaranteed stable across a call.
"""

from ._halfmesh import (
    close_holes,
    remesh,
    remove_small_components,
    repair,
    simplify,
    smooth,
    version,
)

__version__ = version()
__all__ = [
    "close_holes",
    "remesh",
    "remove_small_components",
    "repair",
    "simplify",
    "smooth",
    "version",
    "__version__",
]

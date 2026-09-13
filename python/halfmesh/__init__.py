"""halfmesh — fast, compact half-edge triangle mesh processing.

Python bindings over the C++20 halfmesh library. All array-taking functions
accept float32 [N,3] vertices and uint32 [M,3] faces (numpy) and return NEW
arrays: halfmesh's half-edge construction auto-repairs non-manifold input, so
vertex/face indices are never guaranteed stable across a call.
"""

from ._halfmesh import (
    Mesh,
    close_holes,
    estimate_square_texture_size,
    pack_rectangles,
    remesh,
    remove_long_edge_faces,
    remove_long_edge_faces_capped,
    remove_long_edge_faces_local,
    remove_small_components,
    remove_spikes,
    remove_spurious_components,
    remove_vertices_and_fill,
    repair,
    simplify,
    smooth,
    unwrap,
    version,
)

__version__ = version()
__all__ = [
    "Mesh",
    "close_holes",
    "estimate_square_texture_size",
    "pack_rectangles",
    "remesh",
    "remove_long_edge_faces",
    "remove_long_edge_faces_capped",
    "remove_long_edge_faces_local",
    "remove_small_components",
    "remove_spikes",
    "remove_spurious_components",
    "remove_vertices_and_fill",
    "repair",
    "simplify",
    "smooth",
    "unwrap",
    "version",
    "__version__",
]

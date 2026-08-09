# AGENTS.md — examples (CLI tools)

Small command-line demos of the public API, built only with `-DHALFMESH_BUILD_TOOLS=ON`
(targets defined in `examples/CMakeLists.txt`). Each is a single `.cpp` using
`<halfmesh/...>` and namespace `halfmesh::`. See root AGENTS.md for conventions.

- `Decimate.cpp` — `decimate <in> <out> <ratio>`: QEM `Mesh::Simplify(ratio)`.
- `Remesh.cpp` — `remesh <in> <out>`: `RemeshIsotropic` (auto edge length from bbox diagonal).
- `Unwrap.cpp` — `unwrap <in> <out> <resolution>`: UV parametrize + atlas generation.
- `Smooth.cpp` — `smooth <in> <out> [iterations=5] [algo=hc]`: dispatches `hc`/`taubin`
  through the unified `Mesh::Smooth`; prints bbox-diagonal and mean-radius ratios so the
  two methods can be compared without a diffing pass.
- `TextureBakeTool.cpp` — `texturebake …`: rebake / defrag / texture-fidelity comparison
  harness; its fidelity sampler uses the half-texel (pixel-center) convention (`uv*size - 0.5`).

Decimate / Remesh / Unwrap first run a weld+clean preamble
(`RemoveDuplicateVertices(0)` → `RemoveDegenerateFaces(0)` → `RemoveUnreferencedVertices`)
so seam-split or dirty inputs process cleanly before the main op.

Usage (after building with tools):
```sh
./make/examples/decimate tests/data/mesh.ply /tmp/dec.ply 0.5
./make/examples/remesh   tests/data/mesh.ply /tmp/rem.ply
./make/examples/unwrap   tests/data/mesh.ply /tmp/uv.ply 1024
./make/examples/smooth   tests/data/mesh.ply /tmp/smooth.ply 20 taubin
```
These double as integration smoke tests and as copy-paste API examples. Keep them minimal
and dependency-free (just the halfmesh public headers).

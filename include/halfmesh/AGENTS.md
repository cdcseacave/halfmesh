# AGENTS.md — include/halfmesh (public API)

Public headers, namespace `halfmesh::`, all `#pragma once`. Consumers include
`<halfmesh/Mesh.h>` etc. Installed to `include/halfmesh/`; the CMake target is
`halfmesh::halfmesh`. See the root AGENTS.md for conventions (PascalCase, `ASSERT`,
no origin references).

## Header map
- `Mesh.h` — the main user-facing class `halfmesh::Mesh`: vertex/face/texcoord/texture
  containers + a `HalfMesh halfMesh` member; Load/Save (PLY/glTF), Simplify, RemeshIsotropic,
  SmoothHCLaplacian, SmoothTaubin, Smooth (unified dispatcher), CloseHoles, FixNonManifold, Remove*/repair, normals/area/AABB, adjacency helpers.
- `HalfMesh.h` — `halfmesh::HalfMesh`, the compact half-edge structure: 5 parallel index
  arrays (`v_halfedges/f_halfedges/he_nexts/he_vertices/he_faces`), twin = `h^1`,
  edge = `h/2`; O(1) adjacency iterators (V/F/E AdjacentX); edge ops `EFlip`, `ERemove`,
  and collapse-validity predicates `EIsCollapseValid{Topologically,Geometrically}`.
  `HALFMESH_TRIS=1` selects the triangle-optimised layout.
- `Types.h` — scalar/Eigen/OpenCV type aliases (`real`, `Vector3`, `Pixel`, `Image3u`, …),
  `math::NO_ID`, and the `cv::DataType<Pixel>` registration that makes `cv::Mat_<Pixel>` work.
- `Quadric.h` — `TQuadric` (QEM error quadric) used by decimation.
- `PriorityQueue.h` — `TPriorityQueue` (indexed mutable PQ) used by exact-mode decimation.
- `TriangleKDTree.h` — `TriangleKdTree`: nearest-point + ray-intersection spatial index;
  both queries take an optional `max_dist` search bound (default unbounded).
- `TriangleBVH.h` — `TriangleBVH`: binned-SAH BVH spatial index; `NearestPoint` /
  `IntersectedPoint` take an optional `max_dist` bound and `NearestPoint` a `hint_face` warm
  start (both default to off: unbounded, no hint).
- `OrientedBoundingBox.h` — `OBB` (used by `Mesh::RemoveFacesOutside`).
- `AtlasCharting.h` / `AtlasPacking.h` / `Parametrize.h` — the UV pipeline API (chart
  segmentation, flattening, density + atlas packing).
- `Version.h` — version constants.
- `InteropOpenMVS.h` — opt-in `halfmesh::Mesh ↔ MVS::Mesh` converters, guarded by
  `#if __has_include(<MVS/Mesh.h>)`; a no-op (and never compiled) without openMVS on the path.
- `Util/` — small header-only helpers; see below.

## Util/
`Assert.h` (the `ASSERT` macro — guarded, ASSERT-only), `Log.h` (`REPORT_*`/`TIMER_*`
via std::format), `Loop.h` (`FOREACH`/`RFOREACH`… index/pointer loop macros),
`Maths.h` (`SQUARE`/`D2R`/`IsZero`/`AreEqual`/`RoundCast`, tolerance constants),
`Geometry.h` (point/triangle/ray/box distance + intersection, in `namespace math`),
`Hash.h`, `Accumulator.h` (`WeightedAccumulator`).

## Notes
- Headers stay ISO-clean (the implementation `.cpp` files may use compiler extensions).
- `Mesh.h` exposes same-name const/non-const accessor overloads (e.g. `FVertex`) — no
  `Mutable`-suffixed accessors.

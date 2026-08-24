# T0 — M0: the representation contract

**Goal:** land plan §2.1/§2.2 exactly: `faces` becomes a derived snapshot with
the empty-`faces` flag, the `ListHalfEdges()` gate becomes exact, and every
existing public behavior is preserved (callers still see a populated `faces`
on return). Perf-neutral or better; no algorithm conversion happens here.

**Depends on:** nothing. Everything else depends on this.

## Context (verified against 4b9b538)

- The freshness gate compares BOTH counts (`src/Mesh.cpp:206-215`) — it
  misses count-preserving remaps and forces conservative clears everywhere.
- `HalfMesh::FFaces` APPENDS (`include/halfmesh/HalfMesh.h:596-603`) — it
  must only ever be called on an empty vector.
- These array mutators do NOT `halfMesh.Clear()` today and rely on the count
  gate: `RemoveDegenerateFaces` (both overloads, `src/MeshRepair.cpp:263,376`),
  `RemoveDuplicateFaces` (`:199`), `RemoveFacesOutside` (`:391`),
  `FixNonManifold` (`:410`), `RemoveUnreferencedVertices` (`src/Mesh.cpp:312`).
  The exact gate is UNSOUND until they all Clear.
- `Build` on a mesh with unreferenced vertices fails in RELEASE (NO_ID is odd
  → `ConnectBorders()` rejects the anchor, `src/HalfMesh.cpp:236-239`) and
  falls back to the expensive `ListHalfEdgesSafe`.

## Spec

1. **`Mesh::InvalidateFaces()`** (new): clears `faces`, `faceTexcoords`,
   `faceTexblobs`, `faceNormals`, `texturesDiffuse`, `vertexFaces`. If any
   attribute array was non-empty, emit a one-time
   `REPORT_WARNING("face attributes dropped: processing methods expect untextured meshes")`.
   Never call before a successful `ListHalfEdges()` (it would drop the only
   topology source).
2. **`Mesh::SyncFaces()`** (new):
   `if (faces.empty() && !halfMesh.Empty()) halfMesh.FFaces(faces);`
   Non-const (mutates `faces`).
3. **`Mesh::InvalidateHalfMesh()`** (new, public): `halfMesh.Clear();`.
   Document on the `faces` member in `include/halfmesh/Mesh.h`: users who
   hand-edit `faces` MUST call it — the exact gate cannot detect hand edits
   (the old count heuristic caught count-changing ones; this is a real
   behavior change, so the doc comment is mandatory, plus a line in
   `docs/FEATURES.md`).
4. **Exact gate** in `ListHalfEdges()` (`src/Mesh.cpp:206`): replace the
   count comparison with `if (!halfMesh.Empty()) return;`. Rebuild only from
   a non-empty `faces`. Assert the lost-topology state:
   `ASSERT(!(halfMesh.Empty() && faces.empty() && !vertices.empty()))`.
   Delete the now-stale gate comment.
5. **`halfMesh.Clear()` on every array mutator** listed in Context. Safest
   pattern: Clear at the exit of each public array mutator (not inside the
   `RemoveFaces`/`RemoveVertices` helpers — `RemoveSpuriousComponents` calls
   those mid-op and rebuilds after; a helper-level Clear works but audit each
   call site if you choose it, and say so in the commit message).
6. **Early-out rewrite.** HE-capable methods must not treat `faces.empty()`
   as "no triangles". Verified sites:
   - `Simplify` (`src/MeshSimplify.cpp:64`), `Smooth*`
     (`src/MeshSmooth.cpp:51,133`): early-out becomes
     `vertices.empty() || (faces.empty() && halfMesh.Empty())`. NOTE:
     Simplify then needs `SyncFaces()` on entry — its quadric setup reads
     `faces[iF]` aligned with HE face `iF` (`src/MeshSimplify.cpp:90-150`).
     Same for `RemeshIsotropic`.
   - `CloseHoles` (`src/MeshHoles.cpp:1257`), `RemoveVerticesAndFill`
     (`:1297`): same rewrite (keep their defensive Clear+Build for now — T5
     removes it).
   - Public array-native methods (`FixNonManifold`,
     `RemoveSpuriousComponents` `src/MeshRepair.cpp:570`, IO, atlas entry,
     `RemoveFacesOutside`): call `SyncFaces()` on entry instead. (T4/T6
     convert some of these later; entry-sync keeps M0 behavior-identical.)
7. **`SyncFaces()` at every public exit** that can leave HE-only state, and
   at the entry of every remaining `faces` consumer. Grep for `faces`
   iteration: MeshIO save paths, `ListVertexFaces`, `ComputeFaceNormals`,
   TriangleBVH/KdTree constructors, atlas/texture entry, python
   `to_arrays`/`n_faces` (`python/binding.cpp:180,198` — the lambdas take
   `const Mesh&`; make them take `Mesh&` or call SyncFaces at op exit before
   returning the object). In M0 most of these are no-ops (faces never empty
   yet) — the wiring is what M2+ relies on.
8. **Invariant asserts** (debug, at native-op entry/exit once ops exist; add
   the helper now):
   `faces.empty() || halfMesh.Empty() || faces.size() == halfMesh.FSize()`
   and `halfMesh.Empty() || vertices.size() == halfMesh.VSize()`.
9. **`ValidateHalfMesh()`** (debug/test helper): FFaces into a scratch
  vector, `Build` a scratch `HalfMesh`, validate the live five-array structure,
  and compare semantic topology (face triples, vertex adjacency/incidence,
  boundaries, and V/E/F counts). Do NOT compare raw arrays after a native
  mutation: swap-compaction, face anchors, and legal odd representatives make
  positional identity with a fresh all-even Build impossible. Expose to the
  test infra (used by T1-T6).
10. **Do NOT** touch Simplify/Remesh `FFaces` epilogues (plan §4.8).

## Pitfalls

- `FFaces` appends — every call site must guarantee an empty destination.
- Python `n_faces` on an HE-only mesh must not return 0 — that is why exits
  sync before returning to python.
- `Mesh::Empty()` (`include/halfmesh/Mesh.h:64-68`) already permits
  vertices+HE with empty faces; keep its assert satisfied in the new states.

## Acceptance

- Full suite green (507/507), including python tests.
- New tests: exact-gate test (hand-edit faces → stale HE without
  `InvalidateHalfMesh`, fresh after), state-machine round-trip
  (arrays-only → build → InvalidateFaces → SyncFaces → identical faces
  array), invariant + ValidateHalfMesh smoke test, warn-on-drop test.
- No measurable perf regression on the perf suite (`HALFMESH_BUILD_PERF`).

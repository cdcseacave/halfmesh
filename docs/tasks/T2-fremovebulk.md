# T2 — M1b: `FRemoveBulk` + Mesh wrapper + `VRemoveUnreferenced`

**Goal:** the workhorse primitive — remove an arbitrary set of faces in one
pass, keeping the half-edge valid AND manifold, with vertex bookkeeping
reported so the Mesh layer keeps `vertices`/`vertexColors` in lockstep.

**Depends on:** T0, T1 (uses the parity-agnostic relink core).

## Context (verified against 4b9b538)

- Existing compaction batch primitives (built for exactly this, descending
  order): `FRemoveOnly(FIndex*, unsigned)`, `ERemoveOnly(EIndex*, unsigned)`,
  `VRemoveOnly(VIndex*, unsigned)` (`include/halfmesh/HalfMesh.h:452-454,
  623-625, 752-754`).
- `FRemove` does NOT support faces with >1 border edge (doc at
  `HalfMesh.h:607-617` + the standing `TODO(Dan)` this task resolves);
  `FRemoveCorner` handles only the single-face corner case. Bulk removal of
  scattered faces WILL hit 2-3-border-edge triangles — this is a new
  primitive, not a loop around `FRemove`.
- Lockstep model to mirror: `Mesh::ECollapse` (`src/Mesh.cpp:296-310`) —
  `halfMesh.ERemove` reports removed verts via `RemovedData`, wrapper
  swap-pops `vertices`/`vertexColors` in the same order.
- Pinch-split model to mirror: array `FixNonManifold`
  (`src/MeshRepair.cpp:477-506`) — duplicates the vertex with
  `vertices.emplace_back(v)` and reports through the `duplicatedVertices`
  out-param.
- Boundary-representative invariant: `include/halfmesh/HalfMesh.h:87-92` —
  load-bearing for `VIsBoundary`/`EnumerateHoles`/boundary loops.

## Spec

```cpp
// Remove a set of faces in one pass, keeping the half-edge valid and manifold.
// removedVerts: every vertex that lost its last face, in the same descending
//   order VRemoveOnly uses.
// splitSrcVerts: for every vertex DUPLICATED by a pinch split, the SOURCE
//   vertex index — one entry per appended vHalfedges slot, in append order.
void FRemoveBulk(std::vector<FIndex>& faceRemoves,
                 std::vector<VIndex>& removedVerts,
                 std::vector<VIndex>& splitSrcVerts);
```

Algorithm, O(removed + affected-border) — connectivity surgery on
pre-compact indices, compact last:

1. Mark removed faces (bitset).
2. Half-edge of a removed face whose twin's face survives → twin becomes a
   border half-edge (`heFaces = NO_ID`). If both sides die — twin's face also
   removed, OR twin was already border (removal at an existing boundary) —
   the edge dies (`ERemoveOnly` batch).
3. Vertices whose every incident face died → collect (`VRemoveOnly` batch +
   report via `removedVerts`). Survivors: repoint via `SetVHalfedge` to the
   **boundary-canonical representative** (interior-side half-edge whose twin
   is the new border half-edge) — NOT to an arbitrary surviving
   out-half-edge; hole enumeration dereferences it blindly.
4. **Pinch vertices:** a surviving vertex whose surviving faces form ≥2
   edge-connected fans → duplicate the vertex (append a `vHalfedges` slot),
   rewire one fan to the duplicate, report the source index via
   `splitSrcVerts`. Do NOT try to keep one `vHalfedges[v]` for two fans and
   do NOT "close through ConnectBorders" — `Build` rejects bow-ties and
   `ConnectBorders` cannot split them. Leaving a pinch makes the T6
   `FixNonManifold` short-circuit unsound.
5. Re-link border `heNexts` by circulating each vertex touched in (2)/(4),
   using T1's parity-agnostic relink core.
6. Compact with the descending-order batch overloads.

**Mesh wrapper** `Mesh::RemoveFacesHalfEdge(std::vector<FIndex>&)`:
calls `FRemoveBulk`; appends one position (+color if `vertexColors`
non-empty) per `splitSrcVerts` entry (copy of the source vertex); THEN
replays `removedVerts` as swap-pops (mirror of `ECollapse`, same index
order — both index streams refer to the post-append numbering, since a
descending swap-pop may relocate an appended duplicate); calls
`InvalidateFaces()`. Exit assert: `vertices.size() == halfMesh.VSize()`.

**`VRemoveUnreferenced(std::vector<VIndex>& removedVerts)`** (trivial): one
pass over `vHalfedges`, batch-`VRemoveOnly` every `NO_ID` slot, report for
lockstep. (Native pipelines only; the array
`Mesh::RemoveUnreferencedVertices` stays for ingest — a valid HE normally has
no `NO_ID` slots, see plan §3.3.)

## Pitfalls

- Whole-component removal is the degenerate easy case (no new borders); do
  not let its simplicity drive the design.
- Pinch detection must run on the SURVIVING fan structure, not the original.
- Do not invent a second application order for the wrapper replay — the
  primitive dictates it.

## Acceptance

Tests (each followed by `ValidateHalfMesh` + semantic comparison against a
from-scratch `Build` of the harvested faces; never raw array identity):
- multi-border-edge faces (2 and 3 border edges);
- a removal pattern that REQUIRES a pinch split (e.g. remove two opposite
  faces of a 6-fan interior vertex); verify `splitSrcVerts`, verify
  `vertices.size() == VSize()` after the wrapper, verify a follow-up
  `Build` of harvested faces succeeds (no bow-tie);
- pinch at an existing boundary vertex;
- removal at an existing boundary (twin already border → edge dies);
- scattered faces across a big mesh; a whole component; ALL faces;
- cascade to isolated vertices (spike-like), reported in descending order;
- run on a mutated HE (post `ESplit`/`EFlip`, `alwaysEven == false`).
- Full suite green.

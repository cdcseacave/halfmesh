# T3 — M1c: `FAdd` isolated-vertex + two-new-edges support

**Goal:** extend `FAdd` so hole-fill harvests can attach patch triangles that
introduce new (isolated) vertices — finishing the existing TODO and lifting
the two verified blockers.

**Depends on:** T0, T1 (FAdd failure propagation must already be in).

## Context (verified against 4b9b538)

`FAdd` (`src/HalfMesh.cpp:511-585`) has two blockers:

1. `ASSERT(VIsBoundary(face[v]))` (`:524`): `VIsBoundary` is
   `EHeIsBoundary(VHalfedge(v))`; for an isolated vertex `VHalfedge` is
   `NO_ID`, so this OOB-indexes `heFaces[NO_ID]` — a crash, not a soft
   failure. Isolated verts must become a first-class legal corner (skip the
   boundary test; never index with NO_ID).
2. `numNewEdges[v] > 1 → return NO_ID` (`:538-540`) rejects the actual grow
   step: a triangle sharing ONE existing boundary edge plus a new/isolated
   third vertex creates TWO new edges at that vertex. Isolated vertices must
   be allowed two new edges. Existing boundary vertices keep the ≤1 rule —
   that is the non-manifold-vertex guard, do not weaken it.

Caller contract (unchanged): the caller pre-appends positions to
`Mesh::vertices` and `NO_ID` slots to `vHalfedges` before calling.

Attach-order reality: BFS-from-one-edge does NOT always pass `FAdd`. Two
workable orders exist — ear-clip/two-existing-edges (works with today's FAdd
for boundary-only patches) and the retry queue `TriangulateHole` already
implements (`src/HalfMesh.cpp:415-435`: walk the DP tree, `FAdd`, on `NO_ID`
enqueue and retry). Reuse that pattern; do not invent a weaker BFS attacher.

## Spec

1. Lift blocker (1): treat `vHalfedges[face[v]] == NO_ID` as "isolated
   corner". All paths that read `VHalfedge(face[v])` must guard it.
2. Lift blocker (2): allow `numNewEdges[v] == 2` iff vertex `v` is isolated;
   keep `> 1 → NO_ID` for connected vertices.
3. New-edge creation for an isolated tail must set its representative (via
   `SetVHalfedge`) and end with valid border links for both new border
   half-edges — the T1 parity-agnostic relink core handles the circulation.
4. Provide (or verify TriangulateHole-reuse covers) a bulk disk attach for
   pre-triangulated patches: iterate patch triangles through the retry
   queue; a patch that still cannot attach is skipped wholesale (same
   outcome as today's `ok[j]==0` path in the holes code — the mesh must be
   left untouched by the skipped patch, which T1's clean-reject guarantees
   per face; back out any partially attached patch faces via
   T2's `FRemoveBulk`).
5. Downstream reality check (for T5): `RemoveVerticesAndFill` uses
   `refine=false` (boundary-only patches — retry/ear-clip suffices);
   `CloseHoles` uses Liepa+refine (interior Steiner vertices) — that path
   REQUIRES this task's isolated-vertex + two-new-edges support.

## Acceptance

- Full suite green; goldens unchanged (no behavior change for existing
  callers — TriangulateHole paths must be bit-identical).
- New tests: attach a triangle with one existing boundary edge + isolated
  third vertex; attach a full pre-triangulated disk (with interior Steiner
  vertex) onto a hole via the retry queue; pathological patch → clean skip,
  structure unchanged; `ValidateHalfMesh` after each; never index
  `heFaces[NO_ID]` under ASAN/debug.

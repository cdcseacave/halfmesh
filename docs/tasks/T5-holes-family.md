# T5 — M3: holes family native (`CloseHoles` / `RemoveVerticesAndFill`)

**Goal:** delete the defensive entry rebuilds and make the patch harvest
append natively via `FAdd` — hole filling with zero Builds.

**Depends on:** T0, T1 (CRITICAL — see below), T2, T3.

## Context (verified against 4b9b538, `src/MeshHoles.cpp`)

- `CloseHoles` (`:1254`): early-out on `faces.empty()` (rewritten in T0),
  then defensive `halfMesh.Clear(); ListHalfEdges();` (`:1259-1264` — the
  comment "caches by vertex count" is stale; delete it with the code).
- `RemoveVerticesAndFill` (`:1295`): same defensive `Clear+Build`
  (`:1303-1304`); removal phase via array `RemoveVertices(…, false)`
  (`:1417`); calls `FillBoundaryLoops(…, refine=false, rebuildHalfMesh=false)`
  (`:1458`).
- `FillBoundaryLoops` epilogue (`:1227-1240`): drops face attributes, clears
  `vertexFaces` + `halfMesh`, optional rebuild. Liepa fill computes per-hole
  on local copies — KEEP that isolation; only the harvest changes.
- **Why T1 must land first:** the defensive rebuilds are what currently
  guarantee `FAdd`'s all-even `ConnectBorders` assumption. Deleting them
  without T1 arms a silent-corruption path on the first post-Simplify hole
  (plan §3.4). Do not start this task if T1 is not merged.

## Spec

1. Delete both defensive `Clear()+ListHalfEdges()` entries — the §2.1
   contract plus T1 make them unnecessary. Entry becomes plain
   `ListHalfEdges()` (no-op when HE valid).
2. `RemoveVerticesAndFill` removal phase → `Mesh::RemoveFacesHalfEdge` (T2)
   with the vertex set's incident faces; drop the `vertexFaces` dependency.
3. Harvest, native: append interior (Steiner) vertices — positions to
   `Mesh::vertices`, `NO_ID` slots to `vHalfedges` (T3 caller contract) —
   then add patch triangles via `FAdd` using **`TriangulateHole`'s retry
   queue and/or ear-clip order** (plan §3.2/§4.3). NOT naive BFS — a
   triangle sharing only one edge is the `numNewEdges==2` case and only
   works for corners T3 legalized. `refine=false` paths (boundary-only
   patches) need retry/ear-clip only; the Liepa+refine path (CloseHoles)
   exercises T3's isolated-vertex support.
4. A patch that still cannot attach → skip that hole, mesh untouched (same
   contract as today's `ok[j]==0`); back out partial attachment via
   `FRemoveBulk`.
5. `holesFaces` output: indices stay valid across a later `SyncFaces`
   because `FAdd` appends and `FFaces` walks `fHalfedges` in index order —
   valid ONLY while no face removal/compaction runs between harvest and
   sync. Assert/document that.
6. `FillBoundaryLoops` epilogue: replace the attribute-drop + `Clear()` with
   `InvalidateFaces()` semantics (T0) — no `Clear()`, no rebuild.
   `rebuildHalfMesh` parameter dies. Public exits `SyncFaces()`.

## Acceptance

- MeshHoles test suite green with goldens (fills are computed by the same
  Liepa/refine/fair code on the same local copies — surviving geometry must
  be identical; if harvest ORDER changes face indices, regen goldens and
  say so explicitly in the commit message).
- Zero-`Build` gate: CloseHoles on a prebuilt HE = 0 Builds; the plan's
  Truck-scale criterion "holes-with-zero-fills costs ~0" gets a perf test
  (today it is 6.2 s of pure rebuild).
- Post-Simplify integration test: build → `Simplify` (mutated, never-rebuilt
  HE) → `CloseHoles` → `ValidateHalfMesh` (this is the §3.4 trap regression
  test).
- `RemoveVerticesAndFill` + `holesFaces` consumers (OpenMVS uses it) —
  python/goldens green.

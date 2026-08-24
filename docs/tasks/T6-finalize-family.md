# T6 — M4: finalize family (degenerates / unref / FixNonManifold short-circuit)

**Goal:** native geometric-degenerate cleanup, dispatching unref, and the
`FixNonManifold` short-circuit that makes the half-edge a manifoldness
certificate.

**Depends on:** T0, T2 (uses `VRemoveUnreferenced`; T1 transitively).

## Context (verified against 4b9b538)

- `RemoveDegenerateFaces(Type thArea)` (`src/MeshRepair.cpp:263-374`):
  `vertexFaces`-based scan; welds via ad-hoc index remaps (an edge collapse
  in disguise) with NO Hoppe link/fold checks — it can create
  non-manifoldness (header comment already says follow with unref +
  FixNonManifold). Iterated wrapper at `:376`.
- `RemoveDegenerateFaces(0)` is topology-only (repeated-index faces) and is
  what `ListHalfEdgesSafe` uses (`:58`) — MUST stay array-native for ingest.
- `FixNonManifold` (`:410-534`): array bow-tie splitter — the manifoldizer.
  A valid HE cannot represent what it fixes.
- `Mesh::IsManifold` (`:68-92`) is edge-only — bow-ties PASS it. It is NOT a
  substitute for `Build`'s return value.

## Spec

1. **Native geometric degenerates** (`RemoveDegenerateFacesHalfEdge`):
   scan faces via `fHalfedges` for area ≤ threshold (positions from
   `Mesh::vertices`); **needles** → `ERemove` on the shortest edge (existing
   validity checks apply); **caps** → `EFlip` the long edge, or collapse.
   Duplicate-index faces cannot exist in a valid HE (Build rejects them) —
   that branch lives only in the array arm. Native `ERemove` will REFUSE
   some welds the array path performs: do NOT promise identical removed
   counts. Expect golden regen + metamorphic checks (Hausdorff / remaining
   sub-threshold area), not bit-identity. Counts are tiny; per-element
   `ERemove` cost is irrelevant — the win is deleting `ListVertexFaces` +
   the follow-up rebuild.
2. **Dispatch (plan §4.9):** public `RemoveDegenerateFaces(thArea)`
   dispatches: HE non-empty → native arm; arrays-only → today's array arm +
   `halfMesh.Clear()` (T0 already added the Clear). Both arms public
   (`...Arrays` / `...HalfEdge`). Same treatment for
   `RemoveUnreferencedVertices` (array arm `src/Mesh.cpp:312`; native arm =
   T2's `VRemoveUnreferenced` + wrapper lockstep swap-pops).
3. **`FixNonManifold` short-circuit:** first line becomes
   `if (!halfMesh.Empty()) return 0;` — a valid HE is a manifoldness
   certificate (Build rejects violations; native mutators preserve
   manifoldness because `FRemoveBulk` splits pinches). The array
   implementation stays for ingest via `ListHalfEdgesSafe` — which `Clear()`s
   first, so the short-circuit cannot misfire there. Keep the T0 entry
   `SyncFaces()` for the arrays-only path.
4. Do NOT weaken `ListHalfEdgesSafe`'s sequence
   (`src/MeshRepair.cpp:36-63`) — it is the one orchestrator of the ingest
   repairs, and its steps run on the soup BEFORE `Build`.

## Acceptance

- Corpus metamorphic tests: degenerate conversion exempt from count
  identity (assert Hausdorff ≈ 0 on surviving geometry + no face above the
  old threshold survives that the array path would have removed AND the
  native validity checks permit removing); unref identical-by-construction.
- Short-circuit test: manifold mesh → build → `FixNonManifold()` returns 0
  without touching anything; soup (bow-tie fixture) → array path still
  splits it; `ListHalfEdgesSafe` end-to-end unchanged (goldens).
- Zero-`Build` gate for a spurious→spikes→degenerates→unref native chain.
- Full suite green (regen degenerate goldens if needed — call it out).

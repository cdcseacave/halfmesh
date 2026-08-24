# T4 — M2: repair family native (spurious / small components / spikes)

**Goal:** convert `RemoveSpuriousComponents`, `RemoveSmallComponents`,
`RemoveSpikes` to half-edge-native (zero rebuilds), and introduce the §4.9
representation-dispatch public wrappers.

**Depends on:** T0, T1, T2.

## Context (verified against 4b9b538, all in `src/MeshRepair.cpp`)

- `RemoveSpuriousComponents` (`:568-642`): HE edge-length percentiles
  (already native), then long-edge faces via ARRAY scan + `RemoveFaces` +
  `RemoveUnreferencedVertices` + full `ListHalfEdges` rebuild (`:609`), then
  `ConnectedComponents` (native) + array surgery + final `Clear()` (`:638`).
  ~2 rebuilds paid, 3rd forced on the next consumer.
- `RemoveSmallComponents` (`:539-566`): `ListHalfEdges` →
  `ConnectedComponents` → swap-pop small-component faces →
  `RemoveUnreferencedVertices` → `Clear()`.
- `RemoveSpikes` (`:644-670`): per-iteration `ListVertexFaces()` (full
  O(V+F) third-representation rebuild), collect `vertexFaces[v].size() <= 1`,
  `RemoveVertices(spikes, true)`, final `Clear()`.

## Spec

1. **Spurious, native:** percentile loop unchanged; collect over-long-edge
   incident faces DURING that loop (it already touches every edge — the
   set is identical to the array scan: a face is long iff one of its three
   edges is long; border half-edges contribute no face). Remove via
   `Mesh::RemoveFacesHalfEdge` (pinch-split may fire — scattered long-edge
   drops can pinch). `ConnectedComponents` unchanged; small-component
   removal via `RemoveFacesHalfEdge` (whole components — the easy case).
   Zero rebuilds, zero `vertexFaces`.
2. **Small components, native:** same pattern, whole-component
   `RemoveFacesHalfEdge`, no unref pass needed (`FRemoveBulk` cascades), no
   `Clear()`.
3. **Spikes, native:** detect by circulating `vHalfedges[v]` — valence-1
   (exactly one incident face). NOTE: isolated (`NO_ID`) slots cannot exist
   in a valid HE (plan §3.3), so the array `size() <= 1` zero-face case has
   no native counterpart. First pass O(V); collect incident faces;
   `RemoveFacesHalfEdge`. Cascade via a worklist seeded with the neighbours
   of removed faces — iterations after the first are O(affected).
4. **Dispatch wrappers (plan §4.9):**
   - `RemoveSpikes` is genuinely dual — keep the array implementation as
     `RemoveSpikesArrays` (works on non-manifold soup, no Build), native as
     `RemoveSpikesHalfEdge`, public name dispatches on
     `halfMesh.Empty()`. Both suffixed variants public.
   - Spurious/small-components already require the HE today → native-only
     (entry stays `ListHalfEdges()`), no array arm to keep. Their exit no
     longer Clears; public exit calls `SyncFaces()`.
5. Every native path ends with the mesh in half-edge-primary state
   (`InvalidateFaces` semantics via `RemoveFacesHalfEdge`) + `SyncFaces()`
   at the public exit (T0 convention).

## Texture note (plan §2.2)

The array arms happen to preserve `faceTexcoords`/`faceTexblobs`/
`faceNormals` (via `Mesh::RemoveFaces` lockstep swap-pop) — mark this as a
documented bonus in the header comments; the native arms drop attributes by
design with the one-time warning. Do not engineer attribute preservation
into the native arms.

## Acceptance

- Metamorphic tests old-vs-new on the manifold test corpus: IDENTICAL
  removed-face sets for spikes and both component removers (these are
  set-identical by construction — assert it), Hausdorff ≈ 0 on surviving
  geometry.
- Zero-`Build` gate: instrument a Build counter (add to
  `HALFMESH_BUILD_PERF` suite); a spurious→spikes pipeline on a prebuilt HE
  performs 0 Builds.
- Pinch regression test: a spurious run whose long-edge drops create a pinch
  (construct one) → output still Build-able, `ValidateHalfMesh` green.
- Full suite green; python `remove_small_components` behavior unchanged.

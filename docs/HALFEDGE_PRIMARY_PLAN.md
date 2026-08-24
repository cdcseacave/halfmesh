# Half-edge–primary mesh processing — implementation plan

Status: **planned** (hand-off to the halfmesh repo agent) · Branch: `openmvs-integration`
Driven by the OpenMVS `Mesh::Clean` integration; measured numbers below are from
Tanks&Temples *Truck* (6.1M faces) via `ReconstructMesh --mesh-file` clean-only runs.

## 1. Problem

`Mesh` carries two topology representations: the `faces` array and `halfMesh`
(connectivity-only: `vHalfedges`, `fHalfedges`, `heNexts`, `heVertices`, `heFaces`;
positions always live in `Mesh::vertices` — only *faces* is duplicated state).
The algorithms split into two families:

- **Half-edge-native** — `Simplify` (MeshSimplify.cpp), `RemeshIsotropic`
  (MeshRemesh.cpp), `Smooth*` (positions only). They mutate through
  `ERemove`/`ESplit`/`EFlip` and resync the array with `FFaces(faces)`.
- **Array-native** — everything in MeshRepair.cpp and MeshHoles.cpp
  (`RemoveSpuriousComponents`, `RemoveSpikes`, `CloseHoles`,
  `RemoveVerticesAndFill`, `RemoveDegenerateFaces`,
  `RemoveUnreferencedVertices`, `FixNonManifold`, …). They do swap-pop/append
  surgery on `faces`/`vertices` (often via a *third* derived structure,
  `vertexFaces`) and end with `halfMesh.Clear()`.

A pipeline that interleaves the families (OpenMVS `Clean`: spurious → spikes →
simplify → holes → smooth → remesh → finalize) pays a full `HalfMesh::Build`
at every family transition. `Build` is O(F) but heavy — an
`unordered_map<uint64,HIndex>` over ~3F edges ≈ **3 s per rebuild on Truck** —
and `Clean` pays 3–4 of them: hole closing costs 6.2 s before filling a single
hole, and total clean time is 21.7 s vs 13.4 s for the old CGAL path (which
built its connectivity once and mutated it in place through every stage).

Secondary problem: there is no sound way to know the two representations still
agree. The freshness gate in `Mesh::ListHalfEdges()` compares element *counts*,
which misses any count-preserving remap, so algorithms must clear conservatively
(`CloseHoles` even does a defensive `Clear()` + rebuild on entry).

## 2. Target architecture

**The half-edge is the working representation; `faces` is a derived snapshot.**

- `ImportMesh`/first `ListHalfEdges()` builds the half-edge **once**.
- Every topology-mutating algorithm operates on the half-edge and keeps it
  valid; none of them clears it.
- `faces` is regenerated **once**, on demand, via `FFaces()` (export, save, or
  entry into a remaining array-consumer).
- `vertices` (+ `vertexColors`) is always live: any primitive that removes a
  vertex swap-pops the position arrays in lockstep, exactly as
  `Mesh::ECollapse` already does around `HalfMesh::ERemove`.

### 2.1 Validity contract (problem 2 — decision)

Two options were considered:
(a) a dirty flag set in the mutating primitives;
(b) topology-changing half-edge algorithms always drop the `faces` array, and
faces are regenerated once at the end.

**Decision: (b), with the empty `faces` array itself acting as the flag** — no
new state member at all. The rules:

1. `Mesh::InvalidateFaces()` (new, trivial): clears `faces` **and every
   face-keyed attribute** (`faceTexcoords`, `faceTexblobs`, `faceNormals`) plus
   `vertexFaces`. Called on entry by every half-edge-native topology algorithm
   (Simplify, Remesh, CloseHoles, RemoveSpuriousComponents, …). This is exactly
   the invalidation `FillBoundaryLoops` already performs today, made uniform.
2. `Mesh::SyncFaces()` (new, trivial): `if (faces.empty() && !halfMesh.Empty())
   halfMesh.FFaces(faces);`. Called by `ExportMesh`-style consumers: Save/IO,
   `ListVertexFaces`, `RemoveFacesOutside`, the atlas/texture pipeline entry
   points — any code that iterates `faces`.
3. Array-mutating algorithms (ingest-time repairs that stay array-native, §4.7)
   keep today's convention: mutate arrays, then `halfMesh.Clear()`.
4. `Mesh::ListHalfEdges()` gate becomes exact: `if (!halfMesh.Empty()) return;`
   — a non-empty half-edge is valid **by contract** (any array surgery cleared
   it; any native mutation maintained it). The count-comparison heuristic is
   deleted. Rebuild happens only from a non-empty `faces` array (call
   `SyncFaces()` first is a bug — assert on it).

Class invariant (assert in debug at algorithm entry/exit):

```
faces.empty() || halfMesh.Empty() || faces.size() == halfMesh.FSize()
```

States: *arrays-only* (fresh load, post-ingest-repair), *half-edge-only*
(mid-pipeline, faces dropped), *both-and-consistent* (right after `Build` or
`SyncFaces`). Nothing else is representable, which is the point.

Option (a) was rejected because a `bool facesValid` member would have to be
touched in exactly the same set of functions while adding a second source of
truth that can contradict the arrays; (b) encodes the state in data that cannot
lie. Debug builds additionally get `ValidateHalfMesh()` (rebuild into a scratch
`HalfMesh` and compare) wired into the test corpus after each native op.

Note the OpenMVS `Clean` bridge (`MeshHalfMesh.cpp`) then does exactly:
one `ImportMesh` → stages (no clears, no rebuilds) → one `SyncFaces` in
`ExportMesh`. One `Build` + one `FFaces` per clean, total.

## 3. New/changed HalfMesh primitives

### 3.1 `FRemoveBulk` — the workhorse (new)

```cpp
// Remove a set of faces in one pass, keeping the half-edge valid.
// Appends to removedVerts (in application order) every vertex that lost its
// last face, so the caller can swap-pop positions/colors in lockstep.
void FRemoveBulk(std::vector<FIndex>& faceRemoves, std::vector<VIndex>& removedVerts);
```

Algorithm, O(removed + affected-border):
1. Mark removed faces (bitset).
2. For every half-edge of a removed face whose twin's face survives, the twin
   edge becomes a border half-edge (`heFaces = NO_ID`); if both sides die, the
   edge dies (`ERemoveOnly` batch).
3. Vertices whose every incident face died are collected (`VRemoveOnly` batch +
   reported through `removedVerts`); survivors get `SetVHalfedge` repointed to a
   surviving out-half-edge.
4. Re-link border `heNexts` by circulating each affected vertex — the same
   local rule `ConnectBorders(HIndex&)` implements; only vertices touched in
   (2) need visiting.
5. Compact with the existing descending-sort `FRemoveOnly`/`ERemoveOnly`/
   `VRemoveOnly` batch overloads (they were built for exactly this).

This subsumes the single-face `FRemove` + its `ConnectBorders`-discipline
footnotes and resolves the standing
`TODO(Dan): re-factor to keep connectivity and remove ConnectBorders requirement`.
Whole-component removal is the degenerate easy case (no new borders at all).

Mesh-level wrapper `Mesh::RemoveFacesHalfEdge(std::vector<FIndex>&)`: calls
`FRemoveBulk`, replays `removedVerts` as swap-pops on `vertices`/`vertexColors`
(mirror of `Mesh::ECollapse`), calls `InvalidateFaces()`.

### 3.2 `FAdd` isolated-vertex support (finish the existing TODO)

`FAdd` already stitches a face onto border edges/vertices with full
`ConnectBorders` handling; it only lacks
`TODO(Dan): add support for creating new vertices`. Extend it to accept
vertices with `vHalfedges[v] == NO_ID` (isolated: treated as boundary with no
incident edge). The caller pre-appends positions to `Mesh::vertices` and
`NO_ID` slots to `vHalfedges`.

### 3.3 Unreferenced-vertex sweep (new, trivial)

`VRemoveUnreferenced(std::vector<VIndex>& removedVerts)`: one pass over
`vHalfedges`, batch `VRemoveOnly` every `NO_ID` slot, report for lockstep.
Replaces the `vertexFaces`-based `Mesh::RemoveUnreferencedVertices` inside
native pipelines.

## 4. Per-operation conversion audit

Each stage of the OpenMVS `Clean` pipeline, current vs. native design, and the
speed verdict. All are convertible at equal or better asymptotic *and*
practical cost; none needs the `faces` array or `vertexFaces`.

### 4.1 `RemoveSpuriousComponents` — native, strictly faster
Current: edge-length percentiles over half-edge edges (already native), then
long-edge faces via **array scan** + `RemoveFaces` + `RemoveUnreferencedVertices`
+ **full rebuild**, then `ConnectedComponents` (native) + component removal via
array surgery + final `Clear()`. Pays ~2 rebuilds.
Native: percentiles unchanged; long-edge faces found by iterating half-edge
edges directly (the loop at the top already computes every edge length —
collect the incident faces of over-long edges there); remove via `FRemoveBulk`;
`ConnectedComponents` unchanged; component removal via `FRemoveBulk` (easy
case). Zero rebuilds, no `vertexFaces`.

### 4.2 `RemoveSpikes` — native, strictly faster
Current: per-iteration `ListVertexFaces()` (full O(V+F) rebuild of the third
representation), scan for vertices with ≤1 incident face, `RemoveVertices(…,
true)`, final `Clear()`.
Native: a spike vertex is detected by circulating `vHalfedges[v]` — valence-1
(one incident face) or isolated. First pass O(V); collect the incident faces,
`FRemoveBulk`. Cascade via a **worklist**: only neighbours of removed faces can
become spikes, so iterations after the first are O(affected) instead of O(V).

### 4.3 `CloseHoles` / `RemoveVerticesAndFill` — native, strictly faster
Current: defensive `Clear()`+`Build` on entry, native loop enumeration, Liepa
fill computed per-hole on local copies (already isolated — keep as is), then
harvest **appends to `vertices`/`faces`** and ends with `Clear()` (+ another
`Build` when `rebuildHalfMesh`). Two full rebuilds to fill holes whose patches
total a few thousand triangles.
Native: entry rebuild deleted (the §2.1 contract makes it unnecessary);
`RemoveVerticesAndFill`'s removal phase uses `FRemoveBulk`; harvest appends
interior vertices (positions + `NO_ID` `vHalfedges` slots) and adds patch
triangles via `FAdd` in **advancing-front (BFS) order** over the patch dual
graph starting from a loop-adjacent triangle — each added face then shares at
least one existing edge, so `FAdd`'s manifold checks always pass on a disk
patch. O(patch) per hole. (If a pathological patch is ever rejected by `FAdd`,
fall back to skipping that hole — same outcome as today's `ok[j]==0` path.)

### 4.4 `RemoveDegenerateFaces` — native, equal-or-faster
Current: `vertexFaces`-based scan + face removal + manual vertex-welding via
index remaps (an ad-hoc edge collapse), iterated up to `maxIterations`.
Native: scan faces for area ≤ threshold via `fHalfedges` (positions from
`Mesh::vertices`); **needles** → `ERemove` (collapse shortest edge, existing
validity checks); **caps** → `EFlip` the long edge (existing) or collapse;
duplicate-index faces cannot exist inside a valid half-edge (Build rejects
them), so that branch survives only in the array-native ingest variant.
The welding semantics *are* edge collapses — the native form is the honest
implementation of what the array code approximates. Degenerate counts are tiny,
so per-element `ERemove` cost is irrelevant; the win is deleting the
`ListVertexFaces` + post-hoc rebuild.

### 4.5 `RemoveUnreferencedVertices` — native, trivial (§3.3).

### 4.6 `FixNonManifold` — becomes an ingest-only concern
A valid half-edge **is** a manifoldness certificate: the structure cannot
represent non-manifold topology, `Build` detects violations and falls back to
the repairing `ListHalfEdgesSafe`, and every native mutator preserves
manifoldness. So inside a native pipeline `FixNonManifold` short-circuits:
`if (!halfMesh.Empty()) return 0;`. The array implementation stays, used
exactly once per mesh lifetime — at ingest, via `ListHalfEdgesSafe` — instead
of being re-run as a finalize stage.

### 4.7 Deliberately left array-native
`RemoveDuplicateVertices`, `RemoveDuplicateFaces`, `RemoveFacesOutside`, IO,
and the atlas/texture pipeline consume or repair raw arrays (pre-half-edge
ingest, or read-only scans). They call `SyncFaces()` on entry (read-only ones)
or keep the mutate-then-`halfMesh.Clear()` convention (repairs). No conversion;
they are not on the `Clean` hot path.

### 4.8 `Simplify` / `RemeshIsotropic` / `Smooth*` — already native
Change only their epilogue: replace the unconditional `FFaces(faces)` resync
with `InvalidateFaces()` on entry (the snapshot regenerates once at export
instead of once per stage). `Smooth*` touches positions only and keeps both
representations valid — it must call neither.

## 5. Milestones

Each lands independently, tests green (`ctest`: 507/507 today) before the next.

- **M0 — contract.** `InvalidateFaces()` + `SyncFaces()` + exact
  `ListHalfEdges()` gate + class invariant asserts + debug `ValidateHalfMesh()`.
  Wire `SyncFaces()` into every array consumer (grep for `faces` iteration).
  Behavior-neutral for existing callers.
- **M1 — bulk primitive.** `FRemoveBulk` + `Mesh::RemoveFacesHalfEdge` +
  `VRemoveUnreferenced` + `FAdd` isolated-vertex support. Unit tests: scattered
  faces, whole components, all faces, border-adjacent faces, cascade to
  isolated vertices — each validated against a from-scratch rebuild
  (`ValidateHalfMesh`).
- **M2 — repair family.** §4.1 spurious + §4.2 spikes native.
- **M3 — holes family.** §4.3 native harvest; delete the defensive entry
  rebuild in `CloseHoles`.
- **M4 — finalize family.** §4.4 degenerate faces, §4.5 unreferenced sweep,
  §4.6 `FixNonManifold` short-circuit.
- **M5 — epilogue cleanup.** §4.8: drop per-stage `FFaces` resyncs; update
  `docs/FEATURES.md` + `AGENTS.md` (the representation-authority contract is a
  standing convention from then on).

Perf gates (add to the `HALFMESH_BUILD_PERF` suite so they are repeatable):
- `Build`-count per simulated Clean pipeline == 1 (instrument with a counter).
- Truck-scale synthetic (≥5M faces): spurious+spikes+holes end-to-end must not
  regress vs. the M0 baseline, and holes-with-zero-fills must cost ~0 (today:
  6.2 s of pure rebuild).
- OpenMVS side (run by the OpenMVS agent after M5): `Clean` wall time on Truck
  ≤ 13.4 s (old CGAL path), F1 parity via the paired `--mesh-file` A/B harness.

Correctness gates:
- Metamorphic old-vs-new per converted op on the test corpus: identical removed
  counts on manifold inputs, Hausdorff ≈ 0 on surviving geometry.
- OpenMVS `Tests.exe 2` (mesh suite) green with the rebuilt library.

## 6. Risks / notes

- `FRemoveBulk` border relinking is the only genuinely subtle code (step 4 of
  §3.1); it reuses `ConnectBorders(HIndex&)` logic per affected vertex —
  validate exhaustively against rebuilds in M1 before anything builds on it.
- Bulk removals that slice a component in two are fine (no global bookkeeping
  is kept), but removals that create a non-manifold *pinch* vertex (two fans
  meeting at one vertex) must be handled the way `Build` handles them: such a
  vertex keeps one fan and the other fan's border is closed through it —
  `ConnectBorders(HIndex&)` already resolves multi-border vertices; add an
  explicit test.
- Attribute arrays (`faceTexcoords` etc.) are *dropped* on topology change,
  matching today's `FillBoundaryLoops` behavior — the UV/texture pipeline runs
  before or after, never across, a topology-mutating pass. Assert-empty at the
  entry of native mutators to make the assumption loud.
- `HALFMESH_TRIS == 1` only, as everywhere else in the OpenMVS integration.

## 7. vcpkg port policy (both repos)

**The port must never touch the library.** We own both repos: any source change
lands as a commit on `openmvs-integration` in *this* repo, never as a port
patch. (`ports/halfmesh/openmvs-integration.patch` in OpenMVS was exactly that
anti-pattern; it also went stale silently.)

- **This repo** now carries `vcpkg-overlay-ports/halfmesh/`: a portfile that
  builds straight from this working checkout (`SOURCE_PATH` = repo root,
  resolved relative to the port dir — no download, no patches, no SHA).
  Dev loop against OpenMVS:

  ```
  vcpkg install --triplet x64-windows --x-manifest-root=<openMVS> \
    --x-install-root=<openMVS>/make/vcpkg_installed \
    --overlay-ports=C:/Pro/halfmesh/vcpkg-overlay-ports \
    --overlay-ports=<openMVS>/ports --no-binarycaching
  ```

  `--no-binarycaching` is required: vcpkg hashes the portfile, not the source
  tree, so the cache would serve stale binaries after a source-only edit. The
  overlay's `port-version` is set to 999 so it always out-ranks the OpenMVS
  registry port.
- **OpenMVS repo** (after this plan ships): tag `v0.3.0` here, then point
  `ports/halfmesh/portfile.cmake` at that tag (`REF v${VERSION}` + real
  SHA512), **delete `openmvs-integration.patch`** and the `PATCHES` clause,
  reset `port-version` to 0 with the version bump.

# T7 — M5: Remesh (and Simplify setup) off `mesh.faces` + docs

**Goal:** the last `faces` dependencies inside the native family die; the
OpenMVS pipeline may then skip per-stage `FFaces`. Documentation catches up,
including the §2.2 texture marking of every public method.

**Depends on:** T4, T5, T6 (the pipeline exists); T0 conventions.

## Context (verified against 4b9b538)

- **Remesh keeps `faces` live** and this is why it is its own milestone:
  `TagCreaseEdges` reads `mesh.faces` (`src/MeshRemesh.cpp:177-217`);
  `fvSelection` is sized `faces.size()*3` (`:178`); split pass resyncs
  mid-run (`faces.clear(); m.FFaces(faces);` `:421-422`); collapse pass
  again (`:697,736`); flip (`:871-922`) and smooth (`:993+`) read
  `mesh.faces`; `RemeshData` copies the input
  (`originalMesh(StripToGeometry(_mesh))`, `:96-100`) and builds the
  projection BVH from that copy at construction — emptying `faces` on entry
  would silently empty the BVH.
- **Simplify** reads `faces` only during quadric setup
  (`src/MeshSimplify.cpp:90-150`, assumes `faces[iF]` == HE face `iF`),
  then empties it (`:150`) and `FFaces`s at exit.
- Until this task, both methods keep their entry `SyncFaces()` (T0) and
  their exit resyncs — the Clean win so far is zero extra `Build`s, not
  zero `FFaces`.

## Spec

1. **Simplify:** build `facesQuadric` from `halfMesh.F(iF)` instead of
   `faces[iF]` (the setup loop already iterates `FAdjacentHalfedges`).
   Entry `SyncFaces()` then becomes unnecessary — remove it. Exit `FFaces`
   → `SyncFaces()` (public contract).
2. **Remesh:** replace every `mesh.faces` read with `HalfMesh::F(i)` /
   half-edge walks; key `fvSelection` by `FSize()*3`; delete both mid-pass
   resyncs; `RemeshData`'s `originalMesh` must take its geometry from
   vertices + HE (an `FFaces` into the copy is fine — it IS a copy) so the
   BVH is correct under HE-only entry. Bit-identical `fvSelection` marks
   and goldens are the bar wherever the iteration order is preserved; if an
   order must change, regen goldens and say so.
3. **Pipeline opt-out:** with 1+2 done, the internal (OpenMVS-driven)
   pipeline may skip per-stage `SyncFaces` entirely — one `FFaces` in
   `ExportMesh`. Public entry points still `SyncFaces()` on exit.
   Perf gate upgrade: Build-count == 1 AND FFaces-count == 1 per simulated
   Clean (extend the T4 counter).
4. **Docs:**
   - `docs/FEATURES.md`: representation-authority contract (HE primary,
     `faces` derived; `InvalidateHalfMesh()` rule for hand-edited faces);
     per-method §2.2 texture marking table — every public processing method
     labeled `untextured-only` or `attribute-preserving (bonus)`:
     array-arm removers preserve (bonus), `Smooth*` preserves positions-only
     (bonus), Simplify/Remesh/CloseHoles/native arms = untextured-only.
   - `AGENTS.md`: the contract is a standing convention from now on.
   - `docs/HALFEDGE_PRIMARY_PLAN.md`: flip Status to shipped-through-M5.

## Acceptance

- Remesh/Simplify goldens green (or regen with justification); python green.
- Simulated Clean pipeline test: ImportMesh-equivalent → spurious → spikes →
  simplify → holes → smooth → remesh → SyncFaces: Build-count == 1,
  FFaces-count == 1, `ValidateHalfMesh` at the end.
- Truck-scale synthetic (≥5M faces) perf suite: end-to-end not regressed vs
  the M0 baseline (plan §5 gates).

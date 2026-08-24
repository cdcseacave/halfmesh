# Half-edge–primary rework — hand-off tasks

Standalone tasks decomposing `docs/HALFEDGE_PRIMARY_PLAN.md` (read it first —
it is the authority; these files carry the per-task detail). All file/line
references are **as of commit `4b9b538`** on branch `openmvs-integration`.

## Dependency order

```
T0 (contract)
  └─ T1 (parity borders)
       ├─ T2 (FRemoveBulk)  ──┬─ T4 (repair family)
       └─ T3 (FAdd ext)     ──┴─ T5 (holes family)   [T5 needs T1+T2+T3]
            T2 ───────────────── T6 (finalize family)
                 T4+T5+T6 ────── T7 (faces-free Remesh/Simplify + docs)
T8 (Build speed) — independent, optional
```

Land each task with the full suite green before starting the next.

## Shared conventions (binding for every task)

- Branch: `openmvs-integration`. Build & test:
  `cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON -DHALFMESH_BUILD_TOOLS=ON`
  → `cmake --build make -j` → `ctest --test-dir make --output-on-failure`.
  507 tests green at time of writing; tasks only add tests, never delete.
- **Texture policy (plan §2.2):** all processing methods target UNTEXTURED
  meshes. Native mutators drop face-keyed attributes via `InvalidateFaces()`
  (one-time `REPORT_WARNING` when non-empty). A method that happens to keep
  attributes consistent is a documented bonus, never a contract.
- **Representation contract (plan §2.1):** three states only — arrays-only,
  half-edge-only, both-and-consistent. Public methods `SyncFaces()` on exit.
  Array mutators end with `halfMesh.Clear()`. Half-edge mutators never Clear.
- All new writes to a vertex representative go through `SetVHalfedge`
  (`include/halfmesh/HalfMesh.h:108`) — never raw `vHalfedges[v] =`.
- `GuaranteeAlwaysEven` is a full in-place rebuild — it is NEVER an
  acceptable fix for a parity problem in these tasks.
- Every new native mutator is exercised on the test corpus followed by
  `ValidateHalfMesh()` (T0 adds it): harvest faces, rebuild scratch, validate
  structural invariants and compare semantic topology. Never compare raw
  half-edge arrays after an in-place mutation: numbering and representatives
  are intentionally not canonical outside a fresh `Build`.
- Commits: no Co-Authored-By trailer (repo convention).

## Task index

| Task | Plan §§ | Milestone | Summary |
|------|---------|-----------|---------|
| [T0](T0-contract.md) | 2.1, 2.2 | M0 | InvalidateFaces / SyncFaces / InvalidateHalfMesh / exact gate / invariants |
| [T1](T1-parity-borders.md) | 3.4 | M1a | Parity-agnostic border relink; FAdd propagates ConnectBorders failure |
| [T2](T2-fremovebulk.md) | 3.1, 3.3 | M1b | FRemoveBulk (+pinch split reporting), Mesh wrapper, VRemoveUnreferenced |
| [T3](T3-fadd-extension.md) | 3.2 | M1c | FAdd isolated-vertex + two-new-edges; disk attach via retry queue |
| [T4](T4-repair-family.md) | 4.1, 4.1b, 4.2, 4.9 | M2 | Spurious / small components / spikes native + dispatch |
| [T5](T5-holes-family.md) | 4.3 | M3 | CloseHoles / RemoveVerticesAndFill native harvest; delete entry rebuilds |
| [T6](T6-finalize-family.md) | 4.4–4.6, 4.9 | M4 | Native degenerates, unref dispatch, FixNonManifold short-circuit |
| [T7](T7-faces-free-remesh-simplify.md) | 4.8, 5(M5) | M5 | Remesh/Simplify off `mesh.faces`; docs + texture marking |
| [T8](T8-build-speed.md) | 5(M6) | M6 | Optional: faster HalfMesh::Build (kill the unordered_map) |

# Golden fixtures — frozen known-good regression snapshots (docs/TESTING.md §4)

These files are **committed** so the golden regression tests
(`tests/golden/GoldenDiffTest.cpp`) run in CI in the **default build** without
any external dependency.

## What's here

For each `(input mesh × op)` spec in `tests/golden/GoldenSpecs.h`, two files
frozen as known-good outputs:

| file | content |
|------|---------|
| `<mesh>__<op>.ply`  | frozen result mesh — binary PLY, **lossless** float positions + integer faces |
| `<mesh>__<op>.json` | scalar metrics: counts, op count, surface area, bbox, Hausdorff-to-input, edge-length stats |

Current fixtures (14 — small corpus only, no `LargeMesh`). The `CompareMode`
column mirrors `tests/golden/GoldenSpecs.h` and drives how strict the check is
(see the tolerance section below):

| fixture (`<mesh>__<op>`) | op | CompareMode |
|--------------------------|----|-------------|
| `DirtyDuplicateFaces__RemoveDuplicateFaces`           | RemoveDuplicateFaces       | `CANONICAL_EXACT` |
| `DirtyDegenerateFaces__RemoveDegenerateFaces`         | RemoveDegenerateFaces      | `CANONICAL_EXACT` |
| `DirtyUnreferencedVertices__RemoveUnreferencedVertices` | RemoveUnreferencedVertices | `CANONICAL_EXACT` |
| `DirtyManyComponents__RemoveSmallComponents`          | RemoveSmallComponents      | `CANONICAL_EXACT` |
| `DirtyBowTie__FixNonManifold`                         | FixNonManifold             | `CANONICAL_EXACT` |
| `GridPlane__Simplify`                                 | Simplify                   | `CANONICAL_EXACT` |
| `UVSphere__Simplify`                                  | Simplify                   | `TOLERANT` |
| `Torus__Simplify`                                     | Simplify                   | `TOLERANT` |
| `UVSphere__SimplifyFast`                              | Simplify (aggressive)      | `TOLERANT` |
| `UVSphere__SimplifyMinEdge`                           | Simplify (min-edge)        | `TOLERANT` |
| `UVSphere__RemeshIsotropic`                           | RemeshIsotropic            | `TOLERANT` |
| `UVSphere__SmoothTaubin`                              | SmoothTaubin (5 iters)     | `CANONICAL_EXACT` |
| `UVSphere__SmoothHCLaplacian`                         | SmoothHCLaplacian (5 iters) | `CANONICAL_EXACT` |
| `OpenCylinder__SmoothTaubin`                          | SmoothTaubin (5 iters, border-curve rule) | `CANONICAL_EXACT` |

The five combinatorial repair ops are exact. Simplify is exact **only** on the
flat `GridPlane` (rank-1 quadrics take the FP-order-insensitive fallback paths);
on curved meshes QEM collapse ordering depends on float LSBs that shift between
build configs, so those four are tolerant. `RemeshIsotropic` is tolerant for the
same build-flag reason (see the classification comment in `GoldenSpecs.h`).

`CloseHoles` is intentionally **absent**: its Liepa fill is covered directly by
`MeshHolesTest.cpp` (topology + fairness assertions), which is a stronger check
than a frozen snapshot of a hole-filling result.

The three smoothing fixtures were added 2026-08 when the vcglib crosscheck was
removed along with the `vcgCompatible` mode: they freeze the current
(unit-pinned) smoother behavior against drift, since no external ground truth
remains in the tree. Deterministic vertex order, but plain Eigen float
arithmetic — exact only under default build flags, hence listed in the CI
`GOLDEN_EXACT_EXCLUDE` set like the other `CANONICAL_EXACT` fixtures.

## How the regression test compares (tolerance, §5)

- **`CANONICAL_EXACT`** (the five repair ops + `GridPlane__Simplify`): exact
  canonical mesh equality (up to relabeling), exact op count, surface area
  within 1e-4 relative, bbox within 1e-4 relative.
- **`TOLERANT`** (the four curved Simplify fixtures + `RemeshIsotropic`):
  element counts within 5% relative, symmetric Hausdorff(result, golden) ≤ 1e-2
  of the bbox diagonal, and the real net — quality vs the **input**:
  `hausdorff_to_input ≤ max(1.5 × baseline, 1e-3 × diag)`. These ops make
  split/collapse decisions on float edge-length comparisons; result counts can
  drift a few percent across compiler versions. See the test comment.
- **Remesh edge-length ratchet** (`RemeshIsotropic` only): because uniform edge
  length is the op's contract and the Hausdorff net cannot see isotropy loss,
  `edge_len_mean` is additionally pinned within ±10% and
  `edge_len_min`/`edge_len_max` within ±25% of the frozen baseline
  (baseline-relative, named constants — see `GoldenDiffTest.cpp`).

## Fixture policy

The frozen fixtures represent the behaviour of the algorithm as it stands, **not
the other way around**: genuine algorithmic improvements no longer have to be
bit-identical to an old fixture — the fixtures follow the algorithm. Correct /
optimized code takes priority over byte-stability. What does **not** change:
every regeneration must be justified, reviewed, and recorded, and the
determinism invariants (run-twice identical, serial == parallel) are never
relaxed — they are core product guarantees, not fixture locks.

Concretely: when a behaviour change legitimately shifts a fixture, regenerate it
and **paste the regen delta report** (printed by the regen run, one
`[golden-regen]` line per metric) into the commit message so the change is
quantified, and add a row to the regen log below.

## Regenerating (intentional, reviewed)

The regen path is opt-in via the `GOLDEN_REGEN` env var. `GOLDEN_REGEN` may be
`1`/`all` (regenerate every spec) or a substring that matches the fixture stem
(e.g. `RemeshIsotropic`). Before overwriting, the run prints a delta report
comparing the previous `.json` against the new metrics, flagging a
`QUALITY REGRESSION` if `hausdorff_to_input` rose by more than 10%:

```sh
# build the test (tree already configured under make/)
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON
cmake --build make -j --target golden_diff_test

# regenerate the matching fixture(s); review the printed [golden-regen] delta
GOLDEN_REGEN=RemeshIsotropic ./make/tests/golden_diff_test --gtest_filter='*Regenerate*'

# stage the regenerated .ply + .json TOGETHER with the code change that caused
# it, and paste the delta report into the commit message
git add tests/data/golden/UVSphere__RemeshIsotropic.ply \
        tests/data/golden/UVSphere__RemeshIsotropic.json
```

If the rewrite is byte-identical (same algorithm, no drift) `git status
tests/data/golden` stays clean and there is nothing to commit — that is the
expected result when no behaviour changed.

## Regen log

Add a row here (and the delta report to the commit message) whenever a fixture is
regenerated.

| commit | date | fixture(s) | reason |
|--------|------|------------|--------|
| — | — | all (initial) | initial fixtures |

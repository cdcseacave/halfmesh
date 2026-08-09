# AGENTS.md — tests/golden (golden-fixture regression)

Self-contained regression layer: each op is checked against committed, frozen **known-good
output fixtures**, so no external tree is needed at run time.

## Pieces
- `GoldenSpecs.h` — the single source of truth: `GoldenCorpus()` returns a vector of
  `GoldenSpec { meshName, opName, CompareMode, makeInput(), runOp() }`. `CompareMode`
  is `CANONICAL_EXACT` (deterministic ops → exact canonical mesh equality + counts) or
  `TOLERANT` (float/order-sensitive, e.g. Remesh → element counts ±ε + Hausdorff≈0).
- `GoldenDiffTest.cpp` — parameterized over `GoldenCorpus()`: rebuilds the input, runs the
  op, and compares to the frozen fixture (`MatchesFrozenGolden`); also an `OpDeterministic`
  re-run check. This runs in the DEFAULT build.
- `GoldenIO.{h,cpp}` — load/save fixture `.ply` (binary, lossless) + `.json` metrics sidecar,
  and the `tests/data/golden/` path helper + metrics computation.
- Fixtures live in `tests/data/golden/<mesh>__<op>.{ply,json}` (committed).

Covered ops include the repair ops (RemoveDuplicate/Degenerate/Unreferenced,
RemoveSmallComponents, FixNonManifold), Simplify (UVSphere, Torus genus-1, GridPlane
open/boundary, fast-mode, min-edge-length), and RemeshIsotropic.

## Adding a fixture
1. Add a `GoldenSpec` to `GoldenCorpus()` in `GoldenSpecs.h` (input generator from
   `hmtest::corpus` + the op lambda + a `CompareMode`).
2. Generate the fixture: run the golden test with `GOLDEN_REGEN=<substring>|1|all`
   (`RegenerateFixtureIfRequested`), which writes the `.ply` + `.json` and prints a
   `[golden-regen]` delta report.
3. Review the delta report and the fixture diff, then commit both.
4. `GoldenDiffTest` validates against the fixture from then on.

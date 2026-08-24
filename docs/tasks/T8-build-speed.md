# T8 — M6 (optional backlog): faster `HalfMesh::Build`

**Goal:** shrink the one `Build` the architecture cannot delete (ingest; two
on dirty ingest). Independent of every other task.

## Context (verified against 4b9b538)

`Build` (`src/HalfMesh.cpp:38+`) pairs twin half-edges with
`std::unordered_map<uint64_t, HIndex> createdHalfedges` (`:67`) over ~3F
directed edges — measured ≈ 3 s per rebuild on a 6.1M-face mesh; the map is
the cost center (hashing, allocation, cache misses).

## Spec

Replace the map with one of (benchmark both, keep the winner):

1. **Sort-based pairing:** emit one record per directed edge
   `(packed min:max key, corner id)`, sort (radix or `std::sort` on u64),
   pair adjacent equal keys. Two passes, no per-node allocation.
2. **Flat open-addressing map** (linear probing, power-of-two capacity
   reserved at 3F) keyed by the packed u64.

Constraints:

- The half-edge SLOT ASSIGNMENT must remain byte-identical to today's
  (pairs created in first-encounter face order; `HeTwin` is the odd/even
  pair sibling; `vHalfedges` all-even canonical form; `alwaysEven == true`
  post-Build). If the chosen approach cannot reproduce the exact numbering,
  every structural golden/invariant test and `tests/InternalsTest.cpp` must
  be audited — prefer the approach that keeps numbering identical (the flat
  map trivially does; sort-based needs a stable first-encounter pass).
- All rejection paths must behave identically: self-edges, duplicate
  directed edges, opposite-orientation duplicates, and the bow-tie
  single-fan check (`:134-156`) are untouched.
- `Build` still returns false (never throws/crashes) on non-manifold input.

## Acceptance

- Full suite green with NO golden changes.
- Perf test in `HALFMESH_BUILD_PERF`: ≥2× Build speedup on a ≥5M-face
  synthetic; report the measured number in the commit message.
- Peak-memory not worse than the unordered_map baseline (report).

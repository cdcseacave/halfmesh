# T8 — M6: faster `HalfMesh::Build`

**Status:** completed 2026-08-24. The flat open-addressing candidate won and
replaced `std::unordered_map`.

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

## Implemented approach

`HalfMesh::BuildImpl` now uses two preallocated arrays as a power-of-two flat
hash table: packed undirected `uint64_t` keys and even `HIndex` values. A
MurmurHash3 finalizer distributes the packed vertex indices; collisions use
linear probing. Key zero is the empty sentinel, which is safe because only the
rejected self-edge `0 -> 0` can produce it.

The existing face/corner walk still owns all half-edge creation. The table is
never iterated, so the first encounter creates each even/odd pair exactly where
the old map did. The rejection checks, border connection, and single-fan
validation remain in the same control flow. `HalfMeshTest` pins all five output
arrays for a tetrahedron and exercises a third edge use after both valid twin
directions have been occupied.

`PerfHarness.Build_5m` is the permanent acceptance workload. It asserts a
successful build of at least five million generated faces and records the
isolated Build duration.

## Measured results

Benchmark environment: Release, MSVC 14.51.36231, `x64-windows`, 13th Gen Intel
Core i7-13700KF (16 cores / 24 logical processors), 31.8 GiB RAM. The input was
`LargeMesh(5'000'000)`, which generated 5,003,552 faces. Each sample used a
fresh process. Time covers only `HalfMesh::Build`; peak working set was sampled
from the whole process and therefore includes the input mesh, output HalfMesh,
and temporary pairing storage.

| Pairing implementation | Samples | Median Build | Speedup | Peak working set |
| --- | ---: | ---: | ---: | ---: |
| `std::unordered_map` baseline (`ec1df68`) | 3 | 1.969 s | 1.00x | 904.2 MiB |
| `std::sort` records + face-order replay | 5 | 0.718 s | 2.74x | 423.1 MiB |
| Flat map, retained | 5 | 0.608 s | **3.24x** | 495.8 MiB |

Retained-flat samples were 0.608, 0.618, 0.606, 0.594, and 0.610 s; a final
post-selection run measured 0.607 s. The sort samples were 0.718, 0.717, 0.717,
0.731, and 0.719 s. Baseline samples were 1.967, 1.969, and 1.983 s.

The retained table reduced measured peak working set by 408.4 MiB (45.2%) from
the baseline. At this input size its capacity is 16,777,216 slots and its two
arrays reserve 192 MiB. It costs 72.7 MiB more process peak than the sort
candidate, but is 15.2% faster.

## Costs and tradeoffs

- **Memory:** 12 bytes of table storage per power-of-two slot, alive alongside
  the output arrays. This is substantially below the measured node-map peak,
  but above the sort candidate's staged peak. Worst-case table capacity is the
  next power of two strictly above `3 * face_count` so even disconnected
  triangles cannot fill it.
- **Code complexity:** the production change is 33 added / 24 removed lines
  (9 net), replacing a standard container with explicit capacity, sentinel,
  hash-mixing, and probing logic. The invariants are local to `BuildImpl` and
  documented next to the table.
- **CPU:** expected linear average time with good hash distribution; unlike
  sorting, there is no `O(F log F)` ordering pass. Pathological clustering is
  bounded only by table capacity, as with conventional open addressing.
- **Compatibility:** no public API, numbering, rejection behavior, or golden
  fixture changed. The speedup is machine-dependent; the structural and ≥5M
  workload tests are permanent, while the 3.24x comparison is the recorded
  measurement for this environment.

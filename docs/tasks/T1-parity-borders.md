# T1 — M1a: parity-agnostic border relink + FAdd failure propagation

**Goal:** remove the freshly-built-HE (all-even) assumption from the border
relinking machinery and make `FAdd` fail cleanly instead of corrupting. This
is the prerequisite that makes deleting the defensive rebuilds (T5) sound.

**Depends on:** T0.

## Context (verified against 4b9b538)

- `ConnectBorders(HIndex&)` (`src/HalfMesh.cpp:215-246`) takes the vertex
  representative BY REFERENCE (it repoints `vHalfedges[v]` to the
  boundary-canonical half-edge — that is a feature to keep) but hard-fails on
  odd representatives: `if ((iHeStart & 1) != 0) return false;` (line ~231).
  The batch `ConnectBorders()` (`:248-265`) rejects any odd entry the same
  way.
- The all-even form is guaranteed only by a fresh `Build`. `EFlip`/`ESplit`
  legally leave odd representatives on interior vertices — see the invariant
  note `include/halfmesh/HalfMesh.h:73-102` and the `alwaysEven` flag.
- `FAdd`'s closing border-rewire loop (`src/HalfMesh.cpp:577-583`) calls
  `ConnectBorders(iHe)` and IGNORES the return value: on failure it still
  returns a valid face index with border connectivity un-rewired — silent
  corruption. Today unreachable only because every `FAdd` call site
  (`TriangulateHole`, `src/HalfMesh.cpp:290+`) runs on a freshly built HE —
  exactly the defensive rebuilds the plan deletes in T5.
- `FAdd` also writes `vHalfedges[idxTail] = iHe;` directly (`:556`),
  bypassing `SetVHalfedge`.
- The LOAD-BEARING part of the invariant (must survive this task): for a
  boundary vertex the representative is the interior-side half-edge of one of
  its boundary edges, so `VIsBoundary` (`HalfMesh.h:191`), `EnumerateHoles`
  (`src/HalfMesh.cpp:267-288`), and boundary-loop iteration work.

## Spec

1. Make the per-vertex border-relink core **parity-agnostic**: identify
   boundary status via `heFaces` (`EHeIsBoundary`), never via `(iHe & 1)`.
   Keep the `guardLimit` hang protection. Route every representative write
   through `SetVHalfedge` (this keeps the `alwaysEven` bookkeeping honest —
   an odd representative stored during relink must clear the flag).
2. The strict all-even variant may remain for the `Build()`-internal path
   (Build genuinely guarantees evenness there), but the primitive reachable
   from `FAdd` — and later from `FRemoveBulk` (T2) — must accept odd
   representatives. Decide: either one parity-agnostic implementation for
   both (preferred if goldens stay identical), or a documented split.
3. `FAdd`: check the `ConnectBorders` result. On failure, back out the
   partially-inserted face (new pair slots, `heFaces`/`heNexts` writes,
   `fHalfedges` entry) and return `NO_ID` — the caller contract is already
   "NO_ID = cannot attach" (TriangulateHole's retry queue handles it). If
   after (1) you can PROVE failure unreachable, replace with an ASSERT and
   say so in a comment.
4. Fix the raw `vHalfedges[idxTail] =` write in `FAdd` → `SetVHalfedge`.

## Pitfalls

- Do not "fix" this with `GuaranteeAlwaysEven` — it is a full in-place
  rebuild (`src/HalfMesh.cpp:201-212`), the exact cost T5 deletes.
- Fresh-Build outputs must stay byte-identical (existing golden and
  invariant tests) — the parity-agnostic core must produce the same result
  as the old code when everything happens to be even.
- An odd representative on a BOUNDARY vertex: the HalfMesh.h note claims the
  canonical (even) form is always maintained for boundary vertices. Write a
  test that tries to violate it via `ERemove`/`EFlip` sequences near a
  boundary; if it is genuinely unreachable, assert it and document — that
  assert protects T2's repointing logic.

## Acceptance

- Full suite green; goldens unchanged.
- New tests (extend `tests/HalfMeshTest.cpp` / `HalfMeshInvariantsTest.cpp`):
  - `FAdd` on a HE mutated by `ESplit`+`EFlip` until `alwaysEven == false`
    (never rebuilt), then `ValidateHalfMesh` + semantic comparison against a
    from-scratch rebuild (raw arrays cannot match after in-place mutation).
  - `FAdd` on a HE mutated by `ERemove` (decimation-style), same validation.
  - `EnumerateHoles` correct after the above mutations.
  - A rejected `FAdd` leaves the structure bit-identical to before the call.

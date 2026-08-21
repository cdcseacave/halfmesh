# Unwrap Atlas Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the 1M-face UV unwrap from 3h45m to ≤10 min by de-quadraticizing atlas packing, making the fit-to-resolution loop analytic, and re-merging chart fragments after flip repair — plus an opt-in remesh lever in radiance's postprocess stage.

**Architecture:** All halfmesh C++ changes live in `src/AtlasPacking.cpp` (two-tier pack pass, analytic fit shrink) and `src/AtlasCharting.cpp` (+1 param in `include/halfmesh/Parametrize.h`, +1 field in `include/halfmesh/AtlasCharting.h`), on the `worktree-python-bindings` branch. The Python wheel is rebuilt from the same branch; radiance needs zero API changes. The radiance repo gains one opt-in postprocess flag and a measurement study.

**Tech Stack:** C++20 / Eigen / gtest / CMake+vcpkg (halfmesh); Python 3.11 / numpy / pytest (radiance); conda env `radiance`.

**Spec:** `docs/superpowers/specs/2026-08-21-unwrap-atlas-speedup-design.md` (same repo/branch as this plan)

## Global Constraints

- All halfmesh changes on branch `worktree-python-bindings`, worktree `/home/wii/halfmesh/.claude/worktrees/python-bindings` (per project owner — NO new branch).
- All radiance changes on branch `halfmesh-integration`, worktree `/home/wii/radiance-halfmesh`.
- Unwrap stays geometry-preserving: weld + degenerate/manifold repairs only — never remesh inside unwrap.
- halfmesh code style: tabs, `ColumnLimit: 0` (repo `.clang-format`; run `clang-format -i` on touched files), `ASSERT(expr)` only (no `ASSERT_EQ` outside gtest), `#pragma once`, MIT header block on new files, namespace `halfmesh::`.
- halfmesh build/test (run from the worktree root): `cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON -DHALFMESH_BUILD_TOOLS=ON && cmake --build make -j8 && ctest --test-dir make --output-on-failure`. `$VCPKG_ROOT=/home/wii/vcpkg` must be exported.
- radiance tests: `cd /home/wii/radiance-halfmesh && /home/wii/miniconda3/envs/radiance/bin/python -m pytest tests/unit -x -q` (MUST run with cwd=/home/wii/radiance-halfmesh — the editable install resolves to the sibling checkout otherwise).
- Reference meshes (already on disk, do not regenerate): 320k-face `/home/wii/bench_runs/unwrap_spike/post200k/mesh.ply`; 1M-face `/home/wii/bench_runs/task4_validation/post/mesh.ply`.
- Baseline numbers to beat (installed wheel, before this plan): 1M @ padding=4 → 13,473 s / 78.8 % occupancy / 161,627 charts; 320k @ padding=4 → 2,864 s / 77.2 % / 112,047 charts; 320k @ padding=2 → 353 s / 82.0 %.
- Perf gates (release wheel, `resolution=4096, padding=4`): 1M ≤ 600 s, occupancy ≥ 0.75, pages == 1, charts ≤ 161,627; 320k ≤ 120 s.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Two-tier pack pass (spec §3, Fix 1a)

**Files:**
- Modify: `src/AtlasPacking.cpp` (function `PackRects`, lines ~291–371)
- Test: `tests/AtlasTest.cpp`

**Interfaces:**
- Consumes: existing `SkylineBin::Insert(float rw, float rh, bool allowRotation, Rect& out)`, `struct ChartRect {float w,h,uvMinX,uvMinY,uvMaxX,uvMaxY; bool degenerate;}`, `struct Placement {float x,y; bool rotated; unsigned page;}` — all TU-local in `src/AtlasPacking.cpp`.
- Produces: `PackRects` with identical signature and placement semantics (`placements[ci] = {x+pad, y+pad, rotated, page}`, `packedArea` = Σ placed padded rect areas). Task 2 modifies the caller (`PackAtlas` §1.5) only.

- [ ] **Step 1: Write the failing tests**

Add to `tests/AtlasTest.cpp` (after `BuildSyntheticCharts`, reusing the existing `ChartBBoxes` / `BoundingRectsDisjoint` helpers already in the file):

```cpp
// Build `n` UNIFORM tiny square charts (side `side` world units) plus `nBig`
// large ones (side 40·side). Mimics the production regime: a huge tail of
// near-identical tiny charts and a small head of large ones.
static void BuildMixedCharts(Mesh& mesh,
                             std::vector<unsigned>& faceChart,
                             unsigned& numCharts,
                             unsigned n, unsigned nBig, float side)
{
	numCharts = n + nBig;
	faceChart.clear();
	mesh.vertices.clear();
	mesh.faces.clear();
	mesh.faceTexcoords.clear();
	float offset = 0.f;
	for (unsigned c = 0; c < numCharts; ++c) {
		const float s = (c < nBig) ? 40.f * side : side;
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({offset, 0.f, 0.f});
		mesh.vertices.push_back({offset + s, 0.f, 0.f});
		mesh.vertices.push_back({offset + s, s, 0.f});
		mesh.vertices.push_back({offset, s, 0.f});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 1.f});
		offset += s + 2.f;
	}
}

// ---------------------------------------------------------------------------
// Two-tier packing: a production-shaped input (2000 tiny + 4 large charts)
// must place every chart overlap-free at sane occupancy — and fast (the old
// full-scan skyline is quadratic in chart count; the shelf tier is what makes
// this test complete in milliseconds instead of seconds).
// ---------------------------------------------------------------------------
TEST(PackAtlas, TwoTierManyTinyChartsDisjointAndDense)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildMixedCharts(mesh, faceChart, numCharts, 2000u, 4u, 1.f);

	AtlasParams params;
	params.resolution = 1024;
	params.padding = 2;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	ASSERT_EQ(res.chartPage.size(), numCharts);
	for (unsigned c = 0; c < numCharts; ++c)
		EXPECT_LT(res.chartPage[c], res.numPages);
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x()));
		ASSERT_TRUE(std::isfinite(uv.y()));
		EXPECT_GE(uv.x(), 0.f - 1e-4f);
		EXPECT_LE(uv.x(), 1.f + 1e-4f);
		EXPECT_GE(uv.y(), 0.f - 1e-4f);
		EXPECT_LE(uv.y(), 1.f + 1e-4f);
	}
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts)) << "two charts overlap in the atlas";
	EXPECT_GT(res.occupancy, 0.4f) << "shelf tier wastes too much: " << res.occupancy;

	std::printf("[PackAtlas] TwoTier: pages=%u occupancy=%.3f dims=%ux%u\n",
	            res.numPages, res.occupancy, res.width, res.height);
}

// ---------------------------------------------------------------------------
// Shelf-tier rotation: tall skinny tiny charts must be laid down (rotated) in
// shelves without breaking the winding-preserving 90° UV-rewrite convention —
// disjointness + in-bounds UVs + positive UV area per face prove it.
// ---------------------------------------------------------------------------
TEST(PackAtlas, TwoTierShelfRotationKeepsWindingAndBounds)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	mesh.vertices.clear();
	// 4 big charts absorb the auto density so the 500 skinny ones land UNDER
	// the pageW/32 tier threshold (in the shelf tier) — without them the
	// skinny charts normalize to ~60 texels tall and take the skyline path.
	const unsigned nBig = 4u, n = nBig + 500u;
	float off = 0.f;
	for (unsigned c = 0; c < n; ++c) {
		const float w = (c < nBig) ? 200.f : 1.f;
		const float h = (c < nBig) ? 200.f : 6.f; // tall & skinny → shelf tier wants them rotated
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({off, 0.f, 0.f});
		mesh.vertices.push_back({off + w, 0.f, 0.f});
		mesh.vertices.push_back({off + w, h, 0.f});
		mesh.vertices.push_back({off, h, 0.f});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, h});
		off += 210.f;
	}
	AtlasParams params;
	params.resolution = 512;
	params.padding = 2;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, n, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, n, params);

	const auto rects = ChartBBoxes(mesh, faceChart, n, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, n));
	// Winding preserved: every face keeps POSITIVE signed UV area (a mirrored
	// placement would flip the sign).
	for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
		const float a2 = SignedDoubleArea2D(mesh.faceTexcoords[f * 3 + 0],
		                                    mesh.faceTexcoords[f * 3 + 1],
		                                    mesh.faceTexcoords[f * 3 + 2]);
		EXPECT_GT(a2, 0.f) << "face " << f << " mirrored by shelf rotation";
	}
}
```

- [ ] **Step 2: Build and run the new tests to verify they fail (or expose the perf hole)**

```bash
cd /home/wii/halfmesh/.claude/worktrees/python-bindings
export VCPKG_ROOT=/home/wii/vcpkg
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON && cmake --build make -j8
ctest --test-dir make -R "PackAtlas.TwoTier" --output-on-failure
```

Expected: both tests COMPILE and PASS functionally on the old packer (it is correct, just slow) — verify instead that `TwoTierManyTinyChartsDisjointAndDense` takes noticeably long (the ctest timing line; the old full-scan packer needs several seconds for 2000 charts) and record that number for Step 4's comparison. If either test FAILS on the old code, stop and re-check the fixture.

- [ ] **Step 3: Replace `PackRects` with the two-tier implementation**

In `src/AtlasPacking.cpp`, replace the body of `PackRects` (keep the signature and the sort + page-dims preamble; replace everything from `placements.assign(...)` down) with:

```cpp
	placements.assign(numCharts, Placement{});
	std::vector<SkylineBin> bins;
	bins.emplace_back(static_cast<float>(pageW), static_cast<float>(pageH));
	float packedArea = 0.f;

	// ------------------------------------------------------------------
	// Two-tier split. Insert() is a full min-waste scan — O(#segments) per
	// PROBE and every rect probes every segment — so packing 100k+ tiny
	// charts through it is quadratic in chart count (measured: 78% of a
	// 3h45m production unwrap). Tiny rects don't need min-waste placement:
	// shelve them. Threshold: a padded long side under pageW/32 goes to the
	// shelf tier; at most ~(32)² rects per page can exceed that, so the
	// skyline tier stays small and keeps its quality where it matters.
	// ------------------------------------------------------------------
	const float tierThreshold = static_cast<float>(pageW) / 32.f;
	struct TailRect
	{
		unsigned ci;
		float rw, rh;
		bool rot;
	};
	std::vector<unsigned> head;
	std::vector<TailRect> tail;
	for (unsigned ci : order) {
		const ChartRect& cr = crects[ci];
		if (cr.w <= 0.f || cr.h <= 0.f) {
			// After the ≥1-texel clamp this is unreachable; keep it as a safety net.
			placements[ci] = {0.f, 0.f, false, 0u};
			continue;
		}
		float rw = cr.w + 2.f * static_cast<float>(pad);
		float rh = cr.h + 2.f * static_cast<float>(pad);
		if (std::max(rw, rh) >= tierThreshold) {
			head.push_back(ci);
			continue;
		}
		// Shelf tier: pre-decide the rotation (lowest profile: height ≤ width),
		// same winding-preserving 90° convention as the skyline placements.
		bool rot = false;
		if (params.allowRotation && rh > rw) {
			std::swap(rw, rh);
			rot = true;
		}
		tail.push_back({ci, rw, rh, rot});
	}

	// Head: unchanged skyline min-waste first-fit-decreasing over growing bins.
	for (unsigned ci : head) {
		const ChartRect& cr = crects[ci];
		const float rw = cr.w + 2.f * static_cast<float>(pad);
		const float rh = cr.h + 2.f * static_cast<float>(pad);
		Rect placed;
		unsigned page = 0;
		bool ok = false;
		for (unsigned p = 0; p < static_cast<unsigned>(bins.size()); ++p) {
			if (bins[p].Insert(rw, rh, params.allowRotation, placed)) {
				page = p;
				ok = true;
				break;
			}
		}
		if (!ok) {
			page = static_cast<unsigned>(bins.size());
			bins.emplace_back(static_cast<float>(pageW), static_cast<float>(pageH));
			if (!bins[page].Insert(rw, rh, params.allowRotation, placed)) {
				// Unreachable: grow-to-fit above guarantees a fresh page fits every chart.
				ASSERT(false && "PackAtlas: chart still does not fit a fresh page after "
				                "grow-to-fit — invariant violated");
				placed = {0.f, 0.f, rw, rh, false};
			}
		}
		placements[ci] = {placed.x + static_cast<float>(pad),
		                  placed.y + static_cast<float>(pad),
		                  placed.rotated, page};
		packedArea += placed.w * placed.h;
	}

	// Tail: shelf rows, height-descending so each shelf's FIRST rect is its
	// tallest and everything after it fits the shelf height. Each shelf is
	// allocated THROUGH the skyline as one wide pseudo-rect (shelves nestle
	// into the contour the head left; multi-page logic is untouched); inside a
	// shelf, placement is O(1) per rect. Unfilled shelf remainder counts as
	// waste in `occupancy` (packedArea sums only real rect areas) — honest,
	// and small under the height sort.
	std::sort(tail.begin(), tail.end(), [](const TailRect& a, const TailRect& b) {
		return a.rh > b.rh;
	});
	struct Shelf
	{
		float x = 0.f, y = 0.f, w = 0.f, h = 0.f, cursor = 0.f;
		unsigned page = 0;
		bool open = false;
	};
	Shelf shelf;
	const auto openShelf = [&](float rw, float rh) {
		// Prefer wide shelves; fall back to narrower ones that still slot into
		// leftover contour gaps; a fresh page always fits (grow-to-fit).
		for (const int denom : {1, 2, 4, 8}) {
			const float sw = std::max(rw, static_cast<float>(pageW) / static_cast<float>(denom));
			for (unsigned p = 0; p < static_cast<unsigned>(bins.size()); ++p) {
				Rect r;
				if (bins[p].Insert(sw, rh, false, r)) {
					shelf = {r.x, r.y, sw, rh, r.x, p, true};
					return;
				}
			}
		}
		const unsigned page = static_cast<unsigned>(bins.size());
		bins.emplace_back(static_cast<float>(pageW), static_cast<float>(pageH));
		Rect r;
		const float sw = std::max(rw, static_cast<float>(pageW));
		if (!bins[page].Insert(sw, rh, false, r)) {
			ASSERT(false && "PackAtlas: shelf does not fit a fresh page — invariant violated");
			r = {0.f, 0.f, sw, rh, false};
		}
		shelf = {r.x, r.y, sw, rh, r.x, page, true};
	};
	for (const TailRect& t : tail) {
		if (!shelf.open || shelf.cursor + t.rw > shelf.x + shelf.w + 1e-3f || t.rh > shelf.h + 1e-3f)
			openShelf(t.rw, t.rh);
		placements[t.ci] = {shelf.cursor + static_cast<float>(pad),
		                    shelf.y + static_cast<float>(pad), t.rot, shelf.page};
		shelf.cursor += t.rw;
		packedArea += t.rw * t.rh;
	}

	outPages = static_cast<unsigned>(bins.size());
	outPw = pageW;
	outPh = pageH;
	outPackedArea = packedArea;
```

Note: the UV-rewrite step 5 of `PackAtlas` already handles `pl.rotated` for any placement — the shelf tier only has to set the flag; do NOT touch step 5.

- [ ] **Step 4: Build, run the full atlas suite, verify pass + speed**

```bash
cmake --build make -j8 && ctest --test-dir make --output-on-failure
```

Expected: ALL tests pass (the pre-existing `PackAtlas.*`, `GenerateAtlas.*`, `NormalizeChartDensity.*`, `Parametrize.*` included — the two-tier change may legitimately alter layouts, and those tests assert properties, not layouts). `TwoTierManyTinyChartsDisjointAndDense` now runs in well under a second (compare Step 2's recorded time and mention both numbers in the commit message).

- [ ] **Step 5: Format and commit**

```bash
clang-format -i src/AtlasPacking.cpp tests/AtlasTest.cpp
cmake --build make -j8 && ctest --test-dir make -R PackAtlas --output-on-failure
git add src/AtlasPacking.cpp tests/AtlasTest.cpp
git commit -m "perf(atlas): two-tier pack pass — shelf rows for the tiny-chart tail

Full-scan min-waste skyline is O(N*S); at 100k+ charts (fragmented MVS
meshes) packing dominated the whole unwrap (78% of samples, 3h45m at 1M
faces). Rects with padded long side < pageW/32 now pack into shelf rows
allocated through the skyline; the min-waste head keeps its quality.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Analytic fit-to-resolution shrink + `fitAttempts` (spec §4, Fix 1b)

**Files:**
- Modify: `src/AtlasPacking.cpp` (function `PackAtlas`, the §1.5 `fitToResolution` block, lines ~515–601)
- Modify: `include/halfmesh/AtlasCharting.h` (struct `AtlasResult`, ~line 198)
- Test: `tests/AtlasTest.cpp`

**Interfaces:**
- Consumes: Task 1's `PackRects` (unchanged signature).
- Produces: `AtlasResult::fitAttempts` (`unsigned`, 0 when `fitToResolution` is off, ≥1 when on) — Task 5 exposes it through the Python binding as `meta["fit_attempts"]`.

- [ ] **Step 1: Add the result field**

In `include/halfmesh/AtlasCharting.h`, add to `struct AtlasResult` after `occupancy`:

```cpp
	// fit-to-resolution probe packs performed (0 = fitToResolution off). A
	// converging fit takes 1-2; values near the internal cap (8) mean the
	// shrink loop struggled — a diagnosability hook for huge chart counts.
	unsigned fitAttempts = 0;
```

- [ ] **Step 2: Write the failing test**

Add to `tests/AtlasTest.cpp`:

```cpp
// ---------------------------------------------------------------------------
// Analytic fit shrink: on a padding-dominated tiny-chart input the fit loop
// must converge in ≤3 probe packs (the old blind ×0.95 ladder burned up to 8 —
// the measured 8× wall-time multiplier at production padding=4).
// ---------------------------------------------------------------------------
TEST(PackAtlas, FitToResolutionConvergesInFewAttempts)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildMixedCharts(mesh, faceChart, numCharts, 2000u, 4u, 1.f);

	AtlasParams params;
	params.resolution = 512;
	params.padding = 4; // padding-dominated: the production pathology
	params.allowRotation = true;
	params.fitToResolution = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	std::printf("[PackAtlas] FitConverges: attempts=%u pages=%u occupancy=%.3f\n",
	            res.fitAttempts, res.numPages, res.occupancy);
	EXPECT_EQ(res.numPages, 1u);
	EXPECT_GE(res.fitAttempts, 1u);
	EXPECT_LE(res.fitAttempts, 3u) << "fit loop is still ladder-stepping";
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts));
}
```

- [ ] **Step 3: Run it to verify it fails**

```bash
cmake --build make -j8 && ctest --test-dir make -R FitToResolutionConverges --output-on-failure
```

Expected: FAIL — `res.fitAttempts` is 0 (field never set). If it PASSES, the field default is masking the assertion — re-check Step 2's `EXPECT_GE(res.fitAttempts, 1u)` is present.

- [ ] **Step 4: Implement the analytic shrink**

In `PackAtlas` §1.5, replace the probe loop (the `for (int attempt = 0; attempt < 8; ++attempt)` block) with:

```cpp
				double kf = k;
				std::vector<ChartRect> trial(crects);
				std::vector<Placement> probe;
				unsigned probePages = 0, probePw = 0, probePh = 0;
				float probeArea = 0.f;
				unsigned attempts = 0;
				for (int attempt = 0; attempt < 8; ++attempt) {
					++attempts;
					const float kk = static_cast<float>(kf);
					for (unsigned c = 0; c < numCharts; ++c) {
						if (crects[c].degenerate)
							continue;
						trial[c].w = crects[c].w * kk;
						trial[c].h = crects[c].h * kk;
					}
					PackRects(trial, numCharts, params, pad, probe, probePages, probePw, probePh, probeArea);
					if (probePages <= 1 && probePw <= params.resolution && probePh <= params.resolution)
						break;
					// Analytic shrink: the probe placed `probeArea` padded texels
					// against a one-page budget of targetFill·R². Step k by the
					// square root of the area ratio — proportional to the actual
					// overflow — instead of a blind ×0.95. Upper clamp 0.95 keeps
					// waste-driven overflows (area under budget, layout still >1
					// page) converging at least as fast as the old ladder; lower
					// clamp 0.80 stops one noisy probe from collapsing the scale.
					const double budgetArea = targetFill * R * R;
					double shrink = std::sqrt(budgetArea / std::max(static_cast<double>(probeArea), 1.0));
					shrink = std::clamp(shrink, 0.80, 0.95);
					kf *= shrink;
				}
				result.fitAttempts = attempts;
```

(The trailing `const float kfinal = ...; UV/rect rescale` code after the loop stays exactly as-is.)

Note the upper clamp is 0.95, not the spec's 0.99 — 0.99 would converge *slower* than the old ladder on waste-driven (not area-driven) overflows. Also dropped: the spec's "sort order reused across probes" micro-optimization — after Task 1 the per-probe sort is O(N log N) ≈ milliseconds at 161k rects, so the extra parameter buys nothing measurable. Record both as deviations-with-reason in the executing session's ledger.

- [ ] **Step 5: Build, run the full suite**

```bash
cmake --build make -j8 && ctest --test-dir make --output-on-failure
```

Expected: ALL pass, including the pre-existing `FitToResolutionIteratesToOnePage` and `FitToResolutionHonorsPageDimensions` (both property-based; the analytic shrink still reaches their fixed points).

- [ ] **Step 6: Format and commit**

```bash
clang-format -i src/AtlasPacking.cpp include/halfmesh/AtlasCharting.h tests/AtlasTest.cpp
git add src/AtlasPacking.cpp include/halfmesh/AtlasCharting.h tests/AtlasTest.cpp
git commit -m "perf(atlas): overflow-proportional fit-to-resolution shrink + fitAttempts

The blind x0.95 ladder re-packed up to 8 times; at production padding=4
that was a measured 8x wall-time multiplier. Shrink now steps by
sqrt(budget/packed-area), clamped [0.80, 0.95], and AtlasResult reports
the probe count for diagnosability.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Post-repair re-merge (spec §5, Fix 2)

**Files:**
- Modify: `include/halfmesh/Parametrize.h` (struct `ParametrizeParams`, after `developableFlipRepairRounds` ~line 111)
- Modify: `src/AtlasCharting.cpp` (`RepairDevelopableFlips` ~line 1114; `detail::SegmentCharts` ~line 1234)
- Test: `tests/AtlasTest.cpp`

**Interfaces:**
- Consumes: `unsigned DevelopableMerge(const SegmentState&, const ParametrizeParams&, std::vector<unsigned>& chart, unsigned numCharts, double budget)` (unchanged); `unsigned EnforceConnectivity(const SegmentState&, std::vector<unsigned>&)`; `unsigned Compact(std::vector<unsigned>&)`; `bool detail::ChartFacesFold(const Mesh&, const std::vector<Mesh::FIndex>&, const ParametrizeParams&)` (3-arg overload, `src/Parametrize.cpp:2103`).
- Produces: `ParametrizeParams::postRepairMergeRounds` (`unsigned`, default 2, 0 = old behavior); `RepairDevelopableFlips(..., const std::vector<unsigned>* frontierIn = nullptr)` — when non-null, skips `EnforceConnectivity` (precondition: `chart` compact and per-chart connected) and repairs only the listed chart ids.

- [ ] **Step 1: Add the parameter**

In `include/halfmesh/Parametrize.h`, immediately after the `developableFlipRepairRounds` member:

```cpp
	// Post-repair merge↔repair rounds. Flip repair bisects folding charts but
	// nothing recombined the fragments afterward — on noisy MVS meshes that
	// leaves ~6-face charts (seam + padding blowup, quadratic packing input).
	// Each round re-runs DevelopableMerge over the post-repair partition (same
	// cone-error + vertex-defect gates) and then one flip-repair wave over the
	// merged charts only, so a merge that re-folds is split right back (never
	// a regression). Stops early when a round changes <1% of charts.
	// 0 restores the pre-2026-08 behavior.
	unsigned postRepairMergeRounds = 2;
```

- [ ] **Step 2: Write the failing test**

Add to `tests/AtlasTest.cpp` (needs `#include <random>` at the top with the other includes, and this extern seam next to the existing `ComputeSegmentationSeeds` extern if one is present in this file — otherwise add both lines shown):

```cpp
namespace halfmesh {
namespace detail {
// Test seam: 3-arg fold verdict (defined in src/Parametrize.cpp).
bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params);
} // namespace detail
} // namespace halfmesh
```

```cpp
// Deterministic "staircase terrain": an n×n grid whose vertex heights are
// quantized random levels — many high-angle-defect vertices, like a
// tetra-extracted MVS surface. Cone-Lloyd fragments it, flip repair splits
// further; the post-repair merge must claw a meaningful share back.
static void BuildStaircaseTerrain(Mesh& mesh, unsigned n, float step)
{
	std::mt19937 rng(42u);
	std::uniform_int_distribution<int> lvl(0, 4);
	std::vector<float> h((n + 1) * (n + 1));
	for (float& z : h)
		z = step * static_cast<float>(lvl(rng));
	for (unsigned y = 0; y <= n; ++y)
		for (unsigned x = 0; x <= n; ++x)
			mesh.vertices.push_back({static_cast<float>(x), static_cast<float>(y), h[y * (n + 1) + x]});
	for (unsigned y = 0; y < n; ++y)
		for (unsigned x = 0; x < n; ++x) {
			const unsigned a = y * (n + 1) + x, b = a + 1, c = a + (n + 1), d = c + 1;
			mesh.faces.push_back({a, b, d});
			mesh.faces.push_back({a, d, c});
		}
}

TEST(SegmentCharts, PostRepairMergeReducesChartsFoldFree)
{
	Mesh base;
	BuildStaircaseTerrain(base, 48u, 0.75f);

	ParametrizeParams p0;
	p0.postRepairMergeRounds = 0;
	Mesh m0 = base;
	std::vector<unsigned> chart0;
	const unsigned n0 = SegmentCharts(m0, p0, chart0);

	ParametrizeParams p2;
	p2.postRepairMergeRounds = 2;
	Mesh m2 = base;
	std::vector<unsigned> chart2;
	const unsigned n2 = SegmentCharts(m2, p2, chart2);

	std::printf("[SegmentCharts] PostRepairMerge: %u -> %u charts\n", n0, n2);
	// Precondition: the fixture actually fragments (otherwise the test is vacuous).
	ASSERT_GT(n0, 50u) << "fixture did not fragment — increase `step` or grid size";
	EXPECT_LT(n2, n0) << "re-merge recombined nothing";

	// Every face charted, ids compact.
	ASSERT_EQ(chart2.size(), m2.faces.size());
	std::vector<char> seen(n2, 0);
	for (unsigned c : chart2) {
		ASSERT_LT(c, n2);
		seen[c] = 1;
	}
	for (unsigned c = 0; c < n2; ++c)
		EXPECT_TRUE(seen[c]) << "chart id " << c << " is empty";

	// Fold-free guarantee survives the merge: no >2-face chart folds.
	std::vector<std::vector<Mesh::FIndex>> fl(n2);
	for (Mesh::FIndex f = 0; f < static_cast<Mesh::FIndex>(m2.faces.size()); ++f)
		fl[chart2[f]].push_back(f);
	for (unsigned c = 0; c < n2; ++c) {
		if (fl[c].size() <= 2)
			continue;
		EXPECT_FALSE(detail::ChartFacesFold(m2, fl[c], p2)) << "chart " << c << " folds after re-merge";
	}
}
```

If the `ASSERT_GT(n0, 50u)` precondition fails, tune the fixture (`step` up toward 1.0, or grid 64) until it fragments — the precondition assert exists exactly so this tuning is explicit, not silent.

- [ ] **Step 3: Run it to verify it fails**

```bash
cmake --build make -j8 && ctest --test-dir make -R PostRepairMerge --output-on-failure
```

Expected: compile FAILS on `p0.postRepairMergeRounds` (member doesn't exist) before Step 1, or — with Step 1 done — the test runs and FAILS at `EXPECT_LT(n2, n0)` (both runs identical: nothing consumes the param yet).

- [ ] **Step 4: Implement the frontier-restricted repair + the merge loop**

(a) In `src/AtlasCharting.cpp`, change `RepairDevelopableFlips`'s signature and preamble:

```cpp
unsigned RepairDevelopableFlips(SegmentState& s, const ParametrizeParams& params,
                                std::vector<unsigned>& chart, unsigned numCharts,
                                detail::ChartFlattenCache* cache,
                                const std::vector<unsigned>* frontierIn = nullptr)
```

and replace the two lines

```cpp
	numCharts = EnforceConnectivity(s, chart);
```
(keep) … then where the frontier is initialised (`std::vector<unsigned> frontier(numCharts); std::iota(...)`), with:

```cpp
	// frontierIn restricts repair to the listed chart ids (precondition: the
	// caller guarantees `chart` is compact and per-chart connected — true
	// after DevelopableMerge, which only unions TopoNeighbor-adjacent charts
	// and ends with Compact). Skipping EnforceConnectivity keeps those ids
	// stable so the restriction is meaningful.
	if (frontierIn == nullptr)
		numCharts = EnforceConnectivity(s, chart);
	std::vector<std::vector<Mesh::FIndex>> fl(numCharts);
	for (Mesh::FIndex f = 0; f < s.numFaces; ++f)
		fl[chart[f]].push_back(f);
	...
	std::vector<unsigned> frontier;
	if (frontierIn != nullptr)
		frontier = *frontierIn;
	else {
		frontier.resize(numCharts);
		std::iota(frontier.begin(), frontier.end(), 0u);
	}
```

(b) In `detail::SegmentCharts` (~line 1266), replace

```cpp
	if (params.developableFlipRepairRounds > 0)
		numCharts = RepairDevelopableFlips(s, params, chart, numCharts, cache);
```

with:

```cpp
	if (params.developableFlipRepairRounds > 0) {
		numCharts = RepairDevelopableFlips(s, params, chart, numCharts, cache);
		// Post-repair re-merge: recombine the bisection fragments the repair
		// left behind (nothing else ever merges again), then repair ONLY the
		// merged charts — a merge that re-folds is split right back, so the
		// fold-free guarantee is preserved and a round can never regress.
		for (unsigned round = 0; round < params.postRepairMergeRounds; ++round) {
			const unsigned before = numCharts;
			std::vector<unsigned> pre(chart);
			numCharts = DevelopableMerge(s, params, chart, numCharts,
			                             static_cast<double>(params.developableMaxConeError));
			// Charts containing faces from ≥2 pre-merge charts are the merged
			// ("dirty") ones — the only ones whose fold verdict changed.
			std::vector<unsigned> firstPre(numCharts, NONE);
			std::vector<char> dirtyFlag(numCharts, 0);
			for (FIndex f = 0; f < s.numFaces; ++f) {
				const unsigned c = chart[f];
				if (firstPre[c] == NONE)
					firstPre[c] = pre[f];
				else if (firstPre[c] != pre[f])
					dirtyFlag[c] = 1;
			}
			std::vector<unsigned> dirty;
			for (unsigned c = 0; c < numCharts; ++c)
				if (dirtyFlag[c])
					dirty.push_back(c);
			if (dirty.empty())
				break; // nothing merged — converged
			numCharts = RepairDevelopableFlips(s, params, chart, numCharts, cache, &dirty);
#ifdef HM_ATLAS_DEBUG
			std::cerr << "[re-merge] round " << round << ": " << before << " -> " << numCharts << " charts\n";
#endif
			if (before - numCharts < before / 100)
				break; // <1% net change — not worth another round
		}
	}
```

Stale-cache safety: `ChartFlattenCache` keys by smallest face id and verifies the full face list before reuse (`src/ChartFlattenCache.h` header comment) — a merged chart misses and recomputes; never corrupts.

- [ ] **Step 5: Build, run the full suite**

```bash
cmake --build make -j8 && ctest --test-dir make --output-on-failure
```

Expected: ALL pass — including `SegmentQuality.SegmentDeterministicRunTwice` (the merge loop is deterministic: heap ties break on ids, dirty list is ascending) and `Parametrize.CachedPipelineMatchesUncachedTwoCallPipeline` (cache misses recompute). `PostRepairMergeReducesChartsFoldFree` now passes. If `SegmentQuality.CorpusTableAndFloors` moves, inspect: chart-count floors should IMPROVE (fewer charts); a quality-floor regression is a stop-and-investigate.

- [ ] **Step 6: Format and commit**

```bash
clang-format -i src/AtlasCharting.cpp include/halfmesh/Parametrize.h tests/AtlasTest.cpp
git add src/AtlasCharting.cpp include/halfmesh/Parametrize.h tests/AtlasTest.cpp
git commit -m "feat(atlas): post-repair merge rounds recombine flip-repair fragments

Flip repair bisected 103k charts to 161k on a 1M-face MVS mesh and
nothing ever merged again. Bounded merge<->repair rounds (default 2,
same cone-error/vertex-defect gates, repair restricted to the merged
charts via a frontier parameter) recombine the dust: fewer seams, less
padding waste, smaller packing input.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: End-to-end perf validation + CLI padding arg + CHANGELOG

**Files:**
- Modify: `examples/Unwrap.cpp` (usage + argv parsing, lines ~30–41 and the `aparams` block ~line 63)
- Modify: `CHANGELOG.md`
- No new test files (this task's "test" is the release-build gate runs)

**Interfaces:**
- Consumes: Tasks 1–3 (the assembled pipeline).
- Produces: the measured numbers Task 5 and Task 7 cite; `unwrap` CLI accepts `[padding]` as argv[4].

- [ ] **Step 1: Add the padding argument to the example CLI**

In `examples/Unwrap.cpp`: usage line becomes `"Usage: unwrap <in.ply> <out.ply> [resolution=1024] [padding=2]\n"`, accept `argc` up to 5, parse `const unsigned padding = (argc == 5) ? static_cast<unsigned>(std::stoul(argv[4])) : 2u;` and use `aparams.padding = padding;`.

- [ ] **Step 2: Release build + gate runs**

```bash
cd /home/wii/halfmesh/.claude/worktrees/python-bindings
export VCPKG_ROOT=/home/wii/vcpkg
cmake -S . -B make-rel -DCMAKE_BUILD_TYPE=Release -DHALFMESH_BUILD_TOOLS=ON
cmake --build make-rel -j8
time make-rel/examples/unwrap /home/wii/bench_runs/unwrap_spike/post200k/mesh.ply /tmp/claude-1007/-home-wii-radiance-halfmesh/ba888238-b4cf-44fa-a95f-7325aefcc82f/scratchpad/t4_320k_uv.ply 4096 4
time make-rel/examples/unwrap /home/wii/bench_runs/task4_validation/post/mesh.ply /tmp/claude-1007/-home-wii-radiance-halfmesh/ba888238-b4cf-44fa-a95f-7325aefcc82f/scratchpad/t4_1m_uv.ply 4096 4
```

Gates (from the plan's Global Constraints): 320k ≤ 120 s; 1M ≤ 600 s, occupancy ≥ 0.75 (the CLI prints it), 1 page, charts ≤ 161,627. Record all printed numbers in the task report. A missed gate is a stop-and-report, not a silent pass.

- [ ] **Step 3: CHANGELOG entry**

Add under the unreleased 0.2.0 section of `CHANGELOG.md` (create the section if absent, matching the file's existing style):

```markdown
- Atlas packing: two-tier pack pass (skyline head + shelf-row tail) removes the
  quadratic regime at 100k+ charts; fit-to-resolution shrink is now
  overflow-proportional (was a blind ×0.95 ladder) and `AtlasResult` reports
  `fitAttempts`. Measured on a 1M-face MVS mesh (161k charts, padding 4,
  4096²): 13,473 s → <numbers from Step 2> s.
- Atlas segmentation: post-repair merge rounds
  (`ParametrizeParams::postRepairMergeRounds`, default 2) recombine
  flip-repair fragments — fewer charts, fewer seams, less padding waste.
```

Replace `<numbers from Step 2>` with the measured 1M wall time before committing (a literal placeholder in the committed file is a task failure).

- [ ] **Step 4: Commit**

```bash
clang-format -i examples/Unwrap.cpp
git add examples/Unwrap.cpp CHANGELOG.md
git commit -m "docs: CHANGELOG for the atlas speedup; unwrap CLI padding arg

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Wheel rebuild, binding `fit_attempts`, radiance validation

**Files:**
- Modify: `python/binding.cpp` (the `unwrap` lambda's meta dict, ~line 229, and its docstring)
- Modify: `python/tests/` — the binding test that asserts unwrap meta keys (find it with `grep -rn "occupancy" python/tests/`); extend for `fit_attempts`
- No radiance file changes in this task

**Interfaces:**
- Consumes: `AtlasResult::fitAttempts` (Task 2).
- Produces: installed wheel in the `radiance` conda env; `hm.unwrap()` meta gains `"fit_attempts"` (additive — radiance reads only its known keys).

- [ ] **Step 1: Expose fit_attempts in the binding**

In `python/binding.cpp`, after `meta["occupancy"] = result.occupancy;` add `meta["fit_attempts"] = result.fitAttempts;` and extend the docstring's returns list to `{charts, pages, width, height, occupancy, fit_attempts, vertices, faces}`.

- [ ] **Step 2: Extend the binding test**

In the python test that checks unwrap meta keys, add `fit_attempts` to the expected-keys assertion and assert `meta["fit_attempts"] >= 1` (the binding always packs with `fitToResolution` on via `GenerateAtlas`). Follow the exact existing test's structure — this is a one-assertion extension, not a new file.

- [ ] **Step 3: Rebuild + install the wheel, run binding tests**

```bash
cd /home/wii/halfmesh/.claude/worktrees/python-bindings
export VCPKG_ROOT=/home/wii/vcpkg
/home/wii/miniconda3/envs/radiance/bin/python -m pip install . --no-deps --force-reinstall
/home/wii/miniconda3/envs/radiance/bin/python -m pytest python/tests -x -q
```

Expected: install succeeds (incremental — `build-dir` is persistent), binding tests pass, `python -c "import halfmesh; print(halfmesh.__version__)"` still reports 0.2.0.

- [ ] **Step 4: Radiance validation — unit suite + the production-path gate run**

```bash
cd /home/wii/radiance-halfmesh
/home/wii/miniconda3/envs/radiance/bin/python -m pytest tests/unit -x -q
/home/wii/miniconda3/envs/radiance/bin/python -m radiance.mesh.unwrap --mesh /home/wii/bench_runs/task4_validation/post/mesh.ply --out /tmp/claude-1007/-home-wii-radiance-halfmesh/ba888238-b4cf-44fa-a95f-7325aefcc82f/scratchpad/t5_1m_uv.ply --resolution 4096
```

Expected: full radiance unit suite green with zero code changes; the production-path unwrap (radiance defaults, padding=4) meets the 1M gates: `[timing] unwrap<=600`, occupancy ≥ 75 %, 1 page, charts ≤ 161,627. Record the `[unwrap]` and `[timing]` lines in the task report — Task 7's docs cite them.

- [ ] **Step 5: Commit (halfmesh side only)**

```bash
cd /home/wii/halfmesh/.claude/worktrees/python-bindings
git add python/binding.cpp python/tests
git commit -m "feat(python): expose fit_attempts in unwrap() meta

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Opt-in remesh in radiance postprocess (spec §6, radiance repo)

**Files:**
- Modify: `/home/wii/radiance-halfmesh/radiance/mesh/postprocess.py`
- Test: `/home/wii/radiance-halfmesh/tests/unit/test_mesh_postprocess_unit.py` (validation + CLI), `/home/wii/radiance-halfmesh/tests/unit/test_mesh_postprocess_ops.py` (real halfmesh run)

**Interfaces:**
- Consumes: `hm.remesh(vertices, faces, edge_length, iterations=3) -> tuple` (already in the installed wheel).
- Produces: `postprocess_mesh(..., remesh_edge_length: float = 0.0, remesh_iterations: int = 3)`; CLI flags `--remesh-edge-length`, `--remesh-iterations`; provenance keys `remesh_edge_length`, `remesh_iterations`; timing lap `remesh`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_mesh_postprocess_unit.py`, extend `test_cli_exposes_postprocess_flags` with the two new flags (`--remesh-edge-length`, `--remesh-iterations`, matching how the existing flags are asserted there) and add:

```python
def test_negative_remesh_edge_length_raises_before_any_io(tmp_path):
    from radiance.mesh.postprocess import postprocess_mesh

    with pytest.raises(ValueError, match="remesh_edge_length"):
        postprocess_mesh(
            str(tmp_path / "missing.ply"),
            str(tmp_path / "out.ply"),
            remesh_edge_length=-0.1,
        )


def test_zero_remesh_iterations_raises_before_any_io(tmp_path):
    from radiance.mesh.postprocess import postprocess_mesh

    with pytest.raises(ValueError, match="remesh_iterations"):
        postprocess_mesh(
            str(tmp_path / "missing.ply"),
            str(tmp_path / "out.ply"),
            remesh_edge_length=0.5,
            remesh_iterations=0,
        )
```

In `tests/unit/test_mesh_postprocess_ops.py` add (self-contained; follows the file's existing use of real halfmesh):

```python
def test_remesh_is_off_by_default_and_changes_mesh_when_on(tmp_path):
    import json

    import numpy as np
    import torch

    from radiance.mesh.extract import _write_triangle_mesh_ply
    from radiance.mesh.postprocess import postprocess_mesh

    # A 12x12 unit-cell grid: uniform edge ~1.0, so remesh to 0.5 must refine.
    n = 12
    xs, ys = np.meshgrid(np.arange(n + 1, dtype=np.float32), np.arange(n + 1, dtype=np.float32))
    verts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros((n + 1) ** 2, dtype=np.float32)])
    faces = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            faces.append([a, a + 1, a + n + 2])
            faces.append([a, a + n + 2, a + n + 1])
    faces = np.asarray(faces, dtype=np.int64)
    src = tmp_path / "grid.ply"
    _write_triangle_mesh_ply(str(src), torch.from_numpy(verts), torch.from_numpy(faces))

    common = dict(smooth_iterations=0, decimate_target=1.0, close_holes=0)

    out_off = tmp_path / "off" / "mesh.ply"
    postprocess_mesh(str(src), str(out_off), **common)
    env_off = json.loads((out_off.parent / "env.json").read_text())
    assert env_off["extra"]["remesh_edge_length"] == 0.0

    out_on = tmp_path / "on" / "mesh.ply"
    postprocess_mesh(str(src), str(out_on), remesh_edge_length=0.5, **common)
    env_on = json.loads((out_on.parent / "env.json").read_text())
    assert env_on["extra"]["remesh_edge_length"] == 0.5
    assert env_on["extra"]["remesh_iterations"] == 3
    # Refinement toward edge 0.5 must increase the face count; off must not.
    assert env_on["extra"]["output_faces"] > env_off["extra"]["output_faces"]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /home/wii/radiance-halfmesh
/home/wii/miniconda3/envs/radiance/bin/python -m pytest tests/unit/test_mesh_postprocess_unit.py tests/unit/test_mesh_postprocess_ops.py -x -q
```

Expected: FAIL — `postprocess_mesh() got an unexpected keyword argument 'remesh_edge_length'`.

- [ ] **Step 3: Implement**

In `radiance/mesh/postprocess.py`:

(a) Signature: add `remesh_edge_length: float = 0.0, remesh_iterations: int = 3,` to `postprocess_mesh`'s keyword-only params.

(b) Validation, with the existing `ValueError` block:

```python
    if remesh_edge_length < 0.0:
        raise ValueError(
            f"remesh_edge_length must be >= 0 (0 disables), got {remesh_edge_length!r}"
        )
    if remesh_iterations < 1:
        raise ValueError(f"remesh_iterations must be >= 1, got {remesh_iterations!r}")
```

(c) Stage — after the `min_component_size` block and its `sw.lap("components")`, before the torch import:

```python
    # Optional isotropic remesh toward a uniform edge length — the LAST geometry
    # step so charting sees the final topology. Opt-in experiment lever (see
    # docs/mesh-postprocess.md): the output is no longer the exact F1-gated
    # geometry, so presets may only enable it after passing the same bench gates.
    if remesh_edge_length > 0.0:
        faces_before = len(f)
        v, f = hm.remesh(v, f, float(remesh_edge_length), int(remesh_iterations))
        print(
            f"[postprocess] remeshed {faces_before} -> {len(f)} faces "
            f"(edge_length {remesh_edge_length}, iterations {remesh_iterations})"
        )
    sw.lap("remesh")
```

(d) Provenance: add to `params` (after `components_removed`):

```python
        "remesh_edge_length": remesh_edge_length,
        "remesh_iterations": remesh_iterations,
```

(e) CLI: two `parser.add_argument` entries following the file's style —

```python
    parser.add_argument(
        "--remesh-edge-length",
        type=float,
        default=0.0,
        help="isotropic remesh toward this edge length (world units) as the "
        "final geometry step; 0 disables (default: 0). Experiment lever: "
        "output is no longer the exact F1-gated geometry.",
    )
    parser.add_argument(
        "--remesh-iterations",
        type=int,
        default=3,
        help="remesh relaxation iterations (default: 3)",
    )
```

and pass both through in `main()` (`remesh_edge_length=args.remesh_edge_length, remesh_iterations=args.remesh_iterations`).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/home/wii/miniconda3/envs/radiance/bin/python -m pytest tests/unit/test_mesh_postprocess_unit.py tests/unit/test_mesh_postprocess_ops.py -q
```

Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit (radiance repo)**

```bash
cd /home/wii/radiance-halfmesh
git add radiance/mesh/postprocess.py tests/unit/test_mesh_postprocess_unit.py tests/unit/test_mesh_postprocess_ops.py
git commit -m "feat(postprocess): opt-in isotropic remesh as the final geometry step

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Remesh evaluation study + docs refresh (spec §6/§8, radiance repo — judgment task)

**Files:**
- Create: `/home/wii/radiance-halfmesh/docs/superpowers/specs/2026-08-21-remesh-unwrap-study.md` (results + verdict)
- Modify: `/home/wii/radiance-halfmesh/docs/mesh-postprocess.md` (known-limitation section + new flag docs + new measured numbers)

**Interfaces:**
- Consumes: Task 5's measured 1M unwrap numbers (from its task report); Task 6's flag; the existing metric machinery in `scripts/bench_postprocess.py` (`mesh_distance_metrics`, `thin_metrics`, `build_references`, `evaluate_gates`, `remeasure` — read that file and `tests/unit/test_bench_postprocess.py` for the call pattern; the frozen GT protocol is `dense_tau2_cull20`, tau=0.005, `Truck_final_transform.npy`).
- Produces: an enable-or-document verdict for remesh, recorded with evidence.

This is an experiment, so its steps are measurements and a decision, not TDD:

- [ ] **Step 1: Build the arms.** Median input edge length: compute from `/home/wii/bench_runs/task4_validation/post/mesh.ply` with numpy (load via `halfmesh.Mesh().load(...)/.to_arrays()`, median over unique-edge lengths of a 100k-face random subset, seed 42). Run `radiance.mesh.postprocess` on `/home/wii/bench_runs/Truck_halfmesh_baseline/mesh/mesh.ply` with the shipped defaults PLUS each of: `--remesh-edge-length 0` (control — must byte-match the existing post mesh pipeline settings), `--remesh-edge-length <median>`, `--remesh-edge-length <1.5*median>`. Outputs under `~/bench_runs/remesh_study/<arm>/mesh.ply`.
- [ ] **Step 2: Geometry gates.** For each arm, compute the exact-distance metrics vs GT exactly as the 2026-08-19 study did (same functions, same tau, same GT-backed thin subset). Gates: F1 ≥ 0.98× the no-remesh arm, thin metrics ≥ 0.95× — identical gate structure to the postprocess study.
- [ ] **Step 3: Unwrap effect.** `radiance.mesh.unwrap` (defaults, new wheel) on each arm; record wall time, charts, occupancy, fit_attempts (from env.json / stdout).
- [ ] **Step 4: Verdict.** Enable remesh in the documented recommended pipeline ONLY if an arm passes Step 2's gates AND improves Step 3 meaningfully (≥20 % time or ≥20 % charts). Otherwise: document as opt-in with the table. Either way, write the study file with the full table (all arms × all metrics), the verdict, and the reasoning — mirroring `docs/superpowers/specs/2026-08-19-halfmesh-postprocess-tuning.md`'s structure.
- [ ] **Step 5: Docs refresh.** In `docs/mesh-postprocess.md`: (a) replace the unwrap known-limitation text (the 3h45m wall-time cause is fixed — cite the old and new 1M numbers from Task 5's report and name the halfmesh change); (b) document `--remesh-edge-length`/`--remesh-iterations` and the study verdict; (c) add the two new provenance keys to the env.json key list (§ "unwrap's extra" section has the pattern).
- [ ] **Step 6: Commit (radiance repo)**

```bash
cd /home/wii/radiance-halfmesh
git add docs/superpowers/specs/2026-08-21-remesh-unwrap-study.md docs/mesh-postprocess.md
git commit -m "docs: remesh-for-unwrap study verdict + unwrap speedup numbers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task order and model notes for the executor

Tasks 1→2→3 are sequential (same files, layered interfaces). Task 4 follows 3; Task 5 follows 4; Task 6 can run any time after Task 5's wheel install (it needs `hm.remesh` — already in the current wheel, but run its ops test against the rebuilt one); Task 7 requires 5 and 6. Tasks 1–3 are algorithmic C++ with complete code in this plan (mid-tier implementers; reviewers should check the code against the spec's complexity claims). Task 7 is judgment-heavy (evaluation + verdict) — use a capable model, mirroring the 2026-08-19 study's staffing.

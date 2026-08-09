/*
* GoldenSpecs.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/golden/GoldenSpecs.h — the SINGLE source of truth for which
// (input mesh × op × parameters) combinations are frozen as golden fixtures,
// used by GoldenDiffTest.cpp to drive regression checks.
//
// Each spec carries:
//   - a stable fixture name (mesh + op),
//   - a builder for the INPUT mesh (deterministic corpus generator),
//   - the op applied to a copy of the input (returns the op count),
//   - a comparison mode telling the regression test how strict to be.
//
// Namespace: hmtest::golden
#pragma once

#include "Corpus.h"

#include <functional>
#include <string>
#include <vector>

namespace hmtest {
namespace golden {

// How the golden test compares the recomputed result to the frozen fixture
// (docs/TESTING.md §5).
enum class CompareMode {
	// Deterministic op order, build-flag-stable: exact canonical mesh equality
	// (up to relabeling) + exact op count + tight metrics.
	CANONICAL_EXACT,
	// Float/op-order or build-flag sensitive (Remesh): element counts within a
	// small relative ε + Hausdorff(result,golden) ≈ 0 + metrics within looser ε.
	TOLERANT,
};

struct GoldenSpec
{
	std::string meshName;
	std::string opName;
	CompareMode mode;
	// Build the deterministic input mesh.
	std::function<halfmesh::Mesh()> makeInput;
	// Apply the op to a copy of the input in place; return the op count
	// (fix/removal count, or 0 for ops that mutate without a count).
	std::function<uint32_t(halfmesh::Mesh&)> runOp;
};

// The frozen corpus.  Kept small (no LargeMesh): clean analytic meshes for
// Simplify/Remesh and the dirty synthesizers for the repair ops.
inline std::vector<GoldenSpec> GoldenCorpus()
{
	std::vector<GoldenSpec> specs;

	// --- Repair: RemoveDuplicateFaces (cube + 3 duplicate pairs) ------------
	specs.push_back({"DirtyDuplicateFaces", "RemoveDuplicateFaces", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::DirtyDuplicateFaces(3); },
	                 [](halfmesh::Mesh& m) {
		                 const uint32_t n = static_cast<uint32_t>(m.RemoveDuplicateFaces(true));
		                 m.RemoveUnreferencedVertices();
		                 return n;
	                 }});

	// --- Repair: RemoveDegenerateFaces (cube + 3 degenerate faces) ----------
	specs.push_back({"DirtyDegenerateFaces", "RemoveDegenerateFaces", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::DirtyDegenerateFaces(3); },
	                 [](halfmesh::Mesh& m) {
		                 const uint32_t n = static_cast<uint32_t>(m.RemoveDegenerateFaces(1e-5f));
		                 m.RemoveUnreferencedVertices();
		                 return n;
	                 }});

	// --- Repair: RemoveUnreferencedVertices (cube + 5 orphans) --------------
	specs.push_back({"DirtyUnreferencedVertices", "RemoveUnreferencedVertices", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::DirtyUnreferencedVertices(5); },
	                 [](halfmesh::Mesh& m) {
		                 return static_cast<uint32_t>(m.RemoveUnreferencedVertices());
	                 }});

	// --- Repair: RemoveSmallComponents (cube + far triangle, drop the small one) ---
	// Threshold 2 drops the single-triangle component (size 1) and keeps the cube
	// (size 12): a meaningful "remove the small one" case. (The old threshold 13
	// treated BOTH components as small, so once the inverted-count bug was fixed it
	// would have emptied the mesh; 2 exercises the real behaviour instead.)
	specs.push_back({"DirtyManyComponents", "RemoveSmallComponents", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::DirtyManyComponents(2); },
	                 [](halfmesh::Mesh& m) { return m.RemoveSmallComponents(2); }});

	// --- Repair: FixNonManifold (bow-tie vertex) ---------------------------
	specs.push_back({"DirtyBowTie", "FixNonManifold", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::DirtyBowTie(); },
	                 [](halfmesh::Mesh& m) { return m.FixNonManifold(0.01f); }});

	// --- Simplify — float/build-flag sensitive ------------------------------
	// QEM collapse ordering depends on float LSBs of the accumulated quadric
	// costs: FMA contraction and Eigen's vectorized reductions change those
	// LSBs between build configs, flipping near-tied pops and cascading to a
	// different (equally valid) result. Empirically: a Debug (-O0) build
	// diverges on Torus; a Release build with -ffp-contract=off diverges on
	// all four fixtures below. Only GridPlane (flat: rank-1 quadrics take the
	// FP-order-insensitive fallback paths) reproduces canonically across the
	// probed configs and keeps the exact compare.
	specs.push_back({"UVSphere", "Simplify", CompareMode::TOLERANT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) { m.Simplify(0.5f); return 0u; }});

	specs.push_back({"Torus", "Simplify", CompareMode::TOLERANT,
	                 [] { return hmtest::corpus::TorusMesh(24, 16); },
	                 [](halfmesh::Mesh& m) { m.Simplify(0.5f); return 0u; }});

	// --- Simplify (open grid with boundary, 0.5 decimate) ------------------
	specs.push_back({"GridPlane", "Simplify", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::GridPlane(12); },
	                 [](halfmesh::Mesh& m) { m.Simplify(0.5f); return 0u; }});

	specs.push_back({"UVSphere", "SimplifyFast", CompareMode::TOLERANT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) { m.Simplify(0.5f, 0.f, 5.f); return 0u; }});

	specs.push_back({"UVSphere", "SimplifyMinEdge", CompareMode::TOLERANT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) { m.Simplify(1.0f, 0.2f); return 0u; }});

	// --- Remesh (UV sphere, isotropic) — float/build-flag sensitive --------
	specs.push_back({"UVSphere", "RemeshIsotropic", CompareMode::TOLERANT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) {
		                 halfmesh::Mesh::RemeshParams p;
		                 p.SetEdgeLength(0.25f);
		                 p.iterations = 3;
		                 p.adapt = false;
		                 p.checkSurfDist = false;
		                 m.RemeshIsotropic(p);
		                 return 0u;
	                 }});

	// --- Smoothing (added 2026-08): the vcg crosscheck was removed with the
	// vcgCompatible mode, leaving the smoothers with no external ground truth —
	// these fixtures freeze the current (unit-pinned) behavior against drift.
	// Deterministic vertex order but plain Eigen float arithmetic, so exact
	// equality holds only under default build flags (excluded on -march CI legs
	// via GOLDEN_EXACT_EXCLUDE, like the other CANONICAL_EXACT specs).
	// UVSphere exercises the closed-surface formulas of both filters;
	// OpenCylinder freezes Taubin's border-curve rule (deliberately different
	// from HC's uniform ring — see MeshSmooth.cpp).
	specs.push_back({"UVSphere", "SmoothTaubin", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) { m.SmoothTaubin(5); return 0u; }});

	specs.push_back({"UVSphere", "SmoothHCLaplacian", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::UVSphere(16, 24); },
	                 [](halfmesh::Mesh& m) { m.SmoothHCLaplacian(5); return 0u; }});

	specs.push_back({"OpenCylinder", "SmoothTaubin", CompareMode::CANONICAL_EXACT,
	                 [] { return hmtest::corpus::OpenCylinder(24, 8); },
	                 [](halfmesh::Mesh& m) { m.SmoothTaubin(5); return 0u; }});

	return specs;
}

} // namespace golden
} // namespace hmtest

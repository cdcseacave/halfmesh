/*
* MeshRemeshShared.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Shared remeshing primitives used by BOTH the isotropic remesher
// (MeshRemesh.cpp) and the hole-filler (MeshHoles.cpp), so the two code paths
// stop reimplementing the same logic on their respective data structures:
//   - triangle shape-quality measures,
//   - the ideal-valence rule and the valence-flip score, and
//   - a k-harmonic (membrane / thin-plate) fairing solve.
//
// Library-internal header (lives under src/, not include/halfmesh/): no public
// API surface. Namespace halfmesh::detail.
#pragma once

#include <halfmesh/Types.h>

#include <Eigen/Core>

#include <vector>

namespace halfmesh {
namespace detail {

// ---------------------------------------------------------------------------
// TriangleQuality
// cross-product-squared / max-squared-edge-length.  Range [0, inf).
// Equilateral -> high; degenerate -> 0.
// ---------------------------------------------------------------------------
template <class Scalar>
Scalar TriangleQuality(const TPoint3<Scalar>& p0,
                       const TPoint3<Scalar>& p1,
                       const TPoint3<Scalar>& p2)
{
	const TPoint3<Scalar> d10 = p1 - p0;
	const TPoint3<Scalar> d20 = p2 - p0;
	const TPoint3<Scalar> d12 = p1 - p2;
	const TPoint3<Scalar> x = d10.cross(d20);
	const Scalar a = x.squaredNorm();
	if (a == 0)
		return 0;
	Scalar b = d10.squaredNorm();
	if (b == 0)
		return 0;
	Scalar t = d20.squaredNorm();
	if (b < t)
		b = t;
	t = d12.squaredNorm();
	if (b < t)
		b = t;
	return a / b;
}

// ---------------------------------------------------------------------------
// TriangleQualityRadii
// in-radius / circumscribed-radius.  Range [0, 1].  Equilateral -> ~1.
// ---------------------------------------------------------------------------
template <class Scalar>
Scalar TriangleQualityRadii(const TPoint3<Scalar>& p0,
                            const TPoint3<Scalar>& p1,
                            const TPoint3<Scalar>& p2)
{
	const Scalar a = (p1 - p0).norm();
	const Scalar b = (p2 - p0).norm();
	const Scalar c = (p1 - p2).norm();
	const Scalar sum = (a + b + c) * Scalar(0.5);
	// area^2 from the cross product, not the naive Heron form s(s-a)(s-b)(s-c):
	// for a needle triangle (c ~ a+b) the Heron factors subtract nearly-equal
	// values and lose most significant bits in float, making the degeneracy
	// threshold (TagCreaseEdges, 1e-4) noisy. |(p1-p0)x(p2-p0)|^2/4 == area^2 with
	// no cancellation (~1e-5 vs ~0.5% relative error at the threshold), matching
	// the sibling TriangleQuality. Well-shaped triangles are far from the threshold,
	// so their crease classification is unchanged.
	const Scalar area2 = (p1 - p0).cross(p2 - p0).squaredNorm() * Scalar(0.25);
	if (area2 <= 0)
		return Scalar(0);
	return (8 * area2) / (a * b * c * sum);
}

// ---------------------------------------------------------------------------
// IdealValence — the single target-valence rule: 4 on boundary/locked, 6 interior.
// ---------------------------------------------------------------------------
inline int IdealValence(bool isBoundaryOrLocked)
{
	return isBoundaryOrLocked ? 4 : 6;
}

// ---------------------------------------------------------------------------
// FlipImprovesValence
// Does flipping the edge (a,b)->(c,d) reduce the summed squared valence
// deviation?  After a flip a and b each lose a neighbour while c and d each gain
// one.  oX is the ideal valence of vertex X (see IdealValence).
// ---------------------------------------------------------------------------
inline bool FlipImprovesValence(int va, int vb, int vc, int vd,
                                int oa, int ob, int oc, int od)
{
	auto sq = [](int x) { return x * x; };
	const int before = sq(va - oa) + sq(vb - ob) + sq(vc - oc) + sq(vd - od);
	const int after = sq((va - 1) - oa) + sq((vb - 1) - ob) + sq((vc + 1) - oc) + sq((vd + 1) - od);
	return before > after;
}

// ---------------------------------------------------------------------------
// FairMesh
// Fair `points` in place by minimising a k-harmonic energy with a Laplacian L:
// k=1 is the membrane energy ||L x|| (area minimising, the classic umbrella
// relaxation), k=2 is the thin-plate energy ||L^2 x|| (curvature minimising —
// PMP's minimize_curvature).  Vertices flagged in `locked` are held fixed
// (Dirichlet), the rest solved via Eigen SimplicialLDLT.  Deterministic
// (vertex-index ordering, fixed sparsity).  Returns false (positions left
// unchanged) if there are no free vertices or the SPD solve fails, so callers can
// fall back gracefully.
//
// `cotangent` selects the edge weights of L: false = the uniform (unit) graph
// Laplacian (used by the k=1 tangential relaxation, which relies on uniform
// weights to place free vertices at neighbour barycentres and equalise density);
// true = per-edge cotangent weights (PMP/Liepa minimize_curvature), which remove
// tessellation-density bias in the thin-plate solve. Negative cotangent weights
// (obtuse triangles) are clamped to a small epsilon to preserve SPD.
//
// `tris` are triangles as local vertex-index triples; `points`/`locked` are
// indexed by the same local vertex id.
bool FairMesh(std::vector<Eigen::Matrix<double, 3, 1>>& points,
              const std::vector<Eigen::Matrix<int, 3, 1>>& tris,
              const std::vector<bool>& locked,
              unsigned k = 2,
              bool cotangent = false);

} // namespace detail
} // namespace halfmesh

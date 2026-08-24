/*
* MeshRemesh.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// RemeshData and TestHausdorff are placed in an anonymous namespace inside
// namespace halfmesh to hide them from other TUs.

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleBVH.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Util/Maths.h>
#include <halfmesh/Util/Geometry.h>
#include <halfmesh/Util/Accumulator.h>
#include <halfmesh/Types.h>

#include "MeshRemeshShared.h"
#include "ParallelFor.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <BS_thread_pool.hpp>

namespace halfmesh {
namespace {

// Triangle shape-quality measures are shared with the hole-filler.
using detail::TriangleQuality;
using detail::TriangleQualityRadii;

// Pool-backed parallel-for (see src/ParallelFor.h).
using detail::ParallelForPool;

// ---------------------------------------------------------------------------
// TestHausdorff
// Return true iff every vertex in `vertices` is within `maxDist` of the
// original surface (via the BVH).  If newNormal != NULL also check the normal
// alignment of the nearest face with *newNormal (cosine >= 0.7).
//
// The query is TOLERANCE-BOUNDED: the search radius is seeded with maxDist so
// branch-and-bound prunes every subtree farther than the tolerance from the
// root, instead of resolving the exact global nearest and only then comparing.
// The bound is nudged up by one ULP (nextafter towards +inf) so a surface point
// at EXACTLY maxDist is still found and passes the strict `dist > maxDist`
// reject below — keeping the accept/reject decision identical to the old
// unbounded query while touching far fewer nodes.
// ---------------------------------------------------------------------------
bool TestHausdorff(const TriangleBVH& bvh,
                   const std::vector<Mesh::Vertex>& vertices,
                   Mesh::Type maxDist,
                   const Mesh::Vertex* newNormal = nullptr)
{
	const Mesh::Type searchRadius =
	    std::nextafter(maxDist, std::numeric_limits<Mesh::Type>::infinity());
	for (const Mesh::Vertex& vertex : vertices) {
		const TriangleBVH::NearestNeighbor nn = bvh.NearestPoint(vertex, searchRadius);
		if (!nn.IsValid())
			return false;
		if (nn.dist > maxDist)
			return false;
		if (newNormal != nullptr) {
			// Compare the COSINE of the angle between the new and nearest-original
			// face normals against 0.7 (== cos 45deg), not the raw dot: both normals
			// are unnormalized cross products whose magnitudes carry twice the face
			// area, so a raw dot scales with area^2 and makes the gate scale-dependent
			// (rejects every collapse on small meshes, never fires on large ones).
			// Zero-norm guard mirrors CheckFacesAfterCollapse.
			const Mesh::Vertex faceNormal = bvh.GetMesh().ComputeFaceNormal(nn.idxFace);
			const Mesh::Type normSq = newNormal->squaredNorm() * faceNormal.squaredNorm();
			if (normSq <= Mesh::Type(0) || newNormal->dot(faceNormal) / std::sqrt(normSq) < Mesh::Type(0.7))
				return false;
		}
	}
	return true;
}

// ---------------------------------------------------------------------------
// RemeshData
// Internal helper that drives all remeshing passes.
// ---------------------------------------------------------------------------
class RemeshData
{
	public:
	RemeshData(Mesh& _mesh, Mesh::RemeshParams& _params) :
	    originalMesh(StripToGeometry(_mesh)),
	    mesh(_mesh),
	    m(mesh.halfMesh),
	    params(_params),
	    bvh(originalMesh) {}

	void TagCreaseEdges(bool forceTag = false);
	void ClassifyFeatureVertices();
	void BuildSizingField();
	unsigned SplitLongEdges();
	unsigned CollapseShortEdges();
	unsigned ImproveValence();
	void VertexCoordLaplacian(int iterations = 1,
	                          Mesh::Type delta = 0.2f,
	                          bool cotangentCheck = false);
	void TangentialSmoothing(int iterations, Mesh::Type delta);
	void ProjectVerticesToSurface();

	private:
	bool CheckVertexCanMoveOnCollapse(HalfMesh::HIndex iHe, bool relaxed);
	bool CheckFacesAfterCollapse(HalfMesh::HIndex iHe,
	                             const TPoint3<Mesh::Type>& newPos,
	                             bool relaxed);
	bool CheckCollapseFacesAroundFirstVertex(HalfMesh::HIndex iHe,
	                                         Mesh::Vertex& newPos,
	                                         unsigned relaxed);
	bool TestEdgeCollapse(HalfMesh::HIndex iHe,
	                      Mesh::Vertex& newPos,
	                      bool relaxed = false);
	Mesh::VIndex IdealValence(Mesh::VIndex iV);
	bool TestEdgeFlip(const HalfMesh::HIndex iHe, float cosAngleNormals);
	// Per-edge length thresholds: uniform unless adaptive sizing is active, in
	// which case they follow the per-vertex sizing field (PMP is_too_long/short).
	Mesh::Type SplitThreshold(Mesh::VIndex a, Mesh::VIndex b) const;
	Mesh::Type CollapseThreshold(Mesh::VIndex a, Mesh::VIndex b) const;

	// The BVH/Hausdorff gate only ever reads vertices+faces of the immutable
	// input, so the reference copy strips the (largest) half-edge connectivity
	// and the per-face texcoords: geometry is byte-identical, peak RAM is not.
	static Mesh StripToGeometry(const Mesh& src)
	{
		Mesh geo;
		geo.vertices = src.vertices;
		geo.faces = src.faces;
		return geo;
	}

	// ---- inline mark/selection helpers (verbatim from original) ----
	void ResetVertexMarkDirty() { ++vMarkDirty; }
	void VertexMarkDirty(size_t i) { vSelectionDirty[i] = vMarkDirty; }
	bool IsVertexMarkedDirty(size_t i) const { return vSelectionDirty[i] == vMarkDirty; }

	void ResetFaceVertexMark() { ++fvMark; }
	void FaceVertexMark(size_t i) { fvSelection[i] = fvMark; }
	void FaceVertexUnmark(size_t i) { fvSelection[i] = fvMark - 1; }
	bool IsFaceVertexMarked(size_t i) const { return fvSelection[i] == fvMark; }

	private:
	const Mesh originalMesh;
	Mesh& mesh;
	HalfMesh& m;
	Mesh::RemeshParams& params;
	const TriangleBVH bvh; // over originalMesh; faster than the kd-tree for the gate/projection queries
	BS::light_thread_pool pool; // persistent worker pool for the whole remesh stage
	std::vector<Mesh::FIndex> projectHint; // per-vertex last nearest face (ProjectVerticesToSurface warm start; empty until first projection)
	std::vector<uint8_t> vSelectionDirty; // dirty flag per vertex
	std::vector<uint8_t> fvSelection; // selection per face-vertex (faces*3)
	std::vector<uint8_t> vFeatureDegree; // # incident feature edges per vertex (ClassifyFeatureVertices)
	std::vector<std::vector<Mesh::VIndex>> vFeatureNbrs; // feature-edge neighbours per vertex
	std::vector<float> sizing; // per-vertex target edge length (adaptive sizing; empty unless adapt)
	uint8_t vMarkDirty; // current "dirty" sentinel
	uint8_t fvMark; // current face-vertex "marked" sentinel
};

// ---------------------------------------------------------------------------
// TagCreaseEdges
// Mark face-vertices that lie on boundary or crease edges.
// ---------------------------------------------------------------------------
void RemeshData::TagCreaseEdges(bool /*forceTag*/)
{
	const Mesh::Type thQualityRadii = Mesh::Type(0.0001);
	fvSelection.clear();
	fvSelection.resize(mesh.faces.size() * 3, fvMark = 0);
	ResetFaceVertexMark();
	// Precompute the per-face normal and a per-face quality-pass byte once, in
	// parallel. The old marking scan recomputed the own-face normal ~6x per interior
	// face (3 own halfedges + 3 neighbour visits) and the adjacent quality per crease
	// candidate. Both are pure functions of the face vertices, so the cached values
	// are bitwise identical -> fvSelection marks and goldens are unchanged.
	const std::size_t nf = m.FSize();
	std::vector<Mesh::Vertex> faceNormal(nf);
	std::vector<uint8_t> qualityOk(nf);
	ParallelForPool(pool, nf, [&](std::size_t f) {
		const Mesh::Face& face = mesh.faces[f];
		faceNormal[f] = mesh.ComputeFaceNormal(static_cast<HalfMesh::FIndex>(f));
		const Mesh::Type q = TriangleQualityRadii(mesh.vertices[face[0]],
		                                          mesh.vertices[face[1]],
		                                          mesh.vertices[face[2]]);
		qualityOk[f] = q > thQualityRadii ? uint8_t(1) : uint8_t(0);
	});
	FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
		if (!qualityOk[idxFace])
			continue;
		for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
			if (m.EHeIsBoundary(iHe)) {
				FaceVertexMark(idxFace * 3 + m.FVertexIth(iHe, mesh.faces[idxFace]));
				continue;
			}
			// check the value of the scalar prod to the cos of the crease threshold
			const Mesh::FIndex idxFaceAdj = m.FAdjacent(iHe);
			const Mesh::Vertex& normal = faceNormal[idxFace];
			const Mesh::Vertex& adjNormal = faceNormal[idxFaceAdj];
			const Mesh::Type normalNormSq = normal.squaredNorm() * adjNormal.squaredNorm();
			if (normalNormSq <= 0)
				continue;
			const Mesh::Type cosAngle = normal.dot(adjNormal) / std::sqrt(normalNormSq);
			const Mesh::Type cosMaxAngle = -0.98f;
			if (cosAngle > params.thCreaseCosAngle || cosAngle < cosMaxAngle)
				continue;
			if (qualityOk[idxFaceAdj]) {
				const Mesh::Face& faceAdj = mesh.faces[idxFaceAdj];
				FaceVertexMark(idxFace * 3 + m.FVertexIth(iHe, mesh.faces[idxFace]));
				FaceVertexMark(idxFaceAdj * 3 + m.FVertexIth(m.HeTwin(iHe), faceAdj));
			}
		}
	}
}

// ---------------------------------------------------------------------------
// ClassifyFeatureVertices
// From the per-corner crease/boundary marks (TagCreaseEdges) derive, per vertex,
// the number of incident feature edges and the feature-edge neighbours.  Drives
// PMP-style corner locking and 1D crease sliding: a vertex with exactly two
// feature edges lies on a smooth feature curve (it may slide along it), while
// any other non-zero count is a corner / junction / endpoint (fully locked).
// ---------------------------------------------------------------------------
void RemeshData::ClassifyFeatureVertices()
{
	vFeatureDegree.assign(mesh.vertices.size(), 0);
	vFeatureNbrs.assign(mesh.vertices.size(), {});
	std::vector<bool> edgeDone(m.ESize(), false);
	FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
		const Mesh::Face& face = mesh.faces[idxFace];
		for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
			const HalfMesh::EIndex e = m.HeEdge(iHe);
			if (edgeDone[e])
				continue;
			edgeDone[e] = true;
			if (!IsFaceVertexMarked(idxFace * 3 + m.FVertexIth(iHe, face)))
				continue; // not a feature edge
			const Mesh::VIndex a = m.HeTailVertex(iHe);
			const Mesh::VIndex b = m.HeHeadVertex(iHe);
			++vFeatureDegree[a];
			++vFeatureDegree[b];
			vFeatureNbrs[a].push_back(b);
			vFeatureNbrs[b].push_back(a);
		}
	}
}

// ---------------------------------------------------------------------------
// BuildSizingField
// Curvature-adaptive per-vertex target edge length (PMP adaptive remeshing).
// Computed once on the input mesh; carried through split (midpoint average) and
// collapse (compaction).  High-curvature vertices get a shorter target, flat
// ones a longer target, each clamped to [min,max]_adaptive_mult * L.
// ---------------------------------------------------------------------------
void RemeshData::BuildSizingField()
{
	const size_t nv = mesh.vertices.size();
	// Discrete cotangent Laplace-Beltrami of the positions -> mean-curvature mag,
	// plus per-vertex angle-defect Gaussian curvature (accumulated in the same
	// face loop) so the sizing field can use the MAX absolute principal curvature
	// kappaMax = |H| + sqrt(max(0, H^2 - K)) (Dunyach/PMP), not |H| alone. Using
	// |H| alone over-coarsens saddles (kappa1 ~ -kappa2 => H ~ 0) and developable
	// regions (cylinder: |H| = 1/2r halves the true kappaMax = 1/r).
	std::vector<Mesh::Vertex> lap(nv, Mesh::Vertex::Zero());
	std::vector<Mesh::Type> area(nv, 0);
	std::vector<Mesh::Type> angleSum(nv, 0); // sum of incident triangle corner angles
	auto cot = [](const Mesh::Vertex& a, const Mesh::Vertex& b, const Mesh::Vertex& c) {
		// cotangent of the angle at vertex a (between a->b and a->c)
		const Mesh::Vertex e1 = b - a;
		const Mesh::Vertex e2 = c - a;
		const Mesh::Type cr = e1.cross(e2).norm();
		return cr > 0 ? e1.dot(e2) / cr : Mesh::Type(0);
	};
	auto cornerAngle = [](const Mesh::Vertex& a, const Mesh::Vertex& b, const Mesh::Vertex& c) {
		// interior angle at vertex a (between a->b and a->c), 0 if degenerate
		const Mesh::Vertex e1 = b - a;
		const Mesh::Vertex e2 = c - a;
		const Mesh::Type n = std::sqrt(e1.squaredNorm() * e2.squaredNorm());
		if (n <= 0)
			return Mesh::Type(0);
		return std::acos(CLAMP(e1.dot(e2) / n, Mesh::Type(-1), Mesh::Type(1)));
	};
	FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
		const Mesh::Face& f = mesh.faces[idxFace];
		const Mesh::Vertex& p0 = mesh.vertices[f[0]];
		const Mesh::Vertex& p1 = mesh.vertices[f[1]];
		const Mesh::Vertex& p2 = mesh.vertices[f[2]];
		const Mesh::Type thirdArea = mesh.ComputeFaceDoubleArea(idxFace) * Mesh::Type(0.5) / Mesh::Type(3);
		area[f[0]] += thirdArea;
		area[f[1]] += thirdArea;
		area[f[2]] += thirdArea;
		angleSum[f[0]] += cornerAngle(p0, p1, p2);
		angleSum[f[1]] += cornerAngle(p1, p2, p0);
		angleSum[f[2]] += cornerAngle(p2, p0, p1);
		const Mesh::Type c0 = cot(p0, p1, p2); // weight for opposite edge (1,2)
		const Mesh::Type c1 = cot(p1, p2, p0); // weight for opposite edge (2,0)
		const Mesh::Type c2 = cot(p2, p0, p1); // weight for opposite edge (0,1)
		lap[f[1]] += c0 * (p2 - p1);
		lap[f[2]] += c0 * (p1 - p2);
		lap[f[2]] += c1 * (p0 - p2);
		lap[f[0]] += c1 * (p2 - p0);
		lap[f[0]] += c2 * (p1 - p0);
		lap[f[1]] += c2 * (p0 - p1);
	}
	const Mesh::Type L = params.edgeMaxLength * Mesh::Type(0.75); // SetEdgeLength: edge_max = 4/3 L
	const Mesh::Type lo = L * params.minAdaptiveMult;
	const Mesh::Type hi = L * params.maxAdaptiveMult;
	Mesh::Type e = params.approxError;
	if (e <= 0)
		e = params.edgeMinLength * Mesh::Type(0.1);
	const Mesh::Type eps = std::numeric_limits<Mesh::Type>::epsilon();
	sizing.assign(nv, hi);
	for (size_t v = 0; v < nv; ++v) {
		if (area[v] <= 0)
			continue; // isolated / degenerate -> coarsest
		const Mesh::Type H = lap[v].norm() / (Mesh::Type(4) * area[v]); // |mean curvature|
		// Angle-defect Gaussian curvature. Reference total angle is 2*pi at an
		// interior vertex and pi at a boundary vertex (a boundary vertex spans only
		// a half-disk); using 2*pi there would give a spurious positive K that the
		// max(0, H^2-K) clamp silently neutralises, delivering no boundary gain.
		const Mesh::Type ref = m.VIsBoundary(static_cast<Mesh::VIndex>(v))
		                           ? static_cast<Mesh::Type>(M_PI)
		                           : static_cast<Mesh::Type>(2 * M_PI);
		const Mesh::Type K = (ref - angleSum[v]) / area[v];
		// Max absolute principal curvature (Dunyach/PMP): kappa1,2 = H +/- sqrt(H^2-K).
		const Mesh::Type kappaMax = H + std::sqrt(std::max(Mesh::Type(0), H * H - K));
		const Mesh::Type r = Mesh::Type(1) / std::max(kappaMax, eps);
		Mesh::Type h = (e < r) ? std::sqrt(Mesh::Type(6) * e * r - Mesh::Type(3) * e * e)
		                       : Mesh::Type(3) * e / std::sqrt(Mesh::Type(3));
		sizing[v] = std::min(std::max(h, lo), hi);
	}
	// Smooth the field over the one-ring (2 passes, uniform) so target sizes vary
	// gradually; values stay within [lo,hi].
	for (int it = 0; it < 2; ++it) {
		std::vector<Mesh::Type> acc(nv, 0);
		std::vector<int> cnt(nv, 0);
		FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
			for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
				const Mesh::VIndex a = m.HeTailVertex(iHe);
				const Mesh::VIndex b = m.HeHeadVertex(iHe);
				acc[a] += sizing[b];
				++cnt[a];
				acc[b] += sizing[a];
				++cnt[b];
			}
		}
		for (size_t v = 0; v < nv; ++v)
			if (cnt[v] > 0)
				sizing[v] = Mesh::Type(0.5) * sizing[v] + Mesh::Type(0.5) * (acc[v] / cnt[v]);
	}
}

// ---------------------------------------------------------------------------
// SplitThreshold / CollapseThreshold
// ---------------------------------------------------------------------------
Mesh::Type RemeshData::SplitThreshold(Mesh::VIndex a, Mesh::VIndex b) const
{
	if (!params.adapt || sizing.empty())
		return params.edgeMaxLength;
	return Mesh::Type(4) / Mesh::Type(3) * std::min(sizing[a], sizing[b]);
}

Mesh::Type RemeshData::CollapseThreshold(Mesh::VIndex a, Mesh::VIndex b) const
{
	if (!params.adapt || sizing.empty())
		return params.edgeMinLength;
	return Mesh::Type(4) / Mesh::Type(5) * std::min(sizing[a], sizing[b]);
}

// ---------------------------------------------------------------------------
// SplitLongEdges
// Subdivide every edge longer than edgeMaxLength by inserting a midpoint.
// ---------------------------------------------------------------------------
unsigned RemeshData::SplitLongEdges()
{
	// Incremental edge split (HalfMesh::ESplit): split each over-long edge in
	// place, avoiding the full HalfMesh rebuild the old batch subdivision needed.
	// A triangle with several long edges is subdivided one edge at a time (this
	// drops the old 2-edge diagonal-swap heuristic, so the exact triangulation
	// differs slightly). An edge longer than 2x the threshold yields halves that
	// are still over-long, so sweep until no split fires — each sweep halves the
	// worst-case length, giving O(log(len/th)) sweeps.
	const bool maintainSizing = params.adapt && !sizing.empty();
	const bool maintainHint = !projectHint.empty();
	unsigned numSplits = 0;
	for (unsigned sweepSplits = 1; sweepSplits != 0;) {
		sweepSplits = 0;
		const HalfMesh::EIndex sweepEdges = m.ESize();
		for (HalfMesh::EIndex iE = 0; iE < sweepEdges; ++iE) {
			const HalfMesh::HIndex iHe = m.EHalfedge(iE);
			const HalfMesh::VIndex a = m.HeTailVertex(iHe);
			const HalfMesh::VIndex b = m.HeHeadVertex(iHe);
			const Mesh::Type th = SplitThreshold(a, b);
			if ((mesh.vertices[a] - mesh.vertices[b]).squaredNorm() <= th * th)
				continue;
			[[maybe_unused]] const HalfMesh::VIndex mNew = m.ESplit(iE);
			ASSERT(mNew == static_cast<HalfMesh::VIndex>(mesh.vertices.size()));
			mesh.vertices.push_back((mesh.vertices[a] + mesh.vertices[b]) * Mesh::Type(0.5));
			if (maintainSizing)
				sizing.push_back(Mesh::Type(0.5) * (sizing[a] + sizing[b]));
			if (maintainHint)
				projectHint.push_back(math::NO_ID); // split-created vertex: cold projection next time
			++sweepSplits;
		}
		numSplits += sweepSplits;
	}
	if (numSplits == 0)
		return 0;

	// Sync the flat face list from the updated half-edge structure, then refresh
	// the per-face-vertex crease marks (recomputed from the new geometry) so the
	// following collapse pass sees a consistently sized fvSelection.
	mesh.faces.clear();
	m.FFaces(mesh.faces);
#if REMESH_DEBUG_OUTPUT
	mesh.Save("mesh_split_update.ply");
#endif
	TagCreaseEdges();
	return numSplits;
}

// ---------------------------------------------------------------------------
// CheckVertexCanMoveOnCollapse
// Return true iff the vertex at the head of iHe can be moved (crease check).
// ---------------------------------------------------------------------------
bool RemeshData::CheckVertexCanMoveOnCollapse(HalfMesh::HIndex iHe, bool relaxed)
{
	ASSERT(!m.HeIsBoundary(iHe));
	if (relaxed)
		return !m.VIsBoundary(m.HeVertex(iHe));
	unsigned numIncidentFeatures = 0;
	// Per-call scratch: which 1-ring neighbours have already been counted as a
	// feature endpoint (was a shared v_selection_/v_mark_ mark; made call-local
	// so TestEdgeCollapse is a pure read and safe to evaluate in parallel).
	std::vector<HalfMesh::VIndex> seen;
	const auto IsSeen = [&](HalfMesh::VIndex v) {
		return std::find(seen.begin(), seen.end(), v) != seen.end();
	};
	const Mesh::Type cosMaxAngleBetweenAdjFaces = 0.9f;
	const Mesh::FIndex idxFace = m.HeFace(iHe);
	const Mesh::Face& face = mesh.faces[idxFace];
	const Mesh::Vertex edgeVector =
	    (mesh.vertices[m.HeTailVertex(iHe)] - mesh.vertices[m.HeHeadVertex(iHe)]).normalized();
	for (HalfMesh::HIndex iHeAdj : m.VOutgoingHalfedges(m.HeVertex(iHe))) {
		if (m.HeIsBoundary(iHeAdj))
			continue;
		const Mesh::FIndex idxFaceAdj = m.HeFace(iHeAdj);
		const Mesh::Face& faceAdj = mesh.faces[idxFaceAdj];
		if (IsFaceVertexMarked(idxFaceAdj * 3 + m.FVertexIth(iHeAdj, faceAdj)) && !IsSeen(m.HeHeadVertex(iHeAdj))) {
			seen.push_back(m.HeHeadVertex(iHeAdj));
			const Mesh::Vertex movingEdgeVector0 =
			    (mesh.vertices[m.HeHeadVertex(iHeAdj)] - mesh.vertices[m.HeTailVertex(iHeAdj)]).normalized();
			if (std::abs(movingEdgeVector0.dot(edgeVector)) < cosMaxAngleBetweenAdjFaces || !IsFaceVertexMarked(idxFace * 3 + m.FVertexIth(iHe, face)))
				return false;
			++numIncidentFeatures;
		}
		const HalfMesh::HIndex iHeAdjPrev = m.HePrev(iHeAdj);
		const Mesh::FIndex idxFaceAdjPrev = m.HeFace(iHeAdjPrev);
		const Mesh::Face& faceAdjPrev = mesh.faces[idxFaceAdjPrev];
		if (IsFaceVertexMarked(idxFaceAdjPrev * 3 + m.FVertexIth(iHeAdjPrev, faceAdjPrev)) && !IsSeen(m.HeTailVertex(iHeAdjPrev))) {
			seen.push_back(m.HeTailVertex(iHeAdjPrev));
			const Mesh::Vertex movingEdgeVector1 =
			    (mesh.vertices[m.HeTailVertex(iHeAdjPrev)] - mesh.vertices[m.HeHeadVertex(iHeAdjPrev)]).normalized();
			if (std::abs(movingEdgeVector1.dot(edgeVector)) < cosMaxAngleBetweenAdjFaces || !IsFaceVertexMarked(idxFace * 3 + m.FVertexIth(iHe, face)))
				return false;
			++numIncidentFeatures;
		}
	}
	return numIncidentFeatures < 3;
}

// ---------------------------------------------------------------------------
// CheckFacesAfterCollapse
// Verify that collapsing the vertex at HeVertex(iHe) to newPos does not
// invert, degenerate, or move faces too far from the original surface.
// ---------------------------------------------------------------------------
bool RemeshData::CheckFacesAfterCollapse(HalfMesh::HIndex iHe,
                                         const TPoint3<Mesh::Type>& newPos,
                                         bool relaxed)
{
	const Mesh::Type cosMaxAngleBetweenOldNewFaces = 0.7f;
	for (HalfMesh::HIndex iHeAdj : m.VOutgoingHalfedges(m.HeVertex(iHe))) {
		if (iHe != iHeAdj && !m.HeIsBoundary(iHeAdj)) {
			const HalfMesh::VIndex v0 = m.HeTailVertex(iHeAdj);
			const HalfMesh::VIndex v1 = m.HeHeadVertex(iHeAdj);
			const HalfMesh::VIndex v2 = m.HeHeadVertex(m.HeNext(iHeAdj));

			if (v1 == m.HeHeadVertex(iHe) || v2 == m.HeHeadVertex(iHe))
				continue;

			// check on new face quality
			{
				const Mesh::Type newQuality =
				    TriangleQuality(newPos, mesh.vertices[v1], mesh.vertices[v2]);
				const Mesh::Type oldQuality =
				    TriangleQuality(mesh.vertices[v0], mesh.vertices[v1], mesh.vertices[v2]);
				if (newQuality <= Mesh::Type(0.5) * oldQuality)
					return false;
			}

			// we prevent collapse that makes edges too long (except for cross);
			// bound by the local split threshold (== edgeMaxLength when not
			// adaptive) so adaptive mode can coarsen flat regions whose sizing
			// target exceeds the uniform cap
			if (!relaxed && ((newPos - mesh.vertices[v1]).norm() > SplitThreshold(m.HeVertex(iHe), v1) || (newPos - mesh.vertices[v2]).norm() > SplitThreshold(m.HeVertex(iHe), v2)))
				return false;

			const auto oldNormal = mesh.ComputeFaceNormal(m.HeFace(iHeAdj));
			const auto newNormal =
			    Mesh::ComputeTriangleNormal(newPos, mesh.vertices[v1], mesh.vertices[v2]);
			const Mesh::Type oldNormalNormSq = oldNormal.squaredNorm();
			const Mesh::Type newNormalNormSq = newNormal.squaredNorm();
			if (newNormalNormSq <= 0 || (oldNormalNormSq > 0 && oldNormal.dot(newNormal) / std::sqrt(oldNormalNormSq * newNormalNormSq) < cosMaxAngleBetweenOldNewFaces))
				return false;

			// check on new face distance from original mesh
			if (params.checkSurfDist) {
				const std::vector<Mesh::Vertex> points{
				    (mesh.vertices[v1] + newPos) * Mesh::Type(0.5),
				    (mesh.vertices[v2] + newPos) * Mesh::Type(0.5),
				    newPos,
				};
				if (!TestHausdorff(bvh, points, params.maxSurfDist) || !TestHausdorff(bvh, {Mesh::Vertex{(newPos + mesh.vertices[v1] + mesh.vertices[v2]) / Mesh::Type(3)}}, params.maxSurfDist, &newNormal))
					return false;
			}
		}
	}
	return true;
}

// ---------------------------------------------------------------------------
// CheckCollapseFacesAroundFirstVertex
//  - relaxed: 0 - full check, 1 - full check only move, 2 - partial check
// ---------------------------------------------------------------------------
bool RemeshData::CheckCollapseFacesAroundFirstVertex(HalfMesh::HIndex iHe,
                                                     Mesh::Vertex& newPos,
                                                     unsigned relaxed)
{
	// check crease-move-ability
	const bool moveable0 = CheckVertexCanMoveOnCollapse(iHe, relaxed > 1);
	const bool moveable1 = CheckVertexCanMoveOnCollapse(m.HeTwin(iHe), relaxed > 1);

	// if both movable => go to midpoint, else collapse on movable one
	if (!moveable0 && !moveable1)
		return false;

	newPos = (mesh.vertices[m.HeTailVertex(iHe)] * static_cast<int>(moveable1) + mesh.vertices[m.HeHeadVertex(iHe)] * static_cast<int>(moveable0)) / (static_cast<int>(moveable0) + static_cast<int>(moveable1));

	return CheckFacesAfterCollapse(iHe, newPos, relaxed > 0) && CheckFacesAfterCollapse(m.HeTwin(iHe), newPos, relaxed > 0);
}

// ---------------------------------------------------------------------------
// TestEdgeCollapse
// Return true iff the edge represented by iHe should be collapsed.
// ---------------------------------------------------------------------------
bool RemeshData::TestEdgeCollapse(HalfMesh::HIndex iHe,
                                  Mesh::Vertex& newPos,
                                  bool relaxed)
{
	const Mesh::Type minAreaPercentage = 0.01f;
	const Mesh::Type minLenghPercentage = 0.1f;
	if (m.EHeIsBoundary(iHe))
		return false;
	if (IsVertexMarkedDirty(m.HeTailVertex(iHe)) || IsVertexMarkedDirty(m.HeHeadVertex(iHe)))
		return false;
	if (relaxed)
		return CheckCollapseFacesAroundFirstVertex(iHe, newPos, relaxed ? 1 : 0);

	const Mesh::Type th = CollapseThreshold(m.HeTailVertex(iHe), m.HeHeadVertex(iHe));
	const Mesh::Type dist =
	    (mesh.vertices[m.HeTailVertex(iHe)] - mesh.vertices[m.HeHeadVertex(iHe)]).norm();
	const Mesh::Type area = mesh.ComputeFaceDoubleArea(m.HeFace(iHe)) * 0.5f;
	return (dist < th || area < SQUARE(params.edgeMinLength) * minAreaPercentage) && CheckCollapseFacesAroundFirstVertex(iHe, newPos, dist < th * minLenghPercentage ? 2 : (relaxed ? 1 : 0));
}

// ---------------------------------------------------------------------------
// CollapseShortEdges
// Collapse short edges (< edgeMinLength) and low-valence cross edges.
// ---------------------------------------------------------------------------
unsigned RemeshData::CollapseShortEdges()
{
	// dirty marks gate the greedy independent-set selection (serial)
	vSelectionDirty.clear();
	vSelectionDirty.resize(mesh.vertices.size(), vMarkDirty = 0);
	ResetVertexMarkDirty();

	struct Cand
	{
		HalfMesh::HIndex he;
		Mesh::Vertex newPos;
		bool valid;
	};
	const std::size_t nf = m.FSize();
	std::vector<Cand> cand(nf);
	std::vector<std::tuple<HalfMesh::VIndex, HalfMesh::VIndex, Mesh::Vertex>> collapseEdges;

	// ---- short edges: parallel geometric predicate (pure reads) ----
	ParallelForPool(pool, nf, [&](std::size_t f) {
		Cand c{0, Mesh::Vertex{}, false};
		const HalfMesh::FIndex idxFace = static_cast<HalfMesh::FIndex>(f);
		for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
			Mesh::Vertex newPos;
			if (TestEdgeCollapse(iHe, newPos) && m.EIsCollapseValidTopologically(m.HeEdge(iHe)) && m.EIsCollapseValidGeometrically(m.HeEdge(iHe), newPos, mesh.vertices)) {
				c = {iHe, newPos, true};
				break;
			}
		}
		cand[f] = c;
	});
	// serial greedy select (dirty gate replicates the original face-order scan)
	for (std::size_t f = 0; f < nf; ++f) {
		if (!cand[f].valid)
			continue;
		const Mesh::VIndex v0 = m.HeTailVertex(cand[f].he);
		const Mesh::VIndex v1 = m.HeHeadVertex(cand[f].he);
		if (IsVertexMarkedDirty(v0) || IsVertexMarkedDirty(v1))
			continue;
		VertexMarkDirty(v0);
		VertexMarkDirty(v1);
		collapseEdges.emplace_back(v0, v1, cand[f].newPos);
	}

	// ---- cross edges: parallel predicate over the post-short dirty snapshot ----
	ParallelForPool(pool, nf, [&](std::size_t f) {
		Cand c{0, Mesh::Vertex{}, false};
		const HalfMesh::FIndex idxFace = static_cast<HalfMesh::FIndex>(f);
		for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
			const Mesh::VIndex iV = m.HeVertex(iHe);
			if (m.VIsBoundary(iV) || IsVertexMarkedDirty(iV))
				continue;
			const Mesh::FIndex numFaces = m.VFaceDegree(iV);
			if (numFaces != 4 && numFaces != 3)
				continue;
			bool dirty = false;
			for (Mesh::VIndex iVN : m.VAdjacentVertices(iV)) {
				if (IsVertexMarkedDirty(iVN)) {
					dirty = true;
					break;
				}
			}
			if (dirty)
				continue;
			Mesh::Vertex newPos;
			if (TestEdgeCollapse(iHe, newPos, true) && m.EIsCollapseValidTopologically(m.HeEdge(iHe)) && m.EIsCollapseValidGeometrically(m.HeEdge(iHe), newPos, mesh.vertices)) {
				c = {iHe, newPos, true};
				break;
			}
		}
		cand[f] = c;
	});
	// serial greedy select (re-check dirty incl. cross-accumulated marks)
	for (std::size_t f = 0; f < nf; ++f) {
		if (!cand[f].valid)
			continue;
		const HalfMesh::HIndex iHe = cand[f].he;
		const Mesh::VIndex iV = m.HeVertex(iHe);
		if (IsVertexMarkedDirty(iV))
			continue;
		bool dirty = false;
		for (Mesh::VIndex iVN : m.VAdjacentVertices(iV)) {
			if (IsVertexMarkedDirty(iVN)) {
				dirty = true;
				break;
			}
		}
		if (dirty)
			continue;
		const Mesh::VIndex v0 = m.HeTailVertex(iHe);
		const Mesh::VIndex v1 = m.HeHeadVertex(iHe);
		VertexMarkDirty(v0);
		VertexMarkDirty(v1);
		collapseEdges.emplace_back(v0, v1, cand[f].newPos);
	}
	// collapse edges
	//
	// Each ERemove swap-pops the removed vertex slot (the last vertex is moved
	// into it), which relabels one vertex.  Queued collapses still reference the
	// pre-collapse indices, so they must track that relabelling.  Rather than
	// rescanning the whole queue per collapse (the old O(collapses^2) hot loop),
	// keep a logical<->physical index map updated in O(1) per collapse:
	//   phys[logical] = current physical index (NO_ID once removed)
	//   logl[physical] = logical id of the vertex currently in that slot
	// Queued endpoints are stored as logical ids (== physical at enqueue time).
	const size_t nv0 = mesh.vertices.size();
	std::vector<Mesh::VIndex> phys(nv0), logl(nv0);
	for (size_t i = 0; i < nv0; ++i)
		phys[i] = logl[i] = static_cast<Mesh::VIndex>(i);

	mesh.faces.clear();
	FOREACH (idxEdgePoint, collapseEdges) {
		const auto& edgePoint = collapseEdges[idxEdgePoint];
		const Mesh::VIndex pa = phys[std::get<0>(edgePoint)];
		const Mesh::VIndex pb = phys[std::get<1>(edgePoint)];
		if (pa == math::NO_ID || pb == math::NO_ID || pa == pb)
			continue;
		const Mesh::EIndex iE = m.EEdge(pa, pb);
		if (iE == math::NO_ID || !m.EIsCollapseValidTopologically(iE) || !m.EIsCollapseValidGeometrically(iE, std::get<2>(edgePoint), mesh.vertices))
			continue;
		HalfMesh::RemovedData removedData;
		const Mesh::VIndex vertexMoved = m.ERemove(iE, removedData);
		ASSERT(removedData.numVerts == 1u);
		const Mesh::VIndex oldIdxVertex = mesh.vertices.size() - 1;
		const Mesh::VIndex newIdxVertex = removedData.verts[0];
		// mirror the half-edge swap-pop on the parallel per-vertex arrays
		mesh.vertices[newIdxVertex] = mesh.vertices.back();
		mesh.vertices.pop_back();
		if (!sizing.empty()) {
			sizing[newIdxVertex] = sizing.back();
			sizing.pop_back();
		}
		if (!projectHint.empty()) {
			// mirror the vertex swap-pop on the warm-start hints (indices reference
			// the immutable originalMesh, so they stay valid across collapses)
			projectHint[newIdxVertex] = projectHint.back();
			projectHint.pop_back();
		}
		// update the index maps for the same swap-pop: the vertex that occupied
		// newIdxVertex is gone; the last vertex (oldIdxVertex) now lives there.
		phys[logl[newIdxVertex]] = math::NO_ID;
		if (newIdxVertex != oldIdxVertex) {
			const Mesh::VIndex movedLogical = logl[oldIdxVertex];
			logl[newIdxVertex] = movedLogical;
			phys[movedLogical] = newIdxVertex;
		}
		logl.pop_back();
		mesh.vertices[vertexMoved] = std::get<2>(edgePoint);
	}
	m.FFaces(mesh.faces);
	return collapseEdges.size();
}

// ---------------------------------------------------------------------------
// IdealValence
// Returns target valence for a vertex: 4 on boundary, 6 interior.
// ---------------------------------------------------------------------------
Mesh::VIndex RemeshData::IdealValence(Mesh::VIndex iV)
{
	return detail::IdealValence(m.VIsBoundary(iV));
}

// ---------------------------------------------------------------------------
// TestEdgeFlip
// Return true iff the edge should be flipped to improve valence / quality.
// ---------------------------------------------------------------------------
bool RemeshData::TestEdgeFlip(const HalfMesh::HIndex iHe, float cosAngleNormals)
{
	if (m.EHeIsBoundary(iHe))
		return false;
	const Mesh::FIndex idxFace = m.HeFace(iHe);
	if (IsFaceVertexMarked(idxFace * 3 + m.FVertexIth(iHe, mesh.faces[idxFace])))
		return false;

	// Stencil: iV0/iV2 are the shared-edge endpoints (they LOSE the edge on a
	// flip, valence -1); iV1/iV3 are the opposite vertices that GAIN the new
	// diagonal (valence +1). The old code decremented all four, mis-scoring every
	// flip (a flip fixing a valence-5 opposite vertex was scored as making it
	// worse). Score with the shared squared-deviation rule (detail::FlipImproves-
	// Valence), which encodes exactly this +1/-1 accounting.
	const Mesh::VIndex iV0 = m.HeTailVertex(iHe);
	const Mesh::VIndex iV1 = m.HeVertex(m.HePrev(iHe));
	const Mesh::VIndex iV2 = m.HeHeadVertex(iHe);
	const Mesh::VIndex iV3 = m.HeVertex(m.HePrev(m.HeTwin(iHe)));

	// va/vb: endpoints (-1); vc/vd: opposite pair (+1)
	const int va = static_cast<int>(m.VDegree(iV0));
	const int vb = static_cast<int>(m.VDegree(iV2));
	const int vc = static_cast<int>(m.VDegree(iV1));
	const int vd = static_cast<int>(m.VDegree(iV3));
	const int oa = static_cast<int>(IdealValence(iV0));
	const int ob = static_cast<int>(IdealValence(iV2));
	const int oc = static_cast<int>(IdealValence(iV1));
	const int od = static_cast<int>(IdealValence(iV3));
	auto sq = [](int x) { return x * x; };
	const int oldDist = sq(va - oa) + sq(vb - ob) + sq(vc - oc) + sq(vd - od);
	const int newDist = sq((va - 1) - oa) + sq((vb - 1) - ob) + sq((vc + 1) - oc) + sq((vd + 1) - od);
	const bool valenceImproves = detail::FlipImprovesValence(va, vb, vc, vd, oa, ob, oc, od);
	ASSERT(valenceImproves == (newDist < oldDist));

	const Mesh::Vertex& v0 = mesh.vertices[iV0];
	const Mesh::Vertex& v1 = mesh.vertices[iV1];
	const Mesh::Vertex& v2 = mesh.vertices[iV2];
	const Mesh::Vertex& v3 = mesh.vertices[iV3];

	const Mesh::Vertex oldNormal0 = Mesh::ComputeTriangleNormal(v1, v0, v2).normalized();
	const Mesh::Vertex oldNormal1 = Mesh::ComputeTriangleNormal(v3, v2, v0).normalized();
	const Mesh::Vertex newNormal0 = Mesh::ComputeTriangleNormal(v0, v3, v1).normalized();
	const Mesh::Vertex newNormal1 = Mesh::ComputeTriangleNormal(v2, v1, v3).normalized();
	if (oldNormal0.dot(newNormal0) < cosAngleNormals)
		return false;
	if (oldNormal0.dot(newNormal1) < cosAngleNormals)
		return false;
	if (oldNormal1.dot(newNormal0) < cosAngleNormals)
		return false;
	if (oldNormal1.dot(newNormal1) < cosAngleNormals)
		return false;

	const Mesh::Type oldQuality =
	    std::min(TriangleQuality(v0, v2, v3), TriangleQuality(v0, v1, v2));
	const Mesh::Type newQuality =
	    std::min(TriangleQuality(v0, v1, v3), TriangleQuality(v2, v3, v1));

	return (valenceImproves && newQuality >= oldQuality * 0.5f) || (newDist == oldDist && newQuality > oldQuality) || newQuality > oldQuality * 1.5f;
}

// ---------------------------------------------------------------------------
// ImproveValence
// Flip edges to optimise vertex valence and triangle quality.
// Parallel phase: per-face flip decision (pure reads, incl. Hausdorff).
// Serial phase: apply a conflict-free (independent) subset via touched-vertex set.
// ---------------------------------------------------------------------------
unsigned RemeshData::ImproveValence()
{
	const float cosAngleNormals = std::cos(D2R(5.f));
	const std::size_t nf = m.FSize();

	// ---- parallel: per-face first qualifying flip (pure reads, incl. Hausdorff) ----
	struct FlipCand
	{
		HalfMesh::HIndex he;
		bool want;
	};
	std::vector<FlipCand> cand(nf);
	ParallelForPool(pool, nf, [&](std::size_t f) {
		FlipCand c{0, false};
		const HalfMesh::FIndex idxFaceA = static_cast<HalfMesh::FIndex>(f);
		for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFaceA)) {
			if (!m.EIsFlipValid(m.HeEdge(iHe), mesh.vertices) || !TestEdgeFlip(iHe, cosAngleNormals))
				continue;
			const Mesh::HIndex iHa2 = m.HeNext(m.HeNext(iHe));
			const Mesh::HIndex iHb0 = m.HeTwin(iHe);
			const Mesh::HIndex iHb2 = m.HeNext(m.HeNext(iHb0));
			if (!params.checkSurfDist || TestHausdorff(bvh, {Mesh::Vertex{(mesh.vertices[m.HeVertex(iHa2)] + mesh.vertices[m.HeVertex(iHb2)]) / 2}}, params.maxSurfDist)) {
				c = {iHe, true};
				break;
			}
		}
		cand[f] = c;
	});

	// ---- serial: apply a conflict-free (independent) subset ----
	// An applied flip touches its 4 stencil vertices; any later candidate sharing
	// a touched vertex is skipped (its precomputed decision is now stale) and left
	// for the next remesh iteration. Untouched candidates have unchanged local
	// topology, so the precomputed decision stays valid.
	unsigned numFlips = 0;
	std::vector<uint8_t> touched(mesh.vertices.size(), 0u);
	for (std::size_t f = 0; f < nf; ++f) {
		if (!cand[f].want)
			continue;
		const HalfMesh::HIndex iHe = cand[f].he;
		const Mesh::HIndex iHa2 = m.HeNext(m.HeNext(iHe));
		const Mesh::HIndex iHb0 = m.HeTwin(iHe);
		const Mesh::HIndex iHb2 = m.HeNext(m.HeNext(iHb0));
		const Mesh::VIndex iV0 = m.HeVertex(iHe);
		const Mesh::VIndex iV2 = m.HeVertex(iHb0);
		const Mesh::VIndex iV1 = m.HeVertex(iHa2);
		const Mesh::VIndex iV3 = m.HeVertex(iHb2);
		if (touched[iV0] || touched[iV1] || touched[iV2] || touched[iV3])
			continue;
		const Mesh::FIndex idxFaceA = static_cast<Mesh::FIndex>(f);
		ASSERT(m.HeFace(iHa2) == idxFaceA);
		const Mesh::FIndex idxFaceB = m.HeFace(iHb2);
		Mesh::Face& faceA = mesh.faces[idxFaceA];
		Mesh::Face& faceB = mesh.faces[idxFaceB];
		const Mesh::VIndex idxFaceAVertex0 = m.FVertexIth(iHe, faceA);
		const Mesh::VIndex idxFaceBVertex2 = m.FVertexIth(iHb0, faceB);
		const Mesh::VIndex idxFaceAVertex1 = m.FVertexIth(iHa2, faceA);
		const Mesh::VIndex idxVertex1 = faceA[idxFaceAVertex1];
		const Mesh::VIndex idxFaceBVertex3 = m.FVertexIth(iHb2, faceB);
		const Mesh::VIndex idxVertex3 = faceB[idxFaceBVertex3];
		const bool creaseFa = IsFaceVertexMarked(idxFaceA * 3 + idxFaceAVertex1);
		const bool creaseFb = IsFaceVertexMarked(idxFaceB * 3 + idxFaceBVertex3);
		m.EFlip(m.HeEdge(iHe));
		faceA[idxFaceAVertex0] = idxVertex3;
		faceB[idxFaceBVertex2] = idxVertex1;
		if (creaseFa)
			FaceVertexMark(idxFaceB * 3 + idxFaceBVertex2);
		else
			FaceVertexUnmark(idxFaceB * 3 + idxFaceBVertex2);
		if (creaseFb)
			FaceVertexMark(idxFaceA * 3 + idxFaceAVertex0);
		else
			FaceVertexUnmark(idxFaceA * 3 + idxFaceAVertex0);
		touched[iV0] = touched[iV1] = touched[iV2] = touched[iV3] = 1u;
		++numFlips;
	}
	return numFlips;
}

// ---------------------------------------------------------------------------
// VertexCoordLaplacian
// Tangential Laplacian smoothing (uniform or cotangent weights).
// ---------------------------------------------------------------------------
void RemeshData::VertexCoordLaplacian(int iterations,
                                      Mesh::Type delta,
                                      bool cotangentCheck)
{
	std::vector<bool> fixedVertices(mesh.vertices.size(), false);
	FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
		const Mesh::Face& face = mesh.faces[idxFace];
		for (int i = 0; i < 3; ++i) {
			if (!fixedVertices[face[i]] && (m.VIsBoundary(face[i]) || IsFaceVertexMarked(idxFace * 3 + i)))
				fixedVertices[face[i]] = true;
		}
	}
	typedef WeightedAccumulator<Mesh::Vertex> LaplacianInfo;
	std::vector<LaplacianInfo> lpis(mesh.vertices.size());
	// Upper clamp for cotangent weights: cot(1deg) ~ 57.29 (only used when the
	// cotangent-weight path is enabled).
	const float maxCot = 1.f / std::tan(D2R(1.f));
	for (int iter = 0; iter < iterations; ++iter) {
		std::fill(lpis.begin(), lpis.end(), LaplacianInfo(Mesh::Vertex::Zero(), 0));
		FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
			[[maybe_unused]] const Mesh::Face& face = mesh.faces[idxFace];
			for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
				const Mesh::VIndex iV0 = m.HeTailVertex(iHe);
				const Mesh::VIndex iV1 = m.HeHeadVertex(iHe);
				float weight = 1.f;
				if (cotangentCheck) {
					// Cotangent weight via dot/cross (no acos+tan transcendentals),
					// clamped to [0, cot(1deg)]. A bare cotangent is negative for
					// obtuse opposite angles and unbounded for near-degenerate ones;
					// clamping keeps the one-ring weight sum stable (Botsch et al.,
					// Polygon Mesh Processing) instead of catapulting vertices.
					const Mesh::VIndex iV2 = m.HeHeadVertex(m.HeNext(iHe));
					weight = CLAMP(math::CotAngleBetweenRays(
					                   mesh.vertices[iV1] - mesh.vertices[iV2],
					                   mesh.vertices[iV0] - mesh.vertices[iV2]),
					               0.f, maxCot);
				}
				if (!fixedVertices[iV0])
					lpis[iV0].Add(mesh.vertices[iV1], weight);
				if (!fixedVertices[iV1])
					lpis[iV1].Add(mesh.vertices[iV0], weight);
			}
		}
		FOREACHIDX (Mesh::VIndex, idxVert, mesh.vertices) {
			LaplacianInfo& lpi = lpis[idxVert];
			if (lpi.Empty())
				continue;
			Mesh::Vertex& vertex = mesh.vertices[idxVert];
			lpi.Add(vertex, 1.f);
			const Mesh::Vertex newPos = lpi.Normalized();
			if (!params.checkSurfDist || TestHausdorff(bvh, {newPos}, params.maxSurfDist))
				vertex = vertex * (1 - delta) + newPos * delta;
		}
	}
}

// ---------------------------------------------------------------------------
// TangentialSmoothing
// PMP-style tangential relaxation: move each free vertex toward its one-ring
// centroid, with the displacement projected onto the vertex tangent plane
// (normal component removed so the move stays on the surface), repeated
// `iterations` times and Hausdorff-gated.  Boundary and crease vertices stay
// fixed (1D crease sliding is a separate upgrade).  Markedly more uniform than
// the single uniform-Laplacian pass.  Opt-in via params.smoothTangential.
// ---------------------------------------------------------------------------
void RemeshData::TangentialSmoothing(int iterations, Mesh::Type delta)
{
	const size_t nv = mesh.vertices.size();
	// Per-vertex smoothing mode: free (2D tangential), crease (1D slide), fixed.
	enum { FREE2D = 0,
		   CREASE1D = 1,
		   FIXED = 2 };
	std::vector<uint8_t> mode(nv, FREE2D);
	if (params.featureCorners) {
		// PMP-style: lock corners (feature degree != 2), let smooth crease
		// vertices (degree == 2) slide along the feature curve. Mesh boundary
		// stays fully fixed for now.
		ClassifyFeatureVertices();
		for (size_t v = 0; v < nv; ++v) {
			if (m.VIsBoundary(static_cast<Mesh::VIndex>(v)))
				mode[v] = FIXED;
			else if (vFeatureDegree[v] == 0)
				mode[v] = FREE2D;
			else if (vFeatureDegree[v] == 2)
				mode[v] = CREASE1D;
			else
				mode[v] = FIXED;
		}
	} else {
		// otherwise: lock the boundary and every crease-marked corner.
		FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
			const Mesh::Face& face = mesh.faces[idxFace];
			for (int i = 0; i < 3; ++i)
				if (m.VIsBoundary(face[i]) || IsFaceVertexMarked(idxFace * 3 + i))
					mode[face[i]] = FIXED;
		}
	}

	// In adaptive mode weight each one-ring neighbour by 1/sizing so the target is
	// the Dunyach graded barycenter, not the equal-length centroid: uniform weights
	// pull vertices toward equal edge lengths and fight the sizing field, undoing
	// the split/collapse grading every pass (later-iteration churn). Weight by
	// 1/L (NOT L: weighting by L drifts toward coarse regions, the wrong way — a
	// graded configuration is a smoothing fixed point only for w proportional to
	// 1/L). Uniform mode keeps weight 1 (bit-identical to before).
	const bool weightBySizing = params.adapt && !sizing.empty();

	typedef WeightedAccumulator<Mesh::Vertex> LaplacianInfo;
	std::vector<LaplacianInfo> lpis(nv);
	std::vector<Mesh::Vertex> normals(nv);
	std::vector<Mesh::Vertex> freePos(nv); // precomputed FREE2D targets
	std::vector<uint8_t> freeOk(nv); // FREE2D Hausdorff accept flags
	for (int iter = 0; iter < iterations; ++iter) {
		// Single fused face pass: accumulate area-weighted vertex normals
		// (unnormalised face normals carry the area) AND the one-ring neighbour
		// centroid for the 2D-free vertices (was two separate face traversals).
		//
		// Kept serial, but NOT for determinism: a per-vertex GATHER that sums each
		// vertex's incident-face terms in ascending-face order and replays its
		// centroid Add()s in the scatter's exact per-vertex order is byte-identical
		// to this scatter (the reduction order is reproduced, so it does not depend
		// on thread count) — verified against this loop over 535 smoothing
		// iterations (uniform/adaptive/feature paths), zero mismatches. So the
		// frozen golden does NOT force serial execution. It stays serial because
		// the gather measured NO net win at the shipping default: it needs a
		// per-call incident-face/neighbour CSR (connectivity changes every outer
		// remesh iteration, so it is rebuilt each call), whose random-access build
		// only amortises past ~8-20 smoothing passes. At smoothIterations=5 the
		// gather ran ~15% SLOWER on a dense UVSphere(128,192) (0.0141s vs 0.0122s
		// smoothSeconds); it wins (1.28x) only at heavy pass counts. Serial is the
		// faster variant at the default, so it is retained.
		std::fill(normals.begin(), normals.end(), Mesh::Vertex::Zero());
		std::fill(lpis.begin(), lpis.end(), LaplacianInfo(Mesh::Vertex::Zero(), 0));
		FOREACHRAWIDX (HalfMesh::FIndex, idxFace, m.FSize()) {
			const Mesh::Vertex fn = mesh.ComputeFaceNormal(idxFace);
			const Mesh::Face& face = mesh.faces[idxFace];
			normals[face[0]] += fn;
			normals[face[1]] += fn;
			normals[face[2]] += fn;
			for (HalfMesh::HIndex iHe : m.FAdjacentHalfedges(idxFace)) {
				const Mesh::VIndex iV0 = m.HeTailVertex(iHe);
				const Mesh::VIndex iV1 = m.HeHeadVertex(iHe);
				if (mode[iV0] == FREE2D)
					lpis[iV0].Add(mesh.vertices[iV1], weightBySizing ? 1.f / sizing[iV1] : 1.f);
				if (mode[iV1] == FREE2D)
					lpis[iV1].Add(mesh.vertices[iV0], weightBySizing ? 1.f / sizing[iV0] : 1.f);
			}
		}
		// FREE2D move + Hausdorff accept is a pure Jacobi step over the frozen
		// lpis/normals and the vertex's OWN pre-pass position — no FREE2D vertex
		// reads another's move — so precompute them in parallel (like the projection
		// pass). Bit-identical to the old serial update; the const BVH is safe to
		// query concurrently. The few CREASE1D vertices read feature-neighbour
		// positions and stay in the serial apply below.
		ParallelForPool(pool, nv, [&](std::size_t i) {
			freeOk[i] = 0;
			if (mode[i] != FREE2D)
				return;
			LaplacianInfo& lpi = lpis[i];
			if (lpi.Empty())
				return;
			const Mesh::Vertex& vertex = mesh.vertices[i];
			Mesh::Vertex u = lpi.Normalized() - vertex; // toward (weighted) neighbour centroid
			// project the displacement onto the tangent plane (remove normal part)
			const Mesh::Type nlen2 = normals[i].squaredNorm();
			if (nlen2 > 0) {
				const Mesh::Vertex n = normals[i] / std::sqrt(nlen2);
				u -= n * n.dot(u);
			}
			const Mesh::Vertex np = vertex + u * delta;
			if (!params.checkSurfDist || TestHausdorff(bvh, {np}, params.maxSurfDist)) {
				freePos[i] = np;
				freeOk[i] = 1;
			}
		});
		// Serial apply in index order: FREE2D from the precompute, CREASE1D computed
		// in-loop (it slides toward its two feature neighbours, whose positions may
		// already have been updated this pass — the sequential dependence the old
		// code had, preserved bitwise).
		FOREACHIDX (Mesh::VIndex, idxVert, mesh.vertices) {
			if (mode[idxVert] == FREE2D) {
				if (freeOk[idxVert])
					mesh.vertices[idxVert] = freePos[idxVert];
			} else if (mode[idxVert] == CREASE1D) {
				Mesh::Vertex& vertex = mesh.vertices[idxVert];
				// slide along the feature curve: move toward the midpoint of the two
				// feature neighbours, projected onto the crease tangent.
				const std::vector<Mesh::VIndex>& nbrs = vFeatureNbrs[idxVert];
				if (nbrs.size() != 2)
					continue;
				const Mesh::Vertex& w0 = mesh.vertices[nbrs[0]];
				const Mesh::Vertex& w1 = mesh.vertices[nbrs[1]];
				Mesh::Vertex t = w0 - w1;
				const Mesh::Type tlen2 = t.squaredNorm();
				if (tlen2 <= 0)
					continue;
				t /= std::sqrt(tlen2);
				const Mesh::Vertex u = t * t.dot((w0 + w1) * Mesh::Type(0.5) - vertex);
				const Mesh::Vertex newPos = vertex + u * delta;
				if (!params.checkSurfDist || TestHausdorff(bvh, {newPos}, params.maxSurfDist))
					vertex = newPos;
			}
			// FIXED: nothing
		}
	}
}

// ---------------------------------------------------------------------------
// ProjectVerticesToSurface
// Snap every vertex to the nearest point on the original surface.
// ---------------------------------------------------------------------------
void RemeshData::ProjectVerticesToSurface()
{
	// Each vertex's nearest-point query is independent and reads the const BVH
	// (no mutable state) -> embarrassingly parallel, result order-independent.
	//
	// Warm start: seed each query with the vertex's nearest face from the PREVIOUS
	// projection. The immutable originalMesh keeps face indices stable, and
	// vertices barely move between iterations (delta-bounded, Hausdorff-gated), so
	// the hint's point-to-triangle distance is a tight upper bound that prunes
	// almost the whole tree from iteration 2 on. Seeding with a valid upper bound
	// never prunes a strictly-closer triangle, so the projected point is the exact
	// nearest — bit-identical to the cold query. projectHint is swap-popped
	// alongside vertices in CollapseShortEdges and gets NO_ID for split-created
	// vertices, so it stays aligned; when empty (first projection) every query is
	// cold (hint = NO_ID, ignored by the BVH as out of range).
	if (projectHint.size() != mesh.vertices.size())
		projectHint.assign(mesh.vertices.size(), math::NO_ID);
	// Phase 1 (parallel): pure nearest-point queries into a candidate buffer;
	// vertices are not touched, so queries stay order-independent.
	std::vector<Mesh::Vertex> candidate(mesh.vertices.size());
	std::vector<char> hasCandidate(mesh.vertices.size(), 0);
	ParallelForPool(pool, mesh.vertices.size(), [&](std::size_t i) {
		const TriangleBVH::NearestNeighbor nn = bvh.NearestPoint(
		    mesh.vertices[i], std::numeric_limits<Mesh::Type>::max(), projectHint[i]);
		if (nn.IsValid()) {
			candidate[i] = nn.nearest;
			projectHint[i] = nn.idxFace;
			hasCandidate[i] = 1;
		}
	});
	// Phase 2 (serial, vertex-index order => deterministic): accept each
	// candidate unless it NEWLY flattens an incident face to near-exact
	// collinearity. On thin structures a vertex's nearest original-surface
	// point can lie exactly in the plane through two of its neighbours; the
	// unchecked snap then produces a zero-area face whose edges are all of
	// ordinary length, which the length-based collapse pass can never remove
	// (measured on the challenge fixture: 2 collinear faces re-created by
	// every projection pass after smoothing had pulled them apart). Rejecting
	// the move keeps the pre-projection position, which the next smoothing
	// pass re-relaxes. Threshold: double-area below 1e-6 of the squared
	// longest edge (relative height < 1e-6) — three decades below any
	// float-robust triangle — and only when flatter than the face already was,
	// so already-degenerate neighbourhoods are not frozen in place.
	const auto flatness = [this](const Eigen::Vector3d& p, Mesh::VIndex a,
	                             Mesh::VIndex b) {
		const Eigen::Vector3d pa = mesh.vertices[a].cast<double>();
		const Eigen::Vector3d pb = mesh.vertices[b].cast<double>();
		const double n2 = (pa - p).cross(pb - p).squaredNorm(); // (2*area)^2
		const double l2 = std::max({(pa - p).squaredNorm(), (pb - p).squaredNorm(),
		                            (pa - pb).squaredNorm()});
		return std::make_pair(n2, l2);
	};
	FOREACHIDX (Mesh::VIndex, i, mesh.vertices) {
		if (!hasCandidate[i])
			continue;
		const Eigen::Vector3d cand = candidate[i].cast<double>();
		const Eigen::Vector3d old = mesh.vertices[i].cast<double>();
		bool ok = true;
		for (const HalfMesh::HIndex iHe : m.VOutgoingHalfedges(i)) {
			if (m.HeIsBoundary(iHe))
				continue;
			const Mesh::VIndex a = m.HeHeadVertex(iHe);
			const Mesh::VIndex b = m.HeHeadVertex(m.HeNext(iHe));
			const auto [n2new, l2new] = flatness(cand, a, b);
			if (n2new >= 1e-12 * l2new * l2new)
				continue; // comfortably non-degenerate
			const auto [n2old, l2old] = flatness(old, a, b);
			(void)l2old;
			if (n2new < n2old) {
				ok = false;
				break; // newly flattened -> keep the pre-projection position
			}
		}
		if (ok)
			mesh.vertices[i] = candidate[i];
	}
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Mesh::RemeshIsotropic
// Isotropic remeshing: repeated split / collapse / flip / smooth / project.
// ---------------------------------------------------------------------------
void Mesh::RemeshIsotropic(RemeshParams params, RemeshStats* stats)
{
	static_assert(HALFMESH_TRIS, "mesh faces must be triangs");
	if (vertices.empty() || (faces.empty() && halfMesh.Empty()))
		return;
	SyncFaces();
	ASSERT(ValidateInvariants());
	// Edge lengths have no safe default (a default-constructed RemeshParams
	// zero-initializes them): a non-finite or non-positive edgeMaxLength
	// would make SplitLongEdges subdivide forever (each sweep halves lengths,
	// the threshold never wins) and a negative/NaN edgeMinLength breaks the
	// collapse criterion the same way. Refuse loudly and no-op.
	if (!std::isfinite(params.edgeMaxLength) || !(params.edgeMaxLength > 0.f) || !std::isfinite(params.edgeMinLength) || !(params.edgeMinLength >= 0.f)) {
		REPORT_WARNING("RemeshIsotropic: invalid edge-length params (min={}, max={}); no-op",
		               params.edgeMinLength, params.edgeMaxLength);
		return;
	}
	ListHalfEdges();
	SyncFaces();
	RemeshData data(*this, params);
	data.TagCreaseEdges();
	if (params.adapt)
		data.BuildSizingField();
	RemeshStats acc;
	using RClock = std::chrono::steady_clock;
	auto secs = [](RClock::time_point t0) {
		return std::chrono::duration<double>(RClock::now() - t0).count();
	};
	for (int iter = 0; iter < params.iterations; ++iter) {
		if (params.doSplit) {
			const auto t0 = RClock::now();
			acc.splitCount += data.SplitLongEdges();
			acc.splitSeconds += secs(t0);
		}
#if REMESH_DEBUG_OUTPUT
		Save(std::string("mesh_") + std::to_string(iter) + "_split.ply");
#endif
		if (params.doCollapse) {
			const auto t0 = RClock::now();
			acc.collapseCount += data.CollapseShortEdges();
			acc.collapseSeconds += secs(t0);
		}
#if REMESH_DEBUG_OUTPUT
		Save(std::string("mesh_") + std::to_string(iter) + "_collapse.ply");
#endif
		{
			const auto t0 = RClock::now();
			data.TagCreaseEdges();
			acc.tagSeconds += secs(t0);
		}
		if (params.doFlip) {
			const auto t0 = RClock::now();
			acc.flipCount += data.ImproveValence();
			acc.flipSeconds += secs(t0);
		}
#if REMESH_DEBUG_OUTPUT
		Save(std::string("mesh_") + std::to_string(iter) + "_valence.ply");
#endif
		if (params.doSmooth) {
			const auto t0 = RClock::now();
			if (params.smoothTangential)
				data.TangentialSmoothing(params.smoothIterations, params.smoothDelta);
			else
				data.VertexCoordLaplacian(params.smoothIterations, params.smoothDelta);
			acc.smoothSeconds += secs(t0);
		}
#if REMESH_DEBUG_OUTPUT
		Save(std::string("mesh_") + std::to_string(iter) + "_smooth.ply");
#endif
		if (params.doProject) {
			const auto t0 = RClock::now();
			data.ProjectVerticesToSurface();
			acc.projectSeconds += secs(t0);
		}
#if REMESH_DEBUG_OUTPUT
		Save(std::string("mesh_") + std::to_string(iter) + "_project.ply");
#endif
	}
	if (stats != nullptr)
		*stats = acc;
	SyncFaces();
	ASSERT(ValidateInvariants());
}

} // namespace halfmesh

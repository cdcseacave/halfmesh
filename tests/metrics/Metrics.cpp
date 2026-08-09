/*
* Metrics.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/metrics/Metrics.cpp — shared analytic metrics toolkit
//
// Implementation of hmtest::metrics — uses only the public halfmesh API.

#include "Metrics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace hmtest {
namespace metrics {

using halfmesh::HalfMesh;
using halfmesh::Mesh;
using halfmesh::TriangleKdTree;

// ============================================================
// Internal helpers
// ============================================================

namespace {

// Build a local HalfMesh from the mesh (avoids mutating input).
HalfMesh BuildHM(const Mesh& mesh)
{
	return HalfMesh(mesh);
}

// Signed area of a UV triangle (positive = CCW, negative = CW).
static float SignedUVArea(const Mesh::TexCoord& t0,
                          const Mesh::TexCoord& t1,
                          const Mesh::TexCoord& t2)
{
	return 0.5f * ((t1.x() - t0.x()) * (t2.y() - t0.y()) - (t2.x() - t0.x()) * (t1.y() - t0.y()));
}

// Symmetric-Dirichlet energy of one triangle: map 3-D triangle isometrically
// to 2-D (using edge lengths), compute 2×2 Jacobian to UV space, extract
// singular values s0,s1 → E = s0²+s1²+1/s0²+1/s1² (minimum = 4 for isometry).
// Returns {energy, 3D_area}.
static std::pair<double, double> TriSymDir(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector2d& u0, const Eigen::Vector2d& u1, const Eigen::Vector2d& u2)
{
	// 3D edges
	Eigen::Vector3d e1 = p1 - p0;
	Eigen::Vector3d e2 = p2 - p0;
	const double l1 = e1.norm();
	const double l2 = e2.norm();
	if (l1 < 1e-12 || l2 < 1e-12)
		return {4.0, 0.0};

	// Local 2-D reference frame for the 3-D triangle
	Eigen::Vector2d r1(l1, 0.0);
	Eigen::Vector2d r2(e1.dot(e2) / l1, e1.cross(e2).norm() / l1);
	const double refArea = 0.5 * std::abs(r1.x() * r2.y());
	if (refArea < 1e-12)
		return {4.0, 0.0};

	// UV edges
	Eigen::Vector2d du1 = u1 - u0;
	Eigen::Vector2d du2 = u2 - u0;

	// Jacobian J such that r_i → du_i:  J * r1 = du1, J * r2 = du2
	// J = [du1 | du2] * inv([r1 | r2])
	Eigen::Matrix2d R;
	R.col(0) = r1;
	R.col(1) = r2;
	Eigen::Matrix2d dU;
	dU.col(0) = du1;
	dU.col(1) = du2;

	const double detR = R.determinant();
	if (std::abs(detR) < 1e-12)
		return {4.0, 0.0};

	Eigen::Matrix2d J = dU * R.inverse();
	Eigen::JacobiSVD<Eigen::Matrix2d> svd(J);
	double s0 = svd.singularValues()(0);
	double s1 = svd.singularValues()(1);
	s0 = std::max(s0, 1e-9);
	s1 = std::max(s1, 1e-9);
	const double e = s0 * s0 + s1 * s1 + 1.0 / (s0 * s0) + 1.0 / (s1 * s1);
	const double a3d = 0.5 * (p1 - p0).cross(p2 - p0).norm();
	return {e, a3d};
}

// Sander L2 stretch for one triangle: sqrt(mean(s0²+s1²)/2) where s0,s1 are
// singular values of the Jacobian from 3-D to UV space. Returns {stretch, area}.
static std::pair<double, double> TriStretchL2(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector2d& u0, const Eigen::Vector2d& u1, const Eigen::Vector2d& u2)
{
	Eigen::Vector3d e1 = p1 - p0;
	Eigen::Vector3d e2 = p2 - p0;
	const double l1 = e1.norm();
	if (l1 < 1e-12)
		return {1.0, 0.0};
	Eigen::Vector2d r1(l1, 0.0);
	Eigen::Vector2d r2(e1.dot(e2) / l1, e1.cross(e2).norm() / l1);
	const double refArea = 0.5 * std::abs(r1.x() * r2.y());
	if (refArea < 1e-12)
		return {1.0, 0.0};
	Eigen::Matrix2d R;
	R.col(0) = r1;
	R.col(1) = r2;
	Eigen::Matrix2d dU;
	dU.col(0) = u1 - u0;
	dU.col(1) = u2 - u0;
	const double detR = R.determinant();
	if (std::abs(detR) < 1e-12)
		return {1.0, 0.0};
	Eigen::Matrix2d J = dU * R.inverse();
	Eigen::JacobiSVD<Eigen::Matrix2d> svd(J);
	double s0 = svd.singularValues()(0);
	double s1 = svd.singularValues()(1);
	const double stretch = std::sqrt((s0 * s0 + s1 * s1) * 0.5);
	const double a3d = 0.5 * (p1 - p0).cross(p2 - p0).norm();
	return {stretch, a3d};
}

// Point-to-segment squared distance helper.
static float PointSegDistSq(const Mesh::Vertex& p,
                            const Mesh::Vertex& a,
                            const Mesh::Vertex& b)
{
	const Mesh::Vertex ab = b - a;
	const float t = std::max(0.f, std::min(1.f, (p - a).dot(ab) / ab.squaredNorm()));
	const Mesh::Vertex proj = a + t * ab;
	return (p - proj).squaredNorm();
}

} // anonymous namespace

// ============================================================
// TOPOLOGY
// ============================================================

TopologyCounts ComputeTopology(const Mesh& mesh)
{
	TopologyCounts tc;
	if (mesh.vertices.empty())
		return tc;

	tc.numVertices = static_cast<uint32_t>(mesh.vertices.size());
	tc.numFaces = static_cast<uint32_t>(mesh.faces.size());

	// Edge manifold: each undirected edge shared by at most 2 faces.
	// Computed from raw faces BEFORE building HalfMesh, because HalfMesh::Build
	// can crash or produce undefined results on non-manifold inputs.
	// Key = min(u,v)<<32 | max(u,v), value = face count sharing that edge.
	// Also count unique edges for later use.
	std::unordered_map<uint64_t, uint32_t> edgeFaceCount;
	{
		edgeFaceCount.reserve(mesh.faces.size() * 3);
		bool edgeManifold = true;
		for (const auto& f : mesh.faces) {
			for (int k = 0; k < 3; ++k) {
				const uint32_t u = f[k];
				const uint32_t v = f[(k + 1) % 3];
				const uint64_t key = (static_cast<uint64_t>(std::min(u, v)) << 32)
				                     | static_cast<uint64_t>(std::max(u, v));
				if (++edgeFaceCount[key] > 2) {
					edgeManifold = false;
				}
			}
		}
		tc.isEdgeManifold = edgeManifold;
	}

	// Vertex manifold: faces around each vertex form a single connected fan/cycle.
	// A bow-tie vertex has ≥2 disjoint fans sharing only that vertex.
	// Computed from raw faces BEFORE building HalfMesh (HalfMesh crashes on bow-ties).
	//
	// For vertex v in face (a,b,c): in the one-ring of v, the edge (c→b) represents
	// the face's contribution to v's fan. We build a directed graph: prev → next
	// where prev and next are neighbors of v in each incident face.
	//
	// Manifold condition:
	//   - No duplicate "prev" key (each neighbor appears at most once as the "incoming"
	//     edge of the fan — if duplicated, two faces share the same oriented edge at v,
	//     meaning the one-ring is not simple).
	//   - The (prev→next) pairs form exactly ONE connected component when traversed.
	//     Count "chain starts" = nodes that appear as some pair's next but NOT as any
	//     pair's prev.  A manifold vertex has at most 1 chain start:
	//       0 = closed interior cycle, 1 = open boundary fan.
	//     ≥2 chain starts = disconnected fans = bow-tie = vertex non-manifold.
	{
		const uint32_t nv = static_cast<uint32_t>(mesh.vertices.size());
		std::vector<std::vector<std::pair<uint32_t, uint32_t>>> vPairs(nv);
		for (const auto& f : mesh.faces) {
			for (int k = 0; k < 3; ++k) {
				const uint32_t vi = f[k];
				const uint32_t prev = f[(k + 2) % 3];
				const uint32_t next = f[(k + 1) % 3];
				vPairs[vi].emplace_back(prev, next);
			}
		}
		bool vertexManifold = true;
		for (uint32_t vi = 0; vi < nv && vertexManifold; ++vi) {
			const auto& pairs = vPairs[vi];
			if (pairs.empty())
				continue;
			// Build map: prev → next.  Duplicate prev means two faces share the same
			// oriented half-edge at v, which is non-manifold.
			std::unordered_map<uint32_t, uint32_t> chain;
			chain.reserve(pairs.size());
			for (const auto& [p, n] : pairs) {
				if (!chain.emplace(p, n).second) {
					vertexManifold = false;
					break;
				}
			}
			if (!vertexManifold)
				break;

			// Count chain starts: nodes that appear as some pair's "next" value
			// but are NOT any pair's "prev" key.
			// Each such node is the start of a new chain segment.
			std::unordered_set<uint32_t> prevKeys;
			std::unordered_set<uint32_t> nextVals;
			prevKeys.reserve(pairs.size());
			nextVals.reserve(pairs.size());
			for (const auto& [p, n] : pairs) {
				prevKeys.insert(p);
				nextVals.insert(n);
			}
			uint32_t chainStarts = 0;
			for (const uint32_t n : nextVals) {
				if (prevKeys.find(n) == prevKeys.end())
					++chainStarts;
			}
			// chainStarts == 0: closed interior cycle (valid)
			// chainStarts == 1: open boundary fan (valid)
			// chainStarts >= 2: disconnected fans = bow-tie (non-manifold)
			if (chainStarts >= 2) {
				vertexManifold = false;
			}
		}
		tc.isVertexManifold = vertexManifold;
	}

	// Build HalfMesh only on fully manifold meshes (both edge and vertex).
	// HalfMesh::Build crashes on bow-tie vertices and non-manifold edges.
	// For non-manifold meshes, count edges from the raw edge map and skip hole enumeration.
	if (tc.isEdgeManifold && tc.isVertexManifold) {
		HalfMesh hm(mesh);
		tc.numEdges = hm.ESize();
		tc.euler = static_cast<int32_t>(tc.numVertices)
		           - static_cast<int32_t>(tc.numEdges)
		           + static_cast<int32_t>(tc.numFaces);

		// Boundary loops via EnumerateHoles.
		std::vector<std::vector<HalfMesh::VIndex>> holes;
		hm.EnumerateHoles(holes);
		tc.numBoundaryLoops = static_cast<uint32_t>(holes.size());
		tc.isWatertight = (tc.numBoundaryLoops == 0);

		// genus = (2 - χ - numBoundaryLoops) / 2  (for orientable surface)
		tc.genus = (2 - tc.euler - static_cast<int32_t>(tc.numBoundaryLoops)) / 2;
	} else {
		// Non-manifold: use unique edge count from raw map; boundary/genus undefined.
		tc.numEdges = static_cast<uint32_t>(edgeFaceCount.size());
		tc.euler = static_cast<int32_t>(tc.numVertices)
		           - static_cast<int32_t>(tc.numEdges)
		           + static_cast<int32_t>(tc.numFaces);
		tc.numBoundaryLoops = 0; // undefined for non-manifold
		tc.isWatertight = false;
		tc.genus = 0; // undefined
	}

	return tc;
}

std::map<uint32_t, uint32_t> ComputeValenceHistogram(const Mesh& mesh)
{
	std::map<uint32_t, uint32_t> hist;
	if (mesh.vertices.empty())
		return hist;
	HalfMesh hm(mesh);
	for (HalfMesh::VIndex v = 0; v < hm.VSize(); ++v) {
		const uint32_t val = hm.VFaceDegree(v);
		hist[val]++;
	}
	return hist;
}

// ============================================================
// GEOMETRY
// ============================================================

double ComputeSurfaceArea(const Mesh& mesh)
{
	double area = 0.0;
	for (const auto& f : mesh.faces) {
		const Mesh::Vertex& a = mesh.vertices[f[0]];
		const Mesh::Vertex& b = mesh.vertices[f[1]];
		const Mesh::Vertex& c = mesh.vertices[f[2]];
		area += 0.5 * (b - a).cross(c - a).norm();
	}
	return area;
}

AABB ComputeAABB(const Mesh& mesh)
{
	AABB box;
	if (mesh.vertices.empty()) {
		box.minPt = Mesh::Vertex::Zero();
		box.maxPt = Mesh::Vertex::Zero();
		return box;
	}
	box.minPt = mesh.vertices[0];
	box.maxPt = mesh.vertices[0];
	for (const auto& v : mesh.vertices) {
		box.minPt = box.minPt.cwiseMin(v);
		box.maxPt = box.maxPt.cwiseMax(v);
	}
	return box;
}

double ComputeSignedVolume(const Mesh& mesh)
{
	// Divergence theorem: V = (1/6) * sum_faces( (a · (b × c)) )
	// where a, b, c are the three vertices of each face.
	double vol = 0.0;
	for (const auto& f : mesh.faces) {
		const Eigen::Vector3d a = mesh.vertices[f[0]].cast<double>();
		const Eigen::Vector3d b = mesh.vertices[f[1]].cast<double>();
		const Eigen::Vector3d c = mesh.vertices[f[2]].cast<double>();
		vol += a.dot(b.cross(c));
	}
	return vol / 6.0;
}

TriangleQuality ComputeTriangleQuality(
    const Mesh::Vertex& a, const Mesh::Vertex& b, const Mesh::Vertex& c)
{
	TriangleQuality q;
	// Edge lengths
	const float la = (b - c).norm();
	const float lb = (a - c).norm();
	const float lc = (a - b).norm();
	const float eps = 1e-12f;
	if (la < eps && lb < eps && lc < eps)
		return q;

	// Angles via law of cosines: cos(A) = (b²+c²-a²)/(2bc)
	auto safeAcos = [](float v) {
		return std::acos(std::max(-1.f, std::min(1.f, v)));
	};
	const float cosA = (lb * lb + lc * lc - la * la) / (2.f * lb * lc + eps);
	const float cosB = (la * la + lc * lc - lb * lb) / (2.f * la * lc + eps);
	const float cosC = (la * la + lb * lb - lc * lc) / (2.f * la * lb + eps);
	const float angA = safeAcos(cosA) * (180.f / static_cast<float>(M_PI));
	const float angB = safeAcos(cosB) * (180.f / static_cast<float>(M_PI));
	const float angC = safeAcos(cosC) * (180.f / static_cast<float>(M_PI));

	q.minAngleDeg = std::min({angA, angB, angC});
	q.maxAngleDeg = std::max({angA, angB, angC});

	// Aspect ratio = longest / shortest edge
	const float lMax = std::max({la, lb, lc});
	const float lMin = std::min({la, lb, lc});
	q.aspectRatio = (lMin > eps) ? lMax / lMin : 0.f;

	// Radius ratio = 2 * inradius / circumradius (1 for equilateral)
	const float s = (la + lb + lc) * 0.5f;
	const float area = (b - a).cross(c - a).norm() * 0.5f;
	const float inradius = (area > eps) ? area / s : 0.f;
	const float circum = (area > eps) ? (la * lb * lc) / (4.f * area) : 0.f;
	q.radiusRatio = (circum > eps) ? 2.f * inradius / circum : 0.f;

	return q;
}

std::vector<TriangleQuality> ComputeAllTriangleQualities(const Mesh& mesh)
{
	std::vector<TriangleQuality> result;
	result.reserve(mesh.faces.size());
	for (const auto& f : mesh.faces) {
		result.push_back(ComputeTriangleQuality(
		    mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]));
	}
	return result;
}

EdgeLengthStats ComputeEdgeLengthStats(const Mesh& mesh)
{
	EdgeLengthStats s;
	if (mesh.faces.empty())
		return s;

	// Collect unique edge lengths via HalfMesh edge iteration.
	HalfMesh hm(mesh);
	std::vector<double> lengths;
	lengths.reserve(hm.ESize());
	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e) {
		const HalfMesh::HIndex iHe = hm.EHalfedge(e);
		const HalfMesh::VIndex v0 = hm.HeVertex(iHe);
		const HalfMesh::VIndex v1 = hm.HeHeadVertex(iHe);
		lengths.push_back((mesh.vertices[v0] - mesh.vertices[v1]).norm());
	}
	if (lengths.empty())
		return s;

	s.minLen = *std::min_element(lengths.begin(), lengths.end());
	s.maxLen = *std::max_element(lengths.begin(), lengths.end());
	s.meanLen = std::accumulate(lengths.begin(), lengths.end(), 0.0)
	            / static_cast<double>(lengths.size());
	double var = 0.0;
	for (double l : lengths)
		var += (l - s.meanLen) * (l - s.meanLen);
	s.stddev = std::sqrt(var / static_cast<double>(lengths.size()));
	return s;
}

// ============================================================
// DISTANCE (KD-tree + brute-force)
// ============================================================

float PointToTriangleDistSq(
    const Mesh::Vertex& p,
    const Mesh::Vertex& a,
    const Mesh::Vertex& b,
    const Mesh::Vertex& c)
{
	// Source: Christer Ericson "Real-Time Collision Detection" §5.1.5
	const Mesh::Vertex ab = b - a;
	const Mesh::Vertex ac = c - a;
	const Mesh::Vertex ap = p - a;

	const float d1 = ab.dot(ap);
	const float d2 = ac.dot(ap);
	if (d1 <= 0.f && d2 <= 0.f)
		return (p - a).squaredNorm(); // vertex a

	const Mesh::Vertex bp = p - b;
	const float d3 = ab.dot(bp);
	const float d4 = ac.dot(bp);
	if (d3 >= 0.f && d4 <= d3)
		return (p - b).squaredNorm(); // vertex b

	const Mesh::Vertex cp = p - c;
	const float d5 = ab.dot(cp);
	const float d6 = ac.dot(cp);
	if (d6 >= 0.f && d5 <= d6)
		return (p - c).squaredNorm(); // vertex c

	const float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
		const float v = d1 / (d1 - d3);
		const Mesh::Vertex proj = a + v * ab;
		return (p - proj).squaredNorm();
	}

	const float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
		const float w = d2 / (d2 - d6);
		const Mesh::Vertex proj = a + w * ac;
		return (p - proj).squaredNorm();
	}

	const float va = d3 * d6 - d5 * d4;
	if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
		const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		const Mesh::Vertex proj = b + w * (c - b);
		return (p - proj).squaredNorm();
	}

	// Interior of triangle
	const float denom = 1.f / (va + vb + vc);
	const float v = vb * denom;
	const float w = vc * denom;
	const Mesh::Vertex proj = a + v * ab + w * ac;
	return (p - proj).squaredNorm();
}

// Generate a set of sample points on the surface of a mesh.
// For each triangle we include the 3 vertices, 3 edge midpoints, and the centroid
// (7 samples per face). Vertices are also added up front so isolated vertices count.
// Returns sample points (vertex duplicates from shared vertices are acceptable).
// The full set is |V| + 4·|F| (each vertex, the 3 edge midpoints and the centroid
// of each face). When maxSamples > 0 and the full set would exceed it, points are
// taken with a fixed stride over that sequence — a deterministic, uniform subsample
// that bounds the O(samples) Hausdorff cost on huge meshes (tens of millions of
// samples would otherwise dominate). stride==1 (the small-mesh case) reproduces the
// full, exact set, so unit tests are unaffected.
static std::vector<Mesh::Vertex> SurfaceSamples(const Mesh& mesh, std::size_t maxSamples = 0)
{
	const std::size_t total = mesh.vertices.size() + mesh.faces.size() * 4;
	const std::size_t stride =
	    (maxSamples > 0 && total > maxSamples) ? (total + maxSamples - 1) / maxSamples : 1;
	std::vector<Mesh::Vertex> pts;
	pts.reserve(stride > 1 ? total / stride + 1 : total);
	std::size_t i = 0;
	const auto take = [&](const Mesh::Vertex& p) {
		if (i++ % stride == 0)
			pts.push_back(p);
	};
	for (const auto& v : mesh.vertices)
		take(v);
	for (const auto& f : mesh.faces) {
		const Mesh::Vertex& a = mesh.vertices[f[0]];
		const Mesh::Vertex& b = mesh.vertices[f[1]];
		const Mesh::Vertex& c = mesh.vertices[f[2]];
		take(0.5f * (a + b)); // midpoint ab
		take(0.5f * (b + c)); // midpoint bc
		take(0.5f * (a + c)); // midpoint ac
		take((a + b + c) / 3.0f); // centroid
	}
	return pts;
}

DistanceResult ComputeDistanceKdTree(const Mesh& meshA, const Mesh& meshB)
{
	DistanceResult res;
	if (meshA.vertices.empty() || meshB.vertices.empty())
		return res;

	TriangleKdTree kdB(meshB);
	TriangleKdTree kdA(meshA);

	// Sample surface points from each mesh (vertices + midpoints + centroids),
	// capped so the symmetric Hausdorff stays tractable on multi-million-face
	// meshes (a uniform ~1M-point subsample estimates it to within sampling noise;
	// smaller meshes are sampled in full).
	constexpr std::size_t maxSamples = 1'000'000;
	const auto samplesA = SurfaceSamples(meshA, maxSamples);
	const auto samplesB = SurfaceSamples(meshB, maxSamples);

	// h(A → B): for each sample in A, find nearest on B.
	double hAb = 0.0, meanSum = 0.0;
	for (const auto& v : samplesA) {
		const auto nn = kdB.NearestPoint(v);
		const double d = static_cast<double>(nn.dist);
		hAb = std::max(hAb, d);
		meanSum += d;
	}
	const double meanAb = meanSum / static_cast<double>(samplesA.size());

	// h(B → A): for each sample in B, find nearest on A.
	double hBa = 0.0, meanSumBa = 0.0;
	for (const auto& v : samplesB) {
		const auto nn = kdA.NearestPoint(v);
		const double d = static_cast<double>(nn.dist);
		hBa = std::max(hBa, d);
		meanSumBa += d;
	}
	const double meanBa = meanSumBa / static_cast<double>(samplesB.size());

	res.hausdorffSymmetric = std::max(hAb, hBa);
	res.meanSurfaceDist = (meanAb + meanBa) * 0.5;
	return res;
}

DistanceResult ComputeDistanceBruteForce(const Mesh& meshA, const Mesh& meshB)
{
	DistanceResult res;
	if (meshA.vertices.empty() || meshB.vertices.empty())
		return res;

	// Use the same surface samples as ComputeDistanceKdTree for consistent cross-check.
	const auto samplesA = SurfaceSamples(meshA);
	const auto samplesB = SurfaceSamples(meshB);

	// h(A → B): for each sample in A, find nearest triangle in B.
	double hAb = 0.0, meanSum = 0.0;
	for (const auto& p : samplesA) {
		float best = std::numeric_limits<float>::max();
		for (const auto& f : meshB.faces) {
			const float d2 = PointToTriangleDistSq(
			    p, meshB.vertices[f[0]], meshB.vertices[f[1]], meshB.vertices[f[2]]);
			best = std::min(best, d2);
		}
		const double d = std::sqrt(static_cast<double>(best));
		hAb = std::max(hAb, d);
		meanSum += d;
	}
	const double meanAb = meanSum / static_cast<double>(samplesA.size());

	// h(B → A): for each sample in B, find nearest triangle in A.
	double hBa = 0.0, meanSumBa = 0.0;
	for (const auto& p : samplesB) {
		float best = std::numeric_limits<float>::max();
		for (const auto& f : meshA.faces) {
			const float d2 = PointToTriangleDistSq(
			    p, meshA.vertices[f[0]], meshA.vertices[f[1]], meshA.vertices[f[2]]);
			best = std::min(best, d2);
		}
		const double d = std::sqrt(static_cast<double>(best));
		hBa = std::max(hBa, d);
		meanSumBa += d;
	}
	const double meanBa = meanSumBa / static_cast<double>(samplesB.size());

	res.hausdorffSymmetric = std::max(hAb, hBa);
	res.meanSurfaceDist = (meanAb + meanBa) * 0.5;
	return res;
}

// ============================================================
// UV METRICS
// ============================================================

UVMetrics ComputeUVMetrics(const Mesh& mesh)
{
	UVMetrics uv;
	const size_t nf = mesh.faces.size();
	if (mesh.faceTexcoords.size() != nf * 3) {
		uv.allFinite = false;
		return uv;
	}
	uv.numFaces = static_cast<int>(nf);

	// Compute signed UV areas.
	// Convention: CCW in UV space = positive signed area = "correct" orientation.
	// A "flip" is any triangle with negative signed UV area.
	// This matches the SLIM/ARAP convention (injective = all triangles CCW).
	//
	// When comparing against the 3-D face orientation we look at whether the
	// UV signed area agrees with the 3-D face's own orientation sign. Since
	// all 3-D faces are consistently wound (outward normals), we use the
	// signed UV area directly: negative = flip.
	//
	// NOTE: if the entire atlas is consistently CW (all negative), all triangles
	// are counted as flipped. Callers that want majority-relative flip counts
	// can subtract the minimum(posCount, neg_count) from the result.
	std::vector<float> signedAreas(nf);
	for (size_t fi = 0; fi < nf; ++fi) {
		const Mesh::TexCoord& t0 = mesh.faceTexcoords[fi * 3 + 0];
		const Mesh::TexCoord& t1 = mesh.faceTexcoords[fi * 3 + 1];
		const Mesh::TexCoord& t2 = mesh.faceTexcoords[fi * 3 + 2];
		if (!std::isfinite(t0.x()) || !std::isfinite(t0.y()) || !std::isfinite(t1.x()) || !std::isfinite(t1.y()) || !std::isfinite(t2.x()) || !std::isfinite(t2.y())) {
			uv.allFinite = false;
		}
		signedAreas[fi] = SignedUVArea(t0, t1, t2);
	}

	// Count flips: triangles with negative UV signed area.
	for (size_t fi = 0; fi < nf; ++fi) {
		if (signedAreas[fi] < 0.f)
			++uv.flipCount;
	}

	// Symmetric-Dirichlet and Stretch L2 (area-weighted mean).
	double sumSd = 0.0, sumSt = 0.0, sumArea = 0.0;
	for (size_t fi = 0; fi < nf; ++fi) {
		const Mesh::Face& f = mesh.faces[fi];
		const Eigen::Vector3d p0 = mesh.vertices[f[0]].cast<double>();
		const Eigen::Vector3d p1 = mesh.vertices[f[1]].cast<double>();
		const Eigen::Vector3d p2 = mesh.vertices[f[2]].cast<double>();
		const Eigen::Vector2d u0 = mesh.faceTexcoords[fi * 3 + 0].cast<double>();
		const Eigen::Vector2d u1 = mesh.faceTexcoords[fi * 3 + 1].cast<double>();
		const Eigen::Vector2d u2 = mesh.faceTexcoords[fi * 3 + 2].cast<double>();
		auto [sd, a3d] = TriSymDir(p0, p1, p2, u0, u1, u2);
		auto [st, _a] = TriStretchL2(p0, p1, p2, u0, u1, u2);
		sumSd += a3d * sd;
		sumSt += a3d * st;
		sumArea += a3d;
	}
	if (sumArea > 1e-12) {
		uv.symDirichlet = sumSd / sumArea;
		uv.stretchL2 = sumSt / sumArea;
	}

	// Atlas occupancy: fraction of [0,1]² covered. Approximate via UV triangle areas.
	// (Simple sum of |uvArea| — can be > 1 if charts overlap, but we clamp to 1.)
	double totalUvArea = 0.0;
	for (size_t fi = 0; fi < nf; ++fi)
		totalUvArea += std::abs(static_cast<double>(signedAreas[fi]));
	uv.atlasOccupancy = std::min(1.0, totalUvArea);

	// Overlap detection: bounding-box test on UV triangles.
	// Build axis-aligned bounding boxes per face in UV space, check for pairwise
	// overlaps of distinct faces (O(n²) — acceptable for the toolkit's self-tests).
	// Only check if nf is small enough to be practical.
	uv.hasBboxOverlaps = false;
	if (nf <= 2000) {
		struct UVBox
		{
			float x0, y0, x1, y1;
		};
		std::vector<UVBox> boxes(nf);
		for (size_t fi = 0; fi < nf; ++fi) {
			float xmin = std::numeric_limits<float>::max();
			float ymin = xmin;
			float xmax = std::numeric_limits<float>::lowest();
			float ymax = xmax;
			for (int k = 0; k < 3; ++k) {
				const Mesh::TexCoord& t = mesh.faceTexcoords[fi * 3 + k];
				xmin = std::min(xmin, t.x());
				xmax = std::max(xmax, t.x());
				ymin = std::min(ymin, t.y());
				ymax = std::max(ymax, t.y());
			}
			boxes[fi] = {xmin, ymin, xmax, ymax};
		}
		// We only flag overlap if two faces are non-adjacent and their bounding
		// boxes overlap with a meaningful margin (ignoring shared-edge cases).
		// For the toolkit's purposes this is sufficient.
		const float overlapTol = 1e-4f;
		for (size_t i = 0; i < nf && !uv.hasBboxOverlaps; ++i) {
			for (size_t j = i + 1; j < nf && !uv.hasBboxOverlaps; ++j) {
				const UVBox& bi = boxes[i];
				const UVBox& bj = boxes[j];
				const bool sepX = (bi.x1 + overlapTol < bj.x0) || (bj.x1 + overlapTol < bi.x0);
				const bool sepY = (bi.y1 + overlapTol < bj.y0) || (bj.y1 + overlapTol < bi.y0);
				if (!sepX && !sepY)
					uv.hasBboxOverlaps = true;
			}
		}
	}

	return uv;
}

std::vector<double> ComputeChartTexelDensity(
    const Mesh& mesh,
    const std::vector<unsigned>& faceChart,
    unsigned numCharts)
{
	std::vector<double> worldArea(numCharts, 0.0);
	std::vector<double> uvArea(numCharts, 0.0);

	const size_t nf = mesh.faces.size();
	if (mesh.faceTexcoords.size() != nf * 3 || faceChart.size() != nf)
		return std::vector<double>(numCharts, 0.0);

	for (size_t fi = 0; fi < nf; ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;
		const Mesh::Face& f = mesh.faces[fi];
		const double da3d = (mesh.vertices[f[1]] - mesh.vertices[f[0]])
		                        .cross(mesh.vertices[f[2]] - mesh.vertices[f[0]])
		                        .norm();
		worldArea[cid] += 0.5 * da3d;
		const float sa = std::abs(SignedUVArea(
		    mesh.faceTexcoords[fi * 3 + 0],
		    mesh.faceTexcoords[fi * 3 + 1],
		    mesh.faceTexcoords[fi * 3 + 2]));
		uvArea[cid] += static_cast<double>(sa);
	}

	std::vector<double> density(numCharts, 0.0);
	for (unsigned c = 0; c < numCharts; ++c) {
		if (worldArea[c] > 1e-12)
			density[c] = std::sqrt(uvArea[c] / worldArea[c]);
	}
	return density;
}

// ============================================================
// CHART-PARTITION METRICS
// ============================================================

namespace {

// Undirected edge key (sorted vertex pair).
using EdgeKey = std::pair<uint32_t, uint32_t>;
inline EdgeKey MakeEdge(uint32_t a, uint32_t b)
{
	return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
}

// Map every undirected edge → list of incident face indices.
std::map<EdgeKey, std::vector<uint32_t>> BuildEdgeFaceMap(const Mesh& mesh)
{
	std::map<EdgeKey, std::vector<uint32_t>> edges;
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		const auto& face = mesh.faces[f];
		for (int c = 0; c < 3; ++c) {
			const uint32_t a = face[c];
			const uint32_t b = face[(c + 1) % 3];
			edges[MakeEdge(a, b)].push_back(f);
		}
	}
	return edges;
}

inline double EdgeLength(const Mesh& mesh, const EdgeKey& e)
{
	return static_cast<double>((mesh.vertices[e.first] - mesh.vertices[e.second]).norm());
}

constexpr double UNSET = -1.0; // matches hmbench::UNSET (BenchTypes.h:39)

} // namespace

double ComputeBoundaryCutLength(const Mesh& mesh,
                                const std::vector<unsigned>& faceChart)
{
	if (faceChart.size() != mesh.faces.size())
		return UNSET;
	const auto edges = BuildEdgeFaceMap(mesh);
	double total = 0.0;
	for (const auto& [edge, faces] : edges) {
		bool seam = false;
		if (faces.size() == 1) {
			seam = true; // mesh-boundary edge bounds its chart
		} else {
			const unsigned c0 = faceChart[faces[0]];
			for (size_t i = 1; i < faces.size(); ++i)
				if (faceChart[faces[i]] != c0) {
					seam = true;
					break;
				}
		}
		if (seam)
			total += EdgeLength(mesh, edge);
	}
	return total;
}

// ---------------------------------------------------------------------------
std::vector<double> ComputeChartPlanarityError(const Mesh& mesh,
                                               const std::vector<unsigned>& faceChart,
                                               unsigned numCharts)
{
	std::vector<double> err(numCharts, 0.0);
	if (faceChart.size() != mesh.faces.size() || numCharts == 0)
		return err;

	// Pass 1: area-weighted mean normal per chart.
	std::vector<Mesh::Normal> meanN(numCharts, Mesh::Normal::Zero());
	std::vector<double> chartArea(numCharts, 0.0);
	std::vector<Mesh::Normal> faceUnit(mesh.faces.size());
	std::vector<double> faceArea(mesh.faces.size(), 0.0);
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		Mesh::Normal n = mesh.ComputeFaceNormal(f);
		const double nn = static_cast<double>(n.norm());
		const double area = 0.5 * nn;
		faceArea[f] = area;
		if (nn > 1e-20)
			faceUnit[f] = n / static_cast<float>(nn);
		else
			faceUnit[f] = Mesh::Normal::Zero();
		const unsigned c = faceChart[f];
		if (c < numCharts) {
			meanN[c] += faceUnit[f] * static_cast<float>(area);
			chartArea[c] += area;
		}
	}
	for (unsigned c = 0; c < numCharts; ++c) {
		const double nn = static_cast<double>(meanN[c].norm());
		if (nn > 1e-20)
			meanN[c] /= static_cast<float>(nn);
	}

	// Pass 2: area-weighted mean angle to the chart's mean normal.
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		const unsigned c = faceChart[f];
		if (c >= numCharts || chartArea[c] <= 0.0)
			continue;
		double cosang = static_cast<double>(faceUnit[f].dot(meanN[c]));
		cosang = std::clamp(cosang, -1.0, 1.0);
		err[c] += faceArea[f] * std::acos(cosang);
	}
	for (unsigned c = 0; c < numCharts; ++c)
		if (chartArea[c] > 0.0)
			err[c] /= chartArea[c];
	return err;
}

// ---------------------------------------------------------------------------
std::vector<double> ComputeChartCompactness(const Mesh& mesh,
                                            const std::vector<unsigned>& faceChart,
                                            unsigned numCharts)
{
	std::vector<double> compact(numCharts, UNSET);
	if (faceChart.size() != mesh.faces.size() || numCharts == 0)
		return compact;

	std::vector<double> area(numCharts, 0.0);
	std::vector<double> perim(numCharts, 0.0);
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		const unsigned c = faceChart[f];
		if (c < numCharts)
			area[c] += 0.5 * static_cast<double>(mesh.ComputeFaceDoubleArea(f));
	}
	const auto edges = BuildEdgeFaceMap(mesh);
	for (const auto& [edge, faces] : edges) {
		// Distinct charts incident to this edge.
		std::array<unsigned, 8> seen{};
		unsigned nseen = 0;
		auto note = [&](unsigned c) {
			for (unsigned i = 0; i < nseen; ++i)
				if (seen[i] == c)
					return;
			if (nseen < seen.size())
				seen[nseen++] = c;
		};
		for (uint32_t fidx : faces)
			note(faceChart[fidx]);
		const bool boundary = (faces.size() == 1);
		if (boundary || nseen > 1) {
			const double len = EdgeLength(mesh, edge);
			for (unsigned i = 0; i < nseen; ++i)
				if (seen[i] < numCharts)
					perim[seen[i]] += len;
		}
	}
	for (unsigned c = 0; c < numCharts; ++c)
		compact[c] = (area[c] > 1e-20)
		                 ? (perim[c] * perim[c]) / (4.0 * M_PI * area[c])
		                 : UNSET;
	return compact;
}

// ---------------------------------------------------------------------------
bool ComputeChartCoverage(const Mesh& mesh,
                          const std::vector<unsigned>& faceChart,
                          unsigned numCharts)
{
	if (faceChart.size() != mesh.faces.size())
		return false;
	for (unsigned c : faceChart)
		if (c >= numCharts)
			return false;
	return true;
}

// ============================================================
// ROBUSTNESS
// ============================================================

bool ScanFinite(const Mesh& mesh)
{
	for (const auto& v : mesh.vertices) {
		if (!v.allFinite())
			return false;
	}
	for (const auto& n : mesh.faceNormals) {
		if (!n.allFinite())
			return false;
	}
	for (const auto& t : mesh.faceTexcoords) {
		if (!std::isfinite(t.x()) || !std::isfinite(t.y()))
			return false;
	}
	for (const auto& col : mesh.vertexColors) {
		// Pixel is uint8, always "finite", but check cast.
		(void)col;
	}
	return true;
}

// ============================================================
// CANONICALIZATION
// ============================================================

CanonicalForm Canonicalize(const Mesh& mesh, float posTol)
{
	CanonicalForm cf;
	if (mesh.vertices.empty())
		return cf;

	const uint32_t nv = static_cast<uint32_t>(mesh.vertices.size());
	const uint32_t nf = static_cast<uint32_t>(mesh.faces.size());

	// Step 1: Sort vertices lexicographically (x, y, z) with EXACT comparison.
	// Tolerance is applied only during the final CanonicallyEqual vertex comparison,
	// not in the sort predicate (which must satisfy strict weak ordering).
	std::vector<uint32_t> sortedIdx(nv);
	std::iota(sortedIdx.begin(), sortedIdx.end(), 0u);
	std::sort(sortedIdx.begin(), sortedIdx.end(),
	          [&](uint32_t a, uint32_t b) {
		          const Mesh::Vertex& va = mesh.vertices[a];
		          const Mesh::Vertex& vb = mesh.vertices[b];
		          if (va.x() != vb.x())
			          return va.x() < vb.x();
		          if (va.y() != vb.y())
			          return va.y() < vb.y();
		          return va.z() < vb.z();
	          });

	// Build canonicalId[original_index] = new_index (position in sorted order).
	std::vector<uint32_t> canonicalId(nv);
	for (uint32_t i = 0; i < nv; ++i)
		canonicalId[sortedIdx[i]] = i;

	// Store sorted vertex list.
	cf.sortedVertices.resize(nv);
	for (uint32_t i = 0; i < nv; ++i)
		cf.sortedVertices[i] = mesh.vertices[sortedIdx[i]];

	// Step 2: For each face, remap to canonical vertex IDs and apply cyclic
	// rotation to start at the smallest canonical ID.
	cf.sortedFaces.resize(nf);
	for (uint32_t fi = 0; fi < nf; ++fi) {
		std::array<uint32_t, 3> tri = {
		    canonicalId[mesh.faces[fi][0]],
		    canonicalId[mesh.faces[fi][1]],
		    canonicalId[mesh.faces[fi][2]]};
		// Cyclic rotation: find the index of the minimum element.
		uint32_t minPos = 0;
		if (tri[1] < tri[minPos])
			minPos = 1;
		if (tri[2] < tri[minPos])
			minPos = 2;
		cf.sortedFaces[fi] = {
		    tri[minPos],
		    tri[(minPos + 1) % 3],
		    tri[(minPos + 2) % 3]};
	}

	// Step 3: Sort faces lexicographically.
	std::sort(cf.sortedFaces.begin(), cf.sortedFaces.end());

	return cf;
}

bool CanonicallyEqual(const Mesh& a, const Mesh& b, float posTol)
{
	if (a.vertices.size() != b.vertices.size())
		return false;
	if (a.faces.size() != b.faces.size())
		return false;

	const CanonicalForm ca = Canonicalize(a, posTol);
	const CanonicalForm cb = Canonicalize(b, posTol);

	// Compare sorted vertices within tolerance.
	for (size_t i = 0; i < ca.sortedVertices.size(); ++i) {
		if ((ca.sortedVertices[i] - cb.sortedVertices[i]).norm() > posTol)
			return false;
	}
	// Compare sorted faces exactly (canonical IDs must match).
	return ca.sortedFaces == cb.sortedFaces;
}

// ============================================================
// ANGLE DEFECT (Gauss–Bonnet)
// ============================================================

double ComputeAngleDefect(const Mesh& mesh)
{
	if (mesh.vertices.empty() || mesh.faces.empty())
		return 0.0;
	// Accumulate incident face angles per vertex.
	std::vector<double> angleSum(mesh.vertices.size(), 0.0);
	for (const auto& f : mesh.faces) {
		const Mesh::Vertex& p0 = mesh.vertices[f[0]];
		const Mesh::Vertex& p1 = mesh.vertices[f[1]];
		const Mesh::Vertex& p2 = mesh.vertices[f[2]];
		// Interior angle at each vertex of the triangle.
		auto angleAt = [](const Mesh::Vertex& v,
		                  const Mesh::Vertex& a,
		                  const Mesh::Vertex& b) -> double {
			const Mesh::Vertex ea = (a - v).normalized();
			const Mesh::Vertex eb = (b - v).normalized();
			const double dot = static_cast<double>(ea.dot(eb));
			// Clamp to [-1, 1] to guard against float rounding.
			return std::acos(std::max(-1.0, std::min(1.0, dot)));
		};
		angleSum[f[0]] += angleAt(p0, p1, p2);
		angleSum[f[1]] += angleAt(p1, p2, p0);
		angleSum[f[2]] += angleAt(p2, p0, p1);
	}
	// Angle defect at interior vertex i = 2π − Σ angles.
	// For boundary vertices the nominal sum is π (half-disk), so we skip them.
	// To determine boundary vertices: count how many edges appear only once.
	// Build edge → face count map.
	std::unordered_map<uint64_t, uint32_t> edgeCount;
	edgeCount.reserve(mesh.faces.size() * 3);
	for (const auto& f : mesh.faces) {
		for (int k = 0; k < 3; ++k) {
			const uint32_t u = f[k];
			const uint32_t v = f[(k + 1) % 3];
			const uint64_t key = (static_cast<uint64_t>(std::min(u, v)) << 32)
			                     | static_cast<uint64_t>(std::max(u, v));
			++edgeCount[key];
		}
	}
	// A vertex is on the boundary if any of its edges appears only once.
	std::vector<bool> onBoundary(mesh.vertices.size(), false);
	for (const auto& [key, cnt] : edgeCount) {
		if (cnt == 1) {
			const uint32_t u = static_cast<uint32_t>(key >> 32);
			const uint32_t v = static_cast<uint32_t>(key & 0xFFFFFFFF);
			onBoundary[u] = true;
			onBoundary[v] = true;
		}
	}
	const double twoPi = 2.0 * std::acos(-1.0);
	double totalDefect = 0.0;
	for (size_t i = 0; i < mesh.vertices.size(); ++i) {
		if (!onBoundary[i]) {
			totalDefect += twoPi - angleSum[i];
		}
	}
	return totalDefect;
}

} // namespace metrics
} // namespace hmtest

/*
* Metrics.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/metrics/Metrics.h — shared analytic metrics toolkit
//
// A reusable static library (halfmesh_metrics) used by all later test suites.
// All functions operate on the public halfmesh API only.
// Namespace: hmtest::metrics
#pragma once

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleKDTree.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace hmtest {
namespace metrics {

// ============================================================
// TOPOLOGY METRICS
// ============================================================

// Counts derived directly from mesh buffers (no half-edge needed).
struct TopologyCounts
{
	uint32_t numVertices = 0;
	uint32_t numEdges = 0; // unique undirected edges (from HalfMesh)
	uint32_t numFaces = 0;
	int32_t euler = 0; // χ = V - E + F
	// For a closed orientable surface: genus = (2 - χ - numBoundaryLoops) / 2
	// For an open surface with B boundary loops: genus = (2 - χ - B) / 2
	int32_t genus = 0;
	uint32_t numBoundaryLoops = 0;
	bool isWatertight = false; // true iff 0 boundary loops
	bool isEdgeManifold = false; // every edge shared by at most 2 faces
	bool isVertexManifold = false; // every vertex's one-ring is a disk/half-disk
};

// Compute topology counts. Builds HalfMesh internally if mesh.halfMesh is empty.
// NOTE: isVertexManifold check uses HalfMesh — only valid on edge-manifold meshes.
TopologyCounts ComputeTopology(const halfmesh::Mesh& mesh);

// Valence histogram: map from valence value → number of vertices with that valence.
// Valence = number of incident faces for a vertex (VFaceDegree in HalfMesh).
std::map<uint32_t, uint32_t> ComputeValenceHistogram(const halfmesh::Mesh& mesh);

// ============================================================
// GEOMETRY METRICS
// ============================================================

struct AABB
{
	halfmesh::Mesh::Vertex minPt;
	halfmesh::Mesh::Vertex maxPt;
	halfmesh::Mesh::Vertex size() const { return maxPt - minPt; }
};

// Surface area (sum of triangle areas).
double ComputeSurfaceArea(const halfmesh::Mesh& mesh);

// Axis-aligned bounding box of the mesh.
AABB ComputeAABB(const halfmesh::Mesh& mesh);

// Signed volume via divergence theorem (oriented tetrahedra).
// Meaningful only for closed (watertight) meshes with consistent outward normals.
double ComputeSignedVolume(const halfmesh::Mesh& mesh);

// Per-triangle quality metrics (all in [0,1] or [0, +inf)).
struct TriangleQuality
{
	float minAngleDeg = 0.f; // minimum interior angle in degrees
	float maxAngleDeg = 0.f; // maximum interior angle in degrees
	// aspectRatio = longest_edge / shortest_edge (1 = equilateral)
	float aspectRatio = 0.f;
	// radiusRatio = inradius / circumradius * 2 (1 = equilateral)
	float radiusRatio = 0.f;
};

// Compute quality stats for a single triangle defined by three 3-D points.
TriangleQuality ComputeTriangleQuality(
    const halfmesh::Mesh::Vertex& a,
    const halfmesh::Mesh::Vertex& b,
    const halfmesh::Mesh::Vertex& c);

// Compute quality stats for each face; results indexed by face index.
std::vector<TriangleQuality> ComputeAllTriangleQualities(const halfmesh::Mesh& mesh);

// Edge-length distribution statistics.
struct EdgeLengthStats
{
	double minLen = 0.0;
	double maxLen = 0.0;
	double meanLen = 0.0;
	double stddev = 0.0;
};

EdgeLengthStats ComputeEdgeLengthStats(const halfmesh::Mesh& mesh);

// Integrated angle defect (Gauss–Bonnet): sum over all vertices of (2π - sum of
// incident face angles). For a closed genus-0 surface equals 4π; for a genus-1
// surface (torus) equals 0.
// Only meaningful on edge-manifold meshes; ignores boundary contributions.
double ComputeAngleDefect(const halfmesh::Mesh& mesh);

// ============================================================
// DISTANCE METRICS (KD-tree + brute-force cross-check)
// ============================================================

// Distance results.
struct DistanceResult
{
	double hausdorffSymmetric = 0.0; // max(h(A→B), h(B→A))
	double meanSurfaceDist = 0.0; // mean of per-vertex nearest-surface dist
};

// Compute distance metrics between two meshes using TriangleKdTree.
// Samples: all vertices of each mesh against the other's surface.
DistanceResult ComputeDistanceKdTree(
    const halfmesh::Mesh& meshA,
    const halfmesh::Mesh& meshB);

// Brute-force O(n·m) point-to-triangle distance (for cross-checking on small meshes).
// Returns the distance from point p to the nearest point on triangle (a, b, c).
float PointToTriangleDistSq(
    const halfmesh::Mesh::Vertex& p,
    const halfmesh::Mesh::Vertex& a,
    const halfmesh::Mesh::Vertex& b,
    const halfmesh::Mesh::Vertex& c);

// Brute-force version of ComputeDistanceKdTree — O(|V_A|·|F_B| + |V_B|·|F_A|).
// Use on small meshes only. Results should match KD-tree within float epsilon.
DistanceResult ComputeDistanceBruteForce(
    const halfmesh::Mesh& meshA,
    const halfmesh::Mesh& meshB);

// ============================================================
// UV / ATLAS METRICS
// ============================================================

// UV metrics computed from mesh.faceTexcoords (layout: 3 TexCoords per face).
struct UVMetrics
{
	int flipCount = 0; // triangles with UV orientation opposite to majority
	double symDirichlet = 0.0; // area-weighted mean symmetric-Dirichlet energy (min = 4)
	double stretchL2 = 0.0; // Sander L2 stretch metric
	double atlasOccupancy = 0.0; // fraction of [0,1]² atlas covered by UV triangles
	bool hasBboxOverlaps = false; // true if any two triangles' UV bounding boxes overlap
	// (cheap axis-aligned test, not pixel-exact)
	int numFaces = 0;
	bool allFinite = true;
};

// Compute UV metrics. Requires mesh.faceTexcoords.size() == mesh.faces.size()*3.
UVMetrics ComputeUVMetrics(const halfmesh::Mesh& mesh);

// Per-chart texel density: sqrt(uvArea / worldArea) for each chart.
// faceChart[i] = chart id for face i; numCharts = number of distinct charts.
std::vector<double> ComputeChartTexelDensity(
    const halfmesh::Mesh& mesh,
    const std::vector<unsigned>& faceChart,
    unsigned numCharts);

// ============================================================
// CHART-PARTITION METRICS (segmentation quality)
// ============================================================

// Sum of 3-D lengths of edges separating two different charts (plus chart-
// partition boundary edges).  Lower = fewer/shorter seams.
double ComputeBoundaryCutLength(const halfmesh::Mesh& mesh,
                                const std::vector<unsigned>& faceChart);

// Per-chart area-weighted mean angle (radians) between each face normal and the
// chart's area-weighted mean normal.  Index = chart id.  Lower = more planar.
std::vector<double> ComputeChartPlanarityError(const halfmesh::Mesh& mesh,
                                               const std::vector<unsigned>& faceChart,
                                               unsigned numCharts);

// Per-chart compactness = perimeter^2 / (4*pi*area) of the 3-D chart boundary
// (1 = disk-like, larger = more ragged).  Index = chart id.
std::vector<double> ComputeChartCompactness(const halfmesh::Mesh& mesh,
                                            const std::vector<unsigned>& faceChart,
                                            unsigned numCharts);

// true iff faceChart.size()==faces.size() and every entry is in [0,numCharts).
bool ComputeChartCoverage(const halfmesh::Mesh& mesh,
                          const std::vector<unsigned>& faceChart,
                          unsigned numCharts);

// ============================================================
// ROBUSTNESS / NaN-INF SCAN
// ============================================================

// Returns true if all mesh buffers (vertices, faceNormals, faceTexcoords) are
// finite (no NaN or Inf). Returns false on any non-finite value found.
bool ScanFinite(const halfmesh::Mesh& mesh);

// ============================================================
// CANONICALIZATION
// ============================================================

// Canonical signature for a triangulated mesh: compare up to vertex/face
// relabeling and per-face cyclic rotation (same orientation).
// Two meshes are "canonically equal" if they represent the same abstract
// triangulation (same combinatorial + geometric structure within posTol).
//
// Algorithm:
//   1. Sort vertices by (x, y, z) within posTol → assign canonical vertex IDs.
//   2. For each face, rotate the 3 vertex indices to start at the smallest
//      canonical ID → gives a canonical face triple.
//   3. Sort the canonical face triples.
//   4. Compare the sorted lists.
struct CanonicalForm
{
	// Sorted vertex list (each row is a vertex, sorted lexicographically)
	std::vector<halfmesh::Mesh::Vertex> sortedVertices;
	// Sorted face triples using canonical vertex IDs
	std::vector<std::array<uint32_t, 3>> sortedFaces;
};

CanonicalForm Canonicalize(const halfmesh::Mesh& mesh, float posTol = 1e-5f);

// Returns true if two meshes are canonically equal (same topology + geometry
// up to relabeling and cyclic rotation, within posTol).
bool CanonicallyEqual(
    const halfmesh::Mesh& a,
    const halfmesh::Mesh& b,
    float posTol = 1e-5f);

} // namespace metrics
} // namespace hmtest

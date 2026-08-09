/*
* MeshHolesTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for MeshHoles.cpp:
//   Mesh::CloseHoles — Liepa minimum-weight hole filling (+ refine + fairing).
//
// Tests:
//   1. Synthetic hexagonal hole: a planar ring of triangles with the centre
//      missing -> exactly one boundary loop; CloseHoles closes it, mesh becomes
//      watertight (no boundary loops), face count grows, holesFaces reports the
//      new faces, filled patch stays near the hole plane.
//   2. mesh.ply with a contiguous patch of faces removed (vertices kept) -> one
//      new hole; CloseHoles closes it; boundary-loop count drops back; result is
//      a valid manifold (ListHalfEdges rebuilds, no degenerate faces, returned
//      count matches), face count increases.
//   3. Watertight mesh has no holes: CloseHoles returns 0, mesh unchanged.
//   4. nCloseHoles cap: with several holes only the requested number are closed.

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleKDTree.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string TestMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

// Count boundary loops (holes) via the library's own hole enumeration.
// Force a rebuild: Mesh::ListHalfEdges() caches by vertex count, so after a
// face-only edit (RemoveFaces keeps vertices) we must clear it to rebuild.
static unsigned CountBoundaryLoops(Mesh& m)
{
	m.halfMesh.Clear();
	m.ListHalfEdges();
	if (m.halfMesh.Empty())
		return 0;
	std::vector<std::vector<Mesh::VIndex>> holes;
	m.halfMesh.EnumerateHoles(holes);
	return static_cast<unsigned>(holes.size());
}

// Number of duplicate faces (winding-agnostic: sorted vertex triple).
static unsigned CountDuplicateFaces(const Mesh& m)
{
	std::map<std::array<Mesh::VIndex, 3>, int> seen;
	for (const auto& f : m.faces) {
		std::array<Mesh::VIndex, 3> key{f[0], f[1], f[2]};
		std::sort(key.begin(), key.end());
		++seen[key];
	}
	unsigned dup = 0;
	for (const auto& kv : seen)
		if (kv.second > 1)
			dup += static_cast<unsigned>(kv.second - 1);
	return dup;
}

// Number of faces with a repeated or NaN vertex (should be zero).
static unsigned CountDegenerateFaces(const Mesh& m)
{
	unsigned bad = 0;
	for (const auto& f : m.faces) {
		if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) {
			++bad;
			continue;
		}
		const Mesh::Vertex& a = m.vertices[f[0]];
		const Mesh::Vertex& b = m.vertices[f[1]];
		const Mesh::Vertex& c = m.vertices[f[2]];
		if ((b - a).cross(c - a).norm() <= 0.f)
			++bad;
	}
	return bad;
}

// Build a flat (z=0) triangulated grid of (W x H) quads, each split into two
// triangles with consistent +z winding, then remove the faces touching the
// single interior vertex nearest the centre to carve an interior polygonal hole
// surrounded entirely by mesh (so the hole is a separate boundary loop, in
// addition to the outer rim).  Returns the holed mesh.
static Mesh MakeGridWithInteriorHole(int W = 4, int H = 4)
{
	Mesh m;
	const int nx = W + 1, ny = H + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y < ny; ++y)
		for (int x = 0; x < nx; ++x)
			m.vertices.emplace_back(static_cast<float>(x),
			                        static_cast<float>(y), 0.f);
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			// two triangles, CCW from +z
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	}
	// carve a hole around an interior vertex (not on the outer rim)
	const int cx = W / 2, cy = H / 2;
	const int center = vid(cx, cy);
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f) {
		const auto& fc = m.faces[f];
		if (fc[0] == center || fc[1] == center || fc[2] == center)
			rm.push_back(f);
	}
	m.RemoveFaces(rm); // keep vertices -> interior hole
	m.RemoveUnreferencedVertices(); // drop the now-isolated centre vertex
	return m;
}

// ---------------------------------------------------------------------------
// 1. Synthetic hexagonal hole
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, FillsSyntheticPolygonHole)
{
	Mesh m = MakeGridWithInteriorHole(4, 4);

	// The holed grid has exactly TWO boundary loops: the interior hole and the
	// outer rim. We want CloseHoles to fill the smaller (interior) one.
	const unsigned loopsBefore = CountBoundaryLoops(m);
	ASSERT_EQ(loopsBefore, 2u);

	const size_t facesBefore = m.faces.size();

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	const unsigned closed = m.CloseHoles(1, &holesFaces);

	EXPECT_EQ(closed, 1u);
	ASSERT_EQ(holesFaces.size(), 1u);
	EXPECT_FALSE(holesFaces[0].empty());
	EXPECT_GT(m.faces.size(), facesBefore);

	// Exactly one boundary loop should remain (the outer rim).
	const unsigned loopsAfter = CountBoundaryLoops(m);
	EXPECT_EQ(loopsAfter, 1u);

	// No degenerate faces introduced.
	EXPECT_EQ(CountDegenerateFaces(m), 0u);

	// Geometric sanity: every patch vertex lies near the z=0 hole plane.
	std::set<Mesh::VIndex> patchVerts;
	for (Mesh::FIndex f : holesFaces[0])
		for (int e = 0; e < 3; ++e)
			patchVerts.insert(m.faces[f][e]);
	for (Mesh::VIndex v : patchVerts)
		EXPECT_NEAR(m.vertices[v].z(), 0.f, 1e-3f);
}

// ---------------------------------------------------------------------------
// 1b. Wavy (non-planar) hole: the refine-stage valence/cap flips act on
// non-planar quads, where a flip can fold a patch triangle over. After
// filling, no two edge-adjacent faces touching the patch may have opposing
// normals (a fold shows up as adjacent unit-normal dot ≈ -1).
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, WavyHolePatchKeepsOrientation)
{
	const int W = 10, H = 10;
	Mesh m;
	const int nx = W + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y <= H; ++y)
		for (int x = 0; x <= W; ++x)
			m.vertices.emplace_back(
			    static_cast<float>(x), static_cast<float>(y),
			    0.8f * std::sin(1.7f * static_cast<float>(x)) * std::cos(1.3f * static_cast<float>(y)));
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	// carve a 3x3-vertex hole in the middle -> jagged non-planar boundary
	std::vector<Mesh::FIndex> rm;
	auto inHole = [&](int v) {
		const int x = v % nx, y = v / nx;
		return x >= 4 && x <= 6 && y >= 4 && y <= 6;
	};
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inHole(m.faces[f][0]) || inHole(m.faces[f][1]) || inHole(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();
	ASSERT_EQ(CountBoundaryLoops(m), 2u);

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	ASSERT_EQ(m.CloseHoles(1, &holesFaces), 1u);
	ASSERT_EQ(CountBoundaryLoops(m), 1u);
	EXPECT_EQ(CountDegenerateFaces(m), 0u);

	// Orientation consistency around the patch: for every mesh edge shared by
	// two faces of which at least one belongs to the patch, the unit normals
	// must not oppose each other.
	std::set<Mesh::FIndex> patch(holesFaces[0].begin(), holesFaces[0].end());
	std::map<std::pair<Mesh::VIndex, Mesh::VIndex>, std::vector<Mesh::FIndex>> emap;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = m.faces[f][e], b = m.faces[f][(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			emap[{a, b}].push_back(f);
		}
	auto unitNormal = [&](Mesh::FIndex f) {
		const auto& fc = m.faces[f];
		return Eigen::Vector3f((m.vertices[fc[1]] - m.vertices[fc[0]])
		                           .cross(m.vertices[fc[2]] - m.vertices[fc[0]])
		                           .normalized());
	};
	for (const auto& kv : emap) {
		if (kv.second.size() != 2)
			continue;
		const Mesh::FIndex f0 = kv.second[0], f1 = kv.second[1];
		if (!patch.count(f0) && !patch.count(f1))
			continue;
		EXPECT_GT(unitNormal(f0).dot(unitNormal(f1)), -0.5f)
		    << "folded patch triangle: faces " << f0 << " and " << f1
		    << " have opposing normals";
	}
}

// ---------------------------------------------------------------------------
// 1c. Denser wavy hole that triggers refine collapses. A hole-patch edge
// collapse must not fold a surviving triangle over (orientation gate) nor
// create a duplicate face (link condition) — the same guards flips already had.
// The larger hole + short mean boundary edge exercises CollapseShortEdges hard.
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, DenseWavyHolePatchCollapsesStayValid)
{
	const int W = 14, H = 14;
	Mesh m;
	const int nx = W + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y <= H; ++y)
		for (int x = 0; x <= W; ++x)
			m.vertices.emplace_back(
			    static_cast<float>(x), static_cast<float>(y),
			    0.35f * std::sin(0.8f * static_cast<float>(x)) * std::cos(0.8f * static_cast<float>(y)));
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	// carve a 5x5-vertex hole in the middle -> non-planar patch that refine must
	// remesh (split/collapse ops), mild enough that a correct fill has no folds.
	auto inHole = [&](int v) {
		const int x = v % nx, y = v / nx;
		return x >= 5 && x <= 9 && y >= 5 && y <= 9;
	};
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inHole(m.faces[f][0]) || inHole(m.faces[f][1]) || inHole(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();
	ASSERT_EQ(CountBoundaryLoops(m), 2u);

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	ASSERT_EQ(m.CloseHoles(1, &holesFaces), 1u);
	ASSERT_EQ(CountBoundaryLoops(m), 1u);
	EXPECT_EQ(CountDegenerateFaces(m), 0u);
	// Link condition: a collapse must never emit a face duplicating another.
	EXPECT_EQ(CountDuplicateFaces(m), 0u);

	// Orientation: no edge-adjacent pair touching the patch may have opposing
	// (folded) normals.
	std::set<Mesh::FIndex> patch(holesFaces[0].begin(), holesFaces[0].end());
	std::map<std::pair<Mesh::VIndex, Mesh::VIndex>, std::vector<Mesh::FIndex>> emap;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = m.faces[f][e], b = m.faces[f][(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			emap[{a, b}].push_back(f);
		}
	auto unitNormal = [&](Mesh::FIndex f) {
		const auto& fc = m.faces[f];
		return Eigen::Vector3f((m.vertices[fc[1]] - m.vertices[fc[0]])
		                           .cross(m.vertices[fc[2]] - m.vertices[fc[0]])
		                           .normalized());
	};
	for (const auto& kv : emap) {
		if (kv.second.size() != 2)
			continue;
		const Mesh::FIndex f0 = kv.second[0], f1 = kv.second[1];
		if (!patch.count(f0) && !patch.count(f1))
			continue;
		EXPECT_GT(unitNormal(f0).dot(unitNormal(f1)), -0.5f)
		    << "folded patch triangle: faces " << f0 << " and " << f1;
	}
}

// ---------------------------------------------------------------------------
// 1d. Analytic sphere (dome) hole: thin-plate fairing must keep the filled
// patch ON the surrounding surface. A patch-only bi-Laplacian is blind to the
// surrounding tangent and sags the fill toward the chord; augmenting the solve
// with the parent one-ring around the boundary makes the patch follow the
// sphere. Every dome vertex sits at distance R from the origin, so patch
// vertices (incl. refine/fairing interior ones) must stay near radius R.
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, DomeHolePatchFollowsSphereCurvature)
{
	const int W = 16, H = 16;
	const float R = 10.f, span = 10.f;
	Mesh m;
	const int nx = W + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y <= H; ++y)
		for (int x = 0; x <= W; ++x) {
			const float fx = span * (static_cast<float>(x) / W - 0.5f);
			const float fy = span * (static_cast<float>(y) / H - 0.5f);
			const float z = std::sqrt(std::max(0.f, R * R - fx * fx - fy * fy));
			m.vertices.emplace_back(fx, fy, z); // exactly on the sphere of radius R
		}
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	auto inHole = [&](int v) { const int x = v % nx, y = v / nx; return x >= 5 && x <= 11 && y >= 5 && y <= 11; };
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inHole(m.faces[f][0]) || inHole(m.faces[f][1]) || inHole(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();
	ASSERT_EQ(CountBoundaryLoops(m), 2u);

	const size_t vBefore = m.vertices.size();
	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	ASSERT_EQ(m.CloseHoles(1, &holesFaces), 1u);
	EXPECT_EQ(CountDegenerateFaces(m), 0u);

	std::set<Mesh::VIndex> patchVerts;
	for (const auto& hf : holesFaces)
		for (Mesh::FIndex f : hf)
			for (int e = 0; e < 3; ++e)
				patchVerts.insert(m.faces[f][e]);
	// refine must have added interior vertices (else fairing never runs and the
	// test is vacuous).
	bool hasInterior = false;
	float maxDev = 0.f;
	for (Mesh::VIndex v : patchVerts) {
		if (v >= vBefore)
			hasInterior = true;
		maxDev = std::max(maxDev, std::fabs(m.vertices[v].norm() - R));
	}
	EXPECT_TRUE(hasInterior) << "refine added no interior vertices";
	// Ring-supported fairing follows the sphere (< ~0.5% of R here); a patch-only
	// bi-Laplacian sags the fill (> 3% of R).
	EXPECT_LT(maxDev, 0.015f * R)
	    << "patch drifted off the sphere: max radial deviation " << maxDev;
}

// ---------------------------------------------------------------------------
// 1e. Valence-driven flips must score hole-boundary vertices at their full-mesh
// (post-fill) valence, not their patch-only edge count. Counting only patch
// edges makes a boundary vertex look under-connected (2-3 vs target), so flips
// pile diagonals onto vertices already crowded by the surrounding mesh. Seeding
// with the parent one-ring degree keeps the closed-mesh valence near ideal (6)
// around the seam.
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, HolePatchFlipsUseFullMeshValence)
{
	const int W = 14, H = 14;
	Mesh m;
	const int nx = W + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y <= H; ++y)
		for (int x = 0; x <= W; ++x)
			m.vertices.emplace_back(
			    static_cast<float>(x), static_cast<float>(y),
			    0.35f * std::sin(0.8f * static_cast<float>(x)) * std::cos(0.8f * static_cast<float>(y)));
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	auto inHole = [&](int v) { const int x = v % nx, y = v / nx; return x >= 5 && x <= 9 && y >= 5 && y <= 9; };
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inHole(m.faces[f][0]) || inHole(m.faces[f][1]) || inHole(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();

	const size_t vBefore = m.vertices.size();
	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	ASSERT_EQ(m.CloseHoles(1, &holesFaces), 1u);

	// original vertices that end up in the patch = the hole boundary loop; each
	// becomes fully interior, so ideal valence is 6.
	std::set<Mesh::VIndex> loopVerts;
	for (const auto& hf : holesFaces)
		for (Mesh::FIndex f : hf)
			for (int e = 0; e < 3; ++e)
				if (m.faces[f][e] < vBefore)
					loopVerts.insert(m.faces[f][e]);
	ASSERT_FALSE(loopVerts.empty());

	std::map<Mesh::VIndex, int> degree;
	std::set<std::pair<Mesh::VIndex, Mesh::VIndex>> edges;
	for (const auto& f : m.faces)
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = f[e], b = f[(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			edges.insert({a, b});
		}
	for (const auto& pr : edges) {
		++degree[pr.first];
		++degree[pr.second];
	}
	int maxValence = 0;
	long energy = 0;
	for (Mesh::VIndex v : loopVerts) {
		const int d = degree[v];
		maxValence = std::max(maxValence, d);
		energy += static_cast<long>(d - 6) * (d - 6);
	}
	// Patch-local valence over-connects the seam (a valence-8 vertex, energy ~11);
	// full-mesh seeding keeps it near ideal (max 7, energy ~2).
	EXPECT_LE(maxValence, 7) << "hole-boundary vertex over-connected";
	EXPECT_LT(energy, 7) << "seam valence-deviation energy too high: " << energy;
}

// ---------------------------------------------------------------------------
// 1f. Refine must apply ALL non-conflicting short-edge collapses per snapshot,
// not one-per-rebuild. The old cap (10 collapses per refine iteration) starved
// collapses on larger patches, leaving many interior edges below the 0.7*mean
// floor; batching drives almost all patch edges above it.
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, HolePatchCollapsesReachEdgeLengthFloor)
{
	const int W = 14, H = 14;
	Mesh m;
	const int nx = W + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y <= H; ++y)
		for (int x = 0; x <= W; ++x)
			m.vertices.emplace_back(
			    static_cast<float>(x), static_cast<float>(y),
			    0.35f * std::sin(0.8f * static_cast<float>(x)) * std::cos(0.8f * static_cast<float>(y)));
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	auto inHole = [&](int v) { const int x = v % nx, y = v / nx; return x >= 5 && x <= 9 && y >= 5 && y <= 9; };
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inHole(m.faces[f][0]) || inHole(m.faces[f][1]) || inHole(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	ASSERT_EQ(m.CloseHoles(1, &holesFaces), 1u);

	// collect the patch's undirected edges and their lengths
	std::set<std::pair<Mesh::VIndex, Mesh::VIndex>> pedges;
	for (Mesh::FIndex f : holesFaces[0])
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = m.faces[f][e], b = m.faces[f][(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			pedges.insert({a, b});
		}
	ASSERT_GT(pedges.size(), 40u) << "patch too small to exercise collapse batching";
	double sum = 0.0;
	for (const auto& pr : pedges)
		sum += (m.vertices[pr.first] - m.vertices[pr.second]).norm();
	const double mean = sum / static_cast<double>(pedges.size());
	unsigned shortEdges = 0;
	for (const auto& pr : pedges)
		if ((m.vertices[pr.first] - m.vertices[pr.second]).norm() < 0.7 * mean)
			++shortEdges;
	// Batched collapses leave < ~5% of patch edges below the floor; the old
	// one-collapse-per-rebuild cap leaves ~30% on this patch.
	EXPECT_LT(shortEdges, pedges.size() / 10)
	    << shortEdges << " of " << pedges.size() << " patch edges below 0.7*mean";
}

// ---------------------------------------------------------------------------
// 2. mesh.ply with a removed contiguous patch
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, FillsHoleInRealMesh)
{
	Mesh m;
	ASSERT_TRUE(m.Load(TestMeshPath())) << "could not load " << TestMeshPath();

	// Clean the mesh to a manifold so a single-face removal carves a clean hole.
	m.RemoveDuplicateFaces();
	m.RemoveUnreferencedVertices();
	m.RemoveDegenerateFaces();
	m.RemoveUnreferencedVertices();
	m.ListVertexFaces();
	m.FixNonManifold();

	const unsigned loopsBefore = CountBoundaryLoops(m);

	// Build a clean half-edge view to identify strictly-interior vertices.
	m.ListHalfEdges();
	ASSERT_FALSE(m.halfMesh.Empty());
	const HalfMesh& hm = m.halfMesh;

	// edge -> incidence count, from m.faces (independent of half-edge state).
	std::map<std::pair<Mesh::VIndex, Mesh::VIndex>, int> edgeCount;
	auto ekey = [](Mesh::VIndex a, Mesh::VIndex b) {
		return std::make_pair(std::min(a, b), std::max(a, b));
	};
	for (const auto& f : m.faces)
		for (int e = 0; e < 3; ++e)
			++edgeCount[ekey(f[e], f[(e + 1) % 3])];

	// Pick a face whose three edges are each shared by exactly two faces AND
	// whose three vertices are all strictly interior (on no existing boundary
	// loop). Removing it then carves a clean, isolated triangular hole that adds
	// exactly one new boundary loop (it cannot merge with an existing one).
	Mesh::FIndex seed = math::NO_ID;
	for (Mesh::FIndex f = 0; f < m.faces.size() && seed == math::NO_ID; ++f) {
		const Mesh::Face& fc = m.faces[f];
		if (hm.VIsBoundary(fc[0]) || hm.VIsBoundary(fc[1]) || hm.VIsBoundary(fc[2]))
			continue;
		bool clean = true;
		for (int e = 0; e < 3; ++e)
			if (edgeCount[ekey(fc[e], fc[(e + 1) % 3])] != 2) {
				clean = false;
				break;
			}
		if (clean)
			seed = f;
	}
	ASSERT_NE(seed, math::NO_ID) << "no clean strictly-interior face found";

	std::vector<Mesh::FIndex> removes{seed};
	const size_t patchSize = removes.size();
	m.RemoveFaces(removes); // keeps vertices -> creates a triangular hole

	const unsigned loopsHoled = CountBoundaryLoops(m);
	ASSERT_GT(loopsHoled, loopsBefore)
	    << "removing the patch should have created a new boundary loop";

	const size_t facesAfterRemove = m.faces.size();

	// snapshot the holed surface for a one-sided surface-fidelity check
	const Mesh holedSnapshot = m;
	const float bboxDiag = holedSnapshot.ComputeAABBox().diagonal().norm();

	// Fill only the smallest hole (= the triangular hole we just carved, which is
	// far smaller than the mesh's pre-existing boundary loops). This isolates the
	// assertion to the patch we created.
	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	const unsigned closed = m.CloseHoles(1, &holesFaces);

	EXPECT_EQ(closed, 1u);
	EXPECT_EQ(closed, holesFaces.size());
	EXPECT_GT(m.faces.size(), facesAfterRemove);

	// Exactly one boundary loop closed -> loop count drops by exactly one, back
	// to the original count.
	const unsigned loopsClosed = CountBoundaryLoops(m);
	EXPECT_EQ(loopsClosed, loopsBefore);
	EXPECT_EQ(loopsClosed, loopsHoled - 1);

	// Result is a valid manifold: half-edge rebuild succeeds, no degenerates.
	m.ListHalfEdges();
	EXPECT_FALSE(m.halfMesh.Empty());
	EXPECT_EQ(CountDegenerateFaces(m), 0u);

	// Filled at least as many triangles as we removed (a hole needs covering).
	size_t added = 0;
	for (const auto& hf : holesFaces)
		added += hf.size();
	EXPECT_GE(added, patchSize > 0 ? 1u : 0u);

	// Surface fidelity: every vertex of every patch face stays close to the
	// surrounding (holed) surface — a one-sided distance check that also covers
	// any interior (refine/fairing) vertices added by the patch.
	{
		TriangleKdTree kdtree(holedSnapshot);
		std::set<Mesh::VIndex> patchVerts;
		for (const auto& hf : holesFaces)
			for (Mesh::FIndex f : hf)
				for (int e = 0; e < 3; ++e)
					patchVerts.insert(m.faces[f][e]);
		ASSERT_FALSE(patchVerts.empty());
		for (Mesh::VIndex v : patchVerts) {
			const auto nn = kdtree.NearestPoint(m.vertices[v]);
			ASSERT_TRUE(nn.IsValid());
			EXPECT_LT(nn.dist, 0.05f * bboxDiag)
			    << "patch vertex " << v << " drifted from the surface";
		}
	}
}

// ---------------------------------------------------------------------------
// 3. Watertight mesh -> nothing to close
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, NoHolesReturnsZero)
{
	// A closed tetrahedron.
	Mesh m;
	m.vertices = {
	    Mesh::Vertex(0, 0, 0),
	    Mesh::Vertex(1, 0, 0),
	    Mesh::Vertex(0, 1, 0),
	    Mesh::Vertex(0, 0, 1),
	};
	m.faces = {
	    Mesh::Face(0, 2, 1),
	    Mesh::Face(0, 1, 3),
	    Mesh::Face(0, 3, 2),
	    Mesh::Face(1, 2, 3),
	};
	ASSERT_EQ(CountBoundaryLoops(m), 0u);

	const size_t facesBefore = m.faces.size();
	const unsigned closed = m.CloseHoles();
	EXPECT_EQ(closed, 0u);
	EXPECT_EQ(m.faces.size(), facesBefore);
}

// ---------------------------------------------------------------------------
// 4. nCloseHoles cap respected
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, RespectsCloseLimit)
{
	// Two independent holed grids, each with its own interior hole + outer rim.
	Mesh a = MakeGridWithInteriorHole(4, 4);
	Mesh m = a;
	// offset and append a second holed grid
	const Mesh::VIndex voff = static_cast<Mesh::VIndex>(m.vertices.size());
	for (const auto& v : a.vertices)
		m.vertices.emplace_back(v + Mesh::Vertex(100, 0, 0));
	for (const auto& f : a.faces)
		m.faces.emplace_back(Mesh::Face(f[0] + voff, f[1] + voff, f[2] + voff));

	// 2 interior holes + 2 outer rims = 4 boundary loops.
	ASSERT_EQ(CountBoundaryLoops(m), 4u);

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	const unsigned closed = m.CloseHoles(1, &holesFaces); // cap at 1
	EXPECT_EQ(closed, 1u);
	EXPECT_EQ(holesFaces.size(), 1u);

	// Three boundary loops remain (one inner hole still open + two rims).
	EXPECT_EQ(CountBoundaryLoops(m), 3u);
}

// ---------------------------------------------------------------------------
// 4b. Parallel fill determinism: closing many holes in one call runs the fills
// on a thread pool and harvests them in a fixed order, so two runs of the same
// input must produce byte-identical output (no unordered-container / thread
// scheduling leakage).
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, ParallelCloseHolesDeterministic)
{
	// Tile a 3x3 grid of holed patches so CloseHoles has many independent holes.
	Mesh tile = MakeGridWithInteriorHole(4, 4);
	Mesh m;
	for (int gx = 0; gx < 3; ++gx)
		for (int gy = 0; gy < 3; ++gy) {
			const Mesh::VIndex voff = static_cast<Mesh::VIndex>(m.vertices.size());
			for (const auto& v : tile.vertices)
				m.vertices.emplace_back(v + Mesh::Vertex(gx * 20.f, gy * 20.f, 0.f));
			for (const auto& f : tile.faces)
				m.faces.emplace_back(Mesh::Face(f[0] + voff, f[1] + voff, f[2] + voff));
		}

	Mesh a = m, b = m;
	const unsigned ca = a.CloseHoles(9);
	const unsigned cb = b.CloseHoles(9);
	ASSERT_EQ(ca, cb);
	ASSERT_GE(ca, 2u) << "expected several holes filled";
	ASSERT_EQ(a.vertices.size(), b.vertices.size());
	ASSERT_EQ(a.faces.size(), b.faces.size());
	for (size_t i = 0; i < a.vertices.size(); ++i)
		EXPECT_TRUE(a.vertices[i] == b.vertices[i]) << "vertex " << i << " differs between runs";
	for (size_t i = 0; i < a.faces.size(); ++i)
		EXPECT_TRUE(a.faces[i] == b.faces[i]) << "face " << i << " differs between runs";
}

// Deterministic irregular NON-planar hole: perturb a grid (fixed seed) in x,y,z
// and carve a hole around the central block of vertices, so both the hole
// boundary and its surrounding surface are non-planar (a saddle-like patch).
static Mesh MakeIrregularHole(int W, int H, unsigned seed, int block)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> d(-0.45f, 0.45f);
	std::uniform_real_distribution<float> dz(-1.2f, 1.2f);
	Mesh m;
	const int nx = W + 1, ny = H + 1;
	auto vid = [nx](int x, int y) { return y * nx + x; };
	for (int y = 0; y < ny; ++y)
		for (int x = 0; x < nx; ++x)
			m.vertices.emplace_back(x + d(rng), y + d(rng), dz(rng));
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x) {
			const int v00 = vid(x, y), v10 = vid(x + 1, y);
			const int v01 = vid(x, y + 1), v11 = vid(x + 1, y + 1);
			m.faces.emplace_back(Mesh::Face(v00, v10, v11));
			m.faces.emplace_back(Mesh::Face(v00, v11, v01));
		}
	const int cx = W / 2, cy = H / 2;
	auto inBlock = [&](int v) {
		const int x = v % nx, y = v / nx;
		return x >= cx && x < cx + block && y >= cy && y < cy + block;
	};
	std::vector<Mesh::FIndex> rm;
	for (Mesh::FIndex f = 0; f < m.faces.size(); ++f)
		if (inBlock(m.faces[f][0]) || inBlock(m.faces[f][1]) || inBlock(m.faces[f][2]))
			rm.push_back(f);
	m.RemoveFaces(rm);
	m.RemoveUnreferencedVertices();
	return m;
}

// Fill the smallest hole of m via the raw Liepa DP (HalfMesh::TriangulateHole,
// no refine/fairing) and return the summed area of the fill triangles.
static double FillDPArea(const Mesh& m)
{
	HalfMesh hm(m);
	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	EXPECT_FALSE(holes.empty());
	if (holes.empty())
		return -1.0;
	size_t smallest = 0;
	for (size_t i = 1; i < holes.size(); ++i)
		if (holes[i].size() < holes[smallest].size())
			smallest = i;
	const HalfMesh::HIndex iHeStart = hm.HeTwin(hm.VHalfedge(holes[smallest][0]));
	const HalfMesh::FIndex before = hm.FSize();
	hm.TriangulateHole(iHeStart, m.vertices);
	std::vector<HalfMesh::Face> faces;
	hm.FFaces(faces);
	double area = 0.0;
	for (size_t f = before; f < faces.size(); ++f) {
		const auto& fc = faces[f];
		const Eigen::Vector3d a = m.vertices[fc[0]].cast<double>();
		const Eigen::Vector3d b = m.vertices[fc[1]].cast<double>();
		const Eigen::Vector3d c = m.vertices[fc[2]].cast<double>();
		area += 0.5 * (b - a).cross(c - a).norm();
	}
	return area;
}

// Independent oracle for LiepaMinimizesSummedAreaNotSquared. Reimplements the
// Klincsek min-weight hole-triangulation DP FROM SCRATCH (it does NOT call the
// production HalfMesh::TriangulateHole) with the SQUARED-area term |cross|^2 as
// the lexicographic tie-break after the max-dihedral angle — i.e. the metric the
// DP used before the Liepa summed-area fix. It returns the SUMMED REAL area of
// the triangulation that criterion selects on m's smallest hole. The DP mirrors
// the production weight structure exactly (same dihedral-angle term, same
// interior-edge guard, same boundary enumeration) so only the area metric
// differs; it runs in double for stability across build flags.
static double SquaredCriterionFillArea(const Mesh& m)
{
	HalfMesh hm(m);
	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	EXPECT_FALSE(holes.empty());
	if (holes.empty())
		return -1.0;
	size_t smallest = 0;
	for (size_t i = 1; i < holes.size(); ++i)
		if (holes[i].size() < holes[smallest].size())
			smallest = i;
	// Enumerate the hole boundary in the SAME order production's TriangulateHole
	// does (so the DP triangulates the identical indexed polygon).
	const HalfMesh::HIndex iHeStart = hm.HeTwin(hm.VHalfedge(holes[smallest][0]));
	std::vector<HalfMesh::VIndex> hv;
	HalfMesh::HIndex iHe = iHeStart;
	do {
		hv.emplace_back(hm.HeVertex(iHe));
	} while ((iHe = hm.HeNext(iHe)) != iHeStart);
	const int n = static_cast<int>(hv.size());
	if (n < 3)
		return -1.0;

	const auto pos = [&](HalfMesh::VIndex v) -> Eigen::Vector3d {
		return m.vertices[v].cast<double>();
	};
	const auto normalOf = [&](HalfMesh::VIndex a, HalfMesh::VIndex b,
	                          HalfMesh::VIndex c) -> Eigen::Vector3d {
		return (pos(b) - pos(a)).cross(pos(c) - pos(a)).normalized();
	};
	// opposite vertex across the boundary edge at hole vertex i (production VOpposite)
	const auto vOpp = [&](int i) -> HalfMesh::VIndex {
		return hm.HeHeadVertex(hm.HeNext(hm.VHalfedge(hv[i])));
	};
	const auto eIsInterior = [&](HalfMesh::VIndex v0, HalfMesh::VIndex v1) -> bool {
		const HalfMesh::EIndex e = hm.EEdge(v0, v1);
		return e != math::NO_ID ? !hm.EIsBoundary(e) : false;
	};

	struct W
	{
		double angle = std::numeric_limits<double>::max();
		double area = std::numeric_limits<double>::max();
		W() = default;
		W(double a, double ar) :
		    angle(a), area(ar) {}
		W operator+(const W& r) const { return W(std::max(angle, r.angle), area + r.area); }
		bool operator<(const W& r) const
		{
			return angle < r.angle || (angle == r.angle && area < r.area);
		}
	};
	std::vector<std::vector<W>> weight(n, std::vector<W>(n));
	std::vector<std::vector<int>> index(n, std::vector<int>(n, -1));

	const auto computeWeight = [&](int i, int j, int k) -> W {
		const HalfMesh::VIndex a = hv[i], b = hv[j], c = hv[k];
		if (eIsInterior(a, b) || eIsInterior(b, c) || eIsInterior(c, a))
			return W(); // would duplicate an existing edge -> infinite weight
		const Eigen::Vector3d nScaled = (pos(b) - pos(a)).cross(pos(c) - pos(a));
		const Eigen::Vector3d nrm = nScaled.normalized();
		HalfMesh::VIndex d = (i + 1 == j) ? vOpp(j) : hv[index[i][j]];
		double angle = std::max(0.0, 1.0 - nrm.dot(normalOf(a, d, b)));
		d = (j + 1 == k) ? vOpp(k) : hv[index[j][k]];
		angle = std::max(angle, 1.0 - nrm.dot(normalOf(b, d, c)));
		if (i == 0 && k + 1 == n) {
			d = vOpp(0);
			angle = std::max(angle, 1.0 - nrm.dot(normalOf(c, d, a)));
		}
		return W(angle, nScaled.squaredNorm()); // SQUARED-area tie-break
	};

	for (int i = 0; i < n - 1; ++i) {
		weight[i][i + 1] = W(0.0, 0.0);
		index[i][i + 1] = -1;
	}
	for (int j = 2; j < n; ++j)
		for (int i = 0; i < n - j; ++i) {
			const int k = i + j;
			W wmin;
			int mmin = -1;
			for (int mm = i + 1; mm < k; ++mm) {
				const W w = weight[i][mm] + computeWeight(i, mm, k) + weight[mm][k];
				if (w < wmin) {
					wmin = w;
					mmin = mm;
				}
			}
			weight[i][k] = wmin;
			index[i][k] = mmin;
		}

	// Reconstruct the chosen triangulation and sum its REAL (0.5*|cross|) areas.
	double area = 0.0;
	std::vector<std::pair<int, int>> stack;
	stack.reserve(n * 2);
	stack.emplace_back(0, n - 1);
	while (!stack.empty()) {
		const std::pair<int, int> r = stack.back();
		stack.pop_back();
		if (r.second - r.first < 2)
			continue;
		const int split = index[r.first][r.second];
		if (split < 0)
			return -1.0; // no valid triangulation
		const Eigen::Vector3d A = pos(hv[r.first]);
		const Eigen::Vector3d B = pos(hv[split]);
		const Eigen::Vector3d C = pos(hv[r.second]);
		area += 0.5 * (B - A).cross(C - A).norm();
		stack.emplace_back(r.first, split);
		stack.emplace_back(split, r.second);
	}
	return area;
}

// Liepa 2003 minimizes SUMMED triangle area after the max-dihedral term. The DP
// stored |cross|^2 (= 4*area^2) as the tie-break, so it minimized the sum of
// SQUARED areas, which can pick a different (larger-surface) triangulation on
// non-planar holes. The test searches a fixed family of irregular saddles for a
// case where SquaredCriterionFillArea independently selects that different
// family. This stays comparative rather than pinning platform-sensitive area
// values or relying on one seed's floating-point tie-breaking.
TEST(MeshHolesTest, LiepaMinimizesSummedAreaNotSquared)
{
	// Some seeds pick the same triangulation under either tie-break because their
	// max-dihedral objective has one clear winner. Search a fixed, small family
	// of irregular saddles for a genuine tie-break case; this keeps the test
	// discriminating without depending on one platform's floating-point ties.
	bool foundDifference = false;
	for (unsigned seed = 1; seed <= 64; ++seed) {
		const Mesh m = MakeIrregularHole(8, 8, seed, /*block=*/2);
		const double fillArea = FillDPArea(m);
		const double squaredArea = SquaredCriterionFillArea(m);
		ASSERT_GT(fillArea, 0.0) << "hole was not filled (seed=" << seed << ")";
		ASSERT_GT(squaredArea, 0.0) << "squared-criterion oracle failed to triangulate (seed=" << seed << ")";
		if (squaredArea - fillArea > 0.01 * fillArea) {
			foundDifference = true;
			break;
		}
	}
	EXPECT_TRUE(foundDifference) << "summed-area and squared-area criteria selected no distinct triangulations";
}

// ---------------------------------------------------------------------------
// 5. Collinear-only hole: must be refused, not filled with zero-area garbage
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, CloseHolesRefusesCollinearOnlyHole)
{
	Mesh m;
	m.vertices = {Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(1.f, 0.f, 0.f),
	              Mesh::Vertex(2.f, 0.f, 0.f), Mesh::Vertex(1.f, 1.f, 1.f)};
	m.faces = {Mesh::Face(1, 0, 3), Mesh::Face(2, 1, 3), Mesh::Face(0, 2, 3)};
	const unsigned closed = m.CloseHoles();
	EXPECT_EQ(closed, 0u) << "a hole with only a zero-area triangulation must stay open";
	EXPECT_EQ(m.faces.size(), 3u) << "no zero-area fill face may be appended";
}

// ---------------------------------------------------------------------------
// 5b. Tiny-scale hole: the degenerate-triangle guard must reject by SHAPE, not
// size. At mesh scale 1e-4 every candidate triangle's squared cross norm
// (~1e-16, length^4 units) sat below the old absolute 1e-12 threshold, so the
// DP refused perfectly healthy holes (safe failure: the hole stayed open).
// The scale-relative guard fills it.
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, CloseHolesFillsTinyScaleHole)
{
	const float s = 1e-4f;
	Mesh m;
	m.vertices = {
	    Mesh::Vertex(0, 0, 0),
	    Mesh::Vertex(s, 0, 0),
	    Mesh::Vertex(0, s, 0),
	    Mesh::Vertex(0, 0, s),
	};
	// tetrahedron with face (1,2,3) missing -> one triangular hole
	m.faces = {
	    Mesh::Face(0, 2, 1),
	    Mesh::Face(0, 1, 3),
	    Mesh::Face(0, 3, 2),
	};
	ASSERT_EQ(CountBoundaryLoops(m), 1u);
	const unsigned closed = m.CloseHoles();
	EXPECT_EQ(closed, 1u) << "healthy tiny-scale hole must fill";
	EXPECT_GT(m.faces.size(), 3u) << "the fill must append at least one face";
	EXPECT_EQ(CountBoundaryLoops(m), 0u) << "mesh must be watertight after fill";
	EXPECT_EQ(CountDegenerateFaces(m), 0u);
}

// ---------------------------------------------------------------------------
// 5c. HalfMesh::TriangulateHole (the raw DP oracle, no refine/fairing) repeated
// the HOLES-3 NaN-swallow: a zero-area candidate's normalized() NaN was
// discarded by std::max(0, NaN), so the degenerate triangle scored as perfect
// and won the DP -- appending a zero-area face for a collinear-only hole. It
// must refuse instead (and the reconstruction must tolerate the all-forbidden
// span rather than indexing holeVertices[NO_ID]).
// ---------------------------------------------------------------------------
TEST(MeshHolesTest, TriangulateHoleOracleRefusesCollinearOnlyHole)
{
	Mesh m;
	m.vertices = {Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(1.f, 0.f, 0.f),
	              Mesh::Vertex(2.f, 0.f, 0.f), Mesh::Vertex(1.f, 1.f, 1.f)};
	m.faces = {Mesh::Face(1, 0, 3), Mesh::Face(2, 1, 3), Mesh::Face(0, 2, 3)};
	HalfMesh hm(m);
	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	ASSERT_EQ(holes.size(), 1u);
	const HalfMesh::HIndex iHeStart = hm.HeTwin(hm.VHalfedge(holes[0][0]));
	const HalfMesh::FIndex facesBefore = hm.FSize();
	hm.TriangulateHole(iHeStart, m.vertices);
	EXPECT_EQ(hm.FSize(), facesBefore)
	    << "a hole with only a zero-area triangulation must stay open";
}

// ---------------------------------------------------------------------------
// Large-area hole with a fine boundary: patch refinement is budgeted
// ---------------------------------------------------------------------------
// Refine matches the patch density to the hole BOUNDARY's mean edge length;
// nothing ties that target to the patch's area. A hemisphere whose 800-vertex
// rim bounds the whole equatorial disk used to refine its ~798 Liepa triangles
// toward ~52,000 (the challenge fixture's 2,739-edge outer boundary went
// 2,737 -> 701,000+ and ground for >15 minutes in the O(T log T) refine
// passes). The budget floors the target edge length so the patch stays within
// max(16384, 8*nb) triangles; disk-like holes, which density-match to ~nb^2/5
// triangles, are untouched up to nb ~ 280.
TEST(MeshHolesTest, LargeHoleRefinementIsBudgeted)
{
	// Open hemisphere: latitude rings from the pole to the equator; the
	// equator rim (800 fine edges) is the single boundary loop. Interior
	// spacing is ~30x the rim edge length, mirroring a scan whose outer
	// boundary is far finer than its interior density.
	constexpr int S = 800; // rim segments
	constexpr int R = 6; // latitude rings pole->equator
	Mesh m;
	m.vertices.emplace_back(0.f, 0.f, 1.f); // pole
	for (int r = 1; r <= R; ++r) {
		const double phi = (M_PI / 2.0) * r / R; // 0 = pole .. pi/2 = equator
		for (int s = 0; s < S; ++s) {
			const double th = 2.0 * M_PI * s / S;
			m.vertices.emplace_back(static_cast<float>(std::sin(phi) * std::cos(th)),
			                        static_cast<float>(std::sin(phi) * std::sin(th)),
			                        static_cast<float>(std::cos(phi)));
		}
	}
	const auto ring = [&](int r, int s) -> Mesh::VIndex { // r >= 1
		return static_cast<Mesh::VIndex>(1 + (r - 1) * S + (s % S));
	};
	for (int s = 0; s < S; ++s) // pole fan
		m.faces.emplace_back(Mesh::Face(0, ring(1, s), ring(1, s + 1)));
	for (int r = 1; r < R; ++r)
		for (int s = 0; s < S; ++s) {
			m.faces.emplace_back(Mesh::Face(ring(r, s), ring(r + 1, s), ring(r + 1, s + 1)));
			m.faces.emplace_back(Mesh::Face(ring(r, s), ring(r + 1, s + 1), ring(r, s + 1)));
		}

	std::vector<std::vector<Mesh::FIndex>> holesFaces;
	const unsigned closed = m.CloseHoles(1, &holesFaces);
	ASSERT_EQ(closed, 1u);
	ASSERT_EQ(holesFaces.size(), 1u);
	EXPECT_LE(holesFaces[0].size(), 24000u)
	    << "patch refinement escaped the max(16384, 8*nb) triangle budget";

	// The filled mesh is a valid watertight manifold.
	m.ListHalfEdges();
	ASSERT_FALSE(m.halfMesh.Empty());
	std::vector<std::vector<HalfMesh::VIndex>> holes;
	m.halfMesh.EnumerateHoles(holes);
	EXPECT_TRUE(holes.empty());
}

} // namespace
} // namespace halfmesh

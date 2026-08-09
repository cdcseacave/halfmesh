/*
* TriangleBVHTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// TriangleBVH parity tests: the flattened BVH must return the same nearest-point
// and ray-intersection results as a brute-force scan (and hence as the existing
// TriangleKdTree), on a synthetic grid and on mesh.ply.

#include <gtest/gtest.h>

#include <halfmesh/TriangleBVH.h>
#include <halfmesh/Util/Geometry.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

using halfmesh::Mesh;
using halfmesh::TriangleBVH;

namespace {

Mesh BuildGridMesh(int N)
{
	Mesh mesh;
	for (int z = 0; z <= N; ++z)
		for (int x = 0; x <= N; ++x) {
			const float y = 0.05f * static_cast<float>(std::rand()) / RAND_MAX;
			mesh.vertices.push_back(Mesh::Vertex(static_cast<float>(x), y, static_cast<float>(z)));
		}
	for (int z = 0; z < N; ++z)
		for (int x = 0; x < N; ++x) {
			const Mesh::VIndex v00 = static_cast<Mesh::VIndex>(z * (N + 1) + x);
			const Mesh::VIndex v10 = v00 + 1;
			const Mesh::VIndex v01 = v00 + static_cast<Mesh::VIndex>(N + 1);
			const Mesh::VIndex v11 = v01 + 1;
			mesh.faces.push_back(Mesh::Face(v00, v10, v01));
			mesh.faces.push_back(Mesh::Face(v10, v11, v01));
		}
	return mesh;
}

struct Hit
{
	float dist = std::numeric_limits<float>::max();
	Mesh::Vertex nearest;
};

Hit BruteForceNearest(const Mesh& mesh, const Mesh::Vertex& q)
{
	Hit best;
	Mesh::Vertex n;
	for (size_t i = 0; i < mesh.faces.size(); ++i) {
		const Mesh::Face& f = mesh.faces[i];
		const float d = math::DistanceBetweenTriangleAndPoint(
		    mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]], q, &n);
		if (d < best.dist) {
			best.dist = d;
			best.nearest = n;
		}
	}
	return best;
}

Hit BruteForceIntersect(const Mesh& mesh, const Eigen::ParametrizedLine<float, 3>& ray)
{
	Hit best;
	for (size_t i = 0; i < mesh.faces.size(); ++i) {
		const Mesh::Face& f = mesh.faces[i];
		float d;
		if (math::RayTriangleIntersect(ray, mesh.vertices[f[0]], mesh.vertices[f[1]],
		                               mesh.vertices[f[2]], 0.f, best.dist, &d)) {
			best.dist = d;
			best.nearest = ray.pointAt(d);
		}
	}
	return best;
}

std::string MeshPlyPath()
{
	return (std::filesystem::path(__FILE__).parent_path() / "data" / "mesh.ply").string();
}

} // namespace

TEST(TriangleBVHTest, NearestPointMatchesBruteForceGrid)
{
	std::srand(42);
	const int N = 16; // 512 triangles
	const Mesh mesh = BuildGridMesh(N);
	const TriangleBVH bvh(mesh);

	for (int q = 0; q < 300; ++q) {
		const float x = (static_cast<float>(std::rand()) / RAND_MAX) * (N + 2) - 1.f;
		const float y = (static_cast<float>(std::rand()) / RAND_MAX) * 4.f - 0.5f;
		const float z = (static_cast<float>(std::rand()) / RAND_MAX) * (N + 2) - 1.f;
		const Mesh::Vertex query(x, y, z);

		const auto r = bvh.NearestPoint(query);
		const Hit b = BruteForceNearest(mesh, query);
		ASSERT_TRUE(r.IsValid()) << "q=" << q;
		EXPECT_NEAR(r.dist, b.dist, 1e-4f) << "q=" << q;
		EXPECT_NEAR((r.nearest - b.nearest).norm(), 0.f, 1e-4f) << "q=" << q;
	}
}

TEST(TriangleBVHTest, IntersectedPointMatchesBruteForceGrid)
{
	std::srand(42);
	const int N = 16;
	const Mesh mesh = BuildGridMesh(N);
	const TriangleBVH bvh(mesh);

	for (int q = 0; q < 300; ++q) {
		const float tx = 0.5f + (static_cast<float>(std::rand()) / RAND_MAX) * (N - 1.f);
		const float tz = 0.5f + (static_cast<float>(std::rand()) / RAND_MAX) * (N - 1.f);
		const Mesh::Vertex target(tx, 0.f, tz);
		const float depth = 3.f + (static_cast<float>(std::rand()) / RAND_MAX) * 4.f;
		const Mesh::Vertex origin(tx, -depth, tz);
		const Eigen::Vector3f dir = (target - origin).normalized();
		const Eigen::ParametrizedLine<float, 3> ray(origin, dir);

		const auto r = bvh.IntersectedPoint(ray);
		const Hit b = BruteForceIntersect(mesh, ray);
		ASSERT_TRUE(r.IsValid()) << "q=" << q;
		EXPECT_NEAR(r.dist, b.dist, 1e-4f) << "q=" << q;
		EXPECT_NEAR((r.nearest - b.nearest).norm(), 0.f, 1e-4f) << "q=" << q;
	}
}

// A mesh of disjoint tiny triangles, one per float octave: centroid x ~ 2^o for
// o = -140..120 (denormals through near-max float) — a scan-outlier streak across
// the representable octave range, the adversarial case for the build's partitioning.
// On such a geometric spread a midpoint-style partition peels only the top
// octave(s) per level (and at the huge octaves the SAH surface areas overflow to
// inf, disabling cost ranking), so an unbalanced fallback drives the tree height
// far past any fixed traversal stack -> out-of-bounds stack write. The build must
// force a TRULY balanced (count/2) median split so the height bound
// FORCE_MEDIAN_DEPTH + ceil(log2 n) actually holds.
Mesh BuildFloatOctaveSpreadMesh(int oMin, int oMax)
{
	Mesh mesh;
	for (int o = oMin; o <= oMax; ++o) {
		const float c = std::ldexp(1.f, o);
		const float s = std::ldexp(1.f, o - 2); // c/4: stays disjoint from octave o+1
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back(Mesh::Vertex(c, 0.f, 0.f));
		mesh.vertices.push_back(Mesh::Vertex(c + s, 0.f, 0.f));
		mesh.vertices.push_back(Mesh::Vertex(c, s, 0.f));
		mesh.faces.push_back(Mesh::Face(base, base + 1, base + 2));
	}
	return mesh; // one disjoint triangle per octave
}

// Adversarial deep-tree build: height must stay within the build's hard bound
// (else the fixed traversal stack overflows) and queries must remain exact.
TEST(TriangleBVHTest, DeepTreeBoundedHeightAndExactQueries)
{
	// Full representable spread (261 octaves, denormals to near-max float): the
	// height bound must hold even where the SAH surface areas overflow to inf.
	// The build's hard ceiling is FORCE_MEDIAN_DEPTH(31) + ceil(log2(n)) <= 63
	// edges, so traversal-stack occupancy (height + 1) can never exceed 64 —
	// well below the 128-slot stack. Fatal assert: never run queries against an
	// over-deep tree (they would write past the fixed stack).
	{
		const Mesh mesh = BuildFloatOctaveSpreadMesh(-140, 120);
		const TriangleBVH bvh(mesh);
		ASSERT_LE(bvh.Height(), 64u) << "tree height escaped the depth bound";
	}

	// Query-parity mesh: still an extreme geometric spread (165 octaves), but
	// capped at octave 24 so the triangle-distance kernel stays in its exact
	// regime. (For edges ~2^32 and beyond, |e1 x e2|^2 — L^4 in edge length —
	// overflows float and DistanceBetweenTriangleAndPointSquared's overflow
	// guard degrades gracefully to a conservative boundary distance: safe, but
	// not the exact value brute force computes, so parity queries against such
	// faces are still meaningless.)
	const Mesh mesh = BuildFloatOctaveSpreadMesh(-140, 24);
	const TriangleBVH bvh(mesh);
	ASSERT_LE(bvh.Height(), 64u) << "tree height escaped the depth bound";

	// Queries across the whole octave range must match brute force exactly.
	for (int o = -138; o <= 22; o += 4) {
		const float c = std::ldexp(1.f, o);
		for (const float f : {0.75f, 1.4f}) {
			const Mesh::Vertex query(f * c, 0.5f * c, 0.25f * c);
			const auto r = bvh.NearestPoint(query);
			const Hit b = BruteForceNearest(mesh, query);
			ASSERT_TRUE(r.IsValid()) << "o=" << o << " f=" << f;
			EXPECT_FLOAT_EQ(r.dist, b.dist) << "o=" << o << " f=" << f;
		}
	}
}

// 300 sliver triangles all sharing two opposite corners of the unit cube, so every
// face AABB is EXACTLY the full cube: no split plane can shrink a child box, the
// SAH split cost equals the leaf cost everywhere, and an uncapped "SAH says leaf"
// termination keeps the whole set as ONE giant leaf -> O(n) scans for every query
// touching it (the same giant-leaf pathology the kd-tree's maxTriangles guard
// exists for). The build must cap leaf sizes by splitting anyway (balanced median)
// while keeping answers exact.
TEST(TriangleBVHTest, OverlappingFanCapsLeafSize)
{
	Mesh mesh;
	std::srand(99);
	auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX; };
	for (int i = 0; i < 300; ++i) {
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back(Mesh::Vertex(0.f, 0.f, 0.f));
		mesh.vertices.push_back(Mesh::Vertex(1.f, 1.f, 1.f));
		mesh.vertices.push_back(Mesh::Vertex(rnd(), rnd(), rnd()));
		mesh.faces.push_back(Mesh::Face(base, base + 1, base + 2));
	}
	const TriangleBVH bvh(mesh);

	// Leaves stay bounded: leafSize(4) on the normal path, the SAH-leaf cap (8)
	// when the SAH cost said "leaf" but the count was too large to accept.
	EXPECT_LE(bvh.MaxLeafSize(), 8u) << "unbounded SAH leaf termination";
	EXPECT_LE(bvh.Height(), 64u);

	std::srand(7);
	for (int q = 0; q < 100; ++q) {
		const Mesh::Vertex query(rnd() * 2.f - 0.5f, rnd() * 2.f - 0.5f, rnd() * 2.f - 0.5f);
		const auto r = bvh.NearestPoint(query);
		const Hit b = BruteForceNearest(mesh, query);
		ASSERT_TRUE(r.IsValid()) << "q=" << q;
		EXPECT_FLOAT_EQ(r.dist, b.dist) << "q=" << q;
	}
}

// Bounded search radius + warm-start hint must not change the answer when the true
// nearest is within the bound, must reject when the bound is too tight, and the hint
// must yield the same distance as the un-hinted query.
TEST(TriangleBVHTest, BoundedRadiusAndWarmStartHint)
{
	std::srand(42);
	const int N = 16;
	const Mesh mesh = BuildGridMesh(N);
	const TriangleBVH bvh(mesh);

	for (int q = 0; q < 200; ++q) {
		auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX; };
		const Mesh::Vertex query(rnd() * N, rnd() * 2.f - 0.5f, rnd() * N);

		const auto unbounded = bvh.NearestPoint(query);
		ASSERT_TRUE(unbounded.IsValid()) << "q=" << q;

		// Generous bound: identical result.
		const auto within = bvh.NearestPoint(query, unbounded.dist + 1.f);
		ASSERT_TRUE(within.IsValid()) << "q=" << q;
		EXPECT_FLOAT_EQ(within.dist, unbounded.dist) << "q=" << q;
		EXPECT_EQ(within.idxFace, unbounded.idxFace) << "q=" << q;

		// Tight bound below the true distance: rejected.
		const auto tootight = bvh.NearestPoint(query, unbounded.dist * 0.5f);
		EXPECT_FALSE(tootight.IsValid()) << "q=" << q;

		// Warm start with the true nearest face: same distance (and same face here,
		// since the grid queries have no exact ties).
		const auto hinted = bvh.NearestPoint(query, std::numeric_limits<float>::max(),
		                                     unbounded.idxFace);
		ASSERT_TRUE(hinted.IsValid()) << "q=" << q;
		EXPECT_FLOAT_EQ(hinted.dist, unbounded.dist) << "q=" << q;

		// A stale/incorrect hint must not corrupt the answer (bound still governs).
		const auto stale = bvh.NearestPoint(query, std::numeric_limits<float>::max(),
		                                    Mesh::FIndex(0));
		ASSERT_TRUE(stale.IsValid()) << "q=" << q;
		EXPECT_FLOAT_EQ(stale.dist, unbounded.dist) << "q=" << q;
	}
}

// maxDist must bound the ray parameter during traversal: identical hit when the
// true hit is within the bound, no hit when the bound is shorter than the hit.
TEST(TriangleBVHTest, BoundedRayMaxDist)
{
	std::srand(42);
	const int N = 16;
	const Mesh mesh = BuildGridMesh(N);
	const TriangleBVH bvh(mesh);

	for (int q = 0; q < 200; ++q) {
		auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX; };
		const float tx = 0.5f + rnd() * (N - 1.f);
		const float tz = 0.5f + rnd() * (N - 1.f);
		const Mesh::Vertex target(tx, 0.f, tz);
		const float depth = 3.f + rnd() * 4.f;
		const Mesh::Vertex origin(tx, -depth, tz);
		const Eigen::Vector3f dir = (target - origin).normalized();
		const Eigen::ParametrizedLine<float, 3> ray(origin, dir);

		const auto unbounded = bvh.IntersectedPoint(ray);
		ASSERT_TRUE(unbounded.IsValid()) << "q=" << q;

		const auto within = bvh.IntersectedPoint(ray, unbounded.dist + 1.f);
		ASSERT_TRUE(within.IsValid()) << "q=" << q;
		EXPECT_FLOAT_EQ(within.dist, unbounded.dist) << "q=" << q;
		EXPECT_EQ(within.idxFace, unbounded.idxFace) << "q=" << q;

		const auto tootight = bvh.IntersectedPoint(ray, unbounded.dist * 0.5f);
		EXPECT_FALSE(tootight.IsValid()) << "q=" << q;
	}
}

TEST(TriangleBVHTest, NearestPointMatchesBruteForceMeshPly)
{
	if (!std::filesystem::exists(MeshPlyPath()))
		GTEST_SKIP() << "mesh.ply not found";
	Mesh mesh;
	ASSERT_TRUE(mesh.Load(MeshPlyPath()));
	ASSERT_FALSE(mesh.faces.empty());

	const TriangleBVH bvh(mesh);
	const auto box = mesh.ComputeAABBox();
	const Mesh::Vertex c = box.center();
	const Mesh::Vertex ext = box.diagonal();

	std::srand(7);
	for (int q = 0; q < 200; ++q) {
		auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX - 0.5f; };
		const Mesh::Vertex query = c + Mesh::Vertex(rnd() * ext.x() * 1.5f, rnd() * ext.y() * 1.5f, rnd() * ext.z() * 1.5f);
		const auto r = bvh.NearestPoint(query);
		const Hit b = BruteForceNearest(mesh, query);
		ASSERT_TRUE(r.IsValid()) << "q=" << q;
		EXPECT_NEAR(r.dist, b.dist, 1e-3f) << "q=" << q;
	}
}

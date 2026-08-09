/*
* TriangleKDTreeTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// TriangleKDTree tests — inline-mesh variant.
// This test builds the mesh inline — a triangulated 8x8 grid plane with mild
// z-perturbation — and validates NearestPoint and IntersectedPoint against a
// brute-force scan using math::DistanceBetweenTriangleAndPoint /
// math::RayTriangleIntersect.
#include <gtest/gtest.h>

#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/Util/Geometry.h>

#include <cmath>
#include <cstdlib>
#include <limits>

namespace {

// Build a triangulated N×N quad grid in XZ, with mild Y perturbation.
// Grid spans [0,N] × [0,N] in X and Z.
halfmesh::Mesh BuildGridMesh(int N)
{
	halfmesh::Mesh mesh;

	// vertices: (N+1)*(N+1) grid
	mesh.vertices.reserve(static_cast<std::size_t>((N + 1) * (N + 1)));
	for (int z = 0; z <= N; ++z) {
		for (int x = 0; x <= N; ++x) {
			// mild z-perturbation so the mesh is not perfectly planar
			const float y = 0.05f * static_cast<float>(std::rand()) / RAND_MAX;
			mesh.vertices.push_back(halfmesh::Mesh::Vertex(
			    static_cast<float>(x),
			    y,
			    static_cast<float>(z)));
		}
	}

	// faces: 2 triangles per quad
	mesh.faces.reserve(static_cast<std::size_t>(2 * N * N));
	for (int z = 0; z < N; ++z) {
		for (int x = 0; x < N; ++x) {
			const halfmesh::Mesh::VIndex v00 = static_cast<halfmesh::Mesh::VIndex>(z * (N + 1) + x);
			const halfmesh::Mesh::VIndex v10 = v00 + 1;
			const halfmesh::Mesh::VIndex v01 = v00 + static_cast<halfmesh::Mesh::VIndex>(N + 1);
			const halfmesh::Mesh::VIndex v11 = v01 + 1;
			// lower-left triangle
			halfmesh::Mesh::Face f0;
			f0[0] = v00;
			f0[1] = v10;
			f0[2] = v01;
			mesh.faces.push_back(f0);
			// upper-right triangle
			halfmesh::Mesh::Face f1;
			f1[0] = v10;
			f1[1] = v11;
			f1[2] = v01;
			mesh.faces.push_back(f1);
		}
	}

	return mesh;
}

// Brute-force nearest point over all faces.
halfmesh::TriangleKdTree::NearestNeighbor BruteForceNearest(
    const halfmesh::Mesh& mesh,
    const halfmesh::Mesh::Vertex& query)
{
	halfmesh::TriangleKdTree::NearestNeighbor best;
	halfmesh::Mesh::Vertex n;
	for (halfmesh::Mesh::FIndex i = 0; i < static_cast<halfmesh::Mesh::FIndex>(mesh.faces.size()); ++i) {
		const halfmesh::Mesh::Face& f = mesh.faces[i];
		const float d = math::DistanceBetweenTriangleAndPoint(
		    mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]],
		    query, &n);
		if (d < best.dist) {
			best.dist = d;
			best.nearest = n;
			best.idxFace = i;
		}
	}
	return best;
}

// A mesh with a dense cluster of thousands of near-coincident tiny triangles
// (plus a sparse background), reproducing the distribution that made the old
// spatial-middle split pile a whole cluster into one giant leaf (a 0.7M-triangle
// leaf was observed, turning nearest-point queries into linear scans). Used to
// guard the object-median split: nearest-point must stay exact on such input.
halfmesh::Mesh BuildClusteredMesh()
{
	halfmesh::Mesh mesh;
	std::srand(7);
	auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX; };
	auto addTri = [&](const halfmesh::Mesh::Vertex& c, float s) {
		const auto base = static_cast<halfmesh::Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back(c);
		mesh.vertices.push_back(c + halfmesh::Mesh::Vertex(s, 0.f, 0.f));
		mesh.vertices.push_back(c + halfmesh::Mesh::Vertex(0.f, s, 0.f));
		halfmesh::Mesh::Face f;
		f[0] = base;
		f[1] = base + 1;
		f[2] = base + 2;
		mesh.faces.push_back(f);
	};
	// sparse background spread over [0,10]^3
	for (int i = 0; i < 200; ++i)
		addTri(halfmesh::Mesh::Vertex(rnd() * 10, rnd() * 10, rnd() * 10), 0.3f);
	// dense cluster of distinct-but-near-coincident triangles within a 1e-3 ball
	const halfmesh::Mesh::Vertex P(5.f, 5.f, 5.f);
	for (int i = 0; i < 3000; ++i)
		addTri(P + halfmesh::Mesh::Vertex((rnd() - 0.5f) * 1e-3f, (rnd() - 0.5f) * 1e-3f, (rnd() - 0.5f) * 1e-3f),
		       1e-5f);
	return mesh;
}

// Brute-force nearest ray-triangle hit over all faces.
halfmesh::TriangleKdTree::NearestNeighbor BruteForceIntersect(
    const halfmesh::Mesh& mesh,
    const Eigen::ParametrizedLine<float, 3>& ray)
{
	halfmesh::TriangleKdTree::NearestNeighbor best;
	for (halfmesh::Mesh::FIndex i = 0; i < static_cast<halfmesh::Mesh::FIndex>(mesh.faces.size()); ++i) {
		const halfmesh::Mesh::Face& f = mesh.faces[i];
		float d;
		if (math::RayTriangleIntersect(
		        ray,
		        mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]],
		        float(0), best.dist, &d)) {
			best.dist = d;
			best.nearest = ray.pointAt(d);
			best.idxFace = i;
		}
	}
	return best;
}

} // namespace

// ---------------------------------------------------------------------------
// NearestPoint: KD-tree distance == brute-force distance (tight tolerance)
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, NearestPoint)
{
	std::srand(42);

	const int N = 8; // 8×8 quads → 128 triangles (forces real branching; maxTriangles=8)
	const halfmesh::Mesh mesh = BuildGridMesh(N);
	ASSERT_EQ(mesh.faces.size(), static_cast<std::size_t>(2 * N * N));

	const halfmesh::TriangleKdTree tree(mesh);

	const int numQueries = 100;
	for (int q = 0; q < numQueries; ++q) {
		// random query point around / above the mesh
		const float x = (static_cast<float>(std::rand()) / RAND_MAX) * (N + 2) - 1.0f;
		const float y = (static_cast<float>(std::rand()) / RAND_MAX) * 4.0f - 0.5f;
		const float z = (static_cast<float>(std::rand()) / RAND_MAX) * (N + 2) - 1.0f;
		const halfmesh::Mesh::Vertex query(x, y, z);

		const auto kdResult = tree.NearestPoint(query);
		const auto bruteResult = BruteForceNearest(mesh, query);

		ASSERT_TRUE(kdResult.IsValid()) << "q=" << q << " KD result invalid";
		ASSERT_TRUE(bruteResult.IsValid()) << "q=" << q << " Brute-force result invalid";

		// distance must match brute-force to within 1e-4 (float precision)
		EXPECT_NEAR(kdResult.dist, bruteResult.dist, 1e-4f)
		    << "q=" << q << " distance mismatch: kd=" << kdResult.dist
		    << " brute=" << bruteResult.dist;

		// nearest point must also match
		EXPECT_NEAR((kdResult.nearest - bruteResult.nearest).norm(), 0.0f, 1e-4f)
		    << "q=" << q << " nearest point mismatch";
	}
}

// ---------------------------------------------------------------------------
// Dense near-coincident cluster: KD-tree nearest still matches brute force.
// Regression for the object-median split (a spatial-middle split could not
// separate such a cluster, producing a giant leaf and ~quadratic queries).
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, ClusteredMeshNearestMatchesBruteForce)
{
	const halfmesh::Mesh mesh = BuildClusteredMesh();
	const halfmesh::TriangleKdTree tree(mesh);

	std::srand(123);
	for (int q = 0; q < 200; ++q) {
		// half the queries probe near the dense cluster, half the whole volume
		const bool nearCluster = (q % 2) == 0;
		const float scale = nearCluster ? 0.01f : 10.f;
		const halfmesh::Mesh::Vertex base = nearCluster ? halfmesh::Mesh::Vertex(5.f, 5.f, 5.f)
		                                                : halfmesh::Mesh::Vertex(0.f, 0.f, 0.f);
		const halfmesh::Mesh::Vertex query(
		    base.x() + (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * scale,
		    base.y() + (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * scale,
		    base.z() + (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * scale);

		const auto kdResult = tree.NearestPoint(query);
		const auto bruteResult = BruteForceNearest(mesh, query);
		ASSERT_TRUE(kdResult.IsValid()) << "q=" << q;
		EXPECT_NEAR(kdResult.dist, bruteResult.dist, 1e-4f) << "q=" << q;
	}
}

// ---------------------------------------------------------------------------
// IntersectedPoint: KD-tree ray hit == brute-force ray hit
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, IntersectedPoint)
{
	std::srand(42);

	const int N = 8;
	const halfmesh::Mesh mesh = BuildGridMesh(N);

	const halfmesh::TriangleKdTree tree(mesh);

	// Verify triangle winding by checking a known face normal direction.
	// BuildGridMesh: face(0) = v00,v10,v01 where v00=(0,y,0), v10=(1,y',0), v01=(0,y'',1).
	// normal ~ (v10-v00).cross(v01-v00) ≈ (1,0,0)×(0,0,1) = (0,-1,0) → points DOWN.
	// So rays must travel UPWARD (positive Y) to hit the front face.

	const int numRays = 100;
	for (int q = 0; q < numRays; ++q) {
		// pick a random point on the grid surface interior (x,z in [0.5,N-0.5])
		const float tx = 0.5f + (static_cast<float>(std::rand()) / RAND_MAX) * (N - 1.0f);
		const float tz = 0.5f + (static_cast<float>(std::rand()) / RAND_MAX) * (N - 1.0f);
		const halfmesh::Mesh::Vertex target(tx, 0.0f, tz);

		// origin is directly BELOW the target (negative Y): normals point down,
		// so rays going upward (+Y) hit the front face.
		const float depth = 3.0f + (static_cast<float>(std::rand()) / RAND_MAX) * 4.0f;
		const halfmesh::Mesh::Vertex origin(tx, -depth, tz);

		// direction straight up (toward the mesh front face)
		const Eigen::Vector3f dir = (target - origin).normalized();
		const Eigen::ParametrizedLine<float, 3> ray(origin, dir);

		const auto kdResult = tree.IntersectedPoint(ray);
		const auto bruteResult = BruteForceIntersect(mesh, ray);

		// both should find a hit (ray goes straight down onto the grid)
		ASSERT_TRUE(kdResult.IsValid())
		    << "q=" << q << " KD-tree found no intersection (expected one)";
		ASSERT_TRUE(bruteResult.IsValid())
		    << "q=" << q << " Brute-force found no intersection (expected one)";

		// hit parameter (ray t) must match
		EXPECT_NEAR(kdResult.dist, bruteResult.dist, 1e-4f)
		    << "q=" << q << " ray hit distance mismatch: kd=" << kdResult.dist
		    << " brute=" << bruteResult.dist;

		// hit point must match
		EXPECT_NEAR((kdResult.nearest - bruteResult.nearest).norm(), 0.0f, 1e-4f)
		    << "q=" << q << " ray hit point mismatch";
	}
}

// ---------------------------------------------------------------------------
// Planar mesh (all vertices z==0): KD node boxes are zero-thickness on z, so the
// slab test's entry==exit. A vertical ray must still hit — a strict `<` slab
// test would reject every such box and report no hit.
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, PlanarMeshRayHits)
{
	halfmesh::Mesh mesh;
	const int N = 4; // 2*N*N = 32 faces > maxTriangles(8) => internal nodes exist
	for (int y = 0; y <= N; ++y)
		for (int x = 0; x <= N; ++x)
			mesh.vertices.push_back(halfmesh::Mesh::Vertex(static_cast<float>(x), static_cast<float>(y), 0.f));
	for (int y = 0; y < N; ++y)
		for (int x = 0; x < N; ++x) {
			const halfmesh::Mesh::VIndex v00 = static_cast<halfmesh::Mesh::VIndex>(y * (N + 1) + x);
			const halfmesh::Mesh::VIndex v10 = v00 + 1;
			const halfmesh::Mesh::VIndex v01 = v00 + static_cast<halfmesh::Mesh::VIndex>(N + 1);
			const halfmesh::Mesh::VIndex v11 = v01 + 1;
			halfmesh::Mesh::Face f0;
			f0[0] = v00;
			f0[1] = v10;
			f0[2] = v01;
			mesh.faces.push_back(f0);
			halfmesh::Mesh::Face f1;
			f1[0] = v10;
			f1[1] = v11;
			f1[2] = v01;
			mesh.faces.push_back(f1);
		}

	const halfmesh::TriangleKdTree tree(mesh);

	// Face normals point +z (winding v00,v10,v01 => (1,0,0)x(0,1,0)=+z), so a ray
	// travelling in -z hits the front face.
	const Eigen::ParametrizedLine<float, 3> ray(
	    halfmesh::Mesh::Vertex(1.5f, 1.5f, 1.f), halfmesh::Mesh::Vertex(0.f, 0.f, -1.f));
	const auto hit = tree.IntersectedPoint(ray);
	ASSERT_TRUE(hit.IsValid()) << "planar-mesh vertical ray must hit";
	EXPECT_NEAR(hit.dist, 1.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// GetAABBox must report the tight mesh bounds even when the mesh is small enough
// that the root stays a leaf (faces <= maxTriangles). If BuildRecurse
// early-returned a default (empty) box for such meshes, GetAABBox() would be
// empty and diverge from Mesh::ComputeAABBox() and the BVH.
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, SmallMeshBBoxIsTight)
{
	// Two-triangle unit quad: 2 faces <= maxTriangles(8) => root stays a leaf.
	halfmesh::Mesh mesh;
	mesh.vertices.push_back(halfmesh::Mesh::Vertex(0.f, 0.f, 0.f));
	mesh.vertices.push_back(halfmesh::Mesh::Vertex(1.f, 0.f, 0.f));
	mesh.vertices.push_back(halfmesh::Mesh::Vertex(1.f, 1.f, 0.f));
	mesh.vertices.push_back(halfmesh::Mesh::Vertex(0.f, 1.f, 0.f));
	halfmesh::Mesh::Face f0, f1;
	f0[0] = 0;
	f0[1] = 1;
	f0[2] = 2;
	f1[0] = 0;
	f1[1] = 2;
	f1[2] = 3;
	mesh.faces.push_back(f0);
	mesh.faces.push_back(f1);

	const halfmesh::TriangleKdTree tree(mesh);
	const auto box = tree.GetAABBox();
	ASSERT_FALSE(box.isEmpty()) << "kd-tree bbox must be tight for a small (leaf-root) mesh";

	const auto ref = mesh.ComputeAABBox();
	EXPECT_NEAR((box.min() - ref.min()).norm(), 0.f, 1e-6f);
	EXPECT_NEAR((box.max() - ref.max()).norm(), 0.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Bounded queries: a finite maxDist must reproduce the unbounded answer when the
// true hit is within the bound, and report IsValid()==false when it is not.
// ---------------------------------------------------------------------------
TEST(TriangleKdTreeTest, BoundedQueriesRespectMaxDist)
{
	std::srand(42);
	const int N = 8;
	const halfmesh::Mesh mesh = BuildGridMesh(N);
	const halfmesh::TriangleKdTree tree(mesh);

	// NearestPoint: query well above the grid so the true distance is ~2.
	const halfmesh::Mesh::Vertex q(4.f, 2.f, 4.f);
	const auto unbounded = tree.NearestPoint(q);
	ASSERT_TRUE(unbounded.IsValid());
	const float trueDist = unbounded.dist;

	// A generous bound reproduces the unbounded result exactly.
	const auto within = tree.NearestPoint(q, trueDist + 1.f);
	ASSERT_TRUE(within.IsValid());
	EXPECT_FLOAT_EQ(within.dist, unbounded.dist);
	EXPECT_EQ(within.idxFace, unbounded.idxFace);

	// A bound below the true distance rejects the hit.
	const auto tootight = tree.NearestPoint(q, trueDist * 0.5f);
	EXPECT_FALSE(tootight.IsValid());

	// IntersectedPoint: ray straight up onto the grid (see IntersectedPoint test).
	const halfmesh::Mesh::Vertex origin(4.f, -3.f, 4.f);
	const Eigen::Vector3f dir = (halfmesh::Mesh::Vertex(4.f, 0.f, 4.f) - origin).normalized();
	const Eigen::ParametrizedLine<float, 3> ray(origin, dir);
	const auto rayUnbounded = tree.IntersectedPoint(ray);
	ASSERT_TRUE(rayUnbounded.IsValid());

	const auto rayWithin = tree.IntersectedPoint(ray, rayUnbounded.dist + 1.f);
	ASSERT_TRUE(rayWithin.IsValid());
	EXPECT_FLOAT_EQ(rayWithin.dist, rayUnbounded.dist);

	const auto rayTootight = tree.IntersectedPoint(ray, rayUnbounded.dist * 0.5f);
	EXPECT_FALSE(rayTootight.IsValid());
}

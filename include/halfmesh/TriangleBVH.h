/*
* TriangleBVH.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include <halfmesh/Mesh.h>

namespace halfmesh {

// Triangle bounding-volume hierarchy.
//
// A flattened (array-of-nodes, depth-first, children stored as adjacent pairs)
// BVH over the mesh triangles — a cache-friendly, SIMD-friendly alternative to
// TriangleKdTree with the same query surface. The two are independent and both
// correct: the BVH prunes with tight per-node AABBs (fewer triangle tests) and
// traverses a contiguous array (no pointer chasing), which makes it markedly
// faster for the millions of queries a texture bake issues; the kd-tree is kept
// as a simpler, equally-valid reference.
class TriangleBVH
{
	public:
	typedef Mesh::Type Scalar;
	typedef Mesh::Vertex Point;
	typedef Eigen::AlignedBox<Scalar, 3> AABBox;

	// Result of a nearest-point / ray query (mirrors TriangleKdTree).
	// For NearestPoint, `dist` is the Euclidean distance; for IntersectedPoint,
	// `dist` is the ray parameter t and `nearest` the hit point.
	struct NearestNeighbor
	{
		Scalar dist;
		Point nearest;
		Mesh::FIndex idxFace;

		NearestNeighbor() :
		    dist(std::numeric_limits<Scalar>::max()), idxFace(0) {}
		bool IsValid() const { return dist < std::numeric_limits<Scalar>::max(); }
	};

	explicit TriangleBVH(const Mesh& mesh, unsigned leafSize = 4) :
	    mesh(mesh)
	{
		Build(mesh, leafSize);
	}

	// Build the BVH from the given mesh (leafSize = max triangles per leaf).
	void Build(const Mesh& mesh, unsigned leafSize = 4);

	// Nearest point on the surface to `p`.
	//   maxDist  — optional search-radius bound: only surface points strictly
	//               closer than maxDist are considered (default: unbounded).
	//   hintFace — optional warm start: a previously-known nearest face is tested
	//               first to tighten the search bound before traversal, cutting the
	//               node/triangle tests when successive queries move coherently
	//               (default: no hint). Answers are unchanged for the default call.
	NearestNeighbor NearestPoint(
	    const Point& p,
	    Scalar maxDist = std::numeric_limits<Scalar>::max(),
	    Mesh::FIndex hintFace = std::numeric_limits<Mesh::FIndex>::max()) const;

	// Nearest front-facing ray-surface intersection (t >= 0). `maxDist` bounds the
	// ray parameter DURING traversal (subtrees entered beyond it are pruned) rather
	// than post-filtering the hit; default is unbounded (identical to prior behavior).
	NearestNeighbor IntersectedPoint(
	    const Eigen::ParametrizedLine<Scalar, 3>& ray,
	    Scalar maxDist = std::numeric_limits<Scalar>::max()) const;

	const AABBox& GetAABBox() const { return bbox; }
	const Mesh& GetMesh() const { return mesh; }

	// Tree height in edges (deepest root-to-leaf path); 0 for a single-leaf/empty
	// tree. The build bounds this below the traversal stack capacity, so it is safe
	// to use as a diagnostic / invariant check.
	unsigned Height() const;

	// Largest leaf primitive count (0 for an empty tree). The build caps this at
	// max(leafSize, SAH-leaf cap): oversized "SAH says leaf" sets are split anyway,
	// so no query ever degrades to a giant linear scan.
	unsigned MaxLeafSize() const;

	private:
	// Flattened node: internal nodes (count == 0) store the index of their left
	// child; the right child is the next slot (leftFirst + 1). Leaves store the
	// first primitive index and the primitive count.
	struct Node
	{
		Scalar bmin[3];
		Scalar bmax[3];
		uint32_t leftFirst; // internal: left child index; leaf: first prim index
		uint32_t count; // 0 = internal, > 0 = leaf primitive count
	};

	void Subdivide(uint32_t nodeIdx, uint32_t first, uint32_t count, unsigned leafSize,
	               unsigned depth);
	unsigned HeightRecurse(uint32_t nodeIdx) const;

	const Mesh& mesh;
	std::vector<Node> nodes;
	std::vector<uint32_t> prims; // reordered face indices (leaves index ranges)
	std::vector<Point> centroids; // per-face centroid (build only, kept small)
	std::vector<AABBox> faceBoxes; // per-face AABB (build scratch; cleared after Build)
	AABBox bbox;
};

} // namespace halfmesh

/*
* TriangleKDTree.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <memory>

#include <halfmesh/Util/Assert.h>
#include <halfmesh/Mesh.h>

namespace halfmesh {

// triangles K-d tree
// inspired by pmp-library implementation
class TriangleKdTree
{
	public:
	typedef Mesh::Type Scalar;
	typedef Mesh::Vertex Point;
	typedef Eigen::AlignedBox<Scalar, 3> AABBox;

	// Result of a nearest-point / ray query (mirrors TriangleBVH).
	// For NearestPoint, `dist` is the Euclidean distance; for IntersectedPoint,
	// `dist` is the ray parameter t and `nearest` the hit point.
	struct NearestNeighbor
	{
		Scalar dist;
		Point nearest;
		Mesh::FIndex idxFace;

		NearestNeighbor()
		{
			dist = std::numeric_limits<Scalar>::max();
			idxFace = 0; // mirror TriangleBVH::NearestNeighbor (0 sentinel, not left uninitialized)
		}
		bool IsValid() const
		{
			return dist < std::numeric_limits<Scalar>::max();
		}
	};

	explicit TriangleKdTree(const Mesh& mesh, unsigned maxTriangles = 8, unsigned maxDepth = 32) :
	    mesh(mesh)
	{
		Build(mesh, maxTriangles, maxDepth);
	}
	~TriangleKdTree() {}

	// create the K-d tree from the given mesh
	void Build(const Mesh&, unsigned maxTriangles = 8, unsigned maxDepth = 32);

	// find the nearest point on the surface for the given point.
	// `maxDist` bounds the search: only surface points strictly closer than
	// maxDist are considered (default: unbounded).
	NearestNeighbor NearestPoint(const Point& p,
	                             Scalar maxDist = std::numeric_limits<Scalar>::max()) const;

	// find the nearest FRONT-FACING point on the surface intersected by the
	// given ray: back-facing hits are culled by the Moller-Trumbore determinant
	// sign (math::RayTriangleIntersect, Util/Geometry.h) — same convention as
	// TriangleBVH. On CW-wound meshes (negative signed volume) rays from
	// outside therefore pass through front surfaces they approach from behind.
	// `maxDist` bounds the ray parameter DURING traversal (subtrees whose entry t
	// is >= maxDist are pruned), rather than post-filtering the hit; default is
	// unbounded.
	NearestNeighbor IntersectedPoint(const Eigen::ParametrizedLine<Scalar, 3>& ray,
	                                 Scalar maxDist = std::numeric_limits<Scalar>::max()) const;

	// get the axis-aligned bounding box of the mesh
	const AABBox& GetAABBox() const { return bbox; }

	// get elements of the input mesh
	const Mesh& GetMesh() const { return mesh; }
	const Mesh::Face& GetFace(Mesh::FIndex idxFace) const { return mesh.faces[idxFace]; }
	const Mesh::Vertex& GetVertex(Mesh::VIndex idxVert) const { return mesh.vertices[idxVert]; }

	private:
	// tree node containing parent, children and splitting plane
	struct NodeLeaf
	{
		uint8_t axis;
		union UNodeLeaf {
			struct UNode
			{
				Scalar split;
				std::unique_ptr<NodeLeaf> leftChild;
				std::unique_ptr<NodeLeaf> rightChild;
			} node;
			struct ULeaf
			{
				std::vector<Mesh::FIndex> triangles;
			} leaf;

			UNodeLeaf() {}
			~UNodeLeaf() {}
		} nodeleaf;

		NodeLeaf() :
		    axis(255)
		{
			new (&nodeleaf.leaf.triangles) std::vector<Mesh::FIndex>();
		}
		explicit NodeLeaf(std::vector<Mesh::FIndex>&& _triangles) :
		    axis(255)
		{
			new (&nodeleaf.leaf.triangles) std::vector<Mesh::FIndex>(std::move(_triangles));
		}
		explicit NodeLeaf(uint8_t _axis, Scalar _split, std::unique_ptr<NodeLeaf>&& _left_child, std::unique_ptr<NodeLeaf>&& _right_child) :
		    axis(_axis)
		{
			ASSERT(!IsLeaf());
			nodeleaf.node.split = _split;
			new (&nodeleaf.node.leftChild) std::unique_ptr<NodeLeaf>(std::move(_left_child));
			new (&nodeleaf.node.rightChild) std::unique_ptr<NodeLeaf>(std::move(_right_child));
		}
		~NodeLeaf()
		{
			if (IsLeaf()) {
				nodeleaf.leaf.triangles.~vector();
			} else {
				nodeleaf.node.leftChild.~unique_ptr();
				nodeleaf.node.rightChild.~unique_ptr();
			}
		}
		bool IsLeaf() const { return axis >= 3; }
		const UNodeLeaf::UNode& Node() const
		{
			ASSERT(!IsLeaf());
			return nodeleaf.node;
		}
		UNodeLeaf::UNode& Node()
		{
			ASSERT(!IsLeaf());
			return nodeleaf.node;
		}
		const UNodeLeaf::ULeaf& Leaf() const
		{
			ASSERT(IsLeaf());
			return nodeleaf.leaf;
		}
		UNodeLeaf::ULeaf& Leaf()
		{
			ASSERT(IsLeaf());
			return nodeleaf.leaf;
		}
	};

	// recursive tree build. `scratch` is a single centroid buffer threaded through
	// the recursion and reused at every node (resized, never reallocated) to avoid
	// a per-node heap allocation.
	void BuildRecurse(NodeLeaf&, unsigned maxHandles, unsigned depth, std::vector<Scalar>& scratch);

	// recursive tree traverse
	void NearestRecurse(const NodeLeaf&, const Point&, NearestNeighbor&) const;

	// recursive tree traverse
	void IntersectedPointRecurse(const NodeLeaf&, const AABBox&, const Eigen::ParametrizedLine<Scalar, 3>&, const Point& invDir, NearestNeighbor&) const;

	const Mesh& mesh;
	NodeLeaf root;
	AABBox bbox;
};

} // namespace halfmesh

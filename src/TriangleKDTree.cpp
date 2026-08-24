/*
* TriangleKDTree.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/TriangleKDTree.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include <halfmesh/Util/Geometry.h>

namespace halfmesh {

// A KD-tree over the mesh triangles for fast nearest-point and ray-intersection
// queries. Each internal node splits space by an axis-aligned plane at the middle of
// the longest side of the node's bounding box. A triangle that straddles the plane is
// stored in BOTH children (a "loose"/overlapping partition), so a single query never
// has to reconstruct triangles cut by a split. Recursion stops when a node holds few
// enough triangles, the depth budget runs out, or a split fails to separate anything.

void TriangleKdTree::Build(const Mesh& mesh, unsigned maxTriangles, unsigned maxDepth)
{
	mesh.SyncFacesConst();
	// insert triangles
	root.Leaf().triangles.resize(mesh.faces.size());
	std::iota(root.Leaf().triangles.begin(), root.Leaf().triangles.end(), Mesh::FIndex(0));

	// Compute the tight root bounding box over all face vertices here, so GetAABBox()
	// is valid even when the mesh is small enough that the root stays a leaf (the
	// recursion below returns an empty box in that case). For a subdividing root this
	// equals the box the recursion would compute, so ray traversal is unaffected.
	bbox.setEmpty();
	for (const Mesh::Face& face : mesh.faces)
		for (unsigned i = 0; i < 3; ++i)
			bbox.extend(mesh.vertices[face[i]]);

	// call recursive tree builder (one reusable centroid scratch buffer)
	std::vector<Scalar> scratch;
	BuildRecurse(root, maxTriangles, maxDepth, scratch);
}

void TriangleKdTree::BuildRecurse(NodeLeaf& node, unsigned maxTriangles, unsigned depth, std::vector<Scalar>& scratch)
{
	if (depth == 0 || node.Leaf().triangles.size() <= maxTriangles)
		return;

	// recompute bounding box as most likely some triangles are partially outside (cut by the previous split);
	// this step slows down the construction ~4x, but is necessary for fast queries (otherwise ~3000x slower)
	AABBox bbox;
	for (Mesh::FIndex idxFace : node.Leaf().triangles) {
		const Mesh::Face& face = mesh.faces[idxFace];
		for (unsigned i = 0; i < 3; ++i) {
			bbox.extend(mesh.vertices[face[i]]);
		}
	}

	// split longest side of bounding box
	const Point bb = bbox.sizes();
	uint8_t axis = 0;
	Scalar length = bb[0];
	if (bb[1] > length)
		length = bb[axis = 1];
	if (bb[2] > length)
		length = bb[axis = 2];

	// Split at the OBJECT MEDIAN (median of triangle-centroid coords along axis),
	// not the spatial middle of the bbox. A middle split makes no progress on a
	// dense / near-coincident cluster of triangles — they all fall on one side —
	// so such a cluster piles into a single giant leaf at maxDepth, and then every
	// query touching it degrades to an O(leaf) linear scan (observed: a 0.7M-triangle
	// leaf -> a ~14 h symmetric-Hausdorff on a 28M-face mesh). The median guarantees
	// ~half the centroids land on each side, keeping the tree balanced and leaves
	// bounded. Nearest-point results are unaffected — only tree balance changes.
	// Reuse the caller-supplied scratch buffer (resize, don't reallocate): centroids
	// are fully consumed by nth_element before we recurse, so one buffer threaded
	// down the recursion replaces the per-node heap allocation.
	scratch.resize(node.Leaf().triangles.size());
	std::size_t ci = 0;
	for (Mesh::FIndex idxFace : node.Leaf().triangles) {
		const Mesh::Face& face = mesh.faces[idxFace];
		scratch[ci++] = (mesh.vertices[face[0]][axis] + mesh.vertices[face[1]][axis] + mesh.vertices[face[2]][axis]) / Scalar(3);
	}
	const std::ptrdiff_t mid = static_cast<std::ptrdiff_t>(scratch.size() / 2);
	std::nth_element(scratch.begin(), scratch.begin() + mid, scratch.end());
	const Scalar split = scratch[static_cast<std::size_t>(mid)];

	// create children
	std::unique_ptr<NodeLeaf> left(new NodeLeaf());
	left->Leaf().triangles.reserve(node.Leaf().triangles.size() / 2);
	std::unique_ptr<NodeLeaf> right(new NodeLeaf());
	right->Leaf().triangles.reserve(node.Leaf().triangles.size() / 2);

	// partition left and right children
	for (Mesh::FIndex idxFace : node.Leaf().triangles) {
		const Mesh::Face& face = mesh.faces[idxFace];
		bool l = false, r = false;
		if (mesh.vertices[face[0]][axis] <= split)
			l = true;
		else
			r = true;
		if (mesh.vertices[face[1]][axis] <= split)
			l = true;
		else
			r = true;
		if (mesh.vertices[face[2]][axis] <= split)
			l = true;
		else
			r = true;
		if (l)
			left->Leaf().triangles.emplace_back(idxFace);
		if (r)
			right->Leaf().triangles.emplace_back(idxFace);
	}

	// Decide whether to keep this node as a leaf. Stop subdividing when either:
	//  - a child holds every triangle (they all straddle the plane => no progress), or
	//  - the split duplicates too many straddling triangles. With the loose partition
	//    a triangle spanning the plane lands in BOTH children, so left+right can reach
	//    2n; sliver/large-triangle meshes hit this at nearly every level, exploding the
	//    tree (~90x triangle references were observed) — which both balloons memory and
	//    slows every query by bloating the tree. Capping the per-split duplication at
	//    1.5n keeps it bounded; tiny well-separated clusters straddle little, so they
	//    keep subdividing (preserving the giant-leaf fix from the median split).
	// Otherwise turn this leaf into an internal node in place: NodeLeaf is a tagged
	// union of leaf (triangle list) and internal (axis/split/children) data, so we
	// destroy the leaf payload and placement-new the internal payload over the same
	// storage, then recurse into both children.
	const std::size_t n = node.Leaf().triangles.size();
	const std::size_t produced = left->Leaf().triangles.size() + right->Leaf().triangles.size();
	if (left->Leaf().triangles.size() == n || right->Leaf().triangles.size() == n || produced > n + n / 2) {
		// free memory
		node.Leaf().triangles.shrink_to_fit();
	} else {
		// free memory
		node.~NodeLeaf();
		// store internal data
		new (&node) NodeLeaf(axis, split, std::move(left), std::move(right));
		// recurse children (scratch is free to be reused — split already computed)
		BuildRecurse(*node.Node().leftChild, maxTriangles, depth - 1, scratch);
		BuildRecurse(*node.Node().rightChild, maxTriangles, depth - 1, scratch);
	}
}

TriangleKdTree::NearestNeighbor TriangleKdTree::NearestPoint(const Point& p, Scalar maxDist) const
{
	NearestNeighbor data;
	// data.dist tracks the best SQUARED distance during recursion (avoids a sqrt
	// per candidate triangle); convert back to Euclidean once at the end. A finite
	// maxDist seeds the squared bound so nothing farther is considered; the default
	// max() reproduces the unbounded search exactly. Validity is decided by whether
	// the search strictly improved on the seeded bound (so a bounded query that finds
	// nothing reports IsValid()==false, like the empty-tree case).
	const Scalar bound2 = (maxDist < std::numeric_limits<Scalar>::max())
	                          ? maxDist * maxDist
	                          : std::numeric_limits<Scalar>::max();
	data.dist = bound2;
	NearestRecurse(root, p, data);
	if (data.dist < bound2)
		data.dist = std::sqrt(data.dist);
	else
		data.dist = std::numeric_limits<Scalar>::max();
	return data;
}

void TriangleKdTree::NearestRecurse(const NodeLeaf& node, const Point& point, NearestNeighbor& data) const
{
	if (node.IsLeaf()) {
		// terminal node
		Point n;
		for (Mesh::FIndex idxFace : node.Leaf().triangles) {
			const Mesh::Face& face = mesh.faces[idxFace];
			const Scalar d2 = math::DistanceBetweenTriangleAndPointSquared(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]], point, &n);
			if (data.dist > d2) {
				data.dist = d2;
				data.nearest = n;
				data.idxFace = idxFace;
			}
		}
	} else {
		// Branch-and-bound: descend first into the child whose half-space contains
		// the query point (signed distance to the splitting plane decides the side).
		// Only cross over to the far child if the plane is closer than the best hit
		// found so far — otherwise nothing on the far side can beat the current best.
		// data.dist is squared, so the plane distance is squared to match (SQUARE).
		const Scalar dist = point[node.axis] - node.Node().split;
		if (dist <= 0) {
			NearestRecurse(*node.Node().leftChild, point, data);
			if (SQUARE(dist) < data.dist) {
				NearestRecurse(*node.Node().rightChild, point, data);
			}
		} else {
			NearestRecurse(*node.Node().rightChild, point, data);
			if (SQUARE(dist) < data.dist) {
				NearestRecurse(*node.Node().leftChild, point, data);
			}
		}
	}
}

TriangleKdTree::NearestNeighbor TriangleKdTree::IntersectedPoint(const Eigen::ParametrizedLine<Scalar, 3>& ray, Scalar maxDist) const
{
	NearestNeighbor data;
	// Compute the reciprocal ray direction once and thread it through the
	// recursion so every slab test multiplies instead of dividing (6 divides
	// per box call -> 3 divides per query). data.dist doubles as the current best
	// hit t and the traversal t-bound; seed it with maxDist so subtrees entered
	// beyond that distance are pruned DURING traversal (default max() = unbounded).
	const Point invDir = ray.direction().cwiseInverse();
	data.dist = maxDist;
	IntersectedPointRecurse(root, bbox, ray, invDir, data);
	if (!(data.dist < maxDist))
		data.dist = std::numeric_limits<Scalar>::max(); // no hit within the bound
	return data;
}

void TriangleKdTree::IntersectedPointRecurse(const NodeLeaf& node, const AABBox& box, const Eigen::ParametrizedLine<Scalar, 3>& ray, const Point& invDir, NearestNeighbor& data) const
{
	if (node.IsLeaf()) {
		// terminal node
		Scalar d;
		for (Mesh::FIndex idxFace : node.Leaf().triangles) {
			const Mesh::Face& face = mesh.faces[idxFace];
			if (math::RayTriangleIntersect(ray, mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]], Scalar(0), data.dist, &d)) {
				data.dist = d;
				data.nearest = ray.pointAt(d);
				data.idxFace = idxFace;
			}
		}
	} else {
		// Split the node's box by the plane into the two child sub-boxes and compute
		// each child's ray entry t. Visit the NEARER child first (front-to-back), so a
		// hit found there can tighten data.dist and prune the farther subtree before it
		// is descended — matching the BVH's ordered traversal. A child is entered only
		// if its entry t is nearer than the closest hit so far.
		AABBox boxLeft = box;
		boxLeft.max()[node.axis] = node.Node().split;
		AABBox boxRight = box;
		boxRight.min()[node.axis] = node.Node().split;
		Scalar tLeft, tRight;
		const bool hitLeft = math::RayBoxIntersect(ray, boxLeft, invDir, &tLeft);
		const bool hitRight = math::RayBoxIntersect(ray, boxRight, invDir, &tRight);

		// order the two children by entry t (nearer first)
		const NodeLeaf* nearChild;
		const NodeLeaf* farChild;
		const AABBox* nearBox;
		const AABBox* farBox;
		Scalar tNear, tFar;
		bool hitNear, hitFar;
		if (hitLeft && (!hitRight || tLeft <= tRight)) {
			nearChild = node.Node().leftChild.get();
			nearBox = &boxLeft;
			tNear = tLeft;
			hitNear = hitLeft;
			farChild = node.Node().rightChild.get();
			farBox = &boxRight;
			tFar = tRight;
			hitFar = hitRight;
		} else {
			nearChild = node.Node().rightChild.get();
			nearBox = &boxRight;
			tNear = tRight;
			hitNear = hitRight;
			farChild = node.Node().leftChild.get();
			farBox = &boxLeft;
			tFar = tLeft;
			hitFar = hitLeft;
		}
		if (hitNear && tNear < data.dist)
			IntersectedPointRecurse(*nearChild, *nearBox, ray, invDir, data);
		if (hitFar && tFar < data.dist)
			IntersectedPointRecurse(*farChild, *farBox, ray, invDir, data);
	}
}

} // namespace halfmesh

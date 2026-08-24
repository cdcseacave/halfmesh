/*
* TriangleBVH.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/TriangleBVH.h>

#include <algorithm>
#include <cmath>

#include <halfmesh/Util/Geometry.h>

namespace halfmesh {

namespace {

// Traversal stack capacity. The build bounds the tree height at
// FORCE_MEDIAN_DEPTH + ceil(log2 n) <= 63 edges for any uint32 face count (the
// forced split is a true count/2 halving, see medianSplit), so worst-case stack
// occupancy (height + 1) can never exceed 64; 128 leaves a wide safety margin and
// eliminates the overflow class entirely (no ASSERT, which NDEBUG compiles out).
constexpr int BVH_STACK_SIZE = 128;

// Depth at/after which the build forces a TRULY balanced (positional count/2 via
// nth_element) median split instead of SAH, so adversarial centroid distributions
// (which could otherwise peel one primitive per level) cannot drive the tree past
// a bounded height. From this depth the primitive count halves every level, so
// leaves land no deeper than FORCE_MEDIAN_DEPTH + ceil(log2(n)) <= 31 + 32 = 63
// edges for any uint32 face count.
constexpr unsigned FORCE_MEDIAN_DEPTH = 31;

// Number of SAH bins along the split axis (Wald 2007). 16 is the usual sweet spot.
constexpr int SAH_BINS = 16;

// Max primitives the SAH cost termination may leave in a single leaf. When the
// best split cost is not below the leaf cost (heavily overlapping primitives:
// every child box ~ the parent box) a leaf is locally optimal — but an UNBOUNDED
// leaf turns every query touching it into an O(leaf) linear scan, the giant-leaf
// pathology the kd-tree's maxTriangles guard exists for (mirrored here). Past
// this cap the build splits anyway with the balanced median.
constexpr uint32_t SAH_MAX_LEAF = 8;

// Half the AABB surface area (dx*dy + dy*dz + dz*dx); the factor of 2 cancels in
// every SAH cost ratio, so it is dropped. Empty boxes contribute 0.
inline TriangleBVH::Scalar HalfArea(const TriangleBVH::AABBox& b)
{
	if (b.isEmpty())
		return TriangleBVH::Scalar(0);
	const TriangleBVH::Point d = b.sizes();
	return d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
}

// Squared distance from point p to the AABB [bmin, bmax] (0 if inside).
// Vectorised over the three axes (Eigen lowers this to SIMD).
inline TriangleBVH::Scalar PointAABBDist2(const TriangleBVH::Point& p,
                                          const TriangleBVH::Scalar* bmin,
                                          const TriangleBVH::Scalar* bmax)
{
	using S = TriangleBVH::Scalar;
	const Eigen::Array<S, 3, 1> lo(bmin[0], bmin[1], bmin[2]);
	const Eigen::Array<S, 3, 1> hi(bmax[0], bmax[1], bmax[2]);
	const Eigen::Array<S, 3, 1> e =
	    (lo - p.array()).max(S(0)) + (p.array() - hi).max(S(0));
	return (e * e).sum();
}

// Ray/AABB slab test; returns true on overlap and writes the entry distance
// max(t_enter, 0) to tNear. invDir = 1/ray.direction() (component-wise).
inline bool RayAABB(const TriangleBVH::Point& o, const TriangleBVH::Point& invDir,
                    const TriangleBVH::Scalar* bmin, const TriangleBVH::Scalar* bmax,
                    TriangleBVH::Scalar tMax, TriangleBVH::Scalar& tNear)
{
	using S = TriangleBVH::Scalar;
	S tmin = S(0), tmax = tMax;
	for (int a = 0; a < 3; ++a) {
		const S t1 = (bmin[a] - o[a]) * invDir[a];
		const S t2 = (bmax[a] - o[a]) * invDir[a];
		// Boundary-robust slab test (tavianator 2022): pass the known-non-NaN
		// accumulator first so a 0*inf=NaN from an axis-parallel ray on a slab
		// face is dropped instead of poisoning the interval.
		tmin = std::min(std::max(tmin, t1), std::max(tmin, t2));
		tmax = std::max(std::min(tmax, t1), std::min(tmax, t2));
	}
	tNear = tmin;
	return tmax >= tmin;
}

} // namespace

void TriangleBVH::Build(const Mesh& mesh, unsigned leafSize)
{
	const_cast<Mesh&>(mesh).SyncFaces();
	nodes.clear();
	prims.clear();
	centroids.clear();
	faceBoxes.clear();
	bbox.setEmpty();

	const uint32_t n = static_cast<uint32_t>(mesh.faces.size());
	if (n == 0)
		return;

	prims.resize(n);
	centroids.resize(n);
	// Per-face AABBs, computed once here (like centroids) so Subdivide extends node
	// boxes with one gather + 6 scalars per face instead of chasing faces[] -> 3
	// scattered vertices (9 scalars) at every tree level. AABB-of-AABBs is bit-
	// identical to AABB-of-vertices, so node boxes and query answers are unchanged.
	faceBoxes.resize(n);
	for (uint32_t i = 0; i < n; ++i) {
		prims[i] = i;
		const Mesh::Face& f = mesh.faces[i];
		const Point& a = mesh.vertices[f[0]];
		const Point& b = mesh.vertices[f[1]];
		const Point& c = mesh.vertices[f[2]];
		centroids[i] = (a + b + c) / Scalar(3);
		AABBox& box = faceBoxes[i];
		box.setEmpty();
		box.extend(a);
		box.extend(b);
		box.extend(c);
	}

	nodes.reserve(2 * n);
	nodes.emplace_back(); // root at index 0
	Subdivide(0, 0, n, std::max(1u, leafSize), 0);

	bbox.min() = Point(nodes[0].bmin[0], nodes[0].bmin[1], nodes[0].bmin[2]);
	bbox.max() = Point(nodes[0].bmax[0], nodes[0].bmax[1], nodes[0].bmax[2]);

	// faceBoxes is build-only scratch; release it (centroids is kept small).
	faceBoxes.clear();
	faceBoxes.shrink_to_fit();
}

void TriangleBVH::Subdivide(uint32_t nodeIdx, uint32_t first, uint32_t count,
                            unsigned leafSize, unsigned depth)
{
	// AABB over this node's triangles (union of precomputed per-face boxes), plus
	// centroid bounds. Union of face AABBs == union of all vertices, exactly.
	AABBox box;
	box.setEmpty();
	AABBox cbox;
	cbox.setEmpty();
	for (uint32_t k = first; k < first + count; ++k) {
		box.extend(faceBoxes[prims[k]]);
		cbox.extend(centroids[prims[k]]);
	}

	auto writeBox = [&](Node& nd) {
		for (int a = 0; a < 3; ++a) {
			nd.bmin[a] = box.min()[a];
			nd.bmax[a] = box.max()[a];
		}
	};

	auto makeLeaf = [&]() {
		Node& nd = nodes[nodeIdx];
		writeBox(nd);
		nd.leftFirst = first;
		nd.count = count;
	};

	if (count <= leafSize) {
		makeLeaf();
		return;
	}

	// Widest centroid axis (the split axis); SAH and the median fallback both use it.
	const Point ext = cbox.diagonal();
	int axis = 0;
	if (ext[1] > ext[axis])
		axis = 1;
	if (ext[2] > ext[axis])
		axis = 2;
	const Scalar cmin = cbox.min()[axis];
	const Scalar cext = ext[axis];

	// TRULY balanced median split on the split axis: positional count/2 via
	// nth_element over the centroids, ties broken by primitive index so the
	// partition is fully deterministic. BOTH children get floor/ceil(count/2)
	// primitives, so once this split is forced (depth >= FORCE_MEDIAN_DEPTH) the
	// count halves every level and the height bound — hence the traversal-stack
	// bound — is real. (A midpoint-of-bounds partition is NOT balanced: on a
	// geometric/exponential centroid spread it peels O(1) primitives per level,
	// which is exactly the overflow class this split exists to eliminate.)
	auto medianSplit = [&]() -> uint32_t {
		uint32_t* begin = prims.data() + first;
		uint32_t* end = prims.data() + first + count;
		const uint32_t half = count / 2;
		std::nth_element(begin, begin + half, end, [&](uint32_t a, uint32_t b) {
			const Scalar ca = centroids[a][axis];
			const Scalar cb = centroids[b][axis];
			if (ca != cb)
				return ca < cb;
			return a < b; // deterministic tie handling
		});
		return half; // count > leafSize >= 1, so both sides are non-empty
	};

	uint32_t leftCount = 0;
	const Scalar scale = (cext > Scalar(0)) ? Scalar(SAH_BINS) / cext : Scalar(0);
	if (depth >= FORCE_MEDIAN_DEPTH || !(cext > Scalar(0)) || !std::isfinite(cext) || !std::isfinite(scale)) {
		// Forced balance (depth cap), degenerate centroid spread, or a spread so
		// extreme that the binning arithmetic is not finite (inf/NaN bin indices
		// would be undefined behavior on the int cast): balanced median split.
		leftCount = medianSplit();
	} else {
		// Binned SAH (Wald 2007): bin primitives by centroid along the widest axis,
		// accumulate per-bin geometric boxes/counts, then sweep prefix (left) and
		// suffix (right) box areas to find the min surface-area-weighted split cost.
		struct Bin
		{
			AABBox box;
			uint32_t count;
		};
		Bin bins[SAH_BINS];
		for (Bin& bn : bins) {
			bn.box.setEmpty();
			bn.count = 0;
		}
		auto binOf = [&](uint32_t p) -> int {
			int b = static_cast<int>((centroids[p][axis] - cmin) * scale);
			if (b < 0)
				b = 0;
			else if (b >= SAH_BINS)
				b = SAH_BINS - 1;
			return b;
		};
		for (uint32_t k = first; k < first + count; ++k) {
			const uint32_t p = prims[k];
			const int b = binOf(p);
			bins[b].box.extend(faceBoxes[p]);
			++bins[b].count;
		}
		// prefix (left) sweep: area & count of bins [0..i]
		Scalar leftArea[SAH_BINS - 1];
		uint32_t leftCnt[SAH_BINS - 1];
		AABBox acc;
		acc.setEmpty();
		uint32_t c = 0;
		for (int i = 0; i < SAH_BINS - 1; ++i) {
			acc.extend(bins[i].box);
			c += bins[i].count;
			leftArea[i] = HalfArea(acc);
			leftCnt[i] = c;
		}
		// suffix (right) sweep: area & count of bins [i+1..end]
		Scalar rightArea[SAH_BINS - 1];
		uint32_t rightCnt[SAH_BINS - 1];
		acc.setEmpty();
		c = 0;
		for (int i = SAH_BINS - 1; i > 0; --i) {
			acc.extend(bins[i].box);
			c += bins[i].count;
			rightArea[i - 1] = HalfArea(acc);
			rightCnt[i - 1] = c;
		}
		// pick the min-cost partition plane (deterministic: first minimum wins)
		Scalar bestCost = std::numeric_limits<Scalar>::max();
		int bestBin = -1;
		for (int i = 0; i < SAH_BINS - 1; ++i) {
			if (leftCnt[i] == 0 || rightCnt[i] == 0)
				continue;
			const Scalar cost =
			    leftArea[i] * Scalar(leftCnt[i]) + rightArea[i] * Scalar(rightCnt[i]);
			if (cost < bestCost) {
				bestCost = cost;
				bestBin = i;
			}
		}

		if (bestBin < 0) {
			// no separating plane (all centroids in one bin) -> median fallback
			leftCount = medianSplit();
		} else if (bestCost >= HalfArea(box) * Scalar(count)) {
			// Splitting does not reduce the surface-area-weighted cost below the
			// leaf cost (heavily overlapping prims). Accept the leaf only while it
			// is small; an unbounded leaf would make every query touching it an
			// O(leaf) linear scan, so past the cap force the balanced median split.
			if (count <= SAH_MAX_LEAF) {
				makeLeaf();
				return;
			}
			leftCount = medianSplit();
		} else {
			uint32_t* begin = prims.data() + first;
			uint32_t* end = prims.data() + first + count;
			uint32_t* split = std::partition(begin, end, [&](uint32_t p) {
				return binOf(p) <= bestBin;
			});
			leftCount = static_cast<uint32_t>(split - begin);
			// The chosen plane sits between two non-empty bin groups, so both sides
			// are non-empty; guard anyway against float re-binning edge cases.
			if (leftCount == 0 || leftCount == count)
				leftCount = medianSplit();
		}
	}

	const uint32_t leftChild = static_cast<uint32_t>(nodes.size());
	nodes.emplace_back();
	nodes.emplace_back();

	{
		Node& nd = nodes[nodeIdx];
		writeBox(nd);
		nd.leftFirst = leftChild;
		nd.count = 0;
	}

	Subdivide(leftChild, first, leftCount, leafSize, depth + 1);
	Subdivide(leftChild + 1, first + leftCount, count - leftCount, leafSize, depth + 1);
}

TriangleBVH::NearestNeighbor TriangleBVH::NearestPoint(const Point& p, Scalar maxDist,
                                                       Mesh::FIndex hintFace) const
{
	NearestNeighbor best;
	if (nodes.empty())
		return best;

	// A finite maxDist seeds the squared search bound (radius^2); the default max()
	// reproduces the unbounded search exactly. Validity is decided against this seed
	// so a bounded query that finds nothing reports IsValid()==false.
	const Scalar boundD2 = (maxDist < std::numeric_limits<Scalar>::max())
	                           ? maxDist * maxDist
	                           : std::numeric_limits<Scalar>::max();
	Scalar bestD2 = boundD2;

	// Warm start: test the caller's hint face first so its distance tightens bestD2
	// before traversal, pruning more nodes when successive queries are coherent.
	if (hintFace < static_cast<Mesh::FIndex>(mesh.faces.size())) {
		Point nearPt;
		const Mesh::Face& f = mesh.faces[hintFace];
		const Scalar d2 = math::DistanceBetweenTriangleAndPointSquared(
		    mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]], p, &nearPt);
		if (d2 < bestD2) {
			bestD2 = d2;
			best.nearest = nearPt;
			best.idxFace = hintFace;
		}
	}

	// Explicit traversal stack (no recursion). Each entry stores the node's
	// point-AABB squared distance computed at PUSH time; pruning at pop reuses it
	// instead of recomputing PointAABBDist2 (halving the box tests). The stored value
	// is invariant, so prune decisions are bit-identical to the recompute. Capacity
	// exceeds the build-bounded height, so it cannot overflow (see BVH_STACK_SIZE).
	struct Entry
	{
		uint32_t node;
		Scalar d2;
	};
	Entry stack[BVH_STACK_SIZE];
	int sp = 0;
	stack[sp++] = {0u, Scalar(0)};

	while (sp > 0) {
		const Entry e = stack[--sp];
		if (e.d2 >= bestD2)
			continue; // nothing in this box can beat the current best
		const Node& nd = nodes[e.node];

		if (nd.count > 0) { // leaf
			Point nearPt;
			for (uint32_t k = nd.leftFirst; k < nd.leftFirst + nd.count; ++k) {
				const Mesh::FIndex fi = prims[k];
				const Mesh::Face& f = mesh.faces[fi];
				// Squared distance: the BVH already prunes in squared space, so
				// this drops the wasted per-triangle sqrt (taken once, at the end).
				const Scalar d2 = math::DistanceBetweenTriangleAndPointSquared(
				    mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]], p, &nearPt);
				if (d2 < bestD2) {
					bestD2 = d2;
					best.nearest = nearPt;
					best.idxFace = fi;
				}
			}
		} else { // internal: descend nearer child first
			const uint32_t c0 = nd.leftFirst, c1 = c0 + 1;
			const Scalar d0 = PointAABBDist2(p, nodes[c0].bmin, nodes[c0].bmax);
			const Scalar d1 = PointAABBDist2(p, nodes[c1].bmin, nodes[c1].bmax);
			// push the farther child first so the nearer one is popped next
			if (d0 < d1) {
				if (d1 < bestD2)
					stack[sp++] = {c1, d1};
				if (d0 < bestD2)
					stack[sp++] = {c0, d0};
			} else {
				if (d0 < bestD2)
					stack[sp++] = {c0, d0};
				if (d1 < bestD2)
					stack[sp++] = {c1, d1};
			}
		}
	}
	// A single sqrt converts the tracked squared distance back to Euclidean. If
	// nothing beat the seeded bound (empty tree or bounded miss), leave the max()
	// sentinel so IsValid() stays false.
	if (bestD2 < boundD2)
		best.dist = std::sqrt(bestD2);
	return best;
}

TriangleBVH::NearestNeighbor TriangleBVH::IntersectedPoint(
    const Eigen::ParametrizedLine<Scalar, 3>& ray, Scalar maxDist) const
{
	NearestNeighbor best;
	if (nodes.empty())
		return best;

	const Point o = ray.origin();
	const Point dir = ray.direction();
	const Point invDir(Scalar(1) / dir[0], Scalar(1) / dir[1], Scalar(1) / dir[2]);
	// bestT bounds the ray parameter DURING traversal: seed it with maxDist so
	// subtrees entered beyond that distance are pruned (default max() = unbounded).
	Scalar bestT = maxDist;

	// Explicit traversal stack; each entry stores the node's ray entry-t (tmin)
	// computed at PUSH time, reused for pruning at pop instead of recomputing
	// RayAABB. tmin is independent of the tMax cap, so it is invariant; the
	// pop-time prune (stored >= bestT) is equivalent to the recompute because a
	// hit at t >= tmin is rejected by the half-open [0, bestT) triangle window.
	// Capacity exceeds the build-bounded height (see BVH_STACK_SIZE).
	struct Entry
	{
		uint32_t node;
		Scalar t;
	};
	Entry stack[BVH_STACK_SIZE];
	int sp = 0;
	stack[sp++] = {0u, Scalar(0)};

	while (sp > 0) {
		const Entry e = stack[--sp];
		if (e.t >= bestT)
			continue; // this box is entered only beyond the current best hit
		const Node& nd = nodes[e.node];

		if (nd.count > 0) { // leaf
			for (uint32_t k = nd.leftFirst; k < nd.leftFirst + nd.count; ++k) {
				const Mesh::FIndex fi = prims[k];
				const Mesh::Face& f = mesh.faces[fi];
				Scalar t;
				if (math::RayTriangleIntersect(ray, mesh.vertices[f[0]], mesh.vertices[f[1]],
				                               mesh.vertices[f[2]], Scalar(0), bestT, &t)) {
					bestT = t;
					best.dist = t;
					best.nearest = ray.pointAt(t);
					best.idxFace = fi;
				}
			}
		} else { // internal: visit nearer child first
			const uint32_t c0 = nd.leftFirst, c1 = c0 + 1;
			Scalar t0, t1;
			const bool h0 = RayAABB(o, invDir, nodes[c0].bmin, nodes[c0].bmax, bestT, t0);
			const bool h1 = RayAABB(o, invDir, nodes[c1].bmin, nodes[c1].bmax, bestT, t1);
			// push the farther child first so the nearer one is popped next
			if (h0 && h1) {
				if (t0 < t1) {
					stack[sp++] = {c1, t1};
					stack[sp++] = {c0, t0};
				} else {
					stack[sp++] = {c0, t0};
					stack[sp++] = {c1, t1};
				}
			} else if (h0) {
				stack[sp++] = {c0, t0};
			} else if (h1) {
				stack[sp++] = {c1, t1};
			}
		}
	}
	return best;
}

unsigned TriangleBVH::MaxLeafSize() const
{
	unsigned maxCount = 0;
	for (const Node& nd : nodes)
		if (nd.count > maxCount)
			maxCount = nd.count;
	return maxCount;
}

unsigned TriangleBVH::Height() const
{
	if (nodes.empty())
		return 0;
	return HeightRecurse(0);
}

unsigned TriangleBVH::HeightRecurse(uint32_t nodeIdx) const
{
	const Node& nd = nodes[nodeIdx];
	if (nd.count > 0) // leaf
		return 0;
	const unsigned hl = HeightRecurse(nd.leftFirst);
	const unsigned hr = HeightRecurse(nd.leftFirst + 1);
	return 1u + (hl > hr ? hl : hr);
}

} // namespace halfmesh

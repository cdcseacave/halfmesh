/*
* IglCrosscheckTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/crosscheck/IglCrosscheckTest.cpp — libigl layer-3 cross-checks
//
// Independent third-party verification (docs/TESTING.md §1 layer 3).
// Compares halfmesh outputs against libigl on the shared test corpus.
//
// Cross-checks implemented:
//   1. Adjacency       — igl::triangle_triangle_adjacency vs HalfMesh::FAdjacentFaces
//   2. Components      — igl::facet_components count vs HalfMesh::ConnectedComponents
//   3. Decimation      — igl::qslim vs Mesh::Simplify Hausdorff ballpark
//   4. UV distortion   — igl::lscm vs halfmesh ARAP on a flat disk chart
//
// Deferred (not in this vcpkg registry):
//   - xatlas         (atlas packing cross-check)
//   - geometry-central (Laplacian / geodesics)
//   - CGAL           (excluded by task brief — too heavy)
//
// Tolerance policy (docs/TESTING.md §5):
//   - Exact checks   : adjacency, component count  (pure combinatorics)
//   - Ballpark check : Hausdorff ratio ∈ [0.5, 2.0] (same order of magnitude)
//   - Ballpark check : UV symmetric-Dirichlet within factor 3× of igl::lscm
//                      (igl::lscm chosen over igl::arap: arap segfaults in
//                      header-only mode after memory-intensive tests on arm64
//                      due to deprecated Eigen SVD static state; lscm is a
//                      clean linear solve with no shared state and is an
//                      equally valid independent conformal UV reference)

#include <gtest/gtest.h>

// halfmesh public API
#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Parametrize.h>

// test support
#include "Corpus.h" // hmtest::corpus
#include "Metrics.h" // hmtest::metrics

// libigl
#include <igl/triangle_triangle_adjacency.h>
#include <igl/facet_components.h>
#include <igl/qslim.h>
#include <igl/lscm.h>
#include <igl/boundary_loop.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Adapter: halfmesh::Mesh → Eigen V (#V×3 double) and F (#F×3 int)
// ---------------------------------------------------------------------------
static void meshToEigen(
    const halfmesh::Mesh& m,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F)
{
	const int nv = static_cast<int>(m.vertices.size());
	const int nf = static_cast<int>(m.faces.size());
	V.resize(nv, 3);
	for (int i = 0; i < nv; ++i) {
		V(i, 0) = static_cast<double>(m.vertices[i].x());
		V(i, 1) = static_cast<double>(m.vertices[i].y());
		V(i, 2) = static_cast<double>(m.vertices[i].z());
	}
	F.resize(nf, 3);
	for (int i = 0; i < nf; ++i) {
		F(i, 0) = static_cast<int>(m.faces[i][0]);
		F(i, 1) = static_cast<int>(m.faces[i][1]);
		F(i, 2) = static_cast<int>(m.faces[i][2]);
	}
}

// ---------------------------------------------------------------------------
// Adapter: Eigen (U, G) → halfmesh::Mesh
// ---------------------------------------------------------------------------
static halfmesh::Mesh eigenToMesh(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& G)
{
	halfmesh::Mesh m;
	m.vertices.reserve(static_cast<size_t>(U.rows()));
	for (int i = 0; i < U.rows(); ++i)
		m.vertices.emplace_back(
		    static_cast<float>(U(i, 0)),
		    static_cast<float>(U(i, 1)),
		    static_cast<float>(U(i, 2)));
	m.faces.reserve(static_cast<size_t>(G.rows()));
	for (int i = 0; i < G.rows(); ++i)
		m.faces.push_back({static_cast<halfmesh::Mesh::FIndex>(G(i, 0)),
		                   static_cast<halfmesh::Mesh::FIndex>(G(i, 1)),
		                   static_cast<halfmesh::Mesh::FIndex>(G(i, 2))});
	return m;
}

// ============================================================
// CHECK 1 — FACE ADJACENCY (exact)
// ============================================================
//
// Strategy: for each face f in the mesh, collect the set of adjacent face
// indices from (a) igl::triangle_triangle_adjacency TT and (b) iterating
// HalfMesh::FAdjacentFaces.  The two sets must be identical on every corpus
// manifold mesh.
//
// Tolerance: EXACT — both are pure combinatorial algorithms with no float
// arithmetic.  A -1 entry in TT denotes a boundary edge (no adjacent face).

class AdjacencyTest : public ::testing::TestWithParam<std::string>
{
};

static void runAdjacency(const halfmesh::Mesh& meshIn)
{
	halfmesh::Mesh mesh = meshIn;
	mesh.ListHalfEdges(); // build HalfMesh

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	meshToEigen(mesh, V, F);
	const int nf = F.rows();

	// libigl adjacency
	Eigen::MatrixXi TT;
	igl::triangle_triangle_adjacency(F, TT);

	const halfmesh::HalfMesh& hm = mesh.halfMesh;

	for (int fi = 0; fi < nf; ++fi) {
		// Collect igl neighbours (exclude -1 = boundary)
		std::set<int> iglNbrs;
		for (int j = 0; j < 3; ++j) {
			if (TT(fi, j) != -1)
				iglNbrs.insert(TT(fi, j));
		}

		// Collect halfmesh neighbours (range-based for over IteratorStateBase)
		std::set<int> hmNbrs;
		for (halfmesh::HalfMesh::FIndex g :
		     hm.FAdjacentFaces(static_cast<halfmesh::HalfMesh::FIndex>(fi))) {
			hmNbrs.insert(static_cast<int>(g));
		}

		EXPECT_EQ(iglNbrs, hmNbrs)
		    << "Face " << fi << ": igl and halfmesh adjacency sets differ";
	}
}

TEST(Adjacency, TetrahedronExact)
{
	runAdjacency(hmtest::corpus::TetrahedronMesh());
}
TEST(Adjacency, CubeExact)
{
	runAdjacency(hmtest::corpus::CubeMesh());
}
TEST(Adjacency, IcosahedronExact)
{
	runAdjacency(hmtest::corpus::IcosahedronMesh());
}
TEST(Adjacency, GridPlaneExact)
{
	runAdjacency(hmtest::corpus::GridPlane(4));
}
TEST(Adjacency, UVSphereExact)
{
	runAdjacency(hmtest::corpus::UVSphere(6, 8));
}

// ============================================================
// CHECK 2 — CONNECTED COMPONENTS (exact)
// ============================================================
//
// Strategy: compare igl::facet_components return value (number of components)
// against HalfMesh::ConnectedComponents count on the same mesh.
//
// Tolerance: EXACT — both count edge-connected face clusters; pure graph BFS.

static void runComponents(const halfmesh::Mesh& meshIn,
                          unsigned expectedCount = 0)
{
	halfmesh::Mesh mesh = meshIn;
	mesh.ListHalfEdges();

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	meshToEigen(mesh, V, F);

	// libigl component count
	Eigen::VectorXi C;
	const int iglCount = igl::facet_components(F, C);

	// halfmesh component count
	std::vector<halfmesh::HalfMesh::FIndex> comp;
	const unsigned hmCount = mesh.halfMesh.ConnectedComponents(comp);

	EXPECT_EQ(static_cast<unsigned>(iglCount), hmCount)
	    << "Component count mismatch: igl=" << iglCount
	    << " halfmesh=" << hmCount;

	if (expectedCount > 0)
		EXPECT_EQ(hmCount, expectedCount);
}

TEST(Components, Tetrahedron)
{
	runComponents(hmtest::corpus::TetrahedronMesh(), 1u);
}
TEST(Components, Cube)
{
	runComponents(hmtest::corpus::CubeMesh(), 1u);
}
TEST(Components, Icosahedron)
{
	runComponents(hmtest::corpus::IcosahedronMesh(), 1u);
}
TEST(Components, GridPlane)
{
	runComponents(hmtest::corpus::GridPlane(4), 1u);
}
TEST(Components, UVSphere)
{
	runComponents(hmtest::corpus::UVSphere(6, 8), 1u);
}
TEST(Components, ManyComponents)
{
	// Build a mesh with 3 disconnected components: tetra + cube + icosahedron
	// by offsetting vertices so they don't overlap.
	halfmesh::Mesh combined;
	auto append = [&](halfmesh::Mesh src, float dx) {
		const auto baseV = static_cast<halfmesh::Mesh::FIndex>(combined.vertices.size());
		for (auto& v : src.vertices) {
			auto vv = v;
			vv.x() += dx;
			combined.vertices.push_back(vv);
		}
		for (auto& f : src.faces) {
			combined.faces.push_back({f[0] + baseV, f[1] + baseV, f[2] + baseV});
		}
	};
	append(hmtest::corpus::TetrahedronMesh(), 0.f);
	append(hmtest::corpus::CubeMesh(), 10.f);
	append(hmtest::corpus::IcosahedronMesh(), 20.f);
	runComponents(combined, 3u);
}

// ============================================================
// CHECK 3 — DECIMATION BALLPARK (Hausdorff ratio)
// ============================================================
//
// Strategy: take a mid-size corpus mesh, decimate to half the face count with
// both halfmesh::Mesh::Simplify (decimateRatio=0.5) and igl::qslim; compute
// symmetric Hausdorff distance from each decimated mesh back to the input.
// Assert the ratio (halfmesh_hausdorff / igl_hausdorff) ∈ [0.5, 2.0].
//
// Tolerance: factor ≤ 2.0× (same order of magnitude).  Justification:
//   - Both implementations use QEM (quadric error metrics) but differ in
//     tie-breaking, heap management, and edge-weight initialization.
//   - The Hausdorff is measured vertex-to-surface (KD-tree), not face-to-face,
//     so minor topological differences accumulate.
//   - 2× is generous but not trivial: a degenerate implementation would give
//     errors 10–100× larger.
//
// Notes:
//   - igl::qslim requires a manifold mesh; we use IcosahedronMesh (closed,
//     clean).  The target is ≥4 faces (igl::qslim minimum).
//   - halfmesh Simplify may produce slightly more or fewer faces than the
//     exact target because it uses ratio-based stopping.

static double symmetricHausdorff(const halfmesh::Mesh& a, const halfmesh::Mesh& b)
{
	return hmtest::metrics::ComputeDistanceKdTree(a, b).hausdorffSymmetric;
}

TEST(Decimation, HausdorffBallpark_Icosahedron)
{
	// Icosahedron: 20 faces → target 10
	const halfmesh::Mesh orig = hmtest::corpus::IcosahedronMesh();
	const int target = 10;
	const float ratio = static_cast<float>(target) / static_cast<float>(orig.faces.size());

	// halfmesh decimation
	halfmesh::Mesh hmResult = orig;
	hmResult.Simplify(ratio);
	ASSERT_GT(hmResult.faces.size(), 0u) << "halfmesh Simplify produced empty mesh";

	// igl::qslim decimation
	Eigen::MatrixXd V, U;
	Eigen::MatrixXi F, G;
	Eigen::VectorXi J, I;
	meshToEigen(orig, V, F);
	const bool ok = igl::qslim(V, F, target, false, U, G, J, I);
	ASSERT_TRUE(ok) << "igl::qslim failed";
	ASSERT_GT(G.rows(), 0) << "igl::qslim produced empty mesh";

	const halfmesh::Mesh iglResult = eigenToMesh(U, G);

	// Hausdorff distances to the input
	const double hmH = symmetricHausdorff(orig, hmResult);
	const double iglH = symmetricHausdorff(orig, iglResult);

	ASSERT_GT(iglH, 0.0) << "igl Hausdorff is zero (sanity check)";
	ASSERT_GT(hmH, 0.0) << "halfmesh Hausdorff is zero (sanity check)";

	const double ratioHh = hmH / iglH;
	RecordProperty("hm_hausdorff_x1000", static_cast<int>(hmH * 1000));
	RecordProperty("igl_hausdorff_x1000", static_cast<int>(iglH * 1000));
	RecordProperty("ratio_x1000", static_cast<int>(ratioHh * 1000));

	EXPECT_GE(ratioHh, 0.5) << "halfmesh Hausdorff unexpectedly much BETTER than igl (>2×)";
	EXPECT_LE(ratioHh, 2.0) << "halfmesh Hausdorff unexpectedly much WORSE than igl (>2×)";
}

TEST(Decimation, HausdorffBallpark_UVSphere)
{
	// UV sphere: 96 faces → target 32
	const halfmesh::Mesh orig = hmtest::corpus::UVSphere(8, 12);
	const int target = 32;
	const float ratio = static_cast<float>(target) / static_cast<float>(orig.faces.size());

	halfmesh::Mesh hmResult = orig;
	hmResult.Simplify(ratio);
	ASSERT_GT(hmResult.faces.size(), 0u);

	Eigen::MatrixXd V, U;
	Eigen::MatrixXi F, G;
	Eigen::VectorXi J, I;
	meshToEigen(orig, V, F);
	const bool ok = igl::qslim(V, F, target, false, U, G, J, I);
	ASSERT_TRUE(ok) << "igl::qslim failed on UVSphere";
	ASSERT_GT(G.rows(), 0);

	const halfmesh::Mesh iglResult = eigenToMesh(U, G);

	const double hmH = symmetricHausdorff(orig, hmResult);
	const double iglH = symmetricHausdorff(orig, iglResult);

	ASSERT_GT(iglH, 0.0);
	ASSERT_GT(hmH, 0.0);

	const double ratioHh = hmH / iglH;
	RecordProperty("hm_hausdorff_x1000", static_cast<int>(hmH * 1000));
	RecordProperty("igl_hausdorff_x1000", static_cast<int>(iglH * 1000));
	RecordProperty("ratio_x1000", static_cast<int>(ratioHh * 1000));

	EXPECT_GE(ratioHh, 0.5);
	EXPECT_LE(ratioHh, 2.0);
}

// ============================================================
// CHECK 4 — UV DISTORTION (igl::lscm vs halfmesh ARAP/SLIM)
// ============================================================
//
// Strategy: flatten a disk-topology mesh (GridPlane 5×5 = 50 triangles) with
// both igl::lscm (Least-Squares Conformal Map — a well-tested conformal UV
// method, linear solve, minimal setup) and halfmesh ARAP.  Compare:
//   (a) Both produce flip-free UVs (flips == 0).
//   (b) Both produce finite UVs.
//   (c) halfmesh symmetric-Dirichlet ≤ 3× igl LSCM's (same quality band).
//
// Note: igl::arap was the original first choice but exhibits a segfault when
// run after memory-intensive tests in the same process (igl ARAP relies on
// static SVD state in the header-only build mode; the crash is a known
// header-only ARAP issue with the vcpkg arm64 build).  igl::lscm is a clean
// linear solve with no shared state and is equally valid as an independent
// third-party UV reference.
//
// Tolerance: factor ≤ 3.0× (generous).  Justification:
//   - LSCM minimizes conformal energy (angle preservation), while halfmesh
//     ARAP minimizes as-rigid-as-possible energy (both angle + scale).
//   - On a flat grid both converge to near-isometric maps, so their
//     symmetric-Dirichlet values are both ≈ 4 per face (minimum possible).
//   - 3× tolerates the different energy formulations without being trivial.
//
// igl::lscm setup:
//   - Pin two opposite corner vertices of the grid to (0,0) and (1,0).
//   - The remaining vertices are solved via a sparse linear system.
//   - No iterative ARAP or dynamic-state issues.

TEST(UVDistortion, LscmVsHalfmeshARAP_FlatGrid)
{
	// Build a flat 5×5 grid (disk topology, 50 triangles, 1 boundary loop)
	halfmesh::Mesh mesh = hmtest::corpus::GridPlane(5);
	mesh.ListHalfEdges();
	mesh.ComputeFaceNormals();

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	meshToEigen(mesh, V, F);

	// ---- igl::lscm parametrization ----
	// Pin vertex 0 → (0,0) and vertex at opposite corner → (1,0).
	// For a GridPlane(5): (5+1)^2 = 36 vertices; last vertex index = 35.
	const int nv = static_cast<int>(V.rows());
	Eigen::VectorXi bLscm(2);
	bLscm(0) = 0;
	bLscm(1) = nv - 1;
	Eigen::MatrixXd bcLscm(2, 2);
	bcLscm << 0.0, 0.0,
	    1.0, 0.0;

	Eigen::MatrixXd V_uv_lscm;
	const bool lscmOk = igl::lscm(V, F, bLscm, bcLscm, V_uv_lscm);
	ASSERT_TRUE(lscmOk) << "igl::lscm failed";
	ASSERT_EQ(V_uv_lscm.rows(), nv) << "lscm UV row count mismatch";
	ASSERT_EQ(V_uv_lscm.cols(), 2) << "lscm UV must be Nx2";

	// Inject LSCM UVs into a copy of the mesh for ComputeUVMetrics
	halfmesh::Mesh meshLscm = mesh;
	meshLscm.faceTexcoords.resize(meshLscm.faces.size() * 3);
	for (size_t fi = 0; fi < meshLscm.faces.size(); ++fi) {
		for (int c = 0; c < 3; ++c) {
			const int vi = static_cast<int>(meshLscm.faces[fi][c]);
			meshLscm.faceTexcoords[fi * 3 + c] = halfmesh::Mesh::TexCoord(
			    static_cast<float>(V_uv_lscm(vi, 0)),
			    static_cast<float>(V_uv_lscm(vi, 1)));
		}
	}
	const hmtest::metrics::UVMetrics lscmUv = hmtest::metrics::ComputeUVMetrics(meshLscm);

	// ---- halfmesh ARAP parametrization ----
	halfmesh::ParametrizeParams params;
	params.method = halfmesh::ParametrizeParams::FlattenMethod::ARAP;
	params.flattenIterations = 5;
	const unsigned ncharts = halfmesh::Parametrize(mesh, params);
	ASSERT_GE(ncharts, 1u) << "halfmesh Parametrize produced no charts";
	const hmtest::metrics::UVMetrics hmUv = hmtest::metrics::ComputeUVMetrics(mesh);

	// Record for visibility
	RecordProperty("igl_lscm_sym_dirichlet_x1000", static_cast<int>(lscmUv.symDirichlet * 1000));
	RecordProperty("hm_arap_sym_dirichlet_x1000", static_cast<int>(hmUv.symDirichlet * 1000));
	RecordProperty("igl_lscm_flips", lscmUv.flipCount);
	RecordProperty("hm_arap_flips", hmUv.flipCount);

	// (a) Both flip-free
	EXPECT_EQ(lscmUv.flipCount, 0) << "igl::lscm produced flipped triangles";
	EXPECT_EQ(hmUv.flipCount, 0) << "halfmesh ARAP produced flipped triangles";

	// (b) Finite UVs from both
	EXPECT_TRUE(lscmUv.allFinite) << "igl::lscm produced non-finite UVs";
	EXPECT_TRUE(hmUv.allFinite) << "halfmesh ARAP produced non-finite UVs";

	// (c) Comparable symmetric-Dirichlet: halfmesh ≤ 3× LSCM's on a flat grid.
	// Both should be ≈ 4 per face (isometric minimum); 3× allows for method differences.
	if (lscmUv.symDirichlet > 0.0 && hmUv.symDirichlet > 0.0) {
		const double ratio = hmUv.symDirichlet / lscmUv.symDirichlet;
		RecordProperty("distortion_ratio_x1000", static_cast<int>(ratio * 1000));
		EXPECT_LE(ratio, 3.0)
		    << "halfmesh ARAP sym-Dirichlet is >3× worse than igl::lscm on a flat grid"
		    << " (hm=" << hmUv.symDirichlet << " lscm=" << lscmUv.symDirichlet << ")";
	}
}

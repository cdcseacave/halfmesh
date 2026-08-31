/*
* FlattenTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for src/Parametrize.cpp Module B:
//   halfmesh::ParametrizeCharts / Parametrize — per-chart UV flattening
//   (Tutte init → ARAP / SLIM local-global).
//
// The tests are property-based and distortion/flip-oracle driven:
//   1. Validity:               all UVs finite; faceTexcoords sized 3*faces.
//   2. Planar → isometric:     a flat grid patch flattens with ~0 distortion
//                              (edge-length ratios near constant; per-triangle
//                              symmetric-Dirichlet energy ≈ minimum). KEY ORACLE.
//   3. No flips:               every chart triangle keeps a consistent UV
//                              orientation (signed area same sign). For SLIM the
//                              count must be exactly 0; for ARAP a few allowed.
//   4. Developable → low dist: a bent (developable) strip flattens with low,
//                              bounded distortion (not necessarily zero).
//   5. mesh.ply end-to-end:    Parametrize(segment+flatten) yields valid finite
//                              UVs, bounded average distortion, and (for SLIM)
//                              zero flips.

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>

// Internal Module A<->B bridge header (src/ on this target's include path — see
// tests/CMakeLists.txt): brings in detail::FoldDiagnosis + the 5-arg
// detail::ChartFacesFold overload the fold-diagnosis test below calls directly
// (mirrors tests/AtlasTest.cpp's use of the same header).
#include "ChartFlattenCache.h"
#include "Corpus.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric> // std::iota (FoldDiagnosisReportsOffendingFaces face list)
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace halfmesh {
namespace {

std::string TestMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

// ---------------------------------------------------------------------------
// A flat grid patch in the z=0 plane (nx*ny quads → 2*nx*ny triangles). This is
// a single chart with zero curvature: its isometric flattening is the identity
// (up to rigid motion + uniform scale).
// ---------------------------------------------------------------------------
Mesh MakeFlatGrid(int nx, int ny, float dx = 1.0f, float dy = 1.0f)
{
	Mesh m;
	auto idx = [&](int i, int j) { return static_cast<uint32_t>(j * (nx + 1) + i); };
	for (int j = 0; j <= ny; ++j)
		for (int i = 0; i <= nx; ++i)
			m.vertices.emplace_back(i * dx, j * dy, 0.0f);
	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i) {
			m.faces.emplace_back(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1));
			m.faces.emplace_back(idx(i, j), idx(i + 1, j + 1), idx(i, j + 1));
		}
	return m;
}

// ---------------------------------------------------------------------------
// A developable strip: a grid bent into a partial cylinder around the y axis.
// Each quad row keeps its (intrinsic) 3D edge lengths, so it can be unrolled to
// the plane with low distortion (cylinders are developable).
// ---------------------------------------------------------------------------
Mesh MakeBentStrip(int nx, int ny, float radius = 4.0f, float arc = 1.2f)
{
	Mesh m;
	auto idx = [&](int i, int j) { return static_cast<uint32_t>(j * (nx + 1) + i); };
	for (int j = 0; j <= ny; ++j)
		for (int i = 0; i <= nx; ++i) {
			const float a = arc * (static_cast<float>(i) / nx - 0.5f);
			const float x = radius * std::sin(a);
			const float z = radius * std::cos(a);
			const float y = static_cast<float>(j);
			m.vertices.emplace_back(x, y, z);
		}
	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i) {
			m.faces.emplace_back(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1));
			m.faces.emplace_back(idx(i, j), idx(i + 1, j + 1), idx(i, j + 1));
		}
	return m;
}

// ---------------------------------------------------------------------------
// Per-chart distortion / flip statistics computed from faceTexcoords.
// ---------------------------------------------------------------------------
struct DistortStats
{
	double maxSdEnergy = 0.0; // worst per-triangle symmetric-Dirichlet energy
	double avgSdEnergy = 0.0; // area-weighted mean symmetric-Dirichlet energy
	double edgeRatioCov = 0.0; // coefficient of variation of UV/3D edge ratio
	int flips = 0; // triangles whose UV signed area sign flips
	int triangles = 0;
	bool allFinite = true;
};

// Symmetric-Dirichlet energy of a triangle: map the 3D triangle isometrically
// to 2D (reference) then compute the 2x2 Jacobian to the UV triangle and its
// singular values s0,s1 → E = s0^2 + s1^2 + 1/s0^2 + 1/s1^2 (min = 4).
double TriSymDir(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                 const Eigen::Vector3d& p2, const Eigen::Vector2d& u0,
                 const Eigen::Vector2d& u1, const Eigen::Vector2d& u2,
                 double* area3d)
{
	const Eigen::Vector3d e1 = p1 - p0, e2 = p2 - p0;
	const Eigen::Vector3d nrm = e1.cross(e2);
	const double a3 = 0.5 * nrm.norm();
	if (area3d)
		*area3d = a3;
	if (a3 < 1e-14)
		return 4.0; // degenerate source — treat as ideal
	const double l1 = e1.norm();
	const Eigen::Vector3d xax = e1 / l1;
	Eigen::Vector3d yax = nrm.cross(xax);
	yax /= yax.norm();
	Eigen::Matrix2d Xr;
	Xr.col(0) = Eigen::Vector2d(l1, 0.0) - Eigen::Vector2d(0.0, 0.0);
	Xr.col(1) = Eigen::Vector2d(e2.dot(xax), e2.dot(yax)) - Eigen::Vector2d(0.0, 0.0);
	Eigen::Matrix2d Uu;
	Uu.col(0) = u1 - u0;
	Uu.col(1) = u2 - u0;
	if (std::abs(Xr.determinant()) < 1e-18)
		return 4.0;
	const Eigen::Matrix2d J = Uu * Xr.inverse();
	Eigen::JacobiSVD<Eigen::Matrix2d> svd(J);
	double s0 = svd.singularValues()[0], s1 = svd.singularValues()[1];
	s0 = std::max(s0, 1e-9);
	s1 = std::max(s1, 1e-9);
	return s0 * s0 + s1 * s1 + 1.0 / (s0 * s0) + 1.0 / (s1 * s1);
}

DistortStats ComputeStats(const Mesh& m)
{
	DistortStats s;
	const size_t nf = m.faces.size();
	if (m.faceTexcoords.size() != nf * 3) {
		s.allFinite = false;
		return s;
	}
	double sumE = 0.0, sumA = 0.0;
	// edge-ratio statistics (UV length / 3D length) over all face edges.
	std::vector<double> ratios;
	ratios.reserve(nf * 3);
	for (size_t f = 0; f < nf; ++f) {
		Eigen::Vector3d p[3];
		Eigen::Vector2d u[3];
		for (int k = 0; k < 3; ++k) {
			p[k] = m.vertices[m.faces[f][k]].cast<double>();
			u[k] = m.faceTexcoords[f * 3 + k].cast<double>();
			if (!std::isfinite(u[k].x()) || !std::isfinite(u[k].y()))
				s.allFinite = false;
		}
		double a3 = 0.0;
		const double e = TriSymDir(p[0], p[1], p[2], u[0], u[1], u[2], &a3);
		s.maxSdEnergy = std::max(s.maxSdEnergy, e);
		sumE += a3 * e;
		sumA += a3;
		++s.triangles;
		// signed UV area (flip if <= 0).
		const double sa = (u[1].x() - u[0].x()) * (u[2].y() - u[0].y()) - (u[2].x() - u[0].x()) * (u[1].y() - u[0].y());
		if (sa <= 0.0)
			++s.flips;
		for (int k = 0; k < 3; ++k) {
			const double l3 = (p[(k + 1) % 3] - p[k]).norm();
			const double l2 = (u[(k + 1) % 3] - u[k]).norm();
			if (l3 > 1e-12)
				ratios.push_back(l2 / l3);
		}
	}
	s.avgSdEnergy = (sumA > 0.0) ? sumE / sumA : 0.0;
	if (!ratios.empty()) {
		double mean = 0.0;
		for (double r : ratios)
			mean += r;
		mean /= ratios.size();
		double var = 0.0;
		for (double r : ratios)
			var += (r - mean) * (r - mean);
		var /= ratios.size();
		s.edgeRatioCov = (mean > 1e-12) ? std::sqrt(var) / mean : 0.0;
	}
	return s;
}

// ---------------------------------------------------------------------------
// 1. Validity: faceTexcoords sized 3*faces; all UVs finite. Both methods.
// ---------------------------------------------------------------------------
TEST(Flatten, ValidityBothMethods)
{
	for (auto method : {ParametrizeParams::FlattenMethod::ARAP,
	                    ParametrizeParams::FlattenMethod::SLIM}) {
		Mesh m = MakeFlatGrid(4, 4);
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeParams params;
		params.method = method;
		const unsigned n = Parametrize(m, params);
		EXPECT_GE(n, 1u);
		ASSERT_EQ(m.faceTexcoords.size(), m.faces.size() * 3);
		DistortStats s = ComputeStats(m);
		EXPECT_TRUE(s.allFinite) << "non-finite UVs produced";
	}
}

// ---------------------------------------------------------------------------
// 2. KEY ORACLE — planar chart flattens near-isometrically (≈0 distortion).
//    A flat grid is one chart; its symmetric-Dirichlet energy must be near the
//    perfect-isometry minimum (4 per triangle) and the UV/3D edge-length ratio
//    must be near-constant (low coefficient of variation). Both methods.
// ---------------------------------------------------------------------------
TEST(Flatten, PlanarIsometricOracle)
{
	for (auto method : {ParametrizeParams::FlattenMethod::ARAP,
	                    ParametrizeParams::FlattenMethod::SLIM}) {
		Mesh m = MakeFlatGrid(5, 5);
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeParams params;
		params.method = method;
		params.flattenIterations = 6;
		// Flatten the whole flat patch as a SINGLE explicit chart so the oracle
		// is unambiguous (segmentation may split a flat patch into >1 chart via
		// farthest-point seeding; each sub-chart is still flat, but we want one
		// chart here to assert global isometry directly).
		std::vector<unsigned> fc(m.faces.size(), 0u);
		ParametrizeCharts(m, fc, 1u, params);

		DistortStats s = ComputeStats(m);
		ASSERT_TRUE(s.allFinite);
		ASSERT_GT(s.triangles, 0);
		// perfect isometry => SD energy == 4 per triangle, edge-ratio cov == 0.
		EXPECT_LT(s.maxSdEnergy, 4.05)
		    << "flat patch not isometric (max SD energy too high), method="
		    << static_cast<int>(method);
		EXPECT_LT(s.avgSdEnergy, 4.01)
		    << "flat patch avg distortion too high";
		EXPECT_LT(s.edgeRatioCov, 0.02)
		    << "flat patch edge-length ratios not constant";
		EXPECT_EQ(s.flips, 0) << "flat patch produced flipped triangles";
	}
}

// ---------------------------------------------------------------------------
// 3. No flips. SLIM must be exactly flip-free on a clean patch; ARAP near 0.
// ---------------------------------------------------------------------------
TEST(Flatten, NoFlipsSLIM)
{
	Mesh m = MakeFlatGrid(6, 6);
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	Parametrize(m, params);
	DistortStats s = ComputeStats(m);
	EXPECT_TRUE(s.allFinite);
	EXPECT_EQ(s.flips, 0) << "SLIM must produce a flip-free (injective) map";
}

// ---------------------------------------------------------------------------
// 4. Developable strip flattens with low (bounded) distortion.
// ---------------------------------------------------------------------------
TEST(Flatten, DevelopableLowDistortion)
{
	Mesh m = MakeBentStrip(8, 4);
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	params.flattenIterations = 8;
	// treat the whole strip as a single chart (curvature is mild / one piece).
	std::vector<unsigned> fc;
	const unsigned n = SegmentCharts(m, params, fc);
	ASSERT_GE(n, 1u);
	ParametrizeCharts(m, fc, n, params);

	DistortStats s = ComputeStats(m);
	ASSERT_TRUE(s.allFinite);
	// developable → low (but not zero) distortion; SD energy modest above min 4.
	EXPECT_LT(s.avgSdEnergy, 4.5)
	    << "developable strip should flatten with low distortion";
	EXPECT_EQ(s.flips, 0) << "SLIM should keep the developable strip flip-free";
}

// ---------------------------------------------------------------------------
// 4b. Closed-form vs SVD symmetric-Dirichlet parity. SLIM's line-search energy
//     evaluates each triangle's symmetric-Dirichlet energy
//     from the Jacobian J directly, via s0^2+s1^2 = ||J||_F^2 and s0^2*s1^2 =
//     det(J)^2 (E = f + f/det^2), replacing a per-triangle JacobiSVD. This asserts
//     the closed form matches the SVD form to ulps on a genuinely flattened chart
//     — the numerical-equivalence bar for that substitution.
// ---------------------------------------------------------------------------
TEST(Flatten, ClosedFormEnergyMatchesSVD)
{
	Mesh m = MakeBentStrip(8, 4);
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	params.flattenIterations = 8;
	std::vector<unsigned> fc(m.faces.size(), 0u);
	ParametrizeCharts(m, fc, 1u, params);
	ASSERT_EQ(m.faceTexcoords.size(), m.faces.size() * 3);

	double maxRel = 0.0;
	int checked = 0;
	for (size_t f = 0; f < m.faces.size(); ++f) {
		Eigen::Vector3d p[3];
		Eigen::Vector2d u[3];
		for (int k = 0; k < 3; ++k) {
			p[k] = m.vertices[m.faces[f][k]].cast<double>();
			u[k] = m.faceTexcoords[f * 3 + k].cast<double>();
		}
		// Rebuild the same reference Jacobian the flattener uses.
		const Eigen::Vector3d e1 = p[1] - p[0], e2 = p[2] - p[0];
		const Eigen::Vector3d nrm = e1.cross(e2);
		const double l1 = e1.norm();
		if (l1 < 1e-12 || nrm.norm() < 1e-14)
			continue;
		const Eigen::Vector3d xax = e1 / l1;
		Eigen::Vector3d yax = nrm.cross(xax);
		yax /= yax.norm();
		Eigen::Matrix2d Xr;
		Xr.col(0) = Eigen::Vector2d(l1, 0.0);
		Xr.col(1) = Eigen::Vector2d(e2.dot(xax), e2.dot(yax));
		if (std::abs(Xr.determinant()) < 1e-18)
			continue;
		Eigen::Matrix2d Uu;
		Uu.col(0) = u[1] - u[0];
		Uu.col(1) = u[2] - u[0];
		const Eigen::Matrix2d J = Uu * Xr.inverse();
		// SVD form (old path).
		Eigen::JacobiSVD<Eigen::Matrix2d> svd(J);
		double s0 = svd.singularValues()[0], s1 = svd.singularValues()[1];
		s0 = std::max(s0, 1e-12);
		s1 = std::max(s1, 1e-12);
		const double eSvd = s0 * s0 + s1 * s1 + 1.0 / (s0 * s0) + 1.0 / (s1 * s1);
		// Closed form (new path).
		const double fF = J.squaredNorm();
		const double d = J.determinant();
		const double eCf = fF + fF / std::max(d * d, 1e-24);
		maxRel = std::max(maxRel, std::abs(eCf - eSvd) / std::max(eSvd, 1.0));
		++checked;
	}
	EXPECT_GT(checked, 0);
	EXPECT_LT(maxRel, 1e-9) << "closed-form symmetric-Dirichlet energy diverged from the SVD form";
}

// ---------------------------------------------------------------------------
// 5. ARAP on the developable strip: finite UVs, 0 flips, bounded distortion.
//    Coverage gap fix: previously ARAP flip assertions only ran on the flat
//    patch; this exercises ARAP on the genuinely curved (but developable)
//    fixture and asserts real numeric properties.
// ---------------------------------------------------------------------------
// Spherical cap of angular extent thmax (radians): non-developable, so a
// single-chart flatten is heavily distorted — the classic ARAP flip stressor.
static Mesh MakeSphericalCap(int nth, int nphi, float thmax)
{
	Mesh m;
	m.vertices.emplace_back(0.f, 0.f, 1.f); // pole
	for (int i = 1; i <= nth; ++i) {
		const float th = thmax * static_cast<float>(i) / static_cast<float>(nth);
		for (int j = 0; j < nphi; ++j) {
			const float ph = 2.f * static_cast<float>(M_PI) * static_cast<float>(j) / static_cast<float>(nphi);
			m.vertices.emplace_back(std::sin(th) * std::cos(ph),
			                        std::sin(th) * std::sin(ph), std::cos(th));
		}
	}
	auto ring = [&](int i, int j) {
		return static_cast<uint32_t>(1 + i * nphi + (j % nphi));
	};
	for (int j = 0; j < nphi; ++j)
		m.faces.emplace_back(0u, ring(0, j), ring(0, j + 1));
	for (int i = 0; i + 1 < nth; ++i)
		for (int j = 0; j < nphi; ++j) {
			m.faces.emplace_back(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1));
			m.faces.emplace_back(ring(i, j), ring(i + 1, j + 1), ring(i, j + 1));
		}
	return m;
}

// ---------------------------------------------------------------------------
// 5b. ARAP flip-free guarantee on an indefinite system: deterministic
// tangential jitter on a deep sliver-fan cap creates obtuse triangles whose
// negative cotangent weights make the ARAP Laplacian indefinite. The
// unguarded global solve folded ~17% of the triangles (159/912) starting
// from an injective Tutte init. ARAP must honor the same flip-free contract
// as SLIM — segmentation flip-repair (ChartFolds) assumes it for both.
// ---------------------------------------------------------------------------
TEST(Flatten, ARAPFlipFreeOnObtuseSlivers)
{
	Mesh m = MakeSphericalCap(10, 48, 2.9f);
	for (size_t i = 1; i < m.vertices.size(); ++i) {
		auto& v = m.vertices[i];
		v.x() += std::sin(37.f * static_cast<float>(i)) * 0.015f;
		v.y() += std::cos(57.f * static_cast<float>(i)) * 0.015f;
		v.normalize(); // stay on the sphere: tangential-ish jitter
	}
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::ARAP;
	params.initMethod = ParametrizeParams::InitMethod::Tutte;
	params.flattenIterations = 20;
	std::vector<unsigned> fc(m.faces.size(), 0u);
	ParametrizeCharts(m, fc, 1u, params);
	DistortStats s = ComputeStats(m);
	ASSERT_TRUE(s.allFinite);
	EXPECT_EQ(s.flips, 0)
	    << "ARAP shipped a folded map (" << s.flips << "/" << s.triangles
	    << " flipped) — the flip-free step guard is not enforced";
}

TEST(Flatten, DevelopableARAPNoFlips)
{
	Mesh m = MakeBentStrip(8, 4);
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::ARAP;
	params.flattenIterations = 10;
	std::vector<unsigned> fc;
	const unsigned n = SegmentCharts(m, params, fc);
	ASSERT_GE(n, 1u);
	ParametrizeCharts(m, fc, n, params);

	DistortStats s = ComputeStats(m);
	ASSERT_TRUE(s.allFinite) << "ARAP on developable strip produced non-finite UVs";
	// A cylinder strip is developable → ARAP should converge to a near-isometric
	// map, and with the flip-free step guard the shipped map is exactly fold-free.
	EXPECT_EQ(s.flips, 0)
	    << "ARAP produced flipped triangles on the developable strip ("
	    << s.flips << "/" << s.triangles << ")";
	// Symmetric-Dirichlet energy: a cylinder is developable so distortion should
	// stay modest (well below the arbitrary-mesh threshold).
	EXPECT_LT(s.avgSdEnergy, 6.0)
	    << "ARAP distortion unexpectedly large on a developable strip";
}

// ---------------------------------------------------------------------------
// LPT parallel scheduling determinism: per-chart tasks are enqueued
// largest-first over the thread pool, but each writes disjoint
// faceTexcoords slots and its computation is chart-local, so the atlas is a pure
// function of chart geometry. Two runs must be bitwise identical (float compare).
// ---------------------------------------------------------------------------
TEST(Flatten, ParallelFlattenDeterministic)
{
	Mesh m;
	if (!m.Load(TestMeshPath()))
		GTEST_SKIP() << "tests/data/mesh.ply not found";
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	std::vector<unsigned> fc;
	const unsigned n = SegmentCharts(m, params, fc);
	ASSERT_GE(n, 1u);

	Mesh a = m, b = m;
	ParametrizeCharts(a, fc, n, params);
	ParametrizeCharts(b, fc, n, params);
	ASSERT_EQ(a.faceTexcoords.size(), b.faceTexcoords.size());
	ASSERT_GT(a.faceTexcoords.size(), 0u);
	bool identical = true;
	for (size_t i = 0; i < a.faceTexcoords.size(); ++i)
		if (a.faceTexcoords[i].x() != b.faceTexcoords[i].x() || a.faceTexcoords[i].y() != b.faceTexcoords[i].y()) {
			identical = false;
			break;
		}
	EXPECT_TRUE(identical) << "parallel flatten produced run-to-run differences (nondeterministic scheduling leaked into output)";
}

// ---------------------------------------------------------------------------
// Shipped-float-precision flip-free contract. SLIM's line search rejects steps
// that fold only AFTER faceTexcoords are quantized to float, exempting
// input-degenerate (sliver) source triangles — the same float check ARAP
// enforces. Assert every NON-sliver triangle of the
// shipped (float-cast) map keeps positive signed area, for BOTH SLIM and ARAP
// (the parity the segmentation fold predicate's fast path assumes).
// ---------------------------------------------------------------------------
static int FloatFoldsNonSliver(const Mesh& m)
{
	const size_t nf = m.faces.size();
	std::vector<double> srcA(nf, 0.0);
	double totA = 0.0;
	for (size_t f = 0; f < nf; ++f) {
		const Eigen::Vector3d p0 = m.vertices[m.faces[f][0]].cast<double>();
		const Eigen::Vector3d p1 = m.vertices[m.faces[f][1]].cast<double>();
		const Eigen::Vector3d p2 = m.vertices[m.faces[f][2]].cast<double>();
		srcA[f] = 0.5 * (p1 - p0).cross(p2 - p0).norm();
		totA += srcA[f];
	}
	const double sliverA = (totA / std::max<size_t>(1, nf)) * 1e-6;
	int folds = 0;
	for (size_t f = 0; f < nf; ++f) {
		if (srcA[f] < sliverA)
			continue; // input-degenerate: orientation is float-quantization noise
		const Mesh::TexCoord& a = m.faceTexcoords[f * 3 + 0];
		const Mesh::TexCoord& b = m.faceTexcoords[f * 3 + 1];
		const Mesh::TexCoord& c = m.faceTexcoords[f * 3 + 2];
		const float sa = (b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y());
		if (sa <= 0.0f)
			++folds;
	}
	return folds;
}

TEST(Flatten, ShippedFloatFlipFreeSLIMandARAP)
{
	for (auto method : {ParametrizeParams::FlattenMethod::SLIM,
	                    ParametrizeParams::FlattenMethod::ARAP}) {
		Mesh m = MakeSphericalCap(10, 48, 2.9f);
		for (size_t i = 1; i < m.vertices.size(); ++i) {
			auto& v = m.vertices[i];
			v.x() += std::sin(37.f * static_cast<float>(i)) * 0.015f;
			v.y() += std::cos(57.f * static_cast<float>(i)) * 0.015f;
			v.normalize(); // tangential-ish jitter → obtuse slivers
		}
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeParams params;
		params.method = method;
		params.initMethod = ParametrizeParams::InitMethod::Tutte;
		params.flattenIterations = 20;
		std::vector<unsigned> fc(m.faces.size(), 0u);
		ParametrizeCharts(m, fc, 1u, params);
		ASSERT_EQ(m.faceTexcoords.size(), m.faces.size() * 3);
		EXPECT_EQ(FloatFoldsNonSliver(m), 0)
		    << "shipped float map folds a non-sliver triangle, method="
		    << static_cast<int>(method);
	}
}

// ---------------------------------------------------------------------------
// LSCM skips a degenerate triangle instead of aborting the chart. One injected
// zero-area triangle would otherwise make LscmInit return false for the WHOLE
// chart, silently dropping the leg to the Tutte-circle fallback. Two properties
// pin the behavior:
//   1. the converged map is near-isometric despite the sliver (the user-visible
//      guarantee — a lone degenerate triangle must not wreck the chart), and
//   2. the conformal init actually SHIPS: with flattenIterations=0 the two
//      init legs emit their RAW init maps, which are bitwise-identical iff the
//      LSCM leg silently fell back to Tutte (both then run the exact same code
//      on the same input). No numeric threshold — robust across build flags.
// (An earlier form asserted the converged Tutte-leg energy stayed 1.25x worse;
// that margin was an artifact of a SLIM line-search stall on the noise-oriented
// sliver — see SlimLineSearchNotVetoedByNoiseOrientedSliver — not of the init.)
// ---------------------------------------------------------------------------
static Mesh MakeBentStripWithDegenTri()
{
	// A long, thin, developable (bent) strip: LSCM unrolls it near-isometrically,
	// while the Tutte-circle init and the PCA planar fallback both distort the arc.
	const int nx = 24, ny = 2;
	Mesh m = MakeBentStrip(nx, ny, 4.0f, 3.0f);
	// Inject one exactly-degenerate triangle while KEEPING the chart a disk: collapse
	// the top-right corner vertex onto its neighbour below it (same arc position, so
	// they coincide). The last column's top face becomes zero-area (all corners
	// coincident); the end simply tapers to a point — no hole, no fold, still one
	// boundary loop. Without the skip, this lone zero-area triangle aborts LSCM for
	// the whole chart, dropping it to the distorting Tutte-circle fallback.
	const uint32_t corner = static_cast<uint32_t>(ny * (nx + 1) + nx); // idx(nx, ny)
	const uint32_t below = static_cast<uint32_t>((ny - 1) * (nx + 1) + nx); // idx(nx, ny-1)
	m.vertices[corner] = m.vertices[below];
	return m;
}

TEST(Flatten, LscmSkipsDegenerateTriangleNotWholeChart)
{
	const Mesh base = MakeBentStripWithDegenTri();
	auto flattenAvg = [&](ParametrizeParams::InitMethod im, unsigned iters) {
		Mesh m = base;
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeParams params;
		params.method = ParametrizeParams::FlattenMethod::SLIM;
		params.initMethod = im;
		params.flattenIterations = iters;
		std::vector<unsigned> fc(m.faces.size(), 0u);
		ParametrizeCharts(m, fc, 1u, params);
		return ComputeStats(m).avgSdEnergy; // slivers are ~zero-area, so ~unweighted
	};

	// 1. LSCM (default) must recover near-isometry despite the sliver.
	const double lscm = flattenAvg(ParametrizeParams::InitMethod::LSCM, 8);
	EXPECT_LT(lscm, 4.3) << "LSCM did not recover isometry — the sliver aborted it? avg=" << lscm;
	// 2. The conformal init must actually ship: at zero refine iterations the two
	// legs emit their raw inits, identical only if LSCM silently fell back to
	// Tutte (the raw conformal and circle maps differ grossly on this strip).
	const double lscmRaw = flattenAvg(ParametrizeParams::InitMethod::LSCM, 0);
	const double tutteRaw = flattenAvg(ParametrizeParams::InitMethod::Tutte, 0);
	EXPECT_NE(lscmRaw, tutteRaw)
	    << "raw LSCM-leg map identical to the Tutte-circle init — LscmInit "
	       "aborted on the sliver and fell back (energy="
	    << lscmRaw << ")";
}

// ---------------------------------------------------------------------------
// SLIM's flip-free line search must EXEMPT input-degenerate slivers, matching
// the policy of LscmInit (skips their rows), CountRealFlips (init acceptance)
// and its own shipped-precision guard: a zero-area triangle's UV orientation is
// unconstrained numerical noise, and when that noise lands negative, counting
// it as a real flip vetoes EVERY line-search candidate — SLIM stalls at the raw
// init map. The Tutte-circle init on the sliver strip reproduces the stall under
// any build flags (the sliver comes out negatively oriented), so refinement must
// strictly beat the init-only map by a wide margin.
// ---------------------------------------------------------------------------

TEST(Flatten, SlimLineSearchNotVetoedByNoiseOrientedSliver)
{
	const Mesh base = MakeBentStripWithDegenTri();
	auto convergedAvg = [&](unsigned iters) {
		Mesh m = base;
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeParams params;
		params.method = ParametrizeParams::FlattenMethod::SLIM;
		params.initMethod = ParametrizeParams::InitMethod::Tutte;
		params.flattenIterations = iters;
		std::vector<unsigned> fc(m.faces.size(), 0u);
		ParametrizeCharts(m, fc, 1u, params);
		return ComputeStats(m).avgSdEnergy;
	};
	const double initOnly = convergedAvg(0);
	const double refined = convergedAvg(8);
	EXPECT_LT(refined, 0.5 * initOnly)
	    << "SLIM made no real progress from the Tutte init — sliver vetoed the "
	       "line search? init="
	    << initOnly << " refined=" << refined;
}

// ---------------------------------------------------------------------------
// Cut-to-disk Euclidean-Dijkstra seams stay deterministic.
// An open tube has two boundary loops, so cutToDisk slits it into a disk via the
// FarthestVertex/ShortestCutEdges Dijkstra. The integer-quantized edge metric makes
// the cut a pure function of geometry: two runs must be bitwise identical.
// ---------------------------------------------------------------------------
static Mesh MakeOpenTube(int nth, int nz, float r, float h)
{
	Mesh m;
	auto idx = [&](int iz, int it) { return static_cast<uint32_t>(iz * nth + (it % nth)); };
	for (int iz = 0; iz <= nz; ++iz)
		for (int it = 0; it < nth; ++it) {
			const float a = 2.f * static_cast<float>(M_PI) * static_cast<float>(it) / static_cast<float>(nth);
			m.vertices.emplace_back(r * std::cos(a), r * std::sin(a), h * static_cast<float>(iz));
		}
	for (int iz = 0; iz < nz; ++iz)
		for (int it = 0; it < nth; ++it) {
			m.faces.emplace_back(idx(iz, it), idx(iz, it + 1), idx(iz + 1, it + 1));
			m.faces.emplace_back(idx(iz, it), idx(iz + 1, it + 1), idx(iz + 1, it));
		}
	return m;
}

TEST(Flatten, CutToDiskDijkstraDeterministic)
{
	const Mesh base = MakeOpenTube(24, 6, 1.0f, 0.5f);
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	params.cutToDisk = true;
	const std::vector<unsigned> fc(base.faces.size(), 0u); // whole tube = one chart

	auto flatten = [&]() {
		Mesh m = base;
		m.ListHalfEdges();
		m.ComputeFaceNormals();
		ParametrizeCharts(m, fc, 1u, params);
		return m.faceTexcoords;
	};
	const auto a = flatten();
	const auto b = flatten();
	ASSERT_EQ(a.size(), base.faces.size() * 3);
	bool identical = a.size() == b.size();
	for (size_t i = 0; identical && i < a.size(); ++i)
		if (a[i].x() != b[i].x() || a[i].y() != b[i].y())
			identical = false;
	EXPECT_TRUE(identical) << "cut-to-disk (Dijkstra seams) flatten is nondeterministic";
	bool allFinite = true;
	for (const auto& t : a)
		if (!std::isfinite(t.x()) || !std::isfinite(t.y()))
			allFinite = false;
	EXPECT_TRUE(allFinite) << "cut-to-disk produced non-finite UVs";
}

// ---------------------------------------------------------------------------
// Numerically stable flip-step quadratic. MaxStepNoFlip solves
// A2 t^2 + A1 t + A0 = 0 for the smallest positive root (the largest step
// keeping every triangle's area positive). The textbook (-A1 +/- sqrt)/(2 A2)
// form cancels catastrophically when the quadratic term is small (the common
// case) and can OVERESTIMATE that root, pushing a step past a fold. This mirrors
// the production formula and asserts, against a long-double reference, that the
// stable form never overestimates — and shows the naive form does.
// ---------------------------------------------------------------------------
template <typename T>
static T SmallestStepRoot(T A0, T A1, T A2, bool naive)
{
	T tmax = T(1);
	const T cmax = std::max(std::abs(A0), std::max(std::abs(A1), std::abs(A2)));
	const T ceps = T(1e-12) * cmax;
	const T reps = T(1e-12);
	if (std::abs(A2) <= ceps) {
		if (std::abs(A1) > ceps) {
			const T r = -A0 / A1;
			if (r > reps && r < tmax)
				tmax = r;
		}
		return tmax;
	}
	const T disc = A1 * A1 - T(4) * A2 * A0;
	if (disc < T(0))
		return tmax;
	const T sq = std::sqrt(disc);
	if (naive) {
		for (T r : {(-A1 - sq) / (T(2) * A2), (-A1 + sq) / (T(2) * A2)})
			if (r > reps && r < tmax)
				tmax = r;
	} else {
		const T q = T(-0.5) * (A1 + std::copysign(sq, A1));
		const T r0 = q / A2, r1 = (q != T(0)) ? A0 / q : r0;
		for (T r : {r0, r1})
			if (r > reps && r < tmax)
				tmax = r;
	}
	return tmax;
}

TEST(Flatten, MaxStepQuadraticStable)
{
	int checked = 0, stableOver = 0, naiveOver = 0;
	for (int se2 : {-1, 1})
		for (int e2 = 6; e2 <= 16; ++e2) // |A2| = 1e-6 .. 1e-16 (near-linear)
			for (int se1 : {-1, 1})
				for (int e1 = -4; e1 <= 4; e1 += 2)
					for (int e0 = -6; e0 <= 3; e0 += 3) {
						const double A2 = se2 * std::pow(10.0, -e2);
						const double A1 = se1 * std::pow(10.0, e1);
						const double A0 = std::pow(10.0, e0); // positive current area
						const long double ref = SmallestStepRoot<long double>(A0, A1, A2, false);
						const double s = SmallestStepRoot<double>(A0, A1, A2, false);
						const double nv = SmallestStepRoot<double>(A0, A1, A2, true);
						const double tol = 1e-9 * std::max(1.0, static_cast<double>(ref));
						if (s > static_cast<double>(ref) + tol)
							++stableOver;
						if (nv > static_cast<double>(ref) + tol)
							++naiveOver;
						++checked;
					}
	EXPECT_GT(checked, 0);
	EXPECT_EQ(stableOver, 0) << "stable form overestimated the flip-free step (would let a fold through)";
	// The naive form overestimates on a meaningful fraction of these near-linear
	// cases — the defect this finding fixes.
	RecordProperty("checked", checked);
	RecordProperty("naive_overestimates", naiveOver);
	EXPECT_GT(naiveOver, 0) << "expected the naive form to overestimate on some near-linear quadratic";
}

// ---------------------------------------------------------------------------
// 6. mesh.ply end-to-end: segment + flatten. Valid finite UVs, bounded average
//    distortion, and flip count reported / asserted per method.
// ---------------------------------------------------------------------------
TEST(Flatten, RealMeshEndToEnd)
{
	Mesh m;
	if (!m.Load(TestMeshPath()))
		GTEST_SKIP() << "tests/data/mesh.ply not found";
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ASSERT_GT(m.faces.size(), 0u);

	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	const unsigned n = Parametrize(m, params);
	EXPECT_GT(n, 0u);
	ASSERT_EQ(m.faceTexcoords.size(), m.faces.size() * 3);

	DistortStats s = ComputeStats(m);
	EXPECT_TRUE(s.allFinite) << "mesh.ply produced non-finite UVs";
	// bounded distortion: a reasonable atlas keeps the area-weighted mean modest.
	EXPECT_LT(s.avgSdEnergy, 50.0)
	    << "mesh.ply average distortion unexpectedly large";
	// report numbers for visibility.
	RecordProperty("charts", static_cast<int>(n));
	RecordProperty("triangles", s.triangles);
	RecordProperty("flips", s.flips);
	RecordProperty("avg_sd_energy_x1000", static_cast<int>(s.avgSdEnergy * 1000));
	RecordProperty("max_sd_energy_x1000", static_cast<int>(s.maxSdEnergy * 1000));
	// SLIM aims for flip-free; allow a small tolerance for closed/cut charts that
	// fall back to PCA projection (documented limitation).
	const double flipFrac = s.triangles > 0
	                            ? static_cast<double>(s.flips) / s.triangles
	                            : 0.0;
	EXPECT_LT(flipFrac, 0.02)
	    << "too many flipped triangles on mesh.ply (" << s.flips << "/"
	    << s.triangles << ")";
}

// ---------------------------------------------------------------------------
// Mean-value Tutte weights keep flip-freeness. Forcing the Tutte init on a
// curved chart must still yield an injective (flip-free) map —
// mean-value weights are strictly positive, so Floater's convex-boundary theorem
// still holds. (The distortion improvement over uniform weights is recorded in
// the task report; here we pin the guarantee that must never regress.)
// ---------------------------------------------------------------------------
TEST(Flatten, MeanValueTutteStaysFlipFree)
{
	Mesh m = MakeBentStrip(16, 4, 5.0f, 2.0f);
	m.ListHalfEdges();
	m.ComputeFaceNormals();
	ParametrizeParams params;
	params.method = ParametrizeParams::FlattenMethod::SLIM;
	params.initMethod = ParametrizeParams::InitMethod::Tutte;
	params.flattenIterations = 5;
	std::vector<unsigned> fc(m.faces.size(), 0u);
	ParametrizeCharts(m, fc, 1u, params);
	DistortStats s = ComputeStats(m);
	ASSERT_TRUE(s.allFinite) << "mean-value Tutte produced non-finite UVs";
	EXPECT_EQ(s.flips, 0) << "mean-value Tutte init lost the flip-free guarantee";
	EXPECT_LT(s.avgSdEnergy, 6.0) << "mean-value Tutte distortion unexpectedly large";
}

// ---------------------------------------------------------------------------
// Fold diagnosis (the carve's prerequisite): detail::ChartFacesFold's 5-arg
// overload must report WHICH faces made a chart fold, as global face ids
// (sorted, deduped), so a future repair can carve around them.
//
// SaddleFan (fan of `n` triangles around an interior apex with total apex
// angle `angleSum` > 2π, rim zig-zagging in z to embed the excess angle in
// 3D) now lives in tests/corpus/Corpus.h as hmtest::corpus::SaddleFan — it is
// shared with tests/ParametrizeTest.cpp's public-path equivalence test,
// so it moved to the repo's shared-test-helper library rather than being
// duplicated. A shipped (LSCM/Tutte + SLIM) flattening of this disk must
// either flip or globally self-overlap.
// ---------------------------------------------------------------------------

TEST(Flatten, FoldDiagnosisReportsOffendingFaces)
{
	Mesh mesh = hmtest::corpus::SaddleFan();
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> faces(mesh.faces.size());
	std::iota(faces.begin(), faces.end(), 0u);
	halfmesh::ParametrizeParams params;
	halfmesh::detail::FoldDiagnosis diag;
	const bool folds = halfmesh::detail::ChartFacesFold(mesh, faces, params, nullptr, &diag);
	// Report the measured fold extent that Corpus.h quotes as the fixture's
	// premise, so that number is observable here rather than a comment that can
	// silently rot.
	std::printf("[flatten_test] SaddleFan: folds=%d badFaces=%zu of %zu\n",
	            static_cast<int>(folds), diag.badFaces.size(), mesh.faces.size());
	// Fixture premise: the saddle fan folds under the shipped flatten. If this
	// ever fails, raise the zig-zag amplitude (SaddleFan's `elevation`) — do not
	// weaken the test.
	ASSERT_TRUE(folds);
	ASSERT_FALSE(diag.badFaces.empty());
	EXPECT_TRUE(std::is_sorted(diag.badFaces.begin(), diag.badFaces.end()));
	for (Mesh::FIndex f : diag.badFaces)
		EXPECT_LT(f, mesh.faces.size());
}

// ---------------------------------------------------------------------------
// Curvature-slit fold rescue: a chart that folds from enclosed
// curvature is re-flattened after cutting a slit from its worst interior
// vertex to the boundary, up to params.foldRescueSlits times — one chart with
// one extra seam instead of a split.
//
// Why a real mesh and not a synthetic fixture: the rescue needs an INTERIOR
// vertex to cut from, and no toy fixture produces a rescuable fold.
//   - SaddleFan has no interior vertex at all (its boundary loop walks
//     apex -> rim[1] -> ... -> rim[n+1] -> apex), so WorstInteriorVertex
//     returns -1 and the rescue is a guaranteed no-op.
//   - Closing that fan into a genuine cone does give an interior apex, but
//     ShortestCutEdges removes only ONE edge incident to the source, which
//     cannot split a fully-cyclic one-ring into two arcs — the cut moves the
//     apex to the boundary without duplicating it, reproducing the same
//     still-folding open-fan topology.
//   - Embedding a cone inside a larger disk (annulus, grid-with-bump,
//     multi-ring taper) flattens flip-free instead: LSCM pins the two
//     farthest-apart BOUNDARY vertices, so once the boundary is a few rings
//     away the solve absorbs the defect as smooth stretch rather than a fold.
// A rescuable fold — a localized bad vertex deep inside a much larger good
// region — only occurs on real, irregular geometry. So this drives the rescue
// on tests/data/mesh.ply's PRE-repair segmentation (developableFlipRepairRounds
// = 0, so folding charts survive to be tested; the shipped pipeline's own
// bisect-repair would otherwise have fixed them all). Deterministic (fixed
// corpus mesh + deterministic segmentation, SegmentQuality.SegmentDeterministicRunTwice),
// and measured to yield 133 folding charts of which the rescue fixes 3.
// ---------------------------------------------------------------------------
TEST(Flatten, FoldRescueSlitRescuesAtLeastOneRealMeshChart)
{
	Mesh mesh;
	if (!mesh.Load(TestMeshPath())) {
		GTEST_SKIP() << "tests/data/mesh.ply not found";
	}
	mesh.ListHalfEdges();
	halfmesh::ParametrizeParams segParams;
	segParams.developableFlipRepairRounds = 0; // keep genuinely-folding charts (no bisect-repair)
	segParams.postRepairMergeRounds = 0;
	std::vector<unsigned> faceChart;
	const unsigned numCharts = halfmesh::SegmentCharts(mesh, segParams, faceChart);
	std::vector<std::vector<Mesh::FIndex>> charts(numCharts);
	for (Mesh::FIndex f = 0; f < mesh.faces.size(); ++f)
		if (faceChart[f] < numCharts)
			charts[faceChart[f]].push_back(f);
	int numFoldOff = 0, numRescued = 0;
	for (unsigned c = 0; c < numCharts; ++c) {
		if (charts[c].size() <= 1)
			continue;
		halfmesh::ParametrizeParams off; // slits off → the pre-repair chart may fold
		if (!halfmesh::detail::ChartFacesFold(mesh, charts[c], off))
			continue;
		++numFoldOff;
		halfmesh::ParametrizeParams on;
		on.foldRescueSlits = 2;
		if (!halfmesh::detail::ChartFacesFold(mesh, charts[c], on))
			++numRescued;
	}
	// Fixture premise: pre-repair segmentation of mesh.ply has genuinely folding
	// charts (if this ever fails, the segmentation or repair defaults changed —
	// do not weaken this test, find another pre-repair fold source instead).
	ASSERT_GT(numFoldOff, 0);
	// The curvature-slit rescue turns at least one of them into a single
	// fold-free chart instead of leaving it to the split safety net.
	EXPECT_GT(numRescued, 0);
}

// ---------------------------------------------------------------------------
// A spike of height h over a unit ring of n vertices, skirted by a flat annulus
// out to radius R. The apex carries a severe angle deficit, so any flattening
// must stretch it — but the skirt keeps the apex INTERIOR and the boundary far
// away, so LSCM absorbs the deficit as smooth stretch rather than as a fold.
// That is the shape this test needs: injective, flip-free, and useless.
// ---------------------------------------------------------------------------
static Mesh MakeSpike(int n, float h, float R = 3.f)
{
	Mesh m;
	m.vertices.emplace_back(0.f, 0.f, h); // 0 = apex
	for (int r = 0; r < 2; ++r) {
		const float rad = (r == 0) ? 1.f : R;
		for (int i = 0; i < n; ++i) {
			const float a = 2.f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(n);
			m.vertices.emplace_back(rad * std::cos(a), rad * std::sin(a), 0.f);
		}
	}
	const auto V = [&](int r, int i) { return static_cast<uint32_t>(1 + r * n + (i % n)); };
	for (int i = 0; i < n; ++i)
		m.faces.emplace_back(0u, V(0, i), V(0, i + 1));
	for (int i = 0; i < n; ++i) {
		m.faces.emplace_back(V(0, i), V(1, i), V(1, i + 1));
		m.faces.emplace_back(V(0, i), V(1, i + 1), V(0, i + 1));
	}
	return m;
}

// ---------------------------------------------------------------------------
// Flip-freedom is not sufficient to ship a chart. A chart whose map is
// injective but stretched past any use must still be split, and with NO
// distortion budget configured — developableMaxUvDistortion defaults to 0, and
// before this bar existed that default meant "no distortion check at all", so
// such a chart shipped. Measured on a 471 814-face Ignatius at defaults, that
// path shipped 31 charts above the bar, the worst at symmetric-Dirichlet 3.3e8
// (a ~18 000x stretch), while the injectivity fallback ladder in
// ParametrizeCharts would have refused to SHIP anything above 200 — the two
// disagreed, and the repair's acceptance is the one that decides.
//
// The `1e9` arm is what makes this a distortion test rather than a fold test:
// the SAME chart ships when the budget is lifted, so its map is flip-free and
// non-self-overlapping and only the bar changes the verdict.
// ---------------------------------------------------------------------------
TEST(Flatten, OverStretchedChartSplitsWithNoDistortionBudgetSet)
{
	Mesh mesh = MakeSpike(24, 30.f);
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> faces(mesh.faces.size());
	for (size_t i = 0; i < faces.size(); ++i)
		faces[i] = static_cast<Mesh::FIndex>(i);

	halfmesh::ParametrizeParams def; // developableMaxUvDistortion == 0
	EXPECT_TRUE(halfmesh::detail::ChartFacesFold(mesh, faces, def))
	    << "an unusably stretched chart must be split even with no budget set";

	halfmesh::ParametrizeParams lifted;
	lifted.developableMaxUvDistortion = 1e9f;
	EXPECT_FALSE(halfmesh::detail::ChartFacesFold(mesh, faces, lifted))
	    << "the chart is flip-free and injective — only the distortion bar rejects it";

	halfmesh::ParametrizeParams tight;
	tight.developableMaxUvDistortion = 4.4f;
	EXPECT_TRUE(halfmesh::detail::ChartFacesFold(mesh, faces, tight))
	    << "an explicit budget must still be honoured";

	// The bar must not fire on a merely-curved chart: the same shape at h = 12
	// stretches, ships, and is only rejected by an explicitly tight budget.
	Mesh mild = MakeSpike(24, 12.f);
	mild.ListHalfEdges();
	std::vector<Mesh::FIndex> mildFaces(mild.faces.size());
	for (size_t i = 0; i < mildFaces.size(); ++i)
		mildFaces[i] = static_cast<Mesh::FIndex>(i);
	EXPECT_FALSE(halfmesh::detail::ChartFacesFold(mild, mildFaces, def))
	    << "the default bar must not split ordinary curved charts";
	EXPECT_TRUE(halfmesh::detail::ChartFacesFold(mild, mildFaces, tight));
}

} // namespace
} // namespace halfmesh

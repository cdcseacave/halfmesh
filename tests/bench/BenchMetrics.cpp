/*
* BenchMetrics.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "BenchMetrics.h"

#include "Metrics.h" // hmtest::metrics (shared toolkit)

#include <halfmesh/AtlasCharting.h> // NormalizeChartDensity (unit-scale normalization)

#include <algorithm>
#include <cmath>
#include <limits>

#include <sys/resource.h>

namespace hmbench {

using halfmesh::Mesh;

namespace {

// Closed-form singular values (s1 >= s2 >= 0) of a 2x2 matrix [[a,b],[c,d]].
inline void SingularValues2x2(double a, double b, double c, double d,
                              double& s1, double& s2)
{
	const double e = (a + d) * 0.5;
	const double f = (a - d) * 0.5;
	const double g = (c + b) * 0.5;
	const double h = (c - b) * 0.5;
	const double q = std::sqrt(e * e + h * h);
	const double r = std::sqrt(f * f + g * g);
	s1 = q + r;
	s2 = std::abs(q - r);
}

// Build the 2x2 Jacobian of the 3D->UV map for face fi.  Returns false on a
// degenerate (zero-area) 3D triangle.  Also returns the 3D area.
bool FaceJacobian(const Mesh& mesh, uint32_t fi,
                  double& a, double& b, double& c, double& d, double& worldArea)
{
	const auto& face = mesh.faces[fi];
	const Mesh::Vertex& P0 = mesh.vertices[face[0]];
	const Mesh::Vertex& P1 = mesh.vertices[face[1]];
	const Mesh::Vertex& P2 = mesh.vertices[face[2]];

	const Mesh::Vertex e1 = P1 - P0;
	const Mesh::Vertex e2 = P2 - P0;
	const double x1 = static_cast<double>(e1.norm());
	if (x1 < 1e-12)
		return false;
	const Mesh::Vertex basisX = e1 / static_cast<float>(x1);
	const double x2 = static_cast<double>(e2.dot(basisX));
	const Mesh::Vertex perp = e2 - basisX * static_cast<float>(x2);
	const double y2 = static_cast<double>(perp.norm());
	if (y2 < 1e-12)
		return false;

	const Mesh::TexCoord& Q0 = mesh.faceTexcoords[fi * 3 + 0];
	const Mesh::TexCoord& Q1 = mesh.faceTexcoords[fi * 3 + 1];
	const Mesh::TexCoord& Q2 = mesh.faceTexcoords[fi * 3 + 2];
	const double u1x = static_cast<double>(Q1.x() - Q0.x());
	const double u1y = static_cast<double>(Q1.y() - Q0.y());
	const double u2x = static_cast<double>(Q2.x() - Q0.x());
	const double u2y = static_cast<double>(Q2.y() - Q0.y());

	// M = U * inv(D), D = [[x1, x2],[0, y2]].
	a = u1x / x1;
	c = u1y / x1;
	b = (-u1x * x2 + u2x * x1) / (x1 * y2);
	d = (-u1y * x2 + u2y * x1) / (x1 * y2);
	worldArea = 0.5 * x1 * y2;
	return true;
}

inline bool HasPerCornerUV(const Mesh& mesh)
{
	return mesh.faceTexcoords.size() == mesh.faces.size() * 3 && !mesh.faces.empty();
}

} // namespace

// ---------------------------------------------------------------------------
double ComputeQuasiConformalDistortion(const Mesh& mesh)
{
	if (!HasPerCornerUV(mesh))
		return UNSET;
	double acc = 0.0, totalArea = 0.0;
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		double a, b, c, d, area;
		if (!FaceJacobian(mesh, f, a, b, c, d, area))
			continue;
		double s1, s2;
		SingularValues2x2(a, b, c, d, s1, s2);
		if (s2 < 1e-20)
			continue;
		acc += area * (s1 / s2);
		totalArea += area;
	}
	return (totalArea > 0.0) ? (acc / totalArea) : UNSET;
}

// ---------------------------------------------------------------------------
AreaDistortion ComputeAreaDistortion(const Mesh& mesh)
{
	AreaDistortion out;
	if (!HasPerCornerUV(mesh))
		return out;
	double totalUv = 0.0, totalWorld = 0.0;
	std::vector<double> det(mesh.faces.size(), 0.0);
	std::vector<char> ok(mesh.faces.size(), 0);
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		double a, b, c, d, area;
		if (!FaceJacobian(mesh, f, a, b, c, d, area))
			continue;
		const double detM = std::abs(a * d - b * c); // uvArea / worldArea
		det[f] = detM;
		ok[f] = 1;
		totalUv += detM * area;
		totalWorld += area;
	}
	if (totalWorld <= 0.0 || totalUv <= 0.0)
		return out;
	const double meanScale = totalUv / totalWorld; // global mean det
	// Robust ratio: 95th / 5th percentile of per-face relative area scale.
	// Plain max/min is dominated by single sliver/degenerate faces (can hit
	// 1e20 on a folded chart), making the metric uninterpretable.
	std::vector<double> rel;
	rel.reserve(mesh.faces.size());
	for (uint32_t f = 0; f < mesh.faces.size(); ++f)
		if (ok[f] && det[f] > 1e-20)
			rel.push_back(det[f] / meanScale);
	if (rel.size() < 4)
		return out;
	std::sort(rel.begin(), rel.end());
	auto pct = [&](double p) {
		const size_t idx = static_cast<size_t>(p * (rel.size() - 1));
		return rel[idx];
	};
	const double p5 = pct(0.05);
	const double p95 = pct(0.95);
	out.scaleMin = p5;
	out.scaleMax = p95;
	out.ratio = (p5 > 1e-20) ? (p95 / p5) : UNSET;
	return out;
}

// ---------------------------------------------------------------------------
void FillSegmentation(AtlasMetrics& m, const Mesh& mesh,
                      const std::vector<unsigned>& faceChart, unsigned numCharts)
{
	m.chartCount = numCharts;
	m.fullCoverage = ComputeChartCoverage(mesh, faceChart, numCharts);
	m.boundaryCutLength = ComputeBoundaryCutLength(mesh, faceChart);
	{
		const auto planar = ComputeChartPlanarityError(mesh, faceChart, numCharts);
		double acc = 0.0;
		unsigned n = 0;
		for (double p : planar)
			if (p != UNSET) {
				acc += p;
				++n;
			}
		m.meanPlanarityError = (n > 0) ? acc / n : UNSET;
	}
	{
		const auto compact = ComputeChartCompactness(mesh, faceChart, numCharts);
		double acc = 0.0;
		unsigned n = 0;
		for (double c : compact)
			if (c != UNSET) {
				acc += c;
				++n;
			}
		m.meanChartCompactness = (n > 0) ? acc / n : UNSET;
	}
}

// ---------------------------------------------------------------------------
void FillParametrization(AtlasMetrics& m, const Mesh& mesh,
                         const std::vector<unsigned>& faceChart, unsigned numCharts)
{
	if (!HasPerCornerUV(mesh)) {
		m.allFinite = false;
		return;
	}
	// Normalize a copy to unit world scale per chart so scale-sensitive energies
	// (symmetric-Dirichlet, Sander stretch) bottom out at their isometric minima.
	Mesh probe = mesh;
	if (faceChart.size() == mesh.faces.size() && numCharts > 0) {
		halfmesh::AtlasParams unit;
		unit.texelsPerUnit = 1.f; // per-chart uvArea == worldArea
		halfmesh::NormalizeChartDensity(probe, faceChart, numCharts, unit);
	}
	const hmtest::metrics::UVMetrics uv = hmtest::metrics::ComputeUVMetrics(probe);
	m.flipCount = uv.flipCount;
	m.symDirichlet = uv.symDirichlet;
	m.stretchL2 = uv.stretchL2;
	m.allFinite = uv.allFinite;
	m.quasiconformal = ComputeQuasiConformalDistortion(probe);
	m.areaDistortionRatio = ComputeAreaDistortion(probe).ratio;
}

// ---------------------------------------------------------------------------
void FillPacking(AtlasMetrics& m, const Mesh& mesh)
{
	if (!HasPerCornerUV(mesh))
		return;
	const hmtest::metrics::UVMetrics uv = hmtest::metrics::ComputeUVMetrics(mesh);
	m.occupancyTri = uv.atlasOccupancy;
	m.hasBboxOverlaps = uv.hasBboxOverlaps;
}

// ---------------------------------------------------------------------------
AtlasMetrics MeasureAtlas(const Mesh& mesh,
                          const std::vector<unsigned>& faceChart,
                          unsigned numCharts)
{
	AtlasMetrics m;
	FillSegmentation(m, mesh, faceChart, numCharts);
	FillParametrization(m, mesh, faceChart, numCharts);
	return m;
}

// ---------------------------------------------------------------------------
std::size_t PeakRssBytes()
{
	struct rusage ru;
	if (getrusage(RUSAGE_SELF, &ru) != 0)
		return 0;
#ifdef __APPLE__
	return static_cast<std::size_t>(ru.ru_maxrss); // bytes on macOS
#else
	return static_cast<std::size_t>(ru.ru_maxrss) * 1024; // KiB on Linux
#endif
}

} // namespace hmbench

/*
* ParamIsolation.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/ParamIsolation.h"
#include "BenchMetrics.h"

#include <Eigen/Eigenvalues>

#include <chrono>
#include <unordered_map>

namespace hmbench {

namespace {
inline double Now()
{
	using clk = std::chrono::steady_clock;
	return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
} // namespace

std::vector<ChartSubmesh> ExtractCharts(const halfmesh::Mesh& mesh,
                                        const std::vector<unsigned>& faceChart,
                                        unsigned numCharts)
{
	std::vector<ChartSubmesh> charts(numCharts);
	std::vector<std::unordered_map<uint32_t, int>> g2l(numCharts);
	for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
		const unsigned c = faceChart[f];
		if (c >= numCharts)
			continue;
		ChartSubmesh& sm = charts[c];
		auto& map = g2l[c];
		std::array<int, 3> lf{};
		for (int k = 0; k < 3; ++k) {
			const uint32_t gv = mesh.faces[f][k];
			auto it = map.find(gv);
			int local;
			if (it == map.end()) {
				local = static_cast<int>(sm.V.size());
				sm.V.push_back(mesh.vertices[gv].cast<double>());
				map.emplace(gv, local);
			} else {
				local = it->second;
			}
			lf[k] = local;
		}
		sm.F.push_back(lf);
		sm.globalFaces.push_back(f);
	}
	return charts;
}

bool PcaProject(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv)
{
	const size_t n = chart.V.size();
	if (n < 3)
		return false;
	Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
	for (const auto& v : chart.V)
		centroid += v;
	centroid /= static_cast<double>(n);
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (const auto& v : chart.V) {
		const Eigen::Vector3d d = v - centroid;
		cov += d * d.transpose();
	}
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
	// Largest two eigenvectors (columns sorted ascending → take last two).
	const Eigen::Vector3d axisU = es.eigenvectors().col(2);
	const Eigen::Vector3d axisV = es.eigenvectors().col(1);
	uv.resize(n);
	for (size_t i = 0; i < n; ++i) {
		const Eigen::Vector3d d = chart.V[i] - centroid;
		uv[i] = Eigen::Vector2d(d.dot(axisU), d.dot(axisV));
	}
	return true;
}

EngineResult RunPerChartFlatten(const std::string& name,
                                const halfmesh::Mesh& mesh,
                                const std::vector<unsigned>& faceChart,
                                unsigned numCharts,
                                const ChartFlattener& flatten)
{
	EngineResult r;
	r.engine = name;
	FillSegmentation(r.metrics, mesh, faceChart, numCharts); // shared partition

	const std::vector<ChartSubmesh> charts = ExtractCharts(mesh, faceChart, numCharts);

	halfmesh::Mesh out = mesh; // keep geometry + faces
	out.faceTexcoords.assign(mesh.faces.size() * 3, halfmesh::Mesh::TexCoord(0.f, 0.f));

	unsigned failures = 0;
	const double t0 = Now();
	for (const ChartSubmesh& chart : charts) {
		std::vector<Eigen::Vector2d> uv;
		bool ok = false;
		try {
			ok = flatten(chart, uv) && uv.size() == chart.V.size();
		} catch (...) {
			ok = false;
		}
		if (!ok) {
			++failures;
			uv.clear();
			if (!PcaProject(chart, uv))
				continue; // degenerate chart (<3 verts) — leave zero UVs
		}
		// Orientation normalization: a parametrization is free up to a global
		// reflection (an isometry the packer handles anyway), but the flip and
		// symmetric-Dirichlet metrics use a fixed CCW-positive convention.  Some
		// flatteners (notably BFF) pick the opposite global handedness per chart,
		// which would otherwise be miscounted as "every triangle flipped" with an
		// exploding energy.  Mirror the chart to net-CCW so every engine is
		// measured on the same convention (a no-op for engines already CCW).
		{
			double net2 = 0.0; // 2x net signed UV area over the chart
			for (const std::array<int, 3>& f : chart.F) {
				const Eigen::Vector2d& a = uv[f[0]];
				const Eigen::Vector2d& b = uv[f[1]];
				const Eigen::Vector2d& c = uv[f[2]];
				net2 += (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
			}
			if (net2 < 0.0)
				for (Eigen::Vector2d& p : uv)
					p.x() = -p.x();
		}
		for (size_t i = 0; i < chart.globalFaces.size(); ++i) {
			const uint32_t gf = chart.globalFaces[i];
			for (int k = 0; k < 3; ++k) {
				const int lv = chart.F[i][k];
				out.faceTexcoords[gf * 3 + k] = halfmesh::Mesh::TexCoord(
				    static_cast<float>(uv[lv].x()), static_cast<float>(uv[lv].y()));
			}
		}
	}
	r.parametrization.wallSeconds = Now() - t0;
	r.parametrization.completed = true;
	r.totalWallSeconds = r.parametrization.wallSeconds;

	FillParametrization(r.metrics, out, faceChart, numCharts);
	if (failures > 0)
		r.note = std::to_string(failures) + "/" + std::to_string(numCharts) + " chart(s) fell back to PCA";
	r.validOutput = r.metrics.allFinite && failures < numCharts;
	r.peakRssBytes = PeakRssBytes();
	return r;
}

} // namespace hmbench

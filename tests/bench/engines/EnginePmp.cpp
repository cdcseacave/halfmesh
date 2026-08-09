/*
* EnginePmp.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EnginePmp.h"
#include "engines/ParamIsolation.h"

#include <pmp/surface_mesh.h>
#include <pmp/algorithms/parameterization.h>

namespace hmbench {

namespace {

// Flatten one chart with pmp LSCM (harmonic=false) or discrete-harmonic.
// Throws on non-disk / non-manifold charts → driver applies the PCA fallback.
bool PmpFlatten(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv, bool harmonic)
{
	if (chart.V.size() < 3 || chart.F.empty())
		return false;
	pmp::SurfaceMesh m;
	std::vector<pmp::Vertex> vh(chart.V.size());
	for (size_t i = 0; i < chart.V.size(); ++i)
		vh[i] = m.add_vertex(pmp::Point(static_cast<pmp::Scalar>(chart.V[i].x()),
		                                static_cast<pmp::Scalar>(chart.V[i].y()),
		                                static_cast<pmp::Scalar>(chart.V[i].z())));
	for (const auto& f : chart.F)
		m.add_triangle(vh[f[0]], vh[f[1]], vh[f[2]]);

	if (harmonic)
		pmp::harmonic_parameterization(m);
	else
		pmp::lscm_parameterization(m);

	auto tex = m.get_vertex_property<pmp::TexCoord>("v:tex");
	if (!tex)
		return false;
	uv.resize(chart.V.size());
	for (size_t i = 0; i < chart.V.size(); ++i) {
		const pmp::TexCoord& t = tex[vh[i]];
		uv[i] = Eigen::Vector2d(static_cast<double>(t[0]), static_cast<double>(t[1]));
	}
	return true;
}

} // namespace

EngineResult RunPmpLscm(const halfmesh::Mesh& mesh,
                        const std::vector<unsigned>& faceChart,
                        unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten("pmp-lscm", mesh, faceChart, numCharts,
	                          [](const ChartSubmesh& c, std::vector<Eigen::Vector2d>& uv) {
		                          return PmpFlatten(c, uv, /*harmonic=*/false);
	                          });
}

EngineResult RunPmpHarmonic(const halfmesh::Mesh& mesh,
                            const std::vector<unsigned>& faceChart,
                            unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten("pmp-harmonic", mesh, faceChart, numCharts,
	                          [](const ChartSubmesh& c, std::vector<Eigen::Vector2d>& uv) {
		                          return PmpFlatten(c, uv, /*harmonic=*/true);
	                          });
}

} // namespace hmbench

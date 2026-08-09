/*
* EngineLibigl.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EngineLibigl.h"
#include "engines/ParamIsolation.h"

#include <igl/boundary_loop.h>
#include <igl/lscm.h>

#include <Eigen/Core>

namespace hmbench {

namespace {

// Flatten one chart with igl::lscm: pin two boundary vertices (opposite ends of
// the boundary loop) to (0,0) and (1,0).  Returns false if the chart has no
// usable boundary loop or lscm fails → driver applies the PCA fallback.
bool IglLscmFlatten(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv)
{
	const int nv = static_cast<int>(chart.V.size());
	const int nf = static_cast<int>(chart.F.size());
	if (nv < 3 || nf < 1)
		return false;

	Eigen::MatrixXd V(nv, 3);
	for (int i = 0; i < nv; ++i)
		V.row(i) = chart.V[i].transpose();
	Eigen::MatrixXi F(nf, 3);
	for (int i = 0; i < nf; ++i)
		for (int k = 0; k < 3; ++k)
			F(i, k) = chart.F[i][k];

	Eigen::VectorXi bnd;
	igl::boundary_loop(F, bnd);
	if (bnd.size() < 2)
		return false;

	Eigen::VectorXi b(2);
	b(0) = bnd(0);
	b(1) = bnd(static_cast<int>(bnd.size()) / 2);
	Eigen::MatrixXd bc(2, 2);
	bc << 0.0, 0.0, 1.0, 0.0;

	Eigen::MatrixXd V_uv;
	if (!igl::lscm(V, F, b, bc, V_uv))
		return false;
	if (V_uv.rows() != nv || V_uv.cols() != 2)
		return false;

	uv.resize(nv);
	for (int i = 0; i < nv; ++i)
		uv[i] = Eigen::Vector2d(V_uv(i, 0), V_uv(i, 1));
	return true;
}

} // namespace

EngineResult RunLibiglLscm(const halfmesh::Mesh& mesh,
                           const std::vector<unsigned>& faceChart,
                           unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten("libigl-lscm", mesh, faceChart, numCharts,
	                          IglLscmFlatten);
}

} // namespace hmbench

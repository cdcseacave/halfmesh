/*
* EngineBff.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EngineBff.h"
#include "engines/ParamIsolation.h"

#include "geometrycentral/surface/boundary_first_flattening.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace hmbench {

namespace {

namespace gc = geometrycentral;
namespace gcs = geometrycentral::surface;

// Flatten one chart with Boundary First Flattening (geometry-central's
// parameterizeBFF, in boundary_first_flattening.h).  BFF needs a manifold
// topological disk; a closed / multiply-connected / non-manifold chart fails the
// boundary check (or throws) → false → PCA fallback.
bool BffFlatten(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv)
{
	if (chart.V.size() < 3 || chart.F.empty())
		return false;

	std::vector<std::vector<std::size_t>> polygons;
	polygons.reserve(chart.F.size());
	for (const std::array<int, 3>& f : chart.F)
		polygons.push_back({static_cast<std::size_t>(f[0]),
		                    static_cast<std::size_t>(f[1]),
		                    static_cast<std::size_t>(f[2])});
	std::vector<gc::Vector3> positions;
	positions.reserve(chart.V.size());
	for (const Eigen::Vector3d& v : chart.V)
		positions.push_back(gc::Vector3{v.x(), v.y(), v.z()});

	try {
		std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
		std::unique_ptr<gcs::VertexPositionGeometry> geom;
		std::tie(mesh, geom) =
		    gcs::makeManifoldSurfaceMeshAndGeometry(polygons, positions);
		if (mesh->nBoundaryLoops() != 1)
			return false; // BFF requires a single boundary (topological disk)

		gcs::VertexData<gc::Vector2> flat = gcs::parameterizeBFF(*mesh, *geom);

		uv.assign(chart.V.size(), Eigen::Vector2d::Zero());
		for (gcs::Vertex v : mesh->vertices()) {
			const std::size_t i = v.getIndex();
			if (i < uv.size())
				uv[i] = Eigen::Vector2d(flat[v].x, flat[v].y);
		}
		return true;
	} catch (const std::exception&) {
		return false;
	}
}

} // namespace

EngineResult RunBffParam(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten("bff", mesh, faceChart, numCharts, BffFlatten);
}

} // namespace hmbench

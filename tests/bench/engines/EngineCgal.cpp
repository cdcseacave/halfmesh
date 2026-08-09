/*
* EngineCgal.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EngineCgal.h"
#include "engines/ParamIsolation.h"
#include "BenchMetrics.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_parameterization/LSCM_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Surface_mesh_parameterization/Error_code.h>
#include <CGAL/Polygon_mesh_processing/measure.h> // longest_border
#include <CGAL/mesh_segmentation.h>

#include <boost/graph/graph_traits.hpp>

#include <chrono>
#include <vector>

namespace hmbench {

namespace {

namespace SMP = CGAL::Surface_mesh_parameterization;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;
using SMesh = CGAL::Surface_mesh<Point_3>;
using vertex_descriptor = boost::graph_traits<SMesh>::vertex_descriptor;
using halfedge_descriptor = boost::graph_traits<SMesh>::halfedge_descriptor;
using face_descriptor = boost::graph_traits<SMesh>::face_descriptor;

inline double Now()
{
	return std::chrono::duration<double>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// Build a CGAL Surface_mesh from a chart submesh; vds[i] maps chart.V[i] to its
// CGAL vertex.  Returns false if any face is rejected (would be non-manifold).
bool BuildSurfaceMesh(const ChartSubmesh& chart, SMesh& sm,
                      std::vector<vertex_descriptor>& vds)
{
	vds.resize(chart.V.size());
	for (std::size_t i = 0; i < chart.V.size(); ++i)
		vds[i] = sm.add_vertex(Point_3(chart.V[i].x(), chart.V[i].y(), chart.V[i].z()));
	for (const std::array<int, 3>& f : chart.F) {
		const face_descriptor fd = sm.add_face(vds[f[0]], vds[f[1]], vds[f[2]]);
		if (fd == SMesh::null_face())
			return false;
	}
	return true;
}

// Flatten one chart with a CGAL parameterizer (LSCM / ARAP); returns false to
// trigger the harness PCA fallback (closed chart, non-manifold, or solver fail).
template <class Parameterizer>
bool CgalFlatten(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv)
{
	if (chart.V.size() < 3 || chart.F.empty())
		return false;
	SMesh sm;
	std::vector<vertex_descriptor> vds;
	if (!BuildSurfaceMesh(chart, sm, vds))
		return false;

	const halfedge_descriptor bhd =
	    CGAL::Polygon_mesh_processing::longest_border(sm).first;
	if (bhd == boost::graph_traits<SMesh>::null_halfedge())
		return false; // no boundary loop (closed chart)

	SMesh::Property_map<vertex_descriptor, Point_2> uvmap =
	    sm.add_property_map<vertex_descriptor, Point_2>("v:uv", Point_2(0, 0)).first;

	Parameterizer parameterizer;
	if (SMP::parameterize(sm, parameterizer, bhd, uvmap) != SMP::OK)
		return false;

	uv.resize(chart.V.size());
	for (std::size_t i = 0; i < chart.V.size(); ++i) {
		const Point_2& p = uvmap[vds[i]];
		uv[i] = Eigen::Vector2d(p.x(), p.y());
	}
	return true;
}

} // namespace

EngineResult RunCgalLscm(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten(
	    "cgal-lscm", mesh, faceChart, numCharts,
	    [](const ChartSubmesh& c, std::vector<Eigen::Vector2d>& uv) {
		    return CgalFlatten<SMP::LSCM_parameterizer_3<SMesh>>(c, uv);
	    });
}

EngineResult RunCgalArap(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& /*cfg*/)
{
	return RunPerChartFlatten(
	    "cgal-arap", mesh, faceChart, numCharts,
	    [](const ChartSubmesh& c, std::vector<Eigen::Vector2d>& uv) {
		    return CgalFlatten<SMP::ARAP_parameterizer_3<SMesh>>(c, uv);
	    });
}

// ---------------------------------------------------------------------------
// CGAL SDF-graph-cut segmentation (segmentation metrics only).
// ---------------------------------------------------------------------------
EngineResult RunCgalSegment(const halfmesh::Mesh& mesh, const BenchConfig& /*cfg*/)
{
	EngineResult r;
	r.engine = "cgal-sdf";

	const double t0 = Now();
	SMesh sm;
	std::vector<vertex_descriptor> vds(mesh.vertices.size());
	for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
		const auto& v = mesh.vertices[i];
		vds[i] = sm.add_vertex(Point_3(v.x(), v.y(), v.z()));
	}
	std::vector<face_descriptor> fds(mesh.faces.size(), SMesh::null_face());
	std::size_t added = 0;
	for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
		const auto& face = mesh.faces[f];
		fds[f] = sm.add_face(vds[face[0]], vds[face[1]], vds[face[2]]);
		if (fds[f] != SMesh::null_face())
			++added;
	}
	if (added == 0) {
		r.note = "CGAL rejected all faces (non-manifold input; repair first)";
		return r;
	}

	// SDF values per face, then graph-cut segmentation.
	SMesh::Property_map<face_descriptor, double> sdf =
	    sm.add_property_map<face_descriptor, double>("f:sdf", 0.0).first;
	CGAL::sdf_values(sm, sdf);
	SMesh::Property_map<face_descriptor, std::size_t> seg =
	    sm.add_property_map<face_descriptor, std::size_t>("f:sid", 0).first;
	// number_of_clusters drives how finely SDF space is quantised; smoothing_lambda
	// trades boundary smoothness for SDF fidelity (CGAL defaults: 5 / 0.26).
	const std::size_t numSegments =
	    CGAL::segmentation_from_sdf_values(sm, sdf, seg, /*clusters=*/8, /*lambda=*/0.3);
	r.segmentation.wallSeconds = Now() - t0;
	r.segmentation.completed = true;

	std::vector<unsigned> faceChart(mesh.faces.size(), 0);
	for (std::size_t f = 0; f < mesh.faces.size(); ++f)
		if (fds[f] != SMesh::null_face())
			faceChart[f] = static_cast<unsigned>(seg[fds[f]]);

	FillSegmentation(r.metrics, mesh, faceChart, static_cast<unsigned>(numSegments));
	r.totalWallSeconds = r.segmentation.wallSeconds;
	r.validOutput = r.metrics.chartCount > 0;
	r.peakRssBytes = PeakRssBytes();
	return r;
}

} // namespace hmbench

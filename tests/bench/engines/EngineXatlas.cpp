/*
* EngineXatlas.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EngineXatlas.h"
#include "BenchMetrics.h"

#include "vendor/xatlas/xatlas.h"

#include <chrono>
#include <cstdint>
#include <vector>

namespace hmbench {

namespace {
inline double Now()
{
	using clk = std::chrono::steady_clock;
	return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
} // namespace

EngineResult RunXatlas(const halfmesh::Mesh& mesh, const BenchConfig& cfg)
{
	EngineResult r;
	r.engine = "xatlas";

	const uint32_t nv = static_cast<uint32_t>(mesh.vertices.size());
	const uint32_t nf = static_cast<uint32_t>(mesh.faces.size());
	if (nv == 0 || nf == 0) {
		r.note = "empty mesh";
		return r;
	}

	// Contiguous position + index buffers (avoid assuming Eigen storage layout).
	std::vector<float> positions(static_cast<size_t>(nv) * 3);
	for (uint32_t v = 0; v < nv; ++v) {
		positions[v * 3 + 0] = mesh.vertices[v].x();
		positions[v * 3 + 1] = mesh.vertices[v].y();
		positions[v * 3 + 2] = mesh.vertices[v].z();
	}
	std::vector<uint32_t> indices(static_cast<size_t>(nf) * 3);
	for (uint32_t f = 0; f < nf; ++f)
		for (int c = 0; c < 3; ++c)
			indices[f * 3 + c] = mesh.faces[f][c];

	xatlas::MeshDecl md;
	md.vertexCount = nv;
	md.vertexPositionData = positions.data();
	md.vertexPositionStride = sizeof(float) * 3;
	md.indexCount = nf * 3;
	md.indexData = indices.data();
	md.indexFormat = xatlas::IndexFormat::UInt32;

	xatlas::Atlas* atlas = xatlas::Create();
	const xatlas::AddMeshError err = xatlas::AddMesh(atlas, md);
	if (err != xatlas::AddMeshError::Success) {
		r.note = std::string("AddMesh failed: ") + xatlas::StringForEnum(err);
		xatlas::Destroy(atlas);
		return r;
	}
	xatlas::AddMeshJoin(atlas);

	xatlas::ChartOptions co; // out-of-the-box defaults
	xatlas::PackOptions po;
	po.resolution = cfg.resolution;
	po.padding = 2;
	po.bruteForce = false;
	po.createImage = false;

	// ComputeCharts fuses charting + LSCM parametrization; time it as the
	// "segmentation" stage and note the fusion.  PackCharts is the pack stage.
	{
		const double t0 = Now();
		xatlas::ComputeCharts(atlas, co);
		r.segmentation.wallSeconds = Now() - t0;
		r.segmentation.completed = true;
	}
	{
		const double t0 = Now();
		xatlas::PackCharts(atlas, po);
		r.packing.wallSeconds = Now() - t0;
		r.packing.completed = true;
	}
	r.parametrization.completed = false; // fused into ComputeCharts
	r.note = "charting+param fused in ComputeCharts";
	r.totalWallSeconds = r.segmentation.wallSeconds + r.packing.wallSeconds;

	if (atlas->meshCount == 0) {
		r.note = "no output mesh";
		xatlas::Destroy(atlas);
		return r;
	}
	const xatlas::Mesh& om = atlas->meshes[0];
	const uint32_t outNf = om.indexCount / 3;
	const double W = atlas->width > 0 ? static_cast<double>(atlas->width) : 1.0;
	const double H = atlas->height > 0 ? static_cast<double>(atlas->height) : 1.0;

	// Reconstruct a halfmesh::Mesh: original vertex positions, faces use xref so
	// cross-seam edges share original vertex ids (seam metrics see them).
	halfmesh::Mesh out;
	out.vertices = mesh.vertices;
	out.faces.resize(outNf);
	out.faceTexcoords.resize(static_cast<size_t>(outNf) * 3);
	for (uint32_t fi = 0; fi < outNf; ++fi) {
		for (int c = 0; c < 3; ++c) {
			const uint32_t ov = om.indexArray[fi * 3 + c];
			const xatlas::Vertex& vv = om.vertexArray[ov];
			out.faces[fi][c] = vv.xref;
			out.faceTexcoords[fi * 3 + c] = halfmesh::Mesh::TexCoord(
			    static_cast<float>(vv.uv[0] / W),
			    static_cast<float>(vv.uv[1] / H));
		}
	}

	// Per-face chart labels (global chart id over all charts in the mesh).
	std::vector<unsigned> faceChart(outNf, 0);
	for (uint32_t c = 0; c < om.chartCount; ++c) {
		const xatlas::Chart& chart = om.chartArray[c];
		for (uint32_t k = 0; k < chart.faceCount; ++k)
			if (chart.faceArray[k] < outNf)
				faceChart[chart.faceArray[k]] = c;
	}
	const unsigned numCharts = om.chartCount;

	FillSegmentation(r.metrics, out, faceChart, numCharts);
	FillParametrization(r.metrics, out, faceChart, numCharts);
	FillPacking(r.metrics, out); // triangle occupancy (note: per-page [0,1])

	// xatlas reports its own texel utilization (rectangle fill, incl. padding).
	const unsigned pages = atlas->atlasCount > 0 ? atlas->atlasCount : 1;
	if (atlas->utilization && atlas->atlasCount > 0) {
		double util = 0.0;
		for (uint32_t i = 0; i < atlas->atlasCount; ++i)
			util += atlas->utilization[i];
		r.metrics.occupancyRect = util / atlas->atlasCount;
	}
	r.metrics.numPages = pages;

	r.validOutput = r.metrics.allFinite && r.metrics.chartCount > 0;
	r.peakRssBytes = PeakRssBytes();
	xatlas::Destroy(atlas);
	return r;
}

} // namespace hmbench

/*
* Unwrap.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// examples/Unwrap.cpp — UV atlas generation CLI example.
//
// Usage: unwrap <in.ply> <out.ply> [resolution=1024]
//
// Loads a triangle mesh, generates a UV texture atlas (chart segmentation →
// per-chart SLIM flattening → density normalisation → rectangle packing),
// and saves the result with packed UV coordinates in faceTexcoords.
// Demonstrates the public API:
//   halfmesh::Mesh::Load / halfmesh::GenerateAtlas / Mesh::Save
#include <halfmesh/Mesh.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
	if (argc < 3 || argc > 4) {
		std::cerr << "Usage: unwrap <in.ply> <out.ply> [resolution=1024]\n"
		          << "  resolution  target atlas side length in texels\n";
		return EXIT_FAILURE;
	}

	const std::string inPath = argv[1];
	const std::string outPath = argv[2];
	const unsigned resolution = (argc == 4)
	                                ? static_cast<unsigned>(std::stoul(argv[3]))
	                                : 1024u;

	halfmesh::Mesh mesh;
	if (!mesh.Load(inPath)) {
		std::cerr << "Error: failed to load '" << inPath << "'\n";
		return EXIT_FAILURE;
	}

	std::cout << "Loaded:  " << mesh.vertices.size() << " vertices, "
	          << mesh.faces.size() << " faces\n";

	// Weld coincident vertices and drop degenerate/orphaned geometry first.
	// Unwelded input (glTF always, some PLYs) leaves every edge a boundary, so
	// SegmentCharts fragments into one chart per face.  Exact weld (epsilon 0) +
	// topology-only degenerate cull (thArea 0, so valid near-zero-area slivers are
	// kept) is lossless: the atlas regenerates the UVs anyway.
	mesh.RemoveDuplicateVertices(0);
	mesh.RemoveDegenerateFaces(0.f);
	mesh.RemoveUnreferencedVertices();

	// Run the full atlas pipeline: SegmentCharts + ParametrizeCharts +
	// NormalizeChartDensity + PackAtlas.
	halfmesh::ParametrizeParams pparams;
	// Default segmentation / flattening settings work well for most meshes.

	halfmesh::AtlasParams aparams;
	aparams.resolution = resolution;
	aparams.padding = 2;
	aparams.allowRotation = true;

	const halfmesh::AtlasResult result =
	    halfmesh::GenerateAtlas(mesh, pparams, aparams);

	const unsigned numCharts =
	    static_cast<unsigned>(result.chartPage.size());

	std::cout << "Atlas:   " << numCharts << " charts, "
	          << result.numPages << " page(s), "
	          << result.width << "x" << result.height << " texels, "
	          << (result.occupancy * 100.f) << "% occupancy\n";

	if (!mesh.Save(outPath)) {
		std::cerr << "Error: failed to save '" << outPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Saved:   " << outPath << "\n";
	return EXIT_SUCCESS;
}

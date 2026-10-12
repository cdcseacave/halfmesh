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
// Usage: unwrap <in.ply> <out.ply> [resolution=1024] [padding=2]
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
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
	if (argc < 3 || argc > 5) {
		std::cerr << "Usage: unwrap <in.ply> <out.ply> [resolution=1024] [padding=2]\n"
		          << "  resolution  target atlas side length in texels\n"
		          << "  padding     texel gap between packed charts\n";
		return EXIT_FAILURE;
	}

	const std::string inPath = argv[1];
	const std::string outPath = argv[2];
	long resolution = 1024;
	long padding = 2;
	try {
		if (argc >= 4)
			resolution = std::stol(argv[3]);
		if (argc == 5)
			padding = std::stol(argv[4]);
	} catch (const std::exception&) {
		std::cerr << "Error: resolution and padding must be integers\n";
		return EXIT_FAILURE;
	}
	if (resolution <= 0 || padding < 0 || 2 * padding >= resolution) {
		std::cerr << "Error: need resolution > 0 and 0 <= 2*padding < resolution, or no chart fits the page\n";
		return EXIT_FAILURE;
	}

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
	aparams.padding = padding;
	aparams.allowRotation = true;

	const halfmesh::AtlasResult result =
	    halfmesh::GenerateAtlas(mesh, pparams, aparams);

	const unsigned numCharts =
	    static_cast<unsigned>(result.chartPage.size());

	std::cout << "Atlas:   " << numCharts << " charts, "
	          << result.numPages << " page(s), "
	          << result.width << "x" << result.height << " texels, "
	          << (result.occupancy * 100.f) << "% occupancy, "
	          << (result.coverage * 100.f) << "% coverage\n";

	if (!mesh.Save(outPath)) {
		std::cerr << "Error: failed to save '" << outPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Saved:   " << outPath << "\n";
	return EXIT_SUCCESS;
}

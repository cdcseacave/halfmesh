/*
* Decimate.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// examples/Decimate.cpp — QEM decimation CLI example.
//
// Usage: decimate <in.ply> <out.ply> [target=0.5] [aggressiveness=0]
//
// Loads a triangle mesh, runs QEM edge-collapse simplification down to the
// requested target, and saves the result.  Demonstrates the public API:
//   halfmesh::Mesh::Load / Simplify / Save
//
// `target` is dual-purpose by magnitude: a value in (0,1] is a ratio of the
// input face count (0.5 = half), while a value > 1 is an absolute target face
// count (300000 = decimate to ~300k faces).  `aggressiveness` selects the
// strategy: 0 is exact QEM (a global priority queue — best quality but heavy on
// huge reductions), while a positive value (e.g. 7) is the fast threshold sweep.
#include <halfmesh/Mesh.h>

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
	if (argc < 3 || argc > 5) {
		std::cerr << "Usage: decimate <in.ply> <out.ply> [target=0.5] [aggressiveness=0]\n"
		          << "  target          (0,1] -> fraction of input faces (0.5 = half)\n"
		          << "                  >1    -> absolute target face count (e.g. 300000)\n"
		          << "  aggressiveness  0 -> exact QEM (best quality); >0 (e.g. 7) -> fast\n"
		          << "                  threshold sweep, recommended for large reductions\n";
		return EXIT_FAILURE;
	}

	const std::string inPath = argv[1];
	const std::string outPath = argv[2];
	const double target = (argc >= 4) ? std::stod(argv[3]) : 0.5;
	const float aggressiveness = (argc == 5) ? std::stof(argv[4]) : 0.f;

	if (target <= 0.0) {
		std::cerr << "Error: target must be > 0 (ratio in (0,1], or absolute face count > 1)\n";
		return EXIT_FAILURE;
	}
	if (aggressiveness < 0.f) {
		std::cerr << "Error: aggressiveness must be >= 0\n";
		return EXIT_FAILURE;
	}

	halfmesh::Mesh mesh;
	if (!mesh.Load(inPath)) {
		std::cerr << "Error: failed to load '" << inPath << "'\n";
		return EXIT_FAILURE;
	}

	const auto facesBefore = static_cast<unsigned>(mesh.faces.size());
	const auto vertsBefore = static_cast<unsigned>(mesh.vertices.size());
	std::cout << "Loaded:  " << vertsBefore << " vertices, "
	          << facesBefore << " faces\n";

	// Weld coincident vertices and drop degenerate/orphaned geometry first.
	// Unwelded input (glTF always, some PLYs) makes every edge a boundary, so
	// Simplify pins x3-weight silhouette quadrics everywhere and barely collapses.
	// Exact weld (epsilon 0) + topology-only degenerate cull (thArea 0, so valid
	// near-zero-area slivers are kept) is lossless for geometry-only meshes.
	mesh.RemoveDuplicateVertices(0);
	mesh.RemoveDegenerateFaces(0.f);
	mesh.RemoveUnreferencedVertices();

	// QEM edge-collapse decimation; Simplify reads `target` by magnitude:
	// (0,1) -> fraction of input faces, >1 -> absolute target face count.
	mesh.Simplify(static_cast<float>(target), /*minEdgeLength=*/0.f, aggressiveness);

	const auto facesAfter = static_cast<unsigned>(mesh.faces.size());
	const auto vertsAfter = static_cast<unsigned>(mesh.vertices.size());
	std::cout << "Decimated: " << vertsAfter << " vertices, "
	          << facesAfter << " faces"
	          << " (kept " << (100.f * facesAfter / facesBefore) << "% of faces)\n";

	if (!mesh.Save(outPath)) {
		std::cerr << "Error: failed to save '" << outPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Saved:   " << outPath << "\n";
	return EXIT_SUCCESS;
}

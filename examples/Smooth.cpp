/*
* Smooth.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// examples/Smooth.cpp — mesh smoothing CLI example.
//
// Usage: smooth <in.ply|in.glb> <out.ply|out.glb> [iterations] [algo]
//   algo: hc     = HC Laplacian, Vollmer'99 anti-shrink update (default)
//         taubin = Taubin'95 lambda|mu band-pass (aggressive, ~zero shrink;
//                  wants more iterations than HC, e.g. 10-100)
//
// Loads a triangle mesh, smooths it, and saves the result.  Prints
// before/after vertex count and two shrinkage indicators (bbox diagonal
// ratio, mean distance-to-centroid ratio) so the methods can be compared
// at a glance.  Demonstrates the public API:
//   halfmesh::Mesh::Load / ComputeAABBox / Smooth (unified dispatcher) / Save
#include <halfmesh/Mesh.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double MeanRadius(const std::vector<halfmesh::Mesh::Vertex>& verts, const halfmesh::Mesh::Vertex& c)
{
	double sum = 0;
	for (const halfmesh::Mesh::Vertex& v : verts)
		sum += static_cast<double>((v - c).norm());
	return verts.empty() ? 0.0 : sum / static_cast<double>(verts.size());
}

halfmesh::Mesh::Vertex Centroid(const std::vector<halfmesh::Mesh::Vertex>& verts)
{
	halfmesh::Mesh::Vertex c(halfmesh::Mesh::Vertex::Zero());
	for (const halfmesh::Mesh::Vertex& v : verts)
		c += v;
	return verts.empty() ? c : halfmesh::Mesh::Vertex(c / static_cast<float>(verts.size()));
}

} // namespace

int main(int argc, char* argv[])
{
	if (argc < 3 || argc > 5) {
		std::cerr << "Usage: smooth <in> <out> [iterations] [algo]\n"
		          << "  iterations  number of smoothing passes (default: 5)\n"
		          << "  algo        hc     = HC Laplacian, Vollmer'99 anti-shrink update (default)\n"
		          << "              taubin = Taubin'95 band-pass (aggressive, ~zero shrink)\n";
		return EXIT_FAILURE;
	}

	const std::string inPath = argv[1];
	const std::string outPath = argv[2];
	int iterations = 5;
	if (argc >= 4) {
		try {
			iterations = std::stoi(argv[3]);
		} catch (const std::exception&) {
			// most likely the algo was passed in the iterations slot, e.g. "smooth in out taubin"
			std::cerr << "Error: invalid iterations '" << argv[3]
			          << "' (expected a number); usage: smooth <in> <out> [iterations] [algo]\n";
			return EXIT_FAILURE;
		}
	}
	const std::string algo = argc >= 5 ? std::string(argv[4]) : std::string("hc");

	halfmesh::Mesh mesh;
	if (!mesh.Load(inPath)) {
		std::cerr << "Error: failed to load '" << inPath << "'\n";
		return EXIT_FAILURE;
	}

	const auto vertsBefore = static_cast<unsigned>(mesh.vertices.size());
	const auto facesBefore = static_cast<unsigned>(mesh.faces.size());
	const auto aabbBefore = mesh.ComputeAABBox();
	const double diagBefore = aabbBefore.diagonal().norm();
	const halfmesh::Mesh::Vertex centroidBefore = Centroid(mesh.vertices);
	const double radiusBefore = MeanRadius(mesh.vertices, centroidBefore);
	std::cout << "Loaded:   " << vertsBefore << " vertices, " << facesBefore << " faces\n";

	// both methods go through the unified Smooth() convenience entry point
	if (algo == "taubin")
		mesh.Smooth(iterations, halfmesh::Mesh::SmoothMethod::Taubin);
	else if (algo == "hc")
		mesh.Smooth(iterations, halfmesh::Mesh::SmoothMethod::HCLaplacian);
	else {
		std::cerr << "Error: unknown algo '" << algo << "' (expected hc|taubin)\n";
		return EXIT_FAILURE;
	}

	const auto vertsAfter = static_cast<unsigned>(mesh.vertices.size());
	const auto facesAfter = static_cast<unsigned>(mesh.faces.size());
	const auto aabbAfter = mesh.ComputeAABBox();
	const double diagAfter = aabbAfter.diagonal().norm();
	// centroid is recomputed post-smoothing: HC Laplacian can repair (weld/remove) the
	// input, so the vertex set — and therefore its centroid — may differ from before.
	const halfmesh::Mesh::Vertex centroidAfter = Centroid(mesh.vertices);
	const double radiusAfter = MeanRadius(mesh.vertices, centroidAfter);
	std::cout << "Smoothed: " << vertsAfter << " vertices, " << facesAfter << " faces"
	          << " (iterations: " << iterations
	          << ", algo: " << algo << ")\n"
	          << "  bbox diagonal ratio:   " << (diagBefore > 0 ? diagAfter / diagBefore : 1.0) << "\n"
	          << "  mean radius ratio:     " << (radiusBefore > 0 ? radiusAfter / radiusBefore : 1.0) << "\n";

	if (!mesh.Save(outPath)) {
		std::cerr << "Error: failed to save '" << outPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Saved:    " << outPath << "\n";
	return EXIT_SUCCESS;
}

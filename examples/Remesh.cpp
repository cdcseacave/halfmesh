/*
* Remesh.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// examples/Remesh.cpp — Isotropic remeshing CLI example.
//
// Usage: remesh <in.ply> <out.ply> [edgeLength]
//
// Loads a triangle mesh, runs isotropic remeshing at the given target edge
// length (default: 1/50 of the bounding-box diagonal), and saves the result.
// Demonstrates the public API:
//   halfmesh::Mesh::Load / ComputeAABBox / RemeshIsotropic / Save
#include <halfmesh/Mesh.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
	if (argc < 3 || argc > 4) {
		std::cerr << "Usage: remesh <in.ply> <out.ply> [edge_length]\n"
		          << "  edge_length  target isotropic edge length in world units\n"
		          << "               (default: 1/50 of the bounding-box diagonal)\n";
		return EXIT_FAILURE;
	}

	const std::string inPath = argv[1];
	const std::string outPath = argv[2];

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
	// Unwelded input (glTF always, some PLYs) leaves every edge a boundary, which
	// starves the collapse/flip/relocate passes.  Exact weld (epsilon 0) +
	// topology-only degenerate cull (thArea 0, so valid near-zero-area slivers are
	// kept) is lossless for geometry-only meshes.
	mesh.RemoveDuplicateVertices(0);
	mesh.RemoveDegenerateFaces(0.f);
	mesh.RemoveUnreferencedVertices();

	// Determine target edge length.
	float edgeLength = 0.f;
	if (argc == 4) {
		edgeLength = std::stof(argv[3]);
	} else {
		const auto aabb = mesh.ComputeAABBox();
		edgeLength = static_cast<float>(aabb.diagonal().norm()) / 50.f;
		std::cout << "Auto edge length: " << edgeLength
		          << " (bbox diagonal / 50)\n";
	}

	if (edgeLength <= 0.f) {
		std::cerr << "Error: edge_length must be positive\n";
		return EXIT_FAILURE;
	}

	// Build remesh parameters and run.
	halfmesh::Mesh::RemeshParams params;
	params.SetEdgeLength(edgeLength);
	// Use default iterations (3) and no adaptive scaling.
	mesh.RemeshIsotropic(params);

	const auto facesAfter = static_cast<unsigned>(mesh.faces.size());
	const auto vertsAfter = static_cast<unsigned>(mesh.vertices.size());
	std::cout << "Remeshed: " << vertsAfter << " vertices, "
	          << facesAfter << " faces"
	          << " (edge target: " << edgeLength << ")\n";

	if (!mesh.Save(outPath)) {
		std::cerr << "Error: failed to save '" << outPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Saved:   " << outPath << "\n";
	return EXIT_SUCCESS;
}

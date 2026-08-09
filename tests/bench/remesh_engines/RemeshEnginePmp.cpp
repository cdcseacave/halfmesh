/*
* RemeshEnginePmp.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// pmp-library isotropic-remeshing baseline for RemeshBench.
// Compiled only when HALFMESH_REMESH_WITH_PMP=ON (links libpmp).

#include <halfmesh/Mesh.h>

#include <pmp/surface_mesh.h>
#include <pmp/algorithms/remeshing.h>

#include <chrono>
#include <vector>

bool RemeshPmp(const halfmesh::Mesh& in, float edgeLen, int iters,
               halfmesh::Mesh& out, double& seconds)
{
	// halfmesh::Mesh -> pmp::SurfaceMesh
	pmp::SurfaceMesh sm;
	std::vector<pmp::Vertex> vmap;
	vmap.reserve(in.vertices.size());
	for (const auto& v : in.vertices)
		vmap.push_back(sm.add_vertex(pmp::Point(v.x(), v.y(), v.z())));
	for (const auto& f : in.faces) {
		if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0])
			continue;
		try {
			sm.add_triangle(vmap[f[0]], vmap[f[1]], vmap[f[2]]);
		} catch (...) {
			// non-manifold insertion — skip this face, keep going
		}
	}
	if (sm.n_faces() == 0)
		return false;

	const auto t0 = std::chrono::steady_clock::now();
	try {
		pmp::uniform_remeshing(sm, edgeLen, static_cast<unsigned>(iters), /*use_projection*/ true);
	} catch (...) {
		return false;
	}
	seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
	              std::chrono::steady_clock::now() - t0)
	              .count();

	// pmp::SurfaceMesh -> halfmesh::Mesh (compact indices after GC)
	sm.garbage_collection();
	out.vertices.clear();
	out.faces.clear();
	out.vertices.reserve(sm.n_vertices());
	for (auto v : sm.vertices()) {
		const pmp::Point& p = sm.position(v);
		out.vertices.emplace_back(p[0], p[1], p[2]);
	}
	for (auto fh : sm.faces()) {
		uint32_t idx[3];
		int k = 0;
		for (auto vh : sm.vertices(fh)) {
			if (k < 3)
				idx[k] = static_cast<uint32_t>(vh.idx());
			++k;
		}
		if (k == 3)
			out.faces.emplace_back(halfmesh::Mesh::Face(idx[0], idx[1], idx[2]));
	}
	return !out.faces.empty();
}

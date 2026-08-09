/*
* Corpus.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/corpus/Corpus.cpp — test mesh corpus library implementation
//
// Generators use only the public halfmesh API.
// All topology values are analytically derived and documented inline.

#include "Corpus.h"

#include <halfmesh/HalfMesh.h>
#include <halfmesh/Mesh.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

namespace hmtest {
namespace corpus {

// ============================================================
// Internal helpers
// ============================================================

namespace {

// Push a triangular face onto m.faces.
static void AddFace(halfmesh::Mesh& m, uint32_t a, uint32_t b, uint32_t c)
{
	halfmesh::HalfMesh::Face f;
	f[0] = a;
	f[1] = b;
	f[2] = c;
	m.faces.push_back(f);
}

// Append a vertex and return its index.
static uint32_t AddVertex(halfmesh::Mesh& m, float x, float y, float z)
{
	const uint32_t idx = static_cast<uint32_t>(m.vertices.size());
	m.vertices.push_back({x, y, z});
	return idx;
}

} // anonymous namespace

// ============================================================
// Triangle
// ============================================================
// V=3 E=3 F=1, χ=1 (disk), g=0, 1 boundary loop.
halfmesh::Mesh Triangle()
{
	halfmesh::Mesh m;
	AddVertex(m, 0.f, 0.f, 0.f);
	AddVertex(m, 1.f, 0.f, 0.f);
	AddVertex(m, 0.f, 1.f, 0.f);
	AddFace(m, 0, 1, 2);
	return m;
}
KnownTopology Triangle_Known()
{
	return {3, 3, 1, 1, 0, 1, false};
}

// ============================================================
// Quad (two triangles sharing a diagonal)
// ============================================================
// V=4 E=5 F=2, χ=1 (disk), g=0, 1 boundary loop.
halfmesh::Mesh Quad()
{
	halfmesh::Mesh m;
	AddVertex(m, 0.f, 0.f, 0.f); // 0
	AddVertex(m, 1.f, 0.f, 0.f); // 1
	AddVertex(m, 1.f, 1.f, 0.f); // 2
	AddVertex(m, 0.f, 1.f, 0.f); // 3
	AddFace(m, 0, 1, 2);
	AddFace(m, 0, 2, 3);
	return m;
}
KnownTopology Quad_Known()
{
	// V=4, F=2. Edges: boundary(4) + 1 interior diagonal = 5. χ=4-5+2=1.
	return {4, 5, 2, 1, 0, 1, false};
}

// ============================================================
// TetrahedronMesh
// ============================================================
// Regular tetrahedron, edge length 1.
// V=4 E=6 F=4, χ=2, g=0, watertight.
halfmesh::Mesh TetrahedronMesh()
{
	halfmesh::Mesh m;
	// Vertices of a regular tetrahedron (edge length 1).
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.5f, std::sqrt(3.f) / 2.f, 0.f},
	    {0.5f, std::sqrt(3.f) / 6.f, std::sqrt(6.f) / 3.f},
	};
	// Outward-facing winding.
	AddFace(m, 0, 2, 1);
	AddFace(m, 0, 1, 3);
	AddFace(m, 1, 2, 3);
	AddFace(m, 0, 3, 2);
	return m;
}
KnownTopology TetrahedronMesh_Known()
{
	return {4, 6, 4, 2, 0, 0, true};
}

// ============================================================
// CubeMesh
// ============================================================
// Unit cube (side s): 8 vertices, 12 triangles (2 per face).
// V=8 E=18 F=12, χ=2 g=0, watertight.
halfmesh::Mesh CubeMesh(float s)
{
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // 0
	    {s, 0.f, 0.f}, // 1
	    {s, s, 0.f}, // 2
	    {0.f, s, 0.f}, // 3
	    {0.f, 0.f, s}, // 4
	    {s, 0.f, s}, // 5
	    {s, s, s}, // 6
	    {0.f, s, s}, // 7
	};
	// 6 faces × 2 triangles each, outward normals.
	// bottom (-z)
	AddFace(m, 0, 2, 1);
	AddFace(m, 0, 3, 2);
	// top (+z)
	AddFace(m, 4, 5, 6);
	AddFace(m, 4, 6, 7);
	// front (-y)
	AddFace(m, 0, 1, 5);
	AddFace(m, 0, 5, 4);
	// back (+y)
	AddFace(m, 3, 7, 6);
	AddFace(m, 3, 6, 2);
	// left (-x)
	AddFace(m, 0, 4, 7);
	AddFace(m, 0, 7, 3);
	// right (+x)
	AddFace(m, 1, 2, 6);
	AddFace(m, 1, 6, 5);
	return m;
}
KnownTopology CubeMesh_Known()
{
	// V=8 E=18 F=12 χ=8-18+12=2 g=0 closed.
	return {8, 18, 12, 2, 0, 0, true};
}

// ============================================================
// IcosahedronMesh
// ============================================================
// Regular icosahedron: V=12 E=30 F=20, χ=2 g=0, watertight.
halfmesh::Mesh IcosahedronMesh()
{
	halfmesh::Mesh m;
	// Golden ratio.
	const float phi = (1.f + std::sqrt(5.f)) / 2.f;
	// 12 vertices on 3 orthogonal golden rectangles.
	m.vertices = {
	    {-1.f, phi, 0.f},
	    {1.f, phi, 0.f},
	    {-1.f, -phi, 0.f},
	    {1.f, -phi, 0.f},
	    {0.f, -1.f, phi},
	    {0.f, 1.f, phi},
	    {0.f, -1.f, -phi},
	    {0.f, 1.f, -phi},
	    {phi, 0.f, -1.f},
	    {phi, 0.f, 1.f},
	    {-phi, 0.f, -1.f},
	    {-phi, 0.f, 1.f},
	};
	// 20 triangular faces with consistent outward winding.
	AddFace(m, 0, 11, 5);
	AddFace(m, 0, 5, 1);
	AddFace(m, 0, 1, 7);
	AddFace(m, 0, 7, 10);
	AddFace(m, 0, 10, 11);

	AddFace(m, 1, 5, 9);
	AddFace(m, 5, 11, 4);
	AddFace(m, 11, 10, 2);
	AddFace(m, 10, 7, 6);
	AddFace(m, 7, 1, 8);

	AddFace(m, 3, 9, 4);
	AddFace(m, 3, 4, 2);
	AddFace(m, 3, 2, 6);
	AddFace(m, 3, 6, 8);
	AddFace(m, 3, 8, 9);

	AddFace(m, 4, 9, 5);
	AddFace(m, 2, 4, 11);
	AddFace(m, 6, 2, 10);
	AddFace(m, 8, 6, 7);
	AddFace(m, 9, 8, 1);
	return m;
}
KnownTopology IcosahedronMesh_Known()
{
	// V=12 E=30 F=20 χ=12-30+20=2 g=0 closed.
	return {12, 30, 20, 2, 0, 0, true};
}

// ============================================================
// GridPlane
// ============================================================
// Flat z=0 grid of n×n quads → (n+1)² verts, 2n² tris.
// V=(n+1)², F=2n², E via Euler: χ=1 (disk) → E=V+F-1.
// 1 boundary loop, g=0.
halfmesh::Mesh GridPlane(unsigned n)
{
	assert(n >= 1);
	halfmesh::Mesh m;
	const unsigned nv = n + 1;
	m.vertices.reserve(nv * nv);
	for (unsigned j = 0; j < nv; ++j)
		for (unsigned i = 0; i < nv; ++i)
			m.vertices.push_back({static_cast<float>(i),
			                      static_cast<float>(j), 0.f});
	m.faces.reserve(2 * n * n);
	for (unsigned j = 0; j < n; ++j) {
		for (unsigned i = 0; i < n; ++i) {
			const uint32_t v00 = j * nv + i;
			const uint32_t v10 = j * nv + (i + 1);
			const uint32_t v01 = (j + 1) * nv + i;
			const uint32_t v11 = (j + 1) * nv + (i + 1);
			AddFace(m, v00, v10, v11);
			AddFace(m, v00, v11, v01);
		}
	}
	return m;
}
KnownTopology GridPlane_Known(unsigned n)
{
	const uint32_t V = (n + 1) * (n + 1);
	const uint32_t F = 2 * n * n;
	// χ=1 (disk) → E = V + F - 1
	const uint32_t E = V + F - 1;
	return {V, E, F, 1, 0, 1, false};
}

// ============================================================
// OpenCylinder
// ============================================================
// Open tube: radialSegs meridional segments, heightSegs stacks.
// Vertices: radial * (height+1) rings.
// Faces: 2 * radial * height triangles (2 per quad).
// 2 boundary loops (top + bottom cap), χ=0 (annulus), g=0.
// Developable: unrolls to a rectangle.
halfmesh::Mesh OpenCylinder(unsigned radialSegs, unsigned heightSegs)
{
	assert(radialSegs >= 3 && heightSegs >= 1);
	halfmesh::Mesh m;
	const float dtheta = 2.f * static_cast<float>(std::numbers::pi) / static_cast<float>(radialSegs);
	const float dh = 1.f / static_cast<float>(heightSegs);

	// radialSegs * (heightSegs+1) vertices.
	m.vertices.reserve(radialSegs * (heightSegs + 1));
	for (unsigned h = 0; h <= heightSegs; ++h) {
		const float y = static_cast<float>(h) * dh;
		for (unsigned r = 0; r < radialSegs; ++r) {
			const float theta = static_cast<float>(r) * dtheta;
			m.vertices.push_back({std::cos(theta), y, std::sin(theta)});
		}
	}
	// Quads → 2 triangles each.
	m.faces.reserve(2 * radialSegs * heightSegs);
	for (unsigned h = 0; h < heightSegs; ++h) {
		for (unsigned r = 0; r < radialSegs; ++r) {
			const uint32_t rn = (r + 1) % radialSegs;
			const uint32_t v00 = h * radialSegs + r;
			const uint32_t v10 = h * radialSegs + rn;
			const uint32_t v01 = (h + 1) * radialSegs + r;
			const uint32_t v11 = (h + 1) * radialSegs + rn;
			AddFace(m, v00, v10, v11);
			AddFace(m, v00, v11, v01);
		}
	}
	return m;
}
KnownTopology OpenCylinder_Known(unsigned radialSegs, unsigned heightSegs)
{
	// V = radial * (height+1), F = 2*radial*height.
	// For an open cylinder (annulus = rectangle with top/bottom open):
	//   Topologically equivalent to an annulus.
	//   χ(annulus)=0, g=0, B=2 boundary loops.
	//   Euler: χ = V - E + F = 0.
	//   E = V + F = radial*(height+1) + 2*radial*height = radial*(3*height+1).
	const uint32_t V = radialSegs * (heightSegs + 1);
	const uint32_t F = 2 * radialSegs * heightSegs;
	const uint32_t E = V + F; // χ=0 → E = V+F
	return {V, E, F, 0, 0, 2, false};
}

// ============================================================
// Cone
// ============================================================
// Open cone: apex at top, base ring at y=0.
// V = radialSegs + 1 (apex), F = radialSegs triangles.
// 1 boundary loop (base circle), χ=1 (disk), g=0.
// Developable: unrolls to a circular sector.
halfmesh::Mesh Cone(unsigned radialSegs)
{
	assert(radialSegs >= 3);
	halfmesh::Mesh m;
	const float dtheta = 2.f * static_cast<float>(std::numbers::pi) / static_cast<float>(radialSegs);
	// Base ring vertices.
	m.vertices.reserve(radialSegs + 1);
	for (unsigned r = 0; r < radialSegs; ++r) {
		const float theta = static_cast<float>(r) * dtheta;
		m.vertices.push_back({std::cos(theta), 0.f, std::sin(theta)});
	}
	// Apex vertex.
	const uint32_t apex = AddVertex(m, 0.f, 1.f, 0.f);
	// Triangles from base ring to apex.
	// Winding: (apex, rn, r) gives outward normals (pointing away from the cone axis).
	// Reversing r and rn w.r.t. the naive (apex, r, rn) order is required because
	// the base ring is parameterized CCW when viewed from below (y<0), so to obtain
	// an outward-facing normal the triangle must be CW in that view → (apex, rn, r).
	m.faces.reserve(radialSegs);
	for (unsigned r = 0; r < radialSegs; ++r) {
		const uint32_t rn = (r + 1) % radialSegs;
		AddFace(m, apex, rn, r);
	}
	return m;
}
KnownTopology Cone_Known(unsigned radialSegs)
{
	// V = radialSegs + 1, F = radialSegs.
	// χ=1 (disk) → E = V + F - 1 = 2*radialSegs.
	const uint32_t V = radialSegs + 1;
	const uint32_t F = radialSegs;
	const uint32_t E = V + F - 1; // χ=1 → E = V+F-1
	return {V, E, F, 1, 0, 1, false};
}

// ============================================================
// UVSphere
// ============================================================
// UV sphere: stacks latitudinal rings, slices meridional segments.
// Poles: 1 north (top) + 1 south (bottom).
// V = slices*(stacks-1) + 2.
// F = 2*slices + 2*(stacks-2)*slices (top cap + bottom cap + body quads).
// Wait: standard UV sphere with stacks rings between poles:
//   body_quads = (stacks-1) rows, each with slices quads = 2*slices tris.
//   top_cap: slices triangles (apex→ring0).
//   bottom_cap: slices triangles (ring_{stacks-1}→southPole).
//   F = slices + 2*(stacks-1)*slices + slices = 2*stacks*slices.
//   V = slices*(stacks-1) + 2.
// Closed, χ=2, g=0.
halfmesh::Mesh UVSphere(unsigned stacks, unsigned slices)
{
	assert(stacks >= 2 && slices >= 3);
	halfmesh::Mesh m;
	// North pole.
	m.vertices.push_back({0.f, 1.f, 0.f});
	// Latitude rings (stacks-1 rings between the poles).
	const float dphi = static_cast<float>(std::numbers::pi) / static_cast<float>(stacks);
	const float dtheta = 2.f * static_cast<float>(std::numbers::pi) / static_cast<float>(slices);
	for (unsigned s = 1; s < stacks; ++s) {
		const float phi = static_cast<float>(s) * dphi;
		const float y = std::cos(phi);
		const float r = std::sin(phi);
		for (unsigned sl = 0; sl < slices; ++sl) {
			const float theta = static_cast<float>(sl) * dtheta;
			m.vertices.push_back({r * std::cos(theta), y, r * std::sin(theta)});
		}
	}
	// South pole.
	m.vertices.push_back({0.f, -1.f, 0.f});

	const uint32_t northPole = 0;
	const uint32_t southPole = static_cast<uint32_t>(m.vertices.size()) - 1;
	// First body ring starts at index 1.
	auto ringIdx = [&](unsigned stackRow, unsigned col) -> uint32_t {
		return 1 + stackRow * slices + col % slices;
	};

	// Top cap: triangles from north pole to first ring.
	for (unsigned sl = 0; sl < slices; ++sl) {
		AddFace(m, northPole,
		        ringIdx(0, sl + 1),
		        ringIdx(0, sl));
	}
	// Body quads: (stacks-2) rows of quad strips.
	for (unsigned s = 0; s < stacks - 2; ++s) {
		for (unsigned sl = 0; sl < slices; ++sl) {
			const uint32_t v00 = ringIdx(s, sl);
			const uint32_t v10 = ringIdx(s, sl + 1);
			const uint32_t v01 = ringIdx(s + 1, sl);
			const uint32_t v11 = ringIdx(s + 1, sl + 1);
			AddFace(m, v00, v10, v11);
			AddFace(m, v00, v11, v01);
		}
	}
	// Bottom cap: triangles from last ring to south pole.
	const unsigned lastRing = stacks - 2;
	for (unsigned sl = 0; sl < slices; ++sl) {
		AddFace(m, southPole,
		        ringIdx(lastRing, sl),
		        ringIdx(lastRing, sl + 1));
	}
	return m;
}
KnownTopology UVSphere_Known(unsigned stacks, unsigned slices)
{
	// V = (stacks-1)*slices + 2.
	// F = slices (top cap) + 2*(stacks-2)*slices (body quads as tris)
	//   + slices (bottom cap).
	//   = 2*slices + 2*(stacks-2)*slices = 2*(stacks-1)*slices.
	// Wait: let's count carefully:
	//   top_cap triangles: slices
	//   body rows: (stacks-2) rows, each slices quads = 2*slices tris
	//              → 2*(stacks-2)*slices tris
	//   bottom_cap triangles: slices
	//   Total F = slices + 2*(stacks-2)*slices + slices
	//           = 2*slices*(stacks-1)... wait:
	//           = 2*slices + 2*(stacks-2)*slices = 2*slices*(1 + stacks-2) = 2*slices*(stacks-1).
	// Hmm, let's just check: stacks=2: F = slices + 0 + slices = 2*slices. Ring=0 only; body=0.
	// stacks=3: F = slices + 2*slices + slices = 4*slices. = 2*(stacks-1)*slices=2*2*slices. OK.
	// V = (stacks-1)*slices + 2.
	// E: for a closed sphere χ=2 → E = V + F - 2.
	const uint32_t V = (stacks - 1) * slices + 2;
	const uint32_t F = 2 * (stacks - 1) * slices;
	// Recheck: top_cap=slices, body=2*(stacks-2)*slices, bottom_cap=slices.
	// Total = 2*slices + 2*(stacks-2)*slices = 2*slices*(stacks-1). Confirmed.
	const uint32_t E = V + F - 2; // χ=2 → E=V+F-2
	return {V, E, F, 2, 0, 0, true};
}

// ============================================================
// TorusMesh
// ============================================================
// Torus: majorSegs × minorSegs grid, both axes wrapped.
// V = majorSegs * minorSegs, F = 2 * majorSegs * minorSegs.
// E = 3 * majorSegs * minorSegs (χ=0 → E=V+F = 3MN, check: V-E+F=MN-3MN+2MN=0 ✓).
// Closed, χ=0 genus=1.
halfmesh::Mesh TorusMesh(unsigned majorSegs, unsigned minorSegs)
{
	assert(majorSegs >= 3 && minorSegs >= 3);
	halfmesh::Mesh m;
	const float R = 1.5f; // major radius
	const float r = 0.5f; // minor radius
	const float dphi = 2.f * static_cast<float>(std::numbers::pi) / static_cast<float>(majorSegs);
	const float dtheta = 2.f * static_cast<float>(std::numbers::pi) / static_cast<float>(minorSegs);

	m.vertices.reserve(majorSegs * minorSegs);
	for (unsigned i = 0; i < majorSegs; ++i) {
		const float phi = static_cast<float>(i) * dphi;
		const float cphi = std::cos(phi);
		const float sphi = std::sin(phi);
		for (unsigned j = 0; j < minorSegs; ++j) {
			const float theta = static_cast<float>(j) * dtheta;
			const float ctheta = std::cos(theta);
			const float stheta = std::sin(theta);
			const float x = (R + r * ctheta) * cphi;
			const float y = r * stheta;
			const float z = (R + r * ctheta) * sphi;
			m.vertices.push_back({x, y, z});
		}
	}
	// Quads → 2 triangles each (both axes wrap around).
	m.faces.reserve(2 * majorSegs * minorSegs);
	for (unsigned i = 0; i < majorSegs; ++i) {
		const unsigned ni = (i + 1) % majorSegs;
		for (unsigned j = 0; j < minorSegs; ++j) {
			const unsigned nj = (j + 1) % minorSegs;
			const uint32_t v00 = i * minorSegs + j;
			const uint32_t v10 = i * minorSegs + nj;
			const uint32_t v01 = ni * minorSegs + j;
			const uint32_t v11 = ni * minorSegs + nj;
			AddFace(m, v00, v10, v11);
			AddFace(m, v00, v11, v01);
		}
	}
	return m;
}
KnownTopology TorusMesh_Known(unsigned majorSegs, unsigned minorSegs)
{
	const uint32_t V = majorSegs * minorSegs;
	const uint32_t F = 2 * majorSegs * minorSegs;
	const uint32_t E = V + F; // χ=0 → E=V+F
	return {V, E, F, 0, 1, 0, true};
}

// ============================================================
// LargeMesh
// ============================================================
// Fine UV sphere approximating targetFaces triangles.
// Compute stacks/slices so F ≈ targetFaces.
// F = 2*(stacks-1)*slices → stacks=slices (approximately square quads).
// 2*(s-1)*s ≈ targetFaces → s ≈ sqrt(targetFaces/2)+1.
halfmesh::Mesh LargeMesh(unsigned targetFaces, unsigned* actualFaces)
{
	if (targetFaces < 4)
		targetFaces = 4;
	// Estimate stacks and slices so that F ≈ targetFaces.
	// Use slices=stacks for near-square quads.
	const unsigned slices = std::max(3u,
	                                 static_cast<unsigned>(std::sqrt(static_cast<double>(targetFaces) / 2.0) + 1));
	const unsigned stacks = std::max(2u, slices);
	halfmesh::Mesh m = UVSphere(stacks, slices);
	if (actualFaces)
		*actualFaces = static_cast<unsigned>(m.faces.size());
	return m;
}

// ============================================================
// DIRTY MESH SYNTHESIZERS
// ============================================================

// Bow-tie: two fans sharing only vertex 0.
halfmesh::Mesh DirtyBowTie(DirtyDefects* defects)
{
	halfmesh::Mesh m;
	// Fan 1: triangles (0,1,2), (0,2,3)
	// Fan 2: triangles (0,4,5), (0,5,6)
	// Vertex 0 is the bow-tie center.
	m.vertices = {
	    {0.f, 0.f, 0.f}, // 0 — bow-tie center
	    {1.f, 0.f, 0.f}, // 1
	    {1.f, 1.f, 0.f}, // 2
	    {0.f, 1.f, 0.f}, // 3
	    {-1.f, 0.f, 0.f}, // 4
	    {-1.f, -1.f, 0.f}, // 5
	    {0.f, -1.f, 0.f}, // 6
	};
	AddFace(m, 0, 1, 2);
	AddFace(m, 0, 2, 3);
	AddFace(m, 0, 4, 5);
	AddFace(m, 0, 5, 6);
	if (defects) {
		defects->hasBowTieVertex = true;
		defects->description =
		    "Bow-tie: vertex 0 shared by two disconnected fans (vertex-non-manifold).";
	}
	return m;
}

// Three triangles sharing edge (0,1): edge-non-manifold.
halfmesh::Mesh DirtyThreeOnEdge(DirtyDefects* defects)
{
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // 0
	    {1.f, 0.f, 0.f}, // 1
	    {0.5f, 1.f, 0.f}, // 2
	    {0.5f, -1.f, 0.f}, // 3
	    {0.5f, 0.f, 1.f}, // 4
	};
	AddFace(m, 0, 1, 2);
	AddFace(m, 0, 1, 3);
	AddFace(m, 0, 1, 4);
	if (defects) {
		defects->hasThreeOnEdge = true;
		defects->description =
		    "Three triangles on edge (0,1): edge-non-manifold.";
	}
	return m;
}

// Cube + duplicate face pairs.
halfmesh::Mesh DirtyDuplicateFaces(unsigned numDupPairs, DirtyDefects* defects)
{
	halfmesh::Mesh m = CubeMesh(1.0f);
	// Inject duplicates of the first numDupPairs faces.
	const unsigned actualDups = std::min(numDupPairs,
	                                     static_cast<unsigned>(m.faces.size()));
	for (unsigned i = 0; i < actualDups; ++i) {
		m.faces.push_back(m.faces[i]); // exact duplicate
	}
	if (defects) {
		// Both copies will be removed by RemoveDuplicateFaces(remove_both=true).
		defects->duplicateFaces = 2 * actualDups;
		defects->description =
		    "Cube + " + std::to_string(actualDups) + " duplicate face pairs (both copies counted).";
	}
	return m;
}

// Cube + degenerate (collinear / zero-area) faces appended.
// Each degenerate face uses its own private set of 3 collinear vertices, offset
// far from the cube so that RemoveDegenerateFaces cannot merge them into cube
// vertices and cascade into additional removals.  This guarantees that exactly
// numDegen faces (and only those) are removed, giving a deterministic count.
halfmesh::Mesh DirtyDegenerateFaces(unsigned numDegen, DirtyDefects* defects)
{
	halfmesh::Mesh m = CubeMesh(1.0f);
	for (unsigned i = 0; i < numDegen; ++i) {
		// Place each degenerate face on its own collinear triple starting at x=100+10*i,
		// well away from the cube and from each other.  The three points are collinear
		// along the x-axis → cross product = 0 → zero area.
		const float base = 100.f + static_cast<float>(i) * 10.f;
		const uint32_t va = AddVertex(m, base, 0.f, 0.f);
		const uint32_t vb = AddVertex(m, base + 1.f, 0.f, 0.f);
		const uint32_t vc = AddVertex(m, base + 2.f, 0.f, 0.f);
		AddFace(m, va, vb, vc); // collinear → zero area
	}
	if (defects) {
		defects->degenerateFaces = numDegen;
		defects->description =
		    "Cube + " + std::to_string(numDegen) + " degenerate (zero-area) faces.";
	}
	return m;
}

// Cube + unreferenced vertices appended.
halfmesh::Mesh DirtyUnreferencedVertices(unsigned numUnref, DirtyDefects* defects)
{
	halfmesh::Mesh m = CubeMesh(1.0f);
	for (unsigned i = 0; i < numUnref; ++i) {
		const float fi = static_cast<float>(i);
		AddVertex(m, 100.f + fi, 100.f, 100.f);
	}
	if (defects) {
		defects->unreferencedVertices = numUnref;
		defects->description =
		    "Cube + " + std::to_string(numUnref) + " unreferenced vertices.";
	}
	return m;
}

// Cube with faces deleted to open holes.
// Each deleted face opens 1 hole (the surrounding edges become boundary).
// We delete non-adjacent faces so each forms its own boundary loop.
// A cube has 12 faces; we can safely delete up to 6 non-adjacent ones.
halfmesh::Mesh DirtyHoles(unsigned numHoles, DirtyDefects* defects)
{
	halfmesh::Mesh m = CubeMesh(1.0f);
	// The cube has 12 faces in pairs (bottom=0,1; top=2,3; front=4,5; back=6,7; left=8,9; right=10,11).
	// Pick one face per pair (even indices) to maximally spread holes.
	// With 6 pairs we can have up to 6 holes.
	const unsigned maxHoles = 6;
	const unsigned nh = std::min(numHoles, maxHoles);
	// Faces to delete: indices 0, 2, 4, 6, 8, 10 (first face of each pair).
	// Collect in reverse order to preserve indices during erasure.
	std::vector<unsigned> toDelete;
	for (unsigned i = 0; i < nh; ++i) {
		toDelete.push_back(i * 2); // 0, 2, 4, ...
	}
	// Erase in reverse order.
	for (unsigned i = nh; i > 0; --i) {
		const unsigned idx = toDelete[i - 1];
		m.faces.erase(m.faces.begin() + idx);
	}
	if (defects) {
		defects->deletedFaces = nh;
		defects->expectedBoundaryLoops = nh;
		defects->description =
		    "Cube with " + std::to_string(nh) + " faces deleted, creating " + std::to_string(nh) + " boundary loops.";
	}
	return m;
}

// One large cube + (numComponents - 1) small single-triangle components.
// The cube has 12 faces (8 verts); each small component has 1 face (3 verts).
// DirtyDefects records:
//   numComponents = numComponents (1 large + k small)
//   smallComponents = numComponents - 1
//   smallComponentThreshold = 2  (removes components with face count < 2)
//   largeComponentFaces = 12, largeComponentVerts = 8
// This structure allows RemoveSmallComponents(2) to remove exactly the k small
// triangle components and leave the cube intact.
halfmesh::Mesh DirtyManyComponents(unsigned numComponents, DirtyDefects* defects)
{
	if (numComponents < 1)
		numComponents = 1;
	halfmesh::Mesh m;

	// Large component: one unit cube, placed at origin.
	{
		halfmesh::Mesh cube = CubeMesh(1.0f);
		for (const auto& v : cube.vertices)
			m.vertices.push_back(v);
		for (auto f : cube.faces) {
			m.faces.push_back(f);
		}
	}

	// Small components: single triangles placed far from the cube.
	const unsigned numSmall = numComponents - 1;
	for (unsigned k = 0; k < numSmall; ++k) {
		const float dx = 100.f + static_cast<float>(k) * 10.f;
		const uint32_t vbase = static_cast<uint32_t>(m.vertices.size());
		m.vertices.push_back({dx, 0.f, 0.f});
		m.vertices.push_back({dx + 1.f, 0.f, 0.f});
		m.vertices.push_back({dx, 1.f, 0.f});
		halfmesh::HalfMesh::Face f;
		f[0] = vbase;
		f[1] = vbase + 1;
		f[2] = vbase + 2;
		m.faces.push_back(f);
	}

	if (defects) {
		defects->numComponents = numComponents;
		defects->smallComponents = numSmall;
		defects->smallComponentThreshold = 2u; // threshold to pass to RemoveSmallComponents
		defects->largeComponentFaces = 12u; // cube face count
		defects->largeComponentVerts = 8u; // cube vertex count
		defects->description =
		    "1 cube (12F) + " + std::to_string(numSmall) + " single-triangle component(s).";
	}
	return m;
}

// ============================================================
// UV GROUND-TRUTH HELPERS FOR DEVELOPABLE GENERATORS
// ============================================================

// GridPlane identity UV.
// GridPlane(n) produces vertices in row-major order: idx = row*(n+1) + col.
// The canonical developable UV maps each vertex directly by its grid position
// normalized to [0,1]: u = col/n, v = row/n.
std::vector<UV2> GridPlane_ExpectedUV(unsigned n)
{
	assert(n >= 1);
	const unsigned nv = n + 1;
	std::vector<UV2> uvs;
	uvs.reserve(nv * nv);
	const float invN = 1.f / static_cast<float>(n);
	for (unsigned row = 0; row < nv; ++row) {
		for (unsigned col = 0; col < nv; ++col) {
			uvs.push_back({static_cast<float>(col) * invN,
			               static_cast<float>(row) * invN});
		}
	}
	return uvs;
}

// OpenCylinder analytic unroll UV.
// OpenCylinder(radialSegs, heightSegs) vertices: idx = h*radialSegs + r.
// The cylinder (radius=1, height=1) unrolls to the rectangle [0,1] × [0,1]
// where u = r/radialSegs (azimuthal fraction) and v = h/heightSegs (height fraction).
std::vector<UV2> OpenCylinder_ExpectedUV(unsigned radialSegs, unsigned heightSegs)
{
	assert(radialSegs >= 3 && heightSegs >= 1);
	std::vector<UV2> uvs;
	uvs.reserve(radialSegs * (heightSegs + 1));
	const float invR = 1.f / static_cast<float>(radialSegs);
	const float invH = 1.f / static_cast<float>(heightSegs);
	for (unsigned h = 0; h <= heightSegs; ++h) {
		for (unsigned r = 0; r < radialSegs; ++r) {
			uvs.push_back({static_cast<float>(r) * invR,
			               static_cast<float>(h) * invH});
		}
	}
	return uvs;
}

// Cone analytic unroll UV.
// Cone(radialSegs) vertices: 0..radialSegs-1 = base ring, radialSegs = apex.
// The cone (base radius=1, height=1) has slant length = sqrt(2).
// It unrolls to a circular sector of radius sqrt(2) and opening angle 2π/sqrt(2).
// We normalize to [0,1]² placing the apex at the centre (0.5, 0.5) and scaling
// the sector arc to lie on the unit circle:
//   apex:      uv = (0.5, 0.5)
//   base r:    phi = r * (2π / radialSegs) / sqrt(2)    (sector angle)
//              uv  = (0.5 + 0.5*cos(phi), 0.5 + 0.5*sin(phi))
std::vector<UV2> Cone_ExpectedUV(unsigned radialSegs)
{
	assert(radialSegs >= 3);
	std::vector<UV2> uvs;
	uvs.reserve(radialSegs + 1);
	const float slant = std::sqrt(2.f); // sqrt(radius² + height²) = sqrt(1+1)
	const float dthetaSector = 2.f * static_cast<float>(std::numbers::pi) / (static_cast<float>(radialSegs) * slant);
	for (unsigned r = 0; r < radialSegs; ++r) {
		const float phi = static_cast<float>(r) * dthetaSector;
		uvs.push_back({0.5f + 0.5f * std::cos(phi),
		               0.5f + 0.5f * std::sin(phi)});
	}
	// Apex is the last vertex.
	uvs.push_back({0.5f, 0.5f});
	return uvs;
}

// ============================================================
// CORPUS OVERVIEW
// ============================================================

std::vector<CorpusEntry> makeCorpus()
{
	std::vector<CorpusEntry> entries;

	auto addClean = [&](const std::string& name, KnownTopology topo) {
		CorpusEntry e;
		e.name = name;
		e.isDirty = false;
		e.cleanTopo = topo;
		entries.push_back(std::move(e));
	};
	auto addDirty = [&](const std::string& name, DirtyDefects dd) {
		CorpusEntry e;
		e.name = name;
		e.isDirty = true;
		e.dirtyInfo = dd;
		entries.push_back(std::move(e));
	};

	// Clean generators
	addClean("Triangle", Triangle_Known());
	addClean("Quad", Quad_Known());
	addClean("Tetrahedron", TetrahedronMesh_Known());
	addClean("Cube", CubeMesh_Known());
	addClean("Icosahedron", IcosahedronMesh_Known());
	addClean("GridPlane(4)", GridPlane_Known(4));
	addClean("OpenCylinder(8,4)", OpenCylinder_Known(8, 4));
	addClean("Cone(8)", Cone_Known(8));
	addClean("UVSphere(8,12)", UVSphere_Known(8, 12));
	addClean("Torus(12,8)", TorusMesh_Known(12, 8));

	// Dirty generators
	{
		DirtyDefects dd;
		DirtyBowTie(&dd);
		addDirty("DirtyBowTie", dd);
	}
	{
		DirtyDefects dd;
		DirtyThreeOnEdge(&dd);
		addDirty("DirtyThreeOnEdge", dd);
	}
	{
		DirtyDefects dd;
		DirtyDuplicateFaces(3, &dd);
		addDirty("DirtyDuplicateFaces(3)", dd);
	}
	{
		DirtyDefects dd;
		DirtyDegenerateFaces(3, &dd);
		addDirty("DirtyDegenerateFaces(3)", dd);
	}
	{
		DirtyDefects dd;
		DirtyUnreferencedVertices(5, &dd);
		addDirty("DirtyUnreferencedVertices(5)", dd);
	}
	{
		DirtyDefects dd;
		DirtyHoles(2, &dd);
		addDirty("DirtyHoles(2)", dd);
	}
	{
		DirtyDefects dd;
		DirtyManyComponents(2, &dd);
		addDirty("DirtyManyComponents(2)", dd);
	}

	return entries;
}

} // namespace corpus
} // namespace hmtest

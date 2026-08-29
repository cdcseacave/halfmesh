/*
* Corpus.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/corpus/Corpus.h — test mesh corpus library
//
// Provides:
//   - Analytic mesh generators with documented known topology (hmtest::corpus)
//   - Dirty-mesh synthesizers recording exact defect counts
//   - makeCorpus(): overview of all generators
//
// Namespace: hmtest::corpus
// All generators use the public halfmesh API only.
#pragma once

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include <cmath> // M_PI (SaddleFan default angleSum)
#include <cstdint>
#include <string>
#include <vector>

namespace hmtest {
namespace corpus {

// ============================================================
// Known topology for each clean generator
// ============================================================

struct KnownTopology
{
	uint32_t numVertices = 0;
	uint32_t numEdges = 0;
	uint32_t numFaces = 0;
	int32_t euler = 0; // χ = V - E + F
	int32_t genus = 0;
	uint32_t numBoundaryLoops = 0;
	bool isWatertight = false;
};

// ============================================================
// CLEAN / ANALYTIC GENERATORS
// ============================================================

// Single triangle (open, 1 boundary loop, V=3 E=3 F=1 χ=1 g=0).
halfmesh::Mesh Triangle();
KnownTopology Triangle_Known();

// Quad as 2 triangles (open, 1 boundary loop, V=4 E=5 F=2 χ=1 g=0).
halfmesh::Mesh Quad();
KnownTopology Quad_Known();

// Regular tetrahedron: V=4 E=6 F=4 χ=2 g=0, closed, edge length=1.
halfmesh::Mesh TetrahedronMesh();
KnownTopology TetrahedronMesh_Known();

// Unit cube (side s): V=8 E=18 F=12 χ=2 g=0, watertight.
// Surface area = 6·s², volume = s³.
halfmesh::Mesh CubeMesh(float side = 1.0f);
KnownTopology CubeMesh_Known();

// Regular icosahedron: V=12 E=30 F=20 χ=2 g=0, watertight.
halfmesh::Mesh IcosahedronMesh();
KnownTopology IcosahedronMesh_Known();

// Flat grid plane of n×n quads: (n+1)² vertices, 2n² triangles, 1 boundary loop.
// χ=1 (disk topology), g=0.
halfmesh::Mesh GridPlane(unsigned n = 4);
KnownTopology GridPlane_Known(unsigned n);

// Open cylinder (tube): radialSegs segments, heightSegs stacks.
// V = radial*(height+1), F = 2*radial*height.
// 2 boundary loops (top + bottom), χ=0 (annulus), g=0.
halfmesh::Mesh OpenCylinder(unsigned radialSegs = 8, unsigned heightSegs = 4);
KnownTopology OpenCylinder_Known(unsigned radialSegs, unsigned heightSegs);

// Open cone (tip at apex): radialSegs segments around the base.
// 1 boundary loop (base), χ=1 (disk), g=0.
halfmesh::Mesh Cone(unsigned radialSegs = 8);
KnownTopology Cone_Known(unsigned radialSegs);

// UV Sphere: stacks latitudinal rings, slices meridional segments.
// V = (stacks-1)*slices+2 (poles), F = 2*slices + 2*(stacks-2)*slices.
// Closed, χ=2 g=0. Gauss–Bonnet: integrated angle defect ≈ 4π.
halfmesh::Mesh UVSphere(unsigned stacks = 8, unsigned slices = 12);
KnownTopology UVSphere_Known(unsigned stacks, unsigned slices);

// Torus: majorSegs × minorSegs grid, wrapped on both axes.
// V=major*minor, F=2*major*minor, E=3*major*minor.
// Closed, χ=0 genus=1.
halfmesh::Mesh TorusMesh(unsigned majorSegs = 12, unsigned minorSegs = 8);
KnownTopology TorusMesh_Known(unsigned majorSegs, unsigned minorSegs);

// Large mesh: fine UV sphere with ~targetFaces faces (for perf tests).
// Returns actual face count in actualFaces (written by the function).
halfmesh::Mesh LargeMesh(unsigned targetFaces, unsigned* actualFaces = nullptr);

// ============================================================
// FOLD-INDUCING FIXTURES
// ============================================================

// Saddle fan: n triangles fanned around an interior apex (vertex 0) whose
// rim wraps `angleSum` radians of azimuth (> 2π by default) while a zig-zag
// elevation embeds the excess angle in 3D (non-self-intersecting). The apex's
// angle-sum exceeds 2π (a saddle point / enclosed negative curvature), which
// folds under a shipped SLIM/ARAP flatten — used by the flatten fold-rescue
// tests (Flatten/Parametrize suites). Fixture premise: SaddleFan(12, 3π)
// folds under the shipped flatten (measured: 10/12 faces flagged) — if this
// ever stops folding, raise the zig-zag amplitude / angleSum, do not weaken
// the tests that rely on it.
halfmesh::Mesh SaddleFan(int n = 12, double angleSum = 3.0 * M_PI);

// ============================================================
// UV GROUND-TRUTH HELPERS FOR DEVELOPABLE GENERATORS
// ============================================================
// These helpers provide analytic 2-D UV coordinates (or zero-stretch checks)
// for the three developable generators: GridPlane, OpenCylinder, and Cone.
// They are used by the UV tests to validate that a computed parametrization
// is correct (stretch ≈ 0 for any correct flattening of a developable surface).
//
// Convention: the returned vectors are indexed by VERTEX, not by face-corner,
// and map directly to the vertex ordering produced by the corresponding generator
// called with the same arguments.

// GridPlane identity UV: vertex i=(col,row) maps to UV=(col, row) normalized
// to [0,1]×[0,1].  GridPlane(n) has (n+1)² vertices in row-major order:
//   vertex_idx = row*(n+1) + col,  uv = (col/n, row/n).
// Returns one UV pair per vertex.
struct UV2
{
	float u, v;
};
std::vector<UV2> GridPlane_ExpectedUV(unsigned n);

// OpenCylinder analytic unroll: the cylinder (radius=1, height=1) unrolls to
// a rectangle [0, 2π] × [0, 1].
//   vertex r at ring h: uv = (r / radialSegs, h / heightSegs).
// Vertex ordering matches OpenCylinder(radialSegs, heightSegs):
//   vertex_idx = h * radialSegs + r.
// Returns one UV pair per vertex.
std::vector<UV2> OpenCylinder_ExpectedUV(unsigned radialSegs, unsigned heightSegs);

// Cone analytic unroll: the cone (base radius=1, height=1, slant=√2) unrolls
// to a circular sector of radius √2 and angle 2π/√2 = π√2.
// Apex maps to the sector origin (u=0, v=0 in sector coords).
// Base vertex r maps to:
//   angle_sector = r * (2π / radialSegs) / √2
//   uv = (√2 * cos(angle_sector), √2 * sin(angle_sector))
// We return normalized coordinates so apex=(0.5, 0.5) and base ring lies on
// the unit circle in the sector.  Specifically:
//   apex UV = (0.5, 0.5)
//   base vertex r: phi = r * (2π/radialSegs) / slant_length
//                  uv  = (0.5 + 0.5*cos(phi), 0.5 + 0.5*sin(phi))
// Vertex ordering matches Cone(radialSegs):
//   indices 0..radialSegs-1 = base ring, index radialSegs = apex.
// Returns one UV pair per vertex.
std::vector<UV2> Cone_ExpectedUV(unsigned radialSegs);

// ============================================================
// DEFECT COUNTS for dirty generators
// ============================================================

struct DirtyDefects
{
	// How many of each defect type were injected.
	uint32_t duplicateFaces = 0; // exact pairs added
	uint32_t degenerateFaces = 0; // collinear / zero-area faces added
	uint32_t unreferencedVertices = 0; // orphan vertices appended
	uint32_t deletedFaces = 0; // faces removed to create holes
	uint32_t expectedBoundaryLoops = 0; // expected boundary loops after holes

	// Non-manifold flags
	bool hasBowTieVertex = false;
	bool hasThreeOnEdge = false;

	// Component count
	uint32_t numComponents = 1;
	// For DirtyManyComponents: number of small components that RemoveSmallComponents
	// should eliminate.  A component is "small" if its face count < smallComponentThreshold.
	uint32_t smallComponents = 0; // number of small (removable) components
	uint32_t smallComponentThreshold = 0; // minComponentSize arg to pass
	uint32_t largeComponentFaces = 0; // faces in the surviving large component
	uint32_t largeComponentVerts = 0; // vertices in the surviving large component

	// Description for documentation
	std::string description;
};

// ============================================================
// DIRTY MESH SYNTHESIZERS
// ============================================================

// Bow-tie: two triangle fans sharing vertex 0 but no edge.
// Vertex-non-manifold; edge-manifold.
// V=7 F=4. DirtyDefects: hasBowTieVertex=true.
halfmesh::Mesh DirtyBowTie(DirtyDefects* defects = nullptr);

// Three triangles sharing edge (0,1): edge-non-manifold.
// V=5 F=3. DirtyDefects: hasThreeOnEdge=true.
halfmesh::Mesh DirtyThreeOnEdge(DirtyDefects* defects = nullptr);

// Cube with N duplicate face pairs injected on top.
// DirtyDefects: duplicateFaces = 2*numDupPairs (both copies counted).
halfmesh::Mesh DirtyDuplicateFaces(unsigned numDupPairs = 3,
                                   DirtyDefects* defects = nullptr);

// Cube with N degenerate (collinear/zero-area) faces appended.
// DirtyDefects: degenerateFaces = numDegen.
halfmesh::Mesh DirtyDegenerateFaces(unsigned numDegen = 3,
                                    DirtyDefects* defects = nullptr);

// Cube with N orphan vertices appended (not referenced by any face).
// DirtyDefects: unreferencedVertices = numUnref.
halfmesh::Mesh DirtyUnreferencedVertices(unsigned numUnref = 5,
                                         DirtyDefects* defects = nullptr);

// Cube with numHoles faces deleted to create open boundary loops.
// Each deleted face opens one hole (1 boundary loop per deleted face for a
// manifold mesh with non-adjacent faces removed).
// DirtyDefects: deletedFaces = numHoles, expectedBoundaryLoops = numHoles.
halfmesh::Mesh DirtyHoles(unsigned numHoles = 2, DirtyDefects* defects = nullptr);

// Two disconnected cubes (2 components).
// DirtyDefects: numComponents = 2.
halfmesh::Mesh DirtyManyComponents(unsigned numCubes = 2,
                                   DirtyDefects* defects = nullptr);

// ============================================================
// CORPUS OVERVIEW
// ============================================================

struct CorpusEntry
{
	std::string name;
	bool isDirty = false;
	KnownTopology cleanTopo; // for clean meshes; zero for dirty
	DirtyDefects dirtyInfo; // for dirty meshes; zero for clean
};

// Returns a list of all corpus entries (name + known topology or defect info).
std::vector<CorpusEntry> makeCorpus();

} // namespace corpus
} // namespace hmtest

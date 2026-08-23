/*
* Mesh.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <halfmesh/HalfMesh.h>
#include <halfmesh/OrientedBoundingBox.h>
#include <halfmesh/Util/Maths.h>

#include <string>
#include <vector>

namespace halfmesh {

// Defines a mesh represented as a list of vertices and triangles (faces).
class Mesh
{
	public:
	typedef HalfMesh::Type Type;
	typedef HalfMesh::Vertex Vertex;
	typedef HalfMesh::VIndex VIndex;
	typedef HalfMesh::Face Face;
	typedef HalfMesh::FIndex FIndex;
	typedef HalfMesh::EIndex EIndex;
	typedef HalfMesh::HIndex HIndex;

	typedef Eigen::Matrix<Type, 2, 1> Vertex2D;

	typedef Eigen::Matrix<Type, 3, 1> Normal;
	typedef Eigen::Matrix<Type, 2, 1> TexCoord;

	typedef std::vector<FIndex> VertexFaces;
	typedef Eigen::Matrix<FIndex, 3, 1> FaceFaces;

	typedef cv::Mat_<FIndex> FaceMap;

	using real = halfmesh::real;
	using Pixel = halfmesh::Pixel;
	using Image3u = halfmesh::Image3u;

	public:
	std::vector<Vertex> vertices;
	std::vector<Face> faces;

	// optional data
	std::vector<Pixel> vertexColors; // color for each vertex
	std::vector<Normal> faceNormals; // normal for each face
	std::vector<TexCoord> faceTexcoords; // absolute texture-coordinates for each face vertex (no.faces*3)
	std::vector<FIndex> faceTexblobs; // for each face, the corresponding texture ID, or empty if only one blob (no.faces or 0)
	std::vector<Image3u> texturesDiffuse; // diffuse color images, one for each texture blob (no.blobs)

	// optional tools
	std::vector<VertexFaces> vertexFaces; // list of incident faces for each vertex (in increasing order)
	HalfMesh halfMesh; // represent mesh connectivity in half-edge format for fast adjacency queries

	public:
	void ReleaseOptional();
	bool Empty() const
	{
		ASSERT(vertices.empty() == faces.empty() || vertices.empty() == halfMesh.Empty());
		return vertices.empty();
	}
	bool HasTextureCoordinates() const { return faceTexcoords.size() == faces.size() * 3 || faceTexcoords.size() == vertices.size(); }
	bool HasTexture() const { return HasTextureCoordinates() && !texturesDiffuse.empty(); }

	// convert a textured mesh from storing the texture coordinates per face,
	// to a new equivalent mesh storing them per vertex, duplicating any
	// vertex on the edge between texture patches
	Mesh ToTexCoordPerVertex() const;
	// UV-only variant: converts per-face-corner UVs to per-vertex layout
	// without requiring a texture image (for atlas/unwrap meshes)
	Mesh ToTexCoordPerVertexUVOnly() const;
	// split a textured mesh in multiple meshes, one per texture
	std::vector<Mesh> ToOneMeshPerTexblob() const;

	// import/export PLY/GLTF mesh
	// Load dispatches on the file extension: .ply -> LoadPLY, .glb/.gltf -> LoadGLTF
	bool Load(const std::string& fileName);
	bool LoadPLY(const std::string& fileName);
	bool LoadGLTF(const std::string& fileName);
	bool Save(const std::string& fileName, bool binary = true) const;
	bool SavePLY(const std::string& fileName, bool binary = true) const;
	bool SaveGLTF(const std::string& fileName, bool binary = true) const;
	bool ExportSeamEdges(std::vector<std::pair<VIndex, VIndex>> seamEdges, const std::string& fileName, bool binary = true) const;
	bool ExportSeamEdges(const std::string& fileName, bool binary = true) const;

	// compute normal for the given face (not normalized)
	static Normal ComputeTriangleNormal(const Vertex& x0, const Vertex& x1, const Vertex& x2)
	{
		return (x1 - x0).cross(x2 - x0);
	}
	Normal ComputeFaceNormal(const Face& face) const
	{
		return ComputeTriangleNormal(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
	}
	Normal ComputeFaceNormal(FIndex idxFace) const
	{
		return ComputeFaceNormal(faces[idxFace]);
	}
	// compute normal for all faces
	void ComputeFaceNormals();
	// smoothen the face normals
	//  - maxAngle: maximum angle between neighbor normals that is allowed to be take into consideration (in degrees)
	//  - currentNormalWeight: weight to use for current normal value when averaging with neighbor normals (0..1]
	//  - iterations: number of times to repeat the smoothening process
	void ComputeSmoothFaceNormals(float maxAngle = 25.f, float currentNormalWeight = 0.3f, unsigned iterations = 5);
	// compute normal for all vertices from the face normals
	std::vector<Normal> ComputeVertexNormals();
	// compute area for a triangle defined by three 2D points
	static Type ComputeTriangleDoubleArea2D(const Vertex2D& x0, const Vertex2D& x1, const Vertex2D& x2)
	{
		const Vertex2D x01(x1 - x0);
		const Vertex2D x02(x2 - x0);
		return x02.x() * x01.y() - x02.y() * x01.x();
	}
	// compute the double area for the given face
	static Type ComputeTriangleDoubleArea(const Vertex& x0, const Vertex& x1, const Vertex& x2)
	{
		return ((x1 - x0).cross(x2 - x0)).norm();
	}
	Type ComputeFaceDoubleArea(const Face& face) const
	{
		return ComputeTriangleDoubleArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
	}
	Type ComputeFaceDoubleArea(FIndex idxFace) const
	{
		return ComputeFaceDoubleArea(faces[idxFace]);
	}
	// compute area for all faces
	real ComputeArea() const;
	// compute area for the given face indices
	real ComputeArea(const std::vector<FIndex>&) const;
	// compute the axis-aligned bounding box of the mesh vertices
	Eigen::AlignedBox<Type, 3> ComputeAABBox() const;

	// enumerate the array of triangles incident to each vertex;
	// the list of faces per vertex is stored in increasing index order
	void ListVertexFaces();
	// check that vertex-faces are correct (debug use only)
	bool ValidateVertexFaces();
	// initialize the half-edge structure to represent the mesh surface.
	// Non-manifold input is detected (the manifold-only build fails cleanly)
	// and auto-repaired via ListHalfEdgesSafe with a logged warning — a
	// geometry-preserving MUTATION of the mesh (duplicate faces reduced to one
	// copy, self-edge faces dropped, non-manifold fans/bow-ties split,
	// unreferenced vertices removed). Every half-edge consumer (Simplify,
	// RemeshIsotropic, CloseHoles, SegmentCharts, ...) therefore operates on,
	// and returns, the manifoldized mesh: identity/clamped calls on
	// non-manifold input return that repaired mesh, not the raw input.
	void ListHalfEdges();
	// initialize the half-edge structure to represent the mesh surface
	// for the general case in which the mesh might be non-manifold:
	// welds coincident vertices, removes duplicate/degenerate faces and fixes
	// non-manifold topology, then builds the half-edge structure
	void ListHalfEdgesSafe();
	// cheap O(faces) test for EDGE-manifoldness (no self-edge faces,
	// no directed-edge duplicated, no edge shared by more than two faces).
	// Necessary but NOT sufficient for the manifold-only half-edge build:
	// HalfMesh::Build additionally rejects non-manifold VERTICES (bow-ties —
	// two face fans glued at one vertex; see the single-fan walk in
	// src/HalfMesh.cpp). E.g. {(0,1,2),(0,3,4)} passes this test yet fails
	// Build. NOT called by ListHalfEdges (which detects non-manifoldness
	// inline) so it never penalizes building a known-manifold mesh
	bool IsManifold() const;
	// extract the list of faces per texture patch from the texture coordinates array;
	// return the number of texture patches
	uint32_t ListTexPatchFaces(std::vector<uint32_t>& facePatchIds) const;

	// find the adjacent vertices to the given vertex (first ring)
	std::vector<VIndex> VAdjacentVertices(VIndex) const;
	// find the vertex position in the given face, or
	// negative if face does not contain the vertex
	VIndex FVertexIdx(FIndex, VIndex) const;
	// fetch the requested face vertex by value
	VIndex FVertex(FIndex iF, VIndex iV) const
	{
		const uint32_t idx(FVertexIdx(iF, iV));
		ASSERT(idx != math::NO_ID);
		return faces[iF][idx];
	}
	VIndex& FVertex(FIndex iF, VIndex iV)
	{
		const uint32_t idx(FVertexIdx(iF, iV));
		ASSERT(idx != math::NO_ID);
		return faces[iF][idx];
	}
	// check if the two faces have the same exactly the same vertices
	bool FSameVertices(FIndex, FIndex) const;
	// get the edge orientation in the given face:
	// false - backward, true - forward
	bool FEdgeOrientation(FIndex, VIndex, VIndex) const;
	// find the adjacent face to given face edge, or
	// NO_ID if no adjacent faces exist OR more than one adjacent face exist OR have opposite orientations
	FIndex FEdgeAdjacentFace(FIndex, VIndex, VIndex) const;
	// normalize texture coordinates
	std::vector<TexCoord> FTexcoordsNormalize() const;
	// normalize and flip Y axis texture coordinates
	std::vector<TexCoord> FTexcoordsNormalizeFlipY() const;
	// unnormalize texture coordinates
	std::vector<TexCoord> FTexcoordsUnNormalize() const;
	// unnormalize and flip Y axis texture coordinates
	std::vector<TexCoord> FTexcoordsUnNormalizeFlipY() const;
	// get the texture blob corresponding to the given face
	FIndex FTexblob(FIndex i) const
	{
		ASSERT(HasTextureCoordinates());
		return faceTexblobs.empty() ? 0 : faceTexblobs[i];
	}
	// collapse the given edge, remove it, remove one of its vertices, and remove both adjacent faces
	void ECollapse(EIndex);

	// remove unreferenced vertices
	VIndex RemoveUnreferencedVertices();
	// merge spatially-coincident vertices into one and remap the faces to the
	// surviving representative (per-corner faceTexcoords are unaffected;
	// per-vertex colors are remapped).  This recovers shared connectivity from
	// formats that split/duplicate vertices at seams (e.g. glTF stores textured
	// meshes unwelded, one disconnected sub-mesh per texture).
	//  - epsilon == 0: exact bit-match (glTF/seam duplicates are bit-identical)
	//  - epsilon  > 0: snap to a grid of this cell size before matching
	// faces that collapse to a repeated vertex are NOT removed here; follow with
	// RemoveDegenerateFaces().  returns the number of vertices removed.
	VIndex RemoveDuplicateVertices(Type epsilon = 0);
	// remove specified vertices along with all faces referencing them
	// (require vertexFaces). Pass updateLists=false to SKIP the face
	// removal: only the swapped-in last vertex's faces are re-indexed, and
	// faces referencing the REMOVED vertices are left untouched -- a fast
	// path for callers that remove those faces themselves.
	void RemoveVertices(std::vector<VIndex>&, bool updateLists = true);
	// remove the selected vertices and their incident faces, then fill only the
	// closed boundary loops created by that removal using Liepa triangulation.
	// Pre-existing holes are never filled; a removed region touching an existing
	// boundary remains open. Returns the number of newly created holes filled.
	unsigned RemoveVerticesAndFill(std::vector<VIndex>);
	// remove duplicate faces (referencing the same vertices)
	//  - removeBothFaces: it is recommended that the input mesh to be manifold
	//                       and so the duplicated faces can only be isolated from
	//                       the rest of the mesh and can be both removed
	// unreferenced vertices can be created,
	// so should be followed by RemoveUnreferencedVertices()
	FIndex RemoveDuplicateFaces(bool removeBothFaces = true);
	// remove degenerate faces (with one or more identical vertices,
	// or very close vertices - disabled if thArea == 0)
	// unreferenced vertices and non-manifold edges/vertices can be created,
	// so should be followed by RemoveUnreferencedVertices() and FixNonManifold()
	FIndex RemoveDegenerateFaces(Type thArea = 1e-5f);
	// removing zero-area-faces can generate some new zero-area-faces,
	// so iterate till no zero-area faces are encountered or max number of iterations is reached
	FIndex RemoveDegenerateFaces(unsigned maxIterations, Type thArea = 1e-5f);
	// remove specified faces
	// (require vertexFaces if updateLists)
	void RemoveFaces(std::vector<FIndex>&, bool updateLists = false);
	// removes all faces outside the given oriented bounding-box
	unsigned RemoveFacesOutside(const halfmesh::OBB&);

	// find all non-manifold edges and vertices and fix topology to manifold
	// by duplicating the non-manifold vertices; assigning the new vertex to
	// the smallest connected set of faces; the mesh should be free of
	// duplicate vertices and degenerate faces;
	//  - thMoveDuplicate: displace duplicate vertices with a percentage of
	//      the average vector from the non manifold vertex to the barycenter
	//      of the incident faces (0 - duplicate only)
	//  - duplicatedVertices: return the indices of the duplicated vertices
	// return number of fixes performed
	unsigned FixNonManifold(float thMoveDuplicate = 0.01f, std::vector<VIndex>* duplicatedVertices = NULL);

	// clean the mesh by removing small components
	//  - minComponentSize: remove components with less number of faces
	unsigned RemoveSmallComponents(unsigned minComponentSize);

	// remove reconstruction debris relative to the mesh's own edge-length
	// distribution: first discard faces containing an edge longer than
	// percentile95(edgeLength)*factor, then discard connected components whose
	// bounding-box diagonal is shorter than percentile55(edgeLength)*factor.
	// return number of faces removed
	FIndex RemoveSpuriousComponents(float factor);

	// remove spike/needle vertices: a vertex incident to at most one face is not
	// part of a surface, it is either isolated or the tip of a dangling triangle.
	// Dropping such a vertex takes its incident face with it, which can starve a
	// neighbour down to a single face, so the sweep repeats until the mesh is
	// stable or maxIterations rounds have run.
	// invalidates vertexFaces and the half-edge structure; unreferenced vertices
	// are not created (the spikes themselves are removed)
	// return number of vertices removed
	unsigned RemoveSpikes(unsigned maxIterations = 100);

	// implement edge collapse mesh simplification approximating the error
	// locally for each vertex using a quadric representation;
	// the fast implementation iterates over all faces and removing
	// those with a low error, increasing the threshold for each iteration
	// until the desired face number is reached;
	//  - decimateRatio : desired face count of the simplified mesh, interpreted
	//      by magnitude: a value in (0,1) is a fraction of the input face count
	//      (0.5 -> keep half); a value > 1 is an absolute target face count
	//      (e.g. 300000 -> ~300k faces, clamped to the input count); exactly 1
	//      disables ratio/count decimation (identity), as used with minEdgeLength
	//  - minEdgeLength : desired minimum edge length of the simplified mesh (0 - disabled);
	//      decimateRatio and minEdgeLength are exclusive, only one can be defined
	//  - aggressiveness : multiplier used to increase the threshold between
	//      consecutive iterations; use smaller rate for higher quality;
	//      recommended value in [5..8] range (default 7, 0 - disabled).
	//      0 (the default) runs the exact priority-queue variant instead; on
	//      adversarial input the threshold variant can stop FARTHER from the
	//      target than the exact one (measured on a needle-fused CAD assembly)
	// An empty mesh and the identity call (decimateRatio == 1, no minEdgeLength)
	// are no-ops. Non-manifold input is first auto-repaired to manifold by the
	// half-edge build (geometry-preserving, warning logged — see
	// ListHalfEdges), so identity-equivalent and clamped absolute targets
	// return the manifoldized mesh. Decimation preserves manifold topology:
	// edges whose collapse would break it are skipped, so meshes fused by
	// needle/T-junction triangles have a reachable floor above the requested
	// target (a warning is logged); genus and boundary loops bound that floor
	// from below at extreme targets.
	// For such input run RemoveDegenerateFaces(1e-5f) + RemoveUnreferencedVertices()
	// + FixNonManifold() first — it dissolves the phantom 3-cycles that block
	// collapses (measured: target reached at +3% area vs +50% at the raw floor).
	void Simplify(float decimateRatio, float minEdgeLength = 0.f, float aggressiveness = 0.f);

	unsigned CloseHoles(unsigned nCloseHoles = 200, std::vector<std::vector<FIndex>>* holesFaces = NULL);

	// smooth the vertex positions in place using HC Laplacian smoothing:
	// "Improved Laplacian Smoothing of Noisy Surface Meshes", Vollmer, Mencl
	// and Mueller, EUROGRAPHICS 1999 — a uniform-Laplacian step followed by a
	// correction pass that pushes each vertex back toward its previous
	// position, largely avoiding the shrinkage of plain Laplacian smoothing.
	// Builds/refreshes the half-edge structure (like RemeshIsotropic, including
	// its repair of non-manifold input, which may weld/remove/add and remap
	// vertices — e.g. unreferenced vertices are removed, bow-tie vertices are
	// split); lockedVertices must index the POST-repair mesh (on manifold-clean
	// input, indices are unchanged). On manifold-clean input, faces, texture
	// coordinates and all other attributes are left untouched. Cached face
	// normals are invalidated by smoothing (vertex motion alone can't be
	// detected by their size-based freshness check).
	//  - iterations: number of smoothing steps (<= 0 is a no-op)
	//  - lockedVertices: optional per-vertex mask (size == vertices.size());
	//      true = the vertex position is held fixed, though it still
	//      contributes to its neighbors' averages
	void SmoothHCLaplacian(int iterations = 1, const std::vector<bool>* lockedVertices = NULL);

	// smooth the vertex positions in place using Taubin lambda|mu smoothing:
	// "A Signal Processing Approach To Fair Surface Design", Taubin, SIGGRAPH
	// 1995 — each iteration applies a uniform-Laplacian shrink step (lambda > 0)
	// followed by a slightly stronger inflate step (mu < -lambda), acting as a
	// band-pass filter: high-frequency noise is removed while the low-frequency
	// shape (and thus volume) is preserved. Compared to SmoothHCLaplacian it
	// smooths far more aggressively at ~zero shrinkage for the same
	// per-iteration cost, but needs more iterations (typically 10-100; its
	// low-frequency gain is slightly above 1 by design, so very long runs
	// slowly inflate — ~1%/100 iterations measured).
	// Border vertices are smoothed along the boundary curve only: their average
	// is seeded with the vertex itself and uses just the border-edge neighbors.
	// SmoothHCLaplacian instead uses the plain uniform ring on borders, so the
	// two methods deliberately differ there.
	// Build/repair, locked-vertex and face-normal-cache semantics are identical
	// to SmoothHCLaplacian above.
	//  - iterations: number of lambda+mu step pairs (<= 0 is a no-op)
	//  - lambda: positive smoothing step scale, in (0, 1)
	//  - mu: negative inflate step scale, |mu| > lambda; stable pairs satisfy
	//      |(1 - 2*lambda)*(1 - 2*mu)| <= 1 — the defaults 0.65/-0.69 are the
	//      strongest such pair
	//  - lockedVertices: optional per-vertex mask (size == vertices.size());
	//      true = the vertex position is held fixed, though it still
	//      contributes to its neighbors' averages
	void SmoothTaubin(int iterations = 1, Type lambda = Type(0.65f), Type mu = Type(-0.69f), const std::vector<bool>* lockedVertices = NULL);

	// algorithm selector for the unified Smooth() entry point below
	enum class SmoothMethod {
		Taubin, // Taubin lambda|mu band-pass (SmoothTaubin); the default
		HCLaplacian, // HC (anti-shrink) Laplacian (SmoothHCLaplacian)
	};

	// unified smoothing entry point: run `iterations` passes of the chosen
	// method, each with its own per-function default parameters (Taubin's
	// lambda/mu, no locked vertices). A convenience wrapper over SmoothTaubin /
	// SmoothHCLaplacian for callers that only need to pick an algorithm and a
	// pass count; call those directly to tune lambda/mu or lock vertices. Note
	// Taubin typically wants more iterations than HC (see SmoothTaubin).
	//  - iterations: number of smoothing steps (<= 0 is a no-op)
	//  - method: which algorithm to run (default: Taubin, the strongest
	//      smoother at ~zero shrinkage)
	void Smooth(int iterations = 1, SmoothMethod method = SmoothMethod::Taubin);

	struct RemeshParams
	{
		float edgeMinLength{0}; // no edge should be shorter than this value (used when collapsing)
		float edgeMaxLength{0}; // no edge should be longer than this value (used when refining);
		// 0 is deliberately INVALID — RemeshIsotropic validates and no-ops
		// (a non-positive threshold would split every edge forever)

		float minAdaptiveMult{1};
		float maxAdaptiveMult{1};

		float thCreaseCosAngle{std::cos(D2R(20.f))}; // min angle to be considered crease: two faces with normals diverging more than th share a crease edge

		bool checkSurfDist{false};
		float maxSurfDist{0.f};

		int iterations{3};

		// Curvature-adaptive sizing. When adapt=false (default) a single uniform
		// target edge length is used everywhere. When adapt=true, a per-vertex
		// target is derived from local curvature: high-curvature regions get
		// shorter edges, flat regions longer ones, so the same surface fidelity is
		// reached with fewer triangles. approxError is the target geometric
		// deviation (0 => derived from edgeMinLength); min/maxAdaptiveMult
		// clamp the per-vertex target to [mult*L] of the base length L (both 1 =>
		// uniform).
		bool adapt{false};
		float approxError{0.f};

		// Smoothing controls. Default is PMP-style tangential smoothing (vertex-
		// normal tangent-plane projection + area-weighted centroid, ~5 passes),
		// which markedly improves edge-length uniformity and triangle angles. Set
		// smoothTangential=false to use a single uniform-Laplacian pass instead.
		int smoothIterations{5};
		bool smoothTangential{true};
		float smoothDelta{0.2f};

		// Per-pass toggles; all on by default. Useful for ablation
		// and for callers that want only some operations.
		bool doSplit{true};
		bool doCollapse{true};
		bool doFlip{true};
		bool doSmooth{true};
		bool doProject{true};

		// Feature handling (used by tangential smoothing). Default on (PMP-style):
		// corners/junctions (a vertex with !=2 incident feature edges) are locked
		// while smooth crease vertices (exactly 2 feature edges) slide ALONG the
		// feature curve — preserving sharp edges without freezing them. Set false
		// to fully lock every crease/boundary vertex. On smooth meshes (no creases)
		// this has no effect.
		bool featureCorners{true};

		void SetEdgeLength(float length)
		{
			edgeMinLength = length * 4.f / 5.f;
			edgeMaxLength = length * 4.f / 3.f;
		}
		void SetCreaseAngle(float angleDegree)
		{
			thCreaseCosAngle = std::cos(D2R(angleDegree));
		}
		void SetMaxSurfaceDistance(float edgeLength, float lengthMul = 0.1f)
		{
			maxSurfDist = edgeLength * lengthMul;
		}
		// Enable curvature-adaptive sizing with the given geometric error tolerance
		// and per-vertex length range [minMult, maxMult] * base length.
		void SetAdaptive(float error, float minMult = 0.25f, float maxMult = 4.f)
		{
			adapt = true;
			approxError = error;
			minAdaptiveMult = minMult;
			maxAdaptiveMult = maxMult;
		}
	};
	// Operation counts + per-pass wall-time accumulated over a RemeshIsotropic run.
	struct RemeshStats
	{
		unsigned splitCount{0};
		unsigned collapseCount{0};
		unsigned flipCount{0};
		double splitSeconds{0};
		double collapseSeconds{0};
		double tagSeconds{0};
		double flipSeconds{0};
		double smoothSeconds{0};
		double projectSeconds{0};
	};

	// explicit remeshing of a triangular mesh, by repeatedly applying edge-flip, edge-collapse,
	// vertex-relax and vertex-refine operations to regularize triangles size and aspect ratio;
	// the input mesh should be manifold and have no duplicate or zero area faces.
	// If stats is non-null it receives per-operation counts.
	void RemeshIsotropic(RemeshParams params, RemeshStats* stats = nullptr);
};

} // namespace halfmesh

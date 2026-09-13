/*
* binding.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// pybind11 bindings for the halfmesh library — the native half of the
// pip-installable `halfmesh` package (see python/halfmesh/__init__.py).
// numpy-only by design: consumers using torch convert via torch.from_numpy().
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <halfmesh/Mesh.h>
#include <halfmesh/Version.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>
#include <halfmesh/RectPacking.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;

using halfmesh::Mesh;

namespace {

using VertArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using FaceArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using BoundArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
// a page size: one int for a square page, or (width, height)
using PageSizeArg = std::variant<int, std::array<int, 2>>;

// The memcpy bulk copies below require the element types to be padding-free
// scalar triples. Size-only, like the asserts guarding the same copies in
// src/MeshIO.cpp: Eigen's fixed-size matrices fail is_trivially_copyable
// (user-provided copy-assignment) even though their storage is a plain array.
static_assert(sizeof(Mesh::Vertex) == 3 * sizeof(float),
              "Mesh::Vertex must be memcpy-compatible with float[3]");
static_assert(sizeof(Mesh::Face) == 3 * sizeof(uint32_t),
              "Mesh::Face must be memcpy-compatible with uint32_t[3]");

Mesh MeshFromArrays(const VertArray& v, const FaceArray& f)
{
	if (v.ndim() != 2 || v.shape(1) != 3)
		throw py::value_error("vertices must have shape [N,3] (float32)");
	if (f.ndim() != 2 || f.shape(1) != 3)
		throw py::value_error("faces must have shape [M,3] (uint32)");
	const auto numVertices = static_cast<uint64_t>(v.shape(0));
	if (numVertices == 0 && f.shape(0) != 0)
		throw py::value_error("faces reference vertices, but vertices array is empty");
	const uint32_t* faceData = f.data();
	const size_t numFaceIndices = static_cast<size_t>(f.shape(0)) * 3;
	for (size_t i = 0; i < numFaceIndices; ++i) {
		if (faceData[i] >= numVertices)
			throw py::value_error("face index " + std::to_string(faceData[i]) + " is out of range for " + std::to_string(numVertices) + " vertices");
	}
	Mesh m;
	// Bulk-copy the buffers (memcpy compatibility static_asserted above). The
	// empty guards matter: a zero-size numpy array may hand out a null data
	// pointer, and memcpy(dst, nullptr, 0) is undefined behavior.
	m.vertices.resize(static_cast<size_t>(v.shape(0)));
	if (!m.vertices.empty())
		std::memcpy(m.vertices.data(), v.data(), sizeof(float) * 3 * m.vertices.size());
	m.faces.resize(static_cast<size_t>(f.shape(0)));
	if (!m.faces.empty())
		std::memcpy(m.faces.data(), f.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return m;
}

py::tuple ArraysFromMesh(Mesh& m)
{
	m.SyncFaces();
	py::array_t<float> v({static_cast<py::ssize_t>(m.vertices.size()), py::ssize_t(3)});
	if (!m.vertices.empty())
		std::memcpy(v.mutable_data(), m.vertices.data(), sizeof(float) * 3 * m.vertices.size());
	py::array_t<uint32_t> f({static_cast<py::ssize_t>(m.faces.size()), py::ssize_t(3)});
	if (!m.faces.empty())
		std::memcpy(f.mutable_data(), m.faces.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return py::make_tuple(std::move(v), std::move(f));
}

// Run a counting op without the GIL and return (vertices, faces, count).
template <typename Op>
py::tuple ArraysWithCount(Mesh& mesh, Op op)
{
	const auto count = [&] {
		py::gil_scoped_release release;
		return op(mesh);
	}();
	py::tuple vf = ArraysFromMesh(mesh);
	return py::make_tuple(vf[0], vf[1], count);
}

// Copy a per-vertex [N] array, never alias it: the op runs without the GIL, so it
// must not read a buffer Python could resize or free underneath it.
std::vector<float> CopyPerVertex(const BoundArray& a, py::ssize_t numVertices, const char* name)
{
	if (a.ndim() != 1 || a.shape(0) != numVertices)
		throw py::value_error(std::string(name) + " must have shape [N] matching the N vertices");
	std::vector<float> values(static_cast<size_t>(numVertices));
	if (!values.empty())
		std::memcpy(values.data(), a.data(), sizeof(float) * values.size());
	return values;
}

// Integer dtypes only: forcecasting a boolean array would read True/False as 1/0,
// and a float array would be truncated, instead of either failing.
Int64Array IntegerArray(const py::array& a, py::ssize_t ndim, const char* error)
{
	const char kind = a.dtype().kind();
	if (a.ndim() != ndim || (kind != 'i' && kind != 'u'))
		throw py::value_error(error);
	return Int64Array::ensure(a);
}

// [N,2] integer (width, height) pairs as the cv::Rect list RectPacking.h consumes.
std::vector<cv::Rect> RectsFromSizes(const py::array& sizes)
{
	const Int64Array s = IntegerArray(sizes, 2, "sizes must be an [N,2] integer array of (width, height)");
	if (s.shape(1) != 2)
		throw py::value_error("sizes must be an [N,2] integer array of (width, height)");
	std::vector<cv::Rect> rects;
	rects.reserve(static_cast<size_t>(s.shape(0)));
	const int64_t* wh = s.data();
	for (py::ssize_t i = 0; i < s.shape(0); ++i) {
		const int64_t w = wh[2 * i];
		const int64_t h = wh[2 * i + 1];
		// a negative size is a caller bug, not the "degenerate, left unpacked" a zero is
		if (w < 0 || h < 0 || w > std::numeric_limits<int>::max() || h > std::numeric_limits<int>::max())
			throw py::value_error("rect sizes must lie in [0, 2^31 - 1], got (" + std::to_string(w) + ", " + std::to_string(h) + ")");
		rects.emplace_back(0, 0, static_cast<int>(w), static_cast<int>(h));
	}
	return rects;
}

cv::Size ToSize(const PageSizeArg& size)
{
	if (const int* side = std::get_if<int>(&size))
		return cv::Size(*side, *side);
	const auto& wh = std::get<std::array<int, 2>>(size);
	return cv::Size(wh[0], wh[1]);
}

// Build the half-edge up front for an op taking an argument stated over INPUT vertex
// indices. A failed build would make the op repair and potentially remap vertices,
// silently misaddressing the argument, so require callers to repair first instead.
void RequireIndexStableBuild(Mesh& mesh, const char* name)
{
	if (mesh.faces.empty())
		return;
	bool built = false;
	{
		py::gil_scoped_release release;
		built = mesh.halfMesh.Build(mesh);
	}
	if (!built)
		throw py::value_error(std::string("input requires topology repair, so ") + name + " may no longer address its vertices; call repair() first and state it over its output");
}

// The recommended pre-pass from the Simplify header docs: dissolves the phantom
// topology that blocks collapses and makes every later half-edge build
// non-mutating.
void RepairInPlace(Mesh& m)
{
	// Same sequence and rationale as Mesh::ListHalfEdgesSafe (src/MeshRepair.cpp):
	// removeBothFaces=false keeps one copy of each duplicated face (the default
	// deletes both, removing valid surface), and thArea=0 drops only faces with a
	// repeated vertex index — auto-repair must be geometry-preserving.
	m.RemoveDuplicateVertices(0);
	m.RemoveDuplicateFaces(false);
	m.RemoveDegenerateFaces(0.f);
	m.RemoveUnreferencedVertices();
	m.FixNonManifold();
}

} // namespace

PYBIND11_MODULE(_halfmesh, m)
{
	m.doc() = "halfmesh — fast half-edge triangle mesh processing "
	          "(repair / smooth / simplify / holes / remesh / UV atlas)";
	m.def("version", []() { return std::string(halfmesh::Version()); }, "halfmesh library version string");

	m.def("repair", [](const VertArray& v, const FaceArray& f) {
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			RepairInPlace(mesh);
		}
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), "Weld duplicate vertices, drop duplicate/degenerate faces and unreferenced vertices, fix non-manifold topology.");

	m.def("smooth", [](const VertArray& v, const FaceArray& f, int iterations, const std::string& method) {
		if (method != "taubin" && method != "hc")
			throw py::value_error("smooth method must be 'taubin' or 'hc', got '" + method + "'");
		if (iterations <= 0)
			throw py::value_error("smooth iterations must be > 0");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			if (method == "taubin")
				mesh.SmoothTaubin(iterations);
			else
				mesh.SmoothHCLaplacian(iterations);
		}
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), py::arg("iterations"), py::arg("method"), "Smooth vertex positions: 'taubin' (band-pass, ~zero shrink) or 'hc' (anti-shrink Laplacian).");

	m.def("simplify", [](const VertArray& v, const FaceArray& f, float target, float aggressiveness, std::optional<BoundArray> vertexMaxError) -> py::object {
		if (target <= 0.f)
			throw py::value_error("simplify target must be > 0 (fraction in (0,1) or absolute count > 1)");
		Mesh mesh = MeshFromArrays(v, f);
		if (!vertexMaxError) {
			{
				py::gil_scoped_release release;
				mesh.Simplify(target, /*minEdgeLength=*/0.f, aggressiveness);
			}
			return ArraysFromMesh(mesh);
		}
		if (aggressiveness > 0.f)
			throw py::value_error("vertex_max_error is exact-mode only: leave aggressiveness at 0");
		// the copy also keeps the input unmutated: Simplify compacts the bound in place
		std::vector<float> bounds = CopyPerVertex(*vertexMaxError, v.shape(0), "vertex_max_error");
		RequireIndexStableBuild(mesh, "vertex_max_error");
		if (!mesh.faces.empty()) {
			py::gil_scoped_release release;
			mesh.Simplify(target, /*minEdgeLength=*/0.f, aggressiveness, bounds);
		}
		py::tuple vf = ArraysFromMesh(mesh);
		// the live prefix Simplify compacted the bound into: one entry per survivor
		bounds.resize(mesh.vertices.size());
		py::array_t<float> out(static_cast<py::ssize_t>(bounds.size()));
		if (!bounds.empty())
			std::memcpy(out.mutable_data(), bounds.data(), sizeof(float) * bounds.size());
		return py::make_tuple(vf[0], vf[1], std::move(out)); }, py::arg("vertices"), py::arg("faces"), py::arg("target"), py::arg("aggressiveness") = 0.f, py::arg("vertex_max_error") = py::none(), "QEM edge-collapse decimation. target in (0,1) = keep-fraction, > 1 = absolute face count.\n\nvertex_max_error: optional [N] float32 per-vertex bound on the collapse error, a SQUARED distance (exact mode only, so leave aggressiveness at 0). An edge collapses only while the mean squared distance of its optimal point to the planes of its merged quadric stays within the smaller bound of its endpoints; zero or less LOCKS its vertex. With target == 1 the bound alone stops the decimation (it is no longer the identity call); with a face target, whichever comes first. Passing it returns a 3-tuple (vertices, faces, vertex_max_error) whose third entry holds each surviving vertex's bound.");

	m.def("close_holes", [](const VertArray& v, const FaceArray& f, unsigned max_hole_edges) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) { return self.CloseHoles(max_hole_edges); }); }, py::arg("vertices"), py::arg("faces"), py::arg("max_hole_edges") = 30u, "Liepa hole filling (fill + refine + fair) of every hole spanned by at most max_hole_edges boundary edges. Returns (vertices, faces, closed).");

	m.def("remove_vertices_and_fill", [](const VertArray& v, const FaceArray& f, const py::array& vertex_indices) {
		const Int64Array indices = IntegerArray(vertex_indices, 1, "vertex_indices must be a 1-D integer array (for a boolean mask pass np.flatnonzero(mask))");
		Mesh mesh = MeshFromArrays(v, f);
		const auto numVertices = static_cast<int64_t>(mesh.vertices.size());
		std::vector<Mesh::VIndex> removes;
		removes.reserve(static_cast<size_t>(indices.shape(0)));
		for (const int64_t idx : std::span(indices.data(), static_cast<size_t>(indices.shape(0)))) {
			// the C++ call drops an out-of-range index silently; a Python caller hears of it
			if (idx < 0 || idx >= numVertices)
				throw py::value_error("vertex index " + std::to_string(idx) + " is out of range for " + std::to_string(numVertices) + " vertices");
			removes.push_back(static_cast<Mesh::VIndex>(idx));
		}
		RequireIndexStableBuild(mesh, "vertex_indices");
		return ArraysWithCount(mesh, [&removes](Mesh& self) { return self.RemoveVerticesAndFill(std::move(removes)); }); }, py::arg("vertices"), py::arg("faces"), py::arg("vertex_indices"), "Remove the given vertices with their incident faces, then close only the holes that removal created (Liepa triangulation, no refinement, so the vertex count always shrinks). Pre-existing holes stay open, as does a removed region reaching an existing boundary. Returns (vertices, faces, filled).");

	m.def("remove_small_components", [](const VertArray& v, const FaceArray& f, unsigned min_faces) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) {
			const unsigned removed = self.RemoveSmallComponents(min_faces);
			self.RemoveUnreferencedVertices();
			return removed;
		}); }, py::arg("vertices"), py::arg("faces"), py::arg("min_faces"), "Remove connected components with fewer than min_faces faces. Returns (vertices, faces, removed components).");

	m.def("remove_spurious_components", [](const VertArray& v, const FaceArray& f, float factor) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) {
			const Mesh::FIndex removed = self.RemoveSpuriousComponents(factor);
			self.RemoveUnreferencedVertices();
			return removed;
		}); }, py::arg("vertices"), py::arg("faces"), py::arg("factor") = 2.f, "Remove connected components whose bounding-box diagonal is shorter than percentile55(edge length) * factor, a cutoff relative to the mesh's own sampling; factor <= 0 disables. Run remove_long_edge_faces first to detach debris hanging off the surface. Returns (vertices, faces, removed faces).");

	m.def("remove_spikes", [](const VertArray& v, const FaceArray& f, unsigned max_iterations) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) { return self.RemoveSpikes(max_iterations); }); }, py::arg("vertices"), py::arg("faces"), py::arg("max_iterations") = 100u, "Remove vertices incident to at most one face (isolated vertices and dangling-triangle tips) with their face, repeating while a removal starves a neighbour down to one face, up to max_iterations rounds. Returns (vertices, faces, removed vertices).");

	m.def("remove_long_edge_faces", [](const VertArray& v, const FaceArray& f, float factor) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) {
			const Mesh::FIndex removed = self.RemoveLongEdgeFaces(factor);
			self.RemoveUnreferencedVertices();
			return removed;
		}); }, py::arg("vertices"), py::arg("faces"), py::arg("factor"), "Remove faces with an edge longer than percentile95(edge length) * factor, one global threshold over the mesh's own edge-length distribution; factor <= 0 disables. Returns (vertices, faces, removed).");

	m.def("remove_long_edge_faces_local", [](const VertArray& v, const FaceArray& f, float factor, unsigned rings) {
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) {
			const Mesh::FIndex removed = self.RemoveLongEdgeFacesLocal(factor, rings);
			self.RemoveUnreferencedVertices();
			return removed;
		}); }, py::arg("vertices"), py::arg("faces"), py::arg("factor"), py::arg("rings") = 3u, "Remove faces whose longest edge exceeds factor x the local edge scale: a vertex's scale is the median length of the edges inside its k-ring (BFS depth < rings), a face's scale the largest of its three vertex scales, so a uniformly sparse surface keeps its own scale and survives while a face spanning between denser regions does not; factor <= 0 disables. Returns (vertices, faces, removed).");

	m.def("remove_long_edge_faces_capped", [](const VertArray& v, const FaceArray& f, float factor, float reach, float cone) {
		if (!std::isfinite(factor) || !std::isfinite(reach) || !std::isfinite(cone) || cone >= 1.f)
			throw py::value_error("remove_long_edge_faces_capped needs finite factor, reach and cone with cone < 1: the face's own plane sits at exactly one probe distance");
		Mesh mesh = MeshFromArrays(v, f);
		return ArraysWithCount(mesh, [=](Mesh& self) {
			const Mesh::FIndex removed = self.RemoveLongEdgeFacesCapped(factor, reach, cone);
			self.RemoveUnreferencedVertices();
			return removed;
		}); }, py::arg("vertices"), py::arg("faces"), py::arg("factor") = 2.f, py::arg("reach") = 4.f, py::arg("cone") = 0.35f, "Remove long-edged faces (longest edge > factor x the median longest edge) that cap a cavity: probes on both sides of the centroid along the normal, at 0.5, 1, 2, ..., reach x the longest edge, hit when the nearest surface lies within cone x the probe distance. A lid across an open box or a sheet under a chassis goes; a coarsely sampled real surface has nothing behind it and stays. factor, reach or cone <= 0 disables. Returns (vertices, faces, removed).");

	m.def("remesh", [](const VertArray& v, const FaceArray& f, float edge_length, int iterations, std::optional<BoundArray> vertexSizing, bool adapt, float approx_error, float min_adaptive_mult, float max_adaptive_mult) {
		if (edge_length <= 0.f)
			throw py::value_error("remesh edge_length must be > 0");
		if (iterations <= 0)
			throw py::value_error("remesh iterations must be > 0");
		// The struct defaults the multipliers to 1/1, which clamps the curvature field
		// flat; SetAdaptive is what supplies the usable 0.25/4 range, so route through it.
		if (adapt) {
			if (approx_error < 0.f || !std::isfinite(approx_error))
				throw py::value_error("approx_error must be finite and >= 0 (0 derives it from edge_length)");
			if (!(min_adaptive_mult > 0.f) || !(max_adaptive_mult >= min_adaptive_mult))
				throw py::value_error("need 0 < min_adaptive_mult <= max_adaptive_mult");
		} else if (approx_error != 0.f) {
			// silently remeshing uniform is the one outcome a caller who asked for a
			// tolerance cannot detect from the result
			throw py::value_error("approx_error has no effect without adapt=True");
		}
		Mesh mesh = MeshFromArrays(v, f);
		Mesh::RemeshParams params;
		params.SetEdgeLength(edge_length);
		params.iterations = iterations;
		if (adapt)
			params.SetAdaptive(approx_error, min_adaptive_mult, max_adaptive_mult);
		if (!vertexSizing) {
			{
				py::gil_scoped_release release;
				mesh.RemeshIsotropic(params);
			}
			return ArraysFromMesh(mesh);
		}
		const std::vector<float> sizing = CopyPerVertex(*vertexSizing, v.shape(0), "vertex_sizing");
		// RemeshIsotropic only warns and carries on without the field; raise instead, so a
		// silently ungraded result is never what a Python caller gets back.
		for (const float len : sizing)
			if (!(len > 0.f) || !std::isfinite(len))
				throw py::value_error("vertex_sizing entries must be finite and > 0");
		RequireIndexStableBuild(mesh, "vertex_sizing");
		params.vertexSizing = sizing;
		{
			py::gil_scoped_release release;
			mesh.RemeshIsotropic(params);
		}
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), py::arg("edge_length"), py::arg("iterations") = 3, py::arg("vertex_sizing") = py::none(), py::arg("adapt") = false, py::arg("approx_error") = 0.f, py::arg("min_adaptive_mult") = 0.25f, py::arg("max_adaptive_mult") = 4.f, "Isotropic remeshing toward a uniform target edge length (world units).\n\nvertex_sizing: optional [N] float32 per-vertex TARGET edge length (world units, one entry per input vertex) replacing the uniform target, so the split, collapse and smoothing passes grade the mesh where the caller asks. Every entry must be finite and > 0. Unlike simplify's vertex_max_error it is read-only, so the return stays (vertices, faces). Being per vertex is what makes a target stated in image pixels expressible (target_edge_px / footprint_v). edge_length is still required (the passes that never consult the field read it); the field's own mean is the natural value.\n\nadapt: curvature-adaptive sizing -- high-curvature regions get shorter edges, flat ones longer, for the same fidelity at fewer triangles. approx_error is the target geometric deviation (0 derives it from edge_length) and min/max_adaptive_mult clamp the per-vertex target to that multiple of the base length. Combined with vertex_sizing the two fields INTERSECT per vertex (the finer target wins), so a caller can ask for no face coarser than its own field allows and none so coarse it leaves the surface.");

	py::class_<Mesh>(m, "Mesh",
	                 "Triangle mesh facade over halfmesh::Mesh (PLY / glTF / GLB I/O).")
	    .def(py::init<>())
	    .def_static("from_arrays", [](const VertArray& v, const FaceArray& f) { return MeshFromArrays(v, f); }, py::arg("vertices"), py::arg("faces"))
	    .def("to_arrays", [](Mesh& self) { return ArraysFromMesh(self); }, "Return (vertices float32 [N,3], faces uint32 [M,3]) copies.")
	    .def("load", [](Mesh& self, const std::string& path) {
		    bool ok;
		    {
			    py::gil_scoped_release release;
			    ok = self.Load(path);
		    }
		    if (!ok)
			    throw std::runtime_error("Mesh.load: failed to load '" + path + "'"); }, py::arg("path"), "Load a .ply / .gltf / .glb mesh (format from extension).")
	    .def("save", [](const Mesh& self, const std::string& path, bool binary) {
		    bool ok;
		    {
			    py::gil_scoped_release release;
			    ok = self.Save(path, binary);
		    }
		    if (!ok)
			    throw std::runtime_error("Mesh.save: failed to save '" + path + "'"); }, py::arg("path"), py::arg("binary") = true, "Save as .ply / .gltf / .glb (format from extension).")
	    .def_property_readonly("n_vertices", [](const Mesh& self) { return self.vertices.size(); })
	    .def_property_readonly("n_faces", [](Mesh& self) { self.SyncFaces(); return self.faces.size(); })
	    .def_property_readonly("has_texcoords", &Mesh::HasTextureCoordinates)
	    .def("__repr__", [](Mesh& self) { self.SyncFaces(); return "<halfmesh.Mesh: " + std::to_string(self.vertices.size()) + " vertices, " + std::to_string(self.faces.size()) + " faces>"; });

	m.def("unwrap", [](const std::string& input_path, const std::string& output_path, unsigned resolution, unsigned padding, bool allow_rotation, float max_cone_error, bool cut_to_disk, float max_uv_distortion, unsigned repair_carve_rings, unsigned fold_rescue_slits, float tiny_chart_side, unsigned debris_chart_faces) {
		if (resolution == 0u)
			throw py::value_error("unwrap resolution must be > 0");
		if (2u * padding >= resolution)
			throw py::value_error("unwrap needs 2*padding < resolution, or no chart fits the page");
		if (!(max_cone_error > 0.f) || !std::isfinite(max_cone_error))
			throw py::value_error("max_cone_error must be finite and > 0 (default 0.05; larger = fewer, larger charts)");
		// tau = 4 is a perfectly isometric map, the FLOOR of the measure, so a budget
		// in (0,4] is unsatisfiable: every chart reads over-distorted and bisects until
		// the mesh is one chart per triangle. 0 is "use the internal ship-ability bar",
		// not "off" -- there is no way to disable the check.
		if (!std::isfinite(max_uv_distortion) || max_uv_distortion < 0.f
		    || (max_uv_distortion > 0.f && max_uv_distortion <= 4.f))
			throw py::value_error("max_uv_distortion must be 0 (internal ship-ability bar) or > 4.0; "
			                      "4.0 is perfect isometry and ~4.4 is a quality-first budget. A value in "
			                      "(0, 4] cannot be met by any chart and splits the mesh toward one chart per triangle");
		// Each slit re-flattens the whole chart and each carve ring widens a re-split;
		// past a handful the chart is shredded and the repair's own split is cheaper.
		if (fold_rescue_slits > 16u)
			throw py::value_error("fold_rescue_slits must be <= 16 (2 is the sane on-value; each attempt re-flattens the chart)");
		if (repair_carve_rings > 16u)
			throw py::value_error("repair_carve_rings must be <= 16 (2 is the sane on-value)");
		if (!std::isfinite(tiny_chart_side) || tiny_chart_side < 0.f)
			throw py::value_error("tiny_chart_side must be finite and >= 0 (0 = off)");
		Mesh mesh;
		unsigned charts = 0;
		halfmesh::AtlasResult result;
		{
			py::gil_scoped_release release;
			if (!mesh.Load(input_path))
				throw std::runtime_error("unwrap: failed to load '" + input_path + "'");
			// Weld + clean first (examples/Unwrap.cpp preamble): unwelded input
			// makes every edge a boundary and SegmentCharts fragments into one
			// chart per face. Lossless: the atlas regenerates the UVs anyway.
			mesh.RemoveDuplicateVertices(0);
			mesh.RemoveDegenerateFaces(0.f);
			mesh.RemoveUnreferencedVertices();

			halfmesh::ParametrizeParams pparams; // defaults tuned for MVS-like meshes
			pparams.developableMaxConeError = max_cone_error;
			pparams.cutToDisk = cut_to_disk;
			pparams.developableMaxUvDistortion = max_uv_distortion;
			pparams.repairCarveRings = repair_carve_rings;
			pparams.foldRescueSlits = fold_rescue_slits;
			halfmesh::AtlasParams aparams;
			aparams.resolution = resolution;
			aparams.padding = padding;
			aparams.allowRotation = allow_rotation;
			aparams.tinyChartSide = tiny_chart_side;
			aparams.debrisChartFaces = debris_chart_faces;
			result = halfmesh::GenerateAtlas(mesh, pparams, aparams);
			charts = static_cast<unsigned>(result.chartPage.size());

			if (!mesh.Save(output_path))
				throw std::runtime_error("unwrap: failed to save '" + output_path + "'");
		}
		py::dict meta;
		meta["charts"] = charts;
		meta["pages"] = result.numPages;
		meta["width"] = result.width;
		meta["height"] = result.height;
		meta["occupancy"] = result.occupancy;
		meta["coverage"] = result.coverage;
		meta["fit_attempts"] = result.fitAttempts;
		meta["fit_scale"] = result.fitScale;
		meta["max_chart_extent"] = result.maxChartExtent;
		py::dict padding_applied;
		padding_applied["nominal"] = padding;
		padding_applied["min"] = result.minPadding;
		padding_applied["n_charts_reduced"] = result.chartsPaddingReduced;
		meta["padding_applied"] = padding_applied;
		meta["vertices"] = mesh.vertices.size();
		meta["faces"] = mesh.faces.size();
		return meta; }, py::arg("input_path"), py::arg("output_path"), py::arg("resolution") = 4096u, py::arg("padding") = 2u, py::arg("allow_rotation") = true, py::arg("max_cone_error") = 0.05f, py::arg("cut_to_disk") = false, py::arg("max_uv_distortion") = 0.f, py::arg("repair_carve_rings") = 0u, py::arg("fold_rescue_slits") = 0u, py::arg("tiny_chart_side") = 0.f, py::arg("debris_chart_faces") = 0u, "Generate a packed UV atlas: load -> weld -> GenerateAtlas -> save. Returns {charts, pages, width, height, occupancy, coverage, fit_attempts, fit_scale, max_chart_extent, padding_applied{nominal,min,n_charts_reduced}, vertices, faces}.");

	m.def("pack_rectangles", [](const py::array& sizes, const PageSizeArg& page_size, const std::string& mode, const std::optional<PageSizeArg>& max_page_size, unsigned padding, bool allow_rotation, bool power_of_two, bool square) {
		halfmesh::RectPackParams params;
		if (mode == "grow")
			params.mode = halfmesh::RectPackMode::GrowSinglePage;
		else if (mode == "single")
			params.mode = halfmesh::RectPackMode::FixedSinglePage;
		else if (mode == "multi")
			params.mode = halfmesh::RectPackMode::FixedMultiPage;
		else
			throw py::value_error("pack_rectangles mode must be 'grow', 'single' or 'multi', got '" + mode + "'");
		params.pageSize = ToSize(page_size);
		if (params.pageSize.width <= 0 || params.pageSize.height <= 0)
			throw py::value_error("page_size must be > 0");
		if (max_page_size) {
			// the fixed modes never resize the page, so a cap there would be silently ignored
			if (params.mode != halfmesh::RectPackMode::GrowSinglePage)
				throw py::value_error("max_page_size only caps mode='grow'; the fixed modes never resize the page");
			params.maxPageSize = ToSize(*max_page_size);
			if (params.maxPageSize.width < 0 || params.maxPageSize.height < 0)
				throw py::value_error("max_page_size entries must be >= 0 (0 leaves that axis unbounded)");
		}
		params.padding = padding;
		params.allowRotation = allow_rotation;
		params.powerOfTwo = power_of_two;
		params.square = square;
		const std::vector<cv::Rect> rects = RectsFromSizes(sizes);
		std::vector<halfmesh::RectPlacement> placements;
		halfmesh::RectPackResult result;
		{
			py::gil_scoped_release release;
			result = halfmesh::PackRectangles(rects, params, placements);
		}
		const auto n = static_cast<py::ssize_t>(placements.size());
		py::array_t<int32_t> outRects({n, py::ssize_t(4)});
		py::array_t<uint32_t> outPage(n);
		py::array_t<bool> outRotated(n);
		py::array_t<bool> outPacked(n);
		auto r = outRects.mutable_unchecked<2>();
		auto pg = outPage.mutable_unchecked<1>();
		auto rot = outRotated.mutable_unchecked<1>();
		auto pk = outPacked.mutable_unchecked<1>();
		for (py::ssize_t i = 0; i < n; ++i) {
			const halfmesh::RectPlacement& p = placements[static_cast<size_t>(i)];
			r(i, 0) = p.rect.x;
			r(i, 1) = p.rect.y;
			r(i, 2) = p.rect.width;
			r(i, 3) = p.rect.height;
			pg(i) = p.page;
			rot(i) = p.rotated;
			pk(i) = p.packed;
		}
		const double pageArea = static_cast<double>(result.pageSize.width) * result.pageSize.height * result.numPages;
		py::dict out;
		out["rects"] = std::move(outRects);
		out["page"] = std::move(outPage);
		out["rotated"] = std::move(outRotated);
		out["packed"] = std::move(outPacked);
		out["pages"] = result.numPages;
		out["n_packed"] = result.numPacked;
		out["width"] = result.pageSize.width;
		out["height"] = result.pageSize.height;
		out["packed_area"] = result.packedArea;
		out["occupancy"] = pageArea > 0. ? static_cast<double>(result.packedArea) / pageArea : 0.;
		return out; }, py::arg("sizes"), py::arg("page_size") = PageSizeArg(1024), py::arg("mode") = "grow", py::arg("max_page_size") = py::none(), py::arg("padding") = 2u, py::arg("allow_rotation") = true, py::arg("power_of_two") = false, py::arg("square") = false, "Pack integer (width, height) rectangles into texture pages, no mesh involved (sprite sheets, lightmaps, texture repacking); the packer unwrap() uses for charts. mode: 'grow' doubles one page until everything fits (up to max_page_size), 'single' uses one fixed page and leaves what does not fit unpacked, 'multi' opens as many fixed pages as needed. Returns {rects [N,4] int32 (x, y, w, h), page [N], rotated [N], packed [N], pages, n_packed, width, height, packed_area, occupancy}, each per-rect array in input order.");

	m.def("estimate_square_texture_size", [](const py::array& sizes, int multiple, float target_occupancy) {
		if (multiple < 0)
			throw py::value_error("multiple must be >= 0 (0 rounds up to a power of two)");
		if (!(target_occupancy > 0.f && target_occupancy <= 1.f))
			throw py::value_error("target_occupancy must lie in (0, 1]");
		const std::vector<cv::Rect> rects = RectsFromSizes(sizes);
		py::gil_scoped_release release;
		return halfmesh::EstimateSquareTextureSize(rects, multiple, target_occupancy); }, py::arg("sizes"), py::arg("multiple") = 0, py::arg("target_occupancy") = 0.9f, "Approximate the smallest square page side holding these (width, height) rectangles at target_occupancy, rounded up to a multiple of `multiple`, or to a power of two when it is 0. A starting page_size for pack_rectangles.");
}

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

#include <halfmesh/Mesh.h>
#include <halfmesh/Version.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

using halfmesh::Mesh;

namespace {

using VertArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using FaceArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

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

py::tuple ArraysFromMesh(const Mesh& m)
{
	py::array_t<float> v({static_cast<py::ssize_t>(m.vertices.size()), py::ssize_t(3)});
	if (!m.vertices.empty())
		std::memcpy(v.mutable_data(), m.vertices.data(), sizeof(float) * 3 * m.vertices.size());
	py::array_t<uint32_t> f({static_cast<py::ssize_t>(m.faces.size()), py::ssize_t(3)});
	if (!m.faces.empty())
		std::memcpy(f.mutable_data(), m.faces.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return py::make_tuple(std::move(v), std::move(f));
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

	m.def("simplify", [](const VertArray& v, const FaceArray& f, float target, float aggressiveness) {
		if (target <= 0.f)
			throw py::value_error("simplify target must be > 0 (fraction in (0,1) or absolute count > 1)");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			mesh.Simplify(target, /*minEdgeLength=*/0.f, aggressiveness);
		}
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), py::arg("target"), py::arg("aggressiveness") = 0.f, "QEM edge-collapse decimation. target in (0,1) = keep-fraction, > 1 = absolute face count.");

	m.def("close_holes", [](const VertArray& v, const FaceArray& f, unsigned max_hole_edges) {
		Mesh mesh = MeshFromArrays(v, f);
		unsigned closed = 0;
		{
			py::gil_scoped_release release;
			closed = mesh.CloseHoles(max_hole_edges);
		}
		py::tuple vf = ArraysFromMesh(mesh);
		return py::make_tuple(vf[0], vf[1], closed); }, py::arg("vertices"), py::arg("faces"), py::arg("max_hole_edges") = 30u, "Liepa hole filling (fill + refine + fair) of every hole spanned by at most max_hole_edges boundary edges.");

	m.def("remove_small_components", [](const VertArray& v, const FaceArray& f, unsigned min_faces) {
		Mesh mesh = MeshFromArrays(v, f);
		unsigned removed = 0;
		{
			py::gil_scoped_release release;
			removed = mesh.RemoveSmallComponents(min_faces);
			mesh.RemoveUnreferencedVertices();
		}
		py::tuple vf = ArraysFromMesh(mesh);
		return py::make_tuple(vf[0], vf[1], removed); }, py::arg("vertices"), py::arg("faces"), py::arg("min_faces"), "Remove connected components with fewer than min_faces faces.");

	m.def("remesh", [](const VertArray& v, const FaceArray& f, float edge_length, int iterations) {
		if (edge_length <= 0.f)
			throw py::value_error("remesh edge_length must be > 0");
		if (iterations <= 0)
			throw py::value_error("remesh iterations must be > 0");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			Mesh::RemeshParams params;
			params.SetEdgeLength(edge_length);
			params.iterations = iterations;
			mesh.RemeshIsotropic(params);
		}
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), py::arg("edge_length"), py::arg("iterations") = 3, "Isotropic remeshing toward a uniform target edge length (world units).");

	py::class_<Mesh>(m, "Mesh",
	                 "Triangle mesh facade over halfmesh::Mesh (PLY / glTF / GLB I/O).")
	    .def(py::init<>())
	    .def_static("from_arrays", [](const VertArray& v, const FaceArray& f) { return MeshFromArrays(v, f); }, py::arg("vertices"), py::arg("faces"))
	    .def("to_arrays", [](const Mesh& self) { return ArraysFromMesh(self); }, "Return (vertices float32 [N,3], faces uint32 [M,3]) copies.")
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
	    .def_property_readonly("n_faces", [](const Mesh& self) { return self.faces.size(); })
	    .def_property_readonly("has_texcoords", &Mesh::HasTextureCoordinates)
	    .def("__repr__", [](const Mesh& self) { return "<halfmesh.Mesh: " + std::to_string(self.vertices.size()) + " vertices, " + std::to_string(self.faces.size()) + " faces>"; });

	m.def("unwrap", [](const std::string& input_path, const std::string& output_path, unsigned resolution, unsigned padding, bool allow_rotation) {
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
			halfmesh::AtlasParams aparams;
			aparams.resolution = resolution;
			aparams.padding = padding;
			aparams.allowRotation = allow_rotation;
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
		meta["fit_attempts"] = result.fitAttempts;
		meta["vertices"] = mesh.vertices.size();
		meta["faces"] = mesh.faces.size();
		return meta; }, py::arg("input_path"), py::arg("output_path"), py::arg("resolution") = 4096u, py::arg("padding") = 4u, py::arg("allow_rotation") = true, "Generate a packed UV atlas: load -> weld -> GenerateAtlas -> save. "
	                                                                                                                                                                                                                                                                                            "Returns {charts, pages, width, height, occupancy, fit_attempts, vertices, faces}.");
}

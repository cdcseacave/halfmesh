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

#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

using halfmesh::Mesh;

namespace {

using VertArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using FaceArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

Mesh MeshFromArrays(const VertArray& v, const FaceArray& f)
{
	if (v.ndim() != 2 || v.shape(1) != 3)
		throw py::value_error("vertices must have shape [N,3] (float32)");
	if (f.ndim() != 2 || f.shape(1) != 3)
		throw py::value_error("faces must have shape [M,3] (uint32)");
	Mesh m;
	// Mesh::Vertex / Mesh::Face are static_asserted memcpy-compatible
	// (src/MeshIO.cpp), so bulk-copy the buffers.
	m.vertices.resize(static_cast<size_t>(v.shape(0)));
	std::memcpy(m.vertices.data(), v.data(), sizeof(float) * 3 * m.vertices.size());
	m.faces.resize(static_cast<size_t>(f.shape(0)));
	std::memcpy(m.faces.data(), f.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return m;
}

py::tuple ArraysFromMesh(const Mesh& m)
{
	py::array_t<float> v({static_cast<py::ssize_t>(m.vertices.size()), py::ssize_t(3)});
	std::memcpy(v.mutable_data(), m.vertices.data(), sizeof(float) * 3 * m.vertices.size());
	py::array_t<uint32_t> f({static_cast<py::ssize_t>(m.faces.size()), py::ssize_t(3)});
	std::memcpy(f.mutable_data(), m.faces.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return py::make_tuple(std::move(v), std::move(f));
}

// The recommended pre-pass from the Simplify header docs: dissolves the phantom
// topology that blocks collapses and makes every later half-edge build
// non-mutating.
void RepairInPlace(Mesh& m)
{
	m.RemoveDuplicateVertices(0);
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
		return ArraysFromMesh(mesh); }, py::arg("vertices"), py::arg("faces"), "Weld duplicates, drop degenerate faces and unreferenced vertices, fix non-manifold topology.");

	m.def("smooth", [](const VertArray& v, const FaceArray& f, int iterations, const std::string& method) {
		if (method != "taubin" && method != "hc")
			throw py::value_error("smooth method must be 'taubin' or 'hc', got '" + method + "'");
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

	m.def("close_holes", [](const VertArray& v, const FaceArray& f, unsigned max_holes) {
		Mesh mesh = MeshFromArrays(v, f);
		unsigned closed = 0;
		{
			py::gil_scoped_release release;
			closed = mesh.CloseHoles(max_holes);
		}
		py::tuple vf = ArraysFromMesh(mesh);
		return py::make_tuple(vf[0], vf[1], closed); }, py::arg("vertices"), py::arg("faces"), py::arg("max_holes") = 200u, "Liepa hole filling (fill + refine + fair), smallest holes first; max_holes is a count.");

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
}

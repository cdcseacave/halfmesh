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

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_halfmesh, m)
{
	m.doc() = "halfmesh — fast half-edge triangle mesh processing "
	          "(repair / smooth / simplify / holes / remesh / UV atlas)";
	m.def("version", []() { return std::string(halfmesh::Version()); }, "halfmesh library version string");
}

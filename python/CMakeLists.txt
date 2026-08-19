find_package(pybind11 CONFIG REQUIRED)

# Target name == module name: pybind11_add_module derives the PyInit__halfmesh
# entry point and the _halfmesh.<abi>.so output name from it.
pybind11_add_module(_halfmesh binding.cpp)
target_link_libraries(_halfmesh PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(_halfmesh)

# COMPONENT python: pyproject.toml installs only this component into the wheel,
# so the static lib / headers / CMake config never leak into site-packages.
install(TARGETS _halfmesh DESTINATION halfmesh COMPONENT python)

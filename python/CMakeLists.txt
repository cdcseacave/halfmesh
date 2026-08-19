find_package(pybind11 CONFIG REQUIRED)

# Target name == module name: pybind11_add_module derives the PyInit__halfmesh
# entry point and the _halfmesh.<abi>.so output name from it.
pybind11_add_module(_halfmesh binding.cpp)
target_link_libraries(_halfmesh PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(_halfmesh)

# The module statically links vcpkg's C libraries (zlib, libpng, libjpeg, ...);
# without this, ld exports every one of their symbols as a dynamic symbol of
# the extension module (~1700 of them), risking symbol interposition with
# other extension modules in the same process that bundle the same libraries.
# --exclude-libs,ALL keeps those symbols hidden while leaving halfmesh's own
# PyInit__halfmesh entry point (and pybind11's required exports) visible.
target_link_options(_halfmesh PRIVATE "LINKER:--exclude-libs,ALL")

# COMPONENT python: pyproject.toml installs only this component into the wheel,
# so the static lib / headers / CMake config never leak into site-packages.
install(TARGETS _halfmesh DESTINATION halfmesh COMPONENT python)

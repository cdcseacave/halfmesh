# pybind11 comes from pip (a [build-system] requirement), not vcpkg.
# Under scikit-build-core the pip package's CMake dir is already on
# CMAKE_PREFIX_PATH (pybind11 registers a cmake.prefix entry point); for a
# raw `cmake -DHALFMESH_BUILD_PYTHON=ON` dev build, ask the interpreter
# (`pip install pybind11` first).
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
	# Development.Module (not just Interpreter): with Python3 pre-found,
	# pybind11's config skips its own find_package and needs
	# python3_add_library, which only that component defines.
	find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
	execute_process(
		COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
		OUTPUT_VARIABLE pybind11_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET)
endif()
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
# GNU/ELF-linker specific (ld64 and MSVC link don't know the flag), so guard
# it: on macOS/Windows pybind11_add_module's default-hidden visibility is the
# (weaker) fallback — those platforms have no wheel/CI coverage today anyway.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_link_options(_halfmesh PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# COMPONENT python: pyproject.toml installs only this component into the wheel,
# so the static lib / headers / CMake config never leak into site-packages.
install(TARGETS _halfmesh DESTINATION halfmesh COMPONENT python)

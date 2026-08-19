cmake_minimum_required(VERSION 3.21)

# Load VCPKG toolchain automatically if VCPKG_ROOT is set and no toolchain was
# explicitly passed.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
	if(NOT DEFINED VCPKG_TARGET_TRIPLET AND DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
		set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
	endif()
endif()

# Options (must be declared before project() so VCPKG_MANIFEST_FEATURES can be
# set before the toolchain consumes them during the project() call).
option(HALFMESH_BUILD_TESTS "Build unit tests" OFF)
option(HALFMESH_BUILD_TOOLS "Build example CLIs" OFF)
option(HALFMESH_BUILD_PYTHON "Build the Python bindings module (pybind11)" OFF)

# Performance regression harness (docs/TESTING.md §7 perf).
# OFF by default — excluded from the fast ctest run.
# Build: cmake -DHALFMESH_BUILD_PERF=ON -DCMAKE_BUILD_TYPE=Release
# Run:   ctest --test-dir <build> -L perf   (or run perf_harness directly)
option(HALFMESH_BUILD_PERF "Build performance regression harness (slow — OFF by default)" OFF)

# Third-party C++ cross-checks against libigl (docs/TESTING.md §1 layer 3).
# Dev/CI-internal only — libigl is NOT pulled into the default build.
option(HALFMESH_BUILD_CROSSCHECKS "Build libigl cross-check tests (dev/CI-internal)" OFF)

# UV-atlas benchmark harness (tests/bench/atlasbench) — compares halfmesh's
# segmentation / parametrization / packing against SOTA baselines using the
# shared halfmesh_metrics library.  OFF by default; not in the fast ctest run.
# Build: cmake -DHALFMESH_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
#   xatlas is vendored (always on); the heavier baselines are opt-in sub-flags
#   so the harness still builds when a given baseline is absent.
option(HALFMESH_BUILD_BENCH "Build the atlasbench UV-atlas benchmark harness" OFF)
option(HALFMESH_BENCH_WITH_LIBIGL "atlasbench: enable the libigl parametrization baseline" OFF)
option(HALFMESH_BENCH_WITH_PMP    "atlasbench: enable the pmp-library parametrization baseline" OFF)
option(HALFMESH_BENCH_WITH_BFF    "atlasbench: enable the BFF/geometry-central parametrization baseline" OFF)
option(HALFMESH_BENCH_WITH_CGAL   "atlasbench: enable the CGAL segmentation/parametrization baseline" OFF)
# remeshbench (tests/bench/RemeshBench) isotropic-remeshing baseline (opt-in):
option(HALFMESH_REMESH_WITH_PMP   "remeshbench: compare against pmp::uniform_remeshing (links libpmp)" OFF)

# Wire vcpkg manifest features from the build options.
if(HALFMESH_BUILD_TESTS OR HALFMESH_BUILD_PERF)
	list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()
if(HALFMESH_BUILD_TOOLS)
	list(APPEND VCPKG_MANIFEST_FEATURES "tools")
endif()
if(HALFMESH_BUILD_PYTHON)
	# No vcpkg feature: pybind11 comes from pip, not vcpkg — the vcpkg port
	# depends on vcpkg's own from-source CPython (openssl, libffi, an autotools
	# chain, ~20 extra ports) just to provide a header-only library the Python
	# build environment already has. See python/CMakeLists.txt for resolution.
	# The extension is a shared module, so every static archive linked into it
	# (halfmesh itself and the vcpkg deps: opencv, tinyply, libjpeg, libpng, zlib)
	# must be PIC. The overlay triplet builds the vcpkg side with
	# -DCMAKE_POSITION_INDEPENDENT_CODE=ON; the halfmesh target is handled below.
	# Note: the triplet is CACHEd, so toggling HALFMESH_BUILD_PYTHON back OFF in
	# a reused build directory keeps building deps as x64-linux-pic (harmless —
	# only adds -fPIC — but use a fresh build dir to return to plain x64-linux).
	if(NOT DEFINED VCPKG_TARGET_TRIPLET AND NOT DEFINED ENV{VCPKG_DEFAULT_TRIPLET}
	   AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
		set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets" CACHE STRING "")
		set(VCPKG_TARGET_TRIPLET "x64-linux-pic" CACHE STRING "")
	endif()
endif()
if(HALFMESH_BUILD_CROSSCHECKS)
	list(APPEND VCPKG_MANIFEST_FEATURES "crosschecks")
endif()
# The bench harness uses GTest (sanity tests) → pull the 'tests' feature.
if(HALFMESH_BUILD_BENCH)
	list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()
# The libigl bench baseline reuses the existing 'crosschecks' feature (libigl).
if(HALFMESH_BENCH_WITH_LIBIGL)
	list(APPEND VCPKG_MANIFEST_FEATURES "crosschecks")
endif()
# CGAL / BFF baselines need their own heavier deps (see vcpkg.json 'bench' feature).
if(HALFMESH_BENCH_WITH_CGAL OR HALFMESH_BENCH_WITH_BFF)
	list(APPEND VCPKG_MANIFEST_FEATURES "bench")
endif()

# Include local CMake helpers. CMAKE_CURRENT_SOURCE_DIR is set before project()
# so an absolute path works on all CMake versions >= 3.21.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils.cmake")

project(halfmesh VERSION 0.2.0 LANGUAGES CXX)

# vcpkg.json cannot read the version above, so it carries a manual copy —
# fail the configure the moment the two drift apart (a version bump must
# touch both files; the Python package version is regex-extracted from the
# project() line by scikit-build-core, so it needs no check).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json" _halfmesh_vcpkg_json)
string(REGEX MATCH "\"version\"[ \t]*:[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${_halfmesh_vcpkg_json}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
	message(FATAL_ERROR "vcpkg.json version (${CMAKE_MATCH_1}) does not match project() version (${PROJECT_VERSION}) — update vcpkg.json")
endif()
unset(_halfmesh_vcpkg_json)

# AddressSanitizer + UndefinedBehaviorSanitizer build (docs/TESTING.md §7).
# Declared after project() because it controls compiler flags (not vcpkg manifest
# features), so the toolchain is already known and the option does not need to
# influence VCPKG_MANIFEST_FEATURES.
# Adds -fsanitize=address,undefined to the library and all test binaries.
# Use: cmake -DHALFMESH_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
option(HALFMESH_SANITIZE "Enable ASan + UBSan instrumentation" OFF)

# Verbose atlas-pipeline diagnostics (cone-Lloyd / flip-repair / flatten init
# tallies on stderr). Off by default and gated at compile time: when OFF the
# logging — and the env lookups/counters that drove it — are not compiled in, so
# the pipeline carries zero diagnostic overhead. Use: -DHALFMESH_ATLAS_DEBUG=ON.
option(HALFMESH_ATLAS_DEBUG "Compile in verbose atlas segmentation/flatten diagnostics" OFF)

# -----------------------------------------------------------------------
# Opt-in performance build flags (all OFF by default).
# The default build MUST stay bit-identical: the golden fixtures compare exact
# output and are build-flag sensitive, so these are never auto-enabled. They are
# intended for the perf / bench / tools builds where the exact goldens do not run.
# -----------------------------------------------------------------------
# Target-CPU SIMD. Lets Eigen's fixed-size float kernels (quadrics, cone
# covariance moments, kd-tree distances, 2x2/3x3 solves) use AVX2+FMA instead of
# the x86-64 SSE2 baseline. Applied as PUBLIC compile options (below) because
# Eigen types cross the public headers, so mixing arch levels across the
# library/consumer boundary is ABI-unsafe. Either flag changes floating-point
# results, so keep them OFF whenever the exact-equality goldens run.
#   -DHALFMESH_NATIVE=ON    -> -march=native  (tune for this machine)
#   -DHALFMESH_ARCH=<isa>   -> -march=<isa>    (e.g. x86-64-v3 for portable AVX2)
option(HALFMESH_NATIVE "Build for the host CPU (-march=native). Opt-in; breaks bit-identical goldens" OFF)
set(HALFMESH_ARCH "" CACHE STRING "Target CPU architecture for -march=<arch> (e.g. x86-64-v3); empty = compiler default")

# Interprocedural optimization (LTO/IPO). Inlines hot cross-TU calls — kd-tree
# nearest/ray queries, half-edge ESplit/EFlip/ERemove, bake sampling — across the
# static-library boundary into the example/test binaries. Applied project-wide on
# the optimized configs so the consumer link actually runs LTO. Opt-in; may shift
# float rounding, so keep OFF whenever the exact-equality goldens run.
option(HALFMESH_LTO "Enable interprocedural optimization / LTO on optimized builds. Opt-in" OFF)

# C++20 is required throughout.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release when using a single-config generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Build type: None, Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

# Apply the opt-in interprocedural optimization (HALFMESH_LTO). Set the config-
# scoped CMAKE_INTERPROCEDURAL_OPTIMIZATION_* variables BEFORE any target is
# defined so the library AND its consumers (examples/tests) inherit the property
# and the final links run LTO. check_ipo_supported degrades gracefully on
# toolchains that cannot honor it.
if(HALFMESH_LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT _hm_ipo_ok OUTPUT _hm_ipo_msg LANGUAGES CXX)
	if(_hm_ipo_ok)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
		message(STATUS "halfmesh: LTO/IPO enabled for Release/RelWithDebInfo")
	else()
		message(WARNING "halfmesh: HALFMESH_LTO requested but IPO is unavailable: ${_hm_ipo_msg}")
	endif()
endif()

# Install directories (GNUInstallDirs gives us lib/cmake, include, etc.).
include(GNUInstallDirs)
set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/halfmesh")

# -----------------------------------------------------------------------
# Find dependencies
# -----------------------------------------------------------------------
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tinyply CONFIG REQUIRED)
# tinygltf is header-only; vcpkg installs tiny_gltf.h directly into the
# include path — no CMake config/target is shipped by the port.
find_path(TINYGLTF_INCLUDE_DIR tiny_gltf.h REQUIRED)
# bshoshany-thread-pool is header-only; the vcpkg port installs only the header
# (no CMake config target), like tinygltf. PRIVATE: used by the parallel setup in
# MeshRemesh.cpp and MeshSimplify.cpp.
find_path(BS_THREAD_POOL_INCLUDE_DIR BS_thread_pool.hpp REQUIRED)

# -----------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------
add_library(halfmesh
	src/Version.cpp
	src/Log.cpp
	src/HalfMesh.cpp
	src/TriangleKDTree.cpp
	src/TriangleBVH.cpp
	src/Mesh.cpp
	src/MeshIO.cpp
	src/MeshRepair.cpp
	src/MeshSimplify.cpp
	src/MeshRemesh.cpp
	src/MeshRemeshShared.cpp
	src/MeshHoles.cpp
	src/MeshSmooth.cpp
	src/Parametrize.cpp
	src/AtlasCharting.cpp
	src/AtlasPacking.cpp
	src/TextureBake.cpp
	src/TinyGLTFImpl.cpp
)
add_library(halfmesh::halfmesh ALIAS halfmesh)

target_compile_features(halfmesh PUBLIC cxx_std_20)

# Single source of truth for the version string: inject PROJECT_VERSION into
# Version.cpp so the runtime Version() can never drift from the package version
# (it was previously hardcoded in both places).
set_source_files_properties(src/Version.cpp PROPERTIES
	COMPILE_DEFINITIONS HALFMESH_VERSION_STRING=\"${PROJECT_VERSION}\")

target_include_directories(halfmesh
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(halfmesh
	PUBLIC
		Eigen3::Eigen
		${OpenCV_LIBS}
		tinyply        # tinyply exports the 'tinyply' target (confirmed from tinyplyTargets.cmake)
)

# tinygltf is header-only; expose its include path as a usage requirement.
target_include_directories(halfmesh PUBLIC $<BUILD_INTERFACE:${TINYGLTF_INCLUDE_DIR}>)
target_include_directories(halfmesh PRIVATE ${BS_THREAD_POOL_INCLUDE_DIR})

halfmesh_set_warnings(halfmesh)

# Compile in the verbose atlas diagnostics when HALFMESH_ATLAS_DEBUG=ON. PRIVATE:
# the HM_ATLAS_DEBUG macro is referenced only inside the library's own .cpp files
# (AtlasCharting.cpp, Parametrize.cpp), never in a public header, so it must not
# leak as a usage requirement onto consumers.
if(HALFMESH_ATLAS_DEBUG)
    target_compile_definitions(halfmesh PRIVATE HM_ATLAS_DEBUG)
    message(STATUS "halfmesh: verbose atlas diagnostics enabled (HM_ATLAS_DEBUG)")
endif()

# Apply sanitizer flags to the library target when HALFMESH_SANITIZE=ON.
# The same interface flags propagate to all linked test executables via INTERFACE
# usage requirements; no per-test repetition needed.
if(HALFMESH_SANITIZE)
    if(MSVC)
        # MSVC does not accept the GNU spelling and ships no UBSan; /fsanitize=address
        # is the closest equivalent. Note ASan and /RTC are mutually exclusive, and the
        # Debug CRT is not instrumented, so use a Release-ish config on Windows.
        set(_san_compile "/fsanitize=address")
        set(_san_link "")
        message(STATUS "halfmesh: ASan enabled (MSVC /fsanitize=address; UBSan unavailable)")
    else()
        set(_san_compile "-fsanitize=address,undefined" "-fno-omit-frame-pointer" "-g")
        set(_san_link ${_san_compile})
        message(STATUS "halfmesh: ASan+UBSan enabled")
    endif()
    target_compile_options(halfmesh PUBLIC ${_san_compile})
    if(_san_link)
        target_link_options(halfmesh PUBLIC ${_san_link})
    endif()
endif()

# Apply the opt-in target-CPU SIMD flags (HALFMESH_NATIVE / HALFMESH_ARCH).
# PUBLIC because Eigen types cross the public headers, so the arch level must
# match on both sides of the library/consumer boundary (mixed-arch ABI is
# unsafe). HALFMESH_ARCH (explicit ISA) takes precedence over HALFMESH_NATIVE.
if(HALFMESH_NATIVE OR HALFMESH_ARCH)
    if(MSVC)
        # MSVC has no -march; AVX2 is the closest portable analogue.
        target_compile_options(halfmesh PUBLIC /arch:AVX2)
        message(STATUS "halfmesh: SIMD arch enabled (/arch:AVX2)")
    else()
        if(HALFMESH_ARCH)
            set(_hm_arch_flag "-march=${HALFMESH_ARCH}")
        else()
            set(_hm_arch_flag "-march=native")
        endif()
        target_compile_options(halfmesh PUBLIC ${_hm_arch_flag})
        message(STATUS "halfmesh: SIMD arch enabled (${_hm_arch_flag})")
    endif()
endif()

# -----------------------------------------------------------------------
# Python bindings (pybind11 module halfmesh._halfmesh)
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_PYTHON)
	set_property(TARGET halfmesh PROPERTY POSITION_INDEPENDENT_CODE ON)
	add_subdirectory(python)
endif()

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_TESTS OR HALFMESH_BUILD_PERF)
	enable_testing()
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
		add_subdirectory(tests)
	endif()
endif()

# -----------------------------------------------------------------------
# Third-party cross-checks (libigl) — docs/TESTING.md §1 layer 3
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_CROSSCHECKS)
	enable_testing()
	find_package(GTest CONFIG REQUIRED)
	find_package(libigl CONFIG REQUIRED)
	include(GoogleTest)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/crosscheck/CMakeLists.txt")
		add_subdirectory(tests/crosscheck)
	endif()
endif()

# -----------------------------------------------------------------------
# UV-atlas benchmark harness (tests/bench) — halfmesh vs SOTA baselines
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_BENCH)
	enable_testing()
	find_package(GTest CONFIG REQUIRED)
	include(GoogleTest)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench/CMakeLists.txt")
		add_subdirectory(tests/bench)
	endif()
endif()

# -----------------------------------------------------------------------
# Examples / tools
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_TOOLS)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
		add_subdirectory(examples)
	endif()
endif()

# -----------------------------------------------------------------------
# Install & export
# -----------------------------------------------------------------------
install(TARGETS halfmesh
	EXPORT halfmeshTargets
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY include/halfmesh
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	# repo-internal AI-agent notes are not part of the public API surface
	PATTERN "AGENTS.md" EXCLUDE
)

install(EXPORT halfmeshTargets
	NAMESPACE halfmesh::
	DESTINATION "${INSTALL_CMAKE_DIR}"
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/halfmeshConfig.cmake.in"
	"${PROJECT_BINARY_DIR}/halfmeshConfig.cmake"
	INSTALL_DESTINATION "${INSTALL_CMAKE_DIR}"
)

write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/halfmeshConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

install(FILES
	"${PROJECT_BINARY_DIR}/halfmeshConfig.cmake"
	"${PROJECT_BINARY_DIR}/halfmeshConfigVersion.cmake"
	DESTINATION "${INSTALL_CMAKE_DIR}"
)

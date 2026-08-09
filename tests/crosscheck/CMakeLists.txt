# ---------------------------------------------------------------------------
# tests/crosscheck/CMakeLists.txt — libigl cross-check tests
#
# Built ONLY when HALFMESH_BUILD_CROSSCHECKS=ON.
# GTest, libigl, halfmesh_metrics, halfmesh_corpus are already found/declared
# by the parent CMakeLists.txt before add_subdirectory(tests/crosscheck).
# ---------------------------------------------------------------------------

# halfmesh_metrics + halfmesh_corpus are built in the tests/ subtree when
# HALFMESH_BUILD_TESTS=ON, but crosschecks may be built WITHOUT the tests
# subtree.  Build them here as needed, guarding against double-definition.
if(NOT TARGET halfmesh_metrics)
    add_library(halfmesh_metrics STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../metrics/Metrics.cpp
    )
    target_include_directories(halfmesh_metrics PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../metrics
    )
    target_link_libraries(halfmesh_metrics PUBLIC
        halfmesh::halfmesh
    )
endif()

if(NOT TARGET halfmesh_corpus)
    add_library(halfmesh_corpus STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../corpus/Corpus.cpp
    )
    target_include_directories(halfmesh_corpus PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../corpus
    )
    target_link_libraries(halfmesh_corpus PUBLIC
        halfmesh::halfmesh
    )
endif()

# ---------------------------------------------------------------------------
# raytri.c reproducibility fix
#
# igl::qslim → AABB.h → ray_mesh_intersect.cpp includes "raytri.c" with a
# relative (quote-style) include.  The vcpkg libigl port does NOT install
# raytri.c into the include/igl/ directory (packaging gap).  On a fresh CI
# checkout this causes a fatal compile error.
#
# Fix: at configure time, copy the vendored raytri.c into the build-tree's
# vcpkg installed include/igl/ directory.  The build tree is per-checkout,
# so this does NOT modify any shared or system-wide installation.
# The vendored source (tests/crosscheck/vendor/igl/raytri.c) is committed and
# is the source of truth; vcpkg buildtrees are NOT required.
# ---------------------------------------------------------------------------
get_target_property(_igl_includes igl::igl_core INTERFACE_INCLUDE_DIRECTORIES)
if(_igl_includes)
    # _igl_includes may be a list; grab the first real path (the igl root)
    foreach(_idir IN LISTS _igl_includes)
        if(EXISTS "${_idir}/igl/AABB.h")
            set(_igl_root_include "${_idir}")
            break()
        endif()
    endforeach()
endif()
if(NOT _igl_root_include)
    # fallback: vcpkg conventional layout relative to binary dir
    set(_igl_root_include "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include")
endif()
set(_raytri_dest "${_igl_root_include}/igl/raytri.c")
if(NOT EXISTS "${_raytri_dest}")
    message(STATUS "halfmesh crosscheck: copying vendored raytri.c → ${_raytri_dest}")
    file(COPY_FILE
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/igl/raytri.c"
        "${_raytri_dest}"
    )
else()
    message(STATUS "halfmesh crosscheck: raytri.c already present at ${_raytri_dest}")
endif()
unset(_igl_root_include)
unset(_igl_includes)
unset(_raytri_dest)

# Shared PRE_TEST discovery mode (mirrors tests/CMakeLists.txt).
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

# ---------------------------------------------------------------------------
# igl_crosscheck_test — all four cross-checks in one binary.
# ---------------------------------------------------------------------------
add_executable(igl_crosscheck_test
    IglCrosscheckTest.cpp
)

target_include_directories(igl_crosscheck_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../metrics
    ${CMAKE_CURRENT_SOURCE_DIR}/../corpus
)

target_link_libraries(igl_crosscheck_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
    halfmesh_metrics
    halfmesh_corpus
    igl::igl_core
)

# LABELS "crosscheck" so the CI crosscheck job's `ctest -L crosscheck` actually
# selects these tests (without it, -L crosscheck matches nothing → false green).
gtest_discover_tests(igl_crosscheck_test PROPERTIES LABELS "crosscheck")

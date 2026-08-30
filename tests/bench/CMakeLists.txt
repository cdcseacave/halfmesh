# ---------------------------------------------------------------------------
# tests/bench/CMakeLists.txt — atlasbench UV-atlas benchmark harness
#
# Built ONLY when HALFMESH_BUILD_BENCH=ON.  Produces:
#   - atlasbench         : the comparison CLI
#   - atlasbench_sanity_test : GTest self-checks (identity / determinism / self-report)
#
# xatlas is vendored (always compiled).  The heavier baselines (libigl, pmp,
# BFF, CGAL) are opt-in via HALFMESH_BENCH_WITH_* so the harness still builds
# when a baseline is absent.  Each enabled baseline appends its sources to
# BENCH_ENGINE_SOURCES, its libs to BENCH_ENGINE_LIBS, and a compile define.
# ---------------------------------------------------------------------------

# Defer gtest discovery to `ctest` (PRE_TEST) instead of a POST_BUILD step, which
# can silently drop atlasbench_sanity_test from the suite under a parallel build
# (`cmake --build -j`) if it misses the default 5s discovery timeout. Mirrors the
# same fix in tests/CMakeLists.txt and tests/crosscheck/CMakeLists.txt — this
# subtree can be added standalone (HALFMESH_BUILD_BENCH=ON without
# HALFMESH_BUILD_TESTS), so it needs its own copy rather than inheriting theirs.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

# halfmesh_metrics may already be defined by the tests/ subtree; guard it.
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

# Core harness library sources (everything except the CLI main) + the always-on
# xatlas baseline (vendored, zero external deps).
set(BENCH_LIB_SOURCES
    BenchMetrics.cpp
    BenchReport.cpp
    engines/EngineHalfmesh.cpp
    engines/EngineXatlas.cpp
    engines/ParamIsolation.cpp
    vendor/xatlas/xatlas.cpp
)

# Opt-in baseline engines accumulate here.
set(BENCH_ENGINE_SOURCES "")
set(BENCH_ENGINE_LIBS "")
set(BENCH_ENGINE_INCLUDES "")
set(BENCH_ENGINE_DEFINES "HMBENCH_WITH_XATLAS=1")

# --- pmp-library parametrization baseline (LSCM + harmonic) ----------------
if(HALFMESH_BENCH_WITH_PMP)
    set(HALFMESH_PMP_DIR "$ENV{HOME}/Pro/pmp-library" CACHE PATH "pmp-library source dir")
    find_library(PMP_LIB NAMES pmp HINTS "${HALFMESH_PMP_DIR}/build")
    if(PMP_LIB AND EXISTS "${HALFMESH_PMP_DIR}/src/pmp/algorithms/parameterization.h")
        list(APPEND BENCH_ENGINE_SOURCES engines/EnginePmp.cpp)
        list(APPEND BENCH_ENGINE_LIBS ${PMP_LIB})
        list(APPEND BENCH_ENGINE_INCLUDES ${HALFMESH_PMP_DIR}/src)
        list(APPEND BENCH_ENGINE_DEFINES HMBENCH_WITH_PMP=1)
        message(STATUS "atlasbench: pmp baseline enabled (${PMP_LIB})")
    else()
        message(WARNING "atlasbench: HALFMESH_BENCH_WITH_PMP=ON but libpmp not found under "
            "${HALFMESH_PMP_DIR}/build — build it first: "
            "cmake -S ${HALFMESH_PMP_DIR} -B ${HALFMESH_PMP_DIR}/build && "
            "cmake --build ${HALFMESH_PMP_DIR}/build --target pmp. Skipping pmp.")
    endif()
endif()

# --- libigl parametrization baseline (LSCM) --------------------------------
if(HALFMESH_BENCH_WITH_LIBIGL)
    find_package(libigl CONFIG REQUIRED)
    list(APPEND BENCH_ENGINE_SOURCES engines/EngineLibigl.cpp)
    list(APPEND BENCH_ENGINE_LIBS igl::igl_core)
    list(APPEND BENCH_ENGINE_DEFINES HMBENCH_WITH_LIBIGL=1)
    message(STATUS "atlasbench: libigl baseline enabled")
endif()

# --- CGAL baseline (SDF segmentation + LSCM/ARAP parametrization) ----------
if(HALFMESH_BENCH_WITH_CGAL)
    find_package(CGAL CONFIG REQUIRED)
    list(APPEND BENCH_ENGINE_SOURCES engines/EngineCgal.cpp)
    list(APPEND BENCH_ENGINE_LIBS CGAL::CGAL)
    list(APPEND BENCH_ENGINE_DEFINES HMBENCH_WITH_CGAL=1)
    message(STATUS "atlasbench: CGAL baseline enabled")
endif()

# --- BFF baseline via geometry-central (conformal flattening) --------------
if(HALFMESH_BENCH_WITH_BFF)
    include(FetchContent)
    FetchContent_Declare(geometry-central
        GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
        GIT_TAG master
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(geometry-central)
    list(APPEND BENCH_ENGINE_SOURCES engines/EngineBff.cpp)
    list(APPEND BENCH_ENGINE_LIBS geometry-central)
    list(APPEND BENCH_ENGINE_DEFINES HMBENCH_WITH_BFF=1)
    message(STATUS "atlasbench: BFF (geometry-central) baseline enabled")
endif()

# Shared core library: engines + metrics + report, linked by both the CLI and
# the sanity test so they exercise identical code.
add_library(atlasbench_core STATIC
    ${BENCH_LIB_SOURCES}
    ${BENCH_ENGINE_SOURCES}
)
target_include_directories(atlasbench_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # BenchTypes.h, BenchMetrics.h, engines/
    ${CMAKE_CURRENT_SOURCE_DIR}/../metrics
    ${CMAKE_SOURCE_DIR}/src              # ChartFlattenCache.h: EngineHalfmesh
                                         # shares the flatten cache across
                                         # Modules A/B the way GenerateAtlas
                                         # does, so it measures the cost a
                                         # caller actually pays
    ${BENCH_ENGINE_INCLUDES}
)
target_link_libraries(atlasbench_core PUBLIC
    halfmesh::halfmesh
    halfmesh_metrics
    ${BENCH_ENGINE_LIBS}
)
if(BENCH_ENGINE_DEFINES)
    target_compile_definitions(atlasbench_core PUBLIC ${BENCH_ENGINE_DEFINES})
endif()

# The comparison CLI.
add_executable(atlasbench AtlasBench.cpp)
target_link_libraries(atlasbench PRIVATE atlasbench_core)

# Harness self-verification (identity / determinism / metric normalization).
add_executable(atlasbench_sanity_test AtlasBenchSanityTest.cpp)
target_link_libraries(atlasbench_sanity_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    atlasbench_core
)
gtest_discover_tests(atlasbench_sanity_test PROPERTIES LABELS "bench")

message(STATUS "halfmesh: atlasbench harness enabled")

# ---------------------------------------------------------------------------
# remeshbench — isotropic-remeshing quality + speed benchmark.
#
# Sweeps Mesh::RemeshIsotropic over a corpus at several target edge lengths and
# reports edge-length uniformity, triangle-angle quality, valence regularity,
# surface fidelity and wall-time (shared hmtest::metrics).  Optional baseline:
#   -DHALFMESH_REMESH_WITH_PMP=ON   pmp::uniform_remeshing      (links libpmp)
# It runs on identical inputs/params so quality + time are directly comparable.
# ---------------------------------------------------------------------------
set(REMESHBENCH_SOURCES RemeshBench.cpp)
set(REMESHBENCH_LIBS halfmesh::halfmesh halfmesh_metrics halfmesh_corpus)
set(REMESHBENCH_DEFINES "")
set(REMESHBENCH_INCLUDES "")

if(HALFMESH_REMESH_WITH_PMP)
    set(HALFMESH_PMP_DIR "$ENV{HOME}/Pro/pmp-library" CACHE PATH "pmp-library source dir")
    find_library(PMP_LIB NAMES pmp HINTS "${HALFMESH_PMP_DIR}/build")
    if(PMP_LIB AND EXISTS "${HALFMESH_PMP_DIR}/src/pmp/algorithms/remeshing.h")
        list(APPEND REMESHBENCH_SOURCES remesh_engines/RemeshEnginePmp.cpp)
        list(APPEND REMESHBENCH_LIBS ${PMP_LIB})
        list(APPEND REMESHBENCH_INCLUDES ${HALFMESH_PMP_DIR}/src)
        list(APPEND REMESHBENCH_DEFINES HMBENCH_WITH_PMP=1)
        message(STATUS "remeshbench: pmp baseline enabled (${PMP_LIB})")
    else()
        message(WARNING "remeshbench: HALFMESH_REMESH_WITH_PMP=ON but libpmp not found under "
            "${HALFMESH_PMP_DIR}/build — build it first. Skipping pmp baseline.")
    endif()
endif()

add_executable(remeshbench ${REMESHBENCH_SOURCES})
target_include_directories(remeshbench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # tests/ root → "metrics/...", "corpus/..."
    ${REMESHBENCH_INCLUDES}
)
target_link_libraries(remeshbench PRIVATE ${REMESHBENCH_LIBS})
if(REMESHBENCH_DEFINES)
    target_compile_definitions(remeshbench PRIVATE ${REMESHBENCH_DEFINES})
endif()

# Fast smoke run for ctest (label "bench"): default corpus sweep; non-zero exit
# on an empty mesh or non-finite metric.
add_test(NAME remeshbench_smoke COMMAND remeshbench)
set_tests_properties(remeshbench_smoke PROPERTIES LABELS "bench" TIMEOUT 600)

message(STATUS "halfmesh: remeshbench harness enabled")

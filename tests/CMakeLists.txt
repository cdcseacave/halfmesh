find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# Discover tests at ctest run time (PRE_TEST) instead of as a post-build step.
# POST_BUILD discovery runs each test binary's --gtest_list_tests during the build;
# under a parallel build (`cmake --build -j`) that can hit the default 5s discovery
# timeout and silently mark a test target NOT_BUILT (dropping it from the suite).
# PRE_TEST defers discovery to `ctest`, avoiding the build-time race entirely.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

# Under ASan/UBSan the sanitizer runtime adds ~2-3s of startup overhead to
# every test binary launch, so the default 5s discovery timeout is too short.
# Use 60s when HALFMESH_SANITIZE is ON.
if(HALFMESH_SANITIZE)
    set(HALFMESH_GTEST_DISCOVERY_TIMEOUT 60)
else()
    set(HALFMESH_GTEST_DISCOVERY_TIMEOUT 5)
endif()

# Convenience wrapper: gtest_discover_tests with the project-wide discovery
# timeout applied automatically.  Use in place of gtest_discover_tests().
macro(halfmesh_discover_tests target)
    gtest_discover_tests(${target} DISCOVERY_TIMEOUT ${HALFMESH_GTEST_DISCOVERY_TIMEOUT})
endmacro()

# util_test: parity tests for types + assert/log/loop/math/hash
add_executable(util_test UtilTest.cpp)
target_link_libraries(util_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(util_test)

# sampler_test: generic Linear/Cubic image sampler (Util/Sampler.h)
add_executable(sampler_test SamplerTest.cpp)
target_link_libraries(sampler_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(sampler_test)

# raster_test: UV-space triangle rasterizer + gutter dilation (Util/Raster.h)
add_executable(raster_test RasterTest.cpp)
target_link_libraries(raster_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(raster_test)

# texture_bake_test: rebake/defrag engine + resolvers (TextureBake.h)
add_executable(texture_bake_test TextureBakeTest.cpp)
target_link_libraries(texture_bake_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh::halfmesh
)
halfmesh_discover_tests(texture_bake_test)

# geometry_test: DistanceBetweenTriangleAndPoint
add_executable(geometry_test GeometryTest.cpp)
target_link_libraries(geometry_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(geometry_test)

# oriented_bounding_box_test: OBB Contains + Transform
add_executable(oriented_bounding_box_test OrientedBoundingBoxTest.cpp)
target_link_libraries(oriented_bounding_box_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(oriented_bounding_box_test)

# accumulator_test: WeightedAccumulator parity
add_executable(accumulator_test AccumulatorTest.cpp)
target_link_libraries(accumulator_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(accumulator_test)

# priority_queue_test: indexed mutable PQ ordering + TQuadric parity
add_executable(priority_queue_test PriorityQueueTest.cpp)
target_link_libraries(priority_queue_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(priority_queue_test)

# halfmesh_test: HalfMesh half-edge core topology
add_executable(halfmesh_test HalfMeshTest.cpp)
target_link_libraries(halfmesh_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(halfmesh_test)

# mesh_core_test: Mesh foundational methods (normals, area, AABB, adjacency, edit primitives)
add_executable(mesh_core_test MeshCoreTest.cpp)
target_link_libraries(mesh_core_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_core_test)

# triangle_kdtree_test: NearestPoint + IntersectedPoint vs brute-force on inline grid mesh
add_executable(triangle_kdtree_test TriangleKDTreeTest.cpp)
target_link_libraries(triangle_kdtree_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(triangle_kdtree_test)

# triangle_bvh_test: flattened BVH NearestPoint/IntersectedPoint vs brute-force
add_executable(triangle_bvh_test TriangleBVHTest.cpp)
target_link_libraries(triangle_bvh_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(triangle_bvh_test)

# mesh_io_test: PLY round-trip + glTF smoke-save + mesh.ply-based KdTree tests
add_executable(mesh_io_test MeshIOTest.cpp)
# Needs src/ on the include path for GltfImageCodec.h: the tests that drive a
# tinygltf::TinyGLTF directly must install the same OpenCV image codec MeshIO
# does, since the build compiles tinygltf without stb_image.
target_include_directories(mesh_io_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mesh_io_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_io_test)

# mesh_repair_test: duplicate/degenerate/non-manifold/small-component/OBB-cull
add_executable(mesh_repair_test MeshRepairTest.cpp)
target_include_directories(mesh_repair_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(mesh_repair_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_repair_test)

# mesh_simplify_test: QEM decimation + RoundCast parity
add_executable(mesh_simplify_test MeshSimplifyTest.cpp)
target_include_directories(mesh_simplify_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(mesh_simplify_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_simplify_test)

# mesh_remesh_test: isotropic remeshing + quality-metric regression baselines
add_executable(mesh_remesh_test MeshRemeshTest.cpp)
target_include_directories(mesh_remesh_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(mesh_remesh_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_remesh_test)

# mesh_smooth_test: Mesh::SmoothHCLaplacian — HC Laplacian smoothing
add_executable(mesh_smooth_test MeshSmoothTest.cpp)
target_include_directories(mesh_smooth_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(mesh_smooth_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_smooth_test)

# mesh_holes_test: CloseHoles via Liepa hole-filling
add_executable(mesh_holes_test MeshHolesTest.cpp)
target_link_libraries(mesh_holes_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(mesh_holes_test)

# parametrize_test: D-Charts chart segmentation
add_executable(parametrize_test ParametrizeTest.cpp)
target_link_libraries(parametrize_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh::halfmesh
)
halfmesh_discover_tests(parametrize_test)

# flatten_test: per-chart UV flattening — Tutte + ARAP/SLIM
add_executable(flatten_test FlattenTest.cpp)
target_link_libraries(flatten_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(flatten_test)

# atlas_test: uniform texel-density normalisation
add_executable(atlas_test AtlasTest.cpp)
target_link_libraries(atlas_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(atlas_test)

# segment_quality_test: segmentation-quality harness — seam length, chart
# compactness, packed occupancy over the corpus (D-Charts distance-term plan).
add_executable(segment_quality_test SegmentQualityTest.cpp)
target_link_libraries(segment_quality_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
    halfmesh_metrics
    halfmesh_corpus
)
halfmesh_discover_tests(segment_quality_test)

# internals_test: src/-internal units (halfmesh::detail) with no public
# header — FairMesh (MeshRemeshShared.h) + ParallelForPool (ParallelFor.h).
# Needs src/ on the include path and the header-only BS thread pool
# (BS_THREAD_POOL_INCLUDE_DIR is cached by the top-level find_path).
add_executable(internals_test InternalsTest.cpp)
target_include_directories(internals_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${BS_THREAD_POOL_INCLUDE_DIR}
)
target_link_libraries(internals_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh::halfmesh
)
halfmesh_discover_tests(internals_test)

# ---------------------------------------------------------------------------
# Shared metrics toolkit
# ---------------------------------------------------------------------------

# halfmesh_metrics: reusable static library of topology/geometry/distance/UV/
# robustness/canonicalization metrics (namespace hmtest::metrics).
# Sources live in tests/metrics/ so they can be linked by any test executable.
add_library(halfmesh_metrics STATIC
    metrics/Metrics.cpp
)
target_include_directories(halfmesh_metrics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(halfmesh_metrics PUBLIC
    halfmesh::halfmesh
)

# metrics_test: self-validation of the metrics toolkit against analytic answers.
add_executable(metrics_test metrics/MetricsTest.cpp)
target_link_libraries(metrics_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_metrics
    halfmesh_corpus
)
halfmesh_discover_tests(metrics_test)

# ---------------------------------------------------------------------------
# Test corpus library
# ---------------------------------------------------------------------------

# halfmesh_corpus: analytic mesh generators + dirty synthesizers.
# Namespace: hmtest::corpus. Links halfmesh::halfmesh only (no metrics in lib).
add_library(halfmesh_corpus STATIC
    corpus/Corpus.cpp
)
target_include_directories(halfmesh_corpus PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
)
target_link_libraries(halfmesh_corpus PUBLIC
    halfmesh::halfmesh
)

# corpus_test: validates every generator via the metrics toolkit.
add_executable(corpus_test corpus/CorpusTest.cpp)
target_include_directories(corpus_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(corpus_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
)
halfmesh_discover_tests(corpus_test)

# ---------------------------------------------------------------------------
# Golden fixtures (docs/TESTING.md §4): frozen known-good regression snapshots
#
# halfmesh_golden: shared helpers (mesh<->PLY result + metrics<->JSON sidecar)
# and the spec table (GoldenSpecs.h) used by the golden regression test.
add_library(halfmesh_golden STATIC
    golden/GoldenIO.cpp
)
target_include_directories(halfmesh_golden PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/golden
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(halfmesh_golden PUBLIC
    halfmesh_corpus
    halfmesh_metrics
)

# golden_diff_test: current output vs committed frozen snapshots — runs in the default build.
# Recomputes each op and compares to the committed golden fixtures.
add_executable(golden_diff_test golden/GoldenDiffTest.cpp)
target_link_libraries(golden_diff_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_golden
)
halfmesh_discover_tests(golden_diff_test)

# ---------------------------------------------------------------------------
# Robustness layer (docs/TESTING.md §7)
# ---------------------------------------------------------------------------
# Determinism + property/fuzz + metamorphic tests using hmtest::metrics +
# hmtest::corpus.  Fixed seeds throughout; prints seed+sequence on failure.
add_executable(robustness_test robustness/RobustnessTest.cpp)
target_include_directories(robustness_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(robustness_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(robustness_test)

# ---------------------------------------------------------------------------
# HalfMesh structural-invariant + edge-collapse tests (durable, no reference).
# ---------------------------------------------------------------------------
add_executable(half_mesh_invariants_test HalfMeshInvariantsTest.cpp)
target_include_directories(half_mesh_invariants_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics
)
target_link_libraries(half_mesh_invariants_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    halfmesh_corpus
    halfmesh_metrics
    halfmesh::halfmesh
)
halfmesh_discover_tests(half_mesh_invariants_test)

# ---------------------------------------------------------------------------
# Performance regression harness (docs/TESTING.md §7 perf)
#
# Only built when HALFMESH_BUILD_PERF=ON.  Excluded from the default fast
# ctest run (labelled 'perf').  Run with:
#   ctest --test-dir <build> -L perf --output-on-failure
# or directly: ./tests/perf/perf_harness
#
# Requires halfmesh_corpus (for LargeMesh generator).
# Build in RELEASE for realistic numbers:
#   cmake -DHALFMESH_BUILD_PERF=ON -DCMAKE_BUILD_TYPE=Release
# ---------------------------------------------------------------------------
if(HALFMESH_BUILD_PERF)
    # halfmesh_corpus may not exist yet (if HALFMESH_BUILD_TESTS=OFF).
    # Define it if not already defined.
    if(NOT TARGET halfmesh_corpus)
        add_library(halfmesh_corpus STATIC
            corpus/Corpus.cpp
        )
        target_include_directories(halfmesh_corpus PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/corpus
        )
        target_link_libraries(halfmesh_corpus PUBLIC
            halfmesh::halfmesh
        )
    endif()

    add_executable(perf_harness perf/PerfHarness.cpp)
    target_include_directories(perf_harness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/corpus
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(perf_harness PRIVATE
        GTest::gtest
        halfmesh_corpus
        halfmesh::halfmesh
    )
    # Use a generous discovery timeout — the harness is slow.
    set_target_properties(perf_harness PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/perf"
    )
    # Register with ctest under the 'perf' label and a long per-test timeout.
    # Using add_test directly (not gtest_discover_tests) to avoid running
    # the harness during the PRE_TEST discovery phase and to tag with 'perf'.
    add_test(
        NAME perf_harness
        COMMAND perf_harness
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/perf"
    )
    set_tests_properties(perf_harness PROPERTIES
        LABELS "perf"
        TIMEOUT 3600   # generous: 1 hour for slow machines
        # The harness times wall-clock, so co-scheduled tests read as a slowdown:
        # under `ctest -j` the Build scaling ratio drifts past its bound purely
        # from contention. RUN_SERIAL keeps the machine to itself.
        RUN_SERIAL TRUE
    )
    message(STATUS "halfmesh: perf harness enabled (ctest -L perf)")
endif()

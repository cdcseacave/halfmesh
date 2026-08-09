# examples/CMakeLists.txt — example CLI tools (guarded by HALFMESH_BUILD_TOOLS).
#
# Each example is a small self-contained program that exercises one area of the
# public halfmesh API.  Build with:
#   cmake .. -DHALFMESH_BUILD_TOOLS=ON && cmake --build . -j

cmake_minimum_required(VERSION 3.21)

# --- decimate --------------------------------------------------------------- #
add_executable(decimate Decimate.cpp)
target_link_libraries(decimate PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(decimate)

# --- remesh ----------------------------------------------------------------- #
add_executable(remesh Remesh.cpp)
target_link_libraries(remesh PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(remesh)

# --- unwrap ----------------------------------------------------------------- #
add_executable(unwrap Unwrap.cpp)
target_link_libraries(unwrap PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(unwrap)

# --- smooth ------------------------------------------------------------------ #
add_executable(smooth Smooth.cpp)
target_link_libraries(smooth PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(smooth)

# --- texturebake (rebake / defrag / fidelity comparison harness) ------------ #
add_executable(texturebake TextureBakeTool.cpp)
target_link_libraries(texturebake PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(texturebake)

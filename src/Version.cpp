/*
* Version.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "halfmesh/Version.h"

// Injected by the build from project(VERSION ...) in CMakeLists.txt so the
// runtime string cannot drift from the package version; the fallback only
// covers compiling this file outside the CMake build.
#ifndef HALFMESH_VERSION_STRING
	#define HALFMESH_VERSION_STRING "0.0.0-unversioned"
#endif

namespace halfmesh {
const char* Version() noexcept { return HALFMESH_VERSION_STRING; }
} // namespace halfmesh

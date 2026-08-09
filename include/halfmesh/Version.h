/*
* Version.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once
namespace halfmesh {
// Library version string ("<major>.<minor>.<patch>", injected from the CMake
// project version at build time).
const char* Version() noexcept;
} // namespace halfmesh

/*
* Log.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/Util/Log.h>

namespace halfmesh {
namespace {

// Destination of the REPORT_STATUS* progress logs; &std::cout preserves the
// library's historical default. See the declaration notes in Util/Log.h.
std::ostream* gStatusLog = &std::cout;

} // namespace

void SetStatusLog(std::ostream* os) noexcept
{
	gStatusLog = os;
}

std::ostream* GetStatusLog() noexcept
{
	return gStatusLog;
}

} // namespace halfmesh

/*
* Log.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Logging shim: uses std::format (C++20) by default.
// Define HALFMESH_USE_FMT before including to use fmt::format instead.
#pragma once

#include <halfmesh/Util/Assert.h>

// The macros below forward their whole argument list as a single __VA_ARGS__
// rather than splitting it into (msg, ...) and re-joining with __VA_OPT__(, ).
// Both spellings behave identically, but the __VA_OPT__ one requires a
// C++20-conformant preprocessor: MSVC's default (traditional) preprocessor does
// NOT implement __VA_OPT__ and /std:c++20 does not switch it on, so it would
// need /Zc:preprocessor. This header is reachable from the public API
// (InteropOpenMVS.h), and a library must not force a preprocessor mode on its
// consumers — forwarding __VA_ARGS__ wholesale compiles everywhere with no flags.
#ifdef HALFMESH_USE_FMT
	#include <fmt/format.h>
	#define HALFMESH_FORMAT(...) fmt::format(__VA_ARGS__)
#else
	#include <format>
	#include <string>
	#define HALFMESH_FORMAT(...) std::format(__VA_ARGS__)
#endif

#include <iostream>
#include <chrono>
#include <cstdlib> // ::exit (THROW_FATAL)
#include <string>

namespace halfmesh {
// Runtime destination for the REPORT_STATUS* progress logs the library emits
// (defined in src/Log.cpp). Defaults to &std::cout — pass any std::ostream to
// redirect, or nullptr to silence them entirely. Warnings (REPORT_WARNING*)
// always go to stderr. Not synchronized: set once before starting library work.
void SetStatusLog(std::ostream* os) noexcept;
std::ostream* GetStatusLog() noexcept;
} // namespace halfmesh

// ---------------------------------------------------------------------------
// REPORT_* macros
// ---------------------------------------------------------------------------
#ifndef REPORT_STATUS
	#define REPORT_STATUS(...)                                                 \
		do {                                                                   \
			if (::std::ostream* hmStatusOs_ = ::halfmesh::GetStatusLog())      \
				(*hmStatusOs_) << HALFMESH_FORMAT(__VA_ARGS__) << ::std::endl; \
		} while (0)
#endif
#ifndef REPORT_WARNING
	#define REPORT_WARNING(...) (std::cerr << HALFMESH_FORMAT(__VA_ARGS__) << std::endl)
#endif
#ifndef REPORT_STATUS_NOW
	#define REPORT_STATUS_NOW(...) REPORT_STATUS(__VA_ARGS__)
#endif
#ifndef REPORT_WARNING_NOW
	#define REPORT_WARNING_NOW(...) (std::cerr << HALFMESH_FORMAT(__VA_ARGS__) << std::endl)
#endif
#ifndef THROW_FATAL
	#define THROW_FATAL(...)                 \
		do {                                 \
			REPORT_WARNING_NOW(__VA_ARGS__); \
			::exit(-1);                      \
		} while (0)
#endif

// ---------------------------------------------------------------------------
// TIMER_* macros
//   TIMER_START(...)  — declare + start a high-resolution timer
//   TIMER_STR()       — return elapsed time as a C-string "<N>ms"
// ---------------------------------------------------------------------------
#ifndef TIMER_START
	#define TIMER_START(...)                                   \
		const ::std::chrono::high_resolution_clock::time_point \
		    timerStart = ::std::chrono::high_resolution_clock::now()
#endif
#ifndef TIMER_STR
	#define TIMER_STR()                                                                \
		(HALFMESH_FORMAT("{}ms",                                                       \
		                 ::std::chrono::duration_cast<::std::chrono::milliseconds>(    \
		                     ::std::chrono::high_resolution_clock::now() - timerStart) \
		                     .count())                                                 \
		     .c_str())
#endif

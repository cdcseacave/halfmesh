/*
* Assert.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <cassert>

// ASSERT: debug-mode assertion macro.
// In Debug builds (NDEBUG not defined), behaves like assert(); in Release it
// expands to nothing. Variadic arguments are accepted but ignored.
// The #ifndef guard lets an external ASSERT (e.g. openMVS's) win when both
// libraries are compiled together.
#ifndef ASSERT
	#ifndef NDEBUG
		#define ASSERT(exp, ...) assert(exp)
	#else
		#define ASSERT(exp, ...)
	#endif
#endif

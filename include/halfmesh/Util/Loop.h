/*
* Loop.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <type_traits>

// ---------------------------------------------------------------------------
// Helper: size_type of an array/vector
// ---------------------------------------------------------------------------
#ifndef ARR2IDX
	#define ARR2IDX(arr) typename std::remove_reference<decltype(arr)>::type::size_type
#endif

// ---------------------------------------------------------------------------
// Forward iteration by index
// ---------------------------------------------------------------------------
#ifndef FOREACH
	#define FOREACH(var, arr) \
		for (ARR2IDX(arr) var = 0, var##Size = (arr).size(); var < var##Size; ++var)
#endif

// ---------------------------------------------------------------------------
// Reverse iteration by index (post-decrement sentinel trick from original)
// ---------------------------------------------------------------------------
#ifndef RFOREACH
	#define RFOREACH(var, arr) \
		for (ARR2IDX(arr) var = (arr).size(); var-- > 0;)
#endif

// ---------------------------------------------------------------------------
// Forward iteration by typed index
// ---------------------------------------------------------------------------
#ifndef FOREACHIDX
	#define FOREACHIDX(Type, var, arr) \
		for (Type var = 0, var##Size = static_cast<Type>((arr).size()); var < var##Size; ++var)
#endif

// ---------------------------------------------------------------------------
// Reverse iteration by typed index
// ---------------------------------------------------------------------------
#ifndef RFOREACHIDX
	#define RFOREACHIDX(Type, var, arr) \
		for (Type var = static_cast<Type>((arr).size()); var-- > 0;)
#endif

// ---------------------------------------------------------------------------
// Forward iteration by pointer
// ---------------------------------------------------------------------------
#ifndef FOREACHPTR
	#define FOREACHPTR(var, arr) \
		for (auto var = (arr).data(), var##End = var + (arr).size(); var != var##End; ++var)
#endif

// ---------------------------------------------------------------------------
// Reverse iteration by pointer
// ---------------------------------------------------------------------------
#ifndef RFOREACHPTR
    // NOTE: the decrement must be guarded by the comparison — an unconditional
    // `var-- != var##Begin` also decrements on the final (false) test, forming a
    // pointer one before the array (UB), and on an empty container with null
    // data() that is arithmetic on a null pointer (caught by UBSan).
	#define RFOREACHPTR(var, arr) \
		for (auto var##Begin = (arr).data(), var = var##Begin + (arr).size(); var != var##Begin && (--var, true);)
#endif

// ---------------------------------------------------------------------------
// Forward iteration by typed index over a raw size (not a container)
// ---------------------------------------------------------------------------
#ifndef FOREACHRAWIDX
	#define FOREACHRAWIDX(Type, var, sz) for (Type var = 0, var##Size = (sz); var < var##Size; ++var)
#endif

#ifndef RFOREACHRAWIDX
	#define RFOREACHRAWIDX(Type, var, sz) for (Type var = (sz); var-- > 0;)
#endif

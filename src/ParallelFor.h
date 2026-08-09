/*
* ParallelFor.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Pool-backed parallel-for helper shared by the mesh-processing TUs (Mesh.cpp,
// MeshIO.cpp, MeshSimplify.cpp, MeshRemesh.cpp), so the pool logic lives in one
// place rather than in a file-local copy per TU.
//
// Library-internal header (lives under src/, not include/halfmesh/): no public
// API surface, never installed. Namespace halfmesh::detail.
#pragma once

#include <cstddef>
#include <exception>
#include <mutex>

#include <BS_thread_pool.hpp>

namespace halfmesh {
namespace detail {

// Pool-backed parallel-for: run fn(i) for i in [0,n) across `pool`'s threads,
// split into contiguous blocks (one per thread). Blocks until all i complete;
// reuses the persistent pool so there is no per-call thread spawn/join.
//
// fn MUST be safe to call concurrently for distinct i (writes only to per-i
// outputs, no shared mutable state). Because every fn(i) is a self-contained map
// with no cross-i accumulation, the result is independent of the block split and
// thread order (deterministic).
//
// Exceptions: BS::light_thread_pool workers swallow task exceptions (empty
// catch(...)), so an exception escaping fn would silently drop the rest of
// that block's indices and wait() would return as if complete. Capture the
// FIRST worker exception here and rethrow it after wait(): a failed parallel
// region is loud, never a silent partial result. (Indices after the throwing
// one in the same block are still skipped — the exception is the signal.)
template <class Fn>
void ParallelForPool(BS::light_thread_pool& pool, std::size_t n, Fn&& fn)
{
	if (n == 0)
		return;
	if (n == 1) {
		fn(std::size_t(0));
		return;
	}
	std::exception_ptr firstError = nullptr;
	std::mutex errorMutex;
	pool.detach_blocks(std::size_t(0), n,
	                   [&](std::size_t begin, std::size_t end) {
		                   try {
			                   for (std::size_t i = begin; i < end; ++i)
				                   fn(i);
		                   } catch (...) {
			                   const std::lock_guard<std::mutex> lock(errorMutex);
			                   if (firstError == nullptr)
				                   firstError = std::current_exception();
		                   }
	                   });
	pool.wait();
	if (firstError != nullptr)
		std::rethrow_exception(firstError);
}

} // namespace detail
} // namespace halfmesh

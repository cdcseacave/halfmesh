/*
* Hash.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Provides std::hash specializations for std::pair and std::tuple using
// boost-style HashCombine mixing.
#pragma once

#include <functional>
#include <tuple>
#include <utility>

namespace halfmesh {
namespace detail {

// -------------------------------------------------------------------------
// HashCombine — Boost 1.81-style seed mixing.
// Add the 64-bit golden-ratio constant 0x9e3779b97f4a7c15 = floor(2^64/phi),
// then run a murmur3/splitmix64 finalizer. The pre-1.81 boost mixer
// (32-bit 0x9e3779b9 + shift-6/shift-2) disperses structured integer keys —
// e.g. quantized lattice coordinates — poorly; this multiply-based avalanche
// mixes far better while staying order-sensitive (needed for directed-edge keys).
// On a 32-bit size_t platform the constants truncate to their low 32 bits, which
// is a weaker but still valid mixer; the hot paths here run on 64-bit size_t.
// -------------------------------------------------------------------------
template <class T>
inline void HashCombine(std::size_t& seed, T const& v)
{
	std::size_t x = seed + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + std::hash<T>()(v);
	x ^= x >> 33;
	x *= static_cast<std::size_t>(0xff51afd7ed558ccdULL);
	x ^= x >> 33;
	x *= static_cast<std::size_t>(0xc4ceb9fe1a85ec53ULL);
	x ^= x >> 33;
	seed = x;
}

// Recursive tuple hashing
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
	static void apply(size_t& seed, Tuple const& tuple)
	{
		HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
		halfmesh::detail::HashCombine(seed, std::get<Index>(tuple));
	}
};
template <class Tuple>
struct HashValueImpl<Tuple, 0>
{
	static void apply(size_t& seed, Tuple const& tuple)
	{
		halfmesh::detail::HashCombine(seed, std::get<0>(tuple));
	}
};

} // namespace detail
} // namespace halfmesh

namespace std {

// -------------------------------------------------------------------------
// std::hash<std::pair<T, U>>
// -------------------------------------------------------------------------
template <typename T, typename U>
struct hash<std::pair<T, U>>
{
	std::size_t operator()(const std::pair<T, U>& x) const
	{
		// Seed from 0 and combine BOTH elements (as the tuple hash and Boost do):
		// the avalanche mixer is additive before its finalizer, so seeding with the
		// raw hash(first) and combining only the second would make the result
		// symmetric in (first, second) — losing the order sensitivity directed-edge
		// keys require. Mixing first before adding second keeps it order-sensitive.
		size_t hVal = 0;
		halfmesh::detail::HashCombine<T>(hVal, x.first);
		halfmesh::detail::HashCombine<U>(hVal, x.second);
		return hVal;
	}
};

// -------------------------------------------------------------------------
// std::hash<std::tuple<TT...>>
// -------------------------------------------------------------------------
template <typename... TT>
struct hash<std::tuple<TT...>>
{
	size_t operator()(std::tuple<TT...> const& tt) const
	{
		size_t seed = 0;
		halfmesh::detail::HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
		return seed;
	}
};

} // namespace std

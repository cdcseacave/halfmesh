/*
* PriorityQueue.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <type_traits>
#include <vector>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Types.h>

namespace halfmesh {

// Priority queue, based on vector structure, with support for update and remove;
// Key has to be an unsigned integer value
template <typename KEY, typename PRIORITY, bool ASCENDING = true>
class TPriorityQueue
{
	public:
	typedef KEY Key;
	typedef PRIORITY Priority;
	enum : bool { Ascending = ASCENDING };
	static_assert(std::is_integral<Key>::value && std::is_unsigned<Key>::value, "Unsigned integral required.");

	struct Node
	{
		Key key;
		Priority priority;

		bool operator<(const Node& pqn2) const { return Compare(priority, pqn2.priority); }
	};

	static inline bool Compare(Priority p0, Priority p1)
	{
		if (ASCENDING) {
			return p0 < p1;
		} else {
			return p0 > p1;
		}
	}

	protected:
	enum : Key { NO_INDEX = Key(-1) };

	std::vector<Node> heap;
	std::vector<Key> key2heap;

	public:
	void reserve(Key s)
	{
		heap.reserve(s);
		key2heap.reserve(s);
	}
	// pre-sizes the key->slot map so keys in [0, numKeys) are addressable BEFORE
	// they are emplaced. Callers that emplace only a SUBSET of the key space but
	// still relabel/move across the whole range (e.g. Simplify's build-time min-edge
	// filter, whose pop/move relabel renumbers tail keys into freed slots) must call
	// this: it lets move()/pop()/contains() index key2heap in-bounds, with a
	// never-emplaced key reading as empty (NO_INDEX). Already-emplaced slots are
	// left untouched, so this never perturbs an existing heap.
	void ReserveKeys(Key numKeys)
	{
		if (numKeys > key2heap.size()) {
			key2heap.resize(numKeys, NO_INDEX);
		}
	}
	void clear()
	{
		// Whole-queue reset: the key->slot map is reset in place (every slot back
		// to NO_INDEX) instead of freed, so a queue that pre-sized its key space
		// (ReserveKeys) and relabels/moves across the whole key range — e.g.
		// Simplify's build-time-filtered min-edge queue — keeps every later
		// move()/pop() index in-bounds; freeing the map here would put a later
		// move() out of bounds. Not on any hot path: the single-element
		// pop()/pop(key) drain shortcuts do not route through clear(), resetting
		// only their one live slot in O(1) instead of paying this O(key-space)
		// sweep on every transient drain. The queue is logically empty afterwards:
		// empty()/size()==0 and contains() is false for every key.
		heap.clear();
		key2heap.assign(key2heap.size(), NO_INDEX);
	}
	bool empty() const { return heap.empty(); }
	Key size() const { return heap.size(); }

	// fetches top key
	const Node& peek() const
	{
		return heap.front();
	}

	// fetches given key
	const Node* peek(Key key) const
	{
		ASSERT(key < key2heap.size());
		const Key pos = key2heap[key];
		return pos == NO_INDEX ? NULL : &heap[pos];
	}

	// true iff key currently occupies a heap slot (was emplaced and not yet popped).
	// The sanctioned guard for callers that may hold a never-emplaced key: pair it
	// with emplace() (contains() ? update() : emplace()) so update()/move() are only
	// ever handed a live key. Bounds-safe for any key value (no precondition).
	bool contains(Key key) const
	{
		return key < key2heap.size() && key2heap[key] != NO_INDEX;
	}

	// inserts given key
	void emplace(Key key, Priority priority)
	{
		const Key n = heap.size();
		if (key >= key2heap.size()) {
			key2heap.resize(key + 1, NO_INDEX);
		}
		ASSERT(key2heap[key] == NO_INDEX);
		key2heap[key] = n;
		heap.emplace_back(Node{key, priority});
		SiftUp(n);
	}

	// updates given key or emplace it if missing
	void update(Key key, Priority priority)
	{
		if (key >= key2heap.size() || key2heap[key] == NO_INDEX) {
			emplace(key, priority);
		} else {
			UpdateHeap(key2heap[key], priority);
		}
	}

	// moves given key to the given empty position
	void move(Key key, Key keyEmpty)
	{
		// Precondition: both keys are within the addressable range — callers size
		// the key space via emplace()/ReserveKeys(). These bounds asserts are the
		// OOB guard for the build-time-filtered simplify path, where `key` may name
		// an edge that was never emplaced (then pos == NO_INDEX below and the move is
		// a correct no-op). Loud in Debug/sanitizer, compiled out (zero-cost) in
		// Release.
		ASSERT(key < key2heap.size() && keyEmpty < key2heap.size());
		ASSERT(key2heap[keyEmpty] == NO_INDEX);
		Key& pos = key2heap[key];
		if (pos != NO_INDEX) {
			heap[key2heap[keyEmpty] = pos].key = keyEmpty;
			pos = NO_INDEX;
		}
	}

	// removes top key
	void pop()
	{
		ASSERT(!empty() && key2heap[heap.front().key] == 0);
		if (heap.size() == 1) {
			// O(1) single-element drain: exactly one key2heap slot is live (the
			// heap<->key2heap invariant IsFaultless() states), so reset just that
			// slot. Deliberately NOT clear(): its whole-key-space reset is
			// O(key2heap.size()), and this shortcut is hot — min-edge Simplify's
			// small filtered queue transiently drains to empty repeatedly.
			key2heap[heap.front().key] = NO_INDEX;
			heap.clear();
			return;
		}
		key2heap[heap.front().key] = NO_INDEX;
		heap.front() = heap.back();
		heap.pop_back();
		SiftDown(0);
	}

	// removes given key if exists
	bool pop(Key key)
	{
		if (empty() || key >= key2heap.size()) {
			return false;
		}
		Key& pos = key2heap[key];
		if (pos == NO_INDEX) {
			return false;
		}
		ASSERT(heap[pos].key == key);
		if (heap.size() == 1) {
			// O(1) single-element drain — see pop() above; `pos` is the only live
			// slot, so resetting it alone empties the map.
			pos = NO_INDEX;
			heap.clear();
			return true;
		}
		const Node val = heap.back();
		heap[pos].key = val.key;
		heap.pop_back();
		if (pos < heap.size()) {
			UpdateHeap(pos, val.priority);
		}
		pos = NO_INDEX;
		return true;
	}

	// check the mapping is correct
	bool IsFaultless() const
	{
		FOREACH (i, heap) {
			if (key2heap[heap[i].key] != i) {
				return false;
			}
		}
		return true;
	}

	private:
	// maintains queue ordering
	void UpdateHeap(Key pos, Priority newPriority)
	{
		ASSERT(pos != NO_INDEX);
		Priority& priority = heap[pos].priority;
		if (Compare(newPriority, priority)) {
			priority = newPriority;
			SiftUp(pos);
		} else if (Compare(priority, newPriority)) {
			priority = newPriority;
			SiftDown(pos);
		} else {
			key2heap[heap[pos].key] = pos;
		}
	}

	// moves given key towards the queue top at the correct position
	void SiftUp(Key k)
	{
		const Node val = heap[k];
		Key child;
		while (val < heap[child = (k >> 1)]) {
			key2heap[(heap[k] = heap[child]).key] = k;
			if ((k = child) == 0) {
				break;
			}
		}
		key2heap[(heap[k] = val).key] = k;
	}

	// moves given key towards the queue bottom at the correct position
	void SiftDown(Key k)
	{
		const Key len = heap.size();
		const Node val = heap[k];
		Key child;
		while ((child = (k << 1)) < len) {
			if (child + 1 < len && heap[child + 1] < heap[child]) {
				child++;
			}
			if (val < heap[child]) {
				break;
			}
			key2heap[(heap[k] = heap[child]).key] = k;
			if ((k = child) == 0) {
				break;
			}
		}
		key2heap[(heap[k] = val).key] = k;
	}
};

} // namespace halfmesh

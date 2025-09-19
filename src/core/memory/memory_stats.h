#pragma once

#ifndef CORE_MEMORY_MEMORY_STATS_H
#define CORE_MEMORY_MEMORY_STATS_H

#include <cstddef>

namespace fem::core::memory {

struct MemoryStats {
    std::size_t total_allocated = 0;   // cumulative bytes allocated
    std::size_t total_freed = 0;       // cumulative bytes freed
    std::size_t live_bytes = 0;        // currently allocated bytes (total_allocated - total_freed)
    std::size_t peak_bytes = 0;        // peak live bytes
    std::size_t allocation_count = 0;  // number of allocation calls
    std::size_t free_count = 0;        // number of free calls
};

inline void accumulate(MemoryStats& into, const MemoryStats& from) {
    into.total_allocated += from.total_allocated;
    into.total_freed += from.total_freed;
    into.live_bytes += from.live_bytes;
    into.peak_bytes = into.peak_bytes > from.peak_bytes ? into.peak_bytes : from.peak_bytes;
    into.allocation_count += from.allocation_count;
    into.free_count += from.free_count;
}

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_STATS_H


#pragma once

#ifndef CORE_MEMORY_CACHE_LINE_H
#define CORE_MEMORY_CACHE_LINE_H

#include <cstddef>
#include <cstdint>

#include <config/config.h>

#include "aligned_storage.h"

namespace fem::core::memory {

inline constexpr std::size_t cache_line_size() noexcept { return fem::config::CACHE_LINE_SIZE; }

inline constexpr std::size_t cache_line_mask() noexcept { return cache_line_size() - 1; }

inline constexpr std::size_t pad_to_cache_line(std::size_t n) noexcept {
    return align_up(n, cache_line_size());
}

inline bool same_cache_line(const void* a, const void* b) noexcept {
    auto ua = reinterpret_cast<std::uintptr_t>(a);
    auto ub = reinterpret_cast<std::uintptr_t>(b);
    return (ua & ~cache_line_mask()) == (ub & ~cache_line_mask());
}

template<typename T>
using cache_aligned = CacheAligned<T, fem::config::CACHE_LINE_SIZE>;

} // namespace fem::core::memory

#endif // CORE_MEMORY_CACHE_LINE_H


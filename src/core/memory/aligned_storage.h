#pragma once

#ifndef CORE_MEMORY_ALIGNED_STORAGE_H
#define CORE_MEMORY_ALIGNED_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <memory>

#include <config/config.h>

namespace fem::core::memory {

// Constants
constexpr std::size_t kDefaultAlignment = fem::config::DEFAULT_ALIGNMENT;
constexpr std::size_t kCacheLineSize   = fem::config::CACHE_LINE_SIZE;

// Power-of-two check (constexpr)
constexpr bool is_power_of_two(std::size_t x) noexcept {
    return x && ((x & (x - 1)) == 0);
}

// Align integer up to the next multiple of alignment (alignment must be power-of-two)
constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return is_power_of_two(alignment)
           ? (value + alignment - 1) & ~(alignment - 1)
           : ((value + alignment - 1) / alignment) * alignment;
}

// Align integer down (alignment must be power-of-two)
constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept {
    return is_power_of_two(alignment) ? (value & ~(alignment - 1))
                                      : (value / alignment) * alignment;
}

// Pointer alignment helpers
inline void* align_up(void* p, std::size_t alignment) noexcept {
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    addr = align_up(addr, alignment);
    return reinterpret_cast<void*>(addr);
}

inline const void* align_up(const void* p, std::size_t alignment) noexcept {
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    addr = align_up(addr, alignment);
    return reinterpret_cast<const void*>(addr);
}

template<typename T>
inline T* align_up(T* p, std::size_t alignment) noexcept {
    return static_cast<T*>(align_up(static_cast<void*>(p), alignment));
}

// Alignment predicates
inline bool is_aligned(const void* p, std::size_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

template<typename T>
inline bool is_aligned(const T* p, std::size_t alignment) noexcept {
    return is_aligned(static_cast<const void*>(p), alignment);
}

// Assume-aligned wrapper (uses std::assume_aligned if available)
#if defined(__cpp_lib_assume_aligned) && __cpp_lib_assume_aligned >= 201811L
    using std::assume_aligned;
#else
    template<std::size_t Alignment, class T>
    [[nodiscard]] constexpr T* assume_aligned(T* ptr) noexcept { return ptr; }
    template<std::size_t Alignment, class T>
    [[nodiscard]] constexpr const T* assume_aligned(const T* ptr) noexcept { return ptr; }
#endif

// Raw byte buffer with explicit alignment
template<std::size_t N, std::size_t Align = alignof(std::max_align_t)>
struct AlignedBuffer {
    static_assert(Align != 0, "Alignment must be non-zero");
    alignas(Align) std::byte data[N];

    [[nodiscard]] std::byte*       begin() noexcept { return data; }
    [[nodiscard]] std::byte*       end()   noexcept { return data + N; }
    [[nodiscard]] const std::byte* begin() const noexcept { return data; }
    [[nodiscard]] const std::byte* end()   const noexcept { return data + N; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }
};

// Cache-line aligned wrapper to avoid false sharing for hot fields
template<typename T, std::size_t Align = kCacheLineSize>
struct CacheAligned {
    alignas(Align) T value{};

    CacheAligned() = default;
    template<typename... Args>
    explicit CacheAligned(Args&&... args) : value(std::forward<Args>(args)...) {}

    [[nodiscard]] T*       operator->() noexcept { return &value; }
    [[nodiscard]] const T* operator->() const noexcept { return &value; }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_ALIGNED_STORAGE_H


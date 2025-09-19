#pragma once

#ifndef CORE_MEMORY_MEMORY_UTILS_H
#define CORE_MEMORY_MEMORY_UTILS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <span>

#include <config/config.h>

#include "aligned_storage.h"

namespace fem::core::memory {

// Overflow checks (size_t)
inline constexpr bool add_would_overflow(std::size_t a, std::size_t b) noexcept {
    return a > (std::numeric_limits<std::size_t>::max)() - b;
}

inline constexpr bool mul_would_overflow(std::size_t a, std::size_t b) noexcept {
    if (a == 0 || b == 0) return false;
    return a > (std::numeric_limits<std::size_t>::max)() / b;
}

inline constexpr bool size_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (mul_would_overflow(a, b)) return false;
    out = a * b; return true;
}

inline constexpr bool size_add(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (add_would_overflow(a, b)) return false;
    out = a + b; return true;
}

// Checked narrow cast
template<typename To, typename From>
inline constexpr bool checked_narrow(From value, To& out) noexcept {
    static_assert(std::is_integral_v<From> && std::is_integral_v<To>, "checked_narrow: integral only");
    if constexpr (std::is_signed_v<From> == std::is_signed_v<To>) {
        if (value < static_cast<From>(std::numeric_limits<To>::min()) || value > static_cast<From>(std::numeric_limits<To>::max())) return false;
        out = static_cast<To>(value); return true;
    } else if constexpr (std::is_signed_v<From> && !std::is_signed_v<To>) {
        if (value < 0 || static_cast<std::make_unsigned_t<From>>(value) > std::numeric_limits<To>::max()) return false;
        out = static_cast<To>(value); return true;
    } else { // From unsigned, To signed
        if (value > static_cast<std::make_unsigned_t<To>>(std::numeric_limits<To>::max())) return false;
        out = static_cast<To>(value); return true;
    }
}

// Byte spans
template<class T>
inline std::span<const std::byte> as_bytes(std::span<const T> s) noexcept {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size_bytes()};
}

template<class T>
inline std::span<std::byte> as_writable_bytes(std::span<T> s) noexcept {
    return {reinterpret_cast<std::byte*>(s.data()), s.size_bytes()};
}

// Align wrappers (re-export for convenience)
inline constexpr std::size_t align_up_size(std::size_t v, std::size_t a) noexcept { return align_up(v, a); }
inline constexpr std::size_t align_down_size(std::size_t v, std::size_t a) noexcept { return align_down(v, a); }

template<class T>
inline T* align_up_ptr(T* p, std::size_t a) noexcept { return align_up(p, a); }

template<class T>
inline bool is_aligned_ptr(const T* p, std::size_t a) noexcept { return is_aligned(p, a); }

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_UTILS_H


#pragma once

#ifndef CORE_MEMORY_PREFETCH_H
#define CORE_MEMORY_PREFETCH_H

#include <cstddef>
#include <type_traits>

#include <config/config.h>

namespace fem::core::memory {

// Prefetch locality hints. Higher = keep in closer cache longer.
enum class PrefetchLocality : int {
    NTA = 0, // Non-temporal (streaming)
    L3  = 1,
    L2  = 2,
    L1  = 3
};

#if defined(__GNUC__) || defined(__clang__)

inline void prefetch_read(const void* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    switch (locality) {
        case PrefetchLocality::NTA: __builtin_prefetch(p, 0, 0); break;
        case PrefetchLocality::L3:  __builtin_prefetch(p, 0, 1); break;
        case PrefetchLocality::L2:  __builtin_prefetch(p, 0, 2); break;
        case PrefetchLocality::L1:  __builtin_prefetch(p, 0, 3); break;
    }
}

inline void prefetch_write(const void* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    switch (locality) {
        case PrefetchLocality::NTA: __builtin_prefetch(p, 1, 0); break;
        case PrefetchLocality::L3:  __builtin_prefetch(p, 1, 1); break;
        case PrefetchLocality::L2:  __builtin_prefetch(p, 1, 2); break;
        case PrefetchLocality::L1:  __builtin_prefetch(p, 1, 3); break;
    }
}

#elif defined(_MSC_VER)

// MSVC: use _mm_prefetch (read). For write, fall back to read prefetch.
#  include <immintrin.h>

inline void prefetch_read(const void* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    int hint = _MM_HINT_T0;
    switch (locality) {
        case PrefetchLocality::NTA: hint = _MM_HINT_NTA; break;
        case PrefetchLocality::L3:  hint = _MM_HINT_T2;  break;
        case PrefetchLocality::L2:  hint = _MM_HINT_T1;  break;
        case PrefetchLocality::L1:  hint = _MM_HINT_T0;  break;
    }
    _mm_prefetch(static_cast<const char*>(p), hint);
}

inline void prefetch_write(const void* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    // No dedicated write-prefetch intrinsic; use read prefetch as a best-effort.
    prefetch_read(p, locality);
}

#else

inline void prefetch_read(const void*, PrefetchLocality = PrefetchLocality::L1) noexcept {}
inline void prefetch_write(const void*, PrefetchLocality = PrefetchLocality::L1) noexcept {}

#endif

template<class T>
inline void prefetch_read(const T* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    prefetch_read(static_cast<const void*>(p), locality);
}

template<class T>
inline void prefetch_write(const T* p, PrefetchLocality locality = PrefetchLocality::L1) noexcept {
    prefetch_write(static_cast<const void*>(p), locality);
}

// Convenience wrappers
template<class T> inline void prefetch_l1(const T* p) noexcept { prefetch_read(p, PrefetchLocality::L1); }
template<class T> inline void prefetch_l2(const T* p) noexcept { prefetch_read(p, PrefetchLocality::L2); }
template<class T> inline void prefetch_l3(const T* p) noexcept { prefetch_read(p, PrefetchLocality::L3); }
template<class T> inline void prefetch_nta(const T* p) noexcept { prefetch_read(p, PrefetchLocality::NTA); }

} // namespace fem::core::memory

#endif // CORE_MEMORY_PREFETCH_H


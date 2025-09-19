#pragma once

#ifndef CORE_MEMORY_MEMORY_BARRIER_H
#define CORE_MEMORY_MEMORY_BARRIER_H

#include <atomic>
#include <thread>

#include <config/config.h>

namespace fem::core::memory {

// Thread/Compiler fences
inline void signal_fence_acquire() noexcept { std::atomic_signal_fence(std::memory_order_acquire); }
inline void signal_fence_release() noexcept { std::atomic_signal_fence(std::memory_order_release); }
inline void signal_fence_seq_cst() noexcept { std::atomic_signal_fence(std::memory_order_seq_cst); }

inline void thread_fence_acquire() noexcept { std::atomic_thread_fence(std::memory_order_acquire); }
inline void thread_fence_release() noexcept { std::atomic_thread_fence(std::memory_order_release); }
inline void thread_fence_acq_rel() noexcept { std::atomic_thread_fence(std::memory_order_acq_rel); }
inline void thread_fence_seq_cst() noexcept { std::atomic_thread_fence(std::memory_order_seq_cst); }

// Shorthand barriers
inline void read_barrier() noexcept { thread_fence_acquire(); }
inline void write_barrier() noexcept { thread_fence_release(); }
inline void full_barrier() noexcept { thread_fence_seq_cst(); }

// CPU relax/yield hint for spin-wait loops
inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    _mm_pause();
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUG__) || defined(__clang__))
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

// Convenience: spin until predicate is true with bounded spins
template<class Pred>
inline bool spin_until(Pred&& pred, int max_spins = 1024) {
    while (max_spins-- > 0) {
        if (pred()) return true;
        cpu_relax();
    }
    return pred();
}

// Acquire/Release helpers for atomics
template<class T>
inline T load_acquire(const std::atomic<T>& a) noexcept { return a.load(std::memory_order_acquire); }

template<class T>
inline void store_release(std::atomic<T>& a, T v) noexcept { a.store(v, std::memory_order_release); }

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_BARRIER_H


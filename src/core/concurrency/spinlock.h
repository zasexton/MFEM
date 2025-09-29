#pragma once

#ifndef CORE_CONCURRENCY_SPINLOCK_H
#define CORE_CONCURRENCY_SPINLOCK_H

#include <atomic>
#include <chrono>
#include <thread>
#include <cstddef>
#include <immintrin.h>  // For _mm_pause

#include <core/config/config.h>
#include <core/config/debug.h>

namespace fem::core::concurrency {

/**
 * @brief Basic spinlock implementation using test-and-set
 *
 * A simple spinlock that continuously polls an atomic flag until it can
 * acquire the lock. Suitable for very short critical sections where the
 * overhead of blocking would be higher than spinning.
 *
 * Features:
 * - Test-and-set implementation with exponential backoff
 * - CPU pause instructions to reduce power consumption
 * - Configurable maximum spin count before yielding
 * - Compatible with std::lock_guard and similar RAII wrappers
 *
 * Example usage:
 * @code
 * Spinlock spin_mutex;
 *
 * void critical_section() {
 *     std::lock_guard<Spinlock> lock(spin_mutex);
 *     // Short critical section
 * }
 * @endcode
 */
class Spinlock {
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    static constexpr std::size_t max_spins_ = 1000;

public:
    /**
     * @brief Construct a spinlock (initially unlocked)
     */
    Spinlock() = default;

    /**
     * @brief Acquire the spinlock (blocking)
     *
     * Spins until the lock is acquired. Uses exponential backoff
     * to reduce contention and CPU power consumption.
     */
    void lock() {
        std::size_t spin_count = 0;
        std::size_t backoff = 1;

        while (flag_.test_and_set(std::memory_order_acquire)) {
            if (++spin_count >= max_spins_) {
                // Yield to other threads if spinning too long
                std::this_thread::yield();
                spin_count = 0;
                backoff = 1;
            } else {
                // Exponential backoff with CPU pause
                for (std::size_t i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                    _mm_pause();  // x86 pause instruction
#elif defined(__arm__) || defined(__aarch64__)
                    __asm__ __volatile__("yield");  // ARM yield instruction
#else
                    std::this_thread::yield();  // Fallback
#endif
                }
                backoff = std::min(backoff * 2, std::size_t(64));
            }
        }
    }

    /**
     * @brief Try to acquire the spinlock without blocking
     * @return true if lock was acquired, false otherwise
     */
    bool try_lock() noexcept {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    /**
     * @brief Release the spinlock
     */
    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

    // Disable copy and move operations
    Spinlock(const Spinlock&) = delete;
    Spinlock& operator=(const Spinlock&) = delete;
    Spinlock(Spinlock&&) = delete;
    Spinlock& operator=(Spinlock&&) = delete;
};

/**
 * @brief Ticket-based spinlock for fairness
 *
 * A fair spinlock that uses a ticket-based system to ensure FIFO ordering
 * of lock acquisition. Prevents starvation that can occur with basic spinlocks
 * under high contention.
 *
 * Features:
 * - FIFO fairness guarantees
 * - No thread starvation under contention
 * - Cache-line aligned counters to prevent false sharing
 * - Exponential backoff for better performance
 *
 * Example usage:
 * @code
 * TicketSpinlock fair_mutex;
 *
 * void fair_critical_section() {
 *     std::lock_guard<TicketSpinlock> lock(fair_mutex);
 *     // Critical section with fairness guarantees
 * }
 * @endcode
 */
class TicketSpinlock {
private:
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::size_t> next_ticket_{0};
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::size_t> serving_ticket_{0};

public:
    /**
     * @brief Construct a ticket spinlock (initially unlocked)
     */
    TicketSpinlock() = default;

    /**
     * @brief Acquire the spinlock (blocking with fairness)
     */
    void lock() {
        const std::size_t my_ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);

        std::size_t backoff = 1;
        while (serving_ticket_.load(std::memory_order_acquire) != my_ticket) {
            // Exponential backoff
            for (std::size_t i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
                __asm__ __volatile__("yield");
#else
                std::this_thread::yield();
#endif
            }
            backoff = std::min(backoff * 2, std::size_t(64));
        }
    }

    /**
     * @brief Try to acquire the spinlock without blocking
     * @return true if lock was acquired, false otherwise
     */
    bool try_lock() noexcept {
        std::size_t expected = serving_ticket_.load(std::memory_order_acquire);
        return next_ticket_.compare_exchange_weak(expected, expected + 1,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed);
    }

    /**
     * @brief Release the spinlock
     */
    void unlock() noexcept {
        serving_ticket_.fetch_add(1, std::memory_order_release);
    }

    // Disable copy and move operations
    TicketSpinlock(const TicketSpinlock&) = delete;
    TicketSpinlock& operator=(const TicketSpinlock&) = delete;
    TicketSpinlock(TicketSpinlock&&) = delete;
    TicketSpinlock& operator=(TicketSpinlock&&) = delete;
};

/**
 * @brief Reader-writer spinlock
 *
 * A spinlock that allows multiple concurrent readers or a single writer.
 * Optimized for read-heavy workloads where reads are frequent and writes
 * are rare.
 *
 * Features:
 * - Multiple concurrent readers
 * - Exclusive writer access
 * - Reader-preferring policy
 * - Optimized for read-heavy workloads
 *
 * Example usage:
 * @code
 * RWSpinlock rw_mutex;
 *
 * void read_operation() {
 *     std::shared_lock<RWSpinlock> lock(rw_mutex);
 *     // Read-only access
 * }
 *
 * void write_operation() {
 *     std::unique_lock<RWSpinlock> lock(rw_mutex);
 *     // Exclusive write access
 * }
 * @endcode
 */
class RWSpinlock {
private:
    static constexpr std::uint32_t WRITER_BIT = 0x80000000U;
    static constexpr std::uint32_t READER_MASK = 0x7FFFFFFFU;

    std::atomic<std::uint32_t> state_{0};

public:
    /**
     * @brief Construct an RW spinlock (initially unlocked)
     */
    RWSpinlock() = default;

    /**
     * @brief Acquire a shared (read) lock
     */
    void lock_shared() {
        std::size_t backoff = 1;

        while (true) {
            std::uint32_t expected = state_.load(std::memory_order_relaxed);

            // Check if writer is active or waiting
            if (expected & WRITER_BIT) {
                // Wait for writer to finish
                do {
                    for (std::size_t i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                        _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
                        __asm__ __volatile__("yield");
#else
                        std::this_thread::yield();
#endif
                    }
                    backoff = std::min(backoff * 2, std::size_t(64));
                    expected = state_.load(std::memory_order_relaxed);
                } while (expected & WRITER_BIT);
                backoff = 1;
            }

            // Try to increment reader count
            if (state_.compare_exchange_weak(expected, expected + 1,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
                break;
            }
        }
    }

    /**
     * @brief Try to acquire a shared (read) lock without blocking
     * @return true if lock was acquired, false otherwise
     */
    bool try_lock_shared() noexcept {
        std::uint32_t expected = state_.load(std::memory_order_relaxed);

        // Can only acquire if no writer is active
        if (expected & WRITER_BIT) {
            return false;
        }

        return state_.compare_exchange_weak(expected, expected + 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed);
    }

    /**
     * @brief Release a shared (read) lock
     */
    void unlock_shared() noexcept {
        state_.fetch_sub(1, std::memory_order_release);
    }

    /**
     * @brief Acquire an exclusive (write) lock
     */
    void lock() {
        std::size_t backoff = 1;

        while (true) {
            std::uint32_t expected = 0;

            // Try to set writer bit when state is 0 (no readers/writers)
            if (state_.compare_exchange_weak(expected, WRITER_BIT,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
                break;
            }

            // Wait for all readers/writers to finish
            for (std::size_t i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
                __asm__ __volatile__("yield");
#else
                std::this_thread::yield();
#endif
            }
            backoff = std::min(backoff * 2, std::size_t(64));
        }
    }

    /**
     * @brief Try to acquire an exclusive (write) lock without blocking
     * @return true if lock was acquired, false otherwise
     */
    bool try_lock() noexcept {
        std::uint32_t expected = 0;
        return state_.compare_exchange_weak(expected, WRITER_BIT,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed);
    }

    /**
     * @brief Release an exclusive (write) lock
     */
    void unlock() noexcept {
        state_.store(0, std::memory_order_release);
    }

    // Disable copy and move operations
    RWSpinlock(const RWSpinlock&) = delete;
    RWSpinlock& operator=(const RWSpinlock&) = delete;
    RWSpinlock(RWSpinlock&&) = delete;
    RWSpinlock& operator=(RWSpinlock&&) = delete;
};

/**
 * @brief Adaptive spinlock that switches to blocking
 *
 * A hybrid synchronization primitive that starts as a spinlock but
 * falls back to a blocking mutex if contention is detected. Provides
 * the best of both worlds for unknown contention patterns.
 *
 * Features:
 * - Adaptive behavior based on contention detection
 * - Low latency for uncontended cases
 * - Graceful degradation under high contention
 * - Configurable spin threshold
 *
 * Example usage:
 * @code
 * AdaptiveSpinlock adaptive_mutex;
 *
 * void variable_contention_section() {
 *     std::lock_guard<AdaptiveSpinlock> lock(adaptive_mutex);
 *     // Critical section with unknown contention pattern
 * }
 * @endcode
 */
class AdaptiveSpinlock {
private:
    std::atomic_flag spin_flag_ = ATOMIC_FLAG_INIT;
    std::atomic<bool> use_blocking_{false};
    std::mutex blocking_mutex_;
    std::atomic<std::size_t> contention_count_{0};

    static constexpr std::size_t CONTENTION_THRESHOLD = 10;
    static constexpr std::size_t MAX_SPIN_COUNT = 100;

public:
    /**
     * @brief Construct an adaptive spinlock (initially unlocked)
     */
    AdaptiveSpinlock() = default;

    /**
     * @brief Acquire the lock (adaptive spinning/blocking)
     */
    void lock() {
        if (use_blocking_.load(std::memory_order_acquire)) {
            blocking_mutex_.lock();
            return;
        }

        // Try spinning first
        std::size_t spin_count = 0;
        while (spin_flag_.test_and_set(std::memory_order_acquire)) {
            if (++spin_count >= MAX_SPIN_COUNT) {
                // Too much spinning, switch to blocking mode
                contention_count_.fetch_add(1, std::memory_order_relaxed);

                if (contention_count_.load(std::memory_order_relaxed) >= CONTENTION_THRESHOLD) {
                    use_blocking_.store(true, std::memory_order_release);

                    // Release spin lock and acquire blocking mutex
                    spin_flag_.clear(std::memory_order_release);
                    blocking_mutex_.lock();
                    return;
                }

                std::this_thread::yield();
                spin_count = 0;
            } else {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
                __asm__ __volatile__("yield");
#else
                std::this_thread::yield();
#endif
            }
        }
    }

    /**
     * @brief Try to acquire the lock without blocking
     * @return true if lock was acquired, false otherwise
     */
    bool try_lock() noexcept {
        if (use_blocking_.load(std::memory_order_acquire)) {
            return blocking_mutex_.try_lock();
        }

        return !spin_flag_.test_and_set(std::memory_order_acquire);
    }

    /**
     * @brief Release the lock
     */
    void unlock() noexcept {
        if (use_blocking_.load(std::memory_order_acquire)) {
            blocking_mutex_.unlock();
        } else {
            spin_flag_.clear(std::memory_order_release);
        }
    }

    /**
     * @brief Reset contention tracking (for testing/debugging)
     */
    void reset_contention_tracking() {
        contention_count_.store(0, std::memory_order_relaxed);
        use_blocking_.store(false, std::memory_order_relaxed);
    }

    /**
     * @brief Get current contention count
     * @return Number of detected contention events
     */
    std::size_t contention_count() const noexcept {
        return contention_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if currently using blocking mode
     * @return true if using blocking mutex, false if spinning
     */
    bool is_blocking_mode() const noexcept {
        return use_blocking_.load(std::memory_order_acquire);
    }

    // Disable copy and move operations
    AdaptiveSpinlock(const AdaptiveSpinlock&) = delete;
    AdaptiveSpinlock& operator=(const AdaptiveSpinlock&) = delete;
    AdaptiveSpinlock(AdaptiveSpinlock&&) = delete;
    AdaptiveSpinlock& operator=(AdaptiveSpinlock&&) = delete;
};

/**
 * @brief RAII wrapper for shared locks
 *
 * Provides automatic shared lock management for reader-writer spinlocks.
 * Compatible with std::shared_lock interface.
 */
template<typename RWLock>
class shared_spinlock_guard {
private:
    RWLock& lock_;
    bool owns_lock_;

public:
    explicit shared_spinlock_guard(RWLock& lock) : lock_(lock), owns_lock_(true) {
        lock_.lock_shared();
    }

    ~shared_spinlock_guard() {
        if (owns_lock_) {
            lock_.unlock_shared();
        }
    }

    void unlock() {
        if (owns_lock_) {
            lock_.unlock_shared();
            owns_lock_ = false;
        }
    }

    // Disable copy and move operations
    shared_spinlock_guard(const shared_spinlock_guard&) = delete;
    shared_spinlock_guard& operator=(const shared_spinlock_guard&) = delete;
    shared_spinlock_guard(shared_spinlock_guard&&) = delete;
    shared_spinlock_guard& operator=(shared_spinlock_guard&&) = delete;
};

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_SPINLOCK_H
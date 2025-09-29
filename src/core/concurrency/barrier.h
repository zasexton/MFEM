#pragma once

#ifndef CORE_CONCURRENCY_BARRIER_H
#define CORE_CONCURRENCY_BARRIER_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include <core/config/config.h>
#include <core/config/debug.h>

namespace fem::core::concurrency {

/**
 * @brief A reusable synchronization barrier for coordinating multiple threads
 *
 * A barrier allows a set of threads to wait until all threads have reached
 * the barrier point before any thread can proceed. This is useful for
 * implementing phased algorithms where all threads must complete one phase
 * before any can begin the next.
 *
 * Features:
 * - Lock-free implementation using atomic operations
 * - Reusable across multiple synchronization cycles
 * - Generation-based to handle spurious wakeups
 * - Support for dynamic thread count changes
 * - Exception safe
 *
 * Example usage:
 * @code
 * Barrier barrier(4);  // 4 threads
 *
 * // In each thread:
 * void worker_thread() {
 *     // Phase 1 work
 *     do_phase1_work();
 *
 *     barrier.arrive_and_wait();  // Wait for all threads
 *
 *     // Phase 2 work
 *     do_phase2_work();
 * }
 * @endcode
 */
class Barrier {
private:
    // Threshold number of threads
    std::atomic<std::size_t> threshold_;

    // Current count of threads that haven't arrived yet
    std::atomic<std::size_t> count_;

    // Generation counter to distinguish barrier cycles
    std::atomic<std::size_t> generation_;

public:
    /**
     * @brief Construct a barrier for the specified number of threads
     * @param participants Number of threads that must arrive before barrier opens
     */
    explicit Barrier(std::size_t participants)
        : threshold_(participants), count_(participants), generation_(0) {
        FEM_DEBUG_ASSERT(participants > 0);
    }

    /**
     * @brief Arrive at the barrier and wait for all threads
     *
     * This function blocks until all threads have called arrive_and_wait().
     * Once all threads arrive, they are all released simultaneously.
     * The barrier is then reset for the next cycle.
     */
    void arrive_and_wait() {
        auto gen = generation_.load(std::memory_order_acquire);

        // Decrement count atomically and check if we're the last
        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Last thread to arrive - reset for next cycle and release all
            count_.store(threshold_.load(std::memory_order_relaxed), std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
        } else {
            // Wait for generation to change (indicating barrier has opened)
            while (generation_.load(std::memory_order_acquire) == gen) {
                std::this_thread::yield();
            }
        }
    }

    /**
     * @brief Arrive at barrier and immediately drop participation
     *
     * This allows a thread to signal its arrival but not participate in
     * future barrier cycles. Useful when threads finish their work early.
     */
    void arrive_and_drop() {
        // Decrease both count and threshold
        auto old_count = count_.fetch_sub(1, std::memory_order_acq_rel);
        threshold_.fetch_sub(1, std::memory_order_acq_rel);

        // If this was the last thread needed, release the barrier
        if (old_count == 1) {
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    /**
     * @brief Get the current number of participating threads
     * @return Number of threads expected at the barrier
     */
    std::size_t thread_count() const noexcept {
        return threshold_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get the current generation number
     * @return Current barrier generation (increments each cycle)
     */
    std::size_t generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

    /**
     * @brief Check how many threads are still expected
     * @return Number of threads that haven't arrived yet
     */
    std::size_t waiting_count() const noexcept {
        return count_.load(std::memory_order_acquire);
    }

    /**
     * @brief Reset the barrier for a new set of threads
     * @param new_count New number of participating threads
     *
     * Warning: This should only be called when no threads are waiting
     * at the barrier, otherwise behavior is undefined.
     */
    void reset(std::size_t new_count) {
        FEM_DEBUG_ASSERT(new_count > 0);
        threshold_.store(new_count, std::memory_order_release);
        count_.store(new_count, std::memory_order_release);
        // Don't reset generation - let it continue incrementing
    }

    // Disable copy operations
    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;

    // Disable move operations (barriers are meant to be long-lived)
    Barrier(Barrier&&) = delete;
    Barrier& operator=(Barrier&&) = delete;
};

/**
 * @brief Enhanced barrier with completion callback and timeout support
 *
 * FlexBarrier extends the basic Barrier with additional features:
 * - Completion callback executed by the last arriving thread
 * - Timeout support for arrive_and_wait operations
 * - More flexible thread management
 *
 * Example usage:
 * @code
 * FlexBarrier barrier(4, []() {
 *     std::cout << "All threads synchronized!" << std::endl;
 * });
 *
 * // In worker thread:
 * if (barrier.arrive_and_wait_for(std::chrono::seconds(5))) {
 *     // Synchronization successful
 * } else {
 *     // Timeout occurred
 * }
 * @endcode
 */
class FlexBarrier {
private:
    std::atomic<std::size_t> threshold_;
    std::atomic<std::size_t> count_;
    std::atomic<std::size_t> generation_;
    std::function<void()> completion_func_;

    // For timeout support
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;

public:
    /**
     * @brief Construct a flexible barrier
     * @param participants Number of threads that must arrive
     * @param on_completion Function to call when all threads arrive (optional)
     */
    explicit FlexBarrier(std::size_t participants,
                        std::function<void()> on_completion = nullptr)
        : threshold_(participants), count_(participants), generation_(0),
          completion_func_(std::move(on_completion)) {
        FEM_DEBUG_ASSERT(participants > 0);
    }

    /**
     * @brief Arrive at barrier and wait for all threads
     */
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_.load(std::memory_order_acquire);

        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Last thread to arrive
            if (completion_func_) {
                completion_func_();
            }

            // Reset for next cycle
            count_.store(threshold_.load(std::memory_order_relaxed), std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);

            // Wake up all waiting threads
            cv_.notify_all();
        } else {
            // Wait for generation to change
            cv_.wait(lock, [this, gen]() {
                return generation_.load(std::memory_order_acquire) != gen;
            });
        }
    }

    /**
     * @brief Arrive and wait with timeout
     * @param timeout Maximum time to wait
     * @return true if barrier opened, false if timeout occurred
     */
    template<typename Rep, typename Period>
    bool arrive_and_wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_.load(std::memory_order_acquire);

        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Last thread to arrive
            if (completion_func_) {
                completion_func_();
            }

            // Reset for next cycle
            count_.store(threshold_.load(std::memory_order_relaxed), std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);

            // Wake up all waiting threads
            cv_.notify_all();
            return true;
        } else {
            // Wait for generation to change with timeout
            return cv_.wait_for(lock, timeout, [this, gen]() {
                return generation_.load(std::memory_order_acquire) != gen;
            });
        }
    }

    /**
     * @brief Arrive and drop participation
     */
    void arrive_and_drop() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto old_count = count_.fetch_sub(1, std::memory_order_acq_rel);
        threshold_.fetch_sub(1, std::memory_order_acq_rel);

        if (old_count == 1) {
            if (completion_func_) {
                completion_func_();
            }
            generation_.fetch_add(1, std::memory_order_acq_rel);
            cv_.notify_all();
        }
    }

    /**
     * @brief Set or change the completion callback
     * @param callback New completion function (can be nullptr)
     *
     * Warning: This should only be called when no threads are waiting
     */
    void set_completion_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        completion_func_ = std::move(callback);
    }

    /**
     * @brief Get the current number of participating threads
     */
    std::size_t thread_count() const noexcept {
        return threshold_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get the current generation number
     */
    std::size_t generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

    /**
     * @brief Check how many threads are still expected
     */
    std::size_t waiting_count() const noexcept {
        return count_.load(std::memory_order_acquire);
    }

    /**
     * @brief Reset the barrier
     * @param new_count New number of participating threads
     * @param new_callback New completion callback (optional)
     */
    void reset(std::size_t new_count, std::function<void()> new_callback = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        FEM_DEBUG_ASSERT(new_count > 0);

        threshold_.store(new_count, std::memory_order_release);
        count_.store(new_count, std::memory_order_release);

        if (new_callback) {
            completion_func_ = std::move(new_callback);
        }
    }

    // Disable copy and move operations
    FlexBarrier(const FlexBarrier&) = delete;
    FlexBarrier& operator=(const FlexBarrier&) = delete;
    FlexBarrier(FlexBarrier&&) = delete;
    FlexBarrier& operator=(FlexBarrier&&) = delete;
};

/**
 * @brief RAII helper for automatic barrier participation
 *
 * Automatically calls arrive_and_wait() on destruction, ensuring
 * barrier participation even in the presence of exceptions.
 *
 * Example usage:
 * @code
 * void worker_function(Barrier& barrier) {
 *     BarrierGuard guard(barrier);
 *
 *     // Do work - barrier will be triggered even if exception occurs
 *     risky_operation();
 *
 *     // barrier.arrive_and_wait() called automatically here
 * }
 * @endcode
 */
class BarrierGuard {
private:
    Barrier* barrier_;
    bool armed_;

public:
    /**
     * @brief Construct a barrier guard
     * @param barrier Reference to barrier to participate in
     */
    explicit BarrierGuard(Barrier& barrier) : barrier_(&barrier), armed_(true) {}

    /**
     * @brief Destructor automatically calls arrive_and_wait if still armed
     */
    ~BarrierGuard() {
        if (armed_) {
            barrier_->arrive_and_wait();
        }
    }

    /**
     * @brief Manually trigger barrier and disarm guard
     */
    void trigger() {
        if (armed_) {
            barrier_->arrive_and_wait();
            armed_ = false;
        }
    }

    /**
     * @brief Drop participation and disarm guard
     */
    void drop() {
        if (armed_) {
            barrier_->arrive_and_drop();
            armed_ = false;
        }
    }

    /**
     * @brief Disarm guard without triggering barrier
     */
    void disarm() {
        armed_ = false;
    }

    // Disable copy and move operations
    BarrierGuard(const BarrierGuard&) = delete;
    BarrierGuard& operator=(const BarrierGuard&) = delete;
    BarrierGuard(BarrierGuard&&) = delete;
    BarrierGuard& operator=(BarrierGuard&&) = delete;
};

/**
 * @brief Counting semaphore for resource management
 *
 * A semaphore maintains a count and allows threads to acquire/release
 * permits. Useful for controlling access to a limited number of resources.
 *
 * Example usage:
 * @code
 * Semaphore pool_semaphore(3);  // 3 connection pool
 *
 * void use_connection() {
 *     pool_semaphore.acquire();  // Get a connection
 *
 *     // Use connection
 *     process_request();
 *
 *     pool_semaphore.release();  // Return connection
 * }
 * @endcode
 */
class Semaphore {
private:
    std::atomic<std::ptrdiff_t> count_;
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;

public:
    /**
     * @brief Construct semaphore with initial count
     * @param initial_count Initial number of available permits
     */
    explicit Semaphore(std::ptrdiff_t initial_count = 0)
        : count_(initial_count) {
        FEM_DEBUG_ASSERT(initial_count >= 0);
    }

    /**
     * @brief Acquire a permit (blocking)
     */
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_.load() > 0; });
        count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    /**
     * @brief Try to acquire a permit without blocking
     * @return true if permit acquired, false otherwise
     */
    bool try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto current = count_.load();
        if (current > 0) {
            count_.fetch_sub(1, std::memory_order_acq_rel);
            return true;
        }
        return false;
    }

    /**
     * @brief Try to acquire with timeout
     * @param timeout Maximum time to wait
     * @return true if permit acquired, false if timeout
     */
    template<typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, timeout, [this] { return count_.load() > 0; })) {
            count_.fetch_sub(1, std::memory_order_acq_rel);
            return true;
        }
        return false;
    }

    /**
     * @brief Release a permit
     */
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count_.fetch_add(1, std::memory_order_acq_rel);
        }
        cv_.notify_one();
    }

    /**
     * @brief Release multiple permits
     * @param count Number of permits to release
     */
    void release(std::ptrdiff_t count) {
        FEM_DEBUG_ASSERT(count > 0);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count_.fetch_add(count, std::memory_order_acq_rel);
        }
        cv_.notify_all();
    }

    /**
     * @brief Get current available permits
     */
    std::ptrdiff_t available() const noexcept {
        return count_.load(std::memory_order_acquire);
    }

    // Disable copy and move operations
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore(Semaphore&&) = delete;
    Semaphore& operator=(Semaphore&&) = delete;
};

/**
 * @brief RAII semaphore guard for automatic resource management
 */
class SemaphoreGuard {
private:
    Semaphore* semaphore_;
    bool acquired_;

public:
    explicit SemaphoreGuard(Semaphore& sem) : semaphore_(&sem), acquired_(false) {
        semaphore_->acquire();
        acquired_ = true;
    }

    ~SemaphoreGuard() {
        if (acquired_) {
            semaphore_->release();
        }
    }

    void release() {
        if (acquired_) {
            semaphore_->release();
            acquired_ = false;
        }
    }

    // Disable copy and move operations
    SemaphoreGuard(const SemaphoreGuard&) = delete;
    SemaphoreGuard& operator=(const SemaphoreGuard&) = delete;
    SemaphoreGuard(SemaphoreGuard&&) = delete;
    SemaphoreGuard& operator=(SemaphoreGuard&&) = delete;
};

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_BARRIER_H
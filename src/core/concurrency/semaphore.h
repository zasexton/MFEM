#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <limits>
#include <thread>
#include <optional>
#include <cassert>
#include <cstdint>

namespace core {
namespace concurrency {

// Forward declarations
template<std::ptrdiff_t LeastMaxValue = std::numeric_limits<std::ptrdiff_t>::max()>
class CountingSemaphore;

using BinarySemaphore = CountingSemaphore<1>;

/**
 * @class CountingSemaphore
 * @brief A counting semaphore implementation with configurable maximum value
 *
 * This class implements a standard counting semaphore with acquire/release
 * semantics. It supports timed operations and try operations.
 */
template<std::ptrdiff_t LeastMaxValue>
class CountingSemaphore {
public:
    static constexpr std::ptrdiff_t max() noexcept { return LeastMaxValue; }

    /**
     * @brief Construct a semaphore with initial count
     * @param desired Initial count value
     */
    explicit CountingSemaphore(std::ptrdiff_t desired)
        : count_(desired) {
        assert(desired >= 0 && desired <= max());
    }

    // Disable copy/move
    CountingSemaphore(const CountingSemaphore&) = delete;
    CountingSemaphore& operator=(const CountingSemaphore&) = delete;
    CountingSemaphore(CountingSemaphore&&) = delete;
    CountingSemaphore& operator=(CountingSemaphore&&) = delete;

    /**
     * @brief Acquire one unit from the semaphore
     *
     * Blocks until the count is positive, then decrements it
     */
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ > 0; });
        --count_;
    }

    /**
     * @brief Try to acquire one unit without blocking
     * @return true if successfully acquired, false otherwise
     */
    bool try_acquire() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ > 0) {
            --count_;
            return true;
        }
        return false;
    }

    /**
     * @brief Try to acquire with timeout
     * @param rel_time Relative timeout duration
     * @return true if successfully acquired, false on timeout
     */
    template<typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, rel_time, [this] { return count_ > 0; })) {
            --count_;
            return true;
        }
        return false;
    }

    /**
     * @brief Try to acquire until absolute time point
     * @param abs_time Absolute time point for timeout
     * @return true if successfully acquired, false on timeout
     */
    template<typename Clock, typename Duration>
    bool try_acquire_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_until(lock, abs_time, [this] { return count_ > 0; })) {
            --count_;
            return true;
        }
        return false;
    }

    /**
     * @brief Release one unit to the semaphore
     * @param update Number of units to release (default 1)
     */
    void release(std::ptrdiff_t update = 1) {
        assert(update >= 0);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(count_ + update <= max());
            count_ += update;
        }
        if (update == 1) {
            cv_.notify_one();
        } else {
            cv_.notify_all();
        }
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::ptrdiff_t count_;
};

/**
 * @class LightweightSemaphore
 * @brief A lightweight semaphore using atomic operations where possible
 *
 * This implementation tries to avoid mutex operations for common cases
 * by using atomic operations and only falling back to mutex/condvar
 * when necessary.
 */
template<std::ptrdiff_t LeastMaxValue = std::numeric_limits<std::ptrdiff_t>::max()>
class LightweightSemaphore {
public:
    static constexpr std::ptrdiff_t max() noexcept { return LeastMaxValue; }

    explicit LightweightSemaphore(std::ptrdiff_t desired)
        : count_(desired) {
        assert(desired >= 0 && desired <= max());
    }

    // Disable copy/move
    LightweightSemaphore(const LightweightSemaphore&) = delete;
    LightweightSemaphore& operator=(const LightweightSemaphore&) = delete;
    LightweightSemaphore(LightweightSemaphore&&) = delete;
    LightweightSemaphore& operator=(LightweightSemaphore&&) = delete;

    void acquire() {
        // Fast path: try atomic decrement
        std::ptrdiff_t old_count = count_.load(std::memory_order_relaxed);
        while (old_count > 0) {
            if (count_.compare_exchange_weak(old_count, old_count - 1,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return;
            }
        }

        // Slow path: must wait
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiters_;
        cv_.wait(lock, [this] {
            std::ptrdiff_t c = count_.load(std::memory_order_relaxed);
            if (c > 0) {
                count_.store(c - 1, std::memory_order_relaxed);
                return true;
            }
            return false;
        });
        --waiters_;
    }

    bool try_acquire() noexcept {
        std::ptrdiff_t old_count = count_.load(std::memory_order_relaxed);
        while (old_count > 0) {
            if (count_.compare_exchange_weak(old_count, old_count - 1,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) {
        // First try fast path
        if (try_acquire()) {
            return true;
        }

        // Slow path with timeout
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiters_;
        bool acquired = cv_.wait_for(lock, rel_time, [this] {
            std::ptrdiff_t c = count_.load(std::memory_order_relaxed);
            if (c > 0) {
                count_.store(c - 1, std::memory_order_relaxed);
                return true;
            }
            return false;
        });
        --waiters_;
        return acquired;
    }

    void release(std::ptrdiff_t update = 1) {
        assert(update >= 0);

        std::ptrdiff_t old_count = count_.fetch_add(update, std::memory_order_release);
        assert(old_count + update <= max());

        // Wake waiters if necessary
        if (old_count < 0 || waiters_.load(std::memory_order_relaxed) > 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (update == 1) {
                cv_.notify_one();
            } else {
                cv_.notify_all();
            }
        }
    }

private:
    alignas(64) std::atomic<std::ptrdiff_t> count_;
    alignas(64) std::atomic<std::ptrdiff_t> waiters_{0};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

/**
 * @class FairSemaphore
 * @brief A fair semaphore that ensures FIFO ordering of waiters
 *
 * This implementation maintains fairness by serving waiters in
 * first-come-first-served order.
 */
template<std::ptrdiff_t LeastMaxValue = std::numeric_limits<std::ptrdiff_t>::max()>
class FairSemaphore {
public:
    static constexpr std::ptrdiff_t max() noexcept { return LeastMaxValue; }

    explicit FairSemaphore(std::ptrdiff_t desired)
        : count_(desired), next_ticket_(0), now_serving_(0) {
        assert(desired >= 0 && desired <= max());
    }

    // Disable copy/move
    FairSemaphore(const FairSemaphore&) = delete;
    FairSemaphore& operator=(const FairSemaphore&) = delete;
    FairSemaphore(FairSemaphore&&) = delete;
    FairSemaphore& operator=(FairSemaphore&&) = delete;

    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Take a ticket
        uint64_t my_ticket = next_ticket_++;

        // Wait for our turn and for count to be positive
        cv_.wait(lock, [this, my_ticket] {
            return my_ticket == now_serving_ && count_ > 0;
        });

        --count_;
        ++now_serving_;
        cv_.notify_all();  // Wake up next waiter
    }

    bool try_acquire() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        // Can only acquire if no one is waiting and count is positive
        if (next_ticket_ == now_serving_ && count_ > 0) {
            --count_;
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Take a ticket
        uint64_t my_ticket = next_ticket_++;

        // Wait with timeout
        bool acquired = cv_.wait_for(lock, rel_time, [this, my_ticket] {
            return my_ticket == now_serving_ && count_ > 0;
        });

        if (acquired) {
            --count_;
            ++now_serving_;
            cv_.notify_all();
        } else {
            // Timed out - need to handle ticket cancellation
            // This is complex in general case, simplified here
            if (my_ticket == now_serving_) {
                ++now_serving_;
                cv_.notify_all();
            }
        }

        return acquired;
    }

    void release(std::ptrdiff_t update = 1) {
        assert(update >= 0);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(count_ + update <= max());
            count_ += update;
        }
        cv_.notify_all();  // Wake all waiters to check their turn
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::ptrdiff_t count_;
    uint64_t next_ticket_;
    uint64_t now_serving_;
};

/**
 * @class WeightedSemaphore
 * @brief A semaphore that supports weighted acquire/release operations
 *
 * This allows acquiring/releasing multiple units at once, useful for
 * resource management scenarios.
 */
class WeightedSemaphore {
public:
    explicit WeightedSemaphore(std::ptrdiff_t initial_count)
        : count_(initial_count) {
        assert(initial_count >= 0);
    }

    // Disable copy/move
    WeightedSemaphore(const WeightedSemaphore&) = delete;
    WeightedSemaphore& operator=(const WeightedSemaphore&) = delete;
    WeightedSemaphore(WeightedSemaphore&&) = delete;
    WeightedSemaphore& operator=(WeightedSemaphore&&) = delete;

    /**
     * @brief Acquire multiple units from the semaphore
     * @param weight Number of units to acquire
     */
    void acquire(std::ptrdiff_t weight = 1) {
        assert(weight > 0);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, weight] { return count_ >= weight; });
        count_ -= weight;
    }

    /**
     * @brief Try to acquire multiple units without blocking
     * @param weight Number of units to acquire
     * @return true if successfully acquired, false otherwise
     */
    bool try_acquire(std::ptrdiff_t weight = 1) noexcept {
        assert(weight > 0);
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ >= weight) {
            count_ -= weight;
            return true;
        }
        return false;
    }

    /**
     * @brief Try to acquire multiple units with timeout
     * @param weight Number of units to acquire
     * @param rel_time Relative timeout duration
     * @return true if successfully acquired, false on timeout
     */
    template<typename Rep, typename Period>
    bool try_acquire_for(std::ptrdiff_t weight,
                         const std::chrono::duration<Rep, Period>& rel_time) {
        assert(weight > 0);
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, rel_time, [this, weight] { return count_ >= weight; })) {
            count_ -= weight;
            return true;
        }
        return false;
    }

    /**
     * @brief Release multiple units to the semaphore
     * @param weight Number of units to release
     */
    void release(std::ptrdiff_t weight = 1) {
        assert(weight > 0);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count_ += weight;
        }
        cv_.notify_all();  // Wake all waiters since we don't know their weights
    }

    /**
     * @brief Get current count (for debugging)
     * @return Current semaphore count
     */
    std::ptrdiff_t available() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::ptrdiff_t count_;
};

/**
 * @class SemaphoreGuard
 * @brief RAII guard for automatic semaphore acquire/release
 */
template<typename Semaphore>
class SemaphoreGuard {
public:
    explicit SemaphoreGuard(Semaphore& sem)
        : semaphore_(&sem), owns_(true) {
        semaphore_->acquire();
    }

    explicit SemaphoreGuard(Semaphore& sem, std::defer_lock_t) noexcept
        : semaphore_(&sem), owns_(false) {}

    explicit SemaphoreGuard(Semaphore& sem, std::try_to_lock_t)
        : semaphore_(&sem), owns_(sem.try_acquire()) {}

    template<typename Rep, typename Period>
    SemaphoreGuard(Semaphore& sem,
                   const std::chrono::duration<Rep, Period>& timeout)
        : semaphore_(&sem), owns_(sem.try_acquire_for(timeout)) {}

    ~SemaphoreGuard() {
        if (owns_) {
            semaphore_->release();
        }
    }

    // Disable copy, enable move
    SemaphoreGuard(const SemaphoreGuard&) = delete;
    SemaphoreGuard& operator=(const SemaphoreGuard&) = delete;

    SemaphoreGuard(SemaphoreGuard&& other) noexcept
        : semaphore_(other.semaphore_), owns_(other.owns_) {
        other.owns_ = false;
    }

    SemaphoreGuard& operator=(SemaphoreGuard&& other) noexcept {
        if (this != &other) {
            if (owns_) {
                semaphore_->release();
            }
            semaphore_ = other.semaphore_;
            owns_ = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }

    void acquire() {
        if (!owns_) {
            semaphore_->acquire();
            owns_ = true;
        }
    }

    bool try_acquire() {
        if (!owns_) {
            owns_ = semaphore_->try_acquire();
        }
        return owns_;
    }

    template<typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& timeout) {
        if (!owns_) {
            owns_ = semaphore_->try_acquire_for(timeout);
        }
        return owns_;
    }

    void release() {
        if (owns_) {
            semaphore_->release();
            owns_ = false;
        }
    }

    bool owns_lock() const noexcept { return owns_; }
    explicit operator bool() const noexcept { return owns_; }

    Semaphore* mutex() const noexcept { return semaphore_; }

private:
    Semaphore* semaphore_;
    bool owns_;
};

/**
 * @class MultiResourceSemaphore
 * @brief A semaphore that manages multiple resource types
 *
 * Useful for scenarios where you need to acquire multiple different
 * resources atomically.
 */
template<size_t NumResources>
class MultiResourceSemaphore {
public:
    /**
     * @brief Construct with initial counts for each resource
     * @param initial_counts Array of initial counts
     */
    explicit MultiResourceSemaphore(const std::array<std::ptrdiff_t, NumResources>& initial_counts)
        : counts_(initial_counts) {
        for (auto count : counts_) {
            assert(count >= 0);
        }
    }

    /**
     * @brief Acquire specified amounts of each resource
     * @param weights Array of weights for each resource
     */
    void acquire(const std::array<std::ptrdiff_t, NumResources>& weights) {
        for (auto weight : weights) {
            assert(weight >= 0);
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, &weights] {
            for (size_t i = 0; i < NumResources; ++i) {
                if (counts_[i] < weights[i]) {
                    return false;
                }
            }
            return true;
        });

        for (size_t i = 0; i < NumResources; ++i) {
            counts_[i] -= weights[i];
        }
    }

    /**
     * @brief Try to acquire specified amounts without blocking
     * @param weights Array of weights for each resource
     * @return true if all resources acquired, false otherwise
     */
    bool try_acquire(const std::array<std::ptrdiff_t, NumResources>& weights) noexcept {
        for (auto weight : weights) {
            assert(weight >= 0);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < NumResources; ++i) {
            if (counts_[i] < weights[i]) {
                return false;
            }
        }

        for (size_t i = 0; i < NumResources; ++i) {
            counts_[i] -= weights[i];
        }
        return true;
    }

    /**
     * @brief Release specified amounts of each resource
     * @param weights Array of weights for each resource
     */
    void release(const std::array<std::ptrdiff_t, NumResources>& weights) {
        for (auto weight : weights) {
            assert(weight >= 0);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < NumResources; ++i) {
                counts_[i] += weights[i];
            }
        }
        cv_.notify_all();
    }

    /**
     * @brief Get current counts for all resources
     * @return Array of current counts
     */
    std::array<std::ptrdiff_t, NumResources> available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return counts_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::array<std::ptrdiff_t, NumResources> counts_;
};

} // namespace concurrency
} // namespace core
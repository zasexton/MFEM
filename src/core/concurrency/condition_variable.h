// File: core/concurrency/condition_variable.h
// License: MIT
// Description: Enhanced condition variable implementations for the core library

#pragma once

#include <condition_variable>
#include <mutex>
#include <chrono>
#include <atomic>
#include <functional>
#include <queue>
#include <vector>
#include <optional>
#include <cassert>
#include <limits>
#include <memory>

namespace core {
namespace concurrency {

/**
 * @class ConditionVariableAny
 * @brief Enhanced condition_variable_any with additional features
 *
 * This class extends std::condition_variable_any with debugging and
 * performance monitoring capabilities.
 */
class ConditionVariableAny {
public:
    ConditionVariableAny() = default;

    template<typename Lock>
    void wait(Lock& lock) {
        ++waiters_count_;
        cv_.wait(lock);
        --waiters_count_;
        ++wake_count_;
    }

    template<typename Lock, typename Predicate>
    void wait(Lock& lock, Predicate pred) {
        while (!pred()) {
            wait(lock);
        }
    }

    template<typename Lock, typename Clock, typename Duration>
    std::cv_status wait_until(Lock& lock,
                              const std::chrono::time_point<Clock, Duration>& abs_time) {
        ++waiters_count_;
        auto status = cv_.wait_until(lock, abs_time);
        --waiters_count_;
        if (status == std::cv_status::no_timeout) {
            ++wake_count_;
        } else {
            ++timeout_count_;
        }
        return status;
    }

    template<typename Lock, typename Clock, typename Duration, typename Predicate>
    bool wait_until(Lock& lock,
                    const std::chrono::time_point<Clock, Duration>& abs_time,
                    Predicate pred) {
        while (!pred()) {
            if (wait_until(lock, abs_time) == std::cv_status::timeout) {
                return pred();
            }
        }
        return true;
    }

    template<typename Lock, typename Rep, typename Period>
    std::cv_status wait_for(Lock& lock,
                            const std::chrono::duration<Rep, Period>& rel_time) {
        return wait_until(lock, std::chrono::steady_clock::now() + rel_time);
    }

    template<typename Lock, typename Rep, typename Period, typename Predicate>
    bool wait_for(Lock& lock,
                  const std::chrono::duration<Rep, Period>& rel_time,
                  Predicate pred) {
        return wait_until(lock, std::chrono::steady_clock::now() + rel_time, pred);
    }

    void notify_one() noexcept {
        ++notify_one_count_;
        cv_.notify_one();
    }

    void notify_all() noexcept {
        ++notify_all_count_;
        cv_.notify_all();
    }

    // Statistics
    size_t get_waiters_count() const noexcept { return waiters_count_.load(); }
    size_t get_wake_count() const noexcept { return wake_count_.load(); }
    size_t get_timeout_count() const noexcept { return timeout_count_.load(); }
    size_t get_notify_one_count() const noexcept { return notify_one_count_.load(); }
    size_t get_notify_all_count() const noexcept { return notify_all_count_.load(); }

private:
    std::condition_variable_any cv_;
    std::atomic<size_t> waiters_count_{0};
    std::atomic<size_t> wake_count_{0};
    std::atomic<size_t> timeout_count_{0};
    std::atomic<size_t> notify_one_count_{0};
    std::atomic<size_t> notify_all_count_{0};
};

/**
 * @class EventConditionVariable
 * @brief A condition variable that can be signaled without holding a lock
 *
 * This is useful for event-based synchronization where the notifier
 * doesn't need to hold a lock.
 */
class EventConditionVariable {
public:
    EventConditionVariable() : signaled_(false), generation_(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;
        cv_.wait(lock, [this, gen] { return signaled_ || generation_ != gen; });
        if (auto_reset_ && signaled_) {
            signaled_ = false;
        }
    }

    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;
        bool result = cv_.wait_for(lock, rel_time,
                                   [this, gen] { return signaled_ || generation_ != gen; });
        if (result && auto_reset_ && signaled_) {
            signaled_ = false;
        }
        return result;
    }

    template<typename Clock, typename Duration>
    bool wait_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;
        bool result = cv_.wait_until(lock, abs_time,
                                     [this, gen] { return signaled_ || generation_ != gen; });
        if (result && auto_reset_ && signaled_) {
            signaled_ = false;
        }
        return result;
    }

    void signal() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_ = true;
            ++generation_;
        }
        cv_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        signaled_ = false;
    }

    bool is_signaled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return signaled_;
    }

    void set_auto_reset(bool auto_reset) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_reset_ = auto_reset;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool signaled_;
    bool auto_reset_{false};
    size_t generation_;
};

/**
 * @class BroadcastConditionVariable
 * @brief A condition variable optimized for broadcast patterns
 *
 * This implementation is optimized for scenarios where notify_all is
 * more common than notify_one.
 */
class BroadcastConditionVariable {
public:
    BroadcastConditionVariable() : epoch_(0) {}

    template<typename Lock>
    void wait(Lock& lock) {
        uint64_t my_epoch = epoch_.load(std::memory_order_acquire);

        std::unique_lock<std::mutex> wait_lock(wait_mutex_);
        ++waiters_;

        lock.unlock();
        cv_.wait(wait_lock, [this, my_epoch] {
            return epoch_.load(std::memory_order_acquire) != my_epoch;
        });
        --waiters_;
        wait_lock.unlock();

        lock.lock();
    }

    template<typename Lock, typename Predicate>
    void wait(Lock& lock, Predicate pred) {
        while (!pred()) {
            wait(lock);
        }
    }

    template<typename Lock, typename Rep, typename Period>
    std::cv_status wait_for(Lock& lock,
                            const std::chrono::duration<Rep, Period>& rel_time) {
        uint64_t my_epoch = epoch_.load(std::memory_order_acquire);

        std::unique_lock<std::mutex> wait_lock(wait_mutex_);
        ++waiters_;

        lock.unlock();
        bool result = cv_.wait_for(wait_lock, rel_time, [this, my_epoch] {
            return epoch_.load(std::memory_order_acquire) != my_epoch;
        });
        --waiters_;
        wait_lock.unlock();

        lock.lock();

        return result ? std::cv_status::no_timeout : std::cv_status::timeout;
    }

    void notify_all() noexcept {
        epoch_.fetch_add(1, std::memory_order_release);
        std::lock_guard<std::mutex> lock(wait_mutex_);
        cv_.notify_all();
    }

    void notify_one() noexcept {
        // For broadcast-optimized CV, notify_one is implemented as notify_all
        // This is less efficient for single wake but maintains correctness
        notify_all();
    }

    size_t get_waiters_count() const {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        return waiters_;
    }

private:
    mutable std::mutex wait_mutex_;
    std::condition_variable cv_;
    std::atomic<uint64_t> epoch_;
    size_t waiters_{0};
};

/**
 * @class PriorityConditionVariable
 * @brief A condition variable that wakes waiters based on priority
 *
 * Higher priority waiters are woken first when notify_one is called.
 */
template<typename Priority = int>
class PriorityConditionVariable {
public:
    PriorityConditionVariable() = default;

    template<typename Lock>
    void wait(Lock& lock, Priority priority = Priority{}) {
        auto waiter = std::make_shared<Waiter>(priority);

        {
            std::unique_lock<std::mutex> queue_lock(queue_mutex_);
            waiters_.push(waiter);
        }

        std::unique_lock<std::mutex> waiter_lock(waiter->mutex);
        lock.unlock();
        waiter->cv.wait(waiter_lock, [waiter] { return waiter->signaled; });
        lock.lock();
    }

    template<typename Lock, typename Predicate>
    void wait(Lock& lock, Priority priority, Predicate pred) {
        while (!pred()) {
            wait(lock, priority);
        }
    }

    template<typename Lock, typename Rep, typename Period>
    std::cv_status wait_for(Lock& lock,
                            const std::chrono::duration<Rep, Period>& rel_time,
                            Priority priority = Priority{}) {
        auto waiter = std::make_shared<Waiter>(priority);

        {
            std::unique_lock<std::mutex> queue_lock(queue_mutex_);
            waiters_.push(waiter);
        }

        std::unique_lock<std::mutex> waiter_lock(waiter->mutex);
        lock.unlock();
        bool result = waiter->cv.wait_for(waiter_lock, rel_time,
                                          [waiter] { return waiter->signaled; });
        lock.lock();

        if (!result) {
            // Timed out - remove from queue if still there
            std::unique_lock<std::mutex> queue_lock(queue_mutex_);
            // Note: In production, would need proper removal logic
        }

        return result ? std::cv_status::no_timeout : std::cv_status::timeout;
    }

    void notify_one() {
        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        if (!waiters_.empty()) {
            auto waiter = waiters_.top();
            waiters_.pop();
            queue_lock.unlock();

            {
                std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
                waiter->signaled = true;
            }
            waiter->cv.notify_one();
        }
    }

    void notify_all() {
        std::vector<std::shared_ptr<Waiter>> to_notify;

        {
            std::unique_lock<std::mutex> queue_lock(queue_mutex_);
            while (!waiters_.empty()) {
                to_notify.push_back(waiters_.top());
                waiters_.pop();
            }
        }

        for (auto& waiter : to_notify) {
            {
                std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
                waiter->signaled = true;
            }
            waiter->cv.notify_one();
        }
    }

    size_t get_waiters_count() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return waiters_.size();
    }

private:
    struct Waiter {
        Priority priority;
        std::mutex mutex;
        std::condition_variable cv;
        bool signaled{false};

        explicit Waiter(Priority p) : priority(p) {}
    };

    struct WaiterCompare {
        bool operator()(const std::shared_ptr<Waiter>& a,
                       const std::shared_ptr<Waiter>& b) const {
            // Higher priority = smaller value in priority queue (min-heap by default)
            return a->priority < b->priority;
        }
    };

    mutable std::mutex queue_mutex_;
    std::priority_queue<std::shared_ptr<Waiter>,
                       std::vector<std::shared_ptr<Waiter>>,
                       WaiterCompare> waiters_;
};

/**
 * @class BarrierConditionVariable
 * @brief A condition variable that waits for N threads before proceeding
 *
 * Similar to a barrier but using condition variable semantics.
 */
class BarrierConditionVariable {
public:
    explicit BarrierConditionVariable(size_t count)
        : threshold_(count), count_(count), generation_(0) {
        assert(count > 0);
    }

    template<typename Lock>
    bool wait(Lock& lock) {
        std::unique_lock<std::mutex> barrier_lock(mutex_);
        size_t gen = generation_;

        if (--count_ == 0) {
            // Last thread to arrive
            generation_++;
            count_ = threshold_;
            barrier_lock.unlock();
            cv_.notify_all();
            return true;  // This thread is the trigger
        }

        // Wait for the barrier to be triggered
        lock.unlock();
        cv_.wait(barrier_lock, [this, gen] { return generation_ != gen; });
        lock.lock();
        return false;  // This thread was waiting
    }

    template<typename Lock, typename Rep, typename Period>
    std::optional<bool> wait_for(Lock& lock,
                                 const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> barrier_lock(mutex_);
        size_t gen = generation_;

        if (--count_ == 0) {
            // Last thread to arrive
            generation_++;
            count_ = threshold_;
            barrier_lock.unlock();
            cv_.notify_all();
            return true;  // This thread is the trigger
        }

        // Wait for the barrier to be triggered
        lock.unlock();
        bool result = cv_.wait_for(barrier_lock, rel_time,
                                   [this, gen] { return generation_ != gen; });
        lock.lock();

        if (!result) {
            // Timed out - restore count
            ++count_;
            return std::nullopt;
        }
        return false;  // This thread was waiting
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = threshold_;
        generation_++;
    }

    size_t get_waiting_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return threshold_ - count_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const size_t threshold_;
    size_t count_;
    size_t generation_;
};

/**
 * @class FutureConditionVariable
 * @brief A condition variable that can be waited on once and automatically cleaned up
 *
 * Similar to std::future but using condition variable interface.
 */
template<typename T>
class FutureConditionVariable {
public:
    FutureConditionVariable() = default;

    template<typename Lock>
    T wait(Lock& lock) {
        std::unique_lock<std::mutex> future_lock(mutex_);
        lock.unlock();
        cv_.wait(future_lock, [this] { return ready_ || exception_; });
        lock.lock();

        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return std::move(value_);
    }

    template<typename Lock, typename Rep, typename Period>
    std::optional<T> wait_for(Lock& lock,
                              const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> future_lock(mutex_);
        lock.unlock();
        bool result = cv_.wait_for(future_lock, rel_time,
                                   [this] { return ready_ || exception_; });
        lock.lock();

        if (!result) {
            return std::nullopt;
        }

        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return std::move(value_);
    }

    void set_value(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_ || exception_) {
                throw std::logic_error("FutureConditionVariable already set");
            }
            value_ = std::move(value);
            ready_ = true;
        }
        cv_.notify_all();
    }

    void set_exception(std::exception_ptr ex) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_ || exception_) {
                throw std::logic_error("FutureConditionVariable already set");
            }
            exception_ = ex;
        }
        cv_.notify_all();
    }

    bool is_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_ || exception_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    T value_;
    bool ready_{false};
    std::exception_ptr exception_;
};

// Specialization for void
template<>
class FutureConditionVariable<void> {
public:
    FutureConditionVariable() = default;

    template<typename Lock>
    void wait(Lock& lock) {
        std::unique_lock<std::mutex> future_lock(mutex_);
        lock.unlock();
        cv_.wait(future_lock, [this] { return ready_ || exception_; });
        lock.lock();

        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

    template<typename Lock, typename Rep, typename Period>
    bool wait_for(Lock& lock,
                  const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> future_lock(mutex_);
        lock.unlock();
        bool result = cv_.wait_for(future_lock, rel_time,
                                   [this] { return ready_ || exception_; });
        lock.lock();

        if (result && exception_) {
            std::rethrow_exception(exception_);
        }
        return result;
    }

    void set_value() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_ || exception_) {
                throw std::logic_error("FutureConditionVariable already set");
            }
            ready_ = true;
        }
        cv_.notify_all();
    }

    void set_exception(std::exception_ptr ex) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_ || exception_) {
                throw std::logic_error("FutureConditionVariable already set");
            }
            exception_ = ex;
        }
        cv_.notify_all();
    }

    bool is_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_ || exception_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool ready_{false};
    std::exception_ptr exception_;
};

/**
 * @class CountdownConditionVariable
 * @brief A condition variable that triggers after N events
 *
 * Useful for waiting until multiple operations complete.
 */
class CountdownConditionVariable {
public:
    explicit CountdownConditionVariable(size_t count)
        : count_(count), initial_count_(count) {
        assert(count > 0);
    }

    template<typename Lock>
    void wait(Lock& lock) {
        std::unique_lock<std::mutex> countdown_lock(mutex_);
        lock.unlock();
        cv_.wait(countdown_lock, [this] { return count_ == 0; });
        lock.lock();
    }

    template<typename Lock, typename Rep, typename Period>
    bool wait_for(Lock& lock,
                  const std::chrono::duration<Rep, Period>& rel_time) {
        std::unique_lock<std::mutex> countdown_lock(mutex_);
        lock.unlock();
        bool result = cv_.wait_for(countdown_lock, rel_time,
                                   [this] { return count_ == 0; });
        lock.lock();
        return result;
    }

    void count_down(size_t n = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (n > count_) {
            count_ = 0;
        } else {
            count_ -= n;
        }

        if (count_ == 0) {
            lock.unlock();
            cv_.notify_all();
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = initial_count_;
    }

    size_t get_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t count_;
    const size_t initial_count_;
};

} // namespace concurrency
} // namespace core
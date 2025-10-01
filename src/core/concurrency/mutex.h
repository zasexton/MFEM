// Enhanced mutex types with debugging support, timeouts, and RAII wrappers
//
// Provides:
// - TimedMutex: mutex with try_lock_for/try_lock_until support
// - RecursiveMutex: reentrant mutex with ownership tracking
// - DebugMutex: mutex with deadlock detection and ownership tracking
// - SharedMutex: reader-writer mutex (multiple readers, single writer)
// - PriorityMutex: mutex with priority-based waiting
// - AdaptiveMutex: hybrid spin-then-block mutex
//
// All mutex types are move-only and provide RAII lock guards.

#pragma once

#include <mutex>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace fem::core::concurrency {

// Forward declarations
class TimedMutex;
class RecursiveMutex;
class DebugMutex;
class SharedMutex;
class PriorityMutex;
class AdaptiveMutex;

// Lock result enumeration
enum class LockResult {
    Success,
    Timeout,
    WouldBlock,
    DeadlockDetected,
    Error
};

// -----------------------------------------------------------------------------
// TimedMutex: Basic mutex with timeout support
// -----------------------------------------------------------------------------
class TimedMutex {
private:
    std::timed_mutex mutex_;

public:
    TimedMutex() = default;
    ~TimedMutex() = default;

    // Non-copyable, non-movable (mutexes should not be moved)
    TimedMutex(const TimedMutex&) = delete;
    TimedMutex& operator=(const TimedMutex&) = delete;
    TimedMutex(TimedMutex&&) = delete;
    TimedMutex& operator=(TimedMutex&&) = delete;

    void lock() {
        mutex_.lock();
    }

    bool try_lock() {
        return mutex_.try_lock();
    }

    template<typename Rep, typename Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout) {
        return mutex_.try_lock_for(timeout);
    }

    template<typename Clock, typename Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& deadline) {
        return mutex_.try_lock_until(deadline);
    }

    void unlock() {
        mutex_.unlock();
    }

    // Get native handle for platform-specific operations
    std::timed_mutex::native_handle_type native_handle() {
        return mutex_.native_handle();
    }
};

// -----------------------------------------------------------------------------
// RecursiveMutex: Reentrant mutex with ownership tracking
// -----------------------------------------------------------------------------
class RecursiveMutex {
private:
    std::recursive_timed_mutex mutex_;
    std::atomic<std::thread::id> owner_thread_{};
    std::atomic<size_t> recursion_count_{0};

public:
    RecursiveMutex() = default;
    ~RecursiveMutex() = default;

    RecursiveMutex(const RecursiveMutex&) = delete;
    RecursiveMutex& operator=(const RecursiveMutex&) = delete;
    RecursiveMutex(RecursiveMutex&&) = delete;
    RecursiveMutex& operator=(RecursiveMutex&&) = delete;

    void lock() {
        mutex_.lock();
        owner_thread_ = std::this_thread::get_id();
        ++recursion_count_;
    }

    bool try_lock() {
        if (mutex_.try_lock()) {
            owner_thread_ = std::this_thread::get_id();
            ++recursion_count_;
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout) {
        if (mutex_.try_lock_for(timeout)) {
            owner_thread_ = std::this_thread::get_id();
            ++recursion_count_;
            return true;
        }
        return false;
    }

    void unlock() {
        assert(owner_thread_ == std::this_thread::get_id());
        if (--recursion_count_ == 0) {
            owner_thread_ = std::thread::id{};
        }
        mutex_.unlock();
    }

    bool is_locked_by_current_thread() const {
        return owner_thread_ == std::this_thread::get_id();
    }

    size_t recursion_level() const {
        return recursion_count_;
    }

    std::thread::id owner() const {
        return owner_thread_;
    }
};

// -----------------------------------------------------------------------------
// DebugMutex: Mutex with deadlock detection and debugging support
// -----------------------------------------------------------------------------
class DebugMutex {
private:
    mutable std::mutex mutex_;
    mutable std::mutex metadata_mutex_;  // Separate mutex for metadata
    std::thread::id owner_thread_{};
    std::chrono::steady_clock::time_point lock_time_;
    std::string lock_location_;
    std::atomic<size_t> wait_count_{0};

    // Global deadlock detection (simplified - would need global registry)
    static thread_local std::vector<const DebugMutex*> held_locks_;

    bool check_deadlock() const {
        // Simple cycle detection: check if we already hold this lock
        return std::find(held_locks_.begin(), held_locks_.end(), this) != held_locks_.end();
    }

public:
    DebugMutex() = default;
    ~DebugMutex() {
        // Ensure mutex is not locked on destruction
        assert(owner_thread_ == std::thread::id{});
    }

    DebugMutex(const DebugMutex&) = delete;
    DebugMutex& operator=(const DebugMutex&) = delete;
    DebugMutex(DebugMutex&&) = delete;
    DebugMutex& operator=(DebugMutex&&) = delete;

    void lock(const std::string& location = "") {
        if (check_deadlock()) {
            throw std::runtime_error("Deadlock detected at: " + location);
        }

        ++wait_count_;
        mutex_.lock();

        {
            std::lock_guard<std::mutex> meta_lock(metadata_mutex_);
            owner_thread_ = std::this_thread::get_id();
            lock_time_ = std::chrono::steady_clock::now();
            lock_location_ = location;
        }
        held_locks_.push_back(this);
    }

    bool try_lock(const std::string& location = "") {
        if (check_deadlock()) {
            return false;
        }

        if (mutex_.try_lock()) {
            {
                std::lock_guard<std::mutex> meta_lock(metadata_mutex_);
                owner_thread_ = std::this_thread::get_id();
                lock_time_ = std::chrono::steady_clock::now();
                lock_location_ = location;
            }
            held_locks_.push_back(this);
            return true;
        }
        return false;
    }

    void unlock() {
        {
            std::lock_guard<std::mutex> meta_lock(metadata_mutex_);
            assert(owner_thread_ == std::this_thread::get_id());
            owner_thread_ = std::thread::id{};
            lock_location_.clear();
        }

        auto it = std::find(held_locks_.begin(), held_locks_.end(), this);
        if (it != held_locks_.end()) {
            held_locks_.erase(it);
        }

        mutex_.unlock();
    }

    // Debugging info
    std::thread::id owner() const {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        return owner_thread_;
    }

    std::chrono::milliseconds held_duration() const {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        if (owner_thread_ != std::thread::id{}) {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(now - lock_time_);
        }
        return std::chrono::milliseconds{0};
    }

    std::string lock_location() const {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        return lock_location_;
    }

    size_t contention_count() const {
        return wait_count_;
    }
};

// Static member definition
thread_local std::vector<const DebugMutex*> DebugMutex::held_locks_;

// -----------------------------------------------------------------------------
// SharedMutex: Reader-writer mutex (multiple readers, single writer)
// -----------------------------------------------------------------------------
class SharedMutex {
private:
    std::shared_timed_mutex mutex_;
    std::atomic<size_t> reader_count_{0};
    std::atomic<bool> writer_active_{false};
    std::thread::id writer_thread_{};

public:
    SharedMutex() = default;
    ~SharedMutex() = default;

    SharedMutex(const SharedMutex&) = delete;
    SharedMutex& operator=(const SharedMutex&) = delete;
    SharedMutex(SharedMutex&&) = delete;
    SharedMutex& operator=(SharedMutex&&) = delete;

    // Exclusive (write) lock
    void lock() {
        mutex_.lock();
        writer_active_ = true;
        writer_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        if (mutex_.try_lock()) {
            writer_active_ = true;
            writer_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout) {
        if (mutex_.try_lock_for(timeout)) {
            writer_active_ = true;
            writer_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    void unlock() {
        assert(writer_thread_ == std::this_thread::get_id());
        writer_active_ = false;
        writer_thread_ = std::thread::id{};
        mutex_.unlock();
    }

    // Shared (read) lock
    void lock_shared() {
        mutex_.lock_shared();
        ++reader_count_;
    }

    bool try_lock_shared() {
        if (mutex_.try_lock_shared()) {
            ++reader_count_;
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_lock_shared_for(const std::chrono::duration<Rep, Period>& timeout) {
        if (mutex_.try_lock_shared_for(timeout)) {
            ++reader_count_;
            return true;
        }
        return false;
    }

    void unlock_shared() {
        --reader_count_;
        mutex_.unlock_shared();
    }

    // Info
    size_t reader_count() const { return reader_count_; }
    bool has_writer() const { return writer_active_; }
    bool is_writer() const {
        return writer_active_ && writer_thread_ == std::this_thread::get_id();
    }
};

// -----------------------------------------------------------------------------
// PriorityMutex: Mutex with priority-based waiting
// -----------------------------------------------------------------------------
class PriorityMutex {
private:
    struct Waiter {
        std::thread::id thread_id;
        int priority;
        std::condition_variable cv;
        bool ready = false;

        Waiter(std::thread::id tid, int prio)
            : thread_id(tid), priority(prio) {}
    };

    mutable std::mutex mutex_;
    std::thread::id owner_thread_{};
    struct WaiterCompare {
        bool operator()(const std::shared_ptr<Waiter>& a,
                       const std::shared_ptr<Waiter>& b) const {
            return a->priority < b->priority;
        }
    };

    std::priority_queue<std::shared_ptr<Waiter>,
                        std::vector<std::shared_ptr<Waiter>>,
                        WaiterCompare> waiters_;

public:
    PriorityMutex() = default;
    ~PriorityMutex() = default;

    PriorityMutex(const PriorityMutex&) = delete;
    PriorityMutex& operator=(const PriorityMutex&) = delete;
    PriorityMutex(PriorityMutex&&) = delete;
    PriorityMutex& operator=(PriorityMutex&&) = delete;

    void lock(int priority = 0) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (owner_thread_ == std::thread::id{}) {
            owner_thread_ = std::this_thread::get_id();
            return;
        }

        auto waiter = std::make_shared<Waiter>(std::this_thread::get_id(), priority);
        waiters_.push(waiter);

        waiter->cv.wait(lock, [&] { return waiter->ready; });
        owner_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (owner_thread_ == std::thread::id{}) {
            owner_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(owner_thread_ == std::this_thread::get_id());

        owner_thread_ = std::thread::id{};

        if (!waiters_.empty()) {
            auto next = waiters_.top();
            waiters_.pop();
            next->ready = true;
            lock.unlock();
            next->cv.notify_one();
        }
    }

    size_t waiting_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return waiters_.size();
    }
};

// -----------------------------------------------------------------------------
// AdaptiveMutex: Hybrid spin-then-block mutex for short critical sections
// -----------------------------------------------------------------------------
class AdaptiveMutex {
private:
    std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
    std::mutex fallback_mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> spin_count_{1000};
    std::atomic<size_t> contention_count_{0};
    std::thread::id owner_thread_{};

    static constexpr size_t MIN_SPIN = 100;
    static constexpr size_t MAX_SPIN = 10000;
    static constexpr size_t ADAPTATION_RATE = 10;

public:
    AdaptiveMutex() = default;
    ~AdaptiveMutex() = default;

    AdaptiveMutex(const AdaptiveMutex&) = delete;
    AdaptiveMutex& operator=(const AdaptiveMutex&) = delete;
    AdaptiveMutex(AdaptiveMutex&&) = delete;
    AdaptiveMutex& operator=(AdaptiveMutex&&) = delete;

    void lock() {
        // First try spinning
        size_t spin_limit = spin_count_.load(std::memory_order_relaxed);
        bool had_contention = false;
        for (size_t i = 0; i < spin_limit; ++i) {
            if (!locked_.test_and_set(std::memory_order_acquire)) {
                owner_thread_ = std::this_thread::get_id();

                // Track contention if we had to spin at all
                if (i > 0 && !had_contention) {
                    ++contention_count_;
                }

                // Successful spin - increase spin count for next time
                size_t new_spin = std::min(MAX_SPIN, spin_limit + ADAPTATION_RATE);
                spin_count_.store(new_spin, std::memory_order_relaxed);
                return;
            }

            // We're spinning, so we have contention
            if (!had_contention) {
                ++contention_count_;
                had_contention = true;
            }

            // Pause to reduce contention
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #elif defined(__aarch64__)
                __asm__ __volatile__("yield");
            #else
                std::this_thread::yield();
            #endif
        }

        // Spinning failed - fall back to blocking
        // (contention already counted above)

        // Reduce spin count for next time
        size_t new_spin = std::max(MIN_SPIN, spin_limit - ADAPTATION_RATE);
        spin_count_.store(new_spin, std::memory_order_relaxed);

        // Block until lock is available
        std::unique_lock<std::mutex> lock(fallback_mutex_);
        cv_.wait(lock, [this] {
            return !locked_.test_and_set(std::memory_order_acquire);
        });
        owner_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        if (!locked_.test_and_set(std::memory_order_acquire)) {
            owner_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    void unlock() {
        assert(owner_thread_ == std::this_thread::get_id());
        owner_thread_ = std::thread::id{};
        locked_.clear(std::memory_order_release);
        cv_.notify_one();
    }

    // Tuning interface
    void set_spin_count(size_t count) {
        spin_count_ = std::clamp(count, MIN_SPIN, MAX_SPIN);
    }

    size_t get_spin_count() const {
        return spin_count_;
    }

    size_t contention_count() const {
        return contention_count_;
    }

    void reset_statistics() {
        contention_count_ = 0;
        spin_count_ = 1000;  // Reset to default
    }
};

// -----------------------------------------------------------------------------
// RAII Lock Guards
// -----------------------------------------------------------------------------

// Scoped lock for any mutex type
template<typename Mutex>
class ScopedLock {
private:
    Mutex& mutex_;
    bool owns_lock_;

public:
    explicit ScopedLock(Mutex& m) : mutex_(m), owns_lock_(true) {
        mutex_.lock();
    }

    ScopedLock(Mutex& m, std::defer_lock_t) : mutex_(m), owns_lock_(false) {}

    ScopedLock(Mutex& m, std::try_to_lock_t) : mutex_(m), owns_lock_(mutex_.try_lock()) {}

    ScopedLock(Mutex& m, std::adopt_lock_t) : mutex_(m), owns_lock_(true) {}

    ~ScopedLock() {
        if (owns_lock_) {
            mutex_.unlock();
        }
    }

    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;

    ScopedLock(ScopedLock&& other) noexcept
        : mutex_(other.mutex_), owns_lock_(other.owns_lock_) {
        other.owns_lock_ = false;
    }

    ScopedLock& operator=(ScopedLock&& other) noexcept = delete;

    void lock() {
        if (!owns_lock_) {
            mutex_.lock();
            owns_lock_ = true;
        }
    }

    bool try_lock() {
        if (!owns_lock_) {
            owns_lock_ = mutex_.try_lock();
        }
        return owns_lock_;
    }

    void unlock() {
        if (owns_lock_) {
            mutex_.unlock();
            owns_lock_ = false;
        }
    }

    bool owns_lock() const { return owns_lock_; }
    explicit operator bool() const { return owns_lock_; }

    Mutex* mutex() const noexcept { return &mutex_; }
    Mutex* release() noexcept {
        owns_lock_ = false;
        return &mutex_;
    }
};

// Shared lock for SharedMutex
template<typename SharedMutex>
class SharedLock {
private:
    SharedMutex& mutex_;
    bool owns_lock_;

public:
    explicit SharedLock(SharedMutex& m) : mutex_(m), owns_lock_(true) {
        mutex_.lock_shared();
    }

    SharedLock(SharedMutex& m, std::defer_lock_t) : mutex_(m), owns_lock_(false) {}

    SharedLock(SharedMutex& m, std::try_to_lock_t)
        : mutex_(m), owns_lock_(mutex_.try_lock_shared()) {}

    SharedLock(SharedMutex& m, std::adopt_lock_t) : mutex_(m), owns_lock_(true) {}

    ~SharedLock() {
        if (owns_lock_) {
            mutex_.unlock_shared();
        }
    }

    SharedLock(const SharedLock&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;

    SharedLock(SharedLock&& other) noexcept
        : mutex_(other.mutex_), owns_lock_(other.owns_lock_) {
        other.owns_lock_ = false;
    }

    void lock() {
        if (!owns_lock_) {
            mutex_.lock_shared();
            owns_lock_ = true;
        }
    }

    bool try_lock() {
        if (!owns_lock_) {
            owns_lock_ = mutex_.try_lock_shared();
        }
        return owns_lock_;
    }

    void unlock() {
        if (owns_lock_) {
            mutex_.unlock_shared();
            owns_lock_ = false;
        }
    }

    bool owns_lock() const { return owns_lock_; }
    explicit operator bool() const { return owns_lock_; }
};

// Convenience type aliases
using TimedLock = ScopedLock<TimedMutex>;
using RecursiveLock = ScopedLock<RecursiveMutex>;
using DebugLock = ScopedLock<DebugMutex>;
using WriteLock = ScopedLock<SharedMutex>;
using ReadLock = SharedLock<SharedMutex>;
using PriorityLock = ScopedLock<PriorityMutex>;
using AdaptiveLock = ScopedLock<AdaptiveMutex>;

} // namespace fem::core::concurrency
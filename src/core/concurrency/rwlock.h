// Specialized reader-writer lock implementations
//
// Provides various reader-writer lock implementations optimized for different
// access patterns and performance characteristics:
//
// - ReaderWriterLock: Basic fair reader-writer lock
// - WriterPreferredRWLock: Prioritizes writers to prevent starvation
// - ReaderPreferredRWLock: Prioritizes readers for read-heavy workloads
// - UpgradableRWLock: Supports upgrading read locks to write locks
// - SequenceLock: Optimistic lock for mostly-read scenarios
// - StampedLock: High-performance lock with optimistic reading
// - HierarchicalRWLock: Prevents deadlocks through lock ordering
// - FairRWLock: Ensures strict FIFO ordering between readers and writers

#pragma once

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <queue>
#include <unordered_map>
#include <optional>
#include <cassert>
#include <stdexcept>
#include <limits>

namespace fem::core::concurrency {

// -----------------------------------------------------------------------------
// ReaderWriterLock: Basic fair reader-writer lock
// -----------------------------------------------------------------------------
class ReaderWriterLock {
private:
    mutable std::mutex mutex_;
    std::condition_variable read_cv_;
    std::condition_variable write_cv_;

    int active_readers_ = 0;
    int waiting_readers_ = 0;
    int active_writers_ = 0;  // 0 or 1
    int waiting_writers_ = 0;

    std::thread::id writer_thread_;

public:
    ReaderWriterLock() = default;
    ~ReaderWriterLock() = default;

    ReaderWriterLock(const ReaderWriterLock&) = delete;
    ReaderWriterLock& operator=(const ReaderWriterLock&) = delete;
    ReaderWriterLock(ReaderWriterLock&&) = delete;
    ReaderWriterLock& operator=(ReaderWriterLock&&) = delete;

    // Reader operations
    void lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_readers_;

        read_cv_.wait(lock, [this] {
            return active_writers_ == 0 && waiting_writers_ == 0;
        });

        --waiting_readers_;
        ++active_readers_;
    }

    bool try_lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (active_writers_ == 0 && waiting_writers_ == 0) {
            ++active_readers_;
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_lock_shared_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_readers_;

        bool acquired = read_cv_.wait_for(lock, timeout, [this] {
            return active_writers_ == 0 && waiting_writers_ == 0;
        });

        --waiting_readers_;
        if (acquired) {
            ++active_readers_;
        }
        return acquired;
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_readers_ > 0);

        --active_readers_;

        if (active_readers_ == 0 && waiting_writers_ > 0) {
            write_cv_.notify_one();
        }
    }

    // Writer operations
    void lock() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_writers_;

        write_cv_.wait(lock, [this] {
            return active_readers_ == 0 && active_writers_ == 0;
        });

        --waiting_writers_;
        active_writers_ = 1;
        writer_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (active_readers_ == 0 && active_writers_ == 0) {
            active_writers_ = 1;
            writer_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    template<typename Rep, typename Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_writers_;

        bool acquired = write_cv_.wait_for(lock, timeout, [this] {
            return active_readers_ == 0 && active_writers_ == 0;
        });

        --waiting_writers_;
        if (acquired) {
            active_writers_ = 1;
            writer_thread_ = std::this_thread::get_id();
        }
        return acquired;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_writers_ == 1);
        assert(writer_thread_ == std::this_thread::get_id());

        active_writers_ = 0;
        writer_thread_ = std::thread::id{};

        // Fair policy: wake writers first if any, otherwise wake all readers
        if (waiting_writers_ > 0) {
            write_cv_.notify_one();
        } else if (waiting_readers_ > 0) {
            read_cv_.notify_all();
        }
    }

    // Statistics
    int active_readers() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_readers_;
    }

    bool has_writer() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_writers_ > 0;
    }

    int waiting_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return waiting_readers_ + waiting_writers_;
    }
};

// -----------------------------------------------------------------------------
// WriterPreferredRWLock: Prioritizes writers to prevent writer starvation
// -----------------------------------------------------------------------------
class WriterPreferredRWLock {
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    int active_readers_ = 0;
    int waiting_writers_ = 0;
    bool writer_active_ = false;
    std::thread::id writer_thread_;

public:
    WriterPreferredRWLock() = default;
    ~WriterPreferredRWLock() = default;

    WriterPreferredRWLock(const WriterPreferredRWLock&) = delete;
    WriterPreferredRWLock& operator=(const WriterPreferredRWLock&) = delete;

    void lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Readers must wait if there's an active writer or waiting writers
        cv_.wait(lock, [this] {
            return !writer_active_ && waiting_writers_ == 0;
        });

        ++active_readers_;
    }

    bool try_lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!writer_active_ && waiting_writers_ == 0) {
            ++active_readers_;
            return true;
        }
        return false;
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_readers_ > 0);

        --active_readers_;

        if (active_readers_ == 0) {
            cv_.notify_all();  // Wake any waiting writer
        }
    }

    void lock() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_writers_;

        cv_.wait(lock, [this] {
            return !writer_active_ && active_readers_ == 0;
        });

        --waiting_writers_;
        writer_active_ = true;
        writer_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!writer_active_ && active_readers_ == 0) {
            writer_active_ = true;
            writer_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(writer_active_);
        assert(writer_thread_ == std::this_thread::get_id());

        writer_active_ = false;
        writer_thread_ = std::thread::id{};

        cv_.notify_all();  // Wake all waiting threads
    }
};

// -----------------------------------------------------------------------------
// ReaderPreferredRWLock: Prioritizes readers for read-heavy workloads
// -----------------------------------------------------------------------------
class ReaderPreferredRWLock {
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    int active_readers_ = 0;
    bool writer_active_ = false;
    int waiting_writers_ = 0;
    std::thread::id writer_thread_;

public:
    ReaderPreferredRWLock() = default;
    ~ReaderPreferredRWLock() = default;

    ReaderPreferredRWLock(const ReaderPreferredRWLock&) = delete;
    ReaderPreferredRWLock& operator=(const ReaderPreferredRWLock&) = delete;

    void lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Readers only wait if there's an active writer
        cv_.wait(lock, [this] {
            return !writer_active_;
        });

        ++active_readers_;
    }

    bool try_lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!writer_active_) {
            ++active_readers_;
            return true;
        }
        return false;
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_readers_ > 0);

        --active_readers_;

        if (active_readers_ == 0 && waiting_writers_ > 0) {
            cv_.notify_one();  // Wake one waiting writer
        }
    }

    void lock() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_writers_;

        // Writers must wait for all active readers to finish
        cv_.wait(lock, [this] {
            return !writer_active_ && active_readers_ == 0;
        });

        --waiting_writers_;
        writer_active_ = true;
        writer_thread_ = std::this_thread::get_id();
    }

    bool try_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!writer_active_ && active_readers_ == 0) {
            writer_active_ = true;
            writer_thread_ = std::this_thread::get_id();
            return true;
        }
        return false;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(writer_active_);
        assert(writer_thread_ == std::this_thread::get_id());

        writer_active_ = false;
        writer_thread_ = std::thread::id{};

        cv_.notify_all();  // Wake all waiting threads (readers get priority)
    }
};

// -----------------------------------------------------------------------------
// UpgradableRWLock: Supports upgrading from read to write lock
// -----------------------------------------------------------------------------
class UpgradableRWLock {
public:
    enum class LockState {
        Unlocked,
        SharedLocked,
        UpgradeLocked,
        ExclusiveLocked
    };

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    int shared_count_ = 0;
    bool upgrade_lock_held_ = false;
    bool exclusive_lock_held_ = false;

    std::thread::id upgrade_holder_;
    std::thread::id exclusive_holder_;

    // Track what kind of lock each thread holds
    thread_local static LockState thread_lock_state_;

public:
    UpgradableRWLock() = default;
    ~UpgradableRWLock() = default;

    UpgradableRWLock(const UpgradableRWLock&) = delete;
    UpgradableRWLock& operator=(const UpgradableRWLock&) = delete;

    // Shared lock operations
    void lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] {
            return !exclusive_lock_held_;
        });

        ++shared_count_;
        thread_lock_state_ = LockState::SharedLocked;
    }

    bool try_lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!exclusive_lock_held_) {
            ++shared_count_;
            thread_lock_state_ = LockState::SharedLocked;
            return true;
        }
        return false;
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(shared_count_ > 0);
        assert(thread_lock_state_ == LockState::SharedLocked);

        --shared_count_;
        thread_lock_state_ = LockState::Unlocked;

        if (shared_count_ == 0) {
            cv_.notify_all();
        }
    }

    // Upgrade lock operations (can be upgraded to exclusive)
    void lock_upgrade() {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] {
            return !exclusive_lock_held_ && !upgrade_lock_held_;
        });

        upgrade_lock_held_ = true;
        upgrade_holder_ = std::this_thread::get_id();
        ++shared_count_;  // Upgrade lock counts as a shared lock
        thread_lock_state_ = LockState::UpgradeLocked;
    }

    bool try_lock_upgrade() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!exclusive_lock_held_ && !upgrade_lock_held_) {
            upgrade_lock_held_ = true;
            upgrade_holder_ = std::this_thread::get_id();
            ++shared_count_;
            thread_lock_state_ = LockState::UpgradeLocked;
            return true;
        }
        return false;
    }

    void unlock_upgrade() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(upgrade_lock_held_);
        assert(upgrade_holder_ == std::this_thread::get_id());
        assert(thread_lock_state_ == LockState::UpgradeLocked);

        upgrade_lock_held_ = false;
        upgrade_holder_ = std::thread::id{};
        --shared_count_;
        thread_lock_state_ = LockState::Unlocked;

        cv_.notify_all();
    }

    // Upgrade from upgrade lock to exclusive lock
    void upgrade_to_exclusive() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(upgrade_lock_held_);
        assert(upgrade_holder_ == std::this_thread::get_id());
        assert(thread_lock_state_ == LockState::UpgradeLocked);

        // Wait for all other shared locks to be released
        cv_.wait(lock, [this] {
            return shared_count_ == 1;  // Only our upgrade lock remains
        });

        upgrade_lock_held_ = false;
        upgrade_holder_ = std::thread::id{};
        --shared_count_;

        exclusive_lock_held_ = true;
        exclusive_holder_ = std::this_thread::get_id();
        thread_lock_state_ = LockState::ExclusiveLocked;
    }

    bool try_upgrade_to_exclusive() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(upgrade_lock_held_);
        assert(upgrade_holder_ == std::this_thread::get_id());
        assert(thread_lock_state_ == LockState::UpgradeLocked);

        if (shared_count_ == 1) {  // Only our upgrade lock
            upgrade_lock_held_ = false;
            upgrade_holder_ = std::thread::id{};
            --shared_count_;

            exclusive_lock_held_ = true;
            exclusive_holder_ = std::this_thread::get_id();
            thread_lock_state_ = LockState::ExclusiveLocked;
            return true;
        }
        return false;
    }

    // Downgrade from exclusive to shared lock
    void downgrade_to_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(exclusive_lock_held_);
        assert(exclusive_holder_ == std::this_thread::get_id());
        assert(thread_lock_state_ == LockState::ExclusiveLocked);

        exclusive_lock_held_ = false;
        exclusive_holder_ = std::thread::id{};
        ++shared_count_;
        thread_lock_state_ = LockState::SharedLocked;

        cv_.notify_all();  // Wake readers waiting for exclusive lock to release
    }

    // Exclusive lock operations
    void lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] {
            return !exclusive_lock_held_ && !upgrade_lock_held_ && shared_count_ == 0;
        });

        exclusive_lock_held_ = true;
        exclusive_holder_ = std::this_thread::get_id();
        thread_lock_state_ = LockState::ExclusiveLocked;
    }

    bool try_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!exclusive_lock_held_ && !upgrade_lock_held_ && shared_count_ == 0) {
            exclusive_lock_held_ = true;
            exclusive_holder_ = std::this_thread::get_id();
            thread_lock_state_ = LockState::ExclusiveLocked;
            return true;
        }
        return false;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(exclusive_lock_held_);
        assert(exclusive_holder_ == std::this_thread::get_id());
        assert(thread_lock_state_ == LockState::ExclusiveLocked);

        exclusive_lock_held_ = false;
        exclusive_holder_ = std::thread::id{};
        thread_lock_state_ = LockState::Unlocked;

        cv_.notify_all();
    }

    // Query current thread's lock state
    LockState current_thread_state() const {
        return thread_lock_state_;
    }
};

// Static member definition
thread_local UpgradableRWLock::LockState UpgradableRWLock::thread_lock_state_ =
    UpgradableRWLock::LockState::Unlocked;

// -----------------------------------------------------------------------------
// SequenceLock: Optimistic lock for mostly-read scenarios
// -----------------------------------------------------------------------------
template<typename T>
class SequenceLock {
private:
    alignas(64) std::atomic<uint64_t> sequence_{0};
    alignas(64) T data_{};
    mutable std::mutex writer_mutex_;

    static constexpr uint64_t WRITER_BIT = 1ULL;

public:
    SequenceLock() = default;
    explicit SequenceLock(const T& initial) : data_(initial) {}
    ~SequenceLock() = default;

    SequenceLock(const SequenceLock&) = delete;
    SequenceLock& operator=(const SequenceLock&) = delete;

    // Optimistic read - may retry if concurrent write detected
    T read() const {
        while (true) {
            // Read sequence number before reading data
            uint64_t seq1 = sequence_.load(std::memory_order_acquire);

            // If writer is active, wait and retry
            if (seq1 & WRITER_BIT) {
                std::this_thread::yield();
                continue;
            }

            // Read the data
            T copy = data_;

            // Memory fence to ensure data read completes before sequence check
            std::atomic_thread_fence(std::memory_order_acquire);

            // Check sequence number hasn't changed
            uint64_t seq2 = sequence_.load(std::memory_order_acquire);

            if (seq1 == seq2) {
                return copy;  // Success - no concurrent write
            }

            // Sequence changed, retry
            std::this_thread::yield();
        }
    }

    // Try to read without retrying
    std::optional<T> try_read() const {
        uint64_t seq1 = sequence_.load(std::memory_order_acquire);

        if (seq1 & WRITER_BIT) {
            return std::nullopt;  // Writer active
        }

        T copy = data_;

        std::atomic_thread_fence(std::memory_order_acquire);

        uint64_t seq2 = sequence_.load(std::memory_order_acquire);

        if (seq1 == seq2) {
            return copy;
        }

        return std::nullopt;  // Concurrent modification detected
    }

    // Write operation - exclusive access
    void write(const T& value) {
        std::lock_guard<std::mutex> lock(writer_mutex_);

        // Increment sequence and set writer bit
        uint64_t seq = sequence_.load(std::memory_order_relaxed);
        sequence_.store(seq + WRITER_BIT, std::memory_order_release);

        // Update data
        data_ = value;

        // Increment sequence again and clear writer bit
        sequence_.store(seq + 2, std::memory_order_release);
    }

    // Apply a function to modify the data
    template<typename Func>
    void update(Func&& func) {
        std::lock_guard<std::mutex> lock(writer_mutex_);

        uint64_t seq = sequence_.load(std::memory_order_relaxed);
        sequence_.store(seq + WRITER_BIT, std::memory_order_release);

        func(data_);

        sequence_.store(seq + 2, std::memory_order_release);
    }

    // Get current sequence number (for debugging)
    uint64_t sequence() const {
        return sequence_.load(std::memory_order_acquire) & ~WRITER_BIT;
    }
};

// -----------------------------------------------------------------------------
// StampedLock: High-performance lock with optimistic reading
// -----------------------------------------------------------------------------
class StampedLock {
private:
    static constexpr uint64_t WRITER_BIT = 1ULL;
    static constexpr uint64_t READER_OVERFLOW = 1ULL << 32;
    static constexpr uint64_t READER_MASK = READER_OVERFLOW - 1;
    static constexpr uint64_t STAMP_INC = 1ULL << 33;

    alignas(64) std::atomic<uint64_t> state_{STAMP_INC};  // Start with non-zero stamp
    std::mutex mutex_;
    std::condition_variable read_cv_;
    std::condition_variable write_cv_;

    bool is_writer_active(uint64_t s) const {
        return (s & WRITER_BIT) != 0;
    }

    uint32_t reader_count(uint64_t s) const {
        return static_cast<uint32_t>((s >> 1) & READER_MASK);
    }

    uint64_t get_stamp(uint64_t s) const {
        // Extract the stamp from the high bits
        return s >> 33;
    }

public:
    using stamp_t = uint64_t;

    StampedLock() = default;
    ~StampedLock() = default;

    StampedLock(const StampedLock&) = delete;
    StampedLock& operator=(const StampedLock&) = delete;

    // Optimistic read - returns a stamp for validation
    stamp_t try_optimistic_read() {
        uint64_t s = state_.load(std::memory_order_acquire);
        return is_writer_active(s) ? 0 : get_stamp(s);
    }

    // Validate an optimistic read stamp
    bool validate(stamp_t stamp) {
        if (stamp == 0) return false;

        // Memory fence to ensure any reads happen before validation
        std::atomic_thread_fence(std::memory_order_acquire);

        uint64_t s = state_.load(std::memory_order_acquire);
        return get_stamp(s) == stamp && !is_writer_active(s);
    }

    // Read lock
    stamp_t read_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        while (true) {
            uint64_t s = state_.load(std::memory_order_acquire);

            if (!is_writer_active(s) && reader_count(s) < READER_MASK) {
                // Add reader (no stamp change for readers)
                uint64_t new_state = s + 2;  // Increment reader count
                if (state_.compare_exchange_weak(s, new_state)) {
                    return get_stamp(new_state);
                }
            } else {
                read_cv_.wait(lock);
            }
        }
    }

    // Try read lock
    stamp_t try_read_lock() {
        uint64_t s = state_.load(std::memory_order_acquire);

        if (!is_writer_active(s) && reader_count(s) < READER_MASK) {
            uint64_t new_state = s + 2;
            if (state_.compare_exchange_strong(s, new_state)) {
                return get_stamp(new_state);
            }
        }

        return 0;  // Failed
    }

    // Unlock read
    void unlock_read([[maybe_unused]] stamp_t stamp) {
        std::unique_lock<std::mutex> lock(mutex_);

        uint64_t s = state_.load(std::memory_order_acquire);
        assert(reader_count(s) > 0);

        uint64_t new_state = s - 2;  // Decrement reader count
        state_.store(new_state, std::memory_order_release);

        if (reader_count(new_state) == 0) {
            write_cv_.notify_one();
        }
    }

    // Write lock
    stamp_t write_lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        while (true) {
            uint64_t s = state_.load(std::memory_order_acquire);

            if (!is_writer_active(s) && reader_count(s) == 0) {
                // Set writer bit and increment stamp
                uint64_t new_state = (s | WRITER_BIT) + STAMP_INC;
                if (state_.compare_exchange_weak(s, new_state)) {
                    return new_state;
                }
            } else {
                write_cv_.wait(lock);
            }
        }
    }

    // Try write lock
    stamp_t try_write_lock() {
        uint64_t s = state_.load(std::memory_order_acquire);

        if (!is_writer_active(s) && reader_count(s) == 0) {
            uint64_t new_state = (s | WRITER_BIT) + STAMP_INC;
            if (state_.compare_exchange_strong(s, new_state)) {
                return new_state;
            }
        }

        return 0;  // Failed
    }

    // Unlock write
    void unlock_write([[maybe_unused]] stamp_t stamp) {
        std::unique_lock<std::mutex> lock(mutex_);

        uint64_t s = state_.load(std::memory_order_acquire);
        assert(is_writer_active(s));

        // Clear writer bit
        uint64_t new_state = s & ~WRITER_BIT;
        state_.store(new_state, std::memory_order_release);

        // Wake all waiting threads
        read_cv_.notify_all();
        write_cv_.notify_one();
    }

    // Try to convert optimistic read to read lock
    stamp_t try_convert_to_read_lock(stamp_t stamp) {
        if (stamp == 0) return 0;

        uint64_t s = state_.load(std::memory_order_acquire);

        // Check if stamp is still valid
        if (get_stamp(s) == stamp && !is_writer_active(s) && reader_count(s) < READER_MASK) {
            uint64_t new_state = s + 2;
            if (state_.compare_exchange_strong(s, new_state)) {
                return get_stamp(new_state);
            }
        }

        return 0;
    }

    // Try to convert read lock to write lock
    stamp_t try_convert_to_write_lock(stamp_t stamp) {
        if (stamp == 0) return 0;

        std::unique_lock<std::mutex> lock(mutex_);

        uint64_t s = state_.load(std::memory_order_acquire);

        // Can only upgrade if we're the only reader
        if (reader_count(s) == 1 && !is_writer_active(s)) {
            // Convert: remove reader, add writer, increment stamp
            uint64_t new_state = ((s - 2) | WRITER_BIT) + STAMP_INC;
            if (state_.compare_exchange_strong(s, new_state)) {
                return new_state;
            }
        }

        return 0;
    }
};

// -----------------------------------------------------------------------------
// HierarchicalRWLock: Prevents deadlocks through lock ordering
// -----------------------------------------------------------------------------
class HierarchicalRWLock {
private:
    const unsigned hierarchy_value_;
    unsigned previous_hierarchy_value_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    int readers_ = 0;
    bool writer_ = false;
    std::thread::id writer_thread_;

    static thread_local unsigned current_hierarchy_value_;

    void check_hierarchy_violation() {
        if (current_hierarchy_value_ <= hierarchy_value_) {
            throw std::logic_error("Hierarchy violation! Attempted to acquire lock with hierarchy " +
                                  std::to_string(hierarchy_value_) +
                                  " while holding hierarchy " +
                                  std::to_string(current_hierarchy_value_));
        }
    }

public:
    explicit HierarchicalRWLock(unsigned hierarchy_value)
        : hierarchy_value_(hierarchy_value), previous_hierarchy_value_(0) {}

    ~HierarchicalRWLock() = default;

    HierarchicalRWLock(const HierarchicalRWLock&) = delete;
    HierarchicalRWLock& operator=(const HierarchicalRWLock&) = delete;

    void lock_shared() {
        check_hierarchy_violation();

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !writer_; });

        ++readers_;

        previous_hierarchy_value_ = current_hierarchy_value_;
        current_hierarchy_value_ = hierarchy_value_;
    }

    void unlock_shared() {
        if (current_hierarchy_value_ != hierarchy_value_) {
            throw std::logic_error("Unlock hierarchy mismatch!");
        }

        current_hierarchy_value_ = previous_hierarchy_value_;

        std::unique_lock<std::mutex> lock(mutex_);
        --readers_;

        if (readers_ == 0) {
            cv_.notify_all();
        }
    }

    void lock() {
        check_hierarchy_violation();

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !writer_ && readers_ == 0; });

        writer_ = true;
        writer_thread_ = std::this_thread::get_id();

        previous_hierarchy_value_ = current_hierarchy_value_;
        current_hierarchy_value_ = hierarchy_value_;
    }

    void unlock() {
        if (current_hierarchy_value_ != hierarchy_value_) {
            throw std::logic_error("Unlock hierarchy mismatch!");
        }

        current_hierarchy_value_ = previous_hierarchy_value_;

        std::unique_lock<std::mutex> lock(mutex_);
        writer_ = false;
        writer_thread_ = std::thread::id{};

        cv_.notify_all();
    }

    unsigned hierarchy_value() const { return hierarchy_value_; }
};

// Static member definition
thread_local unsigned HierarchicalRWLock::current_hierarchy_value_ =
    std::numeric_limits<unsigned>::max();

// -----------------------------------------------------------------------------
// FairRWLock: Ensures strict FIFO ordering between readers and writers
// -----------------------------------------------------------------------------
class FairRWLock {
private:
    struct Request {
        enum Type { READ, WRITE };
        Type type;
        std::condition_variable cv;
        bool ready = false;
    };

    mutable std::mutex mutex_;
    std::queue<std::shared_ptr<Request>> queue_;

    int active_readers_ = 0;
    bool active_writer_ = false;
    std::thread::id writer_thread_;

    void process_queue() {
        while (!queue_.empty()) {
            auto& front = queue_.front();

            if (front->type == Request::READ) {
                // Can start if no active writer
                if (!active_writer_) {
                    ++active_readers_;
                    front->ready = true;
                    front->cv.notify_one();
                    queue_.pop();
                    // Continue processing to allow multiple readers
                } else {
                    break;  // Wait for writer to finish
                }
            } else {  // WRITE
                // Can start if no active readers or writer
                if (active_readers_ == 0 && !active_writer_) {
                    active_writer_ = true;
                    front->ready = true;
                    front->cv.notify_one();
                    queue_.pop();
                    break;  // Only one writer at a time
                } else {
                    break;  // Wait for current operations to finish
                }
            }
        }
    }

public:
    FairRWLock() = default;
    ~FairRWLock() = default;

    FairRWLock(const FairRWLock&) = delete;
    FairRWLock& operator=(const FairRWLock&) = delete;

    void lock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);

        auto request = std::make_shared<Request>();
        request->type = Request::READ;

        // Always queue if there are other requests waiting (FIFO order)
        // Only acquire immediately if queue is empty AND no writer
        if (queue_.empty() && !active_writer_) {
            ++active_readers_;
        } else {
            // Join the queue to maintain FIFO order
            queue_.push(request);
            request->cv.wait(lock, [request] { return request->ready; });
        }
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_readers_ > 0);

        --active_readers_;

        if (active_readers_ == 0) {
            process_queue();
        }
    }

    void lock() {
        std::unique_lock<std::mutex> lock(mutex_);

        auto request = std::make_shared<Request>();
        request->type = Request::WRITE;

        // If no queue and no active operations, acquire immediately
        if (queue_.empty() && active_readers_ == 0 && !active_writer_) {
            active_writer_ = true;
            writer_thread_ = std::this_thread::get_id();
        } else {
            // Join the queue
            queue_.push(request);
            request->cv.wait(lock, [request] { return request->ready; });
            writer_thread_ = std::this_thread::get_id();
        }
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(active_writer_);
        assert(writer_thread_ == std::this_thread::get_id());

        active_writer_ = false;
        writer_thread_ = std::thread::id{};

        process_queue();
    }

    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// -----------------------------------------------------------------------------
// RAII Lock Guards for RW Locks
// -----------------------------------------------------------------------------

template<typename RWLock>
class ReadLockGuard {
private:
    RWLock& lock_;
    bool owns_lock_;

public:
    explicit ReadLockGuard(RWLock& lock) : lock_(lock), owns_lock_(true) {
        lock_.lock_shared();
    }

    ~ReadLockGuard() {
        if (owns_lock_) {
            lock_.unlock_shared();
        }
    }

    ReadLockGuard(const ReadLockGuard&) = delete;
    ReadLockGuard& operator=(const ReadLockGuard&) = delete;

    ReadLockGuard(ReadLockGuard&& other) noexcept
        : lock_(other.lock_), owns_lock_(other.owns_lock_) {
        other.owns_lock_ = false;
    }

    void unlock() {
        if (owns_lock_) {
            lock_.unlock_shared();
            owns_lock_ = false;
        }
    }

    bool owns_lock() const { return owns_lock_; }
};

template<typename RWLock>
class WriteLockGuard {
private:
    RWLock& lock_;
    bool owns_lock_;

public:
    explicit WriteLockGuard(RWLock& lock) : lock_(lock), owns_lock_(true) {
        lock_.lock();
    }

    ~WriteLockGuard() {
        if (owns_lock_) {
            lock_.unlock();
        }
    }

    WriteLockGuard(const WriteLockGuard&) = delete;
    WriteLockGuard& operator=(const WriteLockGuard&) = delete;

    WriteLockGuard(WriteLockGuard&& other) noexcept
        : lock_(other.lock_), owns_lock_(other.owns_lock_) {
        other.owns_lock_ = false;
    }

    void unlock() {
        if (owns_lock_) {
            lock_.unlock();
            owns_lock_ = false;
        }
    }

    bool owns_lock() const { return owns_lock_; }
};

template<typename RWLock>
class UpgradeLockGuard {
private:
    RWLock& lock_;
    bool owns_lock_;
    bool upgraded_;
    bool downgraded_to_shared_;

public:
    explicit UpgradeLockGuard(RWLock& lock)
        : lock_(lock), owns_lock_(true), upgraded_(false), downgraded_to_shared_(false) {
        lock_.lock_upgrade();
    }

    ~UpgradeLockGuard() {
        if (owns_lock_) {
            if (upgraded_) {
                lock_.unlock();
            } else if (downgraded_to_shared_) {
                lock_.unlock_shared();
            } else {
                lock_.unlock_upgrade();
            }
        }
    }

    UpgradeLockGuard(const UpgradeLockGuard&) = delete;
    UpgradeLockGuard& operator=(const UpgradeLockGuard&) = delete;

    void upgrade() {
        if (owns_lock_ && !upgraded_) {
            lock_.upgrade_to_exclusive();
            upgraded_ = true;
            downgraded_to_shared_ = false;
        }
    }

    bool try_upgrade() {
        if (owns_lock_ && !upgraded_) {
            if (lock_.try_upgrade_to_exclusive()) {
                upgraded_ = true;
                downgraded_to_shared_ = false;
                return true;
            }
        }
        return false;
    }

    void downgrade() {
        if (owns_lock_ && upgraded_) {
            lock_.downgrade_to_shared();
            upgraded_ = false;
            downgraded_to_shared_ = true;
        }
    }

    bool owns_lock() const { return owns_lock_; }
    bool is_upgraded() const { return upgraded_; }
};

// Convenience aliases
template<typename RWLock> using ReadGuard = ReadLockGuard<RWLock>;
template<typename RWLock> using WriteGuard = WriteLockGuard<RWLock>;
template<typename RWLock> using UpgradeGuard = UpgradeLockGuard<RWLock>;

} // namespace fem::core::concurrency
// One-time synchronization primitives
//
// Provides various latch and countdown implementations for coordinating
// threads at synchronization points:
//
// - Latch: Single-use countdown latch (C++20 std::latch equivalent)
// - CountdownEvent: Resettable countdown with dynamic increment/decrement
// - CompletionLatch: Tracks completion of multiple operations with callbacks
// - BarrierLatch: Combines latch with barrier-like behavior
// - TimedLatch: Latch with timeout support
// - MultiStageLatch: Coordinates multi-stage pipeline synchronization
// - DependentLatch: Latch with dependency tracking
// - GroupLatch: Multiple named latches for group coordination

#pragma once

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <cassert>
#include <limits>

namespace fem::core::concurrency {

// -----------------------------------------------------------------------------
// Latch: Single-use countdown latch (C++20 std::latch equivalent)
// -----------------------------------------------------------------------------
class Latch {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::ptrdiff_t count_;
    const std::ptrdiff_t initial_count_;

public:
    explicit Latch(std::ptrdiff_t count)
        : count_(count), initial_count_(count) {
        if (count < 0) {
            throw std::invalid_argument("Latch count must be non-negative");
        }
    }

    ~Latch() = default;

    // Non-copyable, non-movable
    Latch(const Latch&) = delete;
    Latch& operator=(const Latch&) = delete;
    Latch(Latch&&) = delete;
    Latch& operator=(Latch&&) = delete;

    // Decrement counter and block until it reaches zero
    void count_down_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (count_ <= 0) {
            return;  // Already at zero
        }

        --count_;

        if (count_ == 0) {
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return count_ == 0; });
        }
    }

    // Decrement counter without blocking
    void count_down(std::ptrdiff_t n = 1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (n <= 0) {
            return;
        }

        if (count_ > 0) {
            count_ = std::max(std::ptrdiff_t{0}, count_ - n);

            if (count_ == 0) {
                cv_.notify_all();
            }
        }
    }

    // Test if counter has reached zero
    bool try_wait() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ == 0;
    }

    // Block until counter reaches zero
    void wait() const {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ == 0; });
    }

    // Arrive at the latch (decrement) and continue
    void arrive() {
        count_down();
    }

    // Get current count
    std::ptrdiff_t count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    // Get initial count
    std::ptrdiff_t initial_count() const noexcept {
        return initial_count_;
    }

    // Check if latch is ready (count == 0)
    bool is_ready() const noexcept {
        return try_wait();
    }
};

// -----------------------------------------------------------------------------
// CountdownEvent: Resettable countdown with dynamic increment/decrement
// -----------------------------------------------------------------------------
class CountdownEvent {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::ptrdiff_t count_;
    std::ptrdiff_t initial_count_;
    uint64_t generation_;  // To handle spurious wakeups after reset

public:
    explicit CountdownEvent(std::ptrdiff_t initial_count = 1)
        : count_(initial_count), initial_count_(initial_count), generation_(0) {
        if (initial_count < 0) {
            throw std::invalid_argument("CountdownEvent count must be non-negative");
        }
    }

    ~CountdownEvent() = default;

    CountdownEvent(const CountdownEvent&) = delete;
    CountdownEvent& operator=(const CountdownEvent&) = delete;

    // Add to the count
    void add_count(std::ptrdiff_t signalCount = 1) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (count_ == 0) {
            throw std::logic_error("Cannot add count after countdown reached zero");
        }

        count_ += signalCount;
    }

    // Decrement the count
    bool signal(std::ptrdiff_t signalCount = 1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (signalCount <= 0) {
            return count_ == 0;
        }

        count_ = std::max(std::ptrdiff_t{0}, count_ - signalCount);

        if (count_ == 0) {
            cv_.notify_all();
            return true;
        }
        return false;
    }

    // Wait for count to reach zero
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t gen = generation_;
        cv_.wait(lock, [this, gen] {
            return count_ == 0 || generation_ != gen;
        });
    }

    // Wait with timeout
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t gen = generation_;
        return cv_.wait_for(lock, timeout, [this, gen] {
            return count_ == 0 || generation_ != gen;
        });
    }

    // Reset to initial count
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = initial_count_;
        ++generation_;
    }

    // Reset to specific count
    void reset(std::ptrdiff_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count < 0) {
            throw std::invalid_argument("Count must be non-negative");
        }
        initial_count_ = count;
        count_ = count;
        ++generation_;
    }

    // Get current count
    std::ptrdiff_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    // Check if signaled (count == 0)
    bool is_set() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ == 0;
    }
};

// -----------------------------------------------------------------------------
// CompletionLatch: Tracks completion with callbacks
// -----------------------------------------------------------------------------
class CompletionLatch {
public:
    using CompletionCallback = std::function<void()>;

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::ptrdiff_t pending_count_;
    std::ptrdiff_t completed_count_;
    std::vector<CompletionCallback> callbacks_;
    bool is_completed_;

public:
    explicit CompletionLatch(std::ptrdiff_t expected_completions)
        : pending_count_(expected_completions),
          completed_count_(0),
          is_completed_(false) {
        if (expected_completions <= 0) {
            is_completed_ = true;
        }
    }

    ~CompletionLatch() = default;

    CompletionLatch(const CompletionLatch&) = delete;
    CompletionLatch& operator=(const CompletionLatch&) = delete;

    // Mark one operation as complete
    void complete_one() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (is_completed_) {
            return;
        }

        ++completed_count_;
        --pending_count_;

        if (pending_count_ <= 0) {
            is_completed_ = true;

            // Execute callbacks
            auto callbacks = std::move(callbacks_);
            lock.unlock();

            cv_.notify_all();

            for (const auto& callback : callbacks) {
                if (callback) callback();
            }
        }
    }

    // Mark multiple operations as complete
    void complete(std::ptrdiff_t count) {
        if (count <= 0) return;

        for (std::ptrdiff_t i = 0; i < count; ++i) {
            complete_one();
        }
    }

    // Register a completion callback
    void on_completion(CompletionCallback callback) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (is_completed_) {
            lock.unlock();
            if (callback) callback();
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }

    // Wait for completion
    void wait() const {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return is_completed_; });
    }

    // Wait with timeout
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return is_completed_; });
    }

    // Check if completed
    bool is_completed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_completed_;
    }

    // Get completion progress
    std::pair<std::ptrdiff_t, std::ptrdiff_t> progress() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {completed_count_, completed_count_ + pending_count_};
    }

    // Get completion percentage (0.0 to 1.0)
    double completion_ratio() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto total = completed_count_ + pending_count_;
        return total > 0 ? static_cast<double>(completed_count_) / static_cast<double>(total) : 1.0;
    }
};

// -----------------------------------------------------------------------------
// BarrierLatch: Combines latch with barrier-like behavior
// -----------------------------------------------------------------------------
class BarrierLatch {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    const std::ptrdiff_t threshold_;
    std::ptrdiff_t count_;
    std::ptrdiff_t generation_;
    std::function<void()> completion_callback_;

public:
    explicit BarrierLatch(std::ptrdiff_t num_threads,
                          std::function<void()> on_completion = nullptr)
        : threshold_(num_threads),
          count_(num_threads),
          generation_(0),
          completion_callback_(std::move(on_completion)) {
        if (num_threads <= 0) {
            throw std::invalid_argument("BarrierLatch requires positive thread count");
        }
    }

    ~BarrierLatch() = default;

    BarrierLatch(const BarrierLatch&) = delete;
    BarrierLatch& operator=(const BarrierLatch&) = delete;

    // Arrive and wait for all threads
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);

        auto gen = generation_;

        if (--count_ == 0) {
            // Last thread to arrive - reset for next use
            count_ = threshold_;
            generation_++;

            // Run completion callback if provided
            if (completion_callback_) {
                auto callback = completion_callback_;
                lock.unlock();
                callback();
                lock.lock();
            }

            cv_.notify_all();
        } else {
            // Wait for other threads
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

    // Arrive without waiting
    void arrive_and_drop() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;

            if (completion_callback_) {
                completion_callback_();
            }

            cv_.notify_all();
        }
    }

    // Get the current generation (phase)
    std::ptrdiff_t generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return generation_;
    }

    // Reset the barrier
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = threshold_;
        generation_++;
        cv_.notify_all();
    }
};

// -----------------------------------------------------------------------------
// TimedLatch: Latch with timeout support
// -----------------------------------------------------------------------------
class TimedLatch {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::ptrdiff_t count_;
    bool timed_out_;
    std::chrono::steady_clock::time_point deadline_;

public:
    explicit TimedLatch(std::ptrdiff_t count)
        : count_(count), timed_out_(false) {
        if (count < 0) {
            throw std::invalid_argument("TimedLatch count must be non-negative");
        }
    }

    template<typename Rep, typename Period>
    TimedLatch(std::ptrdiff_t count,
               const std::chrono::duration<Rep, Period>& timeout)
        : count_(count), timed_out_(false) {
        if (count < 0) {
            throw std::invalid_argument("TimedLatch count must be non-negative");
        }
        deadline_ = std::chrono::steady_clock::now() + timeout;
    }

    ~TimedLatch() = default;

    TimedLatch(const TimedLatch&) = delete;
    TimedLatch& operator=(const TimedLatch&) = delete;

    // Count down
    void count_down(std::ptrdiff_t n = 1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (n <= 0 || timed_out_) {
            return;
        }

        count_ = std::max(std::ptrdiff_t{0}, count_ - n);

        if (count_ == 0) {
            cv_.notify_all();
        }
    }

    // Wait for latch or timeout
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (deadline_ != std::chrono::steady_clock::time_point{}) {
            bool success = cv_.wait_until(lock, deadline_, [this] {
                return count_ == 0 || timed_out_;
            });

            if (!success && count_ > 0) {
                timed_out_ = true;
                cv_.notify_all();
                return false;
            }
        } else {
            cv_.wait(lock, [this] { return count_ == 0 || timed_out_; });
        }

        return !timed_out_;
    }

    // Wait with additional timeout
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        bool success = cv_.wait_for(lock, timeout, [this] {
            return count_ == 0 || timed_out_;
        });

        if (!success && count_ > 0) {
            timed_out_ = true;
            cv_.notify_all();
            return false;
        }

        return !timed_out_;
    }

    // Force timeout
    void expire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!timed_out_ && count_ > 0) {
            timed_out_ = true;
            cv_.notify_all();
        }
    }

    // Check if latch is ready or timed out
    std::pair<bool, bool> status() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {count_ == 0, timed_out_};
    }

    std::ptrdiff_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
};

// -----------------------------------------------------------------------------
// MultiStageLatch: Coordinates multi-stage pipeline synchronization
// -----------------------------------------------------------------------------
class MultiStageLatch {
private:
    struct Stage {
        std::ptrdiff_t count;
        std::ptrdiff_t initial_count;
        std::unique_ptr<std::condition_variable> cv;
        bool completed = false;

        Stage(std::ptrdiff_t c)
            : count(c), initial_count(c),
              cv(std::make_unique<std::condition_variable>()),
              completed(c == 0) {}
        Stage(Stage&&) = default;
        Stage& operator=(Stage&&) = default;
        Stage(const Stage&) = delete;
        Stage& operator=(const Stage&) = delete;
    };

    mutable std::mutex mutex_;
    std::vector<Stage> stages_;
    std::size_t current_stage_;
    std::function<void(std::size_t)> stage_callback_;

public:
    explicit MultiStageLatch(const std::vector<std::ptrdiff_t>& stage_counts,
                             std::function<void(std::size_t)> on_stage_complete = nullptr)
        : current_stage_(0), stage_callback_(std::move(on_stage_complete)) {

        if (stage_counts.empty()) {
            throw std::invalid_argument("MultiStageLatch requires at least one stage");
        }

        stages_.reserve(stage_counts.size());
        for (auto count : stage_counts) {
            if (count < 0) {
                throw std::invalid_argument("Stage count must be non-negative");
            }
            stages_.emplace_back(count);
        }
    }

    ~MultiStageLatch() = default;

    MultiStageLatch(const MultiStageLatch&) = delete;
    MultiStageLatch& operator=(const MultiStageLatch&) = delete;

    // Arrive at current stage
    void arrive(std::size_t stage) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (stage >= stages_.size()) {
            throw std::out_of_range("Invalid stage index");
        }

        auto& s = stages_[stage];
        if (s.completed) {
            return;
        }

        if (--s.count <= 0) {
            s.completed = true;
            s.cv->notify_all();

            // Execute stage callback
            if (stage_callback_) {
                auto callback = stage_callback_;
                lock.unlock();
                callback(stage);
                lock.lock();
            }

            // Advance current stage if this was it
            if (stage == current_stage_) {
                while (current_stage_ < stages_.size() &&
                       stages_[current_stage_].completed) {
                    current_stage_++;
                }
            }
        }
    }

    // Wait for specific stage to complete
    void wait(std::size_t stage) const {
        std::unique_lock<std::mutex> lock(mutex_);

        if (stage >= stages_.size()) {
            throw std::out_of_range("Invalid stage index");
        }

        const auto& s = stages_[stage];
        s.cv->wait(lock, [&s] { return s.completed; });
    }

    // Wait for all stages to complete
    void wait_all() const {
        for (std::size_t i = 0; i < stages_.size(); ++i) {
            wait(i);
        }
    }

    // Get current stage
    std::size_t current_stage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_stage_;
    }

    // Check if stage is completed
    bool is_stage_complete(std::size_t stage) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stage >= stages_.size()) {
            return false;
        }
        return stages_[stage].completed;
    }

    // Get total number of stages
    std::size_t num_stages() const {
        return stages_.size();
    }

    // Reset all stages
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& stage : stages_) {
            stage.count = stage.initial_count;
            stage.completed = (stage.initial_count == 0);
        }
        current_stage_ = 0;
    }
};

// -----------------------------------------------------------------------------
// DependentLatch: Latch with dependency tracking
// -----------------------------------------------------------------------------
class DependentLatch : public std::enable_shared_from_this<DependentLatch> {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::ptrdiff_t count_;
    std::vector<std::shared_ptr<DependentLatch>> dependencies_;
    std::vector<std::weak_ptr<DependentLatch>> dependents_;  // Who depends on us
    bool completed_;
    std::function<void()> completion_callback_;

    bool check_dependencies() const {
        for (const auto& dep : dependencies_) {
            if (!dep->is_complete()) {
                return false;
            }
        }
        return true;
    }

    void notify_dependents() {
        std::vector<std::shared_ptr<DependentLatch>> deps_to_notify;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Collect valid dependents
            for (auto it = dependents_.begin(); it != dependents_.end(); ) {
                if (auto dep = it->lock()) {
                    deps_to_notify.push_back(dep);
                    ++it;
                } else {
                    it = dependents_.erase(it);
                }
            }
        }

        // Notify each dependent to recheck
        for (const auto& dep : deps_to_notify) {
            dep->recheck_completion();
        }
    }

    void recheck_completion() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (completed_ || count_ > 0) {
            return;
        }

        if (check_dependencies()) {
            completed_ = true;

            auto callback = completion_callback_;
            lock.unlock();

            // First call our callback
            if (callback) {
                callback();
            }

            // Then notify condition variable and dependents
            cv_.notify_all();
            notify_dependents();
        }
    }

public:
    explicit DependentLatch(std::ptrdiff_t count = 1,
                           std::function<void()> on_complete = nullptr)
        : count_(count), completed_(false), completion_callback_(std::move(on_complete)) {
        if (count < 0) {
            throw std::invalid_argument("DependentLatch count must be non-negative");
        }
        if (count == 0) {
            completed_ = true;
        }
    }

    ~DependentLatch() = default;

    DependentLatch(const DependentLatch&) = delete;
    DependentLatch& operator=(const DependentLatch&) = delete;

    // Add a dependency
    void add_dependency(std::shared_ptr<DependentLatch> dependency) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (completed_) {
            throw std::logic_error("Cannot add dependency to completed latch");
        }

        dependencies_.push_back(dependency);

        // Register ourselves as a dependent of this dependency
        // so it can notify us when it completes
        dependency->add_dependent(this->shared_from_this());
    }

    // Add a dependent (called by other latches that depend on us)
    void add_dependent(std::weak_ptr<DependentLatch> dependent) {
        std::lock_guard<std::mutex> lock(mutex_);
        dependents_.push_back(std::move(dependent));
    }

    // Count down
    void count_down(std::ptrdiff_t n = 1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (n <= 0 || completed_) {
            return;
        }

        count_ = std::max(std::ptrdiff_t{0}, count_ - n);

        if (count_ == 0) {
            // Check dependencies after our count reaches zero
            if (check_dependencies()) {
                completed_ = true;

                auto callback = completion_callback_;
                lock.unlock();

                // First call our callback
                if (callback) {
                    callback();
                }

                // Then notify condition variable and dependents
                cv_.notify_all();
                notify_dependents();  // Notify our dependents that we're complete
            }
        }
    }

    // Wait for this latch and all dependencies
    void wait() const {
        std::unique_lock<std::mutex> lock(mutex_);

        // First check if already completed
        if (completed_) {
            return;
        }

        // Get dependencies list while holding lock to avoid race
        auto deps = dependencies_;
        lock.unlock();

        // Wait for dependencies without holding our lock
        for (const auto& dep : deps) {
            dep->wait();
        }

        // Then wait for this latch
        lock.lock();
        cv_.wait(lock, [this] { return completed_; });
    }

    // Check if complete (including dependencies)
    bool is_complete() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_;
    }

    // Get count
    std::ptrdiff_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    // Get number of dependencies
    std::size_t num_dependencies() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dependencies_.size();
    }
};

// -----------------------------------------------------------------------------
// GroupLatch: Multiple named latches for group coordination
// -----------------------------------------------------------------------------
class GroupLatch {
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::unordered_map<std::string, std::ptrdiff_t> latches_;
    std::unordered_map<std::string, std::ptrdiff_t> initial_counts_;
    std::function<void(const std::string&)> group_callback_;

public:
    explicit GroupLatch(std::function<void(const std::string&)> on_group_complete = nullptr)
        : group_callback_(std::move(on_group_complete)) {}

    ~GroupLatch() = default;

    GroupLatch(const GroupLatch&) = delete;
    GroupLatch& operator=(const GroupLatch&) = delete;

    // Create or reset a named latch
    void create_group(const std::string& name, std::ptrdiff_t count) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (count < 0) {
            throw std::invalid_argument("Group count must be non-negative");
        }

        latches_[name] = count;
        initial_counts_[name] = count;

        if (count == 0) {
            cv_.notify_all();
            if (group_callback_) {
                group_callback_(name);
            }
        }
    }

    // Count down a specific group
    void count_down(const std::string& name, std::ptrdiff_t n = 1) {
        std::unique_lock<std::mutex> lock(mutex_);

        auto it = latches_.find(name);
        if (it == latches_.end()) {
            throw std::invalid_argument("Group does not exist: " + name);
        }

        if (n <= 0 || it->second <= 0) {
            return;
        }

        it->second = std::max(std::ptrdiff_t{0}, it->second - n);

        if (it->second == 0) {
            auto callback = group_callback_;
            lock.unlock();

            cv_.notify_all();

            if (callback) {
                callback(name);
            }
        }
    }

    // Wait for a specific group
    void wait(const std::string& name) const {
        std::unique_lock<std::mutex> lock(mutex_);

        auto it = latches_.find(name);
        if (it == latches_.end()) {
            throw std::invalid_argument("Group does not exist: " + name);
        }

        cv_.wait(lock, [&it] { return it->second == 0; });
    }

    // Wait for all groups
    void wait_all() const {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] {
            for (const auto& [name, count] : latches_) {
                if (count > 0) return false;
            }
            return true;
        });
    }

    // Wait for any group to complete
    std::string wait_any() const {
        std::unique_lock<std::mutex> lock(mutex_);

        std::string completed_group;

        cv_.wait(lock, [this, &completed_group] {
            for (const auto& [name, count] : latches_) {
                if (count == 0) {
                    completed_group = name;
                    return true;
                }
            }
            return false;
        });

        return completed_group;
    }

    // Check if a group is complete
    bool is_complete(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = latches_.find(name);
        if (it == latches_.end()) {
            return false;
        }

        return it->second == 0;
    }

    // Get group count
    std::ptrdiff_t count(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = latches_.find(name);
        if (it == latches_.end()) {
            throw std::invalid_argument("Group does not exist: " + name);
        }

        return it->second;
    }

    // Get all group names
    std::vector<std::string> groups() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::string> names;
        names.reserve(latches_.size());

        for (const auto& [name, _] : latches_) {
            names.push_back(name);
        }

        return names;
    }

    // Reset a group to its initial count
    void reset(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = latches_.find(name);
        if (it == latches_.end()) {
            throw std::invalid_argument("Group does not exist: " + name);
        }

        it->second = initial_counts_[name];
    }

    // Reset all groups
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& [name, count] : latches_) {
            count = initial_counts_[name];
        }
    }

    // Remove a group
    void remove_group(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);

        latches_.erase(name);
        initial_counts_.erase(name);
        cv_.notify_all();
    }
};

// -----------------------------------------------------------------------------
// Utility functions for common latch patterns
// -----------------------------------------------------------------------------

// Execute function when latch completes
template<typename Func>
void on_latch_complete(Latch& latch, Func&& func) {
    std::thread([&latch, func = std::forward<Func>(func)]() {
        latch.wait();
        func();
    }).detach();
}

// Create a latch that auto-counts down after a delay
inline std::shared_ptr<TimedLatch> make_auto_latch(
    std::ptrdiff_t count,
    std::chrono::milliseconds delay) {

    auto latch = std::make_shared<TimedLatch>(count);

    std::thread([latch, delay]() {
        std::this_thread::sleep_for(delay);
        latch->expire();
    }).detach();

    return latch;
}

// Combine multiple latches into one
class CompositeLatch {
private:
    std::vector<std::shared_ptr<Latch>> latches_;

public:
    CompositeLatch() = default;

    void add_latch(std::shared_ptr<Latch> latch) {
        latches_.push_back(std::move(latch));
    }

    void wait() const {
        for (const auto& latch : latches_) {
            latch->wait();
        }
    }

    bool try_wait() const {
        for (const auto& latch : latches_) {
            if (!latch->try_wait()) {
                return false;
            }
        }
        return true;
    }
};

} // namespace fem::core::concurrency
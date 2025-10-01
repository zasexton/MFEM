#pragma once

#ifndef CORE_CONCURRENCY_WORK_STEALING_POOL_H
#define CORE_CONCURRENCY_WORK_STEALING_POOL_H

#include <thread>
#include <vector>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <memory>
#include <chrono>
#include <random>
#include <optional>
#include <type_traits>
#include <utility>
#include <algorithm>

#include <core/config/config.h>
#include <core/config/debug.h>
#include <core/base/singleton.h>
#include <core/error/result.h>

namespace fem::core::concurrency {

// Forward declarations
template<typename T> class WorkHelpingFuture;

/**
 * @brief Work-stealing thread pool for efficient load balancing
 *
 * Implements a work-stealing scheduler where each worker thread has its own
 * deque of tasks. When a thread runs out of work, it attempts to steal tasks
 * from other threads' queues. This provides excellent load balancing for
 * irregular workloads and recursive algorithms.
 *
 * Features:
 * - Per-thread work deques with LIFO local execution
 * - Work stealing with randomized victim selection
 * - Task priorities and affinity hints
 * - Bulk submission support
 * - Adaptive spinning before sleeping
 * - Cache-aware design to minimize contention
 * - Support for nested parallelism
 *
 * Example usage:
 * @code
 * WorkStealingPool pool(4);
 *
 * // Submit a recursive task (fork-join pattern)
 * std::function<int(int, int)> parallel_sum;
 * parallel_sum = [&](int begin, int end) -> int {
 *     if (end - begin <= 1000) {
 *         // Base case: sequential sum
 *         int sum = 0;
 *         for (int i = begin; i < end; ++i) sum += i;
 *         return sum;
 *     }
 *
 *     int mid = begin + (end - begin) / 2;
 *     auto left = pool.submit([&] { return parallel_sum(begin, mid); });
 *     auto right = pool.submit([&] { return parallel_sum(mid, end); });
 *
 *     return left.get() + right.get();
 * };
 *
 * auto result = pool.submit([&] { return parallel_sum(0, 1000000); });
 * std::cout << "Sum: " << result.get() << std::endl;
 * @endcode
 */
class WorkStealingPool {
public:
    /**
     * @brief Task priority levels
     */
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2
    };

    /**
     * @brief Worker affinity hint for task placement
     */
    struct AffinityHint {
        size_t preferred_worker;  // No preference = SIZE_MAX
        bool strict;              // If true, only run on preferred worker

        AffinityHint() : preferred_worker(SIZE_MAX), strict(false) {}
        AffinityHint(size_t worker, bool s = false) : preferred_worker(worker), strict(s) {}
    };

    /**
     * @brief Statistics for monitoring and debugging
     */
    struct Stats {
        std::atomic<size_t> tasks_submitted{0};
        std::atomic<size_t> tasks_completed{0};
        std::atomic<size_t> tasks_stolen{0};
        std::atomic<size_t> steal_attempts{0};
        std::atomic<size_t> failed_steals{0};
        std::atomic<size_t> local_executions{0};
        std::atomic<size_t> spin_iterations{0};
        std::atomic<size_t> sleep_count{0};
        std::chrono::steady_clock::time_point start_time;
    };

private:
    // Task wrapper for internal use
    struct Task {
        std::function<void()> func;
        Priority priority;
        AffinityHint affinity;
        std::chrono::steady_clock::time_point submit_time;

        Task() : priority(Priority::Normal) {}

        Task(std::function<void()> f, Priority p = Priority::Normal, AffinityHint a = AffinityHint{})
            : func(std::move(f)), priority(p), affinity(a),
              submit_time(std::chrono::steady_clock::now()) {}

        // Comparison for priority queue (higher priority first)
        bool operator<(const Task& other) const {
            if (priority != other.priority) {
                return static_cast<int>(priority) < static_cast<int>(other.priority);
            }
            return submit_time > other.submit_time;  // Earlier tasks first
        }
    };

    // Per-worker thread data structure (cache-aligned to prevent false sharing)
    struct alignas(fem::config::CACHE_LINE_SIZE) Worker {
        // Worker's local deque (LIFO for local, FIFO for stealing)
        std::deque<Task> local_queue;
        mutable std::mutex queue_mutex;

        // High-priority queue (always checked first)
        std::priority_queue<Task> priority_queue;

        // Worker thread
        std::thread thread;

        // State flags
        std::atomic<bool> active{true};
        std::atomic<bool> spinning{false};
        std::atomic<bool> sleeping{false};

        // Condition variable for waking sleeping workers
        std::condition_variable wake_cv;

        // Worker ID for debugging
        size_t id;

        // Random engine for steal victim selection
        mutable std::mt19937 rng{std::random_device{}()};

        // Statistics
        std::atomic<size_t> tasks_executed{0};
        std::atomic<size_t> tasks_stolen_from_me{0};
    };

    // Worker threads
    std::vector<std::unique_ptr<Worker>> workers_;

    // Global queue for overflow and initial distribution
    std::queue<Task> global_queue_;
    mutable std::mutex global_mutex_;
    std::condition_variable global_cv_;

    // Idle synchronization for wait_idle()
    mutable std::mutex idle_mutex_;
    std::condition_variable idle_cv_;

    // Pool state
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_workers_{0};
    std::atomic<size_t> next_worker_{0};

    // Configuration
    const size_t num_workers_;
    const size_t max_local_queue_size_;
    const size_t steal_batch_size_;
    const size_t spin_count_;
    const std::chrono::microseconds sleep_duration_;

    // Statistics
    mutable Stats stats_;

    /**
     * @brief Main worker loop
     */
    void worker_loop(size_t worker_id) {
        auto& worker = *workers_[worker_id];
        worker.id = worker_id;

        // Set thread-local storage if needed
        current_worker_id_ = worker_id;

        while (true) {
            Task task;
            bool found_task = false;

            // Phase 1: Try to get task from local queue
            found_task = try_get_local_task(worker, task);

            // Phase 2: Try to steal from other workers
            if (!found_task) {
                found_task = try_steal_task(worker_id, task);
            }

            // Phase 3: Check global queue
            if (!found_task) {
                found_task = try_get_global_task(task);
            }

            // Execute task if found
            if (found_task) {
                execute_task(worker, std::move(task));
            } else {
                // Check if we should stop (and no more tasks)
                if (stop_.load(std::memory_order_acquire)) {
                    // Double-check no tasks remain
                    if (!try_get_local_task(worker, task) &&
                        !try_steal_task(worker_id, task) &&
                        !try_get_global_task(task)) {
                        break;  // Exit worker loop
                    }
                    if (task.func) {
                        execute_task(worker, std::move(task));
                    }
                } else {
                    // No work available - spin then sleep
                    idle_wait(worker);
                }
            }
        }
    }

    /**
     * @brief Try to get task from worker's local queue
     */
    bool try_get_local_task(Worker& worker, Task& task) {
        std::unique_lock<std::mutex> lock(worker.queue_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            return false;  // Avoid blocking on our own queue
        }

        // Check priority queue first
        if (!worker.priority_queue.empty()) {
            task = worker.priority_queue.top();
            worker.priority_queue.pop();
            stats_.local_executions++;
            return true;
        }

        // Check local deque (LIFO - take from back)
        if (!worker.local_queue.empty()) {
            task = std::move(worker.local_queue.back());
            worker.local_queue.pop_back();
            stats_.local_executions++;
            return true;
        }

        return false;
    }

    /**
     * @brief Try to steal tasks from another worker
     */
    bool try_steal_task(size_t thief_id, Task& task) {
        auto& thief = *workers_[thief_id];
        const size_t num_workers = workers_.size();

        // Generate random starting point for victim selection
        std::uniform_int_distribution<size_t> dist(0, num_workers - 2);
        size_t victim_offset = dist(thief.rng);

        stats_.steal_attempts++;

        // Try to steal from each other worker once
        for (size_t i = 0; i < num_workers - 1; ++i) {
            size_t victim_id = (thief_id + 1 + victim_offset + i) % num_workers;

            if (victim_id == thief_id) continue;  // Skip self

            auto& victim = *workers_[victim_id];

            // Try to lock victim's queue
            std::unique_lock<std::mutex> victim_lock(victim.queue_mutex, std::try_to_lock);
            if (!victim_lock.owns_lock()) continue;  // Queue is busy, try next

            // Steal from front of victim's deque (FIFO for stealing)
            if (!victim.local_queue.empty()) {
                // Steal batch of tasks if possible
                size_t steal_count = std::min(steal_batch_size_,
                                             (victim.local_queue.size() + 1) / 2);

                // Lock thief's queue if we need to put tasks there (j > 0)
                std::unique_lock<std::mutex> thief_lock;
                if (steal_count > 1) {
                    thief_lock = std::unique_lock<std::mutex>(thief.queue_mutex);
                }

                for (size_t j = 0; j < steal_count; ++j) {
                    if (j == 0) {
                        task = std::move(victim.local_queue.front());
                        victim.local_queue.pop_front();
                    } else {
                        // Put additional stolen tasks in thief's queue
                        thief.local_queue.push_back(std::move(victim.local_queue.front()));
                        victim.local_queue.pop_front();
                    }
                }

                victim.tasks_stolen_from_me += steal_count;
                stats_.tasks_stolen += steal_count;
                return true;
            }
        }

        stats_.failed_steals++;
        return false;
    }

    /**
     * @brief Try to get task from global queue
     */
    bool try_get_global_task(Task& task) {
        std::unique_lock<std::mutex> lock(global_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || global_queue_.empty()) {
            return false;
        }

        task = std::move(global_queue_.front());
        global_queue_.pop();
        return true;
    }

    /**
     * @brief Execute a task
     */
    void execute_task(Worker& worker, Task task) {
        active_workers_++;
        worker.tasks_executed++;

        try {
            task.func();
            stats_.tasks_completed++;
        } catch (...) {
            // Log error but don't propagate
            stats_.tasks_completed++;  // Count as completed even if failed
        }

        active_workers_--;
    }

    /**
     * @brief Idle waiting with adaptive spinning
     */
    void idle_wait(Worker& worker) {
        // First: Spin for a while
        if (spin_count_ > 0 && !worker.spinning.exchange(true)) {
            stats_.spin_iterations++;

            for (size_t i = 0; i < spin_count_; ++i) {
                if (stop_.load(std::memory_order_acquire)) {
                    worker.spinning = false;
                    return;
                }

                // Yield to other threads
                std::this_thread::yield();

                // Check for new work periodically
                if (i % 10 == 0) {
                    Task task;
                    if (try_get_local_task(worker, task) ||
                        try_steal_task(worker.id, task) ||
                        try_get_global_task(task)) {
                        worker.spinning = false;
                        execute_task(worker, std::move(task));
                        return;
                    }
                }
            }

            worker.spinning = false;
        }

        // Then: Sleep until woken
        if (!worker.sleeping.exchange(true)) {
            stats_.sleep_count++;

            // Use global CV to wake up when ANY work is available
            std::unique_lock<std::mutex> lock(global_mutex_);
            global_cv_.wait_for(lock, sleep_duration_, [this] {
                // Wake immediately if stop signal or work in global queue
                return stop_.load(std::memory_order_acquire) || !global_queue_.empty();
            });

            worker.sleeping = false;
        }
    }

    /**
     * @brief Wake a sleeping worker if available
     */
    void wake_one_worker() {
        for (auto& worker : workers_) {
            if (worker->sleeping.load()) {
                worker->wake_cv.notify_one();
                break;
            }
        }
    }

    /**
     * @brief Select worker for task submission
     */
    size_t select_worker(const AffinityHint& hint) {
        if (hint.preferred_worker < workers_.size()) {
            return hint.preferred_worker;
        }

        // Round-robin distribution
        return next_worker_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    }

    // Thread-local storage for current worker ID
    static thread_local size_t current_worker_id_;

public:
    /**
     * @brief Construct work-stealing pool with specified number of workers
     * @param num_threads Number of worker threads (0 = hardware_concurrency)
     * @param max_local_queue Maximum size of each worker's local queue (0 = unlimited)
     * @param steal_batch Number of tasks to steal at once
     * @param spin_count Number of spin iterations before sleeping
     */
    explicit WorkStealingPool(
        size_t num_threads = 0,
        size_t max_local_queue = 0,
        size_t steal_batch = 4,
        size_t spin_count = 100
    ) : num_workers_(num_threads ? num_threads : std::thread::hardware_concurrency()),
        max_local_queue_size_(max_local_queue),
        steal_batch_size_(steal_batch),
        spin_count_(spin_count),
        sleep_duration_(100) {

        stats_.start_time = std::chrono::steady_clock::now();

        // Create worker threads
        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(std::make_unique<Worker>());
            workers_.back()->thread = std::thread(&WorkStealingPool::worker_loop, this, i);
        }
    }

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~WorkStealingPool() {
        // Ensure clean shutdown
        if (!stop_.load(std::memory_order_acquire)) {
            shutdown();
        }
    }

    // Disable copy operations
    WorkStealingPool(const WorkStealingPool&) = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    /**
     * @brief Submit a task to the pool
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to function
     * @return Future for the task result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return submit_with_options(std::forward<F>(f), Priority::Normal, AffinityHint(),
                                  std::forward<Args>(args)...);
    }

    /**
     * @brief Submit a task with work-helping for recursive algorithms
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to function
     * @return Work-helping future that avoids deadlock in recursive scenarios
     */
    template<typename F, typename... Args>
    auto submit_recursive(F&& f, Args&&... args)
        -> WorkHelpingFuture<std::invoke_result_t<F, Args...>> {
        auto future = submit_with_options(std::forward<F>(f), Priority::Normal, AffinityHint(),
                                         std::forward<Args>(args)...);
        return WorkHelpingFuture<std::invoke_result_t<F, Args...>>(std::move(future), this);
    }

    /**
     * @brief Submit a task with priority
     */
    template<typename F, typename... Args>
    auto submit_with_priority(F&& f, Priority priority, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return submit_with_options(std::forward<F>(f), priority, AffinityHint(),
                                  std::forward<Args>(args)...);
    }

    /**
     * @brief Submit a task with affinity hint
     */
    template<typename F, typename... Args>
    auto submit_with_affinity(F&& f, AffinityHint hint, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return submit_with_options(std::forward<F>(f), Priority::Normal, hint,
                                  std::forward<Args>(args)...);
    }

    /**
     * @brief Submit a task with full options
     */
    template<typename F, typename... Args>
    auto submit_with_options(F&& f, Priority priority, AffinityHint hint, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {

        using return_type = std::invoke_result_t<F, Args...>;

        if (stop_.load(std::memory_order_acquire)) {
            throw std::runtime_error("WorkStealingPool: Cannot submit to stopped pool");
        }

        // Create packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f),
             ...args = std::forward<Args>(args)]() mutable {
                return func(args...);
            }
        );

        auto result = task->get_future();

        // Create task wrapper
        Task wrapped_task([task]() { (*task)(); }, priority, hint);

        // Try to submit to current worker if we're in a worker thread
        if (current_worker_id_ < workers_.size() && !hint.strict) {
            auto& worker = *workers_[current_worker_id_];

            std::unique_lock<std::mutex> lock(worker.queue_mutex, std::try_to_lock);
            if (lock.owns_lock()) {
                if (max_local_queue_size_ == 0 ||
                    worker.local_queue.size() < max_local_queue_size_) {

                    if (priority == Priority::High) {
                        worker.priority_queue.push(wrapped_task);
                    } else {
                        worker.local_queue.push_back(std::move(wrapped_task));
                    }

                    stats_.tasks_submitted++;
                    return result;
                }
            }
        }

        // Submit to selected worker
        size_t worker_id = select_worker(hint);
        auto& worker = *workers_[worker_id];

        {
            std::lock_guard<std::mutex> lock(worker.queue_mutex);

            // Check queue size limit
            if (max_local_queue_size_ > 0 &&
                worker.local_queue.size() >= max_local_queue_size_) {
                // Overflow to global queue
                std::lock_guard<std::mutex> global_lock(global_mutex_);
                global_queue_.push(std::move(wrapped_task));
                global_cv_.notify_one();
            } else {
                if (priority == Priority::High) {
                    worker.priority_queue.push(wrapped_task);
                } else {
                    worker.local_queue.push_back(std::move(wrapped_task));
                }
            }
        }

        // Wake sleeping workers
        global_cv_.notify_all();

        stats_.tasks_submitted++;
        return result;
    }

    /**
     * @brief Submit multiple tasks in bulk
     */
    template<typename F>
    void submit_bulk(F&& f, size_t count, Priority priority = Priority::Normal) {
        std::vector<Task> tasks;
        tasks.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            tasks.emplace_back(
                [func = f, idx = i]() { func(idx); },
                priority
            );
        }

        // Distribute tasks across workers
        size_t tasks_per_worker = (count + num_workers_ - 1) / num_workers_;
        size_t task_idx = 0;

        for (size_t w = 0; w < num_workers_ && task_idx < count; ++w) {
            auto& worker = *workers_[w];
            std::lock_guard<std::mutex> lock(worker.queue_mutex);

            size_t batch_size = std::min(tasks_per_worker, count - task_idx);
            for (size_t i = 0; i < batch_size; ++i) {
                if (priority == Priority::High) {
                    worker.priority_queue.push(tasks[task_idx++]);
                } else {
                    worker.local_queue.push_back(std::move(tasks[task_idx++]));
                }
            }
        }

        // Wake all sleeping workers
        global_cv_.notify_all();

        stats_.tasks_submitted += count;
    }

    /**
     * @brief Submit a range of tasks
     */
    template<typename Iterator, typename F>
    void submit_range(Iterator begin, Iterator end, F&& f, Priority priority = Priority::Normal) {
        size_t count = std::distance(begin, end);
        size_t idx = 0;

        for (auto it = begin; it != end; ++it, ++idx) {
            size_t worker_id = idx % num_workers_;
            auto& worker = *workers_[worker_id];

            Task task([func = f, value = *it]() { func(value); }, priority);

            std::lock_guard<std::mutex> lock(worker.queue_mutex);
            if (priority == Priority::High) {
                worker.priority_queue.push(task);
            } else {
                worker.local_queue.push_back(std::move(task));
            }

            if (worker.sleeping.load()) {
                worker.wake_cv.notify_one();
            }
        }

        stats_.tasks_submitted += count;
    }

    /**
     * @brief Wait until all tasks are complete
     */
    void wait_idle() {
        // More robust wait that avoids lock contention
        size_t spin_count = 0;
        const size_t max_spins = 1000;

        while (true) {
            // First, check active workers
            size_t active = active_workers_.load(std::memory_order_acquire);
            if (active > 0) {
                std::this_thread::yield();
                spin_count = 0;  // Reset spin count when we see activity
                continue;
            }

            // No active workers - verify no pending tasks
            size_t total_tasks = total_queue_size();

            if (total_tasks == 0) {
                // Double-check after a brief delay
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Final verification
                if (active_workers_.load(std::memory_order_acquire) == 0 &&
                    total_queue_size() == 0) {
                    break;  // We're done
                }
            }

            // Increment spin count
            spin_count++;

            // If we've been spinning too long, wake workers to process any stuck tasks
            if (spin_count >= max_spins) {
                // Wake all workers to ensure they process any tasks
                global_cv_.notify_all();
                for (auto& worker : workers_) {
                    worker->wake_cv.notify_all();
                }
                spin_count = 0;  // Reset counter
            }

            // Yield before next iteration
            std::this_thread::yield();
        }
    }

    /**
     * @brief Try to execute one task if available (for work-helping)
     * @return true if a task was executed
     */
    bool try_execute_one() {
        // Only allow work-helping if we're in a worker thread
        if (current_worker_id_ >= workers_.size()) {
            return false;
        }

        auto& worker = *workers_[current_worker_id_];
        Task task;

        // Try to get a task from our own queue first
        if (try_get_local_task(worker, task)) {
            execute_task(worker, std::move(task));
            return true;
        }

        // Try stealing from others
        if (try_steal_task(current_worker_id_, task)) {
            execute_task(worker, std::move(task));
            return true;
        }

        // Try global queue
        if (try_get_global_task(task)) {
            execute_task(worker, std::move(task));
            return true;
        }

        return false;
    }

    /**
     * @brief Shutdown the pool and wait for all tasks
     */
    void shutdown() {
        // Only shutdown once
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // Already shutting down
        }

        // Wake all workers multiple times to ensure they see the stop signal
        for (int i = 0; i < 3; ++i) {
            // Wake via individual worker CVs
            for (auto& worker : workers_) {
                worker->wake_cv.notify_all();
            }
            // Wake via global CV
            global_cv_.notify_all();

            // Brief pause to let notifications propagate
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // Wait for all threads to finish with timeout
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }

        // Clear workers
        workers_.clear();

        // Reset active workers counter
        active_workers_.store(0, std::memory_order_release);
    }

    /**
     * @brief Get current worker ID (if called from worker thread)
     * @return Worker ID or SIZE_MAX if not in worker thread
     */
    static size_t get_current_worker_id() {
        return current_worker_id_;
    }

    // Status queries

    /**
     * @brief Get number of worker threads
     */
    size_t worker_count() const noexcept {
        return num_workers_;
    }

    /**
     * @brief Get total queue size across all workers
     */
    size_t total_queue_size() const {
        size_t total = 0;

        for (const auto& worker : workers_) {
            std::lock_guard<std::mutex> lock(worker->queue_mutex);
            total += worker->local_queue.size();
            total += worker->priority_queue.size();
        }

        std::lock_guard<std::mutex> lock(global_mutex_);
        total += global_queue_.size();

        return total;
    }

    /**
     * @brief Get queue size for specific worker
     */
    size_t worker_queue_size(size_t worker_id) const {
        if (worker_id >= workers_.size()) {
            return 0;
        }

        const auto& worker = *workers_[worker_id];
        std::lock_guard<std::mutex> lock(worker.queue_mutex);
        return worker.local_queue.size() + worker.priority_queue.size();
    }

    /**
     * @brief Get number of active (executing) workers
     */
    size_t active_workers() const noexcept {
        return active_workers_.load();
    }

    /**
     * @brief Check if pool is stopped
     */
    bool is_stopped() const noexcept {
        return stop_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get pool statistics
     */
    const Stats& statistics() const noexcept {
        return stats_;
    }

    /**
     * @brief Get per-worker statistics
     */
    struct WorkerStats {
        size_t tasks_executed;
        size_t tasks_stolen_from_me;
        size_t queue_size;
        bool is_spinning;
        bool is_sleeping;
    };

    WorkerStats worker_statistics(size_t worker_id) const {
        if (worker_id >= workers_.size()) {
            return {};
        }

        const auto& worker = *workers_[worker_id];

        WorkerStats stats;
        stats.tasks_executed = worker.tasks_executed.load();
        stats.tasks_stolen_from_me = worker.tasks_stolen_from_me.load();
        stats.queue_size = worker_queue_size(worker_id);
        stats.is_spinning = worker.spinning.load();
        stats.is_sleeping = worker.sleeping.load();

        return stats;
    }

    /**
     * @brief Reset statistics
     */
    void reset_statistics() {
        stats_.tasks_submitted = 0;
        stats_.tasks_completed = 0;
        stats_.tasks_stolen = 0;
        stats_.steal_attempts = 0;
        stats_.failed_steals = 0;
        stats_.local_executions = 0;
        stats_.spin_iterations = 0;
        stats_.sleep_count = 0;
        stats_.start_time = std::chrono::steady_clock::now();

        for (auto& worker : workers_) {
            worker->tasks_executed = 0;
            worker->tasks_stolen_from_me = 0;
        }
    }

    /**
     * @brief Get steal efficiency (stolen/attempts ratio)
     */
    double steal_efficiency() const {
        size_t attempts = stats_.steal_attempts.load();
        if (attempts == 0) return 0.0;

        size_t stolen = stats_.tasks_stolen.load();
        return static_cast<double>(stolen) / static_cast<double>(attempts);
    }

    /**
     * @brief Get load balance factor (0 = perfect balance, higher = more imbalanced)
     */
    double load_balance_factor() const {
        if (workers_.empty()) return 0.0;

        std::vector<size_t> executions;
        executions.reserve(workers_.size());

        size_t total = 0;
        for (const auto& worker : workers_) {
            size_t executed = worker->tasks_executed.load();
            executions.push_back(executed);
            total += executed;
        }

        if (total == 0) return 0.0;

        double mean = static_cast<double>(total) / static_cast<double>(workers_.size());
        double variance = 0.0;

        for (size_t executed : executions) {
            double diff = static_cast<double>(executed) - mean;
            variance += diff * diff;
        }

        variance /= static_cast<double>(workers_.size());
        return std::sqrt(variance) / mean;  // Coefficient of variation
    }
};

// Static member definition
thread_local size_t WorkStealingPool::current_worker_id_ = SIZE_MAX;

/**
 * @brief Work-helping future wrapper
 *
 * Wraps a std::future and helps execute tasks while waiting to avoid deadlock
 * in recursive task submission scenarios.
 */
template<typename T>
class WorkHelpingFuture {
private:
    std::future<T> future_;
    WorkStealingPool* pool_;

public:
    WorkHelpingFuture(std::future<T>&& f, WorkStealingPool* p)
        : future_(std::move(f)), pool_(p) {}

    WorkHelpingFuture(WorkHelpingFuture&&) = default;
    WorkHelpingFuture& operator=(WorkHelpingFuture&&) = default;

    /**
     * @brief Get the result, helping with work while waiting
     */
    T get() {
        // If we're in a worker thread, help execute tasks while waiting
        if (pool_ && WorkStealingPool::get_current_worker_id() < pool_->worker_count()) {
            while (future_.wait_for(std::chrono::microseconds(10)) != std::future_status::ready) {
                // Try to execute a task to help make progress
                pool_->try_execute_one();
            }
        }
        return future_.get();
    }

    /**
     * @brief Check if the future is ready
     */
    bool ready() const {
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    /**
     * @brief Wait for the result without helping
     */
    void wait() {
        future_.wait();
    }

    /**
     * @brief Check if the future is valid
     */
    bool valid() const {
        return future_.valid();
    }
};

/**
 * @brief Global work-stealing pool singleton
 */
class GlobalWorkStealingPool : public base::Singleton<GlobalWorkStealingPool> {
    friend class base::Singleton<GlobalWorkStealingPool>;

private:
    std::unique_ptr<WorkStealingPool> pool_;
    mutable std::mutex init_mutex_;

    GlobalWorkStealingPool() = default;

public:
    /**
     * @brief Get or create the global work-stealing pool
     */
    WorkStealingPool& get_pool(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!pool_) {
            pool_ = std::make_unique<WorkStealingPool>(num_threads);
        }
        return *pool_;
    }

    /**
     * @brief Reset the global pool
     */
    void reset(size_t num_threads = 0, size_t max_local_queue = 0,
               size_t steal_batch = 4, size_t spin_count = 100) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_) {
            pool_->shutdown();
        }
        pool_ = std::make_unique<WorkStealingPool>(
            num_threads, max_local_queue, steal_batch, spin_count
        );
    }

    /**
     * @brief Shutdown the global pool
     */
    void shutdown() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_) {
            pool_->shutdown();
            pool_.reset();
        }
    }
};

/**
 * @brief Get the global work-stealing pool
 */
inline WorkStealingPool& global_work_stealing_pool() {
    return GlobalWorkStealingPool::instance().get_pool();
}

/**
 * @brief Scoped work context for nested parallelism
 *
 * Automatically submits tasks to parent worker's queue when possible,
 * avoiding unnecessary thread creation in nested parallel regions.
 */
class WorkContext {
private:
    WorkStealingPool* pool_;
    size_t parent_worker_;
    bool owns_pool_;

public:
    /**
     * @brief Create work context with existing pool
     */
    explicit WorkContext(WorkStealingPool& pool)
        : pool_(&pool),
          parent_worker_(WorkStealingPool::get_current_worker_id()),
          owns_pool_(false) {}

    /**
     * @brief Create work context with new pool
     */
    explicit WorkContext(size_t num_threads)
        : pool_(new WorkStealingPool(num_threads)),
          parent_worker_(SIZE_MAX),
          owns_pool_(true) {}

    ~WorkContext() {
        if (owns_pool_) {
            delete pool_;
        }
    }

    // Disable copy
    WorkContext(const WorkContext&) = delete;
    WorkContext& operator=(const WorkContext&) = delete;

    // Move operations
    WorkContext(WorkContext&& other) noexcept
        : pool_(other.pool_),
          parent_worker_(other.parent_worker_),
          owns_pool_(other.owns_pool_) {
        other.pool_ = nullptr;
        other.owns_pool_ = false;
    }

    WorkContext& operator=(WorkContext&& other) noexcept {
        if (this != &other) {
            if (owns_pool_) {
                delete pool_;
            }
            pool_ = other.pool_;
            parent_worker_ = other.parent_worker_;
            owns_pool_ = other.owns_pool_;
            other.pool_ = nullptr;
            other.owns_pool_ = false;
        }
        return *this;
    }

    /**
     * @brief Get the pool for this context
     */
    WorkStealingPool& pool() { return *pool_; }
    const WorkStealingPool& pool() const { return *pool_; }

    /**
     * @brief Check if we're in a nested context
     */
    bool is_nested() const {
        return parent_worker_ != SIZE_MAX;
    }

    /**
     * @brief Submit task with context awareness
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) {
        if (is_nested()) {
            // Prefer parent worker for nested tasks
            return pool_->submit_with_affinity(
                std::forward<F>(f),
                {parent_worker_, false},
                std::forward<Args>(args)...
            );
        } else {
            return pool_->submit(std::forward<F>(f), std::forward<Args>(args)...);
        }
    }
};

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_WORK_STEALING_POOL_H
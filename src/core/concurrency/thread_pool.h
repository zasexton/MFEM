#pragma once

#ifndef CORE_CONCURRENCY_THREAD_POOL_H
#define CORE_CONCURRENCY_THREAD_POOL_H

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <memory>
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <optional>

#include <core/config/config.h>
#include <core/config/debug.h>
#include <core/base/singleton.h>
#include <core/memory/object_pool.h>
#include <core/memory/concurrent_pool.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

namespace fem::core::concurrency {

/**
 * @brief Thread pool for parallel task execution
 *
 * Provides efficient thread reuse and task queuing for parallel workloads.
 * Features include:
 * - Automatic thread count detection
 * - Task prioritization
 * - Bulk submission support
 * - Pause/resume capability
 * - Graceful shutdown
 * - Memory-efficient task storage using object pools
 *
 * Example usage:
 * @code
 * ThreadPool pool(4);  // 4 worker threads
 *
 * // Submit a simple task
 * auto future = pool.submit([]() { return 42; });
 * int result = future.get();  // 42
 *
 * // Submit with arguments
 * auto future2 = pool.submit([](int a, int b) { return a + b; }, 10, 20);
 * int sum = future2.get();  // 30
 *
 * // Bulk submission
 * pool.submit_bulk([](size_t i) { process_item(i); }, 100);
 * @endcode
 */
class ThreadPool {
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
     * @brief Thread pool statistics
     */
    struct Stats {
        std::atomic<size_t> tasks_submitted{0};
        std::atomic<size_t> tasks_completed{0};
        std::atomic<size_t> tasks_failed{0};
        std::atomic<size_t> active_threads{0};
        std::atomic<size_t> queue_size{0};
        std::chrono::steady_clock::time_point start_time;
    };

private:
    // Internal task wrapper
    struct TaskWrapper {
        std::function<void()> func;
        Priority priority;
        std::chrono::steady_clock::time_point submit_time;

        TaskWrapper() : priority(Priority::Normal) {}
        TaskWrapper(std::function<void()> f, Priority p = Priority::Normal)
            : func(std::move(f)), priority(p), submit_time(std::chrono::steady_clock::now()) {}

        // For priority queue ordering (higher priority first)
        bool operator<(const TaskWrapper& other) const {
            if (priority != other.priority) {
                return static_cast<int>(priority) < static_cast<int>(other.priority);
            }
            // Earlier submitted tasks have priority within same priority level
            return submit_time > other.submit_time;
        }
    };

    // Worker threads
    std::vector<std::thread> threads_;

    // Task queue with priority support
    std::priority_queue<TaskWrapper> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_idle_;

    // Thread pool state
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<size_t> active_tasks_{0};

    // Statistics
    mutable Stats stats_;

    // Memory pool for task objects (reuse allocations)
    mutable memory::ObjectPool<TaskWrapper> task_pool_;

    // Configuration
    const size_t max_queue_size_;
    const bool enable_work_stealing_;  // Reserved for future work-stealing implementation

    /**
     * @brief Worker thread main loop
     */
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            // Wait for task or stop signal
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                cv_task_.wait(lock, [this] {
                    return stop_ || (!paused_ && !tasks_.empty());
                });

                if (stop_ && tasks_.empty()) {
                    break;
                }

                if (!tasks_.empty() && !paused_) {
                    TaskWrapper wrapper = tasks_.top();
                    tasks_.pop();
                    task = std::move(wrapper.func);
                    stats_.queue_size = tasks_.size();
                }
            }

            // Execute task outside of lock
            if (task) {
                stats_.active_threads++;
                active_tasks_++;

                task();

                active_tasks_--;
                stats_.active_threads--;

                // Notify if we're now idle
                if (active_tasks_ == 0) {
                    cv_idle_.notify_all();
                }
            }
        }
    }

public:
    /**
     * @brief Construct thread pool with specified number of threads
     * @param num_threads Number of worker threads (0 = hardware_concurrency)
     * @param max_queue_size Maximum task queue size (0 = unlimited)
     */
    explicit ThreadPool(size_t num_threads = 0, size_t max_queue_size = 0)
        : max_queue_size_(max_queue_size),
          enable_work_stealing_(false) {

        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {
                num_threads = 2;  // Fallback to 2 threads
            }
        }

        stats_.start_time = std::chrono::steady_clock::now();

        // Start worker threads
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back(&ThreadPool::worker_loop, this);
        }
    }

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool() {
        shutdown();
    }

    // Disable copy operations
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Move operations
    ThreadPool(ThreadPool&& other) noexcept = default;
    ThreadPool& operator=(ThreadPool&& other) noexcept = default;

    /**
     * @brief Submit a task to the thread pool
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to function
     * @return Future for the task result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return submit_with_priority(std::forward<F>(f), Priority::Normal, std::forward<Args>(args)...);
    }

    /**
     * @brief Submit a task to the thread pool with explicit priority
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param priority Task priority
     * @param args Arguments to pass to function
     * @return Future for the task result
     */
    template<typename F, typename... Args>
    auto submit_with_priority(F&& f, Priority priority, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {

        using return_type = std::invoke_result_t<F, Args...>;

        if (stop_) {
            throw std::runtime_error("ThreadPool: Cannot submit to stopped pool");
        }

        // Create packaged task with bound arguments
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f),
             ...args = std::forward<Args>(args)]() mutable {
                return func(args...);
            }
        );

        auto result = task->get_future();

        // Add to queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Check queue size limit
            if (max_queue_size_ > 0 && tasks_.size() >= max_queue_size_) {
                throw std::runtime_error("ThreadPool: Queue size limit exceeded");
            }

            tasks_.emplace([task, this]() {
                try {
                    (*task)();
                    stats_.tasks_completed++;
                } catch (...) {
                    stats_.tasks_failed++;
                    throw; // Re-throw to preserve exception handling
                }
            }, priority);
            stats_.tasks_submitted++;
            stats_.queue_size = tasks_.size();
        }

        cv_task_.notify_one();
        return result;
    }

    /**
     * @brief Submit a task that returns a Result type for error handling
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function that returns Result<T, E>
     * @param args Arguments to pass to function
     * @param priority Task priority
     * @return Future for the Result
     */
    template<typename F, typename... Args>
    auto submit_with_result(F&& f, Args&&... args, Priority priority = Priority::Normal)
        -> std::future<std::invoke_result_t<F, Args...>>
        requires requires {
            typename std::invoke_result_t<F, Args...>::value_type;
            typename std::invoke_result_t<F, Args...>::error_type;
        } {

        return submit_with_priority(std::forward<F>(f), priority, std::forward<Args>(args)...);
    }

    /**
     * @brief Submit multiple tasks in bulk
     * @tparam F Function type accepting size_t index
     * @param f Function to execute for each index
     * @param count Number of tasks to submit
     * @param priority Task priority for all tasks
     */
    template<typename F>
    void submit_bulk(F&& f, size_t count, Priority priority = Priority::Normal) {
        for (size_t i = 0; i < count; ++i) {
            submit_with_priority([func = f, idx = i]() { func(idx); }, priority);
        }
    }

    /**
     * @brief Submit a range of tasks
     * @tparam Iterator Iterator type
     * @tparam F Function type
     * @param begin Range begin
     * @param end Range end
     * @param f Function to apply to each element
     * @param priority Task priority
     */
    template<typename Iterator, typename F>
    void submit_range(Iterator begin, Iterator end, F&& f, Priority priority = Priority::Normal) {
        for (auto it = begin; it != end; ++it) {
            submit_with_priority([func = f, value = *it]() { func(value); }, priority);
        }
    }

    /**
     * @brief Pause task execution (running tasks continue)
     */
    void pause() {
        paused_ = true;
    }

    /**
     * @brief Resume task execution
     */
    void resume() {
        paused_ = false;
        cv_task_.notify_all();
    }

    /**
     * @brief Wait until all tasks are complete
     */
    void wait_idle() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_idle_.wait(lock, [this] {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

    /**
     * @brief Wait for a specified duration or until idle
     * @param timeout Maximum time to wait
     * @return true if became idle, false if timeout
     */
    bool wait_idle_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return cv_idle_.wait_for(lock, timeout, [this] {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

    /**
     * @brief Stop accepting new tasks and wait for completion
     */
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }

        cv_task_.notify_all();

        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        threads_.clear();
    }

    /**
     * @brief Stop immediately, cancelling pending tasks
     */
    void shutdown_now() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
            // Clear pending tasks
            while (!tasks_.empty()) {
                tasks_.pop();
            }
        }

        cv_task_.notify_all();

        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        threads_.clear();
    }

    /**
     * @brief Try to steal a task (for future work-stealing implementation)
     * @return Optional task function if available
     */
    std::optional<std::function<void()>> try_steal() {
        std::unique_lock<std::mutex> lock(queue_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || tasks_.empty()) {
            return std::nullopt;
        }

        TaskWrapper wrapper = tasks_.top();
        tasks_.pop();
        stats_.queue_size = tasks_.size();
        return std::move(wrapper.func);
    }

    // Status queries

    /**
     * @brief Get number of worker threads
     */
    size_t thread_count() const noexcept {
        return threads_.size();
    }

    /**
     * @brief Get current queue size
     */
    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

    /**
     * @brief Get number of active tasks
     */
    size_t active_tasks() const noexcept {
        return active_tasks_.load();
    }

    /**
     * @brief Check if pool is stopped
     */
    bool is_stopped() const noexcept {
        return stop_.load();
    }

    /**
     * @brief Check if pool is paused
     */
    bool is_paused() const noexcept {
        return paused_.load();
    }

    /**
     * @brief Get thread pool statistics
     */
    const Stats& statistics() const noexcept {
        return stats_;
    }

    /**
     * @brief Reset statistics
     */
    void reset_statistics() {
        stats_.tasks_submitted.store(0);
        stats_.tasks_completed.store(0);
        stats_.tasks_failed.store(0);
        stats_.active_threads.store(0);
        stats_.queue_size.store(0);
        stats_.start_time = std::chrono::steady_clock::now();
    }

    /**
     * @brief Get maximum queue size (0 = unlimited)
     */
    size_t max_queue_size() const noexcept {
        return max_queue_size_;
    }
};

/**
 * @brief Global thread pool singleton
 *
 * Provides a shared thread pool for the entire application.
 * Automatically initialized with hardware_concurrency threads.
 *
 * Example:
 * @code
 * auto& pool = global_thread_pool();
 * auto future = pool.submit([]() { return compute(); });
 * @endcode
 */
class GlobalThreadPool : public base::Singleton<GlobalThreadPool> {
    friend class base::Singleton<GlobalThreadPool>;

private:
    std::unique_ptr<ThreadPool> pool_;
    mutable std::mutex init_mutex_;

    GlobalThreadPool() = default;

public:
    /**
     * @brief Get or create the global thread pool
     * @param num_threads Number of threads (0 = auto-detect), only used on first call
     */
    ThreadPool& get_pool(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!pool_) {
            pool_ = std::make_unique<ThreadPool>(num_threads);
        }
        return *pool_;
    }

    /**
     * @brief Reset the global pool with new configuration
     * @param num_threads Number of threads (0 = auto-detect)
     */
    void reset(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_) {
            pool_->shutdown();
        }
        pool_ = std::make_unique<ThreadPool>(num_threads);
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
 * @brief Get the global thread pool
 * @return Reference to the global thread pool
 */
inline ThreadPool& global_thread_pool() {
    return GlobalThreadPool::instance().get_pool();
}

/**
 * @brief RAII guard for pausing thread pool
 *
 * Pauses the thread pool on construction and resumes on destruction.
 *
 * Example:
 * @code
 * {
 *     ThreadPoolPauseGuard guard(pool);
 *     // Pool is paused here
 *     // Do some work that shouldn't be interleaved with pool tasks
 * }  // Pool resumes here
 * @endcode
 */
class ThreadPoolPauseGuard {
private:
    ThreadPool& pool_;
    bool was_paused_;

public:
    explicit ThreadPoolPauseGuard(ThreadPool& pool)
        : pool_(pool), was_paused_(pool.is_paused()) {
        if (!was_paused_) {
            pool_.pause();
        }
    }

    ~ThreadPoolPauseGuard() {
        if (!was_paused_) {
            pool_.resume();
        }
    }

    // Disable copy/move
    ThreadPoolPauseGuard(const ThreadPoolPauseGuard&) = delete;
    ThreadPoolPauseGuard& operator=(const ThreadPoolPauseGuard&) = delete;
    ThreadPoolPauseGuard(ThreadPoolPauseGuard&&) = delete;
    ThreadPoolPauseGuard& operator=(ThreadPoolPauseGuard&&) = delete;
};

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_THREAD_POOL_H
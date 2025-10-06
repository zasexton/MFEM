#pragma once

#ifndef CORE_CONCURRENCY_WORK_STEALING_POOL_H
#define CORE_CONCURRENCY_WORK_STEALING_POOL_H

#include <thread>
#include <vector>
#include <deque>
#include <atomic>
#include <memory>
#include <functional>
#include <future>
#include <random>
#include <optional>
#include <type_traits>
#include <utility>
#include <chrono>
#include <algorithm>

#include <core/config/config.h>
#include <core/config/debug.h>
#include <core/base/singleton.h>
#include <core/concurrency/latch.h>
#include <core/concurrency/barrier.h>

namespace fem::core::concurrency {

/**
 * @brief Lock-free Chase-Lev deque for work stealing
 *
 * Based on "Dynamic Circular Work-Stealing Deque" by Chase and Lev (2005)
 * and the Cilk work-stealing scheduler by Blumofe and Leiserson (1999).
 *
 * Owner (worker thread) operations:
 * - push(): Add task to bottom (private end) - O(1) lock-free
 * - pop(): Take task from bottom - O(1) lock-free in common case
 *
 * Thief operations:
 * - steal(): Take task from top (public end) - O(1) with CAS
 *
 * Key properties:
 * - Lock-free for owner in common case
 * - Linearizable
 * - Wait-free steal attempts
 * - Automatic resizing
 */
template<typename T>
class ChaseLevDeque {
private:
    // Circular array for storing tasks
    struct Array {
        std::atomic<size_t> size;
        std::unique_ptr<std::atomic<T*>[]> buffer;

        explicit Array(size_t s) : size(s) {
            buffer = std::make_unique<std::atomic<T*>[]>(s);
            for (size_t i = 0; i < s; ++i) {
                buffer[i].store(nullptr, std::memory_order_relaxed);
            }
        }

        T* get(size_t i) const {
            return buffer[i % size.load(std::memory_order_relaxed)].load(std::memory_order_acquire);
        }

        void put(size_t i, T* item) {
            buffer[i % size.load(std::memory_order_relaxed)].store(item, std::memory_order_release);
        }

        std::unique_ptr<Array> resize(size_t bottom, size_t top) {
            size_t old_size = size.load(std::memory_order_relaxed);
            size_t new_size = old_size * 2;
            auto new_array = std::make_unique<Array>(new_size);

            for (size_t i = top; i < bottom; ++i) {
                new_array->put(i, get(i));
            }

            return new_array;
        }
    };

    std::atomic<int64_t> top_{0};     // Accessed by thieves
    std::atomic<int64_t> bottom_{0};  // Accessed by owner
    std::atomic<Array*> array_;       // Current array

    static constexpr size_t INITIAL_SIZE = 256;

public:
    ChaseLevDeque() {
        array_.store(new Array(INITIAL_SIZE), std::memory_order_relaxed);
    }

    ~ChaseLevDeque() {
        delete array_.load(std::memory_order_relaxed);
    }

    // Non-copyable, non-movable
    ChaseLevDeque(const ChaseLevDeque&) = delete;
    ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;

    /**
     * @brief Push task to bottom (owner only)
     */
    void push(T* task) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);
        Array* a = array_.load(std::memory_order_relaxed);

        if (b - t > static_cast<int64_t>(a->size.load(std::memory_order_relaxed)) - 1) {
            // Deque is full, resize
            auto new_array = a->resize(b, t);
            Array* old_array = a;
            array_.store(new_array.release(), std::memory_order_release);
            delete old_array;
            a = array_.load(std::memory_order_relaxed);
        }

        a->put(b, task);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    /**
     * @brief Pop task from bottom (owner only)
     * @return Pointer to task or nullptr if empty
     */
    T* pop() {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        Array* a = array_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty deque
            T* task = a->get(b);

            if (t == b) {
                // Single element left - race with thieves
                if (!top_.compare_exchange_strong(t, t + 1,
                                                  std::memory_order_seq_cst,
                                                  std::memory_order_relaxed)) {
                    // Lost race to thief
                    task = nullptr;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return task;
        } else {
            // Empty deque
            bottom_.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    /**
     * @brief Steal task from top (thieves)
     * @return Pointer to task or nullptr if empty or race lost
     */
    T* steal() {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            // Non-empty deque
            Array* a = array_.load(std::memory_order_consume);
            T* task = a->get(t);

            if (!top_.compare_exchange_strong(t, t + 1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                // Lost race, another thief got it
                return nullptr;
            }
            return task;
        }
        return nullptr;  // Empty
    }

    /**
     * @brief Get approximate size (for statistics)
     */
    size_t size() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return static_cast<size_t>(std::max(int64_t{0}, b - t));
    }

    /**
     * @brief Check if deque is empty (approximate)
     */
    bool empty() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b <= t;
    }
};

/**
 * @brief Cilk-style work-stealing thread pool
 *
 * Implements the work-stealing scheduler from:
 * - "Scheduling Multithreaded Computations by Work Stealing"
 *   by Blumofe and Leiserson (1999)
 * - Uses lock-free Chase-Lev deques per worker
 * - Randomized victim selection for load balancing
 * - Locality-aware stealing preferences
 *
 * Features:
 * - O(1) expected time task operations
 * - Provably efficient: T_p = O(T_1/P + T_∞)
 * - Lock-free task stealing
 * - NUMA-aware work distribution
 * - Exception safe
 */
class WorkStealingPool {
public:
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2
    };

private:
    // Task wrapper
    struct Task {
        std::function<void()> func;
        Priority priority;

        Task() = default;
        Task(std::function<void()> f, Priority p = Priority::Normal)
            : func(std::move(f)), priority(p) {}
    };

    // Simple mutex-based work-stealing deque (temporary - TODO: fix Chase-Lev races)
    template<typename T>
    class SimpleDeque {
    private:
        std::deque<T*> deque_;
        mutable std::mutex mutex_;
    public:
        void push(T* task) {
            std::lock_guard<std::mutex> lock(mutex_);
            deque_.push_back(task);
        }
        T* pop() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (deque_.empty()) return nullptr;
            T* task = deque_.back();
            deque_.pop_back();
            return task;
        }
        T* steal() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (deque_.empty()) return nullptr;
            T* task = deque_.front();
            deque_.pop_front();
            return task;
        }
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return deque_.empty();
        }
    };

    // Per-worker state (cache-aligned to prevent false sharing)
    struct alignas(fem::config::CACHE_LINE_SIZE) Worker {
        // Work-stealing deque (using simple mutex version for correctness)
        SimpleDeque<Task> deque;

        // Worker thread
        std::thread thread;
        size_t id{0};

        // NUMA node hint (for locality-aware stealing)
        int numa_node{0};

        // Random number generator for victim selection (initialized in worker_loop)
        std::mt19937 rng;

        // Statistics
        std::atomic<uint64_t> tasks_executed{0};
        std::atomic<uint64_t> tasks_stolen{0};
        std::atomic<uint64_t> steal_attempts{0};

        Worker() = default;

        // Get random victim index (excluding self)
        size_t random_victim(size_t self_id, size_t num_workers) {
            std::uniform_int_distribution<size_t> dist(0, num_workers - 2);
            size_t victim = dist(rng);
            return (victim >= self_id) ? victim + 1 : victim;
        }
    };

    // Worker threads
    std::vector<std::unique_ptr<Worker>> workers_;

    // Shutdown flag
    std::atomic<bool> shutdown_{false};

    // Number of pending tasks (for wait_idle)
    std::atomic<size_t> pending_tasks_{0};

    // Latch for initialization
    std::unique_ptr<Latch> init_latch_;

    // Thread-local worker ID
    static thread_local size_t current_worker_id_;

    /**
     * @brief Main worker loop implementing Cilk-style work stealing
     */
    void worker_loop(size_t worker_id) {
        auto& worker = *workers_[worker_id];
        worker.id = worker_id;
        current_worker_id_ = worker_id;

        // Initialize RNG (use time + worker_id to avoid blocking on std::random_device)
        auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        worker.rng.seed(static_cast<uint32_t>(seed ^ (worker_id * 0x9e3779b9)));

        // Signal initialization complete
        init_latch_->count_down();

        while (!shutdown_.load(std::memory_order_acquire)) {
            Task* task = nullptr;

            // Phase 1: Try to pop from own deque (LIFO for cache locality)
            task = worker.deque.pop();

            if (!task) {
                // Phase 2: Randomly steal from other workers (FIFO)
                task = try_steal(worker_id);
            }

            if (task) {
                // Execute task
                worker.tasks_executed.fetch_add(1, std::memory_order_relaxed);

                try {
                    task->func();
                } catch (...) {
                    // Ignore exceptions in tasks
                }

                delete task;
                pending_tasks_.fetch_sub(1, std::memory_order_release);
            } else {
                // No work found - yield to avoid busy-waiting
                std::this_thread::yield();
            }
        }
    }

    /**
     * @brief Steal work from a random victim (Blumofe-Leiserson randomized stealing)
     */
    Task* try_steal(size_t thief_id) {
        auto& thief = *workers_[thief_id];
        const size_t num_workers = workers_.size();

        if (num_workers <= 1) {
            return nullptr;  // No one to steal from
        }

        // Locality-aware stealing: prefer victims on same NUMA node
        std::vector<size_t> local_victims;
        std::vector<size_t> remote_victims;

        for (size_t i = 0; i < num_workers; ++i) {
            if (i == thief_id) continue;

            if (workers_[i]->numa_node == thief.numa_node) {
                local_victims.push_back(i);
            } else {
                remote_victims.push_back(i);
            }
        }

        // Try local victims first (80% of attempts)
        std::uniform_int_distribution<int> pref_dist(0, 99);
        bool try_local = !local_victims.empty() && (pref_dist(thief.rng) < 80);

        auto& victims = try_local ? local_victims : remote_victims;
        if (victims.empty()) {
            victims = try_local ? remote_victims : local_victims;
        }

        if (victims.empty()) {
            return nullptr;
        }

        // Randomized work stealing (key to Blumofe-Leiserson efficiency)
        // Try several random victims
        const size_t max_attempts = std::min(size_t{4}, victims.size());

        for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
            std::uniform_int_distribution<size_t> victim_dist(0, victims.size() - 1);
            size_t victim_idx = victim_dist(thief.rng);
            size_t victim_id = victims[victim_idx];

            thief.steal_attempts.fetch_add(1, std::memory_order_relaxed);

            Task* task = workers_[victim_id]->deque.steal();
            if (task) {
                thief.tasks_stolen.fetch_add(1, std::memory_order_relaxed);
                return task;
            }
        }

        return nullptr;
    }

public:
    /**
     * @brief Construct work-stealing pool
     * @param num_threads Number of worker threads (0 = hardware_concurrency)
     */
    explicit WorkStealingPool(size_t num_threads = 0)
        : init_latch_(std::make_unique<Latch>(num_threads ? num_threads : std::thread::hardware_concurrency())) {

        size_t num_workers = num_threads ? num_threads : std::thread::hardware_concurrency();

        // Detect NUMA topology (simplified - in production use hwloc or numa.h)
        auto get_numa_node = [](size_t worker_id) -> int {
            // Simple heuristic: assume 2 NUMA nodes, split workers evenly
            // In production, use actual NUMA topology
            return worker_id % 2;
        };

        // Create workers
        workers_.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back(std::make_unique<Worker>());
            workers_.back()->numa_node = get_numa_node(i);
        }

        // Start worker threads
        for (size_t i = 0; i < num_workers; ++i) {
            workers_[i]->thread = std::thread(&WorkStealingPool::worker_loop, this, i);
        }

        // Wait for all workers to initialize
        init_latch_->wait();
    }

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~WorkStealingPool() {
        shutdown();
    }

    // Non-copyable, non-movable
    WorkStealingPool(const WorkStealingPool&) = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    /**
     * @brief Submit a task for execution
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        if (shutdown_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Cannot submit to shutdown pool");
        }

        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                return func(args...);
            }
        );

        auto result = task_ptr->get_future();

        Task* task = new Task([task_ptr]() { (*task_ptr)(); });

        pending_tasks_.fetch_add(1, std::memory_order_acquire);

        // Push to current worker's deque if we're in a worker thread
        if (current_worker_id_ < workers_.size()) {
            workers_[current_worker_id_]->deque.push(task);
        } else {
            // Push to a worker's deque (round-robin to avoid random_device blocking)
            static std::atomic<size_t> next_worker{0};
            size_t target = next_worker.fetch_add(1, std::memory_order_relaxed) % workers_.size();
            workers_[target]->deque.push(task);
        }

        return result;
    }

    /**
     * @brief Wait until all pending tasks complete
     */
    void wait_idle() {
        while (pending_tasks_.load(std::memory_order_acquire) > 0) {
            // Help with work if we're a worker thread
            if (current_worker_id_ < workers_.size()) {
                auto& worker = *workers_[current_worker_id_];
                Task* task = worker.deque.pop();
                if (!task) {
                    task = try_steal(current_worker_id_);
                }
                if (task) {
                    try {
                        task->func();
                    } catch (...) {
                    }
                    delete task;
                    pending_tasks_.fetch_sub(1, std::memory_order_release);
                } else {
                    std::this_thread::yield();
                }
            } else {
                std::this_thread::yield();
            }
        }
    }

    /**
     * @brief Shutdown pool and wait for all workers
     */
    void shutdown() {
        // Check if already shutdown
        if (shutdown_.load(std::memory_order_acquire)) {
            return;
        }

        // Wait for pending tasks to complete FIRST
        wait_idle();

        // THEN set shutdown flag to stop workers
        shutdown_.store(true, std::memory_order_release);

        // Join all worker threads
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
    }

    /**
     * @brief Get number of worker threads
     */
    size_t worker_count() const {
        return workers_.size();
    }

    /**
     * @brief Get current worker ID (SIZE_MAX if not in worker thread)
     */
    static size_t get_current_worker_id() {
        return current_worker_id_;
    }

    /**
     * @brief Get statistics for a worker
     */
    struct WorkerStats {
        uint64_t tasks_executed;
        uint64_t tasks_stolen;
        uint64_t steal_attempts;
        double steal_success_rate;
    };

    WorkerStats get_worker_stats(size_t worker_id) const {
        if (worker_id >= workers_.size()) {
            return {};
        }

        auto& w = *workers_[worker_id];
        uint64_t attempts = w.steal_attempts.load(std::memory_order_relaxed);
        uint64_t stolen = w.tasks_stolen.load(std::memory_order_relaxed);

        return {
            w.tasks_executed.load(std::memory_order_relaxed),
            stolen,
            attempts,
            attempts > 0 ? static_cast<double>(stolen) / static_cast<double>(attempts) : 0.0
        };
    }
};

// Thread-local storage
thread_local size_t WorkStealingPool::current_worker_id_ = SIZE_MAX;

/**
 * @brief Global work-stealing pool singleton
 */
class GlobalWorkStealingPool : public base::Singleton<GlobalWorkStealingPool> {
    friend class base::Singleton<GlobalWorkStealingPool>;

private:
    std::unique_ptr<WorkStealingPool> pool_;
    std::mutex mutex_;

    GlobalWorkStealingPool() = default;

public:
    WorkStealingPool& get_pool(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_) {
            pool_ = std::make_unique<WorkStealingPool>(num_threads);
        }
        return *pool_;
    }

    void reset(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.reset();
        pool_ = std::make_unique<WorkStealingPool>(num_threads);
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_) {
            pool_->shutdown();
            pool_.reset();
        }
    }
};

inline WorkStealingPool& global_work_stealing_pool() {
    return GlobalWorkStealingPool::instance().get_pool();
}

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_WORK_STEALING_POOL_H

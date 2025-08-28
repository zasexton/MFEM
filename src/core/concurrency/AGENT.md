# Core Concurrency - AGENT.md

## Purpose
The `concurrency/` layer provides comprehensive threading and parallelism infrastructure including thread pools, task scheduling, synchronization primitives, and parallel algorithms. It enables efficient multi-core utilization while maintaining thread safety and avoiding common concurrency pitfalls.

## Architecture Philosophy
- **Task-based parallelism**: Think in tasks, not threads
- **Lock-free when possible**: Minimize contention
- **RAII synchronization**: Automatic lock management
- **Work stealing**: Efficient load balancing
- **Composable abstractions**: Build complex patterns from simple primitives

## Files Overview

### Thread Management
```cpp
thread_pool.hpp        // Basic thread pool implementation
work_stealing_pool.hpp // Work-stealing thread pool
thread_local.hpp       // Thread-local storage utilities
thread_affinity.hpp    // CPU affinity management
thread_priority.hpp    // Thread priority control
```

### Task System
```cpp
task.hpp              // Task abstraction
task_graph.hpp        // DAG task dependencies
task_scheduler.hpp    // Task scheduling strategies
continuation.hpp      // Task continuations
async.hpp            // Async task execution
```

### Synchronization
```cpp
mutex.hpp            // Enhanced mutex types
spinlock.hpp         // Spinlock implementations
rwlock.hpp           // Reader-writer locks
condition_var.hpp    // Condition variables
barrier.hpp          // Thread barriers
latch.hpp           // One-time synchronization
semaphore.hpp       // Counting semaphores
event.hpp           // Manual/auto reset events
```

### Lock-Free Structures
```cpp
atomic_queue.hpp     // Lock-free MPMC queue
atomic_stack.hpp     // Lock-free stack
hazard_pointer.hpp   // Safe memory reclamation
rcu.hpp             // Read-copy-update
seqlock.hpp         // Sequence locks
```

### Futures and Promises
```cpp
future.hpp          // Enhanced futures
promise.hpp         // Enhanced promises
shared_future.hpp   // Shared futures
packaged_task.hpp   // Packaged tasks
when_all.hpp        // Wait for multiple futures
when_any.hpp        // Wait for any future
```

### Parallel Algorithms
```cpp
parallel_for.hpp     // Parallel loops
parallel_reduce.hpp  // Parallel reduction
parallel_scan.hpp    // Parallel prefix sum
parallel_sort.hpp    // Parallel sorting
parallel_pipeline.hpp // Pipeline parallelism
```

### Utilities
```cpp
thread_safe.hpp      // Thread-safe wrapper
concurrent_hash_map.hpp // Thread-safe hash map
object_pool_mt.hpp   // Thread-safe object pool
mpsc_queue.hpp      // Multi-producer single-consumer
spsc_queue.hpp      // Single-producer single-consumer
```

## Detailed Component Specifications

### `thread_pool.hpp`
```cpp
class ThreadPool {
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
    
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    // Submit task
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        auto result = task->get_future();
        {
            std::lock_guard lock(queue_mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        
        return result;
    }
    
    // Bulk submission
    template<typename F>
    void submit_bulk(F&& f, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            submit(f, i);
        }
    }
    
    // Control
    void pause();
    void resume();
    void wait_idle();
    void stop();
    
    // Info
    size_t thread_count() const { return threads_.size(); }
    size_t queue_size() const;
    size_t active_tasks() const { return active_tasks_; }
};

// Global thread pool
ThreadPool& global_thread_pool();
```
**Why necessary**: Efficient thread reuse, controlled parallelism, task queuing.
**Usage**: Background processing, parallel algorithms, async operations.

### `task.hpp`
```cpp
template<typename T>
class Task {
public:
    using result_type = T;
    
    enum class Status {
        Pending,
        Running,
        Completed,
        Failed,
        Cancelled
    };
    
private:
    std::function<T()> func_;
    std::promise<T> promise_;
    std::shared_future<T> future_;
    std::atomic<Status> status_{Status::Pending};
    std::vector<std::function<void()>> continuations_;
    
public:
    explicit Task(std::function<T()> f);
    
    // Execution
    void execute();
    void cancel();
    
    // Results
    T get() { return future_.get(); }
    std::shared_future<T> get_future() { return future_; }
    
    // Status
    Status status() const { return status_; }
    bool is_ready() const { return status_ == Status::Completed; }
    
    // Continuations
    template<typename F>
    auto then(F&& f) -> Task<decltype(f(std::declval<T>()))>;
    
    template<typename F>
    auto then_on(ThreadPool& pool, F&& f);
    
    // Composition
    template<typename U>
    static Task<std::pair<T, U>> when_all(Task<T>& t1, Task<U>& t2);
    
    template<typename U>
    static Task<std::variant<T, U>> when_any(Task<T>& t1, Task<U>& t2);
};

// Task creation helpers
template<typename F>
auto make_task(F&& f) -> Task<decltype(f())>;

template<typename F>
auto async_task(F&& f) -> Task<decltype(f())>;
```
**Why necessary**: Task abstraction, composable async operations, continuation support.
**Usage**: Async workflows, task dependencies, parallel pipelines.

### `parallel_for.hpp`
```cpp
template<typename IndexType, typename F>
void parallel_for(IndexType begin, IndexType end, F&& func) {
    const auto num_threads = std::thread::hardware_concurrency();
    const auto chunk_size = (end - begin) / num_threads;
    
    std::vector<std::future<void>> futures;
    
    for (size_t t = 0; t < num_threads; ++t) {
        IndexType chunk_begin = begin + t * chunk_size;
        IndexType chunk_end = (t == num_threads - 1) ? end : chunk_begin + chunk_size;
        
        futures.push_back(
            std::async(std::launch::async, [=, &func]() {
                for (IndexType i = chunk_begin; i < chunk_end; ++i) {
                    func(i);
                }
            })
        );
    }
    
    for (auto& f : futures) {
        f.wait();
    }
}

// 2D parallel for
template<typename F>
void parallel_for_2d(size_t rows, size_t cols, F&& func) {
    parallel_for(size_t(0), rows, [=, &func](size_t row) {
        for (size_t col = 0; col < cols; ++col) {
            func(row, col);
        }
    });
}

// Parallel for with reduction
template<typename IndexType, typename T, typename F, typename R>
T parallel_for_reduce(IndexType begin, IndexType end, T init, F&& func, R&& reduce) {
    const auto num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<T>> futures;
    
    // ... partition work and reduce results
    
    return final_result;
}

// Chunked parallel for
template<typename Container, typename F>
void parallel_for_each(Container& container, F&& func) {
    parallel_for(size_t(0), container.size(), 
        [&](size_t i) { func(container[i]); });
}
```
**Why necessary**: Simple parallelization of loops, automatic work distribution.
**Usage**: Array processing, matrix operations, independent computations.

### `atomic_queue.hpp`
```cpp
template<typename T>
class AtomicQueue {
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    alignas(64) std::atomic<Node*> head_;
    alignas(64) std::atomic<Node*> tail_;
    
public:
    AtomicQueue();
    ~AtomicQueue();
    
    void enqueue(T item) {
        Node* new_node = new Node;
        T* data = new T(std::move(item));
        new_node->data.store(data);
        
        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
    }
    
    std::optional<T> dequeue() {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return std::nullopt;
        }
        
        T* data = next->data.exchange(nullptr);
        head_.store(next);
        delete head;
        
        T result = std::move(*data);
        delete data;
        return result;
    }
    
    bool empty() const {
        return head_.load()->next.load() == nullptr;
    }
};

// Bounded MPMC queue
template<typename T, size_t Capacity>
class BoundedQueue {
    std::array<std::atomic<T*>, Capacity> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    
public:
    bool try_enqueue(T item);
    std::optional<T> try_dequeue();
    bool empty() const;
    bool full() const;
};
```
**Why necessary**: Lock-free communication between threads, high-performance queuing.
**Usage**: Message passing, work queues, producer-consumer patterns.

### `rcu.hpp`
```cpp
template<typename T>
class RCU {
    struct Version {
        std::shared_ptr<T> data;
        std::atomic<size_t> readers{0};
    };
    
    std::atomic<Version*> current_;
    std::vector<std::unique_ptr<Version>> old_versions_;
    std::mutex cleanup_mutex_;
    
public:
    class ReadGuard {
        Version* version_;
    public:
        explicit ReadGuard(Version* v);
        ~ReadGuard();
        const T& operator*() const { return *version_->data; }
        const T* operator->() const { return version_->data.get(); }
    };
    
    RCU(T initial);
    
    ReadGuard read() {
        Version* v = current_.load();
        v->readers.fetch_add(1);
        return ReadGuard(v);
    }
    
    void update(std::function<T(const T&)> updater) {
        auto old = current_.load();
        auto new_version = std::make_unique<Version>();
        new_version->data = std::make_shared<T>(updater(*old->data));
        
        current_.store(new_version.get());
        
        // Schedule old version for cleanup
        std::lock_guard lock(cleanup_mutex_);
        old_versions_.push_back(std::move(new_version));
        cleanup_old_versions();
    }
    
private:
    void cleanup_old_versions();
};
```
**Why necessary**: Read-heavy concurrent data structures, minimal reader overhead.
**Usage**: Configuration updates, routing tables, caches.

### `work_stealing_pool.hpp`
```cpp
class WorkStealingPool {
    struct Worker {
        std::deque<std::function<void()>> local_queue;
        std::mutex mutex;
        std::condition_variable cv;
        std::thread thread;
        std::atomic<bool> active{true};
    };
    
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<size_t> next_worker_{0};
    
public:
    explicit WorkStealingPool(size_t num_threads);
    
    template<typename F>
    void submit(F&& task) {
        size_t worker_id = next_worker_.fetch_add(1) % workers_.size();
        
        {
            std::lock_guard lock(workers_[worker_id]->mutex);
            workers_[worker_id]->local_queue.push_back(std::forward<F>(task));
        }
        workers_[worker_id]->cv.notify_one();
    }
    
    void worker_loop(size_t id) {
        auto& worker = *workers_[id];
        
        while (worker.active) {
            std::function<void()> task;
            
            // Try to get task from local queue
            {
                std::unique_lock lock(worker.mutex);
                if (!worker.local_queue.empty()) {
                    task = std::move(worker.local_queue.front());
                    worker.local_queue.pop_front();
                }
            }
            
            // Try to steal from other workers
            if (!task) {
                task = try_steal(id);
            }
            
            if (task) {
                task();
            } else {
                std::unique_lock lock(worker.mutex);
                worker.cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }
    
private:
    std::function<void()> try_steal(size_t thief_id);
};
```
**Why necessary**: Better load balancing, reduces idle threads, improves throughput.
**Usage**: Recursive algorithms, irregular workloads, task-based parallelism.

### `barrier.hpp`
```cpp
class Barrier {
    const size_t threshold_;
    std::atomic<size_t> count_;
    std::atomic<size_t> generation_;
    
public:
    explicit Barrier(size_t count) 
        : threshold_(count), count_(count), generation_(0) {}
    
    void arrive_and_wait() {
        auto gen = generation_.load();
        
        if (count_.fetch_sub(1) == 1) {
            // Last thread to arrive
            count_.store(threshold_);
            generation_.fetch_add(1);
        } else {
            // Wait for generation to change
            while (generation_.load() == gen) {
                std::this_thread::yield();
            }
        }
    }
    
    void arrive_and_drop() {
        count_.fetch_sub(1);
        threshold_.fetch_sub(1);
    }
};

// Flexible barrier with completion function
class FlexBarrier {
    std::function<void()> completion_func_;
public:
    FlexBarrier(size_t count, std::function<void()> on_completion);
    void arrive_and_wait();
};
```
**Why necessary**: Synchronization points, phased algorithms, parallel stages.
**Usage**: Parallel simulations, iterative algorithms, multi-stage processing.

## Concurrency Patterns

### Producer-Consumer
```cpp
AtomicQueue<Task> task_queue;

// Producer threads
void producer() {
    while (producing) {
        auto task = generate_task();
        task_queue.enqueue(std::move(task));
    }
}

// Consumer threads
void consumer() {
    while (consuming) {
        if (auto task = task_queue.dequeue()) {
            process(*task);
        }
    }
}
```

### Fork-Join
```cpp
template<typename T>
T parallel_recursive(T begin, T end) {
    if (end - begin < threshold) {
        return sequential_process(begin, end);
    }
    
    T mid = begin + (end - begin) / 2;
    
    auto f1 = async_task([=] { return parallel_recursive(begin, mid); });
    auto f2 = async_task([=] { return parallel_recursive(mid, end); });
    
    return combine(f1.get(), f2.get());
}
```

### Pipeline
```cpp
Pipeline<int, std::string> pipeline;

pipeline.add_stage([](int x) { return x * 2; });
pipeline.add_stage([](int x) { return x + 10; });
pipeline.add_stage([](int x) { return std::to_string(x); });

auto result = pipeline.process(5);  // "20"
```

## Performance Considerations

- **False sharing**: Align to cache lines (64 bytes)
- **Lock contention**: Use lock-free structures where possible
- **Thread pool size**: Usually hardware_concurrency()
- **Task granularity**: Balance between overhead and parallelism
- **Memory ordering**: Use relaxed ordering when safe

## Testing Strategy

- **Race conditions**: ThreadSanitizer, stress testing
- **Deadlocks**: Lock ordering, timeout detection
- **Performance**: Scaling tests, contention analysis
- **Correctness**: Linearizability testing
- **Edge cases**: Thread exhaustion, memory barriers

## Usage Guidelines

1. **Task over thread**: Think in terms of tasks, not threads
2. **Avoid shared state**: Use message passing when possible
3. **RAII locks**: Always use lock_guard/unique_lock
4. **Immutable data**: Prefer immutable structures for concurrent access
5. **Profile first**: Measure before optimizing parallelism

## Anti-patterns to Avoid

- Manual thread management
- Busy waiting without backoff
- Fine-grained locking
- Shared mutable state
- Ignoring false sharing

## Dependencies
- `base/` - For Object patterns
- `memory/` - For allocators
- `error/` - For error handling
- Standard library (C++20)

## Future Enhancements
- Coroutine support
- GPU task scheduling
- Distributed computing abstractions
- Transactional memory
- Deterministic parallelism
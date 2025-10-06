#ifndef CORE_CONCURRENCY_TASK_GRAPH_H_
#define CORE_CONCURRENCY_TASK_GRAPH_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "task.h"
#include "thread_pool.h"

namespace fem {
namespace core {
namespace concurrency {

/**
 * @brief Exception thrown when a cycle is detected in the task graph
 */
class CyclicDependencyException : public std::runtime_error {
public:
    explicit CyclicDependencyException(const std::string& msg)
        : std::runtime_error("Cyclic dependency detected: " + msg) {}
};

/**
 * @brief Exception thrown when a task is not found
 */
class TaskNotFoundException : public std::runtime_error {
public:
    explicit TaskNotFoundException(const std::string& task_id)
        : std::runtime_error("Task not found: " + task_id) {}
};

/**
 * @brief Exception thrown when graph validation fails
 */
class InvalidGraphException : public std::runtime_error {
public:
    explicit InvalidGraphException(const std::string& msg)
        : std::runtime_error("Invalid graph: " + msg) {}
};

/**
 * @brief Node in the task graph representing a single task
 */
template<typename T>
class TaskNode : public std::enable_shared_from_this<TaskNode<T>> {
    template<typename U> friend class TaskGraph;

public:
    using TaskType = std::function<T()>;
    using NodePtr = std::shared_ptr<TaskNode<T>>;
    using WeakNodePtr = std::weak_ptr<TaskNode<T>>;

    enum class Status {
        Pending,      // Not yet ready to run
        Ready,        // All dependencies satisfied, ready to run
        Running,      // Currently executing
        Completed,    // Successfully completed
        Failed,       // Failed with error
        Cancelled     // Cancelled
    };

private:
    std::string id_;
    TaskType task_;
    std::shared_ptr<Task<T>> task_wrapper_;

    // Dependencies
    std::vector<WeakNodePtr> dependencies_;
    std::vector<WeakNodePtr> dependents_;
    std::atomic<size_t> pending_dependencies_{0};

    // Status tracking
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<Status> status_{Status::Pending};
    // Scheduling marker to avoid double-enqueue in parallel execution
    std::atomic<bool> enqueued_{false};

    // Results
    std::conditional_t<std::is_void_v<T>, std::monostate, std::optional<T>> result_;
    std::exception_ptr exception_;

    // Metadata
    int priority_{0};
    std::chrono::steady_clock::time_point created_time_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;

public:
    /**
     * @brief Construct a task node
     * @param id Unique identifier for the task
     * @param task The function to execute
     * @param priority Execution priority (higher = earlier)
     */
    TaskNode(const std::string& id, TaskType task, int priority = 0)
        : id_(id),
          task_(std::move(task)),
          priority_(priority),
          created_time_(std::chrono::steady_clock::now()) {
    }

    /**
     * @brief Get the task ID
     */
    const std::string& id() const { return id_; }

    /**
     * @brief Get the priority
     */
    int priority() const { return priority_; }

    /**
     * @brief Set the priority
     */
    void set_priority(int p) { priority_ = p; }

    /**
     * @brief Get current status
     */
    Status status() const { return status_.load(); }

    /**
     * @brief Check if task is ready to run
     */
    bool is_ready() const {
        return status_.load() == Status::Ready;
    }

    /**
     * @brief Check if task has completed
     */
    bool is_completed() const {
        auto s = status_.load();
        return s == Status::Completed || s == Status::Failed || s == Status::Cancelled;
    }

    /**
     * @brief Add a dependency
     */
    void add_dependency(NodePtr dep) {
        if (!dep) return;

        std::lock_guard<std::mutex> lock(mutex_);
        dependencies_.push_back(dep);
        dep->add_dependent(this->shared_from_this());
        pending_dependencies_.fetch_add(1);

        // If dependency is already complete, decrement counter
        if (dep->is_completed()) {
            notify_dependency_complete();
        }
    }

    /**
     * @brief Add a dependent node
     */
    void add_dependent(NodePtr dep) {
        if (!dep) return;
        dependents_.push_back(dep);
    }

    /**
     * @brief Get dependencies
     */
    std::vector<NodePtr> get_dependencies() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<NodePtr> result;
        for (const auto& weak : dependencies_) {
            if (auto dep = weak.lock()) {
                result.push_back(dep);
            }
        }
        return result;
    }

    /**
     * @brief Get dependents
     */
    std::vector<NodePtr> get_dependents() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<NodePtr> result;
        for (const auto& weak : dependents_) {
            if (auto dep = weak.lock()) {
                result.push_back(dep);
            }
        }
        return result;
    }

    /**
     * @brief Notify that a dependency has completed
     */
    void notify_dependency_complete() {
        size_t remaining = pending_dependencies_.fetch_sub(1) - 1;
        if (remaining == 0 && status_.load() == Status::Pending) {
            status_ = Status::Ready;
            cv_.notify_all();
        }
    }

    /**
     * @brief Execute the task
     * @param pool Thread pool to execute on (optional)
     */
    void execute([[maybe_unused]] ThreadPool* pool = nullptr) {
        Status expected = Status::Ready;
        if (!status_.compare_exchange_strong(expected, Status::Running)) {
            if (expected != Status::Completed) {
                throw std::runtime_error("Task not ready or already executed: " + id_);
            }
            return; // Already completed
        }

        start_time_ = std::chrono::steady_clock::now();

        try {
            if constexpr (std::is_void_v<T>) {
                task_();
            } else {
                result_ = task_();
            }
            status_ = Status::Completed;
        } catch (...) {
            exception_ = std::current_exception();
            status_ = Status::Failed;
        }

        end_time_ = std::chrono::steady_clock::now();

        // Notify dependents only on successful completion
        if (status_.load() == Status::Completed) {
            for (const auto& weak : dependents_) {
                if (auto dep = weak.lock()) {
                    dep->notify_dependency_complete();
                }
            }
        }

        cv_.notify_all();
    }

    /**
     * @brief Cancel the task
     */
    void cancel() {
        Status expected = Status::Pending;
        if (status_.compare_exchange_strong(expected, Status::Cancelled)) {
            cv_.notify_all();

            // Notify dependents
            for (const auto& weak : dependents_) {
                if (auto dep = weak.lock()) {
                    dep->cancel();
                }
            }
        }
    }

    /**
     * @brief Wait for task completion
     */
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return is_completed(); });
    }

    /**
     * @brief Wait for task completion with timeout
     */
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return is_completed(); });
    }

    /**
     * @brief Get the result (blocks until complete)
     */
    T get() {
        wait();

        if (status_.load() == Status::Failed && exception_) {
            std::rethrow_exception(exception_);
        }

        if (status_.load() == Status::Cancelled) {
            throw TaskCancelledException();
        }

        if constexpr (!std::is_void_v<T>) {
            if (!result_.has_value()) {
                throw std::runtime_error("Task result not available");
            }
            return result_.value();
        }
    }

    /**
     * @brief Get execution duration
     */
    std::chrono::nanoseconds duration() const {
        if (start_time_ == std::chrono::steady_clock::time_point{} ||
            end_time_ == std::chrono::steady_clock::time_point{}) {
            return std::chrono::nanoseconds::zero();
        }
        return end_time_ - start_time_;
    }

};

/**
 * @brief DAG-based task graph for managing task dependencies and execution
 */
template<typename T = void>
class TaskGraph {
public:
    using NodePtr = std::shared_ptr<TaskNode<T>>;
    using TaskType = typename TaskNode<T>::TaskType;

private:
    // Graph structure
    std::unordered_map<std::string, NodePtr> nodes_;
    std::vector<NodePtr> roots_;  // Nodes with no dependencies
    std::vector<NodePtr> leaves_; // Nodes with no dependents

    // Synchronization
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Execution state
    std::atomic<bool> is_executing_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<size_t> completed_count_{0};

    // Thread pool for execution
    ThreadPool* thread_pool_{nullptr};

    // Statistics
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;

public:
    /**
     * @brief Constructor
     * @param pool Thread pool to use for execution (optional)
     */
    explicit TaskGraph(ThreadPool* pool = nullptr)
        : thread_pool_(pool) {
    }

    /**
     * @brief Add a task to the graph
     * @param id Unique identifier for the task
     * @param task The function to execute
     * @param priority Execution priority
     * @return The created task node
     */
    NodePtr add_task(const std::string& id, TaskType task, int priority = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (nodes_.find(id) != nodes_.end()) {
            throw std::invalid_argument("Task already exists: " + id);
        }

        auto node = std::make_shared<TaskNode<T>>(id, std::move(task), priority);
        nodes_[id] = node;
        return node;
    }

    /**
     * @brief Add a dependency between tasks
     * @param dependent_id The task that depends on another
     * @param dependency_id The task that must complete first
     */
    void add_dependency(const std::string& dependent_id, const std::string& dependency_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto dependent_it = nodes_.find(dependent_id);
        auto dependency_it = nodes_.find(dependency_id);

        if (dependent_it == nodes_.end()) {
            throw TaskNotFoundException(dependent_id);
        }
        if (dependency_it == nodes_.end()) {
            throw TaskNotFoundException(dependency_id);
        }

        dependent_it->second->add_dependency(dependency_it->second);
    }

    /**
     * @brief Add multiple dependencies
     * @param from_id The dependent task ID
     * @param to_ids List of dependency task IDs
     */
    void add_dependencies(const std::string& from_id,
                          const std::vector<std::string>& to_ids) {
        for (const auto& to_id : to_ids) {
            add_dependency(from_id, to_id);
        }
    }

    /**
     * @brief Get a task node by ID
     */
    NodePtr get_task(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? it->second : nullptr;
    }

    /**
     * @brief Remove a task from the graph
     */
    void remove_task(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            throw TaskNotFoundException(id);
        }

        // Remove from dependencies and dependents
        auto node = it->second;
        for ([[maybe_unused]] auto& dep : node->get_dependencies()) {
            // Remove this node from dep's dependents
        }
        for ([[maybe_unused]] auto& dep : node->get_dependents()) {
            // Remove this node from dep's dependencies
        }

        nodes_.erase(it);
    }

    /**
     * @brief Clear all tasks
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        nodes_.clear();
        roots_.clear();
        leaves_.clear();
        completed_count_ = 0;
        cancel_requested_ = false;
    }

    /**
     * @brief Get the number of tasks
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.size();
    }

    /**
     * @brief Check if the graph is empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.empty();
    }

    /**
     * @brief Validate the graph (check for cycles)
     * @return true if valid (no cycles), false otherwise
     */
    bool validate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !has_cycle();
    }

    /**
     * @brief Detect if there's a cycle in the graph
     * @return true if cycle exists
     */
    bool has_cycle() const {
        std::unordered_set<std::string> visited;
        std::unordered_set<std::string> recursion_stack;

        for (const auto& [id, node] : nodes_) {
            if (visited.find(id) == visited.end()) {
                if (has_cycle_util(node, visited, recursion_stack)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Get topological order of tasks
     * @return Vector of task IDs in topological order
     * @throws CyclicDependencyException if cycle detected
     */
    std::vector<std::string> topological_sort() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (has_cycle()) {
            throw CyclicDependencyException("Cannot sort graph with cycles");
        }

        std::vector<std::string> result;
        std::unordered_set<std::string> visited;

        for (const auto& [id, node] : nodes_) {
            if (visited.find(id) == visited.end()) {
                topological_sort_util(node, visited, result);
            }
        }

        // We traverse dependencies first and push node afterward,
        // so 'result' already has dependencies before dependents.
        return result;
    }

    /**
     * @brief Execute the task graph
     * @param parallel If true, execute independent tasks in parallel
     */
    void execute(bool parallel = true) {
        if (is_executing_.exchange(true)) {
            throw std::runtime_error("Graph is already executing");
        }

        try {
            if (!validate()) {
                throw CyclicDependencyException("Cannot execute graph with cycles");
            }

            identify_roots_and_leaves();

            if (cancel_requested_.load()) {
                throw std::runtime_error("Graph execution cancelled");
            }

            start_time_ = std::chrono::steady_clock::now();
            completed_count_ = 0;

            if (parallel) {
                execute_parallel();
            } else {
                execute_sequential();
            }

            end_time_ = std::chrono::steady_clock::now();

        } catch (...) {
            is_executing_ = false;
            throw;
        }

        is_executing_ = false;
    }

    /**
     * @brief Execute the graph asynchronously
     * @return Future that completes when all tasks are done
     */
    std::future<void> execute_async(bool parallel = true) {
        return std::async(std::launch::async, [this, parallel] {
            execute(parallel);
        });
    }

    /**
     * @brief Cancel execution
     */
    void cancel() {
        cancel_requested_ = true;

        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, node] : nodes_) {
            node->cancel();
        }
        cv_.notify_all();
    }

    /**
     * @brief Wait for all tasks to complete
     */
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return completed_count_.load() == nodes_.size() ||
                   cancel_requested_.load();
        });
    }

    /**
     * @brief Wait with timeout
     */
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] {
            return completed_count_.load() == nodes_.size() ||
                   cancel_requested_.load();
        });
    }

    /**
     * @brief Get execution statistics
     */
    struct Stats {
        size_t total_tasks;
        size_t completed_tasks;
        size_t failed_tasks;
        size_t cancelled_tasks;
        std::chrono::nanoseconds total_duration;
        std::chrono::nanoseconds average_task_duration;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        Stats stats{};
        stats.total_tasks = nodes_.size();

        std::chrono::nanoseconds total_task_time{0};

        for (const auto& [id, node] : nodes_) {
            switch (node->status()) {
                case TaskNode<T>::Status::Completed:
                    stats.completed_tasks++;
                    break;
                case TaskNode<T>::Status::Failed:
                    stats.failed_tasks++;
                    break;
                case TaskNode<T>::Status::Cancelled:
                    stats.cancelled_tasks++;
                    break;
                default:
                    break;
            }
            total_task_time += node->duration();
        }

        if (start_time_ != std::chrono::steady_clock::time_point{} &&
            end_time_ != std::chrono::steady_clock::time_point{}) {
            stats.total_duration = end_time_ - start_time_;
        }

        if (stats.completed_tasks > 0) {
            stats.average_task_duration = total_task_time / stats.completed_tasks;
        }

        return stats;
    }

    /**
     * @brief Get root tasks (no dependencies)
     */
    std::vector<NodePtr> get_roots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return roots_;
    }

    /**
     * @brief Get leaf tasks (no dependents)
     */
    std::vector<NodePtr> get_leaves() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return leaves_;
    }

    /**
     * @brief Set the thread pool for execution
     */
    void set_thread_pool(ThreadPool* pool) {
        thread_pool_ = pool;
    }

private:
    /**
     * @brief Identify root and leaf nodes
     */
    void identify_roots_and_leaves() {
        roots_.clear();
        leaves_.clear();

        for (auto& [id, node] : nodes_) {
            if (node->get_dependencies().empty()) {
                roots_.push_back(node);
                // Only set to Ready if it hasn't been executed yet
                if (node->pending_dependencies_.load() == 0 &&
                    node->status_.load() == TaskNode<T>::Status::Pending) {
                    node->status_ = TaskNode<T>::Status::Ready;
                }
            }
            if (node->get_dependents().empty()) {
                leaves_.push_back(node);
            }
            // Reset enqueue marker before execution begins
            node->enqueued_ = false;
        }
    }

    /**
     * @brief Execute tasks in parallel
     */
    void execute_parallel() {
        // Priority queue for ready tasks
        auto cmp = [](const NodePtr& a, const NodePtr& b) {
            return a->priority() < b->priority();
        };
        std::priority_queue<NodePtr, std::vector<NodePtr>, decltype(cmp)> ready_queue(cmp);

        // Local scheduling state
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        // Tracks tasks scheduled and not yet finished (includes not-yet-started)
        std::atomic<size_t> inflight_tasks{0};

        // Add initial ready tasks
        {
            std::lock_guard<std::mutex> lock_global(mutex_);
            std::lock_guard<std::mutex> lock_queue(queue_mutex);
            for (auto& [id, node] : nodes_) {
                if (node->is_ready()) {
                    bool expected = false;
                    if (node->enqueued_.compare_exchange_strong(expected, true)) {
                        ready_queue.push(node);
                    }
                }
            }
        }

        // Worker function
        auto worker = [&](NodePtr node) {
            
            try {
                if (!cancel_requested_.load()) {
                    node->execute(nullptr);
                }
            } catch (...) {
                // Exception already captured in node
            }

            // Find newly ready tasks
            if (!cancel_requested_.load()) {
                for (auto& dep : node->get_dependents()) {
                    if (dep->is_ready()) {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        bool expected = false;
                        if (dep->enqueued_.compare_exchange_strong(expected, true)) {
                            ready_queue.push(dep);
                            queue_cv.notify_one();
                        }
                    }
                }
            }

            completed_count_.fetch_add(1);
            cv_.notify_all();
            inflight_tasks.fetch_sub(1);
            queue_cv.notify_one();
        };

        // Execute tasks
        while (true) {
            NodePtr next_task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] {
                    return !ready_queue.empty() || inflight_tasks.load() == 0 || cancel_requested_.load();
                });

                if (!ready_queue.empty()) {
                    next_task = ready_queue.top();
                    ready_queue.pop();
                }
            }

            if (next_task) {
                inflight_tasks.fetch_add(1);
                if (thread_pool_) {
                    thread_pool_->submit(worker, next_task);
                } else {
                    std::thread(worker, next_task).detach();
                }
            } else if (inflight_tasks.load() == 0) {
                break; // All done
            } // else: canceled with tasks still finishing; wait again
        }

        // Wait for all active tasks to complete
        while (active_tasks.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
     * @brief Execute tasks sequentially in topological order
     */
    void execute_sequential() {
        auto order = topological_sort();

        for (const auto& id : order) {
            if (cancel_requested_.load()) {
                break;
            }

            auto node = nodes_[id];
            // Only execute nodes that are actually ready
            if (!node->is_ready()) {
                // Root nodes become ready in identify_roots_and_leaves().
                // Dependents become ready when their prerequisites complete successfully.
                // If a node isn't ready (due to failed deps), skip it.
                continue;
            }
            node->execute(nullptr);
            completed_count_.fetch_add(1);
            cv_.notify_all();
        }
    }

    /**
     * @brief Utility for cycle detection
     */
    bool has_cycle_util(const NodePtr& node,
                        std::unordered_set<std::string>& visited,
                        std::unordered_set<std::string>& recursion_stack) const {
        visited.insert(node->id());
        recursion_stack.insert(node->id());

        for (auto& dep : node->get_dependencies()) {
            if (recursion_stack.find(dep->id()) != recursion_stack.end()) {
                return true; // Back edge found
            }
            if (visited.find(dep->id()) == visited.end()) {
                if (has_cycle_util(dep, visited, recursion_stack)) {
                    return true;
                }
            }
        }

        recursion_stack.erase(node->id());
        return false;
    }

    /**
     * @brief Utility for topological sort
     */
    void topological_sort_util(const NodePtr& node,
                               std::unordered_set<std::string>& visited,
                               std::vector<std::string>& result) const {
        visited.insert(node->id());

        for (auto& dep : node->get_dependencies()) {
            if (visited.find(dep->id()) == visited.end()) {
                topological_sort_util(dep, visited, result);
            }
        }

        result.push_back(node->id());
    }
};

// Helper functions

/**
 * @brief Create a simple linear pipeline of tasks
 */
template<typename T>
std::shared_ptr<TaskGraph<T>> make_pipeline(
    const std::vector<std::pair<std::string, std::function<T()>>>& tasks) {

    auto graph = std::make_shared<TaskGraph<T>>();

    std::string prev_id;
    for (const auto& [id, task] : tasks) {
        graph->add_task(id, task);
        if (!prev_id.empty()) {
            graph->add_dependency(id, prev_id);
        }
        prev_id = id;
    }

    return graph;
}

/**
 * @brief Create a fork-join pattern
 */
template<typename T>
std::shared_ptr<TaskGraph<T>> make_fork_join(
    const std::string& fork_task_id,
    const std::vector<std::pair<std::string, std::function<T()>>>& parallel_tasks,
    const std::string& join_task_id,
    std::function<T()> fork_func,
    std::function<T()> join_func) {

    auto graph = std::make_shared<TaskGraph<T>>();

    // Add fork task
    graph->add_task(fork_task_id, fork_func);

    // Add parallel tasks
    for (const auto& [id, task] : parallel_tasks) {
        graph->add_task(id, task);
        graph->add_dependency(id, fork_task_id);
    }

    // Add join task
    graph->add_task(join_task_id, join_func);
    for (const auto& [id, _] : parallel_tasks) {
        graph->add_dependency(join_task_id, id);
    }

    return graph;
}

} // namespace concurrency
} // namespace core
} // namespace fem

#endif // CORE_CONCURRENCY_TASK_GRAPH_H_

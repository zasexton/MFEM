#pragma once

#ifndef CORE_CONCURRENCY_TASK_H
#define CORE_CONCURRENCY_TASK_H

#include <functional>
#include <future>
#include <memory>
#include <atomic>
#include <vector>
#include <variant>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <utility>

#include <core/config/config.h>
#include <core/config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>
#include <core/concurrency/thread_pool.h>

namespace fem::core::concurrency {

// Forward declarations
template<typename T> class Task;

/**
 * @brief Exception thrown when a task is cancelled
 */
class TaskCancelledException : public std::runtime_error {
public:
    TaskCancelledException() : std::runtime_error("Task was cancelled") {}
    explicit TaskCancelledException(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Exception thrown when a task times out
 */
class TaskTimeoutException : public std::runtime_error {
public:
    TaskTimeoutException() : std::runtime_error("Task timed out") {}
    explicit TaskTimeoutException(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Internal shared state for task synchronization
 *
 * This structure holds all the mutable state that needs to be
 * shared between task copies. Using a shared_ptr to this structure
 * solves the mutex move problem.
 */
template<typename T>
struct TaskSharedState {
    // Forward declare Status enum
    enum class Status {
        Pending,
        Running,
        Completed,
        Failed,
        Cancelled
    };

    // Synchronization primitives
    mutable std::mutex mutex;
    std::condition_variable cv;

    // Task state
    std::atomic<bool> executed{false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> cancel_requested{false};
    std::atomic<Status> status{Status::Pending};

    // Result storage
    std::optional<T> result;
    std::exception_ptr exception;

    // Continuations
    std::vector<std::function<void()>> continuations;

    // Deadline
    std::optional<std::chrono::steady_clock::time_point> deadline;
};

// Specialization for void
template<>
struct TaskSharedState<void> {
    // Forward declare Status enum
    enum class Status {
        Pending,
        Running,
        Completed,
        Failed,
        Cancelled
    };

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> executed{false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> cancel_requested{false};
    std::atomic<Status> status{Status::Pending};
    std::exception_ptr exception;
    std::vector<std::function<void()>> continuations;
    std::optional<std::chrono::steady_clock::time_point> deadline;
};

/**
 * @brief Task abstraction for asynchronous computation
 *
 * Provides a composable abstraction for async operations with:
 * - Status tracking (Pending, Running, Completed, Failed, Cancelled)
 * - Future-based result retrieval
 * - Continuation support (then, then_on)
 * - Cancellation support
 * - Task composition (when_all, when_any)
 * - Thread pool integration
 *
 * The implementation uses a shared state pattern to enable
 * efficient copying and moving of tasks while maintaining
 * thread safety.
 *
 * @tparam T Result type of the task
 */
template<typename T>
class Task {
public:
    using result_type = T;

    // Use the Status enum from TaskSharedState
    using Status = typename TaskSharedState<T>::Status;

private:
    // Allow cross-template access for helper wiring
    template<typename> friend class Task;

    // The function to execute
    std::function<T()> func_;

    // Shared state for synchronization
    std::shared_ptr<TaskSharedState<T>> state_;

    // Promise and future for result communication
    std::shared_ptr<std::promise<T>> promise_;
    std::shared_future<T> future_;

    // Current status
    std::atomic<Status> status_{Status::Pending};

    // Cancellation flag
    std::atomic<bool> cancel_requested_{false};

    // Whether execute() should rethrow function exceptions (used for continuations)
    bool propagate_execute_exceptions_{false};

    /**
     * @brief Execute continuations after task completes
     */
    void execute_continuations() {
        if (!state_) return;

        std::vector<std::function<void()>> to_execute;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            to_execute = std::move(state_->continuations);
            state_->continuations.clear();
        }

        for (auto& cont : to_execute) {
            try {
                cont();
            } catch (...) {
                // Log but don't propagate continuation errors
                // In production, consider logging this
            }
        }
    }

    /**
     * @brief Check if deadline has passed
     */
    bool is_deadline_passed() const {
        if (!state_) return false;

        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->deadline) {
            return std::chrono::steady_clock::now() > *state_->deadline;
        }
        return false;
    }

    /**
     * @brief Internal execution logic
     */
    void execute_internal() {
        if (!state_ || !func_) {
            throw std::runtime_error("Invalid task state");
        }

        try {
            // Check for cancellation before execution
            if (cancel_requested_ || state_->cancelled) {
                status_ = Status::Cancelled;
                state_->status = Status::Cancelled;
                promise_->set_exception(
                    std::make_exception_ptr(TaskCancelledException()));
                state_->cv.notify_all();
                execute_continuations();
                return;
            }

            // Check for timeout
            if (is_deadline_passed()) {
                status_ = Status::Cancelled;
                state_->status = Status::Cancelled;
                promise_->set_exception(
                    std::make_exception_ptr(TaskTimeoutException()));
                state_->cv.notify_all();
                execute_continuations();
                return;
            }

            // Execute the function
            if constexpr (std::is_void_v<T>) {
                func_();
                promise_->set_value();
            } else {
                T result = func_();
                {
                    std::lock_guard<std::mutex> lock(state_->mutex);
                    state_->result = std::move(result);
                }
                promise_->set_value(state_->result.value());
            }

            status_ = Status::Completed;
            state_->status = Status::Completed;
            state_->executed = true;
            state_->cv.notify_all();
            execute_continuations();

        } catch (const TaskCancelledException&) {
            // Propagate cancellation for synchronous execute()
            status_ = Status::Cancelled;
            state_->status = Status::Cancelled;
            state_->executed = true;
            state_->cv.notify_all();
            try { promise_->set_exception(std::current_exception()); } catch (...) {}
            execute_continuations();
            throw;
        } catch (const TaskTimeoutException&) {
            // Propagate timeout for synchronous execute()
            status_ = Status::Cancelled;
            state_->status = Status::Cancelled;
            state_->executed = true;
            state_->cv.notify_all();
            try { promise_->set_exception(std::current_exception()); } catch (...) {}
            execute_continuations();
            throw;
        } catch (...) {
            status_ = Status::Failed;
            state_->status = Status::Failed;
            {
                std::lock_guard<std::mutex> lock(state_->mutex);
                state_->exception = std::current_exception();
            }
            state_->executed = true;
            state_->cv.notify_all();

            try {
                promise_->set_exception(std::current_exception());
            } catch (...) {
                // Promise already satisfied
            }

            execute_continuations();
            // Rethrow if requested (continuation semantics), otherwise swallow
            if (propagate_execute_exceptions_) {
                throw;
            }
        }
    }

public:
    /**
     * @brief Construct a task with a function
     * @param f Function to execute
     */
    explicit Task(std::function<T()> f)
        : func_(std::move(f)),
          state_(std::make_shared<TaskSharedState<T>>()),
          promise_(std::make_shared<std::promise<T>>()),
          future_(promise_->get_future().share()) {
    }

    /**
     * @brief Default constructor (empty task)
     */
    Task()
        : state_(std::make_shared<TaskSharedState<T>>()),
          promise_(std::make_shared<std::promise<T>>()),
          future_(promise_->get_future().share()) {
    }

    /**
     * @brief Copy constructor (shared state)
     */
    Task(const Task& other)
        : func_(other.func_),
          state_(other.state_),
          promise_(other.promise_),
          future_(other.future_),
          status_(other.status_.load()),
          cancel_requested_(other.cancel_requested_.load()) {
    }

    /**
     * @brief Move constructor
     */
    Task(Task&& other) noexcept
        : func_(std::move(other.func_)),
          state_(std::move(other.state_)),
          promise_(std::move(other.promise_)),
          future_(std::move(other.future_)),
          status_(other.status_.load()),
          cancel_requested_(other.cancel_requested_.load()) {
        other.status_ = Status::Pending;
        other.cancel_requested_ = false;
    }

    /**
     * @brief Copy assignment
     */
    Task& operator=(const Task& other) {
        if (this != &other) {
            func_ = other.func_;
            state_ = other.state_;
            promise_ = other.promise_;
            future_ = other.future_;
            status_ = other.status_.load();
            cancel_requested_ = other.cancel_requested_.load();
        }
        return *this;
    }

    /**
     * @brief Move assignment
     */
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            func_ = std::move(other.func_);
            state_ = std::move(other.state_);
            promise_ = std::move(other.promise_);
            future_ = std::move(other.future_);
            status_ = other.status_.load();
            cancel_requested_ = other.cancel_requested_.load();
            other.status_ = Status::Pending;
            other.cancel_requested_ = false;
        }
        return *this;
    }

    /**
     * @brief Destructor
     */
    ~Task() = default;

    /**
     * @brief Execute the task immediately on current thread
     */
    void execute() {
        if (!state_) {
            throw std::runtime_error("Invalid task state");
        }

        Status expected = Status::Pending;
        // Update both the local and shared status
        if (!state_->status.compare_exchange_strong(expected, Status::Running)) {
            throw std::runtime_error("Task already executed or cancelled");
        }
        status_ = Status::Running;

        execute_internal();
    }

    /**
     * @brief Execute the task on a thread pool
     * @param pool Thread pool to execute on
     * @param priority Task priority
     */
    void execute_on(ThreadPool& pool,
                   ThreadPool::Priority priority = ThreadPool::Priority::Normal) {
        if (!state_) {
            throw std::runtime_error("Invalid task state");
        }

        Status expected = Status::Pending;
        // Update both the local and shared status
        if (!state_->status.compare_exchange_strong(expected, Status::Running)) {
            throw std::runtime_error("Task already executed or cancelled");
        }
        status_ = Status::Running;

        // Capture everything by value to avoid dangling pointers when task moves
        auto state_copy = state_;
        auto func_copy = func_;
        auto promise_copy = promise_;

        pool.submit_with_priority([state_copy, func_copy, promise_copy]() mutable {
            if (!state_copy || !func_copy) {
                return;
            }

            try {
                // Check for cancellation before execution
                if (state_copy->cancelled || state_copy->cancel_requested) {
                    state_copy->status = Status::Cancelled;
                    promise_copy->set_exception(
                        std::make_exception_ptr(TaskCancelledException()));
                    state_copy->cv.notify_all();
                    // Execute any registered continuations on this task
                    std::vector<std::function<void()>> conts;
                    {
                        std::lock_guard<std::mutex> lock(state_copy->mutex);
                        conts = std::move(state_copy->continuations);
                        state_copy->continuations.clear();
                    }
                    for (auto& c : conts) {
                        try { c(); } catch (...) {}
                    }
                    return;
                }

                // Check for timeout
                {
                    std::lock_guard<std::mutex> lock(state_copy->mutex);
                    if (state_copy->deadline &&
                        std::chrono::steady_clock::now() > *state_copy->deadline) {
                        state_copy->status = Status::Cancelled;
                        promise_copy->set_exception(
                            std::make_exception_ptr(TaskTimeoutException()));
                        state_copy->cv.notify_all();
                        std::vector<std::function<void()>> conts;
                        conts = std::move(state_copy->continuations);
                        state_copy->continuations.clear();
                        for (auto& c : conts) {
                            try { c(); } catch (...) {}
                        }
                        return;
                    }
                }

                // Execute the function
                if constexpr (std::is_void_v<T>) {
                    func_copy();
                    promise_copy->set_value();
                } else {
                    T result = func_copy();
                    {
                        std::lock_guard<std::mutex> lock(state_copy->mutex);
                        state_copy->result = std::move(result);
                    }
                    promise_copy->set_value(state_copy->result.value());
                }

                state_copy->status = Status::Completed;
                state_copy->executed = true;
                state_copy->cv.notify_all();
                // Execute any registered continuations
                std::vector<std::function<void()>> conts;
                {
                    std::lock_guard<std::mutex> lock(state_copy->mutex);
                    conts = std::move(state_copy->continuations);
                    state_copy->continuations.clear();
                }
                for (auto& c : conts) {
                    try { c(); } catch (...) {}
                }

            } catch (...) {
                state_copy->status = Status::Failed;
                {
                    std::lock_guard<std::mutex> lock(state_copy->mutex);
                    state_copy->exception = std::current_exception();
                }
                state_copy->executed = true;
                state_copy->cv.notify_all();

                try {
                    promise_copy->set_exception(std::current_exception());
                } catch (...) {
                    // Promise already satisfied
                }
                // Execute any registered continuations even on failure
                std::vector<std::function<void()>> conts;
                {
                    std::lock_guard<std::mutex> lock(state_copy->mutex);
                    conts = std::move(state_copy->continuations);
                    state_copy->continuations.clear();
                }
                for (auto& c : conts) {
                    try { c(); } catch (...) {}
                }
            }
        }, priority);
    }

    /**
     * @brief Request cancellation of the task
     */
    void cancel() {
        cancel_requested_ = true;
        if (state_) {
            state_->cancelled = true;
            state_->cancel_requested = true;
            state_->cv.notify_all();
        }

        Status current = status_.load();
        if (current == Status::Pending) {
            Status expected = Status::Pending;
            // Update both local and shared status
            if (state_ && state_->status.compare_exchange_strong(expected, Status::Cancelled)) {
                status_ = Status::Cancelled;
                try {
                    promise_->set_exception(
                        std::make_exception_ptr(TaskCancelledException()));
                } catch (...) {
                    // Promise already satisfied
                }
                execute_continuations();
            }
        }
    }

    /**
     * @brief Set a deadline for the task
     * @param timeout Timeout duration from now
     */
    void cancel_after(std::chrono::nanoseconds timeout) {
        if (!state_) return;

        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->deadline = std::chrono::steady_clock::now() + timeout;
    }

    /**
     * @brief Check for cancellation (cooperative cancellation point)
     * @throws TaskCancelledException if cancellation was requested
     */
    void check_cancellation() const {
        if (cancel_requested_ || (state_ && state_->cancelled)) {
            throw TaskCancelledException();
        }

        if (is_deadline_passed()) {
            throw TaskTimeoutException();
        }
    }

    /**
     * @brief Wait for task completion and get result
     * @return Task result
     * @throws Any exception thrown by the task
     */
    T get() {
        return future_.get();
    }

    /**
     * @brief Wait for task completion and get result (const version)
     * @return Task result
     * @throws Any exception thrown by the task
     */
    T get() const {
        return future_.get();
    }

    /**
     * @brief Get the shared future for this task
     * @return Shared future
     */
    std::shared_future<T> get_future() const {
        return future_;
    }

    /**
     * @brief Wait for task completion
     */
    void wait() const {
        future_.wait();
    }

    /**
     * @brief Wait for task completion with timeout
     * @param timeout Maximum time to wait
     * @return Future status
     */
    template<typename Rep, typename Period>
    std::future_status wait_for(
        const std::chrono::duration<Rep, Period>& timeout) const {
        return future_.wait_for(timeout);
    }

    /**
     * @brief Wait for task completion until a time point
     * @param time_point Time point to wait until
     * @return Future status
     */
    template<typename Clock, typename Duration>
    std::future_status wait_until(
        const std::chrono::time_point<Clock, Duration>& time_point) const {
        return future_.wait_until(time_point);
    }

    /**
     * @brief Get current task status
     * @return Current status
     */
    Status status() const noexcept {
        if (state_) {
            return state_->status.load();
        }
        return status_.load();
    }

    /**
     * @brief Check if task is ready (completed, failed, or cancelled)
     * @return true if ready
     */
    bool is_ready() const noexcept {
        auto s = state_ ? state_->status.load() : status_.load();
        return s == Status::Completed || s == Status::Failed || s == Status::Cancelled;
    }

    /**
     * @brief Check if task completed successfully
     * @return true if completed
     */
    bool is_completed() const noexcept {
        return (state_ ? state_->status.load() : status_.load()) == Status::Completed;
    }

    /**
     * @brief Check if task failed
     * @return true if failed
     */
    bool is_failed() const noexcept {
        return (state_ ? state_->status.load() : status_.load()) == Status::Failed;
    }

    /**
     * @brief Check if task was cancelled
     * @return true if cancelled
     */
    bool is_cancelled() const noexcept {
        return (state_ ? state_->status.load() : status_.load()) == Status::Cancelled;
    }

    /**
     * @brief Check if cancellation was requested
     * @return true if cancellation requested
     */
    bool is_cancel_requested() const noexcept {
        return cancel_requested_.load();
    }

    /**
     * @brief Get stored exception (if failed)
     * @return Exception pointer or nullptr
     */
    std::exception_ptr get_exception() const {
        if (!state_) return nullptr;

        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->exception;
    }

    /**
     * @brief Add a continuation to execute after this task
     * @tparam F Function type
     * @param f Continuation function (takes result of this task)
     * @return New task representing the continuation
     */
    template<typename F>
    auto then(F&& f) -> Task<std::invoke_result_t<F, T>>
        requires (!std::is_void_v<T>) {
        using U = std::invoke_result_t<F, T>;

        // Capture the future by value to avoid dangling 'this' pointer
        auto future_copy = future_;
        auto continuation = Task<U>([future_copy, func = std::forward<F>(f)]() mutable {
            T result = future_copy.get();
            if constexpr (std::is_void_v<U>) {
                func(std::move(result));
            } else {
                return func(std::move(result));
            }
        });
        // Propagate exceptions to caller of execute() for continuations
        continuation.propagate_execute_exceptions_ = true;
        // For manual execution, no auto-registration is required
        return continuation;
    }

    /**
     * @brief Add a continuation for void tasks
     * @tparam F Function type
     * @param f Continuation function
     * @return New task representing the continuation
     */
    template<typename F>
    auto then(F&& f) -> Task<std::invoke_result_t<F>>
        requires std::is_void_v<T> {
        using U = std::invoke_result_t<F>;

        // Capture the future by value to avoid dangling 'this' pointer
        auto future_copy = future_;
        auto continuation = Task<U>([future_copy, func = std::forward<F>(f)]() mutable {
            future_copy.get();  // Wait for completion
            if constexpr (std::is_void_v<U>) {
                func();
            } else {
                return func();
            }
        });
        // Propagate exceptions to caller of execute() for continuations
        continuation.propagate_execute_exceptions_ = true;
        // For manual execution, no auto-registration is required
        return continuation;
    }

    /**
     * @brief Add a continuation to execute on a specific thread pool
     * @tparam F Function type
     * @param pool Thread pool to execute on
     * @param f Continuation function
     * @param priority Task priority
     * @return New task representing the continuation
     */
    template<typename F>
    auto then_on(ThreadPool& pool, F&& f,
                ThreadPool::Priority priority = ThreadPool::Priority::Normal)
        -> Task<std::invoke_result_t<F, T>>
        requires (!std::is_void_v<T>) {

        using U = std::invoke_result_t<F, T>;

        // Capture the future by value to avoid dangling 'this' pointer
        auto future_copy = future_;
        auto continuation = Task<U>([future_copy, func = std::forward<F>(f)]() mutable {
            T result = future_copy.get();
            if constexpr (std::is_void_v<U>) {
                func(std::move(result));
            } else {
                return func(std::move(result));
            }
        });

        // Register to auto-execute on pool when ready
        // Capture continuation state to execute it on the pool
        if (state_) {
            auto cont_state = continuation.state_;
            auto cont_func = continuation.func_;
            auto cont_promise = continuation.promise_;

            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->continuations.push_back(
                [cont_state, cont_func, cont_promise, &pool, priority]() mutable {
                    // Submit to pool using the same pattern as execute_on
                    pool.submit_with_priority([cont_state, cont_func, cont_promise]() mutable {
                        if (!cont_state || !cont_func) return;

                        using ContStatus = decltype(cont_state->status.load());
                        auto expected = ContStatus::Pending;
                        if (!cont_state->status.compare_exchange_strong(
                                expected, ContStatus::Running)) {
                            return;
                        }

                        try {
                            // Check for cancellation
                            if (cont_state->cancelled || cont_state->cancel_requested) {
                                cont_state->status = ContStatus::Cancelled;
                                cont_promise->set_exception(
                                    std::make_exception_ptr(TaskCancelledException()));
                                cont_state->cv.notify_all();
                                // Propagate continuations
                                std::vector<std::function<void()>> conts;
                                {
                                    std::lock_guard<std::mutex> lk(cont_state->mutex);
                                    conts = std::move(cont_state->continuations);
                                    cont_state->continuations.clear();
                                }
                                for (auto& c : conts) { try { c(); } catch (...) {} }
                                return;
                            }

                            // Execute the function
                            if constexpr (std::is_void_v<U>) {
                                cont_func();
                                cont_promise->set_value();
                            } else {
                                U cont_result = cont_func();
                                {
                                    std::lock_guard<std::mutex> lock2(cont_state->mutex);
                                    cont_state->result = std::move(cont_result);
                                }
                                cont_promise->set_value(cont_state->result.value());
                            }

                            cont_state->status = ContStatus::Completed;
                            cont_state->executed = true;
                            cont_state->cv.notify_all();
                            // Execute chained continuations
                            std::vector<std::function<void()>> conts;
                            {
                                std::lock_guard<std::mutex> lk(cont_state->mutex);
                                conts = std::move(cont_state->continuations);
                                cont_state->continuations.clear();
                            }
                            for (auto& c : conts) { try { c(); } catch (...) {} }
                        } catch (...) {
                            cont_state->status = ContStatus::Failed;
                            {
                                std::lock_guard<std::mutex> lock3(cont_state->mutex);
                                cont_state->exception = std::current_exception();
                            }
                            cont_state->executed = true;
                            cont_state->cv.notify_all();

                            try {
                                cont_promise->set_exception(std::current_exception());
                            } catch (...) {}
                            // Execute chained continuations even on failure
                            std::vector<std::function<void()>> conts;
                            {
                                std::lock_guard<std::mutex> lk(cont_state->mutex);
                                conts = std::move(cont_state->continuations);
                                cont_state->continuations.clear();
                            }
                            for (auto& c : conts) { try { c(); } catch (...) {} }
                        }
                    }, priority);
                });
        }

        return continuation;
    }

    /**
     * @brief Add a continuation to execute on a specific thread pool (void version)
     */
    template<typename F>
    auto then_on(ThreadPool& pool, F&& f,
                ThreadPool::Priority priority = ThreadPool::Priority::Normal)
        -> Task<std::invoke_result_t<F>>
        requires std::is_void_v<T> {

        using U = std::invoke_result_t<F>;

        // Capture the future by value to avoid dangling 'this'
        auto future_copy = future_;
        auto continuation = Task<U>([future_copy, func = std::forward<F>(f)]() mutable {
            future_copy.get();
            if constexpr (std::is_void_v<U>) {
                func();
            } else {
                return func();
            }
        });

        // Register to auto-execute on pool when ready
        if (state_) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->continuations.push_back(
                [continuation, &pool, priority]() mutable {
                    continuation.execute_on(pool, priority);
                });
        }

        return continuation;
    }

    // Static composition methods

    /**
     * @brief Wait for all tasks to complete
     * @tparam U Result type of second task
     * @param t1 First task
     * @param t2 Second task
     * @return Task returning pair of results or single result if one is void
     */
    template<typename U>
    static auto when_all(Task<T>& t1, Task<U>& t2) {
        if constexpr (std::is_void_v<T> && std::is_void_v<U>) {
            return Task<void>([&t1, &t2]() {
                t1.get();
                t2.get();
            });
        } else if constexpr (std::is_void_v<T>) {
            return Task<U>([&t1, &t2]() {
                t1.get();
                return t2.get();
            });
        } else if constexpr (std::is_void_v<U>) {
            return Task<T>([&t1, &t2]() {
                T r1 = t1.get();
                t2.get();
                return r1;
            });
        } else {
            return Task<std::pair<T, U>>([&t1, &t2]() {
                T r1 = t1.get();
                U r2 = t2.get();
                return std::pair<T, U>{std::move(r1), std::move(r2)};
            });
        }
    }

    /**
     * @brief Wait for any task to complete (race)
     * @tparam U Result type of second task
     * @param t1 First task
     * @param t2 Second task
     * @return Task returning variant of results (first to complete)
     */
    template<typename U>
    static Task<std::variant<T, U>> when_any(Task<T>& t1, Task<U>& t2) {
        return Task<std::variant<T, U>>([&t1, &t2]() {
            // Use wait_for with small timeout to poll both
            while (true) {
                auto s1 = t1.wait_for(std::chrono::milliseconds(1));
                if (s1 == std::future_status::ready) {
                    if constexpr (std::is_void_v<T>) {
                        t1.get();
                        return std::variant<T, U>{std::in_place_index<0>};
                    } else {
                        return std::variant<T, U>{std::in_place_index<0>, t1.get()};
                    }
                }

                auto s2 = t2.wait_for(std::chrono::milliseconds(1));
                if (s2 == std::future_status::ready) {
                    if constexpr (std::is_void_v<U>) {
                        t2.get();
                        return std::variant<T, U>{std::in_place_index<1>};
                    } else {
                        return std::variant<T, U>{std::in_place_index<1>, t2.get()};
                    }
                }

                std::this_thread::yield();
            }
        });
    }

    /**
     * @brief Wait for all tasks in a vector
     * @param tasks Vector of tasks
     * @return Task returning vector of results
     */
    static Task<std::vector<T>> when_all(std::vector<Task<T>>& tasks)
        requires (!std::is_void_v<T>) {
        return Task<std::vector<T>>([&tasks]() {
            std::vector<T> results;
            results.reserve(tasks.size());
            for (auto& task : tasks) {
                results.push_back(task.get());
            }
            return results;
        });
    }

    /**
     * @brief Wait for all void tasks in a vector
     * @param tasks Vector of void tasks
     * @return Task that completes when all complete
     */
    static Task<void> when_all_void(std::vector<Task<void>>& tasks) {
        return Task<void>([&tasks]() {
            for (auto& task : tasks) {
                task.get();
            }
        });
    }
};

// Helper functions

/**
 * @brief Create a task from a function
 * @tparam F Function type
 * @param f Function to execute
 * @return Task wrapping the function
 */
template<typename F>
auto make_task(F&& f) -> Task<std::invoke_result_t<F>> {
    return Task<std::invoke_result_t<F>>(std::forward<F>(f));
}

/**
 * @brief Create and immediately execute a task asynchronously
 * @tparam F Function type
 * @param f Function to execute
 * @param pool Thread pool to execute on (defaults to global pool)
 * @return Task that is already running
 */
template<typename F>
auto async_task(F&& f, ThreadPool* pool = nullptr)
    -> Task<std::invoke_result_t<F>> {

    auto task = make_task(std::forward<F>(f));

    if (pool) {
        task.execute_on(*pool);
    } else {
        task.execute_on(global_thread_pool());
    }

    return task;
}

/**
 * @brief Create a task that is already completed with a value
 * @tparam T Value type
 * @param value Value to wrap
 * @return Completed task
 */
template<typename T>
Task<T> make_ready_task(T value) {
    auto task = Task<T>([v = std::move(value)]() { return v; });
    task.execute();
    return task;
}

/**
 * @brief Create a void task that is already completed
 * @return Completed void task
 */
inline Task<void> make_ready_task() {
    auto task = Task<void>([]() {});
    task.execute();
    return task;
}

/**
 * @brief Create a task that is already failed with an exception
 * @tparam T Result type
 * @param ex Exception to store
 * @return Failed task
 */
template<typename T>
Task<T> make_exceptional_task(std::exception_ptr ex) {
    auto task = Task<T>([ex]() -> T {
        std::rethrow_exception(ex);
    });
    // Execute immediately so the exception is captured and the task is ready
    task.execute();
    return task;
}

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_TASK_H

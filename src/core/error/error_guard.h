#pragma once

#ifndef CORE_ERROR_ERROR_GUARD_H
#define CORE_ERROR_ERROR_GUARD_H

#include <functional>
#include <exception>
#include <optional>
#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include "exception_base.h"
#include "result.h"
#include "nested_exception.h"

namespace fem::core::error {

/**
 * @brief Exception safety guards and RAII helpers
 *
 * Provides strong exception safety guarantees through:
 * - Rollback on exception
 * - Commit/rollback pattern
 * - Resource cleanup
 * - Transaction support
 */

/**
 * @brief Basic scope guard for cleanup
 */
class ScopeGuard {
public:
    using Action = std::function<void()>;

    explicit ScopeGuard(Action cleanup)
        : cleanup_(std::move(cleanup))
        , active_(true) {
    }

    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_))
        , active_(other.active_) {
        other.dismiss();
    }

    ~ScopeGuard() {
        if (active_ && cleanup_) {
            try {
                cleanup_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void dismiss() noexcept {
        active_ = false;
    }

    void execute() {
        if (active_ && cleanup_) {
            cleanup_();
            dismiss();
        }
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&) = delete;

private:
    Action cleanup_;
    bool active_;
};

/**
 * @brief Guard that executes only on exception
 */
class ExceptionGuard {
public:
    using Action = std::function<void()>;

    explicit ExceptionGuard(Action rollback)
        : rollback_(std::move(rollback))
        , exception_count_(std::uncaught_exceptions()) {
    }

    ~ExceptionGuard() {
        if (rollback_ && std::uncaught_exceptions() > exception_count_) {
            try {
                rollback_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void dismiss() noexcept {
        rollback_ = nullptr;
    }

    ExceptionGuard(const ExceptionGuard&) = delete;
    ExceptionGuard& operator=(const ExceptionGuard&) = delete;
    ExceptionGuard(ExceptionGuard&&) = delete;
    ExceptionGuard& operator=(ExceptionGuard&&) = delete;

private:
    Action rollback_;
    int exception_count_;
};

/**
 * @brief Guard that executes only on success (no exception)
 */
class SuccessGuard {
public:
    using Action = std::function<void()>;

    explicit SuccessGuard(Action commit)
        : commit_(std::move(commit))
        , exception_count_(std::uncaught_exceptions()) {
    }

    ~SuccessGuard() {
        if (commit_ && std::uncaught_exceptions() == exception_count_) {
            try {
                commit_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void execute() {
        if (commit_) {
            commit_();
            dismiss();
        }
    }

    void dismiss() noexcept {
        commit_ = nullptr;
    }

    SuccessGuard(const SuccessGuard&) = delete;
    SuccessGuard& operator=(const SuccessGuard&) = delete;
    SuccessGuard(SuccessGuard&&) = delete;
    SuccessGuard& operator=(SuccessGuard&&) = delete;

private:
    Action commit_;
    int exception_count_;
};

/**
 * @brief Transaction guard with commit/rollback
 */
class TransactionGuard {
public:
    using Action = std::function<void()>;

    TransactionGuard(Action commit, Action rollback)
        : commit_(std::move(commit))
        , rollback_(std::move(rollback))
        , committed_(false) {
    }

    ~TransactionGuard() {
        if (!committed_ && rollback_) {
            try {
                rollback_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void commit() {
        if (!committed_ && commit_) {
            commit_();
            committed_ = true;
        }
    }

    void rollback() {
        if (!committed_ && rollback_) {
            rollback_();
            committed_ = true;  // Prevent double rollback
        }
    }

    bool is_committed() const noexcept {
        return committed_;
    }

    TransactionGuard(const TransactionGuard&) = delete;
    TransactionGuard& operator=(const TransactionGuard&) = delete;
    TransactionGuard(TransactionGuard&&) = delete;
    TransactionGuard& operator=(TransactionGuard&&) = delete;

private:
    Action commit_;
    Action rollback_;
    bool committed_;
};

/**
 * @brief Value restoration guard
 */
template<typename T>
class ValueGuard {
public:
    explicit ValueGuard(T& value)
        : value_(value)
        , old_value_(value)
        , should_restore_(true) {
    }

    ValueGuard(T& value, T new_value)
        : value_(value)
        , old_value_(std::move(value))
        , should_restore_(true) {
        value = std::move(new_value);
    }

    ~ValueGuard() {
        if (should_restore_) {
            try {
                value_ = std::move(old_value_);
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void dismiss() noexcept {
        should_restore_ = false;
    }

    void restore() {
        if (should_restore_) {
            value_ = old_value_;
        }
    }

    const T& old_value() const noexcept {
        return old_value_;
    }

    ValueGuard(const ValueGuard&) = delete;
    ValueGuard& operator=(const ValueGuard&) = delete;
    ValueGuard(ValueGuard&&) = delete;
    ValueGuard& operator=(ValueGuard&&) = delete;

private:
    T& value_;
    T old_value_;
    bool should_restore_;
};

/**
 * @brief Error accumulator for multiple operations
 */
class ErrorAccumulator {
public:
    ErrorAccumulator() = default;

    /**
     * @brief Try an operation and accumulate errors
     */
    template<typename F>
    void try_operation(const std::string& description, F&& op) {
        try {
            std::forward<F>(op)();
        } catch (const std::exception& e) {
            errors_.emplace_back(description, std::current_exception());
        }
    }

    /**
     * @brief Try an operation with result
     */
    template<typename F>
    auto try_with_result(const std::string& description, F&& op) 
        -> std::optional<decltype(op())> {
        try {
            return std::forward<F>(op)();
        } catch (const std::exception& e) {
            errors_.emplace_back(description, std::current_exception());
            return std::nullopt;
        }
    }

    /**
     * @brief Check if any errors occurred
     */
    bool has_errors() const noexcept {
        return !errors_.empty();
    }

    /**
     * @brief Get error count
     */
    size_t error_count() const noexcept {
        return errors_.size();
    }

    /**
     * @brief Get all errors
     */
    const std::vector<std::pair<std::string, std::exception_ptr>>& errors() const {
        return errors_;
    }

    /**
     * @brief Throw aggregate exception if errors exist
     */
    void throw_if_errors(const std::string& message = "Multiple errors occurred") {
        if (!errors_.empty()) {
            AggregateException ex(message);
            for (const auto& [desc, ptr] : errors_) {
                ex.add_exception(ptr);
            }
            throw ex;
        }
    }

    /**
     * @brief Get formatted error summary
     */
    std::string error_summary() const {
        if (errors_.empty()) {
            return "No errors";
        }

        std::ostringstream oss;
        oss << "Errors (" << errors_.size() << "):\n";
        
        for (const auto& [desc, ptr] : errors_) {
            oss << "  - " << desc << ": ";
            try {
                std::rethrow_exception(ptr);
            } catch (const std::exception& e) {
                oss << e.what();
            } catch (...) {
                oss << "Unknown error";
            }
            oss << "\n";
        }
        
        return oss.str();
    }

    /**
     * @brief Clear all errors
     */
    void clear() {
        errors_.clear();
    }

private:
    std::vector<std::pair<std::string, std::exception_ptr>> errors_;
};

/**
 * @brief Retry guard for transient failures
 */
class RetryGuard {
public:
    struct Config {
        size_t max_attempts = 3;
        std::chrono::milliseconds initial_delay{100};
        double backoff_multiplier = 2.0;
        std::chrono::milliseconds max_delay{5000};
    };

    explicit RetryGuard(Config config = Config{})
        : config_(config) {
    }

    /**
     * @brief Execute operation with retry
     */
    template<typename F>
    auto execute(F&& op) -> decltype(op()) {
        std::chrono::milliseconds delay = config_.initial_delay;
        std::exception_ptr last_exception;

        for (size_t attempt = 1; attempt <= config_.max_attempts; ++attempt) {
            try {
                return std::forward<F>(op)();
            } catch (...) {
                last_exception = std::current_exception();
                
                if (attempt < config_.max_attempts) {
                    std::this_thread::sleep_for(delay);
                    
                    // Calculate next delay with backoff
                    delay = std::chrono::milliseconds(
                        static_cast<long>(delay.count() * config_.backoff_multiplier));
                    
                    if (delay > config_.max_delay) {
                        delay = config_.max_delay;
                    }
                }
            }
        }

        // All attempts failed, rethrow last exception
        std::rethrow_exception(last_exception);
    }

    /**
     * @brief Execute with custom retry predicate
     */
    template<typename F, typename P>
    auto execute_if(F&& op, P&& should_retry) -> decltype(op()) {
        std::chrono::milliseconds delay = config_.initial_delay;
        std::exception_ptr last_exception;

        for (size_t attempt = 1; attempt <= config_.max_attempts; ++attempt) {
            try {
                return std::forward<F>(op)();
            } catch (...) {
                last_exception = std::current_exception();
                
                bool retry = false;
                try {
                    std::rethrow_exception(last_exception);
                } catch (const std::exception& e) {
                    retry = should_retry(e, attempt);
                } catch (...) {
                    retry = should_retry(nullptr, attempt);
                }

                if (retry && attempt < config_.max_attempts) {
                    std::this_thread::sleep_for(delay);
                    delay = std::chrono::milliseconds(
                        static_cast<long>(delay.count() * config_.backoff_multiplier));
                    
                    if (delay > config_.max_delay) {
                        delay = config_.max_delay;
                    }
                } else if (!retry) {
                    std::rethrow_exception(last_exception);
                }
            }
        }

        std::rethrow_exception(last_exception);
    }

private:
    Config config_;
};

/**
 * @brief State snapshot for rollback
 */
template<typename T>
class StateSnapshot {
public:
    explicit StateSnapshot(const T& state)
        : snapshot_(std::make_unique<T>(state)) {
    }

    void capture(const T& state) {
        snapshot_ = std::make_unique<T>(state);
    }

    void restore(T& state) const {
        if (snapshot_) {
            state = *snapshot_;
        }
    }

    const T& get() const {
        if (!snapshot_) {
            throw std::logic_error("No snapshot available");
        }
        return *snapshot_;
    }

    bool has_snapshot() const noexcept {
        return snapshot_ != nullptr;
    }

private:
    std::unique_ptr<T> snapshot_;
};

/**
 * @brief Helper functions
 */

// Create scope guard
template<typename F>
auto make_scope_guard(F&& f) {
    return ScopeGuard(std::forward<F>(f));
}

// Create exception guard
template<typename F>
auto make_exception_guard(F&& f) {
    return ExceptionGuard(std::forward<F>(f));
}

// Create success guard
template<typename F>
auto make_success_guard(F&& f) {
    return SuccessGuard(std::forward<F>(f));
}

// Create transaction
template<typename C, typename R>
auto make_transaction(C&& commit, R&& rollback) {
    return TransactionGuard(std::forward<C>(commit), std::forward<R>(rollback));
}

// Create value guard
template<typename T>
auto make_value_guard(T& value) {
    return ValueGuard<T>(value);
}

/**
 * @brief Macros for guard creation
 */

#define FEM_SCOPE_EXIT(code) \
    auto FEM_UNIQUE_NAME(scope_guard) = fem::core::error::make_scope_guard([&]() { code; })

#define FEM_ON_EXCEPTION(code) \
    auto FEM_UNIQUE_NAME(exception_guard) = fem::core::error::make_exception_guard([&]() { code; })

#define FEM_ON_SUCCESS(code) \
    auto FEM_UNIQUE_NAME(success_guard) = fem::core::error::make_success_guard([&]() { code; })

#define FEM_VALUE_GUARD(value) \
    auto FEM_UNIQUE_NAME(value_guard) = fem::core::error::make_value_guard(value)

// Helper to create unique names
#define FEM_UNIQUE_NAME(base) FEM_CONCAT(base, __LINE__)
#define FEM_CONCAT(a, b) FEM_CONCAT_IMPL(a, b)
#define FEM_CONCAT_IMPL(a, b) a##b

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_GUARD_H
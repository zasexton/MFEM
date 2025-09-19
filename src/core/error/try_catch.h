#pragma once

#ifndef CORE_ERROR_TRY_CATCH_H
#define CORE_ERROR_TRY_CATCH_H

#include "result.h"
#include "expected.h"
#include "exception_base.h"
#include <functional>
#include <type_traits>

namespace fem::core::error {

/**
 * @brief Convert exception-throwing code to Result
 *
 * Wraps a function that may throw exceptions and converts it to a Result type
 */
template<typename F, typename... Args>
    requires std::invocable<F, Args...>
auto try_catch(F&& f, Args&&... args) -> Result<std::invoke_result_t<F, Args...>, ErrorInfo> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    try {
        if constexpr (std::is_void_v<ReturnType>) {
            std::forward<F>(f)(std::forward<Args>(args)...);
            return Result<void, ErrorInfo>();  // Success for void
        } else {
            return Result<ReturnType, ErrorInfo>(
                std::forward<F>(f)(std::forward<Args>(args)...)
            );
        }
    } catch (const Exception& e) {
        return Result<ReturnType, ErrorInfo>(
            Error<ErrorInfo>(make_error(e.code(), e.what(), e.where()))
        );
    } catch (const std::exception& e) {
        return Result<ReturnType, ErrorInfo>(
            Error<ErrorInfo>(make_error(ErrorCode::Unknown, e.what()))
        );
    } catch (...) {
        return Result<ReturnType, ErrorInfo>(
            Error<ErrorInfo>(make_error(ErrorCode::Unknown, "Unknown exception"))
        );
    }
}

/**
 * @brief Execute with guaranteed cleanup
 *
 * Similar to try-finally in other languages
 */
template<typename F, typename C>
    requires std::invocable<F> && std::invocable<C>
auto try_finally(F&& f, C&& cleanup) {
    struct Cleanup {
        C cleanup_fn;
        ~Cleanup() {
            try {
                cleanup_fn();
            } catch (...) {
                // Suppress exceptions in cleanup
            }
        }
    };

    Cleanup guard{std::forward<C>(cleanup)};
    return std::forward<F>(f)();
}

/**
 * @brief Execute with cleanup on exception
 *
 * Only runs cleanup if an exception is thrown
 */
template<typename F, typename C>
    requires std::invocable<F> && std::invocable<C>
auto try_on_error(F&& f, C&& cleanup) {
    try {
        return std::forward<F>(f)();
    } catch (...) {
        try {
            std::forward<C>(cleanup)();
        } catch (...) {
            // Suppress cleanup exceptions
        }
        throw;  // Rethrow original exception
    }
}

/**
 * @brief Retry an operation with exponential backoff
 */
template<typename F>
    requires std::invocable<F>
auto try_with_retry(F&& f,
                   size_t max_attempts = 3,
                   std::chrono::milliseconds initial_delay = std::chrono::milliseconds(100)) {
    // ReturnType is implicitly determined by the function return

    std::exception_ptr last_exception;
    auto delay = initial_delay;

    for (size_t attempt = 1; attempt <= max_attempts; ++attempt) {
        try {
            return std::forward<F>(f)();
        } catch (...) {
            last_exception = std::current_exception();

            if (attempt < max_attempts) {
                std::this_thread::sleep_for(delay);
                delay *= 2;  // Exponential backoff
            }
        }
    }

    // All attempts failed, rethrow last exception
    std::rethrow_exception(last_exception);
}

/**
 * @brief Convert Result to exception
 *
 * Throws if Result contains an error
 */
template<typename T, typename E>
T unwrap_or_throw(Result<T, E>&& result) {
    if (result.is_error()) {
        if constexpr (std::is_base_of_v<std::exception, E>) {
            throw result.error();
        } else if constexpr (std::is_same_v<E, ErrorInfo>) {
            throw Exception(std::string(result.error().message()),
                          result.error().error_code(),
                          result.error().location());
        } else {
            throw RuntimeError(std::format("Operation failed"));
        }
    }
    if constexpr (!std::is_void_v<T>) {
        return std::move(result).value();
    } else {
        result.value();  // Just checks for errors
    }
}

/**
 * @brief Chain multiple fallible operations
 *
 * Short-circuits on first error
 */
template<typename... Ops>
auto try_chain(Ops... ops) {
    return [ops...](auto&& input) {
        auto result = std::forward<decltype(input)>(input);

        auto chain_op = [&result](auto&& op) {
            if (result.is_ok()) {
                result = op(std::move(result).value());
            }
        };

        (chain_op(ops), ...);

        return result;
    };
}

/**
 * @brief Collect results from multiple operations
 *
 * Returns all results, doesn't short-circuit
 */
template<typename... Ops>
auto try_collect(Ops... ops) {
    return std::make_tuple(ops()...);
}

/**
 * @brief Guard scope with error handling
 */
class TryScope {
public:
    explicit TryScope(std::function<void()> on_error = nullptr)
        : on_error_(std::move(on_error))
        , has_error_(false) {
    }

    ~TryScope() {
        if (has_error_ && on_error_) {
            try {
                on_error_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void set_error() noexcept {
        has_error_ = true;
    }

    void clear_error() noexcept {
        has_error_ = false;
    }

    bool has_error() const noexcept {
        return has_error_;
    }

private:
    std::function<void()> on_error_;
    bool has_error_;
};

/**
 * @brief Macro for automatic error propagation
 *
 * Similar to Rust's ? operator
 */
#define TRY_OR_RETURN(expr) \
    do { \
        auto _result = (expr); \
        if (_result.is_error()) { \
            return Err<decltype(_result.error())>(_result.error()); \
        } \
    } while(0)

/**
 * @brief Macro for logging errors without propagating
 */
#define TRY_LOG(expr) \
    do { \
        auto _result = (expr); \
        if (_result.is_error()) { \
            handle_error(Exception(_result.error().message(), \
                                 _result.error().error_code())); \
        } \
    } while(0)

} // namespace fem::core::error

#endif // CORE_ERROR_TRY_CATCH_H
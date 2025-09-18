#pragma once

#ifndef CORE_ERROR_OUTCOME_H
#define CORE_ERROR_OUTCOME_H

#include <variant>
#include <exception>
#include <optional>
#include "result.h"
#include "expected.h"
#include "exception_base.h"

namespace fem::core::error {

/**
 * @brief Outcome type supporting value, error code, or exception
 *
 * Similar to boost::outcome or std::expected but with three-way state:
 * - Success with value
 * - Error with error code
 * - Exception with exception_ptr
 *
 * This provides maximum flexibility for error handling strategies.
 */
template<typename T, typename E = ErrorCode>
class Outcome {
public:
    using value_type = T;
    using error_type = E;
    using exception_type = std::exception_ptr;

private:
    // Three possible states: value, error, or exception
    std::variant<T, E, std::exception_ptr> data_;

public:
    // Constructors
    Outcome(T value) : data_(std::move(value)) {}
    Outcome(Error<E> error) : data_(std::move(error.get())) {}
    Outcome(std::exception_ptr ex) : data_(std::move(ex)) {}

    // Factory methods
    static Outcome success(T value) {
        return Outcome(std::move(value));
    }

    static Outcome failure(E error) {
        return Outcome(Error<E>(std::move(error)));
    }

    static Outcome exception(std::exception_ptr ex) {
        return Outcome(std::move(ex));
    }

    /**
     * @brief Create outcome from a function that may throw
     */
    template<typename F>
        requires std::invocable<F>
    static Outcome from_call(F&& f) noexcept {
        try {
            if constexpr (std::is_same_v<std::invoke_result_t<F>, void>) {
                std::forward<F>(f)();
                return Outcome(T{});
            } else {
                return Outcome(std::forward<F>(f)());
            }
        } catch (const Exception& e) {
            // Try to extract error code from our exception type
            return Outcome(Error<E>(static_cast<E>(e.code())));
        } catch (...) {
            // Store other exceptions as exception_ptr
            return Outcome(std::current_exception());
        }
    }

    // State queries
    bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    bool has_error() const noexcept {
        return std::holds_alternative<E>(data_);
    }

    bool has_exception() const noexcept {
        return std::holds_alternative<std::exception_ptr>(data_);
    }

    bool has_failure() const noexcept {
        return !has_value();
    }

    explicit operator bool() const noexcept {
        return has_value();
    }

    // Value access
    T& value() & {
        if (has_value()) {
            return std::get<T>(data_);
        }
        if (has_error()) {
            throw RuntimeError(std::format("Outcome has error: {}",
                                         std::get<E>(data_).message()));
        }
        std::rethrow_exception(std::get<std::exception_ptr>(data_));
    }

    const T& value() const& {
        if (has_value()) {
            return std::get<T>(data_);
        }
        if (has_error()) {
            throw RuntimeError(std::format("Outcome has error: {}",
                                         std::get<E>(data_).message()));
        }
        std::rethrow_exception(std::get<std::exception_ptr>(data_));
    }

    T&& value() && {
        if (has_value()) {
            return std::get<T>(std::move(data_));
        }
        if (has_error()) {
            throw RuntimeError(std::format("Outcome has error: {}",
                                         std::get<E>(data_).message()));
        }
        std::rethrow_exception(std::get<std::exception_ptr>(data_));
    }

    // Error access
    E& error() & {
        if (!has_error()) {
            throw LogicError("Outcome does not have an error");
        }
        return std::get<E>(data_);
    }

    const E& error() const& {
        if (!has_error()) {
            throw LogicError("Outcome does not have an error");
        }
        return std::get<E>(data_);
    }

    // Exception access
    std::exception_ptr exception() const {
        if (!has_exception()) {
            throw LogicError("Outcome does not have an exception");
        }
        return std::get<std::exception_ptr>(data_);
    }

    /**
     * @brief Get value or default
     */
    T value_or(T default_value) const {
        return has_value() ? std::get<T>(data_) : std::move(default_value);
    }

    /**
     * @brief Get value or compute default
     */
    template<typename F>
        requires std::invocable<F>
    T value_or_else(F&& f) const {
        return has_value() ? std::get<T>(data_) : std::forward<F>(f)();
    }

    /**
     * @brief Convert to Result
     */
    Result<T, E> to_result() const {
        if (has_value()) {
            return Ok<T, E>(std::get<T>(data_));
        }
        if (has_error()) {
            return Err<E, T>(std::get<E>(data_));
        }

        // Try to extract error from exception
        try {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        } catch (const Exception& e) {
            if constexpr (std::is_same_v<E, ErrorCode>) {
                return Err<E, T>(e.code());
            } else {
                return Err<E, T>(E{});
            }
        } catch (...) {
            if constexpr (std::is_same_v<E, ErrorCode>) {
                return Err<E, T>(ErrorCode::Unknown);
            } else {
                return Err<E, T>(E{});
            }
        }
    }

    /**
     * @brief Convert to Expected
     */
    Expected<T> to_expected() const {
        if (has_value()) {
            return Expected<T>(std::get<T>(data_));
        }
        if (has_exception()) {
            return Expected<T>(std::get<std::exception_ptr>(data_));
        }

        // Convert error to exception
        return Expected<T>(std::make_exception_ptr(
            RuntimeError(std::format("Error: {}", std::get<E>(data_).message()))
        ));
    }

    // Monadic operations
    template<typename F>
        requires std::invocable<F, T>
    auto map(F&& f) -> Outcome<std::invoke_result_t<F, T>, E> {
        if (has_value()) {
            return Outcome<std::invoke_result_t<F, T>, E>::from_call([&]() {
                return std::forward<F>(f)(std::get<T>(data_));
            });
        }
        if (has_error()) {
            return Outcome<std::invoke_result_t<F, T>, E>(
                Error<E>(std::get<E>(data_))
            );
        }
        return Outcome<std::invoke_result_t<F, T>, E>(
            std::get<std::exception_ptr>(data_)
        );
    }

    template<typename F>
        requires std::invocable<F, E>
    auto map_error(F&& f) -> Outcome<T, std::invoke_result_t<F, E>> {
        using NewE = std::invoke_result_t<F, E>;

        if (has_value()) {
            return Outcome<T, NewE>(std::get<T>(data_));
        }
        if (has_error()) {
            return Outcome<T, NewE>(Error<NewE>(
                std::forward<F>(f)(std::get<E>(data_))
            ));
        }
        return Outcome<T, NewE>(std::get<std::exception_ptr>(data_));
    }

    template<typename F>
        requires std::invocable<F, T>
    auto and_then(F&& f) -> std::invoke_result_t<F, T> {
        if (has_value()) {
            return std::forward<F>(f)(std::get<T>(data_));
        }

        using ResultType = std::invoke_result_t<F, T>;
        if (has_error()) {
            return ResultType(Error<E>(std::get<E>(data_)));
        }
        return ResultType(std::get<std::exception_ptr>(data_));
    }

    /**
     * @brief Visit the outcome with separate handlers
     */
    template<typename ValueFunc, typename ErrorFunc, typename ExceptFunc>
    auto visit(ValueFunc&& on_value,
              ErrorFunc&& on_error,
              ExceptFunc&& on_exception) {
        return std::visit([&](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<ArgType, T>) {
                return std::forward<ValueFunc>(on_value)(arg);
            } else if constexpr (std::is_same_v<ArgType, E>) {
                return std::forward<ErrorFunc>(on_error)(arg);
            } else {
                return std::forward<ExceptFunc>(on_exception)(arg);
            }
        }, data_);
    }
};

// Deduction guides
template<typename T>
Outcome(T) -> Outcome<T, ErrorCode>;

template<typename E>
Outcome(Error<E>) -> Outcome<void, E>;

/**
 * @brief Convenience factory for successful outcome
 */
template<typename T, typename E = ErrorCode>
auto make_outcome(T&& value) {
    return Outcome<std::decay_t<T>, E>(std::forward<T>(value));
}

/**
 * @brief Convenience factory for error outcome
 */
template<typename E, typename T = void>
auto make_error_outcome(E&& error) {
    if constexpr (std::is_void_v<T>) {
        return Outcome<std::monostate, std::decay_t<E>>(
            Error<std::decay_t<E>>(std::forward<E>(error))
        );
    } else {
        return Outcome<T, std::decay_t<E>>(
            Error<std::decay_t<E>>(std::forward<E>(error))
        );
    }
}

/**
 * @brief Convenience factory for exception outcome
 */
template<typename T = void, typename E = ErrorCode>
auto make_exception_outcome(std::exception_ptr ex) {
    if constexpr (std::is_void_v<T>) {
        return Outcome<std::monostate, E>(ex);
    } else {
        return Outcome<T, E>(ex);
    }
}

} // namespace fem::core::error

#endif // CORE_ERROR_OUTCOME_H
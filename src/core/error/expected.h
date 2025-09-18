#pragma once

#ifndef CORE_ERROR_EXPECTED_H
#define CORE_ERROR_EXPECTED_H

#include "result.h"
#include "exception_base.h"
#include "logic_error.h"
#include <exception>
#include <functional>

namespace fem::core::error {

/**
 * @brief Expected type with exception_ptr as error type
 *
 * Bridges the gap between exception-based and error-code-based error handling.
 * Similar to Result but specialized for exception handling.
 */
template<typename T>
class Expected {
public:
    using value_type = T;
    using error_type = std::exception_ptr;

private:
    std::variant<T, std::exception_ptr> data_;

public:
    // Constructors
    Expected(T value) : data_(std::move(value)) {}
    Expected(std::exception_ptr error) : data_(std::move(error)) {}

    // Factory methods
    static Expected success(T value) {
        return Expected(std::move(value));
    }

    static Expected failure(std::exception_ptr error) {
        return Expected(std::move(error));
    }

    /**
     * @brief Try to invoke a function and capture any exceptions
     */
    template<typename F>
        requires std::invocable<F>
    static Expected try_invoke(F&& f) noexcept {
        try {
            if constexpr (std::is_same_v<std::invoke_result_t<F>, void>) {
                std::forward<F>(f)();
                return Expected(T{});
            } else {
                return Expected(std::forward<F>(f)());
            }
        } catch (...) {
            return Expected(std::current_exception());
        }
    }

    /**
     * @brief Try to invoke a function with arguments
     */
    template<typename F, typename... Args>
        requires std::invocable<F, Args...>
    static Expected try_invoke(F&& f, Args&&... args) noexcept {
        try {
            if constexpr (std::is_same_v<std::invoke_result_t<F, Args...>, void>) {
                std::forward<F>(f)(std::forward<Args>(args)...);
                return Expected(T{});
            } else {
                return Expected(std::forward<F>(f)(std::forward<Args>(args)...));
            }
        } catch (...) {
            return Expected(std::current_exception());
        }
    }

    // State queries
    bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    bool has_error() const noexcept {
        return std::holds_alternative<std::exception_ptr>(data_);
    }

    explicit operator bool() const noexcept {
        return has_value();
    }

    // Value access
    T& value() & {
        if (has_error()) {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        }
        return std::get<T>(data_);
    }

    const T& value() const& {
        if (has_error()) {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        }
        return std::get<T>(data_);
    }

    T&& value() && {
        if (has_error()) {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        }
        return std::get<T>(std::move(data_));
    }

    /**
     * @brief Get value or throw with custom message
     */
    T& get() {
        return value();
    }

    const T& get() const {
        return value();
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

    // Error access
    std::exception_ptr error() const {
        if (!has_error()) {
            throw LogicError("Expected has no error");
        }
        return std::get<std::exception_ptr>(data_);
    }

    /**
     * @brief Rethrow the contained exception
     */
    [[noreturn]] void rethrow() const {
        if (has_error()) {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        } else {
            throw LogicError("Expected has no error to rethrow");
        }
    }

    // Monadic operations
    template<typename F>
        requires std::invocable<F, T>
    auto map(F&& f) -> Expected<std::invoke_result_t<F, T>> {
        using U = std::invoke_result_t<F, T>;
        if (has_value()) {
            return Expected<U>::try_invoke([&]() {
                return std::forward<F>(f)(std::get<T>(data_));
            });
        }
        return Expected<U>(
            std::get<std::exception_ptr>(data_)
        );
    }

    template<typename F>
        requires std::invocable<F, T>
    auto and_then(F&& f) -> std::invoke_result_t<F, T> {
        if (has_value()) {
            return std::forward<F>(f)(std::get<T>(data_));
        }
        using ResultType = std::invoke_result_t<F, T>;
        return ResultType(std::get<std::exception_ptr>(data_));
    }

    template<typename F>
        requires std::invocable<F, std::exception_ptr>
    auto or_else(F&& f) -> Expected<T> {
        if (has_error()) {
            try {
                return Expected(std::forward<F>(f)(
                    std::get<std::exception_ptr>(data_)
                ));
            } catch (...) {
                return Expected(std::current_exception());
            }
        }
        return *this;
    }

    /**
     * @brief Transform error to Result
     */
    template<typename E = ErrorCode>
    Result<T, E> to_result() const {
        if (has_value()) {
            return Ok<T, E>(T(std::get<T>(data_)));  // Copy the value
        }

        try {
            std::rethrow_exception(std::get<std::exception_ptr>(data_));
        } catch (const Exception& e) {
            if constexpr (std::is_same_v<E, ErrorCode>) {
                return Err<E, T>(e.code());
            } else {
                return Err<E, T>(E{});
            }
        } catch (const std::exception&) {
            if constexpr (std::is_same_v<E, ErrorCode>) {
                return Err<E, T>(ErrorCode::Unknown);
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
};

// Deduction guides
template<typename T>
Expected(T) -> Expected<T>;

/**
 * @brief Convenience function for creating successful Expected
 */
template<typename T>
auto make_expected(T&& value) {
    return Expected<std::decay_t<T>>(std::forward<T>(value));
}

/**
 * @brief Convenience function for creating failed Expected
 */
template<typename T = void>
auto make_expected_error(std::exception_ptr error) {
    if constexpr (std::is_void_v<T>) {
        return Expected<std::monostate>(error);
    } else {
        return Expected<T>(error);
    }
}

} // namespace fem::core::error

#endif // CORE_ERROR_EXPECTED_H
#pragma once

#ifndef CORE_ERROR_RESULT_H
#define CORE_ERROR_RESULT_H

#include <variant>
#include <utility>
#include <type_traits>
#include <concepts>
#include <source_location>
#include <string_view>
#include <format>

namespace fem::core {

// Forward declarations
template<typename T, typename E> class Result;
template<typename E> class Error;

// Concept for error types
template<typename E>
concept ErrorType = requires(E e) {
    { e.code() } -> std::convertible_to<int>;
    { e.message() } -> std::convertible_to<std::string_view>;
};

/**
 * @brief Error wrapper for Result types
 * 
 * Provides explicit error construction to avoid ambiguity
 */
template<typename E>
class Error {
public:
    explicit Error(E error) : error_(std::move(error)) {}
    
    const E& get() const noexcept { return error_; }
    E& get() noexcept { return error_; }
    
private:
    E error_;
};

/**
 * @brief Result type for error handling without exceptions
 * 
 * Similar to Rust's Result<T,E> or C++23's std::expected<T,E>
 * Provides monadic operations for error propagation
 * 
 * @tparam T Value type
 * @tparam E Error type (must satisfy ErrorType concept)
 */
template<typename T, typename E>
    requires ErrorType<E>
class Result {
public:
    using value_type = T;
    using error_type = E;
    
private:
    std::variant<T, E> data_;
    
public:
    // Constructors
    Result(T value) : data_(std::move(value)) {}
    Result(Error<E> error) : data_(std::move(error.get())) {}
    
    // Special members
    Result(const Result&) = default;
    Result(Result&&) noexcept = default;
    Result& operator=(const Result&) = default;
    Result& operator=(Result&&) noexcept = default;
    ~Result() = default;
    
    // State queries
    [[nodiscard]] bool is_ok() const noexcept {
        return std::holds_alternative<T>(data_);
    }
    
    [[nodiscard]] bool is_error() const noexcept {
        return std::holds_alternative<E>(data_);
    }
    
    [[nodiscard]] explicit operator bool() const noexcept {
        return is_ok();
    }
    
    // Value access
    [[nodiscard]] const T& value() const& {
        if (!is_ok()) {
            throw std::runtime_error(std::format("Result contains error: {}", 
                                                  error().message()));
        }
        return std::get<T>(data_);
    }
    
    [[nodiscard]] T& value() & {
        if (!is_ok()) {
            throw std::runtime_error(std::format("Result contains error: {}", 
                                                  error().message()));
        }
        return std::get<T>(data_);
    }
    
    [[nodiscard]] T&& value() && {
        if (!is_ok()) {
            throw std::runtime_error(std::format("Result contains error: {}", 
                                                  error().message()));
        }
        return std::get<T>(std::move(data_));
    }
    
    [[nodiscard]] const T& value_or(const T& default_value) const& noexcept {
        return is_ok() ? std::get<T>(data_) : default_value;
    }
    
    // Error access
    [[nodiscard]] const E& error() const& {
        if (!is_error()) {
            std::terminate(); // Logic error - checking error on success
        }
        return std::get<E>(data_);
    }
    
    [[nodiscard]] E& error() & {
        if (!is_error()) {
            std::terminate(); // Logic error - checking error on success
        }
        return std::get<E>(data_);
    }
    
    // Pointer-like access
    [[nodiscard]] const T* operator->() const {
        return is_ok() ? &std::get<T>(data_) : nullptr;
    }
    
    [[nodiscard]] T* operator->() {
        return is_ok() ? &std::get<T>(data_) : nullptr;
    }
    
    [[nodiscard]] const T& operator*() const& {
        return value();
    }
    
    [[nodiscard]] T& operator*() & {
        return value();
    }
    
    [[nodiscard]] T&& operator*() && {
        return std::move(*this).value();
    }
    
    // Monadic operations
    template<typename F>
        requires std::invocable<F, T>
    auto map(F&& f) const -> Result<std::invoke_result_t<F, T>, E> {
        if (is_ok()) {
            return Result<std::invoke_result_t<F, T>, E>(
                std::forward<F>(f)(std::get<T>(data_))
            );
        }
        return Error<E>(std::get<E>(data_));
    }
    
    template<typename F>
        requires std::invocable<F, T>
    auto and_then(F&& f) const -> std::invoke_result_t<F, T> {
        static_assert(
            std::same_as<typename std::invoke_result_t<F, T>::error_type, E>,
            "and_then function must return Result with same error type"
        );
        
        if (is_ok()) {
            return std::forward<F>(f)(std::get<T>(data_));
        }
        return Error<E>(std::get<E>(data_));
    }
    
    template<typename F>
        requires std::invocable<F, E>
    auto or_else(F&& f) const -> Result<T, std::invoke_result_t<F, E>> {
        if (is_error()) {
            return Error<std::invoke_result_t<F, E>>(
                std::forward<F>(f)(std::get<E>(data_))
            );
        }
        return std::get<T>(data_);
    }
};

// Deduction guides
template<typename T, typename E>
Result(Error<E>) -> Result<T, E>;

// Helper macro for early return on error (similar to Rust's ? operator)
#define TRY(expr) \
    do { \
        auto _result = (expr); \
        if (!_result) { \
            return Error{_result.error()}; \
        } \
    } while(0)

// Convenience function for creating success results
template<typename T, typename E = std::error_code>
[[nodiscard]] auto Ok(T&& value) {
    return Result<std::decay_t<T>, E>(std::forward<T>(value));
}

// Convenience function for creating error results  
template<typename E, typename T = void>
[[nodiscard]] auto Err(E&& error) {
    if constexpr (std::is_void_v<T>) {
        // Type will be deduced from context
        return Error<std::decay_t<E>>(std::forward<E>(error));
    } else {
        return Result<T, std::decay_t<E>>(Error<std::decay_t<E>>(std::forward<E>(error)));
    }
}

} // namespace fem::core

#endif // CORE_ERROR_RESULT_H
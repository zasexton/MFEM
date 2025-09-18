#pragma once

#ifndef CORE_ERROR_PRECONDITION_H
#define CORE_ERROR_PRECONDITION_H

#include <string>
#include <functional>
#include <type_traits>
#include <concepts>
#include <format>
#include <source_location>
#include "logic_error.h"
#include "contract.h"
#include "validation.h"

namespace fem::core::error {

/**
 * @brief Precondition checking utilities
 *
 * Provides comprehensive precondition validation for:
 * - Function arguments
 * - Object state
 * - Resource availability
 * - Type constraints
 */

/**
 * @brief Precondition checker with fluent interface
 */
class Precondition {
public:
    /**
     * @brief Check that a pointer is not null
     */
    template<typename T>
    static void require_not_null(const T* ptr,
                                 const std::string& name,
                                 const std::source_location& loc = std::source_location::current()) {
        if (ptr == nullptr) {
            throw NullPointerError(name, loc);
        }
    }

    /**
     * @brief Check that a value is in range
     */
    template<typename T>
    static void require_in_range(T value, T min, T max,
                                 const std::string& name,
                                 const std::source_location& loc = std::source_location::current()) {
        if (value < min || value > max) {
            throw OutOfRangeError(name, value, min, max, loc);
        }
    }

    /**
     * @brief Check that a container is not empty
     */
    template<typename Container>
    static void require_not_empty(const Container& container,
                                  const std::string& name,
                                  const std::source_location& loc = std::source_location::current()) {
        if (container.empty()) {
            throw InvalidArgumentError(name, "Container must not be empty", loc);
        }
    }

    /**
     * @brief Check that a value is positive
     */
    template<typename T>
        requires std::is_arithmetic_v<T>
    static void require_positive(T value,
                                 const std::string& name,
                                 const std::source_location& loc = std::source_location::current()) {
        if (value <= 0) {
            throw InvalidArgumentError(name, value,
                                      "Value must be positive", loc);
        }
    }

    /**
     * @brief Check that a value is non-negative
     */
    template<typename T>
        requires std::is_arithmetic_v<T>
    static void require_non_negative(T value,
                                     const std::string& name,
                                     const std::source_location& loc = std::source_location::current()) {
        if (value < 0) {
            throw InvalidArgumentError(name, value,
                                      "Value must be non-negative", loc);
        }
    }

    /**
     * @brief Check that a value is finite (not NaN or infinite)
     */
    template<typename T>
        requires std::is_floating_point_v<T>
    static void require_finite(T value,
                               const std::string& name,
                               const std::source_location& loc = std::source_location::current()) {
        if (!std::isfinite(value)) {
            throw InvalidArgumentError(name, 
                                      std::format("Value {} is not finite", value),
                                      "Value must be finite", loc);
        }
    }

    /**
     * @brief Check array index bounds
     */
    static void require_valid_index(size_t index, size_t size,
                                   const std::string& name,
                                   const std::source_location& loc = std::source_location::current()) {
        if (index >= size) {
            throw OutOfRangeError(name, index, size, loc);
        }
    }

    /**
     * @brief Check that two sizes match
     */
    static void require_size_match(size_t size1, size_t size2,
                                  const std::string& name1,
                                  const std::string& name2,
                                  const std::source_location& loc = std::source_location::current()) {
        if (size1 != size2) {
            throw InvalidArgumentError(
                std::format("{} and {}", name1, name2),
                std::format("Size mismatch: {} != {}", size1, size2),
                loc);
        }
    }

    /**
     * @brief Check that a string is not empty
     */
    static void require_not_empty_string(const std::string& str,
                                        const std::string& name,
                                        const std::source_location& loc = std::source_location::current()) {
        if (str.empty()) {
            throw InvalidArgumentError(name, "String must not be empty", loc);
        }
    }

    /**
     * @brief Check custom condition
     */
    static void require(bool condition,
                       const std::string& message,
                       const std::source_location& loc = std::source_location::current()) {
        if (!condition) {
            throw PreconditionError("<unnamed>", "condition", message, loc);
        }
    }
};

/**
 * @brief Precondition builder for complex validations
 */
template<typename T>
class PreconditionBuilder {
public:
    explicit PreconditionBuilder(const T& value, const std::string& name)
        : value_(value), name_(name) {}

    /**
     * @brief Check not null (for pointers)
     */
    PreconditionBuilder& not_null() requires std::is_pointer_v<T> {
        if (value_ == nullptr) {
            violations_.push_back("is null");
        }
        return *this;
    }

    /**
     * @brief Check in range
     */
    template<typename U = T>
        requires std::totally_ordered<U>
    PreconditionBuilder& in_range(U min, U max) {
        if (value_ < min || value_ > max) {
            violations_.push_back(
                std::format("out of range [{}, {}]", min, max));
        }
        return *this;
    }

    /**
     * @brief Check positive
     */
    PreconditionBuilder& positive() requires std::is_arithmetic_v<T> {
        if (value_ <= 0) {
            violations_.push_back("not positive");
        }
        return *this;
    }

    /**
     * @brief Check non-negative
     */
    PreconditionBuilder& non_negative() requires std::is_arithmetic_v<T> {
        if (value_ < 0) {
            violations_.push_back("negative");
        }
        return *this;
    }

    /**
     * @brief Check finite
     */
    PreconditionBuilder& finite() requires std::is_floating_point_v<T> {
        if (!std::isfinite(value_)) {
            violations_.push_back("not finite");
        }
        return *this;
    }

    /**
     * @brief Check custom predicate
     */
    PreconditionBuilder& satisfies(std::function<bool(const T&)> predicate,
                                  const std::string& description) {
        if (!predicate(value_)) {
            violations_.push_back(description);
        }
        return *this;
    }

    /**
     * @brief Check equals
     */
    template<typename U>
        requires std::equality_comparable_with<T, U>
    PreconditionBuilder& equals(const U& expected) {
        if (value_ != expected) {
            violations_.push_back(
                std::format("not equal to expected value"));
        }
        return *this;
    }

    /**
     * @brief Check not equals
     */
    template<typename U>
        requires std::equality_comparable_with<T, U>
    PreconditionBuilder& not_equals(const U& forbidden) {
        if (value_ == forbidden) {
            violations_.push_back(
                std::format("equal to forbidden value"));
        }
        return *this;
    }

    /**
     * @brief Validate and throw if violations exist
     */
    void validate(const std::source_location& loc = std::source_location::current()) {
        if (!violations_.empty()) {
            std::string message = std::format("Precondition violations for '{}':", name_);
            for (const auto& violation : violations_) {
                message += "\n  - " + violation;
            }
            throw PreconditionError("<function>", name_, message, loc);
        }
    }

    /**
     * @brief Check if valid
     */
    bool is_valid() const {
        return violations_.empty();
    }

    /**
     * @brief Get violations
     */
    const std::vector<std::string>& violations() const {
        return violations_;
    }

private:
    const T& value_;
    std::string name_;
    std::vector<std::string> violations_;
};

/**
 * @brief Function argument validator
 */
class ArgumentValidator {
public:
    explicit ArgumentValidator(const std::string& function_name)
        : function_name_(function_name) {}

    /**
     * @brief Validate an argument
     */
    template<typename T>
    PreconditionBuilder<T> arg(const T& value, const std::string& name) {
        return PreconditionBuilder<T>(value, name);
    }

    /**
     * @brief Batch validation
     */
    template<typename... Validators>
    void validate_all(Validators&&... validators) {
        (validators.validate(), ...);
    }

private:
    std::string function_name_;
};

/**
 * @brief Method precondition checker
 */
class MethodPrecondition {
public:
    explicit MethodPrecondition(const std::string& class_name,
                               const std::string& method_name)
        : class_name_(class_name)
        , method_name_(method_name) {}

    /**
     * @brief Check object state
     */
    void require_state(bool condition,
                      const std::string& expected_state,
                      const std::source_location& loc = std::source_location::current()) {
        if (!condition) {
            throw InvalidStateError(
                std::format("{}::{}", class_name_, method_name_),
                expected_state,
                "Invalid state for method call",
                loc);
        }
    }

    /**
     * @brief Check initialized
     */
    void require_initialized(bool initialized,
                           const std::source_location& loc = std::source_location::current()) {
        if (!initialized) {
            throw InvalidStateError(
                std::format("{}::{}", class_name_, method_name_),
                "initialized",
                "uninitialized",
                loc);
        }
    }

private:
    std::string class_name_;
    std::string method_name_;
};

/**
 * @brief Numeric preconditions
 */
class NumericPrecondition {
public:
    /**
     * @brief Check matrix dimensions for multiplication
     */
    static void require_multiplicable(size_t rows1, size_t cols1,
                                     size_t rows2, size_t cols2,
                                     const std::source_location& loc = std::source_location::current()) {
        if (cols1 != rows2) {
            throw InvalidArgumentError(
                "Matrix dimensions",
                std::format("Cannot multiply {}x{} by {}x{} matrices",
                          rows1, cols1, rows2, cols2),
                loc);
        }
    }

    /**
     * @brief Check square matrix
     */
    static void require_square(size_t rows, size_t cols,
                              const std::string& name,
                              const std::source_location& loc = std::source_location::current()) {
        if (rows != cols) {
            throw InvalidArgumentError(
                name,
                std::format("Matrix must be square, got {}x{}", rows, cols),
                loc);
        }
    }

    /**
     * @brief Check positive definite (simplified check)
     */
    template<typename Matrix>
    static void require_positive_definite(const Matrix& mat,
                                         const std::string& name,
                                         const std::source_location& loc = std::source_location::current()) {
        // This would need actual eigenvalue computation
        // Simplified: just check diagonal dominance as a proxy
        for (size_t i = 0; i < mat.rows(); ++i) {
            if (mat(i, i) <= 0) {
                throw InvalidArgumentError(
                    name,
                    "Matrix must be positive definite",
                    loc);
            }
        }
    }
};

/**
 * @brief Helper functions
 */

// Create precondition builder
template<typename T>
auto check_arg(const T& value, const std::string& name) {
    return PreconditionBuilder<T>(value, name);
}

// Quick precondition check
template<typename T>
void require_arg(const T& value, const std::string& name,
                std::function<bool(const T&)> predicate,
                const std::string& message) {
    if (!predicate(value)) {
        throw InvalidArgumentError(name, message);
    }
}

} // namespace fem::core::error

#endif // CORE_ERROR_PRECONDITION_H
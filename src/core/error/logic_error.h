#pragma once

#ifndef CORE_ERROR_LOGIC_ERROR_H
#define CORE_ERROR_LOGIC_ERROR_H

#include "exception_base.h"
#include <type_traits>
#include <typeinfo>

namespace fem::core::error {

/**
 * @brief Base class for logic errors (programming errors)
 *
 * Logic errors represent errors in the program logic that should be
 * caught during development. These are typically bugs, not runtime conditions.
 */
class LogicError : public Exception {
public:
    using Exception::Exception;

    explicit LogicError(const std::string& message,
                       const std::source_location& loc = std::source_location::current())
        : Exception(message, ErrorCode::InvalidArgument, loc) {
    }
};

/**
 * @brief Invalid argument error
 */
class InvalidArgumentError : public LogicError {
public:
    InvalidArgumentError(const std::string& argument_name,
                        const std::string& reason,
                        const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Invalid argument '{}': {}",
                                argument_name, reason), loc)
        , argument_name_(argument_name) {
    }

    template<typename T>
    InvalidArgumentError(const std::string& argument_name,
                        const T& value,
                        const std::string& reason,
                        const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Invalid argument '{}' = {}: {}",
                                argument_name, value, reason), loc)
        , argument_name_(argument_name) {
    }

    const std::string& argument_name() const noexcept { return argument_name_; }

private:
    std::string argument_name_;
};

/**
 * @brief Domain error (mathematical domain violation)
 */
class DomainError : public LogicError {
public:
    DomainError(const std::string& function_name,
               const std::string& reason,
               const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Domain error in {}: {}",
                                function_name, reason), loc)
        , function_name_(function_name) {
    }

    const std::string& function_name() const noexcept { return function_name_; }

private:
    std::string function_name_;
};

/**
 * @brief Length/size error
 */
class LengthError : public LogicError {
public:
    LengthError(const std::string& what,
               size_t requested,
               size_t maximum,
               const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("{}: requested size {} exceeds maximum {}",
                                what, requested, maximum), loc)
        , requested_(requested)
        , maximum_(maximum) {
    }

    size_t requested() const noexcept { return requested_; }
    size_t maximum() const noexcept { return maximum_; }

private:
    size_t requested_;
    size_t maximum_;
};

/**
 * @brief Out of range error
 */
class OutOfRangeError : public LogicError {
public:
    OutOfRangeError(const std::string& what,
                    size_t index,
                    size_t size,
                    const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("{}: index {} out of range [0, {})",
                                what, index, size),
                    ErrorCode::OutOfRange, loc)
        , index_(index)
        , size_(size) {
    }

    template<typename T>
    OutOfRangeError(const std::string& what,
                    T value,
                    T min,
                    T max,
                    const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("{}: value {} out of range [{}, {}]",
                                what, value, min, max),
                    ErrorCode::OutOfRange, loc)
        , index_(0)
        , size_(0) {
    }

    size_t index() const noexcept { return index_; }
    size_t size() const noexcept { return size_; }

private:
    size_t index_;
    size_t size_;
};

/**
 * @brief Invalid state error
 */
class InvalidStateError : public LogicError {
public:
    InvalidStateError(const std::string& object_name,
                     const std::string& expected_state,
                     const std::string& actual_state,
                     const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("{}: invalid state '{}', expected '{}'",
                                object_name, actual_state, expected_state),
                    ErrorCode::InvalidState, loc)
        , object_name_(object_name)
        , expected_state_(expected_state)
        , actual_state_(actual_state) {
    }

    const std::string& object_name() const noexcept { return object_name_; }
    const std::string& expected_state() const noexcept { return expected_state_; }
    const std::string& actual_state() const noexcept { return actual_state_; }

private:
    std::string object_name_;
    std::string expected_state_;
    std::string actual_state_;
};

/**
 * @brief Type mismatch error
 */
class TypeMismatchError : public LogicError {
public:
    TypeMismatchError(const std::string& context,
                     const std::type_info& expected,
                     const std::type_info& actual,
                     const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("{}: type mismatch - expected '{}', got '{}'",
                                context, expected.name(), actual.name()),
                    ErrorCode::TypeMismatch, loc)
        , expected_type_(&expected)
        , actual_type_(&actual) {
    }

    const std::type_info& expected_type() const noexcept { return *expected_type_; }
    const std::type_info& actual_type() const noexcept { return *actual_type_; }

private:
    const std::type_info* expected_type_;
    const std::type_info* actual_type_;
};

/**
 * @brief Null pointer error
 */
class NullPointerError : public LogicError {
public:
    explicit NullPointerError(const std::string& pointer_name,
                             const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Null pointer: '{}'", pointer_name), loc)
        , pointer_name_(pointer_name) {
    }

    const std::string& pointer_name() const noexcept { return pointer_name_; }

private:
    std::string pointer_name_;
};

/**
 * @brief Not implemented error
 */
class NotImplementedError : public LogicError {
public:
    explicit NotImplementedError(const std::string& feature,
                                const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Not implemented: {}", feature),
                    ErrorCode::NotImplemented, loc)
        , feature_(feature) {
    }

    const std::string& feature() const noexcept { return feature_; }

private:
    std::string feature_;
};

/**
 * @brief Precondition violation error
 */
class PreconditionError : public LogicError {
public:
    PreconditionError(const std::string& function_name,
                     const std::string& condition,
                     const std::string& message = "",
                     const std::source_location& loc = std::source_location::current())
        : LogicError(format_message(function_name, condition, message), loc)
        , function_name_(function_name)
        , condition_(condition) {
    }

    const std::string& function_name() const noexcept { return function_name_; }
    const std::string& condition() const noexcept { return condition_; }

private:
    static std::string format_message(const std::string& function_name,
                                     const std::string& condition,
                                     const std::string& message) {
        if (message.empty()) {
            return std::format("Precondition violation in '{}': {}",
                             function_name, condition);
        }
        return std::format("Precondition violation in '{}': {} - {}",
                         function_name, condition, message);
    }

    std::string function_name_;
    std::string condition_;
};

/**
 * @brief Postcondition violation error
 */
class PostconditionError : public LogicError {
public:
    PostconditionError(const std::string& function_name,
                      const std::string& condition,
                      const std::string& message = "",
                      const std::source_location& loc = std::source_location::current())
        : LogicError(format_message(function_name, condition, message), loc)
        , function_name_(function_name)
        , condition_(condition) {
    }

    const std::string& function_name() const noexcept { return function_name_; }
    const std::string& condition() const noexcept { return condition_; }

private:
    static std::string format_message(const std::string& function_name,
                                     const std::string& condition,
                                     const std::string& message) {
        if (message.empty()) {
            return std::format("Postcondition violation in '{}': {}",
                             function_name, condition);
        }
        return std::format("Postcondition violation in '{}': {} - {}",
                         function_name, condition, message);
    }

    std::string function_name_;
    std::string condition_;
};

/**
 * @brief Invariant violation error
 */
class InvariantError : public LogicError {
public:
    InvariantError(const std::string& class_name,
                  const std::string& invariant,
                  const std::string& message = "",
                  const std::source_location& loc = std::source_location::current())
        : LogicError(format_message(class_name, invariant, message), loc)
        , class_name_(class_name)
        , invariant_(invariant) {
    }

    const std::string& class_name() const noexcept { return class_name_; }
    const std::string& invariant() const noexcept { return invariant_; }

private:
    static std::string format_message(const std::string& class_name,
                                     const std::string& invariant,
                                     const std::string& message) {
        if (message.empty()) {
            return std::format("Invariant violation in '{}': {}",
                             class_name, invariant);
        }
        return std::format("Invariant violation in '{}': {} - {}",
                         class_name, invariant, message);
    }

    std::string class_name_;
    std::string invariant_;
};

} // namespace fem::core::error

#endif // CORE_ERROR_LOGIC_ERROR_H
#pragma once

#ifndef CORE_ERROR_VALIDATION_H
#define CORE_ERROR_VALIDATION_H

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <regex>
#include <limits>
#include <cmath>
#include "status.h"
#include "error_chain.h"
#include "logic_error.h"

namespace fem::core::error {

/**
 * @brief Input validation utilities
 *
 * Provides a fluent interface for validating input parameters
 * without using assertions
 */
template<typename T>
class Validator {
public:
    explicit Validator(const T& value, const std::string& name = "value")
        : value_(value), name_(name) {}

    /**
     * @brief Check if value is not null (for pointers)
     */
    Validator& not_null() requires std::is_pointer_v<T> {
        if (value_ == nullptr) {
            add_error("must not be null");
        }
        return *this;
    }

    /**
     * @brief Check if value is in range
     */
    Validator& in_range(T min, T max) requires std::is_arithmetic_v<T> {
        if (value_ < min || value_ > max) {
            add_error(std::format("must be in range [{}, {}], got {}",
                                 min, max, value_));
        }
        return *this;
    }

    /**
     * @brief Check if value is positive
     */
    Validator& positive() requires std::is_arithmetic_v<T> {
        if (value_ <= 0) {
            add_error(std::format("must be positive, got {}", value_));
        }
        return *this;
    }

    /**
     * @brief Check if value is non-negative
     */
    Validator& non_negative() requires std::is_arithmetic_v<T> {
        if (value_ < 0) {
            add_error(std::format("must be non-negative, got {}", value_));
        }
        return *this;
    }

    /**
     * @brief Check if value is finite (for floating point)
     */
    Validator& finite() requires std::is_floating_point_v<T> {
        if (!std::isfinite(value_)) {
            add_error("must be finite");
        }
        return *this;
    }

    /**
     * @brief Custom validation
     */
    Validator& satisfies(std::function<bool(const T&)> predicate,
                        const std::string& message) {
        if (!predicate(value_)) {
            add_error(message);
        }
        return *this;
    }

    /**
     * @brief Get validation status
     */
    Status status() const {
        if (errors_.empty()) {
            return Status::OK();
        }

        std::string message = std::format("{}: ", name_);
        for (size_t i = 0; i < errors_.size(); ++i) {
            if (i > 0) message += "; ";
            message += errors_[i];
        }
        return Status(ErrorCode::InvalidArgument, message);
    }

    /**
     * @brief Check if validation passed
     */
    bool is_valid() const {
        return errors_.empty();
    }

    /**
     * @brief Throw if validation failed
     */
    void throw_if_invalid() const {
        auto s = status();
        if (!s.ok()) {
            throw InvalidArgumentError(name_, std::string(s.message()));
        }
    }

private:
    void add_error(const std::string& error) {
        errors_.push_back(error);
    }

    const T& value_;
    std::string name_;
    std::vector<std::string> errors_;
};

/**
 * @brief String validation utilities
 */
class StringValidator {
public:
    explicit StringValidator(const std::string& value,
                            const std::string& name = "string")
        : value_(value), name_(name) {}

    StringValidator& not_empty() {
        if (value_.empty()) {
            add_error("must not be empty");
        }
        return *this;
    }

    StringValidator& min_length(size_t length) {
        if (value_.length() < length) {
            add_error(std::format("must be at least {} characters", length));
        }
        return *this;
    }

    StringValidator& max_length(size_t length) {
        if (value_.length() > length) {
            add_error(std::format("must be at most {} characters", length));
        }
        return *this;
    }

    StringValidator& matches(const std::regex& pattern,
                            const std::string& description = "pattern") {
        if (!std::regex_match(value_, pattern)) {
            add_error(std::format("must match {}", description));
        }
        return *this;
    }

    StringValidator& one_of(const std::vector<std::string>& values) {
        if (std::find(values.begin(), values.end(), value_) == values.end()) {
            add_error("must be one of the allowed values");
        }
        return *this;
    }

    Status status() const {
        if (errors_.empty()) {
            return Status::OK();
        }

        std::string message = std::format("{}: ", name_);
        for (size_t i = 0; i < errors_.size(); ++i) {
            if (i > 0) message += "; ";
            message += errors_[i];
        }
        return Status(ErrorCode::InvalidArgument, message);
    }

    bool is_valid() const {
        return errors_.empty();
    }

    void throw_if_invalid() const {
        auto s = status();
        if (!s.ok()) {
            throw InvalidArgumentError(name_, std::string(s.message()));
        }
    }

private:
    void add_error(const std::string& error) {
        errors_.push_back(error);
    }

    const std::string& value_;
    std::string name_;
    std::vector<std::string> errors_;
};

/**
 * @brief Collection validation
 */
template<typename Container>
class CollectionValidator {
public:
    explicit CollectionValidator(const Container& container,
                                 const std::string& name = "collection")
        : container_(container), name_(name) {}

    CollectionValidator& not_empty() {
        if (container_.empty()) {
            add_error("must not be empty");
        }
        return *this;
    }

    CollectionValidator& size(size_t expected) {
        if (container_.size() != expected) {
            add_error(std::format("must have size {}, got {}",
                                 expected, container_.size()));
        }
        return *this;
    }

    CollectionValidator& min_size(size_t min) {
        if (container_.size() < min) {
            add_error(std::format("must have at least {} elements, got {}",
                                 min, container_.size()));
        }
        return *this;
    }

    CollectionValidator& max_size(size_t max) {
        if (container_.size() > max) {
            add_error(std::format("must have at most {} elements, got {}",
                                 max, container_.size()));
        }
        return *this;
    }

    template<typename Predicate>
    CollectionValidator& all_satisfy(Predicate pred,
                                     const std::string& message) {
        for (const auto& item : container_) {
            if (!pred(item)) {
                add_error(std::format("all elements must {}", message));
                break;
            }
        }
        return *this;
    }

    Status status() const {
        if (errors_.empty()) {
            return Status::OK();
        }

        std::string message = std::format("{}: ", name_);
        for (size_t i = 0; i < errors_.size(); ++i) {
            if (i > 0) message += "; ";
            message += errors_[i];
        }
        return Status(ErrorCode::InvalidArgument, message);
    }

    bool is_valid() const {
        return errors_.empty();
    }

    void throw_if_invalid() const {
        auto s = status();
        if (!s.ok()) {
            throw InvalidArgumentError(name_, std::string(s.message()));
        }
    }

private:
    void add_error(const std::string& error) {
        errors_.push_back(error);
    }

    const Container& container_;
    std::string name_;
    std::vector<std::string> errors_;
};

/**
 * @brief Factory functions for validators
 */
template<typename T>
auto validate(const T& value, const std::string& name = "value") {
    return Validator<T>(value, name);
}

inline auto validate_string(const std::string& value,
                           const std::string& name = "string") {
    return StringValidator(value, name);
}

template<typename Container>
auto validate_collection(const Container& container,
                        const std::string& name = "collection") {
    return CollectionValidator<Container>(container, name);
}

/**
 * @brief Batch validation
 *
 * Validates multiple values and collects all errors
 */
class BatchValidator {
public:
    BatchValidator() = default;

    template<typename T>
    BatchValidator& add(const Validator<T>& validator) {
        auto status = validator.status();
        if (!status.ok()) {
            chain_.add_error(status.code(), status.message());
        }
        return *this;
    }

    BatchValidator& add(const StringValidator& validator) {
        auto status = validator.status();
        if (!status.ok()) {
            chain_.add_error(status.code(), status.message());
        }
        return *this;
    }

    template<typename Container>
    BatchValidator& add(const CollectionValidator<Container>& validator) {
        auto status = validator.status();
        if (!status.ok()) {
            chain_.add_error(status.code(), status.message());
        }
        return *this;
    }

    Status status() const {
        if (!chain_.has_errors()) {
            return Status::OK();
        }
        return Status(ErrorCode::InvalidArgument, chain_.format());
    }

    bool is_valid() const {
        return !chain_.has_errors();
    }

    void throw_if_invalid() const {
        chain_.throw_if_errors();
    }

private:
    ErrorChain chain_;
};

} // namespace fem::core::error

#endif // CORE_ERROR_VALIDATION_H
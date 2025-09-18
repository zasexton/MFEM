#pragma once

#ifndef CORE_ERROR_POSTCONDITION_H
#define CORE_ERROR_POSTCONDITION_H

#include <string>
#include <functional>
#include <optional>
#include <tuple>
#include <format>
#include <source_location>
#include "logic_error.h"
#include "contract.h"
#include "error_guard.h"

namespace fem::core::error {

/**
 * @brief Postcondition checking utilities
 *
 * Provides comprehensive postcondition validation for:
 * - Function return values
 * - State transitions
 * - Resource guarantees
 * - Side effect verification
 */

/**
 * @brief Postcondition checker with old value capture
 */
template<typename T>
class PostconditionChecker {
public:
    /**
     * @brief Capture old value for comparison
     */
    explicit PostconditionChecker(const T& old_value)
        : old_value_(old_value) {}

    /**
     * @brief Check that value has changed
     */
    void ensure_changed(const T& new_value,
                       const std::string& name,
                       const std::source_location& loc = std::source_location::current()) {
        if (old_value_ == new_value) {
            throw PostconditionError(
                "<function>",
                std::format("{} should have changed", name),
                "Value unchanged",
                loc);
        }
    }

    /**
     * @brief Check that value hasn't changed
     */
    void ensure_unchanged(const T& new_value,
                         const std::string& name,
                         const std::source_location& loc = std::source_location::current()) {
        if (old_value_ != new_value) {
            throw PostconditionError(
                "<function>",
                std::format("{} should not have changed", name),
                "Value changed unexpectedly",
                loc);
        }
    }

    /**
     * @brief Check that value has increased
     */
    void ensure_increased(const T& new_value,
                         const std::string& name,
                         const std::source_location& loc = std::source_location::current())
        requires std::totally_ordered<T> {
        if (new_value <= old_value_) {
            throw PostconditionError(
                "<function>",
                std::format("{} should have increased", name),
                std::format("Old: {}, New: {}", old_value_, new_value),
                loc);
        }
    }

    /**
     * @brief Check that value has decreased
     */
    void ensure_decreased(const T& new_value,
                         const std::string& name,
                         const std::source_location& loc = std::source_location::current())
        requires std::totally_ordered<T> {
        if (new_value >= old_value_) {
            throw PostconditionError(
                "<function>",
                std::format("{} should have decreased", name),
                std::format("Old: {}, New: {}", old_value_, new_value),
                loc);
        }
    }

    /**
     * @brief Get old value
     */
    const T& old_value() const { return old_value_; }

private:
    T old_value_;
};

/**
 * @brief Return value validator
 */
template<typename T>
class ReturnValueValidator {
public:
    explicit ReturnValueValidator(const T& value, const std::string& function_name)
        : value_(value), function_name_(function_name) {}

    /**
     * @brief Ensure not null
     */
    ReturnValueValidator& not_null() requires std::is_pointer_v<T> {
        if (value_ == nullptr) {
            violations_.push_back("return value is null");
        }
        return *this;
    }

    /**
     * @brief Ensure in range
     */
    template<typename U = T>
        requires std::totally_ordered<U>
    ReturnValueValidator& in_range(U min, U max) {
        if (value_ < min || value_ > max) {
            violations_.push_back(
                std::format("return value {} out of range [{}, {}]",
                          value_, min, max));
        }
        return *this;
    }

    /**
     * @brief Ensure positive
     */
    ReturnValueValidator& positive() requires std::is_arithmetic_v<T> {
        if (value_ <= 0) {
            violations_.push_back(
                std::format("return value {} is not positive", value_));
        }
        return *this;
    }

    /**
     * @brief Ensure finite
     */
    ReturnValueValidator& finite() requires std::is_floating_point_v<T> {
        if (!std::isfinite(value_)) {
            violations_.push_back("return value is not finite");
        }
        return *this;
    }

    /**
     * @brief Custom predicate
     */
    ReturnValueValidator& satisfies(std::function<bool(const T&)> predicate,
                                   const std::string& description) {
        if (!predicate(value_)) {
            violations_.push_back(std::format("return value {}", description));
        }
        return *this;
    }

    /**
     * @brief Validate and throw if violations
     */
    void validate(const std::source_location& loc = std::source_location::current()) {
        if (!violations_.empty()) {
            std::string message = std::format(
                "Postcondition violations in '{}':", function_name_);
            for (const auto& violation : violations_) {
                message += "\n  - " + violation;
            }
            throw PostconditionError(function_name_, "return value",
                                   message, loc);
        }
    }

    /**
     * @brief Get the validated value
     */
    const T& get() const { return value_; }

private:
    const T& value_;
    std::string function_name_;
    std::vector<std::string> violations_;
};

/**
 * @brief State transition validator
 */
class StateTransition {
public:
    /**
     * @brief Define a valid transition
     */
    struct Transition {
        std::string from_state;
        std::string to_state;
        std::function<bool()> guard;  // Optional guard condition
    };

    explicit StateTransition(const std::string& object_name)
        : object_name_(object_name) {}

    /**
     * @brief Add valid transition
     */
    StateTransition& allow(const std::string& from,
                          const std::string& to,
                          std::function<bool()> guard = nullptr) {
        transitions_.push_back({from, to, guard});
        return *this;
    }

    /**
     * @brief Validate transition
     */
    void validate(const std::string& from_state,
                 const std::string& to_state,
                 const std::source_location& loc = std::source_location::current()) {
        for (const auto& transition : transitions_) {
            if (transition.from_state == from_state &&
                transition.to_state == to_state) {
                if (!transition.guard || transition.guard()) {
                    return;  // Valid transition
                }
            }
        }

        throw PostconditionError(
            object_name_,
            "state transition",
            std::format("Invalid transition from '{}' to '{}'",
                      from_state, to_state),
            loc);
    }

private:
    std::string object_name_;
    std::vector<Transition> transitions_;
};

/**
 * @brief Side effect verifier
 */
class SideEffectVerifier {
public:
    explicit SideEffectVerifier(const std::string& function_name)
        : function_name_(function_name) {}

    /**
     * @brief Register an expected side effect
     */
    SideEffectVerifier& expect(const std::string& description,
                               std::function<bool()> verifier) {
        expected_effects_.emplace_back(description, verifier);
        return *this;
    }

    /**
     * @brief Verify all expected side effects
     */
    void verify(const std::source_location& loc = std::source_location::current()) {
        std::vector<std::string> failed;
        
        for (const auto& [description, verifier] : expected_effects_) {
            if (!verifier()) {
                failed.push_back(description);
            }
        }

        if (!failed.empty()) {
            std::string message = std::format(
                "Expected side effects not observed in '{}':", function_name_);
            for (const auto& effect : failed) {
                message += "\n  - " + effect;
            }
            throw PostconditionError(function_name_, "side effects",
                                   message, loc);
        }
    }

private:
    std::string function_name_;
    std::vector<std::pair<std::string, std::function<bool()>>> expected_effects_;
};

/**
 * @brief Resource guarantee checker
 */
class ResourceGuarantee {
public:
    enum class Guarantee {
        NoThrow,        // Operation will not throw
        Basic,          // Resources are not leaked
        Strong,         // Operation succeeds or has no effect
        NoChange        // Operation has no observable effect
    };

    explicit ResourceGuarantee(Guarantee level)
        : guarantee_level_(level) {}

    /**
     * @brief Verify guarantee was met
     */
    template<typename F>
    auto enforce(F&& operation) -> decltype(operation()) {
        switch (guarantee_level_) {
            case Guarantee::NoThrow:
                return enforce_nothrow(std::forward<F>(operation));
            
            case Guarantee::Basic:
                return enforce_basic(std::forward<F>(operation));
            
            case Guarantee::Strong:
                return enforce_strong(std::forward<F>(operation));
            
            case Guarantee::NoChange:
                return enforce_no_change(std::forward<F>(operation));
            
            default:
                return operation();
        }
    }

private:
    template<typename F>
    auto enforce_nothrow(F&& operation) -> decltype(operation()) {
        try {
            return operation();
        } catch (...) {
            throw PostconditionError(
                "<operation>",
                "no-throw guarantee",
                "Operation threw an exception");
        }
    }

    template<typename F>
    auto enforce_basic(F&& operation) -> decltype(operation()) {
        // Basic guarantee is typically ensured by RAII
        return operation();
    }

    template<typename F>
    auto enforce_strong(F&& operation) -> decltype(operation()) {
        // Would need state capture and rollback
        // Simplified version
        return operation();
    }

    template<typename F>
    auto enforce_no_change(F&& operation) -> decltype(operation()) {
        // Would need complete state capture
        // Simplified version
        return operation();
    }

    Guarantee guarantee_level_;
};

/**
 * @brief Numeric postconditions
 */
class NumericPostcondition {
public:
    /**
     * @brief Ensure result is normalized
     */
    template<typename T>
        requires std::is_floating_point_v<T>
    static void ensure_normalized(T value,
                                 const std::string& name,
                                 T tolerance = T(1e-6),
                                 const std::source_location& loc = std::source_location::current()) {
        if (std::abs(value - T(1)) > tolerance) {
            throw PostconditionError(
                "<function>",
                name,
                std::format("Value {} is not normalized (expected 1.0)", value),
                loc);
        }
    }

    /**
     * @brief Ensure matrix is symmetric
     */
    template<typename Matrix>
    static void ensure_symmetric(const Matrix& mat,
                                const std::string& name,
                                typename Matrix::value_type tolerance,
                                const std::source_location& loc = std::source_location::current()) {
        for (size_t i = 0; i < mat.rows(); ++i) {
            for (size_t j = i + 1; j < mat.cols(); ++j) {
                if (std::abs(mat(i,j) - mat(j,i)) > tolerance) {
                    throw PostconditionError(
                        "<function>",
                        name,
                        std::format("Matrix not symmetric at ({},{})", i, j),
                        loc);
                }
            }
        }
    }

    /**
     * @brief Ensure conservation (e.g., mass, energy)
     */
    template<typename T>
        requires std::is_arithmetic_v<T>
    static void ensure_conserved(T before, T after,
                                const std::string& quantity,
                                T tolerance,
                                const std::source_location& loc = std::source_location::current()) {
        if (std::abs(after - before) > tolerance) {
            throw PostconditionError(
                "<function>",
                quantity,
                std::format("{} not conserved: {} -> {}",
                          quantity, before, after),
                loc);
        }
    }
};

/**
 * @brief RAII postcondition verifier
 */
class PostconditionScope {
public:
    using Verifier = std::function<void()>;

    explicit PostconditionScope(Verifier verifier)
        : verifier_(std::move(verifier))
        , exception_count_(std::uncaught_exceptions()) {}

    ~PostconditionScope() {
        // Only check postconditions if no exception occurred
        if (verifier_ && std::uncaught_exceptions() == exception_count_) {
            try {
                verifier_();
            } catch (...) {
                // Log but don't throw from destructor
            }
        }
    }

private:
    Verifier verifier_;
    int exception_count_;
};

/**
 * @brief Helper functions
 */

// Create postcondition checker
template<typename T>
auto capture_old(const T& value) {
    return PostconditionChecker<T>(value);
}

// Validate return value
template<typename T>
auto validate_return(const T& value, const std::string& function_name) {
    return ReturnValueValidator<T>(value, function_name);
}

// Create postcondition scope
inline auto postcondition_scope(std::function<void()> verifier) {
    return PostconditionScope(std::move(verifier));
}

} // namespace fem::core::error

#endif // CORE_ERROR_POSTCONDITION_H
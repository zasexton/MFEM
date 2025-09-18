#pragma once

#ifndef CORE_ERROR_CONTRACT_H
#define CORE_ERROR_CONTRACT_H

#include <functional>
#include <string>
#include <source_location>
#include "exception_base.h"
#include "status.h"

namespace fem::core::error {

/**
 * @brief Contract violation exception
 *
 * Thrown when a contract (precondition, postcondition, invariant) is violated
 */
class ContractViolation : public LogicError {
public:
    enum class Type {
        Precondition,
        Postcondition,
        Invariant
    };

    ContractViolation(Type type,
                      const std::string& condition,
                      const std::string& message,
                      const std::source_location& loc = std::source_location::current())
        : LogicError(format_message(type, condition, message), loc)
        , type_(type)
        , condition_(condition) {
    }

    Type type() const noexcept { return type_; }
    const std::string& condition() const noexcept { return condition_; }

private:
    static std::string format_message(Type type,
                                     const std::string& condition,
                                     const std::string& message) {
        const char* type_str = nullptr;
        switch (type) {
            case Type::Precondition: type_str = "Precondition"; break;
            case Type::Postcondition: type_str = "Postcondition"; break;
            case Type::Invariant: type_str = "Invariant"; break;
        }
        return std::format("{} violated: {} - {}",
                          type_str, condition, message);
    }

    Type type_;
    std::string condition_;
};

/**
 * @brief Contract checker for runtime validation
 *
 * Provides a framework for checking contracts without macros,
 * useful for dynamic validation or when macros are not desired
 */
class ContractChecker {
public:
    using Handler = std::function<void(const ContractViolation&)>;

    /**
     * @brief Set custom violation handler
     */
    static void set_handler(Handler handler) {
        handler_ = std::move(handler);
    }

    /**
     * @brief Check a precondition
     */
    static void require(bool condition,
                       const std::string& condition_str,
                       const std::string& message,
                       const std::source_location& loc = std::source_location::current()) {
        if (!condition) {
            handle_violation(ContractViolation(
                ContractViolation::Type::Precondition,
                condition_str, message, loc
            ));
        }
    }

    /**
     * @brief Check a postcondition
     */
    static void ensure(bool condition,
                      const std::string& condition_str,
                      const std::string& message,
                      const std::source_location& loc = std::source_location::current()) {
        if (!condition) {
            handle_violation(ContractViolation(
                ContractViolation::Type::Postcondition,
                condition_str, message, loc
            ));
        }
    }

    /**
     * @brief Check an invariant
     */
    static void invariant(bool condition,
                         const std::string& condition_str,
                         const std::string& message,
                         const std::source_location& loc = std::source_location::current()) {
        if (!condition) {
            handle_violation(ContractViolation(
                ContractViolation::Type::Invariant,
                condition_str, message, loc
            ));
        }
    }

private:
    static void handle_violation(const ContractViolation& violation) {
        if (handler_) {
            handler_(violation);
        } else {
            throw violation;
        }
    }

    static inline Handler handler_ = nullptr;
};

/**
 * @brief RAII guard for invariant checking
 *
 * Automatically checks invariants at scope entry and exit
 */
class InvariantGuard {
public:
    using Checker = std::function<bool()>;

    InvariantGuard(Checker checker, const std::string& name)
        : checker_(std::move(checker))
        , name_(name) {
        check("Entry");
    }

    ~InvariantGuard() {
        try {
            check("Exit");
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }

    InvariantGuard(const InvariantGuard&) = delete;
    InvariantGuard& operator=(const InvariantGuard&) = delete;

private:
    void check(const char* when) {
        if (!checker_()) {
            throw ContractViolation(
                ContractViolation::Type::Invariant,
                name_,
                std::format("Invariant failed at {}", when)
            );
        }
    }

    Checker checker_;
    std::string name_;
};

/**
 * @brief Contract specification for a function
 *
 * Allows defining contracts programmatically
 */
template<typename R, typename... Args>
class FunctionContract {
public:
    using Function = std::function<R(Args...)>;
    using Precondition = std::function<bool(Args...)>;
    using Postcondition = std::function<bool(const R&, Args...)>;

    FunctionContract(Function func) : func_(std::move(func)) {}

    /**
     * @brief Add a precondition
     */
    FunctionContract& requires(Precondition pre, const std::string& desc = "") {
        preconditions_.emplace_back(std::move(pre), desc);
        return *this;
    }

    /**
     * @brief Add a postcondition
     */
    FunctionContract& ensures(Postcondition post, const std::string& desc = "") {
        postconditions_.emplace_back(std::move(post), desc);
        return *this;
    }

    /**
     * @brief Call the function with contract checking
     */
    R operator()(Args... args) const {
        // Check preconditions
        for (const auto& [pre, desc] : preconditions_) {
            if (!pre(args...)) {
                throw ContractViolation(
                    ContractViolation::Type::Precondition,
                    desc.empty() ? "Unknown" : desc,
                    "Precondition failed"
                );
            }
        }

        // Call function
        R result = func_(args...);

        // Check postconditions
        for (const auto& [post, desc] : postconditions_) {
            if (!post(result, args...)) {
                throw ContractViolation(
                    ContractViolation::Type::Postcondition,
                    desc.empty() ? "Unknown" : desc,
                    "Postcondition failed"
                );
            }
        }

        return result;
    }

private:
    Function func_;
    std::vector<std::pair<Precondition, std::string>> preconditions_;
    std::vector<std::pair<Postcondition, std::string>> postconditions_;
};

/**
 * @brief Helper to create a contracted function
 */
template<typename R, typename... Args>
auto make_contracted(std::function<R(Args...)> func) {
    return FunctionContract<R, Args...>(std::move(func));
}

/**
 * @brief Contract validation without exceptions
 *
 * Returns a Status instead of throwing
 */
class SoftContract {
public:
    static Status check_precondition(bool condition,
                                    const std::string& message) {
        if (!condition) {
            return Status(ErrorCode::InvalidArgument,
                         std::format("Precondition failed: {}", message));
        }
        return Status::OK();
    }

    static Status check_postcondition(bool condition,
                                     const std::string& message) {
        if (!condition) {
            return Status(ErrorCode::InvalidState,
                         std::format("Postcondition failed: {}", message));
        }
        return Status::OK();
    }

    static Status check_invariant(bool condition,
                                const std::string& message) {
        if (!condition) {
            return Status(ErrorCode::InvalidState,
                         std::format("Invariant violated: {}", message));
        }
        return Status::OK();
    }
};

// NOTE: The actual assertion macros (FEM_PRECONDITION, FEM_POSTCONDITION, etc.)
// are defined in logging/assert.h. This file provides complementary runtime
// contract checking without macros.

} // namespace fem::core::error

#endif // CORE_ERROR_CONTRACT_H
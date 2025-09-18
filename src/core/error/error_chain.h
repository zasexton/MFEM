#pragma once

#ifndef CORE_ERROR_ERROR_CHAIN_H
#define CORE_ERROR_ERROR_CHAIN_H

#include <vector>
#include <variant>
#include <string>
#include <sstream>
#include "error_code.h"
#include "exception_base.h"
#include "result.h"

namespace fem::core::error {

/**
 * @brief Chain of errors for aggregating multiple failures
 *
 * Useful for batch operations, validation, or any scenario where
 * multiple errors can occur and all need to be reported.
 */
class ErrorChain {
public:
    using ErrorVariant = std::variant<ErrorInfo, Exception>;

    ErrorChain() = default;

    /**
     * @brief Add an error code to the chain
     */
    void add_error(ErrorCode code, std::string_view context = "") {
        errors_.emplace_back(std::in_place_type<ErrorInfo>,
                           code, context);
    }

    /**
     * @brief Add an error info to the chain
     */
    void add_error(const ErrorInfo& error) {
        errors_.emplace_back(error);
    }

    /**
     * @brief Add an exception to the chain
     */
    void add_error(const Exception& exception) {
        errors_.emplace_back(exception);
    }

    /**
     * @brief Check if there are any errors
     */
    bool has_errors() const noexcept {
        return !errors_.empty();
    }

    /**
     * @brief Check if chain is empty (no errors)
     */
    bool empty() const noexcept {
        return errors_.empty();
    }

    /**
     * @brief Get the number of errors
     */
    size_t error_count() const noexcept {
        return errors_.size();
    }

    /**
     * @brief Clear all errors
     */
    void clear() noexcept {
        errors_.clear();
    }

    /**
     * @brief Get all errors
     */
    const std::vector<ErrorVariant>& errors() const noexcept {
        return errors_;
    }

    /**
     * @brief Get first error (if any)
     */
    const ErrorVariant* first_error() const noexcept {
        return errors_.empty() ? nullptr : &errors_.front();
    }

    /**
     * @brief Get last error (if any)
     */
    const ErrorVariant* last_error() const noexcept {
        return errors_.empty() ? nullptr : &errors_.back();
    }

    /**
     * @brief Merge another error chain into this one
     */
    void merge(const ErrorChain& other) {
        errors_.insert(errors_.end(), other.errors_.begin(), other.errors_.end());
    }

    /**
     * @brief Format all errors as a string
     */
    std::string format() const {
        if (errors_.empty()) {
            return "No errors";
        }

        std::ostringstream oss;
        oss << "Error chain with " << errors_.size() << " error(s):\n";

        size_t index = 1;
        for (const auto& error : errors_) {
            oss << "  " << index++ << ". ";

            std::visit([&oss](const auto& e) {
                using T = std::decay_t<decltype(e)>;
                if constexpr (std::is_same_v<T, ErrorInfo>) {
                    oss << e.full_message();
                } else if constexpr (std::is_same_v<T, Exception>) {
                    oss << e.full_message();
                }
            }, error);

            oss << "\n";
        }

        return oss.str();
    }

    /**
     * @brief Convert to Result type
     */
    template<typename T>
    Result<T, ErrorChain> to_result() const {
        if (has_errors()) {
            return Err<ErrorChain, T>(*this);
        }
        return Ok<T, ErrorChain>(T{});
    }

    /**
     * @brief Throw if there are errors
     */
    void throw_if_errors() const {
        if (has_errors()) {
            throw RuntimeError(format());
        }
    }

private:
    std::vector<ErrorVariant> errors_;
};

/**
 * @brief Collect results from multiple operations
 *
 * Executes all operations and collects any errors
 */
template<typename... Ops>
class ErrorCollector {
public:
    explicit ErrorCollector(Ops... ops) : ops_(ops...) {}

    /**
     * @brief Execute all operations and collect results
     */
    auto collect() {
        ErrorChain chain;
        auto results = std::apply([&chain](auto&&... ops) {
            return std::make_tuple(collect_one(chain, ops)...);
        }, ops_);

        if (chain.has_errors()) {
            using ResultTypes = std::tuple<typename std::invoke_result_t<Ops>::value_type...>;
            return Result<ResultTypes, ErrorChain>(Error<ErrorChain>(chain));
        }

        return Result<decltype(results), ErrorChain>(results);
    }

private:
    template<typename Op>
    static auto collect_one(ErrorChain& chain, Op& op) {
        auto result = op();
        if (result.is_error()) {
            chain.add_error(result.error());
        }
        return result;
    }

    std::tuple<Ops...> ops_;
};

/**
 * @brief Validation chain for multiple validations
 */
class ValidationChain {
public:
    using Validator = std::function<Status()>;

    ValidationChain() = default;

    /**
     * @brief Add a validation
     */
    ValidationChain& validate(Validator validator) {
        validators_.push_back(std::move(validator));
        return *this;
    }

    /**
     * @brief Add a named validation
     */
    ValidationChain& validate(const std::string& name, Validator validator) {
        named_validators_.emplace_back(name, std::move(validator));
        return *this;
    }

    /**
     * @brief Run all validations
     */
    Status run() {
        ErrorChain chain;

        // Run unnamed validators
        for (const auto& validator : validators_) {
            auto status = validator();
            if (!status.ok()) {
                chain.add_error(status.code(), status.message());
            }
        }

        // Run named validators
        for (const auto& [name, validator] : named_validators_) {
            auto status = validator();
            if (!status.ok()) {
                chain.add_error(status.code(),
                              std::format("{}: {}", name, status.message()));
            }
        }

        if (chain.has_errors()) {
            return Status(ErrorCode::InvalidArgument, chain.format());
        }

        return Status::OK();
    }

    /**
     * @brief Run validations and stop on first error
     */
    Status run_fast_fail() {
        // Run unnamed validators
        for (const auto& validator : validators_) {
            auto status = validator();
            if (!status.ok()) {
                return status;
            }
        }

        // Run named validators
        for (const auto& [name, validator] : named_validators_) {
            auto status = validator();
            if (!status.ok()) {
                return Status(status.code(),
                            std::format("{}: {}", name, status.message()));
            }
        }

        return Status::OK();
    }

private:
    std::vector<Validator> validators_;
    std::vector<std::pair<std::string, Validator>> named_validators_;
};

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_CHAIN_H
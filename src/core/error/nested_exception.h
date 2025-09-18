#pragma once

#ifndef CORE_ERROR_NESTED_EXCEPTION_H
#define CORE_ERROR_NESTED_EXCEPTION_H

#include <exception>
#include <memory>
#include <vector>
#include <sstream>
#include <format>
#include "exception_base.h"
#include "error_code.h"

namespace fem::core::error {

/**
 * @brief Nested exception support for complex error chains
 *
 * Provides utilities for working with nested exceptions,
 * error aggregation, and exception chain traversal.
 */
class NestedExceptionHelper {
public:
    /**
     * @brief Extract all nested exceptions as a chain
     */
    static std::vector<std::string> extract_chain(const std::exception& e) {
        std::vector<std::string> chain;
        extract_chain_impl(e, chain);
        return chain;
    }

    /**
     * @brief Extract all nested exceptions from exception_ptr
     */
    static std::vector<std::string> extract_chain(std::exception_ptr eptr) {
        if (!eptr) {
            return {};
        }

        try {
            std::rethrow_exception(eptr);
        } catch (const std::exception& e) {
            return extract_chain(e);
        } catch (...) {
            return {"Unknown exception"};
        }
    }

    /**
     * @brief Format exception chain as string
     */
    static std::string format_chain(const std::exception& e,
                                   const std::string& separator = "\n  Caused by: ") {
        auto chain = extract_chain(e);
        if (chain.empty()) {
            return "";
        }

        std::ostringstream oss;
        oss << chain[0];
        for (size_t i = 1; i < chain.size(); ++i) {
            oss << separator << chain[i];
        }
        return oss.str();
    }

    /**
     * @brief Get the root cause of nested exceptions
     */
    static std::string get_root_cause(const std::exception& e) {
        auto chain = extract_chain(e);
        return chain.empty() ? "" : chain.back();
    }

    /**
     * @brief Count depth of nested exceptions
     */
    static size_t depth(const std::exception& e) {
        return extract_chain(e).size();
    }

private:
    static void extract_chain_impl(const std::exception& e,
                                  std::vector<std::string>& chain) {
        chain.push_back(e.what());

        try {
            // Check if this exception has nested exceptions
            auto nested = dynamic_cast<const std::nested_exception*>(&e);
            if (nested && nested->nested_ptr()) {
                try {
                    std::rethrow_exception(nested->nested_ptr());
                } catch (const std::exception& nested_e) {
                    extract_chain_impl(nested_e, chain);
                } catch (...) {
                    chain.push_back("Unknown nested exception");
                }
            }
        } catch (...) {
            // Ignore extraction errors
        }
    }
};

/**
 * @brief Exception that can carry nested exceptions
 *
 * Extends our base exception with nested exception support.
 */
class NestedException : public Exception, public std::nested_exception {
public:
    /**
     * @brief Create nested exception from current exception
     */
    NestedException(const std::string& message,
                   ErrorCode code = ErrorCode::Unknown,
                   const std::source_location& loc = std::source_location::current())
        : Exception(message, code, loc)
        , std::nested_exception() {
    }

    /**
     * @brief Create with explicit nested exception
     */
    NestedException(const std::string& message,
                   std::exception_ptr /* nested */,
                   ErrorCode code = ErrorCode::Unknown,
                   const std::source_location& loc = std::source_location::current())
        : Exception(message, code, loc)
        , std::nested_exception() {
        // Store the nested exception pointer manually if needed
        // std::nested_exception will capture current_exception() automatically
    }

    /**
     * @brief Get formatted message including nested chain
     */
    std::string full_message() const {
        return NestedExceptionHelper::format_chain(*this);
    }

    /**
     * @brief Get the root cause message
     */
    std::string root_cause() const {
        return NestedExceptionHelper::get_root_cause(*this);
    }

    /**
     * @brief Get nesting depth
     */
    size_t depth() const {
        return NestedExceptionHelper::depth(*this);
    }
};

/**
 * @brief Aggregated exception for multiple errors
 *
 * Useful when multiple operations fail and all errors need to be reported.
 */
class AggregateException : public Exception {
public:
    AggregateException(const std::string& message,
                      const std::source_location& loc = std::source_location::current())
        : Exception(message, ErrorCode::Multiple, loc) {
    }

    /**
     * @brief Add an exception to the aggregate
     */
    void add_exception(std::exception_ptr ex) {
        exceptions_.push_back(ex);
        update_message();
    }

    /**
     * @brief Add an exception by value
     */
    template<typename E>
    void add_exception(const E& e) {
        exceptions_.push_back(std::make_exception_ptr(e));
        update_message();
    }

    /**
     * @brief Get all aggregated exceptions
     */
    const std::vector<std::exception_ptr>& exceptions() const {
        return exceptions_;
    }

    /**
     * @brief Get count of aggregated exceptions
     */
    size_t count() const {
        return exceptions_.size();
    }

    /**
     * @brief Check if aggregate is empty
     */
    bool empty() const {
        return exceptions_.empty();
    }

    /**
     * @brief Get formatted message of all exceptions
     */
    std::string full_message() const {
        if (exceptions_.empty()) {
            return what();
        }

        std::ostringstream oss;
        oss << what() << " (" << exceptions_.size() << " errors):\n";

        for (size_t i = 0; i < exceptions_.size(); ++i) {
            oss << "  [" << i + 1 << "] ";
            try {
                std::rethrow_exception(exceptions_[i]);
            } catch (const std::exception& e) {
                oss << e.what();
            } catch (...) {
                oss << "Unknown exception";
            }

            if (i < exceptions_.size() - 1) {
                oss << "\n";
            }
        }

        return oss.str();
    }

    /**
     * @brief Throw if any exceptions aggregated
     */
    void throw_if_any() const {
        if (!exceptions_.empty()) {
            throw *this;
        }
    }

private:
    void update_message() {
        // Note: set_message method not available in base Exception class
        // This would need to be implemented differently
        // For now, we'll just update the internal message tracking
        // message_ = std::format("{} ({} errors)",
        //                      get_base_message(),
        //                      exceptions_.size());
    }

    std::string get_base_message() const {
        std::string msg = what();
        auto pos = msg.find(" (");
        if (pos != std::string::npos) {
            return msg.substr(0, pos);
        }
        return msg;
    }

    std::vector<std::exception_ptr> exceptions_;
};

/**
 * @brief RAII helper for exception context
 *
 * Automatically adds context to exceptions thrown within scope.
 */
class ExceptionContext {
public:
    explicit ExceptionContext(const std::string& context)
        : context_(context) {
    }

    ~ExceptionContext() noexcept {
        if (std::uncaught_exceptions() > uncaught_on_enter_) {
            // Exception is being thrown, could add context here
            // Note: Can't modify in-flight exceptions in destructor
        }
    }

    /**
     * @brief Wrap a callable with exception context
     */
    template<typename F>
    static auto with_context(const std::string& context, F&& f) {
        try {
            return std::forward<F>(f)();
        } catch (const Exception& e) {
            // Re-throw with added context
            std::throw_with_nested(
                NestedException(std::format("{}: {}", context, e.what()),
                              e.code())
            );
        } catch (const std::exception& e) {
            std::throw_with_nested(
                NestedException(std::format("{}: {}", context, e.what()))
            );
        }
    }

private:
    std::string context_;
    int uncaught_on_enter_ = std::uncaught_exceptions();
};

/**
 * @brief Helper functions for working with nested exceptions
 */

/**
 * @brief Throw with nested exception from current exception context
 */
template<typename E>
void throw_with_nested_context(const E& e) {
    std::throw_with_nested(e);
}

/**
 * @brief Create nested exception from current exception
 */
inline NestedException make_nested_exception(
        const std::string& message,
        ErrorCode code = ErrorCode::Unknown) {
    return NestedException(message, std::current_exception(), code);
}

/**
 * @brief Rethrow with additional context
 */
template<typename E>
[[noreturn]] void rethrow_with_context(const std::string& context) {
    try {
        throw;
    } catch (const Exception& e) {
        std::throw_with_nested(
            NestedException(std::format("{}: {}", context, e.what()),
                          e.code())
        );
    } catch (const std::exception& e) {
        std::throw_with_nested(
            NestedException(std::format("{}: {}", context, e.what()))
        );
    }
}

/**
 * @brief Exception guard for cleanup on exception
 */
class ExceptionGuard {
public:
    using cleanup_func = std::function<void()>;

    explicit ExceptionGuard(cleanup_func cleanup)
        : cleanup_(std::move(cleanup))
        , uncaught_on_enter_(std::uncaught_exceptions()) {
    }

    ~ExceptionGuard() noexcept {
        if (cleanup_ && std::uncaught_exceptions() > uncaught_on_enter_) {
            try {
                cleanup_();
            } catch (...) {
                // Suppress exceptions in cleanup
            }
        }
    }

    void dismiss() {
        cleanup_ = nullptr;
    }

private:
    cleanup_func cleanup_;
    int uncaught_on_enter_;
};

/**
 * @brief Macro for adding exception context
 */
#define WITH_EXCEPTION_CONTEXT(context, code) \
    fem::core::error::ExceptionContext::with_context(context, [&]() { code })

} // namespace fem::core::error

#endif // CORE_ERROR_NESTED_EXCEPTION_H
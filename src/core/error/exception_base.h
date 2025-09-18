#pragma once

#ifndef CORE_ERROR_EXCEPTION_BASE_H
#define CORE_ERROR_EXCEPTION_BASE_H

#include <exception>
#include <string>
#include <vector>
#include <source_location>
#include <format>
#include <sstream>
#include <optional>
#include "error_code.h"

namespace fem::core::error {

/**
 * @brief Stack trace information
 *
 * Simplified stack trace for now, can be enhanced with platform-specific
 * implementations later
 */
class StackTrace {
public:
    struct Frame {
        std::string function;
        std::string file;
        int line;
    };

    void capture(int skip_frames = 0);
    const std::vector<Frame>& frames() const { return frames_; }
    std::string format() const;

private:
    std::vector<Frame> frames_;
};

/**
 * @brief Base exception class with rich context
 *
 * Provides detailed error information including:
 * - Error code
 * - Message
 * - Source location
 * - Context stack
 * - Optional stack trace
 */
class Exception : public std::exception {
public:
    /**
     * @brief Construct exception with message
     */
    explicit Exception(const std::string& message,
                      ErrorCode code = ErrorCode::Unknown,
                      const std::source_location& loc = std::source_location::current())
        : message_(message)
        , code_(code)
        , location_(loc) {
    }

    // Copy constructor - deep copy the nested exception if present
    Exception(const Exception& other)
        : message_(other.message_)
        , code_(other.code_)
        , location_(other.location_)
        , context_(other.context_)
        , stack_trace_(other.stack_trace_)
        , nested_(other.nested_ ? std::make_unique<Exception>(*other.nested_) : nullptr) {
    }

    // Move constructor
    Exception(Exception&&) = default;

    // Copy assignment - deep copy the nested exception if present
    Exception& operator=(const Exception& other) {
        if (this != &other) {
            message_ = other.message_;
            code_ = other.code_;
            location_ = other.location_;
            context_ = other.context_;
            stack_trace_ = other.stack_trace_;
            nested_ = other.nested_ ? std::make_unique<Exception>(*other.nested_) : nullptr;
        }
        return *this;
    }

    // Move assignment
    Exception& operator=(Exception&&) = default;

    /**
     * @brief Construct exception with formatted message
     */
    template<typename... Args>
    Exception(ErrorCode code,
             const std::source_location& loc,
             std::format_string<Args...> fmt,
             Args&&... args)
        : message_(std::format(fmt, std::forward<Args>(args)...))
        , code_(code)
        , location_(loc) {
    }

    // Standard exception interface
    const char* what() const noexcept override {
        return message_.c_str();
    }

    // Extended interface
    ErrorCode code() const noexcept { return code_; }
    const std::source_location& where() const noexcept { return location_; }
    const std::vector<std::string>& context() const noexcept { return context_; }

    /**
     * @brief Add context information
     */
    Exception& with_context(const std::string& ctx) {
        context_.push_back(ctx);
        return *this;
    }

    /**
     * @brief Add formatted context
     */
    template<typename... Args>
    Exception& with_context(std::format_string<Args...> fmt, Args&&... args) {
        context_.push_back(std::format(fmt, std::forward<Args>(args)...));
        return *this;
    }

    /**
     * @brief Capture stack trace
     */
    Exception& with_stack_trace() {
        stack_trace_ = StackTrace();
        stack_trace_->capture(2);  // Skip this function and constructor
        return *this;
    }

    /**
     * @brief Add nested exception
     */
    Exception& with_nested(const Exception& nested) {
        nested_ = std::make_unique<Exception>(nested);
        return *this;
    }

    /**
     * @brief Get full formatted message
     */
    std::string full_message() const {
        std::ostringstream oss;

        // Basic info
        oss << "[" << core_error_category().message(static_cast<int>(code_)) << "] ";
        oss << message_ << "\n";

        // Location
        oss << "  at " << location_.file_name() << ":" << location_.line();
        oss << " in " << location_.function_name() << "\n";

        // Context
        if (!context_.empty()) {
            oss << "  Context:\n";
            for (const auto& ctx : context_) {
                oss << "    - " << ctx << "\n";
            }
        }

        // Stack trace
        if (stack_trace_) {
            oss << "  Stack trace:\n" << stack_trace_->format();
        }

        // Nested exception
        if (nested_) {
            oss << "  Caused by:\n";
            std::string nested_msg = nested_->full_message();
            // Indent nested message
            std::istringstream iss(nested_msg);
            std::string line;
            while (std::getline(iss, line)) {
                oss << "    " << line << "\n";
            }
        }

        return oss.str();
    }

    /**
     * @brief Print diagnostic information
     */
    void print_diagnostic(std::ostream& os) const {
        os << full_message();
    }

    /**
     * @brief Describe the exception
     */
    std::string describe() const {
        return std::format("Exception: {} [{}]",
                          message_,
                          core_error_category().message(static_cast<int>(code_)));
    }

protected:
    std::string message_;
    ErrorCode code_;
    std::source_location location_;
    std::vector<std::string> context_;
    std::optional<StackTrace> stack_trace_;
    std::unique_ptr<Exception> nested_;
};

/**
 * @brief Logic error - programming errors
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
 * @brief Runtime error - runtime failures
 */
class RuntimeError : public Exception {
public:
    using Exception::Exception;

    explicit RuntimeError(const std::string& message,
                         const std::source_location& loc = std::source_location::current())
        : Exception(message, ErrorCode::Unknown, loc) {
    }
};

/**
 * @brief System error - OS/system failures
 */
class SystemError : public Exception {
public:
    SystemError(const std::string& message,
                int system_error_code,
                const std::source_location& loc = std::source_location::current())
        : Exception(message, ErrorCode::SystemError, loc)
        , system_error_code_(system_error_code) {
    }

    int system_error_code() const noexcept { return system_error_code_; }

private:
    int system_error_code_;
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

    const std::string& argument_name() const noexcept { return argument_name_; }

private:
    std::string argument_name_;
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
                                what, index, size), loc)
        , index_(index)
        , size_(size) {
    }

    size_t index() const noexcept { return index_; }
    size_t size() const noexcept { return size_; }

private:
    size_t index_;
    size_t size_;
};

/**
 * @brief Not implemented error
 */
class NotImplementedError : public LogicError {
public:
    explicit NotImplementedError(const std::string& feature,
                                const std::source_location& loc = std::source_location::current())
        : LogicError(std::format("Not implemented: {}", feature), loc) {
    }
};

} // namespace fem::core::error

#endif // CORE_ERROR_EXCEPTION_BASE_H
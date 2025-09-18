#pragma once

#ifndef CORE_ERROR_STATUS_H
#define CORE_ERROR_STATUS_H

#include <string>
#include <string_view>
#include "error_code.h"

namespace fem::core::error {

/**
 * @brief Lightweight status code
 *
 * Similar to absl::Status or grpc::Status.
 * More lightweight than Result when you don't need to return a value.
 */
class Status {
public:
    /**
     * @brief Default constructor creates OK status
     */
    Status() noexcept : code_(ErrorCode::Success) {}

    /**
     * @brief Construct with error code
     */
    explicit Status(ErrorCode code) noexcept : code_(code) {}

    /**
     * @brief Construct with error code and message
     */
    Status(ErrorCode code, std::string_view message)
        : code_(code), message_(message) {}

    /**
     * @brief Construct with error code, message and context
     */
    Status(ErrorCode code, std::string_view message, std::string_view context)
        : code_(code), message_(message), context_(context) {}

    // Factory methods
    static Status OK() noexcept {
        return Status();
    }

    static Status Error(ErrorCode code) {
        return Status(code);
    }

    static Status Error(ErrorCode code, std::string_view message) {
        return Status(code, message);
    }

    // State queries
    bool ok() const noexcept {
        return code_ == ErrorCode::Success;
    }

    bool is_error() const noexcept {
        return code_ != ErrorCode::Success;
    }

    explicit operator bool() const noexcept {
        return ok();
    }

    // Accessors
    ErrorCode code() const noexcept {
        return code_;
    }

    std::string_view message() const noexcept {
        if (!message_.empty()) {
            return message_;
        }
        static std::string default_msg;
        default_msg = core_error_category().message(static_cast<int>(code_));
        return default_msg;
    }

    std::string_view context() const noexcept {
        return context_;
    }

    // Update status
    Status& update(ErrorCode code) noexcept {
        code_ = code;
        return *this;
    }

    Status& update(ErrorCode code, std::string_view message) {
        code_ = code;
        message_ = message;
        return *this;
    }

    Status& add_context(std::string_view context) {
        if (context_.empty()) {
            context_ = context;
        } else {
            context_ = std::string(context_) + "; " + std::string(context);
        }
        return *this;
    }

    // Comparison
    friend bool operator==(const Status& a, const Status& b) noexcept {
        return a.code_ == b.code_;
    }

    friend bool operator!=(const Status& a, const Status& b) noexcept {
        return !(a == b);
    }

    // String representation
    std::string to_string() const {
        if (ok()) {
            return "OK";
        }

        std::string result = std::string(message());
        if (!context_.empty()) {
            result += " [" + context_ + "]";
        }
        return result;
    }

private:
    ErrorCode code_;
    std::string message_;
    std::string context_;
};

/**
 * @brief Chainable status updates
 *
 * Allows building up status through multiple operations
 */
class StatusBuilder {
public:
    StatusBuilder() = default;

    StatusBuilder& add_error(const Status& status) {
        if (status.is_error()) {
            if (first_error_.ok()) {
                first_error_ = status;
            }
            error_count_++;
        }
        return *this;
    }

    StatusBuilder& add_error(ErrorCode code, std::string_view message) {
        return add_error(Status(code, message));
    }

    bool ok() const noexcept {
        return first_error_.ok();
    }

    size_t error_count() const noexcept {
        return error_count_;
    }

    Status status() const {
        if (ok()) {
            return Status::OK();
        }

        if (error_count_ == 1) {
            return first_error_;
        }

        // Multiple errors
        return Status(first_error_.code(),
                     std::format("{} (and {} more errors)",
                                first_error_.message(),
                                error_count_ - 1));
    }

private:
    Status first_error_;
    size_t error_count_ = 0;
};

// Convenience macros

/**
 * @brief Return if status is not OK
 */
#define RETURN_IF_ERROR(status) \
    do { \
        auto _status = (status); \
        if (!_status.ok()) { \
            return _status; \
        } \
    } while(0)

/**
 * @brief Check status and return error
 */
#define CHECK_STATUS(expr) \
    do { \
        auto _status = (expr); \
        if (!_status.ok()) { \
            return Status::Error(_status.code(), \
                               std::format("Check failed: {} at {}:{}", \
                                         #expr, __FILE__, __LINE__)); \
        } \
    } while(0)

} // namespace fem::core::error

#endif // CORE_ERROR_STATUS_H
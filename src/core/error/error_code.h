#pragma once

#ifndef CORE_ERROR_ERROR_CODE_H
#define CORE_ERROR_ERROR_CODE_H

#include <string_view>
#include <system_error>
#include <format>

namespace fem::core {

/**
 * @brief Core library error codes
 * 
 * Lightweight error codes for the core infrastructure.
 * Domain-specific layers should define their own error enums.
 */
enum class ErrorCode : int {
    Success = 0,
    
    // General errors (1-99)
    Unknown = 1,
    NotImplemented = 2,
    InvalidArgument = 3,
    InvalidState = 4,
    OutOfRange = 5,
    TypeMismatch = 6,
    
    // Resource errors (100-199)
    OutOfMemory = 100,
    ResourceNotFound = 101,
    ResourceBusy = 102,
    ResourceExhausted = 103,
    AllocationFailed = 104,
    
    // I/O errors (200-299)
    FileNotFound = 200,
    FileAccessDenied = 201,
    FileAlreadyExists = 202,
    InvalidPath = 203,
    IoError = 204,
    EndOfFile = 205,
    
    // Concurrency errors (300-399)
    DeadlockDetected = 300,
    ThreadCreationFailed = 301,
    SynchronizationError = 302,
    TimeoutExpired = 303,
    
    // Configuration errors (400-499)
    ConfigNotFound = 400,
    ConfigInvalid = 401,
    ConfigTypeMismatch = 402,
    RequiredConfigMissing = 403,
    
    // System errors (500-599)
    SystemError = 500,
    PlatformNotSupported = 501,
    FeatureDisabled = 502,
};

/**
 * @brief Error category for core errors
 */
class CoreErrorCategory : public std::error_category {
public:
    const char* name() const noexcept override {
        return "fem::core";
    }
    
    std::string message(int code) const override {
        switch (static_cast<ErrorCode>(code)) {
            case ErrorCode::Success:
                return "Success";
            
            // General errors
            case ErrorCode::Unknown:
                return "Unknown error";
            case ErrorCode::NotImplemented:
                return "Not implemented";
            case ErrorCode::InvalidArgument:
                return "Invalid argument";
            case ErrorCode::InvalidState:
                return "Invalid state";
            case ErrorCode::OutOfRange:
                return "Out of range";
            case ErrorCode::TypeMismatch:
                return "Type mismatch";
            
            // Resource errors
            case ErrorCode::OutOfMemory:
                return "Out of memory";
            case ErrorCode::ResourceNotFound:
                return "Resource not found";
            case ErrorCode::ResourceBusy:
                return "Resource busy";
            case ErrorCode::ResourceExhausted:
                return "Resource exhausted";
            case ErrorCode::AllocationFailed:
                return "Allocation failed";
            
            // I/O errors
            case ErrorCode::FileNotFound:
                return "File not found";
            case ErrorCode::FileAccessDenied:
                return "File access denied";
            case ErrorCode::FileAlreadyExists:
                return "File already exists";
            case ErrorCode::InvalidPath:
                return "Invalid path";
            case ErrorCode::IoError:
                return "I/O error";
            case ErrorCode::EndOfFile:
                return "End of file";
            
            // Concurrency errors
            case ErrorCode::DeadlockDetected:
                return "Deadlock detected";
            case ErrorCode::ThreadCreationFailed:
                return "Thread creation failed";
            case ErrorCode::SynchronizationError:
                return "Synchronization error";
            case ErrorCode::TimeoutExpired:
                return "Timeout expired";
            
            // Configuration errors
            case ErrorCode::ConfigNotFound:
                return "Configuration not found";
            case ErrorCode::ConfigInvalid:
                return "Invalid configuration";
            case ErrorCode::ConfigTypeMismatch:
                return "Configuration type mismatch";
            case ErrorCode::RequiredConfigMissing:
                return "Required configuration missing";
            
            // System errors
            case ErrorCode::SystemError:
                return "System error";
            case ErrorCode::PlatformNotSupported:
                return "Platform not supported";
            case ErrorCode::FeatureDisabled:
                return "Feature disabled";
            
            default:
                return std::format("Unknown error code: {}", code);
        }
    }
};

// Global error category instance
inline const CoreErrorCategory& core_error_category() {
    static CoreErrorCategory instance;
    return instance;
}

// Make ErrorCode compatible with std::error_code
inline std::error_code make_error_code(ErrorCode e) {
    return {static_cast<int>(e), core_error_category()};
}

/**
 * @brief Structured error information
 * 
 * Provides more context than just an error code
 */
class ErrorInfo {
private:
    ErrorCode code_;
    std::string context_;
    std::source_location location_;
    
public:
    ErrorInfo(ErrorCode code, 
              std::string_view context = "",
              const std::source_location& loc = std::source_location::current())
        : code_(code)
        , context_(context)
        , location_(loc) {}
    
    [[nodiscard]] int code() const noexcept { 
        return static_cast<int>(code_); 
    }
    
    [[nodiscard]] ErrorCode error_code() const noexcept { 
        return code_; 
    }
    
    [[nodiscard]] std::string_view message() const noexcept {
        static std::string msg;
        msg = core_error_category().message(static_cast<int>(code_));
        if (!context_.empty()) {
            msg = std::format("{}: {}", msg, context_);
        }
        return msg;
    }
    
    [[nodiscard]] std::string_view context() const noexcept { 
        return context_; 
    }
    
    [[nodiscard]] const std::source_location& location() const noexcept { 
        return location_; 
    }
    
    [[nodiscard]] std::string full_message() const {
        return std::format("[{}:{}] {} - {}", 
                           location_.file_name(),
                           location_.line(),
                           core_error_category().message(static_cast<int>(code_)),
                           context_);
    }
};

// Helper function for creating errors with context
[[nodiscard]] inline ErrorInfo make_error(ErrorCode code, 
                                            std::string_view context = "",
                                            const std::source_location& loc = std::source_location::current()) {
    return ErrorInfo(code, context, loc);
}

} // namespace fem::core

// Enable ErrorCode to be used with std::error_code
namespace std {
    template<>
    struct is_error_code_enum<fem::core::ErrorCode> : std::true_type {};
}

#endif // CORE_ERROR_ERROR_CODE_H
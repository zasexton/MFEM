#pragma once

#ifndef CORE_ERROR_SYSTEM_ERROR_H
#define CORE_ERROR_SYSTEM_ERROR_H

#include <system_error>
#include <cerrno>
#include <cstring>
#include "exception_base.h"
#include "error_code.h"

namespace fem::core::error {

/**
 * @brief System error with OS error code
 *
 * Wraps system/OS errors with additional context.
 * Compatible with std::system_error but provides richer information.
 */
class SystemError : public Exception {
public:
    /**
     * @brief Construct from errno
     */
    explicit SystemError(const std::string& what,
                        int error_code = errno,
                        const std::source_location& loc = std::source_location::current())
        : Exception(format_message(what, error_code),
                   ErrorCode::SystemError, loc)
        , system_error_code_(error_code)
        , error_category_(&std::system_category()) {
    }

    /**
     * @brief Construct from std::error_code
     */
    SystemError(const std::string& what,
               std::error_code ec,
               const std::source_location& loc = std::source_location::current())
        : Exception(format_message(what, ec),
                   ErrorCode::SystemError, loc)
        , system_error_code_(ec.value())
        , error_category_(&ec.category()) {
    }

    /**
     * @brief Construct from std::system_error
     */
    explicit SystemError(const std::system_error& e,
                        const std::source_location& loc = std::source_location::current())
        : Exception(e.what(), ErrorCode::SystemError, loc)
        , system_error_code_(e.code().value())
        , error_category_(&e.code().category()) {
    }

    // Accessors
    int system_error_code() const noexcept { return system_error_code_; }
    const std::error_category& category() const noexcept { return *error_category_; }

    /**
     * @brief Get std::error_code representation
     */
    std::error_code error_code() const noexcept {
        return std::error_code(system_error_code_, *error_category_);
    }

    /**
     * @brief Check if this is a specific system error
     */
    bool is_error(std::errc error) const noexcept {
        return error_code() == std::make_error_code(error);
    }

    /**
     * @brief Create from last system error (errno)
     */
    static SystemError from_errno(const std::string& operation,
                                  const std::source_location& loc = std::source_location::current()) {
        return SystemError(operation, errno, loc);
    }

#ifdef _WIN32
    /**
     * @brief Create from Windows GetLastError()
     */
    static SystemError from_win32(const std::string& operation,
                                  DWORD error_code = ::GetLastError(),
                                  const std::source_location& loc = std::source_location::current()) {
        return SystemError(operation,
                          std::error_code(error_code, std::system_category()),
                          loc);
    }
#endif

private:
    static std::string format_message(const std::string& what, int error_code) {
        return std::format("{}: {} ({})",
                          what,
                          std::strerror(error_code),
                          error_code);
    }

    static std::string format_message(const std::string& what, std::error_code ec) {
        return std::format("{}: {} ({})",
                          what,
                          ec.message(),
                          ec.value());
    }

    int system_error_code_;
    const std::error_category* error_category_;
};

/**
 * @brief File system error
 */
class FileSystemError : public SystemError {
public:
    FileSystemError(const std::string& operation,
                   const std::string& path,
                   int error_code = errno,
                   const std::source_location& loc = std::source_location::current())
        : SystemError(format_fs_message(operation, path), error_code, loc)
        , path_(path) {
    }

    FileSystemError(const std::string& operation,
                   const std::string& path1,
                   const std::string& path2,
                   int error_code = errno,
                   const std::source_location& loc = std::source_location::current())
        : SystemError(format_fs_message(operation, path1, path2), error_code, loc)
        , path_(path1)
        , path2_(path2) {
    }

    const std::string& path() const noexcept { return path_; }
    const std::string& path2() const noexcept { return path2_; }

private:
    static std::string format_fs_message(const std::string& operation,
                                         const std::string& path) {
        return std::format("{}: '{}'", operation, path);
    }

    static std::string format_fs_message(const std::string& operation,
                                         const std::string& path1,
                                         const std::string& path2) {
        return std::format("{}: '{}' -> '{}'", operation, path1, path2);
    }

    std::string path_;
    std::string path2_;
};

/**
 * @brief Network/socket error
 */
class NetworkError : public SystemError {
public:
    NetworkError(const std::string& operation,
                const std::string& address = "",
                int error_code = errno,
                const std::source_location& loc = std::source_location::current())
        : SystemError(format_net_message(operation, address), error_code, loc)
        , address_(address) {
    }

    const std::string& address() const noexcept { return address_; }

private:
    static std::string format_net_message(const std::string& operation,
                                         const std::string& address) {
        if (address.empty()) {
            return operation;
        }
        return std::format("{}: {}", operation, address);
    }

    std::string address_;
};

/**
 * @brief Thread/synchronization error
 */
class ThreadError : public SystemError {
public:
    explicit ThreadError(const std::string& what,
                        int error_code = errno,
                        const std::source_location& loc = std::source_location::current())
        : SystemError(std::format("Thread error: {}", what), error_code, loc) {
    }
};

/**
 * @brief Resource allocation error
 */
class ResourceError : public SystemError {
public:
    enum class ResourceType {
        Memory,
        FileDescriptor,
        ThreadHandle,
        MutexHandle,
        Other
    };

    ResourceError(ResourceType type,
                 const std::string& operation,
                 int error_code = errno,
                 const std::source_location& loc = std::source_location::current())
        : SystemError(format_resource_message(type, operation), error_code, loc)
        , resource_type_(type) {
    }

    ResourceType resource_type() const noexcept { return resource_type_; }

private:
    static std::string format_resource_message(ResourceType type,
                                              const std::string& operation) {
        const char* type_str = nullptr;
        switch (type) {
            case ResourceType::Memory: type_str = "Memory"; break;
            case ResourceType::FileDescriptor: type_str = "File descriptor"; break;
            case ResourceType::ThreadHandle: type_str = "Thread handle"; break;
            case ResourceType::MutexHandle: type_str = "Mutex handle"; break;
            case ResourceType::Other: type_str = "Resource"; break;
        }
        return std::format("{} allocation failed: {}", type_str, operation);
    }

    ResourceType resource_type_;
};

/**
 * @brief Helper to check system call results
 */
class SystemCallChecker {
public:
    /**
     * @brief Check POSIX-style return (negative = error)
     */
    static void check_posix(int result, const std::string& operation) {
        if (result < 0) {
            throw SystemError::from_errno(operation);
        }
    }

    /**
     * @brief Check pointer return (null = error)
     */
    template<typename T>
    static T* check_ptr(T* ptr, const std::string& operation) {
        if (ptr == nullptr) {
            throw SystemError::from_errno(operation);
        }
        return ptr;
    }

#ifdef _WIN32
    /**
     * @brief Check Windows BOOL return
     */
    static void check_win32(BOOL result, const std::string& operation) {
        if (!result) {
            throw SystemError::from_win32(operation);
        }
    }

    /**
     * @brief Check Windows HANDLE return
     */
    static HANDLE check_handle(HANDLE handle, const std::string& operation) {
        if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
            throw SystemError::from_win32(operation);
        }
        return handle;
    }
#endif
};

// Convenience macros for system call checking
#define CHECK_SYSCALL(call) \
    ::fem::core::error::SystemCallChecker::check_posix(call, #call)

#define CHECK_PTR(call) \
    ::fem::core::error::SystemCallChecker::check_ptr(call, #call)

#ifdef _WIN32
#define CHECK_WIN32(call) \
    ::fem::core::error::SystemCallChecker::check_win32(call, #call)

#define CHECK_HANDLE(call) \
    ::fem::core::error::SystemCallChecker::check_handle(call, #call)
#endif

} // namespace fem::core::error

#endif // CORE_ERROR_SYSTEM_ERROR_H
#pragma once

#ifndef CORE_ERROR_RUNTIME_ERROR_H
#define CORE_ERROR_RUNTIME_ERROR_H

#include "exception_base.h"
#include <chrono>
#include <optional>

namespace fem::core::error {

/**
 * @brief I/O error
 */
class IOError : public RuntimeError {
public:
    enum class Operation {
        Read,
        Write,
        Open,
        Close,
        Seek,
        Flush,
        Other
    };

    IOError(Operation op,
           const std::string& resource,
           const std::string& reason,
           const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(op, resource, reason), loc)
        , operation_(op)
        , resource_(resource) {
    }

    Operation operation() const noexcept { return operation_; }
    const std::string& resource() const noexcept { return resource_; }

private:
    static std::string format_message(Operation op,
                                     const std::string& resource,
                                     const std::string& reason) {
        const char* op_str = nullptr;
        switch (op) {
            case Operation::Read: op_str = "read"; break;
            case Operation::Write: op_str = "write"; break;
            case Operation::Open: op_str = "open"; break;
            case Operation::Close: op_str = "close"; break;
            case Operation::Seek: op_str = "seek"; break;
            case Operation::Flush: op_str = "flush"; break;
            case Operation::Other: op_str = "I/O operation"; break;
        }
        return std::format("Failed to {} '{}': {}", op_str, resource, reason);
    }

    Operation operation_;
    std::string resource_;
};

/**
 * @brief Resource not found error
 */
class NotFoundError : public RuntimeError {
public:
    enum class ResourceType {
        File,
        Directory,
        Key,
        Element,
        Object,
        Other
    };

    NotFoundError(ResourceType type,
                 const std::string& name,
                 const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(type, name), loc)
        , resource_type_(type)
        , resource_name_(name) {
    }

    ResourceType resource_type() const noexcept { return resource_type_; }
    const std::string& resource_name() const noexcept { return resource_name_; }

private:
    static std::string format_message(ResourceType type,
                                     const std::string& name) {
        const char* type_str = nullptr;
        switch (type) {
            case ResourceType::File: type_str = "File"; break;
            case ResourceType::Directory: type_str = "Directory"; break;
            case ResourceType::Key: type_str = "Key"; break;
            case ResourceType::Element: type_str = "Element"; break;
            case ResourceType::Object: type_str = "Object"; break;
            case ResourceType::Other: type_str = "Resource"; break;
        }
        return std::format("{} not found: '{}'", type_str, name);
    }

    ResourceType resource_type_;
    std::string resource_name_;
};

/**
 * @brief Resource already exists error
 */
class AlreadyExistsError : public RuntimeError {
public:
    AlreadyExistsError(const std::string& resource_type,
                      const std::string& resource_name,
                      const std::source_location& loc = std::source_location::current())
        : RuntimeError(std::format("{} already exists: '{}'",
                                 resource_type, resource_name), loc)
        , resource_type_(resource_type)
        , resource_name_(resource_name) {
    }

    const std::string& resource_type() const noexcept { return resource_type_; }
    const std::string& resource_name() const noexcept { return resource_name_; }

private:
    std::string resource_type_;
    std::string resource_name_;
};

/**
 * @brief Timeout error
 */
class TimeoutError : public RuntimeError {
public:
    TimeoutError(const std::string& operation,
                std::chrono::milliseconds timeout,
                const std::source_location& loc = std::source_location::current())
        : RuntimeError(std::format("Operation '{}' timed out after {}ms",
                                 operation, timeout.count()), loc)
        , operation_(operation)
        , timeout_(timeout) {
    }

    const std::string& operation() const noexcept { return operation_; }
    std::chrono::milliseconds timeout() const noexcept { return timeout_; }

private:
    std::string operation_;
    std::chrono::milliseconds timeout_;
};

/**
 * @brief Permission denied error
 */
class PermissionError : public RuntimeError {
public:
    enum class Permission {
        Read,
        Write,
        Execute,
        Delete,
        Create,
        Other
    };

    PermissionError(Permission perm,
                   const std::string& resource,
                   const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(perm, resource), loc)
        , permission_(perm)
        , resource_(resource) {
    }

    Permission permission() const noexcept { return permission_; }
    const std::string& resource() const noexcept { return resource_; }

private:
    static std::string format_message(Permission perm,
                                     const std::string& resource) {
        const char* perm_str = nullptr;
        switch (perm) {
            case Permission::Read: perm_str = "read"; break;
            case Permission::Write: perm_str = "write"; break;
            case Permission::Execute: perm_str = "execute"; break;
            case Permission::Delete: perm_str = "delete"; break;
            case Permission::Create: perm_str = "create"; break;
            case Permission::Other: perm_str = "access"; break;
        }
        return std::format("Permission denied to {} '{}'", perm_str, resource);
    }

    Permission permission_;
    std::string resource_;
};

/**
 * @brief Resource exhausted error
 */
class ResourceExhaustedError : public RuntimeError {
public:
    enum class Resource {
        Memory,
        DiskSpace,
        FileHandles,
        Threads,
        Connections,
        Other
    };

    ResourceExhaustedError(Resource resource,
                          std::optional<size_t> limit = std::nullopt,
                          const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(resource, limit), loc)
        , resource_(resource)
        , limit_(limit) {
    }

    Resource resource() const noexcept { return resource_; }
    std::optional<size_t> limit() const noexcept { return limit_; }

private:
    static std::string format_message(Resource resource,
                                     std::optional<size_t> limit) {
        const char* resource_str = nullptr;
        switch (resource) {
            case Resource::Memory: resource_str = "Memory"; break;
            case Resource::DiskSpace: resource_str = "Disk space"; break;
            case Resource::FileHandles: resource_str = "File handles"; break;
            case Resource::Threads: resource_str = "Threads"; break;
            case Resource::Connections: resource_str = "Connections"; break;
            case Resource::Other: resource_str = "Resource"; break;
        }

        if (limit) {
            return std::format("{} exhausted (limit: {})", resource_str, *limit);
        }
        return std::format("{} exhausted", resource_str);
    }

    Resource resource_;
    std::optional<size_t> limit_;
};

/**
 * @brief Configuration error
 */
class ConfigurationError : public RuntimeError {
public:
    ConfigurationError(const std::string& parameter,
                      const std::string& reason,
                      const std::source_location& loc = std::source_location::current())
        : RuntimeError(std::format("Configuration error for '{}': {}",
                                 parameter, reason), loc)
        , parameter_(parameter) {
    }

    const std::string& parameter() const noexcept { return parameter_; }

private:
    std::string parameter_;
};

/**
 * @brief Parsing/format error
 */
class ParseError : public RuntimeError {
public:
    ParseError(const std::string& input,
              size_t position,
              const std::string& expected,
              const std::source_location& loc = std::source_location::current())
        : RuntimeError(std::format("Parse error at position {}: expected {}",
                                 position, expected), loc)
        , input_(input)
        , position_(position)
        , expected_(expected) {
    }

    const std::string& input() const noexcept { return input_; }
    size_t position() const noexcept { return position_; }
    const std::string& expected() const noexcept { return expected_; }

private:
    std::string input_;
    size_t position_;
    std::string expected_;
};

/**
 * @brief Network/communication error
 */
class NetworkError : public RuntimeError {
public:
    enum class Type {
        ConnectionRefused,
        ConnectionLost,
        HostNotFound,
        Timeout,
        ProtocolError,
        Other
    };

    NetworkError(Type type,
                const std::string& endpoint,
                const std::string& details = "",
                const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(type, endpoint, details), loc)
        , type_(type)
        , endpoint_(endpoint) {
    }

    Type type() const noexcept { return type_; }
    const std::string& endpoint() const noexcept { return endpoint_; }

private:
    static std::string format_message(Type type,
                                     const std::string& endpoint,
                                     const std::string& details) {
        const char* type_str = nullptr;
        switch (type) {
            case Type::ConnectionRefused: type_str = "Connection refused"; break;
            case Type::ConnectionLost: type_str = "Connection lost"; break;
            case Type::HostNotFound: type_str = "Host not found"; break;
            case Type::Timeout: type_str = "Network timeout"; break;
            case Type::ProtocolError: type_str = "Protocol error"; break;
            case Type::Other: type_str = "Network error"; break;
        }

        if (details.empty()) {
            return std::format("{}: {}", type_str, endpoint);
        }
        return std::format("{}: {} - {}", type_str, endpoint, details);
    }

    Type type_;
    std::string endpoint_;
};

/**
 * @brief Concurrency error
 */
class ConcurrencyError : public RuntimeError {
public:
    enum class Type {
        Deadlock,
        RaceCondition,
        LockTimeout,
        ThreadCreationFailed,
        Other
    };

    ConcurrencyError(Type type,
                    const std::string& details,
                    const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(type, details), loc)
        , type_(type) {
    }

    Type type() const noexcept { return type_; }

private:
    static std::string format_message(Type type, const std::string& details) {
        const char* type_str = nullptr;
        switch (type) {
            case Type::Deadlock: type_str = "Deadlock detected"; break;
            case Type::RaceCondition: type_str = "Race condition"; break;
            case Type::LockTimeout: type_str = "Lock acquisition timeout"; break;
            case Type::ThreadCreationFailed: type_str = "Thread creation failed"; break;
            case Type::Other: type_str = "Concurrency error"; break;
        }
        return std::format("{}: {}", type_str, details);
    }

    Type type_;
};

/**
 * @brief Numerical computation error
 */
class NumericalError : public RuntimeError {
public:
    enum class Type {
        Overflow,
        Underflow,
        DivisionByZero,
        InvalidOperation,
        Convergence,
        Singularity,
        Other
    };

    NumericalError(Type type,
                  const std::string& operation,
                  const std::string& details = "",
                  const std::source_location& loc = std::source_location::current())
        : RuntimeError(format_message(type, operation, details), loc)
        , type_(type)
        , operation_(operation) {
    }

    Type type() const noexcept { return type_; }
    const std::string& operation() const noexcept { return operation_; }

private:
    static std::string format_message(Type type,
                                     const std::string& operation,
                                     const std::string& details) {
        const char* type_str = nullptr;
        switch (type) {
            case Type::Overflow: type_str = "Numerical overflow"; break;
            case Type::Underflow: type_str = "Numerical underflow"; break;
            case Type::DivisionByZero: type_str = "Division by zero"; break;
            case Type::InvalidOperation: type_str = "Invalid numerical operation"; break;
            case Type::Convergence: type_str = "Convergence failure"; break;
            case Type::Singularity: type_str = "Singularity"; break;
            case Type::Other: type_str = "Numerical error"; break;
        }

        if (details.empty()) {
            return std::format("{} in '{}'", type_str, operation);
        }
        return std::format("{} in '{}': {}", type_str, operation, details);
    }

    Type type_;
    std::string operation_;
};

} // namespace fem::core::error

#endif // CORE_ERROR_RUNTIME_ERROR_H
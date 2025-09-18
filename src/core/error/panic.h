#pragma once

#ifndef CORE_ERROR_PANIC_H
#define CORE_ERROR_PANIC_H

#include <cstdlib>
#include <cstdio>
#include <source_location>
#include <format>
#include <string_view>

namespace fem::core::error {

/**
 * @brief Panic handler for unrecoverable errors
 *
 * Provides controlled termination with diagnostic information
 */
class PanicHandler {
public:
    using Handler = void(*)(const char* message,
                          const std::source_location& location);

    /**
     * @brief Set custom panic handler
     */
    static void set_handler(Handler handler) {
        custom_handler_ = handler;
    }

    /**
     * @brief Panic with message
     */
    [[noreturn]] static void panic(const char* message,
                                   const std::source_location& location) {
        if (custom_handler_) {
            custom_handler_(message, location);
        } else {
            default_panic(message, location);
        }
        std::abort();  // Ensure termination
    }

    /**
     * @brief Panic with formatted message
     */
    template<typename... Args>
    [[noreturn]] static void panic(const std::source_location& location,
                                  std::format_string<Args...> fmt,
                                  Args&&... args) {
        std::string message = std::format(fmt, std::forward<Args>(args)...);
        panic(message.c_str(), location);
    }

private:
    static void default_panic(const char* message,
                             const std::source_location& location) {
        std::fprintf(stderr,
                    "\n=================================\n"
                    "PANIC: Program terminated due to unrecoverable error\n"
                    "=================================\n"
                    "Message: %s\n"
                    "Location: %s:%u\n"
                    "Function: %s\n"
                    "=================================\n",
                    message,
                    location.file_name(),
                    static_cast<unsigned>(location.line()),
                    location.function_name());
        std::fflush(stderr);
    }

    static inline Handler custom_handler_ = nullptr;
};

// Convenience macros

/**
 * @brief Panic with message
 */
#define PANIC(message) \
    ::fem::core::error::PanicHandler::panic( \
        message, std::source_location::current())

/**
 * @brief Panic with formatted message
 */
#define PANIC_FMT(...) \
    ::fem::core::error::PanicHandler::panic( \
        std::source_location::current(), __VA_ARGS__)

/**
 * @brief Panic if condition is false
 */
#define PANIC_IF(condition, message) \
    do { \
        if (!(condition)) { \
            ::fem::core::error::PanicHandler::panic( \
                message, std::source_location::current()); \
        } \
    } while(0)

/**
 * @brief Panic if null
 */
#define PANIC_IF_NULL(ptr, name) \
    do { \
        if ((ptr) == nullptr) { \
            ::fem::core::error::PanicHandler::panic( \
                std::source_location::current(), \
                "{} is null", name); \
        } \
    } while(0)

/**
 * @brief Mark unreachable code
 */
#define UNREACHABLE() \
    ::fem::core::error::PanicHandler::panic( \
        "Unreachable code reached", \
        std::source_location::current())

/**
 * @brief Mark unimplemented code
 */
#define UNIMPLEMENTED() \
    ::fem::core::error::PanicHandler::panic( \
        "Unimplemented functionality", \
        std::source_location::current())

/**
 * @brief Todo with panic
 */
#define TODO(message) \
    ::fem::core::error::PanicHandler::panic( \
        std::source_location::current(), \
        "TODO: {}", message)

} // namespace fem::core::error

#endif // CORE_ERROR_PANIC_H
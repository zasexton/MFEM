#pragma once

#ifndef CORE_CONFIG_DEBUG_H
#define CORE_CONFIG_DEBUG_H

#include "config.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <source_location>
#include <string_view>
#include <fstream>

// ==============================================================================
// Debug and Assertion Utilities
// ==============================================================================
// This file provides debugging utilities, assertions, and error checking
// macros for the core library. These are compile-time configurable based
// on build type.
// ==============================================================================

namespace fem::core::debug {

// ==============================================================================
// Debug Output Utilities
// ==============================================================================

enum class LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Fatal = 5
};

// Simple debug logger (more sophisticated logging in logging/ module)
class DebugLogger {
public:
    static void log(LogLevel level,
                   const std::string& message,
                   const std::source_location& loc = std::source_location::current()) {
        #if CORE_ENABLE_LOGGING
        const char* level_str = "";
        switch (level) {
            case LogLevel::Trace:   level_str = "TRACE"; break;
            case LogLevel::Debug:   level_str = "DEBUG"; break;
            case LogLevel::Info:    level_str = "INFO"; break;
            case LogLevel::Warning: level_str = "WARN"; break;
            case LogLevel::Error:   level_str = "ERROR"; break;
            case LogLevel::Fatal:   level_str = "FATAL"; break;
        }

        std::cerr << "[" << level_str << "] "
                 << loc.file_name() << ":" << loc.line() << " "
                 << "in " << loc.function_name() << ": "
                 << message << std::endl;
        #else
        CORE_UNUSED(level);
        CORE_UNUSED(message);
        CORE_UNUSED(loc);
        #endif
    }
};

// ==============================================================================
// Assertion Handler
// ==============================================================================

class AssertionHandler {
public:
    [[noreturn]] static void handle_assertion(
        const char* expression,
        const char* message,
        const std::source_location& loc) {

        std::stringstream ss;
        ss << "Assertion failed: " << expression << "\n";
        if (message && message[0] != '\0') {
            ss << "Message: " << message << "\n";
        }
        ss << "Location: " << loc.file_name() << ":" << loc.line() << "\n";
        ss << "Function: " << loc.function_name() << "\n";

        #if CORE_ENABLE_LOGGING
        std::cerr << "\n" << ss.str() << std::endl;
        #endif

        #if defined(CORE_DEBUG_BUILD)
        // In debug mode, trigger debugger if attached
        #if defined(CORE_PLATFORM_WINDOWS)
            __debugbreak();
        #elif defined(CORE_COMPILER_GCC) || defined(CORE_COMPILER_CLANG)
            __builtin_trap();
        #endif
        #endif

        std::abort();
    }

    static bool should_break() {
        #if defined(CORE_DEBUG_BUILD)
        return true;
        #else
        return false;
        #endif
    }
};

// ==============================================================================
// Debug Utilities
// ==============================================================================

// Check if debugger is attached (platform-specific)
inline bool is_debugger_attached() {
    #if defined(CORE_PLATFORM_WINDOWS)
        return ::IsDebuggerPresent() != 0;
    #elif defined(CORE_PLATFORM_LINUX)
        // Check /proc/self/status for TracerPid
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 10) == "TracerPid:") {
                return std::stoi(line.substr(10)) != 0;
            }
        }
        return false;
    #else
        return false;
    #endif
}

// Debug break
inline void debug_break() {
    #if defined(CORE_DEBUG_BUILD)
        #if defined(CORE_PLATFORM_WINDOWS)
            __debugbreak();
        #elif defined(CORE_COMPILER_GCC) || defined(CORE_COMPILER_CLANG)
            __builtin_trap();
        #else
            std::abort();
        #endif
    #endif
}

} // namespace fem::core::debug

// ==============================================================================
// Assertion Macros
// ==============================================================================

#if CORE_ENABLE_ASSERTS

// Basic assertion
#define FEM_ASSERT(expr) \
    do { \
        if (CORE_UNLIKELY(!(expr))) { \
            fem::core::debug::AssertionHandler::handle_assertion( \
                #expr, "", std::source_location::current()); \
        } \
    } while(0)

// Assertion with message
#define FEM_ASSERT_MSG(expr, msg) \
    do { \
        if (CORE_UNLIKELY(!(expr))) { \
            fem::core::debug::AssertionHandler::handle_assertion( \
                #expr, msg, std::source_location::current()); \
        } \
    } while(0)

// Numeric-specific assertions (used in object.h)
#define FEM_NUMERIC_ASSERT(expr) FEM_ASSERT(expr)
#define FEM_NUMERIC_ASSERT_MSG(expr, msg) FEM_ASSERT_MSG(expr, msg)

// Debug-only assertion (only in debug builds)
#ifdef CORE_DEBUG_BUILD
    #define FEM_DEBUG_ASSERT(expr) FEM_ASSERT(expr)
    #define FEM_DEBUG_ASSERT_MSG(expr, msg) FEM_ASSERT_MSG(expr, msg)
#else
    #define FEM_DEBUG_ASSERT(expr) ((void)0)
    #define FEM_DEBUG_ASSERT_MSG(expr, msg) ((void)0)
#endif

// Verify (always evaluates expression, asserts in debug)
#ifdef CORE_DEBUG_BUILD
    #define FEM_VERIFY(expr) FEM_ASSERT(expr)
    #define FEM_VERIFY_MSG(expr, msg) FEM_ASSERT_MSG(expr, msg)
#else
    #define FEM_VERIFY(expr) ((void)(expr))
    #define FEM_VERIFY_MSG(expr, msg) ((void)(expr))
#endif

#else // CORE_ENABLE_ASSERTS disabled

#define FEM_ASSERT(expr) ((void)0)
#define FEM_ASSERT_MSG(expr, msg) ((void)0)
#define FEM_NUMERIC_ASSERT(expr) ((void)0)
#define FEM_NUMERIC_ASSERT_MSG(expr, msg) ((void)0)
#define FEM_DEBUG_ASSERT(expr) ((void)0)
#define FEM_DEBUG_ASSERT_MSG(expr, msg) ((void)0)
#define FEM_VERIFY(expr) ((void)(expr))
#define FEM_VERIFY_MSG(expr, msg) ((void)(expr))

#endif // CORE_ENABLE_ASSERTS

// ==============================================================================
// Precondition/Postcondition Macros
// ==============================================================================

#define FEM_REQUIRES(expr) FEM_ASSERT_MSG(expr, "Precondition violation")
#define FEM_ENSURES(expr) FEM_ASSERT_MSG(expr, "Postcondition violation")
#define FEM_INVARIANT(expr) FEM_ASSERT_MSG(expr, "Invariant violation")

// ==============================================================================
// Debug Logging Macros
// ==============================================================================

#if CORE_ENABLE_LOGGING

#define FEM_LOG_TRACE(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Trace, msg)

#define FEM_LOG_DEBUG(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Debug, msg)

#define FEM_LOG_INFO(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Info, msg)

#define FEM_LOG_WARNING(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Warning, msg)

#define FEM_LOG_ERROR(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Error, msg)

#define FEM_LOG_FATAL(msg) \
    fem::core::debug::DebugLogger::log( \
        fem::core::debug::LogLevel::Fatal, msg)

#else

#define FEM_LOG_TRACE(msg) ((void)0)
#define FEM_LOG_DEBUG(msg) ((void)0)
#define FEM_LOG_INFO(msg) ((void)0)
#define FEM_LOG_WARNING(msg) ((void)0)
#define FEM_LOG_ERROR(msg) ((void)0)
#define FEM_LOG_FATAL(msg) ((void)0)

#endif

// ==============================================================================
// Debug-Only Code Blocks
// ==============================================================================

#ifdef CORE_DEBUG_BUILD
    #define FEM_DEBUG_ONLY(code) code
#else
    #define FEM_DEBUG_ONLY(code)
#endif

// ==============================================================================
// Not Implemented Macro
// ==============================================================================

#define FEM_NOT_IMPLEMENTED() \
    do { \
        FEM_LOG_FATAL("Not implemented"); \
        fem::core::debug::AssertionHandler::handle_assertion( \
            "false", "Function not implemented", \
            std::source_location::current()); \
    } while(0)

// ==============================================================================
// Unreachable Code Macro
// ==============================================================================

#define FEM_UNREACHABLE() \
    do { \
        FEM_LOG_FATAL("Unreachable code reached"); \
        fem::core::debug::AssertionHandler::handle_assertion( \
            "false", "Unreachable code", \
            std::source_location::current()); \
    } while(0)

// ==============================================================================
// Static Assertion with Message
// ==============================================================================

#define FEM_STATIC_ASSERT(expr, msg) \
    static_assert(expr, msg)

// ==============================================================================
// Compile-Time Checks
// ==============================================================================

namespace fem::core::debug {

// Compile-time type checks
template<typename T>
struct type_name {
    static constexpr const char* value = "unknown";
};

#define REGISTER_TYPE_NAME(type) \
    template<> struct type_name<type> { \
        static constexpr const char* value = #type; \
    }

// Register common types
REGISTER_TYPE_NAME(bool);
REGISTER_TYPE_NAME(char);
REGISTER_TYPE_NAME(int);
REGISTER_TYPE_NAME(long);
REGISTER_TYPE_NAME(float);
REGISTER_TYPE_NAME(double);
REGISTER_TYPE_NAME(void*);

} // namespace fem::core::debug

#endif // CORE_CONFIG_DEBUG_H
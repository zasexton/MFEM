#pragma once

#ifndef CORE_CONFIG_CONFIG_H
#define CORE_CONFIG_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <limits>

// ==============================================================================
// Core Library Configuration
// ==============================================================================
// This file contains compile-time configuration, type definitions, and
// constants used throughout the core library. For runtime configuration
// management, see config_manager.hpp
// ==============================================================================

namespace fem {
namespace config {

// ==============================================================================
// Version Information
// ==============================================================================

constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;
constexpr const char* VERSION_STRING = "0.1.0";

// ==============================================================================
// Type Definitions
// ==============================================================================

// Index type used throughout the library
using index_t = std::int64_t;
using size_type = std::size_t;

// ID type for objects
using id_type = std::int64_t;

// Floating point precision
using real_t = double;
using float32_t = float;
using float64_t = double;

// ==============================================================================
// Platform Detection
// ==============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define CORE_PLATFORM_WINDOWS 1
    #define CORE_PLATFORM_NAME "Windows"
#elif defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_MAC
        #define CORE_PLATFORM_MACOS 1
        #define CORE_PLATFORM_NAME "macOS"
    #endif
#elif defined(__linux__)
    #define CORE_PLATFORM_LINUX 1
    #define CORE_PLATFORM_NAME "Linux"
#elif defined(__unix__)
    #define CORE_PLATFORM_UNIX 1
    #define CORE_PLATFORM_NAME "Unix"
#else
    #define CORE_PLATFORM_UNKNOWN 1
    #define CORE_PLATFORM_NAME "Unknown"
#endif

// ==============================================================================
// Compiler Detection
// ==============================================================================

#if defined(__clang__)
    #define CORE_COMPILER_CLANG 1
    #define CORE_COMPILER_NAME "Clang"
    #define CORE_COMPILER_VERSION __clang_version__
#elif defined(__GNUC__)
    #define CORE_COMPILER_GCC 1
    #define CORE_COMPILER_NAME "GCC"
    #define CORE_COMPILER_VERSION __VERSION__
#elif defined(_MSC_VER)
    #define CORE_COMPILER_MSVC 1
    #define CORE_COMPILER_NAME "MSVC"
    #define CORE_COMPILER_VERSION _MSC_VER
#else
    #define CORE_COMPILER_UNKNOWN 1
    #define CORE_COMPILER_NAME "Unknown"
#endif

// ==============================================================================
// Build Configuration
// ==============================================================================

// Debug/Release detection
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
    #define CORE_DEBUG_BUILD 1
    #define CORE_BUILD_TYPE "Debug"
#else
    #define CORE_RELEASE_BUILD 1
    #define CORE_BUILD_TYPE "Release"
#endif

// ==============================================================================
// Feature Toggles
// ==============================================================================

// Enable/disable features based on build configuration
#ifndef CORE_ENABLE_LOGGING
    #define CORE_ENABLE_LOGGING 1
#endif

#ifndef CORE_ENABLE_PROFILING
    #ifdef CORE_DEBUG_BUILD
        #define CORE_ENABLE_PROFILING 1
    #else
        #define CORE_ENABLE_PROFILING 0
    #endif
#endif

#ifndef CORE_ENABLE_ASSERTS
    #ifdef CORE_DEBUG_BUILD
        #define CORE_ENABLE_ASSERTS 1
    #else
        #define CORE_ENABLE_ASSERTS 0
    #endif
#endif

#ifndef CORE_ENABLE_THREADING
    #define CORE_ENABLE_THREADING 1
#endif

#ifndef CORE_ENABLE_EXCEPTIONS
    #define CORE_ENABLE_EXCEPTIONS 1
#endif

// ==============================================================================
// Memory Configuration
// ==============================================================================

// Memory alignment
constexpr size_type DEFAULT_ALIGNMENT = alignof(std::max_align_t);
constexpr size_type CACHE_LINE_SIZE = 64;  // Common cache line size

// Memory pool sizes
constexpr size_type SMALL_OBJECT_SIZE = 256;
constexpr size_type PAGE_SIZE = 4096;

// ==============================================================================
// Numeric Limits
// ==============================================================================

// Tolerances for floating point comparisons
constexpr real_t EPSILON = std::numeric_limits<real_t>::epsilon();
constexpr real_t DEFAULT_TOLERANCE = 1e-10;
constexpr real_t STRICT_TOLERANCE = 1e-14;
constexpr real_t LOOSE_TOLERANCE = 1e-6;

// Invalid index marker
constexpr index_t INVALID_INDEX = -1;
constexpr id_type INVALID_ID = -1;

// Maximum values
constexpr size_type MAX_SIZE = std::numeric_limits<size_type>::max();
constexpr index_t MAX_INDEX = std::numeric_limits<index_t>::max();

// ==============================================================================
// String Configuration
// ==============================================================================

constexpr size_type MAX_NAME_LENGTH = 256;
constexpr size_type MAX_PATH_LENGTH = 4096;
constexpr size_type MAX_ERROR_MESSAGE_LENGTH = 1024;

// ==============================================================================
// Thread Configuration
// ==============================================================================

#if CORE_ENABLE_THREADING
    constexpr size_type DEFAULT_THREAD_POOL_SIZE = 0;  // 0 = hardware concurrency
    constexpr size_type MAX_THREAD_POOL_SIZE = 128;
    constexpr size_type DEFAULT_TASK_QUEUE_SIZE = 1024;
#endif

// ==============================================================================
// Export/Import Macros
// ==============================================================================

#if defined(CORE_PLATFORM_WINDOWS)
    #ifdef CORE_BUILD_SHARED
        #ifdef CORE_EXPORTS
            #define CORE_API __declspec(dllexport)
        #else
            #define CORE_API __declspec(dllimport)
        #endif
    #else
        #define CORE_API
    #endif
#else
    #define CORE_API __attribute__((visibility("default")))
#endif

// ==============================================================================
// Inline Macros
// ==============================================================================

#if defined(CORE_COMPILER_MSVC)
    #define CORE_INLINE __forceinline
    #define CORE_NOINLINE __declspec(noinline)
#elif defined(CORE_COMPILER_GCC) || defined(CORE_COMPILER_CLANG)
    #define CORE_INLINE __attribute__((always_inline)) inline
    #define CORE_NOINLINE __attribute__((noinline))
#else
    #define CORE_INLINE inline
    #define CORE_NOINLINE
#endif

// ==============================================================================
// Attribute Macros
// ==============================================================================

// Unused parameter
#define CORE_UNUSED(x) ((void)(x))

// Likely/Unlikely branch hints
#if defined(CORE_COMPILER_GCC) || defined(CORE_COMPILER_CLANG)
    #define CORE_LIKELY(x) __builtin_expect(!!(x), 1)
    #define CORE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define CORE_LIKELY(x) (x)
    #define CORE_UNLIKELY(x) (x)
#endif

// Deprecated marking
#if defined(CORE_COMPILER_MSVC)
    #define CORE_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(CORE_COMPILER_GCC) || defined(CORE_COMPILER_CLANG)
    #define CORE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
    #define CORE_DEPRECATED(msg)
#endif

// ==============================================================================
// Utility Macros
// ==============================================================================

// Stringify
#define CORE_STRINGIFY(x) #x
#define CORE_STRINGIFY_MACRO(x) CORE_STRINGIFY(x)

// Concatenate
#define CORE_CONCAT(a, b) a##b
#define CORE_CONCAT_MACRO(a, b) CORE_CONCAT(a, b)

// Array size
template<typename T, size_type N>
constexpr size_type array_size(T (&)[N]) { return N; }

#define CORE_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// ==============================================================================
// Namespace Aliases
// ==============================================================================

namespace mfem = fem;  // Legacy compatibility alias

} // namespace config

// Make commonly used types available in fem namespace
namespace core {
    using index_t = config::index_t;
    using size_type = config::size_type;
    using id_type = config::id_type;
    using real_t = config::real_t;
} // namespace core

} // namespace fem

#endif // CORE_CONFIG_CONFIG_H
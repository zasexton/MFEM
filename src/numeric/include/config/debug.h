/**
 * @file debug.h
 * @brief Debug utilities, assertions, and error handling for FEM Numeric Library
 *
 * This header provides debugging macros, assertions, logging utilities,
 * and error handling mechanisms for development and debugging.
 */

#ifndef FEM_NUMERIC_DEBUG_H
#define FEM_NUMERIC_DEBUG_H

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <source_location>  // C++20 feature if available

#include "compiler.h"
#include "platform.h"

// ============================================================================
// Debug Mode Detection
// ============================================================================

#ifndef FEM_NUMERIC_DEBUG
  #if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
    #define FEM_NUMERIC_DEBUG 1
  #else
    #define FEM_NUMERIC_DEBUG 0
  #endif
#endif

// ============================================================================
// Logging Levels
// ============================================================================

#define FEM_NUMERIC_LOG_LEVEL_NONE    0
#define FEM_NUMERIC_LOG_LEVEL_ERROR   1
#define FEM_NUMERIC_LOG_LEVEL_WARNING 2
#define FEM_NUMERIC_LOG_LEVEL_INFO    3
#define FEM_NUMERIC_LOG_LEVEL_DEBUG   4
#define FEM_NUMERIC_LOG_LEVEL_TRACE   5

#ifndef FEM_NUMERIC_LOG_LEVEL
  #if FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_LOG_LEVEL FEM_NUMERIC_LOG_LEVEL_DEBUG
  #else
    #define FEM_NUMERIC_LOG_LEVEL FEM_NUMERIC_LOG_LEVEL_ERROR
  #endif
#endif

// ============================================================================
// Error Handling Strategy
// ============================================================================

// Define how to handle assertion failures
#ifndef FEM_NUMERIC_ASSERT_HANDLER
  #if FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_ASSERT_HANDLER(msg) \
      (::FEM_NUMERIC_NAMESPACE::debug::assertion_failed( \
        msg, __FILE__, __LINE__, FEM_NUMERIC_FUNCTION))
  #else
    #define FEM_NUMERIC_ASSERT_HANDLER(msg) ((void)0)
  #endif
#endif

// Define how to handle errors
#ifndef FEM_NUMERIC_ERROR_HANDLER
  #define FEM_NUMERIC_ERROR_HANDLER(msg) \
    throw ::FEM_NUMERIC_NAMESPACE::numeric_error(msg)
#endif

FEM_NUMERIC_BEGIN_NAMESPACE

// ============================================================================
// Exception Classes
// ============================================================================

// Base exception class for numeric library
class numeric_error : public std::runtime_error {
public:
    explicit numeric_error(const std::string& msg)
        : std::runtime_error("FEM Numeric Error: " + msg) {}

    explicit numeric_error(const char* msg)
        : std::runtime_error(std::string("FEM Numeric Error: ") + msg) {}
};

// Specific exception types
class dimension_error : public numeric_error {
public:
    explicit dimension_error(const std::string& msg)
        : numeric_error("Dimension Error: " + msg) {}
};

class singular_matrix_error : public numeric_error {
public:
    explicit singular_matrix_error(const std::string& msg = "Matrix is singular")
        : numeric_error("Singular Matrix: " + msg) {}
};

class convergence_error : public numeric_error {
public:
    explicit convergence_error(const std::string& msg)
        : numeric_error("Convergence Error: " + msg) {}
};

class bounds_error : public numeric_error {
public:
    explicit bounds_error(const std::string& msg)
        : numeric_error("Bounds Error: " + msg) {}
};

class allocation_error : public numeric_error {
public:
    explicit allocation_error(const std::string& msg)
        : numeric_error("Allocation Error: " + msg) {}
};

// ============================================================================
// Debug Utilities Namespace
// ============================================================================

namespace debug {

// Get current timestamp string
inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Format file location
inline std::string format_location(const char* file, int line, const char* func) {
    std::stringstream ss;
    ss << file << ":" << line << " in " << func;
    return ss.str();
}

// Assertion failure handler
[[noreturn]] inline void assertion_failed(
    const char* msg,
    const char* file,
    int line,
    const char* func) {

    std::cerr << "\n===== ASSERTION FAILED =====\n"
              << "Message: " << msg << "\n"
              << "Location: " << format_location(file, line, func) << "\n"
              << "Time: " << timestamp() << "\n"
              << "============================\n" << std::endl;

    #if FEM_NUMERIC_DEBUG
        // In debug mode, trigger debugger if attached
        #if defined(FEM_NUMERIC_OS_WINDOWS)
            __debugbreak();
        #elif defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
            __builtin_trap();
        #else
            std::abort();
        #endif
    #else
        std::abort();
    #endif
}

// Logging implementation
enum class LogLevel {
    None = FEM_NUMERIC_LOG_LEVEL_NONE,
    Error = FEM_NUMERIC_LOG_LEVEL_ERROR,
    Warning = FEM_NUMERIC_LOG_LEVEL_WARNING,
    Info = FEM_NUMERIC_LOG_LEVEL_INFO,
    Debug = FEM_NUMERIC_LOG_LEVEL_DEBUG,
    Trace = FEM_NUMERIC_LOG_LEVEL_TRACE
};

inline const char* log_level_string(LogLevel level) {
    switch(level) {
        case LogLevel::Error:   return "[ERROR]";
        case LogLevel::Warning: return "[WARN ]";
        case LogLevel::Info:    return "[INFO ]";
        case LogLevel::Debug:   return "[DEBUG]";
        case LogLevel::Trace:   return "[TRACE]";
        default: return "[?????]";
    }
}

template<typename... Args>
inline void log(LogLevel level, Args&&... args) {
    if (static_cast<int>(level) <= FEM_NUMERIC_LOG_LEVEL) {
        std::stringstream ss;
        ss << timestamp() << " " << log_level_string(level) << " ";
        ((ss << std::forward<Args>(args)), ...);

        if (level == LogLevel::Error) {
            std::cerr << ss.str() << std::endl;
        } else {
            std::cout << ss.str() << std::endl;
        }
    }
}

// Memory tracking utilities
#if FEM_NUMERIC_ENABLE_MEMORY_TRACKING

class MemoryTracker {
private:
    static inline std::size_t allocated_bytes = 0;
    static inline std::size_t peak_bytes = 0;
    static inline std::size_t allocation_count = 0;
    static inline std::size_t deallocation_count = 0;

public:
    static void record_allocation(std::size_t size) {
        allocated_bytes += size;
        allocation_count++;
        if (allocated_bytes > peak_bytes) {
            peak_bytes = allocated_bytes;
        }
    }

    static void record_deallocation(std::size_t size) {
        allocated_bytes -= size;
        deallocation_count++;
    }

    static std::size_t current_usage() { return allocated_bytes; }
    static std::size_t peak_usage() { return peak_bytes; }
    static std::size_t total_allocations() { return allocation_count; }
    static std::size_t total_deallocations() { return deallocation_count; }

    static void print_stats() {
        log(LogLevel::Info, "Memory Statistics:",
            "\n  Current: ", current_usage(), " bytes",
            "\n  Peak: ", peak_usage(), " bytes",
            "\n  Allocations: ", total_allocations(),
            "\n  Deallocations: ", total_deallocations());
    }

    static void reset() {
        allocated_bytes = 0;
        peak_bytes = 0;
        allocation_count = 0;
        deallocation_count = 0;
    }
};

#endif // FEM_NUMERIC_ENABLE_MEMORY_TRACKING

// Performance timing utilities
class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint start_time;
    std::string name;
    bool running;

public:
    explicit Timer(const std::string& timer_name = "")
        : start_time(Clock::now()), name(timer_name), running(true) {}

    void start() {
        start_time = Clock::now();
        running = true;
    }

    double elapsed_seconds() const {
        if (!running) return 0.0;
        auto end_time = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() * 1e-6;
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

    void stop() {
        if (running && !name.empty()) {
            log(LogLevel::Debug, "Timer '", name, "': ",
                elapsed_milliseconds(), " ms");
        }
        running = false;
    }

    ~Timer() {
        stop();
    }
};

// RAII-based scoped timer
class ScopedTimer {
private:
    Timer timer;

public:
    explicit ScopedTimer(const std::string& name) : timer(name) {}
};

} // namespace debug

FEM_NUMERIC_END_NAMESPACE

// ============================================================================
// Assertion Macros
// ============================================================================

// Basic assertion
#if FEM_NUMERIC_DEBUG
  #define FEM_NUMERIC_ASSERT(cond) \
    do { \
      if (!(cond)) { \
        FEM_NUMERIC_ASSERT_HANDLER("Assertion failed: " #cond); \
      } \
    } while(0)
#else
  #define FEM_NUMERIC_ASSERT(cond) ((void)0)
#endif

// Assertion with custom message
#if FEM_NUMERIC_DEBUG
  #define FEM_NUMERIC_ASSERT_MSG(cond, msg) \
    do { \
      if (!(cond)) { \
        std::stringstream ss; \
        ss << "Assertion failed: " << msg; \
        FEM_NUMERIC_ASSERT_HANDLER(ss.str().c_str()); \
      } \
    } while(0)
#else
  #define FEM_NUMERIC_ASSERT_MSG(cond, msg) ((void)0)
#endif

// Debug-only code block
#if FEM_NUMERIC_DEBUG
  #define FEM_NUMERIC_DEBUG_ONLY(code) do { code } while(0)
#else
  #define FEM_NUMERIC_DEBUG_ONLY(code) ((void)0)
#endif

// ============================================================================
// Bounds Checking Macros
// ============================================================================

#if FEM_NUMERIC_ENABLE_BOUNDS_CHECKING
  #define FEM_NUMERIC_BOUNDS_CHECK(index, size) \
    do { \
      if ((index) >= (size)) { \
        std::stringstream ss; \
        ss << "Index " << (index) << " out of bounds [0, " << (size) << ")"; \
        FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
      } \
    } while(0)

  #define FEM_NUMERIC_BOUNDS_CHECK_2D(row, col, rows, cols) \
    do { \
      if ((row) >= (rows) || (col) >= (cols)) { \
        std::stringstream ss; \
        ss << "Index (" << (row) << ", " << (col) << ") out of bounds [0, " \
           << (rows) << ") x [0, " << (cols) << ")"; \
        FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
      } \
    } while(0)
#else
  #define FEM_NUMERIC_BOUNDS_CHECK(index, size) ((void)0)
  #define FEM_NUMERIC_BOUNDS_CHECK_2D(row, col, rows, cols) ((void)0)
#endif

// ============================================================================
// Dimension Checking Macros
// ============================================================================

#define FEM_NUMERIC_CHECK_DIMENSION(actual, expected) \
  do { \
    if ((actual) != (expected)) { \
      std::stringstream ss; \
      ss << "Dimension mismatch: expected " << (expected) \
         << ", got " << (actual); \
      FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
    } \
  } while(0)

#define FEM_NUMERIC_CHECK_COMPATIBLE_DIMENSIONS(dim1, dim2) \
  do { \
    if ((dim1) != (dim2)) { \
      std::stringstream ss; \
      ss << "Incompatible dimensions: " << (dim1) << " vs " << (dim2); \
      FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
    } \
  } while(0)

#define FEM_NUMERIC_CHECK_SQUARE_MATRIX(rows, cols) \
  do { \
    if ((rows) != (cols)) { \
      std::stringstream ss; \
      ss << "Matrix must be square: " << (rows) << "x" << (cols); \
      FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
    } \
  } while(0)

// ============================================================================
// NaN and Infinity Checking
// ============================================================================

#if FEM_NUMERIC_ENABLE_NAN_CHECKING
  #define FEM_NUMERIC_CHECK_FINITE(value) \
    do { \
      if (!::FEM_NUMERIC_NAMESPACE::is_finite(value)) { \
        std::stringstream ss; \
        ss << "Non-finite value detected: " << (value); \
        FEM_NUMERIC_ERROR_HANDLER(ss.str()); \
      } \
    } while(0)

  #define FEM_NUMERIC_CHECK_NOT_NAN(value) \
    do { \
      if (::FEM_NUMERIC_NAMESPACE::is_nan(value)) { \
        FEM_NUMERIC_ERROR_HANDLER("NaN value detected"); \
      } \
    } while(0)
#else
  #define FEM_NUMERIC_CHECK_FINITE(value) ((void)0)
  #define FEM_NUMERIC_CHECK_NOT_NAN(value) ((void)0)
#endif

// ============================================================================
// Logging Macros
// ============================================================================

#define FEM_NUMERIC_LOG_ERROR(...) \
  ::FEM_NUMERIC_NAMESPACE::debug::log( \
    ::FEM_NUMERIC_NAMESPACE::debug::LogLevel::Error, __VA_ARGS__)

#define FEM_NUMERIC_LOG_WARNING(...) \
  ::FEM_NUMERIC_NAMESPACE::debug::log( \
    ::FEM_NUMERIC_NAMESPACE::debug::LogLevel::Warning, __VA_ARGS__)

#define FEM_NUMERIC_LOG_INFO(...) \
  ::FEM_NUMERIC_NAMESPACE::debug::log( \
    ::FEM_NUMERIC_NAMESPACE::debug::LogLevel::Info, __VA_ARGS__)

#define FEM_NUMERIC_LOG_DEBUG(...) \
  ::FEM_NUMERIC_NAMESPACE::debug::log( \
    ::FEM_NUMERIC_NAMESPACE::debug::LogLevel::Debug, __VA_ARGS__)

#define FEM_NUMERIC_LOG_TRACE(...) \
  ::FEM_NUMERIC_NAMESPACE::debug::log( \
    ::FEM_NUMERIC_NAMESPACE::debug::LogLevel::Trace, __VA_ARGS__)

// ============================================================================
// Performance Assertions
// ============================================================================

// Assert alignment for SIMD operations
#if FEM_NUMERIC_DEBUG && FEM_NUMERIC_ENABLE_SIMD
  #define FEM_NUMERIC_ASSERT_ALIGNED(ptr, alignment) \
    FEM_NUMERIC_ASSERT_MSG( \
      (reinterpret_cast<std::uintptr_t>(ptr) % (alignment)) == 0, \
      "Pointer not aligned to " << (alignment) << " bytes")
#else
  #define FEM_NUMERIC_ASSERT_ALIGNED(ptr, alignment) ((void)0)
#endif

// Assert size is suitable for vectorization
#if FEM_NUMERIC_DEBUG && FEM_NUMERIC_ENABLE_SIMD
  #define FEM_NUMERIC_ASSERT_VECTORIZABLE(size) \
    FEM_NUMERIC_ASSERT_MSG( \
      ((size) % FEM_NUMERIC_SIMD_DOUBLES) == 0, \
      "Size " << (size) << " not divisible by SIMD width " \
      << FEM_NUMERIC_SIMD_DOUBLES)
#else
  #define FEM_NUMERIC_ASSERT_VECTORIZABLE(size) ((void)0)
#endif

// ============================================================================
// Memory Tracking Macros
// ============================================================================

#if FEM_NUMERIC_ENABLE_MEMORY_TRACKING
  #define FEM_NUMERIC_TRACK_ALLOCATION(size) \
    ::FEM_NUMERIC_NAMESPACE::debug::MemoryTracker::record_allocation(size)

  #define FEM_NUMERIC_TRACK_DEALLOCATION(size) \
    ::FEM_NUMERIC_NAMESPACE::debug::MemoryTracker::record_deallocation(size)

  #define FEM_NUMERIC_PRINT_MEMORY_STATS() \
    ::FEM_NUMERIC_NAMESPACE::debug::MemoryTracker::print_stats()
#else
  #define FEM_NUMERIC_TRACK_ALLOCATION(size) ((void)0)
  #define FEM_NUMERIC_TRACK_DEALLOCATION(size) ((void)0)
  #define FEM_NUMERIC_PRINT_MEMORY_STATS() ((void)0)
#endif

// ============================================================================
// Timing Macros
// ============================================================================

#if FEM_NUMERIC_ENABLE_PROFILING
  #define FEM_NUMERIC_TIME_SCOPE(name) \
    ::FEM_NUMERIC_NAMESPACE::debug::ScopedTimer \
      FEM_NUMERIC_UNIQUE_NAME(_timer_)(name)

  #define FEM_NUMERIC_START_TIMER(name) \
    ::FEM_NUMERIC_NAMESPACE::debug::Timer name##_timer(#name); \
    name##_timer.start()

  #define FEM_NUMERIC_STOP_TIMER(name) \
    name##_timer.stop()
#else
  #define FEM_NUMERIC_TIME_SCOPE(name) ((void)0)
  #define FEM_NUMERIC_START_TIMER(name) ((void)0)
  #define FEM_NUMERIC_STOP_TIMER(name) ((void)0)
#endif

// ============================================================================
// Invariant Checking
// ============================================================================

// Check class invariants in debug mode
#if FEM_NUMERIC_DEBUG
  #define FEM_NUMERIC_CHECK_INVARIANTS() check_invariants()
#else
  #define FEM_NUMERIC_CHECK_INVARIANTS() ((void)0)
#endif

// ============================================================================
// Static Assertions
// ============================================================================

// Compile-time assertion with message
#define FEM_NUMERIC_STATIC_ASSERT(cond, msg) \
  static_assert(cond, "FEM Numeric: " msg)

// ============================================================================
// Precondition and Postcondition Contracts
// ============================================================================

#if FEM_NUMERIC_DEBUG
  #define FEM_NUMERIC_REQUIRES(cond) \
    FEM_NUMERIC_ASSERT_MSG(cond, "Precondition violated: " #cond)

  #define FEM_NUMERIC_ENSURES(cond) \
    FEM_NUMERIC_ASSERT_MSG(cond, "Postcondition violated: " #cond)
#else
  #define FEM_NUMERIC_REQUIRES(cond) ((void)0)
  #define FEM_NUMERIC_ENSURES(cond) ((void)0)
#endif

// ============================================================================
// Operation Counting
// ============================================================================

#if FEM_NUMERIC_ENABLE_OP_COUNTING

FEM_NUMERIC_BEGIN_NAMESPACE

class OperationCounter {
private:
    static inline std::size_t additions = 0;
    static inline std::size_t multiplications = 0;
    static inline std::size_t divisions = 0;
    static inline std::size_t comparisons = 0;

public:
    static void count_add() { ++additions; }
    static void count_mul() { ++multiplications; }
    static void count_div() { ++divisions; }
    static void count_cmp() { ++comparisons; }

    static void reset() {
        additions = 0;
        multiplications = 0;
        divisions = 0;
        comparisons = 0;
    }

    static void print_stats() {
        debug::log(debug::LogLevel::Info, "Operation counts:",
                  "\n  Additions: ", additions,
                  "\n  Multiplications: ", multiplications,
                  "\n  Divisions: ", divisions,
                  "\n  Comparisons: ", comparisons);
    }

    static std::size_t flop_count() {
        return additions + multiplications + divisions;
    }
};

FEM_NUMERIC_END_NAMESPACE

  #define FEM_NUMERIC_COUNT_ADD() \
    ::FEM_NUMERIC_NAMESPACE::OperationCounter::count_add()
  #define FEM_NUMERIC_COUNT_MUL() \
    ::FEM_NUMERIC_NAMESPACE::OperationCounter::count_mul()
  #define FEM_NUMERIC_COUNT_DIV() \
    ::FEM_NUMERIC_NAMESPACE::OperationCounter::count_div()
  #define FEM_NUMERIC_COUNT_CMP() \
    ::FEM_NUMERIC_NAMESPACE::OperationCounter::count_cmp()
  #define FEM_NUMERIC_PRINT_OP_STATS() \
    ::FEM_NUMERIC_NAMESPACE::OperationCounter::print_stats()
#else
  #define FEM_NUMERIC_COUNT_ADD() ((void)0)
  #define FEM_NUMERIC_COUNT_MUL() ((void)0)
  #define FEM_NUMERIC_COUNT_DIV() ((void)0)
  #define FEM_NUMERIC_COUNT_CMP() ((void)0)
  #define FEM_NUMERIC_PRINT_OP_STATS() ((void)0)
#endif

// ============================================================================
// Unreachable Code
// ============================================================================

#define FEM_NUMERIC_UNREACHABLE() \
  do { \
    FEM_NUMERIC_ASSERT_MSG(false, "Unreachable code reached"); \
    FEM_NUMERIC_UNREACHABLE(); \
  } while(0)

// ============================================================================
// Deprecated Function Warning
// ============================================================================

#define FEM_NUMERIC_DEPRECATED_MSG(msg) \
  FEM_NUMERIC_DEPRECATED(msg)

#endif // FEM_NUMERIC_DEBUG_H
/**
 * @file config.h
 * @brief Main configuration header for FEM Numeric Library
 *
 * This header aggregates all configuration settings and provides the main
 * entry point for library configuration. It includes compiler detection,
 * platform specifics, feature flags, and sets up the library namespace.
 */

#ifndef FEM_NUMERIC_CONFIG_H
#define FEM_NUMERIC_CONFIG_H

// Include configuration components
#include "compiler.h"
#include "platform.h"
#include "features.h"

// Library version information
#define FEM_NUMERIC_VERSION_MAJOR 1
#define FEM_NUMERIC_VERSION_MINOR 0
#define FEM_NUMERIC_VERSION_PATCH 0
#define FEM_NUMERIC_VERSION_STRING "1.0.0"

// Namespace configuration
#ifndef FEM_NUMERIC_NAMESPACE
  #define FEM_NUMERIC_NAMESPACE fem::numeric
#endif

// Begin namespace macro
#define FEM_NUMERIC_BEGIN_NAMESPACE \
  namespace FEM_NUMERIC_NAMESPACE {

// End namespace macro
#define FEM_NUMERIC_END_NAMESPACE \
  }

// Using namespace macro
#define FEM_NUMERIC_USE_NAMESPACE \
  using namespace FEM_NUMERIC_NAMESPACE;

// Default floating point type
#ifndef FEM_NUMERIC_DEFAULT_SCALAR
  #define FEM_NUMERIC_DEFAULT_SCALAR double
#endif

// Default index type for matrices/vectors
#ifndef FEM_NUMERIC_DEFAULT_INDEX
  #define FEM_NUMERIC_DEFAULT_INDEX std::size_t
#endif

// Small matrix threshold (matrices smaller than this use stack allocation)
#ifndef FEM_NUMERIC_SMALL_MATRIX_SIZE
  #define FEM_NUMERIC_SMALL_MATRIX_SIZE 32
#endif

// Small vector threshold
#ifndef FEM_NUMERIC_SMALL_VECTOR_SIZE
  #define FEM_NUMERIC_SMALL_VECTOR_SIZE 64
#endif

// Memory alignment for SIMD operations
#ifndef FEM_NUMERIC_ALIGNMENT
  #if FEM_NUMERIC_HAS_AVX512
    #define FEM_NUMERIC_ALIGNMENT 64
  #elif FEM_NUMERIC_HAS_AVX
    #define FEM_NUMERIC_ALIGNMENT 32
  #elif FEM_NUMERIC_HAS_SSE
    #define FEM_NUMERIC_ALIGNMENT 16
  #else
    #define FEM_NUMERIC_ALIGNMENT alignof(std::max_align_t)
  #endif
#endif

// Cache line size (for avoiding false sharing)
#ifndef FEM_NUMERIC_CACHE_LINE_SIZE
  #define FEM_NUMERIC_CACHE_LINE_SIZE 64
#endif

// Enable expression templates by default
#ifndef FEM_NUMERIC_DISABLE_EXPRESSION_TEMPLATES
  #define FEM_NUMERIC_USE_EXPRESSION_TEMPLATES 1
#else
  #define FEM_NUMERIC_USE_EXPRESSION_TEMPLATES 0
#endif

// Enable automatic differentiation support
#ifndef FEM_NUMERIC_DISABLE_AUTODIFF
  #define FEM_NUMERIC_HAS_AUTODIFF 1
#else
  #define FEM_NUMERIC_HAS_AUTODIFF 0
#endif

// Thread-local storage for AD tape (reverse mode)
#if FEM_NUMERIC_HAS_THREAD_LOCAL
  #define FEM_NUMERIC_THREAD_LOCAL thread_local
#else
  #define FEM_NUMERIC_THREAD_LOCAL
#endif

// Parallel execution policy
#ifndef FEM_NUMERIC_PARALLEL_THRESHOLD
  #define FEM_NUMERIC_PARALLEL_THRESHOLD 1000  // Min elements for parallel execution
#endif

// Sparse matrix format defaults
#ifndef FEM_NUMERIC_DEFAULT_SPARSE_FORMAT
  #define FEM_NUMERIC_DEFAULT_SPARSE_FORMAT CSR
#endif

// Assembly atomics configuration
#if FEM_NUMERIC_HAS_ATOMIC_DOUBLE
  #define FEM_NUMERIC_USE_ATOMIC_ASSEMBLY 1
#else
  #define FEM_NUMERIC_USE_ATOMIC_ASSEMBLY 0
#endif

// Matrix-free operation configuration
#ifndef FEM_NUMERIC_MATRIX_FREE_BLOCK_SIZE
  #define FEM_NUMERIC_MATRIX_FREE_BLOCK_SIZE 32
#endif

// Block matrix configuration
#ifndef FEM_NUMERIC_MAX_BLOCK_FIELDS
  #define FEM_NUMERIC_MAX_BLOCK_FIELDS 16  // Max fields in block systems
#endif

// Tolerance for numerical comparisons
#ifndef FEM_NUMERIC_EPSILON
  #define FEM_NUMERIC_EPSILON 1e-14
#endif

// Newton solver defaults
#ifndef FEM_NUMERIC_NEWTON_MAX_ITER
  #define FEM_NUMERIC_NEWTON_MAX_ITER 100
#endif

#ifndef FEM_NUMERIC_NEWTON_TOL
  #define FEM_NUMERIC_NEWTON_TOL 1e-10
#endif

// Iterative solver defaults
#ifndef FEM_NUMERIC_GMRES_RESTART
  #define FEM_NUMERIC_GMRES_RESTART 30
#endif

#ifndef FEM_NUMERIC_CG_MAX_ITER
  #define FEM_NUMERIC_CG_MAX_ITER 1000
#endif

// Memory pool configuration for AD
#ifndef FEM_NUMERIC_AD_POOL_CHUNK_SIZE
  #define FEM_NUMERIC_AD_POOL_CHUNK_SIZE 4096
#endif

// Enable bounds checking in debug mode
#ifdef FEM_NUMERIC_DEBUG
  #ifndef FEM_NUMERIC_ENABLE_BOUNDS_CHECK
    #define FEM_NUMERIC_ENABLE_BOUNDS_CHECK 1
  #endif
#else
  #ifndef FEM_NUMERIC_ENABLE_BOUNDS_CHECK
    #define FEM_NUMERIC_ENABLE_BOUNDS_CHECK 0
  #endif
#endif

// Performance profiling
#ifndef FEM_NUMERIC_ENABLE_PROFILING
  #define FEM_NUMERIC_ENABLE_PROFILING 0
#endif

// Logging configuration
#ifndef FEM_NUMERIC_LOG_LEVEL
  #ifdef FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_LOG_LEVEL 3  // DEBUG
  #else
    #define FEM_NUMERIC_LOG_LEVEL 1  // ERROR only
  #endif
#endif

// Export/import macros for shared library builds
#if defined(FEM_NUMERIC_SHARED_LIBRARY)
  #if defined(_WIN32) || defined(_WIN64)
    #ifdef FEM_NUMERIC_EXPORTS
      #define FEM_NUMERIC_API __declspec(dllexport)
    #else
      #define FEM_NUMERIC_API __declspec(dllimport)
    #endif
  #else
    #define FEM_NUMERIC_API __attribute__((visibility("default")))
  #endif
#else
  #define FEM_NUMERIC_API
#endif

// Helper macros for concatenation
#define FEM_NUMERIC_CONCAT_IMPL(a, b) a##b
#define FEM_NUMERIC_CONCAT(a, b) FEM_NUMERIC_CONCAT_IMPL(a, b)

// Unique variable name generation (useful for RAII objects)
#define FEM_NUMERIC_UNIQUE_NAME(base) FEM_NUMERIC_CONCAT(base, __LINE__)

// Stringify macro
#define FEM_NUMERIC_STRINGIFY_IMPL(x) #x
#define FEM_NUMERIC_STRINGIFY(x) FEM_NUMERIC_STRINGIFY_IMPL(x)

// Static assertion with message (pre-C++17 compatibility)
#if FEM_NUMERIC_CPP_VERSION >= 17
  #define FEM_NUMERIC_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
  #define FEM_NUMERIC_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

// Deprecation warnings
#if FEM_NUMERIC_HAS_CPP14_DEPRECATED
  #define FEM_NUMERIC_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(__GNUC__) || defined(__clang__)
  #define FEM_NUMERIC_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
  #define FEM_NUMERIC_DEPRECATED(msg) __declspec(deprecated(msg))
#else
  #define FEM_NUMERIC_DEPRECATED(msg)
#endif

// Performance hints
#if FEM_NUMERIC_HAS_BUILTIN_EXPECT
  #define FEM_NUMERIC_LIKELY(x) __builtin_expect(!!(x), 1)
  #define FEM_NUMERIC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define FEM_NUMERIC_LIKELY(x) (x)
  #define FEM_NUMERIC_UNLIKELY(x) (x)
#endif

// Configure header-only mode
#ifdef FEM_NUMERIC_HEADER_ONLY
  #define FEM_NUMERIC_INLINE inline
#else
  #define FEM_NUMERIC_INLINE
#endif

#endif // FEM_NUMERIC_CONFIG_H

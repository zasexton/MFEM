/**
 * @file compiler.h
 * @brief Compiler detection and compiler-specific feature configuration
 *
 * This header detects the compiler being used and sets up compiler-specific
 * macros for optimizations, intrinsics, and language features.
 */

#ifndef FEM_NUMERIC_COMPILER_H
#define FEM_NUMERIC_COMPILER_H

// Detect C++ standard version
#ifdef _MSVC_LANG
#define FEM_NUMERIC_CPP_VERSION (_MSVC_LANG / 100)
#else
#define FEM_NUMERIC_CPP_VERSION (__cplusplus / 100)
#endif

// Minimum required C++ standard
#if FEM_NUMERIC_CPP_VERSION < 2017
#error "FEM Numeric Library requires C++17 or later"
#endif

// Helper macro to enforce C++20 where required
#define MFEM_REQUIRE_CXX20                                                     \
  static_assert(FEM_NUMERIC_CPP_VERSION >= 2020,                               \
                "MFEM numeric requires C++20 or later")

// Compiler detection
#if defined(__clang__)
#define FEM_NUMERIC_COMPILER_CLANG
#define FEM_NUMERIC_COMPILER_VERSION                                           \
  (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#define FEM_NUMERIC_COMPILER_NAME "Clang"
#elif defined(__INTEL_COMPILER)
#define FEM_NUMERIC_COMPILER_INTEL
#define FEM_NUMERIC_COMPILER_VERSION __INTEL_COMPILER
#define FEM_NUMERIC_COMPILER_NAME "Intel C++"
#elif defined(__GNUC__)
#define FEM_NUMERIC_COMPILER_GCC
#define FEM_NUMERIC_COMPILER_VERSION                                           \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#define FEM_NUMERIC_COMPILER_NAME "GCC"
#elif defined(_MSC_VER)
#define FEM_NUMERIC_COMPILER_MSVC
#define FEM_NUMERIC_COMPILER_VERSION _MSC_VER
#define FEM_NUMERIC_COMPILER_NAME "MSVC"
#else
#warning "Unknown compiler - some optimizations may be disabled"
#define FEM_NUMERIC_COMPILER_UNKNOWN
#define FEM_NUMERIC_COMPILER_NAME "Unknown"
#endif

// Compiler-specific attributes
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_ALWAYS_INLINE __attribute__((always_inline)) inline
#define FEM_NUMERIC_NEVER_INLINE __attribute__((noinline))
#define FEM_NUMERIC_RESTRICT __restrict__
#define FEM_NUMERIC_ALIGN(x) __attribute__((aligned(x)))
#define FEM_NUMERIC_PACKED __attribute__((packed))
#define FEM_NUMERIC_UNUSED __attribute__((unused))
#define FEM_NUMERIC_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#define FEM_NUMERIC_HOT __attribute__((hot))
#define FEM_NUMERIC_COLD __attribute__((cold))
#define FEM_NUMERIC_PURE __attribute__((pure))
#define FEM_NUMERIC_CONST __attribute__((const))
#define FEM_NUMERIC_FLATTEN __attribute__((flatten))
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_ALWAYS_INLINE __forceinline
#define FEM_NUMERIC_NEVER_INLINE __declspec(noinline)
#define FEM_NUMERIC_RESTRICT __restrict
#define FEM_NUMERIC_ALIGN(x) __declspec(align(x))
#define FEM_NUMERIC_PACKED
#define FEM_NUMERIC_UNUSED
#define FEM_NUMERIC_WARN_UNUSED_RESULT _Check_return_
#define FEM_NUMERIC_HOT
#define FEM_NUMERIC_COLD
#define FEM_NUMERIC_PURE
#define FEM_NUMERIC_CONST
#define FEM_NUMERIC_FLATTEN
#else
#define FEM_NUMERIC_ALWAYS_INLINE inline
#define FEM_NUMERIC_NEVER_INLINE
#define FEM_NUMERIC_RESTRICT
#define FEM_NUMERIC_ALIGN(x)
#define FEM_NUMERIC_PACKED
#define FEM_NUMERIC_UNUSED
#define FEM_NUMERIC_WARN_UNUSED_RESULT
#define FEM_NUMERIC_HOT
#define FEM_NUMERIC_COLD
#define FEM_NUMERIC_PURE
#define FEM_NUMERIC_CONST
#define FEM_NUMERIC_FLATTEN
#endif

// Prefetch hints
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_PREFETCH(addr, rw, locality)                               \
  __builtin_prefetch(addr, rw, locality)
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#include <intrin.h>
#define FEM_NUMERIC_PREFETCH(addr, rw, locality)                               \
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#else
#define FEM_NUMERIC_PREFETCH(addr, rw, locality) ((void)0)
#endif

// Branch prediction hints
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_HAS_BUILTIN_EXPECT 1
#else
#define FEM_NUMERIC_HAS_BUILTIN_EXPECT 0
#endif

// Branch prediction convenience macros
#if FEM_NUMERIC_HAS_BUILTIN_EXPECT
#define MFEM_LIKELY(x) __builtin_expect(!!(x), 1)
#define MFEM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MFEM_LIKELY(x) (x)
#define MFEM_UNLIKELY(x) (x)
#endif

// Assume hints for optimizer
#if defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_ASSUME(cond) __builtin_assume(cond)
#elif defined(FEM_NUMERIC_COMPILER_GCC) &&                                     \
    FEM_NUMERIC_COMPILER_VERSION >= 130000
#define FEM_NUMERIC_ASSUME(cond) [[assume(cond)]]
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_ASSUME(cond) __assume(cond)
#else
#define FEM_NUMERIC_ASSUME(cond) ((void)0)
#endif

// Unreachable code marker
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_UNREACHABLE() __builtin_unreachable()
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_UNREACHABLE() __assume(0)
#else
#define FEM_NUMERIC_UNREACHABLE() ((void)0)
#endif

// C++ feature detection
#ifdef __has_cpp_attribute
#define FEM_NUMERIC_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define FEM_NUMERIC_HAS_CPP_ATTRIBUTE(x) 0
#endif

// [[nodiscard]] attribute
#if FEM_NUMERIC_HAS_CPP_ATTRIBUTE(nodiscard)
#define FEM_NUMERIC_NODISCARD [[nodiscard]]
#else
#define FEM_NUMERIC_NODISCARD
#endif

// [[maybe_unused]] attribute
#if FEM_NUMERIC_HAS_CPP_ATTRIBUTE(maybe_unused)
#define FEM_NUMERIC_MAYBE_UNUSED [[maybe_unused]]
#else
#define FEM_NUMERIC_MAYBE_UNUSED FEM_NUMERIC_UNUSED
#endif

// [[fallthrough]] attribute
#if FEM_NUMERIC_HAS_CPP_ATTRIBUTE(fallthrough)
#define FEM_NUMERIC_FALLTHROUGH [[fallthrough]]
#else
#define FEM_NUMERIC_FALLTHROUGH ((void)0)
#endif

// [[deprecated]] attribute
#if FEM_NUMERIC_HAS_CPP_ATTRIBUTE(deprecated)
#define FEM_NUMERIC_HAS_CPP14_DEPRECATED 1
#else
#define FEM_NUMERIC_HAS_CPP14_DEPRECATED 0
#endif

// [[no_unique_address]] attribute (C++20)
#if FEM_NUMERIC_CPP_VERSION >= 2020 &&                                         \
    FEM_NUMERIC_HAS_CPP_ATTRIBUTE(no_unique_address)
#define FEM_NUMERIC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define FEM_NUMERIC_NO_UNIQUE_ADDRESS
#endif

// Thread-local storage
#if defined(FEM_NUMERIC_COMPILER_MSVC)
#if FEM_NUMERIC_COMPILER_VERSION >= 1900
#define FEM_NUMERIC_HAS_THREAD_LOCAL 1
#else
#define FEM_NUMERIC_HAS_THREAD_LOCAL 0
#endif
#else
#define FEM_NUMERIC_HAS_THREAD_LOCAL 1
#endif

// Inline variable support (C++17)
#if FEM_NUMERIC_CPP_VERSION >= 2017
#define FEM_NUMERIC_INLINE_VAR inline
#else
#define FEM_NUMERIC_INLINE_VAR
#endif

// Structured bindings (C++17)
#if FEM_NUMERIC_CPP_VERSION >= 2017
#define FEM_NUMERIC_HAS_STRUCTURED_BINDINGS 1
#else
#define FEM_NUMERIC_HAS_STRUCTURED_BINDINGS 0
#endif

// if constexpr (C++17)
#if FEM_NUMERIC_CPP_VERSION >= 2017
#define FEM_NUMERIC_HAS_IF_CONSTEXPR 1
#define FEM_NUMERIC_IF_CONSTEXPR if constexpr
#else
#define FEM_NUMERIC_HAS_IF_CONSTEXPR 0
#define FEM_NUMERIC_IF_CONSTEXPR if
#endif

// Concepts (C++20)
#if FEM_NUMERIC_CPP_VERSION >= 2020 && defined(__cpp_concepts)
#define FEM_NUMERIC_HAS_CONCEPTS 1
#else
#define FEM_NUMERIC_HAS_CONCEPTS 0
#endif

// Coroutines (C++20)
#if FEM_NUMERIC_CPP_VERSION >= 2020 && defined(__cpp_coroutines)
#define FEM_NUMERIC_HAS_COROUTINES 1
#else
#define FEM_NUMERIC_HAS_COROUTINES 0
#endif

// Three-way comparison (C++20)
#if FEM_NUMERIC_CPP_VERSION >= 2020 && defined(__cpp_impl_three_way_comparison)
#define FEM_NUMERIC_HAS_SPACESHIP 1
#else
#define FEM_NUMERIC_HAS_SPACESHIP 0
#endif

// OpenMP support
#ifdef _OPENMP
#define FEM_NUMERIC_HAS_OPENMP 1
#define FEM_NUMERIC_OPENMP_VERSION _OPENMP
#else
#define FEM_NUMERIC_HAS_OPENMP 0
#endif

// Pragma macros
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_PRAGMA(x) _Pragma(#x)
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_PRAGMA(x) __pragma(x)
#else
#define FEM_NUMERIC_PRAGMA(x)
#endif

// Loop optimization hints
#if defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_VECTORIZE_LOOP                                             \
  FEM_NUMERIC_PRAGMA(clang loop vectorize(enable))
#define FEM_NUMERIC_UNROLL_LOOP(n)                                             \
  FEM_NUMERIC_PRAGMA(clang loop unroll_count(n))
#define FEM_NUMERIC_NO_VECTORIZE                                               \
  FEM_NUMERIC_PRAGMA(clang loop vectorize(disable))
#elif defined(FEM_NUMERIC_COMPILER_GCC)
#define FEM_NUMERIC_VECTORIZE_LOOP FEM_NUMERIC_PRAGMA(GCC ivdep)
#define FEM_NUMERIC_UNROLL_LOOP(n) FEM_NUMERIC_PRAGMA(GCC unroll n)
#define FEM_NUMERIC_NO_VECTORIZE FEM_NUMERIC_PRAGMA(GCC novector)
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_VECTORIZE_LOOP FEM_NUMERIC_PRAGMA(loop(ivdep))
#define FEM_NUMERIC_UNROLL_LOOP(n)
#define FEM_NUMERIC_NO_VECTORIZE
#else
#define FEM_NUMERIC_VECTORIZE_LOOP
#define FEM_NUMERIC_UNROLL_LOOP(n)
#define FEM_NUMERIC_NO_VECTORIZE
#endif

// Diagnostic control
#if defined(FEM_NUMERIC_COMPILER_GCC)
#define FEM_NUMERIC_DIAGNOSTIC_PUSH FEM_NUMERIC_PRAGMA(GCC diagnostic push)
#define FEM_NUMERIC_DIAGNOSTIC_POP FEM_NUMERIC_PRAGMA(GCC diagnostic pop)
#define FEM_NUMERIC_DISABLE_WARNING(w)                                         \
  FEM_NUMERIC_PRAGMA(GCC diagnostic ignored #w)
#elif defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_DIAGNOSTIC_PUSH FEM_NUMERIC_PRAGMA(clang diagnostic push)
#define FEM_NUMERIC_DIAGNOSTIC_POP FEM_NUMERIC_PRAGMA(clang diagnostic pop)
#define FEM_NUMERIC_DISABLE_WARNING(w)                                         \
  FEM_NUMERIC_PRAGMA(clang diagnostic ignored #w)
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_DIAGNOSTIC_PUSH FEM_NUMERIC_PRAGMA(warning(push))
#define FEM_NUMERIC_DIAGNOSTIC_POP FEM_NUMERIC_PRAGMA(warning(pop))
#define FEM_NUMERIC_DISABLE_WARNING(w) FEM_NUMERIC_PRAGMA(warning(disable : w))
#else
#define FEM_NUMERIC_DIAGNOSTIC_PUSH
#define FEM_NUMERIC_DIAGNOSTIC_POP
#define FEM_NUMERIC_DISABLE_WARNING(w)
#endif

// Function name macro
#if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
#define FEM_NUMERIC_FUNCTION __PRETTY_FUNCTION__
#elif defined(FEM_NUMERIC_COMPILER_MSVC)
#define FEM_NUMERIC_FUNCTION __FUNCSIG__
#else
#define FEM_NUMERIC_FUNCTION __func__
#endif

// Check for builtin functions
#ifdef __has_builtin
#define FEM_NUMERIC_HAS_BUILTIN(x) __has_builtin(x)
#else
#define FEM_NUMERIC_HAS_BUILTIN(x) 0
#endif

// Check for include files
#ifdef __has_include
#define FEM_NUMERIC_HAS_INCLUDE(x) __has_include(x)
#else
#define FEM_NUMERIC_HAS_INCLUDE(x) 0
#endif

#endif // FEM_NUMERIC_COMPILER_H
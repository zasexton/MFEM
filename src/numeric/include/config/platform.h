/**
 * @file platform.h
 * @brief Platform and architecture detection for FEM Numeric Library
 *
 * This header detects the operating system, CPU architecture, endianness,
 * and available SIMD instruction sets.
 */

#ifndef FEM_NUMERIC_PLATFORM_H
#define FEM_NUMERIC_PLATFORM_H

// Operating System Detection
#if defined(_WIN32) || defined(_WIN64)
  #define FEM_NUMERIC_OS_WINDOWS
  #ifdef _WIN64
    #define FEM_NUMERIC_OS_WIN64
  #else
    #define FEM_NUMERIC_OS_WIN32
  #endif
  #define FEM_NUMERIC_OS_NAME "Windows"
#elif defined(__APPLE__) && defined(__MACH__)
  #define FEM_NUMERIC_OS_MACOS
  #define FEM_NUMERIC_OS_POSIX
  #define FEM_NUMERIC_OS_NAME "macOS"
  #include <TargetConditionals.h>
  #if TARGET_OS_IOS
    #define FEM_NUMERIC_OS_IOS
  #endif
#elif defined(__linux__)
  #define FEM_NUMERIC_OS_LINUX
  #define FEM_NUMERIC_OS_POSIX
  #define FEM_NUMERIC_OS_NAME "Linux"
#elif defined(__unix__) || defined(__unix)
  #define FEM_NUMERIC_OS_UNIX
  #define FEM_NUMERIC_OS_POSIX
  #define FEM_NUMERIC_OS_NAME "Unix"
#elif defined(__FreeBSD__)
  #define FEM_NUMERIC_OS_FREEBSD
  #define FEM_NUMERIC_OS_POSIX
  #define FEM_NUMERIC_OS_NAME "FreeBSD"
#else
  #warning "Unknown operating system"
  #define FEM_NUMERIC_OS_UNKNOWN
  #define FEM_NUMERIC_OS_NAME "Unknown"
#endif

// Architecture Detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  #define FEM_NUMERIC_ARCH_X86_64
  #define FEM_NUMERIC_ARCH_NAME "x86_64"
  #define FEM_NUMERIC_ARCH_64BIT
#elif defined(__i386__) || defined(_M_IX86)
  #define FEM_NUMERIC_ARCH_X86
  #define FEM_NUMERIC_ARCH_NAME "x86"
  #define FEM_NUMERIC_ARCH_32BIT
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define FEM_NUMERIC_ARCH_ARM64
  #define FEM_NUMERIC_ARCH_NAME "ARM64"
  #define FEM_NUMERIC_ARCH_64BIT
#elif defined(__arm__) || defined(_M_ARM)
  #define FEM_NUMERIC_ARCH_ARM
  #define FEM_NUMERIC_ARCH_NAME "ARM"
  #define FEM_NUMERIC_ARCH_32BIT
#elif defined(__powerpc64__)
  #define FEM_NUMERIC_ARCH_PPC64
  #define FEM_NUMERIC_ARCH_NAME "PowerPC64"
  #define FEM_NUMERIC_ARCH_64BIT
#else
  #warning "Unknown architecture"
  #define FEM_NUMERIC_ARCH_UNKNOWN
  #define FEM_NUMERIC_ARCH_NAME "Unknown"
#endif

// Pointer size detection
#if defined(FEM_NUMERIC_ARCH_64BIT)
  #define FEM_NUMERIC_POINTER_SIZE 8
#elif defined(FEM_NUMERIC_ARCH_32BIT)
  #define FEM_NUMERIC_POINTER_SIZE 4
#else
  #define FEM_NUMERIC_POINTER_SIZE sizeof(void*)
#endif

// Endianness Detection
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  #define FEM_NUMERIC_LITTLE_ENDIAN
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
      __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  #define FEM_NUMERIC_BIG_ENDIAN
#elif defined(_MSC_VER) || defined(__LITTLE_ENDIAN__) || \
      defined(__i386__) || defined(__x86_64__) || \
      defined(_M_IX86) || defined(_M_X64) || defined(_M_ARM)
  #define FEM_NUMERIC_LITTLE_ENDIAN
#elif defined(__BIG_ENDIAN__) || defined(__powerpc__) || defined(__ppc__)
  #define FEM_NUMERIC_BIG_ENDIAN
#else
  #warning "Unknown endianness - assuming little endian"
  #define FEM_NUMERIC_LITTLE_ENDIAN
#endif

// SIMD Instruction Set Detection
#if defined(__AVX512F__)
  #define FEM_NUMERIC_HAS_AVX512 1
  #define FEM_NUMERIC_HAS_AVX2 1
  #define FEM_NUMERIC_HAS_AVX 1
  #define FEM_NUMERIC_HAS_SSE4_2 1
  #define FEM_NUMERIC_HAS_SSE4_1 1
  #define FEM_NUMERIC_HAS_SSSE3 1
  #define FEM_NUMERIC_HAS_SSE3 1
  #define FEM_NUMERIC_HAS_SSE2 1
  #define FEM_NUMERIC_HAS_SSE 1
  #define FEM_NUMERIC_SIMD_WIDTH 512
#elif defined(__AVX2__)
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 1
  #define FEM_NUMERIC_HAS_AVX 1
  #define FEM_NUMERIC_HAS_SSE4_2 1
  #define FEM_NUMERIC_HAS_SSE4_1 1
  #define FEM_NUMERIC_HAS_SSSE3 1
  #define FEM_NUMERIC_HAS_SSE3 1
  #define FEM_NUMERIC_HAS_SSE2 1
  #define FEM_NUMERIC_HAS_SSE 1
  #define FEM_NUMERIC_SIMD_WIDTH 256
#elif defined(__AVX__)
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 0
  #define FEM_NUMERIC_HAS_AVX 1
  #define FEM_NUMERIC_HAS_SSE4_2 1
  #define FEM_NUMERIC_HAS_SSE4_1 1
  #define FEM_NUMERIC_HAS_SSSE3 1
  #define FEM_NUMERIC_HAS_SSE3 1
  #define FEM_NUMERIC_HAS_SSE2 1
  #define FEM_NUMERIC_HAS_SSE 1
  #define FEM_NUMERIC_SIMD_WIDTH 256
#elif defined(__SSE4_2__)
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 0
  #define FEM_NUMERIC_HAS_AVX 0
  #define FEM_NUMERIC_HAS_SSE4_2 1
  #define FEM_NUMERIC_HAS_SSE4_1 1
  #define FEM_NUMERIC_HAS_SSSE3 1
  #define FEM_NUMERIC_HAS_SSE3 1
  #define FEM_NUMERIC_HAS_SSE2 1
  #define FEM_NUMERIC_HAS_SSE 1
  #define FEM_NUMERIC_SIMD_WIDTH 128
#elif defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64)
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 0
  #define FEM_NUMERIC_HAS_AVX 0
  #define FEM_NUMERIC_HAS_SSE4_2 0
  #define FEM_NUMERIC_HAS_SSE4_1 0
  #define FEM_NUMERIC_HAS_SSSE3 0
  #define FEM_NUMERIC_HAS_SSE3 0
  #define FEM_NUMERIC_HAS_SSE2 1
  #define FEM_NUMERIC_HAS_SSE 1
  #define FEM_NUMERIC_SIMD_WIDTH 128
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define FEM_NUMERIC_HAS_NEON 1
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 0
  #define FEM_NUMERIC_HAS_AVX 0
  #define FEM_NUMERIC_HAS_SSE 0
  #define FEM_NUMERIC_SIMD_WIDTH 128
#else
  #define FEM_NUMERIC_HAS_AVX512 0
  #define FEM_NUMERIC_HAS_AVX2 0
  #define FEM_NUMERIC_HAS_AVX 0
  #define FEM_NUMERIC_HAS_SSE 0
  #define FEM_NUMERIC_HAS_NEON 0
  #define FEM_NUMERIC_SIMD_WIDTH 0
#endif

// FMA (Fused Multiply-Add) support
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
  #define FEM_NUMERIC_HAS_FMA 1
#else
  #define FEM_NUMERIC_HAS_FMA 0
#endif

// Number of SIMD doubles that fit in a register
#if FEM_NUMERIC_SIMD_WIDTH > 0
  #define FEM_NUMERIC_SIMD_DOUBLES (FEM_NUMERIC_SIMD_WIDTH / 64)
  #define FEM_NUMERIC_SIMD_FLOATS (FEM_NUMERIC_SIMD_WIDTH / 32)
#else
  #define FEM_NUMERIC_SIMD_DOUBLES 1
  #define FEM_NUMERIC_SIMD_FLOATS 1
#endif

// Atomic operations support
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
  #define FEM_NUMERIC_HAS_ATOMIC_REF 1
#else
  #define FEM_NUMERIC_HAS_ATOMIC_REF 0
#endif

// Check for atomic double support
#include <atomic>
#if defined(__cpp_lib_atomic_is_always_lock_free) && \
    ATOMIC_DOUBLE_LOCK_FREE == 2
  #define FEM_NUMERIC_HAS_ATOMIC_DOUBLE 1
#else
  #define FEM_NUMERIC_HAS_ATOMIC_DOUBLE 0
#endif

// Memory model
#ifdef FEM_NUMERIC_OS_WINDOWS
  #include <malloc.h>
  #define FEM_NUMERIC_ALIGNED_ALLOC(alignment, size) \
    _aligned_malloc(size, alignment)
  #define FEM_NUMERIC_ALIGNED_FREE(ptr) _aligned_free(ptr)
#elif defined(FEM_NUMERIC_OS_POSIX)
  #include <stdlib.h>
  #define FEM_NUMERIC_ALIGNED_ALLOC(alignment, size) \
    aligned_alloc(alignment, size)
  #define FEM_NUMERIC_ALIGNED_FREE(ptr) free(ptr)
#else
  #define FEM_NUMERIC_ALIGNED_ALLOC(alignment, size) malloc(size)
  #define FEM_NUMERIC_ALIGNED_FREE(ptr) free(ptr)
#endif

// Page size
#ifdef FEM_NUMERIC_OS_WINDOWS
  #define FEM_NUMERIC_PAGE_SIZE 4096
#elif defined(FEM_NUMERIC_OS_POSIX)
  #include <unistd.h>
  #define FEM_NUMERIC_PAGE_SIZE sysconf(_SC_PAGESIZE)
#else
  #define FEM_NUMERIC_PAGE_SIZE 4096
#endif

// Hardware concurrency detection
#if __cpp_lib_hardware_interference_size >= 201703L
  #include <new>
  #define FEM_NUMERIC_L1_CACHE_LINE_SIZE \
    std::hardware_destructive_interference_size
#else
  #define FEM_NUMERIC_L1_CACHE_LINE_SIZE 64
#endif

// Math library configuration
#ifdef FEM_NUMERIC_OS_WINDOWS
  #define FEM_NUMERIC_USE_MATH_DEFINES
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif

// Fast math configuration
#if defined(__FAST_MATH__) || defined(_FAST_MATH)
  #define FEM_NUMERIC_FAST_MATH 1
#else
  #define FEM_NUMERIC_FAST_MATH 0
#endif

// Debugging support
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
  #define FEM_NUMERIC_DEBUG 1
#else
  #define FEM_NUMERIC_DEBUG 0
#endif

// Runtime CPU feature detection
#if defined(FEM_NUMERIC_ARCH_X86_64) || defined(FEM_NUMERIC_ARCH_X86)
  #define FEM_NUMERIC_HAS_CPUID 1
  #if defined(FEM_NUMERIC_COMPILER_GCC) || defined(FEM_NUMERIC_COMPILER_CLANG)
    #include <cpuid.h>
    #define FEM_NUMERIC_CPUID(level, a, b, c, d) \
      __cpuid_count(level, 0, a, b, c, d)
  #elif defined(FEM_NUMERIC_COMPILER_MSVC)
    #include <intrin.h>
    #define FEM_NUMERIC_CPUID(level, a, b, c, d) \
      { int info[4]; __cpuidex(info, level, 0); \
        a = info[0]; b = info[1]; c = info[2]; d = info[3]; }
  #endif
#else
  #define FEM_NUMERIC_HAS_CPUID 0
#endif

// NUMA support detection
#if defined(FEM_NUMERIC_OS_LINUX)
  #if FEM_NUMERIC_HAS_INCLUDE(<numa.h>)
    #define FEM_NUMERIC_HAS_NUMA 1
  #else
    #define FEM_NUMERIC_HAS_NUMA 0
  #endif
#elif defined(FEM_NUMERIC_OS_WINDOWS)
  #define FEM_NUMERIC_HAS_NUMA 1  // Windows NUMA API
#else
  #define FEM_NUMERIC_HAS_NUMA 0
#endif

// Large page support
#if defined(FEM_NUMERIC_OS_LINUX)
  #define FEM_NUMERIC_HAS_HUGEPAGES 1
#elif defined(FEM_NUMERIC_OS_WINDOWS)
  #define FEM_NUMERIC_HAS_LARGE_PAGES 1
#else
  #define FEM_NUMERIC_HAS_HUGEPAGES 0
  #define FEM_NUMERIC_HAS_LARGE_PAGES 0
#endif

// Performance monitoring
#if defined(__has_builtin) && __has_builtin(__builtin_readcyclecounter)
  #define FEM_NUMERIC_HAS_CYCLE_COUNTER 1
  #define FEM_NUMERIC_READ_CYCLE_COUNTER() __builtin_readcyclecounter()
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  #define FEM_NUMERIC_HAS_CYCLE_COUNTER 1
  #define FEM_NUMERIC_READ_CYCLE_COUNTER() __rdtsc()
#else
  #define FEM_NUMERIC_HAS_CYCLE_COUNTER 0
  #define FEM_NUMERIC_READ_CYCLE_COUNTER() 0
#endif

// Floating-point environment
#include <cfenv>
#if defined(FE_ALL_EXCEPT)
  #define FEM_NUMERIC_HAS_FENV 1
#else
  #define FEM_NUMERIC_HAS_FENV 0
#endif

// Check for std::execution policies (C++17)
#if __cpp_lib_execution >= 201603L
  #define FEM_NUMERIC_HAS_PARALLEL_STL 1
#else
  #define FEM_NUMERIC_HAS_PARALLEL_STL 0
#endif

// Check for std::bit_cast (C++20)
#if __cpp_lib_bit_cast >= 201806L
  #define FEM_NUMERIC_HAS_BIT_CAST 1
#else
  #define FEM_NUMERIC_HAS_BIT_CAST 0
#endif

// Check for std::numbers (C++20)
#if __cpp_lib_math_constants >= 201907L
  #define FEM_NUMERIC_HAS_MATH_CONSTANTS 1
#else
  #define FEM_NUMERIC_HAS_MATH_CONSTANTS 0
#endif

#endif // FEM_NUMERIC_PLATFORM_H
/**
 * @file features.h
 * @brief Feature flags and optional component configuration
 *
 * This header allows compile-time configuration of library features.
 * Users can define macros before including the library to enable/disable
 * specific features and optimizations.
 */

#ifndef FEM_NUMERIC_FEATURES_H
#define FEM_NUMERIC_FEATURES_H

// ============================================================================
// Core Feature Flags
// ============================================================================

// Enable/disable automatic differentiation support
#ifndef FEM_NUMERIC_ENABLE_AUTODIFF
  #define FEM_NUMERIC_ENABLE_AUTODIFF 1
#endif

// Enable/disable expression templates
#ifndef FEM_NUMERIC_ENABLE_EXPRESSION_TEMPLATES
  #define FEM_NUMERIC_ENABLE_EXPRESSION_TEMPLATES 1
#endif

// Enable/disable lazy evaluation
#ifndef FEM_NUMERIC_ENABLE_LAZY_EVALUATION
  #define FEM_NUMERIC_ENABLE_LAZY_EVALUATION 1
#endif

// Enable/disable SIMD optimizations
#ifndef FEM_NUMERIC_ENABLE_SIMD
  #if FEM_NUMERIC_SIMD_WIDTH > 0
    #define FEM_NUMERIC_ENABLE_SIMD 1
  #else
    #define FEM_NUMERIC_ENABLE_SIMD 0
  #endif
#endif

// Enable/disable parallel algorithms
#ifndef FEM_NUMERIC_ENABLE_PARALLEL
  #if FEM_NUMERIC_HAS_OPENMP || FEM_NUMERIC_HAS_PARALLEL_STL
    #define FEM_NUMERIC_ENABLE_PARALLEL 1
  #else
    #define FEM_NUMERIC_ENABLE_PARALLEL 0
  #endif
#endif

// Enable/disable bounds checking
#ifndef FEM_NUMERIC_ENABLE_BOUNDS_CHECKING
  #ifdef FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_ENABLE_BOUNDS_CHECKING 1
  #else
    #define FEM_NUMERIC_ENABLE_BOUNDS_CHECKING 0
  #endif
#endif

// Enable/disable NaN checking in debug mode
#ifndef FEM_NUMERIC_ENABLE_NAN_CHECKING
  #ifdef FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_ENABLE_NAN_CHECKING 1
  #else
    #define FEM_NUMERIC_ENABLE_NAN_CHECKING 0
  #endif
#endif

// ============================================================================
// Matrix and Vector Features
// ============================================================================

// Enable/disable small matrix optimizations
#ifndef FEM_NUMERIC_ENABLE_SMALL_MATRIX_OPT
  #define FEM_NUMERIC_ENABLE_SMALL_MATRIX_OPT 1
#endif

// Enable/disable static-sized containers
#ifndef FEM_NUMERIC_ENABLE_STATIC_CONTAINERS
  #define FEM_NUMERIC_ENABLE_STATIC_CONTAINERS 1
#endif

// Enable/disable block matrix support
#ifndef FEM_NUMERIC_ENABLE_BLOCK_MATRICES
  #define FEM_NUMERIC_ENABLE_BLOCK_MATRICES 1
#endif

// Enable/disable sparse matrix support
#ifndef FEM_NUMERIC_ENABLE_SPARSE_MATRICES
  #define FEM_NUMERIC_ENABLE_SPARSE_MATRICES 1
#endif

// Sparse matrix formats to include
#ifndef FEM_NUMERIC_ENABLE_CSR
  #define FEM_NUMERIC_ENABLE_CSR 1
#endif

#ifndef FEM_NUMERIC_ENABLE_CSC
  #define FEM_NUMERIC_ENABLE_CSC 1
#endif

#ifndef FEM_NUMERIC_ENABLE_COO
  #define FEM_NUMERIC_ENABLE_COO 1
#endif

#ifndef FEM_NUMERIC_ENABLE_BSR
  #define FEM_NUMERIC_ENABLE_BSR 1  // Block Sparse Row
#endif

// ============================================================================
// Solver Features
// ============================================================================

// Direct solvers
#ifndef FEM_NUMERIC_ENABLE_LU_SOLVER
  #define FEM_NUMERIC_ENABLE_LU_SOLVER 1
#endif

#ifndef FEM_NUMERIC_ENABLE_CHOLESKY_SOLVER
  #define FEM_NUMERIC_ENABLE_CHOLESKY_SOLVER 1
#endif

#ifndef FEM_NUMERIC_ENABLE_QR_SOLVER
  #define FEM_NUMERIC_ENABLE_QR_SOLVER 1
#endif

// Iterative solvers
#ifndef FEM_NUMERIC_ENABLE_CG_SOLVER
  #define FEM_NUMERIC_ENABLE_CG_SOLVER 1
#endif

#ifndef FEM_NUMERIC_ENABLE_GMRES_SOLVER
  #define FEM_NUMERIC_ENABLE_GMRES_SOLVER 1
#endif

#ifndef FEM_NUMERIC_ENABLE_BICGSTAB_SOLVER
  #define FEM_NUMERIC_ENABLE_BICGSTAB_SOLVER 1
#endif

// Eigensolvers
#ifndef FEM_NUMERIC_ENABLE_EIGEN_SOLVER
  #define FEM_NUMERIC_ENABLE_EIGEN_SOLVER 1
#endif

// Nonlinear solvers
#ifndef FEM_NUMERIC_ENABLE_NEWTON_SOLVER
  #define FEM_NUMERIC_ENABLE_NEWTON_SOLVER 1
#endif

// ============================================================================
// Assembly Features
// ============================================================================

// Enable/disable atomic assembly operations
#ifndef FEM_NUMERIC_ENABLE_ATOMIC_ASSEMBLY
  #if FEM_NUMERIC_HAS_ATOMIC_DOUBLE
    #define FEM_NUMERIC_ENABLE_ATOMIC_ASSEMBLY 1
  #else
    #define FEM_NUMERIC_ENABLE_ATOMIC_ASSEMBLY 0
  #endif
#endif

// Enable/disable graph coloring for parallel assembly
#ifndef FEM_NUMERIC_ENABLE_GRAPH_COLORING
  #define FEM_NUMERIC_ENABLE_GRAPH_COLORING 1
#endif

// Enable/disable assembly caching
#ifndef FEM_NUMERIC_ENABLE_ASSEMBLY_CACHE
  #define FEM_NUMERIC_ENABLE_ASSEMBLY_CACHE 1
#endif

// ============================================================================
// Automatic Differentiation Features
// ============================================================================

#if FEM_NUMERIC_ENABLE_AUTODIFF

// Maximum number of directional derivatives for forward-mode AD
#ifndef FEM_NUMERIC_MAX_DUAL_DERIVATIVES
  #define FEM_NUMERIC_MAX_DUAL_DERIVATIVES 16
#endif

// Enable/disable reverse-mode AD (tape-based)
#ifndef FEM_NUMERIC_ENABLE_REVERSE_AD
  #define FEM_NUMERIC_ENABLE_REVERSE_AD 1
#endif

// Enable/disable higher-order AD
#ifndef FEM_NUMERIC_ENABLE_HYPERDUAL
  #define FEM_NUMERIC_ENABLE_HYPERDUAL 1
#endif

// Enable/disable mixed-mode AD
#ifndef FEM_NUMERIC_ENABLE_MIXED_AD
  #define FEM_NUMERIC_ENABLE_MIXED_AD 1
#endif

// Tape optimization strategies
#ifndef FEM_NUMERIC_ENABLE_TAPE_CHECKPOINTING
  #define FEM_NUMERIC_ENABLE_TAPE_CHECKPOINTING 1
#endif

#ifndef FEM_NUMERIC_ENABLE_TAPE_COMPRESSION
  #define FEM_NUMERIC_ENABLE_TAPE_COMPRESSION 0
#endif

#endif // FEM_NUMERIC_ENABLE_AUTODIFF

// ============================================================================
// Optimization Features
// ============================================================================

// Enable/disable matrix-free methods
#ifndef FEM_NUMERIC_ENABLE_MATRIX_FREE
  #define FEM_NUMERIC_ENABLE_MATRIX_FREE 1
#endif

// Enable/disable sum factorization
#ifndef FEM_NUMERIC_ENABLE_SUM_FACTORIZATION
  #define FEM_NUMERIC_ENABLE_SUM_FACTORIZATION 1
#endif

// Enable/disable loop unrolling for small sizes
#ifndef FEM_NUMERIC_ENABLE_LOOP_UNROLLING
  #define FEM_NUMERIC_ENABLE_LOOP_UNROLLING 1
#endif

// Maximum size for compile-time loop unrolling
#ifndef FEM_NUMERIC_MAX_UNROLL_SIZE
  #define FEM_NUMERIC_MAX_UNROLL_SIZE 8
#endif

// ============================================================================
// Memory Management Features
// ============================================================================

// Enable/disable custom memory pools
#ifndef FEM_NUMERIC_ENABLE_MEMORY_POOLS
  #define FEM_NUMERIC_ENABLE_MEMORY_POOLS 1
#endif

// Enable/disable aligned memory allocation
#ifndef FEM_NUMERIC_ENABLE_ALIGNED_ALLOC
  #define FEM_NUMERIC_ENABLE_ALIGNED_ALLOC 1
#endif

// Enable/disable NUMA-aware allocation
#ifndef FEM_NUMERIC_ENABLE_NUMA
  #if FEM_NUMERIC_HAS_NUMA
    #define FEM_NUMERIC_ENABLE_NUMA 1
  #else
    #define FEM_NUMERIC_ENABLE_NUMA 0
  #endif
#endif

// Enable/disable huge pages support
#ifndef FEM_NUMERIC_ENABLE_HUGE_PAGES
  #if FEM_NUMERIC_HAS_HUGEPAGES || FEM_NUMERIC_HAS_LARGE_PAGES
    #define FEM_NUMERIC_ENABLE_HUGE_PAGES 1
  #else
    #define FEM_NUMERIC_ENABLE_HUGE_PAGES 0
  #endif
#endif

// ============================================================================
// Input/Output Features
// ============================================================================

// Enable/disable Matrix Market format support
#ifndef FEM_NUMERIC_ENABLE_MATRIX_MARKET
  #define FEM_NUMERIC_ENABLE_MATRIX_MARKET 1
#endif

// Enable/disable HDF5 support (requires external library)
#ifndef FEM_NUMERIC_ENABLE_HDF5
  #define FEM_NUMERIC_ENABLE_HDF5 0
#endif

// Enable/disable NumPy format support
#ifndef FEM_NUMERIC_ENABLE_NUMPY_FORMAT
  #define FEM_NUMERIC_ENABLE_NUMPY_FORMAT 1
#endif

// ============================================================================
// Debugging and Profiling Features
// ============================================================================

// Enable/disable performance profiling
#ifndef FEM_NUMERIC_ENABLE_PROFILING
  #define FEM_NUMERIC_ENABLE_PROFILING 0
#endif

// Enable/disable memory tracking
#ifndef FEM_NUMERIC_ENABLE_MEMORY_TRACKING
  #ifdef FEM_NUMERIC_DEBUG
    #define FEM_NUMERIC_ENABLE_MEMORY_TRACKING 1
  #else
    #define FEM_NUMERIC_ENABLE_MEMORY_TRACKING 0
  #endif
#endif

// Enable/disable operation counting
#ifndef FEM_NUMERIC_ENABLE_OP_COUNTING
  #define FEM_NUMERIC_ENABLE_OP_COUNTING 0
#endif

// Enable/disable detailed logging
#ifndef FEM_NUMERIC_ENABLE_LOGGING
  #define FEM_NUMERIC_ENABLE_LOGGING 0
#endif

// ============================================================================
// Backend Features
// ============================================================================

// Enable/disable BLAS backend
#ifndef FEM_NUMERIC_ENABLE_BLAS
  #define FEM_NUMERIC_ENABLE_BLAS 0
#endif

// Enable/disable MKL backend
#ifndef FEM_NUMERIC_ENABLE_MKL
  #define FEM_NUMERIC_ENABLE_MKL 0
#endif

// Enable/disable CUDA backend
#ifndef FEM_NUMERIC_ENABLE_CUDA
  #define FEM_NUMERIC_ENABLE_CUDA 0
#endif

// Enable/disable OpenCL backend
#ifndef FEM_NUMERIC_ENABLE_OPENCL
  #define FEM_NUMERIC_ENABLE_OPENCL 0
#endif

// ============================================================================
// Experimental Features
// ============================================================================

// Enable/disable experimental features (may be unstable)
#ifndef FEM_NUMERIC_ENABLE_EXPERIMENTAL
  #define FEM_NUMERIC_ENABLE_EXPERIMENTAL 0
#endif

#if FEM_NUMERIC_ENABLE_EXPERIMENTAL

// Experimental tensor network support
#ifndef FEM_NUMERIC_ENABLE_TENSOR_NETWORKS
  #define FEM_NUMERIC_ENABLE_TENSOR_NETWORKS 0
#endif

// Experimental symbolic computation
#ifndef FEM_NUMERIC_ENABLE_SYMBOLIC
  #define FEM_NUMERIC_ENABLE_SYMBOLIC 0
#endif

// Experimental GPU tensor cores
#ifndef FEM_NUMERIC_ENABLE_TENSOR_CORES
  #define FEM_NUMERIC_ENABLE_TENSOR_CORES 0
#endif

#endif // FEM_NUMERIC_ENABLE_EXPERIMENTAL

// ============================================================================
// Feature Validation
// ============================================================================

// Validate feature combinations
#if FEM_NUMERIC_ENABLE_AUTODIFF && !FEM_NUMERIC_ENABLE_EXPRESSION_TEMPLATES
  #warning "Automatic differentiation works best with expression templates enabled"
#endif

#if FEM_NUMERIC_ENABLE_MATRIX_FREE && !FEM_NUMERIC_ENABLE_SUM_FACTORIZATION
  #warning "Matrix-free methods require sum factorization for optimal performance"
#endif

#if FEM_NUMERIC_ENABLE_PARALLEL && !FEM_NUMERIC_ENABLE_ATOMIC_ASSEMBLY
  #warning "Parallel assembly without atomics may require additional synchronization"
#endif

// ============================================================================
// Feature Summary Macros
// ============================================================================

// Check if any sparse format is enabled
#define FEM_NUMERIC_HAS_ANY_SPARSE \
  (FEM_NUMERIC_ENABLE_SPARSE_MATRICES && \
   (FEM_NUMERIC_ENABLE_CSR || FEM_NUMERIC_ENABLE_CSC || \
    FEM_NUMERIC_ENABLE_COO || FEM_NUMERIC_ENABLE_BSR))

// Check if any direct solver is enabled
#define FEM_NUMERIC_HAS_ANY_DIRECT_SOLVER \
  (FEM_NUMERIC_ENABLE_LU_SOLVER || \
   FEM_NUMERIC_ENABLE_CHOLESKY_SOLVER || \
   FEM_NUMERIC_ENABLE_QR_SOLVER)

// Check if any iterative solver is enabled
#define FEM_NUMERIC_HAS_ANY_ITERATIVE_SOLVER \
  (FEM_NUMERIC_ENABLE_CG_SOLVER || \
   FEM_NUMERIC_ENABLE_GMRES_SOLVER || \
   FEM_NUMERIC_ENABLE_BICGSTAB_SOLVER)

// Check if any external backend is enabled
#define FEM_NUMERIC_HAS_ANY_BACKEND \
  (FEM_NUMERIC_ENABLE_BLAS || \
   FEM_NUMERIC_ENABLE_MKL || \
   FEM_NUMERIC_ENABLE_CUDA || \
   FEM_NUMERIC_ENABLE_OPENCL)

// ============================================================================
// Default Feature Selection
// ============================================================================

// Select default sparse format if none specified
#if FEM_NUMERIC_ENABLE_SPARSE_MATRICES && !FEM_NUMERIC_HAS_ANY_SPARSE
  #undef FEM_NUMERIC_ENABLE_CSR
  #define FEM_NUMERIC_ENABLE_CSR 1
#endif

// Select default solver if none specified
#if !FEM_NUMERIC_HAS_ANY_DIRECT_SOLVER && !FEM_NUMERIC_HAS_ANY_ITERATIVE_SOLVER
  #undef FEM_NUMERIC_ENABLE_LU_SOLVER
  #define FEM_NUMERIC_ENABLE_LU_SOLVER 1
  #undef FEM_NUMERIC_ENABLE_CG_SOLVER
  #define FEM_NUMERIC_ENABLE_CG_SOLVER 1
#endif

#endif // FEM_NUMERIC_FEATURES_H
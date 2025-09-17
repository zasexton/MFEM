# linear_algebra/ AGENT.md

## Purpose
Offer BLAS-like dense and sparse linear algebra routines (vector-vector, matrix-vector, matrix-matrix operations) built on top of the element-wise kernels from `operations/`. This module defines the mid-level abstractions that higher-level solvers and decompositions depend on.

## Responsibilities
- Implement Level 1/2/3 BLAS equivalents for dense and structured storage.
- Provide sparse algebra utilities (CSR/CSC operations, SpMV, SpMM) as extensible interfaces.
- Define interfaces that allow dispatch to optimized backends (BLAS, MKL, CUDA) while preserving a pure C++ fallback.
- Supply helper routines for applying linear operators consistently across storage types (dense, block, matrix-free wrappers).

## Non-Goals
- Factorization algorithms (delegated to `decompositions/`).
- Solver orchestration (delegated to `solvers/`).
- Element-wise kernels (delegated to `operations/`).

## Dependencies
- Depends on `operations/` for elemental kernels.
- Works with storage abstractions from `base/` and `traits/` for layout compatibility.
- Collaborates with `backends/` to select optimized implementations at compile/runtime.

## Planned Layout
```text
linear_algebra/
├── blas_level1.h      # dot, axpy, scal, etc.
├── blas_level2.h      # gemv, ger, syr, etc.
├── blas_level3.h      # gemm, syrk, trsm, etc.
├── sparse_ops.h       # SpMV, SpMM, sparse conversions
└── norms.h            # Vector/matrix norm implementations
```

## Notes
- Provide concept-based overloads to ensure container compatibility (dimensions, storage ordering).
- Backends specialize these interfaces; fallback implementations reside here.

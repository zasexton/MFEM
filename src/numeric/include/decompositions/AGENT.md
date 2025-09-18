# decompositions/ AGENT.md

## Purpose
Provide matrix factorization algorithms (LU, QR, SVD, eigenvalue, Cholesky, etc.) implemented on top of the linear algebra interfaces. These routines are building blocks for solvers and advanced analyses.

## Responsibilities
- Implement dense and sparse factorizations using `linear_algebra/` primitives.
- Expose factorization objects with reusable `apply/solve` interfaces (e.g., `LUFact::solve(rhs)`).
- Support autodiff-aware factorizations where feasible (propagating derivatives through solves).
- Allow backend-specific overrides while offering robust reference implementations.

## Non-Goals
- Reimplement BLAS operations (those come from `linear_algebra/`).
- High-level solver orchestration (belongs to `solvers/`).
- Direct manipulation of storage; rely on container abstractions.

## Dependencies
- Consumes `operations/` and `linear_algebra/`.
- Works with `backends/` to delegate heavy lifting when an optimized routine exists.
- Interacts with `traits/` to ensure factorization requirements (e.g., symmetric positive-definite) are met.

## Planned Layout
```text
decompositions/
├── lu.h          # LU with/without pivoting
├── qr.h          # QR factorizations (Householder, Givens)
├── svd.h         # Singular value decomposition
├── eigen.h       # Eigenvalue algorithms
└── cholesky.h    # Cholesky and LDL^T variants
```

## Notes
- Factorization results should be compatible with autodiff element types (seeded via `operations/`).
- Consider providing fallback iterative refinement routines to improve numerical robustness.

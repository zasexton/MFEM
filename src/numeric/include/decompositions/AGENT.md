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

## Planned Work

The following additions round out practical dense decompositions. Each item notes why it’s useful, when to use it, and expected performance. These are algorithmic building blocks (not preconditioners or high-level solvers).

### High Priority

- LDLT (symmetric indefinite; Bunch–Kaufman/rook pivoting)
  - Useful: Factors general symmetric/Hermitian indefinite matrices as P^T A P = L D L^H with 1×1/2×2 pivots, preserving symmetry and improving stability over naive Gaussian elimination.
  - When: Indefinite systems, KKT-like blocks, constrained formulations where A is symmetric but not SPD; needed to robustly handle negative/zero pivots.
  - Performance: O(n^3) flops with Level-3 BLAS updates; comparable to Cholesky up to constants. Blocked BK algorithms map well to BLAS3. Backend path via LAPACK `xSYTRF/xSYTRS` when enabled.

- QR with column pivoting (CPQR) and rank‑revealing QR (RRQR)
  - Useful: Robust least‑squares and rank detection; chooses informative column ordering and exposes numerical rank via R’s diagonal decay.
  - When: Rectangular A with possible rank deficiency/collinearity; model reduction and column subset/skeletonization seeds.
  - Performance: O(2mn^2 − 2/3 n^3) for m≥n; pivoting adds search/communication overhead (typically 15–40% slower than plain QR). Blocked implementations amortize costs; backend path via LAPACK `xGEQP3` when available.

- Cholesky variants: Pivoted Cholesky (low‑rank) and rank‑1 update/downdate
  - Useful: Pivoted Cholesky builds low‑rank approximations of SPD matrices without forming full factors; rank‑1 update/downdate modifies an existing Cholesky factor efficiently.
  - When: Kernel/GP covariance approximations, Hessian compressions, hierarchical/low‑rank settings; online or sliding‑window problems where A changes by u u^T.
  - Performance: Pivoted Cholesky O(n^2 k) to reach rank k (much cheaper than O(n^3)); stable downdates via hyperbolic rotations. Rank‑1 update/downdate is O(n^2) per update.

- LQ and RQ factorizations (wrappers)
  - Useful: Duals of QR for wide matrices and row‑space orthogonalization; convenient when applying orthogonal/unitary transforms from the right.
  - When: m ≤ n (LQ) or for operations favoring row transformations; building orthonormal bases for row spaces.
  - Performance: Implemented via QR of A^T with the same asymptotic cost as QR; backend via `xGELQF`/`xGERQF` when present.

### Eigen/Schur Building Blocks

- Hessenberg reduction (general, non‑symmetric)
  - Useful: Reduces A to upper Hessenberg H via unitary similarity, the standard preprocessing for QR iterations on general matrices.
  - When: Any dense non‑symmetric eigenvalue computation; prerequisite for real/complex Schur.
  - Performance: O(2/3 n^3). Blocked Householder reduces to BLAS3; backend via LAPACK `xGEHRD`.

- Real/Complex Schur decomposition
  - Useful: A = Q T Q^H with T quasi‑triangular (real Schur) or upper‑triangular (complex Schur). Numerically stable way to obtain eigenvalues/vectors and to build invariant subspaces.
  - When: General eigenproblems, reordering for deflation, matrix functions, stable splitting of spectral subspaces.
  - Performance: QR iteration on Hessenberg, O(n^3) with small constants; blocked multi‑shift implementations are backend‑friendly. Backend via LAPACK `xHSEQR`/`xTREXC`.

- Polar decomposition
  - Useful: A = Q H with Q unitary and H Hermitian positive‑semidefinite; nearest unitary to A in Frobenius norm. Valuable in continuum mechanics and matrix functions.
  - When: Extracting rotation/stretch (e.g., deformation gradients), orthogonalization with positivity, conditioning improvements.
  - Performance: Via SVD costs O(n^3); Newton/Denman–Beavers iterations converge in O(log κ(A)) steps with each step dominated by matrix multiplies/solves (BLAS3‑friendly).

### Nice‑To‑Have

- TSQR / CAQR (tall‑skinny and communication‑avoiding QR)
  - Useful: Efficient QR for m ≫ n with low memory and reduced communication; amenable to blocking/streaming and distributed settings.
  - When: Very tall matrices, out‑of‑core or parallel pipelines, batched accumulation.
  - Performance: O(m n^2) total flops with excellent cache/communication behavior; combines local QR on blocks with small tree reductions of R‑factors.

- Randomized SVD (low‑rank)
  - Useful: Fast approximate SVD for low‑numerical‑rank matrices using randomized range finding with optional power iterations.
  - When: Large dense or implicit operators where only a rank‑k approximation is needed.
  - Performance: O(m n k) + oversampling; few passes over A; BLAS3‑heavy and parallel‑friendly.

- Interpolative Decomposition (ID) and CUR
  - Useful: Low‑rank factorizations using a subset of columns/rows (skeletonization) for interpretability and sparsity preservation.
  - When: Data compression, column/row subset selection, fast approximate matvecs.
  - Performance: Typically built on CPQR/RRQR in O(m n k) time; small core solves of size k.

- Complete Orthogonal Decomposition (COD)
  - Useful: Robust factorization for rank‑deficient least‑squares, exposing orthogonal bases for range/null spaces beyond CPQR alone.
  - When: Diagnostics and numerically stable LS with explicit rank handling and condition estimators.
  - Performance: Similar to CPQR plus additional triangular/orthogonal refinements; O(m n^2) for m ≥ n.

- QR/LS updates (Givens/Householder rank‑1 update/downdate)
  - Useful: Incremental maintenance of QR factors under appended/removed rows/columns or rank‑1 modifications, avoiding refactorization.
  - When: Streaming/regression workflows and iterative refinement where A evolves slightly.
  - Performance: O(n^2) per rank‑1 update or row/column add/remove; numerically stable using Givens sequences.

## Notes
- Factorization results should be compatible with autodiff element types (seeded via `operations/`).
- Consider providing fallback iterative refinement routines to improve numerical robustness.

## Performance Notes (blocked + backends)
- Blocked variants now exist using Level-3 BLAS updates:
  - `cholesky_factor_blocked(A, uplo, block)` (TRSM + SYRK updates)
  - `lu_factor_blocked(A, piv, block)` (right-looking panel + GEMM update)
  - `qr_factor_blocked(A, tau, block)` (WY update with V/T via GEMM/TRMM)
- Optional backend dispatch to LAPACK/MKL is available when configured:
  - Enable with CMake option `-DFEM_NUMERIC_ENABLE_LAPACK=ON` (requires `find_package(LAPACK)`)
  - When enabled and linked, the blocked entry points will attempt backend calls first.

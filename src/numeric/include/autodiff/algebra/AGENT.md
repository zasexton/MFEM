# autodiff/algebra/ AGENT.md

## Purpose
Provide AD-aware bridges and specializations for linear algebra and decompositions so that AD element types work seamlessly with `linear_algebra/` and `decompositions/`.

## Responsibilities
- Expose helpers to treat AD types as scalar-like in BLAS-style routines.
- Offer optional derivative extraction hooks for factorizations/solves.

## Files (planned)
- `dual_linear_algebra.h` — traits/adapters for BLAS-like ops.
- `dual_decompositions.h` — helpers for AD-aware factorization usage.

## Collaborations
- `linear_algebra/`, `decompositions/` for algorithms.
- `autodiff/types` and `autodiff/operations` for semantics.

## Example APIs
```cpp
// Use AD scalars in matrix operations
Matrix<Dual<double,2>> A, x, y;
y = gemv(A, x);      // delegates to linear_algebra with AD-aware traits

// Solve with derivatives preserved
auto lu = lu_factor(A);
auto sol = lu.solve(b);   // Dual derivatives propagate through solve
```

## Notes
- Keep adapters minimal: prefer trait-based customization over branching.


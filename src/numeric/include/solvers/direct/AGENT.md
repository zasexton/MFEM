# solvers/direct/ AGENT.md

## Purpose
Wrap decomposition-based direct solves (LU, Cholesky, QR) with a consistent solver API.

## Responsibilities
- Provide `solve()` wrappers that consume `decompositions/` results.
- Handle basic options (pivoting, SPD assumptions) and report diagnostics.

## Files (planned)
- `lu_solver.h`, `cholesky_solver.h`, `qr_solver.h`.

## Collaborations
- `decompositions/` for factorization implementations.
- `linear_algebra/` for BLAS-level operations.

## Example APIs
```cpp
auto lu = lu_factor(A);
Vector<T> x = lu.solve(b);

auto chol = cholesky(A_spd);
x = chol.solve(b);
```

## Notes
- Keep templated on scalar to support AD; avoid copying factors unnecessarily.


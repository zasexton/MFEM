# solvers/eigen/ AGENT.md

## Purpose
Provide basic eigenvalue solvers suitable for numeric use (power method, Arnoldi/Lanczos stubs) with clean operator interfaces.

## Responsibilities
- Implement simple iterative eigenvalue routines that accept operator apply callbacks.
- Expose options for number of eigenpairs, shift, and tolerance.

## Files (planned)
- `power_method.h`, `arnoldi.h`.

## Collaborations
- `linear_algebra/` for orthonormalization and vector ops.
- `matrix_free/` for operator application.

## Example APIs
```cpp
auto [lambda, v] = power_method(A, /*tol=*/1e-8, /*maxit=*/1000);

Arnoldi arn;
arn.set_subspace_dim(40);
auto result = arn.compute(A, /*k=*/5);
```

## Notes
- Keep algorithms minimal and robust; advanced methods can be added later or delegated to backends.


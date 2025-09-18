# solvers/preconditioners/ AGENT.md

## Purpose
Provide preconditioner interfaces and reference implementations: Jacobi, ILU(0), block field-split, and matrix-free variants.

## Responsibilities
- Define a minimal `apply(z, r)` interface usable by iterative solvers.
- Implement simple, robust preconditioners and adapters for user-defined ones.

## Files (planned)
- `jacobi.h`, `ilu.h`, `block_preconditioner.h`, `matrix_free_preconditioner.h`.

## Collaborations
- `solvers/iterative` consumes these via composition.
- `block/` for field-split logic, `matrix_free/` for operator-based preconditioning.

## Example APIs
```cpp
auto M = Jacobi(A);              // diagonal
auto iters = cg.solve(A, b, x, M);

auto B = FieldSplit{
  {"u", Jacobi(A_uu)},
  {"p", ILU(A_pp)}
};
iters = gmres.solve(K, rhs, x, B);
```

## Notes
- Keep construction cheap; reuse analyzed structures between solves when possible.


# solvers/iterative/ AGENT.md

## Purpose
Provide Krylov and related iterative linear solvers (CG, GMRES, BiCGSTAB, MINRES) with a uniform interface and pluggable preconditioners.

## Responsibilities
- Implement solver loops with configurable stopping criteria and logging.
- Accept linear operators (assembled or matrix-free) and preconditioners.

## Files (planned)
- `cg.h`, `gmres.h`, `bicgstab.h`, `minres.h`.

## Collaborations
- `preconditioners/` for Jacobi/ILU/block/matrix-free preconditioners.
- `matrix_free/` for operator application without assembly.

## Example APIs
```cpp
CG solver;
solver.set_tolerance(1e-8);
solver.set_max_iterations(500);
solver.set_preconditioner(Jacobi(A));

auto iters = solver.solve(A, b, x);
```

## Notes
- Keep interfaces templated on operator and vector types; prefer concept constraints over inheritance.


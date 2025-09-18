# solvers/newton/ AGENT.md

## Purpose
Provide nonlinear solver scaffolding (Newton, quasi-Newton) with line search and trust region strategies, compatible with AD-based Jacobians.

## Responsibilities
- Implement Newton iteration with callback hooks for residual/Jacobian evaluation.
- Provide line search (Armijo/Wolfe) and trust region policies.

## Files (planned)
- `newton_solver.h`, `line_search.h`, `trust_region.h`.

## Collaborations
- `autodiff/` to form Jacobians automatically when provided residual functors over AD vectors.
- `solvers/iterative` or direct solvers for inner linear solves.

## Example APIs
```cpp
NewtonSolver solver;
solver.set_line_search(Backtracking{0.5, 1e-4});
solver.set_tolerance(1e-8);
solver.set_max_iterations(50);

auto residual = [&](const Vector<Var<double>>& u){ return R(u); };
solver.set_residual(residual);   // AD Jacobian computed internally
auto status = solver.solve(u0);
```

## Notes
- Keep state inspection/simple logging; orchestration belongs to higher layers if needed.


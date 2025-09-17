# optimization/ AGENT.md

## Purpose
Provide optimization algorithms (gradient-based and derivative-free) for use in design optimization, inverse problems, and solver configuration. Builds on autodiff, linear algebra, and solver modules.

## Responsibilities
- Implement gradient-based optimizers (gradient descent variants, L-BFGS, conjugate gradient methods for optimization).
- Provide nonlinear least-squares solvers (Levenberg–Marquardt) with Jacobian/Hessian handling.
- Offer derivative-free methods (optional) for robustness.
- Integrate line search and trust-region strategies with customizable policies.

## Non-Goals
- Implement PDE-constrained optimization frameworks (application layer).
- Duplicate solver modules; instead call `solvers/` for inner solves when necessary.

## Dependencies
- Heavily relies on `autodiff/` to compute gradients/Hessians.
- Uses `linear_algebra/`, `solvers/`, and `operations/` for underlying computations.
- Works with `support/` for error handling and `diagnostics/` for convergence reporting.

## Planned Layout
```text
optimization/
├── gradient_descent.h
├── momentum_methods.h
├── lbfgs.h
├── conjugate_gradient.h
├── line_search.h
├── trust_region.h
└── levenberg_marquardt.h
```

## Notes
- Keep APIs flexible: accept functors returning value/gradient pairs.
- Provide hooks for user-defined termination and monitoring.

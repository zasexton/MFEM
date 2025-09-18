# solvers/ AGENT.md

## Purpose
Provide linear and nonlinear solver interfaces (direct, iterative, eigenvalue, Newton-type) built on top of `linear_algebra/` and `decompositions/`. Solvers expose consistent APIs for the rest of the numeric library while keeping backend choices pluggable.

## Responsibilities
- Implement wrappers for direct solvers (LU/Cholesky-based) and iterative solvers (CG, GMRES, MINRES, etc.).
- Provide eigenvalue solvers leveraging factorization modules.
- Offer nonlinear solvers (Newton, quasi-Newton) that integrate with autodiff for Jacobian/Hessian computation.
- Manage preconditioner interfaces, delegating implementations to dedicated modules when complex (block, matrix-free).

## Non-Goals
- Implement elementary kernels (use `operations/` and `linear_algebra/`).
- Provide application-specific convergence criteria (keep generic, allow user callbacks).

## Dependencies
- Relies on `linear_algebra/` and `decompositions/` for core computations.
- Works with `parallel/`/`matrix_free/` for applying operators efficiently.
- Uses `support/` for error handling and `diagnostics/` for optional profiling.

## Planned Layout
```text
solvers/
├── direct/
│   ├── lu_solver.h
│   ├── cholesky_solver.h
│   └── qr_solver.h
├── iterative/
│   ├── cg.h
│   ├── gmres.h
│   ├── bicgstab.h
│   └── minres.h
├── eigen/
│   ├── power_method.h
│   └── arnoldi.h
├── newton/
│   ├── newton_solver.h
│   ├── line_search.h
│   └── trust_region.h
└── preconditioners/
    ├── jacobi.h
    ├── ilu.h
    ├── block_preconditioner.h
    └── matrix_free_preconditioner.h
```

## Notes
- Provide flexible stopping criteria and logging hooks.
- Ensure autodiff compatibility by templating on scalar type.

# matrix_free/ AGENT.md

## Purpose
Support matrix-free operators and sum-factorization techniques that avoid assembling global matrices while maintaining compatibility with the rest of the numeric stack. These operators rely on `parallel/` for parallel traversal and on `operations/`/`linear_algebra/` for local computations.

## Responsibilities
- Define interfaces for matrix-free operators (`MatrixFreeOperator`, `DiagonalOperator`, etc.).
- Implement sum-factorization kernels for tensor-product finite elements.
- Provide infrastructure for matrix-free preconditioners compatible with `solvers/`.
- Integrate with `parallel/` to distribute element operations efficiently.

## Non-Goals
- Implement generic parallel infrastructure (delegated to `parallel/`).
- Duplicate block or sparse matrix logic; matrix-free operators should interoperate but remain distinct.

## Dependencies
- Relies on `parallel/` for execution.
- Uses kernels from `operations/` and `linear_algebra/` for local computations.
- Works with `traits/` to ensure element kernels match container types.

## Planned Layout
```text
matrix_free/
├── operator.h                 # Base class for matrix-free operators
├── sum_factorization.h        # Sum-factorization kernels
├── apply_kernels.h            # Element-wise application helpers
├── matrix_free_preconditioner.h # Preconditioner interfaces
└── transfer_ops.h             # Restriction/prolongation without assembled matrices
```

## Notes
- Keep operators composable: allow chaining with expression templates and solvers.
- Ensure autodiff compatibility (propagating derivatives through matrix-free evaluations).

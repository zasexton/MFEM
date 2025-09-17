# autodiff/ AGENT.md

## Purpose
Provide automatic differentiation (AD) types and infrastructure—forward-mode (dual), reverse-mode (tape-based), and higher-order variants—that can be composed with all numeric containers (vectors, matrices, tensors, block structures). The module ensures consistent derivative propagation across the entire numeric library without external dependencies.

## Responsibilities
- Define AD numeric types (`Dual`, `HyperDual`, `Var`, etc.) with customizable derivative dimension.
- Implement arithmetic and math operation overloads compatible with `operations/` kernels.
- Supply tape/adjoin infrastructure for reverse-mode AD with utilities to record and replay computational graphs.
- Provide adapter traits so other modules (linear algebra, decompositions, solvers) can detect AD types and adjust algorithms accordingly.
- Expose utilities for seeding derivatives, extracting gradients/Jacobians/Hessians, and interop with block/matrix-free structures.

## Non-Goals
- Implement linear algebra or solver algorithms; they consume these AD types.
- Manage memory outside of AD-specific needs (allocator interfaces remain in `allocators/`).
- Provide automatic differentiation for external libraries; focus on in-house containers first with extension points for backends.

## Dependencies
- Uses core traits (`include/traits/`) for concept checks.
- Relies on `base/dual_base.h` for CRTP scaffolding and expression integration.
- Interacts with `operations/` to ensure AD-specific overloads delegate to generic kernels.

## Planned Layout
```text
autodiff/
├── types/
│   ├── dual.h            # Forward mode dual numbers
│   ├── hyperdual.h       # Second-order forward mode
│   ├── var.h             # Reverse-mode variable wrapper
│   ├── complex_dual.h    # Complex-valued dual numbers
│   └── mixed_dual.h      # Hybrid AD strategies
├── operations/
│   ├── dual_ops.h        # Arithmetic overloads for Dual
│   ├── hyperdual_ops.h   # HyperDual operations
│   └── reverse_ops.h     # Reverse-mode operator bindings
├── tape/
│   ├── tape.h            # Reverse-mode tape implementation
│   ├── tape_storage.h    # Storage strategies (arena, pool)
│   └── tape_primitives.h # Node definitions, adjoint operations
├── algebra/
│   ├── dual_linear_algebra.h   # Specializations for linear algebra
│   └── dual_decompositions.h   # AD-aware factorizations helpers
└── utilities/
    ├── seeding.h         # Seed gradients/hessians
    ├── extraction.h      # Extract gradients/Jacobians/Hessians
    └── traits.h          # Detection utilities for AD types
```

## Notes
- Ensure forward and reverse modes can coexist (e.g., nested AD for higher order derivatives).
- Provide compile-time configuration for derivative dimension to avoid dynamic overhead where possible.

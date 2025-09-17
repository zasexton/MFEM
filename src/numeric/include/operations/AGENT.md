# operations/ AGENT.md

## Purpose
Implement element-wise mathematical kernels and reductions that operate on generic containers (vectors, matrices, tensors) without assuming a specific storage layout. These routines are the building blocks used by higher-level linear algebra and decomposition modules.

## Responsibilities
- Provide templated arithmetic/transcendental kernels operating on scalar or SIMD-friendly types (works with autodiff types as long as the operations are defined).
- Implement reductions (sum, norm, min/max) that work with expressions and concrete storage.
- Expose broadcast-aware helpers for element-wise operations across mixed shapes (leveraging broadcasting rules defined in `base/`).
- Maintain backend-neutral interfaces so that linear algebra modules can swap implementations if acceleration libraries are hooked in.

## Non-Goals
- Dense and sparse BLAS routines (handled in `linear_algebra/`).
- Factorizations like LU/QR/SVD (handled in `decompositions/`).
- Expression template orchestration (`expressions/` wraps these kernels, not the other way around).

## Dependencies
- Uses base expression/view infrastructure for iteration.
- Relies on traits/concepts to constrain template instantiations.
- Coordinates with `backends/` by providing hooks where backends can override kernel implementations (e.g., via `dispatch_operation(op, backend)`).

## Planned Layout
```text
operations/
├── arithmetic.h        # add/sub/mul/div and fused variants
├── transcendental.h    # sin/cos/exp/log etc.
├── reductions.h        # sum, product, min, max, norms
├── tensor_contraction.h# generic tensor contractions (small order)
└── broadcast_ops.h     # broadcast-aware element ops
```

## Notes
- Provide customization points (tag invoke or traits) so backends can specialize kernels.
- Ensure functions accept expression operands and delegate evaluation to `expressions/` when needed.

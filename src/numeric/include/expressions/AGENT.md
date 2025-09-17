# expressions/ AGENT.md

## Purpose
Provide user-facing expression templates and composition helpers that build on the generic CRTP infrastructure defined in `include/base/`. This layer packages reusable expression nodes (element-wise ops, fused kernels, evaluation helpers) without redefining the low-level mechanics already implemented in `numeric::base`.

## Responsibilities
- Offer ready-to-use expression classes (`BinaryExpression`, `UnaryExpression`, fused operations) that assemble the base CRTP mixins.
- Collect frequently used expression combinators (e.g., `map`, `zip`, `where`) so applications don’t assemble them manually.
- Maintain integration points with evaluation backends (dense, sparse, matrix-free) via clean interfaces.
- Provide concise documentation/examples demonstrating how end users compose expressions across vectors, matrices, and tensors.

## Non-Goals
- Reimplement CRTP scaffolding, iterator/view abstractions, or SFINAE guards—that code remains in `include/base/`.
- Define mathematical kernels (those belong to `operations/`, `linear_algebra/`, etc.). Expressions orchestrate kernels but do not implement them.

## Dependencies
- Depends heavily on `numeric::base` (expression_base, container_base, slice_base, view_base, etc.).
- Uses type traits from `include/traits/` for concept checking.
- Consumes kernels from `operations/` and `linear_algebra/`, and is backend-agnostic through those modules.

## Planned Layout
```text
expressions/
├── expression.h          # Central expression facade
├── binary_ops.h          # +, -, *, /, etc. for containers
├── unary_ops.h           # abs, exp, etc. wrappers
├── fused_ops.h           # Combine multiple ops to minimize temporaries
├── evaluation.h          # Evaluate expressions into concrete storage
└── lazy_assign.h         # Expression assignment helpers
```

## Notes
- Keep templates header-only with explicit `requires` clauses for clarity.
- Document examples in module README to help users bridge base infrastructure to practical use cases.

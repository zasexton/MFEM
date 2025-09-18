# math/ AGENT.md

## Purpose
Collect scalar and small-vector math utilities that complement the container-level kernels. This includes specialized functions (e.g., safe comparisons, polynomial evaluations, special functions) that are reused across `operations/`, `linear_algebra/`, and higher-level modules.

## Responsibilities
- Provide numerically robust helper routines (e.g., `safe_hypot`, `almost_equal`).
- Implement polynomial evaluation helpers (Horner schemes) reused in polynomial/quadrature modules.
- Offer small fixed-size math utilities (e.g., 2x2 determinant) that shouldn’t live in heavyweight modules.
- Expose constexpr-friendly functions where possible for compile-time evaluation.

## Non-Goals
- Container operations (handled elsewhere).
- Timer/profiling (moved to `diagnostics/`).
- Error handling (moved to `support/`).

## Planned Layout
```text
math/
├── math_functions.h    # General-purpose helpers
├── comparison.h        # ULP/tolerance comparisons
├── polynomial_utils.h  # Horner, Chebyshev, etc.
└── special_functions.h # (Optional) FEM-related special functions
```

## Notes
- Keep functions templated to support autodiff and custom numeric types.
- Document numerical properties (stability, range) for each helper.

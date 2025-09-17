# polynomial/ AGENT.md

## Purpose
Provide polynomial and quadrature utilities required for FEM shape functions and integration. Includes orthogonal polynomials, interpolation, and quadrature rules that integrate with the rest of the numeric stack without depending on FEM-specific mesh data.

## Responsibilities
- Implement Legendre, Chebyshev, and other orthogonal polynomials with evaluation and derivative routines.
- Offer interpolation helpers (Lagrange basis evaluation, barycentric weights).
- Provide quadrature rules (Gauss, Gauss-Lobatto, etc.) with configurable precision.
- Expose utilities that interact cleanly with AD types and tensors.

## Non-Goals
- Mesh-specific logic (handled by FEM layers).
- Numerical integration over arbitrary geometries—focus on polynomial-based rules.

## Dependencies
- Uses scalar helpers from `math/` for numerical robustness.
- Coordinates with `expressions/` for efficient evaluation patterns.
- May cooperate with `matrix_free/` (sum-factorization uses polynomial evaluations).

## Planned Layout
```text
polynomial/
├── quadrature.h          # Quadrature rule generation
├── legendre.h            # Legendre polynomials & derivatives
├── chebyshev.h           # Chebyshev polynomials
├── interpolation.h       # Interpolation / basis evaluations
└── polynomial_utils.h    # Shared helpers (factored from math/)
```

## Notes
- Ensure functions are constexpr-friendly for compile-time rule generation.
- Provide caching options for expensive polynomial evaluations.

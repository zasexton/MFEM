# constrained/ AGENT.md

## Purpose
Handle constrained systems (Dirichlet, periodic, multi-point constraints) and Schur complement formulations required for FEM problems. Provides tools to build and apply constraint operators consistently across numeric modules.

## Responsibilities
- Implement `ConstraintHandler` for managing boundary conditions and constraints.
- Provide elimination utilities (modify matrices/vectors to enforce constraints).
- Offer Lagrange multiplier system builders and Schur complement helpers.
- Support penalty methods and mortar-like coupling strategies.

## Non-Goals
- Implement solvers/preconditioners (delegated to `solvers/`).
- Duplicate block utilities (defer to `block/`).
- Handle time-dependent constraint logic (applications layer concern).

## Dependencies
- Works with `core/` containers, `sparse/` builders, and `block/` utilities.
- Relies on `parallel/` for constraint assembly in concurrent environments.
- Coordinates with `solvers/` for solving augmented systems.

## Planned Layout
```text
constrained/
├── constraint_handler.h      # Main interface
├── elimination.h             # Dirichlet elimination algorithms
├── lagrange_system.h         # Lagrange multiplier system construction
├── penalty_method.h          # Penalty-based enforcement
└── schur_complement.h        # Schur complement utilities
```

## Notes
- Keep APIs autodiff-aware to allow sensitivity analysis.
- Provide robust error diagnostics when constraints conflict.

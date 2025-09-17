# block/ AGENT.md

## Purpose
Support block-structured systems common in multiphysics FEM, providing assembly utilities, block operations, and interfaces compatible with both assembled and matrix-free workflows.

## Responsibilities
- Implement block extraction/insertion helpers for block vectors and matrices.
- Provide block assembly routines that integrate with `parallel/` and `sparse/` builders.
- Offer block preconditioner interfaces (Schur complement splitting, field-split preconditioners).
- Supply utilities for block permutations, adaptive block sizes, and inter-block communication.

## Non-Goals
- Define base container classes (those live in `core/`).
- Implement solver algorithms beyond simple block operations (advanced solvers reside in `solvers/`).
- Duplicate linear algebra kernels; instead call into `operations/`/`linear_algebra/` with block-aware wrappers.

## Dependencies
- Uses `core/` block containers and `storage/` block storage.
- Coordinates with `constrained/` and `solvers/` for applying block systems.
- Interacts with `parallel/` for block assembly and distribution.

## Planned Layout
```text
block/
├── block_operations.h       # Element-wise block operations
├── block_extraction.h       # Extraction/insertion utilities
├── block_assembly.h         # Assembly helpers for block systems
├── block_preconditioners.h  # Interfaces for block preconditioners
└── block_solvers.h          # Basic block solve wrappers (delegating to solvers/)
```

## Notes
- Ensure compatibility with autodiff types and matrix-free operators.
- Provide comprehensive tests covering variable block sizes.

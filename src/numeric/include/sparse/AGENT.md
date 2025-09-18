# sparse/ AGENT.md

## Purpose
Provide sparse matrix and vector utilities, including format conversions, builders, and operations specific to sparse data structures. Works alongside `storage/` (which supplies formats) and `linear_algebra/` (which executes sparse kernels).

## Responsibilities
- Implement format-agnostic builders for assembling sparse matrices/vectors from element contributions.
- Provide conversion routines between CSR/CSC/COO/hybrid formats.
- Offer sparsity pattern utilities (graph-based ordering, structural analysis).
- Expose lightweight views over sparse data for use in expressions and matrix-free contexts.

## Non-Goals
- Implement the actual sparse storage classes (`storage/sparse/...` does that).
- Provide sparse BLAS operations (located in `linear_algebra/`).
- Manage graph algorithms beyond sparsity pattern needs (shared with `graph/`).

## Dependencies
- Depends on `storage/` for format definitions.
- Coordinates with `graph/` for ordering/coloring algorithms.
- Supports `parallel/` assembly helpers.

## Planned Layout
```text
sparse/
├── builder.h             # Generic sparse builders
├── pattern.h             # Sparsity pattern utilities
├── format_conversion.h   # CSR<->CSC<->COO conversions
├── compress.h            # Compression utilities
└── sparse_view.h         # Lightweight views into sparse data
```

## Notes
- Focus on header-only templated implementations.
- Provide deterministic assembly options for reproducibility.

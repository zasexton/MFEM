# core/ AGENT.md

## Purpose
Define the primary container types (vectors, matrices, tensors, block structures) that serve as the user-visible interface to numeric data. These containers wrap storage backends, views, and expression infrastructure while remaining agnostic to FEM application details.

## Responsibilities
- Provide dense vector/matrix/tensor classes with expression-template integration and AD compatibility.
- Offer block containers (block vectors, block matrices) tailored for multi-field FEM systems.
- Supply small fixed-size containers optimized for stack usage (`SmallMatrix`, `SmallVector`).
- Expose view types (const/non-const, sub-blocks) for efficient slicing/interoperability.
- Bridge storage implementations to high-level algorithms (operations, solvers) through consistent interfaces.

## Non-Goals
- Define storage formats (delegated to `storage/`).
- Implement numerical algorithms (handled by `operations/`, `linear_algebra/`, etc.).
- Manage indexing logic beyond simple accessors (advanced indexing handled by `indexing/`).

## Dependencies
- Relies on `base/` for CRTP base classes, iterators, traits, and expression support.
- Consumes `storage/` implementations for underlying memory layout.
- Works with `indexing/` to expose user-friendly slicing/fancy indexing.

## Planned Layout
```text
core/
├── vector.h            # Dense vector
├── matrix.h            # Dense matrix
├── tensor.h            # N-dimensional tensor
├── block_vector.h      # Block vector wrapper
├── block_matrix.h      # Block matrix wrapper
├── small_matrix.h      # Small compile-time sized matrices
├── small_vector.h      # Small vectors
├── sparse_vector.h     # High-level sparse vector interface
├── sparse_matrix.h     # High-level sparse matrix interface
└── sparse_tensor.h     # Optional: high-level sparse tensor interface (storage in storage/sparse)
```

## Notes
- Keep container interfaces header-only and constexpr-friendly.
- Ensure constructors/operators accept a wide range of storage backends.
-
- Ownership note: Core does not own sparse storage formats. If `sparse_tensor.h` is provided here, it is a thin, high-level interface that relies on implementations under `storage/sparse/*` (CSR/CSC/COO/hybrid) and views. Core should not reimplement sparse storage mechanics.

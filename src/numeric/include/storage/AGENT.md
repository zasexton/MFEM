# storage/ AGENT.md

## Purpose
Implement storage backends for numeric containers—dense, sparse, block, aligned, hybrid—ensuring efficient memory layout and interoperability with the rest of the numeric stack. Storage classes expose raw data access while delegating algorithms to higher layers.

## Responsibilities
- Provide dense storage classes (contiguous, strided, aligned) for vectors/matrices/tensors.
- Implement sparse storage formats (CSR, CSC, COO, hybrid) with builders and iterators.
- Support block and hierarchical storage for multi-physics systems.
- Offer memory alignment, padding, and small-buffer optimizations.
- Expose generic interfaces used by `core/`, `operations/`, and `linear_algebra/` for data access.

## Non-Goals
- Define container interfaces (handled by `core/`).
- Implement algorithms (handled by other modules).
- Manage allocator policies beyond hooking into `allocators/`.

## Dependencies
- Uses allocators from `allocators/` for specialized allocation strategies.
- Works with `traits/` to expose layout information to algorithms.
- Exposes iterators compatible with `expression_base` and view infrastructure.

## Planned Layout
```text
storage/
├── dense_storage.h        # Basic contiguous storage
├── strided_storage.h      # Arbitrary stride layouts
├── aligned_storage.h      # SIMD-aligned variants
├── static_storage.h       # Compile-time sized data
├── dynamic_storage.h      # Runtime sized data
├── small_storage.h        # Small-buffer optimization
├── sparse/
│   ├── csr_storage.h
│   ├── csc_storage.h
│   ├── coo_storage.h
│   └── hybrid_storage.h
└── block_storage.h        # Block/Hierarchical storage structures
```

## Notes
- Provide serialization hooks for storage formats to support `io/`.
- Keep interfaces minimal to encourage reuse across modules.

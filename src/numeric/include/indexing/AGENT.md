# indexing/ AGENT.md

## Purpose
Deliver high-level indexing helpers (fancy indexing, multi-dimensional selectors, mask-based access) that sit atop the core slicing/view infrastructure. The goal is to make vector, matrix, and tensor indexing intuitive without duplicating base iterator logic.

## Responsibilities
- Provide user-friendly APIs for multi-index selection, ellipsis handling, boolean/integer masks, and new-axis insertion.
- Offer utilities to parse NumPy-style slice strings into the base `MultiIndex` structures.
- Supply composable index builders that integrate with `Tensor`, `Matrix`, and `Vector` types defined in the library.
- Maintain consistency with broadcasting and stride semantics from `base/slice_base.h`.

## Non-Goals
- Reimplement `slice_base` or view structures—they live in `include/base/`.
- Implement storage-specific indexing; this module translates high-level intent into base-level offsets.

## Dependencies
- `base/slice_base.h`, `base/view_base.h` for low-level mechanics.
- `traits/` for concept checks (e.g., ensuring mask dimensions match).
- Works with `expressions/` to integrate indexing inside expression trees.

## Planned Layout
```text
indexing/
├── index.h            # Entry point helpers (idx(...))
├── fancy_indexing.h   # Boolean/integer index arrays
├── ellipsis.h         # Ellipsis handling utilities
├── slice_parser.h     # Parse string-based slices
└── newaxis.h          # New axis insertion helpers
```

## Notes
- Provide runtime validation with informative diagnostics to aid developer experience.
- Include unit tests that ensure compatibility with all container types.

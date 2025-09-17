# allocators/ AGENT.md

## Purpose
Provide memory allocation strategies optimized for numeric workloads (aligned, pooled, arena, tracking). Allocators integrate with `storage/` and other modules that require custom allocations while preserving header-only flexibility.

## Responsibilities
- Implement allocator classes conforming to C++ allocator requirements.
- Offer specialized allocators: aligned, pooling for small matrices, arenas for temporary buffers, tracking allocators for debugging.
- Supply thread-local or parallel-aware versions when needed.
- Provide hooks for AD-specific allocations (e.g., tape storage) through adaptor layers.

## Non-Goals
- Implement storage semantics (handled by `storage/`).
- Manage global memory diagnostics (those belong to `diagnostics/`).
- Duplicate allocator features available in the standard library unless optimization demands it.

## Dependencies
- Optionally uses `support/` for error handling/reporting.
- Must cooperate with `parallel/` to ensure thread safety when used in concurrent contexts.

## Planned Layout
```text
allocators/
├── aligned_allocator.h
├── pool_allocator.h
├── small_matrix_pool.h
├── arena_allocator.h
├── stack_allocator.h
├── tracking_allocator.h
└── ad_allocator.h
```

## Notes
- Provide compile-time switches to disable expensive tracking in release builds.
- Document ownership semantics clearly to avoid double frees.

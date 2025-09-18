# parallel/ AGENT.md

## Purpose
Provide concurrency primitives tailored for numeric assembly and evaluation while remaining self-contained (no dependency on the core library). This module supplies thread pools, parallel loop helpers, and reduction utilities that other numeric components (assembly, matrix-free, solvers) build upon.

## Responsibilities
- Implement a lightweight thread pool and task system suitable for CPU-bound numeric workloads.
- Expose parallel algorithms (`parallel_for`, `parallel_reduce`, `parallel_scan`) that respect container shapes and allow partitioning strategies (e.g., graph coloring for assembly).
- Offer parallel assembly helpers (e.g., `ConcurrentBuilder`) with deterministic accumulation for sparse structures.
- Provide interfaces for plugging alternative backends (e.g., TBB, OpenMP) through adapter layers, keeping the default implementation header-only.

## Non-Goals
- Implement GPU offloading (covered in `backends/` where GPU support can be added).
- Offer a generic concurrency framework beyond numeric needs.

## Dependencies
- Works closely with `matrix_free/`, `assembly` routines, and `graph/` algorithms.
- Utilizes `traits/` to ensure thread-safe operations on containers.
- May rely on `utilities/timing` for profiling (see restructuring below).

## Planned Layout
```text
parallel/
├── thread_pool.h        # Core thread pool implementation
├── parallel_for.h       # Range-based parallel loops
├── parallel_reduce.h    # Reduction helpers
├── parallel_scan.h      # Prefix operations
├── parallel_assembly.h  # FEM assembly helpers
└── partitioning.h       # (Optional) partition strategies shared with graph/
```

## Notes
- Keep deterministic assembly options for reproducibility.
- Provide knobs for affinity, grain size, and scheduling policies.

## Scope Notes
- Partitioning algorithms and ordering heuristics are owned by `graph/` (see `graph/partitioning.h`). If `parallel/partitioning.h` exists, it must be a thin adapter that delegates to `graph/` for algorithmic choices and only exposes scheduling-friendly wrappers; do not duplicate graph logic here.

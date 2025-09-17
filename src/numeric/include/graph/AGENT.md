# graph/ AGENT.md

## Purpose
Provide graph algorithms underpinning sparse assembly and parallel partitioning: adjacency structures, coloring, reordering, and partitioning utilities used throughout the numeric library.

## Responsibilities
- Build adjacency graphs from mesh/connectivity or sparsity patterns.
- Implement graph coloring algorithms to enable conflict-free parallel assembly.
- Provide reordering strategies (RCM, AMD, nested dissection) to improve matrix conditioning.
- Supply partitioning helpers for distributing work across threads/processes (lightweight alternatives to external libraries).

## Non-Goals
- Implement heavy-duty partitioning packages (Metis/ParMetis); offer adapters instead.
- Duplicate sparse pattern utilities (delegated to `sparse/`).

## Dependencies
- Interacts with `sparse/` for pattern generation.
- Used by `parallel/` (`parallel_assembly.h`) and `matrix_free/` to schedule work.
- Works with `constrained/` to account for constraint-induced graph changes.

## Planned Layout
```text
graph/
├── adjacency.h          # Build adjacency graphs from patterns
├── coloring.h           # Graph coloring algorithms
├── reordering.h         # Ordering heuristics
├── partitioning.h       # Work partition strategies
└── graph_utils.h        # Shared helpers
```

## Notes
- Keep implementations deterministic where possible for reproducibility.
- Provide both sequential and parallel variants of algorithms.

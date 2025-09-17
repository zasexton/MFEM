# diagnostics/ AGENT.md

## Purpose
Provide lightweight timing, profiling, and instrumentation helpers tailored for the numeric library. These tools stay dependency-free and can be compiled out in release builds if desired.

## Responsibilities
- Implement scoped timers and simple performance counters for assembly/solve phases.
- Offer hooks to emit diagnostics to user-provided sinks (e.g., stdout, callbacks).
- Supply compile-time flags/macros to enable/disable diagnostics with zero overhead when disabled.

## Non-Goals
- Full-featured tracing/metrics systems (those belong to higher layers if needed).
- Error handling or logging (handled in `support/` and application code).

## Planned Layout
```text
diagnostics/
├── timer.h           # Scoped/aggregate timing utilities
├── profiler.h        # Optional sampling hooks
└── instrumentation.h # Macros/helpers for conditional instrumentation
```

## Notes
- Integrate with `parallel/` safely (thread-safe accumulation).
- Provide minimal overhead in hot loops.

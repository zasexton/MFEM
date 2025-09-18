# backends/ AGENT.md

## Purpose
Define abstraction layers and adapters for optional acceleration libraries (e.g., BLAS, MKL, CUDA) while preserving the library’s zero-dependency default. These interfaces allow swapping implementations without rewriting code across `operations/`, `linear_algebra/`, and `decompositions/`.

## Responsibilities
- Provide lightweight backend descriptors and dispatch helpers (e.g., `BackendTag`, `select_backend()`).
- Offer wrappers around third-party libraries, isolated so that build systems can enable/disable them cleanly.
- Supply capability queries (e.g., does backend support batched GEMM, GPU streams?).
- Expose consistent error translation (backend error -> numeric error handling).

## Non-Goals
- Hard-code dependencies on external libraries; defaults must compile without them.
- Own mathematical kernels; those live in `operations/`, `linear_algebra/`, `decompositions/`, which call into backends via these interfaces.

## Planned Layout
```text
backends/
├── backend_registry.h    # Runtime/compile-time registration
├── blas_backend.h        # Wrapper for generic BLAS API
├── mkl_backend.h         # Intel MKL adapter (optional)
├── cuda_backend.h        # CUDA kernels / cuBLAS adapters (optional)
└── backend_traits.h      # Capabilities & feature detection
```

## Notes
- Keep all adapters header-only where practical with inline PIMPL fallback for heavy dependencies.
- Document configuration macros/environment variables that select backends.

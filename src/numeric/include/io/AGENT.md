# io/ AGENT.md

## Purpose
Provide lightweight input/output utilities specific to numeric data structures—file formats, serialization hooks, and interop with external tools (Matrix Market, NumPy, HDF5)—while keeping dependencies optional.

## Responsibilities
- Implement import/export routines for dense and sparse matrices/vectors in common formats (Matrix Market, CSV, simple binary).
- Support NumPy `.npy/.npz` interop for easy testing and data exchange.
- Offer optional HDF5 bindings (enabled via build configuration) without hard dependency.
- Provide stream-based serializers compatible with `storage/` and `core/` containers.

## Non-Goals
- General-purpose filesystem utilities (handled by application code).
- Logging or diagnostics (handled elsewhere).

## Dependencies
- Works with `storage/` to traverse raw data efficiently.
- Leverages `support/` for error handling.
- Coordinates with `backends/` when writing backend-specific data.

## Planned Layout
```text
io/
├── matrix_market.h      # Read/write Matrix Market files
├── numpy_format.h       # NumPy .npy/.npz support
├── binary_io.h          # Simple binary serialization
└── hdf5_io.h            # Optional HDF5 adapters (guarded by macros)
```

## Notes
- Keep interfaces templated and header-only when possible.
- Provide clear error diagnostics for malformed input files.

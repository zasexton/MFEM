# config/ AGENT.md

## Purpose
Centralize configuration headers for the numeric library: compiler/feature detection, platform toggles, precision selection, and debug macros.

## Responsibilities
- Provide compile-time feature flags (constexpr if-defs) used across headers.
- Detect compiler/platform specifics and normalize macros.
- Offer numeric precision policies and debug/assert helpers.

## Files (planned)
- `config.h`, `compiler.h`, `platform.h`, `features.h`, `precision.h`, `debug.h`.

## Collaborations
- Included by `base/` and nearly all submodules to branch on availability and options.
- Coordinates with `backends/` to toggle optional accelerators.

## Example Usage
```cpp
#include <numeric/config/features.h>

#if defined(NUMERIC_HAS_OPENMP)
  // use OpenMP path
#else
  // fallback path
#endif

// Precision policy
using Real = std::conditional_t<NUMERIC_USE_FLOAT, float, double>;
```

## Notes
- Keep macros namespaced (NUMERIC_*). Avoid leaking symbols.


# support/ AGENT.md

## Purpose
House shared support utilities specific to the numeric library—primarily error handling, assertions, and small helper types that don’t fit into math or diagnostics. Keeps the rest of the project lightweight by centralizing these cross-cutting helpers.

## Responsibilities
- Provide numeric-specific error types/exceptions (`numeric_error`, `singular_matrix_error`, etc.).
- Supply assertion/checking macros that can be toggled independently of the core library.
- Offer helper types (e.g., `scope_guard`, `noncopyable`) when a numeric-specific variant is required.

## Non-Goals
- General-purpose utilities already supplied by the C++ standard library.
- Domain-specific math helpers (in `math/`).
- Logging and diagnostics (other modules handle those concerns).

## Planned Layout
```text
support/
├── error_handling.h   # Error classes and propagation helpers
├── assert.h           # Numeric-specific asserts/checks
└── scope_guard.h      # Utility for exception-safe clean-up (optional)
```

## Notes
- Keep headers minimal and header-only to preserve zero-dependency goals.
- Ensure errors integrate with solvers/operations for clear diagnostic messages.

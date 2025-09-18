# autodiff/tape/ AGENT.md

## Purpose
Provide reverse-mode AD tape infrastructure for recording computational graphs and computing gradients/Jacobians efficiently.

## Responsibilities
- Implement a minimal tape with node storage, edges, and adjoint propagation.
- Offer memory strategies (arena/pool) and reset/checkpoint utilities.
- Provide primitives for common ops (add, mul, exp, sin, etc.).

## Files (planned)
- `tape.h`, `tape_storage.h`, `tape_primitives.h`.

## Collaborations
- `autodiff/types::Var` uses the tape to record operations.
- `allocators/` for custom storage backends.

## Example APIs
```cpp
Tape tape;
with_tape(tape, [&]{
  Var<double> a = 2.0;
  Var<double> b = 3.0;
  Var<double> c = a*b + sin(a);
  c.backward();
  // a.grad(), b.grad() populated
});

// Checkpointing
auto cp = tape.checkpoint();
// ... do work ...
tape.restore(cp);
```

## Notes
- Keep ABI simple and header-only by default; allow optional compiled variant.


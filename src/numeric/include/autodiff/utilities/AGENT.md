# autodiff/utilities/ AGENT.md

## Purpose
Provide helper utilities for seeding derivatives, extracting gradients/Jacobians/Hessians, and managing AD-related traits.

## Responsibilities
- Seed forward-mode and reverse-mode derivatives conveniently.
- Extract derivative objects from AD containers.
- Expose detection/trait helpers to identify AD types.

## Files (planned)
- `seeding.h`, `extraction.h`, `traits.h`.

## Collaborations
- `autodiff/types` for AD types.
- `linear_algebra/` for Jacobian/Hessian assembly patterns.

## Example APIs
```cpp
// Seeding forward-mode
Vector<Dual<double,3>> x(n);
seed_direction(x, /*dir=*/1);   // set d/dp1 = 1 for all entries

// Extract gradient
Vector<Dual<double,3>> y = f(x);
Vector<double> g = extract_derivative(y, /*dir=*/1);

// Reverse-mode gradient from scalar objective
Var<double> J = objective(u);
J.backward();
Vector<double> grad = gather_grads(u);
```

## Notes
- Prefer zero-copy views when extracting derivatives to avoid overhead.


# autodiff/operations/ AGENT.md

## Purpose
Provide arithmetic and transcendental operations for AD types that forward to core `operations/` kernels while ensuring correct derivative propagation.

## Responsibilities
- Implement operator overloads and function wrappers for `Dual`, `HyperDual`, and `Var`.
- Ensure mixed-mode operations (AD with scalar) promote types correctly.
- Bridge to `operations/` customization points so backends can specialize.

## Files (planned)
- `dual_ops.h`, `hyperdual_ops.h`, `reverse_ops.h`.

## Collaborations
- `autodiff/types` for numerical AD types.
- `operations/` for element-wise kernels and reductions.

## Example APIs
```cpp
Dual<double, 2> x(1.0), y(2.0);
auto z = x + 2.0*y - cos(x);
static_assert(std::is_same_v<decltype(z), Dual<double,2>>);

Var<double> a = 1.0, b = 2.0;
auto c = pow(a, 3) + log(b);
c.backward(); // populates adjoints
```

## Notes
- Prefer `if constexpr` + concepts to keep overload sets minimal and unambiguous.
- Keep AD overloads header-only and noexcept where possible.


# autodiff/types/ AGENT.md

## Purpose
Define automatic differentiation (AD) numeric types that compose with all numeric containers and algorithms: forward-mode Dual, second-order HyperDual, reverse-mode Var, and hybrid variants.

## Responsibilities
- Provide templated AD number types with value + derivative storage.
- Support compile-time derivative dimension for forward modes.
- Interoperate with `operations/` kernels and `linear_algebra/` by exposing a simple scalar-like API.

## Files (planned)
- `dual.h` — Dual<T, N> forward-mode.
- `hyperdual.h` — HyperDual<T> second-order forward mode.
- `var.h` — Reverse-mode variable wrapper.
- `complex_dual.h` — Complex-valued duals.
- `mixed_dual.h` — Hybrid AD strategies.

## Collaborations
- `autodiff/operations`: operator overloads and math functions.
- `autodiff/tape`: reverse-mode recording for `Var`.
- `linear_algebra/`, `decompositions/`: accept AD scalars transparently.

## Example APIs
```cpp
// Forward-mode (compile-time N)
Dual<double, 3> x(2.0);     // x = 2
x.derivative(0) = 1.0;      // seed dx/dp0 = 1

auto y = sin(x) + x*x;      // derivatives propagate
double val = y.value();
double dy_dp0 = y.derivative(0);

// Reverse-mode
Var<double> a = 2.0;
Var<double> b = 3.0;
auto c = a*b + sin(a);
c.backward();                // accumulate adjoints
double dc_da = a.grad();
double dc_db = b.grad();

// HyperDual for second derivatives
HyperDual<double> u = make_hyperdual(1.0, /*e1=*/1.0, /*e2=*/0.0);
auto f = exp(u*u);
double d2f = second_derivative(f);   // ∂²f/∂e1∂e2
```

## Notes
- Types must be trivially copyable when possible and remain constexpr-friendly.
- Storage should be aligned as needed for vectorization.


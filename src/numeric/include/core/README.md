# Numeric Core Containers

The **numeric/core** module provides the dense vector-, matrix-, and tensor
classes that power MFEMSolver's expression-template system. These types are
designed to feel familiar to NumPy users while remaining fully interoperable
with the C++ toolchain. Each container exposes intuitive indexing, rich
mathematical operators, lazy evaluation, and lightweight views that avoid
unnecessary allocations.

## Design Highlights

- **Expression templates** – All arithmetic operators build lazily-evaluated
  expression trees. Results are only materialised when a container is assigned
  from an expression.
- **Broadcasting & mixed types** – Scalars participate in expressions via
  implicit broadcasting. Mixed numeric types promote to the appropriate
  `std::common_type`.
- **NumPy-inspired indexing** – Containers accept slices, ellipses, and
  `newaxis` insertion through the `MultiIndex` API or helper literals.
- **Views instead of copies** – `view()` and slicing operations return
  lightweight, strided views that reference the original storage.
- **IEEE-compliant numerics** – All operations honour the project-wide numeric
  policies (overflow checks, atomics, etc.).

## Common Building Blocks

- [`Slice`](../base/slice_base.h): Describes half-open ranges with optional
  step, supports negative indices and defaults just like NumPy.
- [`MultiIndex`](../base/slice_base.h): A heterogeneous collection of indices,
  slices, ellipses, and `newaxis` sentinels used for advanced indexing.
- [`ExpressionBase`](../base/expression_base.h): CRTP base class for expression
  templates. All vectors, matrices, tensors, and views inherit from it.

## Vector

Header: [`vector.h`](vector.h)

```cpp
using fem::numeric::VectorD = Vector<double>;

VectorD x(5, 1.0);            // size constructor (filled with 1.0)
VectorD y = {0.0, 1.0, 2.0};  // initializer-list constructor

auto expr = x + 2.0 * y;      // lazy expression
VectorD z = expr;             // materialises results

auto sub = x(Slice(0, 3));    // strided view over first three entries
sub[1] = 42.0;                // writes back into x
```

### Key Features

- `operator[]`, `at()`, and `view()` for element access.
- Slicing via `Slice` objects (`x(Slice(1, None, 2))`).
- Broadcasting with scalars (`x += 2.0`).
- Dot products, norms, normalisation helpers, and element-wise operations.

## Matrix

Header: [`matrix.h`](matrix.h)

```cpp
Matrix<double> A({{1.0, 2.0},
                  {3.0, 4.0}});

auto expr = A + 5.0;               // element-wise scalar broadcast
Matrix<double> B = expr.transpose();

auto row = A.row(0);               // contiguous vector view
auto col = A.col(1);               // strided view

auto block = A.submatrix(0, 2, 0, 1); // 2x1 view (no copy)
```

### Highlights

- Configurable storage order (row-major or column-major).
- Row/column/diagonal views and `transpose()` returning strided views.
- Expression support for matrix–matrix and matrix–scalar operations.
- Slicing semantics mirror NumPy (`A(idx(all, Slice(1, None)))`).

## Tensor

Header: [`tensor.h`](tensor.h)

```cpp
Tensor<double, 3> T({2, 3, 4});

auto slice = T(idx(1, Slice(0, 3, 2), all));
Tensor<double, 2> front(slice);           // copy from view when needed

auto expanded = T(idx(newaxis, all, all, 0));
Tensor<double, 4> promoted(expanded);     // inserted singleton axis

auto expr = (T + 2.0) * 0.5;
Tensor<double, 3> result = expr;          // lazy evaluation

auto perm = T.permute({2, 0, 1});         // arbitrary axis permutation
```

### Why Tensors Stand Out

- Arbitrary-rank dense containers with contiguous storage.
- Lazy arithmetic (`+`, `-`, `*`, scalar broadcasting) mirroring vectors and
  matrices.
- NumPy-inspired multi-index slicing with `Slice`, `All`, `NewAxis`, and
  ellipsis support. Slices produce `TensorView` objects that reference the
  original tensor.
- `permute()` for axis reordering (generalised transpose).

## Working with Views

- Views are created with `view()`, `submatrix()` (for matrices), or tensor
  multi-index slicing. Views are `ExpressionBase` derivatives and can take part
  in expressions.
- Views never copy data; writing through a non-const view updates the source
  container.
- To materialise a copy, construct a new container from the view
  (e.g. `Matrix<double> C(view)` or `Tensor<double, R> copy(view)`).

## Expression Evaluation

- All containers accept assignment from any `ExpressionBase` derived type.
- When assigning, the container resizes to match the expression's shape and
  evaluates element by element or via the expression's `eval_to()` method if
  available.
- Chaining expressions is cheap: `(A + B) * scalar - C` allocates exactly once
  when assigned to the destination container.

## Broadcasting Rules

- Scalar operands (arithmetic, complex, dual numbers) broadcast to the shape of
  the opposing tensor/matrix/vector.
- Tensor–tensor operations require matching shapes. For tensors, a mismatch
  throws during expression construction to alert the developer immediately.

## Recommended Usage Pattern

1. Construct vectors/matrices/tensors using the provided constructors.
2. Build expressions with the natural arithmetic operators.
3. Assign the expression to a concrete container (or reuse an existing
   container) to trigger evaluation.
4. Use `view()` or slicing in hot loops to avoid temporary allocations.

## Example: Chain of Operations

```cpp
using fem::numeric::VectorD;
using fem::numeric::Matrix<double>;
using fem::numeric::Tensor<double, 3>;

VectorD x(100, 1.0), y(100, 2.0);
VectorD z = (x + y) * 0.5;  // Lazy addition + broadcast

Matrix<double> A({{1, 2}, {3, 4}});
Matrix<double> B = (A + 2.0).transpose();

Tensor<double, 3> T({2, 3, 4}, 1.0);
auto view = T(idx(0, Slice(0, 3), Slice(0, 4, 2)));
Tensor<double, 2> sub(view);  // Materialise when necessary

Tensor<double, 3> R = T.permute({2, 0, 1});
```

## Further Reading

- [`base/expression_base.h`](../base/expression_base.h) for the underlying
  expression implementation.
- [`base/slice_base.h`](../base/slice_base.h) for slicing utilities and helper
  literals.
- Unit tests in `src/numeric/tests/unit/core` provide additional usage
  examples and edge-case coverage.

Feel free to contribute additional algorithms, specialised views, or higher
level containers. Follow the existing conventions: prefer views over copies,
expose intuitive NumPy-like APIs, and ensure new operators participate in the
expression system.

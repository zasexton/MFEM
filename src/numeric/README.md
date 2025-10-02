FEM Numeric Library — Quick Start and Overview

What is it?
- A modern C++20, header-first numeric library purpose-built for FEM workflows.
- Provides dense/sparse containers, expression templates, BLAS-like linear algebra, and factorization routines with clean layering and optional accelerated backends.
- Designed to be FEM-agnostic: no dependency on FEM-specific classes; use it standalone or inside larger FEM projects.

Key Capabilities
- Core containers: Vector, Matrix (row/column-major), Tensor, views, and block/sparse interfaces.
- Expressions: lazy element-wise ops and broadcasting to minimize temporaries.
- Operations: arithmetic kernels and reductions (sum, min/max, dot, norms).
- Linear algebra: BLAS-like Level 1/2/3 (dot/axpy, gemv, gemm) in pure C++.
- Decompositions: LU/QR/SVD/Cholesky/LDLT and basic eigenvalue utilities.
- Backends: optional LAPACK/MKL hooks for high-performance, drop-in acceleration.
- Planned: autodiff types, matrix-free operators, parallel helpers, IO utilities.

Install or Use In-Tree
1) As a subdirectory (recommended during development)
- CMakeLists.txt of your project:
  add_subdirectory(path/to/src/numeric)
  target_link_libraries(your_target PRIVATE fem::numeric)

2) As an installed package
- Configure and install this repo:
  cmake -S src/numeric -B build-numeric -DFEM_NUMERIC_HEADER_ONLY=ON
  cmake --build build-numeric --target install
- In your project CMake:
  find_package(FEMNumeric REQUIRED)
  target_link_libraries(your_target PRIVATE fem::numeric)

Optional features
- Enable LAPACK backend:
  -DFEM_NUMERIC_ENABLE_LAPACK=ON
  If your system provides LAPACK (or MKL), the library will automatically wrap those routines for decompositions.
- Header-only mode (default today):
  -DFEM_NUMERIC_HEADER_ONLY=ON
- SIMD and tuning flags are auto-configured based on compiler and build type.

Single-Header Include
- For most users, include the library via the aggregator header:
  #include "numeric.h"
  using namespace fem::numeric; // optional, or qualify with fem::numeric::

Quick Start Examples

1) Vectors and element-wise math
  #include "numeric.h"
  using namespace fem::numeric;
  using fem::numeric::operations::add;      // or use operator helpers directly
  using fem::numeric::operations::sum;

  int main() {
    Vector<double> x{1.0, 2.0, 3.0};
    Vector<double> y{4.0, 5.0, 6.0};

    // Lazy expression; evaluated on assignment
    Vector<double> z = add(x, y);  // z = x + y => {5,7,9}

    // Reductions work on containers or expressions
    double s = sum(z);             // 21.0
    return (s > 0) ? 0 : 1;
  }

2) Matrix creation and matrix–vector (GEMV)
  #include "numeric.h"
  using namespace fem::numeric;
  using namespace fem::numeric::linear_algebra;

  int main() {
    // Row-major by default; choose ColumnMajor for BLAS-style layouts
    Matrix<double> A{{1, 2, 3},
                     {4, 5, 6}};  // 2x3
    Vector<double> x{1, 1, 1};     // 3
    Vector<double> y(2, 0.0);      // 2

    // y = 1.0*A*x + 0.0*y
    gemv(Trans::NoTrans, 1.0, A, x, 0.0, y);  // y = {6, 15}
    return (y[0] == 6.0 && y[1] == 15.0) ? 0 : 1;
  }

3) Matrix–matrix (GEMM)
  #include "numeric.h"
  using namespace fem::numeric;
  using namespace fem::numeric::linear_algebra;

  int main() {
    Matrix<double> A{{1, 2},
                     {3, 4}};   // 2x2
    Matrix<double> B{{5, 6},
                     {7, 8}};   // 2x2
    Matrix<double> C(2, 2, 0.0);

    gemm(Trans::NoTrans, Trans::NoTrans,
         1.0, A, B,
         0.0, C);
    // C = A*B = {{19, 22}, {43, 50}}
    return (C(0,0) == 19.0 && C(1,1) == 50.0) ? 0 : 1;
  }

4) Solve a linear system with LU
  #include "numeric.h"
  using namespace fem::numeric;
  using namespace fem::numeric::decompositions;

  int main() {
    Matrix<double> A{{3, 2},
                     {1, 4}};    // SPD not required (general)
    Vector<double> b{5, 6};

    std::vector<int> piv;
    // A is factorized in-place to LU; piv holds row swaps
    lu_factor(A, piv);
    lu_solve(A, piv, b);  // b becomes the solution x
    // Check solution roughly
    return (std::abs(b[0] - 0.8) < 1e-12 && std::abs(b[1] - 1.3) < 1e-12) ? 0 : 1;
  }

5) Expressions, broadcasting, and in-place helpers
  #include "numeric.h"
  using namespace fem::numeric;
  using namespace fem::numeric::operations;

  int main() {
    Vector<double> v{1,2,3};
    auto expr = mul_scalar(v, 2.0); // lazy 2*v
    Vector<double> out = expr;      // evaluated here

    // In-place axpy: y := y + alpha*x
    axpy_inplace(out, 0.5, v);      // out = 2*v + 0.5*v = 2.5*v
    return (out[2] == 7.5) ? 0 : 1;
  }

Layout, Backends, and Performance Notes
- Storage order: Matrix<T, Storage, StorageOrder> defaults to RowMajor. ColumnMajor is often best for LAPACK/BLAS interop; pick based on your kernels and backend.
- LAPACK backend: When enabled (FEM_NUMERIC_ENABLE_LAPACK=ON) and a provider is found, decompositions automatically dispatch to it when profitable (e.g., ColumnMajor matrices).
- Expressions: Prefer composing expressions then assigning to a destination to reduce temporaries and improve locality.

Build Options (selected)
- FEM_NUMERIC_HEADER_ONLY=ON        Use as header-only (current default)
- FEM_NUMERIC_ENABLE_LAPACK=ON      Enable LAPACK/MKL backend where available
- FEM_NUMERIC_ENABLE_SIMD=ON        Turn on SIMD-friendly flags (default)
- Standard warnings/optimizations are set per compiler; Debug builds add helpful diagnostics.

What’s Planned (roadmap alignment)
- Autodiff types (forward/reverse) integrated by element type.
- Matrix-free operators and preconditioners.
- Parallel helpers (thread pool, parallel_for/reduce, graph coloring).
- IO helpers (Matrix Market, NumPy) and diagnostics.

FAQ
- Is it tied to a specific FEM library?
  No. It’s a standalone numeric layer designed to be embedded under FEM code, but it does not depend on FEM types.

- Do I need LAPACK/MKL?
  No. The library ships with pure C++ implementations. LAPACK just makes many decompositions faster.

- Which namespace should I use?
  All types live in fem::numeric with sub-namespaces like operations, linear_algebra, and decompositions.

Directory Pointers
- Public headers: src/numeric/include
- Aggregator header: src/numeric/include/numeric.h
- BLAS-like ops: src/numeric/include/linear_algebra
- Factorizations: src/numeric/include/decompositions
- Element-wise ops: src/numeric/include/operations

If you’d like a deeper tour or runnable samples, we can add small examples under src/numeric/examples on request.


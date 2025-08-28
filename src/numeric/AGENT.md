# AGENT.md - FEM Numeric Library (Revised)

## Mission
Build a high-performance, FEM-oriented numerical library that provides all mathematical operations required for finite element analysis while remaining modular and independent of FEM-specific code.

## Design Philosophy

### Core Principles
1. **Complete Independence**: No dependencies on FEM core classes
2. **FEM-Oriented Design**: Optimized for FEM assembly and solve patterns
3. **Self-Contained**: All functionality within the numeric namespace
4. **Header-Only Options**: Support for header-only usage for templates
5. **Assembly-First**: Designed for incremental, concurrent assembly
6. **Block-Aware**: Native support for block-structured systems
7. **Zero External Dependencies**: Pure C++ with optional accelerated backends

## Complete Directory Structure

```
numeric/
├── README.md                         # Library overview
├── AGENT.md                          # This document
├── CMakeLists.txt                    # Build configuration
├── numeric.hpp                       # Single header include
│
├── config/                           # Configuration and platform detection
│   ├── config.hpp                    # Main configuration
│   ├── compiler.hpp                  # Compiler detection
│   ├── platform.hpp                  # Platform specifics
│   ├── features.hpp                  # Feature flags
│   ├── precision.hpp                 # Floating-point settings
│   └── debug.hpp                     # Debug/assertion macros
│
├── base/                             # Base infrastructure
│   ├── numeric_base.hpp              # Foundation types
│   ├── container_base.hpp            # Container CRTP base
│   ├── storage_base.hpp              # Storage abstractions
│   ├── expression_base.hpp           # Expression templates
│   ├── iterator_base.hpp             # Iterator interfaces
│   ├── traits_base.hpp               # Base traits
│   ├── allocator_base.hpp            # Allocator interfaces
│   ├── view_base.hpp                 # View interfaces
│   ├── slice_base.hpp                # Slicing operations
│   ├── ops_base.hpp                  # Operation functors
│   └── broadcast_base.hpp            # Broadcasting support
│
├── traits/                           # Type traits and concepts
│   ├── type_traits.hpp               # Basic type traits
│   ├── numeric_traits.hpp            # Numeric properties
│   ├── container_traits.hpp          # Container detection
│   ├── expression_traits.hpp         # Expression traits
│   ├── operation_traits.hpp          # Operation compatibility
│   ├── storage_traits.hpp            # Storage characteristics
│   ├── iterator_traits.hpp           # Iterator properties
│   ├── block_traits.hpp              # Block structure traits
│   ├── concepts.hpp                  # C++20 concepts
│   └── sfinae_helpers.hpp            # SFINAE utilities
│
├── core/                             # Core mathematical objects
│   ├── vector.hpp                    # Dense vector
│   ├── matrix.hpp                    # Dense matrix
│   ├── tensor.hpp                    # N-dimensional tensor
│   ├── block_vector.hpp              # Block vectors
│   ├── block_matrix.hpp              # Block matrices
│   ├── small_matrix.hpp              # Small matrix optimizations
│   ├── sparse_vector.hpp             # Sparse vector
│   ├── sparse_matrix.hpp             # Sparse matrix base
│   ├── sparse_tensor.hpp             # Sparse tensor
│   └── complex.hpp                   # Complex number support
│
├── storage/                          # Storage implementations
│   ├── dense_storage.hpp             # Contiguous memory
│   ├── sparse_storage.hpp            # Sparse formats base
│   ├── block_storage.hpp             # Block storage
│   ├── small_storage.hpp             # SSO for small matrices
│   ├── compressed_storage.hpp        # CSR, CSC, COO
│   ├── strided_storage.hpp           # Strided views
│   ├── static_storage.hpp            # Compile-time sized
│   ├── dynamic_storage.hpp           # Runtime sized
│   ├── aligned_storage.hpp           # SIMD alignment
│   └── hybrid_storage.hpp            # Small-buffer optimization
│
├── allocators/                       # Memory allocators
│   ├── aligned_allocator.hpp         # Aligned allocation
│   ├── pool_allocator.hpp            # Pool-based allocation
│   ├── small_matrix_pool.hpp         # Element matrix pool
│   ├── arena_allocator.hpp           # Arena allocation
│   ├── stack_allocator.hpp           # Stack allocation
│   └── tracking_allocator.hpp        # Memory debugging
│
├── sparse/                           # Sparse matrix support
│   ├── formats/
│   │   ├── csr.hpp                  # Compressed sparse row
│   │   ├── csc.hpp                  # Compressed sparse column
│   │   ├── coo.hpp                  # Coordinate format
│   │   ├── block_csr.hpp            # Block CSR
│   │   ├── block_diagonal.hpp       # Block diagonal
│   │   ├── nested_matrix.hpp        # Hierarchical blocks
│   │   ├── dia.hpp                  # Diagonal format
│   │   └── ell.hpp                  # ELLPACK format
│   ├── operations/
│   │   ├── sparse_blas.hpp          # Sparse BLAS
│   │   ├── sparse_arithmetic.hpp    # Arithmetic ops
│   │   ├── sparse_assembly.hpp      # Assembly utilities
│   │   └── sparse_conversion.hpp    # Format conversion
│   └── builders/
│       ├── triplet_builder.hpp      # Build from triplets
│       ├── concurrent_builder.hpp   # Thread-safe assembly
│       ├── assembly_cache.hpp       # Pattern reuse
│       ├── atomic_operations.hpp    # Atomic adds
│       └── incremental_builder.hpp  # Incremental construction
│
├── block/                           # Block-structured operations
│   ├── block_operations.hpp         # Block-wise operations
│   ├── block_extraction.hpp         # Extract/insert blocks
│   ├── block_assembly.hpp           # Block assembly patterns
│   ├── block_preconditioners.hpp    # Block Jacobi, etc.
│   ├── block_solvers.hpp            # Block-aware solvers
│   └── variable_block.hpp           # Variable-sized blocks
│
├── constrained/                     # Constrained systems
│   ├── constraint_handler.hpp       # Constraint elimination
│   ├── lagrange_system.hpp          # Lagrange multipliers
│   ├── penalty_method.hpp           # Penalty methods
│   ├── null_space.hpp               # Null-space methods
│   ├── periodic_constraints.hpp     # Periodic BCs
│   ├── multipoint_constraints.hpp   # MPC handling
│   └── schur_complement.hpp         # Schur complements
│
├── graph/                           # Graph algorithms
│   ├── adjacency.hpp                # Adjacency structures
│   ├── coloring.hpp                 # Graph coloring
│   ├── reordering.hpp               # Cuthill-McKee, etc.
│   ├── bandwidth.hpp                # Bandwidth reduction
│   ├── partitioning.hpp             # Graph partitioning
│   ├── connectivity.hpp             # Connected components
│   └── dependency.hpp               # Dependency analysis
│
├── polynomial/                      # Polynomial operations
│   ├── quadrature.hpp               # Gauss quadrature rules
│   ├── legendre.hpp                 # Legendre polynomials
│   ├── chebyshev.hpp                # Chebyshev polynomials
│   ├── bernstein.hpp                # Bernstein basis
│   ├── hermite.hpp                  # Hermite polynomials
│   ├── vandermonde.hpp              # Vandermonde matrices
│   ├── interpolation.hpp            # Polynomial interpolation
│   └── integration.hpp              # Numerical integration
│
├── expressions/                      # Expression templates
│   ├── expression.hpp               # Expression machinery
│   ├── binary_ops.hpp               # Binary operations
│   ├── unary_ops.hpp                # Unary operations
│   ├── scalar_ops.hpp               # Scalar operations
│   ├── matrix_product.hpp           # Matrix multiplication
│   ├── block_expressions.hpp        # Block operations
│   ├── petrov_galerkin.hpp          # Non-symmetric ops
│   ├── aliasing.hpp                 # Aliasing detection
│   └── evaluation.hpp               # Evaluation strategies
│
├── operations/                      # Mathematical operations
│   ├── arithmetic.hpp               # Basic arithmetic
│   ├── transcendental.hpp           # sin, cos, exp, log
│   ├── tensor_contraction.hpp       # Tensor contractions
│   ├── kronecker_product.hpp        # Kronecker products
│   ├── stabilization.hpp            # SUPG/GLS matrices
│   ├── skew_symmetric.hpp           # Skew operations
│   ├── reductions.hpp               # Sum, mean, etc.
│   └── element_wise.hpp             # Element-wise ops
│
├── linear_algebra/                  # Linear algebra operations
│   ├── blas_level1.hpp              # Vector operations
│   ├── blas_level2.hpp              # Matrix-vector ops
│   ├── blas_level3.hpp              # Matrix-matrix ops
│   ├── batched_blas.hpp             # Batched operations
│   ├── equilibration.hpp            # Matrix scaling
│   ├── condition_number.hpp         # Condition estimation
│   ├── iterative_refinement.hpp     # Mixed precision
│   ├── decompositions.hpp           # LU, QR, SVD
│   └── norms.hpp                    # Vector/matrix norms
│
├── decompositions/                  # Matrix decompositions
│   ├── lu.hpp                       # LU decomposition
│   ├── qr.hpp                       # QR decomposition
│   ├── svd.hpp                      # Singular values
│   ├── eigen.hpp                    # Eigenvalues
│   ├── cholesky.hpp                 # Cholesky
│   ├── ldlt.hpp                     # LDLT decomposition
│   ├── schur.hpp                    # Schur decomposition
│   └── block_lu.hpp                 # Block LU
│
├── solvers/                         # Linear system solvers
│   ├── direct/
│   │   ├── lu_solver.hpp            # LU solver
│   │   ├── qr_solver.hpp            # QR solver
│   │   ├── cholesky_solver.hpp      # Cholesky solver
│   │   ├── block_solver.hpp         # Block direct solver
│   │   └── band_solver.hpp          # Banded systems
│   ├── iterative/
│   │   ├── conjugate_gradient.hpp   # CG
│   │   ├── gmres.hpp                # GMRES
│   │   ├── bicgstab.hpp             # BiCGSTAB
│   │   ├── minres.hpp               # MINRES
│   │   ├── chebyshev.hpp            # Chebyshev iteration
│   │   └── multigrid.hpp            # Multigrid methods
│   ├── eigen/
│   │   ├── power_method.hpp         # Power iteration
│   │   ├── arnoldi.hpp              # Arnoldi
│   │   ├── lanczos.hpp              # Lanczos
│   │   ├── generalized_eigen.hpp    # K*x = λ*M*x
│   │   ├── shift_invert.hpp         # Shift-invert
│   │   └── subspace_iteration.hpp   # Multiple eigenpairs
│   └── preconditioners/
│       ├── diagonal.hpp             # Jacobi
│       ├── ilu.hpp                  # Incomplete LU
│       ├── block_preconditioner.hpp # Block preconditioning
│       ├── amg.hpp                  # Algebraic multigrid
│       └── field_split.hpp          # Field splitting
│
├── matrix_free/                     # Matrix-free methods
│   ├── operator.hpp                 # Matrix-free base
│   ├── sum_factorization.hpp        # Tensor-product kernels
│   ├── diagonal_approximation.hpp   # Diagonal computation
│   ├── chebyshev_smoother.hpp       # For multigrid
│   └── matrix_free_preconditioner.hpp
│
├── assembly/                        # FEM assembly support
│   ├── dof_map.hpp                  # DOF connectivity
│   ├── local_to_global.hpp          # Index mapping
│   ├── shape_function_cache.hpp     # Cache evaluations
│   ├── integration_points.hpp       # Standard rules
│   ├── element_matrix_pool.hpp      # Memory pool
│   └── assembly_pattern.hpp         # Sparsity patterns
│
├── indexing/                        # Indexing and slicing
│   ├── index.hpp                    # Basic indexing
│   ├── slice.hpp                    # Slice objects
│   ├── fancy_indexing.hpp           # Advanced indexing
│   ├── block_indexing.hpp           # Block indexing
│   ├── dof_indexing.hpp             # DOF-based indexing
│   └── multi_index.hpp              # Multi-dimensional
│
├── parallel/                        # Parallelization support
│   ├── parallel_for.hpp             # Parallel loops
│   ├── parallel_reduce.hpp          # Reductions
│   ├── parallel_assembly.hpp        # Concurrent assembly
│   ├── graph_coloring.hpp           # For conflict-free assembly
│   ├── thread_pool.hpp              # Thread pool
│   └── simd_operations.hpp          # SIMD vectorization
│
├── io/                              # Input/output
│   ├── matrix_market.hpp            # Matrix Market format
│   ├── numpy_format.hpp             # NumPy .npy
│   ├── matlab_io.hpp                # MATLAB .mat
│   ├── hdf5_io.hpp                  # HDF5 support
│   ├── vtk_io.hpp                   # VTK matrix output
│   └── binary_io.hpp                # Binary format
│
├── utilities/                       # Utility functions
│   ├── math_functions.hpp           # Common math
│   ├── comparison.hpp               # Float comparison
│   ├── timer.hpp                    # Performance timing
│   ├── memory_utils.hpp             # Memory utilities
│   ├── fem_utils.hpp                # FEM-specific utilities
│   └── error_handling.hpp           # Error handling
│
├── adapters/                        # Optional FEM integration
│   ├── fem_adapter.hpp              # FEM core adapter
│   ├── fem_matrix_wrapper.hpp       # Wrap as FEM objects
│   └── fem_solver_interface.hpp     # Solver integration
│
├── backends/                        # External library backends
│   ├── blas_backend.hpp             # BLAS/LAPACK
│   ├── mkl_backend.hpp              # Intel MKL
│   ├── cuda_backend.hpp             # CUDA support
│   ├── petsc_backend.hpp            # PETSc wrapper
│   └── trilinos_backend.hpp         # Trilinos wrapper
│
├── tests/                           # Comprehensive testing
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── fem_specific/                # FEM-oriented tests
│   └── performance/                 # Performance tests
│
└── examples/                        # Usage examples
    ├── basic_usage.cpp              # Basic operations
    ├── fem_assembly.cpp             # FEM assembly example
    ├── block_systems.cpp            # Block system solve
    ├── constrained_systems.cpp      # Constraints example
    └── matrix_free.cpp              # Matrix-free example
```

## Key Design Features

### 1. Block-Structured Systems
```cpp
// Native support for multi-field problems
BlockMatrix<double> K;  // System matrix
K.set_block_sizes({3, 3, 1});  // u, v, w, p for Stokes

// Block operations
auto K_uu = K.block(0, 0);  // Velocity block
auto K_up = K.block(0, 3);  // Velocity-pressure coupling

// Block preconditioners
BlockPreconditioner<double> P;
P.set_diagonal_blocks({K_uu.factorize(), K_pp.factorize()});
```

### 2. Constraint Handling
```cpp
// Eliminate Dirichlet constraints
ConstraintHandler constraints;
constraints.add_dirichlet(node_id, dof, value);
constraints.add_periodic(node1, node2);
constraints.eliminate(K, F);  // Modify system

// Lagrange multipliers for constraints
LagrangeSystem<double> sys(K, C);  // K*u + C'*λ = f, C*u = g
sys.solve(u, lambda, f, g);
```

### 3. Incremental Assembly
```cpp
// Thread-safe assembly with graph coloring
ConcurrentBuilder<double> builder(sparsity_pattern);
GraphColoring coloring(mesh.connectivity());

parallel_for(coloring.color(c), [&](int elem) {
    auto K_e = compute_element_matrix(elem);
    builder.add_atomic(elem.dofs(), K_e);  // Thread-safe
});

// Assembly with caching for nonlinear problems
AssemblyCache cache;
if (!cache.pattern_changed()) {
    cache.zero_entries();  // Keep structure
} else {
    cache.rebuild_pattern();
}
```

### 4. Matrix-Free Operations
```cpp
// High-order FEM without storing matrix
template<typename Physics>
class MatrixFreeOperator {
    void apply(const Vector& x, Vector& y) {
        parallel_for(elements, [&](int e) {
            // Sum-factorization for tensor-product elements
            auto y_local = sum_factorization.apply(
                physics.compute_action(e), x.local(e)
            );
            y.add_local(e, y_local);
        });
    }
};

// Use in iterative solver
MatrixFreeOperator<Elasticity> A;
gmres.solve(A, b, x);  // No matrix storage
```

### 5. Small Matrix Optimizations
```cpp
// Stack-allocated small matrices for elements
SmallMatrix<double, 8, 8> K_e;  // 8×8 element stiffness
SmallVector<double, 8> F_e;      // 8×1 element force

// Uses SSO (small storage optimization)
// No heap allocation for typical element sizes
```

### 6. Polynomial/Quadrature Support
```cpp
// Quadrature for element integration
auto quad = GaussQuadrature<3>::create(order=2);  // 3D, order 2
for (auto& qp : quad.points()) {
    double w = qp.weight;
    auto xi = qp.coords;
    // Evaluate shape functions at quadrature point
    auto N = shape_functions.evaluate(xi);
}

// High-order polynomial bases
LegendrePolynomial<3> leg(degree=4);
auto vandermonde = leg.vandermonde_matrix(points);
```

### 7. Graph Algorithms for Assembly
```cpp
// Bandwidth reduction
auto perm = CuthillMcKee::compute(K.sparsity());
K.permute(perm);

// Graph coloring for parallel assembly
GraphColoring coloring(mesh.connectivity());
for (int color = 0; color < coloring.num_colors(); ++color) {
    parallel_for(coloring.elements(color), [&](int e) {
        // Safe parallel assembly - no conflicts
        assemble_element(e);
    });
}
```

### 8. FEM-Specific Eigensolvers
```cpp
// Generalized eigenvalue problem: K*x = λ*M*x
GeneralizedEigenSolver solver;
solver.set_matrices(K, M);
solver.set_target(EigenTarget::SMALLEST_REAL, n=10);
auto [eigenvalues, eigenvectors] = solver.solve();

// Shift-invert for interior eigenvalues
ShiftInvertSolver solver(K, M, sigma=1.5);
solver.find_nearest(n=5);
```

## Implementation Priorities

### Phase 1: FEM Essentials (Weeks 1-2)
1. Block matrix/vector structures
2. Sparse matrix assembly infrastructure
3. Constraint elimination
4. Basic graph algorithms

### Phase 2: Assembly Support (Weeks 3-4)
1. Concurrent assembly with atomics
2. Graph coloring
3. Assembly caching
4. DOF mapping utilities

### Phase 3: Solvers for FEM (Weeks 5-6)
1. Block solvers and preconditioners
2. Constrained system solvers
3. Generalized eigensolvers
4. Multigrid components

### Phase 4: Advanced Features (Weeks 7-8)
1. Matrix-free infrastructure
2. Polynomial/quadrature support
3. Tensor contractions
4. Stabilization matrices

### Phase 5: Optimization (Weeks 9-10)
1. Small matrix optimizations
2. Batched BLAS operations
3. Memory pools for elements
4. SIMD optimizations

### Phase 6: Integration (Weeks 11-12)
1. External solver wrappers
2. FEM adapter layer
3. Performance benchmarks
4. Comprehensive testing

## Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Element Matrix Assembly | < 100ns per 8×8 | Stack allocation, no malloc |
| Sparse Matrix Insert | < 20ns atomic | Lock-free assembly |
| Block Matrix Access | O(1) | Direct block indexing |
| Constraint Elimination | < 5% overhead | Efficient row/col removal |
| Graph Coloring | O(V + E) | Linear time coloring |
| Matrix-Free SpMV | > 20% peak FLOPS | Sum-factorization |
| Small Matrix Multiply | > 80% peak | SIMD, unrolled |

## Memory Model

### Hierarchy
1. **Small matrices**: Stack allocation (< 32×32)
2. **Element matrices**: Pool allocator (reused)
3. **Sparse matrices**: Compressed formats
4. **Block matrices**: Hierarchical storage
5. **Large dense**: Aligned heap allocation

### Concurrent Access
- Lock-free assembly via atomics
- Graph coloring for conflict-free parallel
- Thread-local assembly buffers
- Read-write locks for matrix structure changes

## 🔑 Success Metrics

1. **FEM Assembly**: 10x faster than naive implementation
2. **Block Operations**: Native support, no extraction overhead
3. **Constraint Handling**: Transparent to solver
4. **Memory Usage**: < 2x theoretical minimum for sparse
5. **Parallel Efficiency**: > 85% on 8 cores for assembly
6. **Matrix-Free**: Competitive with matrix-explicit methods
7. **Accuracy**: IEEE 754 compliant, stable numerics

## 🎭 Usage Patterns

### Typical FEM Assembly
```cpp
// Setup
auto mesh = /* ... */;
auto sparsity = compute_sparsity_pattern(mesh);
ConcurrentBuilder<double> K(sparsity);
SmallMatrixPool pool;

// Parallel assembly with coloring
auto coloring = GraphColoring(mesh);
for (int color : coloring.colors()) {
    parallel_for(coloring.elements(color), [&](int e) {
        auto K_e = pool.allocate<8, 8>();
        compute_element_matrix(e, K_e);
        K.add_atomic(mesh.dofs(e), K_e);
        pool.deallocate(K_e);
    });
}

// Apply constraints
constraints.eliminate(K, F);

// Solve
auto solver = make_solver<BlockGMRES>(K);
solver.set_preconditioner(make_block_ilu(K));
solver.solve(F, u);
```

### Multi-Physics Block System
```cpp
// Thermo-mechanical coupling
BlockMatrix<double> K;
K.set_structure({
    {"displacement", 3*n_nodes},
    {"temperature", n_nodes}
});

// Assemble blocks
K.block("displacement", "displacement") = K_uu;
K.block("displacement", "temperature") = K_ut;
K.block("temperature", "displacement") = K_tu;
K.block("temperature", "temperature") = K_tt;

// Field-split preconditioner
FieldSplitPreconditioner P;
P.add_field("displacement", make_amg(K_uu));
P.add_field("temperature", make_ilu(K_tt));

// Solve coupled system
gmres.solve(K, F, u, P);
```

## Future Extensions

1. **GPU Assembly**: Device-side element assembly
2. **Adaptive Precision**: Mixed precision iterative refinement
3. **Hierarchical Matrices**: H-matrices for dense blocks
4. **Tensor Decompositions**: For model reduction
5. **Automatic Differentiation**: For sensitivities

## Key Differences from Original

| Feature | Original | Revised |
|---------|----------|---------|
| Block Matrices | Missing | Native support |
| Constraints | Missing | Full elimination/Lagrange |
| Assembly | Basic sparse | Concurrent with caching |
| Graph Algorithms | Missing | Coloring, reordering |
| Polynomials | Missing | Quadrature, bases |
| Matrix-Free | Missing | Sum-factorization |
| FEM Integration | Minimal | Assembly utilities |
| Small Matrices | Generic | Optimized pools |
| Eigensolvers | Basic | Generalized, shift-invert |

This revised architecture addresses all critical gaps for FEM while maintaining the modular, high-performance design philosophy.
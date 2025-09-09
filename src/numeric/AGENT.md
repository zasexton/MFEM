# AGENT.md - FEM Numeric Library (Revised)

## Mission
Build a high-performance, FEM-oriented numerical library that provides all mathematical operations required for finite element analysis while remaining modular and independent of FEM-specific code. The library uses compositional design where advanced features like automatic differentiation are achieved through element type selection rather than specialized container classes.

## Design Philosophy

### Core Principles
1. **Complete Independence**: No dependencies on FEM core classes
2. **Compositional Design**: Features emerge from type composition (e.g., `Vector<Dual<double, 3>>`)
3. **FEM-Oriented Design**: Optimized for FEM assembly and solve patterns
4. **Self-Contained**: All functionality within the numeric namespace
5. **Header-Only Options**: Support for header-only usage for templates
6. **Assembly-First**: Designed for incremental, concurrent assembly
7. **Block-Aware**: Native support for block-structured systems
8. **Zero External Dependencies**: Pure C++ with optional accelerated backends

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
├── base/                             # Base infrastructure (COMPLETE)
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
│   ├── broadcast_base.hpp            # Broadcasting support
│   └── dual_base.hpp                 # Base for dual number types
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
├── autodiff/                         # Automatic differentiation as element types
│   ├── types/                        # AD numeric types (compose with containers)
│   │   ├── dual.hpp                  # Dual<T, N> for forward-mode
│   │   ├── hyperdual.hpp             # HyperDual<T> for second derivatives
│   │   ├── var.hpp                   # Var<T> for reverse-mode (tape-based)
│   │   ├── complex_dual.hpp          # ComplexDual<T, N> for complex AD
│   │   └── mixed_dual.hpp            # Mixed forward/reverse strategies
│   │
│   ├── traits/                       # AD-specific traits
│   │   ├── ad_traits.hpp             # is_ad_type, value_type extraction
│   │   ├── derivative_traits.hpp     # derivative dimensions and types
│   │   ├── promotion_traits.hpp      # Type promotion for mixed AD/non-AD
│   │   └── tape_traits.hpp           # Tape requirements and properties
│   │
│   ├── operations/                   # AD-aware operations
│   │   ├── ad_arithmetic.hpp         # +, -, *, / for AD types
│   │   ├── ad_transcendental.hpp     # sin, cos, exp, log, etc.
│   │   ├── ad_comparison.hpp         # Comparisons on value part
│   │   ├── ad_special.hpp            # erf, gamma, bessel functions
│   │   └── ad_power.hpp              # pow, sqrt with derivatives
│   │
│   ├── algebra/                      # Linear algebra specializations
│   │   ├── ad_blas.hpp              # BLAS for AD element types
│   │   ├── ad_decompositions.hpp    # LU, QR, Cholesky for AD
│   │   ├── ad_norms.hpp             # Differentiable norm computations
│   │   ├── ad_eigenvalues.hpp       # Eigenvalue derivatives
│   │   └── ad_solvers.hpp           # Linear solvers preserving derivatives
│   │
│   ├── tape/                         # Reverse-mode infrastructure
│   │   ├── tape.hpp                  # Core computation graph
│   │   ├── tape_pool.hpp             # Memory pool for tape nodes
│   │   ├── expression_recording.hpp  # Record operations
│   │   ├── gradient_accumulator.hpp  # Backpropagation engine
│   │   ├── checkpointing.hpp         # Memory-efficient taping
│   │   └── tape_optimizer.hpp        # Graph optimization
│   │
│   └── utilities/                    # AD support utilities
│       ├── seed_strategies.hpp       # Jacobian/Hessian seeding
│       ├── sparsity_detection.hpp    # Detect Jacobian sparsity
│       ├── derivative_checker.hpp    # Finite difference validation
│       ├── ad_io.hpp                 # Printing AD types
│       └── sensitivity_analysis.hpp  # Parameter sensitivity tools
│
├── core/                             # Core mathematical objects
│   ├── vector.hpp                    # Dense vector (works with any T)
│   ├── matrix.hpp                    # Dense matrix (works with any T)
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
│   ├── tracking_allocator.hpp        # Memory debugging
│   └── ad_allocator.hpp              # Specialized for AD types
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
│   │   ├── ell.hpp                  # ELLPACK format
│   │   └── pattern.hpp              # Sparsity patterns
│   ├── operations/
│   │   ├── sparse_blas.hpp          # Sparse BLAS
│   │   ├── sparse_arithmetic.hpp    # Arithmetic ops
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
│   ├── variable_block.hpp           # Variable-sized blocks
│   └── nested_blocks.hpp            # Hierarchical block structures
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
│   ├── aliasing.hpp                 # Aliasing detection
│   └── evaluation.hpp               # Evaluation strategies
│
├── operations/                      # Mathematical operations
│   ├── arithmetic.hpp               # Basic arithmetic
│   ├── transcendental.hpp           # sin, cos, exp, log
│   ├── tensor_contraction.hpp       # Tensor contractions
│   ├── kronecker_product.hpp        # Kronecker products
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
│   ├── newton/                      # Newton-type nonlinear solvers
│   │   ├── newton_raphson.hpp       # Newton-Raphson
│   │   ├── line_search.hpp          # Line search strategies
│   │   ├── trust_region.hpp         # Trust region methods
│   │   ├── quasi_newton.hpp         # BFGS, L-BFGS
│   │   └── continuation.hpp         # Parameter continuation
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
├── indexing/                        # Indexing and slicing
│   ├── index.hpp                    # Basic indexing
│   ├── slice.hpp                    # Slice objects
│   ├── fancy_indexing.hpp           # Advanced indexing
│   ├── block_indexing.hpp           # Block indexing
│   └── multi_index.hpp              # Multi-dimensional
│
├── optimization/                    # Optimization algorithms
│   ├── gradient_descent.hpp         # First-order methods
│   ├── conjugate_gradient_opt.hpp   # Nonlinear CG
│   ├── lbfgs.hpp                    # Limited-memory BFGS
│   ├── gauss_newton.hpp             # For least squares
│   ├── levenberg_marquardt.hpp      # LM algorithm
│   ├── interior_point.hpp           # Constrained optimization
│   └── sensitivity_analysis.hpp     # Design sensitivities
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
│   └── error_handling.hpp           # Error handling
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
│   ├── benchmark/                   # Benchmark tests
│   └── performance/                 # Performance tests
│
└── examples/                        # Usage examples
    ├── basic_usage.cpp              # Basic operations
    ├── fem_assembly.cpp             # FEM assembly example
    ├── block_systems.cpp            # Block system solve
    ├── constrained_systems.cpp      # Constraints example
    ├── matrix_free.cpp              # Matrix-free example
    └── autodiff_examples.cpp        # AD usage patterns
```

## Key Design Features

### 1. Compositional Automatic Differentiation
```cpp
// Element types determine container behavior
using ScalarType = double;
using ForwardAD = Dual<double, 3>;      // 3 directional derivatives
using ReverseAD = Var<double>;          // Tape-based AD
using SecondOrder = HyperDual<double>;  // Second derivatives

// Containers work with any numeric type
Vector<ScalarType> x;        // Regular vector
Vector<ForwardAD> dx;        // Vector with derivatives
Matrix<ReverseAD> J;         // Matrix for reverse-mode
Tensor<SecondOrder> H;       // Tensor with Hessian info

// Mixed types in block systems
BlockMatrix<double> K;                    // Regular blocks
BlockMatrix<Dual<double, 2>> K_sensitive; // With sensitivities
```

### 2. Block-Structured Systems
```cpp
// Native support for multi-field problems
BlockMatrix<double> K;  // System matrix
K.set_block_sizes({3, 3, 1});  // u, v, w, p for Stokes

// Block operations work with any element type
auto K_uu = K.block(0, 0);  // Velocity block
auto K_up = K.block(0, 3);  // Velocity-pressure coupling

// AD-aware block operations
BlockMatrix<Dual<double, 2>> K_ad;
auto dK_uu = K_ad.block(0, 0);  // Derivatives propagate
```

### 3. Constraint Handling
```cpp
// Eliminate Dirichlet constraints
ConstraintHandler constraints;
constraints.add_dirichlet(node_id, dof, value);
constraints.add_periodic(node1, node2);

// Works with AD types
Matrix<Var<double>> K_ad;
Vector<Var<double>> F_ad;
constraints.eliminate(K_ad, F_ad);  // Preserves derivatives

// Lagrange multipliers for constraints
LagrangeSystem<double> sys(K, C);  // K*u + C'*λ = f, C*u = g
sys.solve(u, lambda, f, g);
```

### 4. Incremental Assembly with AD Support
```cpp
// Thread-safe assembly with any element type
template<typename T>
void assemble_system(ConcurrentBuilder<T>& builder) {
    GraphColoring coloring(mesh.connectivity());
    
    parallel_for(coloring.color(c), [&](int elem) {
        // Element matrix can be AD type
        SmallMatrix<T, 8, 8> K_e;
        compute_element_matrix(elem, K_e);
        builder.add_atomic(elem.dofs(), K_e);
    });
}

// Use with different element types
ConcurrentBuilder<double> builder_regular(pattern);
ConcurrentBuilder<Dual<double, 3>> builder_ad(pattern);
```

### 5. Matrix-Free Operations
```cpp
// Matrix-free works with any numeric type
template<typename T, typename Physics>
class MatrixFreeOperator {
    void apply(const Vector<T>& x, Vector<T>& y) {
        parallel_for(elements, [&](int e) {
            auto y_local = sum_factorization.apply(
                physics.compute_action(e), x.local(e)
            );
            y.add_local(e, y_local);
        });
    }
};

// Use with AD for matrix-free Jacobian-vector products
MatrixFreeOperator<Var<double>, Elasticity> A;
gmres.solve(A, b, x);  // Derivatives propagate through solve
```

### 6. Type-Safe AD Operations
```cpp
// Operations dispatch based on element type
template<typename T>
T compute_energy(const Vector<T>& x) {
    T energy = T(0);
    for (size_t i = 0; i < x.size(); ++i) {
        energy += x[i] * x[i];  // Works for double or AD types
    }
    return energy;
}

// Automatic derivative computation
Vector<Dual<double, 1>> x_ad;
seed_derivative(x_ad, 0, 1.0);  // Set derivative direction
auto energy_ad = compute_energy(x_ad);
double dE_dx0 = energy_ad.derivative(0);
```

### 7. Specialized Linear Algebra for AD
```cpp
// LU decomposition preserves derivatives
Matrix<Dual<double, 2>> A;
LUDecomposition<Dual<double, 2>> lu(A);
Vector<Dual<double, 2>> b, x;
lu.solve(b, x);  // x has derivatives w.r.t. parameters

// Eigenvalue derivatives
Matrix<Var<double>> K, M;
GeneralizedEigenSolver<Var<double>> solver(K, M);
auto [eigenvalues, eigenvectors] = solver.solve();
// eigenvalues contain derivatives w.r.t. design parameters
```

### 8. Newton Solvers with Automatic Jacobians
```cpp
// Newton-Raphson with AD-computed Jacobian
template<typename Residual>
class NewtonSolver {
    void solve(Vector<double>& x) {
        while (!converged) {
            // Use AD for Jacobian
            Vector<Var<double>> x_ad = tape_vector(x);
            Vector<Var<double>> r_ad = residual(x_ad);
            
            // Extract Jacobian from tape
            auto J = extract_jacobian(r_ad, x_ad);
            
            // Solve linear system
            Vector<double> dx;
            J.solve(-r_ad.value(), dx);
            x += dx;
        }
    }
};
```

## Implementation Priorities

### Phase 1: Core AD Types (Weeks 1-2)
1. Implement `Dual<T, N>` for forward-mode AD
2. Implement `Var<T>` for reverse-mode AD
3. Basic arithmetic operations for AD types
4. Integration with existing containers

### Phase 2: FEM Essentials (Weeks 3-4)
1. Block matrix/vector structures
2. Sparse matrix assembly infrastructure
3. Constraint elimination
4. Basic graph algorithms

### Phase 3: AD-Aware Operations (Weeks 5-6)
1. Transcendental functions for AD types
2. Linear algebra operations with AD
3. Norm computations with derivatives
4. Matrix decompositions preserving derivatives

### Phase 4: Assembly Support (Weeks 7-8)
1. Concurrent assembly with atomics
2. Graph coloring
3. Assembly caching
4. DOF mapping utilities

### Phase 5: Solvers (Weeks 9-10)
1. Block solvers and preconditioners
2. Constrained system solvers
3. Newton solvers with automatic Jacobians
4. Generalized eigensolvers

### Phase 6: Advanced Features (Weeks 11-12)
1. Matrix-free infrastructure
2. Polynomial/quadrature support
3. Tape optimization for reverse-mode
4. Performance optimization

## Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Element Matrix Assembly | < 100ns per 8×8 | Stack allocation, no malloc |
| AD Overhead (Forward) | < 3x scalar | Inline operations |
| AD Overhead (Reverse) | < 5x scalar | Tape management |
| Sparse Matrix Insert | < 20ns atomic | Lock-free assembly |
| Block Matrix Access | O(1) | Direct block indexing |
| Constraint Elimination | < 5% overhead | Efficient row/col removal |
| Graph Coloring | O(V + E) | Linear time coloring |
| Matrix-Free SpMV | > 20% peak FLOPS | Sum-factorization |

## Memory Model

### Type-Aware Storage
1. **Scalar types**: Standard allocation
2. **AD types**: Aligned allocation for derivative arrays
3. **Small matrices**: Stack allocation (< 32×32)
4. **Sparse matrices**: Compressed formats
5. **Block matrices**: Hierarchical storage

### AD-Specific Memory Management
- **Forward-mode**: Derivative arrays co-located with values
- **Reverse-mode**: Tape nodes in memory pool
- **Mixed-mode**: Nested tape structures
- **Checkpointing**: Selective tape recording

## Usage Patterns

### Basic FEM with Sensitivities
```cpp
// Define element type with 2 design parameters
using ADType = Dual<double, 2>;

// Setup system
Matrix<ADType> K;
Vector<ADType> F, u;

// Assemble with sensitivities
for (auto& elem : mesh) {
    SmallMatrix<ADType, 8, 8> K_e;
    compute_element_matrix(elem, design_params, K_e);
    K.add(elem.dofs(), K_e);
}

// Solve preserves derivatives
solve(K, F, u);

// Extract sensitivities
for (int i = 0; i < 2; ++i) {
    Vector<double> du_dp = extract_derivative(u, i);
}
```

### Nonlinear FEM with Automatic Jacobian
```cpp
// Residual function
auto residual = [&](const Vector<Var<double>>& u) {
    return compute_internal_forces(u) - F_ext;
};

// Newton solver with AD
NewtonSolver solver;
solver.set_residual(residual);
solver.solve(u);  // Jacobian computed automatically
```

### Multi-Physics with Block Structure
```cpp
// Different AD types for different physics
using MechAD = Dual<double, 3>;   // 3 mechanical parameters
using ThermalAD = Dual<double, 2>; // 2 thermal parameters

BlockMatrix<double> K;
K.set_blocks({
    {"displacement", Matrix<MechAD>()},
    {"temperature", Matrix<ThermalAD>()}
});

// Assemble and solve preserving sensitivities
```

## Testing Strategy

### Unit Testing Focus
1. **AD Types**: Correctness of derivatives
2. **Container Compatibility**: All containers work with AD
3. **Operation Dispatch**: Correct selection of AD operations
4. **Memory Safety**: No leaks with tape management
5. **Parallel Safety**: Thread-safe AD operations

### Integration Testing
1. **FEM Assembly**: With and without AD
2. **Solver Convergence**: AD doesn't break solvers
3. **Sensitivity Accuracy**: Validate against finite differences
4. **Performance**: AD overhead within targets

## Future Extensions

1. **GPU Support**: CUDA kernels for AD operations
2. **Distributed AD**: MPI-aware tape management
3. **Higher-Order AD**: Third and fourth derivatives
4. **Symbolic AD**: Integration with symbolic systems
5. **Adjoint Methods**: Specialized reverse-mode for PDEs

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| AD Integration | Element types, not containers | Composability, less duplication |
| Container Design | Template on element type | Works with any numeric type |
| Operation Dispatch | Compile-time (if constexpr) | Zero overhead for non-AD |
| Tape Management | Thread-local by default | Parallel safety |
| Memory Layout | Derivatives with values | Cache efficiency |
| Type Promotion | Explicit only | Type safety |
| Special Functions | AD-aware versions | Correct derivatives |

This revised architecture provides automatic differentiation as a composable feature through element types, maintaining clean separation of concerns while enabling sophisticated sensitivity analysis and optimization capabilities for FEM applications.
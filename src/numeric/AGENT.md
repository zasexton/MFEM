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
├── README.md                          # Library overview
├── AGENT.md                          # This document
├── CMakeLists.txt                    # Build configuration
│
├── cmake/                            # CMake configuration files
│   ├── FEMNumericConfig.cmake.in     # Package config template
│   ├── fem_numeric.pc.in             # pkg-config template
│   ├── CodeCoverage.cmake            # Coverage configuration
│   └── FetchTBB.cmake                # TBB dependency
│
├── scripts/                          # Development and build scripts
│   ├── install-dev-tools.sh          # Install development dependencies
│   ├── coverage.sh                   # Run coverage analysis
│   └── detailed_line_coverage.sh     # Detailed line-by-line coverage
│
├── include/                          # Public header files
│   ├── numeric.h                     # Single header include (planned)
│   │
│   ├── config/                       # Configuration and platform detection (planned)
│   │   ├── config.h                  # Main configuration
│   │   ├── compiler.h                # Compiler detection
│   │   ├── platform.h                # Platform specifics
│   │   ├── features.h                # Feature flags
│   │   ├── precision.h               # Floating-point settings
│   │   └── debug.h                   # Debug/assertion macros
│   │
│   ├── base/                         # Base infrastructure (IN PROGRESS)
│   │   ├── AGENT.md                  # Base module documentation
│   │   ├── README.md                 # Base module overview
│   │   ├── numeric_base.h            # Foundation types ✓
│   │   ├── container_base.h          # Container CRTP base ✓
│   │   ├── storage_base.h            # Storage abstractions ✓
│   │   ├── expression_base.h         # Expression templates ✓
│   │   ├── iterator_base.h           # Iterator interfaces ✓
│   │   ├── traits_base.h             # Base traits ✓
│   │   ├── allocator_base.h          # Allocator interfaces ✓
│   │   ├── view_base.h               # View interfaces ✓
│   │   ├── slice_base.h              # Slicing operations ✓
│   │   ├── ops_base.h                # Operation functors ✓
│   │   ├── broadcast_base.h          # Broadcasting support ✓
│   │   ├── dual_base.h               # Base for dual number types ✓
│   │   ├── dual_math.h               # Math operations for duals ✓
│   │   └── dual_comparison.h         # Comparison operations for duals ✓
│   │
│   ├── traits/                       # Type traits and concepts (IN PROGRESS)
│   │   ├── AGENT.md                  # Traits module documentation
│   │   ├── README.md                 # Traits module overview
│   │   ├── type_traits.h             # Basic type traits ✓
│   │   ├── numeric_traits.h          # Numeric properties ✓
│   │   ├── container_traits.h        # Container detection ✓
│   │   ├── expression_traits.h       # Expression traits ✓
│   │   ├── operation_traits.h        # Operation compatibility ✓
│   │   ├── storage_traits.h          # Storage characteristics ✓
│   │   ├── iterator_traits.h         # Iterator properties ✓
│   │   ├── ad_traits.h               # AD type traits ✓
│   │   ├── concepts.h                # C++20 concepts ✓
│   │   └── SFINAE.h                  # SFINAE utilities ✓
│   │
│   ├── autodiff/                     # Automatic differentiation (PLANNED)
│   │   ├── types/                    # AD numeric types
│   │   │   ├── dual.h                # Dual<T, N> for forward-mode
│   │   │   ├── hyperdual.h           # HyperDual<T> for second derivatives
│   │   │   ├── var.h                 # Var<T> for reverse-mode
│   │   │   ├── complex_dual.h        # ComplexDual<T, N>
│   │   │   └── mixed_dual.h          # Mixed strategies
│   │   ├── operations/               # AD-aware operations
│   │   ├── algebra/                  # Linear algebra specializations
│   │   ├── tape/                     # Reverse-mode infrastructure
│   │   └── utilities/                # AD support utilities
│   │
│   ├── core/                         # Core mathematical objects (PLANNED)
│   │   ├── vector.h                  # Dense vector
│   │   ├── matrix.h                  # Dense matrix
│   │   ├── tensor.h                  # N-dimensional tensor
│   │   ├── block_vector.h            # Block vectors
│   │   ├── block_matrix.h            # Block matrices
│   │   ├── small_matrix.h            # Small matrix optimizations
│   │   ├── sparse_vector.h           # Sparse vector
│   │   ├── sparse_matrix.h           # Sparse matrix base
│   │   └── sparse_tensor.h           # Sparse tensor
│   │
│   ├── storage/                      # Storage implementations (PLANNED)
│   │   ├── dense_storage.h           # Contiguous memory
│   │   ├── sparse_storage.h          # Sparse formats base
│   │   ├── block_storage.h           # Block storage
│   │   ├── small_storage.h           # SSO for small matrices
│   │   ├── compressed_storage.h      # CSR, CSC, COO
│   │   ├── strided_storage.h         # Strided views
│   │   ├── static_storage.h          # Compile-time sized
│   │   ├── dynamic_storage.h         # Runtime sized
│   │   ├── aligned_storage.h         # SIMD alignment
│   │   └── hybrid_storage.h          # Small-buffer optimization
│   │
│   ├── allocators/                   # Memory allocators (PLANNED)
│   │   ├── aligned_allocator.h       # Aligned allocation
│   │   ├── pool_allocator.h          # Pool-based allocation
│   │   ├── small_matrix_pool.h       # Element matrix pool
│   │   ├── arena_allocator.h         # Arena allocation
│   │   ├── stack_allocator.h         # Stack allocation
│   │   ├── tracking_allocator.h      # Memory debugging
│   │   └── ad_allocator.h            # Specialized for AD types
│   │
│   ├── sparse/                       # Sparse matrix support (PLANNED)
│   │   ├── formats/
│   │   ├── operations/
│   │   └── builders/
│   │
│   ├── block/                        # Block-structured operations (PLANNED)
│   │   ├── block_operations.h
│   │   ├── block_extraction.h
│   │   ├── block_assembly.h
│   │   ├── block_preconditioners.h
│   │   └── block_solvers.h
│   │
│   ├── constrained/                  # Constrained systems (PLANNED)
│   │   ├── constraint_handler.h
│   │   ├── lagrange_system.h
│   │   ├── penalty_method.h
│   │   └── schur_complement.h
│   │
│   ├── graph/                        # Graph algorithms (PLANNED)
│   │   ├── adjacency.h
│   │   ├── coloring.h
│   │   ├── reordering.h
│   │   └── partitioning.h
│   │
│   ├── polynomial/                   # Polynomial operations (PLANNED)
│   │   ├── quadrature.h
│   │   ├── legendre.h
│   │   ├── chebyshev.h
│   │   └── interpolation.h
│   │
│   ├── expressions/                  # Expression templates (PLANNED)
│   │   ├── expression.h
│   │   ├── binary_ops.h
│   │   ├── unary_ops.h
│   │   └── evaluation.h
│   │
│   ├── operations/                   # Mathematical operations (PLANNED)
│   │   ├── arithmetic.h
│   │   ├── transcendental.h
│   │   ├── tensor_contraction.h
│   │   └── reductions.h
│   │
│   ├── linear_algebra/               # Linear algebra operations (PLANNED)
│   │   ├── blas_level1.h
│   │   ├── blas_level2.h
│   │   ├── blas_level3.h
│   │   └── norms.h
│   │
│   ├── decompositions/               # Matrix decompositions (PLANNED)
│   │   ├── lu.h
│   │   ├── qr.h
│   │   ├── svd.h
│   │   ├── eigen.h
│   │   └── cholesky.h
│   │
│   ├── solvers/                      # Linear system solvers (PLANNED)
│   │   ├── direct/
│   │   ├── iterative/
│   │   ├── eigen/
│   │   ├── newton/
│   │   └── preconditioners/
│   │
│   ├── matrix_free/                  # Matrix-free methods (PLANNED)
│   │   ├── operator.h
│   │   ├── sum_factorization.h
│   │   └── matrix_free_preconditioner.h
│   │
│   ├── indexing/                     # Indexing and slicing (PLANNED)
│   │   ├── index.h
│   │   ├── slice.h
│   │   └── fancy_indexing.h
│   │
│   ├── optimization/                 # Optimization algorithms (PLANNED)
│   │   ├── gradient_descent.h
│   │   ├── lbfgs.h
│   │   └── levenberg_marquardt.h
│   │
│   ├── parallel/                     # Parallelization support (PLANNED)
│   │   ├── parallel_for.h
│   │   ├── parallel_reduce.h
│   │   ├── parallel_assembly.h
│   │   └── thread_pool.h
│   │
│   ├── io/                           # Input/output (PLANNED)
│   │   ├── matrix_market.h
│   │   ├── numpy_format.h
│   │   └── hdf5_io.h
│   │
│   ├── utilities/                    # Utility functions (PLANNED)
│   │   ├── math_functions.h
│   │   ├── comparison.h
│   │   ├── timer.h
│   │   └── error_handling.h
│   │
│   └── backends/                     # External library backends (PLANNED)
│       ├── blas_backend.h
│       ├── mkl_backend.h
│       └── cuda_backend.h
│
├── tests/                            # Comprehensive testing
│   ├── CMakeLists.txt
│   ├── unit/                         # Unit tests 
│   ├── integration/                  # Integration tests (PLANNED)
│   ├── benchmark/                    # Benchmark tests (PLANNED)
│   └── performance/                  # Performance tests (PLANNED)
│
└── examples/                         # Usage examples (PLANNED)
    ├── basic_usage.cpp
    ├── fem_assembly.cpp
    ├── block_systems.cpp
    └── autodiff_examples.cpp
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
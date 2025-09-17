# AGENT.md - FEM Multi-Physics Solver Project (Revised)

## 🎯 Mission
Build a modular, extensible, high-performance multi-physics finite element method (FEM) solver that leverages modern software engineering patterns, supports multiple coupling strategies, and scales from workstations to GPU-accelerated supercomputers.

## 🏗️ Architecture Philosophy
- **Clean Separation**: Domain-agnostic `core/` vs FEM-specific modules
- **Modularity First**: Each component can function independently
- **Physics Agnostic FEM**: FEM machinery separate from specific physics
- **Composition Over Inheritance**: ECS pattern with static polymorphism in hot paths
- **Zero-Cost Abstractions**: Templates for inner loops, OOP for high-level structure
- **Standards Compliant**: IEEE 754, MPI standards, file format standards
- **GPU-First Design**: Unified memory model with CPU/GPU portability

## 📁 Project Structure

```
fem/
├── core/                    # Domain-agnostic software infrastructure
├── numeric/                 # Mathematical operations library
├── device/                  # Hardware abstraction layer
├── fem/                     # FEM-specific base classes
├── mesh/                    # Mesh generation and manipulation
├── assembly/                # System assembly and constraints
├── solvers/                 # Linear and nonlinear solvers
├── physics/                 # Physics modules
├── coupling/                # Multiphysics coupling strategies
├── analysis/                # Analysis procedures
├── adaptation/              # Adaptive strategies
├── parallel/                # MPI distributed computing
├── io_formats/              # Domain-specific file formats
├── visualization/           # Rendering and post-processing
├── tools/                   # FEM-specific utilities
external/                    # External library interfaces
tests/                       # Test infrastructure
benchmarks/                  # Performance benchmarks
examples/                    # Example problems
docs/                        # Documentation
```

## Module Descriptions

### 1. **core/** - Software Infrastructure Layer

### 2. **numeric/** - Mathematical Engine
**Purpose**: High-performance mathematical operations that remain independent of FEM-specific layers. The numeric library exposes containers, storage backends, expression templates, kernels, linear algebra, factorisations, solvers, autodiff types, indexing helpers, backends, and concurrency utilities.

**Current Sub-modules** (see `numeric/AGENT.md` for detail):

| Sub-module | Role |
|------------|------|
| `base/`, `traits/` | CRTP scaffolding, traits, view/slice infrastructure |
| `core/` | User-facing containers (vectors, matrices, tensors, block types) |
| `storage/`, `allocators/` | Concrete storage backends and memory strategies |
| `autodiff/` | Forward/reverse AD types and tape infrastructure |
| `expressions/` | User-facing expression combinators built on the base layer |
| `operations/` | Element-wise kernels and reductions |
| `linear_algebra/` | BLAS-like routines for dense and sparse data |
| `decompositions/` | LU/QR/SVD/eigen factorisations |
| `solvers/` | Reusable linear/nonlinear solver building blocks (for consumers of the numeric library) |
| `optimization/` | Optimisation algorithms on top of solvers/autodiff |
| `sparse/`, `block/`, `constrained/`, `graph/`, `polynomial/` | Specialised utilities supporting FEM assembly patterns |
| `parallel/`, `matrix_free/` | Numeric-specific threading helpers and matrix-free operators |
| `backends/` | Optional acceleration adapters (BLAS, MKL, CUDA) |
| `math/`, `diagnostics/`, `support/` | Scalar helpers, timing instrumentation, error handling |
| `indexing/`, `io/` | Fancy indexing helpers and numeric I/O formats |

The numeric library offers CPU-only fallback implementations while allowing backends to override kernels for GPU or vendor-tuned BLAS.

### 3. **device/** - Hardware Abstraction Layer (NEW)
**Purpose**: Portable execution on CPU/GPU/accelerators

**Sub-modules**:
- `backend/` - Backend implementations (CPU, CUDA, HIP, SYCL)
- `memory/` - Unified memory management with host-device transfer
- `kernel/` - Kernel launch abstractions
- `executor/` - Execution policies and spaces

**Design Pattern**:
```cpp
// Portable kernel abstraction
template<typename ExecutionSpace>
class DeviceKernel {
    // Abstracts parallel_for, reductions, etc.
};

// Memory that can live on CPU or GPU
template<typename T>
class DeviceArray : public core::Object {
    void sync_to_device();
    void sync_to_host();
    T* device_ptr();
    T* host_ptr();
};
```

### 4. **fem/** - FEM Foundation Classes
**Purpose**: Core FEM abstractions with performance optimizations

**Enhanced Sub-modules**:
- `element/` - Element base classes with static polymorphism option
- `node/` - Node and DOF management
- `material/` - Material model interfaces
- `boundary/` - Boundary condition abstractions
- `field/` - Field variable management with ghost support
- `integration/` - Gauss quadrature (precomputed for GPU)
- `shape/` - Shape functions (templated for inlining)
- `basis/` - Basis function definitions
- `constitutive/` - Constitutive models with tangent operators

**Performance-Critical Design**:
```cpp
// Static polymorphism for inner loops
template<typename ShapeFunction>
class ElementKernel {
    // Inlined shape function evaluation
    static void compute_residual(const Element& elem) {
        // No virtual calls in hot path
    }
};

// ECS for flexibility at higher level
class Element : public core::Entity {
    // Components for physics coupling
};
```

### 5. **mesh/** - Mesh Management
**Purpose**: Static and dynamic mesh handling

**Enhanced Sub-modules**:
- `generators/` - Mesh generation algorithms
- `topology/` - Connectivity and adjacency
- `geometry/` - Geometric operations
- `motion/` - **ALE and mesh deformation** (NEW)
  - `ale/` - Arbitrary Lagrangian-Eulerian formulation
  - `morphing/` - Mesh morphing algorithms
  - `smoothing/` - Mesh smoothing for quality
  - `remeshing/` - Dynamic remeshing strategies
- `refinement/` - h-, p-, hp-refinement
- `partitioning/` - Domain decomposition
- `quality/` - Quality metrics and improvement
- `interpolation/` - Inter-mesh transfer operators

**Mesh Motion Support**:
```cpp
class MovingMesh : public Mesh {
    // ALE mesh velocity
    Field mesh_velocity;
    
    // Update mesh for FSI
    void update_coordinates(const Field& displacement);
    void compute_mesh_velocity(double dt);
    
    // Quality-preserving motion
    void smooth_mesh();
    bool check_element_validity();
};
```

### 6. **assembly/** - System Assembly
**Purpose**: Efficient assembly with caching and GPU support

**Enhanced Sub-modules**:
- `local/` - Element computations (GPU-capable)
- `global/` - Global assembly with atomics for GPU
- `pattern/` - Sparsity pattern optimization
- `constraints/` - Constraint handling
- `cache/` - Assembly caching and reuse
- `matrix_free/` - Matrix-free operators for GPU
- `coloring/` - Graph coloring for parallel assembly

**GPU Assembly Strategy**:
```cpp
// Matrix-free assembly for GPU
template<typename Physics>
class MatrixFreeOperator {
    void apply(const DeviceVector& x, DeviceVector& y) {
        // Compute y = A*x without storing A
        device::parallel_for(elements, [=] __device__ (int e) {
            // Element-local computation
        });
    }
};
```

### 7. **solvers/** - FEM Solution Algorithms
**Purpose**: FEM-level solvers and orchestration (linear/nonlinear/transient) that combine assemblies, constraints, boundary conditions, and physics models. While the numeric library exposes reusable solver building blocks, this module integrates them with FEM-specific data structures and external library interfaces.

**Enhanced Sub-modules**:
- `linear/`
  - `direct/` - Direct solver wrappers
  - `iterative/` - Krylov methods
  - `preconditioners/` - Including block and field-split
  - `multigrid/` - Geometric and algebraic MG
- `nonlinear/`
  - `newton/` - With line search and trust region
  - `anderson/` - Anderson acceleration for fixed-point
  - `continuation/` - Arc-length methods
- `eigen/` - Eigenvalue solvers
- `transient/` - Time integration with adaptivity
  - `adaptive/` - **Adaptive time stepping** (NEW)
  - `error_estimation/` - Local truncation error
  - `step_control/` - Step size selection

**Adaptive Time Integration**:
```cpp
class AdaptiveTimeIntegrator : public TimeIntegrator {
    // Error-based step control
    double compute_error_estimate(const State& y);
    double select_timestep(double error, double tol);
    
    // Embedded RK or BDF with error estimation
    void advance_with_error_control(State& y, double& dt);
};
```

### 8. **coupling/** - Multiphysics Coupling Strategies (NEW)
**Purpose**: Flexible coupling with multiple strategies

**Sub-modules**:
- `monolithic/` - Fully coupled system assembly
- `partitioned/` - Staggered iteration schemes
  - `fixed_point/` - Fixed-point iterations
  - `aitken/` - Aitken relaxation
  - `quasi_newton/` - Interface quasi-Newton
- `interface/` - Interface physics and conditions
- `field_transfer/` - Conservative field mapping
- `convergence/` - Coupling convergence monitors

**Coupling Patterns**:
```cpp
// Monolithic coupling
class MonolithicCoupler : public Coupler {
    void assemble_coupled_system(Matrix& K, Vector& F) {
        // Single system with all physics
        for (auto& physics : physics_modules) {
            physics->add_contribution(K, F);
        }
    }
};

// Partitioned coupling with relaxation
class PartitionedCoupler : public Coupler {
    void solve_staggered() {
        do {
            solve_fluid();
            transfer_interface_loads();
            solve_structure();
            transfer_interface_displacement();
            apply_relaxation();  // Aitken or IQN
        } while (!converged());
    }
};
```

### 9. **physics/** - Physics Modules
**Purpose**: Component-based physics with coupling support

**Enhanced Design**:
```cpp
// Physics module base with coupling interface
class PhysicsModule : public core::Component {
    // Standard interface for all physics
    virtual void compute_residual(Element& elem) = 0;
    virtual void compute_jacobian(Element& elem) = 0;
    
    // Coupling interface
    virtual void export_coupling_fields(CouplingInterface& interface) = 0;
    virtual void import_coupling_fields(const CouplingInterface& interface) = 0;
    
    // GPU kernel option
    template<typename Device>
    void compute_residual_device(ElementBatch& batch);
};

// Coupled physics inherit and combine
class ThermoMechanical : public PhysicsModule {
    ThermalPhysics thermal;
    MechanicsPhysics mechanics;
    
    void compute_residual(Element& elem) override {
        thermal.compute_residual(elem);
        mechanics.compute_residual(elem);
        // Add coupling terms
    }
};
```

### 10. **analysis/** - Analysis Procedures
**Purpose**: Orchestration with adaptive loops

**Enhanced Sub-modules**:
- `static/` - Static analysis
- `dynamic/` - Dynamic with adaptive time stepping
- `continuation/` - Path-following methods
- `optimization/` - Shape and topology optimization
- `sensitivity/` - Gradient computation
- `uncertainty/` - UQ with sampling
- `orchestration/` - **Complex solve sequences** (NEW)
  - `adaptive_loops/` - Solve-estimate-refine
  - `multiphysics_loops/` - Coupling iterations
  - `checkpoint/` - State saving/restart

**Orchestration Framework**:
```cpp
class AnalysisOrchestrator : public core::StateMachine {
    // Composable analysis building blocks
    void add_phase(std::unique_ptr<AnalysisPhase> phase);
    
    // Adaptive solve-refine loop
    void solve_adaptive() {
        do {
            solve();
            estimate_error();
            if (error > tol) {
                refine_mesh();
                transfer_solution();
            }
        } while (error > tol);
    }
    
    // Multiphysics with subcycling
    void solve_coupled_transient() {
        while (time < t_final) {
            // Fast physics with small dt
            for (int i = 0; i < n_subcycles; ++i) {
                advance_fluid(dt_fine);
            }
            // Slow physics with large dt
            advance_structure(dt_coarse);
            exchange_coupling_data();
        }
    }
};
```

### 11. **adaptation/** - Adaptive Strategies (NEW)
**Purpose**: Error estimation and adaptivity control

**Sub-modules**:
- `error_estimators/`
  - `recovery/` - ZZ and SPR recovery
  - `residual/` - Residual-based estimators
  - `goal_oriented/` - Dual-weighted residual
  - `hierarchical/` - Hierarchical estimators
- `refinement_strategies/`
  - `isotropic/` - Uniform refinement
  - `anisotropic/` - Directional refinement
  - `optimization_based/` - Optimal mesh distribution
- `markers/` - Element marking strategies
- `constraints/` - Refinement constraints

**Error-Driven Adaptivity**:
```cpp
class ErrorEstimator : public core::Component {
    virtual void estimate(const Solution& u, 
                         ElementErrors& errors) = 0;
};

class AdaptivityController {
    void adaptive_solve(Analysis& analysis) {
        while (!converged) {
            solution = analysis.solve();
            errors = estimator->estimate(solution);
            
            if (max(errors) > tol) {
                mesh->mark_for_refinement(errors);
                mesh->refine();
                solution.transfer_to_new_mesh();
            }
        }
    }
};
```

### 12. **parallel/** - MPI Distributed Computing
**Enhanced with GPU-aware MPI**:
- `mpi_wrapper/` - MPI abstraction
- `gpu_aware/` - GPU-direct communication
- `decomposition/` - Partitioning strategies
- `communication/` - Ghost exchange patterns
- `load_balancing/` - Dynamic balancing with costs
- `collective/` - Optimized collectives

> Layering note: this module owns inter-node (MPI) concerns. Intra-node threading lives in `numeric/parallel/`, while accelerator execution policies are handled by `device/`.

**GPU-Aware Communication**:
```cpp
class GPUAwareCommunicator : public Communicator {
    void exchange_ghosts(DeviceVector& field) {
        // Direct GPU-to-GPU transfer via MPI
        MPI_Sendrecv(field.device_ptr(), ...);
    }
};
```

### 13. **external/** - External Library Interfaces (NEW)
**Purpose**: Leverage optimized external solvers

**Sub-modules**:
- `petsc/` - PETSc interface wrapper
- `trilinos/` - Trilinos/Tpetra interface
- `hypre/` - Hypre AMG interface
- `suitesparse/` - Direct solver interface
- `cuda_libs/` - cuSPARSE, cuSOLVER wrappers
- `mkl/` - Intel MKL interface

**External Solver Integration**:
```cpp
// Abstract interface
class IExternalSolver : public ISolver {
    virtual void solve(const Matrix& A, 
                      const Vector& b, 
                      Vector& x) = 0;
};

// Concrete wrapper
class PETScSolver : public IExternalSolver {
    void solve(const Matrix& A, 
              const Vector& b, 
              Vector& x) override {
        // Convert to PETSc format
        // Call PETSc solver
        // Convert back
    }
};
```

### 14. **io_formats/** - Domain-Specific I/O
[Unchanged from previous version]

### 15. **visualization/** - Post-Processing
**Enhanced with In-Situ Support**:
- `render/` - Real-time rendering
- `insitu/` - In-situ visualization (Catalyst, ADIOS)
- `contour/` - Isosurface generation
- `statistics/` - Statistical analysis
- `cinema/` - Cinema database generation

### 16. **tools/** - FEM-Specific Utilities
[Unchanged from previous version]

### 17. **tests/** - Testing Infrastructure
**Enhanced Testing**:
```
tests/
├── unit/           # Unit tests with GPU tests
├── integration/    # Cross-module tests
├── regression/     # Regression suite
├── verification/   # MMS and manufactured solutions
├── validation/     # Physical validation
├── parallel/       # MPI and GPU-MPI tests
├── performance/    # Scaling studies
└── convergence/    # Convergence rate verification
```

## Module Interactions

### Layered Architecture with Device Abstraction
```
┌─────────────────────────────────────────────────────┐
│                    Applications                      │
├─────────────────────────────────────────────────────┤
│      analysis/      visualization/    coupling/      │
├─────────────────────────────────────────────────────┤
│   physics/    solvers/    adaptation/    parallel/  │
├─────────────────────────────────────────────────────┤
│      fem/        mesh/        assembly/             │
├─────────────────────────────────────────────────────┤
│            numeric/            external/            │
├─────────────────────────────────────────────────────┤
│                    device/                          │
├─────────────────────────────────────────────────────┤
│    core/ (domain-agnostic infrastructure)           │
└─────────────────────────────────────────────────────┘
```

## Design Patterns & Performance

### Hybrid OOP-Template Design
```cpp
// High-level: OOP with ECS for flexibility
class Analysis : public core::Entity {
    // Components, observers, etc.
};

// Mid-level: CRTP for static polymorphism
template<typename Derived>
class ElementBase {
    void compute() {
        static_cast<Derived*>(this)->compute_impl();
    }
};

// Low-level: Templates for zero-cost abstraction
template<int Dim, int Order>
class ShapeFunction {
    // Compile-time optimized
    constexpr static double evaluate(int i, double xi) {
        // Fully inlined
    }
};
```

### GPU Execution Strategy
```cpp
// Assembly on GPU
assembly::parallel_assemble_gpu(mesh, [=] __device__ (Element e) {
    // Element kernel executes on GPU
    LocalMatrix K_e = physics.compute_element_matrix(e);
    assembly.add_atomic(K_e);  // Thread-safe assembly
});

// Solve on GPU
solver.solve_gpu(K_gpu, F_gpu, u_gpu);

// Post-process on CPU if needed
u_gpu.copy_to_host(u_cpu);
```

## Build Configuration

### CMake with Device Selection
```cmake
# Device backend selection
option(FEM_BACKEND_CPU "Enable CPU backend" ON)
option(FEM_BACKEND_CUDA "Enable CUDA backend" OFF)
option(FEM_BACKEND_HIP "Enable HIP backend" OFF)
option(FEM_BACKEND_SYCL "Enable SYCL backend" OFF)

# External solver interfaces
option(FEM_USE_PETSC "Use PETSc solvers" OFF)
option(FEM_USE_TRILINOS "Use Trilinos solvers" OFF)
option(FEM_USE_HYPRE "Use Hypre AMG" OFF)

# Optimization levels
option(FEM_OPTIMIZE_ASSEMBLY "Assembly optimizations" ON)
option(FEM_MATRIX_FREE "Enable matrix-free" OFF)
```

## Performance Targets

| Operation | CPU Target | GPU Target |
|-----------|------------|------------|
| Assembly | < 10% time | < 5% time |
| Linear Solve | < 60% time | < 40% time |
| Field Transfer | < 5% time | < 2% time |
| MPI Communication | < 15% time | < 10% time |
| Parallel Efficiency | > 85% @ 1000 cores | > 75% @ 100 GPUs |
| Matrix-Free SpMV | - | > 100 GFLOPS/GPU |

## 🔑 Key Improvements from Review

### 1. **Adaptive Capabilities**
- Added `adaptation/` module for error estimation
- Integrated adaptive time stepping in solvers
- Orchestration for solve-adapt-refine loops

### 2. **Mesh Motion & ALE**
- Added `mesh/motion/` for ALE formulations
- Support for FSI with moving boundaries
- Mesh quality preservation during motion

### 3. **GPU Strategy**
- Added `device/` abstraction layer
- Matrix-free assembly for GPU efficiency
- GPU-aware MPI communication
- External GPU solver integration

### 4. **Coupling Clarification**
- Dedicated `coupling/` module
- Support for monolithic and partitioned
- Relaxation and convergence acceleration
- Clear interfaces for physics coupling

### 5. **Solver Orchestration**
- Enhanced `analysis/orchestration/`
- Composable analysis phases
- Support for complex workflows
- Subcycling and multi-rate integration

### 6. **External Libraries**
- Added `external/` for solver integration
- Wrappers for PETSc, Trilinos, Hypre
- Leverage proven HPC solvers
- GPU solver library support

### 7. **Performance Optimization**
- Hybrid OOP-template design
- Static polymorphism in hot paths
- Matrix-free operators
- Graph coloring for parallelism

## Success Metrics (Updated)

1. **Modularity**: Each module compiles independently
2. **Performance**:
  - CPU: Match commercial codes
  - GPU: > 5x speedup on assembly/solve
3. **Scalability**:
  - CPU: Linear to 10,000 cores
  - GPU: Linear to 100+ GPUs
4. **Adaptivity**: Automatic error control to user tolerance
5. **Coupling**: Both monolithic and partitioned with < 10 iterations
6. **Extensibility**: New physics in < 200 lines
7. **Reliability**: > 95% test coverage
8. **Portability**: Single code for CPU/GPU

## Implementation Roadmap (Revised)

### Phase 1: Foundation + Device Layer (Months 1-3)
- Complete device abstraction layer
- Basic CPU/GPU numeric operations
- Simple mesh generation
- Unit test framework with GPU tests

### Phase 2: FEM Core + GPU Assembly (Months 4-6)
- FEM base classes with templates
- GPU-capable assembly
- Direct solvers via external libs
- Static analysis
- Verification tests

### Phase 3: Adaptive Capabilities (Months 7-9)
- Error estimators
- Mesh refinement with solution transfer
- Adaptive time stepping
- ALE mesh motion
- Validation problems

### Phase 4: Multiphysics (Months 10-12)
- Coupling strategies (monolithic/partitioned)
- Interface physics
- FSI demonstration
- Thermo-mechanical coupling
- Parallel coupling

### Phase 5: Production & Optimization (Months 13-15)
- Matrix-free GPU methods
- Performance optimization
- Industrial validation
- Documentation and tutorials
- Python bindings

### Phase 6: Advanced Features (Months 16+)
- Multi-GPU support
- In-situ visualization
- Uncertainty quantification
- Topology optimization
- Cloud deployment

## Critical Implementation Notes

### Memory Management Strategy
```cpp
// Unified memory abstraction
template<typename T>
class UnifiedMemory {
    T* cpu_ptr = nullptr;
    T* gpu_ptr = nullptr;
    bool cpu_dirty = false;
    bool gpu_dirty = false;
    
    void sync_to_gpu() {
        if (cpu_dirty) {
            cudaMemcpy(gpu_ptr, cpu_ptr, ...);
            cpu_dirty = false;
        }
    }
};
```

### Avoiding Virtual Call Overhead
```cpp
// Use variant for compile-time dispatch
using ElementVariant = std::variant<Hex8, Hex20, Tet4, Tet10>;

// Visitor pattern with inlining
std::visit([&](auto& elem) {
    // Compile-time type known, can inline
    elem.compute_residual();
}, element_variant);
```

### Testing Strategy
- **Unit Tests**: Every class, including GPU kernels
- **Verification**: Method of manufactured solutions
- **Validation**: Physical benchmarks (cavity, cylinder flow)
- **Performance**: Scaling studies with profiling
- **Regression**: Automated nightly builds

This revised architecture addresses all major points from the review while maintaining the ECS design philosophy and adding concrete strategies for GPU support, adaptive capabilities, and multiphysics coupling.
### 14. **device/** - Hardware Abstraction Layer (NEW)
**Purpose**: Provide execution-space abstractions (CPU, CUDA, HIP, SYCL) and unified memory management. This layer underpins hardware-specific kernels and complements the numeric `parallel/` (thread-level) and top-level `parallel/` (MPI) modules.

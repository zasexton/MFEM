# AGENT.md - Global Assembly Module

## Mission
Orchestrate the efficient assembly of global system matrices and vectors from element contributions, managing sparsity patterns, parallel assembly strategies, constraint application, and providing high-performance assembly for both CPU and GPU architectures.

## Architecture Philosophy
- **Sparsity-First**: Optimize for sparse matrix patterns from the start
- **Parallel-Native**: Conflict-free concurrent assembly via graph coloring
- **Cache-Aware**: Reuse patterns and assembly workspaces for nonlinear problems
- **Constraint-Integrated**: Seamless handling of Dirichlet, periodic, and MPC constraints
- **Matrix-Optional**: Support both matrix-explicit and matrix-free assembly

## Directory Structure

```
assembly/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── local/                           # Element-level operations
│   ├── element_assembly.hpp        # Element matrix/vector assembly
│   ├── local_matrix.hpp            # Local matrix structures
│   ├── local_vector.hpp            # Local vector structures
│   ├── integration_loop.hpp        # Quadrature loop abstractions
│   ├── kernel_interface.hpp        # Assembly kernel interface
│   ├── vectorized_assembly.hpp     # SIMD-optimized assembly
│   └── gpu_kernels/                # GPU assembly kernels
│       ├── element_kernel.cu       # CUDA element kernels
│       ├── shared_memory_opt.hpp   # Shared memory optimization
│       └── warp_assembly.hpp       # Warp-level primitives
│
├── global/                          # Global system assembly
│   ├── global_assembler.hpp        # Main assembly orchestrator
│   ├── scatter_operation.hpp       # Local-to-global scatter
│   ├── gather_operation.hpp        # Global-to-local gather
│   ├── assembly_strategy.hpp       # Assembly strategy interface
│   ├── sequential_assembly.hpp     # Sequential assembly
│   ├── parallel_assembly.hpp       # Thread-parallel assembly
│   ├── distributed_assembly.hpp    # MPI distributed assembly
│   └── gpu_assembly.hpp            # GPU global assembly
│
├── pattern/                         # Sparsity patterns
│   ├── sparsity_pattern.hpp        # Sparsity pattern base
│   ├── pattern_builder.hpp         # Pattern construction
│   ├── pattern_optimizer.hpp       # Bandwidth/fill reduction
│   ├── block_pattern.hpp           # Block-structured patterns
│   ├── symbolic_assembly.hpp       # Symbolic pattern computation
│   ├── pattern_cache.hpp           # Pattern reuse cache
│   └── adaptive_pattern.hpp        # Dynamic pattern updates
│
├── constraints/                     # Constraint handling
│   ├── constraint_manager.hpp      # Constraint orchestration
│   ├── dirichlet/
│   │   ├── dirichlet_elimination.hpp    # Row/column elimination
│   │   ├── penalty_method.hpp           # Penalty approach
│   │   └── symmetric_elimination.hpp    # Symmetric modification
│   ├── periodic/
│   │   ├── periodic_constraints.hpp     # Periodic BC handling
│   │   ├── master_slave.hpp             # Master-slave approach
│   │   └── periodic_assembly.hpp        # Direct periodic assembly
│   ├── multipoint/
│   │   ├── mpc_handler.hpp              # MPC management
│   │   ├── transformation_matrix.hpp    # Constraint transformation
│   │   └── lagrange_multiplier.hpp      # Lagrange approach
│   └── contact/
│       ├── contact_constraints.hpp      # Contact conditions
│       ├── active_set.hpp               # Active set strategy
│       └── mortar_constraints.hpp       # Mortar coupling
│
├── cache/                           # Assembly caching
│   ├── assembly_cache.hpp          # Cache management
│   ├── pattern_cache.hpp           # Sparsity pattern cache
│   ├── element_cache.hpp           # Element matrix cache
│   ├── integration_cache.hpp       # Quadrature data cache
│   ├── nonlinear_cache.hpp         # Nonlinear problem cache
│   └── memory_pool.hpp             # Memory pool for assembly
│
├── coloring/                        # Graph coloring for parallelism
│   ├── graph_coloring.hpp          # Coloring algorithms
│   ├── greedy_coloring.hpp         # Greedy algorithm
│   ├── distance_2_coloring.hpp    # Distance-2 coloring
│   ├── block_coloring.hpp          # Block-aware coloring
│   ├── gpu_coloring.hpp            # GPU-optimized coloring
│   └── conflict_detection.hpp      # Verify conflict-free
│
├── parallel/                        # Parallel assembly strategies
│   ├── thread_pool_assembly.hpp    # Thread pool approach
│   ├── task_based_assembly.hpp     # Task-based parallelism
│   ├── atomic_assembly.hpp         # Atomic operations
│   ├── reduction_assembly.hpp      # Reduction-based assembly
│   ├── numa_aware_assembly.hpp     # NUMA optimization
│   └── hybrid_assembly.hpp         # CPU-GPU hybrid
│
├── distributed/                     # MPI distributed assembly
│   ├── distributed_assembler.hpp   # Distributed orchestrator
│   ├── ghost_assembly.hpp          # Ghost element handling
│   ├── communication_pattern.hpp   # Communication optimization
│   ├── assembly_partition.hpp      # Work distribution
│   ├── collective_assembly.hpp     # MPI collective operations
│   └── overlap_assembly.hpp        # Compute-comm overlap
│
├── matrix_free/                     # Matrix-free assembly
│   ├── matrix_free_operator.hpp    # Operator interface
│   ├── action_assembly.hpp         # Matrix action assembly
│   ├── cell_batch.hpp              # Cell batching for vectorization
│   ├── sum_factorization.hpp       # Tensor-product optimization
│   ├── on_the_fly_assembly.hpp     # Compute during apply
│   └── gpu_matrix_free.hpp         # GPU matrix-free
│
├── block/                           # Block system assembly
│   ├── block_assembler.hpp         # Block matrix assembly
│   ├── field_split.hpp             # Field-based assembly
│   ├── saddle_point.hpp            # Saddle point systems
│   ├── nested_assembly.hpp         # Nested block structure
│   └── variable_block.hpp          # Variable block sizes
│
├── optimization/                    # Assembly optimization
│   ├── assembly_optimizer.hpp      # Optimization strategies
│   ├── bandwidth_reduction.hpp     # Bandwidth minimization
│   ├── fill_reduction.hpp          # Fill-in minimization
│   ├── cache_blocking.hpp          # Cache-aware blocking
│   ├── vectorization.hpp           # SIMD optimization
│   └── gpu_optimization.hpp        # GPU-specific optimizations
│
├── symbolic/                        # Symbolic operations
│   ├── symbolic_assembly.hpp       # Symbolic computation
│   ├── expression_assembly.hpp     # Expression templates
│   ├── automatic_differentiation.hpp # AD for Jacobians
│   ├── pattern_prediction.hpp      # Predict sparsity
│   └── symbolic_factorization.hpp  # Symbolic LU/Cholesky
│
├── callbacks/                       # Assembly callbacks
│   ├── assembly_callback.hpp       # Callback interface
│   ├── progress_callback.hpp       # Progress reporting
│   ├── error_callback.hpp          # Error handling
│   ├── profiling_callback.hpp      # Performance profiling
│   └── debug_callback.hpp          # Debug information
│
├── io/                             # Assembly I/O
│   ├── matrix_writer.hpp           # Export assembled matrices
│   ├── pattern_visualizer.hpp      # Visualize sparsity
│   ├── assembly_statistics.hpp     # Assembly stats
│   └── debug_output.hpp            # Debug dumps
│
├── utilities/                       # Assembly utilities
│   ├── assembly_traits.hpp         # Type traits
│   ├── assembly_timers.hpp         # Performance timing
│   ├── assembly_counters.hpp       # Operation counting
│   └── assembly_validation.hpp     # Validation checks
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── performance/                # Performance benchmarks
    ├── scaling/                    # Scaling studies
    └── validation/                 # Validation tests
```

## Key Components

### 1. Global Assembly Orchestrator
```cpp
// Main assembly coordinator
template<typename MatrixType, typename VectorType>
class GlobalAssembler {
    SparsityPattern pattern;
    AssemblyCache cache;
    ConstraintManager constraints;
    GraphColoring coloring;
    
    // Optimized assembly pipeline
    void assemble(const Mesh& mesh, 
                  const Physics& physics,
                  MatrixType& K,
                  VectorType& F) {
        // 1. Build/reuse sparsity pattern
        if (cache.pattern_valid()) {
            pattern = cache.get_pattern();
        } else {
            pattern = build_sparsity_pattern(mesh);
            optimize_pattern(pattern);
            cache.store_pattern(pattern);
        }
        
        // 2. Allocate matrix with pattern
        K.allocate(pattern);
        
        // 3. Color elements for parallel assembly
        if (!coloring.is_valid(mesh)) {
            coloring.compute(mesh);
        }
        
        // 4. Parallel assembly by color
        for (int color : coloring.colors()) {
            parallel_for(coloring.elements(color), [&](int elem_id) {
                // Thread-safe assembly for this color
                auto K_e = compute_element_matrix(elem_id, physics);
                auto F_e = compute_element_vector(elem_id, physics);
                
                // Atomic-free scatter (no conflicts in color)
                scatter_add(K, elem.dofs(), K_e);
                scatter_add(F, elem.dofs(), F_e);
            });
        }
        
        // 5. Apply constraints
        constraints.apply(K, F);
    }
};
```

### 2. Sparsity Pattern Management
```cpp
// Efficient sparsity pattern with caching
class SparsityPattern {
    // Compressed storage
    std::vector<int> row_offsets;    // CSR row pointers
    std::vector<int> col_indices;    // Column indices
    
    // Block structure if applicable
    std::optional<BlockStructure> blocks;
    
    // Bandwidth and profile
    int bandwidth;
    int64_t nnz;
    
    // Build from mesh connectivity
    static SparsityPattern build(const Mesh& mesh, 
                                const DOFMap& dofs) {
        SparsityPattern pattern;
        
        // Pre-allocate based on connectivity
        pattern.row_offsets.resize(dofs.n_dofs() + 1);
        
        // Build row by row
        std::vector<std::set<int>> rows(dofs.n_dofs());
        
        for (auto& elem : mesh.elements()) {
            auto elem_dofs = dofs.element_dofs(elem);
            
            // Add connectivity
            for (int i : elem_dofs) {
                for (int j : elem_dofs) {
                    rows[i].insert(j);
                }
            }
        }
        
        // Convert to CSR
        pattern.compress(rows);
        
        // Optimize ordering
        pattern.optimize_bandwidth();
        
        return pattern;
    }
    
    // Bandwidth optimization
    void optimize_bandwidth() {
        auto perm = compute_cuthill_mckee(col_indices, row_offsets);
        apply_permutation(perm);
    }
};
```

### 3. Parallel Assembly with Coloring
```cpp
// Conflict-free parallel assembly
class ColoredAssembly {
    GraphColoring coloring;
    std::vector<std::vector<int>> colored_elements;
    
    // Color mesh for parallel assembly
    void color_mesh(const Mesh& mesh) {
        // Build element adjacency graph
        auto graph = build_element_graph(mesh);
        
        // Apply coloring algorithm
        coloring = graph_coloring_algorithm(graph);
        
        // Group elements by color
        colored_elements.resize(coloring.n_colors());
        for (auto& elem : mesh.elements()) {
            int color = coloring.element_color(elem);
            colored_elements[color].push_back(elem.id());
        }
    }
    
    // Parallel assembly without conflicts
    template<typename AssemblyOp>
    void assemble_parallel(AssemblyOp op) {
        for (int c = 0; c < coloring.n_colors(); ++c) {
            // All elements of same color can be assembled in parallel
            #pragma omp parallel for
            for (int i = 0; i < colored_elements[c].size(); ++i) {
                op(colored_elements[c][i]);  // No write conflicts
            }
        }
    }
};
```

### 4. Constraint Application
```cpp
// Comprehensive constraint handling
class ConstraintManager {
    std::vector<DirichletBC> dirichlet_bcs;
    std::vector<PeriodicBC> periodic_bcs;
    std::vector<MultiPointConstraint> mpcs;
    
    // Apply all constraints
    void apply(Matrix& K, Vector& F) {
        // 1. Apply MPCs via transformation
        if (!mpcs.empty()) {
            auto T = build_transformation_matrix(mpcs);
            K = T.transpose() * K * T;
            F = T.transpose() * F;
        }
        
        // 2. Apply periodic constraints
        for (auto& pbc : periodic_bcs) {
            eliminate_periodic_dofs(K, F, pbc);
        }
        
        // 3. Apply Dirichlet BCs (last to override)
        for (auto& dbc : dirichlet_bcs) {
            // Symmetric elimination
            eliminate_row_column_symmetric(K, F, dbc);
        }
    }
    
    // Symmetric Dirichlet elimination
    void eliminate_row_column_symmetric(Matrix& K, Vector& F, 
                                       const DirichletBC& bc) {
        int dof = bc.dof;
        double value = bc.value;
        
        // Modify RHS for non-zero constraints
        for (auto& [row, k_val] : K.column(dof)) {
            if (row != dof) {
                F[row] -= k_val * value;
            }
        }
        
        // Zero row and column except diagonal
        K.zero_row_column(dof);
        K(dof, dof) = 1.0;
        F[dof] = value;
    }
};
```

### 5. Assembly Cache for Nonlinear Problems
```cpp
// Caching for iterative/nonlinear solvers
class AssemblyCache {
    // Sparsity pattern cache
    SparsityPattern cached_pattern;
    bool pattern_valid = false;
    
    // Element matrix cache for linear problems
    std::unordered_map<int, Matrix> element_matrices;
    
    // Integration point cache
    struct IntegrationCache {
        std::vector<double> weights;
        std::vector<Matrix> shape_gradients;
        std::vector<double> jacobians;
    };
    std::unordered_map<int, IntegrationCache> integration_data;
    
    // Memory pool for workspace
    MemoryPool<double> workspace_pool;
    
    // Reuse pattern between iterations
    void reuse_pattern(Matrix& K) {
        if (pattern_valid) {
            K.zero_entries();  // Keep structure, zero values
        } else {
            K.allocate(cached_pattern);
        }
    }
    
    // Cache element data
    void cache_element_data(int elem_id, 
                           const Matrix& K_e,
                           const IntegrationCache& int_cache) {
        if (is_linear_problem) {
            element_matrices[elem_id] = K_e;
        }
        integration_data[elem_id] = int_cache;
    }
};
```

### 6. GPU Assembly
```cpp
// GPU-accelerated assembly
class GPUAssembler {
    // GPU data structures
    DeviceMatrix d_K;
    DeviceVector d_F;
    DeviceSparsityPattern d_pattern;
    
    // Element assembly kernel
    __global__ void assemble_elements_kernel(
        const Element* elements,
        const double* coords,
        const int* dof_map,
        double* K_values,
        double* F_values,
        int n_elements) {
        
        int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (elem_id >= n_elements) return;
        
        // Shared memory for element matrix
        __shared__ double K_e[64][64];
        __shared__ double F_e[64];
        
        // Compute element matrix in parallel within block
        compute_element_matrix_gpu(elements[elem_id], 
                                  coords, K_e, F_e);
        
        // Scatter to global (atomic operations)
        scatter_atomic(K_values, F_values, 
                      dof_map, elem_id, K_e, F_e);
    }
    
    // Launch assembly
    void assemble_gpu(const Mesh& mesh, Matrix& K, Vector& F) {
        // Transfer mesh to GPU
        copy_mesh_to_device(mesh);
        
        // Launch kernel
        int threads = 256;
        int blocks = (mesh.n_elements() + threads - 1) / threads;
        assemble_elements_kernel<<<blocks, threads>>>(
            d_elements, d_coords, d_dof_map,
            d_K.values(), d_F.values(), mesh.n_elements()
        );
        
        // Copy result back
        K.copy_from_device(d_K);
        F.copy_from_device(d_F);
    }
};
```

### 7. Matrix-Free Assembly
```cpp
// Matrix-free operator assembly
template<typename Physics>
class MatrixFreeAssembler {
    const Mesh& mesh;
    const Physics& physics;
    
    // Assembly on-the-fly during matrix-vector product
    void apply(const Vector& x, Vector& y) {
        y.zero();
        
        // Cell batching for vectorization
        constexpr int batch_size = 8;
        
        #pragma omp parallel for
        for (int batch = 0; batch < n_batches; ++batch) {
            // SIMD-vectorized assembly over batch
            apply_batch<batch_size>(batch, x, y);
        }
    }
    
    // Vectorized batch assembly
    template<int BatchSize>
    void apply_batch(int batch_id, 
                    const Vector& x, Vector& y) {
        // Load batch data
        alignas(64) double x_local[BatchSize][64];
        alignas(64) double y_local[BatchSize][64];
        
        // Gather input
        gather_batch(batch_id, x, x_local);
        
        // Vectorized physics evaluation
        physics.compute_action_vectorized<BatchSize>(
            x_local, y_local
        );
        
        // Scatter result
        scatter_add_batch(batch_id, y_local, y);
    }
};
```

### 8. Block System Assembly
```cpp
// Assembly for multi-field problems
class BlockAssembler {
    struct BlockInfo {
        std::string field_name;
        int start_dof;
        int n_dofs;
    };
    std::vector<BlockInfo> blocks;
    
    // Assemble block system
    void assemble_blocks(const MultiPhysics& physics,
                        BlockMatrix& K,
                        BlockVector& F) {
        // Assemble each block
        for (int i = 0; i < blocks.size(); ++i) {
            for (int j = 0; j < blocks.size(); ++j) {
                auto K_ij = K.block(i, j);
                
                // Inter-field coupling
                if (physics.has_coupling(i, j)) {
                    assemble_coupling(physics, i, j, K_ij);
                }
                // Diagonal blocks
                else if (i == j) {
                    assemble_field(physics.field(i), K_ij);
                }
            }
            
            // RHS for field i
            assemble_rhs(physics.field(i), F.block(i));
        }
    }
};
```

## Performance Optimizations

### Memory Access Patterns
```cpp
// Cache-friendly assembly order
class CacheOptimizedAssembly {
    // Reorder elements for cache locality
    std::vector<int> compute_assembly_order(const Mesh& mesh) {
        // Space-filling curve for cache coherence
        return hilbert_curve_ordering(mesh);
    }
    
    // Prefetch for next element
    void prefetch_element_data(int next_elem) {
        __builtin_prefetch(&coords[next_elem], 0, 3);
        __builtin_prefetch(&connectivity[next_elem], 0, 3);
    }
};
```

### SIMD Vectorization
```cpp
// SIMD-optimized local assembly
void assemble_element_simd(const Element& elem,
                          Matrix& K_e) {
    using Vec = simd::double4;
    
    // Vectorized quadrature loop
    for (int q = 0; q < n_quad; q += 4) {
        Vec weights = load_aligned(&quad_weights[q]);
        
        // Compute 4 quadrature points simultaneously
        Vec J_det = compute_jacobian_simd(elem, q);
        
        // Vectorized physics evaluation
        // ...
    }
}
```

## Integration Points

### With fem/
- Uses `fem::Element` for local matrix computation
- Gets shape functions from `fem::ShapeFunction`
- Receives quadrature rules from `fem::Integration`

### With mesh/
- Gets connectivity from `mesh::Topology`
- Uses element-to-DOF maps from `mesh::DOFManager`
- Receives partitioning info for distributed assembly

### With numeric/
- Builds `numeric::SparseMatrix` structures
- Uses `numeric::GraphColoring` algorithms
- Leverages `numeric::AtomicOps` for thread safety

### With solvers/
- Provides assembled systems to linear solvers
- Supplies matrix-free operators for iterative methods
- Manages constraint-modified systems

## Success Metrics

1. **Assembly Speed**: > 1M DOFs/second on single core
2. **Parallel Efficiency**: > 90% on 16 cores
3. **GPU Speedup**: > 20x for large problems
4. **Pattern Reuse**: < 1% overhead for pattern cache
5. **Memory Usage**: < 2x matrix storage for assembly
6. **Constraint Application**: < 5% of assembly time

## Key Features

1. **Conflict-Free Parallel**: Graph coloring eliminates race conditions
2. **Cache-Aware**: Reuses patterns and workspaces for nonlinear problems
3. **GPU-Ready**: Native GPU assembly with optimized kernels
4. **Constraint-Integrated**: Handles all constraint types efficiently
5. **Matrix-Free Option**: Supports both explicit and matrix-free assembly
6. **Block-Structured**: Native support for multi-field problems

This architecture provides comprehensive assembly capabilities from element-level operations through parallel global assembly, with full support for constraints, caching, and both CPU and GPU execution.
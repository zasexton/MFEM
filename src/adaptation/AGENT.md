# AGENT.md - Adaptive Strategies Module

## Mission
Orchestrate error-driven adaptive refinement and coarsening strategies for finite element analysis, managing error estimation, refinement decisions, solution transfer, and ensuring optimal mesh resolution for accuracy and computational efficiency.

## Architecture Philosophy
- **Error-Driven**: All adaptation based on rigorous error estimation
- **Conservation-First**: Maintain physical quantities during adaptation
- **Multi-Strategy**: Support h-, p-, r-, and hp-adaptivity
- **Goal-Oriented**: Option for quantity-of-interest adaptation
- **Parallel-Aware**: Scalable adaptation for distributed meshes

## Directory Structure

```
adaptation/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── error_estimators/                # Error estimation strategies
│   ├── error_estimator_base.hpp    # Base estimator interface
│   ├── residual/
│   │   ├── element_residual.hpp    # Element residual method
│   │   ├── edge_residual.hpp       # Edge jump estimators
│   │   ├── bubble_estimator.hpp    # Bubble function approach
│   │   └── equilibrated_residual.hpp # Equilibrated residual
│   ├── recovery/
│   │   ├── zz_estimator.hpp        # Zienkiewicz-Zhu
│   │   ├── spr_estimator.hpp       # Superconvergent Patch Recovery
│   │   ├── recovery_estimator.hpp  # General recovery methods
│   │   └── gradient_recovery.hpp   # Gradient-based recovery
│   ├── hierarchical/
│   │   ├── hierarchical_estimator.hpp # Hierarchical bases
│   │   ├── saturation_estimator.hpp   # Saturation assumption
│   │   └── bank_weiser.hpp            # Bank-Weiser estimator
│   ├── goal_oriented/
│   │   ├── dual_weighted_residual.hpp # DWR method
│   │   ├── adjoint_estimator.hpp      # Adjoint-based
│   │   ├── qoi_estimator.hpp          # Quantity of interest
│   │   └── sensitivity_estimator.hpp  # Sensitivity analysis
│   ├── interpolation/
│   │   ├── interpolation_error.hpp    # Interpolation estimates
│   │   └── projection_error.hpp       # Projection-based
│   └── physics_specific/
│       ├── energy_norm.hpp            # Energy norm estimates
│       ├── conservation_error.hpp     # Conservation metrics
│       └── constitutive_error.hpp     # Constitutive relation error
│
├── marking_strategies/              # Element marking for refinement
│   ├── marking_strategy_base.hpp   # Base marking interface
│   ├── threshold/
│   │   ├── fixed_fraction.hpp      # Fixed fraction marking
│   │   ├── fixed_number.hpp        # Fixed number of elements
│   │   ├── absolute_threshold.hpp  # Absolute error threshold
│   │   └── relative_threshold.hpp  # Relative error threshold
│   ├── optimization/
│   │   ├── optimal_marking.hpp     # Optimization-based marking
│   │   ├── equidistribution.hpp    # Error equidistribution
│   │   └── predictive_marking.hpp  # Predictive strategies
│   ├── guaranteed/
│   │   ├── dorfler_marking.hpp     # Dörfler marking
│   │   ├── maximum_marking.hpp     # Maximum strategy
│   │   └── guaranteed_reduction.hpp # Guaranteed error reduction
│   └── parallel/
│       ├── load_balanced_marking.hpp # Load-aware marking
│       └── distributed_marking.hpp   # Distributed decisions
│
├── refinement_strategies/           # Refinement methods
│   ├── refinement_strategy.hpp      # Base strategy
│   ├── h_refinement/
│   │   ├── h_strategy_base.hpp        # h-refinement interface
│   │   ├── isotropic_refinement.hpp   # Uniform subdivision
│   │   ├── adaptive_refinement.hpp    # Error-based refinement
│   │   ├── anisotropic_refinement.hpp # Directional refinement
│   │   ├── bisection_refinement.hpp   # Longest edge bisection
│   │   ├── regular_refinement.hpp     # Regular patterns
│   │   ├── conforming_refinement.hpp  # Maintain conformity
│   │   └── hanging_node_handler.hpp   # Hanging node treatment
│   ├── p_refinement/
│   │   ├── p_strategy_base.hpp        # p-refinement interface
│   │   ├── polynomial_enrichment.hpp
│   │   ├── order_elevation.hpp
│   │   ├── uniform_p_refinement.hpp   # Uniform order increase
│   │   ├── variable_p_refinement.hpp  # Variable order
│   │   ├── modal_enrichment.hpp       # Modal basis enrichment
│   │   └── spectral_refinement.hpp    # Spectral convergence
│   ├── r_refinement/
│   │   ├── r_strategy_base.hpp     # r-refinement (moving nodes)
│   │   ├── spring_analogy.hpp      # Spring-based movement
│   │   ├── variational_r.hpp       # Variational approach
│   │   └── optimal_node_placement.hpp # Optimal positioning
│   ├── hp_refinement/
│   │   ├── hp_strategy_base.hpp    # hp-refinement interface
│   │   ├── smoothness_indicator.hpp # Solution smoothness
│   │   ├── hp_decision.hpp         # h vs p decision logic
│   │   ├── exponential_convergence.hpp # Exponential rates
│   │   └── reference_solution.hpp  # Reference-based hp
│   └── combined/
│       ├── hr_refinement.hpp       # h+r refinement
│       ├── pr_refinement.hpp       # p+r refinement
│       └── hpr_refinement.hpp      # h+p+r refinement
│
├── coarsening_strategies/           # Mesh coarsening
│   ├── coarsening_base.hpp         # Base coarsening interface
│   ├── h_coarsening/
│   │   ├── edge_collapse.hpp       # Edge collapsing
│   │   ├── vertex_removal.hpp      # Vertex removal
│   │   ├── face_removal.hpp        # Face removal
│   │   ├── agglomeration.hpp       # Element agglomeration
│   │   └── unrefinement.hpp        # Reverse refinement
│   ├── p_coarsening/
│   │   ├── order_reduction.hpp     # Polynomial order reduction
│   │   └── modal_truncation.hpp    # Modal basis truncation
│   ├── derefinement_criteria/
│   │   ├── error_based.hpp         # Error-based coarsening
│   │   ├── gradient_based.hpp      # Solution gradient
│   │   └── age_based.hpp           # Time since refinement
│   └── admissibility/
│       ├── topology_preservation.hpp # Maintain valid topology
│       └── quality_preservation.hpp  # Maintain mesh quality
│
├── solution_transfer/               # Solution projection
│   ├── transfer_operator_base.hpp  # Base transfer interface
│   ├── interpolation/
│   │   ├── nodal_interpolation.hpp # Node-based interpolation
│   │   ├── high_order_interpolation.hpp # High-order transfer
│   │   └── patch_interpolation.hpp # Patch-based methods
│   ├── projection/
│   │   ├── l2_projection.hpp       # L2 projection
│   │   ├── h1_projection.hpp       # H1 projection
│   │   ├── galerkin_projection.hpp # Galerkin projection
│   │   ├── hierarchical_transfer.hpp # Hierarchical basis
│   │   └── conservative_projection.hpp # Conservative transfer
│   ├── restriction_prolongation/
│   │   ├── injection.hpp           # Simple injection
│   │   ├── full_weighting.hpp      # Full weighting restriction
│   │   ├── linear_interpolation.hpp # Linear prolongation
│   │   └── high_order_transfer.hpp # High-order operators
│   ├── field_transfer/
│   │   ├── scalar_transfer.hpp     # Scalar field transfer
│   │   ├── vector_transfer.hpp     # Vector field transfer
│   │   ├── tensor_transfer.hpp     # Tensor field transfer
│   │   └── history_transfer.hpp    # History variable transfer
│   │── conservation/
│   │   ├── mass_conservation.hpp   # Conserve mass
│   │   ├── momentum_conservation.hpp # Conserve momentum
│   │   └── energy_conservation.hpp # Conserve energy
│   └── multigrid_transfer/         # MG operators
│       ├── geometric_transfer.hpp  # Geometric MG
│       └── algebraic_transfer.hpp  # AMG transfers
│
├── adaptive_loop/                   # Adaptation orchestration
│   ├── adaptive_solver.hpp         # Main adaptive loop
│   ├── convergence_criteria.hpp    # Convergence checks
│   ├── adaptation_controller.hpp   # Control logic
│   ├── step_controller.hpp         # Adaptation step control
│   ├── history_tracker.hpp         # Track adaptation history
│   └── rollback_manager.hpp        # Rollback on failure
│
├── indicators/                      # Refinement indicators
│   ├── smoothness_indicator.hpp    # Solution smoothness
│   ├── feature_indicator.hpp       # Feature detection
│   ├── gradient_indicator.hpp      # Gradient-based
│   ├── curvature_indicator.hpp     # Curvature detection
│   ├── physics_indicator.hpp       # Physics-based indicators
│   └── geometric_indicator.hpp     # Geometric features
│
├── metrics/                         # Adaptation metrics
│   ├── effectivity_index.hpp       # Effectivity computation
│   ├── convergence_rate.hpp        # Convergence analysis
│   ├── efficiency_metric.hpp       # Computational efficiency
│   ├── quality_metric.hpp          # Mesh quality tracking
│   └── performance_metric.hpp      # Performance monitoring
│
├── constraints/                     # Adaptation constraints
│   ├── level_constraint.hpp        # Max refinement level
│   ├── size_constraint.hpp         # Element size limits
│   ├── anisotropy_constraint.hpp   # Anisotropy limits
│   ├── topology_constraint.hpp     # Topology preservation
│   ├── load_balance_constraint.hpp # Parallel load balance
│   └── memory_constraint.hpp       # Memory limitations
│
├── parallel/                        # Parallel adaptation
│   ├── parallel_estimator.hpp      # Distributed error estimation
│   ├── parallel_marking.hpp        # Distributed marking
│   ├── parallel_refinement.hpp     # Distributed refinement
│   ├── load_rebalancing.hpp        # Post-adaptation balancing
│   ├── ghost_update.hpp            # Ghost layer updates
│   └── migration.hpp               # Element migration
│
├── optimization/                    # Optimal adaptation
│   ├── optimal_mesh.hpp            # Optimal mesh generation
│   ├── error_equilibration.hpp     # Error equilibration
│   ├── cost_optimization.hpp       # Cost vs accuracy
│   ├── resource_optimization.hpp   # Resource constraints
│   └── multi_objective.hpp         # Multi-objective optimization
│
├── special/                         # Special adaptivity
│   ├── space_time/
│   │   ├── space_time_adaptivity.hpp # 4D adaptation
│   │   └── time_slab_refinement.hpp  # Time slab methods
│   ├── moving_mesh/
│   │   ├── ale_adaptivity.hpp      # ALE mesh adaptation
│   │   └── interface_tracking.hpp  # Interface refinement
│   ├── multiscale/
│   │   ├── scale_separation.hpp    # Scale-based adaptation
│   │   └── heterogeneous_adaptivity.hpp
│   └── uncertainty/
│       ├── stochastic_adaptivity.hpp # Stochastic adaptation
│       └── robust_adaptivity.hpp     # Robust refinement
│
├── utilities/                       # Adaptation utilities
│   ├── adaptation_history.hpp      # History tracking
│   ├── statistics.hpp              # Statistical analysis
│   ├── visualization.hpp           # Adaptation visualization
│   └── debugging.hpp               # Debug utilities
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── convergence/                # Convergence studies
    ├── benchmarks/                 # Performance benchmarks
    └── validation/                 # Validation problems
```

## Key Components

### 1. Adaptive Solution Loop
```cpp
// Main adaptive solver orchestration
template<typename Problem>
class AdaptiveSolver {
    ErrorEstimator estimator;
    MarkingStrategy marking;
    RefinementStrategy refinement;
    SolutionTransfer transfer;
    
    // Adaptive solution algorithm
    Solution solve_adaptive(Problem& problem,
                          double tolerance,
                          int max_iterations = 10) {
        Mesh mesh = problem.initial_mesh();
        Solution u;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // 1. Solve on current mesh
            u = solve_problem(problem, mesh);
            
            // 2. Estimate error
            auto error_indicators = estimator.estimate(mesh, u);
            double global_error = compute_global_error(error_indicators);
            
            // 3. Check convergence
            if (global_error < tolerance) {
                return u;  // Converged
            }
            
            // 4. Mark elements for refinement
            auto marked = marking.mark(mesh, error_indicators);
            
            // 5. Adapt mesh
            auto new_mesh = refinement.refine(mesh, marked);
            
            // 6. Transfer solution
            Solution u_new(new_mesh);
            transfer.project(mesh, u, new_mesh, u_new);
            
            // 7. Update for next iteration
            mesh = std::move(new_mesh);
            u = std::move(u_new);
            
            // 8. Optional: coarsen where possible
            coarsen_if_needed(mesh, error_indicators);
        }
        
        return u;
    }
};
```

### 2. Error Estimation
```cpp
// Residual-based error estimator
class ResidualEstimator : public ErrorEstimator {
    // Element-wise error indicators
    std::vector<double> estimate(const Mesh& mesh,
                                const Solution& u) override {
        std::vector<double> eta(mesh.n_elements());
        
        #pragma omp parallel for
        for (int e = 0; e < mesh.n_elements(); ++e) {
            auto elem = mesh.element(e);
            
            // Interior residual
            double r_T = compute_element_residual(elem, u);
            
            // Edge jumps
            double r_E = 0.0;
            for (auto& edge : elem.edges()) {
                if (!edge.is_boundary()) {
                    r_E += compute_edge_jump(edge, u);
                }
            }
            
            // Element error indicator
            double h = elem.diameter();
            eta[e] = h * r_T + sqrt(h) * r_E;
        }
        
        return eta;
    }
};

// Goal-oriented error estimation
class DualWeightedResidual : public ErrorEstimator {
    // Solve adjoint problem
    Solution solve_dual(const Problem& primal, 
                       const QoI& quantity) {
        // Formulate adjoint problem
        AdjointProblem dual(primal, quantity);
        return solve(dual);
    }
    
    std::vector<double> estimate(const Mesh& mesh,
                                const Solution& u_h) override {
        // Solve dual problem
        auto z_h = solve_dual(problem, qoi);
        
        // Compute weighted residuals
        std::vector<double> eta(mesh.n_elements());
        
        for (int e = 0; e < mesh.n_elements(); ++e) {
            // Residual weighted by dual solution
            double r = compute_residual(e, u_h);
            double w = interpolate_dual_weight(e, z_h);
            eta[e] = abs(r * w);
        }
        
        return eta;
    }
};
```

### 3. Marking Strategies
```cpp
// Dörfler marking with guaranteed reduction
class DorflerMarking : public MarkingStrategy {
    double theta = 0.5;  // Dörfler parameter
    
    std::vector<bool> mark(const Mesh& mesh,
                          const std::vector<double>& eta) override {
        // Sort elements by error
        std::vector<int> sorted_elements = argsort(eta);
        
        // Compute total error
        double total_error_sq = 0.0;
        for (auto e : eta) {
            total_error_sq += e * e;
        }
        
        // Mark elements until Dörfler criterion satisfied
        std::vector<bool> marked(mesh.n_elements(), false);
        double marked_error_sq = 0.0;
        
        for (auto elem_id : sorted_elements) {
            marked[elem_id] = true;
            marked_error_sq += eta[elem_id] * eta[elem_id];
            
            if (marked_error_sq >= theta * total_error_sq) {
                break;  // Dörfler criterion satisfied
            }
        }
        
        return marked;
    }
};
```

### 4. hp-Adaptivity Decision
```cpp
// Decide between h- and p-refinement
class HPDecision {
    // Smoothness indicator based on decay of coefficients
    double estimate_smoothness(const Element& elem,
                              const Solution& u) {
        // Project to Legendre basis
        auto coeffs = project_to_legendre(elem, u);
        
        // Estimate decay rate
        double decay_rate = estimate_decay(coeffs);
        
        // Higher decay rate = smoother solution
        return decay_rate;
    }
    
    // Make hp decision
    RefinementType decide(const Element& elem,
                        const Solution& u,
                        double error) {
        double smoothness = estimate_smoothness(elem, u);
        
        if (smoothness > smoothness_threshold) {
            // Smooth solution: p-refinement more efficient
            return RefinementType::P_REFINE;
        } else {
            // Non-smooth: h-refinement needed
            return RefinementType::H_REFINE;
        }
    }
};
```

### 5. Conservative Solution Transfer
```cpp
// Conservative projection for refinement
class ConservativeTransfer : public SolutionTransfer {
    void project(const Mesh& coarse_mesh,
                const Solution& u_coarse,
                const Mesh& fine_mesh,
                Solution& u_fine) override {
        // Ensure conservation of integral quantities
        
        for (auto& fine_elem : fine_mesh.elements()) {
            // Find parent in coarse mesh
            auto parent = coarse_mesh.find_parent(fine_elem);
            
            // L2 projection maintaining conservation
            auto M_fine = compute_mass_matrix(fine_elem);
            auto rhs = compute_projection_rhs(parent, u_coarse, fine_elem);
            
            // Local solve
            u_fine.set_local(fine_elem, M_fine.solve(rhs));
        }
        
        // Verify conservation
        double coarse_integral = integrate(coarse_mesh, u_coarse);
        double fine_integral = integrate(fine_mesh, u_fine);
        assert(abs(coarse_integral - fine_integral) < 1e-12);
    }
};
```

### 6. Parallel Adaptation
```cpp
// Distributed mesh adaptation
class ParallelAdaptation {
    // Parallel error estimation with reduction
    double estimate_global_error(const DistributedMesh& mesh,
                                const Solution& u) {
        // Local error estimation
        auto local_error = estimate_local(mesh.local_elements(), u);
        
        // Global reduction
        double global_error;
        MPI_Allreduce(&local_error, &global_error, 1, 
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        return sqrt(global_error);
    }
    
    // Load-balanced marking
    std::vector<bool> mark_with_load_balance(
        const DistributedMesh& mesh,
        const std::vector<double>& error) {
        
        // Estimate load after refinement
        auto predicted_load = predict_refinement_load(mesh, error);
        
        // Adjust marking for load balance
        auto marked = initial_marking(error);
        balance_marking(marked, predicted_load);
        
        return marked;
    }
    
    // Repartition after adaptation
    void rebalance_after_adaptation(DistributedMesh& mesh) {
        // Compute new partition
        auto new_partition = compute_balanced_partition(mesh);
        
        // Migrate elements
        mesh.migrate_elements(new_partition);
        
        // Update ghost layers
        mesh.update_ghosts();
    }
};
```

### 7. Anisotropic Refinement
```cpp
// Directional refinement based on solution
class AnisotropicRefinement {
    // Compute metric tensor from Hessian
    Matrix compute_metric(const Element& elem,
                         const Solution& u) {
        auto H = compute_hessian(elem, u);
        
        // Eigendecomposition
        auto [eigvals, eigvecs] = eigen_decompose(H);
        
        // Build metric for desired error
        Matrix M;
        for (int i = 0; i < dim; ++i) {
            double lambda = compute_metric_eigenvalue(eigvals[i]);
            M += lambda * outer_product(eigvecs[i], eigvecs[i]);
        }
        
        return M;
    }
    
    // Anisotropic refinement pattern
    void refine_anisotropic(Element& elem,
                           const Matrix& metric) {
        // Determine refinement directions
        auto directions = get_refinement_directions(metric);
        
        // Refine only in necessary directions
        for (auto dir : directions) {
            elem.refine_in_direction(dir);
        }
    }
};
```

## 🚀 Performance Optimizations

### Hierarchical Error Estimation
```cpp
// Use hierarchy for fast estimation
class HierarchicalEstimator {
    // Reuse coarse solution
    std::vector<double> estimate_hierarchical(
        const Mesh& fine_mesh,
        const Solution& u_fine,
        const Solution& u_coarse) {
        
        std::vector<double> eta(fine_mesh.n_elements());
        
        #pragma omp parallel for
        for (int e = 0; e < fine_mesh.n_elements(); ++e) {
            // Use difference as error indicator
            auto diff = u_fine.local(e) - interpolate(u_coarse, e);
            eta[e] = norm(diff);
        }
        
        return eta;
    }
};
```

### Cached Refinement Patterns
```cpp
// Cache common refinement patterns
class RefinementCache {
    std::unordered_map<ElementType, RefinementPattern> patterns;
    
    void refine_with_cache(Element& elem) {
        auto pattern = patterns[elem.type()];
        if (pattern) {
            apply_pattern(elem, pattern);
        } else {
            auto new_pattern = compute_pattern(elem);
            patterns[elem.type()] = new_pattern;
            apply_pattern(elem, new_pattern);
        }
    }
};
```

## Integration Points

### With mesh/
- Triggers mesh refinement/coarsening operations
- Updates mesh topology after adaptation
- Maintains mesh quality during refinement

### With fem/
- Uses element shape functions for error estimation
- Projects between polynomial spaces for p-refinement
- Evaluates weak form residuals

### With solvers/
- Integrates with iterative solvers for adaptive loops
- Provides error estimates for solver tolerance
- Triggers re-solves after adaptation

### With assembly/
- Updates sparsity patterns after refinement
- Rebuilds assembly structures for new mesh
- Handles constraint updates

## Success Metrics

1. **Error Reduction**: Optimal convergence rates achieved
2. **Efficiency Index**: Effectivity index near 1.0
3. **Marking Efficiency**: < 30% elements marked per iteration
4. **Transfer Conservation**: Machine precision conservation
5. **Parallel Scaling**: > 85% efficiency for adaptation
6. **Memory Overhead**: < 20% for adaptation structures

## Key Features

1. **Complete Adaptivity**: h-, p-, r-, and hp-refinement strategies
2. **Goal-Oriented**: Adapt for specific quantities of interest
3. **Conservative Transfer**: Maintains physical conservation laws
4. **Parallel-Ready**: Scalable distributed adaptation
5. **Error Guaranteed**: Rigorous error estimation and reduction
6. **Physics-Aware**: Specialized estimators for different physics

This architecture provides comprehensive adaptive capabilities for optimizing mesh resolution based on solution behavior, ensuring accuracy while minimizing computational cost.
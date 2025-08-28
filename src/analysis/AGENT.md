# AGENT.md - Analysis Procedures Module

## Mission
Orchestrate high-level finite element analysis procedures, managing solution strategies, time evolution, nonlinear iterations, and complex multi-step simulations while coordinating between physics modules, solvers, and adaptive strategies.

## Architecture Philosophy
- **Workflow-Centric**: Define complete analysis workflows
- **State-Machine Based**: Manage complex analysis states
- **Recovery-Capable**: Handle failures and restart gracefully
- **Physics-Agnostic**: Works with any physics modules
- **Adaptive-Aware**: Integrated with adaptation strategies

## Directory Structure

```
analysis/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── core/                            # Core analysis infrastructure
│   ├── analysis_base.hpp           # Base analysis interface
│   ├── analysis_manager.hpp        # Analysis orchestration
│   ├── analysis_state.hpp          # State management
│   ├── analysis_context.hpp        # Analysis context/settings
│   ├── step_controller.hpp         # Step control logic
│   ├── convergence_monitor.hpp     # Convergence monitoring
│   └── analysis_factory.hpp        # Analysis type factory
│
├── static/                          # Static analysis
│   ├── linear_static.hpp           # Linear static analysis
│   ├── nonlinear_static.hpp        # Nonlinear static
│   ├── incremental_loading.hpp     # Load stepping
│   ├── displacement_control.hpp    # Displacement control
│   └── work_control.hpp            # Work control method
│
├── dynamic/                         # Dynamic analysis
│   ├── transient/
│   │   ├── time_integration.hpp    # Time integration base
│   │   ├── explicit_dynamics.hpp   # Explicit time integration
│   │   ├── implicit_dynamics.hpp   # Implicit time integration
│   │   ├── newmark_method.hpp      # Newmark-β method
│   │   ├── hht_alpha.hpp           # HHT-α method
│   │   ├── generalized_alpha.hpp   # Generalized-α
│   │   └── wbz_alpha.hpp           # WBZ-α method
│   ├── modal/
│   │   ├── modal_analysis.hpp      # Modal analysis
│   │   ├── eigenvalue_extraction.hpp # Eigenvalue problems
│   │   ├── modal_superposition.hpp # Mode superposition
│   │   ├── modal_damping.hpp       # Damping models
│   │   └── participation_factors.hpp # Modal participation
│   ├── harmonic/
│   │   ├── harmonic_response.hpp   # Harmonic analysis
│   │   ├── frequency_sweep.hpp     # Frequency domain
│   │   ├── complex_modes.hpp       # Complex eigenvalues
│   │   └── transfer_function.hpp   # Transfer functions
│   └── spectral/
│       ├── response_spectrum.hpp   # Response spectrum
│       ├── random_vibration.hpp    # Random vibration
│       └── psd_analysis.hpp        # PSD analysis
│
├── stability/                       # Stability analysis
│   ├── buckling/
│   │   ├── linear_buckling.hpp     # Linear buckling
│   │   ├── nonlinear_buckling.hpp  # Nonlinear buckling
│   │   ├── post_buckling.hpp       # Post-buckling
│   │   └── imperfection_analysis.hpp # Imperfections
│   ├── limit_load/
│   │   ├── limit_analysis.hpp      # Limit load analysis
│   │   ├── shakedown.hpp           # Shakedown analysis
│   │   └── plastic_collapse.hpp    # Plastic collapse
│   └── bifurcation/
│       ├── bifurcation_tracking.hpp # Bifurcation points
│       ├── branch_switching.hpp     # Branch following
│       └── path_following.hpp       # Solution paths
│
├── continuation/                    # Continuation methods
│   ├── arc_length/
│   │   ├── arc_length_method.hpp   # Arc-length control
│   │   ├── riks_method.hpp         # Riks method
│   │   ├── crisfield.hpp           # Crisfield method
│   │   └── modified_riks.hpp       # Modified Riks
│   ├── predictor_corrector/
│   │   ├── predictor_base.hpp      # Predictor interface
│   │   ├── tangent_predictor.hpp   # Tangent predictor
│   │   ├ното secant_predictor.hpp  # Secant predictor
│   │   └── corrector_iterations.hpp # Corrector schemes
│   └── path_control/
│       ├── load_control.hpp        # Load parameter
│       ├── displacement_control.hpp # Displacement param
│       └── arc_length_control.hpp  # Arc-length param
│
├── multiphysics/                    # Multiphysics analysis
│   ├── coupled_analysis.hpp        # Coupled analysis base
│   ├── staggered/
│   │   ├── staggered_scheme.hpp    # Staggered solution
│   │   ├── isothermal_split.hpp    # Isothermal operator
│   │   └── adiabatic_split.hpp     # Adiabatic operator
│   ├── monolithic/
│   │   ├── fully_coupled.hpp       # Monolithic solution
│   │   └── newton_coupled.hpp      # Coupled Newton
│   └── partitioned/
│       ├── weak_coupling.hpp       # Weak coupling
│       └── strong_coupling.hpp     # Strong coupling
│
├── optimization/                    # Optimization analysis
│   ├── sensitivity/
│   │   ├── sensitivity_analysis.hpp # Sensitivity base
│   │   ├── direct_differentiation.hpp # Direct method
│   │   ├── adjoint_method.hpp      # Adjoint method
│   │   └── finite_difference.hpp   # FD sensitivities
│   ├── shape/
│   │   ├── shape_optimization.hpp  # Shape optimization
│   │   ├── mesh_morphing.hpp       # Mesh deformation
│   │   └── cad_parametric.hpp      # CAD parameters
│   ├── topology/
│   │   ├── topology_optimization.hpp # Topology opt
│   │   ├── simp_method.hpp         # SIMP method
│   │   ├── level_set_topology.hpp  # Level set method
│   │   └── evolutionary.hpp        # Evolutionary methods
│   └── parameter/
│       ├── parameter_identification.hpp # Parameter ID
│       └── inverse_problems.hpp    # Inverse analysis
│
├── stochastic/                      # Stochastic analysis
│   ├── uncertainty/
│   │   ├── monte_carlo.hpp         # Monte Carlo
│   │   ├── latin_hypercube.hpp     # LHS sampling
│   │   ├── polynomial_chaos.hpp    # PC expansion
│   │   └── stochastic_collocation.hpp # Collocation
│   ├── reliability/
│   │   ├── form_sorm.hpp           # FORM/SORM
│   │   ├── importance_sampling.hpp # Importance sampling
│   │   └── subset_simulation.hpp   # Subset simulation
│   └── robust/
│       ├── robust_design.hpp       # Robust optimization
│       └── worst_case.hpp          # Worst-case analysis
│
├── special/                         # Special analyses
│   ├── substructuring/
│   │   ├── craig_bampton.hpp       # Craig-Bampton
│   │   ├── component_modes.hpp     # CMS
│   │   └── superelement.hpp        # Superelements
│   ├── contact/
│   │   ├── contact_analysis.hpp    # Contact problems
│   │   ├── gap_analysis.hpp        # Gap elements
│   │   └── wear_analysis.hpp       # Wear simulation
│   ├── fracture/
│   │   ├── crack_propagation.hpp   # Crack growth
│   │   ├── j_integral.hpp          # J-integral
│   │   └── cohesive_zone.hpp       # CZM
│   └── fatigue/
│       ├── fatigue_analysis.hpp    # Fatigue life
│       ├── sn_curve.hpp            # S-N approach
│       └── crack_growth_rate.hpp   # da/dN
│
├── control/                         # Analysis control
│   ├── load_step/
│   │   ├── load_step_manager.hpp   # Load step control
│   │   ├── adaptive_stepping.hpp   # Adaptive steps
│   │   └── substep_control.hpp     # Substepping
│   ├── convergence/
│   │   ├── convergence_criteria.hpp # Convergence tests
│   │   ├── norm_based.hpp          # Norm criteria
│   │   ├── energy_based.hpp        # Energy criteria
│   │   └── force_based.hpp         # Force criteria
│   └── recovery/
│       ├── bisection_recovery.hpp  # Bisection on failure
│       ├── line_search.hpp         # Line search
│       └── cutback_strategy.hpp    # Time step cutback
│
├── workflow/                        # Analysis workflows
│   ├── workflow_engine.hpp         # Workflow execution
│   ├── sequential_analysis.hpp     # Sequential steps
│   ├── branching_analysis.hpp      # Conditional branches
│   ├── iterative_analysis.hpp      # Iterative workflows
│   └── parallel_workflows.hpp      # Parallel execution
│
├── postprocessing/                  # Analysis postprocessing
│   ├── results_extraction.hpp      # Results extraction
│   ├── history_output.hpp          # Time history
│   ├── envelope_results.hpp        # Min/max envelopes
│   ├── averaging.hpp               # Result averaging
│   └── path_results.hpp            # Path-based results
│
├── utilities/                       # Analysis utilities
│   ├── analysis_monitor.hpp        # Runtime monitoring
│   ├── progress_tracker.hpp        # Progress reporting
│   ├── performance_metrics.hpp     # Performance tracking
│   └── debugging_aids.hpp          # Debug helpers
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── verification/               # Verification tests
    ├── benchmarks/                 # Analysis benchmarks
    └── workflows/                  # Workflow tests
```

## Key Components

### 1. Analysis Manager
```cpp
// High-level analysis orchestration
class AnalysisManager {
    std::unique_ptr<PhysicsModule> physics;
    std::unique_ptr<Solver> solver;
    std::unique_ptr<Mesh> mesh;
    AnalysisContext context;
    
public:
    // Execute complete analysis
    AnalysisResults run(const AnalysisProcedure& procedure) {
        // Initialize
        initialize_analysis(procedure);
        
        // Setup workflow
        auto workflow = create_workflow(procedure);
        
        // Execute with monitoring
        AnalysisResults results;
        try {
            results = workflow->execute(context);
        } catch (const ConvergenceException& e) {
            results = handle_convergence_failure(e);
        } catch (const AnalysisException& e) {
            results = recover_from_failure(e);
        }
        
        // Post-process
        finalize_results(results);
        
        return results;
    }
    
private:
    std::unique_ptr<Workflow> create_workflow(
        const AnalysisProcedure& proc) {
        
        switch (proc.type) {
            case AnalysisType::LinearStatic:
                return std::make_unique<LinearStaticWorkflow>();
                
            case AnalysisType::NonlinearDynamic:
                return create_nonlinear_dynamic_workflow(proc);
                
            case AnalysisType::ArcLength:
                return create_continuation_workflow(proc);
                
            // ... other types
        }
    }
};
```

### 2. Nonlinear Static Analysis
```cpp
// Nonlinear static with load stepping
class NonlinearStatic : public AnalysisProcedure {
    struct LoadStep {
        double load_factor;
        int max_iterations;
        double tolerance;
        int min_substeps;
        int max_substeps;
    };
    
    AnalysisResults execute() override {
        AnalysisResults results;
        
        for (auto& step : load_steps) {
            bool converged = false;
            int substeps = 1;
            
            while (!converged && substeps <= step.max_substeps) {
                double delta_lambda = step.load_factor / substeps;
                
                for (int substep = 0; substep < substeps; ++substep) {
                    // Apply load increment
                    double lambda = current_lambda + delta_lambda;
                    auto F = compute_external_force(lambda);
                    
                    // Newton-Raphson iterations
                    converged = solve_nonlinear(F, step);
                    
                    if (!converged) {
                        // Cutback - increase substeps
                        substeps *= 2;
                        restore_last_converged();
                        break;
                    }
                    
                    // Update and save
                    update_state();
                    results.add_step(current_state);
                    current_lambda = lambda;
                }
            }
            
            if (!converged) {
                throw ConvergenceException("Failed to converge");
            }
        }
        
        return results;
    }
    
private:
    bool solve_nonlinear(const Vector& F_ext, 
                        const LoadStep& step) {
        Vector u = current_solution;
        
        for (int iter = 0; iter < step.max_iterations; ++iter) {
            // Compute residual
            auto F_int = compute_internal_force(u);
            Vector R = F_ext - F_int;
            
            // Check convergence
            if (check_convergence(R, step.tolerance)) {
                current_solution = u;
                return true;
            }
            
            // Compute tangent
            auto K = compute_tangent_stiffness(u);
            
            // Solve for increment
            Vector delta_u = K.solve(R);
            
            // Line search if needed
            double alpha = line_search(u, delta_u, R);
            
            // Update solution
            u += alpha * delta_u;
        }
        
        return false;
    }
};
```

### 3. Transient Dynamic Analysis
```cpp
// Implicit dynamics with Newmark method
class NewmarkDynamics : public DynamicAnalysis {
    double beta = 0.25;
    double gamma = 0.5;
    
    void advance_time_step(double dt) {
        // Predictors
        Vector u_pred = u + dt * v + dt*dt * (0.5 - beta) * a;
        Vector v_pred = v + dt * (1 - gamma) * a;
        
        // Newton iterations for acceleration
        Vector a_new = a;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Update displacement and velocity
            Vector u_new = u_pred + beta * dt * dt * a_new;
            Vector v_new = v_pred + gamma * dt * a_new;
            
            // Compute residual
            Vector R = M * a_new + C * v_new + 
                      compute_internal_force(u_new) - F_ext(t + dt);
            
            // Check convergence
            if (norm(R) < tolerance) {
                u = u_new;
                v = v_new;
                a = a_new;
                return;
            }
            
            // Effective tangent
            Matrix K_eff = K + (gamma / (beta * dt)) * C + 
                          (1.0 / (beta * dt * dt)) * M;
            
            // Solve for increment
            Vector delta_a = K_eff.solve(-R);
            a_new += delta_a;
        }
        
        throw ConvergenceException("Time step failed");
    }
};
```

### 4. Arc-Length Method
```cpp
// Arc-length continuation for snap-through
class ArcLengthMethod : public ContinuationMethod {
    double arc_length;
    
    Solution trace_equilibrium_path() {
        Solution path;
        
        // Initial predictor
        auto [u, lambda] = compute_initial_solution();
        path.add_point(u, lambda);
        
        while (lambda < lambda_max && !reached_limit_point()) {
            // Predictor step
            auto [du_pred, dlambda_pred] = 
                compute_predictor(u, lambda);
            
            // Corrector iterations
            Vector u_new = u + du_pred;
            double lambda_new = lambda + dlambda_pred;
            
            for (int iter = 0; iter < max_iterations; ++iter) {
                // Compute residual
                Vector R = compute_residual(u_new, lambda_new);
                
                // Arc-length constraint
                double g = compute_constraint(u_new - u, 
                                            lambda_new - lambda);
                
                // Check convergence
                if (norm(R) < tol_force && abs(g) < tol_disp) {
                    u = u_new;
                    lambda = lambda_new;
                    path.add_point(u, lambda);
                    break;
                }
                
                // Solve augmented system
                auto [du, dlambda] = solve_augmented_system(
                    R, g, u_new, lambda_new
                );
                
                u_new += du;
                lambda_new += dlambda;
            }
            
            // Adjust arc length for next step
            adapt_arc_length();
        }
        
        return path;
    }
    
private:
    std::pair<Vector, double> solve_augmented_system(
        const Vector& R, double g,
        const Vector& u, double lambda) {
        
        // Augmented system:
        // [K   -P] [du    ]   [-R]
        // [a^T  b] [dlambda] = [-g]
        
        Matrix K = compute_tangent(u);
        Vector P = compute_load_vector();
        
        // Compute constraint derivatives
        Vector a = compute_constraint_derivative_u();
        double b = compute_constraint_derivative_lambda();
        
        // Block solve
        Vector du1 = K.solve(P);
        Vector du2 = K.solve(-R);
        
        double dlambda = (-g - dot(a, du2)) / (b + dot(a, du1));
        Vector du = du2 + dlambda * du1;
        
        return {du, dlambda};
    }
};
```

### 5. Modal Analysis
```cpp
// Modal analysis with subspace iteration
class ModalAnalysis : public Analysis {
    struct ModeInfo {
        double frequency;
        double period;
        Vector mode_shape;
        double effective_mass;
        double participation_factor;
    };
    
    std::vector<ModeInfo> extract_modes(int n_modes) {
        // Assemble mass and stiffness
        auto M = assemble_mass_matrix();
        auto K = assemble_stiffness_matrix();
        
        // Solve generalized eigenvalue problem
        EigenSolver solver;
        auto [eigenvalues, eigenvectors] = 
            solver.solve_generalized(K, M, n_modes);
        
        // Process modes
        std::vector<ModeInfo> modes;
        
        for (int i = 0; i < n_modes; ++i) {
            ModeInfo mode;
            
            // Natural frequency
            mode.frequency = sqrt(eigenvalues[i]) / (2 * M_PI);
            mode.period = 1.0 / mode.frequency;
            
            // Normalize mode shape
            mode.mode_shape = eigenvectors.col(i);
            normalize_mode(mode.mode_shape, M);
            
            // Modal participation
            Vector L = M * mode.mode_shape;
            mode.effective_mass = dot(L, unit_vector);
            mode.participation_factor = 
                mode.effective_mass / total_mass;
            
            modes.push_back(mode);
        }
        
        return modes;
    }
    
    // Modal superposition for dynamic response
    Solution modal_transient(const LoadHistory& loads,
                            const std::vector<ModeInfo>& modes) {
        int n_modes = modes.size();
        
        // Modal coordinates
        std::vector<ModalCoordinate> eta(n_modes);
        
        // Time integration
        for (double t = 0; t < t_final; t += dt) {
            // Modal forces
            Vector F = loads.evaluate(t);
            
            for (int i = 0; i < n_modes; ++i) {
                double f_modal = dot(modes[i].mode_shape, F);
                
                // Single DOF response
                eta[i] = integrate_sdof(eta[i], f_modal, 
                                      modes[i].frequency, dt);
            }
            
            // Superpose to get physical response
            Vector u = superpose_modes(eta, modes);
            solution.add_time_step(t, u);
        }
        
        return solution;
    }
};
```

### 6. Buckling Analysis
```cpp
// Linear buckling analysis
class BucklingAnalysis : public StabilityAnalysis {
    struct BucklingMode {
        double load_factor;
        Vector mode_shape;
    };
    
    std::vector<BucklingMode> compute_buckling_loads(int n_modes) {
        // Reference load
        Vector F_ref = compute_reference_load();
        
        // Linear solution at reference load
        Vector u_ref = K_elastic.solve(F_ref);
        
        // Geometric stiffness at reference state
        Matrix K_g = compute_geometric_stiffness(u_ref, F_ref);
        
        // Solve eigenvalue problem: (K_e + lambda * K_g) * phi = 0
        EigenSolver solver;
        auto [eigenvalues, eigenvectors] = 
            solver.solve_generalized(-K_g, K_elastic, n_modes);
        
        // Extract buckling modes
        std::vector<BucklingMode> modes;
        for (int i = 0; i < n_modes; ++i) {
            modes.push_back({
                .load_factor = eigenvalues[i],
                .mode_shape = eigenvectors.col(i)
            });
        }
        
        return modes;
    }
};
```

### 7. Optimization-Based Analysis
```cpp
// Topology optimization with SIMP
class TopologyOptimization : public OptimizationAnalysis {
    void optimize() {
        // Initialize design variables (density)
        Vector rho(n_elements, initial_density);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // FE analysis with current design
            auto K = assemble_stiffness_SIMP(rho);
            Vector u = K.solve(F);
            
            // Objective function (compliance)
            double compliance = dot(F, u);
            
            // Sensitivity analysis
            Vector dc_drho = compute_sensitivity(u, rho);
            
            // Filter sensitivities
            filter_sensitivities(dc_drho);
            
            // Update design (OC or MMA)
            rho = optimality_criteria_update(rho, dc_drho);
            
            // Check convergence
            double change = norm(rho - rho_old) / norm(rho);
            if (change < tolerance) break;
            
            rho_old = rho;
        }
    }
};
```

## Performance Optimizations

### Adaptive Time Stepping
```cpp
class AdaptiveTimeController {
    double compute_time_step(const State& state,
                            double dt_previous) {
        // Error estimation
        double error = estimate_temporal_error(state);
        
        // Optimal time step
        double dt_optimal = dt_previous * 
            pow(tolerance / error, 1.0 / (order + 1));
        
        // Apply safety factors and limits
        dt_optimal *= safety_factor;
        dt_optimal = std::clamp(dt_optimal, dt_min, dt_max);
        
        return dt_optimal;
    }
};
```

### Predictor Recycling
```cpp
class PredictorCache {
    // Reuse converged solution as predictor
    Vector predict_next_step(const History& history) {
        if (history.size() >= 2) {
            // Polynomial extrapolation
            return polynomial_extrapolate(history);
        }
        return history.back().solution;
    }
};
```

## 🔌 Integration Points

### With physics/
- Calls physics modules for residuals/Jacobians
- Manages physics state updates
- Coordinates multi-physics

### With solvers/
- Uses linear/nonlinear solvers
- Manages solver parameters
- Handles convergence

### With adaptation/
- Triggers adaptive refinement
- Manages solution transfer
- Controls error tolerance

### With mesh/
- Updates mesh for ALE
- Handles mesh motion
- Manages remeshing

## Success Metrics

1. **Robustness**: > 95% analyses complete successfully
2. **Efficiency**: < 20% overhead vs direct solve
3. **Accuracy**: Meet specified tolerances
4. **Scalability**: Linear scaling with DOFs
5. **Recovery**: > 90% recovery from failures
6. **Adaptivity**: Automatic step size control

## Key Features

1. **Complete Coverage**: All standard analysis types
2. **Advanced Methods**: Arc-length, continuation, optimization
3. **Robust Control**: Automatic stepping and recovery
4. **Workflow Support**: Complex multi-step analyses
5. **Failure Handling**: Automatic recovery strategies
6. **Performance**: Optimized for large-scale problems

This architecture provides comprehensive analysis procedures that orchestrate the entire simulation process, from simple linear statics to complex nonlinear dynamics and optimization problems.
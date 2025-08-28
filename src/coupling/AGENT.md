# AGENT.md - Multiphysics Coupling Module

## Mission
Orchestrate the coupling between different physics domains in multiphysics simulations, providing both monolithic and partitioned coupling strategies, managing interface conditions, field transfers, and convergence of coupled systems.

## Architecture Philosophy
- **Strategy-Flexible**: Support monolithic, staggered, and loosely coupled approaches
- **Conservation-Preserving**: Maintain physical conservation at interfaces
- **Stability-Focused**: Ensure numerical stability of coupled systems
- **Scale-Independent**: From tight coupling to weak interaction
- **Parallel-Ready**: Distributed coupling across domains

## Directory Structure

```
coupling/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── base/                            # Core coupling infrastructure
│   ├── coupling_strategy.hpp       # Base coupling strategy
│   ├── coupling_manager.hpp        # Coupling orchestration
│   ├── physics_interface.hpp       # Physics module interface
│   ├── coupling_topology.hpp       # Coupling graph/dependencies
│   ├── convergence_monitor.hpp     # Coupled convergence checks
│   ├── coupling_statistics.hpp     # Performance metrics
│   └── coupling_factory.hpp        # Strategy factory
│
├── monolithic/                      # Monolithic coupling
│   ├── monolithic_assembler.hpp    # Single system assembly
│   ├── block_system.hpp            # Block matrix structure
│   ├── jacobian_assembly.hpp       # Full Jacobian construction
│   ├── monolithic_solver.hpp       # Coupled system solver
│   ├── physics_combiner.hpp        # Combine physics modules
│   └── unified_residual.hpp        # Combined residual evaluation
│
├── partitioned/                     # Partitioned/staggered coupling
│   ├── staggered_solver.hpp        # Staggered iteration
│   ├── fixed_point/
│   │   ├── picard_iteration.hpp    # Picard coupling
│   │   ├── block_gauss_seidel.hpp  # Block GS iteration
│   │   └── parallel_splitting.hpp  # Parallel staggered
│   ├── acceleration/
│   │   ├── aitken_relaxation.hpp   # Aitken acceleration
│   │   ├── anderson_acceleration.hpp # Anderson mixing
│   │   ├── quasi_newton.hpp        # Interface quasi-Newton
│   │   ├── broyden.hpp             # Broyden's method
│   │   └── dynamic_relaxation.hpp  # Adaptive relaxation
│   ├── predictor_corrector/
│   │   ├── predictor_base.hpp      # Predictor interface
│   │   ├── linear_predictor.hpp    # Linear extrapolation
│   │   ├── polynomial_predictor.hpp # Higher-order prediction
│   │   └── corrector_strategy.hpp  # Correction strategies
│   └── subcycling/
│       ├── multirate.hpp           # Multi-rate time stepping
│       ├── subcycle_manager.hpp    # Subcycle orchestration
│       └── time_synchronization.hpp # Time sync points
│
├── interface/                       # Interface management
│   ├── coupling_interface.hpp      # Interface abstraction
│   ├── interface_mesh.hpp          # Interface discretization
│   ├── interface_physics.hpp       # Interface physics (contact, etc.)
│   ├── interface_conditions/
│   │   ├── dirichlet_neumann.hpp   # DN coupling
│   │   ├── neumann_neumann.hpp     # NN coupling  
│   │   ├── robin_robin.hpp         # RR coupling
│   │   └── mortar_interface.hpp    # Mortar methods
│   ├── geometric/
│   │   ├── surface_coupling.hpp    # Surface interfaces
│   │   ├── volume_coupling.hpp     # Volume coupling
│   │   ├── point_coupling.hpp      # Point constraints
│   │   └── line_coupling.hpp       # Line interfaces
│   └── detection/
│       ├── interface_detection.hpp  # Automatic detection
│       ├── proximity_search.hpp     # Proximity-based
│       └── intersection_finder.hpp  # Intersection detection
│
├── field_transfer/                  # Field transfer operators
│   ├── transfer_operator.hpp       # Transfer operator base
│   ├── interpolation/
│   │   ├── consistent_interp.hpp   # Consistent interpolation
│   │   ├── conservative_interp.hpp # Conservative transfer
│   │   ├── radial_basis.hpp        # RBF interpolation
│   │   ├── nearest_neighbor.hpp    # Nearest projection
│   │   └── weighted_residual.hpp   # Weighted residual
│   ├── projection/
│   │   ├── l2_projection.hpp       # L2 projection
│   │   ├── h1_projection.hpp       # H1 projection
│   │   ├── mortar_projection.hpp   # Mortar projection
│   │   └── common_refinement.hpp   # Common refinement
│   ├── conservation/
│   │   ├── force_conservation.hpp  # Force balance
│   │   ├── energy_conservation.hpp # Energy conservation
│   │   ├── mass_conservation.hpp   # Mass conservation
│   │   └── momentum_conservation.hpp # Momentum balance
│   └── nonmatching/
│       ├── nonmatching_transfer.hpp # Non-matching meshes
│       ├── octree_search.hpp       # Octree-based search
│       └── mapping_cache.hpp       # Cache mappings
│
├── convergence/                     # Convergence criteria
│   ├── convergence_criteria.hpp    # Base criteria
│   ├── residual_convergence.hpp    # Residual-based
│   ├── interface_convergence.hpp   # Interface quantities
│   ├── relative_convergence.hpp    # Relative measures
│   ├── absolute_convergence.hpp    # Absolute tolerance
│   └── combined_convergence.hpp    # Multiple criteria
│
├── stability/                       # Stability enhancement
│   ├── stability_analysis.hpp      # Stability assessment
│   ├── added_mass.hpp              # Added mass treatment
│   ├── energy_stability.hpp        # Energy-based stability
│   ├── implicit_coupling.hpp       # Implicit treatment
│   └── stabilization_parameter.hpp # Stabilization params
│
├── specific/                        # Specific coupling types
│   ├── fsi/                       # Fluid-Structure Interaction
│   │   ├── fsi_coupling.hpp       # FSI manager
│   │   ├── ale_fsi.hpp            # ALE-based FSI
│   │   ├── immersed_boundary.hpp  # IB methods
│   │   ├── added_mass_fsi.hpp     # Added mass handling
│   │   └── mesh_motion_fsi.hpp    # Mesh motion
│   ├── thermal_mechanical/
│   │   ├── thermomechanical.hpp   # Thermal-mechanical
│   │   ├── thermal_stress.hpp     # Thermal stresses
│   │   └── heat_generation.hpp    # Mechanical heat
│   ├── electromagnetic/
│   │   ├── em_thermal.hpp         # EM-thermal coupling
│   │   ├── magnetomechanical.hpp  # Magneto-mechanical
│   │   └── joule_heating.hpp      # Joule heating
│   ├── chemical/
│   │   ├── reaction_diffusion.hpp # Reaction-diffusion
│   │   ├── thermochemical.hpp     # Thermo-chemical
│   │   └── chemo_mechanical.hpp   # Chemo-mechanical
│   └── multiscale/
│       ├── scale_bridging.hpp     # Scale bridging
│       ├── heterogeneous.hpp      # Heterogeneous coupling
│       └── homogenization.hpp     # Homogenization
│
├── time_integration/               # Coupled time integration
│   ├── coupled_timestepper.hpp    # Coupled time stepping
│   ├── synchronization.hpp        # Time synchronization
│   ├── implicit_explicit.hpp      # IMEX schemes
│   ├── waveform_iteration.hpp     # Waveform relaxation
│   └── space_time_coupling.hpp    # Space-time methods
│
├── optimization/                    # Coupling optimization
│   ├── parameter_optimization.hpp  # Optimize coupling params
│   ├── relaxation_factor.hpp      # Optimal relaxation
│   ├── interface_placement.hpp    # Optimal interfaces
│   └── coupling_strength.hpp      # Coupling strength analysis
│
├── parallel/                       # Parallel coupling
│   ├── distributed_coupling.hpp   # Distributed domains
│   ├── parallel_interface.hpp     # Parallel interfaces
│   ├── load_balancing.hpp         # Coupled load balance
│   ├── communication_pattern.hpp  # Optimized comm
│   └── heterogeneous_parallel.hpp # CPU-GPU coupling
│
├── diagnostics/                    # Coupling diagnostics
│   ├── coupling_monitor.hpp       # Real-time monitoring
│   ├── energy_balance.hpp         # Energy tracking
│   ├── conservation_check.hpp     # Conservation verification
│   ├── stability_indicator.hpp    # Stability metrics
│   └── performance_analysis.hpp   # Performance profiling
│
├── utilities/                      # Coupling utilities
│   ├── coupling_graph.hpp         # Dependency graph
│   ├── field_mapper.hpp           # Field mapping utilities
│   ├── interpolation_matrix.hpp   # Interpolation matrices
│   └── coupling_io.hpp            # I/O for coupled problems
│
└── tests/                         # Testing
    ├── unit/                      # Unit tests
    ├── convergence/               # Convergence studies
    ├── conservation/              # Conservation tests
    └── benchmarks/                # Coupling benchmarks
```

## Key Components

### 1. Coupling Manager
```cpp
// Main coupling orchestrator
class CouplingManager {
    std::vector<std::unique_ptr<PhysicsModule>> physics;
    std::unique_ptr<CouplingStrategy> strategy;
    std::vector<CouplingInterface> interfaces;
    ConvergenceMonitor convergence;
    
public:
    // Solve coupled problem
    Solution solve(const CoupledProblem& problem) {
        // Initialize physics modules
        for (auto& phys : physics) {
            phys->initialize(problem);
        }
        
        // Setup coupling interfaces
        detect_interfaces();
        build_transfer_operators();
        
        // Time loop
        while (!problem.finished()) {
            // Coupled solve at current time
            strategy->solve_coupled_step(physics, interfaces);
            
            // Check convergence
            if (!convergence.check(physics)) {
                adapt_coupling_parameters();
            }
            
            // Advance time
            problem.advance();
        }
        
        return collect_solution();
    }
};
```

### 2. Monolithic Coupling
```cpp
// Fully coupled monolithic system
class MonolithicCoupling : public CouplingStrategy {
    // Assemble coupled Jacobian
    void assemble_jacobian(BlockMatrix& J,
                          const std::vector<PhysicsModule*>& physics) {
        int n_fields = physics.size();
        
        // Diagonal blocks (single physics)
        for (int i = 0; i < n_fields; ++i) {
            J.block(i, i) = physics[i]->compute_jacobian();
        }
        
        // Off-diagonal blocks (coupling terms)
        for (int i = 0; i < n_fields; ++i) {
            for (int j = 0; j < n_fields; ++j) {
                if (i != j && has_coupling(i, j)) {
                    J.block(i, j) = compute_coupling_jacobian(
                        physics[i], physics[j]
                    );
                }
            }
        }
    }
    
    // Solve monolithic system
    Solution solve_coupled_step(
        std::vector<PhysicsModule*>& physics,
        const std::vector<CouplingInterface>& interfaces) override {
        
        // Build monolithic system
        BlockMatrix J;
        BlockVector F;
        assemble_jacobian(J, physics);
        assemble_residual(F, physics);
        
        // Apply interface conditions
        for (auto& interface : interfaces) {
            interface.apply_to_system(J, F);
        }
        
        // Solve
        BlockVector delta_u;
        monolithic_solver.solve(J, F, delta_u);
        
        // Update physics states
        update_solutions(physics, delta_u);
        
        return extract_solution(physics);
    }
};
```

### 3. Partitioned Coupling with Quasi-Newton
```cpp
// Interface quasi-Newton for strong coupling
class QuasiNewtonCoupling : public PartitionedCoupling {
    Matrix V, W;  // Quasi-Newton matrices
    int reuse_vectors = 4;
    
    Solution solve_coupled_step(
        std::vector<PhysicsModule*>& physics,
        const std::vector<CouplingInterface>& interfaces) override {
        
        auto& fluid = *physics[0];
        auto& structure = *physics[1];
        auto& interface = interfaces[0];
        
        // Initial interface displacement
        Vector d = interface.get_displacement();
        Vector d_old = d;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Solve fluid with interface displacement
            fluid.set_boundary_displacement(d);
            fluid.solve();
            
            // Get interface forces
            Vector f = fluid.get_interface_forces();
            
            // Solve structure with interface forces
            structure.set_boundary_forces(f);
            structure.solve();
            
            // Get new displacement
            Vector d_new = structure.get_interface_displacement();
            
            // Compute residual
            Vector r = d_new - d;
            
            // Check convergence
            if (norm(r) < tolerance) {
                return extract_solution(physics);
            }
            
            // Quasi-Newton update
            if (iter > 0) {
                Vector delta_r = r - r_old;
                Vector delta_d = d - d_old;
                
                // Update quasi-Newton approximation
                update_quasi_newton(V, W, delta_d, delta_r);
                
                // Compute new displacement
                d = d_new - apply_inverse_jacobian(V, W, r);
            } else {
                // Simple relaxation for first iteration
                d = 0.5 * (d + d_new);
            }
            
            d_old = d;
            r_old = r;
        }
        
        return extract_solution(physics);
    }
    
private:
    void update_quasi_newton(Matrix& V, Matrix& W,
                            const Vector& delta_d,
                            const Vector& delta_r) {
        // Limited memory quasi-Newton (IQN-ILS)
        if (V.cols() >= reuse_vectors) {
            // Remove oldest column
            V.remove_column(0);
            W.remove_column(0);
        }
        
        // Add new information
        V.add_column(delta_d);
        W.add_column(delta_r);
    }
};
```

### 4. Field Transfer with Conservation
```cpp
// Conservative field transfer between meshes
class ConservativeTransfer : public FieldTransfer {
    // Build transfer operator with conservation
    Matrix build_transfer_operator(const Mesh& source_mesh,
                                  const Mesh& target_mesh,
                                  const Interface& interface) {
        Matrix T(target_mesh.n_dofs(), source_mesh.n_dofs());
        
        // Build consistent interpolation
        for (auto& target_elem : target_mesh.interface_elements()) {
            for (auto& qp : quadrature_points(target_elem)) {
                // Find source element containing qp
                auto source_elem = source_mesh.locate(qp.coords);
                
                // Evaluate shape functions
                auto N_target = target_elem.shape_functions(qp);
                auto N_source = source_elem.shape_functions(
                    source_elem.to_local(qp.coords)
                );
                
                // Add contribution to transfer matrix
                for (int i : target_elem.dofs()) {
                    for (int j : source_elem.dofs()) {
                        T(i, j) += N_target[i] * N_source[j] * qp.weight;
                    }
                }
            }
        }
        
        // Ensure conservation
        enforce_conservation(T, source_mesh, target_mesh);
        
        return T;
    }
    
    void enforce_conservation(Matrix& T,
                             const Mesh& source,
                             const Mesh& target) {
        // Compute row sums
        Vector row_sums(T.rows());
        for (int i = 0; i < T.rows(); ++i) {
            row_sums[i] = T.row(i).sum();
        }
        
        // Normalize for conservation
        double source_integral = compute_interface_integral(source);
        double target_integral = compute_interface_integral(target);
        double scale = source_integral / target_integral;
        
        T *= scale;
    }
};
```

### 5. FSI-Specific Coupling
```cpp
// Fluid-structure interaction coupling
class FSICoupling : public CouplingStrategy {
    ALEMesh fluid_mesh;
    bool use_added_mass_prediction = true;
    
    Solution solve_coupled_step(FluidSolver& fluid,
                               StructuralSolver& structure,
                               const FSIInterface& interface) {
        // Predict interface motion (for added mass)
        Vector d_pred;
        if (use_added_mass_prediction) {
            d_pred = predict_displacement(structure, interface);
        } else {
            d_pred = interface.get_displacement();
        }
        
        // Move fluid mesh (ALE)
        fluid_mesh.move_boundary(d_pred);
        fluid_mesh.update_interior();
        
        // Compute mesh velocity
        Vector w_mesh = fluid_mesh.compute_velocity(dt);
        
        // Solve fluid with moving mesh
        fluid.set_mesh_velocity(w_mesh);
        fluid.solve();
        
        // Get interface forces
        Vector f = fluid.compute_interface_forces(interface);
        
        // Apply under-relaxation for stability
        if (requires_relaxation(f)) {
            f = relax_forces(f);
        }
        
        // Solve structure
        structure.set_interface_forces(f);
        structure.solve();
        
        // Get actual displacement
        Vector d = structure.get_interface_displacement();
        
        // Check mesh quality
        if (!fluid_mesh.check_quality()) {
            remesh_fluid_domain();
        }
        
        return {fluid.solution(), structure.solution()};
    }
};
```

### 6. Multirate Time Integration
```cpp
// Different time steps for different physics
class MultirateCoupling : public CouplingStrategy {
    struct PhysicsTimeStep {
        PhysicsModule* physics;
        double dt;
        int subcycles;
    };
    
    void solve_coupled_step(std::vector<PhysicsTimeStep>& physics,
                           double dt_global) {
        // Find synchronization points
        double t_sync = compute_sync_time(physics, dt_global);
        
        while (current_time < t_sync) {
            // Fast physics (e.g., fluid)
            for (auto& fast : get_fast_physics(physics)) {
                for (int sub = 0; sub < fast.subcycles; ++sub) {
                    // Interpolate slow physics to current time
                    auto slow_state = interpolate_slow_physics(
                        current_time + sub * fast.dt
                    );
                    
                    // Set interface conditions from slow physics
                    fast.physics->set_interface_state(slow_state);
                    
                    // Advance fast physics
                    fast.physics->advance(fast.dt);
                }
            }
            
            // Slow physics (e.g., structure)
            for (auto& slow : get_slow_physics(physics)) {
                // Average fast physics over interval
                auto fast_avg = average_fast_physics(slow.dt);
                
                // Set averaged interface conditions
                slow.physics->set_interface_state(fast_avg);
                
                // Advance slow physics
                slow.physics->advance(slow.dt);
            }
            
            current_time += dt_global;
        }
    }
};
```

### 7. Stability Enhancement
```cpp
// Added mass treatment for FSI
class AddedMassStabilization {
    // Estimate added mass effect
    Matrix estimate_added_mass(const FluidSolver& fluid,
                              const Interface& interface) {
        // Approximate added mass matrix
        Matrix M_a(interface.n_dofs(), interface.n_dofs());
        
        // Use potential flow approximation
        for (auto& face : interface.faces()) {
            double area = face.area();
            Vector normal = face.normal();
            
            // Added mass coefficient (depends on geometry)
            double C_a = compute_added_mass_coefficient(face);
            
            // Local added mass contribution
            Matrix M_local = C_a * fluid.density() * area * 
                           outer_product(normal, normal);
            
            // Assemble to interface matrix
            assemble_local(M_a, face.dofs(), M_local);
        }
        
        return M_a;
    }
    
    // Robin-Neumann scheme for stability
    void apply_robin_neumann(Vector& interface_force,
                            const Vector& displacement,
                            const Matrix& M_a,
                            double alpha) {
        // Modify interface condition for stability
        interface_force -= alpha * M_a * displacement;
    }
};
```

## Performance Optimizations

### Prediction for Fast Convergence
```cpp
class CouplingPredictor {
    // Polynomial extrapolation
    Vector predict_interface_state(const std::vector<Vector>& history,
                                  double t_target) {
        int order = std::min(3, int(history.size()) - 1);
        
        // Lagrange extrapolation
        Vector prediction(history[0].size(), 0.0);
        for (int i = 0; i <= order; ++i) {
            double L_i = lagrange_basis(i, order, t_target);
            prediction += L_i * history[history.size() - 1 - i];
        }
        
        return prediction;
    }
};
```

### Reuse of Coupling Information
```cpp
// Cache transfer operators
class TransferCache {
    std::unordered_map<std::pair<MeshID, MeshID>, Matrix> operators;
    
    const Matrix& get_transfer_operator(const Mesh& source,
                                       const Mesh& target) {
        auto key = std::make_pair(source.id(), target.id());
        
        if (operators.find(key) == operators.end()) {
            operators[key] = build_transfer_operator(source, target);
        }
        
        return operators[key];
    }
};
```

## Integration Points

### With physics/
- Receives physics modules to couple
- Calls physics residual/Jacobian evaluations
- Manages physics state updates

### With mesh/
- Handles interface mesh generation
- Manages mesh motion for ALE
- Performs mesh-to-mesh transfers

### With solvers/
- Uses linear/nonlinear solvers for coupled systems
- Leverages block preconditioners
- Employs specialized saddle-point solvers

### With assembly/
- Assembles coupled system matrices
- Handles interface constraints
- Manages block structure

## Success Metrics

1. **Convergence Rate**: < 10 iterations for partitioned
2. **Conservation Error**: < 1e-12 for conserved quantities
3. **Stability**: No spurious oscillations at interfaces
4. **Parallel Efficiency**: > 85% for distributed coupling
5. **Transfer Accuracy**: 2nd order for linear fields
6. **Time Step**: Allow CFL > 1 for implicit coupling

## Key Features

1. **Strategy Flexibility**: Monolithic and partitioned options
2. **Strong Coupling**: Quasi-Newton and other acceleration methods
3. **Conservation**: Guaranteed conservation at interfaces
4. **Stability**: Added mass treatment and stabilization
5. **Multirate**: Different time steps for different physics
6. **Parallel**: Distributed domain coupling

This architecture provides comprehensive multiphysics coupling capabilities, from tightly coupled monolithic systems to loosely coupled partitioned approaches, with robust interface handling and conservation guarantees.
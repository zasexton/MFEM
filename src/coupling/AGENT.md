# AGENT.md - Multiphysics Coupling Module

## Mission
Orchestrate the coupling between different physics domains in multiphysics simulations, providing both monolithic and partitioned coupling strategies, managing interface conditions, field transfers, and convergence of coupled systems.

## Architecture Philosophy
- **Physics-Agnostic**: Couples any physics modules through standard interfaces
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
│   ├── physics_registry.hpp        # Registry of available physics from physics/
│   ├── coupling_topology.hpp       # Coupling graph/dependencies
│   ├── convergence_monitor.hpp     # Coupled convergence checks
│   └── coupling_factory.hpp        # Strategy factory
│
├── strategies/                      # Solution strategies
│   ├── monolithic/
│   │   ├── monolithic_assembler.hpp    # Single system assembly
│   │   ├── block_system.hpp            # Block matrix structure
│   │   └── jacobian_coupler.hpp        # Cross-Jacobian terms
│   ├── partitioned/
│   │   ├── staggered_solver.hpp        # Staggered iteration
│   │   ├── picard_iteration.hpp        # Fixed-point iteration
│   │   ├── block_gauss_seidel.hpp      # Block GS iteration
│   │   └── parallel_splitting.hpp      # Parallel staggered
│   └── acceleration/
│       ├── aitken_relaxation.hpp       # Aitken acceleration
│       ├── anderson_acceleration.hpp    # Anderson mixing
│       ├── quasi_newton.hpp            # Interface quasi-Newton
│       └── dynamic_relaxation.hpp      # Adaptive relaxation
│
├── interface/                       # Interface management
│   ├── interface_detector.hpp      # Automatic interface detection
│   ├── interface_mesh.hpp          # Interface discretization
│   ├── coupling_conditions/
│   │   ├── dirichlet_neumann.hpp   # DN coupling
│   │   ├── neumann_neumann.hpp     # NN coupling  
│   │   ├── robin_robin.hpp         # RR coupling
│   │   └── mortar_interface.hpp    # Mortar methods
│   └── geometric/
│       ├── surface_interface.hpp   # Surface coupling
│       ├── volume_overlap.hpp      # Volume coupling
│       └── point_constraint.hpp    # Point coupling
│
├── transfer/                        # Field transfer operators
│   ├── transfer_operator.hpp       # Transfer operator base
│   ├── interpolation/
│   │   ├── consistent_interp.hpp   # Consistent interpolation
│   │   ├── conservative_interp.hpp # Conservative transfer
│   │   ├── radial_basis.hpp        # RBF interpolation
│   │   └── weighted_residual.hpp   # Weighted residual
│   ├── projection/
│   │   ├── l2_projection.hpp       # L2 projection
│   │   ├── h1_projection.hpp       # H1 projection
│   │   └── common_refinement.hpp   # Common refinement
│   └── conservation/
│       ├── force_balance.hpp       # Force conservation
│       ├── energy_balance.hpp      # Energy conservation
│       └── mass_conservation.hpp    # Mass conservation
│
├── convergence/                     # Convergence criteria
│   ├── residual_convergence.hpp    # Residual-based
│   ├── interface_convergence.hpp   # Interface quantities
│   └── combined_criteria.hpp       # Multiple criteria
│
├── stability/                       # Stability enhancement
│   ├── stability_analysis.hpp      # Stability assessment
│   ├── added_mass_treatment.hpp    # Added mass for FSI
│   └── implicit_treatment.hpp      # Implicit coupling
│
├── configurations/                  # Pre-configured physics combinations
│   ├── thermomechanical.hpp       # Heat + LinearElasticity
│   ├── fsi.hpp                    # NavierStokes + Elasticity
│   ├── electromagnetic_thermal.hpp # Maxwell + HeatConduction
│   ├── fluid_porous.hpp           # NavierStokes + DarcyFlow
│   └── reaction_transport.hpp      # AdvectionDiffusion + ChemicalReaction
│
├── time_integration/               # Coupled time integration
│   ├── synchronized_stepping.hpp   # Same dt for all physics
│   ├── subcycling.hpp             # Different dt per physics
│   └── waveform_iteration.hpp     # Waveform relaxation
│
├── parallel/                       # Parallel coupling
│   ├── distributed_coupling.hpp   # Distributed domains
│   ├── parallel_interface.hpp     # Parallel interfaces
│   └── load_balancing.hpp         # Coupled load balance
│
├── diagnostics/                    # Coupling diagnostics
│   ├── coupling_monitor.hpp       # Real-time monitoring
│   ├── conservation_check.hpp     # Conservation verification
│   └── performance_analysis.hpp   # Performance profiling
│
├── utilities/                      # Coupling utilities
│   ├── coupling_graph.hpp         # Dependency graph
│   └── field_mapper.hpp           # Field mapping utilities
│
└── tests/                         # Testing
    ├── unit/                      # Unit tests
    ├── convergence/               # Convergence studies
    └── benchmarks/                # Coupling benchmarks
```

## Key Components

### 1. Coupling Manager
```cpp
// Main coupling orchestrator that combines physics from physics/
class CouplingManager {
    // Physics modules loaded from physics/ subfolder
    std::vector<std::unique_ptr<PhysicsModule>> physics_modules;
    std::unique_ptr<CouplingStrategy> strategy;
    std::vector<CouplingInterface> interfaces;
    
public:
    // Add physics modules from physics/ subfolder
    template<typename PhysicsType>
    void add_physics(const PhysicsParameters& params) {
        // PhysicsType comes from physics/ (e.g., LinearElasticity, NavierStokes)
        physics_modules.push_back(
            std::make_unique<PhysicsType>(params)
        );
    }
    
    // Configure how physics interact
    void setup_coupling(const CouplingConfig& config) {
        // Detect interfaces between physics domains
        detect_interfaces();
        
        // Build transfer operators between meshes
        build_transfer_operators();
        
        // Set coupling strategy
        strategy = create_strategy(config.type);
    }
    
    // Solve the coupled problem
    Solution solve(const CoupledProblem& problem) {
        // Let strategy orchestrate the physics modules
        return strategy->solve_coupled_step(physics_modules, interfaces);
    }
};
```

### 2. Physics Registry - Interface to physics
```cpp
// Registry of available physics modules from physics/ subfolder
class PhysicsRegistry {
    std::map<std::string, PhysicsFactory> factories;
    
public:
    void register_physics() {
        // Register all physics from physics/ subfolder
        register<mechanics::LinearElasticity>("linear_elasticity");
        register<fluid::NavierStokes>("navier_stokes");
        register<thermal::HeatConduction>("heat_conduction");
        register<electromagnetic::Maxwell>("maxwell");
        // ... etc
    }
    
    std::unique_ptr<PhysicsModule> create(
        const std::string& type,
        const Parameters& params) {
        return factories[type](params);
    }
};
```

### 3. Configuration Examples - Pre-defined combinations
```cpp
// configurations/thermomechanical.hpp
// Combines physics from physics/thermal/ and physics/mechanics/
class ThermomechanicalConfig {
public:
    void setup(CouplingManager& manager) {
        // Add physics modules from physics/ subfolder
        manager.add_physics<thermal::HeatConduction>(thermal_params);
        manager.add_physics<mechanics::LinearElasticity>(elastic_params);
        
        // Configure coupling
        manager.set_strategy<MonolithicCoupling>();
        
        // Define interface: temperature affects thermal strain
        manager.add_interface(
            ThermalStrainInterface{
                .from = "temperature",
                .to = "thermal_strain"
            }
        );
    }
};

// configurations/fsi.hpp  
// Combines physics from physics/fluid/ and physics/mechanics/
class FSIConfig {
public:
    void setup(CouplingManager& manager) {
        // Physics from respective subfolders
        manager.add_physics<fluid::NavierStokes>(fluid_params);
        manager.add_physics<mechanics::FiniteStrain>(solid_params);
        
        // Use partitioned coupling with quasi-Newton
        manager.set_strategy<PartitionedCoupling>(
            QuasiNewtonAcceleration{.reuse_vectors = 4}
        );
        
        // Configure interface conditions
        manager.add_interface(
            FSIInterface{
                .condition = DirichletNeumann{},
                .added_mass_treatment = true
            }
        );
    }
};
```

### 4. Coupling Strategy- Orchestrates physics 
```cpp
// Monolithic strategy assembles all physics into one system
class MonolithicCoupling : public CouplingStrategy {
    Solution solve_coupled_step(
        std::vector<PhysicsModule*>& physics,
        const std::vector<CouplingInterface>& interfaces) override {
        
        // Build block system from individual physics
        BlockMatrix J;
        BlockVector F;
        
        // Diagonal blocks - each physics contributes its Jacobian
        for (int i = 0; i < physics.size(); ++i) {
            J.block(i, i) = physics[i]->compute_jacobian();
            F.block(i) = physics[i]->compute_residual();
        }
        
        // Off-diagonal blocks - coupling terms
        for (auto& interface : interfaces) {
            auto [i, j] = interface.physics_indices();
            J.block(i, j) = compute_coupling_jacobian(
                physics[i], physics[j], interface
            );
        }
        
        // Solve monolithic system
        BlockVector delta_u = solver.solve(J, F);
        
        // Update each physics
        for (int i = 0; i < physics.size(); ++i) {
            physics[i]->update_solution(delta_u.block(i));
        }
        
        return collect_solution(physics);
    }
};
```

### 5. Interface Detection and Transfer
```cpp
// Automatically detect interfaces between physics domains
class InterfaceDetector {
    std::vector<CouplingInterface> detect(
        const std::vector<PhysicsModule*>& physics) {
        
        std::vector<CouplingInterface> interfaces;
        
        for (int i = 0; i < physics.size(); ++i) {
            for (int j = i+1; j < physics.size(); ++j) {
                // Check if physics domains share boundaries
                if (shares_boundary(physics[i]->domain(), 
                                  physics[j]->domain())) {
                    
                    // Create appropriate interface
                    interfaces.push_back(
                        create_interface(physics[i], physics[j])
                    );
                }
            }
        }
        return interfaces;
    }
};
```

### 6. Field Transfer Between Physics
```cpp
// Transfer fields between different physics discretizations
class FieldTransfer {
    void transfer(const PhysicsModule* from,
                  PhysicsModule* to,
                  const std::string& field_name) {
        
        // Get field from source physics
        auto source_field = from->get_field(field_name);
        auto source_mesh = from->mesh();
        
        // Get target mesh from destination physics
        auto target_mesh = to->mesh();
        
        // Build transfer operator (cached for efficiency)
        auto T = build_transfer_operator(source_mesh, target_mesh);
        
        // Transfer and set field in target physics
        auto target_field = T * source_field;
        to->set_external_field(field_name, target_field);
    }
};
```

## Integration Points

### With physics/
- The coupling module is a client of the `physics/` module
- Receives physics modules to couple
- Determines when and how physics modules run
- Transfers fields between physics
- Never implements PDEs or constitutive models 
- Provides recipes for common physics combinations
- calls `compute_residual()`, `compute_jacobian()`, `update_state()`

### With mesh/
- Handles interface mesh generation
- Manages mesh motion for ALE
- Manages mesh-to-mesh field transfers
- Performs interface detection based on mesh topology

### With solvers/
- Uses linear/nonlinear solvers for coupled systems
- Leverages block preconditioners
- Manages solver sequence for partitioned schemes

### With assembly/
- Assembles coupled system matrices
- Handles interface constraints
- Manages block structure

## Success Metrics

1. **Modularity**: Any physics from physics/ can be coupled
2. **Convergence**: < 10 iterations for strongly coupled problems
3. **Conservation**: < 1e-12 violation at interfaces
4. **Stability**: No spurious oscillations at interfaces
5. **Parallel Efficiency**: > 85% for distributed coupling
6. **Configuation Simplicity**: < 20 lines to configure standard problems
7. **Transfer Accuracy**: 2nd order for linear fields
8. **Time Step**: Allow CFL > 1 for implicit coupling

## Key Features

1. **Pure Orchestration**: No physics implementation, only coordination
2. **Physics Agnostic**: Works with any PhysicsModule implementation
3. **Strategy Flexibility**: Multiple coupling algorithms available
4. **Pre-configured**: Common multiphysics problems ready to use
5. **Conservation Guaranteed**: Enforces conservation at interfaces
6. **Parallel**: Distributed domain coupling

This architecture provides comprehensive multiphysics coupling capabilities, from tightly coupled monolithic systems to loosely coupled partitioned approaches, with robust interface handling and conservation guarantees.
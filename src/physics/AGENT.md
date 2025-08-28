# AGENT.md - Physics Modules

## Mission
Implement domain-specific physics formulations for finite element analysis, providing modular, extensible physics components that can be combined for multiphysics simulations while maintaining clear interfaces with the FEM infrastructure.

## Architecture Philosophy
- **Formulation-Focused**: Each physics module implements specific PDEs
- **Component-Based**: Physics as pluggable components via ECS
- **Coupling-Ready**: Clean interfaces for multiphysics interaction
- **Material-Flexible**: Support for linear and nonlinear constitutive models
- **Verification-Driven**: Built-in manufactured solutions and benchmarks

## Directory Structure

```
physics/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── base/                            # Common physics infrastructure
│   ├── physics_module.hpp          # Base physics interface
│   ├── weak_form.hpp               # Weak formulation base
│   ├── constitutive_model.hpp      # Material model interface
│   ├── field_variable.hpp          # Physics field definitions
│   ├── physics_traits.hpp          # Physics type traits
│   ├── conservation_law.hpp        # Conservation principles
│   └── physics_factory.hpp         # Physics module factory
│
├── mechanics/                       # Solid/structural mechanics
│   ├── solid/
│   │   ├── linear_elasticity.hpp   # Linear elastic
│   │   ├── finite_strain.hpp       # Large deformation
│   │   ├── hyperelasticity.hpp     # Hyperelastic models
│   │   ├── plasticity/
│   │   │   ├── j2_plasticity.hpp   # Von Mises plasticity
│   │   │   ├── mohr_coulomb.hpp    # Mohr-Coulomb
│   │   │   ├── drucker_prager.hpp  # Drucker-Prager
│   │   │   └── crystal_plasticity.hpp # Crystal plasticity
│   │   ├── viscoelasticity/
│   │   │   ├── maxwell.hpp         # Maxwell model
│   │   │   ├── kelvin_voigt.hpp    # Kelvin-Voigt
│   │   │   └── generalized_maxwell.hpp
│   │   ├── damage/
│   │   │   ├── scalar_damage.hpp   # Isotropic damage
│   │   │   ├── anisotropic_damage.hpp
│   │   │   └── phase_field_fracture.hpp
│   │   └── constitutive/
│   │       ├── neo_hookean.hpp     # Neo-Hookean
│   │       ├── mooney_rivlin.hpp   # Mooney-Rivlin
│   │       └── ogden.hpp           # Ogden model
│   ├── structural/
│   │   ├── beam/
│   │   │   ├── euler_bernoulli.hpp # Euler-Bernoulli beam
│   │   │   ├── timoshenko.hpp      # Timoshenko beam
│   │   │   └── nonlinear_beam.hpp  # Geometrically nonlinear
│   │   ├── plate/
│   │   │   ├── kirchhoff_love.hpp  # Thin plate
│   │   │   ├── mindlin_reissner.hpp # Thick plate
│   │   │   └── von_karman.hpp      # Von Karman plate
│   │   ├── shell/
│   │   │   ├── linear_shell.hpp    # Linear shell
│   │   │   ├── nonlinear_shell.hpp # Nonlinear shell
│   │   │   └── composite_shell.hpp # Layered composites
│   │   └── membrane/
│   │       └── membrane.hpp        # Membrane elements
│   ├── contact/
│   │   ├── contact_mechanics.hpp   # Contact base
│   │   ├── penalty_contact.hpp     # Penalty method
│   │   ├── lagrange_contact.hpp    # Lagrange multipliers
│   │   ├── augmented_lagrange.hpp  # Augmented Lagrangian
│   │   ├── mortar_contact.hpp      # Mortar methods
│   │   └── friction/
│   │       ├── coulomb_friction.hpp # Coulomb friction
│   │       └── adhesion.hpp        # Adhesive contact
│   └── dynamics/
│       ├── elastodynamics.hpp      # Dynamic elasticity
│       ├── wave_propagation.hpp    # Wave equations
│       └── modal_analysis.hpp      # Modal dynamics
│
├── thermal/                         # Heat transfer
│   ├── conduction/
│   │   ├── steady_heat.hpp         # Steady-state heat
│   │   ├── transient_heat.hpp      # Transient heat
│   │   ├── nonlinear_heat.hpp      # Temperature-dependent
│   │   └── anisotropic_heat.hpp    # Anisotropic conduction
│   ├── convection/
│   │   ├── forced_convection.hpp   # Forced convection
│   │   ├── natural_convection.hpp  # Natural convection
│   │   └── mixed_convection.hpp    # Mixed convection
│   ├── radiation/
│   │   ├── surface_radiation.hpp   # Surface-to-surface
│   │   ├── participating_media.hpp # Participating media
│   │   └── view_factors.hpp        # View factor calculation
│   └── phase_change/
│       ├── melting_solidification.hpp # Phase change
│       └── stefan_problem.hpp      # Stefan problem
│
├── fluid/                           # Fluid dynamics
│   ├── incompressible/
│   │   ├── stokes.hpp              # Stokes flow
│   │   ├── navier_stokes.hpp       # Incompressible NS
│   │   ├── stabilized/
│   │   │   ├── supg.hpp            # SUPG stabilization
│   │   │   ├── pspg.hpp            # PSPG stabilization
│   │   │   └── gls.hpp             # GLS method
│   │   └── turbulence/
│   │       ├── rans/
│   │       │   ├── k_epsilon.hpp   # k-epsilon model
│   │       │   ├── k_omega.hpp     # k-omega model
│   │       │   └── spalart_allmaras.hpp
│   │       ├── les/
│   │       │   ├── smagorinsky.hpp # Smagorinsky
│   │       │   └── dynamic_les.hpp # Dynamic LES
│   │       └── dns.hpp             # Direct simulation
│   ├── compressible/
│   │   ├── euler.hpp               # Euler equations
│   │   ├── compressible_ns.hpp     # Compressible NS
│   │   └── shock_capturing.hpp     # Shock capturing
│   ├── porous/
│   │   ├── darcy.hpp               # Darcy flow
│   │   ├── brinkman.hpp            # Brinkman equations
│   │   └── richards.hpp            # Richards equation
│   └── multiphase/
│       ├── level_set.hpp           # Level set method
│       ├── vof.hpp                 # Volume of fluid
│       └── phase_field_flow.hpp    # Phase field
│
├── electromagnetic/                 # Electromagnetics
│   ├── electrostatics/
│   │   ├── laplace.hpp             # Laplace equation
│   │   ├── poisson.hpp             # Poisson equation
│   │   └── dielectric.hpp          # Dielectric materials
│   ├── magnetostatics/
│   │   ├── vector_potential.hpp    # Vector potential
│   │   ├── scalar_potential.hpp    # Scalar potential
│   │   └── nonlinear_magnetics.hpp # Nonlinear B-H
│   ├── electrodynamics/
│   │   ├── maxwell.hpp             # Maxwell's equations
│   │   ├── eddy_current.hpp        # Eddy currents
│   │   ├── wave_equation.hpp       # EM waves
│   │   └── time_harmonic.hpp       # Frequency domain
│   └── coupled/
│       ├── magnetohydrodynamics.hpp # MHD
│       └── electrokinetics.hpp     # Electrokinetic flow
│
├── acoustic/                        # Acoustics
│   ├── helmholtz.hpp               # Helmholtz equation
│   ├── wave_equation.hpp           # Acoustic waves
│   ├── aeroacoustics.hpp           # Flow-induced noise
│   └── vibroacoustics.hpp          # Structure-borne sound
│
├── chemical/                        # Chemical/reaction physics
│   ├── diffusion/
│   │   ├── ficks_law.hpp           # Fick's diffusion
│   │   ├── multispecies.hpp        # Multiple species
│   │   └── cross_diffusion.hpp     # Cross effects
│   ├── reaction/
│   │   ├── reaction_kinetics.hpp   # Chemical kinetics
│   │   ├── combustion.hpp          # Combustion models
│   │   └── catalysis.hpp           # Surface reactions
│   └── transport/
│       ├── advection_diffusion.hpp # Advection-diffusion
│       └── reactive_transport.hpp  # Reactive transport
│
├── quantum/                         # Quantum mechanics
│   ├── schrodinger.hpp             # Schrödinger equation
│   ├── density_functional.hpp      # DFT
│   └── tight_binding.hpp           # Tight-binding
│
├── biological/                      # Biological physics
│   ├── biomechanics/
│   │   ├── soft_tissue.hpp         # Soft tissue mechanics
│   │   ├── bone_remodeling.hpp     # Bone adaptation
│   │   └── muscle_contraction.hpp  # Active materials
│   └── biofluid/
│       ├── blood_flow.hpp          # Hemodynamics
│       └── respiratory_flow.hpp    # Lung mechanics
│
├── coupled/                         # Pre-coupled physics
│   ├── thermomechanical/
│   │   ├── thermoelasticity.hpp    # Coupled thermo-elastic
│   │   └── thermoplasticity.hpp    # Thermo-plastic
│   ├── fluid_structure/
│   │   ├── fsi_monolithic.hpp      # Monolithic FSI
│   │   └── fsi_partitioned.hpp     # Partitioned FSI
│   ├── electromechanical/
│   │   ├── piezoelectric.hpp       # Piezoelectricity
│   │   └── magnetostrictive.hpp    # Magnetostriction
│   └── multiscale/
│       ├── fe2.hpp                 # FE² method
│       └── heterogeneous.hpp       # Heterogeneous
│
├── verification/                    # Verification cases
│   ├── manufactured/
│   │   ├── mms_generator.hpp       # MMS generation
│   │   └── mms_problems.hpp        # Standard MMS
│   ├── benchmarks/
│   │   ├── patch_test.hpp          # Patch tests
│   │   ├── cook_membrane.hpp       # Cook's membrane
│   │   ├── lid_driven_cavity.hpp   # Cavity flow
│   │   └── heat_equation_1d.hpp    # 1D heat
│   └── analytical/
│       ├── beam_deflection.hpp     # Analytical beam
│       └── plate_vibration.hpp     # Plate modes
│
├── utilities/                       # Physics utilities
│   ├── units.hpp                   # Unit system
│   ├── material_library.hpp        # Material database
│   ├── dimensionless.hpp           # Dimensionless numbers
│   └── tensor_operations.hpp       # Tensor utilities
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests per physics
    ├── verification/               # Verification tests
    ├── benchmarks/                 # Performance tests
    └── convergence/                # Convergence studies
```

## Key Components

### 1. Base Physics Module
```cpp
// Base interface for all physics
template<int Dim>
class PhysicsModule : public core::Component {
public:
    // Problem dimension
    static constexpr int dimension = Dim;
    
    // Fields this physics solves for
    virtual std::vector<FieldVariable> fields() const = 0;
    
    // Weak form residual
    virtual void compute_residual(const Element& elem,
                                 const Solution& u,
                                 Vector& R_e) = 0;
    
    // Jacobian/tangent
    virtual void compute_jacobian(const Element& elem,
                                 const Solution& u,
                                 Matrix& K_e) = 0;
    
    // Time derivative terms (for dynamics)
    virtual void compute_mass_matrix(const Element& elem,
                                    Matrix& M_e) { M_e.zero(); }
    
    // Source terms
    virtual void compute_source(const Element& elem,
                               const Time& t,
                               Vector& F_e) { F_e.zero(); }
    
    // Boundary conditions
    virtual void apply_natural_bc(const BoundaryElement& face,
                                 const NaturalBC& bc,
                                 Vector& F_e) = 0;
    
    // Material update (for nonlinear)
    virtual void update_material_state(const Element& elem,
                                      const Solution& u) {}
    
    // Post-processing quantities
    virtual void compute_derived_quantities(const Element& elem,
                                          const Solution& u,
                                          DerivedQuantities& q) {}
};
```

### 2. Linear Elasticity Implementation
```cpp
// Linear elastic solid mechanics
template<int Dim>
class LinearElasticity : public PhysicsModule<Dim> {
    ElasticMaterial material;
    
public:
    std::vector<FieldVariable> fields() const override {
        return {FieldVariable("displacement", Dim)};
    }
    
    void compute_residual(const Element& elem,
                         const Solution& u,
                         Vector& R_e) override {
        // Quadrature loop
        for (auto& qp : elem.quadrature_points()) {
            // Strain-displacement matrix
            auto B = compute_B_matrix(elem, qp);
            
            // Strain
            auto epsilon = B * u.local(elem);
            
            // Stress (linear elastic)
            auto sigma = material.C * epsilon;
            
            // Internal forces
            R_e += B.transpose() * sigma * qp.weight * elem.J(qp);
        }
        
        // External forces
        Vector F_ext = compute_external_forces(elem);
        R_e -= F_ext;
    }
    
    void compute_jacobian(const Element& elem,
                         const Solution& u,
                         Matrix& K_e) override {
        // Stiffness matrix
        for (auto& qp : elem.quadrature_points()) {
            auto B = compute_B_matrix(elem, qp);
            auto C = material.elasticity_tensor();
            
            K_e += B.transpose() * C * B * qp.weight * elem.J(qp);
        }
    }
    
private:
    Matrix compute_B_matrix(const Element& elem,
                          const QuadraturePoint& qp) {
        auto dN = elem.shape_derivatives(qp);
        int n_nodes = elem.n_nodes();
        
        Matrix B(voigt_size(), Dim * n_nodes);
        
        // Build strain-displacement matrix
        for (int i = 0; i < n_nodes; ++i) {
            if constexpr (Dim == 2) {
                B(0, 2*i)     = dN(i, 0);  // ε_xx
                B(1, 2*i + 1) = dN(i, 1);  // ε_yy
                B(2, 2*i)     = dN(i, 1);  // γ_xy
                B(2, 2*i + 1) = dN(i, 0);
            } else {
                // 3D case
                B(0, 3*i)     = dN(i, 0);  // ε_xx
                B(1, 3*i + 1) = dN(i, 1);  // ε_yy
                B(2, 3*i + 2) = dN(i, 2);  // ε_zz
                B(3, 3*i + 1) = dN(i, 2);  // γ_yz
                B(3, 3*i + 2) = dN(i, 1);
                B(4, 3*i)     = dN(i, 2);  // γ_xz
                B(4, 3*i + 2) = dN(i, 0);
                B(5, 3*i)     = dN(i, 1);  // γ_xy
                B(5, 3*i + 1) = dN(i, 0);
            }
        }
        
        return B;
    }
};
```

### 3. Incompressible Navier-Stokes
```cpp
// Incompressible flow with stabilization
template<int Dim>
class NavierStokes : public PhysicsModule<Dim> {
    double density;
    double viscosity;
    StabilizationType stab_type;
    
public:
    std::vector<FieldVariable> fields() const override {
        return {
            FieldVariable("velocity", Dim),
            FieldVariable("pressure", 1)
        };
    }
    
    void compute_residual(const Element& elem,
                         const Solution& u,
                         Vector& R_e) override {
        auto [v, p] = extract_fields(u.local(elem));
        
        // Quadrature loop
        for (auto& qp : elem.quadrature_points()) {
            auto N = elem.shape_functions(qp);
            auto dN = elem.shape_derivatives(qp);
            
            // Velocity gradient
            auto grad_v = compute_gradient(v, dN);
            auto div_v = trace(grad_v);
            
            // Convection
            auto v_qp = interpolate(v, N);
            auto conv = density * (grad_v * v_qp);
            
            // Stress
            auto tau = 2 * viscosity * symm(grad_v);
            auto sigma = -p * I + tau;
            
            // Weak form
            R_v += dN.transpose() * sigma * qp.weight;
            R_v += N * conv * qp.weight;
            
            // Continuity
            R_p += N * div_v * qp.weight;
            
            // Stabilization
            if (stab_type != StabilizationType::None) {
                add_stabilization(elem, qp, v, p, R_v, R_p);
            }
        }
    }
    
private:
    void add_stabilization(const Element& elem,
                          const QuadraturePoint& qp,
                          const Vector& v, double p,
                          Vector& R_v, Vector& R_p) {
        // SUPG/PSPG stabilization
        double h = elem.characteristic_length();
        double U = norm(v);
        double Re_h = density * U * h / (2 * viscosity);
        
        // Stabilization parameters
        double tau_supg = compute_tau_supg(h, U, viscosity);
        double tau_pspg = compute_tau_pspg(h, U, viscosity);
        
        // Residuals
        auto r_mom = compute_momentum_residual(v, p);
        auto r_cont = compute_continuity_residual(v);
        
        // SUPG contribution
        R_v += tau_supg * (grad_N * v) * r_mom * qp.weight;
        
        // PSPG contribution
        R_p += tau_pspg * grad_N * r_mom * qp.weight;
    }
};
```

### 4. Nonlinear Hyperelasticity
```cpp
// Finite strain hyperelasticity
class NeoHookean : public PhysicsModule<3> {
    double mu;     // Shear modulus
    double lambda; // Lamé parameter
    
    void compute_residual(const Element& elem,
                         const Solution& u,
                         Vector& R_e) override {
        for (auto& qp : elem.quadrature_points()) {
            // Deformation gradient
            auto F = compute_deformation_gradient(elem, u, qp);
            
            // Right Cauchy-Green tensor
            auto C = F.transpose() * F;
            
            // Invariants
            double I1 = trace(C);
            double I3 = det(C);
            double J = sqrt(I3);
            
            // First Piola-Kirchhoff stress
            auto P = mu * (F - F.inverse().transpose()) 
                   + lambda * log(J) * F.inverse().transpose();
            
            // Internal forces
            auto dN_dX = elem.shape_derivatives_reference(qp);
            R_e += dN_dX.transpose() * P * qp.weight * elem.J0(qp);
        }
    }
    
    void compute_jacobian(const Element& elem,
                         const Solution& u,
                         Matrix& K_e) override {
        for (auto& qp : elem.quadrature_points()) {
            auto F = compute_deformation_gradient(elem, u, qp);
            
            // Material tangent (4th order tensor)
            auto C_mat = compute_material_tangent(F);
            
            // Geometric stiffness
            auto sigma = compute_cauchy_stress(F);
            auto K_geo = compute_geometric_stiffness(elem, sigma, qp);
            
            // Total tangent
            K_e += K_mat + K_geo;
        }
    }
};
```

### 5. Phase Field Fracture
```cpp
// Phase field fracture model
class PhaseFieldFracture : public PhysicsModule<2> {
    double Gc;    // Critical energy release rate
    double l0;    // Length scale
    
    std::vector<FieldVariable> fields() const override {
        return {
            FieldVariable("displacement", 2),
            FieldVariable("damage", 1)
        };
    }
    
    void compute_residual(const Element& elem,
                         const Solution& u,
                         Vector& R_e) override {
        auto [disp, d] = extract_fields(u.local(elem));
        
        for (auto& qp : elem.quadrature_points()) {
            // Displacement part
            auto epsilon = compute_strain(elem, disp, qp);
            auto [eps_pos, eps_neg] = split_strain(epsilon);
            
            // Degradation function
            double g_d = (1 - d)^2 + k_res;
            
            // Stress
            auto sigma = g_d * C * eps_pos + C * eps_neg;
            
            // Damage driving force
            double psi_pos = 0.5 * dot(eps_pos, C * eps_pos);
            double H = std::max(H_old, psi_pos);  // History field
            
            // Residuals
            R_u += B.transpose() * sigma * qp.weight;
            R_d += (Gc/l0 * d + 2*H*(1-d)) * N * qp.weight;
            R_d += Gc*l0 * grad_d * grad_N * qp.weight;
        }
    }
};
```

### 6. Verification with MMS
```cpp
// Method of manufactured solutions
class ManufacturedSolution {
    std::function<double(Point)> u_exact;
    std::function<Vector(Point)> grad_u_exact;
    
    Vector compute_source_term(const Point& x) {
        // Compute source that makes u_exact a solution
        auto laplacian_u = compute_laplacian(u_exact, x);
        auto f = -laplacian_u;  // For Poisson equation
        return f;
    }
    
    double compute_error(const Solution& u_h) {
        double error_l2 = 0.0;
        
        for (auto& elem : mesh.elements()) {
            for (auto& qp : elem.quadrature_points()) {
                auto x = elem.map_to_physical(qp);
                double u_h_qp = interpolate(u_h.local(elem), qp);
                double u_ex = u_exact(x);
                
                error_l2 += pow(u_h_qp - u_ex, 2) * qp.weight;
            }
        }
        
        return sqrt(error_l2);
    }
};
```

## Performance Optimizations

### Vectorized Element Computation
```cpp
// SIMD-optimized element assembly
template<typename Physics>
void vectorized_assembly(const ElementBatch& batch,
                        Physics& physics,
                        MatrixBatch& K_e) {
    #pragma omp simd
    for (int e = 0; e < batch.size(); ++e) {
        physics.compute_jacobian_vectorized(batch[e], K_e[e]);
    }
}
```

### Material State Caching
```cpp
// Cache material state for nonlinear problems
class MaterialStateCache {
    std::vector<MaterialState> states;
    
    void update_if_needed(const Element& elem,
                         const Solution& u) {
        if (solution_changed(u)) {
            states[elem.id()] = compute_material_state(elem, u);
        }
    }
};
```

## Integration Points

### With fem/
- Uses element shape functions and quadrature
- Implements weak form interfaces
- Adds physics components to elements

### With assembly/
- Provides element matrix/vector computations
- Defines sparsity patterns through fields
- Handles constraint contributions

### With coupling/
- Exposes coupling interfaces
- Provides field transfer operators
- Manages coupled state updates

### With solvers/
- Defines linear/nonlinear systems
- Provides preconditioner hints
- Supplies convergence metrics

## Success Metrics

1. **Accuracy**: Pass all verification tests
2. **Conservation**: < 1e-12 violation of conservation laws
3. **Performance**: > 80% of theoretical FLOPS for assembly
4. **Modularity**: New physics in < 500 lines
5. **Robustness**: Stable for ill-conditioned problems
6. **Convergence**: Optimal convergence rates achieved

## Key Features

1. **Comprehensive Coverage**: All major physics domains
2. **Nonlinear Support**: Full geometric and material nonlinearity
3. **Stabilization**: Built-in stabilization methods
4. **Verification**: MMS and benchmark problems included
5. **Component-Based**: Mix and match physics via ECS
6. **Performance**: Optimized for vectorization and caching

This architecture provides a complete suite of physics implementations that can be combined for complex multiphysics simulations while maintaining modularity and performance.
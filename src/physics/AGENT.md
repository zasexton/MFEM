# AGENT.md - Physics Modules

## Mission
Implement domain-specific physics formulations for finite element analysis, providing modular, extensible physics components that can be combined for multiphysics simulations while maintaining clear interfaces with the FEM infrastructure.

## Architecture Philosophy
- **Formulation-Focused**: Each physics module implements specific PDEs
- **Component-Based**: Physics as pluggable components via ECS
- **Coupling-Ready**: Clean interfaces for multiphysics interaction
- **Material-Flexible**: Support for linear and nonlinear constitutive models
- **Verification-Driven**: Built-in manufactured solutions and benchmarks

## Scope and Non-Goals

To avoid duplication and ensure clean boundaries across libraries:
- Physics does not implement constitutive material models (hyperelasticity, plasticity, damage, CZM, etc.). These live in `materials/` and are consumed via a thin adapter.
- Physics does not define FEM element types. Elements, DOFs, spaces, constraints, and contact enforcement live in `fem/`.
- Physics does not implement multiphysics coupling strategies. Orchestration (monolithic/partitioned), interface handling, and transfer operators live in `coupling/`.
- Physics does not define unit systems or property catalogs. Use core units and the materials registry directly.

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
│   ├── material_adapter.hpp        # Adapter to top-level materials library (consumes materials public API)
│   ├── field_variable.hpp          # Physics field definitions
│   ├── physics_traits.hpp          # Physics type traits
│   ├── coupling_interface.hpp      # Interface exposed to coupling module
│   ├── conservation_law.hpp        # Conservation principles
│   └── physics_factory.hpp         # Physics module factory
│

├── mechanics/                       # Solid/structural mechanics
│   ├── solid/
│   │   ├── formulations/               # Domain-specific weak forms (no element types, no constitutive models)
│   │   │   ├── small_strain_solid.hpp     # Uses materials/ small-strain models via MaterialAdapter
│   │   │   ├── finite_strain_solid.hpp    # Uses materials/ finite-strain models via MaterialAdapter
│   │   │   ├── thermo_mechanical.hpp      # Thermal strain coupling (materials provide tangents)
│   │   │   └── phase_field_fracture.hpp   # Phase-field PDE; energy/tangents from materials
│   │   ├── gradient_enhanced/
│   │   │   ├── strain_gradient.hpp        # Higher-order PDE terms (consumes materials)
│   │   │   ├── micropolar.hpp             # Cosserat kinematics (materials handle constitutive law)
│   │   │   └── nonlocal.hpp               # Nonlocal operators (materials provide length-scale effects)
│   │   ├── enriched_methods/
│   │   │   ├── xfem.hpp                   # Extended FEM enrichment (consumes fem/)
│   │   │   ├── gfem.hpp                   # Generalized FEM enrichment (consumes fem/)
│   │   │   └── phantom_node.hpp           # Phantom node method (consumes fem/)
│   │   └── notes.md                       # Notes: constitutive models are defined in materials/
│   │ 
│   ├── nonlocal/
│   │   ├── peridynamics/
│   │   │   ├── bond_based.hpp           # Bond-based PD
│   │   │   ├── state_based.hpp          # State-based PD
│   │   │   └── correspondence.hpp       # Correspondence PD
│   │   └── lattice/
│   │       ├── lattice_spring.hpp       # Spring networks
│   │       └── discrete_lattice.hpp     # Discrete models
│   │ 
│   ├── structural/
│   │   ├── formulations/                # Structural formulations (use fem element types; no element definitions here)
│   │   │   ├── beam_formulation.hpp       # EB/Timoshenko/geometrically exact beams
│   │   │   ├── plate_formulation.hpp      # KL/Mindlin/Reissner plates
│   │   │   ├── shell_formulation.hpp      # KL/RM/solid-shell shells
│   │   │   └── cable_formulation.hpp      # Cable/rope formulations
│   │ 
│   ├── contact/                        # Contact laws (constraint/enforcement lives in fem/)
│   │   ├── frictionless_law.hpp          # No friction
│   │   ├── coulomb_law.hpp               # Coulomb friction
│   │   ├── stick_slip_law.hpp            # Stick-slip
│   │   ├── adhesive_law.hpp              # Adhesion
│   │   └── thermal_contact_law.hpp       # With heat transfer
│   │
│   └── dynamics/
│       ├── dynamics_formulation.hpp     # Residuals/mass/damping; time stepping in fem/solvers
│       ├── modal_dynamics.hpp           # Modal superposition (post-processing)
│       ├── harmonic_response.hpp        # Frequency-domain formulation (time integrators in fem)
│       └── random_vibration.hpp         # Stochastic loading formulation (time-stepping in fem)
│
├── thermal/                         # Heat transfer
│   ├── conduction/
│   │   ├── fourier.hpp                  # Fourier's law
│   │   ├── anisotropic_conduction.hpp   # Anisotropic
│   │   ├── nonlinear_conduction.hpp     # k(T)
│   │   ├── hyperbolic_heat.hpp          # Non-Fourier
│   │   └── dual_phase_lag.hpp           # DPL model
│   ├── convection/
│   │   ├── forced_convection.hpp        # Forced
│   │   ├── natural_convection.hpp       # Natural
│   │   ├── mixed_convection.hpp         # Mixed
│   │   └── conjugate_heat.hpp           # CHT
│   ├── radiation/
│   │   ├── surface_to_surface.hpp       # View factors
│   │   ├── participating_media.hpp      # P1, DOM
│   │   ├── rosseland.hpp                # Rosseland approx
│   │   └── monte_carlo_radiation.hpp    # MCRT
│   ├── phase_change/
│   │   ├── enthalpy_method.hpp          # Enthalpy
│   │   ├── stefan_problem.hpp           # Stefan
│   │   ├── level_set_stefan.hpp         # Level set
│   │   └── solidification.hpp           # Alloy solidification
│   ├── microscale/
│   │   ├── phonon_transport.hpp         # BTE for phonons
│   │   ├── molecular_heat.hpp           # MD heat transfer
│   │   └── kapitza_resistance.hpp       # Interface resistance
│   └── interface_thermal/
│       └── thermal_interface.hpp        # TIM/TBC models
│
├── fluid/                           # Fluid dynamics
│   ├── incompressible/
│   │   ├── stokes/
│   │   │   ├── stokes.hpp               # Stokes flow
│   │   │   └── brinkman_stokes.hpp      # Brinkman-Stokes
│   │   ├── navier_stokes/
│   │   │   ├── steady_ns.hpp            # Steady NS
│   │   │   ├── unsteady_ns.hpp          # Transient NS
│   │   │   ├── boussinesq.hpp           # Buoyancy-driven
│   │   │   └── non_newtonian.hpp        # Non-Newtonian
│   │   └── stabilized/                  # NOTE: stabilization methods live in fem/ (reused across physics)
│   │       └── (use fem/galerkin/* and fem/stabilized/*; configure from physics)
│   │
│   ├── compressible/
│   │   ├── euler.hpp                    # Euler equations
│   │   ├── navier_stokes_comp.hpp       # Compressible NS
│   │   └── notes.md                     # DG and shock-capturing provided by fem/ (generic across PDEs)
│   │
│   ├── turbulence/
│   │   ├── rans/
│   │   │   ├── k_epsilon.hpp            # k-ε model
│   │   │   ├── k_omega.hpp              # k-ω model
│   │   │   ├── k_omega_sst.hpp          # SST model
│   │   │   ├── spalart_allmaras.hpp     # SA model
│   │   │   └── reynolds_stress.hpp      # RSM
│   │   ├── les/
│   │   │   ├── smagorinsky.hpp          # Smagorinsky
│   │   │   ├── dynamic_smagorinsky.hpp  # Dynamic model
│   │   │   ├── wale.hpp                 # WALE model
│   │   │   └── vreman.hpp               # Vreman model
│   │   └── hybrid/
│   │       ├── des.hpp                  # DES
│   │       └── les_rans.hpp             # Hybrid LES-RANS
│   │
│   ├── multiphase/
│   │   ├── level_set.hpp                # Level set
│   │   ├── vof.hpp                      # Volume of fluid
│   │   ├── phase_field_flow.hpp         # Cahn-Hilliard
│   │   ├── mixture_model.hpp            # Mixture theory
│   │   └── euler_euler.hpp              # Two-fluid model
│   │
│   ├── free_surface/
│   │   ├── ale_free_surface.hpp         # ALE kinematics
│   │   ├── space_time_formulation.hpp   # Space-time weak form (time integrators live in fem)
│   │   └── moving_mesh.hpp              # Moving mesh kinematics
│   │ 
│   ├── porous_media/
│   │   ├── darcy.hpp                    # Darcy's law
│   │   ├── forchheimer.hpp              # Non-Darcy flow
│   │   ├── brinkman.hpp                 # Brinkman equation
│   │   ├── richards.hpp                 # Richards equation
│   │   └── two_phase_darcy.hpp          # Oil-water
│   │ 
│   ├── lattice_boltzmann/
│   │   ├── bgk_lbm.hpp                  # BGK collision
│   │   ├── mrt_lbm.hpp                  # Multiple relaxation
│   │   ├── thermal_lbm.hpp              # Thermal LBM
│   │   └── multiphase_lbm.hpp           # Multiphase LBM
│   │
│   ├── granular/
│   │   ├── kinetic_theory.hpp           # Kinetic theory
│   │   ├── frictional_flow.hpp          # Frictional regime
│   │   └── dense_granular.hpp           # Dense flows
│   │ 
│   └── special_flows/
│       ├── lubrication.hpp              # Thin film
│       ├── hele_shaw.hpp                # Hele-Shaw
│       ├── shallow_water.hpp            # Shallow water
│       └── micropolar.hpp               # Micropolar fluids
│
├── electromagnetic/                 # Electromagnetics
│   ├── static/
│   │   ├── electrostatics.hpp           # Laplace/Poisson
│   │   ├── magnetostatics.hpp           # Vector potential
│   │   └── current_flow.hpp             # DC current
│   ├── quasi_static/
│   │   ├── eddy_current.hpp             # Eddy currents
│   │   ├── magnetic_diffusion.hpp       # Magnetic diffusion
│   │   └── darwin_model.hpp             # Darwin approximation
│   ├── wave/
│   │   ├── maxwell_time.hpp             # Time domain
│   │   ├── maxwell_frequency.hpp        # Frequency domain
│   │   ├── helmholtz_em.hpp             # Helmholtz
│   │   ├── perfectly_matched_layer.hpp  # PML
│   │   └── fdtd.hpp                     # FDTD method
│   └── coupled_em/
│       ├── joule_heating.hpp            # Resistive heating
│       ├── lorentz_force.hpp            # EM forces
│       └── induction_heating.hpp        # Induction
│
├── optics/                          # Optical and photonic physics
│   ├── ray_optics/
│   │   ├── ray_tracing.hpp              # Geometric optics
│   │   └── beam_tracing.hpp             # Gaussian beams
│   ├── wave_optics/
│   │   ├── beam_propagation.hpp         # BPM
│   │   ├── finite_difference_bpm.hpp    # FD-BPM
│   │   └── spectral_methods.hpp         # Spectral propagation
│   ├── nonlinear_optics/
│   │   ├── kerr_effect.hpp              # Kerr nonlinearity
│   │   ├── second_harmonic.hpp          # SHG
│   │   └── four_wave_mixing.hpp         # FWM
│   └── photonic_structures/
│       ├── photonic_crystals.hpp        # Band structure
│       ├── metamaterials_optical.hpp    # Optical metamaterials
│       └── plasmonics.hpp               # Surface plasmons
│
├── acoustic/                        # Acoustics and vibro-acoustics
│   ├── linear_acoustics/
│   │   ├── helmholtz.hpp                # Frequency domain
│   │   ├── wave_equation.hpp            # Time domain
│   │   └── boundary_element.hpp         # BEM acoustics
│   ├── nonlinear_acoustics/
│   │   ├── westervelt.hpp               # Westervelt equation
│   │   └── kzk.hpp                      # KZK equation
│   └── aeroacoustics/
│       ├── lighthill.hpp                # Lighthill analogy
│       ├── ffowcs_williams.hpp          # FW-H
│       └── acoustic_perturbation.hpp    # APE
│
├── chemical/                        # Chemical transport and reactions
│   ├── transport/
│   │   ├── ficks_diffusion.hpp          # Fick's law
│   │   ├── stefan_maxwell.hpp           # Multicomponent
│   │   ├── dusty_gas.hpp                # Dusty gas model
│   │   └── advection_diffusion.hpp      # ADE
│   ├── reaction/
│   │   ├── arrhenius.hpp                # Arrhenius kinetics
│   │   ├── michaelis_menten.hpp         # Enzyme kinetics
│   │   ├── langmuir_hinshelwood.hpp     # Surface reactions
│   │   └── combustion.hpp               # Combustion models
│   ├── electrochemistry/
│   │   ├── nernst_planck.hpp            # Ion transport
│   │   ├── poisson_nernst_planck.hpp    # PNP
│   │   ├── butler_volmer.hpp            # Electrode kinetics
│   │   └── corrosion.hpp                # Corrosion models
│   ├── battery/
│   │   ├── newman_model.hpp             # Newman P2D
│   │   ├── single_particle.hpp          # SPM
│   │   ├── dfn_model.hpp                # Doyle-Fuller-Newman
│   │   └── solid_electrolyte.hpp        # Solid-state batteries
│   └── polymer/
│       ├── polymer_diffusion.hpp        # Polymer transport
│       ├── reptation.hpp                # Reptation model
│       └── viscoelastic_polymer.hpp     # Polymer rheology
│
├── quantum/                         # Quantum mechanics
│   ├── schrodinger/
│   │   ├── time_independent.hpp         # TISE
│   │   └── time_dependent.hpp           # TDSE
│   ├── density_functional/
│   │   ├── kohn_sham.hpp                # Kohn-Sham DFT
│   │   └── orbital_free.hpp             # OF-DFT
│   └── semiclassical/
│       ├── wkb.hpp                      # WKB approximation
│       └── quantum_hydrodynamics.hpp    # QHD
│
├── plasma/                          # Plasma physics
│   ├── magnetohydrodynamics/
│   │   ├── ideal_mhd.hpp                # Ideal MHD
│   │   ├── resistive_mhd.hpp            # Resistive MHD
│   │   └── hall_mhd.hpp                 # Hall MHD
│   ├── kinetic/
│   │   ├── vlasov.hpp                   # Vlasov equation
│   │   ├── fokker_planck.hpp            # Fokker-Planck
│   │   └── particle_in_cell.hpp         # PIC
│   └── fluid/
│       ├── two_fluid_plasma.hpp         # Ion-electron
│       └── drift_diffusion_plasma.hpp   # Drift-diffusion
│
├── biological/                      # Biological physics
│   ├── biomechanics/
│   │   ├── soft_tissue/
│   │   │   ├── neo_hookean_tissue.hpp   # Incompressible
│   │   │   ├── holzapfel_ogden.hpp      # HGO model
│   │   │   ├── fung.hpp                 # Fung model
│   │   │   └── viscoelastic_tissue.hpp  # QLV model
│   │   ├── hard_tissue/
│   │   │   ├── bone_remodeling.hpp      # Wolff's law
│   │   │   ├── bone_damage.hpp          # Fatigue
│   │   │   └── trabecular_bone.hpp      # Microstructure
│   │   └── active_materials/
│   │       ├── muscle_contraction.hpp   # Hill model
│   │       └── cardiac_mechanics.hpp    # Active stress
│   ├── biofluid/
│   │   ├── blood_flow/
│   │   │   ├── newtonian_blood.hpp      # Newtonian
│   │   │   ├── carreau_yasuda.hpp       # Non-Newtonian
│   │   │   └── windkessel.hpp           # 0D models
│   │   └── respiratory/
│   │       └── airway_flow.hpp          # Lung mechanics
│   ├── cell_mechanics/
│   │   ├── cell_membrane.hpp            # Membrane mechanics
│   │   └── cytoskeleton.hpp             # Network models
│   ├── neural/
│   │   ├── hodgkin_huxley.hpp           # HH model
│   │   ├── cable_equation.hpp           # Neural cables
│   │   └── neural_field.hpp             # Field theory
│   ├── tumor/
│   │   ├── tumor_growth.hpp             # Growth models
│   │   ├── angiogenesis.hpp             # Vessel formation
│   │   └── drug_delivery.hpp            # Drug transport
│   └── biofilm/
│       ├── biofilm_growth.hpp           # Biofilm formation
│       └── biofilm_mechanics.hpp        # Mechanical properties
│
├── geomechanics/                   # Geotechnical and geological
│   ├── soil_mechanics/
│   │   ├── cam_clay_soil.hpp            # Cam-Clay
│   │   ├── hardening_soil.hpp           # HS model
│   │   ├── hypoplastic.hpp              # Hypoplasticity
│   │   └── liquefaction.hpp             # Liquefaction
│   ├── rock_mechanics/
│   │   ├── hoek_brown.hpp               # Hoek-Brown
│   │   ├── joint_model.hpp              # Joint elements
│   │   └── discrete_fracture.hpp        # DFN
│   └── geodynamics/
│       ├── mantle_convection.hpp        # Mantle flow
│       └── fault_mechanics.hpp          # Earthquake
│
├── environmental/                   # Environmental physics
│   ├── atmosphere/
│   │   ├── weather_dynamics.hpp         # Weather models
│   │   ├── pollution_dispersion.hpp     # Air quality
│   │   └── cloud_physics.hpp            # Cloud formation
│   ├── ocean/
│   │   ├── ocean_circulation.hpp        # Ocean currents
│   │   ├── wave_dynamics.hpp            # Ocean waves
│   │   └── coastal_processes.hpp        # Coastal erosion
│   ├── ice/
│   │   ├── glacier_flow.hpp             # Glacier dynamics
│   │   ├── sea_ice.hpp                  # Sea ice models
│   │   └── permafrost.hpp               # Permafrost thaw
│   └── hydrology/
│       ├── surface_water.hpp            # Rivers/lakes
│       ├── groundwater.hpp              # Aquifers
│       └── watershed.hpp                # Watershed models
│
├── particle/                        # Particle-based methods
│   ├── discrete_element/
│   │   ├── dem_spheres.hpp              # Spherical DEM
│   │   ├── dem_polyhedra.hpp            # Polyhedral DEM
│   │   └── dem_deformable.hpp           # Deformable particles
│   ├── smoothed_particle/
│   │   ├── sph_fluid.hpp                # Fluid SPH
│   │   ├── sph_solid.hpp                # Total Lagrangian SPH
│   │   └── corrected_sph.hpp            # Corrected SPH
│   ├── material_point/
│   │   ├── mpm_standard.hpp             # Standard MPM
│   │   ├── gimp.hpp                     # GIMP
│   │   └── cpdi.hpp                     # CPDI
│   └── molecular_dynamics/
│       ├── lennard_jones.hpp            # LJ potential
│       ├── embedded_atom.hpp            # EAM
│       └── coarse_grained.hpp           # CG-MD
│
├── intrinsic_coupled/              # Inherently coupled multi-field physics
│   ├── piezoelectric.hpp          # Electro-mechanical
│   ├── piezomagnetic.hpp          # Magneto-mechanical
│   ├── magnetostrictive.hpp       # Magneto-elastic
│   ├── electrostriction.hpp       # Electro-elastic
│   ├── poroelasticity.hpp         # Biot's theory
│   ├── thermoelectric.hpp         # Seebeck/Peltier
│   ├── magnetohydrodynamics.hpp   # MHD (as single physics)
│   ├── electrokinetics.hpp        # Electro-osmosis
│   ├── chemo_mechanics.hpp        # Mechano-chemistry
│   ├── magnetoelectric.hpp        # ME coupling
│   └── flexoelectric.hpp          # Flexoelectricity
│
├── verification/                    # Verification and benchmarks
│   ├── manufactured/
│   │   ├── mms_generator.hpp            # MMS tools
│   │   └── mms_library/                 # MMS solutions by physics
│   ├── analytical/
│   │   └── analytical_library/          # Analytical solutions
│   └── benchmarks/
│       └── benchmark_suite/             # Standard benchmarks
│
├── utilities/                      # Physics utilities
│   └── dimensionless.hpp           # Domain-specific dimensionless numbers (Re, Pr, Gr, etc.)
│                                   # Use core units and materials registry directly; no units/catalog here
│
└── tests/                          # Testing infrastructure
    ├── unit/                       # Unit tests per physics
    ├── convergence/                # Convergence studies
    └── performance/                # Performance benchmarks
```

### Material Adapter (base/material_adapter.hpp)

The adapter provides a thin, zero‑copy bridge between physics internals and the top‑level `materials/` API:

- Maps physics field variables and kinematics to `materials::MaterialInputs` (small vs finite strain, temperature, pore pressure, electric/magnetic fields, gradients, dt).
- Owns a handle to a `materials::IMaterial` instance (via factory/registry); supports configuration and cloning.
- Invokes `evaluate(inputs, state, outputs, &tangents)` and returns stress/tangent blocks in the measures required by the physics formulation (Cauchy/PK, small/finite strain), performing only measure conversion if needed.
- Manages per‑integration‑point material state lifecycles (trial → accept/reject) by collaborating with assembly/element storage; no dynamic allocation in hot paths.
- Provides optional caching of `TangentBlocks` for Jacobian reuse within a time/load step when valid.

Example interface (header sketch):
```cpp
class MaterialAdapter {
public:
    explicit MaterialAdapter(std::shared_ptr<materials::IMaterial> m);

    // Fill MaterialInputs from physics data
    void prepare_inputs(const Kinematics& kin,
                        const Fields& fields,
                        double dt,
                        materials::MaterialInputs& min) const;

    // Evaluate and return stress/tangent blocks
    void evaluate(const materials::MaterialInputs& min,
                  MaterialPointState& mp_state,
                  materials::MaterialOutputs& mout,
                  materials::TangentBlocks* tb = nullptr) const;
};
```

Notes:
- The adapter maps physics kinematics/fields into `materials::MaterialInputs`,
  calls `evaluate()` to obtain `MaterialOutputs` and optional `TangentBlocks`,
  and never reimplements material behavior.
- Choice of constitutive law, regimes (small/finite strain), and coupled tangents
  is entirely delegated to `materials/`.

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
    // Acquire a material from the top-level materials library
    std::shared_ptr<materials::IMaterial> material =
        materials::MaterialFactory::instance().create<materials::IsotropicElastic>();
    
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
            
            // Stress (linear elastic) via unified materials interface
            materials::MaterialInputs min;
            min.eps = epsilon; // small-strain example
            materials::MaterialOutputs mout;
            material->evaluate(min, state_, mout);
            auto sigma = mout.P; // Cauchy stress in small strain
            
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
            // Consistent tangent from materials
            materials::MaterialInputs min;
            materials::MaterialOutputs mout;
            materials::TangentBlocks tb;
            material->evaluate(min, state_, mout, &tb);
            auto C = tb.dP_dEps;
            
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
// Incompressible flow; stabilization terms are provided by fem/ (SUPG/PSPG/GLS/VMS)
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
            
            // Stabilization: integrate fem/stabilized terms here when enabled
            // (configured via fem/ stabilized APIs; not implemented in physics)
        }
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
3. **Stabilization**: Integrates fem/ stabilized and DG methods; not implemented here
4. **Verification**: MMS and benchmark problems included
5. **Component-Based**: Mix and match physics via ECS
6. **Performance**: Optimized for vectorization and caching

This architecture provides a complete suite of physics implementations that can be combined for complex multiphysics simulations while maintaining modularity and performance.

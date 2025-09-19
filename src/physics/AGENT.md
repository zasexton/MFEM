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
│   │   ├── maxwell_time.hpp             # Time-domain weak form (time integration in fem)
│   │   ├── maxwell_frequency.hpp        # Frequency-domain formulation
│   │   ├── helmholtz_em.hpp             # Helmholtz
│   │   └── perfectly_matched_layer.hpp  # PML (absorbing boundary formulation)
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

### 2. Next-Generation Physics Framework (Target Architecture)

The following examples demonstrate our vision for a more intuitive physics module architecture that:
- Expresses physics in mathematical notation identical to textbook weak formulations
- Automatically generates strong forms, boundary terms, documentation, and tests
- Cleanly interfaces with the materials library for constitutive models
- Provides multiple levels of abstraction for different user expertise

#### 2.1 Linear Elasticity - High-Level Variational Approach
```cpp
// Next-generation Linear Elasticity using variational forms
template<int Dim>
class LinearElasticity : public PhysicsModule<Dim> {
public:
    using namespace fem::variational;

    // Define physics entirely through mathematical weak form
    struct PhysicsDefinition {
        // Core weak formulation - reads like mathematical notation
        static auto weak_form() {
            return [](auto u, auto v, auto& ctx) {
                // Kinematic assumption: small strain
                auto strain = sym(grad(u));

                // Constitutive law from material library
                auto stress = ctx.material.evaluate_stress(strain);

                // Weak form: ∫σ:∇v dx = ∫f·v dx + ∫t·v ds
                auto internal_work = integral(inner(stress, grad(v)), ctx.domain);
                auto body_work = integral(inner(ctx.body_force, v), ctx.domain);
                auto traction_work = integral(inner(ctx.traction, v),
                                            ctx.boundary["traction"]);

                return internal_work - body_work - traction_work;
            };
        }

        // Metadata for automatic documentation
        static auto metadata() {
            return PhysicsMetadata()
                .name("Linear Elasticity")
                .description("Small deformation elastic solid mechanics")
                .governing_equation("∇·σ + f = 0")
                .field("u", "displacement", "m", Dim)
                .parameter("E", "Young's modulus", "Pa")
                .parameter("ν", "Poisson's ratio", "-");
        }
    };

    LinearElasticity() {
        // Everything is auto-generated from the weak form
        initialize_from_weak_form(PhysicsDefinition::weak_form());
        generate_artifacts();
    }

private:
    // Material interface (bridges to materials library)
    class MaterialInterface {
        std::shared_ptr<materials::IMaterial> material_;
    public:
        auto evaluate_stress(const auto& strain) {
            // Delegates to materials library
            return SymbolicMaterialEvaluator(material_, strain);
        }
    };

    void generate_artifacts() {
        // Automatic generation of all derived quantities
        generate_strong_form();      // ∇·σ + f = 0
        generate_boundary_terms();    // Natural: σ·n = t, Essential: u = g
        generate_jacobian();         // Tangent stiffness matrix
        generate_documentation();     // Markdown/LaTeX docs
        generate_verification();      // Patch test, MMS, energy conservation
    }

    //==========================================================================
    // AUTOMATIC STRONG FORM DERIVATION
    //==========================================================================
    void generate_strong_form() {
        // From weak form: ∫σ:∇v dx = ∫f·v dx
        // Apply integration by parts: -∫(∇·σ)·v dx + ∫(σ·n)·v ds = ∫f·v dx
        // Extract: ∇·σ + f = 0 in Ω, σ·n = t on Γt
        strong_form_ = symbolic::derive_strong_form(weak_form_);
    }

    //==========================================================================
    // AUTOMATIC DOCUMENTATION GENERATION
    //==========================================================================
    void generate_documentation() {
        documentation_ = DocumentationGenerator()
            .add_section("Governing Equations", {
                {"Strong Form", "∇·σ + f = 0 in Ω"},
                {"Weak Form", "∫σ:∇v dx = ∫f·v dx + ∫t·v ds"},
                {"Constitutive", "σ = C:ε (from material model)"},
                {"Kinematic", "ε = sym(∇u)"}
            })
            .add_section("Boundary Conditions", {
                {"Natural (Neumann)", "σ·n = t on Γt"},
                {"Essential (Dirichlet)", "u = g on Γu"}
            })
            .add_material_models({"LinearElastic", "NeoHookean", "MooneyRivlin"})
            .generate();
    }

    //==========================================================================
    // AUTOMATIC VERIFICATION TEST GENERATION
    //==========================================================================
    void generate_verification() {
        verification_suite_ = VerificationSuite()
            .add_patch_test()              // Constant strain field
            .add_rigid_body_test()         // Zero stress for rigid motion
            .add_manufactured_solution()    // Convergence verification
            .add_energy_conservation()     // Work-energy theorem
            .add_material_objectivity();   // Frame invariance
    }

public:
    // Clean runtime interface
    void compute_residual(const Element& elem, const Solution& u, Vector& R_e) override {
        compiled_residual_->evaluate(elem, u, R_e);
    }

    void compute_jacobian(const Element& elem, const Solution& u, Matrix& K_e) override {
        compiled_jacobian_->evaluate(elem, u, K_e);
    }

    // Material model selection
    void set_material(const std::string& model, const Parameters& params = {}) {
        material_ = materials::MaterialFactory::create(model, params);
    }

    // Access auto-generated content
    std::string get_documentation(Format fmt = Format::Markdown) const {
        return documentation_.format(fmt);
    }

    bool verify(const Mesh& mesh) {
        return verification_suite_.run_all(mesh, *this);
    }
};
```

# Alternative: Physics as Mathematical Notation
*Format A* variational DSL with concrete, UFL‑style examples and knobs for spaces, coefficients, stabilization, coupling, and backend lowering. It’s written to fit the responsibilities and folder layout already defined for `fem/variational/` and to stay inside FEM scope while delegating materials and multiphysics orchestration to their respective modules.

---

## Variational DSL (Format A): Cookbook Examples & Recipes

This section demonstrates how users express PDEs in near‑textbook weak form using the `fem/variational/` DSL and then tailor spaces, coefficients, stabilization, coupling structure, and backend lowering. The DSL is implemented under `fem/variational/` (symbolic operators, measures, IR, lowering) and reuses the existing FEM spaces and quadrature through lightweight adapters; it does **not** re‑implement materials or coupling logic.&#x20;

> **Notation recap (available primitives)**
>
> * **Spaces**: `spaces::H1`, `spaces::L2`, `spaces::Hcurl`, `spaces::Hdiv`, `spaces::DG`, plus helpers for mixed/product spaces via adapters in `variational/spaces/`. These wrap `fem/spaces/` types (e.g., `h1_space.hpp`, `h_curl_space.hpp`).&#x20;
> * **Functions & coeffs**: `trial(V,"u")`, `test(V,"v")`, `coeff::field("k")`, `coeff::vector("beta")`, `coeff::material("LinearElastic", params)` (thin adapter to `materials/`). Materials live outside FEM/variational and are consumed via adapters.&#x20;
> * **Operators**: `grad`, `div`, `curl`, `sym`, `tr`, `inner`; facet operators `n()`, `jump()`, `avg()` for DG/interface terms; time/ALE helpers `dt(u)`, `ale::material(u, v_mesh)`.&#x20;
> * **Measures**: `dx()`, `ds("tag")`, `dS()` for interior facets.&#x20;
> * **Lowering**: backends `lower_assembled{…}`, `lower_matrix_free{…}`, optional `lower_ceed{…}`; the compiler performs analysis (shape inference, CSE, sparsity prediction) before emitting kernels.&#x20;
> * **BCs**: Natural/flux terms are encoded in the integrals; essential (Dirichlet) constraints are applied via FEM boundary APIs (`fem/boundary/*`). The DSL provides convenience glue that calls into those APIs rather than re‑implementing them.&#x20;

> **Scope/ownership reminder**
>
> * **Materials** (constitutive laws, tangents) are owned by `materials/` and consumed here through a thin adapter.&#x20;
> * **Multiphysics orchestration** (monolithic vs. partitioned, interface relaxation) lives in `coupling/` and `analysis/`. Variational forms can be *composed* into block systems; solve policy is out of scope here.&#x20;

---

### 1) Poisson with variable coefficient + mixed boundary conditions

```cpp
using namespace fem::variational;

auto V  = spaces::H1<Dim>(mesh, Order{1});           // scalar H1
auto u  = trial(V, "u");
auto v  = test(V,  "v");

auto k  = coeff::field("k");                         // conductivity
auto f  = coeff::field("f");                         // source
auto gN = coeff::field("gN");                        // Neumann flux

auto a = inner(k * grad(u), grad(v)) * dx();
auto L = f * v * dx() + gN * v * ds("GammaN");

auto prob = Problem(V, a, L)
  .with_dirichlet("GammaD", u, coeff::field("u_D"))  // calls fem/boundary
  .compile(lower_assembled{quadrature=Auto{}});
```

**Tuning**: change `Order{p}`, supply `quadrature=Gauss{q}`, or switch to `lower_matrix_free{sum_factorization=true}`.

---

### 2) Linear elasticity (small strain), vector H¹ space, material from `materials/`

```cpp
auto V  = spaces::H1<Dim>(mesh, Order{1}, Components{Dim}); // vector-valued
auto u  = trial(V, "u");
auto v  = test(V,  "v");

// Material adapter fetches stress/elasticity from materials/
auto mat = coeff::material("LinearElastic", {{"E", 210e9}, {"nu", 0.3}});

auto eps   = sym(grad(u));
auto sigma = mat.evaluate_stress(eps);

auto a = inner(sigma, grad(v)) * dx();              // ∫ σ : ∇v
auto L = inner(coeff::vector("f"), v) * dx()
       + inner(coeff::vector("t"), v) * ds("Gamma_t");

auto prob = Problem(V, a, L)
  .with_dirichlet("Gamma_u", u, coeff::vector("u_D"))
  .compile(lower_assembled{});
```

The stress evaluation delegates to `materials/` via the adapter; no constitutive law is re‑implemented here.&#x20;

---

### 3) Stokes (incompressible): Taylor–Hood or equal‑order + PSPG

```cpp
// Taylor–Hood (H1^k for velocity, H1^{k-1} for pressure)
auto V = spaces::H1<Dim>(mesh, Order{2}, Components{Dim});
auto Q = spaces::H1<Dim>(mesh, Order{1});           // or L2 for pressure
auto [u,p] = trial(V,Q, "u","p");
auto [v,q] = test (V,Q, "v","q");

auto nu = coeff::field("nu");
auto f  = coeff::vector("f");

auto a = nu * inner(grad(u), grad(v)) * dx()
       - p * div(v) * dx()
       + q * div(u) * dx();

auto L = inner(f, v) * dx();

// Optional PSPG/SUPG when using equal-order (H1/H1)
auto stab = forms::stabilization::pspg(u,p,v,q, /*auto_tau=*/Auto{});
auto prob = Problem({V,Q}, a + stab, L).compile(lower_assembled{});
```

Stabilization terms are provided under `variational/forms/stabilization/` and can be toggled or auto‑selected based on the element pair.&#x20;

---

### 4) Advection–diffusion (high Péclet) with SUPG/CIP

```cpp
auto V   = spaces::H1<Dim>(mesh, Order{1});
auto u   = trial(V,"u");
auto v   = test (V,"v");
auto k   = coeff::field("k");              // diffusivity
auto beta= coeff::vector("beta");          // advection velocity
auto f   = coeff::field("f");

auto gal = inner(k*grad(u), grad(v)) * dx()
         + inner(beta · grad(u), v) * dx()
         - f * v * dx();

auto supg = forms::stabilization::supg(u, v, beta, f, Auto{}); // τ auto
auto cip  = forms::stabilization::cip (u, v, k, Auto{});

auto prob = Problem(V, gal + supg + cip, 0.0).compile(lower_assembled{});
```

---

### 5) Transient heat: mass + diffusion with `dt()`

```cpp
auto V  = spaces::H1<Dim>(mesh, Order{1});
auto T  = trial(V,"T");
auto w  = test (V,"w");

auto rho = coeff::field("rho");
auto cp  = coeff::field("cp");
auto k   = coeff::field("k");
auto Q   = coeff::field("Q");

auto a = (rho*cp * dt(T) * w) * dx()                 // mass
       + inner(k * grad(T), grad(w)) * dx();         // diffusion

auto L = Q * w * dx();

auto prob = Problem(V, a, L)
  .time_discretization(TimeScheme::BDF2)             // delegated to solvers/
  .compile(lower_assembled{});
```

Time operators like `dt()` are provided by `variational/symbolic/time_operators.hpp`. Actual time stepping is orchestrated in `solvers/transient/`.&#x20;

---

### 6) Maxwell (H(curl)) with impedance/PML boundary

```cpp
auto V  = spaces::Hcurl<Dim>(mesh, Order{1});
auto E  = trial(V,"E");
auto W  = test (V,"W");

auto mu_inv = coeff::field("mu_inv");      // 1/μ
auto eps    = coeff::field("eps");         // ε
auto omega  = coeff::field("omega");

auto a = inner(mu_inv * curl(E), curl(W)) * dx()
       - (omega*omega) * inner(eps * E, W) * dx();

auto impedance = forms::interface::impedance(E, W, coeff::field("Z"), "Gamma_out");
auto pml       = forms::stabilization::pml(E, W, coeff::tensor("S")); // stretch

auto prob = Problem(V, a + impedance + pml, 0.0).compile(lower_assembled{});
```

H(curl) spaces and interface helpers are exposed via the variational adapters and interface subpackage.&#x20;

---

### 7) Interior‑penalty DG for scalar advection–diffusion

```cpp
auto V  = spaces::DG<Dim>(mesh, Order{1});
auto u  = trial(V,"u");
auto v  = test (V,"v");
auto k  = coeff::field("k");
auto b  = coeff::vector("b");     // advection

using forms::facet; using forms::ip;   // notational sugar

auto a = inner(k*grad(u), grad(v)) * dx()
       + inner(b · grad(u), v) * dx()

       // Symmetric interior penalty diffusion
       + facet::inner( avg(k*grad(u)) · n(), jump(v) ) * dS()
       + facet::inner( avg(k*grad(v)) · n(), jump(u) ) * dS()
       + ip::penalty(Auto{}) * inner(jump(u), jump(v)) * dS();

auto prob = Problem(V, a, 0.0).compile(lower_assembled{});
```

Facet operators `n()`, `jump()`, `avg()` are available in `variational/symbolic/facet_operators.hpp`.&#x20;

---

### 8) Cahn–Hilliard (mixed form)

```cpp
// Unknowns: concentration c and chemical potential μ
auto Vc = spaces::H1<Dim>(mesh, Order{1});
auto Vm = spaces::H1<Dim>(mesh, Order{1});
auto [c, mu] = trial(Vc,Vm, "c","mu");
auto [w, r ] = test (Vc,Vm, "w","r");

auto M  = coeff::field("M");         // mobility
auto eps= coeff::field("eps");       // interface thickness
auto f0 = coeff::field("f0");        // bulk free energy density derivative

auto a = inner( dt(c), w ) * dx()
       + inner( M * grad(mu), grad(w) ) * dx()
       + inner( mu, r ) * dx()
       - inner( f0(c), r ) * dx()
       - inner( -eps*eps * div(grad(c)), r ) * dx();

auto prob = Problem({Vc,Vm}, a, 0.0)
  .time_discretization(TimeScheme::BDF2)
  .compile(lower_assembled{});
```

---

### 9) Phase‑field fracture via **energy functional** (first/second variations auto‑derived)

```cpp
auto Vu = spaces::H1<Dim>(mesh, Order{1}, Components{Dim}); // displacement
auto Vd = spaces::H1<Dim>(mesh, Order{1});                  // damage
auto [u, d] = trial(Vu,Vd, "u","d");

auto mat = coeff::material("NeoHookean", { /* ... */});     // materials/
auto Gc  = coeff::field("Gc");
auto l0  = coeff::field("l0");

auto eps  = sym(grad(u));
auto psi  = 0.5 * inner(eps, mat.elasticity()*eps);
auto g_d  = sqr(1.0 - d) + coeff::field("k_res");

auto energy_density = g_d*psi + (Gc/(2*l0))*(sqr(d) + sqr(l0)*inner(grad(d),grad(d)));

auto Pi = energy_density * dx();     // total potential energy functional

auto prob = Problem::from_energy({Vu,Vd}, Pi)   // δΠ, δ²Π auto via AD
  .with_irreversibility(d >= d_old)             // handled via constraints/solvers
  .compile(lower_assembled{});
```

Variational’s `functional.hpp` and derivative utilities compute first/second variations; constraints are enforced via FEM/solver components.&#x20;

---

### 10) Thermo‑elastic coupling (block form), monolithic assembly

```cpp
auto Vu = spaces::H1<Dim>(mesh, Order{1}, Components{Dim});
auto VT = spaces::H1<Dim>(mesh, Order{1});
auto [u,T] = trial(Vu,VT, "u","T");
auto [v,w] = test (Vu,VT, "v","w");

auto mat  = coeff::material("ThermoElastic", { /* α, E, ν, k ... */});
auto eps  = sym(grad(u));
auto eps_th = mat.thermal_expansion() * T * identity<Dim>();

auto sigma = mat.evaluate_stress(eps - eps_th);
auto q     = -mat.thermal_conductivity() * grad(T);
auto s_mech= mat.mechanical_dissipation(eps);  // optional

auto mech = inner(sigma, grad(v)) * dx()
          - inner(coeff::vector("f"), v) * dx();

auto heat = inner(q, grad(w)) * dx()
          + s_mech * w * dx()
          - coeff::field("Q") * w * dx();

auto prob = Problem({Vu,VT}, mech + heat, 0.0)
  .compile(lower_assembled{block_structure=Auto{}});
```

Coupling *policy* (monolithic vs. partitioned, Aitken/IQN, etc.) is owned by `coupling/`. The block operator produced here plugs into those strategies.&#x20;

---

### 11) Navier–Stokes (steady) with auto‑stabilization & backend switch

```cpp
auto V = spaces::H1<Dim>(mesh, Order{2}, Components{Dim});
auto Q = spaces::L2   (mesh, Order{1});
auto [v,p] = trial(V,Q, "v","p");
auto [w,q] = test (V,Q, "w","q");

auto rho = coeff::field("rho");
auto nu  = coeff::field("nu");
auto f   = coeff::vector("f");

auto convection = inner(grad(v)*v, w);
auto viscous    = nu * inner(grad(v), grad(w));
auto pressure   = - inner(p, div(w)) + inner(div(v), q);

auto base = (rho*convection + viscous + pressure - inner(f,w)) * dx();
auto stab = forms::stabilization::auto_incompressible(v,p,w,q, rho, nu);

auto prob = Problem({V,Q}, base + stab, 0.0)
  .compile(lower_matrix_free{sum_factorization=true, device=Auto{}});
```

---

### 12) ALE advection with material derivative

```cpp
auto V   = spaces::H1<Dim>(mesh, Order{1});
auto phi = trial(V,"phi");
auto psi = test (V,"psi");

auto v_flow = coeff::vector("v");       // physical flow speed
auto v_mesh = coeff::vector("v_mesh");  // ALE mesh speed

auto DphiDt = ale::material(phi, v_flow - v_mesh);   // ∂t φ + (v-v_m)·∇φ

auto form = inner(DphiDt, psi) * dx();
auto prob = Problem(V, form, 0.0).compile(lower_assembled{});
```

ALE/time operators are provided by the variational symbolic layer and lower to standard quadrature operations via integration adapters.&#x20;

---

### Tailoring & Overrides (one‑liners)

* **Spaces**
  Change polynomial order or family: `spaces::H1<Dim>(mesh, Order{p})`, vector components via `Components{Dim}`, switch to `Hcurl`, `Hdiv`, or `DG` as required by the physics (all mapped to `fem/spaces/`).&#x20;

* **Quadrature**
  `compile(lower_assembled{quadrature=Gauss{q}})` or `=Auto{}` for heuristic selection.

* **Stabilization**
  Opt‑in terms from `variational/forms/stabilization/` (SUPG, PSPG, GLS, CIP, grad‑div); or use `auto_*` helpers that inspect the form/space pairing to attach consistent stabilizations.&#x20;

* **Interface/Boundary**
  Use `forms/interface::*` (e.g., Nitsche, mortar, impedance) or add natural terms with `ds("tag")`. Essential BCs route through `fem/boundary/*` via DSL convenience functions.&#x20;

* **Coupling structure**
  Build block problems with `Problem({V1,V2,…}, a_total, L_total)`; pass the resulting operator to `coupling/` for monolithic or partitioned solves (Aitken, IQN, etc.).&#x20;

* **Materials**
  Always pass constitutive behavior via `coeff::material(name, params)`; the adapter maps strain/fields to `materials::evaluate()` and returns stress/tangents without duplicating models.&#x20;

* **Backend lowering**
  Choose `lower_assembled{…}` for classic sparse assembly, `lower_matrix_free{…}` for matrix‑free/sum‑factorized kernels, or `lower_ceed{…}` to target libCEED; the compiler runs analysis passes (shape inference, CSE, sparsity) before emitting kernels.&#x20;

---

**Where this fits**
These examples live under `fem/variational/examples/` and are executable after binding spaces/BCs/coefficients from the FEM layer. They interoperate with `physics/` modules (for end‑to‑end applications), `materials/` (for constitutive response), and `coupling/` (for strategy). The boundaries among modules remain intact: *variational* expresses math; *fem* provides spaces/integration/BC machinery; *materials* supplies constitutive response; *coupling/analysis/solvers* drive solution orchestration.  &#x20;

---


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

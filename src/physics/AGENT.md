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
│   ├── material_adapter.hpp        # Adapter to top-level materials library (consumes materials public API)
│   ├── field_variable.hpp          # Physics field definitions
│   ├── physics_traits.hpp          # Physics type traits
│   ├── coupling_interface.hpp      # Interface exposed to coupling module
│   ├── conservation_law.hpp        # Conservation principles
│   └── physics_factory.hpp         # Physics module factory
│
├── mechanics/                       # Solid/structural mechanics
│   ├── solid/
│   │   ├── linear/
│   │   │   ├── linear_elasticity.hpp    # Hooke's law
│   │   │   ├── anisotropic_elastic.hpp  # Orthotropic/anisotropic
│   │   │   └── thermal_stress.hpp       # Thermoelastic stress
│   │   ├── finite_strain/
│   │   │   ├── total_lagrangian.hpp     # Total Lagrangian
│   │   │   ├── updated_lagrangian.hpp   # Updated Lagrangian
│   │   │   └── corotational.hpp         # Corotational formulation
│   │   ├── hyperelastic/
│   │   │   ├── neo_hookean.hpp          # Neo-Hookean
│   │   │   ├── mooney_rivlin.hpp        # Mooney-Rivlin
│   │   │   ├── ogden.hpp                # Ogden model
│   │   │   ├── yeoh.hpp                 # Yeoh model
│   │   │   ├── arruda_boyce.hpp         # Arruda-Boyce
│   │   │   ├── gent.hpp                 # Gent model
│   │   │   └── holzapfel_gasser.hpp     # HGO
│   │   ├── plasticity/
│   │   │   ├── j2_plasticity.hpp        # Von Mises
│   │   │   ├── tresca.hpp               # Tresca
│   │   │   ├── mohr_coulomb.hpp         # Mohr-Coulomb
│   │   │   ├── drucker_prager.hpp       # Drucker-Prager
│   │   │   ├── cam_clay.hpp             # Modified Cam-Clay
│   │   │   ├── crystal_plasticity.hpp   # Crystal plasticity
│   │   │   ├── johnson_cook.hpp         # Johnson-Cook
│   │   │   └── gurson_tvergaard.hpp     # Gurson model
│   │   ├── viscoelastic/
│   │   │   ├── maxwell.hpp              # Maxwell model
│   │   │   ├── kelvin_voigt.hpp         # Kelvin-Voigt
│   │   │   ├── zener.hpp                # Standard linear solid
│   │   │   ├── generalized_maxwell.hpp  # Prony series
│   │   │   ├── burgers.hpp              # Burgers model
│   │   │   └── fractional_derivative.hpp # Fractional viscoelasticity
│   │   ├── damage/
│   │   │   ├── lemaitre.hpp             # Lemaitre damage
│   │   │   ├── mazars.hpp               # Mazars concrete
│   │   │   ├── gurson_damage.hpp        # Ductile damage
│   │   │   ├── cohesive_zone.hpp        # CZM
│   │   │   └── phase_field_fracture.hpp # Phase field
│   │   ├── gradient_enhanced/
│   │   │   ├── strain_gradient.hpp      # Strain gradient elasticity
│   │   │   ├── micropolar.hpp           # Cosserat continuum
│   │   │   └── nonlocal.hpp             # Nonlocal models
│   │   ├── enriched_methods/
│   │   │   ├── xfem.hpp                 # Extended FEM
│   │   │   ├── gfem.hpp                 # Generalized FEM
│   │   │   └── phantom_node.hpp         # Phantom node method
│   │   └── special/
│   │       ├── shape_memory.hpp         # Shape memory alloys
│   │       ├── swelling.hpp             # Swelling materials
│   │       └── growth.hpp               # Biological growth
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
│   │   ├── beam/
│   │   │   ├── euler_bernoulli.hpp      # Euler-Bernoulli
│   │   │   ├── timoshenko.hpp           # Timoshenko
│   │   │   ├── geometrically_exact.hpp  # Simo-Reissner
│   │   │   └── composite_beam.hpp       # Layered beams
│   │   ├── plate/
│   │   │   ├── kirchhoff_love.hpp       # Thin plate
│   │   │   ├── mindlin_reissner.hpp     # Thick plate
│   │   │   ├── von_karman.hpp           # Large deflection
│   │   │   └── laminated_plate.hpp      # Composite plates
│   │   ├── shell/
│   │   │   ├── kirchhoff_love_shell.hpp # Thin shell
│   │   │   ├── reissner_mindlin_shell.hpp # Thick shell
│   │   │   ├── solid_shell.hpp          # Solid-shell element
│   │   │   └── isogeometric_shell.hpp   # NURBS-based
│   │   └── cable/
│   │       ├── cable_element.hpp        # Cable/rope
│   │       └── catenary.hpp             # Catenary cables
│   │ 
│   ├── contact/
│   │   ├── frictionless_contact.hpp     # No friction
│   │   ├── coulomb_friction.hpp         # Coulomb model
│   │   ├── stick_slip.hpp               # Stick-slip
│   │   ├── adhesive_contact.hpp         # Adhesion
│   │   ├── wear.hpp                     # Archard wear
│   │   └── thermal_contact.hpp          # With heat transfer
│   │
│   └── dynamics/
│       ├── explicit_dynamics.hpp        # Central difference
│       ├── implicit_dynamics.hpp        # Newmark/HHT
│       ├── modal_dynamics.hpp           # Modal superposition
│       ├── harmonic_response.hpp        # Frequency domain
│       └── random_vibration.hpp         # Stochastic dynamics
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
│   │   └── stabilized/
│   │       ├── supg.hpp                 # SUPG
│   │       ├── pspg.hpp                 # PSPG
│   │       ├── gls.hpp                  # GLS
│   │       └── vms.hpp                  # Variational multiscale
│   │
│   ├── compressible/
│   │   ├── euler.hpp                    # Euler equations
│   │   ├── navier_stokes_comp.hpp       # Compressible NS
│   │   ├── discontinuous_galerkin.hpp   # DG methods
│   │   └── shock_capturing.hpp          # Shock capturing
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
│   │   ├── ale_free_surface.hpp         # ALE methods
│   │   ├── space_time_free.hpp          # Space-time
│   │   └── moving_mesh.hpp              # Moving mesh
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
│   ├── units.hpp                   # Unit systems
│   ├── material_catalog.hpp        # (Optional) physics-facing aliases that proxy to materials' property/database utilities
│   └── dimensionless.hpp           # Dimensionless numbers
│
└── tests/                          # Testing infrastructure
    ├── unit/                       # Unit tests per physics
    ├── convergence/                # Convergence studies
    └── performance/                # Performance benchmarks
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

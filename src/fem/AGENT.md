# AGENT.md - FEM Foundation Module

## Mission
Provide core finite element abstractions and base classes that define the fundamental FEM concepts, independent of specific physics or analysis types, while leveraging the ECS architecture from core/ and mathematical operations from numeric/. Constitutive material models are provided by the standalone `materials/` library and are not implemented under `fem/`.

## Architecture Philosophy
- **Physics-Agnostic**: FEM machinery without specific physics implementation
- **Component-Based**: Elements and nodes as entities with pluggable components
- **Performance-Critical**: Template metaprogramming for compile-time optimization
- **Extensible**: Clear interfaces for custom elements, materials, and formulations
- **Cache-Friendly**: Data layout optimized for modern CPUs

## Directory Structure

```
fem/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
├── core/                            # Core FEM infrastructure
│   ├── fem_traits.hpp               # FEM type traits and concepts
│   ├── fem_concepts.hpp             # C++20 concepts for FEM
│   ├── reference_solutions.hpp      # Reference analytical solutions for testing
│   └── performance_monitoring.hpp   # Performance timing and profiling
│
├── element/                         # Element abstractions
│   ├── element_base.hpp            # Base element interface
│   ├── element_traits.hpp          # Element type traits
│   ├── element_topology.hpp        # Topology definitions
│   ├── element_geometry.hpp        # Geometric mappings
│   ├── element_family.hpp          # Element families (Lagrange, Hermite, etc.)
│   ├── element_entity.hpp          # ECS-based element
│   ├── reference_element.hpp       # Reference element mappings
│   ├── element_registry.hpp        # Element type registry
│   ├── element_orientation.hpp     # Edge/Face orientation handling
│   ├── fem_constants.hpp           # FEM-specific mathematical constants
│   └── types/                      # Concrete element types
│       ├── line/                   # 1D line elements
│       │   ├── line2.hpp           # 2-node line (linear)
│       │   ├── line3.hpp           # 3-node line (quadratic)
│       │   ├── line4.hpp           # 4-node line (cubic)
│       │   └── line_p.hpp          # p-refinement line elements
│       ├── triangle/               # 2D triangular elements
│       │   ├── tri3.hpp            # 3-node triangle (linear)
│       │   ├── tri6.hpp            # 6-node triangle (quadratic)
│       │   ├── tri10.hpp           # 10-node triangle (cubic)
│       │   ├── tri15.hpp           # 15-node triangle (quartic)
│       │   └── tri_p.hpp           # p-refinement triangular elements
│       ├── quadrilateral/          # 2D quadrilateral elements
│       │   ├── quad4.hpp           # 4-node quad (bilinear)
│       │   ├── quad8.hpp           # 8-node quad (biquadratic serendipity)
│       │   ├── quad9.hpp           # 9-node quad (biquadratic Lagrange)
│       │   ├── quad12.hpp          # 12-node quad (bicubic serendipity)
│       │   ├── quad16.hpp          # 16-node quad (bicubic Lagrange)
│       │   └── quad_p.hpp          # p-refinement quadrilateral elements
│       ├── tetrahedron/            # 3D tetrahedral elements
│       │   ├── tet4.hpp            # 4-node tetrahedron (linear)
│       │   ├── tet10.hpp           # 10-node tetrahedron (quadratic)
│       │   ├── tet20.hpp           # 20-node tetrahedron (cubic)
│       │   ├── tet35.hpp           # 35-node tetrahedron (quartic)
│       │   └── tet_p.hpp           # p-refinement tetrahedral elements
│       ├── pyramid/                # 3D pyramid elements (transition elements)
│       │   ├── pyr5.hpp            # 5-node pyramid (linear)
│       │   ├── pyr13.hpp           # 13-node pyramid (quadratic)
│       │   ├── pyr14.hpp           # 14-node pyramid (quadratic alternative)
│       │   └── pyr_p.hpp           # p-refinement pyramid elements
│       ├── hexahedron/             # 3D hexahedral elements
│       │   ├── hex8.hpp            # 8-node hexahedron (trilinear)
│       │   ├── hex20.hpp           # 20-node hexahedron (triquadratic serendipity)
│       │   ├── hex27.hpp           # 27-node hexahedron (triquadratic Lagrange)
│       │   ├── hex32.hpp           # 32-node hexahedron (tricubic serendipity)
│       │   ├── hex64.hpp           # 64-node hexahedron (tricubic Lagrange)
│       │   └── hex_p.hpp           # p-refinement hexahedral elements
│       ├── prism/                  # 3D prism/wedge elements
│       │   ├── prism6.hpp          # 6-node prism (linear)
│       │   ├── prism15.hpp         # 15-node prism (quadratic)
│       │   ├── prism18.hpp         # 18-node prism (quadratic alternative)
│       │   ├── prism24.hpp         # 24-node prism (cubic serendipity)
│       │   └── prism_p.hpp         # p-refinement prism elements
│       ├── structural/             # Structural elements
│       │   ├── beam/               # Beam elements
│       │   │   ├── beam2.hpp       # 2-node Euler-Bernoulli beam
│       │   │   ├── beam3.hpp       # 3-node beam with mid-node
│       │   │   ├── timoshenko.hpp  # Timoshenko beam (shear deformation)
│       │   │   ├── curved_beam.hpp # Curved beam elements
│       │   │   └── composite_beam.hpp # Composite/layered beams
│       │   ├── truss/              # Truss elements
│       │   │   ├── truss2.hpp      # 2-node truss (pin-jointed)
│       │   │   ├── truss3.hpp      # 3-node nonlinear truss
│       │   │   ├── cable.hpp       # Cable elements (tension-only)
│       │   │   └── compression_only.hpp # Compression-only struts
│       │   ├── shell/              # Shell elements
│       │   │   ├── shell3.hpp      # 3-node triangular shell
│       │   │   ├── shell4.hpp      # 4-node quadrilateral shell
│       │   │   ├── shell6.hpp      # 6-node triangular shell
│       │   │   ├── shell8.hpp      # 8-node quadrilateral shell
│       │   │   ├── shell9.hpp      # 9-node quadrilateral shell
│       │   │   ├── facet_shell.hpp # Facet shell elements
│       │   │   ├── curved_shell.hpp # Curved shell elements
│       │   │   └── composite_shell.hpp # Layered/composite shells
│       │   ├── plate/              # Plate elements
│       │   │   ├── kirchhoff.hpp   # Kirchhoff (thin) plate
│       │   │   ├── mindlin.hpp     # Mindlin-Reissner (thick) plate
│       │   │   ├── dkt.hpp         # Discrete Kirchhoff Triangle
│       │   │   ├── dkq.hpp         # Discrete Kirchhoff Quadrilateral
│       │   │   └── argyris.hpp     # Argyris plate element
│       │   ├── membrane/           # Membrane elements
│       │   │   ├── membrane3.hpp   # 3-node membrane
│       │   │   ├── membrane4.hpp   # 4-node membrane
│       │   │   └── membrane6.hpp   # 6-node membrane
│       │   └── spring/             # Spring and mass elements
│       │       ├── spring.hpp      # Spring elements
│       │       ├── dashpot.hpp     # Damper elements
│       │       ├── mass_point.hpp  # Point mass elements
│       │       └── rigid_link.hpp  # Rigid body connections
│       ├── contact/                # Contact interface elements
│       │   ├── node_to_node.hpp    # Node-to-node interface
│       │   ├── node_to_surface.hpp # Node-to-surface interface
│       │   ├── surface_to_surface.hpp # Surface-to-surface interface
│       │   ├── mortar.hpp          # Mortar coupling elements
│       │   ├── penalty.hpp         # Penalty-based interface
│       │   └── lagrange_multiplier.hpp # Lagrange multiplier interface
│       ├── mixed/                  # Mixed formulation elements
│       │   ├── mixed_field.hpp     # General mixed field formulation
│       │   ├── enhanced_strain.hpp # Enhanced strain elements
│       │   ├── assumed_strain.hpp  # Assumed strain elements
│       │   ├── hybrid_stress.hpp   # Hybrid stress elements
│       │   └── bubble_enriched.hpp # Bubble function enriched elements
│       ├── enriched/               # Enriched elements
│       │   ├── xfem/               # Extended FEM
│       │   │   ├── discontinuous.hpp # Discontinuity enrichment
│       │   │   ├── singular.hpp    # Singular function enrichment
│       │   │   ├── interface.hpp   # Interface enrichment
│       │   │   └── level_set.hpp   # Level set enrichment
│       │   ├── gfem/               # Generalized FEM
│       │   │   ├── partition_unity.hpp # Partition of unity method
│       │   │   ├── cloud_points.hpp    # Point cloud methods
│       │   │   └── handbook_functions.hpp # Handbook enrichment functions
│       │   ├── meshfree/           # Meshfree methods
│       │   │   ├── moving_least_squares.hpp # MLS approximation
│       │   │   ├── element_free_galerkin.hpp # EFG method
│       │   │   ├── reproducing_kernel.hpp    # RKPM method
│       │   │   └── natural_element.hpp       # Natural element method
│       │   └── multiscale/         # Multiscale elements
│       │       ├── concurrent.hpp  # Concurrent multiscale
│       │       ├── hierarchical.hpp # Hierarchical multiscale
│       │       └── adaptive.hpp    # Adaptive multiscale
│       └── special/                # Special purpose elements
│           ├── infinite.hpp        # Infinite elements
│           ├── interface.hpp       # Interface elements
│           ├── cohesive.hpp        # Cohesive zone elements
│           ├── absorbing.hpp       # Absorbing boundary elements
│           ├── perfectly_matched_layer.hpp # PML elements
│           ├── gap.hpp             # Gap/constraint elements
│           ├── coupling.hpp        # Coupling elements
│           ├── transition.hpp      # Mesh transition elements
│           ├── rve.hpp             # Representative volume elements
│           ├── periodic.hpp        # Periodic boundary elements
│           ├── superelement.hpp    # Superelements/substructures
│           └── adaptive.hpp        # Adaptive elements
│
├── node/                           # Node and DOF management
│   ├── node_base.hpp               # Base node class
│   ├── node_entity.hpp             # ECS-based node
│   ├── nodal_coordinates.hpp       # Coordinate systems and transformations
│   ├── node_set.hpp                # Node set management and queries
│   ├── connectivity/               # Mesh connectivity management
│   │   ├── mesh_topology.hpp       # Topological mesh representation
│   │   ├── adjacency.hpp           # Node-element adjacency graphs
│   │   ├── boundary_detection.hpp  # Automatic boundary node detection
│   │   └── mesh_partitioning.hpp   # Domain decomposition support
│   ├── dof/                        # Degree of freedom management
│   │   ├── dof.hpp                 # Basic DOF definition
│   │   ├── dof_manager.hpp         # Global DOF numbering and management
│   │   ├── dof_map.hpp             # Local-global DOF mapping
│   │   ├── dof_numbering.hpp       # DOF numbering strategies
│   │   ├── dof_ordering.hpp        # DOF reordering for bandwidth optimization
│   │   ├── hierarchical_dof.hpp    # Hierarchical p-refinement DOF
│   │   ├── field_dof.hpp           # Multi-field DOF management
│   │   └── distributed_dof.hpp     # Parallel/distributed DOF management
│   ├── constraints/                # Constraint management
│   │   ├── constraint_base.hpp     # Base constraint interface
│   │   ├── essential_bc.hpp        # Essential boundary conditions (Dirichlet)
│   │   ├── multipoint_constraint.hpp # Multi-point constraints (MPC)
│   │   ├── rigid_body_element.hpp  # Rigid body connections (RBE)
│   │   ├── periodic_constraint.hpp # Periodic boundary constraints
│   │   ├── contact_constraint.hpp  # Contact/interface constraints
│   │   ├── constraint_solver.hpp   # Constraint elimination algorithms
│   │   └── lagrange_multiplier.hpp # Lagrange multiplier constraints
│   ├── equation_systems/           # Equation numbering and assembly
│   │   ├── equation_numbering.hpp  # Global equation numbering
│   │   ├── sparsity_pattern.hpp    # Matrix sparsity pattern management
│   │   ├── assembly_interface.hpp  # DOF to equation mapping for assembly
│   │   └── bandwidth_optimization.hpp # Bandwidth/profile minimization
│   └── utilities/                  # Node utilities
│       ├── node_locator.hpp        # Spatial node location/search
│       ├── node_merge.hpp          # Node merging and tolerance handling
│       ├── coordinate_transform.hpp # Coordinate system transformations
│       └── node_validation.hpp     # Mesh quality and node validation
│
├── shape/                           # Shape functions
│   ├── shape_function_base.hpp     # Base interface for all shape functions
│   ├── shape_function_cache.hpp    # Caching evaluated shapes and derivatives
│   ├── shape_derivatives.hpp       # Shape function derivative computation
│   ├── interpolation_base.hpp      # Base interpolation interface
│   ├── lagrange/                   # Lagrangian shape functions
│   │   ├── lagrange_1d.hpp         # 1D Lagrange polynomials
│   │   ├── lagrange_2d.hpp         # 2D Lagrange polynomials
│   │   ├── lagrange_3d.hpp         # 3D Lagrange polynomials
│   │   ├── complete_lagrange.hpp   # Complete Lagrange families
│   │   └── tensor_product.hpp      # Tensor product Lagrange
│   ├── hermite/                    # Hermite shape functions
│   │   ├── hermite_1d.hpp          # 1D Hermite polynomials
│   │   ├── hermite_2d.hpp          # 2D Hermite polynomials
│   │   ├── hermite_3d.hpp          # 3D Hermite polynomials
│   │   └── cubic_hermite.hpp       # Cubic Hermite interpolation
│   ├── hierarchical/               # Hierarchical bases
│   │   ├── legendre.hpp            # Legendre polynomial basis
│   │   ├── lobatto.hpp             # Lobatto polynomial basis
│   │   ├── chebyshev.hpp           # Chebyshev polynomial basis
│   │   ├── jacobi.hpp              # Jacobi polynomial basis
│   │   ├── hierarchical_1d.hpp     # 1D hierarchical functions
│   │   ├── hierarchical_2d.hpp     # 2D hierarchical functions
│   │   ├── hierarchical_3d.hpp     # 3D hierarchical functions
│   │   └── p_refinement.hpp        # p-refinement infrastructure
│   ├── serendipity/                # Serendipity elements
│   │   ├── serendipity_2d.hpp      # 2D serendipity elements
│   │   ├── serendipity_3d.hpp      # 3D serendipity elements
│   │   ├── rational_serendipity.hpp # Rational serendipity
│   │   └── general_serendipity.hpp # General serendipity families
│   ├── spectral/                   # Spectral elements
│   │   ├── gauss_lobatto.hpp       # Gauss-Lobatto-Legendre points
│   │   ├── gauss_radau.hpp         # Gauss-Radau points
│   │   ├── gauss_legendre.hpp      # Gauss-Legendre points
│   │   ├── chebyshev_gauss.hpp     # Chebyshev-Gauss points
│   │   └── spectral_basis.hpp      # High-order spectral basis
│   ├── rational/                   # Rational basis functions
│   │   ├── rational_base.hpp       # Base rational function interface
│   │   ├── bezier.hpp              # Bézier basis functions
│   │   ├── b_spline.hpp            # B-spline basis functions
│   │   └── t_spline.hpp            # T-spline basis functions
│   ├── nurbs/                      # NURBS shape functions
│   │   ├── nurbs.hpp               # NURBS basis functions
│   │   ├── nurbs_knot_vector.hpp   # Knot vector management
│   │   ├── nurbs_patch.hpp         # NURBS patch operations
│   │   ├── nurbs_refinement.hpp    # h/p/k-refinement for NURBS
│   │   └── isogeometric.hpp        # Isogeometric analysis support
│   ├── vector_valued/              # Vector-valued shape functions
│   │   ├── vector_lagrange.hpp     # Vector Lagrange functions
│   │   ├── nedelec.hpp             # Nédélec (edge) elements
│   │   ├── raviart_thomas.hpp      # Raviart-Thomas (face) elements
│   │   ├── bdm.hpp                 # Brezzi-Douglas-Marini elements
│   │   └── hdiv_hcurl.hpp          # H(div) and H(curl) conforming
│   ├── enriched/                   # XFEM/GFEM enrichment
│   │   ├── enrichment_base.hpp     # Base enrichment interface
│   │   ├── heaviside.hpp           # Heaviside step functions
│   │   ├── singular.hpp            # Singular enrichment functions
│   │   ├── crack_tip.hpp           # Crack tip enrichment
│   │   ├── void_enrichment.hpp     # Void/inclusion enrichment
│   │   ├── ramp_function.hpp       # Ramp enrichment functions
│   │   ├── ridge_function.hpp      # Ridge enrichment functions
│   │   └── partition_unity.hpp     # Partition of unity functions
│   ├── adaptive/                   # Adaptive refinement
│   │   ├── hp_refinement.hpp       # hp-adaptive refinement
│   │   ├── anisotropic.hpp         # Anisotropic refinement
│   │   ├── bubble_functions.hpp    # Bubble function enrichment
│   │   └── error_indicators.hpp    # Shape function error estimation
│   └── utilities/                  # Shape function utilities
│       ├── shape_evaluator.hpp     # Optimized shape function evaluation
│       ├── derivative_computer.hpp # Automatic differentiation of shapes
│       ├── interpolation_points.hpp # Standard interpolation point sets
│       ├── shape_validator.hpp     # Shape function validation/testing
│       └── basis_transformation.hpp # Basis transformation utilities
│
├── integration/                     # Numerical integration
│   ├── quadrature_rule.hpp         # Base quadrature rule interface
│   ├── quadrature_cache.hpp        # Cache quadrature data and evaluations
│   ├── integration_point.hpp       # Integration point abstraction
│   ├── gauss/                      # Gaussian quadrature families
│   │   ├── gauss_1d.hpp            # 1D Gauss-Legendre rules
│   │   ├── gauss_triangle.hpp      # Triangular Gaussian quadrature
│   │   ├── gauss_quadrilateral.hpp # Quadrilateral Gaussian quadrature
│   │   ├── gauss_tetrahedron.hpp   # Tetrahedral Gaussian quadrature
│   │   ├── gauss_hexahedron.hpp    # Hexahedral Gaussian quadrature
│   │   ├── gauss_pyramid.hpp       # Pyramid Gaussian quadrature
│   │   ├── gauss_prism.hpp         # Prism Gaussian quadrature
│   │   └── gauss_tensor.hpp        # Tensor product construction
│   ├── newton_cotes/               # Newton-Cotes rules
│   │   ├── newton_cotes_1d.hpp     # 1D Newton-Cotes rules
│   │   ├── simpson.hpp             # Simpson's rule
│   │   ├── trapezoidal.hpp         # Trapezoidal rule
│   │   └── composite_rules.hpp     # Composite Newton-Cotes
│   ├── spectral/                   # Spectral integration
│   │   ├── gauss_lobatto.hpp       # Gauss-Lobatto-Legendre
│   │   ├── gauss_radau.hpp         # Gauss-Radau rules
│   │   ├── chebyshev_gauss.hpp     # Chebyshev-Gauss points
│   │   ├── clenshaw_curtis.hpp     # Clenshaw-Curtis quadrature
│   │   └── fejer.hpp               # Fejér quadrature rules
│   ├── adaptive/                   # Adaptive quadrature
│   │   ├── adaptive_1d.hpp         # 1D adaptive quadrature
│   │   ├── adaptive_multidim.hpp   # Multi-dimensional adaptive
│   │   ├── error_estimation.hpp    # Integration error estimation
│   │   ├── subdivision_strategy.hpp # Domain subdivision strategies
│   │   └── sparse_grids.hpp        # Sparse grid integration
│   ├── special/                    # Special integration methods
│   │   ├── singular.hpp            # Singular integral handling
│   │   ├── oscillatory.hpp         # Oscillatory integrand methods
│   │   ├── surface.hpp             # Surface integration
│   │   ├── line_integral.hpp       # Line integral computation
│   │   ├── weakly_singular.hpp     # Weakly singular integrals
│   │   └── boundary_element.hpp    # Boundary element integration
│   ├── cut_cell/                   # Cut-cell integration
│   │   ├── cut_cell_base.hpp       # Base cut-cell integration
│   │   ├── level_set.hpp           # Level set cut-cell
│   │   ├── moment_fitting.hpp      # Moment fitting methods
│   │   ├── decomposition.hpp       # Cell decomposition strategies
│   │   └── stabilization.hpp       # Cut-cell stabilization
│   ├── isogeometric/               # IGA-specific integration
│   │   ├── nurbs_quadrature.hpp    # NURBS quadrature rules
│   │   ├── optimal_rules.hpp       # ML-optimized quadrature
│   │   ├── reduced_rules.hpp       # Reduced integration schemes
│   │   └── element_splitting.hpp   # Element splitting for IGA
│   ├── high_order/                 # High-order integration
│   │   ├── exactness_rules.hpp     # Polynomial exactness requirements
│   │   ├── degree_freedom.hpp      # Degree of freedom mapping
│   │   ├── integration_order.hpp   # Order selection strategies
│   │   └── curved_elements.hpp     # Curved element integration
│   └── utilities/                  # Integration utilities
│       ├── rule_generator.hpp      # Quadrature rule generation
│       ├── weight_optimizer.hpp    # Weight optimization algorithms
│       ├── point_optimizer.hpp     # Point location optimization
│       ├── accuracy_test.hpp       # Integration accuracy testing
│       └── performance_timer.hpp   # Integration performance profiling
│
├── field/                           # Field variables and operations
│   ├── field_base.hpp              # Base field interface
│   ├── field_registry.hpp          # Field type registration and management
│   ├── field_descriptor.hpp        # Field metadata and properties
│   ├── field_component.hpp         # ECS field component
│   ├── types/                      # Field type definitions
│   │   ├── scalar_field.hpp        # Scalar field variables
│   │   ├── vector_field.hpp        # Vector field variables
│   │   ├── tensor_field.hpp        # Tensor field variables
│   │   ├── complex_field.hpp       # Complex-valued fields
│   │   └── quaternion_field.hpp    # Quaternion fields
│   ├── operations/                 # Field operations
│   │   ├── field_arithmetic.hpp    # Field arithmetic operations
│   │   ├── field_interpolation.hpp # Field interpolation between points
│   │   ├── field_gradient.hpp      # Gradient computation
│   │   ├── field_divergence.hpp    # Divergence computation
│   │   ├── field_curl.hpp          # Curl computation
│   │   ├── field_laplacian.hpp     # Laplacian computation
│   │   └── field_projection.hpp    # Field projection operations
│   ├── storage/                    # Field storage and access
│   │   ├── field_storage.hpp       # Field value storage strategies
│   │   ├── field_history.hpp       # Time history storage
│   │   ├── distributed_field.hpp   # Distributed field storage
│   │   ├── compressed_field.hpp    # Compressed field storage
│   │   └── cached_field.hpp        # Cached field computations
│   ├── transfer/                   # Field transfer operations
│   │   ├── mesh_transfer.hpp       # Field transfer between meshes
│   │   ├── interpolation_transfer.hpp # Interpolation-based transfer
│   │   ├── projection_transfer.hpp # Projection-based transfer
│   │   ├── conservative_transfer.hpp # Conservative field transfer
│   │   └── mortar_transfer.hpp     # Mortar-based field transfer
│   ├── boundaries/                 # Field boundary handling
│   │   ├── boundary_extraction.hpp # Extract boundary field values
│   │   ├── boundary_conditions.hpp # Field-specific boundary conditions
│   │   ├── flux_computation.hpp    # Boundary flux computation
│   │   └── normal_computation.hpp  # Normal vector computation
│   └── utilities/                  # Field utilities
│       ├── field_validator.hpp     # Field validation and consistency
│       ├── field_analyzer.hpp      # Field analysis and statistics
│       ├── field_io.hpp            # Field input/output operations
│       └── field_visualization.hpp # Field visualization support
│
├── boundary/                        # Boundary conditions
│   ├── boundary_condition_base.hpp # Base BC interface
│   ├── dirichlet.hpp               # Dirichlet BCs
│   ├── neumann.hpp                 # Neumann BCs
│   ├── robin.hpp                   # Robin BCs
│   ├── periodic.hpp                # Periodic BCs
│   ├── multipoint_constraint.hpp   # MPC
│   ├── boundary_integral.hpp       # Boundary integrals
│   ├── natural_bc.hpp              # Natural BCs
│   ├── nitsche.hpp                 # Nitsche BCs
│   ├── periodic_affine.hpp         # Periodic Affine BCs
│   ├── mortar.hpp                  # Mortar coupling
│   └── essential_bc.hpp            # Essential BCs
│
├── basis/                           # Mathematical basis abstractions
│   ├── basis_base.hpp              # Base mathematical basis interface
│   ├── basis_properties.hpp        # Basis mathematical properties
│   ├── basis_factory.hpp           # Basis creation and management
│   ├── polynomial/                 # Polynomial basis families
│   │   ├── monomial.hpp            # Monomial basis (1, x, x², ...)
│   │   ├── bernstein.hpp           # Bernstein polynomial basis
│   │   ├── newton.hpp              # Newton polynomial basis
│   │   ├── lagrange_basis.hpp      # Lagrange interpolating basis
│   │   └── hermite_basis.hpp       # Hermite polynomial basis
│   ├── orthogonal/                 # Orthogonal polynomial families
│   │   ├── legendre.hpp            # Legendre polynomials
│   │   ├── chebyshev.hpp           # Chebyshev polynomials (1st/2nd kind)
│   │   ├── jacobi.hpp              # Jacobi polynomials
│   │   ├── laguerre.hpp            # Laguerre polynomials
│   │   ├── hermite_orthogonal.hpp  # Orthogonal Hermite polynomials
│   │   └── zernike.hpp             # Zernike polynomials
│   ├── spectral/                   # Spectral basis functions
│   │   ├── fourier.hpp             # Fourier series basis
│   │   ├── sine_cosine.hpp         # Trigonometric basis functions
│   │   ├── exponential.hpp         # Complex exponential basis
│   │   └── spherical_harmonics.hpp # Spherical harmonic basis
│   ├── wavelets/                   # Wavelet basis families
│   │   ├── haar.hpp                # Haar wavelets
│   │   ├── daubechies.hpp          # Daubechies wavelets
│   │   ├── biorthogonal.hpp        # Biorthogonal wavelets
│   │   ├── coiflets.hpp            # Coiflet wavelets
│   │   └── meyer.hpp               # Meyer wavelets
│   ├── splines/                    # Spline basis functions
│   │   ├── b_spline_basis.hpp      # B-spline basis functions
│   │   ├── nurbs_basis.hpp         # NURBS basis functions
│   │   ├── catmull_rom.hpp         # Catmull-Rom spline basis
│   │   └── cubic_spline.hpp        # Cubic spline basis
│   ├── radial/                     # Radial basis functions
│   │   ├── multiquadric.hpp        # Multiquadric RBF
│   │   ├── gaussian.hpp            # Gaussian RBF
│   │   ├── thin_plate_spline.hpp   # Thin plate spline RBF
│   │   ├── inverse_multiquadric.hpp # Inverse multiquadric RBF
│   │   └── polyharmonic.hpp        # Polyharmonic RBF
│   ├── modal/                      # Modal basis representations
│   │   ├── modal_base.hpp          # Base modal representation
│   │   ├── eigenmode.hpp           # Eigenmode basis
│   │   ├── frequency_domain.hpp    # Frequency domain basis
│   │   └── vibration_mode.hpp      # Structural vibration modes
│   ├── nodal/                      # Nodal basis representations
│   │   ├── nodal_base.hpp          # Base nodal representation
│   │   ├── point_evaluation.hpp    # Point evaluation functionals
│   │   ├── cardinal_basis.hpp      # Cardinal basis functions
│   │   └── interpolation_basis.hpp # Interpolation-based basis
│   └── utilities/                  # Basis utilities
│       ├── basis_transformation.hpp # Basis change transformations
│       ├── orthogonalization.hpp  # Gram-Schmidt orthogonalization
│       ├── normalization.hpp      # Basis normalization methods
│       ├── conditioning.hpp       # Basis conditioning analysis
│       └── completeness.hpp       # Basis completeness verification
│
├── mapping/                        # Coordinate mappings
│   ├── mapping_base.hpp            # Base mapping interface
│   ├── bezier_extraction.hpp       # IGA patch space abstraction
│   ├── isoparametric.hpp           # Isoparametric mapping
│   ├── subparametric.hpp           # Subparametric mapping
│   ├── superparametric.hpp         # Superparametric mapping
│   ├── jacobian.hpp                # Jacobian computation
│   ├── coordinate_transform.hpp    # Coordinate transformations
│   └── mapping_cache.hpp           # Cache mapping data
│
├── interpolation/                   # Interpolation methods
│   ├── interpolator_base.hpp       # Base interpolator
│   ├── nodal_interpolation.hpp     # Node-based interpolation
│   ├── least_squares.hpp           # L2 projection
│   ├── patch_recovery.hpp          # SPR/ZZ recovery
│   ├── moving_least_squares.hpp    # MLS interpolation
│   └── radial_basis.hpp            # RBF interpolation
│
├── formulation/                     # FEM formulations
│   ├── formulation_base.hpp        # Base formulation interface
│   ├── weak_form_interface.hpp     # Weak form mathematical interface
│   ├── assembly_strategy.hpp       # Assembly strategy patterns
│   ├── galerkin/                   # Galerkin method family
│   │   ├── standard_galerkin.hpp   # Standard Bubnov-Galerkin
│   │   ├── petrov_galerkin.hpp     # Petrov-Galerkin method
│   │   ├── weighted_galerkin.hpp   # Weighted residual methods
│   │   ├── streamline_upwind.hpp   # SUPG/PSPG stabilization
│   │   └── galerkin_least_squares.hpp # GLS stabilization
│   ├── discontinuous/              # Discontinuous Galerkin methods
│   │   ├── dg_base.hpp             # Base DG formulation
│   │   ├── interior_penalty.hpp    # Interior penalty DG
│   │   ├── local_dg.hpp            # Local discontinuous Galerkin
│   │   ├── hybridizable_dg.hpp     # Hybridizable DG (HDG)
│   │   ├── embedded_dg.hpp         # Embedded discontinuous Galerkin
│   │   ├── flux_reconstruction.hpp # Flux reconstruction methods
│   │   └── dg_stabilization.hpp    # DG stabilization techniques
│   ├── mixed/                      # Mixed formulations
│   │   ├── mixed_base.hpp          # Base mixed formulation
│   │   ├── two_field_mixed.hpp     # Two-field mixed methods
│   │   ├── three_field_mixed.hpp   # Three-field mixed methods
│   │   ├── hybrid_mixed.hpp        # Hybrid mixed methods
│   │   ├── enhanced_strain.hpp     # Enhanced strain formulations
│   │   ├── assumed_strain.hpp      # Assumed strain methods
│   │   └── saddle_point.hpp        # Saddle point problem handling
│   ├── stabilized/                 # Stabilized methods
│   │   ├── stabilization_base.hpp  # Base stabilization interface
│   │   ├── supg.hpp                # Streamline upwind Petrov-Galerkin
│   │   ├── pspg.hpp                # Pressure stabilizing Petrov-Galerkin
│   │   ├── gls.hpp                 # Galerkin least squares
│   │   ├── cip.hpp                 # Continuous interior penalty
│   │   ├── residual_free_bubbles.hpp # Residual-free bubble methods
│   │   └── variational_multiscale.hpp # Variational multiscale methods
│   ├── least_squares/              # Least squares methods
│   │   ├── standard_ls.hpp         # Standard least squares FEM
│   │   ├── weighted_ls.hpp         # Weighted least squares
│   │   ├── first_order_ls.hpp      # First-order system least squares
│   │   ├── div_ls.hpp              # Divergence-based least squares
│   │   └── constrained_ls.hpp      # Constrained least squares
│   ├── collocation/                # Collocation methods
│   │   ├── point_collocation.hpp   # Point collocation methods
│   │   ├── subdomain_collocation.hpp # Subdomain collocation
│   │   ├── boundary_collocation.hpp # Boundary collocation
│   │   └── spectral_collocation.hpp # Spectral collocation
│   ├── meshfree/                   # Meshfree formulations
│   │   ├── meshfree_base.hpp       # Base meshfree formulation
│   │   ├── element_free_galerkin.hpp # Element-free Galerkin
│   │   ├── moving_least_squares.hpp # Moving least squares
│   │   ├── reproducing_kernel.hpp  # Reproducing kernel particle method
│   │   ├── natural_element.hpp     # Natural element method
│   │   └── partition_unity_fem.hpp # Partition of unity finite element
│   ├── multiscale/                 # Multiscale formulations
│   │   ├── multiscale_base.hpp     # Base multiscale interface
│   │   ├── variational_multiscale.hpp # Variational multiscale method
│   │   ├── heterogeneous_multiscale.hpp # Heterogeneous multiscale method
│   │   ├── computational_homogenization.hpp # Computational homogenization
│   │   └── concurrent_multiscale.hpp # Concurrent multiscale coupling
│   ├── nonlinear/                  # Nonlinear formulations
│   │   ├── nonlinear_base.hpp      # Base nonlinear formulation (interfaces residual/Jacobian assembly)
│   │   └── incremental_form.hpp    # Incremental update patterns (delegates solves to solver library)
│   ├── time_dependent/             # Time-dependent formulations
│   │   ├── mixed_form_time.hpp     # Space-time weak forms and residual/Jacobian assembly
│   │   └── space_time_basis.hpp    # Space-time basis support (delegates time stepping to solvers)
│   └── utilities/                  # Formulation utilities
│       ├── formulation_factory.hpp # Formulation creation and selection
│       ├── convergence_monitor.hpp # Convergence monitoring
│       ├── residual_computer.hpp   # Residual computation utilities
│       ├── jacobian_computer.hpp   # Jacobian computation utilities
│       └── method_validator.hpp    # Formulation validation and testing
│
├── variational/                     # Variational form language (UFL-inspired)
│   ├── README.md                   # Overview of variational forms system
│   ├── form_language.hpp           # Core DSL for variational forms
│   ├── form_compiler.hpp           # Orchestrates IR generation and lowering
│   ├── symbolic/                   # Symbolic expression system
│   │   ├── expression.hpp          # Base symbolic expression
│   │   ├── test_function.hpp       # Test functions (v)
│   │   ├── trial_function.hpp      # Trial functions (u)
│   │   ├── coefficient.hpp         # Problem coefficients
│   │   ├── operators.hpp           # Differential operators (grad, div, curl)
│   │   ├── facet_operators.hpp     # DG/interface helpers (n(), jump(), avg())
│   │   ├── time_operators.hpp      # dt(), transient weighting helpers
│   │   ├── ale_operators.hpp       # Material derivative, pullbacks for ALE
│   │   ├── functionals.hpp         # Linear/bilinear functionals
│   │   └── algebra.hpp             # Symbolic algebra operations
│   ├── forms/                      # Form definitions & reusable terms
│   │   ├── bilinear_form.hpp       # a(u,v) bilinear forms
│   │   ├── linear_form.hpp         # L(v) linear forms
│   │   ├── nonlinear_form.hpp      # Nonlinear forms F(u;v)
│   │   ├── mixed_form.hpp          # Mixed variational forms
│   │   ├── block_form.hpp          # Block/multi-field helpers
│   │   ├── stabilization/          # Stabilization terms (SUPG/PSPG/GLS/grad-div)
│   │   ├── interface/              # Interface weak enforcement (Nitsche, mortar)
│   │   └── functional.hpp          # Energy/objective functionals
│   ├── integration/                # Variational integration (wrapping fem/integration)
│   │   ├── variational_integrator.hpp # Bridges to fem/integration quadrature
│   │   ├── domain_integrator.hpp   # Domain integrals
│   │   ├── boundary_integrator.hpp # Boundary integrals
│   │   ├── interface_integrator.hpp # Interior/exterior facet integrals
│   │   └── measure.hpp             # Integration measures (dx, ds, dS)
│   ├── spaces/                     # Space adapters (lightweight)
│   │   ├── space_adapter.hpp       # Bridge fem/spaces types into DSL metadata
│   │   ├── mixed_space_adapter.hpp # Mixed/product space helpers
│   │   └── space_hierarchy.hpp     # Metadata for refinement (uses adaptation module for actual AMR)
│   ├── ir/                         # Intermediate representation & analysis passes
│   │   ├── form_ir.hpp             # Normalized IR (nodes, annotations)
│   │   ├── shape_inference.hpp     # Rank/shape checking, conformity analysis
│   │   ├── analysis_passes.hpp     # CSE, constant folding, strength reduction
│   │   ├── sparsity_inference.hpp  # Symbolic sparsity pattern inference
│   │   └── rewrite_rules.hpp       # Algebraic identities/simplifications
│   ├── lowering/                   # Lowering IR to execution backends
│   │   ├── lowering_config.hpp     # Backend/optimization flags
│   │   ├── lower_assembled.hpp     # Emit assembled sparse/block operators
│   │   ├── lower_matrix_free.hpp   # Emit matrix-free/sum-factorized kernels
│   │   └── lower_ceed.hpp          # Optional libCEED/libParanumal bridges
│   ├── assembly/                   # Consumers of lowered artifacts
│   │   ├── form_assembler.hpp      # Assemble using lowered operators
│   │   ├── assembly_kernel.hpp     # Generated assembly kernels
│   │   ├── code_generation.hpp     # C++/GPU code emission
│   │   ├── sparsity_pattern.hpp    # Preallocation helper using IR sparsity
│   │   └── optimization.hpp        # Kernel/block optimization utilities
│   ├── examples/                   # Example variational forms
│   │   ├── poisson.hpp             # Poisson equation
│   │   ├── elasticity.hpp          # Linear elasticity
│   │   ├── stokes.hpp              # Stokes flow
│   │   ├── navier_stokes.hpp       # Navier-Stokes
│   │   └── heat_equation.hpp       # Heat equation
│   └── utilities/                  # Variational utilities
│       ├── form_printer.hpp        # Pretty print forms
│       ├── form_validator.hpp      # Validate form consistency
│       ├── derivative_computer.hpp # Automatic differentiation / linearization
│       ├── form_parser.hpp         # Parse mathematical notation
│       ├── form_profiler.hpp       # FLOP/cost estimation for kernels
│       └── error_messages.hpp      # Friendly diagnostics (shape/space mismatch)
│
├── spaces/                          # Function spaces
│   ├── function_space_base.hpp     # Base function space interface
│   ├── space_properties.hpp        # Mathematical space properties
│   ├── space_factory.hpp           # Function space creation and management
│   ├── sobolev/                    # Sobolev spaces
│   │   ├── h1_space.hpp            # H¹(Ω) conforming spaces
│   │   ├── h2_space.hpp            # H²(Ω) spaces for C¹ problems
│   │   ├── h_curl_space.hpp        # H(curl) spaces for electromagnetics
│   │   ├── h_div_space.hpp         # H(div) spaces for fluid mechanics
│   │   ├── h_grad_space.hpp        # H(grad) spaces for potential problems
│   │   └── broken_sobolev.hpp      # Broken Sobolev spaces for DG
│   ├── lebesgue/                   # Lebesgue spaces
│   │   ├── l2_space.hpp            # L²(Ω) spaces
│   │   ├── lp_space.hpp            # Lᵖ(Ω) spaces (p ≠ 2)
│   │   ├── l_infinity.hpp          # L^∞(Ω) spaces
│   │   └── weighted_l2.hpp         # Weighted L² spaces
│   ├── mixed/                      # Mixed and product spaces
│   │   ├── product_space.hpp       # Cartesian product spaces
│   │   ├── composite_space.hpp     # Composite field spaces
│   │   ├── taylor_hood.hpp         # Taylor-Hood elements (velocity-pressure)
│   │   ├── mini_element.hpp        # MINI elements with bubble enrichment
│   │   ├── crouzeix_raviart.hpp    # Crouzeix-Raviart nonconforming
│   │   └── mixed_space_builder.hpp # General mixed space construction
│   ├── enriched/                   # Enriched and extended spaces
│   │   ├── xfem_space.hpp          # XFEM enriched spaces
│   │   ├── gfem_space.hpp          # GFEM enriched spaces
│   │   ├── partition_unity.hpp     # Partition of unity spaces
│   │   ├── bubble_enriched.hpp     # Bubble function enrichment
│   │   ├── singular_enriched.hpp   # Singular function enrichment
│   │   └── meshfree_space.hpp      # Meshfree approximation spaces
│   ├── conforming/                 # Conforming spaces
│   │   ├── c0_continuous.hpp       # C⁰ continuous spaces
│   │   ├── c1_continuous.hpp       # C¹ continuous spaces
│   │   ├── global_continuous.hpp   # Globally continuous spaces
│   │   └── conforming_constraints.hpp # Conformity constraint handling
│   ├── nonconforming/              # Nonconforming spaces
│   │   ├── discontinuous.hpp       # Discontinuous Galerkin spaces
│   │   ├── crouzeix_raviart.hpp    # Crouzeix-Raviart spaces
│   │   ├── morley.hpp              # Morley elements
│   │   ├── nonconforming_base.hpp  # Base nonconforming interface
│   │   └── jump_operators.hpp      # Jump and average operators
│   ├── spectral/                   # Spectral spaces
│   │   ├── fourier_space.hpp       # Fourier basis spaces
│   │   ├── legendre_space.hpp      # Legendre polynomial spaces
│   │   ├── chebyshev_space.hpp     # Chebyshev polynomial spaces
│   │   ├── spectral_element.hpp    # High-order spectral elements
│   │   └── modal_space.hpp         # Modal basis representations
│   ├── adaptive/                   # Adaptive spaces
│   │   ├── hp_adaptive.hpp         # hp-adaptive spaces
│   │   ├── hierarchical_space.hpp  # Hierarchical refinement
│   │   ├── space_hierarchy.hpp     # Multi-level space hierarchies
│   │   ├── local_refinement.hpp    # Local space refinement
│   │   └── anisotropic_space.hpp   # Anisotropic refinement
│   ├── trace/                      # Trace and boundary spaces
│   │   ├── trace_space.hpp         # Trace spaces on boundaries
│   │   ├── boundary_space.hpp      # Boundary element spaces
│   │   ├── mortar_space.hpp        # Mortar coupling spaces
│   │   ├── interface_space.hpp     # Interface spaces
│   │   └── skeleton_space.hpp      # Skeleton spaces for HDG
│   └── utilities/                  # Space utilities
│       ├── space_validator.hpp     # Space consistency validation
│       ├── space_analyzer.hpp      # Space property analysis
│       ├── dof_extraction.hpp      # DOF extraction from spaces
│       ├── space_transfer.hpp      # Transfer between spaces
│       └── space_visualization.hpp # Space visualization support
│
├── error/                           # Error estimation
│   ├── error_estimator_base.hpp    # Base estimator
│   ├── residual_estimator.hpp      # Residual-based
│   ├── recovery_estimator.hpp      # Recovery-based
│   ├── goal_oriented.hpp           # Goal-oriented
│   ├── error_indicator.hpp         # Error indicators
│   └── effectivity_index.hpp       # Effectivity computation
│
├── python/                         # Python bindings for FEM (pybind11-based)
│   ├── CMakeLists.txt              # Build configuration for Python module
│   ├── module.cpp                  # Top-level pybind11 module definition
│   ├── element_bindings.cpp        # Element/topology bindings
│   ├── node_bindings.cpp           # Node/DOF management bindings
│   ├── shape_bindings.cpp          # Shape function bindings
│   ├── field_bindings.cpp          # Field variable bindings
│   ├── integration_bindings.cpp    # Quadrature/integration bindings
│   ├── assembly_bindings.cpp       # Assembly helpers (maps lowered forms to element assembly)
│   ├── formulation_bindings.cpp    # Weak-form assembly wrappers (consume variational outputs)
│   ├── solver_bindings.cpp         # Bridges to top-level solvers (assembly/solve entry points)
│   ├── materials_bindings.cpp      # Optional bridge to top-level materials library
│   ├── utilities_bindings.cpp      # Diagnostics/profiling/IO helpers
│   └── __init__.py                 # Python package initializer
│
├── special/                         # Special FEM methods
│   ├── xfem/                       # Extended FEM
│   │   ├── level_set.hpp
│   │   ├── enrichment.hpp
│   │   └── crack.hpp
│   ├── gfem/                       # Generalized FEM
│   │   └── partition_of_unity.hpp
│   ├── sfem/                       # Smooth FEM
│   │   └── smooth_basis.hpp
│   ├── mfem/                       # Meshfree FEM
│   │   └── meshfree_shape.hpp
│   └── vfem/                       # Virtual FEM
│       └── virtual_element.hpp
│
│
├── tests/                           # Testing
│   ├── unit/                        # Unit tests
│   ├── convergence/                 # Convergence tests
│   └── patch/                       # Patch tests
└── benchmarks/                      # Performance benchmarks
```

## Element Classification System

The fem/element/types/ directory provides a comprehensive taxonomy of finite element types organized by topology, mathematical formulation, and computational features. Elements are physics-agnostic building blocks that physics modules compose to create domain-specific formulations:

### Standard Topology Elements

#### **1D Line Elements**
- **Linear (line2)**: Basic truss, beam backbone
- **Quadratic (line3)**: Higher-order accuracy, curved geometry
- **Cubic (line4)**: High-order approximation
- **p-refinement**: Variable polynomial order for adaptive accuracy

#### **2D Planar Elements**
- **Triangular Family**: Natural for complex geometry, automatic meshing
  - `tri3`: Linear, minimal DOF, basic analysis
  - `tri6`: Quadratic, good balance of accuracy/cost
  - `tri10`: Cubic, high accuracy applications
  - `tri15`: Quartic, spectral-level accuracy
- **Quadrilateral Family**: Structured meshes, better aspect ratio handling
  - `quad4`: Bilinear, computational efficiency
  - `quad8/9`: Biquadratic, curved boundaries
  - `quad12/16`: Bicubic, high-order analysis

#### **3D Volumetric Elements**
- **Tetrahedral Elements**: Automatic meshing, complex geometries
  - `tet4`: Linear, poor performance, use sparingly
  - `tet10`: Quadratic, general purpose 3D analysis
  - `tet20/35`: High-order for demanding accuracy
- **Hexahedral Elements**: Best performance, structured meshes
  - `hex8`: Trilinear, computational workhorse
  - `hex20/27`: Triquadratic, curved geometry handling
  - `hex32/64`: Tricubic, high-fidelity analysis
- **Pyramid Elements**: Critical for hybrid hex-tet meshes
  - `pyr5/13/14`: Transition between hex and tet regions
- **Prism/Wedge Elements**: Boundary layer meshing, extrusion
  - `prism6/15/18/24`: Good for layered domains

### Structural Engineering Elements

#### **Beam Elements**
- **Euler-Bernoulli**: Classical beam theory, slender members
- **Timoshenko**: Shear deformation effects, thick beams
- **Curved**: Non-straight members, arches
- **Composite**: Multi-material layered construction

#### **Shell Elements**
- **Triangular/Quadrilateral**: Membrane + bending, thin structures
- **Facet**: Discrete shell modeling approach
- **Curved**: Smooth shell surface representation
- **Composite**: Layered shell construction

#### **Plate Elements**
- **Kirchhoff**: Thin plate theory, classical approach
- **Mindlin-Reissner**: Thick plate, shear deformation
- **DKT/DKQ**: Discrete Kirchhoff formulations

### Interface and Coupling Elements

#### **Contact/Interface Elements**
- **Node-to-node**: Point coupling and constraint enforcement
- **Node-to-surface**: Point-to-manifold coupling
- **Surface-to-surface**: Manifold-to-manifold coupling
- **Mortar**: Mathematically optimal interface treatment
- **Penalty**: Penalty-based constraint enforcement
- **Lagrange multiplier**: Exact constraint enforcement

### Advanced Mathematical Formulations

#### **Mixed Elements**
- **Mixed field formulation**: General multi-field coupling
- **Enhanced strain**: Improved element performance
- **Assumed strain**: Elimination of locking phenomena
- **Hybrid stress**: Stress-based formulations
- **Bubble enriched**: Interior enrichment functions

#### **Enriched Elements**
- **XFEM**: Discontinuity modeling without mesh conformity
  - Arbitrary discontinuities, singular enrichment, level set interfaces
- **GFEM**: Partition of unity enrichment
  - Handbook functions, cloud-based methods, local enrichment
- **Meshfree**: Non-mesh-based approximation
  - MLS, EFG, RKPM, natural element methods
- **Multiscale**: Scale-bridging formulations
  - Concurrent, hierarchical, adaptive approaches

### Special Purpose Elements

- **Infinite elements**: Far-field boundary conditions
- **Interface elements**: General interface modeling
- **Cohesive elements**: Zone-based interface modeling
- **Absorbing elements**: Wave absorption at boundaries
- **PML elements**: Perfectly matched layers
- **Gap elements**: Constraint gap modeling
- **Coupling elements**: Multi-domain coupling
- **Transition elements**: Mesh transition handling
- **Superelement**: Reduced-order substructures

### Element Selection Guidelines

#### **By Geometry Complexity**
- **Simple regular geometry**: Hexahedral elements preferred
- **Moderate complexity**: Mixed hex-tet with pyramid transitions
- **Complex arbitrary geometry**: Tetrahedral elements
- **Thin structures**: Shell/plate elements
- **Slender members**: Beam/truss elements

#### **By Mathematical Requirements**
- **Standard field problems**: Continuum elements (hex, tet, prism)
- **Incompressible formulations**: Mixed elements with pressure DOF
- **Multi-field coupling**: Mixed formulation elements
- **Interface problems**: Contact/mortar elements
- **Discontinuity modeling**: XFEM enriched elements
- **High-gradient regions**: Enhanced strain or bubble elements

#### **By Accuracy Requirements**
- **Engineering accuracy**: Linear/quadratic elements
- **Research accuracy**: Higher-order elements
- **Spectral accuracy**: p-refinement elements
- **Adaptive analysis**: hp-refinement capable elements

#### **By Computational Requirements**
- **Performance-critical**: Low-order elements (tet4, hex8, quad4)
- **Accuracy-performance balance**: Mid-order elements (tet10, hex20)
- **High-fidelity**: High-order elements (tet20, hex27, p-refinement)
- **Memory-constrained**: Linear elements with adaptive refinement

## Physics-Agnostic Building Block Approach

The fem/ library provides mathematical and topological element abstractions that physics modules compose to create domain-specific formulations. This architecture separates concerns:

### **FEM Library Responsibilities**
- **Element topology**: Node connectivity and geometric mapping
- **Shape functions**: Basis function evaluation and derivatives
- **Numerical integration**: Quadrature rules and point management
- **DOF management**: Global numbering and constraint handling
- **Assembly interface**: Matrix/vector assembly from element contributions
- **Mathematical formulations**: Mixed methods, enrichment, interface coupling

### **Physics Module Responsibilities**
- **Field definitions**: What variables are solved for (displacement, temperature, etc.)
- **Constitutive relations**: Material laws and governing equations
- **Boundary conditions**: Physics-specific boundary conditions
- **Source terms**: Body forces, heat sources, electromagnetic sources
- **Coupling terms**: Inter-field coupling in multiphysics problems

### **Composition Examples**

#### **Structural Mechanics**
```cpp
// Physics module composes FEM building blocks
auto element = fem::ElementFactory::create("hex20");
element->add_field("displacement", 3);  // 3D displacement field
element->add_material(material_component);
element->set_formulation(fem::GalerkinFormulation{});

// Physics defines the weak form using variational language
auto weak_form = inner(sigma(grad(u)), grad(v)) * dx;
```

#### **Fluid Mechanics**
```cpp
// Same element, different physics
auto element = fem::ElementFactory::create("hex20");
element->add_field("velocity", 3);
element->add_field("pressure", 1);
element->set_formulation(fem::MixedFormulation{});

// Physics defines Stokes/Navier-Stokes equations
auto weak_form = inner(grad(u), grad(v)) * dx +
                 inner(grad(p), v) * dx;
```

#### **Thermal Analysis**
```cpp
// Same element, thermal physics
auto element = fem::ElementFactory::create("hex20");
element->add_field("temperature", 1);
element->add_material(thermal_material);

// Physics defines heat equation
auto weak_form = inner(k * grad(T), grad(v)) * dx +
                 rho * cp * inner(dT_dt, v) * dx;
```

### **Benefits of This Approach**

1. **Reusability**: Same element can support multiple physics
2. **Extensibility**: New physics modules use existing element infrastructure
3. **Maintainability**: Physics-specific code isolated from mathematical infrastructure
4. **Performance**: Element evaluation optimized independently of physics
5. **Consistency**: Common interface across all physics domains
6. **Modularity**: Physics modules can be developed independently

### **Element-Physics Interface**

Elements provide standardized interfaces that physics modules use:

```cpp
// Generic element interface
class Element {
    // Topology and geometry
    virtual Matrix shape_functions(Point xi) const = 0;
    virtual Tensor shape_derivatives(Point xi) const = 0;
    virtual double jacobian_determinant(Point xi) const = 0;

    // DOF management
    virtual void add_field(string name, int components) = 0;
    virtual DOFMap get_dof_map(string field) const = 0;

    // Assembly interface
    virtual void contribute_to_matrix(string field1, string field2,
                                    MatrixContribution& contrib) = 0;
    virtual void contribute_to_vector(string field,
                                    VectorContribution& contrib) = 0;
};
```

This design allows physics modules to focus on their domain expertise while leveraging robust, optimized mathematical infrastructure.

## Node and DOF Management Architecture

The fem/node/ directory provides comprehensive infrastructure for node management, degree of freedom handling, and constraint enforcement - all physics-agnostic building blocks.

### **Node Management Responsibilities**

#### **Basic Node Operations**
- **Node entities**: ECS-based nodes with coordinate and connectivity information
- **Coordinate systems**: Support for Cartesian, cylindrical, spherical coordinate systems
- **Node sets**: Efficient node grouping and query capabilities
- **Spatial operations**: Node location, merging, and validation

#### **Mesh Connectivity**
- **Topological representation**: Complete mesh topology with node-element relationships
- **Adjacency graphs**: Efficient node-element and element-element adjacency
- **Boundary detection**: Automatic identification of boundary nodes and surfaces
- **Mesh partitioning**: Support for domain decomposition in parallel computing

#### **DOF Management**
- **Multi-field DOF**: Support for arbitrary field variables (displacement, temperature, pressure, etc.)
- **Hierarchical DOF**: p-refinement with hierarchical basis functions
- **Distributed DOF**: Parallel/distributed memory DOF management
- **DOF numbering**: Optimized numbering strategies for bandwidth minimization

#### **Constraint Handling**
- **Essential boundary conditions**: Dirichlet-type constraints
- **Multi-point constraints (MPC)**: Linear relationships between DOF
- **Rigid body elements (RBE)**: Rigid connections and interpolation
- **Periodic constraints**: Periodic boundary condition enforcement
- **Contact constraints**: Interface constraint management

### **Shape Function Architecture**

The fem/shape/ directory provides a comprehensive taxonomy of shape functions and basis functions for finite element interpolation.

#### **Classical Shape Function Families**

##### **Lagrange Shape Functions**
- **Complete polynomial spaces**: Full polynomial completeness for accuracy
- **Tensor product construction**: Efficient construction for hexahedral elements
- **Arbitrary order**: Linear through very high-order polynomials
- **Nodal basis**: Point-wise interpolation properties

##### **Hermite Shape Functions**
- **C¹ continuity**: First derivative continuity across elements
- **Higher-order continuity**: Support for beam, plate, and shell elements
- **Mixed interpolation**: Position and derivative degrees of freedom

##### **Serendipity Elements**
- **Edge-based construction**: Nodes only on element edges
- **Reduced DOF count**: Fewer nodes than complete Lagrange
- **Rational variants**: Improved performance for specific geometries

#### **Advanced Shape Function Types**

##### **Hierarchical Bases**
- **p-refinement ready**: Natural support for adaptive order refinement
- **Orthogonal construction**: Legendre, Chebyshev, Jacobi polynomial bases
- **Condition number control**: Better numerical conditioning than Lagrange

##### **Spectral Elements**
- **High-order accuracy**: Exponential convergence for smooth solutions
- **Optimal point distributions**: Gauss-Lobatto-Legendre and variants
- **Numerical integration**: Exact integration of polynomial integrands

##### **Rational Basis Functions**
- **NURBS**: Non-uniform rational B-splines for CAD integration
- **Isogeometric analysis**: Exact geometry representation
- **Bézier functions**: Computer graphics integration
- **T-splines**: Local refinement capabilities

##### **Vector-Valued Shape Functions**
- **H(div) conforming**: Raviart-Thomas, BDM elements for fluid mechanics
- **H(curl) conforming**: Nédélec elements for electromagnetics
- **Mixed formulations**: Support for incompressible flow, electromagnetics

#### **Enrichment and Adaptive Technologies**

##### **XFEM/GFEM Enrichment**
- **Discontinuity modeling**: Cracks, voids, material interfaces
- **Singular enrichment**: Crack tip fields, corner singularities
- **Partition of unity**: General enrichment framework

##### **Adaptive Refinement (via adaptation/)**
- Shape functions expose the capabilities needed for p-/hp-adaptivity (e.g., hierarchical bases, modal enrichment), but adaptation orchestration, error estimation, and mesh refinement are owned by the high-level `adaptation/` module.
- Refer to `src/adaptation/AGENT.md` for refinement strategies (h/p/r/hp), anisotropy, and error estimators. The `fem/shape/` plan avoids duplicating those responsibilities.

### **Physics-Agnostic Design Principles**

#### **Node/DOF Separation of Concerns**
- **Geometric information**: Node coordinates, connectivity (physics-independent)
- **DOF management**: Field-agnostic DOF numbering and constraint handling
- **Physics composition**: Physics modules define field types and coupling

#### **Shape Function Modularity**
- **Mathematical properties**: Polynomial completeness, continuity, orthogonality
- **Computational optimization**: Efficient evaluation, caching, vectorization
- **Physics independence**: Shape functions work with any field variable type

#### **Interface Design**
```cpp
// Physics-agnostic node interface
class Node {
    // Geometric properties
    Point coordinates() const;
    std::vector<ElementId> adjacent_elements() const;

    // DOF management (field-type independent)
    void add_dof(FieldId field, ComponentId component);
    DOFId get_dof_id(FieldId field, ComponentId component) const;

    // Constraint handling
    void add_constraint(std::unique_ptr<Constraint> constraint);
};

// Physics-agnostic shape function interface
template<int Dim, int Order>
class ShapeFunction {
    // Evaluation at parametric coordinates
    Vector shape_values(const Point<Dim>& xi) const;
    Matrix shape_derivatives(const Point<Dim>& xi) const;

    // Properties
    int polynomial_order() const;
    ContinuityType continuity() const;
    bool has_bubble_functions() const;
};
```

#### **Composition Examples**
```cpp
// Structural mechanics: displacement field
auto node = NodeFactory::create(coordinates);
node->add_dof(FieldRegistry::get("displacement"), ComponentId::X);
node->add_dof(FieldRegistry::get("displacement"), ComponentId::Y);
node->add_dof(FieldRegistry::get("displacement"), ComponentId::Z);

// Fluid mechanics: velocity + pressure
auto node = NodeFactory::create(coordinates);
node->add_dof(FieldRegistry::get("velocity"), ComponentId::X);
node->add_dof(FieldRegistry::get("velocity"), ComponentId::Y);
node->add_dof(FieldRegistry::get("velocity"), ComponentId::Z);
node->add_dof(FieldRegistry::get("pressure"), ComponentId::SCALAR);

// Electromagnetics: electric and magnetic fields
auto node = NodeFactory::create(coordinates);
node->add_dof(FieldRegistry::get("electric_field"), ComponentId::X);
node->add_dof(FieldRegistry::get("magnetic_field"), ComponentId::Y);
```

This architecture ensures that node and shape function infrastructure can be optimized for mathematical and computational properties while remaining completely independent of physics domains.

## Utilities Reorganization

The fem/utilities/ folder has been eliminated to prevent it from becoming a dumping ground for miscellaneous code. Instead, utility files have been distributed to their appropriate conceptual homes:

### **Redistributed Files**

#### **Element Infrastructure** (moved to element/)
- **element_orientation.hpp**: Edge/face orientation handling - logically belongs with element topology
- **fem_constants.hpp**: FEM-specific mathematical constants - needed primarily for element computations

#### **Core FEM Infrastructure** (moved to core/)
- **fem_traits.hpp**: Type traits for FEM - fundamental infrastructure used across all FEM components
- **fem_concepts.hpp**: C++20 concepts - core programming abstractions for type safety
- **reference_solutions.hpp**: Reference analytical solutions - testing and validation infrastructure
- **performance_monitoring.hpp**: Performance timing and profiling - development and optimization tools

### **Benefits of Reorganization**

1. **Conceptual Clarity**: Each file is placed with related functionality rather than in a catch-all folder
2. **Discoverability**: Developers find orientation handling in element/, not hidden in utilities/
3. **Maintainability**: Clear ownership and responsibility for each component
4. **Prevents Code Smell**: Eliminates the "junk drawer" anti-pattern that utilities/ folders often become
5. **Logical Grouping**: Files are grouped by what they do, not by being "miscellaneous"

### **Guidelines for Future Development**

- **No utilities/ folder**: Never create utilities/, misc/, or common/ folders
- **Context-specific placement**: Always place code with its primary use case
- **Shared infrastructure**: Truly shared code goes in core/ with clear responsibility
- **Domain-specific tools**: Tools specific to elements, shapes, integration, etc. go in their respective domain folders

This organization ensures that the fem/ library maintains clean architectural boundaries and prevents the accumulation of poorly organized code.

## FEM Subfolder Responsibilities and Relationships

The fem/ library is organized into specialized subfolders, each with distinct responsibilities that work together to provide comprehensive finite element capabilities. Understanding these relationships is crucial for effective usage and extension.

### **Core Mathematical Infrastructure**

#### **basis/ - Pure Mathematical Foundations**
**Responsibility**: Provides abstract mathematical basis functions independent of finite element context
- **Polynomial families**: Monomial, Bernstein, Lagrange, Hermite bases
- **Orthogonal systems**: Legendre, Chebyshev, Jacobi polynomials
- **Spectral bases**: Fourier series, spherical harmonics
- **Specialized bases**: Wavelets, radial basis functions, splines
- **Mathematical properties**: Orthogonality, completeness, conditioning

**Physics Independence**: Completely physics-agnostic mathematical abstractions

#### **shape/ - Finite Element Basis Functions**
**Responsibility**: Adapts mathematical bases for finite element interpolation on specific element topologies
- **Element-specific adaptation**: Takes basis/ functions and adapts them to triangles, hexahedra, etc.
- **Finite element properties**: C⁰, C¹ continuity, nodal/modal forms
- **Computational optimization**: Efficient evaluation, derivative computation, caching
- **Advanced FEM techniques**: XFEM enrichment, hierarchical p-refinement

**Relationship to basis/**: Uses and extends basis/ mathematical foundations for FEM-specific requirements

### **Geometric and Topological Infrastructure**

#### **element/ - Element Topology and Geometry**
**Responsibility**: Defines element shapes, connectivity, and geometric properties
- **Topological definitions**: Node connectivity patterns for all element types
- **Reference elements**: Standard element definitions in parametric coordinates
- **Element families**: Complete taxonomy of 1D/2D/3D element types
- **Geometric mappings**: Coordinate transformations from reference to physical elements

**Relationship to shape/**: Provides the geometric context (element topology) that shape functions are defined on

#### **node/ - Node and Connectivity Management**
**Responsibility**: Manages discrete points, connectivity, and degree of freedom assignment
- **Node entities**: Physical points with coordinates and connectivity information
- **DOF management**: Assignment of field variables to nodes (physics-agnostic)
- **Constraints**: Boundary conditions, multi-point constraints, rigid connections
- **Mesh topology**: Adjacency relationships, boundary detection

**Relationship to element/**: Provides the discrete points that elements connect; manages element-node relationships

### **Computational Infrastructure**

#### **integration/ - Numerical Integration**
**Responsibility**: Provides quadrature rules and integration strategies for finite element computations
- **Quadrature families**: Gauss, Newton-Cotes, spectral quadrature rules
- **Element-specific rules**: Optimized integration for each element topology
- **Advanced techniques**: Adaptive quadrature, cut-cell integration, sparse grids
- **Performance optimization**: Cached evaluations, ML-optimized rules

**Relationship to element/**: Provides integration rules tailored to each element type
**Relationship to shape/**: Integration points where shape functions are evaluated

#### **field/ - Field Variable Management**
**Responsibility**: Manages field variables (solution unknowns) independently of their physics meaning
- **Field types**: Scalar, vector, tensor, complex field abstractions
- **Field operations**: Interpolation, gradients, divergence, curl operations
- **Storage strategies**: Distributed, compressed, cached field storage
- **Transfer operations**: Field mapping between different meshes

**Physics Independence**: Field operations work regardless of whether fields represent displacement, temperature, pressure, etc.
**Relationship to node/**: Fields are associated with DOF at nodes
**Relationship to shape/**: Field interpolation uses shape functions

### **Mathematical Formulation Infrastructure**

#### **formulation/ - Numerical Method Implementation**
**Responsibility**: Implements weak-form evaluation and residual/Jacobian assembly for finite element methods; delegates nonlinear/time stepping and solver orchestration to the `solvers/` module.
- **Method families**: Galerkin, Petrov-Galerkin, discontinuous Galerkin
- **Advanced methods**: Mixed formulations, stabilized methods, least-squares
- **Implementation strategies**: Element matrix/vector computation, stabilization terms
- **Computational patterns**: Assembly algorithms, constraint handling (but not Newton/time stepping control)

**Relationship to shape/**: Uses shape functions to implement specific numerical methods
**Relationship to integration/**: Uses quadrature rules for numerical integration

#### **variational/ - High-Level Mathematical Expression**
**Responsibility**: Provides the domain-specific language for expressing weak forms, builds an intermediate representation, runs symbolic/analytical passes, and lowers to execution backends while delegating solver policy to other modules.
- **Symbolic representation**: Core DSL with differential, facet, time, and ALE operators so users write forms as in textbook notation.
- **Reusable form components**: Library of stabilization, interface, and block-form utilities that return `Form` objects composable with user-defined residuals.
- **IR & analysis**: Normalized form IR plus passes for shape inference, CSE, constant folding, sparsity inference, and algebraic rewrites.
- **Lowering**: Configurable lowering targets (assembled sparse, matrix-free/sum-factorized, libCEED) producing artifacts consumed by assembly/solvers.
- **Space/integration adapters**: Light wrappers over `fem/spaces/` and `fem/integration/` so the DSL reuses existing function space and quadrature infrastructure instead of duplicating it.
- **Physics interface**: Physics modules depend on the DSL to declare problems, then call the lowering/assembly pipeline.

**Relationship to formulation/**: Compiled forms emit kernels/operators that formulation/ (and downstream assembly) execute.
**Relationship to integration/**: Uses adapters to select quadrature rules from `fem/integration/` appropriate to the form.
**Relationship to spaces/**: Uses adapters to reference spaces from `fem/spaces/`; all space construction/refinement logic lives there.

### **Workflow and Data Flow**

#### **Element Assembly Workflow**
1. **element/**: Define element topology and reference geometry
2. **node/**: Assign DOF and manage constraints
3. **shape/**: Evaluate shape functions at integration points
4. **integration/**: Provide quadrature points and weights
5. **field/**: Interpolate field values and compute gradients
6. **formulation/**: Apply specific FEM method (Galerkin, mixed, etc.)
7. **Result**: Element matrix and vector contributions

#### **Physics Module Integration**
1. **variational/**: Physics expresses problem as mathematical weak form
2. **field/**: Register physics field types (displacement, pressure, etc.)
3. **Compilation**: variational/ builds form IR, runs analysis (CSE, sparsity), and lowers to the selected backend (assembled, matrix-free, libCEED)
4. **Runtime**: Lowered operators are executed via formulation/ and the assembly/solver stack for efficient computation

### **Separation of Concerns Examples**

#### **Structural Mechanics Example**
```cpp
// Physics module defines the problem
auto weak_form = inner(sigma(grad(u)), grad(v)) * dx;

// fem/ infrastructure handles the mathematics:
// - element/: Provides hex8 element topology
// - node/: Manages displacement DOF at nodes
// - shape/: Evaluates Lagrange shape functions
// - integration/: Uses 2x2x2 Gauss quadrature
// - field/: Computes displacement gradients
// - formulation/: Applies Galerkin method
```

#### **Fluid Mechanics Example**
```cpp
// Same fem/ infrastructure, different physics
auto weak_form = inner(grad(u), grad(v)) * dx + inner(grad(p), v) * dx;

// fem/ handles:
// - element/: Same hex8 topology
// - node/: Velocity + pressure DOF
// - shape/: Same shape functions, different fields
// - integration/: Same quadrature rules
// - formulation/: Mixed formulation for incompressibility
```

### **Key Design Principles**

1. **Layered Architecture**: Lower layers (basis/, element/) provide foundations for higher layers (formulation/, variational/)

2. **Separation of Mathematical and Physical Concerns**: All fem/ components work with any physics

3. **Composability**: Physics modules compose fem/ building blocks rather than inherit from them

4. **Optimization Independence**: Each layer can be optimized independently while maintaining interfaces

5. **Extensibility**: New element types, shape functions, or methods can be added without affecting other components

This architecture enables physics modules to focus on their domain expertise while leveraging highly optimized, reusable mathematical infrastructure.

## Key Components

### 1. Element Architecture
```cpp
// Component-based element using ECS
class Element : public core::Entity {
    // Core components
    ElementTopology topology;        // Connectivity
    ElementGeometry geometry;        // Coordinates
    ShapeFunction shapes;           // Basis functions
    IntegrationRule quadrature;     // Quadrature
    
    // Optional physics components (added by physics modules)
    // MechanicsComponent, ThermalComponent, etc.
    
    // Performance-critical kernel
    template<typename Physics>
    void compute_local_matrix(Matrix& K_e) {
        // Static polymorphism for performance
        Physics::compute(this, K_e);
    }
};

// Static polymorphism for compile-time optimization
template<int Dim, int Nodes>
class IsoparametricElement {
    static constexpr int n_dofs = Dim * Nodes;
    using LocalMatrix = SmallMatrix<double, n_dofs, n_dofs>;
    
    // Compile-time optimized operations
    void compute_jacobian() { /* unrolled loops */ }
};
```

### 2. Shape Function System
```cpp
// Cached shape function evaluation
class ShapeFunctionCache {
    // Pre-compute at quadrature points
    void initialize(const Element& elem, const QuadratureRule& quad) {
        for (auto& qp : quad.points()) {
            N[qp.id] = elem.shape_values(qp.xi);
            dN[qp.id] = elem.shape_derivatives(qp.xi);
        }
    }
    
    // Fast access during assembly
    const auto& values(int qp) const { return N[qp]; }
    const auto& derivatives(int qp) const { return dN[qp]; }
};

// Hierarchical shape functions for p-refinement
template<int Order>
class HierarchicalBasis {
    // Legendre polynomials up to Order
    static auto evaluate(double xi) {
        return legendre_polynomial<Order>(xi);
    }
};
```

### 3. DOF Management
```cpp
// Efficient DOF numbering and constraints
class DOFManager {
    // Local to global mapping
    std::vector<int> local_to_global;
    
    // Constraint handling
    void apply_dirichlet(int node, int dof, double value) {
        constrained_dofs.insert({node, dof}, value);
    }
    
    // Optimized numbering for bandwidth
    void optimize_numbering() {
        auto perm = compute_cuthill_mckee(connectivity);
        renumber(perm);
    }
};

// Component for field DOFs
class FieldDOFs : public core::Component {
    std::vector<int> indices;      // Global DOF indices
    std::vector<double> values;    // Current values
    std::vector<double> velocities; // For dynamics
};
```

### 4. Material Interface
```cpp
// Material as component for elements
class MaterialComponent : public core::Component {
    // Pure virtual for physics to implement
    virtual void compute_stress(const Strain& E, Stress& S) = 0;
    virtual void compute_tangent(const Strain& E, Tangent& C) = 0;
    
    // State management
    MaterialState state;
    MaterialHistory history;
};

// Material point for integration points
struct MaterialPoint {
    Vector stress;
    Matrix tangent;
    double energy;
    std::any internal_variables;  // Physics-specific
};
```

### 5. Quadrature Integration
```cpp
// Optimized quadrature with caching
template<typename Element>
class QuadratureData {
    // Pre-computed values
    std::vector<double> weights;
    std::vector<Point> points;
    std::vector<Matrix> shape_values;
    std::vector<Tensor> shape_gradients;
    std::vector<double> jacobian_dets;
    
    void precompute(const Element& elem) {
        for (int q = 0; q < n_points; ++q) {
            shape_values[q] = elem.N(points[q]);
            shape_gradients[q] = elem.dN(points[q]);
            jacobian_dets[q] = elem.J_det(points[q]);
        }
    }
};
```
### 6. Local Assembly
```cpp
// fem/assembly/element_assembler.hpp
template<typename Physics>
class ElementAssembler {
// Defines the mathematical formulation
Matrix compute_stiffness(const Element& elem) {
Matrix K_e(elem.n_dofs(), elem.n_dofs());

        for (auto& qp : elem.quadrature_points()) {
            auto B = elem.strain_displacement_matrix(qp);
            auto D = physics.material_stiffness(qp);
            auto w = qp.weight * elem.jacobian(qp);
            
            K_e += B.T() * D * B * w;  // The physics equation
        }
        return K_e;
    }
};
```
### 6. Weak Form Abstraction
```cpp
// Define weak form for physics modules
template<typename TestSpace, typename TrialSpace>
class WeakForm {
    // Bilinear form a(u,v)
    virtual double bilinear(const TestFunc& v,
                           const TrialFunc& u) = 0;

    // Linear form f(v)
    virtual double linear(const TestFunc& v) = 0;

    // For nonlinear problems
    virtual void residual(const Solution& u, Vector& R) = 0;
    virtual void jacobian(const Solution& u, Matrix& J) = 0;
};
```

### 7. Variational Form Language (UFL-Inspired)
```cpp
// Symbolic variational form definition
template<typename FunctionSpace>
class VariationalForm {
    // Define test and trial functions symbolically
    TestFunction<FunctionSpace> v;
    TrialFunction<FunctionSpace> u;

    // Express weak form in mathematical notation
    auto weak_form = inner(grad(u), grad(v)) * dx
                   + f * v * dx;

    // Automatic code generation for assembly
    auto assembler = compile_form(weak_form);

    // Generated optimized assembly kernel
    Matrix assemble() {
        return assembler.compute_matrix();
    }
};

// Example: Poisson equation in natural mathematical notation
auto poisson_form() {
    auto V = FunctionSpace<P1>(mesh);
    auto u = TrialFunction<V>();
    auto v = TestFunction<V>();
    auto f = Coefficient<V>("source");

    return inner(grad(u), grad(v)) * dx == f * v * dx;
}
```

### 8. Form Compilation and Optimization
```cpp
// Compile-time form analysis and code generation
template<typename Form>
class FormCompiler {
    // Analyze form structure
    void analyze_form(const Form& form) {
        tensor_rank = determine_rank(form);
        integration_domains = extract_domains(form);
        required_derivatives = analyze_derivatives(form);
    }

    // Generate optimized assembly code
    auto generate_kernel() -> AssemblyKernel {
        return AssemblyKernel{
            .element_tensor_computation = generate_element_tensor(),
            .quadrature_loop = generate_quadrature_code(),
            .basis_evaluation = generate_basis_code()
        };
    }

    // Automatic differentiation for Jacobians
    auto compute_jacobian(const Form& residual) {
        return automatic_differentiate(residual);
    }
};
```

### 9. Error Estimation
```cpp
// Adaptive refinement based on error
class ErrorEstimator : public core::Component {
    // Element-wise error indicators
    std::vector<double> compute_indicators(const Solution& u) {
        std::vector<double> eta(mesh.n_elements());
        
        for (auto& elem : mesh.elements()) {
            eta[elem.id()] = estimate_element_error(elem, u);
        }
        return eta;
    }
    
    // Mark elements for refinement
    void mark_elements(const std::vector<double>& eta) {
        double threshold = compute_threshold(eta);
        for (size_t e = 0; e < eta.size(); ++e) {
            if (eta[e] > threshold) {
                mesh.mark_for_refinement(e);
            }
        }
    }
};
```

## Performance Considerations

### Memory Layout
- **AoS for Elements**: Each element contains its data
- **SoA for Nodes**: Separate arrays for coordinates, DOFs
- **Cache Blocking**: Group elements for cache efficiency
- **Memory Pools**: Reuse element workspace

### Compile-Time Optimization
```cpp
// Template recursion for unrolling
template<int I, int N>
struct UnrolledLoop {
    template<typename F>
    static void apply(F&& f) {
        f(I);
        UnrolledLoop<I+1, N>::apply(f);
    }
};

// Specialization to end recursion
template<int N>
struct UnrolledLoop<N, N> {
    template<typename F>
    static void apply(F&&) {}
};
```

### Vectorization
```cpp
// SIMD-friendly operations
alignas(32) double N[8];      // Aligned shape functions
alignas(32) double dN[8][3];  // Aligned derivatives

// Vectorized evaluation
void evaluate_shape_simd(const double* xi, double* N) {
    #pragma omp simd
    for (int i = 0; i < n_nodes; ++i) {
        N[i] = compute_shape(i, xi);
    }
}
```

## Variational vs. Formulation: Key Differences

The fem/ module includes both a `formulation/` folder and a new `variational/` folder. While they may seem similar, they serve fundamentally different purposes:

### formulation/ - Implementation-Level Abstractions
The `formulation/` folder provides **concrete implementation strategies** for finite element methods:
- **Galerkin methods**: Standard, Petrov-Galerkin variants
- **Specialized techniques**: Discontinuous Galerkin, stabilized methods, least-squares
- **Implementation patterns**: How to actually implement these mathematical concepts in code
- **Low-level interfaces**: Direct interaction with element matrices and assembly

```cpp
// formulation/ - Implementation-focused
class GalerkinFormulation {
    void assemble_element_matrix(const Element& elem, Matrix& K_e) {
        // Direct implementation of Galerkin method
        for (auto& qp : elem.quadrature_points()) {
            auto B = elem.strain_displacement_matrix(qp);
            auto D = material.stiffness_matrix(qp);
            K_e += B.T() * D * B * qp.weight;
        }
    }
};
```

### variational/ - User-Level Mathematical Expressions
The `variational/` folder provides a **domain-specific language** for expressing mathematical problems:
- **Mathematical notation**: Write weak forms as they appear in textbooks
- **Symbolic computation**: Automatic differentiation, form manipulation
- **High-level abstraction**: Users think in terms of mathematical operators
- **Code generation**: Automatically generates optimized assembly code

```cpp
// variational/ - Mathematics-focused
auto linear_elasticity() {
    auto V = VectorFunctionSpace<P1>(mesh, 3);  // 3D vector space
    auto u = TrialFunction<V>();
    auto v = TestFunction<V>();
    auto f = Coefficient<V>("body_force");

    // Express exactly as in mathematical formulation
    auto a = inner(sigma(u), epsilon(v)) * dx;
    auto L = inner(f, v) * dx;

    return VariationalProblem(a == L);
}
```

### Relationship and Workflow

1. **User Experience**: Physics modules and applications use `variational/` to express their mathematical problems in natural notation

2. **Code Generation**: The variational form compiler analyzes the symbolic expressions and generates efficient assembly code

3. **Implementation**: The generated code uses components from `formulation/` for the actual numerical implementation

4. **Optimization**: The compiler can choose optimal formulation strategies based on the mathematical structure

```cpp
// Workflow example:
auto problem = poisson_equation();          // variational/ - define mathematically
auto compiled = compile_form(problem);      // variational/ - analyze and optimize
auto assembler = compiled.get_assembler();  // Uses formulation/ - implementation
auto matrix = assembler.assemble();         // Efficient numerical computation
```

### Benefits of Separation

- **Mathematical Clarity**: Users work with familiar mathematical notation
- **Implementation Flexibility**: Multiple implementation strategies can target the same mathematical expression
- **Automatic Optimization**: Compiler can choose best implementation based on problem structure
- **Maintainability**: Mathematical expressions remain stable while implementations can evolve
- **Performance**: Generated code can be highly optimized for specific problem types

## Integration Points

### With numeric/
- Uses `numeric::Matrix` for element matrices
- Uses `numeric::QuadratureRule` for integration
- Uses `numeric::PolynomialBasis` for shape functions
- Uses `numeric::SparseBuilder` for assembly

### With core/
- Elements inherit from `core::Entity`
- Materials are `core::Component`
- Shape functions use `core::Factory`
- DOF manager uses `core::Registry`

### With physics/
- Physics modules add components to elements
- Physics implement `WeakForm` interface
- Physics define material models
- Physics specify field variables

### With variational/
- Physics modules express problems using variational form language
- Automatic code generation from mathematical expressions
- Integration with `formulation/` for optimized implementation
- Symbolic manipulation of weak forms

## Design Patterns

### Factory Pattern for Elements
```cpp
auto elem = ElementFactory::create("hex20");
elem->set_nodes(node_coords);
elem->set_material(material);
```

### Visitor Pattern for Assembly
```cpp
class AssemblyVisitor : public ElementVisitor {
    void visit(const Element& elem) override {
        auto K_e = elem.compute_matrix();
        assembler.add(elem.dofs(), K_e);
    }
};
```

### Strategy Pattern for Quadrature
```cpp
element.set_quadrature_strategy(
    make_gauss_quadrature(order=2)
);
```

### Expression Template Pattern for Variational Forms
```cpp
// Build complex expressions using operator overloading
auto weak_form = inner(grad(u), grad(v)) * dx +
                 alpha * inner(u, v) * dx;

// Automatic optimization and code generation
auto optimized = compile_form(weak_form);
```

### Visitor Pattern for Form Analysis
```cpp
class FormAnalyzer : public FormVisitor {
    void visit(const BilinearForm& form) override {
        analyze_symmetry(form);
        analyze_sparsity(form);
    }

    void visit(const NonlinearForm& form) override {
        compute_jacobian_sparsity(form);
    }
};
```

## Success Metrics

### **Performance Metrics**
1. **Element Evaluation**: < 1μs for 8-node hex element matrix computation
2. **Shape Function Cache**: 10x speedup vs recomputation
3. **DOF Numbering**: < O(n log n) complexity for bandwidth optimization
4. **Memory Usage**: < 1KB per element overhead
5. **Assembly Interface**: Zero-copy where possible

### **Architecture Metrics**
6. **Extensibility**: New element type in < 100 lines of code
7. **Element Coverage**: 150+ mathematical element types as physics-agnostic building blocks
8. **Topology Support**: Complete 1D/2D/3D element families with p-refinement
9. **Advanced Methods**: XFEM, meshfree, multiscale, enriched formulations

### **Function Space Metrics**
10. **Space Coverage**: 50+ function space types covering all major FEM applications
11. **Sobolev Spaces**: Complete H¹, H², H(curl), H(div) space families
12. **Mixed Spaces**: Taylor-Hood, MINI, Crouzeix-Raviart, and general mixed constructions
13. **Adaptive Spaces**: hp-adaptive, hierarchical, and anisotropic refinement support

### **Formulation Metrics**
14. **Method Coverage**: 60+ formulation types covering all major FEM methods
15. **Galerkin Family**: Standard, Petrov, stabilized, and least-squares variants
16. **DG Methods**: Interior penalty, HDG, flux reconstruction, and embedded DG
17. **Advanced Formulations**: Multiscale, meshfree, collocation, and time-dependent methods

### **Variational Form Metrics**
18. **Form Compilation**: < 10ms for complex variational forms
19. **Generated Code**: Performance within 5% of hand-optimized assembly
20. **Mathematical Notation**: 1:1 correspondence with textbook weak forms
21. **Form Analysis**: Automatic sparsity pattern detection and optimization

### **Code Quality Metrics**
22. **No Utilities Dumping**: Zero miscellaneous or utilities folders
23. **Clear Ownership**: Every file belongs to a specific conceptual domain
24. **Architectural Boundaries**: Clean separation between mathematical layers

## Key Innovations

1. **ECS for FEM**: Elements/nodes as entities with components
2. **Compile-Time Elements**: Template metaprogramming for performance
3. **Cached Operations**: Pre-compute everything possible
4. **Physics Agnostic**: Clean separation from physics
5. **Comprehensive Element Taxonomy**: 150+ element types systematically organized
6. **Unified Element Interface**: Common API across all element families
7. **Topology-Physics Separation**: Element topology independent of physics
8. **Variational Form Language**: UFL-inspired DSL for mathematical expression
9. **Automatic Code Generation**: From symbolic forms to optimized assembly
10. **Dual-Layer Architecture**: High-level mathematical + low-level implementation
11. **Modern C++**: Concepts, ranges, modules where applicable

This architecture provides a solid, performant foundation for finite element computations while maintaining flexibility through the component system and enabling high performance through compile-time optimization and caching strategies. The physics-agnostic element taxonomy covering 100+ mathematical element types serves as reusable building blocks that physics modules compose to create domain-specific formulations. This separation of concerns ensures that mathematical infrastructure can be optimized independently while physics modules focus on their domain expertise. Combined with the variational form language, this creates a complete solution where users can express their mathematical problems in natural notation, compose appropriate element types for their mathematical requirements, and achieve performance comparable to hand-written specialized code.

# AGENT.md - FEM Foundation Module

## Mission
Provide core finite element abstractions and base classes that define the fundamental FEM concepts, independent of specific physics or analysis types, while leveraging the ECS architecture from core/ and mathematical operations from numeric/.

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
│   └── types/                      # Concrete element types
│       ├── line/
│       │   ├── line2.hpp           # 2-node line
│       │   └── line3.hpp           # 3-node line
│       ├── triangle/
│       │   ├── tri3.hpp            # 3-node triangle
│       │   ├── tri6.hpp            # 6-node triangle
│       │   └── tri10.hpp           # 10-node triangle
│       ├── quadrilateral/
│       │   ├── quad4.hpp           # 4-node quad
│       │   ├── quad8.hpp           # 8-node quad
│       │   └── quad9.hpp           # 9-node quad
│       ├── tetrahedron/
│       │   ├── tet4.hpp            # 4-node tetrahedron
│       │   └── tet10.hpp           # 10-node tetrahedron
│       ├── hexahedron/
│       │   ├── hex8.hpp            # 8-node hexahedron
│       │   ├── hex20.hpp           # 20-node hexahedron
│       │   └── hex27.hpp           # 27-node hexahedron
│       ├── prism/
│       │   ├── prism6.hpp          # 6-node prism
│       │   └── prism15.hpp         # 15-node prism
│       └── special/
│           ├── infinite.hpp        # Infinite elements
│           ├── interface.hpp       # Interface elements
│           └── cohesive.hpp        # Cohesive elements
│
├── node/                           # Node and DOF management
│   ├── node_base.hpp               # Base node class
│   ├── node_entity.hpp             # ECS-based node
│   ├── dof.hpp                     # Degree of freedom
│   ├── dof_manager.hpp             # DOF numbering and management
│   ├── dof_map.hpp                 # Local-global DOF mapping
│   ├── dof_constraints.hpp         # DOF constraint definitions
│   ├── dof_ordering.hpp            # DOF ordering/reordering strategies
│   ├── nodal_coordinates.hpp       # Coordinate systems
│   └── node_set.hpp                # Node set management
│
├── shape/                           # Shape functions
│   ├── shape_function_base.hpp     # Base interface
│   ├── shape_function_cache.hpp    # Caching evaluated shapes
│   ├── lagrange/                   # Lagrangian shape functions
│   │   ├── lagrange_1d.hpp
│   │   ├── lagrange_2d.hpp
│   │   └── lagrange_3d.hpp
│   ├── hermite/                    # Hermite shape functions
│   │   └── hermite.hpp
│   ├── hierarchical/               # Hierarchical bases
│   │   ├── legendre.hpp
│   │   └── lobatto.hpp
│   ├── serendipity/                # Serendipity elements
│   │   └── serendipity.hpp
│   ├── nurbs/                      # NURBS shape functions
│   │   ├── nurbs.hpp
│   │   ├── nurbs_knot_vector.hpp
│   │   └── nurbs_patch.hpp
│   └── enriched/                   # XFEM/GFEM enrichment
│       ├── heaviside.hpp
│       └── crack_tip.hpp
│
├── integration/                     # Numerical integration
│   ├── quadrature_rule.hpp         # Base quadrature rule
│   ├── gauss/                      # Gaussian quadrature
│   │   ├── gauss_1d.hpp
│   │   ├── gauss_triangle.hpp
│   │   ├── gauss_quadrilateral.hpp
│   │   ├── gauss_tetrahedron.hpp   
│   │   ├── gauss_tensor.hpp        # Tensor product quadrature
│   │   └── gauss_hexahedron.hpp
│   ├── newton_cotes/               # Newton-Cotes rules
│   │   └── newton_cotes.hpp
│   ├── adaptive/                   # Adaptive quadrature
│   │   └── adaptive_quadrature.hpp
│   ├── special/                    # Special integration
│   │   ├── singular.hpp           # Singular integrals
│   │   └── surface.hpp            # Surface integration
│   ├── cut_cell.hpp               # Embedded interface quadrature   
│   └── quadrature_cache.hpp        # Cache quadrature data
│
├── material/                        # Material model interfaces
│   ├── material_base.hpp           # Base material interface
│   ├── material_component.hpp      # ECS material component
│   ├── material_point.hpp          # Material point data
│   ├── constitutive_base.hpp       # Constitutive model interface
│   ├── material_tangent.hpp        # Tangent operators
│   ├── material_state.hpp          # State variables
│   ├── material_properties.hpp     # Property container
│   ├── integration_schemes.hpp     # Constitutive update schemes
│   └── material_factory.hpp        # Material creation factory
│
├── field/                           # Field variables
│   ├── field_base.hpp              # Base field class
│   ├── scalar_field.hpp            # Scalar fields
│   ├── vector_field.hpp            # Vector fields
│   ├── tensor_field.hpp            # Tensor fields
│   ├── field_interpolation.hpp     # Field interpolation
│   ├── field_gradient.hpp          # Gradient computation
│   ├── field_component.hpp         # ECS field component
│   └── field_history.hpp           # Time history storage
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
├── basis/                           # Basis functions
│   ├── basis_base.hpp              # Base basis class
│   ├── polynomial_basis.hpp        # Polynomial bases
│   ├── spectral_basis.hpp          # Spectral bases
│   ├── wavelet_basis.hpp           # Wavelet bases
│   ├── modal_basis.hpp             # Modal bases
│   └── nodal_basis.hpp             # Nodal bases
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
│   ├── weak_form.hpp               # Weak form interface
│   ├── galerkin.hpp                # Standard Galerkin
│   ├── petrov_galerkin.hpp         # Petrov-Galerkin
│   ├── least_squares.hpp           # Least-squares FEM
│   ├── mixed.hpp                   # Mixed formulations
│   ├── discontinuous_galerkin.hpp  # DG methods
│   ├── stabilized.hpp              # Stabilized methods
│   └── variational.hpp             # Variational principles
│
├── spaces/                          # Function spaces
│   ├── function_space.hpp          # Function space base
│   ├── h1_space.hpp                # H1 conforming
│   ├── h_curl_space.hpp            # H(curl) conforming
│   ├── h_div_space.hpp             # H(div) conforming
│   ├── l2_space.hpp                # L2 space
│   ├── composite_space.hpp         # Product spaces
│   └── enriched_space.hpp          # Enriched spaces
│
├── error/                           # Error estimation
│   ├── error_estimator_base.hpp    # Base estimator
│   ├── residual_estimator.hpp      # Residual-based
│   ├── recovery_estimator.hpp      # Recovery-based
│   ├── goal_oriented.hpp           # Goal-oriented
│   ├── error_indicator.hpp         # Error indicators
│   └── effectivity_index.hpp       # Effectivity computation
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
├── utilities/                       # FEM utilities
│   ├── fem_constants.hpp           # FEM-specific constants
│   ├── fem_traits.hpp              # FEM type traits
│   ├── fem_concepts.hpp            # C++20 concepts
│   ├── reference_values.hpp        # Reference solutions
│   ├── orientation.hpp             # Edge/Face orientation handling
│   └── fem_timers.hpp              # Performance timing
│
├── tests/                           # Testing
│   ├── unit/                        # Unit tests
│   ├── convergence/                 # Convergence tests
│   └── patch/                       # Patch tests
└── benchmarks/                      # Performance benchmarks
```

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

### 7. Error Estimation
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

## Success Metrics

1. **Element Evaluation**: < 1μs for 8-node hex
2. **Shape Function Cache**: 10x speedup vs recomputation
3. **DOF Numbering**: < O(n log n) complexity
4. **Memory Usage**: < 1KB per element overhead
5. **Assembly Interface**: Zero-copy where possible
6. **Extensibility**: New element in < 100 lines

## Key Innovations

1. **ECS for FEM**: Elements/nodes as entities with components
2. **Compile-Time Elements**: Template metaprogramming for performance
3. **Cached Operations**: Pre-compute everything possible
4. **Physics Agnostic**: Clean separation from physics
5. **Modern C++**: Concepts, ranges, modules where applicable

This architecture provides a solid, performant foundation for finite element computations while maintaining flexibility through the component system and enabling high performance through compile-time optimization and caching strategies.
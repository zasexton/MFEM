# Material Library Infrastructure - AGENT.md

## 🎯 Mission
Build a high-performance, extensible material model infrastructure for multiphysics FEM that supports nonlinear constitutive behavior, history-dependent materials, multi-field coupling, and seamless CPU/GPU execution while maintaining physical consistency and numerical stability.

## 🏗️ Architecture Philosophy
- **Performance-Critical Design**: Static polymorphism for inner loops, dynamic for configuration
- **Physics Agnostic Core**: Base material interfaces independent of specific physics
- **Composition Pattern**: Build complex materials from simple behaviors
- **Zero-Cost Abstractions**: Templates for evaluation, virtuals only for setup
- **State Management**: Efficient history variable tracking with minimal memory
- **GPU-First**: Unified material evaluation on CPU/GPU
- **Verification Built-In**: Automatic tangent checking and consistency validation

## 📁 Module Structure

```
materials/
├── base/                    # Core material interfaces
│   ├── material_base.h      # Base material class
│   ├── material_point.h     # Material state at integration point
│   ├── material_data.h      # Material parameters storage
│   ├── material_traits.h    # Compile-time material properties
│   └── material_factory.h   # Dynamic material creation
│
├── state/                   # State variable management
│   ├── state_variables.h    # History variable container
│   ├── state_manager.h      # State update/storage orchestration
│   ├── state_view.h         # Lightweight state access
│   └── checkpointing.h      # State serialization
│
├── properties/              # Material property definitions
│   ├── property_base.h      # Property interface
│   ├── elastic_properties.h # Elastic constants
│   ├── thermal_properties.h # Thermal parameters
│   ├── plastic_properties.h # Plasticity parameters
│   ├── damage_properties.h  # Damage/fracture parameters
│   ├── viscous_properties.h # Rate-dependent parameters
│   └── field_dependent.h    # Temperature/field-dependent props
│
├── models/                  # Concrete material models
│   ├── linear/              # Linear models
│   │   ├── isotropic_elastic.h
│   │   ├── orthotropic_elastic.h
│   │   ├── anisotropic_elastic.h
│   │   └── thermoelastic.h
│   │
│   ├── nonlinear/           # Nonlinear models
│   │   ├── hyperelastic/    # Rubber-like materials
│   │   │   ├── neo_hookean.h
│   │   │   ├── mooney_rivlin.h
│   │   │   ├── ogden.h
│   │   │   └── arruda_boyce.h
│   │   │
│   │   ├── plasticity/      # Plastic models
│   │   │   ├── von_mises.h
│   │   │   ├── drucker_prager.h
│   │   │   ├── mohr_coulomb.h
│   │   │   ├── crystal_plasticity.h
│   │   │   └── gurson_tvergaard.h
│   │   │
│   │   ├── damage/          # Damage models
│   │   │   ├── scalar_damage.h
│   │   │   ├── anisotropic_damage.h
│   │   │   ├── cohesive_zone.h
│   │   │   └── phase_field_fracture.h
│   │   │
│   │   └── viscoplasticity/ # Rate-dependent
│   │       ├── perzyna.h
│   │       ├── chaboche.h
│   │       └── anand.h
│   │
│   ├── multiphysics/        # Coupled models
│   │   ├── thermo_mechanical.h
│   │   ├── thermo_plastic.h
│   │   ├── poro_elastic.h
│   │   ├── electro_mechanical.h
│   │   ├── magneto_mechanical.h
│   │   └── chemo_mechanical.h
│   │
│   └── user_defined/        # User materials
│       ├── umat_interface.h # User material interface
│       ├── python_material.h # Python-defined materials
│       └── expression_material.h # Math expression materials
│
├── kernels/                 # Computational kernels
│   ├── stress_update.h      # Stress integration algorithms
│   ├── tangent_computation.h # Tangent operator computation
│   ├── eigen_decomposition.h # Spectral decompositions
│   ├── return_mapping.h     # Return mapping algorithms
│   └── gpu/                 # GPU-specific kernels
│       ├── cuda_kernels.cu
│       └── hip_kernels.cpp
│
├── interfaces/              # External interfaces
│   ├── abaqus_umat.h        # ABAQUS UMAT compatibility
│   ├── mfront_interface.h   # MFront integration
│   └── constitutive_api.h   # Generic API
│
├── utilities/               # Support utilities (thin wrappers)
│   ├── measures.h           # Stress/strain measure conversions (builds on numeric)
│   ├── transformations.h    # Coordinate transformations
│   ├── units.h              # Unit conversions (optional; prefer core units if available)
│   └── verification.h       # Tangent checking (uses numeric/autodiff)
│
└── database/                # Material database
    ├── material_library.h    # Predefined materials
    ├── property_database.h   # Material properties DB
    └── calibration_data.h    # Experimental data storage
```

> Note on examples: Code snippets in this document use generic placeholders for device annotations (e.g., `DEVICE_FUNC`) and tensor types. In implementation, use the project’s numeric containers and the appropriate device annotations from the device layer.

## Library Dependencies & Boundaries

Materials is a standalone shared library that reuses other high‑level libraries instead of duplicating them:

| Dependency | Type | Purpose | Notes |
|------------|------|---------|-------|
| `numeric`  | public | Containers (vector/matrix/tensor), autodiff, tensor ops, invariants, linear algebra | All math lives here; no reimplementation in materials |
| `device`   | public | Execution spaces, device annotations, kernel launch helpers, host–device buffers | Device code must be allocation‑free and trivially copyable |
| `core`     | private | Factory/registry, configuration, serialization, diagnostics | Do not create a parallel config/logging stack in materials |
| `assembly` | consumer | Calls materials from element kernels | Materials expose evaluate() only; no element loops here |
| `io_formats` | optional | Persist material properties/calibration data | Materials provide schemas/adapters, not parsers |

Non‑dependencies: `physics`, `coupling`, `analysis`. Physics pass field values via `MaterialInputs`; materials return constitutive responses and tangents.

## Reuse‑Over‑Reimplement Checklist

- Use `numeric/operations` and `numeric/math` for tensor algebra, invariants, and spectral utilities.
- Reuse `numeric/autodiff` for verification/production tangents; do not embed a separate AD machinery.
- Use `device` for kernels/launch; no device wrappers inside materials.
- For property I/O, prefer `io_formats` or application code; materials define property schemas only.
- For optimization/calibration, rely on `numeric/optimization`; materials expose property accessors and residual/Jacobian callbacks.

## Unified Evaluation Contracts

To support multiphysics consistently, all material models share a common evaluation interface with structured inputs, outputs, and coupled tangents.

```cpp
// Inputs available to every material (fill only the fields you need)
struct MaterialInputs {
    // Kinematics
    Tensor2 F;          // deformation gradient (finite strain)
    SymTensor2 eps;     // small-strain tensor (small strain)
    Tensor2 L;          // velocity gradient (rate-dependent)
    // Field variables
    double T = 293.15;  // temperature
    double p = 0.0;     // pore pressure
    Vector3 E;          // electric field
    Vector3 B;          // magnetic flux density
    double c = 0.0;     // concentration/chemical potential proxy
    // Gradients (for nonlocal/phase-field)
    Vector3 grad_T;
    Vector3 grad_c;
    // Time increment
    double dt = 0.0;
    // Regime flags
    enum class Kinematics { SmallStrain, FiniteStrain } kin = Kinematics::FiniteStrain;
    enum class Domain { PlaneStress, PlaneStrain, ThreeD } domain = Domain::ThreeD;
    bool compressible = true;
};

// Outputs to be consumed by assembly
struct MaterialOutputs {
    Tensor2 P;          // first Piola-Kirchhoff (finite strain) or Cauchy (small strain)
    Tensor4 C;          // consistent tangent in the chosen stress/strain measure
    double psi = 0.0;   // free energy density (optional)
    Vector3 q;          // heat flux (optional)
    Vector3 j;          // mass/charge flux (optional)
};

// Block-structured coupled tangents (optional, when multiphysics present)
struct TangentBlocks {
    Tensor4 dP_dEps;    // ∂σ/∂ε or ∂P/∂F (consistent measure)
    Tensor2 dP_dT;      // thermal stress tangent
    Tensor2 dP_dp;      // poroelastic coupling
    Matrix dFlux_dGrad; // generic flux vs gradient (Fourier/Fick)
    // Extend with other blocks as needed; unused blocks can be empty.
};

// Canonical evaluate contract
template<class State>
void evaluate(const MaterialInputs& in, State& state, MaterialOutputs& out,
              TangentBlocks* tb = nullptr) const;
```

Guidelines:
- Choose one kinematics regime per model instance (small vs finite strain) and document the returned stress/tangent measures.
- If multiple fields are coupled, populate `TangentBlocks` consistently so global assembly can place blocks without guesswork.

## Packaging & Build

- Build materials as an independent shared library that publicly depends on `numeric` and `device`, and privately on `core`.
- Export only public headers needed by consumers (model interfaces, `MaterialInputs/Outputs/TangentBlocks`, adapters). Keep internal kernels/state manifests private.

## Kinematics and Regimes

- Small-strain vs finite-strain: expose a model trait or configuration setting and use a consistent stress/strain measure (e.g., Cauchy/ε for small strain, P/F for finite strain). Provide helpers to convert measures where necessary.
- Plane stress/plane strain/3D: declare domain in `MaterialInputs` and implement enforcement (e.g., plane stress iterative enforcement of σ_zz = 0).
- Inelastic kinematics: for plasticity/viscoplasticity adopt multiplicative split F = Fe·Fp and store Fe, Fp (or their invariants) in state. Provide invariant helpers (I1, J2, etc.) and spectral utilities for anisotropy.


## Core Design Patterns

### 1. **Static Polymorphism for Performance**

```cpp
// CRTP base for compile-time dispatch in hot paths
template<typename Derived, typename StateType>
class MaterialModel {
public:
    // Interface that derived classes must implement
    template<typename StressType, typename StrainType>
    DEVICE_FUNC inline void compute_stress(
        const StrainType& strain,
        StateType& state,
        StressType& stress) const {
        static_cast<const Derived*>(this)->compute_stress_impl(strain, state, stress);
    }
    
    // Tangent computation with automatic differentiation option
    template<typename TangentType>
    DEVICE_FUNC inline void compute_tangent(
        const StateType& state,
        TangentType& tangent) const {
        static_cast<const Derived*>(this)->compute_tangent_impl(state, tangent);
    }
};

// Concrete implementation with zero virtual overhead
template<typename ScalarType>
class NeoHookean : public MaterialModel<NeoHookean<ScalarType>, 
                                       NeoHookeanState<ScalarType>> {
    using State = NeoHookeanState<ScalarType>;
    
    template<typename StressType, typename StrainType>
    DEVICE_FUNC inline void compute_stress_impl(
        const StrainType& F,  // Deformation gradient
        State& state,
        StressType& P) const { // First Piola-Kirchhoff stress
        // Direct implementation, fully inlinable
        auto J = F.determinant();
        auto Finv = F.inverse();
        P = mu_ * (F - Finv.transpose()) + lambda_ * std::log(J) * Finv.transpose();
    }
    
private:
    ScalarType lambda_, mu_;
};
```

### 2. **Dynamic Dispatch for Configuration**

```cpp
// Runtime polymorphism for material selection/configuration
class IMaterial : public fem::core::Object {
public:
    // Virtual interface for non-performance-critical operations
    virtual void initialize(const fem::core::Config& config) = 0;
    virtual std::unique_ptr<IMaterial> clone() const = 0;
    virtual size_t state_size() const = 0;
    virtual void serialize(fem::core::Serializer& s) const = 0;
    
    // Get material properties for GUI/reporting
    virtual PropertySet get_properties() const = 0;
    virtual std::string material_type() const = 0;
    
    // Factory registration
    static void register_material(const std::string& name,
                                 std::function<std::unique_ptr<IMaterial>()> creator) {
        get_factory().register_creator(name, creator);
    }
    
protected:
    // Template method pattern for common operations
    virtual void validate_parameters() const = 0;
    virtual void compute_derived_quantities() = 0;
};

// Bridge between dynamic and static interfaces
template<typename ConcreteModel>
class MaterialAdapter : public IMaterial {
    ConcreteModel model_;
    
public:
    // Delegate to CRTP implementation
    void compute_response(MaterialPoint& pt) override {
        model_.compute_stress(pt.strain(), pt.state(), pt.stress());
        model_.compute_tangent(pt.state(), pt.tangent());
    }
};
```

### 3. **Material Point State Management**

```cpp
// Lightweight material point with SoA layout for vectorization
template<typename ScalarType, int Dim>
class MaterialPoint {
public:
    using Tensor2 = Eigen::Matrix<ScalarType, Dim, Dim>;
    using Tensor4 = Eigen::Matrix<ScalarType, Dim*Dim, Dim*Dim>;
    using StateVector = Eigen::VectorX<ScalarType>;
    
    // Kinematic quantities
    const Tensor2& deformation_gradient() const { return F_; }
    const Tensor2& velocity_gradient() const { return L_; }
    ScalarType temperature() const { return T_; }
    ScalarType time_increment() const { return dt_; }
    
    // Response quantities (mutable for const material models)
    mutable Tensor2 stress_;
    mutable Tensor4 tangent_;
    mutable StateVector history_;
    
    // State update management
    void push_state() { history_old_ = history_; }
    void pull_state() { history_ = history_old_; }
    bool converged() const { return converged_; }
    
private:
    Tensor2 F_, L_;
    ScalarType T_, dt_;
    StateVector history_old_;
    bool converged_ = false;
};

// Array-of-Structures-of-Arrays for GPU efficiency
template<typename ScalarType, int Dim>
class MaterialPointArray {
    // Coalesced memory access pattern for GPU
    DeviceArray<ScalarType> F_data_;      // All F components
    DeviceArray<ScalarType> stress_data_; // All stress components
    DeviceArray<ScalarType> state_data_;  // All history variables
    
public:
    DEVICE_FUNC MaterialPointView<ScalarType, Dim> 
    operator[](int idx) {
        return MaterialPointView<ScalarType, Dim>(
            &F_data_[idx * Dim * Dim],
            &stress_data_[idx * Dim * Dim],
            &state_data_[idx * state_size_]
        );
    }
};
```

State update discipline:
- Provide a two-phase update API: `begin_trial()` to write trial history variables; `accept()` to commit at convergence; `reject()` to roll back on line-search failure. This guarantees consistent states across Newton iterations.
- Prefer compile-time sized state where feasible; otherwise maintain a manifest that maps indices to named variables to keep SoA layouts stable and versionable.

## Property System Enhancements

- Support scalar, vector, and tensor-valued properties with clear units. Include orientation frames for anisotropic materials.
- Allow temperature/field-dependent property curves with unit-aware evaluation and bounds checking.
- Provide calibration hooks so `database/` and `optimization/` can drive parameter identification (cost functions, priors, constraints).

### 4. **Multiphysics Coupling Interface**

```cpp
// Material coupling for multiphysics
template<typename... PhysicsTypes>
class CoupledMaterial : public IMaterial {
    std::tuple<PhysicsTypes...> physics_materials_;
    
public:
    // Coupled stress computation
    template<size_t... Is>
    void compute_coupled_stress(MaterialPoint& pt, std::index_sequence<Is...>) {
        // Compute individual contributions
        (std::get<Is>(physics_materials_).compute_stress(pt), ...);
        
        // Add coupling terms
        compute_coupling_contributions(pt);
    }
    
    // Field sensitivities for coupled problems
    auto compute_sensitivities(const MaterialPoint& pt) {
        return std::make_tuple(
            compute_temperature_sensitivity(pt),
            compute_pressure_sensitivity(pt),
            compute_electric_field_sensitivity(pt)
        );
    }
    
private:
    void compute_coupling_contributions(MaterialPoint& pt) {
        // Thermo-mechanical coupling
        if constexpr(has_thermal<PhysicsTypes...>()) {
            pt.stress() -= thermal_stress_contribution(pt.temperature());
        }
        
        // Poro-mechanical coupling  
        if constexpr(has_porous<PhysicsTypes...>()) {
            pt.stress() -= biot_coefficient_ * pt.pressure() * Identity;
        }
    }
};
```

Coupling specifics:
- Provide reference coupling terms (e.g., thermal stresses, Biot poroelastic coupling, electro/magnetostriction) and populate the appropriate `TangentBlocks` entries.
- Expose field sensitivities via `MaterialOutputs` or a dedicated API to streamline block assembly in global systems.

### 5. **GPU Material Evaluation**

```cpp
// GPU kernel for material evaluation
template<typename MaterialKernel, typename ScalarType, int Dim>
__global__ void evaluate_material_gpu(
    MaterialKernel material,
    MaterialPointArray<ScalarType, Dim> points,
    int n_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    
    // Get thread-local view of material point
    auto pt = points[idx];
    
    // Evaluate material model (fully inlined)
    material.compute_stress(pt.strain(), pt.state(), pt.stress());
    material.compute_tangent(pt.state(), pt.tangent());
}

// Host-side interface
template<typename Material>
class GPUMaterialEvaluator {
    Material material_;
    
public:
    void evaluate(MaterialPointArray& points) {
        const int threads = 256;
        const int blocks = (points.size() + threads - 1) / threads;
        
        evaluate_material_gpu<<<blocks, threads>>>(
            material_, points, points.size()
        );
        
        cudaDeviceSynchronize();
    }
};
```

Device/parallel requirements:
- No dynamic allocation or virtual dispatch in device code. Use trivially copyable functors and pre-sized SoA buffers.
- Define memory layout and alignment guarantees; ensure coalesced access patterns (document indexing formulas).
- Batch evaluation by material type where possible to improve SIMD and cache efficiency.

### 6. **History Variable Management**

```cpp
// Type-safe history variable storage
template<typename... Variables>
class StateVariables {
    std::tuple<Variables...> variables_;
    
public:
    // Compile-time access by type
    template<typename T>
    T& get() { return std::get<T>(variables_); }
    
    // Runtime access by index for serialization
    void serialize(std::vector<double>& data) const {
        serialize_impl(data, std::index_sequence_for<Variables...>{});
    }
    
    // Automatic differentiation support
    auto to_dual() const {
        return std::apply([](const auto&... vars) {
            return std::make_tuple(make_dual(vars)...);
        }, variables_);
    }
};

// Example: J2 plasticity state
struct J2PlasticityState {
    using Variables = StateVariables<
        Tensor<double, 3, 3>,  // Plastic strain
        double,                // Equivalent plastic strain
        Tensor<double, 3, 3>   // Back stress (kinematic hardening)
    >;
    
    Variables vars;
    
    // Named accessors for clarity
    auto& plastic_strain() { return vars.get<0>(); }
    auto& equivalent_plastic_strain() { return vars.get<1>(); }
    auto& back_stress() { return vars.get<2>(); }
};
```

### 7. **Material Property System**

```cpp
// Flexible property definition with units and bounds
class MaterialProperty : public fem::core::Component {
public:
    using Value = std::variant<double, Vector3d, Matrix3d, std::function<double(double)>>;
    
    MaterialProperty(const std::string& name, 
                    Value default_value,
                    const std::string& units = "",
                    std::optional<std::pair<double, double>> bounds = {})
        : name_(name), value_(default_value), units_(units), bounds_(bounds) {}
    
    // Temperature-dependent properties
    void set_temperature_dependence(std::function<Value(double)> func) {
        temp_dependent_ = func;
    }
    
    Value evaluate(double temperature = 293.15) const {
        if (temp_dependent_) {
            return temp_dependent_(temperature);
        }
        return value_;
    }
    
    // Validation
    bool validate() const {
        if (bounds_ && std::holds_alternative<double>(value_)) {
            double val = std::get<double>(value_);
            return val >= bounds_->first && val <= bounds_->second;
        }
        return true;
    }
    
private:
    std::string name_;
    Value value_;
    std::string units_;
    std::optional<std::pair<double, double>> bounds_;
    std::optional<std::function<Value(double)>> temp_dependent_;
};

// Property set for material models
class PropertySet {
    std::unordered_map<std::string, MaterialProperty> properties_;
    
public:
    // Builder pattern for property definition
    PropertySet& add_scalar(const std::string& name, 
                           double default_val,
                           const std::string& units = "",
                           double min = -inf, 
                           double max = inf) {
        properties_.emplace(name, 
            MaterialProperty(name, default_val, units, std::make_pair(min, max)));
        return *this;
    }
    
    // Load from configuration
    void load_from_config(const fem::core::Config& config) {
        for (auto& [name, prop] : properties_) {
            if (config.has(name)) {
                prop.set_value(config.get<MaterialProperty::Value>(name));
            }
        }
    }
};
```

### 8. **Automatic Tangent Verification**

```cpp
// Verify tangent consistency using finite differences
template<typename Material>
class TangentVerifier {
    Material& material_;
    double tolerance_ = 1e-6;
    double perturbation_ = 1e-8;
    
public:
    bool verify(const MaterialPoint& pt) {
        // Compute analytical tangent
        Tensor4 C_analytical;
        material_.compute_tangent(pt.state(), C_analytical);
        
        // Compute numerical tangent via finite differences
        Tensor4 C_numerical = compute_numerical_tangent(pt);
        
        // Compare
        double error = (C_analytical - C_numerical).norm() / C_analytical.norm();
        
        if (error > tolerance_) {
            LOG_WARNING("Tangent verification failed: error = {}", error);
            LOG_DEBUG("Analytical:\n{}", C_analytical);
            LOG_DEBUG("Numerical:\n{}", C_numerical);
            return false;
        }
        
        return true;
    }
    
private:
    Tensor4 compute_numerical_tangent(const MaterialPoint& pt) {
        Tensor4 C_num;
        Tensor2 stress_plus, stress_minus;
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Perturb strain
                auto strain_plus = pt.strain();
                auto strain_minus = pt.strain();
                strain_plus(i, j) += perturbation_;
                strain_minus(i, j) -= perturbation_;
                
                // Compute stresses
                material_.compute_stress(strain_plus, pt.state(), stress_plus);
                material_.compute_stress(strain_minus, pt.state(), stress_minus);
                
                // Finite difference
                C_num.col(i * 3 + j) = 
                    (stress_plus - stress_minus).reshaped() / (2 * perturbation_);
            }
        }
        
        return C_num;
    }
};
```

## Integration with FEM Infrastructure

### Element Integration

```cpp
// Material evaluation at element level
template<typename Element, typename Material>
class MaterialElementKernel {
public:
    void compute_element_residual(const Element& elem,
                                 const Material& material,
                                 Vector& residual) {
        // Loop over integration points
        for (const auto& ip : elem.integration_points()) {
            // Get material point state
            MaterialPoint pt = extract_material_point(elem, ip);
            
            // Evaluate material model
            material.compute_stress(pt.strain(), pt.state(), pt.stress());
            
            // Accumulate residual
            residual += ip.weight() * elem.B_matrix(ip).transpose() * pt.stress();
        }
    }
    
    void compute_element_tangent(const Element& elem,
                                const Material& material,
                                Matrix& tangent) {
        // Similar pattern for tangent assembly
    }
};
```

## Cross-Module Dependencies

- Depends on `src/numeric` for containers, autodiff types, and tensor utilities.
- Uses `src/device` for device annotations and execution policies.
- Integrates with `src/assembly` for element-level integration and global assembly.
- May use `src/core` factory/configuration helpers. Materials should not depend on physics modules; physics supply field values via `MaterialInputs`.

### Factory Registration

```cpp
// Automatic registration using core::Factory
class MaterialFactory : public fem::core::Factory<IMaterial> {
public:
    static MaterialFactory& instance() {
        static MaterialFactory factory;
        return factory;
    }
    
    // Convenient creation with type safety
    template<typename MaterialType, typename... Args>
    std::unique_ptr<IMaterial> create(Args&&... args) {
        auto material = std::make_unique<MaterialType>(std::forward<Args>(args)...);
        material->validate_parameters();
        return material;
    }
};

// Registration helper
template<typename MaterialType>
class MaterialRegistrar {
public:
    MaterialRegistrar(const std::string& name) {
        MaterialFactory::instance().register_creator(name, []() {
            return std::make_unique<MaterialType>();
        });
    }
};

// Automatic registration via static initialization
#define REGISTER_MATERIAL(Type, Name) \
    static MaterialRegistrar<Type> registrar_##Type(Name);

// Usage
REGISTER_MATERIAL(NeoHookean, "NeoHookean")
REGISTER_MATERIAL(VonMisesPlasticity, "VonMises")
```

## External Interfaces and Adapters

Adapters normalize external material interfaces (UMAT/MFront/Python) into the unified contract:
- Map external inputs/outputs into `MaterialInputs`/`MaterialOutputs` consistently.
- Provide configuration-time dynamic selection; evaluation-time paths must be static and allocation-free for performance.
- Python materials are for prototyping and validation, not hot loops.

## Performance Optimization Strategies

### 1. **Vectorization**
```cpp
// SIMD-friendly material evaluation
template<typename Material>
void evaluate_material_vectorized(const MaterialPointArray& points,
                                 const Material& material) {
    #pragma omp simd
    for (int i = 0; i < points.size(); ++i) {
        material.compute_stress(points[i]);
    }
}
```

### 2. **Cache Optimization**
```cpp
// Group materials by type to improve cache locality
class MaterialBatcher {
    std::unordered_map<std::type_index, std::vector<int>> batches_;
    
public:
    void evaluate_all(const std::vector<IMaterial*>& materials,
                     MaterialPointArray& points) {
        // Process each material type together
        for (const auto& [type, indices] : batches_) {
            evaluate_batch(materials[indices[0]], points, indices);
        }
    }
};
```

### 3. **Expression Templates**
```cpp
// Lazy evaluation of tensor operations
template<typename Expr>
class TensorExpression {
    // Delay computation until assignment
    template<typename Tensor>
    void assign_to(Tensor& result) const {
        static_cast<const Expr*>(this)->eval(result);
    }
};
```

## Return Mapping and Consistent Tangents

Standardize plasticity/viscoplasticity interfaces:
- Trial state computation, yield/check, projection/return mapping, and consistent tangent formation.
- Provide convergence safeguards, optional line search, and robust stopping criteria.
- Encapsulate hardening laws as interchangeable policies.

## Automatic Differentiation Integration

Use AD (numeric/autodiff) for tangent verification and, where feasible, production tangents. Document when analytical tangents are recommended (e.g., stiff inelastic regimes). Provide switches to compare AD vs analytical during testing.

## Nonlocal / Gradient-Enhanced Models

For damage/phase-field and other gradient-enhanced materials:
- Accept and use field gradients (e.g., ∇c, ∇T) from `MaterialInputs`.
- Support neighborhood-based nonlocal averaging (element patch or graph neighborhoods) via extension points.
- Clarify communication needs for assembly so neighboring data is available.

## Testing Infrastructure

```cpp
// Material model test suite
template<typename Material>
class MaterialTestSuite : public ::testing::Test {
protected:
    void test_objectivity() {
        // Frame indifference test
    }
    
    void test_thermodynamic_consistency() {
        // Energy conservation
    }
    
    void test_tangent_consistency() {
        // Tangent verification (multi-field): check all coupled blocks
    }
    
    void test_convergence() {
        // Newton convergence with material tangent
    }
    
    void test_gpu_parity() {
        // CPU vs GPU evaluation parity within tolerances
    }
};

// Benchmark suite
template<typename Material>
class MaterialBenchmark {
    void benchmark_stress_computation();
    void benchmark_tangent_computation();
    void benchmark_gpu_evaluation();
};
```

## Usage Examples

### Simple Linear Elastic Material
```cpp
auto material = MaterialFactory::instance().create<IsotropicElastic>();
material->set_property("young_modulus", 210e9);
material->set_property("poisson_ratio", 0.3);
```

### Complex Multiphysics Material
```cpp
// Thermo-elasto-plastic with damage
auto material = MaterialFactory::instance().create<CoupledMaterial<
    ThermoElastic,
    J2Plasticity,
    ScalarDamage
>>();

material->configure(R"(
    thermal:
        conductivity: 50.0
        specific_heat: 450.0
        expansion: 1.2e-5
    plasticity:
        yield_stress: 250e6
        hardening_modulus: 10e9
    damage:
        threshold: 0.001
        evolution_parameter: 1000.0
)");
```

### GPU Evaluation
```cpp
// Prepare material for GPU
auto gpu_material = material->to_gpu();

// Evaluate on device
GPUMaterialEvaluator<decltype(gpu_material)> evaluator(gpu_material);
evaluator.evaluate(device_points);
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
1. Material base classes and traits
2. State variable management
3. Property system
4. Basic factory infrastructure

### Phase 2: Linear Models (Week 3)
5. Isotropic/orthotropic/anisotropic elasticity
6. Thermal expansion
7. Basic multiphysics coupling interfaces

### Phase 3: Nonlinear Models (Week 4-5)
8. Hyperelastic models (Neo-Hookean, Mooney-Rivlin)
9. J2 plasticity with hardening
10. Return mapping algorithms

### Phase 4: Advanced Features (Week 6-7)
11. Damage and fracture models
12. Crystal plasticity
13. User-defined materials

### Phase 5: Optimization (Week 8)
14. GPU kernel implementation
15. Vectorization and cache optimization
16. Performance benchmarking

## Success Metrics

1. **Performance**: < 5% overhead vs hand-coded materials
2. **Extensibility**: New material in < 100 lines
3. **GPU Efficiency**: > 80% bandwidth utilization
4. **Accuracy**: Machine precision tangents
5. **Testing**: > 95% code coverage
6. **Memory**: < 100 bytes/integration point overhead

## Critical Design Decisions

1. **CRTP over Virtual**: 10-50x performance in tight loops
2. **SoA Layout**: 2-4x GPU memory bandwidth
3. **Expression Templates**: Eliminate temporaries
4. **Compile-Time Dispatch**: Zero overhead material selection
5. **Unified Memory**: Seamless CPU/GPU execution

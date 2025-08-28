# AGENT.md - FEM Tools Module

## Mission
Provide high-performance, FEM-specific utilities that support preprocessing, postprocessing, verification, and analysis workflows while maintaining clean interfaces with the core FEM solver modules.

## Architecture Philosophy
- **Standalone Utilities**: Each tool can function independently
- **Performance Focused**: Optimized for large-scale FEM data
- **GPU-Aware**: Support device memory where applicable
- **Scriptable**: Python bindings for workflow automation
- **Standards Compliant**: Follow FEM community conventions

## Module Structure

```
tools/
├── preprocessors/           # Mesh and problem setup utilities
│   ├── mesh_converters/    # Format conversion utilities
│   ├── mesh_quality/       # Quality checking and improvement
│   ├── boundary_marking/   # Automatic boundary detection
│   ├── load_generators/    # Load pattern generation
│   └── material_assignment/ # Material property mapping
├── postprocessors/          # Result analysis utilities
│   ├── field_calculators/  # Derived field computation
│   ├── integrators/        # Domain/boundary integration
│   ├── extractors/         # Point/line/surface extraction
│   ├── comparators/        # Solution comparison tools
│   └── statistics/         # Statistical analysis
├── verification/            # Verification utilities
│   ├── mms_generators/     # Manufactured solution generators
│   ├── convergence/        # Convergence rate calculators
│   ├── benchmarks/         # Standard benchmark problems
│   └── error_norms/        # Error norm computations
├── mesh_tools/             # Advanced mesh utilities
│   ├── decomposers/        # Domain decomposition tools
│   ├── refinement_tools/   # Manual refinement utilities
│   ├── morphing/           # Mesh deformation tools
│   ├── stitching/          # Mesh merging/stitching
│   └── generators/         # Parametric mesh generators
├── solution_tools/         # Solution manipulation
│   ├── interpolators/      # Solution interpolation
│   ├── projectors/         # Field projection tools
│   ├── restart/            # Checkpoint/restart utilities
│   ├── sensitivity/        # Sensitivity analysis
│   └── reducers/           # Model order reduction
├── debugging/              # Debug and diagnostic tools
│   ├── matrix_spy/         # Sparse matrix visualization
│   ├── convergence_monitor/# Solver convergence tracking
│   ├── memory_profiler/    # Memory usage analysis
│   ├── performance/        # Performance profiling
│   └── validators/         # Solution validation
├── workflow/               # Workflow automation
│   ├── parameter_sweep/    # Parametric studies
│   ├── optimization/       # Design optimization tools
│   ├── uncertainty/        # UQ workflow tools
│   ├── job_submission/     # HPC job management
│   └── report_generation/  # Automated reporting
└── bindings/              # Language bindings
    ├── python/            # Python interfaces
    ├── matlab/            # MATLAB interfaces
    └── julia/             # Julia interfaces
```

## Module Descriptions

### 1. **preprocessors/** - Mesh and Problem Setup

#### **mesh_converters/**
```cpp
class MeshConverter : public core::Object {
public:
    // Universal conversion interface
    void convert(const std::string& input_file,
                const std::string& output_file,
                const ConversionOptions& opts);
    
    // Format detection
    MeshFormat detect_format(const std::string& file);
    
    // Batch conversion
    void convert_batch(const std::vector<std::string>& files);
    
    // GPU-aware conversion for large meshes
    template<typename Device>
    void convert_large_mesh(const std::string& input,
                           const std::string& output);
};

// Specific converters
class AbaqusToNative : public MeshConverter { };
class AnsysToNative : public MeshConverter { };
class GmshToNative : public MeshConverter { };
class NastranToNative : public MeshConverter { };
```

#### **mesh_quality/**
```cpp
class MeshQualityChecker : public core::Component {
    // Quality metrics
    struct QualityMetrics {
        double min_jacobian;
        double max_aspect_ratio;
        double min_angle;
        double max_angle;
        double volume_ratio;
        std::vector<int> inverted_elements;
    };
    
    QualityMetrics analyze(const Mesh& mesh);
    
    // GPU-accelerated for large meshes
    QualityMetrics analyze_gpu(const DeviceMesh& mesh);
    
    // Automatic fixing
    void fix_inverted_elements(Mesh& mesh);
    void smooth_poor_quality(Mesh& mesh);
    
    // Report generation
    void generate_quality_report(const Mesh& mesh,
                                const std::string& filename);
};
```

#### **boundary_marking/**
```cpp
class BoundaryMarker : public core::Object {
    // Automatic detection based on geometry
    void mark_by_normal(Mesh& mesh, 
                       const Vector3d& normal,
                       double tolerance,
                       int marker_id);
    
    void mark_by_position(Mesh& mesh,
                         std::function<bool(Vector3d)> predicate,
                         int marker_id);
    
    // Feature detection
    void detect_edges(Mesh& mesh, double angle_threshold);
    void detect_corners(Mesh& mesh);
    
    // Interactive marking (with GUI callback)
    void mark_interactive(Mesh& mesh,
                         ISelectionCallback* callback);
};
```

### 2. **postprocessors/** - Result Analysis

#### **field_calculators/**
```cpp
class FieldCalculator : public core::Component {
    // Derived fields from primary solution
    Field compute_von_mises_stress(const Field& stress_tensor);
    Field compute_strain_energy(const Field& stress, const Field& strain);
    Field compute_vorticity(const Field& velocity);
    Field compute_pressure_gradient(const Field& pressure);
    
    // GPU kernels for large fields
    template<typename Device>
    DeviceField compute_derived_gpu(const DeviceField& primary,
                                   FieldOperation op);
    
    // Custom field operations via expressions
    Field evaluate_expression(const std::string& expr,
                            const FieldMap& fields);
};
```

#### **integrators/**
```cpp
class DomainIntegrator : public core::Object {
    // Volume integrals
    double integrate_volume(const Field& field,
                           const Mesh& mesh);
    
    // Surface integrals  
    double integrate_surface(const Field& field,
                            const Mesh& mesh,
                            int boundary_id);
    
    // Line integrals
    double integrate_line(const Field& field,
                         const Path& path);
    
    // Flux calculations
    double compute_flux(const Field& field,
                       const Vector3d& normal,
                       int boundary_id);
    
    // Parallel integration with MPI reduction
    double integrate_parallel(const Field& field,
                            const DistributedMesh& mesh);
};
```

#### **extractors/**
```cpp
class DataExtractor : public core::Component {
    // Point data extraction
    double extract_at_point(const Field& field,
                           const Point3d& point);
    
    // Line extraction for plotting
    std::vector<double> extract_along_line(const Field& field,
                                          const Line& line,
                                          int n_samples);
    
    // Surface extraction
    Field extract_on_surface(const Field& field,
                            int boundary_id);
    
    // Time history extraction
    TimeSeries extract_history(const Field& field,
                              const Point3d& point,
                              const TimeRange& range);
    
    // Probe arrays
    class ProbeArray {
        std::vector<Point3d> locations;
        void sample(const Field& field, double time);
        void write_history(const std::string& filename);
    };
};
```

### 3. **verification/** - Verification Utilities

#### **mms_generators/**
```cpp
class MMSGenerator : public core::Object {
    // Generate manufactured solutions
    struct MMSSolution {
        std::function<double(Point3d, double)> exact_solution;
        std::function<Vector3d(Point3d, double)> exact_gradient;
        std::function<double(Point3d, double)> source_term;
    };
    
    // Problem-specific MMS
    MMSSolution generate_poisson_mms(int polynomial_order);
    MMSSolution generate_elasticity_mms(int order);
    MMSSolution generate_heat_mms(int spatial_order, int temporal_order);
    MMSSolution generate_navier_stokes_mms(int order);
    
    // Automatic source term derivation
    template<typename PDEOperator>
    auto derive_source_term(const std::function<double(Point3d)>& u_exact,
                           PDEOperator L) {
        // Symbolic/automatic differentiation
        return L(u_exact);
    }
};
```

#### **convergence/**
```cpp
class ConvergenceAnalyzer : public core::Object {
    struct ConvergenceData {
        std::vector<double> h_values;
        std::vector<double> errors;
        double convergence_rate;
        double theoretical_rate;
        bool is_optimal;
    };
    
    // Spatial convergence
    ConvergenceData analyze_spatial(const Problem& problem,
                                   const std::vector<int>& refinement_levels);
    
    // Temporal convergence  
    ConvergenceData analyze_temporal(const Problem& problem,
                                    const std::vector<double>& timesteps);
    
    // hp-convergence
    ConvergenceData analyze_hp(const Problem& problem,
                              const HPStrategy& strategy);
    
    // Plotting
    void plot_convergence(const ConvergenceData& data,
                         const std::string& filename);
};
```

### 4. **mesh_tools/** - Advanced Mesh Utilities

#### **decomposers/**
```cpp
class DomainDecomposer : public core::Component {
    // Partitioning strategies
    enum Strategy {
        METIS,
        PARMETIS,
        SCOTCH,
        GEOMETRIC,
        SPECTRAL
    };
    
    // Decompose mesh for parallel execution
    std::vector<SubDomain> decompose(const Mesh& mesh,
                                    int n_partitions,
                                    Strategy strategy);
    
    // Load balancing with weights
    void decompose_weighted(const Mesh& mesh,
                           const std::vector<double>& element_weights,
                           int n_partitions);
    
    // Dynamic repartitioning
    void rebalance(DistributedMesh& mesh,
                  const std::vector<double>& current_loads);
    
    // GPU-aware partitioning
    void partition_for_gpu(const Mesh& mesh,
                          int n_gpus,
                          const GPUTopology& topology);
};
```

#### **morphing/**
```cpp
class MeshMorpher : public core::Object {
    // Shape morphing
    void morph_rbf(Mesh& mesh,
                  const std::vector<ControlPoint>& control_points,
                  const std::vector<Vector3d>& displacements);
    
    // FFD (Free Form Deformation)
    void morph_ffd(Mesh& mesh,
                  const FFDBox& control_box);
    
    // Parametric morphing
    void morph_parametric(Mesh& mesh,
                         const ParameterSet& params);
    
    // Quality-preserving morphing
    void morph_with_smoothing(Mesh& mesh,
                             const BoundaryDisplacement& disp);
};
```

### 5. **solution_tools/** - Solution Manipulation

#### **interpolators/**
```cpp
class SolutionInterpolator : public core::Component {
    // Inter-mesh interpolation
    void interpolate(const Field& source_field,
                    const Mesh& source_mesh,
                    Field& target_field,
                    const Mesh& target_mesh);
    
    // Conservative interpolation
    void interpolate_conservative(const Field& source,
                                 Field& target);
    
    // High-order interpolation
    template<int Order>
    void interpolate_high_order(const Field& source,
                               Field& target);
    
    // GPU-accelerated for large problems
    void interpolate_gpu(const DeviceField& source,
                        DeviceField& target);
};
```

#### **restart/**
```cpp
class RestartManager : public core::Object {
    // Checkpoint writing
    void write_checkpoint(const State& state,
                         const std::string& filename);
    
    // Restart reading with verification
    State read_checkpoint(const std::string& filename);
    
    // Incremental checkpointing
    void write_incremental(const State& state,
                          int checkpoint_id);
    
    // Parallel I/O for large-scale
    void write_parallel(const DistributedState& state,
                       const std::string& prefix);
    
    // Automatic checkpoint management
    class CheckpointScheduler {
        void schedule(double interval);
        void cleanup_old(int keep_last_n);
    };
};
```

### 6. **debugging/** - Debug and Diagnostic Tools

#### **matrix_spy/**
```cpp
class MatrixSpy : public core::Component {
    // Visualize sparsity pattern
    void spy(const SparseMatrix& A,
            const std::string& filename);
    
    // Interactive matrix explorer
    void explore_interactive(const SparseMatrix& A);
    
    // Condition number estimation
    double estimate_condition_number(const SparseMatrix& A);
    
    // Bandwidth analysis
    BandwidthInfo analyze_bandwidth(const SparseMatrix& A);
    
    // Symmetry checking
    SymmetryInfo check_symmetry(const SparseMatrix& A,
                               double tolerance);
};
```

#### **convergence_monitor/**
```cpp
class ConvergenceMonitor : public core::Observer {
    // Real-time monitoring
    void attach_to_solver(ISolver* solver);
    
    // Convergence metrics
    struct Metrics {
        std::vector<double> residual_history;
        std::vector<double> error_estimates;
        int iteration_count;
        double convergence_rate;
    };
    
    // Live plotting
    void plot_convergence_live(const Metrics& metrics);
    
    // Stagnation detection
    bool detect_stagnation(const Metrics& metrics);
    
    // Adaptive tolerance adjustment
    double suggest_tolerance(const Metrics& metrics);
};
```

#### **memory_profiler/**
```cpp
class MemoryProfiler : public core::Component {
    // Track allocations
    void start_profiling();
    void stop_profiling();
    
    // Memory usage by component
    struct MemoryUsage {
        size_t peak_usage;
        size_t current_usage;
        std::map<std::string, size_t> component_usage;
    };
    
    MemoryUsage get_profile();
    
    // GPU memory tracking
    MemoryUsage get_gpu_profile();
    
    // Memory leak detection
    std::vector<MemoryLeak> detect_leaks();
    
    // Optimization suggestions
    std::vector<std::string> suggest_optimizations();
};
```

### 7. **workflow/** - Workflow Automation

#### **parameter_sweep/**
```cpp
class ParameterSweep : public core::Object {
    // Define parameter space
    class ParameterSpace {
        void add_parameter(const std::string& name,
                          double min, double max, int n_samples);
        void add_discrete(const std::string& name,
                         const std::vector<double>& values);
    };
    
    // Sweep execution
    void sweep(const Problem& base_problem,
              const ParameterSpace& space,
              std::function<void(const Results&)> callback);
    
    // Parallel sweep with MPI
    void sweep_parallel(const Problem& base_problem,
                       const ParameterSpace& space);
    
    // Adaptive sampling
    void sweep_adaptive(const Problem& base_problem,
                       const ParameterSpace& space,
                       const ResponseMetric& metric);
    
    // Database storage
    void store_results(const std::string& database);
};
```

#### **optimization/**
```cpp
class DesignOptimizer : public core::Component {
    // Optimization problem definition
    struct OptimizationProblem {
        std::function<double(const DesignVector&)> objective;
        std::vector<Constraint> constraints;
        Bounds bounds;
    };
    
    // Gradient-based optimization
    DesignVector optimize_gradient(const OptimizationProblem& problem);
    
    // Gradient-free optimization
    DesignVector optimize_genetic(const OptimizationProblem& problem);
    
    // Topology optimization
    Field optimize_topology(const StructuralProblem& problem,
                           double volume_fraction);
    
    // Shape optimization
    Mesh optimize_shape(const FluidProblem& problem,
                       const ShapeParameters& params);
    
    // Sensitivity computation
    Sensitivities compute_sensitivities(const Problem& problem,
                                       const DesignVector& design);
};
```

## Integration Examples

### Example 1: Verification Workflow
```cpp
// MMS convergence study
void verify_solver() {
    MMSGenerator mms;
    auto solution = mms.generate_poisson_mms(4);
    
    ConvergenceAnalyzer analyzer;
    std::vector<int> refinements = {10, 20, 40, 80, 160};
    
    auto data = analyzer.analyze_spatial(solution, refinements);
    
    // Check optimal convergence
    assert(std::abs(data.convergence_rate - 2.0) < 0.1);
    
    // Generate report
    analyzer.plot_convergence(data, "convergence.png");
}
```

### Example 2: Preprocessing Pipeline
```cpp
// Convert and prepare mesh
void prepare_mesh(const std::string& input_file) {
    // Convert from commercial format
    MeshConverter converter;
    converter.convert(input_file, "mesh.fem", {.optimize = true});
    
    // Check quality
    Mesh mesh = io::read_mesh("mesh.fem");
    MeshQualityChecker checker;
    auto metrics = checker.analyze(mesh);
    
    if (metrics.min_jacobian < 0.1) {
        checker.fix_inverted_elements(mesh);
        checker.smooth_poor_quality(mesh);
    }
    
    // Mark boundaries
    BoundaryMarker marker;
    marker.mark_by_position(mesh, 
        [](auto p) { return std::abs(p.x) < 1e-6; },
        WALL_BOUNDARY);
    
    // Decompose for parallel
    DomainDecomposer decomposer;
    auto partitions = decomposer.decompose(mesh, n_procs, 
                                          DomainDecomposer::METIS);
}
```

### Example 3: Postprocessing Analysis
```cpp
// Extract and analyze results
void analyze_results(const Solution& solution) {
    FieldCalculator calc;
    auto von_mises = calc.compute_von_mises_stress(solution.stress);
    
    // Domain integration
    DomainIntegrator integrator;
    double total_energy = integrator.integrate_volume(
        solution.strain_energy, mesh);
    
    // Extract along line
    DataExtractor extractor;
    Line centerline({0,0,0}, {1,0,0});
    auto profile = extractor.extract_along_line(
        solution.velocity, centerline, 100);
    
    // Time history at probe
    ProbeArray probes;
    probes.add_location({0.5, 0.5, 0});
    probes.sample(solution.pressure, time);
}
```

### Example 4: Parameter Study with Optimization
```cpp
// Automated parameter study
void optimize_design() {
    ParameterSpace space;
    space.add_parameter("thickness", 0.01, 0.1, 20);
    space.add_parameter("young_modulus", 1e9, 1e11, 10);
    
    ParameterSweep sweep;
    sweep.sweep_parallel(base_problem, space);
    
    // Optimize based on results
    DesignOptimizer optimizer;
    OptimizationProblem opt_problem;
    opt_problem.objective = [](const DesignVector& x) {
        return compute_compliance(x);
    };
    opt_problem.constraints.push_back(
        VolumeConstraint(0.3));
    
    auto optimal = optimizer.optimize_gradient(opt_problem);
}
```

## Python Bindings

```python
# Python interface example
import femtools as ft

# Mesh conversion
converter = ft.MeshConverter()
converter.convert("model.inp", "model.fem")

# Quality check
checker = ft.MeshQualityChecker()
mesh = ft.read_mesh("model.fem")
metrics = checker.analyze(mesh)
print(f"Min Jacobian: {metrics.min_jacobian}")

# Parameter sweep
sweep = ft.ParameterSweep()
space = ft.ParameterSpace()
space.add_parameter("pressure", 1e5, 1e6, 50)

def callback(results):
    print(f"Max stress: {results.max_stress}")

sweep.sweep(problem, space, callback)

# Convergence analysis
analyzer = ft.ConvergenceAnalyzer()
data = analyzer.analyze_spatial(problem, [10, 20, 40, 80])
analyzer.plot_convergence(data, "convergence.png")
```

## Performance Considerations

### GPU Acceleration
- Large mesh operations use GPU kernels
- Field calculations leverage device memory
- Parallel algorithms for data extraction

### Memory Management
```cpp
// Streaming for large datasets
class StreamingProcessor : public core::Object {
    void process_large_dataset(const std::string& file) {
        // Process in chunks to avoid memory overflow
        while (has_more_data()) {
            auto chunk = read_chunk();
            process_chunk(chunk);
            write_result(chunk);
        }
    }
};
```

### Parallel Execution
- MPI-aware tools for distributed meshes
- Collective operations for reductions
- Parallel I/O for large files

## Testing Strategy

```
tools/tests/
├── unit/              # Unit tests for each tool
├── integration/       # Tool chain tests
├── performance/       # Benchmark tests
├── validation/        # Validation against analytical
└── regression/        # Regression suite
```

## Build Configuration

```cmake
# Tools module options
option(FEM_TOOLS_BUILD_ALL "Build all tools" ON)
option(FEM_TOOLS_PYTHON "Build Python bindings" ON)
option(FEM_TOOLS_MATLAB "Build MATLAB interface" OFF)
option(FEM_TOOLS_GPU "Enable GPU acceleration" ON)

# External dependencies
find_package(VTK QUIET)
find_package(CGAL QUIET)
find_package(Eigen3 REQUIRED)

# Conditional compilation
if(FEM_TOOLS_GPU)
    add_subdirectory(tools/gpu_kernels)
endif()
```

## Success Metrics

1. **Performance**: Process 100M element mesh in < 60 seconds
2. **Accuracy**: MMS convergence rates within 1% of theoretical
3. **Usability**: Complete workflow in < 10 Python commands
4. **Robustness**: Handle degenerate meshes gracefully
5. **Scalability**: Linear scaling to 1000+ cores
6. **Coverage**: Support all major commercial formats
7. **Integration**: Seamless with main solver

## Implementation Priority

### Phase 1: Core Tools (Essential)
- Mesh converters (Gmsh, Abaqus)
- Basic quality checking
- Simple postprocessing
- MMS generators

### Phase 2: Advanced Tools (Important)
- Domain decomposition
- Convergence analysis
- Field calculators
- Python bindings

### Phase 3: Specialized Tools (Nice-to-have)
- Optimization framework
- Parameter sweep automation
- Interactive debugging
- Advanced morphing

## Notes

- All tools inherit from `core::Object` for consistency
- GPU kernels share device abstraction with main solver
- Python bindings use pybind11 for zero-copy arrays
- Tools can be built and tested independently
- Documentation generated from inline comments
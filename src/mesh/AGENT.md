# AGENT.md - Mesh Management Module (Revised)

## Mission
Provide comprehensive mesh generation, manipulation, adaptation, and management capabilities for finite element analysis, supporting both static and dynamic meshes, high-order curved elements, with efficient data structures for parallel and adaptive computations.

## Architecture Philosophy
- **Topology-First**: Clear separation between topology and geometry
- **High-Order Native**: First-class support for curved elements
- **Parallel-Ready**: Distributed mesh support from the ground up
- **Adaptation-Native**: Built-in support for h-, p-, hp-refinement with solution transfer
- **GPU-Aware**: Data structures designed for heterogeneous computing
- **Motion-Capable**: Robust ALE and deforming mesh support for FSI

## Directory Structure

```
mesh/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── topology/                        # Mesh connectivity
│   ├── topology_base.hpp           # Base topology interface
│   ├── cell.hpp                    # Cell abstraction
│   ├── face.hpp                    # Face abstraction
│   ├── edge.hpp                    # Edge abstraction
│   ├── vertex.hpp                  # Vertex abstraction
│   ├── connectivity.hpp            # Connectivity tables
│   ├── adjacency.hpp               # Adjacency relationships
│   ├── incidence.hpp               # Incidence matrices
│   ├── orientation.hpp             # Element orientation
│   ├── boundary_topology.hpp       # Boundary mesh topology
│   ├── high_order_topology.hpp     # High-order node connectivity (NEW)
│   ├── nonconforming_topology.hpp  # Hanging nodes/mortar (NEW)
│   └── dual_mesh.hpp               # Dual mesh construction
│
├── geometry/                        # Geometric information
│   ├── point.hpp                   # Point in space
│   ├── coordinate_system.hpp       # Coordinate systems
│   ├── bounding_box.hpp            # AABB structures
│   ├── geometric_entity.hpp        # Geometric primitives
│   ├── curve.hpp                   # Curve definitions
│   ├── surface.hpp                 # Surface definitions
│   ├── volume.hpp                  # Volume definitions
│   ├── transformation.hpp          # Geometric transformations
│   ├── distance.hpp                # Distance calculations
│   ├── intersection.hpp            # Intersection tests
│   ├── high_order_mapping.hpp      # Curved element geometry (NEW)
│   ├── cad_geometry.hpp            # CAD-based geometry (NEW)
│   └── isoparametric.hpp           # Isoparametric mappings (NEW)
│
├── mesh_types/                      # Concrete mesh types
│   ├── mesh_base.hpp               # Base mesh interface
│   ├── unstructured_mesh.hpp       # General unstructured
│   ├── structured_mesh.hpp         # Structured/regular
│   ├── cartesian_mesh.hpp          # Cartesian grids
│   ├── hybrid_mesh.hpp             # Mixed element types
│   ├── surface_mesh.hpp            # 2D manifold meshes
│   ├── volume_mesh.hpp             # 3D solid meshes
│   ├── hierarchical_mesh.hpp       # Multi-level meshes
│   ├── high_order_mesh.hpp         # Curved mesh support (NEW)
│   ├── nonconforming_mesh.hpp      # Nonconforming interfaces (NEW)
│   └── patch_mesh.hpp              # Mesh patches
│
├── attributes/                      # Mesh metadata (NEW)
│   ├── mesh_tags.hpp               # Element/face/edge tags
│   ├── region_markers.hpp          # Physical region IDs
│   ├── material_attributes.hpp     # Material assignments
│   ├── bc_attributes.hpp           # BC markers
│   ├── field_attachment.hpp        # Field-mesh coupling
│   └── solver_metadata.hpp         # Solver-specific data
│
├── generators/                      # Mesh generation
│   ├── generator_base.hpp          # Generator interface
│   ├── structured/
│   │   ├── cartesian_generator.hpp # Cartesian grids
│   │   ├── cylindrical_generator.hpp
│   │   ├── spherical_generator.hpp
│   │   ├── mapped_generator.hpp    # Transfinite interpolation
│   │   └── tensor_product.hpp      # Tensor-product meshes
│   ├── unstructured/
│   │   ├── triangulator_2d.hpp     # Delaunay triangulation
│   │   ├── tetrahedralizer_3d.hpp  # Tet mesh generation
│   │   ├── advancing_front.hpp     # Advancing front method
│   │   ├── octree_generator.hpp    # Octree-based
│   │   └── frontal_delaunay.hpp    # Frontal-Delaunay
│   ├── parametric/
│   │   ├── parametric_surface.hpp  # Surface meshing
│   │   ├── parametric_volume.hpp   # Volume from CAD
│   │   └── curved_mesh_generator.hpp # High-order curved (NEW)
│   ├── extruded/
│   │   ├── extrusion.hpp           # 2D to 3D extrusion
│   │   └── revolution.hpp          # Rotational sweep
│   └── special/
│       ├── voronoi.hpp             # Voronoi diagrams
│       ├── boundary_layer.hpp      # Anisotropic BL meshes
│       └── lattice.hpp             # Lattice structures
│
├── quality/                         # Mesh quality
│   ├── quality_metric.hpp          # Quality metric base
│   ├── metrics/
│   │   ├── aspect_ratio.hpp        # Element aspect ratio
│   │   ├── skewness.hpp            # Skewness measure
│   │   ├── jacobian_quality.hpp    # Jacobian-based
│   │   ├── condition_number.hpp    # Condition metrics
│   │   ├── volume_ratio.hpp        # Volume measures
│   │   └── curved_quality.hpp      # High-order metrics (NEW)
│   ├── improvement/
│   │   ├── smoothing.hpp           # Laplacian smoothing
│   │   ├── optimization_smoothing.hpp
│   │   ├── edge_swapping.hpp       # Topological improvement
│   │   ├── curved_smoothing.hpp    # High-order smoothing (NEW)
│   │   └── sliver_removal.hpp      # Remove degenerate elements
│   ├── monitoring/                 # Quality tracking (NEW)
│   │   ├── quality_monitor.hpp     # Real-time monitoring
│   │   ├── deformation_tracker.hpp # Large deformation detect
│   │   └── quality_history.hpp     # Quality evolution
│   └── validation/
│       ├── mesh_validator.hpp      # Mesh validity checks
│       ├── topology_check.hpp      # Topology consistency
│       ├── geometry_check.hpp      # Geometric validity
│       └── high_order_check.hpp    # Curved element validity (NEW)
│
├── partitioning/                    # Domain decomposition
│   ├── partitioner_base.hpp        # Partitioner interface
│   ├── graph_partitioner.hpp       # Graph-based methods
│   ├── geometric_partitioner.hpp   # Coordinate-based
│   ├── recursive_bisection.hpp     # Recursive bisection
│   ├── space_filling_curve.hpp     # SFC partitioning
│   ├── load_balancer.hpp           # Dynamic load balancing
│   ├── partition_interface.hpp     # Inter-partition boundaries
│   ├── gpu_partitioner.hpp         # GPU-aware partitioning (NEW)
│   └── external/
│       ├── metis_wrapper.hpp       # METIS interface
│       ├── scotch_wrapper.hpp      # SCOTCH interface
│       ├── zoltan_wrapper.hpp      # Zoltan interface
│       └── parmetis_wrapper.hpp    # ParMETIS (NEW)
│
├── motion/                         # Mesh motion (ALE)
│   ├── mesh_motion_base.hpp        # Motion interface
│   ├── ale/
│   │   ├── ale_mesh.hpp            # ALE mesh class
│   │   ├── mesh_velocity.hpp       # Mesh velocity computation
│   │   ├── ale_update.hpp          # ALE coordinate update
│   │   └── quality_preserving.hpp  # Quality maintenance (NEW)
│   ├── morphing/
│   │   ├── rbf_morphing.hpp        # RBF-based morphing
│   │   ├── laplacian_morphing.hpp  # Laplacian smoothing
│   │   ├── elastic_morphing.hpp    # Pseudo-structural
│   │   ├── idr_morphing.hpp        # IDR morphing
│   │   └── large_deformation.hpp   # Large deform handling (NEW)
│   ├── remeshing/
│   │   ├── local_remeshing.hpp     # Local remeshing
│   │   ├── global_remeshing.hpp    # Complete remesh
│   │   ├── remesh_criteria.hpp     # Enhanced criteria (ENHANCED)
│   │   └── automatic_remesh.hpp    # Auto regeneration (NEW)
│   └── tracking/
│       ├── interface_tracking.hpp  # Interface motion
│       ├── front_tracking.hpp      # Explicit fronts
│       └── level_set_mesh.hpp      # Level-set based
│
├── interpolation/                  # Inter-mesh transfer
│   ├── interpolator_base.hpp       # Interpolation interface
│   ├── conservative_transfer.hpp   # Conservative interpolation
│   ├── consistent_transfer.hpp     # Consistent interpolation
│   ├── projection.hpp              # L2 projection
│   ├── nearest_neighbor.hpp        # Nearest point
│   ├── radial_basis.hpp            # RBF interpolation
│   ├── mortar_projection.hpp       # Mortar methods
│   ├── nonconforming_interface.hpp # Nonconforming support (NEW)
│   └── high_order_transfer.hpp     # Curved mesh transfer (NEW)
│
├── search/                         # Spatial search structures
│   ├── search_tree.hpp             # Base search interface
│   ├── kdtree.hpp                  # k-d tree
│   ├── octree.hpp                  # Octree
│   ├── bvh.hpp                     # Bounding volume hierarchy
│   ├── rtree.hpp                   # R-tree
│   ├── point_locator.hpp           # Point-in-element
│   ├── neighbor_search.hpp         # Nearest neighbors
│   └── gpu_search.hpp              # GPU-accelerated search
│
├── io/                              # Mesh I/O
│   ├── mesh_reader.hpp             # Reader interface
│   ├── mesh_writer.hpp             # Writer interface
│   ├── formats/
│   │   ├── gmsh_io.hpp             # Gmsh format
│   │   ├── exodus_io.hpp           # Exodus II
│   │   ├── vtk_io.hpp              # VTK/VTU format
│   │   ├── xdmf_io.hpp             # XDMF/HDF5 format
│   │   ├── hdf5_io.hpp             # Native HDF5
│   │   ├── abaqus_io.hpp           # Abaqus input
│   │   ├── nastran_io.hpp          # Nastran bulk
│   │   ├── cgns_io.hpp             # CGNS format
│   │   ├── stl_io.hpp              # STL format
│   │   └── native_io.hpp           # Native format
│   └── parallel_io/
│       ├── distributed_reader.hpp  # Parallel read
│       ├── distributed_writer.hpp  # Parallel write
│       └── mpi_hdf5_io.hpp         # Parallel HDF5
│
├── parallel/                       # Distributed mesh
│   ├── distributed_mesh.hpp        # Distributed mesh class
│   ├── ghost_cells.hpp             # Ghost element layer
│   ├── halo_exchange.hpp           # Halo communication
│   ├── mesh_migration.hpp          # Element migration
│   ├── parallel_refinement.hpp     # Parallel adaptation
│   ├── gpu_refinement.hpp          # GPU-aware refinement (NEW)
│   ├── partition_boundary.hpp      # Partition interfaces
│   └── scalable_algorithms.hpp     # Scalable mesh ops (NEW)
│
├── boundary/                        # Boundary mesh
│   ├── boundary_mesh.hpp           # Boundary extraction
│   ├── surface_extraction.hpp      # Extract surfaces
│   ├── edge_extraction.hpp         # Extract edges
│   ├── boundary_markers.hpp        # Boundary identification
│   ├── periodic_boundary.hpp       # Periodic mesh
│   └── boundary_layer_mesh.hpp     # BL mesh generation
│
├── algorithms/                      # Mesh algorithms
│   ├── connectivity_algorithms.hpp  # Connectivity computation
│   ├── mesh_boolean.hpp            # Boolean operations
│   ├── mesh_intersection.hpp       # Intersection algorithms
│   ├── convex_hull.hpp             # Convex hull
│   ├── mesh_simplification.hpp     # Decimation
│   ├── feature_detection.hpp       # Sharp features
│   └── mortar_assembly.hpp         # Mortar methods (NEW)
│
├── data_structures/                 # Efficient storage
│   ├── compact_storage.hpp         # Memory-efficient storage
│   ├── array_of_structs.hpp        # AoS layout
│   ├── struct_of_arrays.hpp        # SoA layout
│   ├── compressed_row.hpp          # CSR-like storage
│   ├── hierarchical_storage.hpp    # Multi-level storage
│   ├── gpu_storage.hpp             # GPU-friendly layouts (NEW)
│   └── hybrid_storage.hpp          # CPU-GPU unified (NEW)
│
├── utilities/                       # Mesh utilities
│   ├── mesh_traits.hpp             # Type traits
│   ├── mesh_iterators.hpp          # Iterator interfaces
│   ├── mesh_statistics.hpp         # Mesh statistics
│   ├── mesh_converter.hpp          # Format conversion
│   ├── mesh_diagnostics.hpp        # Diagnostic tools
│   └── performance_monitor.hpp     # Performance tracking (NEW)
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── quality/                    # Quality tests
    ├── performance/                # Performance benchmarks
    ├── validation/                 # Validation cases
    └── gpu/                        # GPU-specific tests (NEW)
```

## Key Components (Enhanced)

### 1. High-Order Curved Mesh Support
```cpp
// High-order mesh with curved elements
class HighOrderMesh : public Mesh {
    // Curved geometry representation
    struct CurvedElement {
        std::vector<Point> nodes;           // All nodes (vertex + edge + face)
        std::vector<int> orders;            // Polynomial order per edge/face
        GeometricMapping mapping;           // Isoparametric/subparametric
    };
    
    // CAD-based curving
    void curve_to_geometry(const CADGeometry& cad) {
        for (auto& elem : boundary_elements()) {
            // Project high-order nodes to CAD
            elem.curve_to_surface(cad.surface(elem.bc_tag));
            
            // Ensure validity (no inversions)
            if (!elem.is_valid()) {
                elem.uncurve();  // Fall back to linear
            }
        }
    }
    
    // High-order quality metrics
    double min_jacobian() const {
        double min_j = 1e10;
        for (auto& elem : elements()) {
            // Check Jacobian at many points
            auto j_min = elem.compute_min_jacobian(100);
            min_j = std::min(min_j, j_min);
        }
        return min_j;
    }
};
```

### 2. Solution Transfer Infrastructure
```cpp
// Conservative solution transfer for AMR
class SolutionTransfer {
    // L2 projection for refinement
    void project_to_refined(const Mesh& coarse_mesh,
                           const Solution& coarse_sol,
                           const Mesh& fine_mesh,
                           Solution& fine_sol) {
        // Conservative transfer
        for (auto& fine_elem : fine_mesh.elements()) {
            auto parent = coarse_mesh.parent_element(fine_elem);
            
            // L2 projection on element
            auto mass = compute_mass_matrix(fine_elem);
            auto rhs = compute_projection_rhs(parent, coarse_sol);
            fine_sol.local(fine_elem) = mass.solve(rhs);
        }
    }
    
    // Restriction for coarsening
    void restrict_to_coarse(const Solution& fine_sol,
                           Solution& coarse_sol) {
        // Volume-weighted average
        for (auto& coarse_elem : coarse_mesh.elements()) {
            double total_vol = 0;
            Vector weighted_sum(n_dofs);
            
            for (auto& child : fine_mesh.children(coarse_elem)) {
                auto vol = child.volume();
                weighted_sum += vol * fine_sol.local(child);
                total_vol += vol;
            }
            
            coarse_sol.local(coarse_elem) = weighted_sum / total_vol;
        }
    }
};
```

### 3. Mesh Attributes & Metadata
```cpp
// Rich metadata for solver coupling
class MeshAttributes {
    // Region/material tags
    std::unordered_map<int, std::string> region_names;
    std::unordered_map<int, int> element_materials;
    
    // Boundary conditions
    struct BCMarker {
        BCType type;
        int physics_id;
        std::any data;
    };
    std::unordered_map<int, BCMarker> boundary_conditions;
    
    // Field attachments
    std::vector<FieldInfo> attached_fields;
    
    // Solver-specific metadata
    std::unordered_map<std::string, std::any> solver_data;
    
    // Easy access
    auto elements_in_region(const std::string& name) {
        return ElementRange(region_map[name]);
    }
};
```

### 4. Robust ALE with Quality Preservation
```cpp
// Enhanced ALE with automatic quality control
class RobustALEMesh : public ALEMesh {
    QualityMonitor monitor;
    
    // Update with quality preservation
    void update_motion(const BoundaryMotion& motion, double dt) {
        // Store initial quality
        auto initial_quality = monitor.compute_quality(*this);
        
        // Apply boundary motion
        apply_boundary_displacement(motion);
        
        // Iterative smoothing with quality checks
        for (int iter = 0; iter < max_smoothing_iters; ++iter) {
            smooth_interior();
            
            auto quality = monitor.compute_quality(*this);
            if (quality.min_jacobian < critical_jacobian) {
                // Trigger local remeshing
                local_remesh(quality.bad_elements());
            }
            
            if (quality.satisfactory()) break;
        }
        
        // Full remesh if quality still bad
        if (monitor.requires_full_remesh()) {
            auto new_mesh = regenerate_mesh();
            transfer_solution(new_mesh);
            *this = std::move(new_mesh);
        }
    }
};
```

### 5. GPU-Aware Data Structures
```cpp
// GPU-optimized mesh storage
class GPUMesh {
    // SoA layout for coalesced access
    struct DeviceData {
        float* x, *y, *z;              // Coordinates
        int* cells;                    // Connectivity (flat)
        int* cell_offsets;            // CSR offsets
        int* cell_types;              // Element types
    };
    
    DeviceData d_data;
    bool on_device = false;
    
    // Transfer to GPU
    void to_device() {
        if (!on_device) {
            cudaMemcpy(d_data.x, h_data.x, ...);
            // ... copy all arrays
            on_device = true;
        }
    }
    
    // GPU kernels for refinement
    __global__ void mark_refinement_gpu(const float* error,
                                       int* refinement_flags) {
        int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e < n_elements) {
            if (error[e] > threshold) {
                refinement_flags[e] = 1;
            }
        }
    }
};
```

### 6. Nonconforming Mesh Support
```cpp
// Mortar methods for nonconforming interfaces
class NonconformingMesh : public Mesh {
    // Mortar spaces at interfaces
    struct MortarInterface {
        std::vector<int> master_faces;
        std::vector<int> slave_faces;
        MortarSpace mortar_space;
        Matrix coupling_matrix;  // Mortar constraints
    };
    
    std::vector<MortarInterface> interfaces;
    
    // Build mortar coupling
    void build_mortar_coupling() {
        for (auto& interface : interfaces) {
            // Project slave to mortar space
            auto M_sm = compute_mortar_projection(
                interface.slave_faces,
                interface.mortar_space
            );
            
            // Project master to mortar space
            auto M_mm = compute_mortar_projection(
                interface.master_faces,
                interface.mortar_space
            );
            
            interface.coupling_matrix = M_sm.transpose() * M_mm;
        }
    }
};
```

### 7. Multigrid Transfer Operators
```cpp
// Transfer operators for multigrid
class MultigridTransfer {
    // Geometric multigrid transfers
    Matrix prolongation;   // Coarse to fine
    Matrix restriction;    // Fine to coarse
    
    void build_geometric_transfer(const Mesh& fine,
                                 const Mesh& coarse) {
        // Build prolongation (interpolation)
        for (auto& fine_node : fine.nodes()) {
            auto coarse_elem = coarse.locate(fine_node.coords());
            auto N = coarse_elem.shape_functions(fine_node.coords());
            
            for (int i = 0; i < coarse_elem.n_nodes(); ++i) {
                prolongation(fine_node.id(), coarse_elem.node(i)) = N[i];
            }
        }
        
        // Restriction = transpose (Galerkin)
        restriction = prolongation.transpose();
    }
};
```

## Performance & Robustness

### Quality Monitoring System
```cpp
class QualityMonitor {
    struct QualityMetrics {
        double min_jacobian;
        double max_aspect_ratio;
        double min_volume;
        std::vector<int> bad_elements;
        
        bool satisfactory() const {
            return min_jacobian > 0.1 && 
                   max_aspect_ratio < 100;
        }
    };
    
    // Real-time monitoring during deformation
    void monitor_deformation(const Mesh& mesh) {
        auto metrics = compute_quality(mesh);
        
        if (metrics.min_jacobian < 0.05) {
            trigger_emergency_remesh();
        } else if (metrics.min_jacobian < 0.2) {
            trigger_local_remesh(metrics.bad_elements);
        }
    }
};
```

## Enhanced Integration Points

### With fem/
- Provides high-order element connectivity
- Supplies curved geometry for shape functions
- Manages field attachments and metadata

### With solvers/
- Provides multigrid transfer operators
- Supplies mesh hierarchy for geometric MG
- Manages solution transfer for AMR

### With physics/
- Region/material attributes drive physics assignment
- BC markers connect to physics boundary conditions
- Mortar interfaces for multi-physics coupling

## Updated Success Metrics

1. **High-Order Support**: Curved elements with positive Jacobian
2. **Solution Transfer**: Conservative to machine precision
3. **ALE Robustness**: Handle 10x initial deformation
4. **GPU Efficiency**: 100x speedup for refinement marking
5. **Parallel AMR**: < 5% overhead for dynamic refinement
6. **I/O Performance**: 1GB/s for HDF5 parallel I/O

## Key Enhancements

1. **High-Order Native**: Full support for curved elements and CAD integration
2. **Robust Solution Transfer**: Conservative prolongation/restriction operators
3. **Quality Preservation**: Automatic monitoring and recovery for large deformations
4. **GPU Acceleration**: GPU-aware data structures and algorithms
5. **HPC I/O**: XDMF/HDF5 for scalable parallel I/O
6. **Nonconforming Support**: Mortar methods for interface coupling
7. **Metadata Rich**: Comprehensive attributes for physics coupling

This revised architecture addresses all identified gaps while maintaining the comprehensive coverage of the original design.
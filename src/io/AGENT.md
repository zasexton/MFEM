# AGENT.md - Input/Output Module

## Mission
Provide comprehensive input/output capabilities for finite element simulations, supporting multiple file formats, parallel I/O, in-situ visualization, checkpoint/restart, and efficient data management for both pre-processing and post-processing.

## Architecture Philosophy
- **Format-Agnostic**: Unified interface for multiple file formats
- **Parallel-First**: Scalable I/O for distributed simulations
- **Stream-Capable**: Support for both file-based and streaming I/O
- **Metadata-Rich**: Preserve simulation context and provenance
- **Performance-Optimized**: Minimize I/O bottlenecks

## Directory Structure

```
io/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── core/                            # Core I/O infrastructure
│   ├── io_manager.hpp              # I/O orchestration
│   ├── file_handler.hpp            # File handling abstraction
│   ├── stream_handler.hpp          # Stream I/O interface
│   ├── io_context.hpp              # I/O context and metadata
│   ├── io_error.hpp                # Error handling
│   ├── io_traits.hpp               # I/O type traits
│   └── io_factory.hpp              # Factory for format selection
│
├── formats/                         # File format implementations
│   ├── native/
│   │   ├── fem_native.hpp          # Native FEM format
│   │   ├── binary_format.hpp       # Binary representation
│   │   └── schema_version.hpp      # Format versioning
│   ├── mesh/
│   │   ├── gmsh/
│   │   │   ├── gmsh_reader.hpp     # Gmsh mesh reader
│   │   │   ├── gmsh_writer.hpp     # Gmsh mesh writer
│   │   │   └── gmsh_physical.hpp   # Physical groups
│   │   ├── exodus/
│   │   │   ├── exodus_reader.hpp   # Exodus II reader
│   │   │   ├── exodus_writer.hpp   # Exodus II writer
│   │   │   └── exodus_metadata.hpp # Exodus metadata
│   │   ├── abaqus/
│   │   │   ├── abaqus_reader.hpp   # Abaqus input file
│   │   │   └── abaqus_keywords.hpp # Keyword parsing
│   │   ├── nastran/
│   │   │   ├── nastran_reader.hpp  # Nastran bulk data
│   │   │   └── nastran_cards.hpp   # Card definitions
│   │   ├── cgns/
│   │   │   ├── cgns_reader.hpp     # CGNS format
│   │   │   └── cgns_writer.hpp
│   │   └── stl/
│   │       ├── stl_reader.hpp      # STL format
│   │       └── stl_ascii.hpp       # ASCII STL
│   ├── visualization/
│   │   ├── vtk/
│   │   │   ├── vtk_legacy.hpp      # Legacy VTK
│   │   │   ├── vtu_writer.hpp      # VTU XML format
│   │   │   ├── pvtu_writer.hpp     # Parallel VTU
│   │   │   └── vtk_attributes.hpp  # Field attributes
│   │   ├── xdmf/
│   │   │   ├── xdmf_writer.hpp     # XDMF metadata
│   │   │   ├── xdmf_reader.hpp
│   │   │   └── xdmf_schema.hpp     # XDMF schema
│   │   ├── ensight/
│   │   │   ├── ensight_gold.hpp    # EnSight Gold
│   │   │   └── ensight_case.hpp    # Case file
│   │   └── tecplot/
│   │       └── tecplot_ascii.hpp   # Tecplot format
│   ├── scientific/
│   │   ├── hdf5/
│   │   │   ├── hdf5_handler.hpp    # HDF5 base handler
│   │   │   ├── hdf5_dataset.hpp    # Dataset management
│   │   │   ├── hdf5_attributes.hpp # Metadata attributes
│   │   │   └── hdf5_parallel.hpp   # Parallel HDF5
│   │   ├── netcdf/
│   │   │   └── netcdf_handler.hpp  # NetCDF support
│   │   └── matlab/
│   │       └── mat_file.hpp        # MATLAB .mat files
│   └── specialized/
│       ├── csv/
│       │   └── csv_handler.hpp     # CSV import/export
│       ├── json/
│       │   └── json_handler.hpp    # JSON format
│       └── xml/
│           └── xml_handler.hpp     # XML parsing
│
├── parallel/                        # Parallel I/O
│   ├── mpi_io/
│   │   ├── mpi_file_handler.hpp    # MPI-IO wrapper
│   │   ├── collective_io.hpp       # Collective operations
│   │   ├── file_view.hpp           # MPI file views
│   │   └── io_aggregation.hpp      # I/O aggregation
│   ├── distributed/
│   │   ├── partitioned_writer.hpp  # Partitioned output
│   │   ├── partitioned_reader.hpp  # Partitioned input
│   │   ├── gather_scatter_io.hpp   # Gather/scatter patterns
│   │   └── parallel_merge.hpp      # Merge distributed files
│   └── load_balance/
│       ├── io_scheduling.hpp       # I/O scheduling
│       └── bandwidth_sharing.hpp   # Bandwidth management
│
├── streaming/                       # Streaming I/O
│   ├── stream_manager.hpp          # Stream management
│   ├── in_situ/
│   │   ├── catalyst_adapter.hpp    # ParaView Catalyst
│   │   ├── adios_streaming.hpp     # ADIOS streaming
│   │   └── visit_libsim.hpp        # VisIt in-situ
│   ├── socket/
│   │   ├── socket_stream.hpp       # Socket-based I/O
│   │   └── tcp_server.hpp          # TCP server
│   └── compression/
│       ├── compression_filter.hpp  # Compression filters
│       ├── zlib_compression.hpp    # zlib compression
│       └── lz4_compression.hpp     # LZ4 compression
│
├── checkpoint/                      # Checkpoint/restart
│   ├── checkpoint_manager.hpp      # Checkpoint orchestration
│   ├── state_serialization.hpp     # State serialization
│   ├── incremental_checkpoint.hpp  # Incremental saves
│   ├── restart_handler.hpp         # Restart from checkpoint
│   ├── versioning/
│   │   ├── version_control.hpp     # Version management
│   │   └── migration.hpp           # Version migration
│   └── resilience/
│       ├── redundant_checkpoint.hpp # Redundant storage
│       └── recovery.hpp            # Recovery strategies
│
├── metadata/                        # Metadata management
│   ├── simulation_metadata.hpp     # Simulation info
│   ├── provenance.hpp              # Data provenance
│   ├── annotations.hpp             # User annotations
│   ├── time_series.hpp             # Time series metadata
│   └── unit_system.hpp             # Units and dimensions
│
├── converters/                      # Format conversion
│   ├── converter_base.hpp          # Converter interface
│   ├── mesh_converter.hpp          # Mesh format conversion
│   ├── result_converter.hpp        # Result conversion
│   └── batch_converter.hpp         # Batch processing
│
├── validation/                      # I/O validation
│   ├── format_validator.hpp        # Format validation
│   ├── mesh_validator.hpp          # Mesh consistency
│   ├── checksum.hpp                # Data integrity
│   └── schema_validator.hpp        # Schema validation
│
├── performance/                     # Performance optimization
│   ├── buffering/
│   │   ├── buffer_manager.hpp      # Buffer management
│   │   ├── double_buffering.hpp    # Double buffering
│   │   └── async_buffer.hpp        # Asynchronous buffers
│   ├── caching/
│   │   ├── file_cache.hpp          # File caching
│   │   └── metadata_cache.hpp      # Metadata cache
│   └── profiling/
│       ├── io_profiler.hpp         # I/O profiling
│       └── bandwidth_monitor.hpp   # Bandwidth monitoring
│
├── utilities/                       # I/O utilities
│   ├── file_system.hpp             # File system operations
│   ├── path_utils.hpp              # Path manipulation
│   ├── endianness.hpp              # Endian conversion
│   ├── string_parsing.hpp          # String parsing utilities
│   └── progress_reporter.hpp       # Progress reporting
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── format_tests/               # Format-specific tests
    ├── parallel_io/                # Parallel I/O tests
    └── performance/                # I/O benchmarks
```

## Key Components

### 1. Unified I/O Manager
```cpp
// Central I/O orchestration
class IOManager {
    std::unique_ptr<IOFactory> factory;
    IOContext context;
    
public:
    // Automatic format detection
    template<typename Data>
    void write(const std::string& filename, 
               const Data& data,
               const IOOptions& options = {}) {
        // Detect format from extension or options
        auto format = detect_format(filename, options);
        
        // Create appropriate writer
        auto writer = factory->create_writer(format);
        
        // Set context (parallel info, metadata, etc.)
        writer->set_context(context);
        
        // Perform write
        if (context.is_parallel()) {
            write_parallel(writer, data, options);
        } else {
            write_serial(writer, data, options);
        }
    }
    
    template<typename Data>
    Data read(const std::string& filename,
              const IOOptions& options = {}) {
        // Auto-detect format
        auto format = detect_format_from_content(filename);
        
        // Create reader
        auto reader = factory->create_reader(format);
        
        // Read with appropriate strategy
        if (is_partitioned_file(filename)) {
            return read_partitioned<Data>(reader, filename);
        } else {
            return read_monolithic<Data>(reader, filename);
        }
    }
    
private:
    // Parallel write with optimization
    void write_parallel(Writer* writer,
                       const auto& data,
                       const IOOptions& opts) {
        if (opts.use_collective) {
            writer->write_collective(data);
        } else if (opts.use_aggregation) {
            aggregate_and_write(writer, data, opts.aggregators);
        } else {
            writer->write_independent(data);
        }
    }
};
```

### 2. HDF5 Parallel I/O
```cpp
// Parallel HDF5 handler
class ParallelHDF5Handler {
    hid_t file_id;
    hid_t plist_id;
    MPI_Comm comm;
    
    void write_mesh(const DistributedMesh& mesh) {
        // Create file with parallel access
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
        
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                           H5P_DEFAULT, plist_id);
        
        // Write coordinates collectively
        write_coordinates_collective(mesh);
        
        // Write connectivity with hyperslab selection
        write_connectivity_hyperslab(mesh);
        
        // Write fields with compression
        write_fields_compressed(mesh);
        
        H5Fclose(file_id);
    }
    
private:
    void write_coordinates_collective(const DistributedMesh& mesh) {
        // Create dataset
        hsize_t global_dims[2] = {mesh.global_n_nodes(), 3};
        hid_t dataspace = H5Screate_simple(2, global_dims, NULL);
        
        hid_t dataset = H5Dcreate(file_id, "/coordinates",
                                 H5T_NATIVE_DOUBLE, dataspace,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Select hyperslab for this process
        hsize_t offset[2] = {mesh.local_node_offset(), 0};
        hsize_t count[2] = {mesh.local_n_nodes(), 3};
        H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
                           offset, NULL, count, NULL);
        
        // Create memory dataspace
        hid_t memspace = H5Screate_simple(2, count, NULL);
        
        // Collective write
        hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                xfer_plist, mesh.coordinates().data());
        
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Sclose(memspace);
    }
    
    void write_fields_compressed(const DistributedMesh& mesh) {
        // Enable compression for large fields
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 6);  // Compression level 6
        
        // Chunking for compression
        hsize_t chunk_dims[2] = {1024, mesh.n_dofs_per_node()};
        H5Pset_chunk(dcpl, 2, chunk_dims);
        
        // Write field data
        // ...
    }
};
```

### 3. VTK/VTU Writer
```cpp
// Parallel VTU writer with PVTU master file
class ParallelVTUWriter {
    void write(const std::string& basename,
              const DistributedMesh& mesh,
              const Fields& fields,
              int time_step) {
        // Each process writes its piece
        write_piece(basename, mesh, fields, time_step);
        
        // Rank 0 writes PVTU master file
        if (rank == 0) {
            write_pvtu_master(basename, time_step);
        }
    }
    
private:
    void write_piece(const std::string& basename,
                    const DistributedMesh& mesh,
                    const Fields& fields,
                    int time_step) {
        // Generate piece filename
        std::string filename = fmt::format("{}_{:04d}_{:04d}.vtu",
                                         basename, time_step, rank);
        
        XMLWriter xml(filename);
        
        // VTU header
        xml.start_element("VTKFile")
           .attribute("type", "UnstructuredGrid")
           .attribute("version", "1.0")
           .attribute("byte_order", get_endianness());
        
        xml.start_element("UnstructuredGrid");
        
        // Write piece
        xml.start_element("Piece")
           .attribute("NumberOfPoints", mesh.local_n_nodes())
           .attribute("NumberOfCells", mesh.local_n_elements());
        
        // Points
        write_points(xml, mesh);
        
        // Cells
        write_cells(xml, mesh);
        
        // Point data
        xml.start_element("PointData");
        for (auto& field : fields.point_fields()) {
            write_field(xml, field);
        }
        xml.end_element();
        
        // Cell data
        xml.start_element("CellData");
        for (auto& field : fields.cell_fields()) {
            write_field(xml, field);
        }
        xml.end_element();
        
        xml.end_element();  // Piece
        xml.end_element();  // UnstructuredGrid
        xml.end_element();  // VTKFile
    }
    
    void write_pvtu_master(const std::string& basename, int time_step) {
        std::string filename = fmt::format("{}_{:04d}.pvtu",
                                         basename, time_step);
        
        XMLWriter xml(filename);
        
        xml.start_element("VTKFile")
           .attribute("type", "PUnstructuredGrid");
        
        xml.start_element("PUnstructuredGrid")
           .attribute("GhostLevel", "0");
        
        // Declare data arrays
        xml.start_element("PPoints");
        xml.empty_element("PDataArray")
           .attribute("type", "Float64")
           .attribute("NumberOfComponents", "3");
        xml.end_element();
        
        // Reference piece files
        for (int i = 0; i < comm_size; ++i) {
            std::string piece = fmt::format("{}_{:04d}_{:04d}.vtu",
                                          basename, time_step, i);
            xml.empty_element("Piece")
               .attribute("Source", piece);
        }
        
        xml.end_element();
        xml.end_element();
    }
};
```

### 4. Checkpoint/Restart System
```cpp
// Comprehensive checkpoint management
class CheckpointManager {
    struct CheckpointMetadata {
        int version;
        double simulation_time;
        int time_step;
        int num_processes;
        size_t data_size;
        uint64_t checksum;
    };
    
    void checkpoint(const SimulationState& state) {
        // Determine checkpoint strategy
        if (should_incremental()) {
            checkpoint_incremental(state);
        } else {
            checkpoint_full(state);
        }
    }
    
private:
    void checkpoint_full(const SimulationState& state) {
        // Create checkpoint directory
        std::string dir = create_checkpoint_dir();
        
        // Write metadata
        CheckpointMetadata meta{
            .version = CHECKPOINT_VERSION,
            .simulation_time = state.time,
            .time_step = state.step,
            .num_processes = comm_size,
            .data_size = state.size()
        };
        
        if (rank == 0) {
            write_metadata(dir + "/metadata.json", meta);
        }
        
        // Parallel write of state data
        ParallelHDF5Handler h5;
        h5.open(dir + "/checkpoint.h5", comm);
        
        // Write mesh
        h5.write_group("/mesh");
        h5.write_dataset("/mesh/coordinates", state.mesh.coordinates());
        h5.write_dataset("/mesh/connectivity", state.mesh.connectivity());
        
        // Write solution fields
        h5.write_group("/solution");
        for (auto& [name, field] : state.fields) {
            h5.write_dataset("/solution/" + name, field);
        }
        
        // Write history variables
        h5.write_group("/history");
        h5.write_dataset("/history/material_state", 
                        state.material_history);
        
        h5.close();
        
        // Compute and verify checksum
        meta.checksum = compute_checksum(dir);
        verify_checkpoint(dir, meta);
    }
    
    void checkpoint_incremental(const SimulationState& state) {
        // Only save changed data
        auto changes = detect_changes(state, last_checkpoint);
        
        std::string dir = get_current_checkpoint_dir();
        
        // Write delta file
        HDF5Handler h5;
        h5.open(dir + fmt::format("/delta_{}.h5", state.step));
        
        for (auto& change : changes) {
            h5.write_dataset(change.path, change.data);
        }
        
        // Update metadata
        update_incremental_metadata(dir, state.step);
    }
    
    void restart(SimulationState& state, const std::string& checkpoint) {
        // Read metadata
        auto meta = read_metadata(checkpoint + "/metadata.json");
        
        // Handle different number of processes
        if (comm_size != meta.num_processes) {
            restart_with_redistribution(state, checkpoint, meta);
        } else {
            restart_direct(state, checkpoint, meta);
        }
    }
};
```

### 5. In-Situ Visualization
```cpp
// ParaView Catalyst integration
class CatalystAdapter {
    vtkCPProcessor* processor;
    vtkCPDataDescription* dataDescription;
    
    void initialize(const std::string& script) {
        processor = vtkCPProcessor::New();
        processor->Initialize();
        
        // Add python script
        vtkCPPythonScriptPipeline* pipeline = 
            vtkCPPythonScriptPipeline::New();
        pipeline->Initialize(script.c_str());
        processor->AddPipeline(pipeline);
    }
    
    void coprocess(const SimulationState& state, double time) {
        // Create data description
        dataDescription = vtkCPDataDescription::New();
        dataDescription->SetTimeData(time, state.time_step);
        
        // Check if processing needed
        if (!processor->RequestDataDescription(dataDescription)) {
            return;
        }
        
        // Convert to VTK data structures
        vtkUnstructuredGrid* grid = create_vtk_grid(state.mesh);
        
        // Add fields
        for (auto& [name, field] : state.fields) {
            add_field_to_grid(grid, name, field);
        }
        
        // Add to coprocessor
        dataDescription->GetInputDescription(0)->SetGrid(grid);
        
        // Execute pipeline
        processor->CoProcess(dataDescription);
        
        dataDescription->Delete();
    }
};
```

### 6. Format Converter
```cpp
// Multi-format converter
class FormatConverter {
    void convert(const std::string& input,
                const std::string& output,
                const ConversionOptions& options) {
        // Detect formats
        auto input_format = detect_format(input);
        auto output_format = detect_format(output);
        
        // Create reader and writer
        auto reader = create_reader(input_format);
        auto writer = create_writer(output_format);
        
        // Read mesh
        auto mesh = reader->read_mesh(input);
        
        // Convert if needed
        if (options.transform_coordinates) {
            transform_coordinates(mesh, options.transformation);
        }
        
        if (options.renumber_nodes) {
            renumber_for_bandwidth(mesh);
        }
        
        // Read and convert fields
        if (reader->has_fields()) {
            auto fields = reader->read_fields(input);
            
            // Interpolate if mesh changed
            if (options.interpolate_fields) {
                fields = interpolate_fields(fields, mesh);
            }
            
            writer->write_fields(output, fields);
        }
        
        // Write output
        writer->write_mesh(output, mesh);
    }
};
```

### 7. Streaming I/O
```cpp
// Real-time streaming output
class StreamingOutput {
    std::unique_ptr<StreamProtocol> protocol;
    CircularBuffer buffer;
    std::thread streaming_thread;
    
    void start_streaming(const std::string& endpoint) {
        // Setup connection
        if (endpoint.starts_with("tcp://")) {
            protocol = std::make_unique<TCPStream>(endpoint);
        } else if (endpoint.starts_with("adios://")) {
            protocol = std::make_unique<ADIOSStream>(endpoint);
        }
        
        protocol->connect();
        
        // Start streaming thread
        streaming_thread = std::thread([this] {
            stream_loop();
        });
    }
    
    void send_frame(const SimulationState& state) {
        // Serialize state
        auto data = serialize_state(state);
        
        // Add to buffer (non-blocking)
        if (!buffer.try_push(data)) {
            // Buffer full, drop frame or wait
            handle_buffer_overflow();
        }
    }
    
private:
    void stream_loop() {
        while (streaming) {
            if (auto data = buffer.pop()) {
                // Compress if requested
                if (compression_enabled) {
                    data = compress(data);
                }
                
                // Send over protocol
                protocol->send(data);
            }
        }
    }
};
```

## Performance Optimizations

### Asynchronous I/O
```cpp
class AsyncIOManager {
    void write_async(const std::string& filename,
                    const Data& data) {
        // Double buffering
        auto& buffer = get_next_buffer();
        buffer.copy_from(data);
        
        // Start async write
        auto future = std::async(std::launch::async, [=] {
            writer->write(filename, buffer);
        });
        
        pending_operations.push_back(std::move(future));
    }
};
```

### I/O Aggregation
```cpp
class IOAggregator {
    void aggregate_write(const DistributedData& data) {
        // Select aggregator processes
        int aggregators = select_optimal_aggregators();
        
        // Gather to aggregators
        if (is_aggregator()) {
            auto gathered = gather_from_clients(data);
            write_aggregated(gathered);
        } else {
            send_to_aggregator(data);
        }
    }
};
```

## Integration Points

### With mesh/
- Reads/writes mesh formats
- Handles mesh metadata
- Supports partitioned meshes

### With solvers/
- Checkpoint solver state
- Export solution fields
- Support restart

### With visualization/
- Export to visualization formats
- In-situ visualization support
- Time series output

### With parallel/
- Parallel I/O operations
- Distributed file handling
- Collective operations

## Success Metrics

1. **I/O Bandwidth**: > 80% of hardware maximum
2. **Parallel Efficiency**: > 90% weak scaling
3. **Checkpoint Overhead**: < 5% of computation time
4. **Format Support**: > 15 common formats
5. **Metadata Preservation**: 100% fidelity
6. **Compression Ratio**: > 3:1 for fields

## Key Features

1. **Multi-Format**: Support for all major FEM formats
2. **Parallel I/O**: Scalable distributed I/O
3. **In-Situ**: Real-time visualization support
4. **Checkpoint/Restart**: Comprehensive resilience
5. **Streaming**: Real-time data streaming
6. **Metadata Rich**: Complete provenance tracking

This architecture provides comprehensive I/O capabilities for all aspects of FEM simulation, from mesh input through solution output, with careful attention to performance and scalability.
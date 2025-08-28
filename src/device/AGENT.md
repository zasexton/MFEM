# AGENT.md - Device Abstraction Layer

## Mission
Provide a unified abstraction layer for heterogeneous computing, enabling seamless execution on CPUs, GPUs, and accelerators while managing memory, kernels, and execution policies across different hardware backends.

## Architecture Philosophy
- **Backend-Agnostic**: Single code path for multiple hardware targets
- **Zero-Cost Abstraction**: No overhead for device abstraction
- **Memory-Unified**: Automatic memory management across devices
- **Kernel-Portable**: Write once, run on any backend
- **Performance-First**: Optimal code generation per device

## Directory Structure

```
device/
├── README.md                         # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── core/                            # Core device abstractions
│   ├── device_base.hpp             # Base device interface
│   ├── device_traits.hpp           # Device capability traits
│   ├── device_config.hpp           # Device configuration
│   ├── device_registry.hpp         # Device registration system
│   ├── execution_space.hpp         # Execution space abstraction
│   ├── memory_space.hpp            # Memory space abstraction
│   ├── device_selector.hpp         # Automatic device selection
│   └── device_error.hpp            # Error handling
│
├── backend/                         # Backend implementations
│   ├── backend_base.hpp            # Backend interface
│   ├── cpu/
│   │   ├── cpu_backend.hpp         # CPU backend
│   │   ├── cpu_executor.hpp        # CPU execution
│   │   ├── simd_kernels.hpp        # SIMD optimizations
│   │   └── openmp_executor.hpp     # OpenMP parallelism
│   ├── cuda/
│   │   ├── cuda_backend.hpp        # CUDA backend
│   │   ├── cuda_executor.hpp       # CUDA kernel launch
│   │   ├── cuda_memory.hpp         # CUDA memory management
│   │   ├── cuda_stream.hpp         # Stream management
│   │   └── cuda_error.hpp          # CUDA error handling
│   ├── hip/
│   │   ├── hip_backend.hpp         # AMD HIP backend
│   │   ├── hip_executor.hpp        # HIP execution
│   │   └── hip_memory.hpp          # HIP memory
│   ├── sycl/
│   │   ├── sycl_backend.hpp        # Intel SYCL backend
│   │   ├── sycl_executor.hpp       # SYCL execution
│   │   └── sycl_queue.hpp          # SYCL queue management
│   ├── opencl/
│   │   ├── opencl_backend.hpp      # OpenCL backend
│   │   ├── opencl_kernel.hpp       # OpenCL kernels
│   │   └── opencl_context.hpp      # Context management
│   └── metal/
│       ├── metal_backend.hpp       # Apple Metal backend
│       └── metal_executor.hpp      # Metal execution
│
├── memory/                          # Memory management
│   ├── memory_manager.hpp          # Unified memory manager
│   ├── device_array.hpp            # Device array abstraction
│   ├── device_buffer.hpp           # Raw buffer management
│   ├── memory_pool.hpp             # Device memory pools
│   ├── pinned_memory.hpp           # Pinned host memory
│   ├── unified_memory.hpp          # Unified memory (UVM)
│   ├── memory_transfer.hpp         # Host-device transfers
│   ├── memory_view.hpp             # Non-owning views
│   └── memory_traits.hpp           # Memory access traits
│
├── kernel/                          # Kernel abstractions
│   ├── kernel_base.hpp             # Kernel interface
│   ├── kernel_launcher.hpp         # Kernel launch interface
│   ├── kernel_config.hpp           # Launch configuration
│   ├── parallel_for.hpp            # Parallel loop abstraction
│   ├── parallel_reduce.hpp         # Parallel reduction
│   ├── parallel_scan.hpp           # Parallel scan/prefix
│   ├── team_policy.hpp             # Hierarchical parallelism
│   ├── kernel_fusion.hpp           # Kernel fusion optimization
│   └── jit_compiler.hpp            # JIT compilation support
│
├── executor/                        # Execution policies
│   ├── executor_base.hpp           # Executor interface
│   ├── execution_policy.hpp        # Execution policies
│   ├── synchronous_executor.hpp    # Synchronous execution
│   ├── asynchronous_executor.hpp   # Async execution
│   ├── stream_executor.hpp         # Stream-based execution
│   ├── graph_executor.hpp          # Task graph execution
│   ├── dynamic_executor.hpp        # Dynamic scheduling
│   └── work_queue.hpp              # Work queue abstraction
│
├── synchronization/                 # Synchronization primitives
│   ├── device_event.hpp            # Event abstraction
│   ├── device_barrier.hpp          # Barrier synchronization
│   ├── device_mutex.hpp            # Device-side mutex
│   ├── atomic_operations.hpp       # Atomic operations
│   ├── fence.hpp                   # Memory fences
│   └── stream_sync.hpp             # Stream synchronization
│
├── algorithms/                      # Device algorithms
│   ├── algorithm_base.hpp          # Algorithm interface
│   ├── sort.hpp                    # Parallel sorting
│   ├── scan.hpp                    # Prefix scan
│   ├── reduce.hpp                  # Reductions
│   ├── transform.hpp               # Transform operations
│   ├── partition.hpp               # Partitioning
│   ├── merge.hpp                   # Merging
│   └── radix_sort.hpp              # Radix sort
│
├── math/                           # Device math operations
│   ├── blas_device.hpp             # Device BLAS operations
│   ├── lapack_device.hpp           # Device LAPACK
│   ├── sparse_device.hpp           # Sparse operations
│   ├── fft_device.hpp              # FFT on device
│   ├── random_device.hpp           # Random number generation
│   └── special_functions.hpp       # Special math functions
│
├── profiling/                       # Performance profiling
│   ├── profiler_base.hpp           # Profiler interface
│   ├── timer.hpp                   # Device timers
│   ├── event_trace.hpp             # Event tracing
│   ├── performance_counters.hpp    # Hardware counters
│   ├── memory_profiler.hpp         # Memory usage tracking
│   └── kernel_profiler.hpp         # Kernel performance
│
├── interop/                         # Interoperability
│   ├── cuda_interop.hpp            # CUDA interop
│   ├── opencl_interop.hpp          # OpenCL interop
│   ├── vulkan_interop.hpp          # Vulkan interop
│   ├── opengl_interop.hpp          # OpenGL interop
│   └── external_memory.hpp         # External memory import
│
├── codegen/                         # Code generation
│   ├── kernel_generator.hpp        # Kernel code generation
│   ├── template_engine.hpp         # Template-based codegen
│   ├── ast_builder.hpp             # AST construction
│   ├── optimization_passes.hpp     # Code optimization
│   └── backend_lowering.hpp        # Backend-specific lowering
│
├── utilities/                       # Device utilities
│   ├── device_info.hpp             # Device query utilities
│   ├── device_selector.hpp         # Automatic selection
│   ├── capability_check.hpp        # Capability checking
│   ├── debugging.hpp               # Debug utilities
│   └── error_checking.hpp          # Error check macros
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── performance/                # Performance tests
    ├── multi_device/               # Multi-device tests
    └── backends/                   # Backend-specific tests
```

## Key Components

### 1. Unified Device Interface
```cpp
// Device abstraction for any backend
class Device {
    std::unique_ptr<Backend> backend;
    MemoryManager memory_manager;
    ExecutionSpace exec_space;
    
public:
    // Device selection
    static Device select_device(DeviceType type = DeviceType::Auto) {
        if (type == DeviceType::Auto) {
            // Automatic selection based on problem size/type
            if (cuda_available() && problem_size > threshold) {
                return Device(CUDABackend());
            }
            return Device(CPUBackend());
        }
        return Device(create_backend(type));
    }
    
    // Memory allocation
    template<typename T>
    DeviceArray<T> allocate(size_t n) {
        return memory_manager.allocate<T>(n);
    }
    
    // Kernel execution
    template<typename Kernel, typename... Args>
    void execute(const LaunchConfig& config, 
                Kernel kernel, Args&&... args) {
        backend->launch(config, kernel, std::forward<Args>(args)...);
    }
    
    // Synchronization
    void synchronize() {
        backend->synchronize();
    }
};
```

### 2. Unified Memory Management
```cpp
// Device array with automatic memory management
template<typename T>
class DeviceArray {
    Device* device;
    T* device_ptr = nullptr;
    T* host_ptr = nullptr;
    size_t size_;
    bool host_dirty = false;
    bool device_dirty = false;
    
public:
    // Automatic synchronization
    T* data() {
        sync_to_device();
        device_dirty = true;
        return device_ptr;
    }
    
    const T* host_data() {
        sync_to_host();
        host_dirty = true;
        return host_ptr;
    }
    
    // Lazy synchronization
    void sync_to_device() {
        if (host_dirty && device_ptr) {
            device->copy_to_device(host_ptr, device_ptr, size_);
            host_dirty = false;
        }
    }
    
    void sync_to_host() {
        if (device_dirty && host_ptr) {
            device->copy_to_host(device_ptr, host_ptr, size_);
            device_dirty = false;
        }
    }
    
    // View for kernel arguments
    DeviceView<T> view() {
        sync_to_device();
        return DeviceView<T>(device_ptr, size_);
    }
};
```

### 3. Portable Kernel Abstraction
```cpp
// Write-once kernel for any backend
template<typename T>
struct VectorAddKernel {
    // Portable kernel implementation
    DEVICE_FUNCTION
    void operator()(int i, const T* a, const T* b, T* c) const {
        c[i] = a[i] + b[i];
    }
    
    // Metadata for optimization
    static constexpr bool is_simple = true;
    static constexpr int vector_width = 4;
};

// Usage - same code for CPU/GPU
template<typename Device>
void vector_add(Device& device, 
                const DeviceArray<double>& a,
                const DeviceArray<double>& b,
                DeviceArray<double>& c) {
    int n = a.size();
    
    // Automatic backend selection
    device.parallel_for(n, VectorAddKernel<double>{}, 
                       a.view(), b.view(), c.view());
}
```

### 4. Backend Implementations
```cpp
// CUDA backend implementation
class CUDABackend : public Backend {
    cudaStream_t stream;
    
    template<typename Kernel, typename... Args>
    void launch(const LaunchConfig& config,
               Kernel kernel, Args... args) override {
        // Calculate launch configuration
        int threads = 256;
        int blocks = (config.n + threads - 1) / threads;
        
        // Launch CUDA kernel
        cuda_kernel_wrapper<<<blocks, threads, 0, stream>>>(
            kernel, args...
        );
        
        check_cuda_error();
    }
    
    void* allocate(size_t bytes) override {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
};

// CPU backend with SIMD
class CPUBackend : public Backend {
    template<typename Kernel, typename... Args>
    void launch(const LaunchConfig& config,
               Kernel kernel, Args... args) override {
        #pragma omp parallel for simd
        for (int i = 0; i < config.n; ++i) {
            kernel(i, args...);
        }
    }
};
```

### 5. Hierarchical Parallelism
```cpp
// Team-based parallelism (like Kokkos)
template<typename Device>
class TeamPolicy {
    int league_size;  // Number of teams
    int team_size;    // Threads per team
    
    template<typename Kernel>
    void execute(Device& device, Kernel kernel) {
        device.parallel_for_teams(
            league_size, team_size,
            [=] DEVICE_LAMBDA (const TeamMember& team) {
                // Team-level operations
                kernel(team);
                
                // Nested parallelism
                team.parallel_for(team.size(), 
                    [&](int i) {
                        // Thread-level work
                    });
                
                // Team synchronization
                team.barrier();
            }
        );
    }
};
```

### 6. Memory Pool Management
```cpp
// Device memory pool for efficient allocation
template<typename Device>
class DeviceMemoryPool {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    Device* device;
    std::vector<Block> blocks;
    size_t total_allocated = 0;
    
    void* allocate(size_t bytes) {
        // Try to find existing block
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= bytes) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        size_t alloc_size = std::max(bytes, min_block_size);
        void* ptr = device->allocate_raw(alloc_size);
        blocks.push_back({ptr, alloc_size, true});
        total_allocated += alloc_size;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
};
```

### 7. Kernel Fusion
```cpp
// Fuse multiple operations for efficiency
template<typename Device>
class KernelFusion {
    // Fused AXPY + norm kernel
    struct FusedAxpyNorm {
        double alpha;
        
        DEVICE_FUNCTION
        double operator()(int i, const double* x, 
                         double* y) const {
            y[i] += alpha * x[i];
            return y[i] * y[i];  // Return for reduction
        }
    };
    
    double axpy_and_norm(Device& device,
                         double alpha,
                         const DeviceArray<double>& x,
                         DeviceArray<double>& y) {
        // Single kernel does AXPY and computes norm
        return device.parallel_reduce(
            x.size(),
            FusedAxpyNorm{alpha},
            x.view(), y.view(),
            0.0,
            [] DEVICE_LAMBDA (double a, double b) { 
                return a + b; 
            }
        );
    }
};
```

### 8. Multi-Device Support
```cpp
// Manage multiple devices
class MultiDevice {
    std::vector<Device> devices;
    
    // Distribute work across devices
    template<typename Work>
    void distribute_work(const std::vector<Work>& work_items) {
        int n_devices = devices.size();
        std::vector<std::future<void>> futures;
        
        for (int d = 0; d < n_devices; ++d) {
            futures.push_back(
                std::async([&, d] {
                    // Set device context
                    devices[d].make_current();
                    
                    // Process assigned work
                    int start = d * work_items.size() / n_devices;
                    int end = (d + 1) * work_items.size() / n_devices;
                    
                    for (int i = start; i < end; ++i) {
                        process(devices[d], work_items[i]);
                    }
                })
            );
        }
        
        // Wait for all devices
        for (auto& f : futures) {
            f.wait();
        }
    }
};
```

## Performance Optimizations

### Automatic Kernel Tuning
```cpp
class KernelTuner {
    struct TuningParameters {
        int block_size;
        int items_per_thread;
        bool use_shared_memory;
    };
    
    TuningParameters auto_tune(const Kernel& kernel,
                              const ProblemSize& size) {
        std::vector<TuningParameters> candidates = 
            generate_candidates(kernel, size);
        
        TuningParameters best;
        double best_time = std::numeric_limits<double>::max();
        
        for (auto& params : candidates) {
            double time = benchmark_kernel(kernel, params);
            if (time < best_time) {
                best_time = time;
                best = params;
            }
        }
        
        return best;
    }
};
```

### Memory Transfer Optimization
```cpp
// Overlapped transfers and computation
class PipelinedExecution {
    void execute_pipelined(Device& device,
                          const std::vector<WorkItem>& items) {
        auto stream1 = device.create_stream();
        auto stream2 = device.create_stream();
        
        for (int i = 0; i < items.size(); ++i) {
            auto& current_stream = (i % 2) ? stream1 : stream2;
            auto& other_stream = (i % 2) ? stream2 : stream1;
            
            // Copy input to device
            current_stream.copy_async(items[i].input);
            
            // Wait for previous computation
            if (i > 0) {
                other_stream.synchronize();
            }
            
            // Launch kernel
            current_stream.launch(process_kernel, items[i]);
            
            // Copy result back
            current_stream.copy_async(items[i].output);
        }
        
        // Wait for all operations
        stream1.synchronize();
        stream2.synchronize();
    }
};
```

## Integration Points

### With numeric/
- Provides device-accelerated math operations
- Manages device memory for matrices/vectors
- Implements device BLAS/LAPACK

### With assembly/
- GPU-accelerated element assembly
- Device-side sparsity pattern construction
- Parallel constraint application

### With solvers/
- Device-accelerated linear solvers
- GPU preconditioners
- Distributed multi-GPU solvers

## Success Metrics

1. **Abstraction Overhead**: < 1% vs native backend code
2. **Memory Transfer**: Overlap with computation > 90%
3. **Kernel Launch**: < 10μs overhead
4. **Auto-tuning**: Within 95% of hand-tuned performance
5. **Multi-device Scaling**: > 90% efficiency
6. **Memory Pool**: < 5% fragmentation

## Key Features

1. **Write-Once**: Single kernel code for all backends
2. **Automatic Memory**: Transparent host-device synchronization
3. **Performance Portable**: Optimal code generation per device
4. **Multi-Device**: Seamless multi-GPU/accelerator support
5. **Interoperable**: Works with existing CUDA/OpenCL code
6. **Auto-Tuning**: Automatic performance optimization

This architecture provides a complete abstraction layer for heterogeneous computing, enabling FEM codes to run efficiently on any available hardware without code modification.
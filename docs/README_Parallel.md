# Parallel Computing in MFEMSolver

This project supports three complementary parallelism models that can be used individually or in combination for optimal performance on modern HPC systems.

## Parallelism Models

### 1. **OpenMP** (Shared Memory - Single Node)
- **Purpose**: Thread-based parallelism within a single compute node
- **Best for**: Loop-level parallelism, SIMD operations, local matrix operations
- **Threads**: Typically 1-128 threads per node
- **Configuration**: `ENABLE_OPENMP=ON`

### 2. **MPI** (Distributed Memory - Multiple Nodes)
- **Purpose**: Process-based parallelism across multiple compute nodes
- **Best for**: Domain decomposition, distributed matrices, cluster computing
- **Processes**: 1 to thousands of processes
- **Configuration**: `ENABLE_MPI=ON`

### 3. **Intel TBB** (Task-Based - Single Node)
- **Purpose**: Task-based parallelism with work stealing
- **Best for**: Irregular parallelism, nested parallelism, concurrent data structures
- **Tasks**: Dynamic task scheduling
- **Configuration**: `TBB_FETCH=ON`

## Build Configuration

### Basic Build
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENMP=ON \
    -DENABLE_MPI=ON \
    -DTBB_FETCH=ON
make -j
```

### OpenMP-Only Build (Single Node)
```bash
cmake .. -DENABLE_OPENMP=ON -DENABLE_MPI=OFF
```

### MPI-Only Build (Multi-Node)
```bash
cmake .. -DENABLE_MPI=ON -DENABLE_OPENMP=OFF
```

### Hybrid MPI+OpenMP Build (Recommended for HPC)
```bash
cmake .. \
    -DENABLE_MPI=ON \
    -DENABLE_OPENMP=ON \
    -DENABLE_HYBRID_PARALLEL=ON
```

### Advanced Options

#### Custom OpenMP Configuration
```bash
cmake .. \
    -DENABLE_OPENMP=ON \
    -DOPENMP_RUNTIME=libomp \           # Force specific runtime
    -DOPENMP_NUM_THREADS_DEFAULT=8 \    # Default thread count
    -DFETCH_OPENMP=ON                   # Build OpenMP if not found
```

#### Custom MPI Configuration
```bash
cmake .. \
    -DENABLE_MPI=ON \
    -DBUNDLED_MPI=ON \        # Build our own MPI
    -DMPI_VENDOR=openmpi      # or mpich
```

#### TBB with Custom Memory Allocator
```bash
cmake .. \
    -DTBB_FETCH=ON \
    -DTBB_USE_MALLOC=ON       # Use TBB's scalable allocator
```

#### Suppressing Atomic Assembly Warning
When parallel assembly runs without atomic updates the library emits a compile-time
note to remind you that external synchronization is required. If you deliberately
handle synchronization yourself, this note can be silenced:
```bash
cmake .. -DFEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING=ON
```
Be aware that parallel assembly without atomics risks data races unless your code
uses proper synchronization mechanisms.

## Runtime Configuration

### OpenMP Thread Control
```bash
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./your_program
```

### MPI Process Launch
```bash
# Single node, 4 processes
mpirun -np 4 ./your_program

# Multiple nodes (SLURM)
srun -N 4 -n 16 ./your_program

# With hostfile
mpirun -np 8 -hostfile hosts.txt ./your_program
```

### Hybrid MPI+OpenMP Execution
```bash
# 4 MPI ranks, 8 OpenMP threads each
export OMP_NUM_THREADS=8
mpirun -np 4 ./your_program

# SLURM submission
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=16
srun ./your_program
```

## Performance Guidelines

### When to Use Each Model

| Scenario | Recommended | Configuration |
|----------|-------------|---------------|
| Desktop/Workstation | OpenMP or TBB | `-DENABLE_OPENMP=ON -DENABLE_MPI=OFF` |
| Single HPC Node | OpenMP + TBB | `-DENABLE_OPENMP=ON -DTBB_FETCH=ON` |
| HPC Cluster | MPI + OpenMP | `-DENABLE_MPI=ON -DENABLE_OPENMP=ON` |
| Irregular Workloads | TBB | `-DTBB_FETCH=ON` |
| GPU Offloading | OpenMP | `-DENABLE_OPENMP=ON` (with target directives) |

### Avoiding Oversubscription

In hybrid parallel mode, ensure total threads don't exceed available cores:

```bash
# Example: 2 nodes, 32 cores each
# BAD: 4 MPI × 32 OpenMP = 128 threads on 64 cores
# GOOD: 4 MPI × 8 OpenMP = 32 threads per node

# Per-node configuration
CORES_PER_NODE=32
MPI_PER_NODE=4
OMP_THREADS=$((CORES_PER_NODE / MPI_PER_NODE))
export OMP_NUM_THREADS=$OMP_THREADS
```

## Code Examples

### Using OpenMP in Your Code
```cpp
#ifdef ENABLE_OPENMP
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // Parallel work
}
#endif
```

### Using MPI in Your Code
```cpp
#ifdef ENABLE_MPI
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
// Distributed work
#endif
```

### Using TBB in Your Code
```cpp
#ifdef ENABLE_TBB
tbb::parallel_for(0, n, [](int i) {
    // Parallel work
});
#endif
```

### Hybrid Example
```cpp
#ifdef ENABLE_MPI
// Domain decomposition across MPI ranks
int rank = get_mpi_rank();
auto local_domain = partition_domain(rank);
#endif

#ifdef ENABLE_OPENMP
// Thread parallelism within each domain
#pragma omp parallel for
for (auto& element : local_domain) {
    process_element(element);
}
#endif
```

## Troubleshooting

### OpenMP Not Found
- **Linux**: Install `libomp-dev` or `libgomp-dev`
- **macOS**: `brew install libomp`
- **Fallback**: Use `-DFETCH_OPENMP=ON` to build from source

### MPI Not Found
- **Linux**: Install `libopenmpi-dev` or `mpich`
- **macOS**: `brew install open-mpi`
- **Fallback**: Use `-DBUNDLED_MPI=ON` to build from source

### Thread Affinity Issues
```bash
# OpenMP affinity
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# MPI process binding
mpirun --bind-to core --map-by socket ./program
```

### Performance Profiling
```bash
# OpenMP profiling
export OMP_DISPLAY_ENV=TRUE
export OMP_DISPLAY_AFFINITY=TRUE

# MPI profiling
mpirun -np 4 --report-bindings ./program

# Combined
export OMP_NUM_THREADS=4
mpirun -np 2 --report-bindings --bind-to socket ./program
```

## CMake Targets

After configuration, the following targets are available:
- `openmp::openmp` - OpenMP support
- `mpi::bundled` - MPI support
- `TBB::tbb` - Intel TBB support
- `femsolver_parallel` - Combined parallel support

Link your targets:
```cmake
target_link_libraries(my_solver PRIVATE femsolver_parallel)
```

## Testing

Run parallel tests:
```bash
# All tests
ctest

# Specific parallel tests
ctest -R OpenMP
ctest -R MPI
ctest -R Hybrid

# With verbose output
ctest -V -R Parallel
```

## References

- [OpenMP Specification](https://www.openmp.org/specifications/)
- [MPI Standard](https://www.mpi-forum.org/docs/)
- [Intel TBB Documentation](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/current/overview.html)
- [Hybrid MPI+OpenMP Best Practices](https://www.mcs.anl.gov/~itf/dbpp/text/node94.html)
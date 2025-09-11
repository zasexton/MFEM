# Core Library Benchmarks

## Structure

The benchmark folder mirrors the `core/` source folder structure:

```
benchmarks/
├── CMakeLists.txt           # Benchmark build configuration
├── benchmark_main.cpp       # Optional custom benchmark runner
├── base/                    # Benchmarks for core/base/
│   ├── object_benchmark.cpp
│   ├── factory_benchmark.cpp
│   └── ...
├── memory/                  # Benchmarks for core/memory/
│   ├── arena_benchmark.cpp
│   ├── pool_benchmark.cpp
│   └── ...
├── concurrency/            # Benchmarks for core/concurrency/
│   ├── threadpool_benchmark.cpp
│   └── ...
└── ...                     # Other module benchmarks
```

## Quick Start

### Build and Run
```bash
# Configure with benchmarks enabled
cmake .. -DBUILD_PERFORMANCE_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# Build benchmarks
make core_benchmarks

# Run all benchmarks
make benchmark_core

# Run quick benchmarks (less time)
make benchmark_core_quick

# Run detailed benchmarks with statistics  
make benchmark_core_detailed
```

### Run Specific Benchmarks
```bash
# Run benchmarks for specific module
make benchmark_core_base      # Only base module
make benchmark_core_memory    # Only memory module

# Or use filter directly
./core_benchmarks --benchmark_filter="Object"
./core_benchmarks --benchmark_filter="BM_ObjectCreation"
./core_benchmarks --benchmark_filter=".*Memory.*"
```

## Output Formats

### Console Output (Default)
```bash
./core_benchmarks
```

### JSON Output
```bash
# Save to file
./core_benchmarks --benchmark_format=json --benchmark_out=results.json

# Or use make target
make benchmark_core_json
```

### CSV Output
```bash
./core_benchmarks --benchmark_format=csv --benchmark_out=results.csv
```

## Benchmark Options

### Time Control
```bash
# Set minimum time per benchmark
./core_benchmarks --benchmark_min_time=2.0  # 2 seconds minimum

# Set specific number of iterations
./core_benchmarks --benchmark_iterations=1000
```

### Statistical Analysis
```bash
# Run multiple repetitions for statistics
./core_benchmarks --benchmark_repetitions=10

# Show only aggregate statistics
./core_benchmarks --benchmark_report_aggregates_only=true

# Show both individual and aggregate
./core_benchmarks --benchmark_display_aggregates_only=false
```

### Time Units
```bash
# Change time unit display
./core_benchmarks --benchmark_time_unit=ns  # nanoseconds
./core_benchmarks --benchmark_time_unit=us  # microseconds
./core_benchmarks --benchmark_time_unit=ms  # milliseconds
```

## Writing Benchmarks

### Basic Benchmark
```cpp
#include <benchmark/benchmark.h>
#include "core/base/object.h"

using namespace fem::core::base;

// Simple benchmark
static void BM_ObjectCreation(benchmark::State& state) {
    for (auto _ : state) {
        Object obj("Test");
        benchmark::DoNotOptimize(obj.id());  // Prevent optimization
    }
}
BENCHMARK(BM_ObjectCreation);
```

### Benchmark with Arguments
```cpp
static void BM_VectorReserve(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> v;
        v.reserve(state.range(0));
        benchmark::DoNotOptimize(v.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_VectorReserve)->Range(8, 8<<10)->Complexity();
```

### Custom Counters
```cpp
static void BM_DataProcessing(benchmark::State& state) {
    size_t bytes_processed = 0;
    for (auto _ : state) {
        auto data = ProcessData(state.range(0));
        bytes_processed += data.size();
    }
    
    state.SetBytesProcessed(bytes_processed);
    state.SetItemsProcessed(state.iterations());
    
    // Custom counters
    state.counters["BytesPerOp"] = bytes_processed / state.iterations();
    state.counters["OpsPerSec"] = benchmark::Counter(
        state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_DataProcessing)->Arg(1024)->Arg(8192);
```

### Multi-threaded Benchmark
```cpp
static void BM_ConcurrentAccess(benchmark::State& state) {
    static SharedResource resource;
    
    for (auto _ : state) {
        resource.access();
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ConcurrentAccess)->ThreadRange(1, 8);
```

### Setup and Teardown
```cpp
static void BM_ComplexOperation(benchmark::State& state) {
    // Setup (not timed)
    auto data = GenerateTestData(state.range(0));
    
    for (auto _ : state) {
        // Only this is timed
        auto result = ProcessData(data);
        benchmark::DoNotOptimize(result);
    }
    
    // Teardown (not timed)
    // Happens automatically
}
BENCHMARK(BM_ComplexOperation)->Range(100, 10000);
```

### Manual Timing
```cpp
static void BM_FileIO(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();  // Pause for setup
        auto filename = GenerateTempFile();
        state.ResumeTiming();  // Resume for measurement
        
        WriteFile(filename, state.range(0));
        
        state.PauseTiming();  // Pause for cleanup
        DeleteFile(filename);
        state.ResumeTiming();
    }
}
BENCHMARK(BM_FileIO)->Range(1<<10, 1<<20);
```

## Analyzing Results

### Compare Benchmarks
```bash
# Run baseline
./core_benchmarks --benchmark_out=baseline.json --benchmark_out_format=json

# Make changes and run again
./core_benchmarks --benchmark_out=new.json --benchmark_out_format=json

# Compare (requires benchmark tools)
compare.py baseline.json new.json
```

### Complexity Analysis
```cpp
// Add complexity analysis to benchmark
static void BM_Algorithm(benchmark::State& state) {
    std::vector<int> data(state.range(0));
    for (auto _ : state) {
        SortAlgorithm(data);
    }
    state.SetComplexityN(state.range(0));
}
// Specify expected complexity
BENCHMARK(BM_Algorithm)->RangeMultiplier(2)->Range(8, 8<<10)->Complexity(benchmark::oN);
// Or let it infer
BENCHMARK(BM_Algorithm)->RangeMultiplier(2)->Range(8, 8<<10)->Complexity();
```

## Performance Tips

### System Preparation
```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set -g performance

# Set CPU affinity
taskset -c 0 ./core_benchmarks

# Increase process priority
nice -n -20 ./core_benchmarks
```

### Benchmark Best Practices
1. **Use Release mode**: Always benchmark in Release/RelWithDebInfo
2. **Warm up**: Benchmarks do automatic warm-up
3. **Prevent optimization**: Use `benchmark::DoNotOptimize()`
4. **Multiple runs**: Use `--benchmark_repetitions` for stability
5. **Isolate system**: Close other applications
6. **Check variance**: High variance indicates interference

### Common Pitfalls
- Benchmarking Debug builds (too slow)
- Not preventing compiler optimizations
- Including setup/teardown in measurements
- Too short benchmark duration
- System interference (background processes)

## Continuous Benchmarking

### Integration with CI
```yaml
# Example GitHub Actions
- name: Run Benchmarks
  run: |
    make core_benchmarks
    ./core_benchmarks --benchmark_format=json --benchmark_out=results.json
    
- name: Store Benchmark Result
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'googlecpp'
    output-file-path: results.json
```

### Track Performance Over Time
```bash
# Save results with timestamp
./core_benchmarks --benchmark_out=$(date +%Y%m%d_%H%M%S).json --benchmark_out_format=json

# Create performance dashboard
# Use tools like Google Benchmark's compare.py or custom scripts
```

## Profiling with Benchmarks

### Using perf (Linux)
```bash
# Profile specific benchmark
perf record -g ./core_benchmarks --benchmark_filter="BM_ObjectCreation"
perf report

# Or use make target
make profile_core
```

### Using Instruments (macOS)
```bash
# Run with Instruments
instruments -t "Time Profiler" ./core_benchmarks
```

### Using VTune (Intel)
```bash
# Collect hotspots
vtune -collect hotspots ./core_benchmarks
```

## Benchmark Naming Conventions

### Function Names
- Prefix with `BM_` for benchmarks
- Use descriptive names: `BM_<Component><Operation>`
- Examples:
    - `BM_ObjectCreation`
    - `BM_MemoryPoolAllocate`
    - `BM_ThreadPoolSubmitTask`

### File Names
- End with `_benchmark.cpp`
- Match source file: `object.cpp` → `object_benchmark.cpp`

### Benchmark Groups
```cpp
// Group related benchmarks
static void BM_Vector_PushBack(benchmark::State& state) { /* ... */ }
static void BM_Vector_Reserve(benchmark::State& state) { /* ... */ }
static void BM_Vector_Iteration(benchmark::State& state) { /* ... */ }
```

## Troubleshooting

### Benchmarks Not Found
- Check file location: `benchmarks/<module>/*_benchmark.cpp`
- Verify BENCHMARK() macro usage
- Ensure BUILD_PERFORMANCE_TESTS=ON
- Rebuild after adding files

### High Variance
- Check for CPU frequency scaling
- Close other applications
- Use `--benchmark_repetitions`
- Increase `--benchmark_min_time`
- Check for thermal throttling

### Unexpected Results
- Verify Release build mode
- Check `DoNotOptimize()` usage
- Ensure proper setup/teardown
- Look for cache effects
- Consider alignment issues

## Resources

- [Google Benchmark GitHub](https://github.com/google/benchmark)
- [User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Complexity Analysis](https://github.com/google/benchmark#asymptotic-complexity)
- [Benchmark Tools](https://github.com/google/benchmark/tree/main/tools)
- [Best Practices](https://github.com/google/benchmark/blob/main/docs/perf_tips.md)
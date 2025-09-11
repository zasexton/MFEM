# Core Library Test Suite

## Directory Structure

The test infrastructure is organized into two main directories:

```
tests/                      # All testing related files
├── CMakeLists.txt         # Test build configuration
├── test_main.cpp          # Optional custom test runner
├── test_utils.h           # Shared test utilities and helpers
├── unit/                  # Unit tests (mirrors core/ structure)
│   ├── base/
│   │   ├── object_test.cpp
│   │   ├── factory_test.cpp
│   │   └── ...
│   ├── memory/
│   │   ├── arena_test.cpp
│   │   └── ...
│   └── ...
└── integration/           # Integration tests (future)

benchmarks/                # Performance benchmarks
├── CMakeLists.txt        # Benchmark build configuration
├── benchmark_main.cpp    # Optional custom benchmark runner
├── base/                 # Benchmarks (mirrors core/ structure)
│   ├── object_benchmark.cpp
│   └── ...
├── memory/
│   ├── arena_benchmark.cpp
│   └── ...
└── ...
```

## 🏃 Running Tests

### Unit Tests

#### Build and Run All Tests
```bash
# From build directory
make core_unit_tests        # Build tests
make test_core              # Run all core tests via CTest
make test_core_verbose      # Run with detailed output

# Or run directly
./tests/core_unit_tests     # Run all tests
./tests/core_unit_tests --help    # See all Google Test options
```

#### Run Specific Tests
```bash
# Run tests matching a pattern
./tests/core_unit_tests --gtest_filter="ObjectTest.*"           # All ObjectTest cases
./tests/core_unit_tests --gtest_filter="*Construction*"         # Tests with "Construction" in name
./tests/core_unit_tests --gtest_filter="ObjectTest.UniqueID*"   # Specific test
./tests/core_unit_tests --gtest_filter="-*Death*"               # Exclude death tests
```

#### Test Output Options
```bash
# Colored output (default if terminal supports it)
./tests/core_unit_tests --gtest_color=yes

# XML output for CI integration
./tests/core_unit_tests --gtest_output=xml:test_results.xml

# JSON output
./tests/core_unit_tests --gtest_output=json:test_results.json

# Brief output
./tests/core_unit_tests --gtest_brief=1

# Show test execution time
./tests/core_unit_tests --gtest_print_time=1
```

#### Other Useful Options
```bash
# List all tests without running
./tests/core_unit_tests --gtest_list_tests

# Run tests multiple times
./tests/core_unit_tests --gtest_repeat=10

# Shuffle test order (detect dependencies)
./tests/core_unit_tests --gtest_shuffle

# Run until failure
./tests/core_unit_tests --gtest_repeat=-1 --gtest_break_on_failure
```

## Performance Benchmarks

Performance benchmarks are in a separate `benchmarks/` directory:

### Build and Run Benchmarks
```bash
# Build benchmarks (requires BUILD_PERFORMANCE_TESTS=ON)
make core_benchmarks

# Run all benchmarks
make benchmark_core

# Run quick benchmarks (fewer iterations)
make benchmark_core_quick

# Run detailed benchmarks with statistics
make benchmark_core_detailed

# Output to JSON for analysis
make benchmark_core_json

# Run benchmarks for specific module
make benchmark_core_base      # Only base module benchmarks
make benchmark_core_memory    # Only memory module benchmarks
```

### Benchmark Options
```bash
# Run with custom settings
./benchmarks/core_benchmarks --benchmark_format=console
./benchmarks/core_benchmarks --benchmark_out=results.json --benchmark_out_format=json
./benchmarks/core_benchmarks --benchmark_filter="Object"  # Filter benchmarks
./benchmarks/core_benchmarks --benchmark_time_unit=ns     # Change time unit
./benchmarks/core_benchmarks --benchmark_min_time=1.0     # Min time per benchmark

# Compare benchmarks
./benchmarks/core_benchmarks --benchmark_filter="Object" --benchmark_out=old.json
# ... make changes ...
./benchmarks/core_benchmarks --benchmark_filter="Object" --benchmark_out=new.json
# Use benchmark tools/compare.py old.json new.json
```

## Code Coverage

Enable coverage when configuring CMake:

```bash
# Configure with coverage
cmake .. -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug

# Build and run tests
make core_unit_tests
make coverage_core       # Generate coverage report

# View report
open coverage/html/index.html      # macOS
xdg-open coverage/html/index.html  # Linux
firefox coverage/html/index.html   # Direct browser
```

Coverage excludes benchmark code to focus on actual library coverage.

## Memory Checking

Run tests with Valgrind for memory leak detection:

```bash
# If Valgrind is installed
make memcheck_core

# Or manually
valgrind --leak-check=full --show-leak-kinds=all ./tests/core_unit_tests

# With suppressions file
valgrind --suppressions=valgrind.supp ./tests/core_unit_tests
```

## Performance Profiling

Profile benchmarks on Linux:

```bash
# If perf is available
make profile_core

# Or manually with perf
perf record -g ./benchmarks/core_benchmarks
perf report

# With specific benchmark filter
perf record -g ./benchmarks/core_benchmarks --benchmark_filter="ObjectCreation"
```

## Writing Tests

### Unit Test Structure

Unit tests go in `tests/unit/` with the same directory structure as the source:

```cpp
// tests/unit/base/object_test.cpp
#include <gtest/gtest.h>
#include "core/base/object.h"
#include "test_utils.h"  // Test utilities from tests/

using namespace fem::core::base;

// Simple test
TEST(ObjectTest, ConstructorCreatesUniqueID) {
    Object obj1;
    Object obj2;
    EXPECT_NE(obj1.id(), obj2.id());
}

// Test fixture for shared setup
class ObjectTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        obj_ = std::make_unique<Object>();
    }
    
    std::unique_ptr<Object> obj_;
};

TEST_F(ObjectTestFixture, IsValidAfterConstruction) {
    EXPECT_TRUE(obj_->is_valid());
}
```

### Benchmark Structure

Benchmarks go in `benchmarks/` with the same directory structure as the source:

```cpp
// benchmarks/base/object_benchmark.cpp
#include <benchmark/benchmark.h>
#include "core/base/object.h"

using namespace fem::core::base;

static void BM_ObjectCreation(benchmark::State& state) {
    for (auto _ : state) {
        Object obj("BenchmarkObject");
        benchmark::DoNotOptimize(obj.id());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectCreation);

// Benchmark with arguments
static void BM_VectorCreation(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<Object> objects(state.range(0));
        benchmark::DoNotOptimize(objects.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_VectorCreation)->Range(8, 8<<10)->Complexity();
```

### Using Test Utilities

The `test_utils.h` file provides common testing helpers:

```cpp
#include "test_utils.h"

using namespace fem::core::test;

TEST(ExampleTest, UsingUtilities) {
    // Timer utility
    Timer timer;
    expensive_operation();
    EXPECT_LT(timer.elapsed_ms(), 100.0);
    
    // Temp directory utility
    TempDirectory temp_dir("test_");
    auto file = temp_dir.create_file("data.txt", "content");
    EXPECT_TRUE(std::filesystem::exists(file));
    
    // Parallel execution helper
    std::atomic<int> counter{0};
    run_parallel(4, [&counter](int thread_id) {
        counter.fetch_add(1);
    });
    EXPECT_EQ(counter, 4);
    
    // Custom matchers
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_THAT(data, IsSorted());
    
    double value = 3.14159;
    EXPECT_THAT(value, IsNear(3.14, 0.01));
    
    // Performance assertions
    EXPECT_PERFORMANCE([]{ quick_operation(); }, 1.0, 1000);
}
```

## Naming Conventions

### Directory Structure
- Unit tests: `tests/unit/<module>/<source>_test.cpp`
- Benchmarks: `benchmarks/<module>/<source>_benchmark.cpp`
- Test utilities: `tests/test_utils.h`

### Test Names
- Test suites: Match the class name (e.g., `ObjectTest`)
- Test cases: Describe what is being tested (e.g., `ConstructorCreatesUniqueID`)
- Benchmarks: Prefix with `BM_` (e.g., `BM_ObjectCreation`)

### File Names
- Unit test files: `*_test.cpp`
- Benchmark files: `*_benchmark.cpp`
- Match source file: `object.cpp` → `object_test.cpp` & `object_benchmark.cpp`

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution (< 100ms per test)
- No external dependencies
- Mock external interfaces

### Integration Tests (`tests/integration/`) - Future
- Test component interactions
- May use file system, network
- Slower execution acceptable
- Test real configurations

### Benchmarks (`benchmarks/`)
- Measure performance characteristics
- Identify bottlenecks
- Track performance over time
- Compare implementations

## Debugging

### Debug Single Test
```bash
# With GDB
gdb ./tests/core_unit_tests
(gdb) break object_test.cpp:42
(gdb) run --gtest_filter="ObjectTest.UniqueIDGeneration"

# With LLDB
lldb ./tests/core_unit_tests
(lldb) b object_test.cpp:42
(lldb) run --gtest_filter="ObjectTest.UniqueIDGeneration"
```

### Debug Benchmark
```bash
gdb ./benchmarks/core_benchmarks
(gdb) break object_benchmark.cpp:20
(gdb) run --benchmark_filter="BM_ObjectCreation"
```

### Break on Test Failure
```bash
./tests/core_unit_tests --gtest_break_on_failure
```

## Troubleshooting

### Tests Not Found
- Check file location: `tests/unit/<module>/*_test.cpp`
- Verify TEST() macro syntax
- Rebuild after adding new files
- Check CMakeLists.txt includes the module

### Benchmarks Not Found
- Check file location: `benchmarks/<module>/*_benchmark.cpp`
- Verify BENCHMARK() macro syntax
- Ensure BUILD_PERFORMANCE_TESTS=ON
- Rebuild after adding new files

### Compilation Errors
- Include paths: Use `"core/module/header.h"`
- Test utilities: Include `"test_utils.h"`
- Link libraries: Check CMakeLists.txt

### Performance Issues
- Run benchmarks in Release mode
- Disable CPU frequency scaling
- Close other applications
- Use `--benchmark_repetitions` for stability

## Resources

### Testing
- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [Google Test Advanced Guide](https://google.github.io/googletest/advanced.html)
- [Google Mock Documentation](https://google.github.io/googletest/gmock_for_dummies.html)

### Benchmarking
- [Google Benchmark](https://github.com/google/benchmark)
- [Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Benchmark Complexity Analysis](https://github.com/google/benchmark#asymptotic-complexity)

### Profiling
- [Perf Wiki](https://perf.wiki.kernel.org/index.php/Main_Page)
- [Valgrind Documentation](https://valgrind.org/docs/manual/quick-start.html)
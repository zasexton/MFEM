#include <gtest/gtest.h>
#include <core/memory/prefetch.h>
#include <vector>
#include <array>
#include <memory>
#include <cstring>
#include <thread>
#include <chrono>
#include <numeric>

namespace fcm = fem::core::memory;

class PrefetchTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Test data structures
    struct TestStruct {
        int a, b, c, d;
        TestStruct(int val = 0) : a(val), b(val+1), c(val+2), d(val+3) {}
    };

    struct LargeStruct {
        double data[128];  // 1KB structure
        LargeStruct() { std::fill(std::begin(data), std::end(data), 3.14); }
    };
};

// Test enum class PrefetchLocality
TEST_F(PrefetchTest, LocalityEnumValues) {
    // Verify enum values match expected ordering
    EXPECT_EQ(static_cast<int>(fcm::PrefetchLocality::NTA), 0);
    EXPECT_EQ(static_cast<int>(fcm::PrefetchLocality::L3), 1);
    EXPECT_EQ(static_cast<int>(fcm::PrefetchLocality::L2), 2);
    EXPECT_EQ(static_cast<int>(fcm::PrefetchLocality::L1), 3);

    // Verify the ordering makes sense (higher value = closer cache)
    EXPECT_LT(static_cast<int>(fcm::PrefetchLocality::NTA), static_cast<int>(fcm::PrefetchLocality::L3));
    EXPECT_LT(static_cast<int>(fcm::PrefetchLocality::L3), static_cast<int>(fcm::PrefetchLocality::L2));
    EXPECT_LT(static_cast<int>(fcm::PrefetchLocality::L2), static_cast<int>(fcm::PrefetchLocality::L1));
}

// Basic functionality tests - void* versions
TEST_F(PrefetchTest, PrefetchReadVoidPtr_AllLocalities) {
    int data = 42;
    const void* ptr = &data;

    // Should not crash or throw for any locality
    EXPECT_NO_THROW(fcm::prefetch_read(ptr, fcm::PrefetchLocality::NTA));
    EXPECT_NO_THROW(fcm::prefetch_read(ptr, fcm::PrefetchLocality::L3));
    EXPECT_NO_THROW(fcm::prefetch_read(ptr, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(ptr, fcm::PrefetchLocality::L1));

    // Default locality should also work
    EXPECT_NO_THROW(fcm::prefetch_read(ptr));
}

TEST_F(PrefetchTest, PrefetchWriteVoidPtr_AllLocalities) {
    int data = 42;
    const void* ptr = &data;

    // Should not crash or throw for any locality
    EXPECT_NO_THROW(fcm::prefetch_write(ptr, fcm::PrefetchLocality::NTA));
    EXPECT_NO_THROW(fcm::prefetch_write(ptr, fcm::PrefetchLocality::L3));
    EXPECT_NO_THROW(fcm::prefetch_write(ptr, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_write(ptr, fcm::PrefetchLocality::L1));

    // Default locality should also work
    EXPECT_NO_THROW(fcm::prefetch_write(ptr));
}

// Typed pointer template versions
TEST_F(PrefetchTest, PrefetchReadTypedPtr_BasicTypes) {
    int int_val = 123;
    double double_val = 3.14159;
    char char_val = 'A';

    // Test different basic types
    EXPECT_NO_THROW(fcm::prefetch_read(&int_val, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_read(&double_val, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(&char_val, fcm::PrefetchLocality::L3));
}

TEST_F(PrefetchTest, PrefetchWriteTypedPtr_BasicTypes) {
    int int_val = 123;
    double double_val = 3.14159;
    char char_val = 'A';

    // Test different basic types
    EXPECT_NO_THROW(fcm::prefetch_write(&int_val, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_write(&double_val, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_write(&char_val, fcm::PrefetchLocality::L3));
}

TEST_F(PrefetchTest, PrefetchReadStructTypes) {
    TestStruct small_struct(10);
    LargeStruct large_struct;

    // Test with custom struct types
    EXPECT_NO_THROW(fcm::prefetch_read(&small_struct, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_read(&large_struct, fcm::PrefetchLocality::L2));

    // Test const structs
    const TestStruct const_struct(20);
    EXPECT_NO_THROW(fcm::prefetch_read(&const_struct, fcm::PrefetchLocality::L3));
}

TEST_F(PrefetchTest, PrefetchWriteStructTypes) {
    TestStruct small_struct(10);
    LargeStruct large_struct;

    // Test with custom struct types
    EXPECT_NO_THROW(fcm::prefetch_write(&small_struct, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_write(&large_struct, fcm::PrefetchLocality::L2));

    // Note: const structs work for write prefetch (it's just a hint)
    const TestStruct const_struct(20);
    EXPECT_NO_THROW(fcm::prefetch_write(&const_struct, fcm::PrefetchLocality::L3));
}

// Array and container tests
TEST_F(PrefetchTest, PrefetchArrayTypes) {
    int static_array[100];
    std::array<double, 50> std_array;
    std::vector<TestStruct> vector_data(25);

    // Initialize data
    for (int i = 0; i < 100; ++i) static_array[i] = i;
    std_array.fill(2.718);
    for (size_t i = 0; i < vector_data.size(); ++i) {
        vector_data[i] = TestStruct(static_cast<int>(i));
    }

    // Test array prefetching
    EXPECT_NO_THROW(fcm::prefetch_read(static_array, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_read(std_array.data(), fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(vector_data.data(), fcm::PrefetchLocality::L3));

    // Test individual elements
    EXPECT_NO_THROW(fcm::prefetch_read(&static_array[50], fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_read(&std_array[25], fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(&vector_data[10], fcm::PrefetchLocality::L3));
}

// Convenience wrapper tests
TEST_F(PrefetchTest, ConvenienceWrappers) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    const int* ptr = data.data();

    // Test all convenience wrappers
    EXPECT_NO_THROW(fcm::prefetch_l1(ptr));
    EXPECT_NO_THROW(fcm::prefetch_l2(ptr + 100));
    EXPECT_NO_THROW(fcm::prefetch_l3(ptr + 200));
    EXPECT_NO_THROW(fcm::prefetch_nta(ptr + 300));

    // Test with different types
    TestStruct struct_data;
    EXPECT_NO_THROW(fcm::prefetch_l1(&struct_data));
    EXPECT_NO_THROW(fcm::prefetch_l2(&struct_data));
    EXPECT_NO_THROW(fcm::prefetch_l3(&struct_data));
    EXPECT_NO_THROW(fcm::prefetch_nta(&struct_data));
}

TEST_F(PrefetchTest, ConvenienceWrappersWithArrays) {
    double array[1000];
    for (int i = 0; i < 1000; ++i) array[i] = i * 0.1;

    // Test with array elements
    EXPECT_NO_THROW(fcm::prefetch_l1(&array[0]));
    EXPECT_NO_THROW(fcm::prefetch_l2(&array[250]));
    EXPECT_NO_THROW(fcm::prefetch_l3(&array[500]));
    EXPECT_NO_THROW(fcm::prefetch_nta(&array[750]));
}

// Edge cases and safety tests
TEST_F(PrefetchTest, NullPtrSafety) {
    // Prefetching nullptr should be safe (no-op)
    EXPECT_NO_THROW(fcm::prefetch_read(static_cast<void*>(nullptr)));
    EXPECT_NO_THROW(fcm::prefetch_write(static_cast<void*>(nullptr)));

    EXPECT_NO_THROW(fcm::prefetch_read(static_cast<int*>(nullptr)));
    EXPECT_NO_THROW(fcm::prefetch_write(static_cast<int*>(nullptr)));

    // Convenience wrappers with nullptr
    EXPECT_NO_THROW(fcm::prefetch_l1(static_cast<int*>(nullptr)));
    EXPECT_NO_THROW(fcm::prefetch_l2(static_cast<int*>(nullptr)));
    EXPECT_NO_THROW(fcm::prefetch_l3(static_cast<int*>(nullptr)));
    EXPECT_NO_THROW(fcm::prefetch_nta(static_cast<int*>(nullptr)));
}

TEST_F(PrefetchTest, AlignmentAndPadding) {
    // Test prefetch with various alignments
    alignas(64) int aligned_data = 42;

    EXPECT_NO_THROW(fcm::prefetch_read(&aligned_data, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_write(&aligned_data, fcm::PrefetchLocality::L1));

    // Test with unaligned access (should still be safe)
    char buffer[128];
    for (int i = 0; i < 128; ++i) buffer[i] = static_cast<char>(i);

    // Prefetch at various offsets
    for (int offset = 0; offset < 8; ++offset) {
        EXPECT_NO_THROW(fcm::prefetch_read(&buffer[offset], fcm::PrefetchLocality::L1));
        EXPECT_NO_THROW(fcm::prefetch_write(&buffer[offset], fcm::PrefetchLocality::L1));
    }
}

// Cross-platform compatibility test
TEST_F(PrefetchTest, CrossPlatformBehavior) {
    int data = 100;

    // These should work on all platforms (no-op on unsupported platforms)
    EXPECT_NO_THROW(fcm::prefetch_read(&data, fcm::PrefetchLocality::NTA));
    EXPECT_NO_THROW(fcm::prefetch_read(&data, fcm::PrefetchLocality::L3));
    EXPECT_NO_THROW(fcm::prefetch_read(&data, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(&data, fcm::PrefetchLocality::L1));

    EXPECT_NO_THROW(fcm::prefetch_write(&data, fcm::PrefetchLocality::NTA));
    EXPECT_NO_THROW(fcm::prefetch_write(&data, fcm::PrefetchLocality::L3));
    EXPECT_NO_THROW(fcm::prefetch_write(&data, fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_write(&data, fcm::PrefetchLocality::L1));
}

// Performance pattern tests
TEST_F(PrefetchTest, LinearAccessPattern) {
    const size_t size = 1000;
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);

    double sum = 0.0;
    const size_t prefetch_distance = 8;

    // Simulate linear traversal with prefetch
    for (size_t i = 0; i < size; ++i) {
        // Prefetch ahead
        if (i + prefetch_distance < size) {
            fcm::prefetch_read(&data[i + prefetch_distance], fcm::PrefetchLocality::L1);
        }

        // Process current element
        sum += data[i];
    }

    // Verify computation is correct
    double expected = size * (size + 1) / 2.0;
    EXPECT_DOUBLE_EQ(sum, expected);
}

TEST_F(PrefetchTest, StridedAccessPattern) {
    const size_t rows = 50;
    const size_t cols = 50;
    std::vector<int> matrix(rows * cols);

    // Initialize matrix
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<int>(i + 1);
    }

    // Column-wise traversal with prefetch
    std::vector<long> col_sums(cols, 0);

    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            size_t idx = row * cols + col;

            // Prefetch next element in same column
            if (row + 1 < rows) {
                size_t next_idx = (row + 1) * cols + col;
                fcm::prefetch_read(&matrix[next_idx], fcm::PrefetchLocality::L1);
            }

            col_sums[col] += matrix[idx];
        }
    }

    // Verify sums are correct
    for (size_t col = 0; col < cols; ++col) {
        long expected = 0;
        for (size_t row = 0; row < rows; ++row) {
            expected += static_cast<long>((row * cols + col) + 1);
        }
        EXPECT_EQ(col_sums[col], expected);
    }
}

// Multithreaded safety test
TEST_F(PrefetchTest, ThreadSafety) {
    const size_t data_size = 10000;
    std::vector<int> shared_data(data_size);
    std::iota(shared_data.begin(), shared_data.end(), 1);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<long> thread_sums(num_threads, 0);

    // Each thread processes a section with prefetch
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * (data_size / num_threads);
            size_t end = (t + 1) * (data_size / num_threads);
            if (t == num_threads - 1) end = data_size; // Last thread takes remainder

            for (size_t i = start; i < end; ++i) {
                // Prefetch ahead
                if (i + 4 < end) {
                    fcm::prefetch_read(&shared_data[i + 4], fcm::PrefetchLocality::L1);
                }

                thread_sums[t] += shared_data[i];
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify total sum
    long total_sum = 0;
    for (long sum : thread_sums) {
        total_sum += sum;
    }

    long expected_sum = static_cast<long>(data_size * (data_size + 1) / 2);
    EXPECT_EQ(total_sum, expected_sum);
}

// Comprehensive functionality verification
TEST_F(PrefetchTest, ComprehensiveApiTest) {
    // Test all combinations of functions and localities
    TestStruct test_struct(42);

    // Void pointer versions
    const void* void_ptr = &test_struct;
    fcm::prefetch_read(void_ptr, fcm::PrefetchLocality::NTA);
    fcm::prefetch_read(void_ptr, fcm::PrefetchLocality::L3);
    fcm::prefetch_read(void_ptr, fcm::PrefetchLocality::L2);
    fcm::prefetch_read(void_ptr, fcm::PrefetchLocality::L1);
    fcm::prefetch_read(void_ptr); // Default

    fcm::prefetch_write(void_ptr, fcm::PrefetchLocality::NTA);
    fcm::prefetch_write(void_ptr, fcm::PrefetchLocality::L3);
    fcm::prefetch_write(void_ptr, fcm::PrefetchLocality::L2);
    fcm::prefetch_write(void_ptr, fcm::PrefetchLocality::L1);
    fcm::prefetch_write(void_ptr); // Default

    // Typed pointer versions
    fcm::prefetch_read(&test_struct, fcm::PrefetchLocality::NTA);
    fcm::prefetch_read(&test_struct, fcm::PrefetchLocality::L3);
    fcm::prefetch_read(&test_struct, fcm::PrefetchLocality::L2);
    fcm::prefetch_read(&test_struct, fcm::PrefetchLocality::L1);
    fcm::prefetch_read(&test_struct); // Default

    fcm::prefetch_write(&test_struct, fcm::PrefetchLocality::NTA);
    fcm::prefetch_write(&test_struct, fcm::PrefetchLocality::L3);
    fcm::prefetch_write(&test_struct, fcm::PrefetchLocality::L2);
    fcm::prefetch_write(&test_struct, fcm::PrefetchLocality::L1);
    fcm::prefetch_write(&test_struct); // Default

    // Convenience wrappers
    fcm::prefetch_l1(&test_struct);
    fcm::prefetch_l2(&test_struct);
    fcm::prefetch_l3(&test_struct);
    fcm::prefetch_nta(&test_struct);

    // If we reach here without crashes, all APIs work
    SUCCEED();
}

// Memory ordering and behavior test
TEST_F(PrefetchTest, MemoryBehaviorTest) {
    const size_t buffer_size = 1024;
    std::vector<char> buffer(buffer_size);

    // Fill with pattern
    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i] = static_cast<char>(i % 256);
    }

    // Prefetch entire buffer in chunks
    const size_t chunk_size = 64; // Typical cache line size
    for (size_t offset = 0; offset < buffer_size; offset += chunk_size) {
        fcm::prefetch_read(&buffer[offset], fcm::PrefetchLocality::L1);

        // Small delay to allow prefetch to potentially complete
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }

    // Verify buffer contents are unchanged (prefetch shouldn't modify data)
    for (size_t i = 0; i < buffer_size; ++i) {
        EXPECT_EQ(buffer[i], static_cast<char>(i % 256));
    }
}

// Edge case: Very large structure
TEST_F(PrefetchTest, LargeStructurePrefetch) {
    struct VeryLargeStruct {
        double data[1024];  // 8KB structure
        VeryLargeStruct() {
            for (int i = 0; i < 1024; ++i) {
                data[i] = i * 0.5;
            }
        }
    };

    VeryLargeStruct large_obj;

    // Should handle large objects gracefully
    EXPECT_NO_THROW(fcm::prefetch_read(&large_obj, fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_write(&large_obj, fcm::PrefetchLocality::L1));

    // Prefetch different parts of the large structure
    EXPECT_NO_THROW(fcm::prefetch_read(&large_obj.data[0], fcm::PrefetchLocality::L1));
    EXPECT_NO_THROW(fcm::prefetch_read(&large_obj.data[512], fcm::PrefetchLocality::L2));
    EXPECT_NO_THROW(fcm::prefetch_read(&large_obj.data[1023], fcm::PrefetchLocality::L3));
}
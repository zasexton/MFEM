#include <gtest/gtest.h>
#include <core/memory/cache_line.h>
#include <core/memory/prefetch.h>
#include <vector>
#include <array>
#include <chrono>
#include <numeric>

using namespace fem::core::memory;

class CacheAndPrefetchTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Cache Line Tests
// ============================================================================

TEST_F(CacheAndPrefetchTest, CacheLineSize) {
    size_t size = cache_line_size();

    // Common cache line sizes are 32, 64, or 128 bytes
    EXPECT_TRUE(size == 32 || size == 64 || size == 128);
    EXPECT_TRUE(is_power_of_two(size));

    // Verify it matches config
    EXPECT_EQ(fem::config::CACHE_LINE_SIZE, size);
}

TEST_F(CacheAndPrefetchTest, CacheLineMask) {
    size_t mask = cache_line_mask();
    size_t size = cache_line_size();

    EXPECT_EQ(size - 1, mask);

    // Verify mask properties
    EXPECT_EQ(0u, (size & mask));  // Size aligned to itself should mask to 0
    EXPECT_EQ(mask, ((size - 1) & mask));  // One less than size should preserve all bits
}

TEST_F(CacheAndPrefetchTest, PadToCacheLine) {
    size_t line_size = cache_line_size();

    EXPECT_EQ(0u, pad_to_cache_line(0));
    EXPECT_EQ(line_size, pad_to_cache_line(1));
    EXPECT_EQ(line_size, pad_to_cache_line(line_size - 1));
    EXPECT_EQ(line_size, pad_to_cache_line(line_size));
    EXPECT_EQ(2 * line_size, pad_to_cache_line(line_size + 1));

    // Test with various sizes
    for (size_t i = 0; i <= line_size * 3; i += 7) {
        size_t padded = pad_to_cache_line(i);
        EXPECT_EQ(0u, padded % line_size) << "Failed for size " << i;
        EXPECT_GE(padded, i) << "Failed for size " << i;
    }
}

TEST_F(CacheAndPrefetchTest, SameCacheLine) {
    size_t line_size = cache_line_size();

    // Allocate aligned buffer
    alignas(128) std::byte buffer[512] = {};

    // Pointers in same cache line
    void* p1 = &buffer[0];
    void* p2 = &buffer[1];
    void* p3 = &buffer[line_size - 1];

    EXPECT_TRUE(same_cache_line(p1, p2));
    EXPECT_TRUE(same_cache_line(p1, p3));
    EXPECT_TRUE(same_cache_line(p2, p3));

    // Pointers in different cache lines
    void* p4 = &buffer[line_size];
    void* p5 = &buffer[line_size + 1];

    EXPECT_FALSE(same_cache_line(p1, p4));
    EXPECT_FALSE(same_cache_line(p1, p5));
    EXPECT_FALSE(same_cache_line(p3, p4));

    EXPECT_TRUE(same_cache_line(p4, p5));

    // Far apart
    void* p6 = &buffer[256];
    EXPECT_FALSE(same_cache_line(p1, p6));
    EXPECT_FALSE(same_cache_line(p4, p6));
}

TEST_F(CacheAndPrefetchTest, CacheAlignedType) {
    struct TestData {
        int x;
        double y;
    };

    cache_aligned<TestData> aligned_data;

    // Check alignment
    EXPECT_TRUE(is_aligned(&aligned_data.value, cache_line_size()));

    // Test usage
    aligned_data.value.x = 42;
    aligned_data.value.y = 3.14;

    EXPECT_EQ(42, aligned_data.value.x);
    EXPECT_DOUBLE_EQ(3.14, aligned_data.value.y);
}

TEST_F(CacheAndPrefetchTest, CacheAlignedArray) {
    const size_t count = 4;
    cache_aligned<int> values[count];

    // Each element should be cache-line aligned
    for (size_t i = 0; i < count; ++i) {
        EXPECT_TRUE(is_aligned(&values[i].value, cache_line_size()));
        values[i].value = static_cast<int>(i * 10);
    }

    // Verify no false sharing - elements should be in different cache lines
    for (size_t i = 0; i < count - 1; ++i) {
        EXPECT_FALSE(same_cache_line(&values[i].value, &values[i + 1].value));
    }

    // Verify values
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(static_cast<int>(i * 10), values[i].value);
    }
}

// ============================================================================
// Prefetch Tests
// ============================================================================

TEST_F(CacheAndPrefetchTest, PrefetchRead_BasicTypes) {
    // These should compile and not crash
    int int_value = 42;
    double double_value = 3.14;
    std::array<char, 256> array_value{};

    // Test with different localities
    prefetch_read(&int_value, PrefetchLocality::L1);
    prefetch_read(&int_value, PrefetchLocality::L2);
    prefetch_read(&int_value, PrefetchLocality::L3);
    prefetch_read(&int_value, PrefetchLocality::NTA);

    prefetch_read(&double_value);  // Default locality
    prefetch_read(array_value.data());

    // Prefetch nullptr should be safe (no-op)
    prefetch_read(nullptr);

    SUCCEED();  // If we get here without crashing, test passes
}

TEST_F(CacheAndPrefetchTest, PrefetchWrite_BasicTypes) {
    int int_value = 42;
    double double_value = 3.14;
    std::vector<int> vec(100);

    // Test with different localities
    prefetch_write(&int_value, PrefetchLocality::L1);
    prefetch_write(&int_value, PrefetchLocality::L2);
    prefetch_write(&int_value, PrefetchLocality::L3);
    prefetch_write(&int_value, PrefetchLocality::NTA);

    prefetch_write(&double_value);  // Default locality
    prefetch_write(vec.data());

    // Prefetch nullptr should be safe
    prefetch_write(nullptr);

    SUCCEED();
}

TEST_F(CacheAndPrefetchTest, PrefetchRead_TypedPointers) {
    struct LargeStruct {
        double data[64];  // 512 bytes, multiple cache lines
    };

    LargeStruct s{};
    const LargeStruct* ps = &s;

    // Typed prefetch
    prefetch_read(ps, PrefetchLocality::L1);
    prefetch_read(ps, PrefetchLocality::L2);
    prefetch_read(ps, PrefetchLocality::L3);
    prefetch_read(ps, PrefetchLocality::NTA);

    // Array of structs
    std::vector<LargeStruct> vec(10);
    for (size_t i = 0; i < vec.size(); ++i) {
        prefetch_read(&vec[i]);
    }

    SUCCEED();
}

TEST_F(CacheAndPrefetchTest, PrefetchWrite_TypedPointers) {
    struct TestStruct {
        int x, y, z;
    };

    TestStruct s{};
    TestStruct* ps = &s;

    prefetch_write(ps, PrefetchLocality::L1);
    prefetch_write(ps, PrefetchLocality::L2);
    prefetch_write(ps, PrefetchLocality::L3);
    prefetch_write(ps, PrefetchLocality::NTA);

    SUCCEED();
}

TEST_F(CacheAndPrefetchTest, PrefetchHelpers_L1L2L3NTA) {
    std::vector<int> data(1000);

    // Test convenience wrappers
    prefetch_l1(data.data());
    prefetch_l2(&data[100]);
    prefetch_l3(&data[200]);
    prefetch_nta(&data[300]);

    // Should work with const pointers too
    const int* cdata = data.data();
    prefetch_l1(cdata);
    prefetch_l2(cdata + 100);
    prefetch_l3(cdata + 200);
    prefetch_nta(cdata + 300);

    SUCCEED();
}

TEST_F(CacheAndPrefetchTest, PrefetchPattern_LinearAccess) {
    // Test a realistic prefetch pattern for linear array traversal
    const size_t size = 10000;
    std::vector<double> data(size);

    // Initialize data
    std::iota(data.begin(), data.end(), 0.0);

    // Simulate processing with prefetch
    const size_t prefetch_distance = 8;  // Prefetch 8 elements ahead
    double sum = 0;

    for (size_t i = 0; i < size; ++i) {
        // Prefetch ahead for future iterations
        if (i + prefetch_distance < size) {
            prefetch_read(&data[i + prefetch_distance], PrefetchLocality::L1);
        }

        // Process current element
        sum += data[i] * data[i];
    }

    // Verify computation
    double expected = 0;
    for (size_t i = 0; i < size; ++i) {
        expected += static_cast<double>(i * i);
    }
    EXPECT_DOUBLE_EQ(expected, sum);
}

TEST_F(CacheAndPrefetchTest, PrefetchPattern_StridedAccess) {
    // Test prefetch with strided access pattern
    const size_t rows = 100;
    const size_t cols = 100;
    std::vector<int> matrix(rows * cols);

    // Initialize
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<int>(i);
    }

    // Column-wise sum with prefetch (poor cache pattern without prefetch)
    std::vector<int> col_sums(cols, 0);

    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            // Prefetch next row in same column
            if (row + 1 < rows) {
                prefetch_read(&matrix[(row + 1) * cols + col], PrefetchLocality::L1);
            }

            col_sums[col] += matrix[row * cols + col];
        }
    }

    // Verify sums
    for (size_t col = 0; col < cols; ++col) {
        int expected = 0;
        for (size_t row = 0; row < rows; ++row) {
            expected += static_cast<int>(row * cols + col);
        }
        EXPECT_EQ(expected, col_sums[col]) << "Column " << col;
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CacheAndPrefetchTest, FalseSharing_Demonstration) {
    // Demonstrate false sharing prevention using cache alignment
    struct Counter {
        alignas(1) volatile long count = 0;  // Deliberately misaligned
    };

    struct AlignedCounter {
        cache_aligned<volatile long> count{0};
    };

    // Regular counters might share cache lines
    Counter regular_counters[4];

    // Aligned counters won't share cache lines
    AlignedCounter aligned_counters[4];

    // Check that regular counters might share cache lines
    // On most systems, adjacent Counter objects will share cache lines
    // But we can't guarantee this, so just check our aligned version
    for (int i = 0; i < 3; ++i) {
        // Just check that we can call same_cache_line with volatile pointers
        [[maybe_unused]] bool shared = same_cache_line(
            const_cast<const void*>(static_cast<volatile void*>(&regular_counters[i].count)),
            const_cast<const void*>(static_cast<volatile void*>(&regular_counters[i + 1].count)));
    }

    // Check that aligned counters don't share cache lines
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(same_cache_line(const_cast<const void*>(static_cast<volatile void*>(&aligned_counters[i].count.value)),
                                    const_cast<const void*>(static_cast<volatile void*>(&aligned_counters[i + 1].count.value))))
            << "Aligned counters " << i << " and " << (i + 1)
            << " should not share cache lines";
    }
}

TEST_F(CacheAndPrefetchTest, CacheLineUtilities_Consistency) {
    size_t line_size = cache_line_size();

    // Test consistency between different utilities
    for (size_t addr = 0; addr < line_size * 4; addr += 17) {
        void* p1 = reinterpret_cast<void*>(addr);
        void* p2 = reinterpret_cast<void*>(addr + 1);

        // If in same cache line, should round to same address
        if (same_cache_line(p1, p2)) {
            size_t aligned1 = reinterpret_cast<size_t>(p1) & ~cache_line_mask();
            size_t aligned2 = reinterpret_cast<size_t>(p2) & ~cache_line_mask();
            EXPECT_EQ(aligned1, aligned2);
        }
    }
}

TEST_F(CacheAndPrefetchTest, PrefetchLocality_EnumValues) {
    // Verify enum values are as expected
    EXPECT_EQ(0, static_cast<int>(PrefetchLocality::NTA));
    EXPECT_EQ(1, static_cast<int>(PrefetchLocality::L3));
    EXPECT_EQ(2, static_cast<int>(PrefetchLocality::L2));
    EXPECT_EQ(3, static_cast<int>(PrefetchLocality::L1));
}
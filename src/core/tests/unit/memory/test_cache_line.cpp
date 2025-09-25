#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <atomic>
#include <thread>
#include <chrono>
#include <type_traits>

#include <core/memory/cache_line.h>
#include <config/config.h>

namespace fem::core::memory {
namespace {

// Test fixture for cache line tests
class CacheLineTest : public ::testing::Test {
protected:
    static constexpr std::size_t expected_cache_line_size = fem::config::CACHE_LINE_SIZE;
};

// Test cache_line_size function
TEST_F(CacheLineTest, CacheLineSize) {
    constexpr std::size_t size = cache_line_size();

    // Check that it returns the configured cache line size
    EXPECT_EQ(size, expected_cache_line_size);

    // Check that it's a power of 2
    EXPECT_TRUE((size & (size - 1)) == 0);

    // Check that it's a reasonable size (typically 64 or 128 bytes)
    EXPECT_GE(size, 32u);
    EXPECT_LE(size, 256u);
}

TEST_F(CacheLineTest, CacheLineSizeIsConstexpr) {
    // Verify that cache_line_size() can be used in constexpr context
    constexpr std::size_t compile_time_size = cache_line_size();
    EXPECT_EQ(compile_time_size, expected_cache_line_size);

    // Use in array size (requires compile-time constant)
    std::array<char, cache_line_size()> arr;
    EXPECT_EQ(arr.size(), expected_cache_line_size);
}

// Test cache_line_mask function
TEST_F(CacheLineTest, CacheLineMask) {
    constexpr std::size_t mask = cache_line_mask();
    constexpr std::size_t size = cache_line_size();

    // Mask should be size - 1
    EXPECT_EQ(mask, size - 1);

    // All lower bits should be set
    for (std::size_t i = 0; i < 8; ++i) {
        if ((1u << i) < size) {
            EXPECT_TRUE(mask & (1u << i));
        }
    }
}

TEST_F(CacheLineTest, CacheLineMaskIsConstexpr) {
    // Verify constexpr usage
    constexpr std::size_t compile_time_mask = cache_line_mask();
    EXPECT_EQ(compile_time_mask, cache_line_size() - 1);
}

// Test pad_to_cache_line function
TEST_F(CacheLineTest, PadToCacheLine) {
    constexpr std::size_t cls = cache_line_size();

    // Test exact multiples
    EXPECT_EQ(pad_to_cache_line(0), 0u);
    EXPECT_EQ(pad_to_cache_line(cls), cls);
    EXPECT_EQ(pad_to_cache_line(2 * cls), 2 * cls);
    EXPECT_EQ(pad_to_cache_line(3 * cls), 3 * cls);

    // Test values that need padding
    EXPECT_EQ(pad_to_cache_line(1), cls);
    EXPECT_EQ(pad_to_cache_line(cls - 1), cls);
    EXPECT_EQ(pad_to_cache_line(cls + 1), 2 * cls);
    EXPECT_EQ(pad_to_cache_line(2 * cls - 1), 2 * cls);
    EXPECT_EQ(pad_to_cache_line(2 * cls + 1), 3 * cls);
}

TEST_F(CacheLineTest, PadToCacheLineVariousSizes) {
    constexpr std::size_t cls = cache_line_size();

    // Test various common sizes
    for (std::size_t size = 1; size <= cls * 4; ++size) {
        std::size_t padded = pad_to_cache_line(size);

        // Result should be aligned to cache line
        EXPECT_EQ(padded % cls, 0u);

        // Result should be at least the input size
        EXPECT_GE(padded, size);

        // Result should be less than input + cache line size
        EXPECT_LT(padded, size + cls);
    }
}

TEST_F(CacheLineTest, PadToCacheLineIsConstexpr) {
    // Verify constexpr usage
    constexpr std::size_t padded1 = pad_to_cache_line(10);
    constexpr std::size_t padded2 = pad_to_cache_line(100);

    EXPECT_EQ(padded1, cache_line_size());
    EXPECT_GE(padded2, 100u);
    EXPECT_EQ(padded2 % cache_line_size(), 0u);
}

// Test same_cache_line function
TEST_F(CacheLineTest, SameCacheLineBasic) {
    constexpr std::size_t cls = cache_line_size();

    // Allocate aligned memory for testing
    alignas(cls) char buffer[cls * 4] = {};

    // Pointers in the same cache line should return true
    EXPECT_TRUE(same_cache_line(&buffer[0], &buffer[0]));
    EXPECT_TRUE(same_cache_line(&buffer[0], &buffer[1]));
    EXPECT_TRUE(same_cache_line(&buffer[0], &buffer[cls - 1]));

    // Pointers in different cache lines should return false
    EXPECT_FALSE(same_cache_line(&buffer[0], &buffer[cls]));
    EXPECT_FALSE(same_cache_line(&buffer[0], &buffer[cls + 1]));
    EXPECT_FALSE(same_cache_line(&buffer[0], &buffer[2 * cls]));
    EXPECT_FALSE(same_cache_line(&buffer[cls], &buffer[2 * cls]));
}

TEST_F(CacheLineTest, SameCacheLineBoundaries) {
    constexpr std::size_t cls = cache_line_size();

    alignas(cls) char buffer[cls * 3] = {};

    // Test boundary conditions
    // Last byte of first cache line vs first byte of second cache line
    EXPECT_FALSE(same_cache_line(&buffer[cls - 1], &buffer[cls]));

    // First and last byte of same cache line
    EXPECT_TRUE(same_cache_line(&buffer[cls], &buffer[2 * cls - 1]));

    // Test with various offsets
    for (std::size_t i = 0; i < cls; ++i) {
        for (std::size_t j = 0; j < cls; ++j) {
            EXPECT_TRUE(same_cache_line(&buffer[cls + i], &buffer[cls + j]));
        }
    }
}

TEST_F(CacheLineTest, SameCacheLineNullptr) {
    // Test with nullptr - implementation defined but should not crash
    bool result = same_cache_line(nullptr, nullptr);
    (void)result; // Result is implementation-defined

    // Test nullptr with valid pointer
    int x = 42;
    result = same_cache_line(nullptr, &x);
    EXPECT_FALSE(result); // Different addresses, likely different cache lines
}

TEST_F(CacheLineTest, SameCacheLineUnaligned) {
    constexpr std::size_t cls = cache_line_size();

    // Test with unaligned pointers
    char* buffer = new char[cls * 4 + 17];

    // Find an unaligned starting point
    char* unaligned = buffer + 7;

    // Points within cache line size should still be detected as same line
    EXPECT_TRUE(same_cache_line(unaligned, unaligned + 1));

    // Points far apart should be different
    EXPECT_FALSE(same_cache_line(unaligned, unaligned + cls * 2));

    delete[] buffer;
}

// Test cache_aligned template
TEST_F(CacheLineTest, CacheAlignedAlignment) {
    // Test that cache_aligned provides proper alignment
    struct TestStruct {
        int x;
        double y;
    };

    using CacheAlignedInt = cache_aligned<int>;
    using CacheAlignedStruct = cache_aligned<TestStruct>;

    // Check alignment requirements
    EXPECT_EQ(alignof(CacheAlignedInt), cache_line_size());
    EXPECT_EQ(alignof(CacheAlignedStruct), cache_line_size());

    // Check that size is padded appropriately
    EXPECT_GE(sizeof(CacheAlignedInt), sizeof(int));
    EXPECT_GE(sizeof(CacheAlignedStruct), sizeof(TestStruct));
}

TEST_F(CacheLineTest, CacheAlignedArray) {
    constexpr std::size_t cls = cache_line_size();

    // Create an array of cache-aligned integers
    using CacheAlignedInt = cache_aligned<int>;
    CacheAlignedInt arr[4];

    // Each element should be in a different cache line
    for (int i = 0; i < 4; ++i) {
        arr[i].value = i;
    }

    // Verify values
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(arr[i].value, i);
    }

    // Verify that consecutive elements are not in the same cache line
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(same_cache_line(&arr[i], &arr[i + 1]));
    }

    // Check that each element is properly aligned
    for (int i = 0; i < 4; ++i) {
        auto addr = reinterpret_cast<std::uintptr_t>(&arr[i]);
        EXPECT_EQ(addr % cls, 0u);
    }
}

TEST_F(CacheLineTest, CacheAlignedPreventsFalseSharing) {
    // Test that cache_aligned can prevent false sharing
    struct Counter {
        std::atomic<int> count{0};
    };

    struct PaddedCounter {
        cache_aligned<std::atomic<int>> count{0};
    };

    // Regular counters might share cache lines
    Counter regular_counters[2];

    // Padded counters should not share cache lines
    PaddedCounter padded_counters[2];

    // Check that padded counters are in different cache lines
    EXPECT_FALSE(same_cache_line(&padded_counters[0].count,
                                 &padded_counters[1].count));

    // Regular counters might be in the same cache line (depending on size)
    // This is not guaranteed but likely for small atomic types
    bool regular_same_line = same_cache_line(&regular_counters[0].count,
                                            &regular_counters[1].count);

    // If they're small enough, they might share a line
    if (sizeof(Counter) < cache_line_size() / 2) {
        // Small counters could share a cache line
        (void)regular_same_line; // May or may not be true
    } else {
        // Large counters shouldn't share a cache line
        EXPECT_FALSE(regular_same_line);
    }
}

TEST_F(CacheLineTest, CacheAlignedWithLargeTypes) {
    // Test with types larger than cache line
    constexpr std::size_t cls = cache_line_size();

    struct LargeStruct {
        char data[cls * 2];
    };

    using CacheAlignedLarge = cache_aligned<LargeStruct>;

    CacheAlignedLarge obj;
    obj.value.data[0] = 'A';
    obj.value.data[cls * 2 - 1] = 'Z';

    EXPECT_EQ(obj.value.data[0], 'A');
    EXPECT_EQ(obj.value.data[cls * 2 - 1], 'Z');

    // Should still be aligned
    auto addr = reinterpret_cast<std::uintptr_t>(&obj);
    EXPECT_EQ(addr % cls, 0u);
}

// Performance-related tests
TEST_F(CacheLineTest, CacheLineSizePerformance) {
    // Test that cache_line_size() is efficient (should be inlined)
    constexpr int iterations = 1000000;

    auto start = std::chrono::high_resolution_clock::now();

    std::size_t sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += cache_line_size();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be very fast (essentially free after optimization)
    EXPECT_LT(duration.count(), 10000); // Less than 10ms for 1M calls

    // Use sum to prevent optimization
    EXPECT_EQ(sum, cache_line_size() * iterations);
}

TEST_F(CacheLineTest, PadToCacheLineEdgeCases) {
    // Test with maximum values
    constexpr std::size_t cls = cache_line_size();

    // Large values
    std::size_t large = SIZE_MAX - cls;
    std::size_t padded = pad_to_cache_line(large);

    // Should handle overflow gracefully
    if (padded < large) {
        // Overflow occurred, which is acceptable
        EXPECT_EQ(padded, 0u); // Or another defined overflow behavior
    } else {
        EXPECT_GE(padded, large);
        EXPECT_EQ(padded % cls, 0u);
    }
}

// Test interaction with aligned_storage (from dependency)
TEST_F(CacheLineTest, IntegrationWithAlignedStorage) {
    constexpr std::size_t cls = cache_line_size();

    // Test that align_up (used by pad_to_cache_line) works correctly
    for (std::size_t i = 0; i <= cls * 2; ++i) {
        std::size_t padded = pad_to_cache_line(i);

        if (i == 0) {
            EXPECT_EQ(padded, 0u);
        } else if (i % cls == 0) {
            EXPECT_EQ(padded, i);
        } else {
            EXPECT_EQ(padded, ((i / cls) + 1) * cls);
        }
    }
}

// Compile-time tests using static_assert
namespace CompileTimeTests {
    // These tests run at compile time
    static_assert(cache_line_size() > 0);
    static_assert((cache_line_size() & (cache_line_size() - 1)) == 0); // Power of 2
    static_assert(cache_line_mask() == cache_line_size() - 1);
    static_assert(pad_to_cache_line(1) == cache_line_size());
    static_assert(pad_to_cache_line(cache_line_size()) == cache_line_size());
    static_assert(pad_to_cache_line(cache_line_size() + 1) == 2 * cache_line_size());
}

// Test typical use cases
TEST_F(CacheLineTest, TypicalUseCaseFalseSharing) {
    // Simulate a typical use case: preventing false sharing in multi-threaded code
    struct ThreadData {
        cache_aligned<std::atomic<int>> counter;
        cache_aligned<int> thread_id;
        cache_aligned<bool> done;
    };

    ThreadData data[4];

    // Initialize
    for (int i = 0; i < 4; ++i) {
        data[i].counter.value = 0;
        data[i].thread_id.value = i;
        data[i].done.value = false;
    }

    // Verify that each thread's data is in separate cache lines
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(same_cache_line(&data[i].counter, &data[i + 1].counter));
        EXPECT_FALSE(same_cache_line(&data[i].thread_id, &data[i + 1].thread_id));
        EXPECT_FALSE(same_cache_line(&data[i].done, &data[i + 1].done));

        // Within same thread, different fields might be in different cache lines
        EXPECT_FALSE(same_cache_line(&data[i].counter, &data[i].thread_id));
        EXPECT_FALSE(same_cache_line(&data[i].counter, &data[i].done));
        EXPECT_FALSE(same_cache_line(&data[i].thread_id, &data[i].done));
    }
}

TEST_F(CacheLineTest, TypicalUseCaseDataLayout) {
    // Test data structure layout optimization
    constexpr std::size_t cls = cache_line_size();

    struct OptimizedStruct {
        // Hot data in first cache line
        int hot_field1;
        int hot_field2;
        char padding1[cls - 2 * sizeof(int)];

        // Cold data in separate cache line
        int cold_field1;
        int cold_field2;
        char padding2[cls - 2 * sizeof(int)];
    };

    static_assert(sizeof(OptimizedStruct) >= 2 * cls);

    OptimizedStruct obj = {};

    // Hot fields should be in same cache line
    EXPECT_TRUE(same_cache_line(&obj.hot_field1, &obj.hot_field2));

    // Hot and cold fields should be in different cache lines
    EXPECT_FALSE(same_cache_line(&obj.hot_field1, &obj.cold_field1));
    EXPECT_FALSE(same_cache_line(&obj.hot_field2, &obj.cold_field2));

    // Cold fields should be in same cache line
    EXPECT_TRUE(same_cache_line(&obj.cold_field1, &obj.cold_field2));
}

} // namespace
} // namespace fem::core::memory
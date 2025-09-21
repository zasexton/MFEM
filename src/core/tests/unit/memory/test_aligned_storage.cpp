#include <gtest/gtest.h>
#include <core/memory/aligned_storage.h>
#include <cstdint>
#include <vector>
#include <memory>

using namespace fem::core::memory;

class AlignedStorageTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// is_power_of_two tests
TEST_F(AlignedStorageTest, IsPowerOfTwo_ValidPowers) {
    EXPECT_TRUE(is_power_of_two(1));
    EXPECT_TRUE(is_power_of_two(2));
    EXPECT_TRUE(is_power_of_two(4));
    EXPECT_TRUE(is_power_of_two(8));
    EXPECT_TRUE(is_power_of_two(16));
    EXPECT_TRUE(is_power_of_two(32));
    EXPECT_TRUE(is_power_of_two(64));
    EXPECT_TRUE(is_power_of_two(128));
    EXPECT_TRUE(is_power_of_two(256));
    EXPECT_TRUE(is_power_of_two(1024));
    EXPECT_TRUE(is_power_of_two(4096));
    EXPECT_TRUE(is_power_of_two(size_t(1) << 20));
    EXPECT_TRUE(is_power_of_two(size_t(1) << 30));
}

TEST_F(AlignedStorageTest, IsPowerOfTwo_InvalidValues) {
    EXPECT_FALSE(is_power_of_two(0));
    EXPECT_FALSE(is_power_of_two(3));
    EXPECT_FALSE(is_power_of_two(5));
    EXPECT_FALSE(is_power_of_two(6));
    EXPECT_FALSE(is_power_of_two(7));
    EXPECT_FALSE(is_power_of_two(9));
    EXPECT_FALSE(is_power_of_two(10));
    EXPECT_FALSE(is_power_of_two(15));
    EXPECT_FALSE(is_power_of_two(17));
    EXPECT_FALSE(is_power_of_two(100));
    EXPECT_FALSE(is_power_of_two(127));
    EXPECT_FALSE(is_power_of_two(1023));
}

// align_up tests for size_t
TEST_F(AlignedStorageTest, AlignUp_SizeT_PowerOfTwo) {
    // Align to 8
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(8u)));
    EXPECT_EQ(8u, align_up(std::size_t(1u), std::size_t(8u)));
    EXPECT_EQ(8u, align_up(std::size_t(7u), std::size_t(8u)));
    EXPECT_EQ(8u, align_up(std::size_t(8u), std::size_t(8u)));
    EXPECT_EQ(16u, align_up(std::size_t(9u), std::size_t(8u)));
    EXPECT_EQ(16u, align_up(std::size_t(15u), std::size_t(8u)));
    EXPECT_EQ(16u, align_up(std::size_t(16u), std::size_t(8u)));

    // Align to 16
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(16u)));
    EXPECT_EQ(16u, align_up(std::size_t(1u), std::size_t(16u)));
    EXPECT_EQ(16u, align_up(std::size_t(15u), std::size_t(16u)));
    EXPECT_EQ(16u, align_up(std::size_t(16u), std::size_t(16u)));
    EXPECT_EQ(32u, align_up(std::size_t(17u), std::size_t(16u)));

    // Align to 64 (cache line)
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(64u)));
    EXPECT_EQ(64u, align_up(std::size_t(1u), std::size_t(64u)));
    EXPECT_EQ(64u, align_up(std::size_t(63u), std::size_t(64u)));
    EXPECT_EQ(64u, align_up(std::size_t(64u), std::size_t(64u)));
    EXPECT_EQ(128u, align_up(std::size_t(65u), std::size_t(64u)));

    // Align to 4096 (page)
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(4096u)));
    EXPECT_EQ(4096u, align_up(std::size_t(1u), std::size_t(4096u)));
    EXPECT_EQ(4096u, align_up(std::size_t(4095u), std::size_t(4096u)));
    EXPECT_EQ(4096u, align_up(std::size_t(4096u), std::size_t(4096u)));
    EXPECT_EQ(8192u, align_up(std::size_t(4097u), std::size_t(4096u)));
}

TEST_F(AlignedStorageTest, AlignUp_SizeT_NonPowerOfTwo) {
    // Non-power-of-two alignments should still work (slower path)
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(3u)));
    EXPECT_EQ(3u, align_up(std::size_t(1u), std::size_t(3u)));
    EXPECT_EQ(3u, align_up(std::size_t(2u), std::size_t(3u)));
    EXPECT_EQ(3u, align_up(std::size_t(3u), std::size_t(3u)));
    EXPECT_EQ(6u, align_up(std::size_t(4u), std::size_t(3u)));

    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(10u)));
    EXPECT_EQ(10u, align_up(std::size_t(1u), std::size_t(10u)));
    EXPECT_EQ(10u, align_up(std::size_t(9u), std::size_t(10u)));
    EXPECT_EQ(10u, align_up(std::size_t(10u), std::size_t(10u)));
    EXPECT_EQ(20u, align_up(std::size_t(11u), std::size_t(10u)));
}

// align_down tests for size_t
TEST_F(AlignedStorageTest, AlignDown_SizeT_PowerOfTwo) {
    // Align to 8
    EXPECT_EQ(0u, align_down(std::size_t(0u), std::size_t(8u)));
    EXPECT_EQ(0u, align_down(std::size_t(1u), std::size_t(8u)));
    EXPECT_EQ(0u, align_down(std::size_t(7u), std::size_t(8u)));
    EXPECT_EQ(8u, align_down(std::size_t(8u), std::size_t(8u)));
    EXPECT_EQ(8u, align_down(std::size_t(9u), std::size_t(8u)));
    EXPECT_EQ(8u, align_down(std::size_t(15u), std::size_t(8u)));
    EXPECT_EQ(16u, align_down(std::size_t(16u), std::size_t(8u)));

    // Align to 64
    EXPECT_EQ(0u, align_down(std::size_t(0u), std::size_t(64u)));
    EXPECT_EQ(0u, align_down(std::size_t(1u), std::size_t(64u)));
    EXPECT_EQ(0u, align_down(std::size_t(63u), std::size_t(64u)));
    EXPECT_EQ(64u, align_down(std::size_t(64u), std::size_t(64u)));
    EXPECT_EQ(64u, align_down(std::size_t(65u), std::size_t(64u)));
    EXPECT_EQ(64u, align_down(std::size_t(127u), std::size_t(64u)));
    EXPECT_EQ(128u, align_down(std::size_t(128u), std::size_t(64u)));
}

TEST_F(AlignedStorageTest, AlignDown_SizeT_NonPowerOfTwo) {
    EXPECT_EQ(0u, align_down(std::size_t(0u), std::size_t(3u)));
    EXPECT_EQ(0u, align_down(std::size_t(1u), std::size_t(3u)));
    EXPECT_EQ(0u, align_down(std::size_t(2u), std::size_t(3u)));
    EXPECT_EQ(3u, align_down(std::size_t(3u), std::size_t(3u)));
    EXPECT_EQ(3u, align_down(std::size_t(4u), std::size_t(3u)));
    EXPECT_EQ(3u, align_down(std::size_t(5u), std::size_t(3u)));
    EXPECT_EQ(6u, align_down(std::size_t(6u), std::size_t(3u)));
}

// align_up tests for pointers
TEST_F(AlignedStorageTest, AlignUp_Pointer) {
    // Create a byte buffer to test alignment
    std::vector<std::byte> buffer(256);

    // Test aligning void*
    void* p1 = &buffer[1];
    void* aligned8 = align_up(p1, 8);
    EXPECT_TRUE(is_aligned(aligned8, 8));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned8), reinterpret_cast<uintptr_t>(p1));

    void* p2 = &buffer[17];
    void* aligned16 = align_up(p2, 16);
    EXPECT_TRUE(is_aligned(aligned16, 16));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned16), reinterpret_cast<uintptr_t>(p2));

    void* p3 = &buffer[33];
    void* aligned64 = align_up(p3, 64);
    EXPECT_TRUE(is_aligned(aligned64, 64));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned64), reinterpret_cast<uintptr_t>(p3));
}

TEST_F(AlignedStorageTest, AlignUp_ConstPointer) {
    std::vector<std::byte> buffer(256);

    const void* p1 = &buffer[5];
    const void* aligned = align_up(p1, 16);
    EXPECT_TRUE(is_aligned(aligned, 16));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned), reinterpret_cast<uintptr_t>(p1));
}

TEST_F(AlignedStorageTest, AlignUp_TypedPointer) {
    std::vector<int> buffer(64);

    int* p1 = &buffer[1];
    int* aligned = align_up(p1, 16);
    EXPECT_TRUE(is_aligned(aligned, 16));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned), reinterpret_cast<uintptr_t>(p1));
}

// is_aligned tests
TEST_F(AlignedStorageTest, IsAligned_Void) {
    std::vector<std::byte> buffer(256);

    // Get an aligned address
    void* base = align_up(buffer.data(), 64);

    EXPECT_TRUE(is_aligned(base, 1));
    EXPECT_TRUE(is_aligned(base, 2));
    EXPECT_TRUE(is_aligned(base, 4));
    EXPECT_TRUE(is_aligned(base, 8));
    EXPECT_TRUE(is_aligned(base, 16));
    EXPECT_TRUE(is_aligned(base, 32));
    EXPECT_TRUE(is_aligned(base, 64));

    // Offset by 1
    void* offset1 = static_cast<std::byte*>(base) + 1;
    EXPECT_TRUE(is_aligned(offset1, 1));
    EXPECT_FALSE(is_aligned(offset1, 2));
    EXPECT_FALSE(is_aligned(offset1, 4));
    EXPECT_FALSE(is_aligned(offset1, 8));

    // Offset by 8
    void* offset8 = static_cast<std::byte*>(base) + 8;
    EXPECT_TRUE(is_aligned(offset8, 1));
    EXPECT_TRUE(is_aligned(offset8, 2));
    EXPECT_TRUE(is_aligned(offset8, 4));
    EXPECT_TRUE(is_aligned(offset8, 8));
    EXPECT_FALSE(is_aligned(offset8, 16));
    EXPECT_FALSE(is_aligned(offset8, 64));
}

TEST_F(AlignedStorageTest, IsAligned_TypedPointer) {
    alignas(64) int values[16] = {};

    EXPECT_TRUE(is_aligned(&values[0], 4));
    EXPECT_TRUE(is_aligned(&values[0], 8));
    EXPECT_TRUE(is_aligned(&values[0], 16));
    EXPECT_TRUE(is_aligned(&values[0], 64));

    // values[1] is 4 bytes offset from aligned base
    EXPECT_TRUE(is_aligned(&values[1], 4));
    EXPECT_FALSE(is_aligned(&values[1], 8));
    EXPECT_FALSE(is_aligned(&values[1], 16));
}

// assume_aligned tests
TEST_F(AlignedStorageTest, AssumeAligned) {
    alignas(64) int buffer[16];

    int* p1 = assume_aligned<64>(&buffer[0]);
    EXPECT_EQ(&buffer[0], p1);
    EXPECT_TRUE(is_aligned(p1, 64));

    const int* p2 = assume_aligned<16>(&buffer[0]);
    EXPECT_EQ(&buffer[0], p2);
    EXPECT_TRUE(is_aligned(p2, 16));
}

// AlignedBuffer tests
TEST_F(AlignedStorageTest, AlignedBuffer_DefaultAlignment) {
    AlignedBuffer<128> buffer{};

    EXPECT_EQ(128u, buffer.size());
    EXPECT_EQ(buffer.data, buffer.begin());
    EXPECT_EQ(buffer.data + 128, buffer.end());
    EXPECT_TRUE(is_aligned(buffer.begin(), alignof(std::max_align_t)));
}

TEST_F(AlignedStorageTest, AlignedBuffer_CustomAlignment) {
    AlignedBuffer<256, 64> buffer{};

    EXPECT_EQ(256u, buffer.size());
    EXPECT_TRUE(is_aligned(buffer.begin(), 64));

    // Fill buffer and verify
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer.data[i] = std::byte(i & 0xFF);
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(std::byte(i & 0xFF), buffer.data[i]);
    }
}

TEST_F(AlignedStorageTest, AlignedBuffer_ConstMethods) {
    const AlignedBuffer<64, 32> buffer{};

    EXPECT_EQ(64u, buffer.size());
    EXPECT_TRUE(is_aligned(buffer.begin(), 32));

    const std::byte* begin = buffer.begin();
    const std::byte* end = buffer.end();
    EXPECT_EQ(64, end - begin);
}

// CacheAligned tests
TEST_F(AlignedStorageTest, CacheAligned_DefaultConstruction) {
    CacheAligned<int> aligned_int;

    EXPECT_EQ(0, aligned_int.value);
    EXPECT_TRUE(is_aligned(&aligned_int.value, kCacheLineSize));
}

TEST_F(AlignedStorageTest, CacheAligned_ValueConstruction) {
    CacheAligned<int> aligned_int(42);

    EXPECT_EQ(42, aligned_int.value);
    EXPECT_TRUE(is_aligned(&aligned_int.value, kCacheLineSize));
}

TEST_F(AlignedStorageTest, CacheAligned_ComplexType) {
    struct TestStruct {
        int x;
        double y;
        char z[17];

        TestStruct(int a, double b) : x(a), y(b) {
            for (size_t i = 0; i < sizeof(z); ++i) {
                z[i] = static_cast<char>(i);
            }
        }
    };

    CacheAligned<TestStruct> aligned_struct(10, 3.14);

    EXPECT_EQ(10, aligned_struct.value.x);
    EXPECT_DOUBLE_EQ(3.14, aligned_struct.value.y);
    EXPECT_TRUE(is_aligned(&aligned_struct.value, kCacheLineSize));
}

TEST_F(AlignedStorageTest, CacheAligned_ArrowOperator) {
    struct Point {
        int x, y;
        int sum() const { return x + y; }
    };

    CacheAligned<Point> aligned_point;
    aligned_point.value.x = 3;
    aligned_point.value.y = 4;

    EXPECT_EQ(3, aligned_point->x);
    EXPECT_EQ(4, aligned_point->y);
    EXPECT_EQ(7, aligned_point->sum());

    const CacheAligned<Point>& const_ref = aligned_point;
    EXPECT_EQ(3, const_ref->x);
    EXPECT_EQ(7, const_ref->sum());
}

TEST_F(AlignedStorageTest, CacheAligned_CustomAlignment) {
    CacheAligned<int, 256> aligned_int(100);

    EXPECT_EQ(100, aligned_int.value);
    EXPECT_TRUE(is_aligned(&aligned_int.value, 256));
}

TEST_F(AlignedStorageTest, CacheAligned_Array) {
    // Array of cache-aligned integers
    CacheAligned<int> values[4];

    for (int i = 0; i < 4; ++i) {
        values[i].value = i * i;
        EXPECT_TRUE(is_aligned(&values[i].value, kCacheLineSize));
    }

    // Verify values
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i * i, values[i].value);
    }

    // Check that each element is in a different cache line (no false sharing)
    for (int i = 0; i < 3; ++i) {
        auto addr1 = reinterpret_cast<uintptr_t>(&values[i].value);
        auto addr2 = reinterpret_cast<uintptr_t>(&values[i + 1].value);
        EXPECT_GE(addr2 - addr1, kCacheLineSize);
    }
}

// Constants tests
TEST_F(AlignedStorageTest, Constants) {
    EXPECT_EQ(fem::config::DEFAULT_ALIGNMENT, kDefaultAlignment);
    EXPECT_EQ(fem::config::CACHE_LINE_SIZE, kCacheLineSize);

    EXPECT_TRUE(is_power_of_two(kDefaultAlignment));
    EXPECT_TRUE(is_power_of_two(kCacheLineSize));
}

// Edge cases
TEST_F(AlignedStorageTest, EdgeCases_ZeroAlignment) {
    // align_up with alignment of 1 should return the value unchanged
    EXPECT_EQ(0u, align_up(std::size_t(0u), std::size_t(1u)));
    EXPECT_EQ(5u, align_up(std::size_t(5u), std::size_t(1u)));
    EXPECT_EQ(100u, align_up(std::size_t(100u), std::size_t(1u)));

    EXPECT_EQ(0u, align_down(std::size_t(0u), std::size_t(1u)));
    EXPECT_EQ(5u, align_down(std::size_t(5u), std::size_t(1u)));
    EXPECT_EQ(100u, align_down(std::size_t(100u), std::size_t(1u)));
}

TEST_F(AlignedStorageTest, EdgeCases_LargeAlignment) {
    size_t large_align = size_t(1) << 20;  // 1MB alignment

    EXPECT_EQ(0u, align_up(std::size_t(0u), large_align));
    EXPECT_EQ(large_align, align_up(std::size_t(1u), large_align));
    EXPECT_EQ(large_align, align_up(large_align - 1, large_align));
    EXPECT_EQ(large_align, align_up(large_align, large_align));
    EXPECT_EQ(2 * large_align, align_up(large_align + 1, large_align));
}

TEST_F(AlignedStorageTest, EdgeCases_MaxValue) {
    size_t max_val = std::numeric_limits<size_t>::max();

    // These should not overflow
    // max_val is typically 0xFFFFFFFFFFFFFFFF which when aligned down to 8 becomes 0xFFFFFFFFFFFFFFF8
    EXPECT_EQ(max_val - (max_val % 8), align_down(max_val, std::size_t(8u)));

    // max_val - 7 is 0xFFFFFFFFFFFFFFF8 which is already aligned to 8
    EXPECT_EQ(max_val - 7, align_down(max_val - 7, std::size_t(8u)));
}

// Compile-time tests (these verify constexpr)
TEST_F(AlignedStorageTest, ConstexprFunctions) {
    constexpr bool power_2 = is_power_of_two(64);
    constexpr bool not_power_2 = is_power_of_two(63);

    EXPECT_TRUE(power_2);
    EXPECT_FALSE(not_power_2);

    constexpr size_t aligned_val = align_up(std::size_t(100u), std::size_t(16u));
    EXPECT_EQ(112u, aligned_val);

    constexpr size_t aligned_down_val = align_down(std::size_t(100u), std::size_t(16u));
    EXPECT_EQ(96u, aligned_down_val);
}
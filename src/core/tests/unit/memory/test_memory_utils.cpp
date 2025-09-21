#include <gtest/gtest.h>
#include <core/memory/memory_utils.h>
#include <vector>
#include <array>
#include <limits>

using namespace fem::core::memory;

class MemoryUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Overflow check tests - add_would_overflow
TEST_F(MemoryUtilsTest, AddWouldOverflow_NoOverflow) {
    EXPECT_FALSE(add_would_overflow(0u, 0u));
    EXPECT_FALSE(add_would_overflow(1u, 1u));
    EXPECT_FALSE(add_would_overflow(100u, 200u));
    EXPECT_FALSE(add_would_overflow(1000u, 2000u));

    size_t max = std::numeric_limits<size_t>::max();
    EXPECT_FALSE(add_would_overflow(0u, max));
    EXPECT_FALSE(add_would_overflow(max, 0u));
    EXPECT_FALSE(add_would_overflow(max / 2, max / 2));
}

TEST_F(MemoryUtilsTest, AddWouldOverflow_Overflow) {
    size_t max = std::numeric_limits<size_t>::max();

    EXPECT_TRUE(add_would_overflow(max, 1u));
    EXPECT_TRUE(add_would_overflow(1u, max));
    EXPECT_TRUE(add_would_overflow(max, max));
    EXPECT_TRUE(add_would_overflow(max - 10, 11u));
    EXPECT_TRUE(add_would_overflow(max / 2 + 1, max / 2 + 1));
}

// Overflow check tests - mul_would_overflow
TEST_F(MemoryUtilsTest, MulWouldOverflow_NoOverflow) {
    EXPECT_FALSE(mul_would_overflow(0u, 0u));
    EXPECT_FALSE(mul_would_overflow(0u, 100u));
    EXPECT_FALSE(mul_would_overflow(100u, 0u));
    EXPECT_FALSE(mul_would_overflow(1u, 1u));
    EXPECT_FALSE(mul_would_overflow(10u, 10u));
    EXPECT_FALSE(mul_would_overflow(1000u, 1000u));

    size_t max = std::numeric_limits<size_t>::max();
    EXPECT_FALSE(mul_would_overflow(1u, max));
    EXPECT_FALSE(mul_would_overflow(max, 1u));
}

TEST_F(MemoryUtilsTest, MulWouldOverflow_Overflow) {
    size_t max = std::numeric_limits<size_t>::max();

    EXPECT_TRUE(mul_would_overflow(max, 2u));
    EXPECT_TRUE(mul_would_overflow(2u, max));
    EXPECT_TRUE(mul_would_overflow(max, max));

    size_t large = size_t(1) << 32;  // 4GB on 64-bit
    if (sizeof(size_t) == 8) {
        EXPECT_FALSE(mul_would_overflow(large, large / 2));
        EXPECT_TRUE(mul_would_overflow(large, large));
    }

    // Square root of max should overflow when squared
    // For 64-bit: sqrt(2^64) ≈ 2^32, so 2^32 * 2^32 = 2^64 which overflows
    // For 32-bit: sqrt(2^32) ≈ 2^16, so 2^16 * 2^16 = 2^32 which overflows
    size_t sqrt_max = size_t(1) << (sizeof(size_t) * 4);
    EXPECT_TRUE(mul_would_overflow(sqrt_max, sqrt_max));

    // But something smaller should be safe
    size_t sqrt_max_minus_1 = sqrt_max - 1;
    EXPECT_FALSE(mul_would_overflow(sqrt_max_minus_1, sqrt_max_minus_1));
}

// size_add tests
TEST_F(MemoryUtilsTest, SizeAdd_Success) {
    size_t result;

    EXPECT_TRUE(size_add(0u, 0u, result));
    EXPECT_EQ(0u, result);

    EXPECT_TRUE(size_add(100u, 200u, result));
    EXPECT_EQ(300u, result);

    EXPECT_TRUE(size_add(1000u, 2000u, result));
    EXPECT_EQ(3000u, result);

    size_t max = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(size_add(0u, max, result));
    EXPECT_EQ(max, result);

    EXPECT_TRUE(size_add(max / 2, max / 2, result));
    EXPECT_EQ(max - 1, result);  // max is odd, so max/2 + max/2 = max - 1
}

TEST_F(MemoryUtilsTest, SizeAdd_Failure) {
    size_t result = 0;
    size_t max = std::numeric_limits<size_t>::max();

    EXPECT_FALSE(size_add(max, 1u, result));
    EXPECT_EQ(0u, result);  // Result should be unchanged on failure

    EXPECT_FALSE(size_add(max, max, result));
    EXPECT_EQ(0u, result);

    EXPECT_FALSE(size_add(max - 10, 11u, result));
    EXPECT_EQ(0u, result);
}

// size_mul tests
TEST_F(MemoryUtilsTest, SizeMul_Success) {
    size_t result;

    EXPECT_TRUE(size_mul(0u, 0u, result));
    EXPECT_EQ(0u, result);

    EXPECT_TRUE(size_mul(0u, 100u, result));
    EXPECT_EQ(0u, result);

    EXPECT_TRUE(size_mul(10u, 20u, result));
    EXPECT_EQ(200u, result);

    EXPECT_TRUE(size_mul(1000u, 1000u, result));
    EXPECT_EQ(1000000u, result);

    size_t max = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(size_mul(1u, max, result));
    EXPECT_EQ(max, result);
}

TEST_F(MemoryUtilsTest, SizeMul_Failure) {
    size_t result = 0;
    size_t max = std::numeric_limits<size_t>::max();

    EXPECT_FALSE(size_mul(max, 2u, result));
    EXPECT_EQ(0u, result);

    EXPECT_FALSE(size_mul(max, max, result));
    EXPECT_EQ(0u, result);

    size_t large = size_t(1) << 32;
    if (sizeof(size_t) == 8) {
        EXPECT_FALSE(size_mul(large, large, result));
        EXPECT_EQ(0u, result);
    }
}

// checked_narrow tests
TEST_F(MemoryUtilsTest, CheckedNarrow_SameSignedTypes) {
    // int64 to int32
    int32_t result32;

    EXPECT_TRUE(checked_narrow<int32_t>(int64_t(0), result32));
    EXPECT_EQ(0, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int64_t(100), result32));
    EXPECT_EQ(100, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int64_t(-100), result32));
    EXPECT_EQ(-100, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int64_t(INT32_MAX), result32));
    EXPECT_EQ(INT32_MAX, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int64_t(INT32_MIN), result32));
    EXPECT_EQ(INT32_MIN, result32);

    // Out of range
    EXPECT_FALSE(checked_narrow<int32_t>(int64_t(INT32_MAX) + 1, result32));
    EXPECT_FALSE(checked_narrow<int32_t>(int64_t(INT32_MIN) - 1, result32));
}

TEST_F(MemoryUtilsTest, CheckedNarrow_SameUnsignedTypes) {
    // uint64 to uint32
    uint32_t result32;

    EXPECT_TRUE(checked_narrow<uint32_t>(uint64_t(0), result32));
    EXPECT_EQ(0u, result32);

    EXPECT_TRUE(checked_narrow<uint32_t>(uint64_t(100), result32));
    EXPECT_EQ(100u, result32);

    EXPECT_TRUE(checked_narrow<uint32_t>(uint64_t(UINT32_MAX), result32));
    EXPECT_EQ(UINT32_MAX, result32);

    // Out of range
    EXPECT_FALSE(checked_narrow<uint32_t>(uint64_t(UINT32_MAX) + 1, result32));
    EXPECT_FALSE(checked_narrow<uint32_t>(UINT64_MAX, result32));
}

TEST_F(MemoryUtilsTest, CheckedNarrow_SignedToUnsigned) {
    // int32 to uint16
    uint16_t result16;

    EXPECT_TRUE(checked_narrow<uint16_t>(int32_t(0), result16));
    EXPECT_EQ(0u, result16);

    EXPECT_TRUE(checked_narrow<uint16_t>(int32_t(100), result16));
    EXPECT_EQ(100u, result16);

    EXPECT_TRUE(checked_narrow<uint16_t>(int32_t(UINT16_MAX), result16));
    EXPECT_EQ(UINT16_MAX, result16);

    // Negative values should fail
    EXPECT_FALSE(checked_narrow<uint16_t>(int32_t(-1), result16));
    EXPECT_FALSE(checked_narrow<uint16_t>(int32_t(-100), result16));

    // Out of range positive
    EXPECT_FALSE(checked_narrow<uint16_t>(int32_t(UINT16_MAX) + 1, result16));
}

TEST_F(MemoryUtilsTest, CheckedNarrow_UnsignedToSigned) {
    // uint32 to int16
    int16_t result16;

    EXPECT_TRUE(checked_narrow<int16_t>(uint32_t(0), result16));
    EXPECT_EQ(0, result16);

    EXPECT_TRUE(checked_narrow<int16_t>(uint32_t(100), result16));
    EXPECT_EQ(100, result16);

    EXPECT_TRUE(checked_narrow<int16_t>(uint32_t(INT16_MAX), result16));
    EXPECT_EQ(INT16_MAX, result16);

    // Out of range
    EXPECT_FALSE(checked_narrow<int16_t>(uint32_t(INT16_MAX) + 1, result16));
    EXPECT_FALSE(checked_narrow<int16_t>(uint32_t(UINT16_MAX), result16));
    EXPECT_FALSE(checked_narrow<int16_t>(UINT32_MAX, result16));
}

TEST_F(MemoryUtilsTest, CheckedNarrow_WideningIsAllowed) {
    // int16 to int32 (widening should always succeed)
    int32_t result32;

    EXPECT_TRUE(checked_narrow<int32_t>(int16_t(0), result32));
    EXPECT_EQ(0, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int16_t(INT16_MAX), result32));
    EXPECT_EQ(INT16_MAX, result32);

    EXPECT_TRUE(checked_narrow<int32_t>(int16_t(INT16_MIN), result32));
    EXPECT_EQ(INT16_MIN, result32);

    // uint16 to uint32 (widening)
    uint32_t uresult32;

    EXPECT_TRUE(checked_narrow<uint32_t>(uint16_t(0), uresult32));
    EXPECT_EQ(0u, uresult32);

    EXPECT_TRUE(checked_narrow<uint32_t>(uint16_t(UINT16_MAX), uresult32));
    EXPECT_EQ(UINT16_MAX, uresult32);
}

// as_bytes tests
TEST_F(MemoryUtilsTest, AsBytes_ConstSpan) {
    std::array<int, 4> data = {1, 2, 3, 4};
    std::span<const int> int_span(data);

    auto byte_span = as_bytes(int_span);

    EXPECT_EQ(data.size() * sizeof(int), byte_span.size());
    EXPECT_EQ(reinterpret_cast<const std::byte*>(data.data()), byte_span.data());

    // Verify data is accessible
    const int* as_ints = reinterpret_cast<const int*>(byte_span.data());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], as_ints[i]);
    }
}

TEST_F(MemoryUtilsTest, AsBytes_EmptySpan) {
    std::span<const int> empty_span;

    auto byte_span = as_bytes(empty_span);

    EXPECT_EQ(0u, byte_span.size());
    EXPECT_EQ(nullptr, byte_span.data());
}

TEST_F(MemoryUtilsTest, AsBytes_Struct) {
    struct TestStruct {
        int x;
        double y;
        char z[8];
    };

    TestStruct data = {42, 3.14, "hello"};
    std::span<const TestStruct> struct_span(&data, 1);

    auto byte_span = as_bytes(struct_span);

    EXPECT_EQ(sizeof(TestStruct), byte_span.size());
    EXPECT_EQ(reinterpret_cast<const std::byte*>(&data), byte_span.data());
}

// as_writable_bytes tests
TEST_F(MemoryUtilsTest, AsWritableBytes_MutableSpan) {
    std::array<int, 4> data = {1, 2, 3, 4};
    std::span<int> int_span(data);

    auto byte_span = as_writable_bytes(int_span);

    EXPECT_EQ(data.size() * sizeof(int), byte_span.size());
    EXPECT_EQ(reinterpret_cast<std::byte*>(data.data()), byte_span.data());

    // Modify through byte span
    int* as_ints = reinterpret_cast<int*>(byte_span.data());
    as_ints[0] = 100;
    as_ints[1] = 200;

    EXPECT_EQ(100, data[0]);
    EXPECT_EQ(200, data[1]);
}

TEST_F(MemoryUtilsTest, AsWritableBytes_EmptySpan) {
    std::span<int> empty_span;

    auto byte_span = as_writable_bytes(empty_span);

    EXPECT_EQ(0u, byte_span.size());
    EXPECT_EQ(nullptr, byte_span.data());
}

// Alignment wrapper tests
TEST_F(MemoryUtilsTest, AlignUpSize_Wrapper) {
    EXPECT_EQ(0u, align_up_size(0u, 8u));
    EXPECT_EQ(8u, align_up_size(1u, 8u));
    EXPECT_EQ(8u, align_up_size(7u, 8u));
    EXPECT_EQ(8u, align_up_size(8u, 8u));
    EXPECT_EQ(16u, align_up_size(9u, 8u));

    EXPECT_EQ(64u, align_up_size(1u, 64u));
    EXPECT_EQ(64u, align_up_size(63u, 64u));
    EXPECT_EQ(64u, align_up_size(64u, 64u));
}

TEST_F(MemoryUtilsTest, AlignDownSize_Wrapper) {
    EXPECT_EQ(0u, align_down_size(0u, 8u));
    EXPECT_EQ(0u, align_down_size(1u, 8u));
    EXPECT_EQ(0u, align_down_size(7u, 8u));
    EXPECT_EQ(8u, align_down_size(8u, 8u));
    EXPECT_EQ(8u, align_down_size(15u, 8u));

    EXPECT_EQ(0u, align_down_size(63u, 64u));
    EXPECT_EQ(64u, align_down_size(64u, 64u));
    EXPECT_EQ(64u, align_down_size(127u, 64u));
}

TEST_F(MemoryUtilsTest, AlignUpPtr_Wrapper) {
    std::vector<std::byte> buffer(256);

    int* p = reinterpret_cast<int*>(&buffer[5]);
    int* aligned = align_up_ptr(p, 16);

    EXPECT_TRUE(is_aligned(aligned, 16));
    EXPECT_GE(reinterpret_cast<uintptr_t>(aligned), reinterpret_cast<uintptr_t>(p));
}

TEST_F(MemoryUtilsTest, IsAlignedPtr_Wrapper) {
    alignas(64) int buffer[16] = {};

    EXPECT_TRUE(is_aligned_ptr(&buffer[0], 4));
    EXPECT_TRUE(is_aligned_ptr(&buffer[0], 8));
    EXPECT_TRUE(is_aligned_ptr(&buffer[0], 16));
    EXPECT_TRUE(is_aligned_ptr(&buffer[0], 64));

    EXPECT_TRUE(is_aligned_ptr(&buffer[1], 4));
    EXPECT_FALSE(is_aligned_ptr(&buffer[1], 8));
    EXPECT_FALSE(is_aligned_ptr(&buffer[1], 16));
    EXPECT_FALSE(is_aligned_ptr(&buffer[1], 64));
}

// Constexpr tests
TEST_F(MemoryUtilsTest, ConstexprFunctions) {
    // Test that these functions can be used at compile time
    constexpr bool add_ok = !add_would_overflow(100u, 200u);
    EXPECT_TRUE(add_ok);

    constexpr bool mul_ok = !mul_would_overflow(10u, 20u);
    EXPECT_TRUE(mul_ok);

    constexpr size_t aligned = align_up_size(100u, 16u);
    EXPECT_EQ(112u, aligned);

    constexpr size_t aligned_down = align_down_size(100u, 16u);
    EXPECT_EQ(96u, aligned_down);
}

// Edge cases
TEST_F(MemoryUtilsTest, EdgeCases_ZeroSizes) {
    size_t result;

    EXPECT_TRUE(size_add(0u, 0u, result));
    EXPECT_EQ(0u, result);

    EXPECT_TRUE(size_mul(0u, 0u, result));
    EXPECT_EQ(0u, result);

    EXPECT_TRUE(size_mul(0u, std::numeric_limits<size_t>::max(), result));
    EXPECT_EQ(0u, result);
}

TEST_F(MemoryUtilsTest, EdgeCases_CheckedNarrowSameType) {
    int32_t result;

    EXPECT_TRUE(checked_narrow<int32_t>(int32_t(100), result));
    EXPECT_EQ(100, result);

    EXPECT_TRUE(checked_narrow<int32_t>(int32_t(INT32_MAX), result));
    EXPECT_EQ(INT32_MAX, result);

    EXPECT_TRUE(checked_narrow<int32_t>(int32_t(INT32_MIN), result));
    EXPECT_EQ(INT32_MIN, result);
}

TEST_F(MemoryUtilsTest, EdgeCases_ByteSpanFromSingleElement) {
    int value = 42;
    std::span<const int> int_span(&value, 1);

    auto byte_span = as_bytes(int_span);

    EXPECT_EQ(sizeof(int), byte_span.size());
    EXPECT_EQ(reinterpret_cast<const std::byte*>(&value), byte_span.data());

    // Verify we can read the value back
    const int* read_back = reinterpret_cast<const int*>(byte_span.data());
    EXPECT_EQ(42, *read_back);
}
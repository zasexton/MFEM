// test_broadcast_base.cpp - Unit tests for broadcast functionality
#include <gtest/gtest.h>
#include <base/broadcast_base.h>
#include <base/numeric_base.h>
#include <vector>
#include <numeric>

using namespace fem::numeric;

// ============================================================================
// BroadcastHelper Tests
// ============================================================================

TEST(BroadcastHelper, AreCompatible_SameShape) {
    Shape shape1({3, 4, 5});
    Shape shape2({3, 4, 5});

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape1, shape2));
}

TEST(BroadcastHelper, AreCompatible_ScalarWithAny) {
    Shape scalar({});  // Scalar (rank 0)
    Shape matrix({3, 4});

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(scalar, matrix));
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(matrix, scalar));
}

TEST(BroadcastHelper, AreCompatible_OneDimension) {
    Shape shape1({1, 4});
    Shape shape2({3, 4});

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape1, shape2));
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape2, shape1));
}

TEST(BroadcastHelper, AreCompatible_DifferentRanks) {
    Shape shape1({5});        // 1D
    Shape shape2({3, 5});     // 2D
    Shape shape3({2, 3, 5});  // 3D

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape1, shape2));
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape1, shape3));
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape2, shape3));
}

TEST(BroadcastHelper, AreCompatible_BroadcastableOnes) {
    Shape shape1({1, 3, 1});
    Shape shape2({2, 1, 4});

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(shape1, shape2));
}

TEST(BroadcastHelper, AreIncompatible_DifferentDimensions) {
    Shape shape1({3, 4});
    Shape shape2({3, 5});  // Different second dimension

    EXPECT_FALSE(BroadcastHelper::are_broadcastable(shape1, shape2));
}

TEST(BroadcastHelper, AreIncompatible_NonBroadcastable) {
    Shape shape1({3, 4});
    Shape shape2({5, 6});

    EXPECT_FALSE(BroadcastHelper::are_broadcastable(shape1, shape2));
}

// ============================================================================
// Broadcast Shape Computation Tests
// ============================================================================

TEST(BroadcastHelper, BroadcastShape_SameShape) {
    Shape shape1({3, 4});
    Shape shape2({3, 4});

    Shape result = BroadcastHelper::broadcast_shape(shape1, shape2);
    EXPECT_EQ(result, Shape({3, 4}));
}

TEST(BroadcastHelper, BroadcastShape_ScalarWithMatrix) {
    Shape scalar({});
    Shape matrix({3, 4});

    Shape result = BroadcastHelper::broadcast_shape(scalar, matrix);
    EXPECT_EQ(result, Shape({3, 4}));
}

TEST(BroadcastHelper, BroadcastShape_OneDimBroadcast) {
    Shape shape1({1, 5});
    Shape shape2({3, 5});

    Shape result = BroadcastHelper::broadcast_shape(shape1, shape2);
    EXPECT_EQ(result, Shape({3, 5}));
}

TEST(BroadcastHelper, BroadcastShape_MultiDimBroadcast) {
    Shape shape1({1, 3, 1});
    Shape shape2({2, 1, 4});

    Shape result = BroadcastHelper::broadcast_shape(shape1, shape2);
    EXPECT_EQ(result, Shape({2, 3, 4}));
}

TEST(BroadcastHelper, BroadcastShape_DifferentRanks) {
    Shape shape1({5});      // (5,)
    Shape shape2({3, 5});   // (3, 5)

    Shape result = BroadcastHelper::broadcast_shape(shape1, shape2);
    EXPECT_EQ(result, Shape({3, 5}));
}

TEST(BroadcastHelper, BroadcastShape_ComplexBroadcast) {
    Shape shape1({1, 1, 3, 1});
    Shape shape2({2, 4, 1, 5});

    Shape result = BroadcastHelper::broadcast_shape(shape1, shape2);
    EXPECT_EQ(result, Shape({2, 4, 3, 5}));
}

TEST(BroadcastHelper, BroadcastShape_IncompatibleThrows) {
    Shape shape1({3, 4});
    Shape shape2({3, 5});

    EXPECT_THROW(BroadcastHelper::broadcast_shape(shape1, shape2), DimensionError);
}

// ============================================================================
// Index Mapping Tests
// ============================================================================

TEST(BroadcastHelper, MapIndex_NoActualBroadcast) {
    Shape original({2, 3});
    Shape broadcast({2, 3});

    // Test all indices map correctly when no broadcasting occurs
    for (size_t i = 0; i < 6; ++i) {
        size_t mapped = BroadcastHelper::map_broadcast_index(i, broadcast, original);
        EXPECT_EQ(mapped, i);
    }
}

TEST(BroadcastHelper, MapIndex_ScalarBroadcast) {
    Shape original({1});  // Scalar as 1-element array
    Shape broadcast({3, 4});

    // All broadcast indices should map to index 0
    for (size_t i = 0; i < 12; ++i) {
        size_t mapped = BroadcastHelper::map_broadcast_index(i, broadcast, original);
        EXPECT_EQ(mapped, 0u);
    }
}

TEST(BroadcastHelper, MapIndex_RowBroadcast) {
    Shape original({1, 3});   // Row vector
    Shape broadcast({4, 3});  // Broadcast to 4 rows

    // Indices should repeat: 0,1,2, 0,1,2, 0,1,2, 0,1,2
    std::vector<size_t> expected = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

    for (size_t i = 0; i < 12; ++i) {
        size_t mapped = BroadcastHelper::map_broadcast_index(i, broadcast, original);
        EXPECT_EQ(mapped, expected[i]);
    }
}

TEST(BroadcastHelper, MapIndex_ColumnBroadcast) {
    Shape original({3, 1});   // Column vector
    Shape broadcast({3, 4});  // Broadcast to 4 columns

    // Indices should be: 0,0,0,0, 1,1,1,1, 2,2,2,2
    std::vector<size_t> expected = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

    for (size_t i = 0; i < 12; ++i) {
        size_t mapped = BroadcastHelper::map_broadcast_index(i, broadcast, original);
        EXPECT_EQ(mapped, expected[i]);
    }
}

TEST(BroadcastHelper, MapIndex_ComplexBroadcast) {
    Shape original({1, 2, 1});   // Middle dimension is 2
    Shape broadcast({3, 2, 4});  // Broadcast outer and inner

    // With shape (1,2,1) the valid indices are 0 and 1
    // Broadcast pattern should repeat these appropriately
    for (size_t i = 0; i < 24; ++i) {
        size_t mapped = BroadcastHelper::map_broadcast_index(i, broadcast, original);

        // The middle dimension alternates between 0 and 1
        size_t expected = ((i / 4) % 2);
        EXPECT_EQ(mapped, expected);
    }
}

// ============================================================================
// Stride Computation Tests
// ============================================================================

TEST(BroadcastHelper, ComputeStrides_NoReallBroadcast) {
    Shape original({2, 3});
    Shape broadcast({2, 3});

    auto strides = BroadcastHelper::compute_broadcast_strides(original, broadcast);

    EXPECT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 3);  // Stride for first dimension
    EXPECT_EQ(strides[1], 1);  // Stride for second dimension
}

TEST(BroadcastHelper, ComputeStrides_ScalarBroadcast) {
    Shape original({1});
    Shape broadcast({3, 4});

    auto strides = BroadcastHelper::compute_broadcast_strides(original, broadcast);

    EXPECT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 0);  // Both strides are 0 for scalar broadcast
    EXPECT_EQ(strides[1], 0);
}

TEST(BroadcastHelper, ComputeStrides_RowBroadcast) {
    Shape original({1, 4});
    Shape broadcast({3, 4});

    auto strides = BroadcastHelper::compute_broadcast_strides(original, broadcast);

    EXPECT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 0);  // First dimension is broadcast (stride 0)
    EXPECT_EQ(strides[1], 1);  // Second dimension normal stride
}

TEST(BroadcastHelper, ComputeStrides_ColumnBroadcast) {
    Shape original({3, 1});
    Shape broadcast({3, 4});

    auto strides = BroadcastHelper::compute_broadcast_strides(original, broadcast);

    EXPECT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 1);  // First dimension normal stride
    EXPECT_EQ(strides[1], 0);  // Second dimension is broadcast (stride 0)
}

TEST(BroadcastHelper, ComputeStrides_AddDimensions) {
    Shape original({3});      // 1D
    Shape broadcast({2, 3});  // Broadcast to 2D

    auto strides = BroadcastHelper::compute_broadcast_strides(original, broadcast);

    EXPECT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 0);  // New dimension has stride 0
    EXPECT_EQ(strides[1], 1);  // Original dimension keeps stride
}

// ============================================================================
// BroadcastIterator Tests
// ============================================================================

TEST(BroadcastIterator, ScalarBroadcast) {
    double data[] = {5.0};
    Shape original({1});
    Shape broadcast({3, 3});

    BroadcastIterator<double> iter(data, original, broadcast);

    // All iterations should return the same value
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(*iter, 5.0);
        ++iter;
    }
}

TEST(BroadcastIterator, RowBroadcast) {
    double data[] = {1.0, 2.0, 3.0};
    Shape original({1, 3});
    Shape broadcast({2, 3});

    BroadcastIterator<double> iter(data, original, broadcast);

    std::vector<double> expected = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(*iter, expected[i]);
        ++iter;
    }
}

TEST(BroadcastIterator, ColumnBroadcast) {
    double data[] = {1.0, 2.0};
    Shape original({2, 1});
    Shape broadcast({2, 3});

    BroadcastIterator<double> iter(data, original, broadcast);

    std::vector<double> expected = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(*iter, expected[i]);
        ++iter;
    }
}

TEST(BroadcastIterator, ComplexBroadcast) {
    double data[] = {1.0, 2.0};  // Shape (2, 1)
    Shape original({2, 1});
    Shape broadcast({2, 3});

    auto start = BroadcastHelper::make_broadcast_iterator(data, original, broadcast);
    auto end = BroadcastIterator<double>(data, original, broadcast, 6);

    std::vector<double> result;
    for (auto it = start; it != end; ++it) {
        result.push_back(*it);
    }

    std::vector<double> expected = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    EXPECT_EQ(result, expected);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(BroadcastHelper, EmptyShape) {
    Shape empty1({0});
    Shape empty2({0, 3});
    Shape normal({3, 4});

    // Empty shapes should be broadcastable with each other
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(empty1, empty2));

    // But not with non-empty shapes (in general)
    EXPECT_FALSE(BroadcastHelper::are_broadcastable(empty1, normal));
}

TEST(BroadcastHelper, LargeRankDifference) {
    Shape small({3});
    Shape large({2, 3, 4, 5, 3});  // 5D

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(small, large));

    Shape result = BroadcastHelper::broadcast_shape(small, large);
    EXPECT_EQ(result, large);
}

TEST(BroadcastHelper, AllOnes) {
    Shape ones1({1, 1, 1});
    Shape ones2({1, 1});
    Shape shape({3, 4, 5});

    EXPECT_TRUE(BroadcastHelper::are_broadcastable(ones1, shape));
    EXPECT_TRUE(BroadcastHelper::are_broadcastable(ones2, shape));

    Shape result = BroadcastHelper::broadcast_shape(ones1, shape);
    EXPECT_EQ(result, shape);
}
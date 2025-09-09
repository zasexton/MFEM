// test_view_base.cpp - Unit tests for view classes
#include <gtest/gtest.h>
#include <base/view_base.h>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

using namespace fem::numeric;

// ============================================================================
// ViewBase Tests
// ============================================================================

TEST(ViewBase, DefaultConstruction) {
    ViewBase<int> view;
    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), 0u);
    EXPECT_TRUE(view.empty());
    EXPECT_FALSE(view.is_valid());
    EXPECT_TRUE(view.is_view());
}

TEST(ViewBase, ConstructionFromPointer) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    ViewBase<int> view(vec.data(), vec.size());

    EXPECT_EQ(view.data(), vec.data());
    EXPECT_EQ(view.size(), 5u);
    EXPECT_FALSE(view.empty());
    EXPECT_TRUE(view.is_valid());
}

TEST(ViewBase, ElementAccess) {
    std::vector<int> vec{10, 20, 30, 40, 50};
    ViewBase<int> view(vec.data(), vec.size());

    // operator[]
    EXPECT_EQ(view[0], 10);
    EXPECT_EQ(view[4], 50);

    // at() with bounds checking
    EXPECT_EQ(view.at(2), 30);
    EXPECT_THROW(view.at(5), std::out_of_range);

    // front() and back()
    EXPECT_EQ(view.front(), 10);
    EXPECT_EQ(view.back(), 50);
}

TEST(ViewBase, MutableAccess) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    ViewBase<int> view(vec.data(), vec.size());

    // Modify through view
    view[0] = 100;
    view.at(2) = 300;

    // Check original data is modified
    EXPECT_EQ(vec[0], 100);
    EXPECT_EQ(vec[2], 300);
}

TEST(ViewBase, ConstView) {
    const std::vector<int> vec{1, 2, 3, 4, 5};
    ViewBase<const int> view(vec.data(), vec.size());

    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view.at(4), 5);

    // Should not compile:
    // view[0] = 10;
}

TEST(ViewBase, Subview) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8};
    ViewBase<int> view(vec.data(), vec.size());

    auto sub = view.subview(2, 3);
    EXPECT_EQ(sub.size(), 3u);
    EXPECT_EQ(sub[0], 3);
    EXPECT_EQ(sub[1], 4);
    EXPECT_EQ(sub[2], 5);

    // Modify through subview
    sub[1] = 400;
    EXPECT_EQ(vec[3], 400);

    // Out of range subview
    EXPECT_THROW(view.subview(6, 3), std::out_of_range);
}

TEST(ViewBase, SpanConversion) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    ViewBase<int> view(vec.data(), vec.size());

    std::span<int> span = view;
    EXPECT_EQ(span.size(), 5u);
    EXPECT_EQ(span[0], 1);

    const ViewBase<int> const_view(vec.data(), vec.size());
    std::span<const int> const_span = const_view;
    EXPECT_EQ(const_span.size(), 5u);
}

TEST(ViewBase, Overlaps) {
    std::vector<int> vec(10);
    ViewBase<int> view1(vec.data(), 5);
    ViewBase<int> view2(vec.data() + 3, 5);
    ViewBase<int> view3(vec.data() + 5, 5);

    EXPECT_TRUE(view1.overlaps(view2));   // Overlapping
    EXPECT_TRUE(view2.overlaps(view1));   // Symmetric
    EXPECT_FALSE(view1.overlaps(view3));  // Adjacent but not overlapping

    ViewBase<int> empty;
    EXPECT_FALSE(view1.overlaps(empty));  // Empty view doesn't overlap
}

TEST(ViewBase, Reshape) {
    std::vector<int> vec(12);
    std::iota(vec.begin(), vec.end(), 1);  // 1, 2, 3, ..., 12

    ViewBase<int> view(vec.data(), vec.size());

    // Reshape to 3x4
    auto reshaped_3x4 = view.reshape<2>({3, 4});
    ASSERT_TRUE(reshaped_3x4.has_value());
    auto& view_2d = reshaped_3x4.value();
    EXPECT_EQ(view_2d(0, 0), 1);
    EXPECT_EQ(view_2d(0, 3), 4);
    EXPECT_EQ(view_2d(2, 3), 12);

    // Reshape to 2x2x3
    auto reshaped_2x2x3 = view.reshape<3>({2, 2, 3});
    ASSERT_TRUE(reshaped_2x2x3.has_value());

    // Invalid reshape (wrong total size)
    auto invalid = view.reshape<2>({5, 3});
    EXPECT_FALSE(invalid.has_value());
}

// ============================================================================
// StridedView Tests
// ============================================================================

TEST(StridedView, BasicConstruction) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8};
    StridedView<int> view(vec.data(), 4, 2);  // Every other element

    EXPECT_EQ(view.size(), 4u);
    EXPECT_EQ(view.stride(), 2);
    EXPECT_FALSE(view.is_contiguous());

    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[1], 3);
    EXPECT_EQ(view[2], 5);
    EXPECT_EQ(view[3], 7);
}

TEST(StridedView, MutableStridedAccess) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8};
    StridedView<int> view(vec.data(), 4, 2);

    view[1] = 30;  // Modifies vec[2]
    view[3] = 70;  // Modifies vec[6]

    EXPECT_EQ(vec[2], 30);
    EXPECT_EQ(vec[6], 70);
}

TEST(StridedView, ContiguousStride) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    StridedView<int> view(vec.data(), 5, 1);

    EXPECT_TRUE(view.is_contiguous());
    EXPECT_EQ(view[2], 3);
}

TEST(StridedView, NegativeStride) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    StridedView<int> view(vec.data() + 4, 5, -1);  // Reverse view

    EXPECT_EQ(view[0], 5);
    EXPECT_EQ(view[1], 4);
    EXPECT_EQ(view[2], 3);
    EXPECT_EQ(view[3], 2);
    EXPECT_EQ(view[4], 1);
}

TEST(StridedView, StridedSubview) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    StridedView<int> view(vec.data(), 5, 2);  // 1, 3, 5, 7, 9

    auto sub = view.subview(1, 3);  // Skip first, take 3
    EXPECT_EQ(sub.size(), 3u);
    EXPECT_EQ(sub.stride(), 2);
    EXPECT_EQ(sub[0], 3);
    EXPECT_EQ(sub[1], 5);
    EXPECT_EQ(sub[2], 7);
}

TEST(StridedView, BoundsChecking) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    StridedView<int> view(vec.data(), 3, 2);

    EXPECT_EQ(view.at(0), 1);
    EXPECT_EQ(view.at(2), 5);
    EXPECT_THROW(view.at(3), std::out_of_range);
}

// ============================================================================
// MultiDimView Tests
// ============================================================================

TEST(MultiDimView, Construction2D) {
    std::vector<int> vec(12);
    std::iota(vec.begin(), vec.end(), 1);  // 1-12

    MultiDimView<int, 2> view = MultiDimView<int, 2>::from_contiguous(
        vec.data(), {3, 4});

    EXPECT_EQ(view.size(), 12u);
    EXPECT_EQ(view.size(0), 3u);
    EXPECT_EQ(view.size(1), 4u);
    EXPECT_TRUE(view.is_contiguous());

    // Access elements
    EXPECT_EQ(view(0, 0), 1);
    EXPECT_EQ(view(0, 3), 4);
    EXPECT_EQ(view(1, 0), 5);
    EXPECT_EQ(view(2, 3), 12);
}

TEST(MultiDimView, Construction3D) {
    std::vector<int> vec(24);
    std::iota(vec.begin(), vec.end(), 0);

    MultiDimView<int, 3> view = MultiDimView<int, 3>::from_contiguous(
        vec.data(), {2, 3, 4});

    EXPECT_EQ(view.size(), 24u);
    EXPECT_EQ(view(0, 0, 0), 0);
    EXPECT_EQ(view(1, 2, 3), 23);
}

TEST(MultiDimView, MutableMultiDim) {
    std::vector<int> vec(6, 0);
    MultiDimView<int, 2> view = MultiDimView<int, 2>::from_contiguous(
        vec.data(), {2, 3});

    view(0, 1) = 10;
    view(1, 2) = 20;

    EXPECT_EQ(vec[1], 10);
    EXPECT_EQ(vec[5], 20);
}

TEST(MultiDimView, NonContiguousStrides) {
    std::vector<int> vec(12);
    std::iota(vec.begin(), vec.end(), 1);

    // Column-major strides for 3x4 matrix
    MultiDimView<int, 2> view(vec.data(), {3, 4}, {1, 3});

    EXPECT_FALSE(view.is_contiguous());
    EXPECT_EQ(view(0, 0), 1);
    EXPECT_EQ(view(0, 1), 4);  // Skip 3 elements
    EXPECT_EQ(view(1, 0), 2);  // Next row
}

TEST(MultiDimView, Slice2D) {
    std::vector<int> vec(12);
    std::iota(vec.begin(), vec.end(), 1);

    MultiDimView<int, 2> view = MultiDimView<int, 2>::from_contiguous(
        vec.data(), {3, 4});

    // Slice along first dimension (select row 1)
    auto row_slice = view.slice(0, 1);
    EXPECT_EQ(row_slice.size(0), 1u);
    EXPECT_EQ(row_slice.size(1), 4u);
    EXPECT_EQ(row_slice(0, 0), 5);
    EXPECT_EQ(row_slice(0, 3), 8);

    // Slice along second dimension (select column 2)
    auto col_slice = view.slice(1, 2);
    EXPECT_EQ(col_slice.size(0), 3u);
    EXPECT_EQ(col_slice.size(1), 1u);
    EXPECT_EQ(col_slice(0, 0), 3);
    EXPECT_EQ(col_slice(1, 0), 7);
    EXPECT_EQ(col_slice(2, 0), 11);
}

TEST(MultiDimView, BoundsChecking) {
    std::vector<int> vec(6);
    MultiDimView<int, 2> view = MultiDimView<int, 2>::from_contiguous(
        vec.data(), {2, 3});

    EXPECT_NO_THROW(view(1, 2));
    EXPECT_THROW(view(2, 0), std::out_of_range);
    EXPECT_THROW(view(0, 3), std::out_of_range);

    EXPECT_THROW(view.slice(0, 2), std::out_of_range);
    EXPECT_THROW(view.slice(2, 0), std::out_of_range);
}

// ============================================================================
// ViewFactory Tests
// ============================================================================

TEST(ViewFactory, CreateBasicView) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    auto view = ViewFactory::create_view(vec.data(), vec.size());

    EXPECT_EQ(view.size(), 5u);
    EXPECT_EQ(view[2], 3);
}

TEST(ViewFactory, CreateStridedView) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    auto view = ViewFactory::create_strided_view(vec.data(), 3, 2);

    EXPECT_EQ(view.size(), 3u);
    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[1], 3);
    EXPECT_EQ(view[2], 5);
}

TEST(ViewFactory, CreateMultiDimView) {
    std::vector<int> vec(12);
    std::iota(vec.begin(), vec.end(), 0);

    auto view = ViewFactory::create_multidim_view<int, 2>(
        vec.data(), {3, 4}, {4, 1});

    EXPECT_EQ(view.size(), 12u);
    EXPECT_EQ(view(0, 0), 0);
    EXPECT_EQ(view(2, 3), 11);
}

TEST(ViewFactory, CreateTransposedView) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6};  // 2x3 matrix
    auto view = ViewFactory::create_transposed_view(vec.data(), 2, 3);

    // Original: [[1,2,3], [4,5,6]]
    // Transposed: [[1,4], [2,5], [3,6]]
    EXPECT_EQ(view.size(0), 3u);  // Now 3 rows
    EXPECT_EQ(view.size(1), 2u);  // Now 2 columns

    EXPECT_EQ(view(0, 0), 1);
    EXPECT_EQ(view(0, 1), 4);
    EXPECT_EQ(view(1, 0), 2);
    EXPECT_EQ(view(1, 1), 5);
    EXPECT_EQ(view(2, 0), 3);
    EXPECT_EQ(view(2, 1), 6);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ViewIntegration, ModifyThroughMultipleViews) {
    std::vector<int> vec(20, 0);

    // Create overlapping views
    ViewBase<int> view1(vec.data(), 10);      // Covers indices 0-9
    ViewBase<int> view2(vec.data() + 5, 10);  // Covers indices 5-14

    // Modify through first view
    for (size_t i = 0; i < view1.size(); ++i) {
        view1[i] = static_cast<int>(i);
    }

    // Modify through second view
    for (size_t i = 0; i < view2.size(); ++i) {
        view2[i] += 100;
    }

    // Check results
    EXPECT_EQ(vec[0], 0);     // Only modified by view1
    EXPECT_EQ(vec[5], 105);   // Modified by both views (was 5, now 5+100)
    EXPECT_EQ(vec[9], 109);   // Modified by both views (was 9, now 9+100)
    EXPECT_EQ(vec[14], 100);  // Only modified by view2 (was 0, now 0+100)
    EXPECT_EQ(vec[15], 0);    // Not modified by either view
}

TEST(ViewIntegration, ChainedSubviews) {
    std::vector<int> vec(100);
    std::iota(vec.begin(), vec.end(), 0);

    ViewBase<int> view1(vec.data(), 100);
    auto view2 = view1.subview(10, 50);  // Elements 10-59
    auto view3 = view2.subview(5, 20);   // Elements 15-34 of original

    EXPECT_EQ(view3.size(), 20u);
    EXPECT_EQ(view3[0], 15);
    EXPECT_EQ(view3[19], 34);
}

TEST(ViewIntegration, MixedViewTypes) {
    std::vector<double> vec(24);
    std::iota(vec.begin(), vec.end(), 1.0);

    // Create 2D view
    auto view_2d = MultiDimView<double, 2>::from_contiguous(
        vec.data(), {4, 6});

    // Create strided view of same data (every 6th element = first column)
    StridedView<double> col_view(vec.data(), 4, 6);

    // Modify through 2D view
    view_2d(1, 0) = 100.0;  // Element at position 6

    // Check through strided view
    EXPECT_EQ(col_view[1], 100.0);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST(ViewEdgeCases, EmptyView) {
    ViewBase<int> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_FALSE(empty.is_valid());

    // Operations on empty view
    EXPECT_THROW(empty.subview(0, 1), std::out_of_range);
}

TEST(ViewEdgeCases, SingleElementView) {
    int value = 42;
    ViewBase<int> single(&value, 1);

    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);
    EXPECT_EQ(single.front(), 42);
    EXPECT_EQ(single.back(), 42);

    single[0] = 100;
    EXPECT_EQ(value, 100);
}

TEST(ViewEdgeCases, ZeroStrideView) {
    int value = 42;
    StridedView<int> view(&value, 5, 0);  // All elements point to same location

    EXPECT_EQ(view[0], 42);
    EXPECT_EQ(view[1], 42);
    EXPECT_EQ(view[4], 42);

    view[2] = 100;
    EXPECT_EQ(value, 100);
    EXPECT_EQ(view[0], 100);  // All elements see the change
}
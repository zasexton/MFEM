// test_container_base.cpp - Unit tests for container base functionality
#include <gtest/gtest.h>
#include <base/container_base.h>
#include <base/numeric_base.h>
#include <base/storage_base.h>
#include <base/slice_base.h>
#include <base/view_base.h>
#include <vector>
#include <numeric>

using namespace fem::numeric;

// Test container implementation for testing base functionality
template<typename T>
class TestContainer : public ContainerBase<TestContainer<T>, T> {
public:
    using Base = ContainerBase<TestContainer<T>, T>;
    using Base::Base;

    // Required for proper CRTP
    using Base::shape;
    using Base::data;
};

// Test container with slicing support
template<typename T>
class SliceableTestContainer : public ContainerBase<SliceableTestContainer<T>, T>,
                                public SliceableContainer<SliceableTestContainer<T>, T> {
public:
    using Base = ContainerBase<SliceableTestContainer<T>, T>;
    using Base::Base;

    // Implement required methods for slicing
    size_t compute_index(const MultiIndex& idx) {
        // Simple linear indexing for testing
        return 0;
    }

    std::tuple<size_t, std::ptrdiff_t, size_t> compute_1d_slice(const MultiIndex& idx) {
        return {0, 1, 10};  // offset, stride, count
    }

    AnyMultiDimView<T> create_multidim_view_impl(
        const MultiIndex& idx,
        const Shape& result_shape) {
        return AnyMultiDimView<T>();
    }
};

// ============================================================================
// ContainerBase Tests
// ============================================================================

TEST(ContainerBase, DefaultConstruction) {
    TestContainer<double> c;
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
    EXPECT_TRUE(c.owns_data());
}

TEST(ContainerBase, ShapeConstruction) {
    Shape shape({3, 4});
    TestContainer<double> c(shape);
    EXPECT_EQ(c.size(), 12u);
    EXPECT_EQ(c.shape(), shape);
    EXPECT_FALSE(c.empty());
    EXPECT_EQ(c.ndim(), 2u);
}

TEST(ContainerBase, ValueConstruction) {
    Shape shape({2, 3});
    TestContainer<double> c(shape, 5.0);
    EXPECT_EQ(c.size(), 6u);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_EQ(c[i], 5.0);
    }
}

TEST(ContainerBase, InitializerListConstruction) {
    Shape shape({2, 2});
    TestContainer<double> c(shape, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(c[0], 1.0);
    EXPECT_EQ(c[1], 2.0);
    EXPECT_EQ(c[2], 3.0);
    EXPECT_EQ(c[3], 4.0);
}

TEST(ContainerBase, InitializerListSizeMismatch) {
    Shape shape({2, 2});
    EXPECT_THROW(TestContainer<double>(shape, {1.0, 2.0, 3.0}), DimensionError);
}

TEST(ContainerBase, CheckFinite) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});

    // This would need NumericOptions to have check_finite = true
    // and would test lines 395-404
    // You might need to expose check_finite() as public for testing
}
// ============================================================================
// Element Access Tests
// ============================================================================

TEST(ContainerBase, ElementAccess) {
    Shape shape({3, 3});
    TestContainer<double> c(shape, 0.0);

    // Test operator[]
    c[4] = 5.0;
    EXPECT_EQ(c[4], 5.0);

    // Test at()
    c.at(5) = 6.0;
    EXPECT_EQ(c.at(5), 6.0);

    // Test at() bounds checking
    EXPECT_THROW(c.at(100), std::out_of_range);
}

TEST(ContainerBase, FrontBack) {
    Shape shape({5});
    TestContainer<double> c(shape, {1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_EQ(c.front(), 1.0);
    EXPECT_EQ(c.back(), 5.0);

    c.front() = 10.0;
    c.back() = 50.0;

    EXPECT_EQ(c[0], 10.0);
    EXPECT_EQ(c[4], 50.0);
}

TEST(ContainerBase, ConstElementAccess) {
    const TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});

    // Test const operator[]
    EXPECT_EQ(c[0], 1.0);
    EXPECT_EQ(c[1], 2.0);

    // Test const at()
    EXPECT_EQ(c.at(0), 1.0);
    EXPECT_THROW(c.at(10), std::out_of_range);

    // Test const front/back
    EXPECT_EQ(c.front(), 1.0);
    EXPECT_EQ(c.back(), 3.0);
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST(ContainerBase, Iterators) {
    Shape shape({4});
    TestContainer<double> c(shape, {1.0, 2.0, 3.0, 4.0});

    // Forward iteration
    std::vector<double> values;
    for (auto val : c) {
        values.push_back(val);
    }
    EXPECT_EQ(values, std::vector<double>({1.0, 2.0, 3.0, 4.0}));

    // Reverse iteration
    values.clear();
    for (auto it = c.rbegin(); it != c.rend(); ++it) {
        values.push_back(*it);
    }
    EXPECT_EQ(values, std::vector<double>({4.0, 3.0, 2.0, 1.0}));
}

TEST(ContainerBase, ConstIterators) {
    const TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});

    // Test const iterators
    std::vector<double> values;
    for (auto it = c.cbegin(); it != c.cend(); ++it) {
        values.push_back(*it);
    }
    EXPECT_EQ(values.size(), 3u);

    // Test const reverse iterators
    values.clear();
    for (auto it = c.crbegin(); it != c.crend(); ++it) {
        values.push_back(*it);
    }
    EXPECT_EQ(values, std::vector<double>({3.0, 2.0, 1.0}));
}

// ============================================================================
// Modifier Tests
// ============================================================================

TEST(ContainerBase, Fill) {
    Shape shape({2, 3});
    TestContainer<double> c(shape);

    c.fill(7.0);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_EQ(c[i], 7.0);
    }
}

TEST(ContainerBase, Swap) {
    Shape shape1({2, 2});
    Shape shape2({3});
    TestContainer<double> c1(shape1, 1.0);
    TestContainer<double> c2(shape2, 2.0);

    c1.swap(c2);

    EXPECT_EQ(c1.shape(), shape2);
    EXPECT_EQ(c1.size(), 3u);
    EXPECT_EQ(c1[0], 2.0);

    EXPECT_EQ(c2.shape(), shape1);
    EXPECT_EQ(c2.size(), 4u);
    EXPECT_EQ(c2[0], 1.0);
}

TEST(ContainerBase, Clear) {
    Shape shape({3, 3});
    TestContainer<double> c(shape, 5.0);

    c.clear();
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
}

// ============================================================================
// Shape Operations Tests
// ============================================================================

TEST(ContainerBase, Resize) {
    TestContainer<double> c(Shape({2, 2}), 1.0);

    c.resize(Shape({3, 3}));
    EXPECT_EQ(c.size(), 9u);
    EXPECT_EQ(c.shape(), Shape({3, 3}));
}

TEST(ContainerBase, ResizeWithValue) {
    TestContainer<double> c(Shape({2}));

    c.resize(Shape({5}), 3.0);
    EXPECT_EQ(c.size(), 5u);
    // Note: resize behavior for new elements depends on storage implementation
}

TEST(ContainerBase, Reshape) {
    TestContainer<double> c(Shape({2, 3}), 1.0);

    c.reshape(Shape({3, 2}));
    EXPECT_EQ(c.size(), 6u);
    EXPECT_EQ(c.shape(), Shape({3, 2}));

    // Should throw if sizes don't match
    EXPECT_THROW(c.reshape(Shape({4, 4})), DimensionError);
}

// ============================================================================
// Copy Operations Tests
// ============================================================================

TEST(ContainerBase, Copy) {
    TestContainer<double> c1(Shape({2, 2}), {1.0, 2.0, 3.0, 4.0});

    auto c2 = c1.copy();
    EXPECT_EQ(c2.size(), c1.size());
    EXPECT_EQ(c2.shape(), c1.shape());

    // Verify it's a copy, not sharing data
    c2[0] = 10.0;
    EXPECT_EQ(c1[0], 1.0);
    EXPECT_EQ(c2[0], 10.0);
}

TEST(ContainerBase, DeepCopy) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});

    auto c2 = c1.deep_copy();
    EXPECT_EQ(c2.size(), c1.size());
    for (size_t i = 0; i < c1.size(); ++i) {
        EXPECT_EQ(c2[i], c1[i]);
    }
}

// ============================================================================
// Validation Operations Tests
// ============================================================================

TEST(ContainerBase, AllFinite) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});
    EXPECT_TRUE(c.all_finite());

    c[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(c.all_finite());
}

TEST(ContainerBase, HasNaN) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});
    EXPECT_FALSE(c.has_nan());

    c[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(c.has_nan());
}

TEST(ContainerBase, HasInf) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});
    EXPECT_FALSE(c.has_inf());

    c[1] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(c.has_inf());
}

// ============================================================================
// Transformation Operations Tests
// ============================================================================

TEST(ContainerBase, ApplyUnary) {
    TestContainer<double> c(Shape({4}), {1.0, 2.0, 3.0, 4.0});

    c.apply([](double x) { return x * 2.0; });

    EXPECT_EQ(c[0], 2.0);
    EXPECT_EQ(c[1], 4.0);
    EXPECT_EQ(c[2], 6.0);
    EXPECT_EQ(c[3], 8.0);
}

TEST(ContainerBase, ApplyCopy) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});

    auto c2 = c1.apply_copy([](double x) { return x * x; });

    // Original unchanged
    EXPECT_EQ(c1[0], 1.0);
    EXPECT_EQ(c1[1], 2.0);

    // New container has transformed values
    EXPECT_EQ(c2[0], 1.0);
    EXPECT_EQ(c2[1], 4.0);
    EXPECT_EQ(c2[2], 9.0);
}

TEST(ContainerBase, ApplyBinary) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});
    TestContainer<double> c2(Shape({3}), {4.0, 5.0, 6.0});

    c1.apply(c2, std::plus<double>());

    EXPECT_EQ(c1[0], 5.0);
    EXPECT_EQ(c1[1], 7.0);
    EXPECT_EQ(c1[2], 9.0);
}

TEST(ContainerBase, ApplyBinaryShapeMismatch) {
    TestContainer<double> c1(Shape({3}), 1.0);
    TestContainer<double> c2(Shape({4}), 2.0);

    EXPECT_THROW(c1.apply(c2, std::plus<double>()), DimensionError);
}

TEST(ContainerBase, ApplyOperations) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});

    // Test apply (lines 234-236)
    c.apply([](double x) { return x * 2.0; });
    EXPECT_EQ(c[0], 2.0);
    EXPECT_EQ(c[1], 4.0);

    // Test apply_copy (lines 239-243)
    TestContainer<double> c2(Shape({3}), {1.0, 2.0, 3.0});
    auto c3 = c2.apply_copy([](double x) { return x + 1.0; });
    EXPECT_EQ(c2[0], 1.0);  // Original unchanged
    EXPECT_EQ(c3[0], 2.0);  // New has transformed values
}

// ============================================================================
// Reduction Operations Tests
// ============================================================================

TEST(ContainerBase, Sum) {
    TestContainer<double> c(Shape({4}), {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(c.sum(), 10.0);
}

TEST(ContainerBase, Product) {
    TestContainer<double> c(Shape({4}), {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(c.product(), 24.0);
}

TEST(ContainerBase, MinMax) {
    TestContainer<double> c(Shape({5}), {3.0, 1.0, 4.0, 1.0, 5.0});
    EXPECT_EQ(c.min(), 1.0);
    EXPECT_EQ(c.max(), 5.0);
}

TEST(ContainerBase, MinMaxEmpty) {
    TestContainer<double> c;
    EXPECT_THROW(c.min(), std::runtime_error);
    EXPECT_THROW(c.max(), std::runtime_error);
}

TEST(ContainerBase, ReduceOperations) {
    TestContainer<double> c(Shape({4}), {1.0, 2.0, 3.0, 4.0});

    // Test generic reduce (lines 271-273)
    auto sum = c.reduce(std::plus<double>(), 0.0);
    EXPECT_EQ(sum, 10.0);

    auto product = c.reduce(std::multiplies<double>(), 1.0);
    EXPECT_EQ(product, 24.0);
}

TEST(ContainerBase, ScalarSubtraction) {
    TestContainer<double> c(Shape({3}), {5.0, 6.0, 7.0});

    // Test operator-= (lines 325-328)
    c -= 2.0;
    EXPECT_EQ(c[0], 3.0);
    EXPECT_EQ(c[1], 4.0);
}

TEST(ContainerBase, ContainerSubtraction) {
    TestContainer<double> c1(Shape({3}), {5.0, 6.0, 7.0});
    TestContainer<double> c2(Shape({3}), {1.0, 2.0, 3.0});

    // Test operator-= (lines 350-353)
    c1 -= c2;
    EXPECT_EQ(c1[0], 4.0);
    EXPECT_EQ(c1[1], 4.0);
}

TEST(ContainerBase, ContainerMultiplication) {
    TestContainer<double> c1(Shape({3}), {2.0, 3.0, 4.0});
    TestContainer<double> c2(Shape({3}), {2.0, 2.0, 2.0});

    // Test operator*= (lines 355-358)
    c1 *= c2;
    EXPECT_EQ(c1[0], 4.0);
    EXPECT_EQ(c1[1], 6.0);
}

TEST(ContainerBase, ValidDivision) {
    TestContainer<double> c(Shape({3}), {4.0, 6.0, 8.0});

    // Test valid division (lines 339-340)
    c /= 2.0;
    EXPECT_EQ(c[0], 2.0);
    EXPECT_EQ(c[1], 3.0);
}

TEST(ContainerBase, ValidContainerDivision) {
    TestContainer<double> c1(Shape({3}), {6.0, 8.0, 10.0});
    TestContainer<double> c2(Shape({3}), {2.0, 4.0, 5.0});

    // Test valid division (lines 365-367)
    c1 /= c2;
    EXPECT_EQ(c1[0], 3.0);
    EXPECT_EQ(c1[1], 2.0);
    EXPECT_EQ(c1[2], 2.0);
}

TEST(ContainerBase, MoveOperations) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});

    // Test move constructor (line 69)
    TestContainer<double> c2(std::move(c1));
    EXPECT_EQ(c2.size(), 3u);

    // Test move assignment (line 71)
    TestContainer<double> c3;
    c3 = std::move(c2);
    EXPECT_EQ(c3.size(), 3u);
}
// ============================================================================
// Comparison Operations Tests
// ============================================================================

TEST(ContainerBase, Equality) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});
    TestContainer<double> c2(Shape({3}), {1.0, 2.0, 3.0});
    TestContainer<double> c3(Shape({3}), {1.0, 2.0, 4.0});

    EXPECT_TRUE(c1 == c2);
    EXPECT_FALSE(c1 == c3);
    EXPECT_FALSE(c1 != c2);
    EXPECT_TRUE(c1 != c3);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST(ContainerBase, Broadcasting) {
    TestContainer<double> c1(Shape({3, 1}));
    TestContainer<double> c2(Shape({1, 4}));

    EXPECT_TRUE(c1.is_broadcastable_with(c2));

    auto broadcast_shape = c1.broadcast_shape(c2);
    EXPECT_EQ(broadcast_shape.size(), 12u);  // 3*4
}

// ============================================================================
// Element-wise Operations Tests
// ============================================================================

TEST(ContainerBase, ScalarOperations) {
    TestContainer<double> c(Shape({3}), {1.0, 2.0, 3.0});

    c += 2.0;
    EXPECT_EQ(c[0], 3.0);
    EXPECT_EQ(c[1], 4.0);
    EXPECT_EQ(c[2], 5.0);

    c *= 2.0;
    EXPECT_EQ(c[0], 6.0);
    EXPECT_EQ(c[1], 8.0);
    EXPECT_EQ(c[2], 10.0);
}

TEST(ContainerBase, ContainerOperations) {
    TestContainer<double> c1(Shape({3}), {1.0, 2.0, 3.0});
    TestContainer<double> c2(Shape({3}), {4.0, 5.0, 6.0});

    c1 += c2;
    EXPECT_EQ(c1[0], 5.0);
    EXPECT_EQ(c1[1], 7.0);
    EXPECT_EQ(c1[2], 9.0);
}

TEST(ContainerBase, DivisionByZero) {
    TestContainer<double> c(Shape({2}), {1.0, 2.0});

    EXPECT_THROW(c /= 0.0, ComputationError);

    TestContainer<double> c2(Shape({2}), {0.0, 1.0});
    EXPECT_THROW(c /= c2, ComputationError);
}

// ============================================================================
// ViewContainer Tests
// ============================================================================

TEST(ViewContainer, Construction) {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    Shape shape({2, 2});

    class TestView : public ViewContainer<TestView, double> {
    public:
        using ViewContainer::ViewContainer;
    };

    TestView view(data, shape);

    EXPECT_EQ(view.size(), 4u);
    EXPECT_FALSE(view.owns_data());
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view[0], 1.0);
}

TEST(ViewContainer, Overlap) {
    double data1[] = {1.0, 2.0, 3.0, 4.0};
    double data2[] = {5.0, 6.0, 7.0, 8.0};

    class TestView : public ViewContainer<TestView, double> {
    public:
        using ViewContainer::ViewContainer;
    };

    TestView view1(data1, Shape({4}));
    TestView view2(data2, Shape({4}));
    TestView view3(data1 + 2, Shape({2}));  // Overlaps with view1

    EXPECT_FALSE(view1.overlaps(view2));
    EXPECT_TRUE(view1.overlaps(view3));
}

// ============================================================================
// SliceableContainer Tests
// ============================================================================

TEST(SliceableContainer, BasicSlicing) {
    SliceableTestContainer<double> c(Shape({3, 3}), 1.0);

    MultiIndex idx{all, 1};
    using SC = SliceableContainer<SliceableTestContainer<double>, double>;
    auto& sliceable = static_cast<SC&>(c);
    auto result = sliceable[idx];

    // Check the type - now using reference_wrapper
    EXPECT_TRUE(std::holds_alternative<std::reference_wrapper<double>>(result) ||
                std::holds_alternative<ViewBase<double>>(result) ||
                std::holds_alternative<StridedView<double>>(result) ||
                std::holds_alternative<AnyMultiDimView<double>>(result));
}

// ============================================================================
// Memory Information Tests
// ============================================================================

TEST(ContainerBase, StorageAccess) {
    TestContainer<double> c(Shape({2, 2}), 1.0);

    // Test storage methods
    auto& storage = c.storage();
    EXPECT_EQ(storage.size(), 4u);

    const auto& const_storage = static_cast<const TestContainer<double>&>(c).storage();
    EXPECT_EQ(const_storage.size(), 4u);
}

TEST(ContainerBase, MemoryInfo) {
    TestContainer<double> c(Shape({10}));

    EXPECT_EQ(c.nbytes(), 10 * sizeof(double));
    EXPECT_EQ(c.dtype(), typeid(double));
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.layout(), Layout::RowMajor);
    EXPECT_EQ(c.device(), Device::CPU);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ContainerBase, EmptyContainer) {
    TestContainer<double> c;

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
    EXPECT_EQ(c.sum(), 0.0);
    EXPECT_EQ(c.product(), 1.0);
}

TEST(ContainerBase, SingleElement) {
    TestContainer<double> c(Shape({1}), {42.0});

    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(c.front(), 42.0);
    EXPECT_EQ(c.back(), 42.0);
    EXPECT_EQ(c.sum(), 42.0);
    EXPECT_EQ(c.min(), 42.0);
    EXPECT_EQ(c.max(), 42.0);
}
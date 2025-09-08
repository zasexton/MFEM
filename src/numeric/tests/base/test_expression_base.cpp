#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

#include <base/expression_base.h>

namespace fem::numeric::test {

// Mock container for testing
template<typename T>
class MockContainer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;

    MockContainer() = default;
    explicit MockContainer(const Shape& shape)
        : shape_(shape), data_(shape.size()) {}

    MockContainer(const Shape& shape, const T& value)
        : shape_(shape), data_(shape.size(), value) {}

    MockContainer(const Shape& shape, std::initializer_list<T> values)
        : shape_(shape), data_(values) {
        if (data_.size() != shape.size()) {
            throw std::runtime_error("Size mismatch");
        }
    }

    const Shape& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& at(size_t i) { return data_.at(i); }
    const T& at(size_t i) const { return data_.at(i); }

    void resize(const Shape& shape) {
        shape_ = shape;
        data_.resize(shape.size());
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    bool is_contiguous() const { return true; }

    pointer data() { return data_.data(); }
    const_pointer data() const { return data_.data(); }

    void reshape(const Shape& new_shape) {
        if (new_shape.size() != size()) {
            throw std::runtime_error("Cannot reshape: size mismatch");
        }
        shape_ = new_shape;
    }

private:
    Shape shape_;
    std::vector<T> data_;
};

// Test fixture
class ExpressionBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default numeric options for testing
        NumericOptions::defaults().check_finite = false;
        NumericOptions::defaults().allow_parallel = false;
    }
};

// ============================================================
// Terminal Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, TerminalExpressionCreation) {
    MockContainer<double> container(Shape({3, 4}), 1.0);
    auto expr = make_expression(container);

    EXPECT_EQ(expr.shape(), Shape({3, 4}));
    EXPECT_EQ(expr.size(), 12);
    EXPECT_TRUE(expr.is_parallelizable());
}

TEST_F(ExpressionBaseTest, TerminalExpressionEvaluation) {
    MockContainer<double> container(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    auto expr = make_expression(container);

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(3), 4.0);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(5), 6.0);
}

TEST_F(ExpressionBaseTest, TerminalExpressionEvalTo) {
    MockContainer<double> source(Shape({2, 2}), {1, 2, 3, 4});
    auto expr = make_expression(source);

    MockContainer<double> dest(Shape({2, 2}));
    expr.eval_to(dest);

    EXPECT_DOUBLE_EQ(dest[0], 1.0);
    EXPECT_DOUBLE_EQ(dest[1], 2.0);
    EXPECT_DOUBLE_EQ(dest[2], 3.0);
    EXPECT_DOUBLE_EQ(dest[3], 4.0);
}

// ============================================================
// Scalar Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, ScalarExpressionCreation) {
    auto expr = make_scalar_expression(5.0, Shape({3, 3}));

    EXPECT_EQ(expr.shape(), Shape({3, 3}));
    EXPECT_EQ(expr.size(), 9);
    EXPECT_DOUBLE_EQ(expr.value(), 5.0);
}

TEST_F(ExpressionBaseTest, ScalarExpressionEvaluation) {
    auto expr = make_scalar_expression(3.14, Shape({2, 2}));

    // All positions should return the same scalar value
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 3.14);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(1), 3.14);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(3), 3.14);
}

TEST_F(ExpressionBaseTest, ScalarExpressionEvalTo) {
    auto expr = make_scalar_expression(7.0, Shape({2, 3}));

    MockContainer<double> dest(Shape({2, 3}));
    expr.eval_to(dest);

    for (size_t i = 0; i < dest.size(); ++i) {
        EXPECT_DOUBLE_EQ(dest[i], 7.0);
    }
}

// ============================================================
// View Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, ViewExpressionCreation) {
    double data[] = {1, 2, 3, 4, 5, 6};
    auto expr = make_view_expression(data, Shape({2, 3}));

    EXPECT_EQ(expr.shape(), Shape({2, 3}));
    EXPECT_EQ(expr.size(), 6);
}

TEST_F(ExpressionBaseTest, ViewExpressionEvaluation) {
    double data[] = {1.1, 2.2, 3.3, 4.4};
    auto expr = make_view_expression(data, Shape({2, 2}));

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 1.1);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(1), 2.2);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(2), 3.3);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(3), 4.4);
}

// ============================================================
// Binary Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, BinaryAddition) {
    MockContainer<double> a(Shape({2, 2}), {1, 2, 3, 4});
    MockContainer<double> b(Shape({2, 2}), {5, 6, 7, 8});

    auto expr_a = make_expression(a);
    auto expr_b = make_expression(b);
    auto result_expr = expr_a + expr_b;

    EXPECT_EQ(result_expr.shape(), Shape({2, 2}));
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), 6.0);  // 1+5
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), 8.0);  // 2+6
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(2), 10.0); // 3+7
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(3), 12.0); // 4+8
}

TEST_F(ExpressionBaseTest, BinarySubtraction) {
    MockContainer<double> a(Shape({3}), {10, 20, 30});
    MockContainer<double> b(Shape({3}), {3, 5, 7});

    auto result_expr = make_expression(a) - make_expression(b);

    MockContainer<double> result(Shape({3}));
    result_expr.eval_to(result);

    EXPECT_DOUBLE_EQ(result[0], 7.0);  // 10-3
    EXPECT_DOUBLE_EQ(result[1], 15.0); // 20-5
    EXPECT_DOUBLE_EQ(result[2], 23.0); // 30-7
}

TEST_F(ExpressionBaseTest, BinaryMultiplication) {
    MockContainer<double> a(Shape({2, 2}), {2, 3, 4, 5});
    MockContainer<double> b(Shape({2, 2}), {10, 10, 10, 10});

    auto result_expr = make_expression(a) * make_expression(b);

    MockContainer<double> result(Shape({2, 2}));
    result_expr.eval_to(result);

    EXPECT_DOUBLE_EQ(result[0], 20.0);
    EXPECT_DOUBLE_EQ(result[1], 30.0);
    EXPECT_DOUBLE_EQ(result[2], 40.0);
    EXPECT_DOUBLE_EQ(result[3], 50.0);
}

TEST_F(ExpressionBaseTest, BinaryDivision) {
    MockContainer<double> a(Shape({2}), {10, 20});
    MockContainer<double> b(Shape({2}), {2, 4});

    auto result_expr = make_expression(a) / make_expression(b);

    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), 5.0);  // 10/2
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), 5.0);  // 20/4
}

// ============================================================
// Unary Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, UnaryNegation) {
    MockContainer<double> a(Shape({3}), {1, -2, 3});

    auto result_expr = -make_expression(a);

    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), -1.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(2), -3.0);
}

TEST_F(ExpressionBaseTest, UnaryAbs) {
    MockContainer<double> a(Shape({4}), {-1.5, 2.5, -3.5, 4.5});

    auto result_expr = abs(make_expression(a));

    MockContainer<double> result(Shape({4}));
    result_expr.eval_to(result);

    EXPECT_DOUBLE_EQ(result[0], 1.5);
    EXPECT_DOUBLE_EQ(result[1], 2.5);
    EXPECT_DOUBLE_EQ(result[2], 3.5);
    EXPECT_DOUBLE_EQ(result[3], 4.5);
}

TEST_F(ExpressionBaseTest, UnarySqrt) {
    MockContainer<double> a(Shape({3}), {4, 9, 16});

    auto result_expr = sqrt(make_expression(a));

    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), 3.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(2), 4.0);
}

TEST_F(ExpressionBaseTest, UnaryExp) {
    MockContainer<double> a(Shape({2}), {0, 1});

    auto result_expr = exp(make_expression(a));

    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), std::exp(1.0));
}

TEST_F(ExpressionBaseTest, UnaryLog) {
    MockContainer<double> a(Shape({2}), {1, std::exp(1.0)});

    auto result_expr = log(make_expression(a));

    EXPECT_NEAR(result_expr.template operator[]<double>(0), 0.0, 1e-10);
    EXPECT_NEAR(result_expr.template operator[]<double>(1), 1.0, 1e-10);
}

// ============================================================
// Trigonometric Function Tests
// ============================================================

TEST_F(ExpressionBaseTest, TrigonometricSin) {
    MockContainer<double> a(Shape({3}), {0, M_PI/2, M_PI});

    auto result_expr = sin(make_expression(a));

    EXPECT_NEAR(result_expr.template operator[]<double>(0), 0.0, 1e-10);
    EXPECT_NEAR(result_expr.template operator[]<double>(1), 1.0, 1e-10);
    EXPECT_NEAR(result_expr.template operator[]<double>(2), 0.0, 1e-10);
}

TEST_F(ExpressionBaseTest, TrigonometricCos) {
    MockContainer<double> a(Shape({3}), {0, M_PI/2, M_PI});

    auto result_expr = cos(make_expression(a));

    EXPECT_NEAR(result_expr.template operator[]<double>(0), 1.0, 1e-10);
    EXPECT_NEAR(result_expr.template operator[]<double>(1), 0.0, 1e-10);
    EXPECT_NEAR(result_expr.template operator[]<double>(2), -1.0, 1e-10);
}

// ============================================================
// Broadcasting Tests
// ============================================================

TEST_F(ExpressionBaseTest, BroadcastScalarToArray) {
    MockContainer<double> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    auto scalar = make_scalar_expression(10.0, Shape({1}));

    // Scalar should broadcast to match array shape
    auto result_expr = make_expression(a) + scalar;

    MockContainer<double> result(Shape({2, 3}));
    result_expr.eval_to(result);

    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], a[i] + 10.0);
    }
}

TEST_F(ExpressionBaseTest, BroadcastCompatibleShapes) {
    MockContainer<double> a(Shape({3, 1}), {1, 2, 3});
    MockContainer<double> b(Shape({1, 4}), {10, 20, 30, 40});

    // Should broadcast to (3, 4)
    auto result_expr = make_expression(a) + make_expression(b);

    EXPECT_EQ(result_expr.shape(), Shape({3, 4}));

    // Check first row: 1 + [10, 20, 30, 40] = [11, 21, 31, 41]
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(1), 21.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(2), 31.0);
    EXPECT_DOUBLE_EQ(result_expr.template operator[]<double>(3), 41.0);
}

// ============================================================
// Complex Expression Tests
// ============================================================

TEST_F(ExpressionBaseTest, ChainedOperations) {
    MockContainer<double> a(Shape({3}), {1, 4, 9});

    // sqrt(abs(a)) + 1
    auto expr = sqrt(abs(make_expression(a))) + make_scalar_expression(1.0, Shape({3}));

    MockContainer<double> result(Shape({3}));
    expr.eval_to(result);

    EXPECT_DOUBLE_EQ(result[0], 2.0);  // sqrt(1) + 1
    EXPECT_DOUBLE_EQ(result[1], 3.0);  // sqrt(4) + 1
    EXPECT_DOUBLE_EQ(result[2], 4.0);  // sqrt(9) + 1
}

TEST_F(ExpressionBaseTest, NestedBinaryOperations) {
    MockContainer<double> a(Shape({2}), {2, 3});
    MockContainer<double> b(Shape({2}), {4, 5});
    MockContainer<double> c(Shape({2}), {1, 1});

    // (a + b) * c
    auto expr = (make_expression(a) + make_expression(b)) * make_expression(c);

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 6.0);  // (2+4)*1
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(1), 8.0);  // (3+5)*1
}

// ============================================================
// IEEE Compliance Tests
// ============================================================

TEST_F(ExpressionBaseTest, IEEEHandlingNaN) {
    MockContainer<double> a(Shape({2}), {1.0, std::numeric_limits<double>::quiet_NaN()});
    MockContainer<double> b(Shape({2}), {2.0, 3.0});

    auto expr = make_expression(a) + make_expression(b);

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 3.0);
    EXPECT_TRUE(std::isnan(expr.template operator[]<double>(1)));
}

TEST_F(ExpressionBaseTest, IEEEHandlingInfinity) {
    MockContainer<double> a(Shape({2}), {1.0, std::numeric_limits<double>::infinity()});
    MockContainer<double> b(Shape({2}), {2.0, 3.0});

    auto expr = make_expression(a) + make_expression(b);

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 3.0);
    EXPECT_TRUE(std::isinf(expr.template operator[]<double>(1)));
}

TEST_F(ExpressionBaseTest, IEEEDivisionByZero) {
    MockContainer<double> a(Shape({2}), {1.0, 2.0});
    MockContainer<double> b(Shape({2}), {2.0, 0.0});

    auto expr = make_expression(a) / make_expression(b);

    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 0.5);
    EXPECT_TRUE(std::isinf(expr.template operator[]<double>(1)));
}

// ============================================================
// Edge Cases and Error Handling
// ============================================================

TEST_F(ExpressionBaseTest, EmptyContainer) {
    MockContainer<double> empty(Shape({}));
    auto expr = make_expression(empty);

    EXPECT_EQ(expr.size(), 0);
    EXPECT_EQ(expr.shape(), Shape({}));
}

TEST_F(ExpressionBaseTest, SingleElementContainer) {
    MockContainer<double> single(Shape({1}), {42.0});
    auto expr = make_expression(single);

    EXPECT_EQ(expr.size(), 1);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 42.0);
}

TEST_F(ExpressionBaseTest, IncompatibleShapesBinaryOp) {
    MockContainer<double> a(Shape({2, 3}), 1.0);
    MockContainer<double> b(Shape({3, 2}), 2.0);

    // These shapes are not broadcastable
    EXPECT_THROW({
        auto expr = make_expression(a) + make_expression(b);
    }, DimensionError);
}

// ============================================================
// Type Conversion Tests
// ============================================================

TEST_F(ExpressionBaseTest, MixedTypeEvaluation) {
    MockContainer<int> a(Shape({3}), {1, 2, 3});
    auto expr = make_expression(a);

    // Evaluate as double
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(expr.template operator[]<double>(2), 3.0);
}

TEST_F(ExpressionBaseTest, TypeConversionInEvalTo) {
    MockContainer<int> source(Shape({2}), {10, 20});
    auto expr = make_expression(source);

    MockContainer<double> dest(Shape({2}));
    expr.eval_to(dest);

    EXPECT_DOUBLE_EQ(dest[0], 10.0);
    EXPECT_DOUBLE_EQ(dest[1], 20.0);
}

// ============================================================
// Performance and Optimization Tests
// ============================================================

TEST_F(ExpressionBaseTest, ComplexityEstimation) {
    MockContainer<double> a(Shape({100}), 1.0);
    MockContainer<double> b(Shape({100}), 2.0);

    auto expr_simple = make_expression(a);
    auto expr_binary = make_expression(a) + make_expression(b);

    EXPECT_EQ(expr_simple.complexity(), 100);
    EXPECT_GT(expr_binary.complexity(), expr_simple.complexity());
}

TEST_F(ExpressionBaseTest, ParallelizableCheck) {
    MockContainer<double> a(Shape({10}), 1.0);

    auto expr = make_expression(a);
    EXPECT_TRUE(expr.is_parallelizable());

    auto unary_expr = abs(expr);
    EXPECT_TRUE(unary_expr.is_parallelizable());
}

TEST_F(ExpressionBaseTest, VectorizableCheck) {
    MockContainer<double> a(Shape({10}), 1.0);

    auto expr = make_expression(a);
    EXPECT_TRUE(expr.is_vectorizable());  // Contiguous data
}

// ============================================================
// Expression Traits Tests
// ============================================================

TEST_F(ExpressionBaseTest, IsExpressionTrait) {
    MockContainer<double> container(Shape({2}), 1.0);
    auto expr = make_expression(container);

    EXPECT_FALSE(is_expression_v<MockContainer<double>>);
    EXPECT_TRUE(is_expression_v<decltype(expr)>);
}

} // namespace fem::numeric::test

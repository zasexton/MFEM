/**
 * @file test_ops_base.cpp
 * @brief Comprehensive unit tests for ops_base.h
 *
 * Tests all operations including:
 * - Arithmetic operations
 * - Transcendental functions
 * - Comparison operations
 * - Reduction operations
 * - IEEE 754 compliance
 * - Edge cases and error conditions
 */

#include <gtest/gtest.h>
#include <complex>
#include <limits>
#include <vector>
#include <cmath>
#include <random>

#include <base/ops_base.h>

using namespace fem::numeric;
using namespace fem::numeric::ops;

// Test fixtures
class OpsBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset to default options
        NumericOptions::defaults() = NumericOptions{};
    }

    // Helper to check if values are approximately equal
    template<typename T>
    bool approx_equal(T a, T b, T tol = std::numeric_limits<T>::epsilon() * 100) {
        if (std::isnan(a) && std::isnan(b)) return true;
        if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
        return std::abs(a - b) <= tol;
    }
};

// ============================================================
// Arithmetic Operations Tests
// ============================================================

TEST_F(OpsBaseTest, PlusOperation) {
    // Integer addition
    plus<int> add_int;
    EXPECT_EQ(add_int(5, 3), 8);
    EXPECT_EQ(add_int(-5, 3), -2);
    EXPECT_EQ(add_int(0, 0), 0);

    // Floating point addition
    plus<double> add_double;
    EXPECT_DOUBLE_EQ(add_double(3.14, 2.86), 6.0);
    EXPECT_DOUBLE_EQ(add_double(-1.5, 1.5), 0.0);

    // Void specialization (generic)
    plus<> add_generic;
    EXPECT_EQ(add_generic(5, 3), 8);
    EXPECT_DOUBLE_EQ(add_generic(3.14, 2.86), 6.0);
}

TEST_F(OpsBaseTest, MinusOperation) {
    minus<int> sub_int;
    EXPECT_EQ(sub_int(5, 3), 2);
    EXPECT_EQ(sub_int(3, 5), -2);

    minus<double> sub_double;
    EXPECT_DOUBLE_EQ(sub_double(5.5, 2.5), 3.0);
}

TEST_F(OpsBaseTest, MultipliesOperation) {
    multiplies<int> mul_int;
    EXPECT_EQ(mul_int(5, 3), 15);
    EXPECT_EQ(mul_int(-5, 3), -15);
    EXPECT_EQ(mul_int(0, 100), 0);

    multiplies<double> mul_double;
    EXPECT_DOUBLE_EQ(mul_double(2.5, 4.0), 10.0);
}

TEST_F(OpsBaseTest, DividesOperation) {
    divides<int> div_int;
    EXPECT_EQ(div_int(10, 2), 5);
    EXPECT_EQ(div_int(7, 2), 3);  // Integer division

    // Division by zero - integer throws
    EXPECT_THROW(div_int(5, 0), ComputationError);

    divides<double> div_double;
    EXPECT_DOUBLE_EQ(div_double(10.0, 2.0), 5.0);
    EXPECT_DOUBLE_EQ(div_double(7.0, 2.0), 3.5);

    // Division by zero - floating point returns inf
    double result = div_double(1.0, 0.0);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0);  // Positive infinity

    result = div_double(-1.0, 0.0);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0);  // Negative infinity
}

TEST_F(OpsBaseTest, ModulusOperation) {
    modulus<int> mod_int;
    EXPECT_EQ(mod_int(10, 3), 1);
    EXPECT_EQ(mod_int(9, 3), 0);
    EXPECT_EQ(mod_int(-10, 3), -1);

    // Modulo by zero throws
    EXPECT_THROW(mod_int(5, 0), ComputationError);

    modulus<double> mod_double;
    EXPECT_DOUBLE_EQ(mod_double(10.5, 3.0), 1.5);
}

// ============================================================
// Assignment Operations Tests
// ============================================================

TEST_F(OpsBaseTest, AssignmentOperations) {
    plus_assign<int> add_assign;
    int a = 5;
    EXPECT_EQ(add_assign(a, 3), 8);
    EXPECT_EQ(a, 8);  // Modified in place

    minus_assign<double> sub_assign;
    double b = 10.5;
    EXPECT_DOUBLE_EQ(sub_assign(b, 2.5), 8.0);
    EXPECT_DOUBLE_EQ(b, 8.0);

    multiplies_assign<int> mul_assign;
    int c = 4;
    EXPECT_EQ(mul_assign(c, 3), 12);
    EXPECT_EQ(c, 12);

    divides_assign<double> div_assign;
    double d = 20.0;
    EXPECT_DOUBLE_EQ(div_assign(d, 4.0), 5.0);
    EXPECT_DOUBLE_EQ(d, 5.0);

    modulus_assign<int> mod_assign;
    int e = 10;
    EXPECT_EQ(mod_assign(e, 3), 1);
    EXPECT_EQ(e, 1);
}

// ============================================================
// Unary Operations Tests
// ============================================================

TEST_F(OpsBaseTest, NegateOperation) {
    negate<int> neg_int;
    EXPECT_EQ(neg_int(5), -5);
    EXPECT_EQ(neg_int(-5), 5);
    EXPECT_EQ(neg_int(0), 0);

    negate<double> neg_double;
    EXPECT_DOUBLE_EQ(neg_double(3.14), -3.14);
}

TEST_F(OpsBaseTest, AbsOperation) {
    abs_op<int> abs_int;
    EXPECT_EQ(abs_int(5), 5);
    EXPECT_EQ(abs_int(-5), 5);
    EXPECT_EQ(abs_int(0), 0);

    abs_op<double> abs_double;
    EXPECT_DOUBLE_EQ(abs_double(-3.14), 3.14);
    EXPECT_DOUBLE_EQ(abs_double(3.14), 3.14);

    // Unsigned types
    abs_op<unsigned> abs_unsigned;
    EXPECT_EQ(abs_unsigned(5u), 5u);
}

TEST_F(OpsBaseTest, SignOperation) {
    sign_op<int> sign_int;
    EXPECT_EQ(sign_int(5), 1);
    EXPECT_EQ(sign_int(-5), -1);
    EXPECT_EQ(sign_int(0), 0);

    sign_op<double> sign_double;
    EXPECT_DOUBLE_EQ(sign_double(3.14), 1.0);
    EXPECT_DOUBLE_EQ(sign_double(-3.14), -1.0);
    EXPECT_DOUBLE_EQ(sign_double(0.0), 0.0);
}

TEST_F(OpsBaseTest, SqrtOperation) {
    sqrt_op<double> sqrt_double;
    EXPECT_DOUBLE_EQ(sqrt_double(4.0), 2.0);
    EXPECT_DOUBLE_EQ(sqrt_double(9.0), 3.0);
    EXPECT_DOUBLE_EQ(sqrt_double(0.0), 0.0);

    // Negative input for floating point returns NaN
    double result = sqrt_double(-1.0);
    EXPECT_TRUE(std::isnan(result));

    // Integer sqrt
    sqrt_op<int> sqrt_int;
    EXPECT_EQ(sqrt_int(4), 2);
    EXPECT_EQ(sqrt_int(9), 3);

    // Negative input for integer throws
    EXPECT_THROW(sqrt_int(-1), ComputationError);
}

// ============================================================
// Transcendental Functions Tests
// ============================================================

TEST_F(OpsBaseTest, TrigonometricOperations) {
    sin_op<double> sin_d;
    cos_op<double> cos_d;
    tan_op<double> tan_d;

    // Test at key angles
    EXPECT_NEAR(sin_d(0.0), 0.0, 1e-10);
    EXPECT_NEAR(sin_d(M_PI/2), 1.0, 1e-10);
    EXPECT_NEAR(sin_d(M_PI), 0.0, 1e-10);

    EXPECT_NEAR(cos_d(0.0), 1.0, 1e-10);
    EXPECT_NEAR(cos_d(M_PI/2), 0.0, 1e-10);
    EXPECT_NEAR(cos_d(M_PI), -1.0, 1e-10);

    EXPECT_NEAR(tan_d(0.0), 0.0, 1e-10);
    EXPECT_NEAR(tan_d(M_PI/4), 1.0, 1e-10);
}

TEST_F(OpsBaseTest, InverseTrigOperations) {
    asin_op<double> asin_d;
    acos_op<double> acos_d;
    atan_op<double> atan_d;

    EXPECT_NEAR(asin_d(0.0), 0.0, 1e-10);
    EXPECT_NEAR(asin_d(1.0), M_PI/2, 1e-10);
    EXPECT_NEAR(asin_d(-1.0), -M_PI/2, 1e-10);

    EXPECT_NEAR(acos_d(1.0), 0.0, 1e-10);
    EXPECT_NEAR(acos_d(0.0), M_PI/2, 1e-10);
    EXPECT_NEAR(acos_d(-1.0), M_PI, 1e-10);

    EXPECT_NEAR(atan_d(0.0), 0.0, 1e-10);
    EXPECT_NEAR(atan_d(1.0), M_PI/4, 1e-10);

    // Domain error for asin/acos
    double result = asin_d(2.0);  // Out of domain [-1, 1]
    EXPECT_TRUE(std::isnan(result));

    result = acos_d(2.0);
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(OpsBaseTest, HyperbolicOperations) {
    sinh_op<double> sinh_d;
    cosh_op<double> cosh_d;
    tanh_op<double> tanh_d;

    EXPECT_NEAR(sinh_d(0.0), 0.0, 1e-10);
    EXPECT_NEAR(cosh_d(0.0), 1.0, 1e-10);
    EXPECT_NEAR(tanh_d(0.0), 0.0, 1e-10);

    // Identity: cosh²(x) - sinh²(x) = 1
    double x = 1.5;
    double sinh_val = sinh_d(x);
    double cosh_val = cosh_d(x);
    EXPECT_NEAR(cosh_val * cosh_val - sinh_val * sinh_val, 1.0, 1e-10);
}

TEST_F(OpsBaseTest, ExponentialAndLogarithm) {
    exp_op<double> exp_d;
    log_op<double> log_d;
    log10_op<double> log10_d;
    log2_op<double> log2_d;

    EXPECT_NEAR(exp_d(0.0), 1.0, 1e-10);
    EXPECT_NEAR(exp_d(1.0), M_E, 1e-10);

    EXPECT_NEAR(log_d(1.0), 0.0, 1e-10);
    EXPECT_NEAR(log_d(M_E), 1.0, 1e-10);

    EXPECT_NEAR(log10_d(1.0), 0.0, 1e-10);
    EXPECT_NEAR(log10_d(10.0), 1.0, 1e-10);
    EXPECT_NEAR(log10_d(100.0), 2.0, 1e-10);

    EXPECT_NEAR(log2_d(1.0), 0.0, 1e-10);
    EXPECT_NEAR(log2_d(2.0), 1.0, 1e-10);
    EXPECT_NEAR(log2_d(8.0), 3.0, 1e-10);

    // Log of zero returns -inf
    double result = log_d(0.0);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0);

    // Log of negative returns NaN
    result = log_d(-1.0);
    EXPECT_TRUE(std::isnan(result));
}

// ============================================================
// Comparison Operations Tests
// ============================================================

TEST_F(OpsBaseTest, ComparisonOperations) {
    equal_to<int> eq;
    not_equal_to<int> ne;
    less<int> lt;
    greater<int> gt;
    less_equal<int> le;
    greater_equal<int> ge;

    EXPECT_TRUE(eq(5, 5));
    EXPECT_FALSE(eq(5, 3));

    EXPECT_TRUE(ne(5, 3));
    EXPECT_FALSE(ne(5, 5));

    EXPECT_TRUE(lt(3, 5));
    EXPECT_FALSE(lt(5, 3));
    EXPECT_FALSE(lt(5, 5));

    EXPECT_TRUE(gt(5, 3));
    EXPECT_FALSE(gt(3, 5));
    EXPECT_FALSE(gt(5, 5));

    EXPECT_TRUE(le(3, 5));
    EXPECT_TRUE(le(5, 5));
    EXPECT_FALSE(le(5, 3));

    EXPECT_TRUE(ge(5, 3));
    EXPECT_TRUE(ge(5, 5));
    EXPECT_FALSE(ge(3, 5));
}

TEST_F(OpsBaseTest, ComparisonWithNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();

    equal_to<double> eq;
    less<double> lt;

    // IEEE 754: NaN comparisons always return false
    EXPECT_FALSE(eq(nan, nan));
    EXPECT_FALSE(eq(nan, 1.0));
    EXPECT_FALSE(eq(1.0, nan));

    EXPECT_FALSE(lt(nan, 1.0));
    EXPECT_FALSE(lt(1.0, nan));
    EXPECT_FALSE(lt(nan, nan));
}

// ============================================================
// Special Operations Tests
// ============================================================

TEST_F(OpsBaseTest, PowerOperation) {
    power_op<double> pow_d;

    EXPECT_DOUBLE_EQ(pow_d(2.0, 3.0), 8.0);
    EXPECT_DOUBLE_EQ(pow_d(4.0, 0.5), 2.0);
    EXPECT_DOUBLE_EQ(pow_d(10.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(pow_d(5.0, -1.0), 0.2);
}

TEST_F(OpsBaseTest, Atan2Operation) {
    atan2_op<double> atan2_d;

    EXPECT_NEAR(atan2_d(0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(atan2_d(1.0, 0.0), M_PI/2, 1e-10);
    EXPECT_NEAR(atan2_d(0.0, -1.0), M_PI, 1e-10);
    EXPECT_NEAR(atan2_d(-1.0, 0.0), -M_PI/2, 1e-10);
}

TEST_F(OpsBaseTest, HypotOperation) {
    hypot_op<double> hypot_d;

    EXPECT_DOUBLE_EQ(hypot_d(3.0, 4.0), 5.0);
    EXPECT_DOUBLE_EQ(hypot_d(5.0, 12.0), 13.0);
    EXPECT_DOUBLE_EQ(hypot_d(0.0, 0.0), 0.0);
}

TEST_F(OpsBaseTest, MinMaxOperations) {
    min_op<int> min_i;
    max_op<int> max_i;

    EXPECT_EQ(min_i(3, 5), 3);
    EXPECT_EQ(min_i(5, 3), 3);
    EXPECT_EQ(max_i(3, 5), 5);
    EXPECT_EQ(max_i(5, 3), 5);

    // NaN handling for floating point
    min_op<double> min_d;
    max_op<double> max_d;
    double nan = std::numeric_limits<double>::quiet_NaN();

    // IEEE 754: min/max with NaN returns the non-NaN value
    EXPECT_DOUBLE_EQ(min_d(nan, 5.0), 5.0);
    EXPECT_DOUBLE_EQ(min_d(5.0, nan), 5.0);
    EXPECT_DOUBLE_EQ(max_d(nan, 5.0), 5.0);
    EXPECT_DOUBLE_EQ(max_d(5.0, nan), 5.0);
}

// ============================================================
// Rounding Operations Tests
// ============================================================

TEST_F(OpsBaseTest, RoundingOperations) {
    round_op<double> round_d;
    floor_op<double> floor_d;
    ceil_op<double> ceil_d;
    trunc_op<double> trunc_d;

    // Round
    EXPECT_DOUBLE_EQ(round_d(3.2), 3.0);
    EXPECT_DOUBLE_EQ(round_d(3.5), 4.0);  // Round half to even
    EXPECT_DOUBLE_EQ(round_d(3.7), 4.0);
    EXPECT_DOUBLE_EQ(round_d(-3.2), -3.0);
    EXPECT_DOUBLE_EQ(round_d(-3.7), -4.0);

    // Floor
    EXPECT_DOUBLE_EQ(floor_d(3.7), 3.0);
    EXPECT_DOUBLE_EQ(floor_d(-3.2), -4.0);

    // Ceil
    EXPECT_DOUBLE_EQ(ceil_d(3.2), 4.0);
    EXPECT_DOUBLE_EQ(ceil_d(-3.7), -3.0);

    // Trunc
    EXPECT_DOUBLE_EQ(trunc_d(3.7), 3.0);
    EXPECT_DOUBLE_EQ(trunc_d(-3.7), -3.0);

    // Integer types should pass through unchanged
    round_op<int> round_i;
    EXPECT_EQ(round_i(5), 5);
}

// ============================================================
// Logical Operations Tests
// ============================================================

TEST_F(OpsBaseTest, LogicalOperations) {
    logical_and land;
    logical_or lor;
    logical_not lnot;
    logical_xor lxor;

    EXPECT_TRUE(land(true, true));
    EXPECT_FALSE(land(true, false));
    EXPECT_FALSE(land(false, true));
    EXPECT_FALSE(land(false, false));

    EXPECT_TRUE(lor(true, true));
    EXPECT_TRUE(lor(true, false));
    EXPECT_TRUE(lor(false, true));
    EXPECT_FALSE(lor(false, false));

    EXPECT_FALSE(lnot(true));
    EXPECT_TRUE(lnot(false));

    EXPECT_FALSE(lxor(true, true));
    EXPECT_TRUE(lxor(true, false));
    EXPECT_TRUE(lxor(false, true));
    EXPECT_FALSE(lxor(false, false));

    // Test with numeric types
    EXPECT_TRUE(land(5, 3));  // Non-zero values
    EXPECT_FALSE(land(5, 0));
    EXPECT_TRUE(lor(0, 3));
    EXPECT_FALSE(lnot(5));
    EXPECT_TRUE(lnot(0));
}

// ============================================================
// Bitwise Operations Tests
// ============================================================

TEST_F(OpsBaseTest, BitwiseOperations) {
    bit_and<int> band;
    bit_or<int> bor;
    bit_xor<int> bxor;
    bit_not<int> bnot;
    left_shift<int> lshift;
    right_shift<int> rshift;

    EXPECT_EQ(band(0b1010, 0b1100), 0b1000);
    EXPECT_EQ(bor(0b1010, 0b1100), 0b1110);
    EXPECT_EQ(bxor(0b1010, 0b1100), 0b0110);
    EXPECT_EQ(bnot(0b1010), ~0b1010);

    EXPECT_EQ(lshift(0b0001, 3), 0b1000);
    EXPECT_EQ(rshift(0b1000, 2), 0b0010);
}

// ============================================================
// Reduction Operations Tests
// ============================================================

TEST_F(OpsBaseTest, SumOperation) {
    sum_op<int> sum_i;
    std::vector<int> vec_i = {1, 2, 3, 4, 5};

    EXPECT_EQ(sum_i(vec_i.begin(), vec_i.end()), 15);
    EXPECT_EQ(sum_i(vec_i), 15);  // Container overload

    sum_op<double> sum_d;
    std::vector<double> vec_d = {1.5, 2.5, 3.0};
    EXPECT_DOUBLE_EQ(sum_d(vec_d), 7.0);

    // Empty sequence
    std::vector<int> empty;
    EXPECT_EQ(sum_i(empty), 0);
}

TEST_F(OpsBaseTest, ProductOperation) {
    product_op<int> prod_i;
    std::vector<int> vec = {2, 3, 4};

    EXPECT_EQ(prod_i(vec), 24);

    // Empty sequence returns identity (1)
    std::vector<int> empty;
    EXPECT_EQ(prod_i(empty), 1);
}

TEST_F(OpsBaseTest, MeanOperation) {
    mean_op<double> mean_d;
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};

    EXPECT_DOUBLE_EQ(mean_d(vec), 3.0);

    // Empty sequence throws
    std::vector<double> empty;
    EXPECT_THROW(mean_d(empty), ComputationError);
}

TEST_F(OpsBaseTest, VarianceOperation) {
    variance_op<double> var_d;
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Population variance
    EXPECT_DOUBLE_EQ(var_d(vec, false), 2.0);

    // Sample variance
    EXPECT_DOUBLE_EQ(var_d(vec, true), 2.5);

    // Single element - sample variance should throw
    std::vector<double> single = {5.0};
    EXPECT_THROW(var_d(single, true), ComputationError);

    // Empty sequence throws
    std::vector<double> empty;
    EXPECT_THROW(var_d(empty), ComputationError);
}

TEST_F(OpsBaseTest, StdDevOperation) {
    stddev_op<double> stddev_d;
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Population standard deviation
    EXPECT_NEAR(stddev_d(vec, false), std::sqrt(2.0), 1e-10);

    // Sample standard deviation
    EXPECT_NEAR(stddev_d(vec, true), std::sqrt(2.5), 1e-10);
}

// ============================================================
// OperationDispatcher Tests
// ============================================================

TEST_F(OpsBaseTest, OperationDispatcherBinary) {
    using OpType = OperationDispatcher::OpType;

    auto add_func = OperationDispatcher::get_binary_op<int>(OpType::ADD);
    EXPECT_EQ(add_func(5, 3), 8);

    auto mul_func = OperationDispatcher::get_binary_op<double>(OpType::MUL);
    EXPECT_DOUBLE_EQ(mul_func(2.5, 4.0), 10.0);

    auto pow_func = OperationDispatcher::get_binary_op<double>(OpType::POW);
    EXPECT_DOUBLE_EQ(pow_func(2.0, 3.0), 8.0);

    // Invalid operation type
    EXPECT_THROW(
        OperationDispatcher::get_binary_op<int>(OpType::SIN),
        std::invalid_argument
    );
}

TEST_F(OpsBaseTest, OperationDispatcherUnary) {
    using OpType = OperationDispatcher::OpType;

    auto neg_func = OperationDispatcher::get_unary_op<int>(OpType::NEG);
    EXPECT_EQ(neg_func(5), -5);

    auto sin_func = OperationDispatcher::get_unary_op<double>(OpType::SIN);
    EXPECT_NEAR(sin_func(M_PI/2), 1.0, 1e-10);

    auto sqrt_func = OperationDispatcher::get_unary_op<double>(OpType::SQRT);
    EXPECT_DOUBLE_EQ(sqrt_func(9.0), 3.0);

    // Invalid operation type
    EXPECT_THROW(
        OperationDispatcher::get_unary_op<int>(OpType::ADD),
        std::invalid_argument
    );
}

TEST_F(OpsBaseTest, OperationDispatcherComparison) {
    using OpType = OperationDispatcher::OpType;

    auto eq_func = OperationDispatcher::get_comparison_op<int>(OpType::EQ);
    EXPECT_TRUE(eq_func(5, 5));
    EXPECT_FALSE(eq_func(5, 3));

    auto lt_func = OperationDispatcher::get_comparison_op<double>(OpType::LT);
    EXPECT_TRUE(lt_func(3.0, 5.0));
    EXPECT_FALSE(lt_func(5.0, 3.0));
}

TEST_F(OpsBaseTest, OperationDispatcherHelpers) {
    using OpType = OperationDispatcher::OpType;

    // Test is_unary
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::NEG));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SIN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SQRT));
    EXPECT_FALSE(OperationDispatcher::is_unary(OpType::ADD));
    EXPECT_FALSE(OperationDispatcher::is_unary(OpType::POW));

    // Test is_reduction
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::SUM));
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::MEAN));
    EXPECT_FALSE(OperationDispatcher::is_reduction(OpType::ADD));
    EXPECT_FALSE(OperationDispatcher::is_reduction(OpType::SIN));

    // Test is_comparison
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::EQ));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::LT));
    EXPECT_FALSE(OperationDispatcher::is_comparison(OpType::ADD));
    EXPECT_FALSE(OperationDispatcher::is_comparison(OpType::SIN));
}

// ============================================================
// IEEE 754 Compliance Tests
// ============================================================

TEST_F(OpsBaseTest, IEEEInfinityHandling) {
    double inf = std::numeric_limits<double>::infinity();
    double ninf = -inf;

    plus<double> add;
    minus<double> sub;
    multiplies<double> mul;
    divides<double> div;

    // inf + finite = inf
    EXPECT_TRUE(std::isinf(add(inf, 5.0)));
    EXPECT_GT(add(inf, 5.0), 0);

    // inf - inf = NaN
    EXPECT_TRUE(std::isnan(sub(inf, inf)));

    // inf * 0 = NaN
    EXPECT_TRUE(std::isnan(mul(inf, 0.0)));

    // finite / inf = 0
    EXPECT_DOUBLE_EQ(div(5.0, inf), 0.0);

    // inf / inf = NaN
    EXPECT_TRUE(std::isnan(div(inf, inf)));
}

TEST_F(OpsBaseTest, IEEENaNPropagation) {
    double nan = std::numeric_limits<double>::quiet_NaN();

    // NaN propagates through arithmetic operations
    plus<double> add;
    EXPECT_TRUE(std::isnan(add(nan, 5.0)));
    EXPECT_TRUE(std::isnan(add(5.0, nan)));

    // NaN propagates through transcendental functions
    sin_op<double> sin_d;
    EXPECT_TRUE(std::isnan(sin_d(nan)));

    exp_op<double> exp_d;
    EXPECT_TRUE(std::isnan(exp_d(nan)));

    // Special: min/max with NaN returns non-NaN value
    min_op<double> min_d;
    EXPECT_DOUBLE_EQ(min_d(nan, 5.0), 5.0);
}

// ============================================================
// Complex Number Support Tests (if applicable)
// ============================================================

TEST_F(OpsBaseTest, ComplexNumberOperations) {
    using Complex = std::complex<double>;

    plus<Complex> add_c;
    Complex c1(3.0, 4.0);
    Complex c2(1.0, 2.0);
    Complex result = add_c(c1, c2);

    EXPECT_DOUBLE_EQ(result.real(), 4.0);
    EXPECT_DOUBLE_EQ(result.imag(), 6.0);

    multiplies<Complex> mul_c;
    result = mul_c(Complex(2.0, 3.0), Complex(1.0, 4.0));
    EXPECT_DOUBLE_EQ(result.real(), -10.0);  // (2*1 - 3*4)
    EXPECT_DOUBLE_EQ(result.imag(), 11.0);   // (2*4 + 3*1)
}

// ============================================================
// Edge Cases and Error Conditions
// ============================================================

TEST_F(OpsBaseTest, EdgeCases) {
    // Very large numbers
    double large = std::numeric_limits<double>::max();
    plus<double> add;
    double result = add(large, large);
    EXPECT_TRUE(std::isinf(result));  // Overflow to infinity

    // Very small numbers (denormalized)
    double tiny = std::numeric_limits<double>::denorm_min();
    multiplies<double> mul;

    // Multiplying denorm_min by 0.5 underflows to zero (correct behavior)
    result = mul(tiny, 0.5);
    EXPECT_EQ(result, 0.0);  // Underflow to zero is expected

    // But multiplying by 2.0 should still give a denormalized number
    result = mul(tiny, 2.0);
    EXPECT_GT(result, 0.0);  // Should be 2 * denorm_min
    EXPECT_EQ(result, 2.0 * tiny);

    // Verify denormalized arithmetic works
    double small_denorm = std::numeric_limits<double>::denorm_min() * 100;
    result = mul(small_denorm, 1.0);
    EXPECT_EQ(result, small_denorm);  // Identity multiplication preserves value

    // Precision limits
    double epsilon = std::numeric_limits<double>::epsilon();
    EXPECT_DOUBLE_EQ(add(1.0, epsilon/2), 1.0);  // Below precision threshold
}

// ============================================================
// Performance Characteristics Tests
// ============================================================

TEST_F(OpsBaseTest, PerformanceCharacteristics) {
    // This test verifies that operations have expected complexity
    // Not timing-based, just structural verification

    // Verify that void specializations exist and work
    plus<> add_void;
    auto result = add_void(5, 3.14);  // Mixed types
    EXPECT_DOUBLE_EQ(result, 8.14);

    // Verify perfect forwarding works
    std::string s1 = "Hello ";
    std::string s2 = "World";
    auto str_result = add_void(s1, s2);
    EXPECT_EQ(str_result, "Hello World");
}

// ============================================================
// Integration Tests
// ============================================================

TEST_F(OpsBaseTest, ChainedOperations) {
    // Test that operations can be chained together
    plus<double> add;
    multiplies<double> mul;
    sqrt_op<double> sqrt;

    // (3 + 4) * 2 = 14, sqrt(14) ≈ 3.74
    double result = sqrt(mul(add(3.0, 4.0), 2.0));
    EXPECT_NEAR(result, std::sqrt(14.0), 1e-10);
}

TEST_F(OpsBaseTest, MixedTypeOperations) {
    // Test operations with mixed types using void specialization
    plus<> add;

    // int + double
    auto result1 = add(5, 3.14);
    EXPECT_DOUBLE_EQ(result1, 8.14);

    // float + int
    auto result2 = add(2.5f, 3);
    EXPECT_FLOAT_EQ(result2, 5.5f);
}

// ============================================================
// Options and Configuration Tests
// ============================================================

TEST_F(OpsBaseTest, NumericOptionsIntegration) {
    // Enable finite checking
    NumericOptions::defaults().check_finite = true;

    // Operations should still work with finite values
    plus<double> add;
    EXPECT_DOUBLE_EQ(add(3.0, 4.0), 7.0);

    // NaN should still propagate (it's allowed)
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(add(nan, 5.0)));

    // Reset options
    NumericOptions::defaults().check_finite = false;
}

/**
 * @file test_ops_base_coverage.cpp
 * @brief Additional tests to achieve complete coverage of ops_base.h
 */

#include <gtest/gtest.h>
#include <complex>
#include <limits>
#include <vector>
#include <cmath>

#include <base/ops_base.h>

using namespace fem::numeric;
using namespace fem::numeric::ops;

class OpsBaseCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {
        NumericOptions::defaults() = NumericOptions{};
    }
};

// Test all untested arithmetic paths
TEST_F(OpsBaseCoverageTest, ArithmeticCoverage) {
    // Test divides with integer division by zero (line 138)
    divides<int> div_i;
    EXPECT_THROW(div_i(10, 0), ComputationError);

    // Test divides_assign with division by zero for integers (line 218)
    divides_assign<int> div_assign_i;
    int val = 10;
    EXPECT_THROW(div_assign_i(val, 0), ComputationError);

    // Test divides_assign with zero for floating point (line 216)
    divides_assign<double> div_assign_d;
    double dval = 10.0;
    double result = div_assign_d(dval, 0.0);
    EXPECT_TRUE(std::isinf(result));

    // Test modulus_assign with zero (line 230)
    modulus_assign<int> mod_assign_i;
    int mval = 10;
    EXPECT_THROW(mod_assign_i(mval, 0), ComputationError);

    // Test modulus_assign floating point path (line 233)
    modulus_assign<double> mod_assign_d;
    double mdval = 10.5;
    EXPECT_DOUBLE_EQ(mod_assign_d(mdval, 3.0), 1.5);
}

// Test all sign_op branches
TEST_F(OpsBaseCoverageTest, SignOperationComplete) {
    sign_op<int> sign_i;

    // Test positive branch (line 286)
    EXPECT_EQ(sign_i(10), 1);

    // Test negative branch (line 287)
    EXPECT_EQ(sign_i(-10), -1);

    // Test zero branch (line 288)
    EXPECT_EQ(sign_i(0), 0);
}

// Test integer paths for transcendental functions
TEST_F(OpsBaseCoverageTest, TranscendentalIntegerPaths) {
    // Test sin_op integer path (line 330)
    sin_op<int> sin_i;
    EXPECT_EQ(sin_i(0), 0);
    EXPECT_EQ(sin_i(1), 0);  // sin(1 radian) ≈ 0.84, rounds to 0

    // Test cos_op integer path (line 342)
    cos_op<int> cos_i;
    EXPECT_EQ(cos_i(0), 1);

    // Test tan_op integer path (line 354)
    tan_op<int> tan_i;
    EXPECT_EQ(tan_i(0), 0);

    // Test asin_op integer paths (lines 366-369)
    asin_op<int> asin_i;
    EXPECT_EQ(asin_i(0), 0);
    EXPECT_EQ(asin_i(1), 1);  // asin(1) = π/2 ≈ 1.57, rounds to 1
    EXPECT_THROW(asin_i(2), ComputationError);  // Out of domain

    // Test acos_op integer paths (lines 381-384)
    acos_op<int> acos_i;
    EXPECT_EQ(acos_i(1), 0);
    EXPECT_THROW(acos_i(2), ComputationError);  // Out of domain

    // Test atan_op integer path (line 396)
    atan_op<int> atan_i;
    EXPECT_EQ(atan_i(0), 0);
    EXPECT_EQ(atan_i(1), 0);  // atan(1) = π/4 ≈ 0.785, rounds to 0

    // Test sinh_op integer path (line 409)
    sinh_op<int> sinh_i;
    EXPECT_EQ(sinh_i(0), 0);

    // Test cosh_op integer path (line 421)
    cosh_op<int> cosh_i;
    EXPECT_EQ(cosh_i(0), 1);

    // Test tanh_op integer path (line 433)
    tanh_op<int> tanh_i;
    EXPECT_EQ(tanh_i(0), 0);

    // Test exp_op integer path (line 445)
    exp_op<int> exp_i;
    EXPECT_EQ(exp_i(0), 1);
    EXPECT_EQ(exp_i(1), 2);  // e^1 ≈ 2.718, rounds to 2

    // Test log_op integer paths (lines 459, 465)
    log_op<int> log_i;
    EXPECT_EQ(log_i(1), 0);
    EXPECT_EQ(log_i(2), 0);  // ln(2) ≈ 0.693, rounds to 0
    EXPECT_THROW(log_i(0), ComputationError);
    EXPECT_THROW(log_i(-1), ComputationError);

    // Test log10_op integer paths (lines 478, 484)
    log10_op<int> log10_i;
    EXPECT_EQ(log10_i(1), 0);
    EXPECT_EQ(log10_i(10), 1);
    EXPECT_THROW(log10_i(0), ComputationError);
    EXPECT_THROW(log10_i(-1), ComputationError);

    // Test log2_op integer paths (lines 497, 503)
    log2_op<int> log2_i;
    EXPECT_EQ(log2_i(1), 0);
    EXPECT_EQ(log2_i(2), 1);
    EXPECT_THROW(log2_i(0), ComputationError);
    EXPECT_THROW(log2_i(-1), ComputationError);
}

// Test floating point log paths with special values
TEST_F(OpsBaseCoverageTest, LogarithmFloatingPointPaths) {
    // Test log_op with zero and negative (lines 457, 463)
    log_op<double> log_d;
    double result = log_d(0.0);
    EXPECT_TRUE(std::isinf(result) && result < 0);
    result = log_d(-1.0);
    EXPECT_TRUE(std::isnan(result));

    // Test log10_op with zero and negative (line 476)
    log10_op<double> log10_d;
    result = log10_d(0.0);
    EXPECT_TRUE(std::isinf(result) && result < 0);
    result = log10_d(-1.0);
    EXPECT_TRUE(std::isnan(result));

    // Test log2_op with zero and negative (line 495)
    log2_op<double> log2_d;
    result = log2_d(0.0);
    EXPECT_TRUE(std::isinf(result) && result < 0);
    result = log2_d(-1.0);
    EXPECT_TRUE(std::isnan(result));
}

// Test all comparison operations with proper types
TEST_F(OpsBaseCoverageTest, ComparisonOperationsComplete) {
    // Test equal_to with floating point and NaN (lines 517-520, 522)
    equal_to<double> eq_d;
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(eq_d(nan, nan));
    EXPECT_FALSE(eq_d(nan, 5.0));
    EXPECT_FALSE(eq_d(5.0, nan));
    EXPECT_TRUE(eq_d(5.0, 5.0));
    EXPECT_FALSE(eq_d(5.0, 3.0));

    // Test equal_to with integers (no NaN check)
    equal_to<int> eq_i;
    EXPECT_TRUE(eq_i(5, 5));
    EXPECT_FALSE(eq_i(5, 3));

    // Test not_equal_to (line 538)
    not_equal_to<int> ne_i;
    EXPECT_TRUE(ne_i(5, 3));
    EXPECT_FALSE(ne_i(5, 5));

    // Test less with NaN (lines 556-559, 561)
    less<double> lt_d;
    EXPECT_FALSE(lt_d(nan, 5.0));
    EXPECT_FALSE(lt_d(5.0, nan));
    EXPECT_FALSE(lt_d(nan, nan));
    EXPECT_TRUE(lt_d(3.0, 5.0));
    EXPECT_FALSE(lt_d(5.0, 3.0));

    // Test greater (line 577)
    greater<int> gt_i;
    EXPECT_TRUE(gt_i(5, 3));
    EXPECT_FALSE(gt_i(3, 5));

    // Test less_equal (line 593)
    less_equal<int> le_i;
    EXPECT_TRUE(le_i(3, 5));
    EXPECT_TRUE(le_i(5, 5));
    EXPECT_FALSE(le_i(5, 3));

    // Test greater_equal (line 609)
    greater_equal<int> ge_i;
    EXPECT_TRUE(ge_i(5, 3));
    EXPECT_TRUE(ge_i(5, 5));
    EXPECT_FALSE(ge_i(3, 5));
}

// Test special operations with integer types
TEST_F(OpsBaseCoverageTest, SpecialOperationsInteger) {
    // Test power_op integer path (line 633)
    power_op<int> pow_i;
    EXPECT_EQ(pow_i(2, 3), 8);
    EXPECT_EQ(pow_i(5, 0), 1);

    // Test atan2_op integer path (line 649)
    atan2_op<int> atan2_i;
    EXPECT_EQ(atan2_i(0, 1), 0);
    EXPECT_EQ(atan2_i(1, 1), 0);  // atan2(1,1) = π/4 ≈ 0.785, rounds to 0

    // Test hypot_op integer path (line 661)
    hypot_op<int> hypot_i;
    EXPECT_EQ(hypot_i(3, 4), 5);
    EXPECT_EQ(hypot_i(5, 12), 13);
}

// Test min/max operations thoroughly
TEST_F(OpsBaseCoverageTest, MinMaxComplete) {
    // Test min_op with NaN in both positions (lines 672-673, 675)
    min_op<double> min_d;
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(min_d(nan, 5.0), 5.0);  // First is NaN
    EXPECT_DOUBLE_EQ(min_d(5.0, nan), 5.0);  // Second is NaN
    EXPECT_DOUBLE_EQ(min_d(3.0, 5.0), 3.0);  // Normal case

    // Test min_op with integers (no NaN path)
    min_op<int> min_i;
    EXPECT_EQ(min_i(3, 5), 3);
    EXPECT_EQ(min_i(5, 3), 3);

    // Test max_op with NaN in both positions (lines 684-685, 687)
    max_op<double> max_d;
    EXPECT_DOUBLE_EQ(max_d(nan, 5.0), 5.0);  // First is NaN
    EXPECT_DOUBLE_EQ(max_d(5.0, nan), 5.0);  // Second is NaN
    EXPECT_DOUBLE_EQ(max_d(3.0, 5.0), 5.0);  // Normal case

    // Test max_op with integers
    max_op<int> max_i;
    EXPECT_EQ(max_i(3, 5), 5);
    EXPECT_EQ(max_i(5, 3), 5);
}

// Test rounding operations with integer types
TEST_F(OpsBaseCoverageTest, RoundingOperationsInteger) {
    // Test round_op integer path (line 702)
    round_op<int> round_i;
    EXPECT_EQ(round_i(5), 5);

    // Test round_op floating point path (line 700)
    round_op<double> round_d;
    EXPECT_DOUBLE_EQ(round_d(3.7), 4.0);
    EXPECT_DOUBLE_EQ(round_d(3.2), 3.0);

    // Test floor_op integer path (line 714)
    floor_op<int> floor_i;
    EXPECT_EQ(floor_i(5), 5);

    // Test floor_op floating point path (line 712)
    floor_op<double> floor_d;
    EXPECT_DOUBLE_EQ(floor_d(3.7), 3.0);

    // Test ceil_op integer path (line 726)
    ceil_op<int> ceil_i;
    EXPECT_EQ(ceil_i(5), 5);

    // Test ceil_op floating point path (line 724)
    ceil_op<double> ceil_d;
    EXPECT_DOUBLE_EQ(ceil_d(3.2), 4.0);

    // Test trunc_op integer path (line 738)
    trunc_op<int> trunc_i;
    EXPECT_EQ(trunc_i(5), 5);

    // Test trunc_op floating point path (line 736)
    trunc_op<double> trunc_d;
    EXPECT_DOUBLE_EQ(trunc_d(3.7), 3.0);
}

// Test logical operations thoroughly
TEST_F(OpsBaseCoverageTest, LogicalOperationsComplete) {
    logical_and land;
    logical_or lor;
    logical_not lnot;
    logical_xor lxor;

    // Test logical_and all paths (line 750)
    EXPECT_TRUE(land(true, true));
    EXPECT_FALSE(land(true, false));
    EXPECT_FALSE(land(false, true));
    EXPECT_FALSE(land(false, false));
    EXPECT_TRUE(land(5, 3));
    EXPECT_FALSE(land(0, 5));

    // Test logical_or all paths (line 757)
    EXPECT_TRUE(lor(true, true));
    EXPECT_TRUE(lor(true, false));
    EXPECT_TRUE(lor(false, true));
    EXPECT_FALSE(lor(false, false));
    EXPECT_TRUE(lor(5, 0));
    EXPECT_FALSE(lor(0, 0));

    // Test logical_not all paths (line 764)
    EXPECT_FALSE(lnot(true));
    EXPECT_TRUE(lnot(false));
    EXPECT_FALSE(lnot(5));
    EXPECT_TRUE(lnot(0));

    // Test logical_xor all paths (line 771)
    EXPECT_FALSE(lxor(true, true));
    EXPECT_TRUE(lxor(true, false));
    EXPECT_TRUE(lxor(false, true));
    EXPECT_FALSE(lxor(false, false));
    EXPECT_FALSE(lxor(5, 3));
    EXPECT_TRUE(lxor(5, 0));
}

// Test bitwise operations
TEST_F(OpsBaseCoverageTest, BitwiseOperationsComplete) {
    // Test bit_and (line 783)
    bit_and<int> band;
    EXPECT_EQ(band(0xFF, 0x0F), 0x0F);
    EXPECT_EQ(band(0b1010, 0b1100), 0b1000);

    // Test bit_or (line 791)
    bit_or<int> bor;
    EXPECT_EQ(bor(0xF0, 0x0F), 0xFF);
    EXPECT_EQ(bor(0b1010, 0b0101), 0b1111);

    // Test bit_xor (line 799)
    bit_xor<int> bxor;
    EXPECT_EQ(bxor(0xFF, 0xF0), 0x0F);
    EXPECT_EQ(bxor(0b1010, 0b1100), 0b0110);

    // Test bit_not (line 807)
    bit_not<int> bnot;
    EXPECT_EQ(bnot(0), ~0);
    EXPECT_EQ(bnot(0xFF), ~0xFF);

    // Test left_shift (line 815)
    left_shift<int> lshift;
    EXPECT_EQ(lshift(1, 4), 16);
    EXPECT_EQ(lshift(0b0001, 3), 0b1000);

    // Test right_shift (line 823)
    right_shift<int> rshift;
    EXPECT_EQ(rshift(16, 4), 1);
    EXPECT_EQ(rshift(0b1000, 2), 0b0010);
}

// Test reduction operations with iterators
TEST_F(OpsBaseCoverageTest, ReductionOperationsIterators) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<double> dvec = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Test sum_op iterator version (lines 835-836)
    sum_op<int> sum_i;
    EXPECT_EQ(sum_i(vec.begin(), vec.end()), 15);

    // Test product_op iterator version (lines 849-850)
    product_op<int> prod_i;
    EXPECT_EQ(prod_i(vec.begin(), vec.end()), 120);

    // Test mean_op iterator version (lines 863-869)
    mean_op<double> mean_d;
    EXPECT_DOUBLE_EQ(mean_d(dvec.begin(), dvec.end()), 3.0);

    // Test empty mean (line 866)
    std::vector<double> empty;
    EXPECT_THROW(mean_d(empty.begin(), empty.end()), ComputationError);

    // Test variance_op iterator version (lines 882-899)
    variance_op<double> var_d;
    EXPECT_DOUBLE_EQ(var_d(dvec.begin(), dvec.end(), false), 2.0);  // Population
    EXPECT_DOUBLE_EQ(var_d(dvec.begin(), dvec.end(), true), 2.5);   // Sample

    // Test variance with empty (line 885)
    EXPECT_THROW(var_d(empty.begin(), empty.end(), false), ComputationError);

    // Test variance with single element for sample (line 888)
    std::vector<double> single = {5.0};
    EXPECT_THROW(var_d(single.begin(), single.end(), true), ComputationError);

    // Test stddev_op iterator version (lines 912-913)
    stddev_op<double> std_d;
    EXPECT_NEAR(std_d(dvec.begin(), dvec.end(), false), std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(std_d(dvec.begin(), dvec.end(), true), std::sqrt(2.5), 1e-10);
}

// Test OperationDispatcher exhaustively
TEST_F(OpsBaseCoverageTest, OperationDispatcherExhaustive) {
    using OpType = OperationDispatcher::OpType;

    // Test all binary operations (lines 957-966)
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::ADD));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::SUB));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::MUL));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::DIV));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<double>(OpType::POW));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::MOD));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::MIN));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<int>(OpType::MAX));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<double>(OpType::ATAN2));
    EXPECT_NO_THROW(OperationDispatcher::get_binary_op<double>(OpType::HYPOT));

    // Test all unary operations (lines 975-995)
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<int>(OpType::NEG));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<int>(OpType::ABS));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<int>(OpType::SIGN));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::SQRT));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::EXP));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::LOG));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::LOG10));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::LOG2));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::SIN));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::COS));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::TAN));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::ASIN));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::ACOS));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::ATAN));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::SINH));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::COSH));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::TANH));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::ROUND));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::FLOOR));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::CEIL));
    EXPECT_NO_THROW(OperationDispatcher::get_unary_op<double>(OpType::TRUNC));

    // Test all comparison operations (lines 1004-1009)
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::EQ));
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::NE));
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::LT));
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::LE));
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::GT));
    EXPECT_NO_THROW(OperationDispatcher::get_comparison_op<int>(OpType::GE));

    // Test helper functions comprehensively (lines 1017-1042)
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::NEG));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::ABS));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SIGN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SQRT));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::EXP));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::LOG));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::LOG10));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::LOG2));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SIN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::COS));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::TAN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::ASIN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::ACOS));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::ATAN));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::SINH));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::COSH));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::TANH));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::ROUND));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::FLOOR));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::CEIL));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::TRUNC));
    EXPECT_TRUE(OperationDispatcher::is_unary(OpType::BIT_NOT));
    EXPECT_FALSE(OperationDispatcher::is_unary(OpType::ADD));

    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::SUM));
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::PRODUCT));
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::MEAN));
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::VARIANCE));
    EXPECT_TRUE(OperationDispatcher::is_reduction(OpType::STDDEV));
    EXPECT_FALSE(OperationDispatcher::is_reduction(OpType::ADD));

    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::EQ));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::NE));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::LT));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::LE));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::GT));
    EXPECT_TRUE(OperationDispatcher::is_comparison(OpType::GE));
    EXPECT_FALSE(OperationDispatcher::is_comparison(OpType::ADD));
}

// Test void specializations thoroughly
TEST_F(OpsBaseCoverageTest, VoidSpecializationsComplete) {
    // Test minus void specialization (line 108)
    minus<> sub;
    EXPECT_EQ(sub(10, 3), 7);
    EXPECT_DOUBLE_EQ(sub(10.5, 3.5), 7.0);

    // Test multiplies void specialization (line 125)
    multiplies<> mul;
    EXPECT_EQ(mul(3, 4), 12);
    EXPECT_DOUBLE_EQ(mul(2.5, 4.0), 10.0);

    // Test divides void specialization (line 150)
    divides<> div;
    EXPECT_EQ(div(10, 2), 5);
    EXPECT_DOUBLE_EQ(div(10.0, 2.5), 4.0);

    // Test modulus void specialization (line 174)
    modulus<> mod;
    EXPECT_EQ(mod(10, 3), 1);

    // Test negate void specialization (line 257)
    negate<> neg;
    EXPECT_EQ(neg(5), -5);
    EXPECT_DOUBLE_EQ(neg(3.14), -3.14);

    // Test abs_op void specialization (line 278)
    abs_op<> abs;
    EXPECT_EQ(abs(-5), 5);
    EXPECT_DOUBLE_EQ(abs(-3.14), 3.14);

    // Test sign_op void specialization (lines 297-299)
    sign_op<> sign;
    EXPECT_EQ(sign(10), 1);
    EXPECT_EQ(sign(-10), -1);
    EXPECT_EQ(sign(0), 0);

    // Test comparison void specializations (lines 531, 547, 570, 586, 602, 618)
    equal_to<> eq;
    not_equal_to<> ne;
    less<> lt;
    greater<> gt;
    less_equal<> le;
    greater_equal<> ge;

    EXPECT_TRUE(eq(5, 5));
    EXPECT_FALSE(eq(5, 3));
    EXPECT_TRUE(ne(5, 3));
    EXPECT_FALSE(ne(5, 5));
    EXPECT_TRUE(lt(3, 5));
    EXPECT_FALSE(lt(5, 3));
    EXPECT_TRUE(gt(5, 3));
    EXPECT_FALSE(gt(3, 5));
    EXPECT_TRUE(le(3, 5));
    EXPECT_TRUE(le(5, 5));
    EXPECT_TRUE(ge(5, 3));
    EXPECT_TRUE(ge(5, 5));
}

// Test check functions with options enabled
TEST_F(OpsBaseCoverageTest, CheckFunctionsEnabled) {
    // Enable finite checking to test the check paths
    NumericOptions::defaults().check_finite = true;

    // Test UnaryOp check_input with NaN (lines 31-35)
    abs_op<double> abs_d;
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW(abs_d(nan));  // NaN is allowed
    EXPECT_NO_THROW(abs_d(5.0));

    // Test with infinity
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_NO_THROW(abs_d(inf));

    // Test BinaryOp check_inputs (lines 48-51)
    plus<double> add_d;
    EXPECT_NO_THROW(add_d(5.0, 3.0));
    EXPECT_NO_THROW(add_d(nan, 5.0));
    EXPECT_NO_THROW(add_d(5.0, nan));
    EXPECT_NO_THROW(add_d(inf, 5.0));

    // Test ReductionOp check_range (lines 62-64)
    sum_op<double> sum_d;
    std::vector<double> vec = {1.0, 2.0, 3.0};
    EXPECT_NO_THROW(sum_d(vec));

    // Reset options
    NumericOptions::defaults().check_finite = false;
}

// Test mixed type operations with void specialization
TEST_F(OpsBaseCoverageTest, MixedTypeVoidOperations) {
    plus<> add;

    // Test the special branches for mixed integral/floating (lines 85-88)
    // int + float
    float f = 3.5f;
    int i = 2;
    auto result1 = add(i, f);  // Should trigger line 88
    EXPECT_FLOAT_EQ(result1, 5.5f);

    // float + int
    auto result2 = add(f, i);  // Should trigger line 86
    EXPECT_FLOAT_EQ(result2, 5.5f);

    // Test the else branch (line 90)
    auto result3 = add(3.5, 2.5);  // Both floating
    EXPECT_DOUBLE_EQ(result3, 6.0);

    auto result4 = add(3, 2);  // Both integral
    EXPECT_EQ(result4, 5);
}
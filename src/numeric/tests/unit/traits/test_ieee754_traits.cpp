#include <base/traits_base.h>
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <complex>

using namespace fem::numeric;

// ============================================================================
// IEEE 754 Compliance Tests for Numeric Traits - CRITICAL MISSING COVERAGE
// ============================================================================

class IEEE754TraitsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save FPU state if needed for rounding mode tests
    }
};

// ============================================================================
// Numeric Traits Special Values Tests
// ============================================================================

TEST_F(IEEE754TraitsTest, FloatSpecialValues) {
    using traits = numeric_traits<float>;
    
    // Basic constants
    EXPECT_EQ(traits::zero(), 0.0f);
    EXPECT_EQ(traits::one(), 1.0f);
    
    // IEEE 754 limits
    EXPECT_EQ(traits::min(), std::numeric_limits<float>::min());
    EXPECT_EQ(traits::max(), std::numeric_limits<float>::max());
    EXPECT_EQ(traits::lowest(), std::numeric_limits<float>::lowest());
    EXPECT_EQ(traits::epsilon(), std::numeric_limits<float>::epsilon());
    
    // Special IEEE values
    if (traits::has_infinity) {
        EXPECT_TRUE(std::isinf(traits::infinity()));
        EXPECT_FALSE(std::signbit(traits::infinity()));  // Positive infinity
        
        EXPECT_TRUE(std::isinf(traits::neg_infinity()));
        EXPECT_TRUE(std::signbit(traits::neg_infinity())); // Negative infinity
    }
    
    if (traits::has_quiet_nan) {
        EXPECT_TRUE(std::isnan(traits::quiet_nan()));
        // NaN should not equal itself
        EXPECT_NE(traits::quiet_nan(), traits::quiet_nan());
    }
}

TEST_F(IEEE754TraitsTest, DoubleSpecialValues) {
    using traits = numeric_traits<double>;
    
    // IEEE characteristics should be true for double
    EXPECT_TRUE(traits::has_infinity);
    EXPECT_TRUE(traits::has_quiet_nan);
    
    // Test special values
    auto inf = traits::infinity();
    auto neg_inf = traits::neg_infinity();
    auto nan = traits::quiet_nan();
    
    EXPECT_TRUE(std::isinf(inf));
    EXPECT_FALSE(std::signbit(inf));
    
    EXPECT_TRUE(std::isinf(neg_inf));
    EXPECT_TRUE(std::signbit(neg_inf));
    
    EXPECT_TRUE(std::isnan(nan));
    EXPECT_NE(nan, nan);  // NaN != NaN
    
    // Arithmetic with special values
    EXPECT_TRUE(std::isinf(inf + 1.0));
    EXPECT_TRUE(std::isinf(inf * 2.0));
    EXPECT_TRUE(std::isnan(inf - inf));
    EXPECT_TRUE(std::isnan(inf / inf));
    EXPECT_TRUE(std::isnan(0.0 * inf));
    
    EXPECT_TRUE(std::isnan(nan + 1.0));
    EXPECT_TRUE(std::isnan(nan * 0.0));
}

TEST_F(IEEE754TraitsTest, IntegerTypeLimits) {
    // Integer types don't have infinity or NaN
    using int_traits = numeric_traits<int>;
    EXPECT_FALSE(int_traits::has_infinity);
    EXPECT_FALSE(int_traits::has_quiet_nan);
    
    // Should fall back to max/lowest for "infinity"
    EXPECT_EQ(int_traits::infinity(), std::numeric_limits<int>::max());
    EXPECT_EQ(int_traits::neg_infinity(), std::numeric_limits<int>::lowest());
    
    // quiet_nan should return zero-initialized value
    EXPECT_EQ(int_traits::quiet_nan(), int{});
}

// ============================================================================
// Complex Number IEEE Compliance Tests
// ============================================================================

TEST_F(IEEE754TraitsTest, ComplexSpecialValues) {
    using traits = numeric_traits<std::complex<double>>;
    
    // Complex-specific constants
    auto zero = traits::zero();
    EXPECT_EQ(zero.real(), 0.0);
    EXPECT_EQ(zero.imag(), 0.0);
    
    auto one = traits::one();
    EXPECT_EQ(one.real(), 1.0);
    EXPECT_EQ(one.imag(), 0.0);
    
    auto i = traits::i();
    EXPECT_EQ(i.real(), 0.0);
    EXPECT_EQ(i.imag(), 1.0);
    
    // IEEE characteristics inherited from underlying type
    EXPECT_TRUE(traits::has_infinity);
    EXPECT_TRUE(traits::has_quiet_nan);
}

TEST_F(IEEE754TraitsTest, ComplexIEEEValues) {
    using traits = numeric_traits<std::complex<double>>;
    
    if (traits::has_quiet_nan) {
        auto nan = traits::quiet_nan();
        EXPECT_TRUE(std::isnan(nan.real()));
        EXPECT_TRUE(std::isnan(nan.imag()));
        
        // NaN propagation in complex arithmetic
        std::complex<double> normal(1.0, 2.0);
        auto result = nan + normal;
        EXPECT_TRUE(std::isnan(result.real()));
        // Note: imag part may or may not be NaN depending on implementation
    }
    
    if (traits::has_infinity) {
        auto inf = traits::infinity();
        EXPECT_TRUE(std::isinf(inf.real()));
        EXPECT_EQ(inf.imag(), 0.0);
        
        // Infinity arithmetic
        std::complex<double> finite(1.0, 1.0);
        auto result = inf + finite;
        EXPECT_TRUE(std::isinf(result.real()));
        EXPECT_EQ(result.imag(), 1.0);
    }
}

// ============================================================================
// Signed Zero and IEEE Edge Cases
// ============================================================================

TEST_F(IEEE754TraitsTest, SignedZeroHandling) {
    // Test that traits work correctly with signed zeros
    constexpr double pos_zero = 0.0;
    constexpr double neg_zero = -0.0;
    
    // Both should be considered equal
    EXPECT_EQ(pos_zero, neg_zero);
    
    // But have different sign bits
    EXPECT_FALSE(std::signbit(pos_zero));
    EXPECT_TRUE(std::signbit(neg_zero));
    
    // Traits should return positive zero
    using traits = numeric_traits<double>;
    auto zero = traits::zero();
    EXPECT_EQ(zero, 0.0);
    EXPECT_FALSE(std::signbit(zero));
}

TEST_F(IEEE754TraitsTest, DenormalNumberHandling) {
    using traits = numeric_traits<double>;
    
    // Test with smallest positive denormal
    auto denorm_min = std::numeric_limits<double>::denorm_min();
    EXPECT_GT(denorm_min, 0.0);
    EXPECT_LT(denorm_min, traits::min());
    
    // Verify it's actually denormal/subnormal
    if (denorm_min > 0.0) {
        EXPECT_EQ(std::fpclassify(denorm_min), FP_SUBNORMAL);
    }
    
    // Arithmetic with denormals should work
    auto denorm2 = denorm_min * 2.0;
    if (denorm2 > 0.0) {
        EXPECT_GT(denorm2, denorm_min);
        EXPECT_LT(denorm2, traits::min());
    }
}

// ============================================================================
// Type Consistency Tests
// ============================================================================

TEST_F(IEEE754TraitsTest, TraitsTypeConsistency) {
    using float_traits = numeric_traits<float>;
    using double_traits = numeric_traits<double>;
    
    // Verify type consistency
    static_assert(std::is_same_v<float_traits::value_type, float>);
    static_assert(std::is_same_v<float_traits::real_type, float>);
    static_assert(std::is_same_v<float_traits::complex_type, std::complex<float>>);
    
    static_assert(std::is_same_v<double_traits::value_type, double>);
    static_assert(std::is_same_v<double_traits::real_type, double>);
    static_assert(std::is_same_v<double_traits::complex_type, std::complex<double>>);
    
    // Complex traits consistency
    using complex_traits = numeric_traits<std::complex<double>>;
    static_assert(std::is_same_v<complex_traits::value_type, std::complex<double>>);
    static_assert(std::is_same_v<complex_traits::real_type, double>);
    static_assert(std::is_same_v<complex_traits::complex_type, std::complex<double>>);
}

TEST_F(IEEE754TraitsTest, IEEEComplianceFlags) {
    // Test IEEE compliance detection
    using float_traits = numeric_traits<float>;
    using double_traits = numeric_traits<double>;
    using int_traits = numeric_traits<int>;
    
    // Floating point types should be IEEE compliant
    EXPECT_TRUE(float_traits::is_ieee_compliant);
    EXPECT_TRUE(double_traits::is_ieee_compliant);
    EXPECT_TRUE(float_traits::is_floating_point);
    EXPECT_TRUE(double_traits::is_floating_point);
    
    // Integer types are not IEEE floating point
    EXPECT_FALSE(int_traits::is_ieee_compliant);
    EXPECT_FALSE(int_traits::is_floating_point);
    EXPECT_TRUE(int_traits::is_integral);
}

// ============================================================================
// Precision and Accuracy Tests
// ============================================================================

TEST_F(IEEE754TraitsTest, EpsilonAccuracy) {
    using float_traits = numeric_traits<float>;
    using double_traits = numeric_traits<double>;
    
    // Verify epsilon is machine epsilon
    EXPECT_EQ(float_traits::epsilon(), std::numeric_limits<float>::epsilon());
    EXPECT_EQ(double_traits::epsilon(), std::numeric_limits<double>::epsilon());
    
    // Test epsilon meaning: 1 + epsilon != 1
    float f_eps = float_traits::epsilon();
    double d_eps = double_traits::epsilon();
    
    EXPECT_NE(1.0f + f_eps, 1.0f);
    EXPECT_NE(1.0 + d_eps, 1.0);
    
    // But 1 + epsilon/2 == 1 (for well-behaved implementations)
    EXPECT_EQ(1.0f + f_eps/2.0f, 1.0f);
    EXPECT_EQ(1.0 + d_eps/2.0, 1.0);
}

TEST_F(IEEE754TraitsTest, RoundingBehavior) {
    // Test that traits values exhibit correct rounding behavior
    using traits = numeric_traits<double>;
    
    // Test operations near epsilon
    auto eps = traits::epsilon();
    auto one = traits::one();
    
    // These should demonstrate IEEE rounding
    double sum = one + eps;
    EXPECT_GT(sum, one);
    EXPECT_LT(sum, one + 2*eps);
    
    // Test with values near limits
    auto max_val = traits::max();
    auto largest_normal = max_val;
    
    // max + 1 should overflow to infinity (if supported)
    if (traits::has_infinity) {
        // This may overflow depending on the value
        // Just test that max is indeed the maximum finite value
        EXPECT_TRUE(std::isfinite(max_val));
        EXPECT_FALSE(std::isinf(max_val));
    }
}

// ============================================================================
// Cross-Platform Consistency Tests  
// ============================================================================

TEST_F(IEEE754TraitsTest, CrossPlatformConsistency) {
    // Test that traits provide consistent values across platforms
    using traits = numeric_traits<double>;
    
    // These should be the same on all IEEE 754 compliant platforms
    EXPECT_EQ(traits::zero(), 0.0);
    EXPECT_EQ(traits::one(), 1.0);
    
    // Size and alignment should be consistent
    EXPECT_EQ(traits::size, sizeof(double));
    EXPECT_EQ(traits::alignment, alignof(double));
    
    // IEEE properties should be consistent
    EXPECT_TRUE(traits::has_infinity);
    EXPECT_TRUE(traits::has_quiet_nan);
    EXPECT_TRUE(traits::is_ieee_compliant);
    EXPECT_TRUE(traits::is_floating_point);
    EXPECT_TRUE(traits::is_signed);
    EXPECT_FALSE(traits::is_integral);
    EXPECT_FALSE(traits::is_complex);
    EXPECT_FALSE(traits::is_dual);
}

// ============================================================================
// Performance and Compile-Time Tests
// ============================================================================

TEST_F(IEEE754TraitsTest, CompileTimeEvaluation) {
    // Verify that trait values can be computed at compile time
    using traits = numeric_traits<double>;
    
    constexpr auto zero = traits::zero();
    constexpr auto one = traits::one();
    constexpr auto eps = traits::epsilon();
    
    static_assert(zero == 0.0);
    static_assert(one == 1.0);
    static_assert(eps > 0.0);
    
    // Boolean traits should be compile-time constants
    static_assert(traits::has_infinity);
    static_assert(traits::has_quiet_nan);
    static_assert(traits::is_ieee_compliant);
    static_assert(traits::is_floating_point);
    static_assert(!traits::is_integral);
    static_assert(!traits::is_complex);
    static_assert(!traits::is_dual);
}
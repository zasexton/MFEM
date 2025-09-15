#include <base/numeric_base.h>
#include <base/traits_base.h>
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <cfenv>
#include <complex>

using namespace fem::numeric;

// Note: Some compilers ignore STDC FENV_ACCESS and warn; we rely on
// volatile temporaries and precise FP flags set in CMake to respect
// rounding-mode changes from std::fesetround.

// ============================================================================
// IEEE 754 Compliance Critical Tests
// ============================================================================

class IEEE754ComplianceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save current FPU state
        saved_rounding_mode_ = std::fegetround();
    }

    void TearDown() override {
        // Restore FPU state
        std::fesetround(saved_rounding_mode_);
    }

    int saved_rounding_mode_;
};

// ============================================================================
// Signed Zero Tests - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, SignedZeroDistinction) {
    constexpr double pos_zero = 0.0;
    constexpr double neg_zero = -0.0;
    
    // Signed zeros should be equal but distinguishable
    EXPECT_EQ(pos_zero, neg_zero);
    EXPECT_FALSE(std::signbit(pos_zero));
    EXPECT_TRUE(std::signbit(neg_zero));
    
    // Test with float precision
    constexpr float pos_zerof = 0.0f;
    constexpr float neg_zerof = -0.0f;
    EXPECT_EQ(pos_zerof, neg_zerof);
    EXPECT_FALSE(std::signbit(pos_zerof));
    EXPECT_TRUE(std::signbit(neg_zerof));
}

TEST_F(IEEE754ComplianceTest, SignedZeroArithmetic) {
    constexpr double pos_zero = 0.0;
    constexpr double neg_zero = -0.0;
    
    // Addition preserves signs correctly
    EXPECT_FALSE(std::signbit(pos_zero + pos_zero));  // +0 + +0 = +0
    EXPECT_FALSE(std::signbit(pos_zero + neg_zero));  // +0 + -0 = +0
    EXPECT_FALSE(std::signbit(neg_zero + pos_zero));  // -0 + +0 = +0
    EXPECT_TRUE(std::signbit(neg_zero + neg_zero));   // -0 + -0 = -0
    
    // Multiplication with signed zeros
    EXPECT_FALSE(std::signbit(pos_zero * 1.0));      // +0 * +1 = +0
    EXPECT_TRUE(std::signbit(pos_zero * -1.0));      // +0 * -1 = -0
    EXPECT_TRUE(std::signbit(neg_zero * 1.0));       // -0 * +1 = -0
    EXPECT_FALSE(std::signbit(neg_zero * -1.0));     // -0 * -1 = +0
}

TEST_F(IEEE754ComplianceTest, SignedZeroDivision) {
    constexpr double pos_zero = 0.0;
    constexpr double neg_zero = -0.0;
    
    // Division by signed zeros produces correct infinities
    EXPECT_TRUE(std::isinf(1.0 / pos_zero));
    EXPECT_FALSE(std::signbit(1.0 / pos_zero));      // +inf
    
    EXPECT_TRUE(std::isinf(1.0 / neg_zero));
    EXPECT_TRUE(std::signbit(1.0 / neg_zero));       // -inf
    
    EXPECT_TRUE(std::isinf(-1.0 / pos_zero));
    EXPECT_TRUE(std::signbit(-1.0 / pos_zero));      // -inf
    
    EXPECT_TRUE(std::isinf(-1.0 / neg_zero));
    EXPECT_FALSE(std::signbit(-1.0 / neg_zero));     // +inf
}

// ============================================================================
// FPU Rounding Mode Tests - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, RoundingModeNearestEven) {
    std::fesetround(FE_TONEAREST);
    
    // Test tie-breaking to even (banker's rounding)
    EXPECT_EQ(std::rint(2.5), 2.0);   // Ties to even: 2.5 -> 2
    EXPECT_EQ(std::rint(3.5), 4.0);   // Ties to even: 3.5 -> 4
    EXPECT_EQ(std::rint(4.5), 4.0);   // Ties to even: 4.5 -> 4
    EXPECT_EQ(std::rint(5.5), 6.0);   // Ties to even: 5.5 -> 6
    
    // Test with negative numbers
    EXPECT_EQ(std::rint(-2.5), -2.0); // Ties to even: -2.5 -> -2
    EXPECT_EQ(std::rint(-3.5), -4.0); // Ties to even: -3.5 -> -4
}

TEST_F(IEEE754ComplianceTest, RoundingModeUpward) {
    if (std::fesetround(FE_UPWARD) != 0) {
        GTEST_SKIP() << "FE_UPWARD rounding mode not supported";
    }
    
    // All operations should round toward positive infinity. Use volatile
    // variables to force runtime evaluation and avoid constant folding.
    volatile double one = 1.0;
    volatile double three = 3.0;
    volatile double t = one / three; // inexact under FE_UPWARD
    double sum = static_cast<double>(t + t + t);
    EXPECT_GT(sum, 1.0);  // Should be > 1.0

    // Test basic arithmetic rounding up, also with volatile to avoid folding
    volatile double result = 1.0;
    result /= 3.0;  // Should round up
    result *= 3.0;  // Should round up
    EXPECT_GT(static_cast<double>(result), 1.0);
}

TEST_F(IEEE754ComplianceTest, RoundingModeDownward) {
    if (std::fesetround(FE_DOWNWARD) != 0) {
        GTEST_SKIP() << "FE_DOWNWARD rounding mode not supported";
    }
    
    // All operations should round toward negative infinity. Use volatile
    // variables to force runtime evaluation and avoid constant folding.
    volatile double one = 1.0;
    volatile double three = 3.0;
    volatile double t = one / three; // inexact under FE_DOWNWARD
    double sum = static_cast<double>(t + t + t);
    EXPECT_LT(sum, 1.0);  // Should be < 1.0

    // Test basic arithmetic rounding down, also with volatile
    volatile double result = 1.0;
    result /= 3.0;  // Should round down
    result *= 3.0;  // Should round down
    EXPECT_LT(static_cast<double>(result), 1.0);
}

TEST_F(IEEE754ComplianceTest, RoundingModeTowardZero) {
    if (std::fesetround(FE_TOWARDZERO) != 0) {
        GTEST_SKIP() << "FE_TOWARDZERO rounding mode not supported";
    }
    
    // Test truncation behavior
    EXPECT_EQ(std::rint(2.7), 2.0);   // Toward zero: 2.7 -> 2
    EXPECT_EQ(std::rint(-2.7), -2.0); // Toward zero: -2.7 -> -2
    EXPECT_EQ(std::rint(2.3), 2.0);   // Toward zero: 2.3 -> 2
    EXPECT_EQ(std::rint(-2.3), -2.0); // Toward zero: -2.3 -> -2
}

// ============================================================================
// Denormal/Subnormal Number Tests - CRITICAL MISSING  
// ============================================================================

TEST_F(IEEE754ComplianceTest, DenormalNumbers) {
    // Test smallest positive denormal double
    constexpr double min_denormal = std::numeric_limits<double>::denorm_min();
    EXPECT_GT(min_denormal, 0.0);
    EXPECT_LT(min_denormal, std::numeric_limits<double>::min());
    
    // Verify denormal arithmetic
    double half_denormal = min_denormal / 2.0;
    if (half_denormal > 0.0) {
        // System supports gradual underflow
        EXPECT_GT(half_denormal, 0.0);
        EXPECT_LT(half_denormal, min_denormal);
    } else {
        // System flushes to zero
        EXPECT_EQ(half_denormal, 0.0);
    }
    
    // Test denormal addition
    double sum = min_denormal + min_denormal;
    EXPECT_GT(sum, min_denormal);
    EXPECT_LT(sum, std::numeric_limits<double>::min());
}

TEST_F(IEEE754ComplianceTest, DenormalFloat) {
    // Test with single precision
    constexpr float min_denormal_f = std::numeric_limits<float>::denorm_min();
    EXPECT_GT(min_denormal_f, 0.0f);
    EXPECT_LT(min_denormal_f, std::numeric_limits<float>::min());
    
    // Test denormal operations preserve precision
    float tiny = min_denormal_f * 1000.0f;
    float result = tiny - tiny;
    EXPECT_EQ(result, 0.0f);
    EXPECT_FALSE(std::signbit(result));  // Should be positive zero
}

// ============================================================================
// Complex Number IEEE Compliance - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, ComplexNaNPropagation) {
    std::complex<double> nan_complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
    std::complex<double> regular(1.0, 2.0);
    
    // NaN should propagate through complex operations
    auto result = nan_complex + regular;
    EXPECT_TRUE(std::isnan(result.real()));
    EXPECT_FALSE(std::isnan(result.imag()));
    
    // Test complex multiplication with NaN
    auto mult_result = nan_complex * regular;
    EXPECT_TRUE(std::isnan(mult_result.real()));
    EXPECT_TRUE(std::isnan(mult_result.imag()));
}

TEST_F(IEEE754ComplianceTest, ComplexInfinityBehavior) {
    std::complex<double> inf_complex(std::numeric_limits<double>::infinity(), 0.0);
    std::complex<double> regular(1.0, 1.0);
    
    // Test infinity propagation
    auto result = inf_complex + regular;
    EXPECT_TRUE(std::isinf(result.real()));
    EXPECT_FALSE(std::isinf(result.imag()));
    
    // Division by an exact zero complex should produce infinities or NaNs
    // in typical IEEE-conforming implementations. We accept either in
    // order to remain portable across libstdc++/libc++ algorithms.
    std::complex<double> zero(0.0, 0.0);
    auto div_zero = regular / zero;
    EXPECT_TRUE(std::isinf(div_zero.real()) || std::isinf(div_zero.imag())
                || std::isnan(div_zero.real()) || std::isnan(div_zero.imag()));
}

// ============================================================================
// Signaling vs Quiet NaN Tests - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, QuietNaNBehavior) {
    double qnan = std::numeric_limits<double>::quiet_NaN();
    
    // Quiet NaN should not raise exceptions in comparison
    EXPECT_FALSE(qnan == qnan);
    EXPECT_FALSE(qnan < 1.0);
    EXPECT_FALSE(qnan > 1.0);
    EXPECT_FALSE(qnan <= 1.0);
    EXPECT_FALSE(qnan >= 1.0);
    
    // Arithmetic with quiet NaN should produce NaN
    EXPECT_TRUE(std::isnan(qnan + 1.0));
    EXPECT_TRUE(std::isnan(qnan * 0.0));
    EXPECT_TRUE(std::isnan(std::sqrt(qnan)));
}

TEST_F(IEEE754ComplianceTest, SignalingNaNBehavior) {
    if (!std::numeric_limits<double>::has_signaling_NaN) {
        GTEST_SKIP() << "Signaling NaN not supported";
    }
    
    double snan = std::numeric_limits<double>::signaling_NaN();
    
    // Signaling NaN should convert to quiet NaN in arithmetic
    double result = snan + 1.0;
    EXPECT_TRUE(std::isnan(result));
    // Note: We can't easily test if exception was raised without signal handling
}

// ============================================================================
// Overflow/Underflow Behavior Tests - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, OverflowToInfinity) {
    double large = std::numeric_limits<double>::max();
    
    // Multiplication overflow should produce infinity
    double overflow_result = large * 2.0;
    EXPECT_TRUE(std::isinf(overflow_result));
    EXPECT_FALSE(std::signbit(overflow_result));  // Positive infinity
    
    // Negative overflow should produce negative infinity
    double neg_overflow = -large * 2.0;
    EXPECT_TRUE(std::isinf(neg_overflow));
    EXPECT_TRUE(std::signbit(neg_overflow));      // Negative infinity
}

TEST_F(IEEE754ComplianceTest, GradualUnderflow) {
    double small = std::numeric_limits<double>::min();
    
    // Division underflow with gradual underflow
    double underflow_result = small / 2.0;
    if (underflow_result > 0.0) {
        // Gradual underflow supported
        EXPECT_GT(underflow_result, 0.0);
        EXPECT_LT(underflow_result, small);
    } else {
        // Flush to zero
        EXPECT_EQ(underflow_result, 0.0);
    }
}

// ============================================================================
// Copy Sign and Sign Bit Tests - CRITICAL MISSING
// ============================================================================

TEST_F(IEEE754ComplianceTest, CopySignOperations) {
    // Test copysign with various combinations
    EXPECT_EQ(std::copysign(3.0, 5.0), 3.0);    // Positive sign
    EXPECT_EQ(std::copysign(3.0, -5.0), -3.0);  // Negative sign
    EXPECT_EQ(std::copysign(-3.0, 5.0), 3.0);   // Change to positive
    EXPECT_EQ(std::copysign(-3.0, -5.0), -3.0); // Keep negative
    
    // Test with zeros
    EXPECT_FALSE(std::signbit(std::copysign(0.0, 1.0)));   // +0
    EXPECT_TRUE(std::signbit(std::copysign(0.0, -1.0)));   // -0
    
    // Test with infinities and NaN
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_FALSE(std::signbit(std::copysign(inf, 1.0)));
    EXPECT_TRUE(std::signbit(std::copysign(inf, -1.0)));
    EXPECT_FALSE(std::signbit(std::copysign(nan, 1.0)));
    EXPECT_TRUE(std::signbit(std::copysign(nan, -1.0)));
}

TEST_F(IEEE754ComplianceTest, SignBitDetection) {
    // Test signbit with normal numbers
    EXPECT_FALSE(std::signbit(1.0));
    EXPECT_TRUE(std::signbit(-1.0));
    
    // Test signbit with zeros
    EXPECT_FALSE(std::signbit(0.0));
    EXPECT_TRUE(std::signbit(-0.0));
    
    // Test signbit with infinities
    EXPECT_FALSE(std::signbit(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(std::signbit(-std::numeric_limits<double>::infinity()));
    
    // Test signbit with NaN
    EXPECT_FALSE(std::signbit(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(std::signbit(std::copysign(std::numeric_limits<double>::quiet_NaN(), -1.0)));
}

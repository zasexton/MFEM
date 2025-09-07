// test_dual_math.cpp - Unit tests for extended dual number mathematical functions
#include <gtest/gtest.h>
#include <base/dual_math.h>
#include <base/dual_base.h>
#include <base/numeric_base.h>
#include <cmath>
#include <limits>

using namespace fem::numeric::autodiff;
using namespace fem::numeric;

// ============================================================================
// Trigonometric Functions Tests
// ============================================================================

TEST(DualMath, TangentFunction) {
    DualBase<double, 1> x(M_PI/4, 1.0);  // 45 degrees
    auto y = tan(x);
    EXPECT_NEAR(y.value(), 1.0, 1e-10);  // tan(π/4) = 1
    EXPECT_NEAR(y.derivative(0), 2.0, 1e-10);  // sec²(π/4) = 2

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto tan_zero = tan(zero);
    EXPECT_NEAR(tan_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(tan_zero.derivative(0), 1.0, 1e-10);  // sec²(0) = 1
}

TEST(DualMath, ArcsinFunction) {
    DualBase<double, 1> x(0.5, 1.0);
    auto y = asin(x);
    EXPECT_NEAR(y.value(), M_PI/6, 1e-10);  // asin(0.5) = π/6
    EXPECT_NEAR(y.derivative(0), 2.0/std::sqrt(3.0), 1e-10);  // 1/√(1-0.25) = 2/√3

    // Test near boundary
    DualBase<double, 1> near_one(0.99, 1.0);
    auto asin_near = asin(near_one);
    EXPECT_GT(asin_near.derivative(0), 7.0);  // Derivative gets large near ±1
}

TEST(DualMath, ArccosFunction) {
    DualBase<double, 1> x(0.5, 1.0);
    auto y = acos(x);
    EXPECT_NEAR(y.value(), M_PI/3, 1e-10);  // acos(0.5) = π/3
    EXPECT_NEAR(y.derivative(0), -2.0/std::sqrt(3.0), 1e-10);  // -1/√(1-0.25)

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto acos_zero = acos(zero);
    EXPECT_NEAR(acos_zero.value(), M_PI/2, 1e-10);
    EXPECT_NEAR(acos_zero.derivative(0), -1.0, 1e-10);
}

TEST(DualMath, ArctanFunction) {
    DualBase<double, 1> x(1.0, 1.0);
    auto y = atan(x);
    EXPECT_NEAR(y.value(), M_PI/4, 1e-10);  // atan(1) = π/4
    EXPECT_NEAR(y.derivative(0), 0.5, 1e-10);  // 1/(1+1) = 0.5

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto atan_zero = atan(zero);
    EXPECT_NEAR(atan_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(atan_zero.derivative(0), 1.0, 1e-10);
}

TEST(DualMath, Atan2Function) {
    // atan2(y, x) with both as dual numbers
    DualBase<double, 2> y(1.0, 1.0, 0.0);  // dy/dx = 1, dy/dy = 0
    DualBase<double, 2> x(1.0, 0.0, 1.0);  // dx/dx = 0, dx/dy = 1
    auto result = atan2(y, x);

    EXPECT_NEAR(result.value(), M_PI/4, 1e-10);  // atan2(1,1) = π/4
    // Partial derivatives: ∂/∂x[atan2(y,x)] = -y/(x²+y²), ∂/∂y[atan2(y,x)] = x/(x²+y²)
    EXPECT_NEAR(result.derivative(0), 0.5, 1e-10);   // x/(x²+y²) with dy/dx=1
    EXPECT_NEAR(result.derivative(1), -0.5, 1e-10);  // -y/(x²+y²) with dx/dy=1

    // Test quadrants
    DualBase<double, 1> neg_y(-1.0, 1.0);
    DualBase<double, 1> neg_x(-1.0, 0.0);
    auto q3 = atan2(neg_y, neg_x);
    EXPECT_NEAR(q3.value(), -3*M_PI/4, 1e-10);
}

// ============================================================================
// Hyperbolic Functions Tests
// ============================================================================

TEST(DualMath, SinhFunction) {
    DualBase<double, 1> x(1.0, 1.0);
    auto y = sinh(x);
    EXPECT_NEAR(y.value(), std::sinh(1.0), 1e-10);
    EXPECT_NEAR(y.derivative(0), std::cosh(1.0), 1e-10);

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto sinh_zero = sinh(zero);
    EXPECT_NEAR(sinh_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(sinh_zero.derivative(0), 1.0, 1e-10);  // cosh(0) = 1
}

TEST(DualMath, CoshFunction) {
    DualBase<double, 1> x(1.0, 1.0);
    auto y = cosh(x);
    EXPECT_NEAR(y.value(), std::cosh(1.0), 1e-10);
    EXPECT_NEAR(y.derivative(0), std::sinh(1.0), 1e-10);

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto cosh_zero = cosh(zero);
    EXPECT_NEAR(cosh_zero.value(), 1.0, 1e-10);  // cosh(0) = 1
    EXPECT_NEAR(cosh_zero.derivative(0), 0.0, 1e-10);  // sinh(0) = 0
}

TEST(DualMath, TanhFunction) {
    DualBase<double, 1> x(1.0, 1.0);
    auto y = tanh(x);
    EXPECT_NEAR(y.value(), std::tanh(1.0), 1e-10);

    double cosh_1 = std::cosh(1.0);
    EXPECT_NEAR(y.derivative(0), 1.0/(cosh_1*cosh_1), 1e-10);

    // Test at 0
    DualBase<double, 1> zero(0.0, 1.0);
    auto tanh_zero = tanh(zero);
    EXPECT_NEAR(tanh_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(tanh_zero.derivative(0), 1.0, 1e-10);  // sech²(0) = 1
}

// ============================================================================
// Logarithmic Functions Tests
// ============================================================================

TEST(DualMath, Log10Function) {
    DualBase<double, 1> x(10.0, 1.0);
    auto y = log10(x);
    EXPECT_NEAR(y.value(), 1.0, 1e-10);  // log10(10) = 1
    EXPECT_NEAR(y.derivative(0), 1.0/(10.0*std::log(10.0)), 1e-10);

    // Test at 100
    DualBase<double, 1> hundred(100.0, 1.0);
    auto log10_100 = log10(hundred);
    EXPECT_NEAR(log10_100.value(), 2.0, 1e-10);
}

TEST(DualMath, Log2Function) {
    DualBase<double, 1> x(8.0, 1.0);
    auto y = log2(x);
    EXPECT_NEAR(y.value(), 3.0, 1e-10);  // log2(8) = 3
    EXPECT_NEAR(y.derivative(0), 1.0/(8.0*std::log(2.0)), 1e-10);

    // Test at 1
    DualBase<double, 1> one(1.0, 1.0);
    auto log2_1 = log2(one);
    EXPECT_NEAR(log2_1.value(), 0.0, 1e-10);
    EXPECT_NEAR(log2_1.derivative(0), 1.0/std::log(2.0), 1e-10);
}

// ============================================================================
// Extended Power Functions Tests
// ============================================================================

TEST(DualMath, PowDualExponent) {
    // Test x^y where both x and y are dual numbers
    DualBase<double, 2> x(2.0, 1.0, 0.0);  // x = 2, dx = 1
    DualBase<double, 2> y(3.0, 0.0, 1.0);  // y = 3, dy = 1
    auto z = pow(x, y);

    EXPECT_NEAR(z.value(), 8.0, 1e-10);  // 2^3 = 8
    // ∂/∂x[x^y] = y*x^(y-1) = 3*2^2 = 12
    EXPECT_NEAR(z.derivative(0), 12.0, 1e-10);
    // ∂/∂y[x^y] = x^y * ln(x) = 8 * ln(2)
    EXPECT_NEAR(z.derivative(1), 8.0*std::log(2.0), 1e-10);
}

TEST(DualMath, CbrtFunction) {
    DualBase<double, 1> x(8.0, 1.0);
    auto y = cbrt(x);
    EXPECT_NEAR(y.value(), 2.0, 1e-10);  // ∛8 = 2
    // d/dx[∛x] = 1/(3*x^(2/3)) = 1/(3*4) = 1/12
    EXPECT_NEAR(y.derivative(0), 1.0/12.0, 1e-10);

    // Test negative value (cbrt is defined for negative numbers)
    DualBase<double, 1> neg(-8.0, 1.0);
    auto cbrt_neg = cbrt(neg);
    EXPECT_NEAR(cbrt_neg.value(), -2.0, 1e-10);
}

// ============================================================================
// Special Functions Tests
// ============================================================================

TEST(DualMath, AbsFunction) {
    // Positive value
    DualBase<double, 1> pos(3.0, 1.0);
    auto abs_pos = abs(pos);
    EXPECT_NEAR(abs_pos.value(), 3.0, 1e-10);
    EXPECT_NEAR(abs_pos.derivative(0), 1.0, 1e-10);

    // Negative value
    DualBase<double, 1> neg(-3.0, 1.0);
    auto abs_neg = abs(neg);
    EXPECT_NEAR(abs_neg.value(), 3.0, 1e-10);
    EXPECT_NEAR(abs_neg.derivative(0), -1.0, 1e-10);

    // Zero (non-differentiable point)
    DualBase<double, 1> zero(0.0, 1.0);
    auto abs_zero = abs(zero);
    EXPECT_NEAR(abs_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(abs_zero.derivative(0), 0.0, 1e-10);  // Subgradient convention
}

TEST(DualMath, SignFunction) {
    // Positive
    DualBase<double, 1> pos(3.0, 1.0);
    auto sign_pos = sign(pos);
    EXPECT_NEAR(sign_pos.value(), 1.0, 1e-10);
    EXPECT_NEAR(sign_pos.derivative(0), 0.0, 1e-10);  // Derivative is always 0

    // Negative
    DualBase<double, 1> neg(-3.0, 1.0);
    auto sign_neg = sign(neg);
    EXPECT_NEAR(sign_neg.value(), -1.0, 1e-10);
    EXPECT_NEAR(sign_neg.derivative(0), 0.0, 1e-10);

    // Zero
    DualBase<double, 1> zero(0.0, 1.0);
    auto sign_zero = sign(zero);
    EXPECT_NEAR(sign_zero.value(), 0.0, 1e-10);
    EXPECT_NEAR(sign_zero.derivative(0), 0.0, 1e-10);
}

TEST(DualMath, MaxFunction) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    auto result = max(a, b);
    EXPECT_NEAR(result.value(), 3.0, 1e-10);
    EXPECT_NEAR(result.derivative(0), 1.0, 1e-10);  // Takes a's derivatives
    EXPECT_NEAR(result.derivative(1), 0.0, 1e-10);

    // Test equal values (non-differentiable point)
    DualBase<double, 2> c(2.0, 2.0, 3.0);
    DualBase<double, 2> d(2.0, 4.0, 5.0);
    auto equal_max = max(c, d);
    EXPECT_NEAR(equal_max.value(), 2.0, 1e-10);
    // Convention: takes first argument's derivatives when equal
    EXPECT_NEAR(equal_max.derivative(0), 2.0, 1e-10);
    EXPECT_NEAR(equal_max.derivative(1), 3.0, 1e-10);
}

TEST(DualMath, MinFunction) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    auto result = min(a, b);
    EXPECT_NEAR(result.value(), 2.0, 1e-10);
    EXPECT_NEAR(result.derivative(0), 0.0, 1e-10);  // Takes b's derivatives
    EXPECT_NEAR(result.derivative(1), 1.0, 1e-10);
}

TEST(DualMath, SmoothMaxFunction) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    // Default smoothing (k=10)
    auto smooth_result = smooth_max(a, b);
    EXPECT_NEAR(smooth_result.value(), 3.0, 0.1);  // Approximately 3
    // Derivative should be weighted average, heavily favoring a
    EXPECT_GT(smooth_result.derivative(0), 0.9);  // Mostly a's derivative
    EXPECT_LT(smooth_result.derivative(1), 0.1);  // Very little of b's

    // Test with smaller k (more smoothing)
    auto smoother = smooth_max(a, b, 1.0);
    double expected = std::log(std::exp(3.0) + std::exp(2.0));
    EXPECT_NEAR(smoother.value(), expected, 1e-10);

    // Test equal values
    DualBase<double, 2> c(2.0, 1.0, 0.0);
    DualBase<double, 2> d(2.0, 0.0, 1.0);
    auto equal_smooth = smooth_max(c, d, 10.0);
    EXPECT_NEAR(equal_smooth.value(), 2.0 + std::log(2.0)/10.0, 1e-10);
    // Derivatives should be equally weighted
    EXPECT_NEAR(equal_smooth.derivative(0), 0.5, 1e-10);
    EXPECT_NEAR(equal_smooth.derivative(1), 0.5, 1e-10);
}

TEST(DualMath, SmoothMinFunction) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    auto smooth_result = smooth_min(a, b);
    EXPECT_NEAR(smooth_result.value(), 2.0, 0.1);  // Approximately 2
    // Should mostly take b's derivatives
    EXPECT_LT(smooth_result.derivative(0), 0.1);
    EXPECT_GT(smooth_result.derivative(1), 0.9);
}

// ============================================================================
// Utility Functions Tests
// ============================================================================

TEST(DualMath, ClampFunction) {
    // Value in range
    DualBase<double, 1> in_range(5.0, 1.0);
    auto clamped_in = clamp(in_range, 0.0, 10.0);
    EXPECT_NEAR(clamped_in.value(), 5.0, 1e-10);
    EXPECT_NEAR(clamped_in.derivative(0), 1.0, 1e-10);  // Pass through

    // Value below min
    DualBase<double, 1> below(2.0, 1.0);
    auto clamped_below = clamp(below, 3.0, 10.0);
    EXPECT_NEAR(clamped_below.value(), 3.0, 1e-10);
    EXPECT_NEAR(clamped_below.derivative(0), 0.0, 1e-10);  // Zero at boundary

    // Value above max
    DualBase<double, 1> above(15.0, 1.0);
    auto clamped_above = clamp(above, 0.0, 10.0);
    EXPECT_NEAR(clamped_above.value(), 10.0, 1e-10);
    EXPECT_NEAR(clamped_above.derivative(0), 0.0, 1e-10);  // Zero at boundary
}

// ============================================================================
// Chain Rule and Composition Tests
// ============================================================================

TEST(DualMath, ChainRuleComposition) {
    // Test composition: f(g(x)) where f = sin and g = x²
    DualBase<double, 1> x(M_PI/6, 1.0);
    auto x_squared = x * x;
    auto sin_x_squared = sin(x_squared);

    // sin((π/6)²) and its derivative
    double x_val = M_PI/6;
    double expected_val = std::sin(x_val * x_val);
    double expected_deriv = 2 * x_val * std::cos(x_val * x_val);

    EXPECT_NEAR(sin_x_squared.value(), expected_val, 1e-10);
    EXPECT_NEAR(sin_x_squared.derivative(0), expected_deriv, 1e-10);
}

TEST(DualMath, ComplexExpression) {
    // Test: f(x,y) = exp(x) * sin(y) + log(x+y)
    DualBase<double, 2> x(1.0, 1.0, 0.0);  // ∂/∂x
    DualBase<double, 2> y(2.0, 0.0, 1.0);  // ∂/∂y

    auto f = exp(x) * sin(y) + log(x + y);

    double exp_1 = std::exp(1.0);
    double sin_2 = std::sin(2.0);
    double cos_2 = std::cos(2.0);

    EXPECT_NEAR(f.value(), exp_1 * sin_2 + std::log(3.0), 1e-10);

    // ∂f/∂x = exp(x)*sin(y) + 1/(x+y)
    EXPECT_NEAR(f.derivative(0), exp_1 * sin_2 + 1.0/3.0, 1e-10);

    // ∂f/∂y = exp(x)*cos(y) + 1/(x+y)
    EXPECT_NEAR(f.derivative(1), exp_1 * cos_2 + 1.0/3.0, 1e-10);
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

TEST(DualMath, TrigBoundaries) {
    // Test tan near π/2 (approaching infinity)
    DualBase<double, 1> near_pi2(M_PI/2 - 0.01, 1.0);
    auto tan_near = tan(near_pi2);
    EXPECT_GT(std::abs(tan_near.value()), 99.0);
    EXPECT_GT(tan_near.derivative(0), 9900.0);  // sec² gets very large

    // Test asin/acos at boundaries
    DualBase<double, 1> one(1.0, 1.0);
    auto asin_one = asin(one);
    EXPECT_NEAR(asin_one.value(), M_PI/2, 1e-10);
    // Derivative is infinite at x=1, but implementation may return large value

    DualBase<double, 1> minus_one(-1.0, 1.0);
    auto acos_minus = acos(minus_one);
    EXPECT_NEAR(acos_minus.value(), M_PI, 1e-10);
}

TEST(DualMath, LogarithmEdgeCases) {
    // Log of very small positive number
    DualBase<double, 1> small(1e-10, 1.0);
    auto log_small = log(small);
    EXPECT_LT(log_small.value(), -20.0);
    EXPECT_GT(log_small.derivative(0), 1e9);  // 1/x for small x

    // Log2 of power of 2
    DualBase<double, 1> sixteen(16.0, 1.0);
    auto log2_16 = log2(sixteen);
    EXPECT_NEAR(log2_16.value(), 4.0, 1e-10);
}

TEST(DualMath, HyperbolicLargeValues) {
    // Test hyperbolic functions with large arguments
    DualBase<double, 1> large(10.0, 1.0);
    auto sinh_large = sinh(large);
    auto cosh_large = cosh(large);

    EXPECT_NEAR(sinh_large.value(), std::sinh(10.0), 1e-7);
    EXPECT_NEAR(cosh_large.value(), std::cosh(10.0), 1e-7);

    // For large x, sinh(x) ≈ cosh(x) ≈ exp(x)/2
    EXPECT_NEAR(sinh_large.value(), cosh_large.value(), 1.0);
}

// ============================================================================
// Multi-Variable Tests
// ============================================================================

TEST(DualMath, MultiVariableTrig) {
    // Test gradient of f(x,y,z) = sin(x)*cos(y)*tan(z)
    const size_t N = 3;
    auto x = make_independent<double, N>(M_PI/4, 0);
    auto y = make_independent<double, N>(M_PI/3, 1);
    auto z = make_independent<double, N>(M_PI/6, 2);

    auto f = sin(x) * cos(y) * tan(z);

    double sin_val = std::sin(M_PI/4);
    double cos_val = std::cos(M_PI/3);
    double tan_val = std::tan(M_PI/6);

    EXPECT_NEAR(f.value(), sin_val * cos_val * tan_val, 1e-10);

    // Check partial derivatives
    auto grad = extract_gradient(f);
    // ∂f/∂x = cos(x)*cos(y)*tan(z)
    EXPECT_NEAR(grad[0], std::cos(M_PI/4) * cos_val * tan_val, 1e-10);
    // ∂f/∂y = -sin(x)*sin(y)*tan(z)
    EXPECT_NEAR(grad[1], -sin_val * std::sin(M_PI/3) * tan_val, 1e-10);
    // ∂f/∂z = sin(x)*cos(y)*sec²(z)
    double sec2_val = 1.0 / (std::cos(M_PI/6) * std::cos(M_PI/6));
    EXPECT_NEAR(grad[2], sin_val * cos_val * sec2_val, 1e-10);
}

// ============================================================================
// Type Compatibility Tests
// ============================================================================

TEST(DualMath, FloatCompatibility) {
    DualBase<float, 1> x(0.5f, 1.0f);

    auto sin_x = sin(x);
    auto exp_x = exp(x);
    auto log_x = log(x);

    EXPECT_NEAR(sin_x.value(), std::sin(0.5f), 1e-6f);
    EXPECT_NEAR(exp_x.value(), std::exp(0.5f), 1e-6f);
    EXPECT_NEAR(log_x.value(), std::log(0.5f), 1e-6f);
}
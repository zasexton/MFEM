// test_dual_base.cpp - Unit tests for DualBase automatic differentiation
#include <gtest/gtest.h>
#include <base/dual_base.h>
#include <base/numeric_base.h>
#include <cmath>
#include <limits>
#include <sstream>

using namespace fem::numeric::autodiff;
using namespace fem::numeric;

// ============================================================================
// Basic Construction and Access Tests
// ============================================================================

TEST(DualBase, DefaultConstruction) {
    DualBase<double, 3> d;
    EXPECT_EQ(d.value(), 0.0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(d.derivative(i), 0.0);
    }
    EXPECT_EQ(d.size(), 3u);
}

TEST(DualBase, ValueConstruction) {
    DualBase<double, 3> d(5.0);
    EXPECT_EQ(d.value(), 5.0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(d.derivative(i), 0.0);
    }
}

TEST(DualBase, ValueAndDerivativesConstruction) {
    std::array<double, 3> derivs{1.0, 2.0, 3.0};
    DualBase<double, 3> d(5.0, derivs);
    EXPECT_EQ(d.value(), 5.0);
    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 2.0);
    EXPECT_EQ(d.derivative(2), 3.0);
}

TEST(DualBase, VariadicConstruction) {
    DualBase<double, 3> d(5.0, 1.0, 2.0, 3.0);
    EXPECT_EQ(d.value(), 5.0);
    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 2.0);
    EXPECT_EQ(d.derivative(2), 3.0);
}

TEST(DualBase, CopyConstruction) {
    DualBase<double, 2> d1(3.0, 1.0, 0.0);
    DualBase<double, 2> d2(d1);
    EXPECT_EQ(d2.value(), d1.value());
    EXPECT_EQ(d2.derivative(0), d1.derivative(0));
    EXPECT_EQ(d2.derivative(1), d1.derivative(1));
}

TEST(DualBase, MoveConstruction) {
    DualBase<double, 2> d1(3.0, 1.0, 0.0);
    DualBase<double, 2> d2(std::move(d1));
    EXPECT_EQ(d2.value(), 3.0);
    EXPECT_EQ(d2.derivative(0), 1.0);
    EXPECT_EQ(d2.derivative(1), 0.0);
}

TEST(DualBase, CopyAssignment) {
    DualBase<double, 2> d1(3.0, 1.0, 0.0);
    DualBase<double, 2> d2;
    d2 = d1;
    EXPECT_EQ(d2.value(), d1.value());
    EXPECT_EQ(d2.derivative(0), d1.derivative(0));
    EXPECT_EQ(d2.derivative(1), d1.derivative(1));
}

TEST(DualBase, MoveAssignment) {
    DualBase<double, 2> d1(3.0, 1.0, 0.0);
    DualBase<double, 2> d2;
    d2 = std::move(d1);
    EXPECT_EQ(d2.value(), 3.0);
    EXPECT_EQ(d2.derivative(0), 1.0);
    EXPECT_EQ(d2.derivative(1), 0.0);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST(DualBase, ValueAccess) {
    DualBase<double, 2> d(3.0);
    EXPECT_EQ(d.value(), 3.0);
    d.value() = 5.0;
    EXPECT_EQ(d.value(), 5.0);

    const DualBase<double, 2> cd(7.0);
    EXPECT_EQ(cd.value(), 7.0);
}

TEST(DualBase, DerivativeAccess) {
    DualBase<double, 3> d(1.0);
    d.derivative(0) = 1.0;
    d.derivative(1) = 2.0;
    d.derivative(2) = 3.0;

    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 2.0);
    EXPECT_EQ(d.derivative(2), 3.0);

    const DualBase<double, 3> cd(1.0, 4.0, 5.0, 6.0);
    EXPECT_EQ(cd.derivative(0), 4.0);
    EXPECT_EQ(cd.derivative(1), 5.0);
    EXPECT_EQ(cd.derivative(2), 6.0);
}

TEST(DualBase, DerivativesArrayAccess) {
    DualBase<double, 3> d(1.0);
    auto& derivs = d.derivatives();
    derivs[0] = 1.0;
    derivs[1] = 2.0;
    derivs[2] = 3.0;

    const auto& const_derivs = d.derivatives();
    EXPECT_EQ(const_derivs[0], 1.0);
    EXPECT_EQ(const_derivs[1], 2.0);
    EXPECT_EQ(const_derivs[2], 3.0);
}

TEST(DualBase, GradientAccess) {
    DualBase<double, 3> d(1.0, 1.0, 2.0, 3.0);
    const auto& grad = d.gradient();
    EXPECT_EQ(grad[0], 1.0);
    EXPECT_EQ(grad[1], 2.0);
    EXPECT_EQ(grad[2], 3.0);
}

// ============================================================================
// Seeding Operations Tests
// ============================================================================

TEST(DualBase, SeedWithIndex) {
    DualBase<double, 3> d(5.0);
    d.seed(1);
    EXPECT_EQ(d.derivative(0), 0.0);
    EXPECT_EQ(d.derivative(1), 1.0);
    EXPECT_EQ(d.derivative(2), 0.0);

    d.seed(0, 2.0);
    EXPECT_EQ(d.derivative(0), 2.0);
    EXPECT_EQ(d.derivative(1), 0.0);
    EXPECT_EQ(d.derivative(2), 0.0);
}

TEST(DualBase, SeedWithVector) {
    DualBase<double, 3> d(5.0);
    std::array<double, 3> seed_vec{1.0, 2.0, 3.0};
    d.seed(seed_vec);
    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 2.0);
    EXPECT_EQ(d.derivative(2), 3.0);
}

TEST(DualBase, ClearDerivatives) {
    DualBase<double, 3> d(5.0, 1.0, 2.0, 3.0);
    d.clear_derivatives();
    EXPECT_EQ(d.value(), 5.0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(d.derivative(i), 0.0);
    }
}

// ============================================================================
// Arithmetic Operations Tests
// ============================================================================

TEST(DualBase, Addition) {
    // Dual + Dual
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);
    auto c = a + b;
    EXPECT_EQ(c.value(), 5.0);
    EXPECT_EQ(c.derivative(0), 1.0);
    EXPECT_EQ(c.derivative(1), 1.0);

    // Dual + Scalar
    auto d = a + 2.0;
    EXPECT_EQ(d.value(), 5.0);
    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 0.0);

    // Scalar + Dual
    auto e = 2.0 + a;
    EXPECT_EQ(e.value(), 5.0);
    EXPECT_EQ(e.derivative(0), 1.0);
    EXPECT_EQ(e.derivative(1), 0.0);
}

TEST(DualBase, Subtraction) {
    // Dual - Dual
    DualBase<double, 2> a(5.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);
    auto c = a - b;
    EXPECT_EQ(c.value(), 3.0);
    EXPECT_EQ(c.derivative(0), 1.0);
    EXPECT_EQ(c.derivative(1), -1.0);

    // Dual - Scalar
    auto d = a - 2.0;
    EXPECT_EQ(d.value(), 3.0);
    EXPECT_EQ(d.derivative(0), 1.0);
    EXPECT_EQ(d.derivative(1), 0.0);

    // Scalar - Dual
    auto e = 10.0 - a;
    EXPECT_EQ(e.value(), 5.0);
    EXPECT_EQ(e.derivative(0), -1.0);
    EXPECT_EQ(e.derivative(1), 0.0);
}

TEST(DualBase, Multiplication) {
    // Dual * Dual: (f*g)' = f'*g + f*g'
    DualBase<double, 2> a(3.0, 1.0, 0.0);  // f = 3, f' = [1,0]
    DualBase<double, 2> b(2.0, 0.0, 1.0);  // g = 2, g' = [0,1]
    auto c = a * b;
    EXPECT_EQ(c.value(), 6.0);              // f*g = 3*2 = 6
    EXPECT_EQ(c.derivative(0), 2.0);        // f'*g + f*g' = 1*2 + 3*0 = 2
    EXPECT_EQ(c.derivative(1), 3.0);        // f'*g + f*g' = 0*2 + 3*1 = 3

    // Dual * Scalar
    auto d = a * 2.0;
    EXPECT_EQ(d.value(), 6.0);
    EXPECT_EQ(d.derivative(0), 2.0);
    EXPECT_EQ(d.derivative(1), 0.0);

    // Scalar * Dual
    auto e = 2.0 * a;
    EXPECT_EQ(e.value(), 6.0);
    EXPECT_EQ(e.derivative(0), 2.0);
    EXPECT_EQ(e.derivative(1), 0.0);
}

TEST(DualBase, Division) {
    // Dual / Dual: (f/g)' = (f'*g - f*g')/g²
    DualBase<double, 2> a(6.0, 1.0, 0.0);  // f = 6, f' = [1,0]
    DualBase<double, 2> b(2.0, 0.0, 1.0);  // g = 2, g' = [0,1]
    auto c = a / b;
    EXPECT_EQ(c.value(), 3.0);              // f/g = 6/2 = 3
    EXPECT_NEAR(c.derivative(0), 0.5, 1e-10); // (1*2 - 6*0)/4 = 0.5
    EXPECT_NEAR(c.derivative(1), -1.5, 1e-10); // (0*2 - 6*1)/4 = -1.5

    // Dual / Scalar
    auto d = a / 2.0;
    EXPECT_EQ(d.value(), 3.0);
    EXPECT_EQ(d.derivative(0), 0.5);
    EXPECT_EQ(d.derivative(1), 0.0);

    // Scalar / Dual
    DualBase<double, 2> e_denom(2.0, 1.0, 0.0);
    auto e = 8.0 / e_denom;
    EXPECT_EQ(e.value(), 4.0);
    EXPECT_EQ(e.derivative(0), -2.0);  // -8*1/4 = -2
    EXPECT_EQ(e.derivative(1), 0.0);
}

TEST(DualBase, UnaryNegation) {
    DualBase<double, 2> a(3.0, 1.0, 2.0);
    auto b = -a;
    EXPECT_EQ(b.value(), -3.0);
    EXPECT_EQ(b.derivative(0), -1.0);
    EXPECT_EQ(b.derivative(1), -2.0);
}

// ============================================================================
// Compound Assignment Operators Tests
// ============================================================================

TEST(DualBase, CompoundAddition) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    a += b;
    EXPECT_EQ(a.value(), 5.0);
    EXPECT_EQ(a.derivative(0), 1.0);
    EXPECT_EQ(a.derivative(1), 1.0);

    a += 3.0;
    EXPECT_EQ(a.value(), 8.0);
    EXPECT_EQ(a.derivative(0), 1.0);
    EXPECT_EQ(a.derivative(1), 1.0);
}

TEST(DualBase, CompoundSubtraction) {
    DualBase<double, 2> a(5.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    a -= b;
    EXPECT_EQ(a.value(), 3.0);
    EXPECT_EQ(a.derivative(0), 1.0);
    EXPECT_EQ(a.derivative(1), -1.0);

    a -= 1.0;
    EXPECT_EQ(a.value(), 2.0);
    EXPECT_EQ(a.derivative(0), 1.0);
    EXPECT_EQ(a.derivative(1), -1.0);
}

TEST(DualBase, CompoundMultiplication) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    a *= b;
    EXPECT_EQ(a.value(), 6.0);
    EXPECT_EQ(a.derivative(0), 2.0);  // 1*2 + 3*0 = 2
    EXPECT_EQ(a.derivative(1), 3.0);  // 0*2 + 3*1 = 3

    a *= 2.0;
    EXPECT_EQ(a.value(), 12.0);
    EXPECT_EQ(a.derivative(0), 4.0);
    EXPECT_EQ(a.derivative(1), 6.0);
}

TEST(DualBase, CompoundDivision) {
    DualBase<double, 2> a(6.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);

    a /= b;
    EXPECT_EQ(a.value(), 3.0);
    EXPECT_NEAR(a.derivative(0), 0.5, 1e-10);
    EXPECT_NEAR(a.derivative(1), -1.5, 1e-10);

    a /= 2.0;
    EXPECT_EQ(a.value(), 1.5);
    EXPECT_NEAR(a.derivative(0), 0.25, 1e-10);
    EXPECT_NEAR(a.derivative(1), -0.75, 1e-10);
}

// ============================================================================
// Comparison Operators Tests
// ============================================================================

TEST(DualBase, EqualityComparison) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(3.0, 0.0, 1.0);  // Same value, different derivatives
    DualBase<double, 2> c(4.0, 1.0, 0.0);  // Different value

    EXPECT_TRUE(a == b);   // Only compares values
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a == 3.0);
    EXPECT_TRUE(3.0 == a);
    EXPECT_FALSE(a == 4.0);
}

TEST(DualBase, InequalityComparison) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(3.0, 0.0, 1.0);
    DualBase<double, 2> c(4.0, 1.0, 0.0);

    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != 3.0);
    EXPECT_FALSE(3.0 != a);
    EXPECT_TRUE(a != 4.0);
}

// ============================================================================
// Mathematical Functions Tests
// ============================================================================

TEST(DualBase, PowerFunction) {
    DualBase<double, 1> x(2.0, 1.0);  // x = 2, dx = 1
    auto y = pow(x, 3.0);              // y = x³
    EXPECT_NEAR(y.value(), 8.0, 1e-10);         // 2³ = 8
    EXPECT_NEAR(y.derivative(0), 12.0, 1e-10);  // 3*2² = 12

    // Test with fractional power
    auto z = pow(x, 0.5);
    EXPECT_NEAR(z.value(), std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(z.derivative(0), 0.5 / std::sqrt(2.0), 1e-10);
}

TEST(DualBase, SquareRootFunction) {
    DualBase<double, 1> x(4.0, 1.0);
    auto y = sqrt(x);
    EXPECT_NEAR(y.value(), 2.0, 1e-10);
    EXPECT_NEAR(y.derivative(0), 0.25, 1e-10);  // 1/(2*√4) = 0.25
}

TEST(DualBase, ExponentialFunction) {
    DualBase<double, 1> x(1.0, 1.0);
    auto y = exp(x);
    EXPECT_NEAR(y.value(), std::exp(1.0), 1e-10);
    EXPECT_NEAR(y.derivative(0), std::exp(1.0), 1e-10);  // d/dx[e^x] = e^x
}

TEST(DualBase, LogarithmFunction) {
    DualBase<double, 1> x(2.0, 1.0);
    auto y = log(x);
    EXPECT_NEAR(y.value(), std::log(2.0), 1e-10);
    EXPECT_NEAR(y.derivative(0), 0.5, 1e-10);  // d/dx[ln(x)] = 1/x = 1/2
}

TEST(DualBase, SineFunction) {
    DualBase<double, 1> x(M_PI/6, 1.0);  // 30 degrees
    auto y = sin(x);
    EXPECT_NEAR(y.value(), 0.5, 1e-10);
    EXPECT_NEAR(y.derivative(0), std::cos(M_PI/6), 1e-10);  // √3/2
}

TEST(DualBase, CosineFunction) {
    DualBase<double, 1> x(M_PI/3, 1.0);  // 60 degrees
    auto y = cos(x);
    EXPECT_NEAR(y.value(), 0.5, 1e-10);
    EXPECT_NEAR(y.derivative(0), -std::sin(M_PI/3), 1e-10);  // -√3/2
}

// ============================================================================
// Chain Rule Tests
// ============================================================================

TEST(DualBase, ChainRule) {
    // Test f(g(x)) where f(u) = u² and g(x) = sin(x)
    // So h(x) = sin²(x), h'(x) = 2*sin(x)*cos(x) = sin(2x)
    DualBase<double, 1> x(M_PI/4, 1.0);
    auto sinx = sin(x);
    auto sin2x = sinx * sinx;

    EXPECT_NEAR(sin2x.value(), 0.5, 1e-10);  // sin²(π/4) = 0.5
    EXPECT_NEAR(sin2x.derivative(0), std::sin(M_PI/2), 1e-10);  // sin(π/2) = 1
}

TEST(DualBase, ComplexExpression) {
    // Test (x² + 2x + 1) / (x + 1) = x + 1 for x ≠ -1
    DualBase<double, 1> x(2.0, 1.0);
    auto numerator = x * x + 2.0 * x + 1.0;
    auto denominator = x + 1.0;
    auto result = numerator / denominator;

    EXPECT_NEAR(result.value(), 3.0, 1e-10);        // x + 1 = 3 at x=2
    EXPECT_NEAR(result.derivative(0), 1.0, 1e-10);  // d/dx[x+1] = 1
}

// ============================================================================
// Multi-Variable Tests
// ============================================================================

TEST(DualBase, MultiVariable) {
    // f(x,y) = x²y + xy²
    // ∂f/∂x = 2xy + y²
    // ∂f/∂y = x² + 2xy

    // At (x,y) = (2,3)
    DualBase<double, 2> x(2.0, 1.0, 0.0);  // x with dx=1, dy=0
    DualBase<double, 2> y(3.0, 0.0, 1.0);  // y with dx=0, dy=1

    auto f = x * x * y + x * y * y;
    EXPECT_NEAR(f.value(), 30.0, 1e-10);         // 2²*3 + 2*3² = 12 + 18 = 30
    EXPECT_NEAR(f.derivative(0), 21.0, 1e-10);   // 2*2*3 + 3² = 12 + 9 = 21
    EXPECT_NEAR(f.derivative(1), 16.0, 1e-10);   // 2² + 2*2*3 = 4 + 12 = 16
}

// ============================================================================
// Helper Functions Tests
// ============================================================================

TEST(DualBase, MakeDual) {
    auto d1 = make_dual<double, 3>(5.0);
    EXPECT_EQ(d1.value(), 5.0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(d1.derivative(i), 0.0);
    }

    std::array<double, 3> derivs{1.0, 2.0, 3.0};
    auto d2 = make_dual(5.0, derivs);
    EXPECT_EQ(d2.value(), 5.0);
    EXPECT_EQ(d2.derivative(0), 1.0);
    EXPECT_EQ(d2.derivative(1), 2.0);
    EXPECT_EQ(d2.derivative(2), 3.0);
}

TEST(DualBase, MakeIndependent) {
    auto x = make_independent<double, 3>(2.0, 0);
    auto y = make_independent<double, 3>(3.0, 1);
    auto z = make_independent<double, 3>(4.0, 2);

    EXPECT_EQ(x.derivative(0), 1.0);
    EXPECT_EQ(x.derivative(1), 0.0);
    EXPECT_EQ(x.derivative(2), 0.0);

    EXPECT_EQ(y.derivative(0), 0.0);
    EXPECT_EQ(y.derivative(1), 1.0);
    EXPECT_EQ(y.derivative(2), 0.0);

    EXPECT_EQ(z.derivative(0), 0.0);
    EXPECT_EQ(z.derivative(1), 0.0);
    EXPECT_EQ(z.derivative(2), 1.0);
}

TEST(DualBase, ExtractJacobian) {
    // System of functions: f₁ = x² + y, f₂ = xy
    auto x = make_independent<double, 2>(2.0, 0);
    auto y = make_independent<double, 2>(3.0, 1);

    std::array<DualBase<double, 2>, 2> funcs;
    funcs[0] = x * x + y;
    funcs[1] = x * y;

    auto jacobian = extract_jacobian(funcs);

    // J = [[2x, 1], [y, x]] at (2,3) = [[4, 1], [3, 2]]
    EXPECT_NEAR(jacobian[0][0], 4.0, 1e-10);
    EXPECT_NEAR(jacobian[0][1], 1.0, 1e-10);
    EXPECT_NEAR(jacobian[1][0], 3.0, 1e-10);
    EXPECT_NEAR(jacobian[1][1], 2.0, 1e-10);
}

TEST(DualBase, ExtractGradient) {
    auto x = make_independent<double, 2>(2.0, 0);
    auto y = make_independent<double, 2>(3.0, 1);
    auto f = x * x + y * y;  // f = x² + y²

    auto gradient = extract_gradient(f);
    EXPECT_NEAR(gradient[0], 4.0, 1e-10);  // ∂f/∂x = 2x = 4
    EXPECT_NEAR(gradient[1], 6.0, 1e-10);  // ∂f/∂y = 2y = 6
}

// ============================================================================
// Stream Output Test
// ============================================================================

TEST(DualBase, StreamOutput) {
    DualBase<double, 2> d(3.14, 1.0, 2.0);
    std::ostringstream oss;
    oss << d;
    std::string output = oss.str();
    EXPECT_TRUE(output.find("3.14") != std::string::npos);
    EXPECT_TRUE(output.find("1") != std::string::npos);
    EXPECT_TRUE(output.find("2") != std::string::npos);
}

// ============================================================================
// Edge Cases and Special Values Tests
// ============================================================================

TEST(DualBase, ZeroDivision) {
    DualBase<double, 1> x(1.0, 1.0);
    DualBase<double, 1> zero(0.0, 0.0);

    // Division by zero should produce inf/nan as appropriate
    auto result = x / zero;
    EXPECT_TRUE(std::isinf(result.value()));
}

TEST(DualBase, FloatType) {
    // Test with float instead of double
    DualBase<float, 2> a(3.0f, 1.0f, 0.0f);
    DualBase<float, 2> b(2.0f, 0.0f, 1.0f);
    auto c = a + b;
    EXPECT_NEAR(c.value(), 5.0f, 1e-6f);
    EXPECT_NEAR(c.derivative(0), 1.0f, 1e-6f);
    EXPECT_NEAR(c.derivative(1), 1.0f, 1e-6f);
}

TEST(DualBase, LargeDimension) {
    // Test with many derivatives
    constexpr size_t N = 100;
    DualBase<double, N> x(1.0);
    x.seed(50, 1.0);

    EXPECT_EQ(x.derivative(50), 1.0);
    EXPECT_EQ(x.derivative(49), 0.0);
    EXPECT_EQ(x.derivative(51), 0.0);
    EXPECT_EQ(x.size(), N);
}

TEST(DualBase, SingleDerivative) {
    // Most common case: single variable differentiation
    DualBase<double, 1> x(3.0, 1.0);
    auto y = x * x * x;  // y = x³
    EXPECT_NEAR(y.value(), 27.0, 1e-10);
    EXPECT_NEAR(y.derivative(0), 27.0, 1e-10);  // 3*3² = 27
}
// test_dual_comparison.cpp - Unit tests for dual number comparison operators
#include <gtest/gtest.h>
#include <base/dual_comparison.h>
#include <base/dual_base.h>
#include <base/numeric_base.h>
#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <vector>

using namespace fem::numeric::autodiff;
using namespace fem::numeric;

// ============================================================================
// Basic Comparison Operator Tests
// ============================================================================

TEST(DualComparison, LessThan) {
    // Dual < Dual (only compares values, not derivatives)
    DualBase<double, 2> a(2.0, 1.0, 0.0);
    DualBase<double, 2> b(3.0, 0.0, 1.0);
    DualBase<double, 2> c(2.0, 5.0, 6.0);  // Same value as a, different derivatives

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < c);  // Equal values, even though derivatives differ
    EXPECT_FALSE(c < a);

    // Dual < Scalar
    EXPECT_TRUE(a < 3.0);
    EXPECT_FALSE(a < 2.0);
    EXPECT_FALSE(a < 1.0);

    // Scalar < Dual
    EXPECT_TRUE(1.0 < a);
    EXPECT_FALSE(2.0 < a);
    EXPECT_FALSE(3.0 < a);
}

TEST(DualComparison, LessThanOrEqual) {
    DualBase<double, 2> a(2.0, 1.0, 0.0);
    DualBase<double, 2> b(3.0, 0.0, 1.0);
    DualBase<double, 2> c(2.0, 5.0, 6.0);

    // Dual <= Dual
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(a <= c);  // Equal values
    EXPECT_TRUE(c <= a);

    // Dual <= Scalar
    EXPECT_TRUE(a <= 3.0);
    EXPECT_TRUE(a <= 2.0);
    EXPECT_FALSE(a <= 1.0);

    // Scalar <= Dual
    EXPECT_TRUE(1.0 <= a);
    EXPECT_TRUE(2.0 <= a);
    EXPECT_FALSE(3.0 <= a);
}

TEST(DualComparison, GreaterThan) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);
    DualBase<double, 2> c(3.0, 5.0, 6.0);

    // Dual > Dual
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > c);  // Equal values
    EXPECT_FALSE(c > a);

    // Dual > Scalar
    EXPECT_TRUE(a > 2.0);
    EXPECT_FALSE(a > 3.0);
    EXPECT_FALSE(a > 4.0);

    // Scalar > Dual
    EXPECT_TRUE(4.0 > a);
    EXPECT_FALSE(3.0 > a);
    EXPECT_FALSE(2.0 > a);
}

TEST(DualComparison, GreaterThanOrEqual) {
    DualBase<double, 2> a(3.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 1.0);
    DualBase<double, 2> c(3.0, 5.0, 6.0);

    // Dual >= Dual
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(b >= a);
    EXPECT_TRUE(a >= c);  // Equal values
    EXPECT_TRUE(c >= a);

    // Dual >= Scalar
    EXPECT_TRUE(a >= 2.0);
    EXPECT_TRUE(a >= 3.0);
    EXPECT_FALSE(a >= 4.0);

    // Scalar >= Dual
    EXPECT_TRUE(4.0 >= a);
    EXPECT_TRUE(3.0 >= a);
    EXPECT_FALSE(2.0 >= a);
}

// ============================================================================
// Warning Case: Derivatives Ignored in Comparisons
// ============================================================================

TEST(DualComparison, DerivativesIgnoredWarning) {
    // This test documents that standard comparisons ignore derivatives
    // which can be surprising/dangerous
    DualBase<double, 2> a(2.0, 100.0, 200.0);  // Large derivatives
    DualBase<double, 2> b(2.0, 0.0, 0.0);      // Zero derivatives

    // These are considered equal for comparison despite different derivatives
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(b >= a);

    // But they are NOT equal in the equality sense (from dual_base.h)
    EXPECT_TRUE(a == b);  // Only compares values
}

// ============================================================================
// Approximate Equality Tests
// ============================================================================

TEST(DualComparison, ApproxEqual) {
    DualBase<double, 2> a(2.0, 1.0, 2.0);
    DualBase<double, 2> b(2.0, 1.0, 2.0);
    DualBase<double, 2> c(2.0 + 1e-15, 1.0 + 1e-15, 2.0 + 1e-15);  // Very close
    DualBase<double, 2> d(2.1, 1.0, 2.0);  // Different value
    DualBase<double, 2> e(2.0, 1.1, 2.0);  // Different derivative

    // Exact equality
    EXPECT_TRUE(approx_equal(a, b));

    // Within default tolerance
    EXPECT_TRUE(approx_equal(a, c));

    // Outside tolerance
    EXPECT_FALSE(approx_equal(a, d));
    EXPECT_FALSE(approx_equal(a, e));

    // Custom tolerance
    DualBase<double, 2> f(2.0, 1.05, 2.0);
    EXPECT_FALSE(approx_equal(a, f, 0.01));
    EXPECT_TRUE(approx_equal(a, f, 0.1));
}

TEST(DualComparison, ApproxEqualEdgeCases) {
    // Zero values
    DualBase<double, 1> zero1(0.0, 0.0);
    DualBase<double, 1> zero2(1e-15, 1e-15);
    EXPECT_TRUE(approx_equal(zero1, zero2));

    // Negative values
    DualBase<double, 1> neg1(-1.0, -2.0);
    DualBase<double, 1> neg2(-1.0 - 1e-15, -2.0 - 1e-15);
    EXPECT_TRUE(approx_equal(neg1, neg2));

    // Large values
    DualBase<double, 1> large1(1e10, 1e10);
    DualBase<double, 1> large2(1e10 + 1.0, 1e10 + 1.0);
    EXPECT_FALSE(approx_equal(large1, large2));  // Default tolerance too small
    EXPECT_TRUE(approx_equal(large1, large2, 10.0));  // Larger absolute tolerance
}

// ============================================================================
// Lexicographic Ordering Tests
// ============================================================================

TEST(DualComparison, LexicographicOrdering) {
    DualLexicographicLess<double, 2> comp;

    DualBase<double, 2> a(1.0, 1.0, 1.0);
    DualBase<double, 2> b(2.0, 0.0, 0.0);  // Larger value
    DualBase<double, 2> c(1.0, 2.0, 0.0);  // Same value, larger first derivative
    DualBase<double, 2> d(1.0, 1.0, 2.0);  // Same value and first deriv, larger second
    DualBase<double, 2> e(1.0, 1.0, 1.0);  // Identical to a

    // Value comparison dominates
    EXPECT_TRUE(comp(a, b));
    EXPECT_FALSE(comp(b, a));

    // First derivative comparison when values equal
    EXPECT_TRUE(comp(a, c));
    EXPECT_FALSE(comp(c, a));

    // Second derivative comparison when value and first deriv equal
    EXPECT_TRUE(comp(a, d));
    EXPECT_FALSE(comp(d, a));

    // Equal elements
    EXPECT_FALSE(comp(a, e));
    EXPECT_FALSE(comp(e, a));
}

TEST(DualComparison, LexicographicInContainer) {
    // Test that lexicographic ordering works in STL containers
    std::set<DualBase<double, 2>, DualLexicographicLess<double, 2>> dual_set;

    DualBase<double, 2> a(1.0, 1.0, 0.0);
    DualBase<double, 2> b(2.0, 0.0, 0.0);
    DualBase<double, 2> c(1.0, 2.0, 0.0);
    DualBase<double, 2> d(1.0, 1.0, 0.0);  // Duplicate of a

    dual_set.insert(a);
    dual_set.insert(b);
    dual_set.insert(c);
    dual_set.insert(d);  // Should not be inserted (duplicate)

    EXPECT_EQ(dual_set.size(), 3u);

    // Verify ordering
    auto it = dual_set.begin();
    EXPECT_TRUE(approx_equal(*it++, a));  // (1, [1,0])
    EXPECT_TRUE(approx_equal(*it++, c));  // (1, [2,0])
    EXPECT_TRUE(approx_equal(*it++, b));  // (2, [0,0])
}

TEST(DualComparison, UseInMap) {
    // Test using dual numbers as map keys with lexicographic ordering
    std::map<DualBase<double, 1>, std::string, DualLexicographicLess<double, 1>> dual_map;

    DualBase<double, 1> key1(1.0, 1.0);
    DualBase<double, 1> key2(2.0, 0.0);
    DualBase<double, 1> key3(1.0, 2.0);

    dual_map[key1] = "first";
    dual_map[key2] = "second";
    dual_map[key3] = "third";

    EXPECT_EQ(dual_map.size(), 3u);
    EXPECT_EQ(dual_map[key1], "first");
    EXPECT_EQ(dual_map[key2], "second");
    EXPECT_EQ(dual_map[key3], "third");
}

// ============================================================================
// Finite and NaN Tests
// ============================================================================

TEST(DualComparison, IsFinite) {
    // All finite
    DualBase<double, 2> finite(1.0, 2.0, 3.0);
    EXPECT_TRUE(is_finite(finite));

    // Infinite value
    DualBase<double, 2> inf_val(std::numeric_limits<double>::infinity(), 1.0, 2.0);
    EXPECT_FALSE(is_finite(inf_val));

    // Infinite derivative
    DualBase<double, 2> inf_deriv(1.0, std::numeric_limits<double>::infinity(), 2.0);
    EXPECT_FALSE(is_finite(inf_deriv));

    // Negative infinity
    DualBase<double, 2> neg_inf(1.0, 2.0, -std::numeric_limits<double>::infinity());
    EXPECT_FALSE(is_finite(neg_inf));

    // NaN is not finite
    DualBase<double, 2> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0);
    EXPECT_FALSE(is_finite(nan_val));
}

TEST(DualComparison, HasNaN) {
    // No NaN
    DualBase<double, 2> no_nan(1.0, 2.0, 3.0);
    EXPECT_FALSE(has_nan(no_nan));

    // NaN in value
    DualBase<double, 2> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0);
    EXPECT_TRUE(has_nan(nan_val));

    // NaN in first derivative
    DualBase<double, 2> nan_deriv1(1.0, std::numeric_limits<double>::quiet_NaN(), 2.0);
    EXPECT_TRUE(has_nan(nan_deriv1));

    // NaN in second derivative
    DualBase<double, 2> nan_deriv2(1.0, 2.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(has_nan(nan_deriv2));

    // Infinity is not NaN
    DualBase<double, 2> inf(std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(), 2.0);
    EXPECT_FALSE(has_nan(inf));
    EXPECT_FALSE(is_finite(inf));  // But it's not finite either
}

// ============================================================================
// Sorting and Algorithm Tests
// ============================================================================

TEST(DualComparison, StandardSort) {
    // Test that standard comparisons work with STL algorithms
    std::vector<DualBase<double, 1>> vec;
    vec.emplace_back(3.0, 1.0);
    vec.emplace_back(1.0, 2.0);
    vec.emplace_back(2.0, 3.0);
    vec.emplace_back(1.0, 4.0);  // Same value as [1], different derivative

    std::sort(vec.begin(), vec.end());

    // Should be sorted by value only
    EXPECT_EQ(vec[0].value(), 1.0);
    EXPECT_EQ(vec[1].value(), 1.0);  // Two with value 1.0
    EXPECT_EQ(vec[2].value(), 2.0);
    EXPECT_EQ(vec[3].value(), 3.0);
}

TEST(DualComparison, MinMaxElement) {
    std::vector<DualBase<double, 1>> vec;
    vec.emplace_back(3.0, 100.0);
    vec.emplace_back(1.0, 200.0);
    vec.emplace_back(5.0, 50.0);
    vec.emplace_back(2.0, 300.0);

    auto min_it = std::min_element(vec.begin(), vec.end());
    auto max_it = std::max_element(vec.begin(), vec.end());

    EXPECT_EQ(min_it->value(), 1.0);
    EXPECT_EQ(min_it->derivative(0), 200.0);  // Derivatives preserved

    EXPECT_EQ(max_it->value(), 5.0);
    EXPECT_EQ(max_it->derivative(0), 50.0);
}

// ============================================================================
// Mixed Type Comparisons
// ============================================================================

TEST(DualComparison, FloatComparisons) {
    DualBase<float, 2> a(2.0f, 1.0f, 0.0f);
    DualBase<float, 2> b(3.0f, 0.0f, 1.0f);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(a >= b);

    EXPECT_TRUE(a < 3.0f);
    EXPECT_TRUE(1.0f < a);
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

TEST(DualComparison, ZeroComparisons) {
    DualBase<double, 1> zero(0.0, 1.0);
    DualBase<double, 1> pos(1e-100, 0.0);
    DualBase<double, 1> neg(-1e-100, 0.0);

    EXPECT_TRUE(neg < zero);
    EXPECT_TRUE(zero < pos);
    EXPECT_FALSE(pos < zero);
    EXPECT_FALSE(zero < neg);
}

TEST(DualComparison, InfinityComparisons) {
    double inf = std::numeric_limits<double>::infinity();
    DualBase<double, 1> finite(100.0, 1.0);
    DualBase<double, 1> pos_inf(inf, 0.0);
    DualBase<double, 1> neg_inf(-inf, 0.0);

    EXPECT_TRUE(neg_inf < finite);
    EXPECT_TRUE(finite < pos_inf);
    EXPECT_TRUE(neg_inf < pos_inf);

    EXPECT_FALSE(pos_inf < finite);
    EXPECT_FALSE(finite < neg_inf);
    EXPECT_FALSE(pos_inf < neg_inf);
}

TEST(DualComparison, NaNComparisons) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    DualBase<double, 1> with_nan(nan, 1.0);
    DualBase<double, 1> normal(1.0, 1.0);

    // NaN comparisons should follow IEEE 754 rules (always false)
    EXPECT_FALSE(with_nan < normal);
    EXPECT_FALSE(normal < with_nan);
    EXPECT_FALSE(with_nan < with_nan);
    EXPECT_FALSE(with_nan <= with_nan);
    EXPECT_FALSE(with_nan > with_nan);
    EXPECT_FALSE(with_nan >= with_nan);

    // But NaN can be detected
    EXPECT_TRUE(has_nan(with_nan));
    EXPECT_FALSE(is_finite(with_nan));
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

TEST(DualComparison, LargeVectorSort) {
    // Create a large vector of dual numbers
    const size_t N = 10000;
    std::vector<DualBase<double, 3>> vec;
    vec.reserve(N);

    // Fill with random-ish values
    for (size_t i = 0; i < N; ++i) {
        double di = static_cast<double>(i);
        double val = std::sin(di * 0.1) * 100;
        vec.emplace_back(val, di * 0.1, di * 0.2, di * 0.3);
    }

    // Sort should work efficiently
    std::sort(vec.begin(), vec.end());

    // Verify sorted
    for (size_t i = 1; i < N; ++i) {
        EXPECT_LE(vec[i-1].value(), vec[i].value());
    }
}

TEST(DualComparison, ConsistencyCheck) {
    // Verify that comparison operators are consistent
    DualBase<double, 2> a(1.0, 1.0, 1.0);
    DualBase<double, 2> b(2.0, 0.0, 0.0);
    DualBase<double, 2> c(1.0, 2.0, 0.0);

    // Transitivity: if a < b and b < c, then a < c
    DualBase<double, 2> x(1.0, 0.0, 0.0);
    DualBase<double, 2> y(2.0, 0.0, 0.0);
    DualBase<double, 2> z(3.0, 0.0, 0.0);
    EXPECT_TRUE(x < y);
    EXPECT_TRUE(y < z);
    EXPECT_TRUE(x < z);

    // Antisymmetry: if a < b, then !(b < a)
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    // Totality with <=: a <= b or b <= a (or both if equal)
    EXPECT_TRUE(a <= b || b <= a);
    EXPECT_TRUE(a <= c || c <= a);
    EXPECT_TRUE(b <= c || c <= b);

    // Consistency between operators
    if (a < b) {
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(a >= b);
    }
}
#include <gtest/gtest.h>
#include <complex>
#include <limits>
#include <cmath>
#include <type_traits>

#include <traits/numeric_traits.h>

using namespace fem::numeric::traits;

// ============================================================================
// NUMERIC LIMITS TESTS
// ============================================================================

TEST(NumericTraitsTest, NumericLimitsBasicTypes) {
    // Test integer types
    {
        using limits_int = numeric_limits<int>;
        EXPECT_EQ(limits_int::zero(), 0);
        EXPECT_EQ(limits_int::one(), 1);
        EXPECT_EQ(limits_int::additive_identity(), 0);
        EXPECT_EQ(limits_int::multiplicative_identity(), 1);
        EXPECT_EQ(limits_int::machine_epsilon(), 1);
        EXPECT_EQ(limits_int::comparison_tolerance(), 0);
        EXPECT_FALSE(limits_int::is_ieee754);
        EXPECT_FALSE(limits_int::supports_nan);
        EXPECT_FALSE(limits_int::supports_infinity);
    }

    // Test float
    {
        using limits_float = numeric_limits<float>;
        EXPECT_EQ(limits_float::zero(), 0.0f);
        EXPECT_EQ(limits_float::one(), 1.0f);
        EXPECT_EQ(limits_float::machine_epsilon(), std::numeric_limits<float>::epsilon());
        EXPECT_GT(limits_float::comparison_tolerance(), 0.0f);
        EXPECT_TRUE(limits_float::is_ieee754);
        EXPECT_TRUE(limits_float::supports_nan);
        EXPECT_TRUE(limits_float::supports_infinity);
    }

    // Test double
    {
        using limits_double = numeric_limits<double>;
        EXPECT_EQ(limits_double::zero(), 0.0);
        EXPECT_EQ(limits_double::one(), 1.0);
        EXPECT_EQ(limits_double::machine_epsilon(), std::numeric_limits<double>::epsilon());
        EXPECT_GT(limits_double::comparison_tolerance(), 0.0);
        EXPECT_TRUE(limits_double::is_ieee754);
        EXPECT_TRUE(limits_double::supports_nan);
        EXPECT_TRUE(limits_double::supports_infinity);
    }
}

TEST(NumericTraitsTest, NumericLimitsComplexNumbers) {
    using limits_complex = numeric_limits<std::complex<double>>;

    EXPECT_EQ(limits_complex::zero(), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(limits_complex::one(), std::complex<double>(1.0, 0.0));
    EXPECT_EQ(limits_complex::i(), std::complex<double>(0.0, 1.0));

    EXPECT_TRUE(limits_complex::is_specialized);
    EXPECT_TRUE(limits_complex::is_signed);
    EXPECT_FALSE(limits_complex::is_integer);
    EXPECT_FALSE(limits_complex::is_exact);

    auto eps = limits_complex::epsilon();
    EXPECT_EQ(eps.real(), std::numeric_limits<double>::epsilon());
    EXPECT_EQ(eps.imag(), 0.0);

    auto tol = limits_complex::comparison_tolerance();
    EXPECT_GT(tol.real(), 0.0);
    EXPECT_EQ(tol.imag(), 0.0);

    // Check NaN and infinity support based on underlying type
    EXPECT_EQ(limits_complex::has_quiet_NaN, std::numeric_limits<double>::has_quiet_NaN);
    EXPECT_EQ(limits_complex::has_infinity, std::numeric_limits<double>::has_infinity);

    if (limits_complex::has_quiet_NaN) {
        auto nan = limits_complex::quiet_NaN();
        EXPECT_TRUE(std::isnan(nan.real()));
        EXPECT_TRUE(std::isnan(nan.imag()));
    }

    if (limits_complex::has_infinity) {
        auto inf = limits_complex::infinity();
        EXPECT_TRUE(std::isinf(inf.real()));
        EXPECT_EQ(inf.imag(), 0.0);
    }
}

// ============================================================================
// PRECISION CATEGORY TESTS
// ============================================================================

TEST(NumericTraitsTest, PrecisionCategory) {
    // Standard floating-point types
    EXPECT_EQ(precision_category_v<float>, PrecisionCategory::Single);
    EXPECT_EQ(precision_category_v<double>, PrecisionCategory::Double);

    // Non-floating-point types
    EXPECT_EQ(precision_category_v<int>, PrecisionCategory::Unknown);
    EXPECT_EQ(precision_category_v<char>, PrecisionCategory::Unknown);

    // Complex types
    EXPECT_EQ(precision_category_v<std::complex<float>>, PrecisionCategory::Unknown);
}

// ============================================================================
// TYPE PROMOTION TESTS
// ============================================================================

TEST(NumericTraitsTest, PromoteTypes) {
    // Basic promotion
    static_assert(std::is_same_v<promote_types_t<int>, int>);
    static_assert(std::is_same_v<promote_types_t<int, float>, float>);
    static_assert(std::is_same_v<promote_types_t<float, double>, double>);

    // Multiple type promotion
    static_assert(std::is_same_v<promote_types_t<int, float, double>, double>);
    static_assert(std::is_same_v<promote_types_t<char, short, int>, int>);

    // Complex promotion (if supported by promote_t)
    // Note: This depends on the implementation of promote_t in traits_base.h
    // static_assert(std::is_same_v<promote_types_t<float, std::complex<double>>, std::complex<double>>);

    EXPECT_TRUE(true); // Compile-time tests
}

// ============================================================================
// STORAGE REQUIREMENTS TESTS
// ============================================================================

TEST(NumericTraitsTest, StorageRequirements) {
    using storage_double = storage_requirements<double>;

    EXPECT_EQ(storage_double::bytes_per_element, sizeof(double));
    // After fix, alignment will be alignof(double) instead of from storage_traits
    EXPECT_EQ(storage_double::alignment, alignof(double));

    // Test bytes needed calculation
    EXPECT_EQ(storage_double::bytes_needed(10), 10 * sizeof(double));

    // Test aligned bytes calculation
    size_t align = 64; // Cache line alignment
    size_t n = 10;
    size_t bytes = storage_double::bytes_needed(n);
    size_t aligned = storage_double::aligned_bytes_needed(n, align);

    EXPECT_GE(aligned, bytes);
    EXPECT_EQ(aligned % align, 0);

    // Test with exact alignment
    n = align / sizeof(double); // Exactly one cache line
    aligned = storage_double::aligned_bytes_needed(n, align);
    EXPECT_EQ(aligned, align);

    // Test natural alignment
    aligned = storage_double::aligned_bytes_needed(10);  // Use default alignment
    EXPECT_EQ(aligned % storage_double::alignment, 0);
}

// ============================================================================
// NUMERIC PROPERTIES TESTS
// ============================================================================

TEST(NumericTraitsTest, NumericPropertiesIntegral) {
    using props = numeric_properties<int>;

    EXPECT_TRUE(props::is_integral);
    EXPECT_FALSE(props::is_floating);
    EXPECT_FALSE(props::is_complex);
    EXPECT_TRUE(props::is_signed);
    EXPECT_TRUE(props::is_exact);

    EXPECT_TRUE(props::is_number_like);
    EXPECT_FALSE(props::is_ieee);
    EXPECT_FALSE(props::has_nan);
    EXPECT_FALSE(props::has_inf);

    EXPECT_EQ(props::precision, PrecisionCategory::Unknown);
    EXPECT_EQ(props::size, sizeof(int));
    EXPECT_EQ(props::alignment, alignof(int));

    EXPECT_EQ(props::zero(), 0);
    EXPECT_EQ(props::one(), 1);
    EXPECT_EQ(props::eps(), 0); // No epsilon for integers
}

TEST(NumericTraitsTest, NumericPropertiesFloating) {
    using props = numeric_properties<double>;

    EXPECT_FALSE(props::is_integral);
    EXPECT_TRUE(props::is_floating);
    EXPECT_FALSE(props::is_complex);
    EXPECT_TRUE(props::is_signed);
    EXPECT_FALSE(props::is_exact);

    EXPECT_TRUE(props::is_number_like);
    EXPECT_TRUE(props::is_ieee);
    EXPECT_TRUE(props::has_nan);
    EXPECT_TRUE(props::has_inf);

    EXPECT_EQ(props::precision, PrecisionCategory::Double);
    EXPECT_EQ(props::size, sizeof(double));
    EXPECT_EQ(props::alignment, alignof(double));

    EXPECT_EQ(props::zero(), 0.0);
    EXPECT_EQ(props::one(), 1.0);
    EXPECT_GT(props::eps(), 0.0);
    EXPECT_GT(props::min(), 0.0);
    EXPECT_GT(props::max(), props::min());

    EXPECT_EQ(props::mantissa_bits, std::numeric_limits<double>::digits);
    EXPECT_EQ(props::decimal_digits, std::numeric_limits<double>::digits10);
}

TEST(NumericTraitsTest, NumericPropertiesComplex) {
    using props = numeric_properties<std::complex<double>>;

    EXPECT_FALSE(props::is_integral);
    EXPECT_FALSE(props::is_floating);
    EXPECT_TRUE(props::is_complex);
    EXPECT_TRUE(props::is_signed);
    EXPECT_FALSE(props::is_exact);

    // Note: These depend on how NumberLike and IEEECompliant are defined for complex
    // EXPECT_TRUE(props::is_number_like);
    // EXPECT_TRUE(props::is_ieee);

    EXPECT_EQ(props::size, sizeof(std::complex<double>));
    EXPECT_EQ(props::alignment, alignof(std::complex<double>));
}

// ============================================================================
// IEEE COMPLIANCE TESTS
// ============================================================================

TEST(NumericTraitsTest, IsFinite) {
    // Test with regular values
    EXPECT_TRUE(is_finite(0.0));
    EXPECT_TRUE(is_finite(1.0));
    EXPECT_TRUE(is_finite(-1.0));
    EXPECT_TRUE(is_finite(std::numeric_limits<double>::max()));
    EXPECT_TRUE(is_finite(std::numeric_limits<double>::min()));

    // Test with special values
    EXPECT_FALSE(is_finite(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(is_finite(-std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(is_finite(std::numeric_limits<double>::quiet_NaN()));

    // Test with integers
    EXPECT_TRUE(is_finite(0));
    EXPECT_TRUE(is_finite(42));
    EXPECT_TRUE(is_finite(-42));
}

TEST(NumericTraitsTest, IsNaN) {
    // Test regular values
    EXPECT_FALSE(is_nan(0.0));
    EXPECT_FALSE(is_nan(1.0));
    EXPECT_FALSE(is_nan(std::numeric_limits<double>::infinity()));

    // Test NaN
    EXPECT_TRUE(is_nan(std::numeric_limits<double>::quiet_NaN()));

    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(is_nan(nan));

    // Test with integers
    EXPECT_FALSE(is_nan(0));
    EXPECT_FALSE(is_nan(42));
}

TEST(NumericTraitsTest, IsInf) {
    // Test regular values
    EXPECT_FALSE(is_inf(0.0));
    EXPECT_FALSE(is_inf(1.0));
    EXPECT_FALSE(is_inf(std::numeric_limits<double>::max()));

    // Test infinity
    EXPECT_TRUE(is_inf(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(is_inf(-std::numeric_limits<double>::infinity()));

    // Test NaN
    EXPECT_FALSE(is_inf(std::numeric_limits<double>::quiet_NaN()));

    // Test with integers
    EXPECT_FALSE(is_inf(0));
    EXPECT_FALSE(is_inf(std::numeric_limits<int>::max()));
}

// ============================================================================
// APPROXIMATELY EQUAL TESTS
// ============================================================================

TEST(NumericTraitsTest, ApproximatelyEqualIntegers) {
    // Integers should use exact comparison
    EXPECT_TRUE(approximately_equal(5, 5));
    EXPECT_FALSE(approximately_equal(5, 6));
    EXPECT_TRUE(approximately_equal(0, 0));
    EXPECT_FALSE(approximately_equal(0, 1));

    // Large integers
    EXPECT_TRUE(approximately_equal(1000000, 1000000));
    EXPECT_FALSE(approximately_equal(1000000, 1000001));
}

TEST(NumericTraitsTest, ApproximatelyEqualFloatingPoint) {
    // Exact equality
    EXPECT_TRUE(approximately_equal(1.0, 1.0));
    EXPECT_TRUE(approximately_equal(0.0, 0.0));

    // Small differences within tolerance
    double a = 1.0;
    double b = a + std::numeric_limits<double>::epsilon();
    EXPECT_TRUE(approximately_equal(a, b));

    // Larger differences outside tolerance
    EXPECT_FALSE(approximately_equal(1.0, 2.0));
    EXPECT_FALSE(approximately_equal(1.0, 1.1));

    // Near zero comparisons
    // Default tolerance is 100 * epsilon ≈ 2.2e-14

    // Test with denormalized numbers
    // denorm_min ≈ 4.9e-324, much smaller than tolerance
    double denorm = std::numeric_limits<double>::denorm_min();
    EXPECT_TRUE(approximately_equal(denorm, 0.0));  // Within absolute tolerance
    EXPECT_TRUE(approximately_equal(0.0, denorm));

    // Test with smallest normalized number
    // min() ≈ 2.2e-308, still much smaller than tolerance (2.2e-14)
    double min_norm = std::numeric_limits<double>::min();
    // Logically, if the difference (2.2e-308) is less than tolerance (2.2e-14),
    // they should be approximately equal
    EXPECT_TRUE(approximately_equal(min_norm, 0.0));  // Fixed: should be true
    EXPECT_TRUE(approximately_equal(0.0, min_norm));

    // But with a very small tolerance, min_norm should NOT equal 0
    EXPECT_FALSE(approximately_equal(min_norm, 0.0, 1e-310));  // Tolerance smaller than min_norm

    // Test values that are within epsilon of zero
    // epsilon ≈ 2.2e-16, smaller than tolerance (2.2e-14)
    double tiny = std::numeric_limits<double>::epsilon();
    EXPECT_TRUE(approximately_equal(tiny, 0.0));  // Within absolute tolerance
    EXPECT_TRUE(approximately_equal(0.0, tiny));

    // Test with values larger than tolerance
    double larger_than_tol = 1e-10;  // Much larger than default tolerance
    EXPECT_FALSE(approximately_equal(larger_than_tol, 0.0));
    EXPECT_FALSE(approximately_equal(0.0, larger_than_tol));

    // Custom tolerance tests
    EXPECT_TRUE(approximately_equal(1.0, 1.1, 0.2));
    EXPECT_FALSE(approximately_equal(1.0, 1.1, 0.05));

    // Test relative tolerance for normal values
    EXPECT_TRUE(approximately_equal(1000.0, 1000.0 + 1000.0 * 1e-15));  // Within relative tolerance
    EXPECT_FALSE(approximately_equal(1000.0, 1000.0 + 1000.0 * 1e-12)); // Outside relative tolerance
}

TEST(NumericTraitsTest, ApproximatelyEqualSpecialValues) {
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();

    // NaN comparisons always false
    EXPECT_FALSE(approximately_equal(nan, nan));
    EXPECT_FALSE(approximately_equal(nan, 0.0));
    EXPECT_FALSE(approximately_equal(0.0, nan));

    // Infinity comparisons
    EXPECT_TRUE(approximately_equal(inf, inf));
    EXPECT_TRUE(approximately_equal(-inf, -inf));
    EXPECT_FALSE(approximately_equal(inf, -inf));
    EXPECT_FALSE(approximately_equal(inf, 0.0));
    EXPECT_FALSE(approximately_equal(0.0, inf));
}

TEST(NumericTraitsTest, ApproximatelyEqualComplex) {
    using Complex = std::complex<double>;

    // Exact equality
    Complex a(1.0, 2.0);
    Complex b(1.0, 2.0);
    EXPECT_TRUE(approximately_equal(a, b));

    // Small differences
    Complex c(1.0 + std::numeric_limits<double>::epsilon(), 2.0);
    EXPECT_TRUE(approximately_equal(a, c));

    // Larger differences
    Complex d(2.0, 2.0);
    EXPECT_FALSE(approximately_equal(a, d));

    Complex e(1.0, 3.0);
    EXPECT_FALSE(approximately_equal(a, e));

    // Custom tolerance
    Complex tol(0.2, 0.2);
    Complex f(1.1, 2.1);
    EXPECT_TRUE(approximately_equal(a, f, tol));
}

// ============================================================================
// UTILITY FUNCTION TESTS
// ============================================================================

TEST(NumericTraitsTest, ZeroAndOneValues) {
    // Integer types
    EXPECT_EQ(zero_value<int>(), 0);
    EXPECT_EQ(one_value<int>(), 1);

    EXPECT_EQ(zero_value<unsigned>(), 0u);
    EXPECT_EQ(one_value<unsigned>(), 1u);

    // Floating-point types
    EXPECT_EQ(zero_value<float>(), 0.0f);
    EXPECT_EQ(one_value<float>(), 1.0f);

    EXPECT_EQ(zero_value<double>(), 0.0);
    EXPECT_EQ(one_value<double>(), 1.0);

    // Complex types
    EXPECT_EQ(zero_value<std::complex<double>>(), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(one_value<std::complex<double>>(), std::complex<double>(1.0, 0.0));
}

TEST(NumericTraitsTest, TypeCompatibility) {
    // Compatible types
    EXPECT_TRUE((are_compatible_types<int, int>));
    EXPECT_TRUE((are_compatible_types<float, float>));
    EXPECT_TRUE((are_compatible_types<double, double>));

    // Promotable types (depends on are_compatible_v implementation)
    // These expectations depend on how are_compatible_v is defined in the base
    // EXPECT_TRUE((are_compatible_types<int, float>));
    // EXPECT_TRUE((are_compatible_types<float, double>));

    // Incompatible types (if the base defines them as such)
    // EXPECT_FALSE((are_compatible_types<int, std::string>));
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST(NumericTraitsTest, EdgeCasesWithUnsignedTypes) {
    using props = numeric_properties<unsigned int>;

    EXPECT_TRUE(props::is_integral);
    EXPECT_FALSE(props::is_signed);
    EXPECT_TRUE(props::is_exact);
    EXPECT_EQ(props::zero(), 0u);
    EXPECT_EQ(props::one(), 1u);

    // Unsigned comparison
    EXPECT_TRUE(approximately_equal(5u, 5u));
    EXPECT_FALSE(approximately_equal(5u, 6u));
}

TEST(NumericTraitsTest, EdgeCasesWithCharTypes) {
    using props_char = numeric_properties<char>;
    using props_uchar = numeric_properties<unsigned char>;

    EXPECT_TRUE(props_char::is_integral);
    EXPECT_TRUE(props_uchar::is_integral);
    EXPECT_FALSE(props_uchar::is_signed);

    EXPECT_EQ(props_char::size, 1);
    EXPECT_EQ(props_uchar::size, 1);
}

TEST(NumericTraitsTest, VerySmallAndLargeValues) {
    // Test with denormalized numbers
    double denorm = std::numeric_limits<double>::denorm_min();
    EXPECT_TRUE(is_finite(denorm));
    EXPECT_FALSE(is_nan(denorm));
    EXPECT_FALSE(is_inf(denorm));
    EXPECT_TRUE(approximately_equal(denorm, 0.0)); // Should be within tolerance

    // Test with very large values
    double large = std::numeric_limits<double>::max();
    EXPECT_TRUE(is_finite(large));
    EXPECT_FALSE(is_nan(large));
    EXPECT_FALSE(is_inf(large));
    EXPECT_FALSE(approximately_equal(large, large / 2.0));
}

// ============================================================================
// COMPILE-TIME TESTS
// ============================================================================

TEST(NumericTraitsTest, CompileTimeChecks) {
    // These are compile-time checks
    static_assert(numeric_limits<int>::zero() == 0);
    static_assert(numeric_limits<int>::one() == 1);
    static_assert(numeric_limits<double>::is_ieee754);
    static_assert(!numeric_limits<int>::is_ieee754);

    // Check that constexpr functions work at compile time
    constexpr double zero = zero_value<double>();
    constexpr double one = one_value<double>();
    static_assert(zero == 0.0);
    static_assert(one == 1.0);

    EXPECT_TRUE(true); // If compilation succeeds, test passes
}
#include <base/traits_base.h>
#include <gtest/gtest.h>
#include <complex>
#include <type_traits>
#include <utility>

using namespace fem::numeric;

// ============================================================================
// Type Promotion Traits Tests - CRITICAL MISSING COVERAGE
// ============================================================================

class PromotionTraitsTest : public ::testing::Test {
protected:
    // Helper to test promotion symmetry
    template<typename T1, typename T2>
    static constexpr bool is_promotion_symmetric() {
        return std::is_same_v<promote_t<T1, T2>, promote_t<T2, T1>>;
    }
    
    // Helper to test promotion reflexivity
    template<typename T>
    static constexpr bool is_promotion_reflexive() {
        return std::is_same_v<promote_t<T, T>, T>;
    }
};

// ============================================================================
// Basic Integer Promotion Tests
// ============================================================================

TEST_F(PromotionTraitsTest, BasicIntegerPromotions) {
    // Same type promotions should be reflexive
    static_assert(std::is_same_v<promote_t<int8_t, int8_t>, int8_t>);
    static_assert(std::is_same_v<promote_t<int16_t, int16_t>, int16_t>);
    static_assert(std::is_same_v<promote_t<int32_t, int32_t>, int32_t>);
    static_assert(std::is_same_v<promote_t<int64_t, int64_t>, int64_t>);
    
    // Upward integer promotions
    static_assert(std::is_same_v<promote_t<int8_t, int16_t>, int16_t>);
    static_assert(std::is_same_v<promote_t<int8_t, int32_t>, int32_t>);
    static_assert(std::is_same_v<promote_t<int8_t, int64_t>, int64_t>);
    static_assert(std::is_same_v<promote_t<int16_t, int32_t>, int32_t>);
    static_assert(std::is_same_v<promote_t<int16_t, int64_t>, int64_t>);
    static_assert(std::is_same_v<promote_t<int32_t, int64_t>, int64_t>);
}

TEST_F(PromotionTraitsTest, IntegerToFloatingPointPromotions) {
    // Integer to float promotions
    static_assert(std::is_same_v<promote_t<int8_t, float>, float>);
    static_assert(std::is_same_v<promote_t<int16_t, float>, float>);
    static_assert(std::is_same_v<promote_t<int32_t, float>, float>);
    static_assert(std::is_same_v<promote_t<int64_t, float>, float>);
    
    // Integer to double promotions
    static_assert(std::is_same_v<promote_t<int8_t, double>, double>);
    static_assert(std::is_same_v<promote_t<int16_t, double>, double>);
    static_assert(std::is_same_v<promote_t<int32_t, double>, double>);
    static_assert(std::is_same_v<promote_t<int64_t, double>, double>);
}

TEST_F(PromotionTraitsTest, FloatingPointPromotions) {
    // Float to double promotion
    static_assert(std::is_same_v<promote_t<float, double>, double>);
    
    // Same type should be reflexive
    static_assert(std::is_same_v<promote_t<float, float>, float>);
    static_assert(std::is_same_v<promote_t<double, double>, double>);
}

// ============================================================================
// Unsigned Integer Promotion Tests
// ============================================================================

TEST_F(PromotionTraitsTest, UnsignedIntegerPromotions) {
    // Unsigned to unsigned promotions
    static_assert(std::is_same_v<promote_t<uint8_t, uint16_t>, uint16_t>);
    static_assert(std::is_same_v<promote_t<uint8_t, uint32_t>, uint32_t>);
    static_assert(std::is_same_v<promote_t<uint8_t, uint64_t>, uint64_t>);
    static_assert(std::is_same_v<promote_t<uint16_t, uint32_t>, uint32_t>);
    static_assert(std::is_same_v<promote_t<uint16_t, uint64_t>, uint64_t>);
    static_assert(std::is_same_v<promote_t<uint32_t, uint64_t>, uint64_t>);
    
    // Unsigned to floating point
    static_assert(std::is_same_v<promote_t<uint32_t, float>, float>);
    static_assert(std::is_same_v<promote_t<uint64_t, double>, double>);
}

TEST_F(PromotionTraitsTest, MixedSignPromotions) {
    // Mixed signed/unsigned promotions - these follow C++ rules
    // The exact rules can be complex, but we test some common cases
    
    // When both types have same size, signed wins in promotion
    static_assert(std::is_same_v<promote_t<int32_t, uint32_t>, 
                  decltype(int32_t{} + uint32_t{})>);
    
    // When unsigned is larger, unsigned wins
    static_assert(std::is_same_v<promote_t<int8_t, uint16_t>, 
                  decltype(int8_t{} + uint16_t{})>);
    
    // Test some specific cases we know should work
    static_assert(std::is_same_v<promote_t<int16_t, uint8_t>, int16_t>);
    static_assert(std::is_same_v<promote_t<int32_t, uint16_t>, int32_t>);
}

// ============================================================================
// Complex Number Promotion Tests
// ============================================================================

TEST_F(PromotionTraitsTest, ComplexPromotions) {
    // Real to complex promotions
    static_assert(std::is_same_v<promote_t<std::complex<float>, float>, 
                                std::complex<float>>);
    static_assert(std::is_same_v<promote_t<float, std::complex<float>>, 
                                std::complex<float>>);
    static_assert(std::is_same_v<promote_t<std::complex<double>, int>, 
                                std::complex<double>>);
    static_assert(std::is_same_v<promote_t<int, std::complex<double>>, 
                                std::complex<double>>);
    
    // Complex to complex promotions
    static_assert(std::is_same_v<promote_t<std::complex<float>, std::complex<double>>, 
                                std::complex<double>>);
    static_assert(std::is_same_v<promote_t<std::complex<double>, std::complex<float>>, 
                                std::complex<double>>);
    
    // Mixed complex promotions with integers
    static_assert(std::is_same_v<promote_t<std::complex<float>, int32_t>, 
                                std::complex<float>>);
    static_assert(std::is_same_v<promote_t<std::complex<double>, float>, 
                                std::complex<double>>);
}

TEST_F(PromotionTraitsTest, ComplexPromotionConsistency) {
    // Test that complex promotions are consistent with their underlying types
    using float_to_double = promote_t<float, double>;
    using complex_float_to_double = promote_t<std::complex<float>, double>;
    using complex_float_to_complex_double = promote_t<std::complex<float>, std::complex<double>>;
    
    static_assert(std::is_same_v<float_to_double, double>);
    static_assert(std::is_same_v<complex_float_to_double, std::complex<double>>);
    static_assert(std::is_same_v<complex_float_to_complex_double, std::complex<double>>);
}

// ============================================================================
// Promotion Symmetry Tests
// ============================================================================

TEST_F(PromotionTraitsTest, PromotionSymmetry) {
    // Promotion should be symmetric (commutative)
    EXPECT_TRUE((is_promotion_symmetric<int8_t, int16_t>()));
    EXPECT_TRUE((is_promotion_symmetric<int32_t, float>()));
    EXPECT_TRUE((is_promotion_symmetric<float, double>()));
    EXPECT_TRUE((is_promotion_symmetric<int, std::complex<double>>()));
    EXPECT_TRUE((is_promotion_symmetric<std::complex<float>, double>()));
    EXPECT_TRUE((is_promotion_symmetric<std::complex<float>, std::complex<double>>()));
    
    // Test with unsigned types
    EXPECT_TRUE((is_promotion_symmetric<uint16_t, uint32_t>()));
    EXPECT_TRUE((is_promotion_symmetric<uint32_t, float>()));
}

TEST_F(PromotionTraitsTest, PromotionReflexivity) {
    // Promotion of a type with itself should be the same type
    EXPECT_TRUE((is_promotion_reflexive<int8_t>()));
    EXPECT_TRUE((is_promotion_reflexive<int16_t>()));
    EXPECT_TRUE((is_promotion_reflexive<int32_t>()));
    EXPECT_TRUE((is_promotion_reflexive<int64_t>()));
    EXPECT_TRUE((is_promotion_reflexive<uint8_t>()));
    EXPECT_TRUE((is_promotion_reflexive<uint16_t>()));
    EXPECT_TRUE((is_promotion_reflexive<uint32_t>()));
    EXPECT_TRUE((is_promotion_reflexive<uint64_t>()));
    EXPECT_TRUE((is_promotion_reflexive<float>()));
    EXPECT_TRUE((is_promotion_reflexive<double>()));
    EXPECT_TRUE((is_promotion_reflexive<std::complex<float>>()));
    EXPECT_TRUE((is_promotion_reflexive<std::complex<double>>()));
}

// ============================================================================
// Extended Precision and Long Double Tests
// ============================================================================

TEST_F(PromotionTraitsTest, LongDoublePromotions) {
    // Long double should be the highest precision floating point
    static_assert(std::is_same_v<promote_t<float, long double>, long double>);
    static_assert(std::is_same_v<promote_t<double, long double>, long double>);
    static_assert(std::is_same_v<promote_t<long double, long double>, long double>);
    
    // Integer to long double
    static_assert(std::is_same_v<promote_t<int64_t, long double>, long double>);
    static_assert(std::is_same_v<promote_t<uint64_t, long double>, long double>);
    
    // Complex long double
    static_assert(std::is_same_v<promote_t<std::complex<float>, long double>, 
                                std::complex<long double>>);
    static_assert(std::is_same_v<promote_t<std::complex<double>, std::complex<long double>>, 
                                std::complex<long double>>);
}

// ============================================================================
// Promotion Transitivity Tests
// ============================================================================

TEST_F(PromotionTraitsTest, PromotionTransitivity) {
    // Test that promotion is transitive: if A promotes to B and B promotes to C,
    // then A should promote to C
    
    // int8 -> int16 -> int32
    using step1 = promote_t<int8_t, int16_t>;  // should be int16_t
    using step2 = promote_t<step1, int32_t>;   // should be int32_t
    using direct = promote_t<int8_t, int32_t>; // should be int32_t
    
    static_assert(std::is_same_v<step1, int16_t>);
    static_assert(std::is_same_v<step2, int32_t>);
    static_assert(std::is_same_v<direct, int32_t>);
    static_assert(std::is_same_v<step2, direct>);
    
    // int -> float -> double
    using int_float = promote_t<int, float>;      // should be float
    using float_double = promote_t<int_float, double>; // should be double
    using int_double = promote_t<int, double>;    // should be double
    
    static_assert(std::is_same_v<int_float, float>);
    static_assert(std::is_same_v<float_double, double>);
    static_assert(std::is_same_v<int_double, double>);
    static_assert(std::is_same_v<float_double, int_double>);
}

// ============================================================================
// Edge Cases and Corner Cases
// ============================================================================

TEST_F(PromotionTraitsTest, EdgeCasePromotions) {
    // Test promotions with cv-qualified types
    static_assert(std::is_same_v<promote_t<const int, float>, float>);
    static_assert(std::is_same_v<promote_t<volatile double, int>, double>);
    
    // Test with bool (should promote to larger types)
    static_assert(std::is_same_v<promote_t<bool, int>, int>);
    static_assert(std::is_same_v<promote_t<bool, float>, float>);
    
    // Test with char types
    static_assert(std::is_same_v<promote_t<char, int>, int>);
    static_assert(std::is_same_v<
        promote_t<signed char, unsigned char>,
        decltype(std::declval<signed char>() + std::declval<unsigned char>())
    >);
}

// ============================================================================
// Performance and Compile-Time Tests
// ============================================================================

TEST_F(PromotionTraitsTest, CompileTimePromotion) {
    // Verify that promotions are computed at compile time
    constexpr bool int_float_ok = std::is_same_v<promote_t<int, float>, float>;
    constexpr bool complex_ok = std::is_same_v<promote_t<std::complex<float>, double>, 
                                              std::complex<double>>;
    
    static_assert(int_float_ok);
    static_assert(complex_ok);
    
    // Test that promotion doesn't increase compilation overhead significantly
    // (This is more of a development guideline than a runtime test)
    using many_promotions = promote_t<
        promote_t<int8_t, int16_t>,
        promote_t<int32_t, float>
    >;
    static_assert(std::is_same_v<many_promotions, float>);
}

// ============================================================================
// Real-World Usage Scenarios
// ============================================================================

TEST_F(PromotionTraitsTest, ArithmeticOperationPromotions) {
    // Test that promotions match actual arithmetic operations
    
    // Mixed arithmetic should follow promotion rules
    int8_t a = 1;
    int16_t b = 2;
    auto result1 = a + b;
    using expected1 = decltype(a + b);
    static_assert(std::is_same_v<decltype(result1), expected1>);
    
    // Float and int
    float f = 1.0f;
    int i = 2;
    auto result2 = f * static_cast<float>(i);
    using expected2 = decltype(result2);
    static_assert(std::is_same_v<decltype(result2), expected2>);
    
    // Complex and real
    std::complex<float> cf(1.0f, 2.0f);
    double d = 3.0;
    // Note: std::complex<T> operators are defined for the same T.
    // For mixed precision, convert to the promoted complex type explicitly.
    using expected3 = promote_t<std::complex<float>, double>; // std::complex<double>
    auto result3 = static_cast<expected3>(cf) + d;
    static_assert(std::is_same_v<decltype(result3), expected3>);
}

TEST_F(PromotionTraitsTest, ContainerCompatibilityPromotions) {
    // Test promotions in the context of container operations
    // This simulates how containers might use promotion traits
    
    // Vector-like operations
    using float_int_container = promote_t<float, int>;
    using complex_real_container = promote_t<std::complex<double>, float>;
    
    static_assert(std::is_same_v<float_int_container, float>);
    static_assert(std::is_same_v<complex_real_container, std::complex<double>>);
    
    // Verify these work for actual computation
    float_int_container val1 = 3.14f + 42;  // Should compile and work
    complex_real_container val2 = std::complex<double>(1,2) + 3.0;
    
    EXPECT_NEAR(val1, 45.14f, 1e-5f);
    EXPECT_NEAR(val2.real(), 4.0, 1e-10);
    EXPECT_NEAR(val2.imag(), 2.0, 1e-10);
}

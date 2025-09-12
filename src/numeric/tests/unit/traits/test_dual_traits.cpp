#include <base/traits_base.h>
#include <base/dual_base.h>
#include <gtest/gtest.h>

using namespace fem::numeric;

// ============================================================================
// Dual Number Traits Tests - CRITICAL MISSING COVERAGE
// ============================================================================

class DualTraitsTest : public ::testing::Test {
protected:
    using Dual1 = DualBase<double, 1>;
    using Dual2 = DualBase<double, 2>;  
    using Dual3 = DualBase<float, 3>;
};

TEST_F(DualTraitsTest, DualNumberDetection) {
    // Basic dual number detection
    EXPECT_TRUE((is_dual_number_v<Dual1>));
    EXPECT_TRUE((is_dual_number_v<Dual2>));
    EXPECT_TRUE((is_dual_number_v<Dual3>));
    
    // Non-dual types should be false
    EXPECT_FALSE(is_dual_number_v<double>);
    EXPECT_FALSE(is_dual_number_v<float>);
    EXPECT_FALSE(is_dual_number_v<int>);
    EXPECT_FALSE((is_dual_number_v<std::complex<double>>));
    
    // Test with cv-qualifiers
    EXPECT_TRUE((is_dual_number_v<const Dual1>));
    EXPECT_TRUE((is_dual_number_v<volatile Dual2>));
    EXPECT_TRUE((is_dual_number_v<const volatile Dual3>));
}

TEST_F(DualTraitsTest, StorableTypeConceptWithDuals) {
    // Dual numbers should satisfy StorableType
    EXPECT_TRUE(StorableType<Dual1>);
    EXPECT_TRUE(StorableType<Dual2>);
    EXPECT_TRUE(StorableType<Dual3>);
    
    // Basic numeric types should also satisfy
    EXPECT_TRUE(StorableType<int>);
    EXPECT_TRUE(StorableType<double>);
    EXPECT_TRUE((StorableType<std::complex<double>>));
    
    // Non-numeric types should not satisfy
    EXPECT_FALSE(StorableType<std::string>);
    EXPECT_FALSE(StorableType<void*>);
}

TEST_F(DualTraitsTest, ScalarTypeExtraction) {
    // Scalar type extraction for dual numbers
    static_assert(std::is_same_v<scalar_type_t<Dual1>, double>);
    static_assert(std::is_same_v<scalar_type_t<Dual2>, double>);
    static_assert(std::is_same_v<scalar_type_t<Dual3>, float>);
    
    // Compare with regular types
    static_assert(std::is_same_v<scalar_type_t<double>, double>);
    static_assert(std::is_same_v<scalar_type_t<std::complex<float>>, float>);
}

TEST_F(DualTraitsTest, DualNumericTraits) {
    using traits = numeric_traits<Dual2>;
    
    // Type information
    EXPECT_FALSE(traits::is_floating_point);
    EXPECT_FALSE(traits::is_integral);
    EXPECT_TRUE(traits::is_signed);
    EXPECT_FALSE(traits::is_complex);
    EXPECT_TRUE(traits::is_dual);
    EXPECT_EQ(traits::num_derivatives, 2);
    
    // IEEE characteristics from underlying type
    EXPECT_TRUE(traits::has_infinity);
    EXPECT_TRUE(traits::has_quiet_nan);
    EXPECT_FALSE(traits::has_signaling_nan);
    
    // Size and alignment
    EXPECT_EQ(traits::size, sizeof(Dual2));
    EXPECT_EQ(traits::alignment, alignof(Dual2));
}

TEST_F(DualTraitsTest, DualNumericTraitsFactoryMethods) {
    using traits = numeric_traits<Dual2>;
    
    // Basic constants
    auto zero = traits::zero();
    EXPECT_EQ(zero.value(), 0.0);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(zero.derivative(i), 0.0);
    }
    
    auto one = traits::one();
    EXPECT_EQ(one.value(), 1.0);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(one.derivative(i), 0.0);
    }
    
    // Independent variable creation
    auto x = traits::make_independent(3.14, 0);
    EXPECT_EQ(x.value(), 3.14);
    EXPECT_EQ(x.derivative(0), 1.0);
    EXPECT_EQ(x.derivative(1), 0.0);
    
    auto y = traits::make_independent(2.71, 1);
    EXPECT_EQ(y.value(), 2.71);
    EXPECT_EQ(y.derivative(0), 0.0);
    EXPECT_EQ(y.derivative(1), 1.0);
}

TEST_F(DualTraitsTest, DualStorageOptimizationTraits) {
    using traits = storage_optimization_traits<Dual2>;
    
    // Dual numbers are not trivially relocatable (have complex structure)
    EXPECT_FALSE(traits::is_trivially_relocatable);
    
    // Dual numbers don't support SIMD (composite type)
    EXPECT_FALSE(traits::supports_simd);
    
    // Dual numbers prefer alignment (composite structure)
    EXPECT_TRUE(traits::prefers_alignment);
    
    // Dual numbers don't support fast fill (not trivially copyable)
    EXPECT_FALSE(traits::supports_fast_fill);
}

TEST_F(DualTraitsTest, DualSIMDTraits) {
    using traits = simd_traits<Dual2>;
    
    // Dual numbers are not vectorizable
    EXPECT_FALSE(traits::is_vectorizable);
    EXPECT_EQ(traits::vector_size, 1);
    EXPECT_EQ(traits::alignment, alignof(Dual2));
}

// ============================================================================
// Integration Tests with Actual Dual Operations
// ============================================================================

TEST_F(DualTraitsTest, DualIntegrationTest) {
    using traits = numeric_traits<Dual2>;
    
    // Create independent variables
    auto x = traits::make_independent(2.0, 0);
    auto y = traits::make_independent(3.0, 1);
    
    // Compute f(x,y) = x^2 + y^2
    auto f = x * x + y * y;
    
    // Verify function value
    EXPECT_EQ(f.value(), 13.0); // 2^2 + 3^2 = 13
    
    // Verify partial derivatives
    EXPECT_EQ(f.derivative(0), 4.0); // df/dx = 2x = 4
    EXPECT_EQ(f.derivative(1), 6.0); // df/dy = 2y = 6
}

TEST_F(DualTraitsTest, DualTypeCompatibility) {
    // Test that dual numbers work with container traits
    using traits = storage_optimization_traits<Dual2>;
    
    // Should be compatible with containers
    EXPECT_TRUE(StorableType<Dual2>);
    
    // Should have consistent scalar type
    static_assert(std::is_same_v<scalar_type_t<Dual2>, double>);
    
    // Should work with numeric traits
    using numeric = numeric_traits<Dual2>;
    EXPECT_TRUE(numeric::is_dual);
    EXPECT_EQ(numeric::num_derivatives, 2);
}
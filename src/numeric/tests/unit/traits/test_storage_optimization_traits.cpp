#include <base/traits_base.h>
#include <base/dual_base.h>
#include <gtest/gtest.h>
#include <complex>

using namespace fem::numeric;

// ============================================================================
// Storage Optimization Traits Tests - CRITICAL MISSING COVERAGE
// ============================================================================

class StorageOptimizationTraitsTest : public ::testing::Test {
protected:
    // Test types
    struct LargeType {
        double data[5];  // 40 bytes - larger than 16
    };
    
    struct SmallType {
        float x, y;      // 8 bytes - smaller than 16
    };
    
    struct NonTrivialType {
        std::unique_ptr<int> ptr;
        NonTrivialType() : ptr(std::make_unique<int>(42)) {}
    };
};

// ============================================================================
// Basic Type Storage Optimization Tests
// ============================================================================

TEST_F(StorageOptimizationTraitsTest, TriviallyRelocatableTraits) {
    // Basic arithmetic types should be trivially relocatable
    EXPECT_TRUE((storage_optimization_traits<int>::is_trivially_relocatable));
    EXPECT_TRUE((storage_optimization_traits<float>::is_trivially_relocatable));
    EXPECT_TRUE((storage_optimization_traits<double>::is_trivially_relocatable));
    
    // Complex numbers should be trivially relocatable
    EXPECT_TRUE((storage_optimization_traits<std::complex<float>>::is_trivially_relocatable));
    EXPECT_TRUE((storage_optimization_traits<std::complex<double>>::is_trivially_relocatable));
    
    // Dual numbers should NOT be trivially relocatable (complex internal structure)
    EXPECT_FALSE((storage_optimization_traits<DualBase<double, 1>>::is_trivially_relocatable));
    EXPECT_FALSE((storage_optimization_traits<DualBase<float, 2>>::is_trivially_relocatable));
    
    // Non-trivial types should not be trivially relocatable
    EXPECT_FALSE((storage_optimization_traits<NonTrivialType>::is_trivially_relocatable));
}

TEST_F(StorageOptimizationTraitsTest, SIMDSupportTraits) {
    // Basic arithmetic types should support SIMD
    EXPECT_TRUE((storage_optimization_traits<int>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<float>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<double>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<int8_t>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<int16_t>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<int32_t>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<int64_t>::supports_simd));
    
    // Complex numbers are arithmetic but composite, may or may not support SIMD
    // This depends on implementation - currently should be true as they're arithmetic
    EXPECT_TRUE((storage_optimization_traits<std::complex<float>>::supports_simd));
    EXPECT_TRUE((storage_optimization_traits<std::complex<double>>::supports_simd));
    
    // Dual numbers should NOT support SIMD (complex internal structure)
    EXPECT_FALSE((storage_optimization_traits<DualBase<double, 1>>::supports_simd));
    EXPECT_FALSE((storage_optimization_traits<DualBase<float, 2>>::supports_simd));
    
    // Non-arithmetic types should not support SIMD
    EXPECT_FALSE((storage_optimization_traits<std::string>::supports_simd));
    EXPECT_FALSE((storage_optimization_traits<NonTrivialType>::supports_simd));
}

TEST_F(StorageOptimizationTraitsTest, AlignmentPreferenceTraits) {
    // Small types (< 16 bytes) should not prefer alignment
    EXPECT_FALSE((storage_optimization_traits<int>::prefers_alignment));
    EXPECT_FALSE((storage_optimization_traits<float>::prefers_alignment));
    EXPECT_FALSE((storage_optimization_traits<double>::prefers_alignment));  // 8 bytes
    EXPECT_FALSE((storage_optimization_traits<std::complex<float>>::prefers_alignment)); // 8 bytes
    
    // Large types (>= 16 bytes) should prefer alignment
    EXPECT_TRUE((storage_optimization_traits<std::complex<double>>::prefers_alignment)); // 16 bytes
    EXPECT_TRUE((storage_optimization_traits<LargeType>::prefers_alignment)); // 40 bytes
    
    // Dual numbers should prefer alignment (complex structure)
    EXPECT_TRUE((storage_optimization_traits<DualBase<double, 1>>::prefers_alignment));
    EXPECT_TRUE((storage_optimization_traits<DualBase<float, 2>>::prefers_alignment));
    EXPECT_TRUE((storage_optimization_traits<DualBase<double, 3>>::prefers_alignment));
}

TEST_F(StorageOptimizationTraitsTest, FastFillTraits) {
    // Small trivially copyable types should support fast fill
    EXPECT_TRUE((storage_optimization_traits<int>::supports_fast_fill));
    EXPECT_TRUE((storage_optimization_traits<float>::supports_fast_fill));
    EXPECT_TRUE((storage_optimization_traits<double>::supports_fast_fill));
    EXPECT_TRUE((storage_optimization_traits<SmallType>::supports_fast_fill));
    
    // Complex numbers <= 16 bytes should support fast fill
    EXPECT_TRUE((storage_optimization_traits<std::complex<float>>::supports_fast_fill)); // 8 bytes
    EXPECT_TRUE((storage_optimization_traits<std::complex<double>>::supports_fast_fill)); // 16 bytes
    
    // Large types (> 16 bytes) should NOT support fast fill
    EXPECT_FALSE((storage_optimization_traits<LargeType>::supports_fast_fill)); // 40 bytes
    
    // Dual numbers should NOT support fast fill (not trivially copyable or > size limit)
    EXPECT_FALSE((storage_optimization_traits<DualBase<double, 1>>::supports_fast_fill));
    EXPECT_FALSE((storage_optimization_traits<DualBase<float, 2>>::supports_fast_fill));
    
    // Non-trivially copyable types should not support fast fill
    EXPECT_FALSE((storage_optimization_traits<NonTrivialType>::supports_fast_fill));
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

TEST_F(StorageOptimizationTraitsTest, SizeBoundaryConditions) {
    // Test exactly 16-byte types
    struct Exactly16Bytes {
        double a, b; // 16 bytes exactly
    };
    
    using traits16 = storage_optimization_traits<Exactly16Bytes>;
    EXPECT_TRUE(traits16::prefers_alignment);    // >= 16 bytes
    EXPECT_TRUE(traits16::supports_fast_fill);   // <= 16 bytes and trivially copyable
    
    // Test exactly 17-byte types  
    struct Exactly17Bytes {
        double a, b;  // 16 bytes
        char c;       // +1 byte = 17 bytes (+ padding)
    };
    
    using traits17 = storage_optimization_traits<Exactly17Bytes>;
    EXPECT_TRUE(traits17::prefers_alignment);     // >= 16 bytes  
    EXPECT_FALSE(traits17::supports_fast_fill);   // > 16 bytes
}

TEST_F(StorageOptimizationTraitsTest, CVQualifierHandling) {
    // CV-qualifiers should not affect storage traits
    EXPECT_EQ((storage_optimization_traits<const double>::is_trivially_relocatable),
              (storage_optimization_traits<double>::is_trivially_relocatable));
    
    EXPECT_EQ((storage_optimization_traits<volatile float>::supports_simd),
              (storage_optimization_traits<float>::supports_simd));
    
    EXPECT_EQ((storage_optimization_traits<const volatile int>::supports_fast_fill),
              (storage_optimization_traits<int>::supports_fast_fill));
    
    EXPECT_EQ((storage_optimization_traits<const DualBase<double,1>>::prefers_alignment),
              (storage_optimization_traits<DualBase<double,1>>::prefers_alignment));
}

// ============================================================================
// SIMD Traits Detailed Testing
// ============================================================================

TEST_F(StorageOptimizationTraitsTest, SIMDVectorSizeCalculation) {
    // Test SIMD vector size calculations for different types
    using float_simd = simd_traits<float>;
    using double_simd = simd_traits<double>;
    using int32_simd = simd_traits<int32_t>;
    using int16_simd = simd_traits<int16_t>;
    using int8_simd = simd_traits<int8_t>;
    
    // Vectorizable types should have reasonable vector sizes
    if constexpr (float_simd::is_vectorizable) {
        EXPECT_GT(float_simd::vector_size, 1);
        EXPECT_LE(float_simd::vector_size, 16);  // Reasonable upper bound
        EXPECT_EQ(float_simd::vector_size * sizeof(float), float_simd::alignment);
    }
    
    if constexpr (double_simd::is_vectorizable) {
        EXPECT_GT(double_simd::vector_size, 1);
        EXPECT_LE(double_simd::vector_size, 8);   // Reasonable upper bound  
        EXPECT_EQ(double_simd::vector_size * sizeof(double), double_simd::alignment);
    }
    
    // Non-vectorizable types should have vector_size = 1
    using dual_simd = simd_traits<DualBase<double,1>>;
    using bool_simd = simd_traits<bool>;
    using string_simd = simd_traits<std::string>;
    
    EXPECT_FALSE(dual_simd::is_vectorizable);
    EXPECT_EQ(dual_simd::vector_size, 1);
    
    EXPECT_FALSE(bool_simd::is_vectorizable);
    EXPECT_EQ(bool_simd::vector_size, 1);
    
    EXPECT_FALSE(string_simd::is_vectorizable);
    EXPECT_EQ(string_simd::vector_size, 1);
}

TEST_F(StorageOptimizationTraitsTest, SIMDAlignmentRequirements) {
    // SIMD alignment should be consistent with vector size
    using float_traits = simd_traits<float>;
    using double_traits = simd_traits<double>;
    
    if constexpr (float_traits::is_vectorizable) {
        EXPECT_EQ(float_traits::alignment, float_traits::vector_size * sizeof(float));
        EXPECT_GE(float_traits::alignment, alignof(float));
    } else {
        EXPECT_EQ(float_traits::alignment, alignof(float));
    }
    
    if constexpr (double_traits::is_vectorizable) {
        EXPECT_EQ(double_traits::alignment, double_traits::vector_size * sizeof(double));
        EXPECT_GE(double_traits::alignment, alignof(double));
    } else {
        EXPECT_EQ(double_traits::alignment, alignof(double));
    }
}

// ============================================================================
// Integration with Real Usage Scenarios
// ============================================================================

TEST_F(StorageOptimizationTraitsTest, ContainerOptimizationDecisions) {
    // Simulate container optimization decisions based on traits
    
    template<typename T>
    constexpr bool should_use_memcpy() {
        return storage_optimization_traits<T>::is_trivially_relocatable;
    }
    
    template<typename T> 
    constexpr bool should_use_simd() {
        return storage_optimization_traits<T>::supports_simd;
    }
    
    template<typename T>
    constexpr bool should_align_allocation() {
        return storage_optimization_traits<T>::prefers_alignment;
    }
    
    template<typename T>
    constexpr bool should_use_fast_fill() {
        return storage_optimization_traits<T>::supports_fast_fill;
    }
    
    // Test optimization decisions for different types
    
    // float: should use all optimizations except alignment
    EXPECT_TRUE(should_use_memcpy<float>());
    EXPECT_TRUE(should_use_simd<float>());
    EXPECT_FALSE(should_align_allocation<float>());
    EXPECT_TRUE(should_use_fast_fill<float>());
    
    // std::complex<double>: alignment needed due to size
    EXPECT_TRUE(should_use_memcpy<std::complex<double>>());
    EXPECT_TRUE(should_use_simd<std::complex<double>>());
    EXPECT_TRUE(should_align_allocation<std::complex<double>>());
    EXPECT_TRUE(should_use_fast_fill<std::complex<double>>());
    
    // DualBase: should avoid most optimizations
    using Dual = DualBase<double, 2>;
    EXPECT_FALSE(should_use_memcpy<Dual>());
    EXPECT_FALSE(should_use_simd<Dual>());
    EXPECT_TRUE(should_align_allocation<Dual>());
    EXPECT_FALSE(should_use_fast_fill<Dual>());
}

TEST_F(StorageOptimizationTraitsTest, MemoryLayoutConsistency) {
    // Verify that traits are consistent with actual memory layout
    
    // Small types should not need special alignment
    EXPECT_LE(alignof(float), 8);
    EXPECT_FALSE((storage_optimization_traits<float>::prefers_alignment));
    
    // Large types should need alignment
    EXPECT_GE(alignof(LargeType), 8);
    EXPECT_TRUE((storage_optimization_traits<LargeType>::prefers_alignment));
    
    // Complex double is exactly 16 bytes
    EXPECT_EQ(sizeof(std::complex<double>), 16);
    EXPECT_TRUE((storage_optimization_traits<std::complex<double>>::prefers_alignment));
}
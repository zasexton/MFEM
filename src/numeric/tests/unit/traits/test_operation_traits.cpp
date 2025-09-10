#include <gtest/gtest.h>
#include <type_traits>
#include <functional>
#include <complex>
#include <limits>

// Include the actual headers with implementations
#include <base/ops_base.h>
#include <base/numeric_base.h>
#include <base/traits_base.h>

#include <traits/type_traits.h>
#include <traits/numeric_traits.h>
#include <traits/operation_traits.h>

// Test fixtures
namespace {

using namespace fem::numeric;
using namespace fem::numeric::traits;

// Custom operation for testing
struct CustomOp {
    double operator()(double x) const { return x * 2.0; }
};

// Non-callable type for negative testing
struct NotAnOperation {};

// Test operation category detection
TEST(OperationTraitsTest, OperationCategory) {
    // Arithmetic
    EXPECT_EQ(operation_category_v<ops::plus<>>, OperationCategory::Arithmetic);
    EXPECT_EQ(operation_category_v<ops::minus<>>, OperationCategory::Arithmetic);
    EXPECT_EQ(operation_category_v<ops::multiplies<>>, OperationCategory::Arithmetic);
    EXPECT_EQ(operation_category_v<ops::divides<>>, OperationCategory::Arithmetic);
    EXPECT_EQ(operation_category_v<ops::modulus<>>, OperationCategory::Arithmetic);
    EXPECT_EQ(operation_category_v<ops::negate<>>, OperationCategory::Arithmetic);

    // Comparison
    EXPECT_EQ(operation_category_v<ops::less<>>, OperationCategory::Comparison);
    EXPECT_EQ(operation_category_v<ops::greater<>>, OperationCategory::Comparison);
    EXPECT_EQ(operation_category_v<ops::equal_to<>>, OperationCategory::Comparison);

    // Transcendental
    EXPECT_EQ(operation_category_v<ops::sin_op<>>, OperationCategory::Transcendental);
    EXPECT_EQ(operation_category_v<ops::cos_op<>>, OperationCategory::Transcendental);
    EXPECT_EQ(operation_category_v<ops::exp_op<>>, OperationCategory::Transcendental);
    EXPECT_EQ(operation_category_v<ops::log_op<>>, OperationCategory::Transcendental);
    EXPECT_EQ(operation_category_v<ops::sqrt_op<>>, OperationCategory::Transcendental);
    EXPECT_EQ((operation_category_v<ops::power_op<double>>), OperationCategory::Transcendental);

    // Unknown
    EXPECT_EQ(operation_category_v<CustomOp>, OperationCategory::Unknown);
}

// Test in-place operation detection
TEST(OperationTraitsTest, InplaceOperations) {
    EXPECT_TRUE(is_inplace_operation_v<ops::plus_assign<>>);
    EXPECT_TRUE(is_inplace_operation_v<ops::minus_assign<>>);
    EXPECT_TRUE(is_inplace_operation_v<ops::multiplies_assign<>>);
    EXPECT_TRUE(is_inplace_operation_v<ops::divides_assign<>>);
    EXPECT_TRUE(is_inplace_operation_v<ops::modulus_assign<>>);

    EXPECT_FALSE(is_inplace_operation_v<ops::plus<>>);
    EXPECT_FALSE(is_inplace_operation_v<ops::minus<>>);
    EXPECT_FALSE(is_inplace_operation_v<ops::sin_op<>>);
    EXPECT_FALSE(is_inplace_operation_v<CustomOp>);
}

// Test element-wise operation detection
TEST(OperationTraitsTest, ElementwiseOperations) {
    // Arithmetic operations are element-wise
    EXPECT_TRUE(is_elementwise_operation_v<ops::plus<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::minus<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::multiplies<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::divides<>>);

    // Transcendental operations are element-wise
    EXPECT_TRUE(is_elementwise_operation_v<ops::sin_op<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::exp_op<>>);

    // Comparison operations are element-wise
    EXPECT_TRUE(is_elementwise_operation_v<ops::less<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::equal_to<>>);

    // Special operations
    EXPECT_TRUE(is_elementwise_operation_v<ops::abs_op<>>);
    EXPECT_TRUE(is_elementwise_operation_v<ops::sign_op<>>);
}

// Test domain restrictions
TEST(OperationTraitsTest, DomainRestrictions) {
    EXPECT_TRUE(has_domain_restrictions_v<ops::sqrt_op<>>);  // x >= 0
    EXPECT_TRUE(has_domain_restrictions_v<ops::log_op<>>);   // x > 0
    EXPECT_TRUE(has_domain_restrictions_v<ops::asin_op<>>);  // |x| <= 1
    EXPECT_TRUE(has_domain_restrictions_v<ops::acos_op<>>);  // |x| <= 1
    // Note: tanh_op instead of atanh_op if that's what's available
    EXPECT_TRUE(has_domain_restrictions_v<ops::tanh_op<>>);

    EXPECT_FALSE(has_domain_restrictions_v<ops::plus<>>);
    EXPECT_FALSE(has_domain_restrictions_v<ops::sin_op<>>);
    EXPECT_FALSE(has_domain_restrictions_v<ops::exp_op<>>);
}

// Test make_inplace transformation
TEST(OperationTraitsTest, MakeInplace) {
    EXPECT_TRUE((std::is_same_v<make_inplace_t<ops::plus<>>, ops::plus_assign<>>));
    EXPECT_TRUE((std::is_same_v<make_inplace_t<ops::minus<>>, ops::minus_assign<>>));
    EXPECT_TRUE((std::is_same_v<make_inplace_t<ops::multiplies<>>, ops::multiplies_assign<>>));
    EXPECT_TRUE((std::is_same_v<make_inplace_t<ops::divides<>>, ops::divides_assign<>>));

    // Operations without in-place versions
    EXPECT_TRUE((std::is_same_v<make_inplace_t<ops::sin_op<>>, void>));
}

// Test inverse operations
TEST(OperationTraitsTest, InverseOperations) {
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::plus<>>, ops::minus<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::minus<>>, ops::plus<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::multiplies<>>, ops::divides<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::divides<>>, ops::multiplies<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::exp_op<>>, ops::log_op<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::log_op<>>, ops::exp_op<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::sin_op<>>, ops::asin_op<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::cos_op<>>, ops::acos_op<>>));
    EXPECT_TRUE((std::is_same_v<inverse_operation_t<ops::tan_op<>>, ops::atan_op<>>));
}

// Test unary/binary operation detection - use extra parentheses for template with comma
TEST(OperationTraitsTest, UnaryBinaryDetection) {
    // Unary operations
    EXPECT_TRUE((is_unary_operation_v<ops::negate<double>, double>));
    EXPECT_TRUE((is_unary_operation_v<ops::sin_op<double>, double>));
    EXPECT_TRUE((is_unary_operation_v<ops::exp_op<double>, double>));
    EXPECT_TRUE((is_unary_operation_v<ops::abs_op<double>, double>));

    EXPECT_FALSE((is_unary_operation_v<ops::plus<double>, double>));
    EXPECT_FALSE((is_unary_operation_v<ops::power_op<double>, double>));

    // Binary operations
    EXPECT_TRUE((is_binary_operation_v<ops::plus<double>, double>));
    EXPECT_TRUE((is_binary_operation_v<ops::minus<double>, double>));
    EXPECT_TRUE((is_binary_operation_v<ops::power_op<double>, double>));
    EXPECT_TRUE((is_binary_operation_v<ops::less<double>, double>));

    EXPECT_FALSE((is_binary_operation_v<ops::sin_op<double>, double>));
    EXPECT_FALSE((is_binary_operation_v<NotAnOperation, double>));
}

// Test operation result types
TEST(OperationTraitsTest, OperationResult) {
    // Arithmetic operations preserve type
    EXPECT_TRUE((std::is_same_v<operation_result_t<ops::plus<>, double, double>, double>));
    EXPECT_TRUE((std::is_same_v<operation_result_t<ops::sin_op<double>, double>, double>));

    // Comparison operations return bool
    EXPECT_TRUE((std::is_same_v<operation_result_t<ops::less<>, double, double>, bool>));
    EXPECT_TRUE((std::is_same_v<operation_result_t<ops::equal_to<>, int, int>, bool>));
}

// Test valid operation checking
TEST(OperationTraitsTest, IsValidOperation) {
    EXPECT_TRUE((is_valid_operation_v<ops::plus<>, double, double>));
    EXPECT_TRUE((is_valid_operation_v<ops::sin_op<double>, double>));
    EXPECT_TRUE((is_valid_operation_v<ops::power_op<double>, double, double>));

    EXPECT_FALSE((is_valid_operation_v<ops::plus<>, double>));  // Binary op, one arg
    EXPECT_FALSE((is_valid_operation_v<ops::sin_op<double>, double, double>));  // Unary op, two args
    EXPECT_FALSE((is_valid_operation_v<NotAnOperation, double>));
}

// Test algebraic properties
TEST(OperationTraitsTest, AlgebraicProperties) {
    // Addition properties
    auto plus_props = algebraic_properties<ops::plus<>>::value;
    EXPECT_TRUE(plus_props.commutative);
    EXPECT_TRUE(plus_props.associative);
    EXPECT_TRUE(plus_props.has_identity);
    EXPECT_TRUE(plus_props.has_inverse);
    EXPECT_FALSE(plus_props.idempotent);

    // Subtraction properties
    auto minus_props = algebraic_properties<ops::minus<>>::value;
    EXPECT_FALSE(minus_props.commutative);
    EXPECT_FALSE(minus_props.associative);
    EXPECT_TRUE(minus_props.has_identity);
    EXPECT_FALSE(minus_props.has_inverse);

    // Multiplication properties
    auto mult_props = algebraic_properties<ops::multiplies<>>::value;
    EXPECT_TRUE(mult_props.commutative);
    EXPECT_TRUE(mult_props.associative);
    EXPECT_TRUE(mult_props.distributive);
    EXPECT_TRUE(mult_props.has_identity);
    EXPECT_TRUE(mult_props.has_inverse);

    // Min/Max properties
    auto min_props = algebraic_properties<ops::min_op<double>>::value;
    EXPECT_TRUE(min_props.commutative);
    EXPECT_TRUE(min_props.associative);
    EXPECT_TRUE(min_props.idempotent);

    // Logical operations
    auto and_props = algebraic_properties<ops::logical_and>::value;
    EXPECT_TRUE(and_props.commutative);
    EXPECT_TRUE(and_props.associative);
    EXPECT_TRUE(and_props.idempotent);
    EXPECT_TRUE(and_props.has_identity);

    // Bitwise XOR
    auto xor_props = algebraic_properties<std::bit_xor<>>::value;
    EXPECT_TRUE(xor_props.commutative);
    EXPECT_TRUE(xor_props.associative);
    EXPECT_FALSE(xor_props.idempotent);
    EXPECT_TRUE(xor_props.has_identity);
    EXPECT_TRUE(xor_props.has_inverse);
}

// Test type preservation
TEST(OperationTraitsTest, TypePreservation) {
    EXPECT_TRUE((preserves_type_v<ops::plus<>, double>));
    EXPECT_TRUE((preserves_type_v<ops::sin_op<double>, double>));
    EXPECT_TRUE((preserves_type_v<ops::negate<double>, double>));

    EXPECT_FALSE((preserves_type_v<ops::less<>, double>));  // Returns bool
    EXPECT_FALSE((preserves_type_v<ops::equal_to<>, int>));  // Returns bool
}

// Test IEEE safety
TEST(OperationTraitsTest, IEEESafety) {
    // Arithmetic and transcendental operations are IEEE-safe
    EXPECT_TRUE((is_ieee_safe_v<ops::plus<>, double>));
    EXPECT_TRUE((is_ieee_safe_v<ops::divides<>, double>));
    EXPECT_TRUE((is_ieee_safe_v<ops::sin_op<>, double>));
    EXPECT_TRUE((is_ieee_safe_v<ops::sqrt_op<>, double>));

    // Most comparisons are not NaN-safe (except !=)
    EXPECT_FALSE((is_ieee_safe_v<ops::less<>, double>));
    EXPECT_FALSE((is_ieee_safe_v<ops::equal_to<>, double>));
    EXPECT_TRUE((is_ieee_safe_v<ops::not_equal_to<>, double>));

    // Non-IEEE types are always "safe"
    EXPECT_TRUE((is_ieee_safe_v<ops::plus<>, int>));
}

// Test overflow detection
TEST(OperationTraitsTest, OverflowDetection) {
    // Floating point operations that can overflow to infinity
    EXPECT_TRUE((can_overflow_v<ops::plus<>, double>));
    EXPECT_TRUE((can_overflow_v<ops::multiplies<>, double>));
    EXPECT_TRUE((can_overflow_v<ops::exp_op<>, double>));
    EXPECT_TRUE((can_overflow_v<ops::power_op<double>, double>));

    // Integer operations that can overflow (UB)
    EXPECT_TRUE((can_overflow_v<ops::plus<>, int>));
    EXPECT_TRUE((can_overflow_v<ops::minus<>, int>));
    EXPECT_TRUE((can_overflow_v<ops::multiplies<>, int>));
    EXPECT_TRUE((can_overflow_v<ops::negate<>, int>));

    // Operations that don't overflow
    EXPECT_FALSE((can_overflow_v<ops::less<>, double>));
    EXPECT_FALSE((can_overflow_v<ops::sin_op<>, double>));  // Result always in [-1, 1]
}

// Test NaN production
TEST(OperationTraitsTest, NaNProduction) {
    EXPECT_TRUE((can_produce_nan_v<ops::divides<>, double>));   // 0/0, inf/inf
    EXPECT_TRUE((can_produce_nan_v<ops::sqrt_op<>, double>));   // sqrt(negative)
    EXPECT_TRUE((can_produce_nan_v<ops::log_op<>, double>));    // log(negative)
    EXPECT_TRUE((can_produce_nan_v<ops::asin_op<>, double>));   // asin(|x| > 1)
    EXPECT_TRUE((can_produce_nan_v<ops::power_op<double>, double>));    // 0^0, inf^0

    EXPECT_FALSE((can_produce_nan_v<ops::plus<>, double>));
    EXPECT_FALSE((can_produce_nan_v<ops::sin_op<>, double>));
    EXPECT_FALSE((can_produce_nan_v<ops::divides<>, int>));     // Integers don't have NaN
}

// Test operation complexity
TEST(OperationTraitsTest, OperationComplexity) {
    using Complexity = operation_complexity<ops::plus<>>::Level;

    EXPECT_EQ(operation_complexity_v<ops::plus<>>, Complexity::Trivial);
    EXPECT_EQ(operation_complexity_v<ops::minus<>>, Complexity::Trivial);
    EXPECT_EQ(operation_complexity_v<ops::multiplies<>>, Complexity::Simple);
    EXPECT_EQ(operation_complexity_v<ops::divides<>>, Complexity::Moderate);
    EXPECT_EQ(operation_complexity_v<ops::sqrt_op<>>, Complexity::Moderate);
    EXPECT_EQ(operation_complexity_v<ops::sin_op<>>, Complexity::Complex);
    EXPECT_EQ(operation_complexity_v<ops::exp_op<>>, Complexity::Complex);
    EXPECT_EQ((operation_complexity_v<ops::power_op<double>>), Complexity::VeryComplex);
}

// Test vectorization benefits
TEST(OperationTraitsTest, VectorizationBenefits) {
    // Simple arithmetic operations benefit from vectorization
    EXPECT_TRUE((benefits_from_vectorization_v<ops::plus<>, double>));
    EXPECT_TRUE((benefits_from_vectorization_v<ops::multiplies<>, float>));

    // Complex operations don't benefit as much
    EXPECT_FALSE((benefits_from_vectorization_v<ops::power_op<double>, double>));

    // Operations that can produce NaN don't vectorize well
    EXPECT_FALSE((benefits_from_vectorization_v<ops::sqrt_op<>, double>));

    // Non-arithmetic types don't vectorize
    struct CustomType {};
    EXPECT_FALSE((benefits_from_vectorization_v<ops::plus<>, CustomType>));
}

// Test operation properties aggregator
TEST(OperationTraitsTest, OperationProperties) {
    using props = operation_properties<ops::plus<>, double>;

    EXPECT_FALSE(props::is_unary);
    EXPECT_TRUE(props::is_binary);
    EXPECT_EQ(props::category, OperationCategory::Arithmetic);
    EXPECT_TRUE(props::preserves_type);
    EXPECT_TRUE((std::is_same_v<props::result_type, double>));
    EXPECT_TRUE(props::algebra.commutative);
    EXPECT_TRUE(props::ieee_safe);
    EXPECT_TRUE(props::might_overflow);
    EXPECT_FALSE(props::might_produce_nan);
    EXPECT_TRUE(props::vectorizable);

    // Test identity element
    EXPECT_EQ(props::identity(), 0.0);

    // Test for multiplication
    using mult_props = operation_properties<ops::multiplies<>, double>;
    EXPECT_EQ(mult_props::identity(), 1.0);
}

// Test fusion compatibility
TEST(OperationTraitsTest, FusionCompatibility) {
    EXPECT_TRUE((can_fuse_operations_v<ops::plus<>, ops::minus<>>));
    EXPECT_TRUE((can_fuse_operations_v<ops::multiplies<>, ops::plus<>>));

    // Complex operations don't fuse well
    EXPECT_FALSE((can_fuse_operations_v<ops::sin_op<>, ops::exp_op<>>));

    // Test fusion safety
    using fusion1 = fusion_compatibility<ops::plus<>, ops::multiplies<>>;
    EXPECT_TRUE(fusion1::is_safe);
    EXPECT_GT(fusion1::speedup_factor, 1.0);
}

// Test specialized implementation hints
TEST(OperationTraitsTest, SpecializedImplementation) {
    // Test with actual operations from ops_base if they exist
    // Otherwise skip these tests

    // SIMD operations
    EXPECT_TRUE((use_specialized_impl<ops::plus<>, double>::use_simd));

    // Parallel operations
    EXPECT_TRUE((use_specialized_impl<ops::sin_op<>, double>::use_parallel));
}

// Test error propagation
TEST(OperationTraitsTest, ErrorPropagation) {
    using add_error = error_propagation<ops::plus<>, double>;
    using div_error = error_propagation<ops::divides<>, double>;
    using exp_error = error_propagation<ops::exp_op<>, double>;

    EXPECT_EQ(add_error::condition_number(), 1.0);
    EXPECT_TRUE(add_error::is_numerically_stable);

    EXPECT_EQ(div_error::condition_number(), 2.0);
    EXPECT_TRUE(div_error::is_numerically_stable);

    EXPECT_EQ(exp_error::condition_number(), 10.0);
    EXPECT_FALSE(exp_error::is_numerically_stable);
}

// Test alignment requirements - use parentheses for template with comma
TEST(OperationTraitsTest, AlignmentRequirements) {
    // SIMD operations need alignment
    EXPECT_GE((requires_alignment_v<ops::plus<>, float>), 16);   // SSE alignment
    EXPECT_GE((requires_alignment_v<ops::plus<>, double>), 32);  // AVX alignment

    // Non-vectorizable operations use default alignment
    EXPECT_EQ((requires_alignment_v<ops::power_op<double>, double>), alignof(double));
}

// Test thread safety
TEST(OperationTraitsTest, ThreadSafety) {
    EXPECT_TRUE(is_thread_safe_v<ops::plus<>>);
    EXPECT_TRUE(is_thread_safe_v<ops::sin_op<>>);

    EXPECT_FALSE(is_thread_safe_v<ops::plus_assign<>>);  // In-place ops modify state
    EXPECT_FALSE(is_thread_safe_v<ops::minus_assign<>>);
}

// Test memory access patterns
TEST(OperationTraitsTest, MemoryAccessPattern) {
    EXPECT_EQ(memory_access_pattern_v<ops::plus<>>, MemoryAccessPattern::Sequential);
    EXPECT_EQ(memory_access_pattern_v<ops::sin_op<>>, MemoryAccessPattern::Sequential);
    EXPECT_EQ(memory_access_pattern_v<ops::sum_op<>>, MemoryAccessPattern::Reduction);
    EXPECT_EQ((memory_access_pattern_v<ops::max_op<double>>), MemoryAccessPattern::Reduction);
}

// Test batch processing hints
TEST(OperationTraitsTest, BatchProcessingHints) {
    using simple_batch = batch_processing_hints<ops::plus<>, double>;
    using complex_batch = batch_processing_hints<ops::sin_op<>, double>;

    // Simple operations can have larger batches
    EXPECT_GT(simple_batch::optimal_batch_size, complex_batch::optimal_batch_size);

    // Minimum batch size based on cache line
    constexpr size_t cache_line = 64;
    EXPECT_EQ(simple_batch::minimum_batch_size, cache_line / sizeof(double));

    // Complex operations benefit from prefetch
    EXPECT_FALSE(simple_batch::use_prefetch);
    EXPECT_TRUE(complex_batch::use_prefetch);
}

// Test special value handling
TEST(OperationTraitsTest, SpecialValueHandling) {
    using div_handling = special_value_handling<ops::divides<>, double>;
    EXPECT_TRUE(div_handling::handle_zero);
    EXPECT_FALSE(div_handling::handle_negative);
    EXPECT_TRUE(div_handling::handle_infinity);
    EXPECT_TRUE(div_handling::handle_nan);

    using sqrt_handling = special_value_handling<ops::sqrt_op<>, double>;
    EXPECT_FALSE(sqrt_handling::handle_zero);
    EXPECT_TRUE(sqrt_handling::handle_negative);

    using log_handling = special_value_handling<ops::log_op<>, double>;
    EXPECT_TRUE(log_handling::handle_zero);
    EXPECT_TRUE(log_handling::handle_negative);
}

// Test reduction operations
TEST(OperationTraitsTest, ReductionOperations) {
    EXPECT_TRUE(is_reduction_operation_v<ops::sum_op<>>);
    EXPECT_TRUE(is_reduction_operation_v<ops::product_op<>>);
    EXPECT_TRUE((is_reduction_operation_v<ops::min_op<double>>));
    EXPECT_TRUE((is_reduction_operation_v<ops::max_op<double>>));
    EXPECT_TRUE(is_reduction_operation_v<ops::mean_op<>>);

    EXPECT_FALSE(is_reduction_operation_v<ops::plus<>>);
    EXPECT_FALSE(is_reduction_operation_v<ops::sin_op<>>);
}

// Test reduction identity values - use parentheses for template with comma
TEST(OperationTraitsTest, ReductionIdentity) {
    EXPECT_EQ((reduction_identity<ops::sum_op<>, double>::value()), 0.0);
    EXPECT_EQ((reduction_identity<ops::product_op<>, double>::value()), 1.0);
    EXPECT_EQ((reduction_identity<ops::min_op<double>, double>::value()),
              std::numeric_limits<double>::max());
    EXPECT_EQ((reduction_identity<ops::max_op<double>, double>::value()),
              std::numeric_limits<double>::lowest());
}

// Test accuracy requirements
TEST(OperationTraitsTest, AccuracyRequirements) {
    using add_acc = accuracy_requirements<ops::plus<>, double>;
    using sqrt_acc = accuracy_requirements<ops::sqrt_op<>, double>;
    using sin_acc = accuracy_requirements<ops::sin_op<>, double>;
    using pow_acc = accuracy_requirements<ops::power_op<double>, double>;

    EXPECT_EQ(add_acc::max_ulp_error, 1);  // Exact
    EXPECT_EQ(sqrt_acc::max_ulp_error, 1);  // Correctly rounded
    EXPECT_EQ(sin_acc::max_ulp_error, 4);   // Transcendental

    EXPECT_FALSE(add_acc::needs_extended_precision);
    EXPECT_TRUE(pow_acc::needs_extended_precision);
}

// Test dispatch strategy
TEST(OperationTraitsTest, DispatchStrategy) {
    using Strategy = DispatchStrategy;

    // Small size uses scalar
    EXPECT_EQ((dispatch_strategy<ops::plus<>, double, 10>::value), Strategy::Scalar);

    // Large size with simple op uses vectorization
    auto large_simple = dispatch_strategy<ops::plus<>, double, 10000>::value;
    EXPECT_TRUE(large_simple == Strategy::Vectorized ||
                large_simple == Strategy::ParallelVectorized);

    // Complex operations use parallel
    auto complex_op = dispatch_strategy<ops::sin_op<>, double, 10000>::value;
    EXPECT_TRUE(complex_op == Strategy::Parallel ||
                complex_op == Strategy::ParallelVectorized);
}

// Compile-time tests
namespace CompileTimeTests {
    // Verify trait values at compile time
    static_assert(is_inplace_operation_v<ops::plus_assign<>>);
    static_assert(!is_inplace_operation_v<ops::plus<>>);

    static_assert(is_elementwise_operation_v<ops::sin_op<>>);
    static_assert(has_domain_restrictions_v<ops::sqrt_op<>>);

    static_assert((is_unary_operation_v<ops::negate<double>, double>));
    static_assert((is_binary_operation_v<ops::plus<double>, double>));

    static_assert(operation_category_v<ops::sin_op<>> == OperationCategory::Transcendental);
    static_assert(operation_category_v<ops::plus<>> == OperationCategory::Arithmetic);

    static_assert((preserves_type_v<ops::plus<>, double>));
    static_assert(!(preserves_type_v<ops::less<>, double>));

    static_assert((can_overflow_v<ops::exp_op<>, double>));
    static_assert((can_produce_nan_v<ops::sqrt_op<>, double>));

    static_assert((benefits_from_vectorization_v<ops::plus<>, float>));
    static_assert(is_thread_safe_v<ops::sin_op<>>);

    static_assert((is_reduction_operation_v<ops::sum_op<>>));

    // Test algebraic properties
    constexpr auto plus_props = algebraic_properties<ops::plus<>>::value;
    static_assert(plus_props.commutative);
    static_assert(plus_props.associative);
}

} // anonymous namespace

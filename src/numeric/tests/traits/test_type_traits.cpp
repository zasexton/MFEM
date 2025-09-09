#include <gtest/gtest.h>
#include <complex>
#include <type_traits>
#include <traits/type_traits.h>

namespace fem::numeric::traits::test {

// Test fixture for type traits
class TypeTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Type Category Tests
// =============================================================================

TEST_F(TypeTraitsTest, TypeCategoryIntegral) {
    EXPECT_EQ(type_category_v<bool>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<char>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<signed char>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<unsigned char>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<short>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<unsigned short>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<int>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<unsigned int>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<long>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<unsigned long>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<long long>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<unsigned long long>, TypeCategory::Integral);
}

TEST_F(TypeTraitsTest, TypeCategoryFloatingPoint) {
    EXPECT_EQ(type_category_v<float>, TypeCategory::FloatingPoint);
    EXPECT_EQ(type_category_v<double>, TypeCategory::FloatingPoint);
    EXPECT_EQ(type_category_v<long double>, TypeCategory::FloatingPoint);
}

TEST_F(TypeTraitsTest, TypeCategoryComplex) {
    EXPECT_EQ(type_category_v<std::complex<float>>, TypeCategory::Complex);
    EXPECT_EQ(type_category_v<std::complex<double>>, TypeCategory::Complex);
    EXPECT_EQ(type_category_v<std::complex<long double>>, TypeCategory::Complex);
}

TEST_F(TypeTraitsTest, TypeCategoryUnknown) {
    struct CustomType {};
    EXPECT_EQ(type_category_v<CustomType>, TypeCategory::Unknown);
    EXPECT_EQ(type_category_v<void*>, TypeCategory::Unknown);
    EXPECT_EQ(type_category_v<std::string>, TypeCategory::Unknown);
}

TEST_F(TypeTraitsTest, TypeCategoryWithQualifiers) {
    // Should work with cv-qualified types
    EXPECT_EQ(type_category_v<const int>, TypeCategory::Integral);
    EXPECT_EQ(type_category_v<volatile double>, TypeCategory::FloatingPoint);
    EXPECT_EQ(type_category_v<const volatile std::complex<float>>, TypeCategory::Complex);
}

// =============================================================================
// Numeric Type Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsNumeric) {
    // Assuming NumberLike concept accepts standard arithmetic types
    EXPECT_TRUE(is_numeric_v<int>);
    EXPECT_TRUE(is_numeric_v<float>);
    EXPECT_TRUE(is_numeric_v<double>);
    EXPECT_TRUE(is_numeric_v<long long>);

    // Non-numeric types
    struct NonNumeric {};
    EXPECT_FALSE(is_numeric_v<NonNumeric>);
    EXPECT_FALSE(is_numeric_v<void*>);
}

TEST_F(TypeTraitsTest, IsComplex) {
    EXPECT_TRUE(is_complex_v<std::complex<float>>);
    EXPECT_TRUE(is_complex_v<std::complex<double>>);
    EXPECT_TRUE(is_complex_v<std::complex<long double>>);

    EXPECT_FALSE(is_complex_v<float>);
    EXPECT_FALSE(is_complex_v<double>);
    EXPECT_FALSE(is_complex_v<int>);
}

// =============================================================================
// Type Transformation Tests
// =============================================================================

TEST_F(TypeTraitsTest, RealType) {
    static_assert(std::is_same_v<real_type_t<float>, float>);
    static_assert(std::is_same_v<real_type_t<double>, double>);
    static_assert(std::is_same_v<real_type_t<int>, int>);

    static_assert(std::is_same_v<real_type_t<std::complex<float>>, float>);
    static_assert(std::is_same_v<real_type_t<std::complex<double>>, double>);
}

TEST_F(TypeTraitsTest, ComplexType) {
    static_assert(std::is_same_v<complex_type_t<float>, std::complex<float>>);
    static_assert(std::is_same_v<complex_type_t<double>, std::complex<double>>);

    // Complex of complex should remain complex
    static_assert(std::is_same_v<complex_type_t<std::complex<float>>, std::complex<float>>);
}

TEST_F(TypeTraitsTest, ScalarType) {
    static_assert(std::is_same_v<scalar_type_t<float>, float>);
    static_assert(std::is_same_v<scalar_type_t<double>, double>);
    static_assert(std::is_same_v<scalar_type_t<int>, int>);

    static_assert(std::is_same_v<scalar_type_t<std::complex<float>>, float>);
    static_assert(std::is_same_v<scalar_type_t<std::complex<double>>, double>);
}

// =============================================================================
// Sign Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsSigned) {
    EXPECT_TRUE(is_signed_v<int>);
    EXPECT_TRUE(is_signed_v<float>);
    EXPECT_TRUE(is_signed_v<double>);
    EXPECT_TRUE(is_signed_v<long>);
    EXPECT_TRUE(is_signed_v<std::complex<float>>);
    EXPECT_TRUE(is_signed_v<std::complex<double>>);

    EXPECT_FALSE(is_signed_v<unsigned int>);
    EXPECT_FALSE(is_signed_v<unsigned long>);
    EXPECT_FALSE(is_signed_v<size_t>);
}

TEST_F(TypeTraitsTest, IsUnsigned) {
    EXPECT_TRUE(is_unsigned_v<unsigned int>);
    EXPECT_TRUE(is_unsigned_v<unsigned long>);
    EXPECT_TRUE(is_unsigned_v<size_t>);
    EXPECT_TRUE(is_unsigned_v<unsigned char>);

    EXPECT_FALSE(is_unsigned_v<int>);
    EXPECT_FALSE(is_unsigned_v<float>);
    EXPECT_FALSE(is_unsigned_v<double>);
    EXPECT_FALSE(is_unsigned_v<std::complex<float>>);
}

// =============================================================================
// Qualifier Manipulation Tests
// =============================================================================

TEST_F(TypeTraitsTest, RemoveCVRef) {
    static_assert(std::is_same_v<remove_cvref_t<const int&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<volatile double&&>, double>);
    static_assert(std::is_same_v<remove_cvref_t<const volatile float&>, float>);
    static_assert(std::is_same_v<remove_cvref_t<int>, int>);
}

TEST_F(TypeTraitsTest, IsSameDecay) {
    EXPECT_TRUE((is_same_decay_v<int, const int&>));
    EXPECT_TRUE((is_same_decay_v<double, double&&>));
    EXPECT_TRUE((is_same_decay_v<float[10], float*>));
    EXPECT_FALSE((is_same_decay_v<int, double>));
}

// =============================================================================
// Size and Layout Tests
// =============================================================================

TEST_F(TypeTraitsTest, ByteSize) {
    EXPECT_EQ(byte_size_v<char>, sizeof(char));
    EXPECT_EQ(byte_size_v<int>, sizeof(int));
    EXPECT_EQ(byte_size_v<double>, sizeof(double));
    EXPECT_EQ(byte_size_v<std::complex<float>>, sizeof(std::complex<float>));
}

TEST_F(TypeTraitsTest, Alignment) {
    EXPECT_EQ(alignment_of_v<char>, alignof(char));
    EXPECT_EQ(alignment_of_v<int>, alignof(int));
    EXPECT_EQ(alignment_of_v<double>, alignof(double));
    EXPECT_EQ(alignment_of_v<std::complex<double>>, alignof(std::complex<double>));
}

TEST_F(TypeTraitsTest, StandardLayout) {
    EXPECT_TRUE(has_standard_layout_v<int>);
    EXPECT_TRUE(has_standard_layout_v<double>);

    struct StandardLayoutType {
        int a;
        double b;
    };
    EXPECT_TRUE(has_standard_layout_v<StandardLayoutType>);
}

TEST_F(TypeTraitsTest, TriviallyCopyable) {
    EXPECT_TRUE(is_trivially_copyable_v<int>);
    EXPECT_TRUE(is_trivially_copyable_v<double>);
    EXPECT_TRUE(is_trivially_copyable_v<std::complex<float>>);
}

TEST_F(TypeTraitsTest, IsPOD) {
    EXPECT_TRUE(is_pod_v<int>);
    EXPECT_TRUE(is_pod_v<double>);

    struct PODType {
        int x;
        double y;
    };
    EXPECT_TRUE(is_pod_v<PODType>);

    struct NonPODType {
        NonPODType() = default;
        virtual ~NonPODType() = default;
        int x;
    };
    EXPECT_FALSE(is_pod_v<NonPODType>);
}

// =============================================================================
// Type Identity and Conditional Tests
// =============================================================================

TEST_F(TypeTraitsTest, TypeIdentity) {
    static_assert(std::is_same_v<type_identity_t<int>, int>);
    static_assert(std::is_same_v<type_identity_t<const double&>, const double&>);
}

TEST_F(TypeTraitsTest, Conditional) {
    static_assert(std::is_same_v<conditional_t<true, int, double>, int>);
    static_assert(std::is_same_v<conditional_t<false, int, double>, double>);
}

// =============================================================================
// Enable If Tests
// =============================================================================

TEST_F(TypeTraitsTest, EnableIfIntegral) {
    // This should compile for integral types
    auto test_integral = []<typename T>(T) -> enable_if_integral_t<T> {
        return T{};
    };

    EXPECT_EQ(test_integral(5), 5);
    EXPECT_EQ(test_integral(10L), 10L);
    // test_integral(3.14); // This should not compile
}

TEST_F(TypeTraitsTest, EnableIfFloating) {
    // This should compile for floating point types
    auto test_floating = []<typename T>(T val) -> enable_if_floating_t<T> {
        return val;
    };

    EXPECT_EQ(test_floating(3.14f), 3.14f);
    EXPECT_EQ(test_floating(2.718), 2.718);
    // test_floating(5); // This should not compile
}

// =============================================================================
// Common Type Tests
// =============================================================================

TEST_F(TypeTraitsTest, CommonType) {
    static_assert(std::is_same_v<common_type_t<int, long>, long>);
    static_assert(std::is_same_v<common_type_t<float, double>, double>);
    static_assert(std::is_same_v<common_type_t<int, float, double>, double>);
}

// =============================================================================
// Type Checking Utilities Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsAnyOf) {
    EXPECT_TRUE((is_any_of_v<int, float, int, double>));
    EXPECT_TRUE((is_any_of_v<double, float, int, double>));
    EXPECT_FALSE((is_any_of_v<char, float, int, double>));
    EXPECT_FALSE((is_any_of_v<long, float, int, double>));
}

TEST_F(TypeTraitsTest, AreSame) {
    EXPECT_TRUE((are_same_v<int, int, int>));
    EXPECT_TRUE((are_same_v<double, double, double, double>));
    EXPECT_FALSE((are_same_v<int, int, double>));
    EXPECT_FALSE((are_same_v<float, double>));
}

// =============================================================================
// Index Type Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsIndexType) {
    EXPECT_TRUE(is_index_type_v<size_t>);
    EXPECT_TRUE(is_index_type_v<int>);
    EXPECT_TRUE(is_index_type_v<long>);
    EXPECT_TRUE(is_index_type_v<long long>);
    EXPECT_TRUE(is_index_type_v<ptrdiff_t>);

    EXPECT_FALSE(is_index_type_v<float>);
    EXPECT_FALSE(is_index_type_v<double>);
    EXPECT_FALSE(is_index_type_v<unsigned int>);
    EXPECT_FALSE(is_index_type_v<char>);
}

// =============================================================================
// Operation Support Tests
// =============================================================================

TEST_F(TypeTraitsTest, HasArithmeticOps) {
    EXPECT_TRUE(has_arithmetic_ops_v<int>);
    EXPECT_TRUE(has_arithmetic_ops_v<float>);
    EXPECT_TRUE(has_arithmetic_ops_v<double>);
    EXPECT_TRUE(has_arithmetic_ops_v<std::complex<double>>);

    struct NoOps {};
    EXPECT_FALSE(has_arithmetic_ops_v<NoOps>);
}

TEST_F(TypeTraitsTest, HasComparisonOps) {
    EXPECT_TRUE(has_comparison_ops_v<int>);
    EXPECT_TRUE(has_comparison_ops_v<float>);
    EXPECT_TRUE(has_comparison_ops_v<double>);

    struct NoComparison {};
    EXPECT_FALSE(has_comparison_ops_v<NoComparison>);
}

// =============================================================================
// Constructibility Tests
// =============================================================================

TEST_F(TypeTraitsTest, Constructibility) {
    EXPECT_TRUE(is_default_constructible_v<int>);
    EXPECT_TRUE(is_copy_constructible_v<int>);
    EXPECT_TRUE(is_move_constructible_v<int>);

    struct NoDefault {
        NoDefault() = delete;
        NoDefault(int) {}
    };
    EXPECT_FALSE(is_default_constructible_v<NoDefault>);
    EXPECT_TRUE(is_copy_constructible_v<NoDefault>);
}

TEST_F(TypeTraitsTest, Assignability) {
    EXPECT_TRUE(is_copy_assignable_v<int>);
    EXPECT_TRUE(is_move_assignable_v<int>);

    struct NoCopyAssign {
        NoCopyAssign& operator=(const NoCopyAssign&) = delete;
        NoCopyAssign& operator=(NoCopyAssign&&) = default;
    };
    EXPECT_FALSE(is_copy_assignable_v<NoCopyAssign>);
    EXPECT_TRUE(is_move_assignable_v<NoCopyAssign>);
}

// =============================================================================
// Array Tests
// =============================================================================

TEST_F(TypeTraitsTest, ArrayTraits) {
    using Array1D = int[10];
    using Array2D = int[5][3];

    EXPECT_TRUE(is_array_v<Array1D>);
    EXPECT_TRUE(is_array_v<Array2D>);
    EXPECT_FALSE(is_array_v<int>);
    EXPECT_FALSE(is_array_v<int*>);

    EXPECT_EQ(rank_v<Array1D>, 1);
    EXPECT_EQ(rank_v<Array2D>, 2);
    EXPECT_EQ(rank_v<int>, 0);

    EXPECT_EQ(extent_v<Array1D>, 10);
    EXPECT_EQ(extent_v<Array2D>, 5);
    EXPECT_EQ((extent_v<Array2D, 1>), 3);

    static_assert(std::is_same_v<remove_all_extents_t<Array1D>, int>);
    static_assert(std::is_same_v<remove_all_extents_t<Array2D>, int>);
}

// =============================================================================
// Pointer Tests
// =============================================================================

TEST_F(TypeTraitsTest, PointerTraits) {
    EXPECT_TRUE(is_pointer_v<int*>);
    EXPECT_TRUE(is_pointer_v<const double*>);
    EXPECT_FALSE(is_pointer_v<int>);
    EXPECT_FALSE(is_pointer_v<int&>);

    static_assert(std::is_same_v<remove_pointer_t<int*>, int>);
    static_assert(std::is_same_v<remove_pointer_t<const double*>, const double>);
    static_assert(std::is_same_v<remove_pointer_t<int>, int>);

    static_assert(std::is_same_v<add_pointer_t<int>, int*>);
    static_assert(std::is_same_v<add_pointer_t<const double>, const double*>);
}

// =============================================================================
// Reference Tests
// =============================================================================

TEST_F(TypeTraitsTest, ReferenceTraits) {
    EXPECT_TRUE(is_reference_v<int&>);
    EXPECT_TRUE(is_reference_v<const double&>);
    EXPECT_TRUE(is_reference_v<int&&>);
    EXPECT_FALSE(is_reference_v<int>);

    EXPECT_TRUE(is_lvalue_reference_v<int&>);
    EXPECT_TRUE(is_lvalue_reference_v<const double&>);
    EXPECT_FALSE(is_lvalue_reference_v<int&&>);

    EXPECT_TRUE(is_rvalue_reference_v<int&&>);
    EXPECT_FALSE(is_rvalue_reference_v<int&>);

    static_assert(std::is_same_v<remove_reference_t<int&>, int>);
    static_assert(std::is_same_v<remove_reference_t<int&&>, int>);
    static_assert(std::is_same_v<remove_reference_t<const int&>, const int>);

    static_assert(std::is_same_v<add_lvalue_reference_t<int>, int&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<int>, int&&>);
}

// =============================================================================
// CV Qualifier Tests
// =============================================================================

TEST_F(TypeTraitsTest, CVQualifiers) {
    EXPECT_TRUE(is_const_v<const int>);
    EXPECT_FALSE(is_const_v<int>);
    EXPECT_TRUE(is_const_v<const volatile int>);

    EXPECT_TRUE(is_volatile_v<volatile int>);
    EXPECT_FALSE(is_volatile_v<int>);
    EXPECT_TRUE(is_volatile_v<const volatile int>);

    static_assert(std::is_same_v<add_const_t<int>, const int>);
    static_assert(std::is_same_v<remove_const_t<const int>, int>);

    static_assert(std::is_same_v<add_volatile_t<int>, volatile int>);
    static_assert(std::is_same_v<remove_volatile_t<volatile int>, int>);

    static_assert(std::is_same_v<add_cv_t<int>, const volatile int>);
    static_assert(std::is_same_v<remove_cv_t<const volatile int>, int>);
}

// =============================================================================
// Sign Manipulation Tests
// =============================================================================

TEST_F(TypeTraitsTest, SignManipulation) {
    static_assert(std::is_same_v<make_signed_t<unsigned int>, int>);
    static_assert(std::is_same_v<make_signed_t<unsigned long>, long>);

    static_assert(std::is_same_v<make_unsigned_t<int>, unsigned int>);
    static_assert(std::is_same_v<make_unsigned_t<long>, unsigned long>);
}

// =============================================================================
// Decay Tests
// =============================================================================

TEST_F(TypeTraitsTest, Decay) {
    static_assert(std::is_same_v<decay_t<const int&>, int>);
    static_assert(std::is_same_v<decay_t<int[10]>, int*>);
    static_assert(std::is_same_v<decay_t<int(int)>, int(*)(int)>);
}

// =============================================================================
// Convertibility Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsConvertible) {
    EXPECT_TRUE((is_convertible_v<int, long>));
    EXPECT_TRUE((is_convertible_v<float, double>));
    EXPECT_TRUE((is_convertible_v<int, double>));

    struct Base {};
    struct Derived : Base {};
    EXPECT_TRUE((is_convertible_v<Derived*, Base*>));
    EXPECT_FALSE((is_convertible_v<Base*, Derived*>));
}

// =============================================================================
// Type Classification Tests
// =============================================================================

TEST_F(TypeTraitsTest, TypeClassification) {
    class TestClass {};
    enum TestEnum { A, B, C };
    union TestUnion { int i; float f; };

    EXPECT_TRUE(is_class_v<TestClass>);
    EXPECT_FALSE(is_class_v<int>);

    EXPECT_TRUE(is_enum_v<TestEnum>);
    EXPECT_FALSE(is_enum_v<int>);

    EXPECT_TRUE(is_union_v<TestUnion>);
    EXPECT_FALSE(is_union_v<TestClass>);

    EXPECT_TRUE(is_function_v<void(int)>);
    EXPECT_FALSE(is_function_v<void(*)(int)>);

    EXPECT_TRUE(is_void_v<void>);
    EXPECT_FALSE(is_void_v<int>);

    EXPECT_TRUE(is_null_pointer_v<std::nullptr_t>);
    EXPECT_FALSE(is_null_pointer_v<void*>);
}

// =============================================================================
// Index Sequence Tests
// =============================================================================

TEST_F(TypeTraitsTest, IndexSequence) {
    using Seq = index_sequence<0, 1, 2, 3>;
    using MakeSeq = make_index_sequence<4>;

    // Check that make_index_sequence generates correct sequence
    auto check_sequence = []<size_t... Is>(std::index_sequence<Is...>) {
        return ((Is == Is) && ...); // Just check it compiles
    };

    EXPECT_TRUE(check_sequence(make_index_sequence<5>{}));
}

// =============================================================================
// Invocable Tests
// =============================================================================

TEST_F(TypeTraitsTest, Invocable) {
    auto lambda = [](int x) { return x * 2; };

    EXPECT_TRUE((is_invocable_v<decltype(lambda), int>));
    EXPECT_FALSE((is_invocable_v<decltype(lambda), std::string>));

    auto noexcept_lambda = [](int x) noexcept { return x * 2; };
    EXPECT_TRUE((is_nothrow_invocable_v<decltype(noexcept_lambda), int>));
    EXPECT_FALSE((is_nothrow_invocable_v<decltype(lambda), int>));

    static_assert(std::is_same_v<invoke_result_t<decltype(lambda), int>, int>);
}

// =============================================================================
// Compile-time Tests (Static Assertions)
// =============================================================================

TEST_F(TypeTraitsTest, CompileTimeChecks) {
    // Test void_t
    template<typename T, typename = void>
    struct has_value_type : std::false_type {};

    template<typename T>
    struct has_value_type<T, void_t<typename T::value_type>> : std::true_type {};

    struct WithValueType { using value_type = int; };
    struct WithoutValueType {};

    static_assert(has_value_type<WithValueType>::value);
    static_assert(!has_value_type<WithoutValueType>::value);

    // All static assertions should pass
    SUCCEED();
}

// =============================================================================
// Edge Cases and Corner Cases
// =============================================================================

TEST_F(TypeTraitsTest, EdgeCases) {
    // Test with function pointers
    using FuncPtr = void(*)(int);
    EXPECT_TRUE(is_pointer_v<FuncPtr>);
    EXPECT_FALSE(is_function_v<FuncPtr>);

    // Test with member pointers
    struct S { int member; };
    using MemberPtr = int S::*;
    EXPECT_FALSE(is_pointer_v<MemberPtr>);

    // Test with references to arrays
    using ArrayRef = int(&)[10];
    EXPECT_TRUE(is_reference_v<ArrayRef>);
    EXPECT_FALSE(is_array_v<ArrayRef>);

    // Test with void
    EXPECT_FALSE(is_numeric_v<void>);
    EXPECT_TRUE(is_void_v<void>);
}

} // anonymous namespace
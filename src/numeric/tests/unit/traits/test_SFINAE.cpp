#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <array>
#include <list>
#include <complex>
#include <memory>
#include <string>

#include <traits/SFINAE.h>

// Mock types for testing
namespace test_types {

    // Simple container-like class with required members
    template<typename T>
    class MockContainer {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using iterator = T*;
        using const_iterator = const T*;

        size_type size() const { return 0; }
        T* data() { return nullptr; }
        const T* data() const { return nullptr; }
        iterator begin() { return nullptr; }
        iterator end() { return nullptr; }
        const_iterator begin() const { return nullptr; }
        const_iterator end() const { return nullptr; }
    };

    // Container without data() method
    template<typename T>
    class IncompleteContainer {
    public:
        using value_type = T;
        std::size_t size() const { return 0; }
        T* begin() { return nullptr; }
        T* end() { return nullptr; }
    };

    // Class with shape and dimensions
    class ShapedType {
    public:
        std::array<size_t, 3> shape() const { return {1, 2, 3}; }
        size_t dimensions() const { return 3; }
    };

    // Class with eval method (expression-like)
    class ExpressionType {
    public:
        double eval() const { return 42.0; }
    };

    // Class with is_view method
    class ViewType {
    public:
        using value_type = double;
        bool is_view() const { return true; }
    };

    // Class with storage_type member
    class WithStorageType {
    public:
        using storage_type = std::vector<double>;
    };

    // Empty class for negative tests
    class EmptyClass {};

    // Template class for specialization testing
    template<typename T, typename U = void>
    class TemplateClass {};

} // namespace test_types

using namespace fem::numeric::traits;
using namespace test_types;


template<typename T>
constexpr bool is_complete_container =
    has_value_type_v<T> &&
    has_size_v<T> &&
    has_data_v<T> &&
    has_begin_v<T> &&
    has_end_v<T>;

template<typename T>
using has_double_value_type = std::enable_if_t<
    has_value_type_v<T> && std::is_same_v<typename T::value_type, double>
>;

namespace {

    struct AddOp {
        template<typename T, typename U>
        auto operator()(T&& t, U&& u) -> decltype(t + u) { return t + u; }
    };

    struct DivOp {
        template<typename T, typename U>
        auto operator()(T&& t, U&& u) -> decltype(t / u) { return t / u; }
    };

    struct NegOp {
        template<typename T>
        auto operator()(T&& t) -> decltype(-t) { return -t; }
    };

    struct DerefOp {
        template<typename T>
        auto operator()(T&& t) -> decltype(*t) { return *t; }
    };

} // namespace


// ============================================================================
// Basic SFINAE utilities tests
// ============================================================================

TEST(SFINAETest, AlwaysFalse) {
    static_assert(!always_false_v<int>);
    static_assert(!always_false_v<double, float, char>);
    static_assert(!always_false_v<>);
}

// ============================================================================
// Detection idiom tests
// ============================================================================

TEST(SFINAETest, IsDetected) {
    // Test with valid operations
    static_assert(is_detected_v<has_size_t, MockContainer<int>>);
    static_assert(is_detected_v<has_data_t, MockContainer<double>>);

    // Test with invalid operations
    static_assert(!is_detected_v<has_size_t, EmptyClass>);
    static_assert(!is_detected_v<has_data_t, IncompleteContainer<int>>);
}

TEST(SFINAETest, DetectedOr) {
    // Default type when detection fails
    using default_type = int;
    using result1 = detected_or_t<default_type, has_value_type_t, EmptyClass>;
    static_assert(std::is_same_v<result1, default_type>);

    // Detected type when detection succeeds
    using result2 = detected_or_t<default_type, has_value_type_t, MockContainer<double>>;
    static_assert(std::is_same_v<result2, double>);
}

TEST(SFINAETest, IsDetectedExact) {
    // Test exact type match
    static_assert(is_detected_exact_v<double, has_value_type_t, MockContainer<double>>);
    static_assert(!is_detected_exact_v<float, has_value_type_t, MockContainer<double>>);
}

// ============================================================================
// Member function detection tests
// ============================================================================

TEST(SFINAETest, HasMemberFunctions) {
    // Positive tests
    static_assert(has_size_v<MockContainer<int>>);
    static_assert(has_data_v<MockContainer<int>>);
    static_assert(has_begin_v<MockContainer<int>>);
    static_assert(has_end_v<MockContainer<int>>);

    // Standard containers
    static_assert(has_size_v<std::vector<int>>);
    static_assert(has_data_v<std::vector<int>>);
    static_assert(has_begin_v<std::vector<int>>);
    static_assert(has_end_v<std::vector<int>>);

    // Negative tests
    static_assert(!has_size_v<EmptyClass>);
    static_assert(!has_data_v<EmptyClass>);
    static_assert(!has_begin_v<EmptyClass>);
    static_assert(!has_end_v<EmptyClass>);

    // Partial implementation
    static_assert(has_size_v<IncompleteContainer<int>>);
    static_assert(!has_data_v<IncompleteContainer<int>>);
    static_assert(has_begin_v<IncompleteContainer<int>>);
    static_assert(has_end_v<IncompleteContainer<int>>);
}

TEST(SFINAETest, HasShapeAndDimensions) {
    static_assert(has_shape_v<ShapedType>);
    static_assert(has_dimensions_v<ShapedType>);

    static_assert(!has_shape_v<EmptyClass>);
    static_assert(!has_dimensions_v<EmptyClass>);
}

TEST(SFINAETest, HasEval) {
    static_assert(has_eval_v<ExpressionType>);
    static_assert(!has_eval_v<EmptyClass>);
}

TEST(SFINAETest, HasIsView) {
    static_assert(has_is_view_v<ViewType>);
    static_assert(!has_is_view_v<EmptyClass>);
}

// ============================================================================
// Member type detection tests
// ============================================================================

TEST(SFINAETest, HasMemberTypes) {
    // Test value_type
    static_assert(has_value_type_v<MockContainer<int>>);
    static_assert(has_value_type_v<std::vector<double>>);
    static_assert(!has_value_type_v<EmptyClass>);

    // Test element_type
    static_assert(has_element_type_v<std::shared_ptr<int>>);
    static_assert(!has_element_type_v<MockContainer<int>>);

    // Test allocator_type
    static_assert(has_allocator_type_v<std::vector<int>>);
    static_assert(!has_allocator_type_v<MockContainer<int>>);

    // Test iterator types
    static_assert(has_iterator_v<MockContainer<int>>);
    static_assert(has_const_iterator_v<MockContainer<int>>);
    static_assert(has_iterator_v<std::vector<int>>);
    static_assert(!has_iterator_v<EmptyClass>);

    // Test size_type and difference_type
    static_assert(has_size_type_v<MockContainer<int>>);
    static_assert(has_size_type_v<std::vector<int>>);
    static_assert(has_difference_type_v<std::vector<int>>);
    static_assert(!has_size_type_v<EmptyClass>);
}

TEST(SFINAETest, HasStorageType) {
    static_assert(has_storage_type_v<WithStorageType>);
    static_assert(!has_storage_type_v<EmptyClass>);
}

// ============================================================================
// Operation detection tests
// ============================================================================

TEST(SFINAETest, BinaryOperations) {
    // Arithmetic types
    static_assert(has_binary_op_v<int, int, AddOp>);
    static_assert(has_binary_op_v<double, float, AddOp>);
    static_assert(has_binary_op_v<int, double, DivOp>);

    // String concatenation
    static_assert(has_binary_op_v<std::string, std::string, AddOp>);

    // Invalid operations
    static_assert(!has_binary_op_v<EmptyClass, EmptyClass, AddOp>);
    static_assert(!has_binary_op_v<std::string, int, DivOp>);
}

TEST(SFINAETest, UnaryOperations) {
    // Arithmetic negation
    static_assert(has_unary_op_v<int, NegOp>);
    static_assert(has_unary_op_v<double, NegOp>);

    // Pointer dereference
    static_assert(has_unary_op_v<int*, DerefOp>);
    static_assert(has_unary_op_v<std::shared_ptr<int>, DerefOp>);

    // Invalid operations
    static_assert(!has_unary_op_v<EmptyClass, NegOp>);
    static_assert(!has_unary_op_v<int, DerefOp>);
}

// ============================================================================
// Iterator detection tests
// ============================================================================

TEST(SFINAETest, IteratorDetection) {
    // Valid iterators
    static_assert(detect_iterator_v<int*>);
    static_assert(detect_iterator_v<std::vector<int>::iterator>);
    static_assert(detect_iterator_v<std::list<double>::const_iterator>);

    // Random access iterators
    static_assert(detect_random_access_iterator_v<int*>);
    static_assert(detect_random_access_iterator_v<std::vector<int>::iterator>);
    static_assert(detect_random_access_iterator_v<std::array<double, 5>::iterator>);

    // Non-random access iterators
    static_assert(!detect_random_access_iterator_v<std::list<int>::iterator>);

    // Non-iterators
    static_assert(!detect_iterator_v<int>);
    static_assert(!detect_iterator_v<EmptyClass>);
}

// ============================================================================
// Type selection helpers tests
// ============================================================================

TEST(SFINAETest, EnableIf) {
    // Test function overloading with enable_if
    auto test_func = []<typename T>(T value) -> enable_if_t<std::is_integral_v<T>, int> {
        return static_cast<int>(value) * 2;
    };

    EXPECT_EQ(test_func(5), 10);
    // test_func(5.5) would not compile
}

TEST(SFINAETest, FirstValidType) {
    // Test with all void types (should return default)
    using result1 = first_valid_type_t<int, void, void, void>;
    static_assert(std::is_same_v<result1, int>);

    // Test with first valid type
    using result2 = first_valid_type_t<int, double, void, float>;
    static_assert(std::is_same_v<result2, double>);

    // Test with void followed by valid type
    using result3 = first_valid_type_t<int, void, void, std::string>;
    static_assert(std::is_same_v<result3, std::string>);
}

TEST(SFINAETest, ConjunctionDisjunction) {
    static_assert(conjunction_v<true, true, true>);
    static_assert(!conjunction_v<true, false, true>);
    static_assert(!conjunction_v<false, false, false>);
    static_assert(conjunction_v<>);  // Empty conjunction is true

    static_assert(disjunction_v<true, false, false>);
    static_assert(disjunction_v<false, false, true>);
    static_assert(!disjunction_v<false, false, false>);
    static_assert(!disjunction_v<>);  // Empty disjunction is false
}

// ============================================================================
// Common type computation tests
// ============================================================================

TEST(SFINAETest, NumericCommonType) {
    // Basic numeric types
    static_assert(std::is_same_v<numeric_common_type_t<int, int>, int>);
    static_assert(std::is_same_v<numeric_common_type_t<int, double>, double>);
    static_assert(std::is_same_v<numeric_common_type_t<float, double>, double>);

    // Complex types
    using complex_float = std::complex<float>;
    using complex_double = std::complex<double>;

    static_assert(std::is_same_v<
        numeric_common_type_t<complex_float, complex_float>,
        complex_float>);

    // Mixed real and complex
    static_assert(std::is_same_v<
        numeric_common_type_t<int, complex_float>,
        complex_float>);

    static_assert(std::is_same_v<
        numeric_common_type_t<double, complex_float>,
        complex_double>);

    // Complex with different precision
    static_assert(std::is_same_v<
        numeric_common_type_t<complex_float, complex_double>,
        complex_double>);
}

// ============================================================================
// Specialization detection tests
// ============================================================================

TEST(SFINAETest, IsSpecialization) {
    // Positive tests
    static_assert(is_specialization_v<std::vector<int>, std::vector>);
    static_assert(is_specialization_v<std::array<double, 5>, std::array>);
    static_assert(is_specialization_v<std::complex<float>, std::complex>);
    static_assert(is_specialization_v<TemplateClass<int>, TemplateClass>);
    static_assert(is_specialization_v<TemplateClass<int, double>, TemplateClass>);

    // Negative tests
    static_assert(!is_specialization_v<int, std::vector>);
    static_assert(!is_specialization_v<EmptyClass, std::vector>);
    static_assert(!is_specialization_v<std::vector<int>, std::array>);
}

// ============================================================================
// Container validation tests
// ============================================================================

TEST(SFINAETest, ValidateContainer) {
    using valid_container = validate_container_type<MockContainer<int>>;
    static_assert(valid_container::has_required_members);
    static_assert(valid_container::has_iterators);
    static_assert(valid_container::is_valid);

    using invalid_container = validate_container_type<IncompleteContainer<int>>;
    static_assert(!invalid_container::has_required_members);  // Missing data()
    static_assert(invalid_container::has_iterators);
    static_assert(!invalid_container::is_valid);

    using empty = validate_container_type<EmptyClass>;
    static_assert(!empty::has_required_members);
    static_assert(!empty::has_iterators);
    static_assert(!empty::is_valid);

    // Standard containers
    using std_vector = validate_container_type<std::vector<int>>;
    static_assert(std_vector::is_valid);
}

// ============================================================================
// Complex detection scenarios
// ============================================================================

TEST(SFINAETest, ComplexDetectionScenarios) {
    // Combine multiple detections
    static_assert(is_complete_container<MockContainer<int>>);
    static_assert(is_complete_container<std::vector<int>>);
    static_assert(!is_complete_container<IncompleteContainer<int>>);
    static_assert(!is_complete_container<EmptyClass>);

    // Detect specific member type values
    static_assert(is_detected_v<has_double_value_type, MockContainer<double>>);
    static_assert(!is_detected_v<has_double_value_type, MockContainer<int>>);
    static_assert(!is_detected_v<has_double_value_type, EmptyClass>);
}

// ============================================================================
// Edge cases and corner cases
// ============================================================================

TEST(SFINAETest, EdgeCases) {
    // Detection with reference types
    static_assert(has_size_v<MockContainer<int>&>);
    static_assert(has_size_v<const MockContainer<int>&>);
    static_assert(has_size_v<MockContainer<int>&&>);

    // Detection with cv-qualified types
    static_assert(has_value_type_v<const MockContainer<int>>);
    static_assert(has_value_type_v<volatile std::vector<int>>);

    // Nested template detection
    using nested_vector = std::vector<std::vector<int>>;
    static_assert(has_value_type_v<nested_vector>);
    static_assert(std::is_same_v<
        typename nested_vector::value_type,
        std::vector<int>>);
}

// ============================================================================
// Runtime tests for detected operations
// ============================================================================

TEST(SFINAETest, RuntimeOperationDetection) {
    MockContainer<int> container;

    if constexpr (has_size_v<decltype(container)>) {
        EXPECT_EQ(container.size(), 0);
    }

    if constexpr (has_data_v<decltype(container)>) {
        EXPECT_EQ(container.data(), nullptr);
    }

    ShapedType shaped;
    if constexpr (has_shape_v<decltype(shaped)>) {
        auto shape = shaped.shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 2);
        EXPECT_EQ(shape[2], 3);
    }

    if constexpr (has_dimensions_v<decltype(shaped)>) {
        EXPECT_EQ(shaped.dimensions(), 3);
    }

    ExpressionType expr;
    if constexpr (has_eval_v<decltype(expr)>) {
        EXPECT_DOUBLE_EQ(expr.eval(), 42.0);
    }

    ViewType view;
    if constexpr (has_is_view_v<decltype(view)>) {
        EXPECT_TRUE(view.is_view());
    }
}
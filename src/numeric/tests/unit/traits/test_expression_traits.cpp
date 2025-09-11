#include <gtest/gtest.h>
#include <type_traits>
#include <complex>

#include <base/expression_base.h>
#include <base/container_base.h>
#include <base/ops_base.h>
#include <base/numeric_base.h>
#include <base/traits_base.h>

#include <traits/type_traits.h>
#include <traits/numeric_traits.h>
#include <traits/operation_traits.h>
#include <traits/expression_traits.h>

// Test fixtures
namespace {

using namespace fem::numeric;
namespace nt = fem::numeric::traits;

// Mock container for testing
template<typename T>
    class MockContainer : public ContainerBase<MockContainer<T>, T> {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    MockContainer(size_t size = 10) : data_(size) {}

    size_type size() const { return data_.size(); }
    Shape shape() const { return Shape{data_.size()}; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    iterator begin() { return data_.data(); }
    const_iterator begin() const { return data_.data(); }
    iterator end() { return data_.data() + data_.size(); }
    const_iterator end() const { return data_.data() + data_.size(); }

private:
    std::vector<T> data_;
};

// Create some test expression types
using TestContainer = MockContainer<double>;
using TestTerminal = TerminalExpression<TestContainer>;
using TestScalar = ScalarExpression<double>;
using TestUnary = UnaryExpression<TestTerminal, ops::negate<double>>;
// BinaryExpression now takes template parameters as <Op, Left, Right>
using TestBinary = BinaryExpression<ops::plus<double>, TestTerminal, TestTerminal>;
using ComplexBinary = BinaryExpression<ops::multiplies<double>, TestUnary, TestBinary>;

// Custom non-expression type for negative testing
struct NotAnExpression {
    double value;
};

// TEST: Expression type detection
TEST(ExpressionTraitsTest, IsExpression) {
    // Positive cases
    EXPECT_TRUE(nt::is_expression_v<TestTerminal>);
    EXPECT_TRUE(nt::is_expression_v<TestScalar>);
    EXPECT_TRUE(nt::is_expression_v<TestUnary>);
    EXPECT_TRUE(nt::is_expression_v<TestBinary>);
    EXPECT_TRUE(nt::is_expression_v<ComplexBinary>);

    // Negative cases
    EXPECT_FALSE(nt::is_expression_v<NotAnExpression>);
    EXPECT_FALSE(nt::is_expression_v<double>);
    EXPECT_FALSE(nt::is_expression_v<TestContainer>);
    EXPECT_FALSE(nt::is_expression_v<int>);
}

// TEST: Terminal expression detection
TEST(ExpressionTraitsTest, IsTerminalExpression) {
    EXPECT_TRUE(nt::is_terminal_expression_v<TestTerminal>);
    EXPECT_TRUE(nt::is_terminal_expression_v<TerminalExpression<MockContainer<float>>>);

    EXPECT_FALSE(nt::is_terminal_expression_v<TestScalar>);
    EXPECT_FALSE(nt::is_terminal_expression_v<TestUnary>);
    EXPECT_FALSE(nt::is_terminal_expression_v<TestBinary>);
    EXPECT_FALSE(nt::is_terminal_expression_v<NotAnExpression>);
}

// TEST: Binary expression detection
TEST(ExpressionTraitsTest, IsBinaryExpression) {
    EXPECT_TRUE(nt::is_binary_expression_v<TestBinary>);
    EXPECT_TRUE(nt::is_binary_expression_v<ComplexBinary>);
    EXPECT_TRUE((nt::is_binary_expression_v<BinaryExpression<ops::minus<double>, TestTerminal, TestScalar>>));

    EXPECT_FALSE(nt::is_binary_expression_v<TestTerminal>);
    EXPECT_FALSE(nt::is_binary_expression_v<TestUnary>);
    EXPECT_FALSE(nt::is_binary_expression_v<TestScalar>);
    EXPECT_FALSE(nt::is_binary_expression_v<NotAnExpression>);
}

// TEST: Unary expression detection
TEST(ExpressionTraitsTest, IsUnaryExpression) {
    EXPECT_TRUE(nt::is_unary_expression_v<TestUnary>);
    EXPECT_TRUE((nt::is_unary_expression_v<UnaryExpression<TestTerminal, ops::abs_op<double>>>));
    EXPECT_TRUE((nt::is_unary_expression_v<UnaryExpression<TestScalar, ops::sin_op<>>>));

    EXPECT_FALSE(nt::is_unary_expression_v<TestTerminal>);
    EXPECT_FALSE(nt::is_unary_expression_v<TestBinary>);
    EXPECT_FALSE(nt::is_unary_expression_v<TestScalar>);
    EXPECT_FALSE(nt::is_unary_expression_v<NotAnExpression>);
}

// TEST: Scalar expression detection
TEST(ExpressionTraitsTest, IsScalarExpression) {
    EXPECT_TRUE(nt::is_scalar_expression_v<TestScalar>);
    EXPECT_TRUE(nt::is_scalar_expression_v<ScalarExpression<float>>);
    EXPECT_TRUE(nt::is_scalar_expression_v<ScalarExpression<int>>);

    EXPECT_FALSE(nt::is_scalar_expression_v<TestTerminal>);
    EXPECT_FALSE(nt::is_scalar_expression_v<TestUnary>);
    EXPECT_FALSE(nt::is_scalar_expression_v<TestBinary>);
    EXPECT_FALSE(nt::is_scalar_expression_v<NotAnExpression>);
}

// TEST: Broadcast expression detection
TEST(ExpressionTraitsTest, IsBroadcastExpression) {
    // Currently, BroadcastExpression is not defined, so all should be false
    EXPECT_FALSE(nt::is_broadcast_expression_v<TestTerminal>);
    EXPECT_FALSE(nt::is_broadcast_expression_v<TestScalar>);
    EXPECT_FALSE(nt::is_broadcast_expression_v<TestUnary>);
    EXPECT_FALSE(nt::is_broadcast_expression_v<TestBinary>);
}

// TEST: Extract value type from expression
TEST(ExpressionTraitsTest, ExpressionValueType) {
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<TestTerminal>, double>));
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<TestScalar>, double>));
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<TestUnary>, double>));
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<TestBinary>, double>));

    // Non-expression types should give void
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<NotAnExpression>, void>));
}

// TEST: Expression category detection
TEST(ExpressionTraitsTest, ExpressionCategory) {
    EXPECT_EQ(nt::expression_category_v<TestTerminal>, nt::ExpressionCategory::Terminal);
    EXPECT_EQ(nt::expression_category_v<TestScalar>, nt::ExpressionCategory::Scalar);
    EXPECT_EQ(nt::expression_category_v<TestUnary>, nt::ExpressionCategory::Unary);
    EXPECT_EQ(nt::expression_category_v<TestBinary>, nt::ExpressionCategory::Binary);
    EXPECT_EQ(nt::expression_category_v<ComplexBinary>, nt::ExpressionCategory::Binary);

    // Non-expressions should be Unknown
    EXPECT_EQ(nt::expression_category_v<NotAnExpression>, nt::ExpressionCategory::Unknown);
    EXPECT_EQ(nt::expression_category_v<double>, nt::ExpressionCategory::Unknown);
}

// TEST: Expression depth calculation
TEST(ExpressionTraitsTest, ExpressionDepth) {
    // Terminal has depth 1
    EXPECT_EQ(nt::expression_depth_v<TestTerminal>, 1);

    // Scalar has depth 0 (not a terminal)
    EXPECT_EQ(nt::expression_depth_v<TestScalar>, 0);

    // Unary: 1 + depth of argument
    EXPECT_EQ(nt::expression_depth_v<TestUnary>, 2);  // 1 + 1 (terminal)

    // Binary: 1 + max(left, right)
    EXPECT_EQ(nt::expression_depth_v<TestBinary>, 2);  // 1 + max(1, 1)

    // Complex binary: 1 + max(unary, binary) = 1 + max(2, 2) = 3
    EXPECT_EQ(nt::expression_depth_v<ComplexBinary>, 3);

    // Non-expressions have depth 0
    EXPECT_EQ(nt::expression_depth_v<NotAnExpression>, 0);
}

// TEST: Operation count
TEST(ExpressionTraitsTest, OperationCount) {
    // Terminals have no operations
    EXPECT_EQ(nt::operation_count_v<TestTerminal>, 0);
    EXPECT_EQ(nt::operation_count_v<TestScalar>, 0);

    // Unary has 1 operation
    EXPECT_EQ(nt::operation_count_v<TestUnary>, 1);

    // Binary has 1 operation + operations in children
    EXPECT_EQ(nt::operation_count_v<TestBinary>, 1);  // 1 + 0 + 0

    // Complex binary: 1 + (1) + (1) = 3
    EXPECT_EQ(nt::operation_count_v<ComplexBinary>, 3);

    // Non-expressions have 0 operations
    EXPECT_EQ(nt::operation_count_v<NotAnExpression>, 0);
}

// TEST: Broadcast detection
TEST(ExpressionTraitsTest, HasBroadcast) {
    // No broadcasts in simple expressions
    EXPECT_FALSE(nt::has_broadcast_v<TestTerminal>);
    EXPECT_FALSE(nt::has_broadcast_v<TestScalar>);
    EXPECT_FALSE(nt::has_broadcast_v<TestUnary>);
    EXPECT_FALSE(nt::has_broadcast_v<TestBinary>);
    EXPECT_FALSE(nt::has_broadcast_v<ComplexBinary>);
}

// TEST: Vectorization capability
TEST(ExpressionTraitsTest, IsVectorizable) {
    // Simple expressions with arithmetic types are vectorizable
    EXPECT_TRUE(nt::is_vectorizable_v<TestTerminal>);
    EXPECT_TRUE(nt::is_vectorizable_v<TestScalar>);
    EXPECT_TRUE(nt::is_vectorizable_v<TestUnary>);
    EXPECT_TRUE(nt::is_vectorizable_v<TestBinary>);

    // Complex expressions are still vectorizable if depth is reasonable
    EXPECT_TRUE(nt::is_vectorizable_v<ComplexBinary>);

    // Non-expressions are not vectorizable (void value_type)
    EXPECT_FALSE(nt::is_vectorizable_v<NotAnExpression>);
}

// TEST: Expression properties aggregator
TEST(ExpressionTraitsTest, ExpressionProperties) {
    using props = nt::expression_properties<TestBinary>;

    EXPECT_TRUE(props::is_expression);
    EXPECT_EQ(props::category, nt::ExpressionCategory::Binary);
    EXPECT_EQ(props::depth, 2);
    EXPECT_EQ(props::operation_count, 1);
    EXPECT_FALSE(props::has_broadcasts);
    EXPECT_TRUE((std::is_same_v<props::value_type, double>));
    EXPECT_TRUE(props::can_vectorize);
    EXPECT_FALSE(props::should_parallelize);  // Not enough operations
    EXPECT_FALSE(props::should_materialize);  // Not deep enough

    // Check evaluation strategy
    EXPECT_EQ(props::recommended_strategy, nt::EvaluationStrategy::Lazy);  // Simple expression
}

// TEST: Extract operation type
TEST(ExpressionTraitsTest, ExpressionOperation) {
    using UnaryOp = nt::expression_operation_t<TestUnary>;
    using BinaryOp = nt::expression_operation_t<TestBinary>;

    EXPECT_TRUE((std::is_same_v<UnaryOp, ops::negate<double>>));
    EXPECT_TRUE((std::is_same_v<BinaryOp, ops::plus<double>>));

    // Terminal/Scalar don't have operations
    EXPECT_TRUE((std::is_same_v<nt::expression_operation_t<TestTerminal>, void>));
    EXPECT_TRUE((std::is_same_v<nt::expression_operation_t<TestScalar>, void>));
}

// TEST: Commutativity check across expression tree
TEST(ExpressionTraitsTest, IsCommutativeExpression) {
    // Plus is commutative
    EXPECT_TRUE(nt::is_commutative_v<TestBinary>);

    // Create non-commutative binary expression
    using NonCommBinary = BinaryExpression<ops::minus<double>, TestTerminal, TestTerminal>;
    EXPECT_FALSE(nt::is_commutative_v<NonCommBinary>);

    // Expressions without binary operations are trivially commutative
    EXPECT_TRUE(nt::is_commutative_v<TestUnary>);
    EXPECT_TRUE(nt::is_commutative_v<TestTerminal>);
}

// TEST: Parallel safety
TEST(ExpressionTraitsTest, IsParallelSafe) {
    EXPECT_TRUE(nt::is_parallel_safe_v<TestTerminal>);
    EXPECT_TRUE(nt::is_parallel_safe_v<TestScalar>);
    EXPECT_TRUE(nt::is_parallel_safe_v<TestUnary>);
    EXPECT_TRUE(nt::is_parallel_safe_v<TestBinary>);

    // Non-expressions are not parallel safe
    EXPECT_FALSE(nt::is_parallel_safe_v<NotAnExpression>);
}

// TEST: Expression compatibility
TEST(ExpressionTraitsTest, AreExpressionsCompatible) {
    // Same value types are compatible
    EXPECT_TRUE((nt::are_expressions_compatible_v<TestTerminal, TestTerminal>));
    EXPECT_TRUE((nt::are_expressions_compatible_v<TestTerminal, TestScalar>));
    EXPECT_TRUE((nt::are_expressions_compatible_v<TestUnary, TestBinary>));

    // Different but compatible numeric types
    using FloatTerminal = TerminalExpression<MockContainer<float>>;
    EXPECT_TRUE((nt::are_expressions_compatible_v<TestTerminal, FloatTerminal>));

    // Incompatible with non-expressions (void value_type)
    EXPECT_FALSE((nt::are_expressions_compatible_v<TestTerminal, NotAnExpression>));
}

// TEST: Terminal count
TEST(ExpressionTraitsTest, TerminalCount) {
    EXPECT_EQ(nt::terminal_count_v<TestTerminal>, 1);
    EXPECT_EQ(nt::terminal_count_v<TestScalar>, 0);  // Scalar is not a terminal
    EXPECT_EQ(nt::terminal_count_v<TestUnary>, 1);   // Contains one terminal
    EXPECT_EQ(nt::terminal_count_v<TestBinary>, 2);  // Contains two terminals

    // Complex expression with mixed types
    using MixedBinary = BinaryExpression<ops::plus<double>, TestUnary, TestScalar>;
    EXPECT_EQ(nt::terminal_count_v<MixedBinary>, 1);  // Only unary contains a terminal
}

// TEST: Should materialize
TEST(ExpressionTraitsTest, ShouldMaterialize) {
    // Simple expressions don't need materialization
    EXPECT_FALSE(nt::should_materialize_v<TestTerminal>);
    EXPECT_FALSE(nt::should_materialize_v<TestUnary>);
    EXPECT_FALSE(nt::should_materialize_v<TestBinary>);

    // Create a deep expression tree
    using Deep1 = BinaryExpression<ops::plus<double>, TestBinary, TestBinary>;
    using Deep2 = BinaryExpression<ops::plus<double>, Deep1, Deep1>;
    using Deep3 = BinaryExpression<ops::plus<double>, Deep2, Deep2>;
    using Deep4 = BinaryExpression<ops::plus<double>, Deep3, Deep3>;

    // Deep4 should have depth > 5 and should be materialized
    EXPECT_GT(nt::expression_depth_v<Deep4>, 5);
    EXPECT_TRUE(nt::should_materialize_v<Deep4>);
}

// TEST: Optimization hints
TEST(ExpressionTraitsTest, OptimizationHints) {
    using hints = nt::optimization_hints<TestBinary>;

    EXPECT_TRUE(hints::use_simd);           // Vectorizable
    EXPECT_FALSE(hints::use_parallel);      // Not enough operations
    EXPECT_FALSE(hints::materialize);       // Not deep enough
    EXPECT_TRUE(hints::can_reorder);        // Plus is commutative
    EXPECT_TRUE(hints::fuse_operations);    // Shallow and no broadcasts

    // Create expression with many operations
    using Deep1 = BinaryExpression<ops::plus<double>, TestBinary, TestBinary>;
    using Deep2 = BinaryExpression<ops::plus<double>, Deep1, Deep1>;
    using hints2 = nt::optimization_hints<Deep2>;

    EXPECT_TRUE(hints2::use_simd);
    EXPECT_FALSE(hints2::materialize);  // Still not deep enough (depth = 4)
    EXPECT_TRUE(hints2::fuse_operations);  // depth <= 4 check passes and no broadcasts
}

// TEST: Evaluation order
TEST(ExpressionTraitsTest, EvaluationOrder) {
    using Order = nt::evaluation_order<TestBinary>::Order;

    // Both sides have same operation count (0), so LeftToRight
    EXPECT_EQ(nt::evaluation_order<TestBinary>::value, Order::LeftToRight);

    // Create asymmetric expression
    using AsymBinary = BinaryExpression<ops::plus<double>, TestUnary, TestTerminal>;
    // Left has 1 operation, right has 0, so RightToLeft
    EXPECT_EQ(nt::evaluation_order<AsymBinary>::value, Order::RightToLeft);

    // Create reverse asymmetric
    using RevAsymBinary = BinaryExpression<ops::plus<double>, TestTerminal, TestUnary>;
    // Left has 0 operations, right has 1, so LeftToRight
    EXPECT_EQ(nt::evaluation_order<RevAsymBinary>::value, Order::LeftToRight);
}

// TEST: Complex expression trees
TEST(ExpressionTraitsTest, ComplexExpressionTrees) {
    // Create a complex expression: (a + b) * sin(c) + d
    using Term1 = TerminalExpression<MockContainer<double>>;
    using Term2 = TerminalExpression<MockContainer<double>>;
    using Term3 = TerminalExpression<MockContainer<double>>;
    using Term4 = TerminalExpression<MockContainer<double>>;

    using Add = BinaryExpression<ops::plus<double>, Term1, Term2>;
    using Sin = UnaryExpression<Term3, ops::sin_op<>>;
    using Mul = BinaryExpression<ops::multiplies<double>, Add, Sin>;
    using Final = BinaryExpression<ops::plus<double>, Mul, Term4>;

    // Check properties
    EXPECT_EQ(nt::expression_category_v<Final>, nt::ExpressionCategory::Binary);
    EXPECT_EQ(nt::expression_depth_v<Final>, 4);
    EXPECT_EQ(nt::operation_count_v<Final>, 4);  // +, sin, *, +
    EXPECT_EQ(nt::terminal_count_v<Final>, 4);   // 4 terminals
    EXPECT_TRUE(nt::is_vectorizable_v<Final>);
    EXPECT_FALSE(nt::should_materialize_v<Final>);  // depth = 4, not > 5

    using props = nt::expression_properties<Final>;
    EXPECT_EQ(props::recommended_strategy, nt::EvaluationStrategy::Lazy);
}

// TEST: Expression with different value types
TEST(ExpressionTraitsTest, MixedValueTypes) {
    using IntContainer = MockContainer<int>;
    using FloatContainer = MockContainer<float>;
    using ComplexContainer = MockContainer<std::complex<double>>;

    using IntTerminal = TerminalExpression<IntContainer>;
    using FloatTerminal = TerminalExpression<FloatContainer>;
    using ComplexTerminal = TerminalExpression<ComplexContainer>;

    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<IntTerminal>, int>));
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<FloatTerminal>, float>));
    EXPECT_TRUE((std::is_same_v<nt::expression_value_type_t<ComplexTerminal>, std::complex<double>>));

    // Complex numbers are not vectorizable in our simple model
    EXPECT_TRUE(nt::is_vectorizable_v<IntTerminal>);
    EXPECT_TRUE(nt::is_vectorizable_v<FloatTerminal>);
    EXPECT_FALSE(nt::is_vectorizable_v<ComplexTerminal>);  // Not arithmetic type
}

// TEST: Edge cases
TEST(ExpressionTraitsTest, EdgeCases) {
    // Empty/minimal expressions
    EXPECT_EQ(nt::expression_depth_v<TestScalar>, 0);
    EXPECT_EQ(nt::operation_count_v<TestScalar>, 0);
    EXPECT_EQ(nt::terminal_count_v<TestScalar>, 0);

    // Self-referential types (if they existed) would need special handling
    // This is just to ensure the traits handle basic recursion correctly
    using Nested1 = UnaryExpression<TestTerminal, ops::negate<double>>;
    using Nested2 = UnaryExpression<Nested1, ops::negate<double>>;
    using Nested3 = UnaryExpression<Nested2, ops::negate<double>>;

    EXPECT_EQ(nt::expression_depth_v<Nested3>, 4);  // 3 unary + 1 terminal
    EXPECT_EQ(nt::operation_count_v<Nested3>, 3);   // 3 negations
    EXPECT_EQ(nt::terminal_count_v<Nested3>, 1);    // 1 terminal at the bottom
}

// Compile-time tests
namespace CompileTimeTests {
    // Verify trait values at compile time
    static_assert(nt::is_expression_v<TestTerminal>);
    static_assert(!nt::is_expression_v<NotAnExpression>);

    static_assert(nt::is_terminal_expression_v<TestTerminal>);
    static_assert(nt::is_unary_expression_v<TestUnary>);
    static_assert(nt::is_binary_expression_v<TestBinary>);
    static_assert(nt::is_scalar_expression_v<TestScalar>);

    static_assert(nt::expression_category_v<TestTerminal> == nt::ExpressionCategory::Terminal);
    static_assert(nt::expression_category_v<TestUnary> == nt::ExpressionCategory::Unary);
    static_assert(nt::expression_category_v<TestBinary> == nt::ExpressionCategory::Binary);

    static_assert(nt::expression_depth_v<TestTerminal> == 1);
    static_assert(nt::expression_depth_v<TestUnary> == 2);

    static_assert(nt::operation_count_v<TestTerminal> == 0);
    static_assert(nt::operation_count_v<TestUnary> == 1);

    static_assert(!nt::has_broadcast_v<TestBinary>);
    static_assert(nt::is_vectorizable_v<TestBinary>);
    static_assert(nt::is_parallel_safe_v<TestBinary>);

    static_assert(nt::terminal_count_v<TestTerminal> == 1);
    static_assert(nt::terminal_count_v<TestBinary> == 2);

    // Test expression properties
    using props = nt::expression_properties<TestBinary>;
    static_assert(props::is_expression);
    static_assert(props::category == nt::ExpressionCategory::Binary);
    static_assert(props::depth == 2);
    static_assert(props::can_vectorize);
}

} // anonymous namespace

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <array>

#include <traits/container_traits.h>
#include <traits/storage_traits.h>


// Test fixtures and mock types
namespace {

using namespace fem::numeric;
using namespace fem::numeric::traits;

// Mock container types for testing
class MockDenseContainer : public ContainerBase<MockDenseContainer, double, DynamicStorage<double>> {
public:
    using base_type = ContainerBase<MockDenseContainer, double, DynamicStorage<double>>;
    using value_type = double;
    using size_type = std::size_t;
    using iterator = double*;
    using shape_type = Shape;

    static constexpr size_t static_dimensions = 2;
    static constexpr MemoryLayout memory_layout = MemoryLayout::RowMajor;
    static constexpr size_t rank = 2;

    MockDenseContainer& operator+=(const MockDenseContainer&) { return *this; }
    MockDenseContainer& operator-=(const MockDenseContainer&) { return *this; }
    MockDenseContainer& operator*=(double) { return *this; }
    MockDenseContainer& operator/=(double) { return *this; }
};

class MockExpression : public ExpressionBase<MockExpression> {
public:
    using value_type = float;
    using size_type = std::size_t;

    size_type size() const { return 10; }
};

class MockView : public ViewBase<MockView> {
public:
    using value_type = int;
    using size_type = std::size_t;
    using iterator = int*;

    size_type size() const { return 5; }
};

// Container with static storage
class MockStaticContainer : public ContainerBase<MockStaticContainer, float, StaticStorage<float, 100>> {
public:
    using value_type = float;
    using size_type = std::size_t;
    using storage_type = StaticStorage<float, 100>;

    static constexpr size_t static_dimensions = 1;
    static constexpr bool use_parallel = true;
};

// Aligned container for SIMD
class alignas(32) MockSIMDContainer : public ContainerBase<MockSIMDContainer, double, AlignedStorage<double, 32>> {
public:
    using value_type = double;
    using size_type = std::size_t;
    using storage_type = AlignedStorage<double, 32>;

    static constexpr MemoryLayout memory_layout = MemoryLayout::RowMajor;
};

// Non-container type for negative testing
struct NotAContainer {
    int value;
};

// Container without arithmetic operations
class NoArithmeticContainer : public ContainerBase<NoArithmeticContainer, int, DynamicStorage<int>> {
public:
    using value_type = int;
    using size_type = std::size_t;
};

} // anonymous namespace

// Test type detection traits
TEST(ContainerTraitsTest, IsContainerBase) {
    EXPECT_TRUE(is_container_base_v<MockDenseContainer>);
    EXPECT_TRUE(is_container_base_v<MockStaticContainer>);
    EXPECT_TRUE(is_container_base_v<MockSIMDContainer>);
    EXPECT_FALSE(is_container_base_v<MockExpression>);
    EXPECT_FALSE(is_container_base_v<MockView>);
    EXPECT_FALSE(is_container_base_v<NotAContainer>);
    EXPECT_FALSE(is_container_base_v<int>);
    EXPECT_FALSE(is_container_base_v<std::vector<int>>);
}

TEST(ContainerTraitsTest, IsExpressionBase) {
    EXPECT_TRUE(is_expression_base_v<MockExpression>);
    EXPECT_FALSE(is_expression_base_v<MockDenseContainer>);
    EXPECT_FALSE(is_expression_base_v<MockView>);
    EXPECT_FALSE(is_expression_base_v<NotAContainer>);
    EXPECT_FALSE(is_expression_base_v<double>);
}

TEST(ContainerTraitsTest, IsViewBase) {
    EXPECT_TRUE(is_view_base_v<MockView>);
    EXPECT_FALSE(is_view_base_v<MockDenseContainer>);
    EXPECT_FALSE(is_view_base_v<MockExpression>);
    EXPECT_FALSE(is_view_base_v<NotAContainer>);
    EXPECT_FALSE((is_view_base_v<std::array<int, 5>>));
}

// Test extended container traits
TEST(ContainerTraitsTest, ExtendedTraitsCategory) {
    using dense_traits = extended_container_traits<MockDenseContainer>;
    using expr_traits = extended_container_traits<MockExpression>;
    using view_traits = extended_container_traits<MockView>;
    using unknown_traits = extended_container_traits<NotAContainer>;

    EXPECT_EQ(dense_traits::category, ContainerCategory::Dense);
    EXPECT_EQ(expr_traits::category, ContainerCategory::Expression);
    EXPECT_EQ(view_traits::category, ContainerCategory::View);
    EXPECT_EQ(unknown_traits::category, ContainerCategory::Unknown);
}

TEST(ContainerTraitsTest, ExtendedTraitsTypes) {
    using traits = extended_container_traits<MockDenseContainer>;

    EXPECT_TRUE((std::is_same_v<traits::value_type, double>));
    EXPECT_TRUE((std::is_same_v<traits::size_type, std::size_t>));
    EXPECT_TRUE((std::is_same_v<traits::storage_type, DynamicStorage<double>>));
    EXPECT_TRUE((std::is_same_v<traits::iterator_type, double*>));
}

TEST(ContainerTraitsTest, ExtendedTraitsProperties) {
    using dense_traits = extended_container_traits<MockDenseContainer>;

    EXPECT_TRUE(dense_traits::has_random_access);
    EXPECT_EQ(dense_traits::static_dimensions, 2);
    EXPECT_TRUE(dense_traits::has_static_shape);
    EXPECT_EQ(dense_traits::layout, MemoryLayout::RowMajor);
    EXPECT_TRUE(dense_traits::is_contiguous);
    EXPECT_TRUE(dense_traits::is_owning);
    EXPECT_FALSE(dense_traits::is_lazy);
}

TEST(ContainerTraitsTest, SIMDSupport) {
    using simd_traits = extended_container_traits<MockSIMDContainer>;
    using no_simd_traits = extended_container_traits<MockView>;

    // MockSIMDContainer should support SIMD (aligned, contiguous, numeric type)
    EXPECT_TRUE(simd_traits::is_contiguous);
    // Note: The actual SIMD support check depends on alignment which may vary

    // View should not support SIMD (not contiguous)
    EXPECT_FALSE(no_simd_traits::is_contiguous);
    EXPECT_FALSE(no_simd_traits::supports_simd);
}

TEST(ContainerTraitsTest, LazyEvaluation) {
    using expr_traits = extended_container_traits<MockExpression>;
    using dense_traits = extended_container_traits<MockDenseContainer>;

    EXPECT_TRUE(expr_traits::is_lazy);
    EXPECT_FALSE(dense_traits::is_lazy);
}

// Test arithmetic operations support
TEST(ContainerTraitsTest, SupportsArithmeticOps) {
    EXPECT_TRUE(supports_arithmetic_ops_v<MockDenseContainer>);
    EXPECT_FALSE(supports_arithmetic_ops_v<NoArithmeticContainer>);
    EXPECT_FALSE(supports_arithmetic_ops_v<NotAContainer>);
    EXPECT_FALSE(supports_arithmetic_ops_v<int>);
}

// Test evaluation strategy
TEST(ContainerTraitsTest, EvaluationStrategy) {
    using expr_strategy = evaluation_strategy<MockExpression>;
    using static_strategy = evaluation_strategy<MockStaticContainer>;
    using dense_strategy = evaluation_strategy<MockDenseContainer>;

    EXPECT_EQ(expr_strategy::value, evaluation_strategy<MockExpression>::Lazy);
    EXPECT_EQ(static_strategy::value, evaluation_strategy<MockStaticContainer>::Parallel);
    // Dense container strategy depends on SIMD support
}

// Test container rank
TEST(ContainerTraitsTest, ContainerRank) {
    EXPECT_EQ(container_rank_v<MockDenseContainer>, 2);
    EXPECT_EQ(container_rank_v<MockStaticContainer>, 1);
    EXPECT_TRUE(container_rank<MockDenseContainer>::is_static);

    // Test with type without rank
    EXPECT_EQ(container_rank_v<NotAContainer>, 0);
    EXPECT_FALSE(container_rank<NotAContainer>::is_static);
}

// Test storage info
TEST(ContainerTraitsTest, ContainerStorageInfo) {
    using dense_storage = container_storage_info<MockDenseContainer>;
    using static_storage = container_storage_info<MockStaticContainer>;

    EXPECT_EQ(dense_storage::element_size, sizeof(double));
    EXPECT_EQ(static_storage::element_size, sizeof(float));

    EXPECT_TRUE(dense_storage::is_dynamic);
    EXPECT_FALSE(static_storage::is_dynamic);

    // Test bytes calculation
    EXPECT_EQ(dense_storage::bytes_for(10), 10 * sizeof(double));
    EXPECT_EQ(static_storage::bytes_for(10), 10 * sizeof(float));
}

// Test aligned storage calculation
TEST(ContainerTraitsTest, AlignedStorageCalculation) {
    // Create a mock type with specific alignment requirements
    struct AlignedContainer : ContainerBase<AlignedContainer, double, AlignedStorage<double, 64>> {
        using value_type = double;
        using storage_type = AlignedStorage<double, 64>;
    };

    using aligned_info = container_storage_info<AlignedContainer>;

    EXPECT_EQ(aligned_info::alignment, 64);
    EXPECT_TRUE(aligned_info::is_aligned);

    // Test that bytes_for rounds up to alignment boundary
    size_t bytes_5 = aligned_info::bytes_for(5);
    EXPECT_EQ(bytes_5 % 64, 0);  // Should be aligned to 64 bytes
    EXPECT_GE(bytes_5, 5 * sizeof(double));  // Should be at least 5 doubles
}

// Test optimal container selection
TEST(ContainerTraitsTest, OptimalContainerSelection) {
    // Dynamic size selection
    using dynamic_optimal = optimal_container_t<double, 0, false>;
    EXPECT_TRUE((std::is_same_v<
        typename dynamic_optimal::storage_type,
        DynamicStorage<double>
    >));

    // Small static size selection (should use static storage)
    using small_optimal = optimal_container_t<char, 32, false>;
    EXPECT_TRUE((std::is_same_v<
        typename small_optimal::storage_type,
        StaticStorage<char, 32>
    >));

    // SIMD-enabled selection
    using simd_optimal = optimal_container_t<float, 128, true>;
    EXPECT_TRUE((std::is_same_v<
        typename simd_optimal::storage_type,
        AlignedStorage<float, 32>
    >));
}

// Test shape type extraction
TEST(ContainerTraitsTest, ShapeTypeExtraction) {
    using dense_shape = shape_type<MockDenseContainer>;
    EXPECT_TRUE((std::is_same_v<dense_shape, Shape>));

    // For types without shape_type, should default to Shape
    struct NoShapeContainer {
        using value_type = int;
        using size_type = size_t;
    };
    using default_shape = shape_type<NoShapeContainer>;
    EXPECT_TRUE((std::is_same_v<default_shape, Shape>));
}

// Compile-time tests using static_assert
namespace CompileTimeTests {
    // Verify trait values at compile time
    static_assert(is_container_base_v<MockDenseContainer>);
    static_assert(!is_container_base_v<MockExpression>);
    static_assert(is_expression_base_v<MockExpression>);
    static_assert(is_view_base_v<MockView>);

    static_assert(extended_container_traits<MockDenseContainer>::static_dimensions == 2);
    static_assert(extended_container_traits<MockDenseContainer>::has_static_shape);
    static_assert(extended_container_traits<MockExpression>::is_lazy);

    static_assert(container_rank_v<MockDenseContainer> == 2);
    static_assert(supports_arithmetic_ops_v<MockDenseContainer>);
}

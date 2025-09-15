/**
 * @file test_block_vector.cpp
 * @brief Comprehensive unit tests for the BlockVector class
 */

#include <gtest/gtest.h>
#include <sstream>
#include <complex>
#include "../../include/core/block_vector.h"

using namespace fem::numeric;

namespace {

// Test fixture for block vector tests
class BlockVectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// === Constructor Tests ===

TEST_F(BlockVectorTest, DefaultConstructor) {
    BlockVector<double> bv;
    
    EXPECT_EQ(bv.num_blocks(), 0);
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

TEST_F(BlockVectorTest, ConstructorWithBlockSizes) {
    BlockVector<double> bv({3, 4, 2});
    
    EXPECT_EQ(bv.num_blocks(), 3);
    EXPECT_EQ(bv.size(), 9);
    EXPECT_FALSE(bv.empty());
    
    EXPECT_TRUE(bv.has_block("block_0"));
    EXPECT_TRUE(bv.has_block("block_1"));
    EXPECT_TRUE(bv.has_block("block_2"));
    
    EXPECT_EQ(bv.block_size("block_0"), 3);
    EXPECT_EQ(bv.block_size("block_1"), 4);
    EXPECT_EQ(bv.block_size("block_2"), 2);
}

TEST_F(BlockVectorTest, ConstructorWithNamedBlocks) {
    BlockVector<double> bv({{"velocity", 6}, {"pressure", 2}});
    
    EXPECT_EQ(bv.num_blocks(), 2);
    EXPECT_EQ(bv.size(), 8);
    
    EXPECT_TRUE(bv.has_block("velocity"));
    EXPECT_TRUE(bv.has_block("pressure"));
    EXPECT_FALSE(bv.has_block("temperature"));
    
    EXPECT_EQ(bv.block_size("velocity"), 6);
    EXPECT_EQ(bv.block_size("pressure"), 2);
}

TEST_F(BlockVectorTest, ConstructorWithStorageMode) {
    BlockVector<double> bv_contiguous({2, 3}, BlockVector<double>::StorageMode::Contiguous);
    BlockVector<double> bv_separate({2, 3}, BlockVector<double>::StorageMode::Separate);
    
    EXPECT_EQ(bv_contiguous.size(), 5);
    EXPECT_EQ(bv_separate.size(), 5);
    
    // Both should behave the same externally
    bv_contiguous[0] = 1.0;
    bv_separate[0] = 1.0;
    
    EXPECT_EQ(bv_contiguous[0], 1.0);
    EXPECT_EQ(bv_separate[0], 1.0);
}

TEST_F(BlockVectorTest, CopyConstructor) {
    BlockVector<double> original({{"u", 3}, {"p", 1}});
    original.block("u")[0] = 1.0;
    original.block("u")[1] = 2.0;
    original.block("p")[0] = 3.0;
    
    BlockVector<double> copy(original);
    
    EXPECT_EQ(copy.num_blocks(), original.num_blocks());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.block("u")[0], 1.0);
    EXPECT_EQ(copy.block("u")[1], 2.0);
    EXPECT_EQ(copy.block("p")[0], 3.0);
    
    // Modify original to ensure deep copy
    original.block("u")[0] = 999.0;
    EXPECT_EQ(copy.block("u")[0], 1.0);
}

TEST_F(BlockVectorTest, MoveConstructor) {
    BlockVector<double> original({{"test", 2}});
    original.block("test")[0] = 42.0;
    
    BlockVector<double> moved(std::move(original));
    
    EXPECT_EQ(moved.num_blocks(), 1);
    EXPECT_EQ(moved.size(), 2);
    EXPECT_EQ(moved.block("test")[0], 42.0);
}

// === Block Management Tests ===

TEST_F(BlockVectorTest, AddBlock) {
    BlockVector<double> bv;
    
    bv.add_block("velocity", 3);
    EXPECT_EQ(bv.num_blocks(), 1);
    EXPECT_EQ(bv.size(), 3);
    EXPECT_TRUE(bv.has_block("velocity"));
    EXPECT_EQ(bv.block_size("velocity"), 3);
    
    bv.add_block("pressure", 1);
    EXPECT_EQ(bv.num_blocks(), 2);
    EXPECT_EQ(bv.size(), 4);
    EXPECT_TRUE(bv.has_block("pressure"));
    EXPECT_EQ(bv.block_size("pressure"), 1);
    
    // Test duplicate name
    EXPECT_THROW(bv.add_block("velocity", 2), std::invalid_argument);
}

TEST_F(BlockVectorTest, RemoveBlock) {
    BlockVector<double> bv({{"a", 2}, {"b", 3}, {"c", 1}});
    
    EXPECT_EQ(bv.num_blocks(), 3);
    EXPECT_EQ(bv.size(), 6);
    
    bv.remove_block("b");
    
    EXPECT_EQ(bv.num_blocks(), 2);
    EXPECT_EQ(bv.size(), 3);
    EXPECT_TRUE(bv.has_block("a"));
    EXPECT_FALSE(bv.has_block("b"));
    EXPECT_TRUE(bv.has_block("c"));
    
    // Test removing non-existent block
    EXPECT_THROW(bv.remove_block("nonexistent"), std::invalid_argument);
}

TEST_F(BlockVectorTest, BlockNames) {
    BlockVector<double> bv({{"velocity", 3}, {"pressure", 1}, {"temperature", 2}});
    
    auto names = bv.block_names();
    EXPECT_EQ(names.size(), 3);
    
    // Check that all expected names are present (order may vary)
    EXPECT_TRUE(std::find(names.begin(), names.end(), "velocity") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "pressure") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "temperature") != names.end());
}

TEST_F(BlockVectorTest, BlockOffset) {
    BlockVector<double> bv({{"a", 2}, {"b", 3}, {"c", 1}});
    
    EXPECT_EQ(bv.block_offset("a"), 0);
    EXPECT_EQ(bv.block_offset("b"), 2);
    EXPECT_EQ(bv.block_offset("c"), 5);
    
    EXPECT_THROW(bv.block_offset("nonexistent"), std::invalid_argument);
}

// === Block Access Tests ===

TEST_F(BlockVectorTest, BlockAccessByName) {
    BlockVector<double> bv({{"u", 3}, {"p", 2}});
    
    auto u_block = bv.block("u");
    u_block[0] = 1.0;
    u_block[1] = 2.0;
    u_block[2] = 3.0;
    
    auto p_block = bv.block("p");
    p_block[0] = 4.0;
    p_block[1] = 5.0;
    
    EXPECT_EQ(u_block[0], 1.0);
    EXPECT_EQ(u_block[1], 2.0);
    EXPECT_EQ(u_block[2], 3.0);
    EXPECT_EQ(p_block[0], 4.0);
    EXPECT_EQ(p_block[1], 5.0);
    
    // Test const access
    const auto& const_bv = bv;
    auto const_u_block = const_bv.block("u");
    EXPECT_EQ(const_u_block[0], 1.0);
}

TEST_F(BlockVectorTest, BlockAccessByIndex) {
    BlockVector<double> bv({{"first", 2}, {"second", 3}});
    
    auto block0 = bv.block(0);
    auto block1 = bv.block(1);
    
    block0[0] = 10.0;
    block1[2] = 20.0;
    
    EXPECT_EQ(block0[0], 10.0);
    EXPECT_EQ(block1[2], 20.0);
    
    // Test out of range
    EXPECT_THROW(bv.block(2), std::out_of_range);
}

TEST_F(BlockVectorTest, NonExistentBlock) {
    BlockVector<double> bv({{"existing", 3}});
    
    EXPECT_THROW(bv.block("nonexistent"), std::invalid_argument);
}

// === Element Access Tests ===

TEST_F(BlockVectorTest, GlobalElementAccess) {
    BlockVector<double> bv({{"a", 2}, {"b", 3}});
    
    // Set values through global indexing
    bv[0] = 1.0;  // First element of block "a"
    bv[1] = 2.0;  // Second element of block "a"
    bv[2] = 3.0;  // First element of block "b"
    bv[3] = 4.0;  // Second element of block "b"
    bv[4] = 5.0;  // Third element of block "b"
    
    // Verify through block access
    EXPECT_EQ(bv.block("a")[0], 1.0);
    EXPECT_EQ(bv.block("a")[1], 2.0);
    EXPECT_EQ(bv.block("b")[0], 3.0);
    EXPECT_EQ(bv.block("b")[1], 4.0);
    EXPECT_EQ(bv.block("b")[2], 5.0);
    
    // Verify through global access
    EXPECT_EQ(bv[0], 1.0);
    EXPECT_EQ(bv[4], 5.0);
}

TEST_F(BlockVectorTest, BoundsChecking) {
    BlockVector<double> bv({{"test", 3}});
    
    EXPECT_NO_THROW(bv.at(0));
    EXPECT_NO_THROW(bv.at(2));
    EXPECT_THROW(bv.at(3), std::out_of_range);
    EXPECT_THROW(bv.at(100), std::out_of_range);
}

TEST_F(BlockVectorTest, DataAccess) {
    BlockVector<double> bv({{"test", 3}}, BlockVector<double>::StorageMode::Contiguous);
    
    auto* data = bv.data();
    EXPECT_NE(data, nullptr);
    
    data[0] = 99.0;
    EXPECT_EQ(bv[0], 99.0);
    EXPECT_EQ(bv.block("test")[0], 99.0);
    
    // Separate storage mode should throw
    BlockVector<double> bv_separate({{"test", 3}}, BlockVector<double>::StorageMode::Separate);
    EXPECT_THROW(bv_separate.data(), std::logic_error);
}

// === Vector Operations Tests ===

TEST_F(BlockVectorTest, Fill) {
    BlockVector<double> bv({{"a", 2}, {"b", 3}});
    
    bv.fill(7.5);
    
    for (size_t i = 0; i < bv.size(); ++i) {
        EXPECT_EQ(bv[i], 7.5);
    }
    
    EXPECT_EQ(bv.block("a")[0], 7.5);
    EXPECT_EQ(bv.block("b")[2], 7.5);
}

TEST_F(BlockVectorTest, Zero) {
    BlockVector<double> bv({{"test", 3}});
    bv.fill(42.0);
    
    bv.zero();
    
    for (size_t i = 0; i < bv.size(); ++i) {
        EXPECT_EQ(bv[i], 0.0);
    }
}

// === Arithmetic Operations Tests ===

TEST_F(BlockVectorTest, Addition) {
    BlockVector<double> a({{"u", 2}, {"p", 1}});
    BlockVector<double> b({{"u", 2}, {"p", 1}});
    
    a[0] = 1.0; a[1] = 2.0; a[2] = 3.0;
    b[0] = 4.0; b[1] = 5.0; b[2] = 6.0;
    
    a += b;
    
    EXPECT_EQ(a[0], 5.0);
    EXPECT_EQ(a[1], 7.0);
    EXPECT_EQ(a[2], 9.0);
    
    // Test block structure consistency
    EXPECT_EQ(a.block("u")[0], 5.0);
    EXPECT_EQ(a.block("u")[1], 7.0);
    EXPECT_EQ(a.block("p")[0], 9.0);
}

TEST_F(BlockVectorTest, AdditionIncompatibleStructure) {
    BlockVector<double> a({{"u", 2}, {"p", 1}});
    BlockVector<double> b({{"u", 3}, {"p", 1}});  // Different block size
    
    EXPECT_THROW(a += b, std::invalid_argument);
    
    BlockVector<double> c({{"u", 2}});  // Different number of blocks
    EXPECT_THROW(a += c, std::invalid_argument);
}

TEST_F(BlockVectorTest, Subtraction) {
    BlockVector<double> a({{"test", 3}});
    BlockVector<double> b({{"test", 3}});
    
    a[0] = 10.0; a[1] = 8.0; a[2] = 6.0;
    b[0] = 1.0; b[1] = 2.0; b[2] = 3.0;
    
    a -= b;
    
    EXPECT_EQ(a[0], 9.0);
    EXPECT_EQ(a[1], 6.0);
    EXPECT_EQ(a[2], 3.0);
}

TEST_F(BlockVectorTest, ScalarMultiplication) {
    BlockVector<double> bv({{"test", 3}});
    
    bv[0] = 2.0; bv[1] = 4.0; bv[2] = 6.0;
    
    bv *= 2.5;
    
    EXPECT_EQ(bv[0], 5.0);
    EXPECT_EQ(bv[1], 10.0);
    EXPECT_EQ(bv[2], 15.0);
}

TEST_F(BlockVectorTest, ScalarDivision) {
    BlockVector<double> bv({{"test", 3}});
    
    bv[0] = 10.0; bv[1] = 20.0; bv[2] = 30.0;
    
    bv /= 10.0;
    
    EXPECT_EQ(bv[0], 1.0);
    EXPECT_EQ(bv[1], 2.0);
    EXPECT_EQ(bv[2], 3.0);
}

// === Norms Tests ===

TEST_F(BlockVectorTest, Norm2) {
    BlockVector<double> bv({{"test", 2}});
    
    bv[0] = 3.0;
    bv[1] = 4.0;
    
    double norm = bv.norm2();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(BlockVectorTest, MaxNorm) {
    BlockVector<double> bv({{"a", 2}, {"b", 2}});
    
    bv[0] = -7.0;
    bv[1] = 3.0;
    bv[2] = 5.0;
    bv[3] = -2.0;
    
    double norm = bv.max_norm();
    EXPECT_EQ(norm, 7.0);
}

// === Conversion Tests ===

TEST_F(BlockVectorTest, ToVector) {
    BlockVector<double> bv({{"a", 2}, {"b", 2}}, BlockVector<double>::StorageMode::Contiguous);
    
    bv[0] = 1.0; bv[1] = 2.0; bv[2] = 3.0; bv[3] = 4.0;
    
    auto vec = bv.to_vector();
    
    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec[0], 1.0);
    EXPECT_EQ(vec[1], 2.0);
    EXPECT_EQ(vec[2], 3.0);
    EXPECT_EQ(vec[3], 4.0);
    
    // Test with separate storage should fail
    BlockVector<double> bv_separate({{"test", 2}}, BlockVector<double>::StorageMode::Separate);
    EXPECT_THROW(bv_separate.to_vector(), std::logic_error);
}

// === Non-member Operations Tests ===

TEST_F(BlockVectorTest, NonMemberAddition) {
    BlockVector<double> a({{"test", 2}});
    BlockVector<int> b({{"test", 2}});
    
    a[0] = 1.5; a[1] = 2.5;
    b[0] = 1; b[1] = 2;
    
    auto result = a + b;
    
    // Check type promotion
    static_assert(std::is_same_v<decltype(result), BlockVector<double>>);
    
    EXPECT_NEAR(result[0], 2.5, 1e-10);
    EXPECT_NEAR(result[1], 4.5, 1e-10);
}

TEST_F(BlockVectorTest, NonMemberSubtraction) {
    BlockVector<double> a({{"test", 2}});
    BlockVector<float> b({{"test", 2}});
    
    a[0] = 5.0; a[1] = 10.0;
    b[0] = 1.0f; b[1] = 3.0f;
    
    auto result = a - b;
    
    EXPECT_NEAR(result[0], 4.0, 1e-6);
    EXPECT_NEAR(result[1], 7.0, 1e-6);
}

TEST_F(BlockVectorTest, ScalarMultiplicationNonMember) {
    BlockVector<double> bv({{"test", 2}});
    
    bv[0] = 3.0; bv[1] = 4.0;
    
    auto result1 = 2.0 * bv;
    auto result2 = bv * 2.0;
    
    EXPECT_EQ(result1[0], 6.0);
    EXPECT_EQ(result1[1], 8.0);
    EXPECT_EQ(result2[0], 6.0);
    EXPECT_EQ(result2[1], 8.0);
}

// === Complex Numbers Tests ===

TEST_F(BlockVectorTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    BlockVector<Complex> bv({{"test", 2}});
    
    bv.block("test")[0] = Complex(1, 2);
    bv.block("test")[1] = Complex(3, -1);
    
    bv *= Complex(2, 0);
    
    EXPECT_EQ(bv.block("test")[0], Complex(2, 4));
    EXPECT_EQ(bv.block("test")[1], Complex(6, -2));
}

// === Multi-Physics Example Tests ===

TEST_F(BlockVectorTest, FluidStructureInteraction) {
    BlockVector<double> solution({
        {"velocity_x", 100},
        {"velocity_y", 100},
        {"pressure", 50},
        {"displacement_x", 25},
        {"displacement_y", 25}
    });
    
    EXPECT_EQ(solution.num_blocks(), 5);
    EXPECT_EQ(solution.size(), 300);
    
    // Set fluid velocity
    auto vel_x = solution.block("velocity_x");
    auto vel_y = solution.block("velocity_y");
    vel_x.fill(1.0);
    vel_y.fill(0.5);
    
    // Set pressure
    auto pressure = solution.block("pressure");
    pressure.fill(0.0);
    
    // Set displacement
    auto disp_x = solution.block("displacement_x");
    auto disp_y = solution.block("displacement_y");
    disp_x.fill(0.01);
    disp_y.fill(0.02);
    
    // Verify structure
    EXPECT_EQ(solution.block_size("velocity_x"), 100);
    EXPECT_EQ(solution.block_size("pressure"), 50);
    EXPECT_EQ(solution.block_size("displacement_x"), 25);
    
    EXPECT_EQ(solution.block_offset("velocity_x"), 0);
    EXPECT_EQ(solution.block_offset("velocity_y"), 100);
    EXPECT_EQ(solution.block_offset("pressure"), 200);
    EXPECT_EQ(solution.block_offset("displacement_x"), 250);
    EXPECT_EQ(solution.block_offset("displacement_y"), 275);
}

// === Stream Output Tests ===

TEST_F(BlockVectorTest, StreamOutput) {
    BlockVector<double> bv({{"u", 2}, {"p", 1}});
    
    bv.block("u")[0] = 1.5;
    bv.block("u")[1] = 2.5;
    bv.block("p")[0] = 3.5;
    
    std::ostringstream oss;
    oss << bv;
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("BlockVector") != std::string::npos);
    EXPECT_TRUE(output.find("3 total") != std::string::npos);
    EXPECT_TRUE(output.find("u") != std::string::npos);
    EXPECT_TRUE(output.find("p") != std::string::npos);
    EXPECT_TRUE(output.find("1.5") != std::string::npos);
    EXPECT_TRUE(output.find("2.5") != std::string::npos);
    EXPECT_TRUE(output.find("3.5") != std::string::npos);
}

// === Error Handling Tests ===

TEST_F(BlockVectorTest, ErrorCases) {
    BlockVector<double> bv;
    
    // Access non-existent block
    EXPECT_THROW(bv.block("nonexistent"), std::invalid_argument);
    EXPECT_THROW(bv.block_size("nonexistent"), std::invalid_argument);
    EXPECT_THROW(bv.block_offset("nonexistent"), std::invalid_argument);
    
    // Out of range block index
    EXPECT_THROW(bv.block(0), std::out_of_range);
    
    // Out of range element access
    bv.add_block("test", 2);
    EXPECT_THROW(bv.at(2), std::out_of_range);
}

// === Performance Tests ===

TEST_F(BlockVectorTest, LargeBlockVector) {
    std::vector<std::pair<std::string, size_t>> large_blocks = {
        {"field1", 10000},
        {"field2", 10000},
        {"field3", 5000}
    };
    
    EXPECT_NO_THROW({
        BlockVector<double> large_bv(large_blocks);
        
        EXPECT_EQ(large_bv.size(), 25000);
        EXPECT_EQ(large_bv.num_blocks(), 3);
        
        // Operations should work efficiently
        large_bv.fill(1.0);
        large_bv *= 2.0;
        
        EXPECT_EQ(large_bv[0], 2.0);
        EXPECT_EQ(large_bv[24999], 2.0);
    });
}

// === Edge Cases ===

TEST_F(BlockVectorTest, EmptyBlocks) {
    BlockVector<double> bv;
    
    bv.add_block("empty", 0);
    EXPECT_EQ(bv.num_blocks(), 1);
    EXPECT_EQ(bv.size(), 0);
    EXPECT_EQ(bv.block_size("empty"), 0);
}

TEST_F(BlockVectorTest, SingleElementBlocks) {
    BlockVector<double> bv({{"a", 1}, {"b", 1}, {"c", 1}});
    
    EXPECT_EQ(bv.num_blocks(), 3);
    EXPECT_EQ(bv.size(), 3);
    
    bv.block("a")[0] = 1.0;
    bv.block("b")[0] = 2.0;
    bv.block("c")[0] = 3.0;
    
    EXPECT_EQ(bv[0], 1.0);
    EXPECT_EQ(bv[1], 2.0);
    EXPECT_EQ(bv[2], 3.0);
}

} // namespace
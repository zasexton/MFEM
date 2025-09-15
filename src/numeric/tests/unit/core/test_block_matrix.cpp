/**
 * @file test_block_matrix.cpp
 * @brief Comprehensive unit tests for the BlockMatrix class
 */

#include <gtest/gtest.h>
#include <sstream>
#include <complex>
#include <core/block_matrix.h>

using namespace fem::numeric;

namespace {

// Test fixture for block matrix tests
class BlockMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// === Constructor Tests ===

TEST_F(BlockMatrixTest, DefaultConstructor) {
    BlockMatrix<double> bm;
    
    EXPECT_EQ(bm.num_block_rows(), 0);
    EXPECT_EQ(bm.num_block_cols(), 0);
    EXPECT_EQ(bm.rows(), 0);
    EXPECT_EQ(bm.cols(), 0);
    EXPECT_EQ(bm.size(), 0);
    EXPECT_TRUE(bm.empty());
}

TEST_F(BlockMatrixTest, SymmetricConstructor) {
    std::vector<std::string> names = {"u", "v", "p"};
    std::vector<size_t> sizes = {3, 3, 1};
    
    BlockMatrix<double> bm(names, sizes);
    
    EXPECT_EQ(bm.num_block_rows(), 3);
    EXPECT_EQ(bm.num_block_cols(), 3);
    EXPECT_EQ(bm.rows(), 7);
    EXPECT_EQ(bm.cols(), 7);
    EXPECT_EQ(bm.size(), 49);
    EXPECT_FALSE(bm.empty());
    EXPECT_TRUE(bm.is_square());
    
    EXPECT_EQ(bm.row_block_size("u"), 3);
    EXPECT_EQ(bm.col_block_size("u"), 3);
    EXPECT_EQ(bm.row_block_size("p"), 1);
    EXPECT_EQ(bm.col_block_size("p"), 1);
}

TEST_F(BlockMatrixTest, GeneralConstructor) {
    std::vector<std::string> row_names = {"displacement", "velocity"};
    std::vector<size_t> row_sizes = {6, 6};
    std::vector<std::string> col_names = {"force", "acceleration", "pressure"};
    std::vector<size_t> col_sizes = {6, 6, 1};
    
    BlockMatrix<double> bm(row_names, row_sizes, col_names, col_sizes);
    
    EXPECT_EQ(bm.num_block_rows(), 2);
    EXPECT_EQ(bm.num_block_cols(), 3);
    EXPECT_EQ(bm.rows(), 12);
    EXPECT_EQ(bm.cols(), 13);
    EXPECT_EQ(bm.size(), 156);
    EXPECT_FALSE(bm.is_square());
}

TEST_F(BlockMatrixTest, CopyConstructor) {
    BlockMatrix<double> original({"u", "p"}, {2, 1});
    
    // Set some values
    original.block("u", "u")(0, 0) = 1.0;
    original.block("u", "p")(0, 0) = 2.0;
    original.block("p", "u")(0, 0) = 3.0;
    original.block("p", "p")(0, 0) = 4.0;
    
    BlockMatrix<double> copy(original);
    
    EXPECT_EQ(copy.num_block_rows(), original.num_block_rows());
    EXPECT_EQ(copy.num_block_cols(), original.num_block_cols());
    
    // Verify deep copy
    EXPECT_EQ(copy.block("u", "u")(0, 0), 1.0);
    EXPECT_EQ(copy.block("u", "p")(0, 0), 2.0);
    EXPECT_EQ(copy.block("p", "u")(0, 0), 3.0);
    EXPECT_EQ(copy.block("p", "p")(0, 0), 4.0);
    
    // Modify original to ensure deep copy
    original.block("u", "u")(0, 0) = 999.0;
    EXPECT_EQ(copy.block("u", "u")(0, 0), 1.0);
}

TEST_F(BlockMatrixTest, MoveConstructor) {
    BlockMatrix<double> original({"test"}, {2});
    original.block("test", "test")(0, 0) = 42.0;
    
    BlockMatrix<double> moved(std::move(original));
    
    EXPECT_EQ(moved.num_block_rows(), 1);
    EXPECT_EQ(moved.num_block_cols(), 1);
    EXPECT_EQ(moved.block("test", "test")(0, 0), 42.0);
}

// === Block Structure Management Tests ===

TEST_F(BlockMatrixTest, SetSymmetricBlockStructure) {
    BlockMatrix<double> bm;
    
    bm.set_block_structure({"velocity", "pressure"}, {6, 2});
    
    EXPECT_EQ(bm.num_block_rows(), 2);
    EXPECT_EQ(bm.num_block_cols(), 2);
    EXPECT_EQ(bm.rows(), 8);
    EXPECT_EQ(bm.cols(), 8);
    EXPECT_TRUE(bm.is_square());
    
    auto row_names = bm.row_block_names();
    auto col_names = bm.col_block_names();
    
    EXPECT_EQ(row_names.size(), 2);
    EXPECT_EQ(col_names.size(), 2);
    EXPECT_TRUE(std::find(row_names.begin(), row_names.end(), "velocity") != row_names.end());
    EXPECT_TRUE(std::find(col_names.begin(), col_names.end(), "pressure") != col_names.end());
}

TEST_F(BlockMatrixTest, SetGeneralBlockStructure) {
    BlockMatrix<double> bm;
    
    std::vector<std::string> row_names = {"row1", "row2"};
    std::vector<size_t> row_sizes = {3, 2};
    std::vector<std::string> col_names = {"col1", "col2", "col3"};
    std::vector<size_t> col_sizes = {2, 2, 1};
    
    bm.set_block_structure(row_names, row_sizes, col_names, col_sizes);
    
    EXPECT_EQ(bm.num_block_rows(), 2);
    EXPECT_EQ(bm.num_block_cols(), 3);
    EXPECT_EQ(bm.rows(), 5);
    EXPECT_EQ(bm.cols(), 5);
    EXPECT_TRUE(bm.is_square());  // Happens to be square
}

TEST_F(BlockMatrixTest, InvalidBlockStructure) {
    BlockMatrix<double> bm;
    
    // Mismatched sizes
    EXPECT_THROW(bm.set_block_structure({"a", "b"}, {1}), std::invalid_argument);
    EXPECT_THROW(bm.set_block_structure({"a"}, {1}, {"b", "c"}, {1}), std::invalid_argument);
}

TEST_F(BlockMatrixTest, BlockSizes) {
    BlockMatrix<double> bm({"u", "v", "p"}, {3, 3, 1});
    
    EXPECT_EQ(bm.row_block_size("u"), 3);
    EXPECT_EQ(bm.row_block_size("v"), 3);
    EXPECT_EQ(bm.row_block_size("p"), 1);
    
    EXPECT_EQ(bm.col_block_size("u"), 3);
    EXPECT_EQ(bm.col_block_size("v"), 3);
    EXPECT_EQ(bm.col_block_size("p"), 1);
    
    EXPECT_THROW(bm.row_block_size("nonexistent"), std::invalid_argument);
    EXPECT_THROW(bm.col_block_size("nonexistent"), std::invalid_argument);
}

// === Block Access Tests ===

TEST_F(BlockMatrixTest, BlockAccessByName) {
    BlockMatrix<double> bm({"u", "p"}, {2, 1});
    
    // Access blocks (this should create them)
    auto& K_uu = bm.block("u", "u");
    auto& K_up = bm.block("u", "p");
    auto& K_pu = bm.block("p", "u");
    auto& K_pp = bm.block("p", "p");
    
    EXPECT_EQ(K_uu.rows(), 2);
    EXPECT_EQ(K_uu.cols(), 2);
    EXPECT_EQ(K_up.rows(), 2);
    EXPECT_EQ(K_up.cols(), 1);
    EXPECT_EQ(K_pu.rows(), 1);
    EXPECT_EQ(K_pu.cols(), 2);
    EXPECT_EQ(K_pp.rows(), 1);
    EXPECT_EQ(K_pp.cols(), 1);
    
    // Set values
    K_uu(0, 0) = 1.0; K_uu(0, 1) = 2.0;
    K_uu(1, 0) = 3.0; K_uu(1, 1) = 4.0;
    K_up(0, 0) = 5.0; K_up(1, 0) = 6.0;
    K_pu(0, 0) = 7.0; K_pu(0, 1) = 8.0;
    K_pp(0, 0) = 9.0;
    
    // Verify values
    EXPECT_EQ(K_uu(0, 0), 1.0);
    EXPECT_EQ(K_uu(1, 1), 4.0);
    EXPECT_EQ(K_up(1, 0), 6.0);
    EXPECT_EQ(K_pu(0, 1), 8.0);
    EXPECT_EQ(K_pp(0, 0), 9.0);
}

TEST_F(BlockMatrixTest, BlockAccessByIndex) {
    BlockMatrix<double> bm({"first", "second"}, {2, 1});
    
    auto& block00 = bm.block(0, 0);
    auto& block01 = bm.block(0, 1);
    auto& block10 = bm.block(1, 0);
    auto& block11 = bm.block(1, 1);
    
    EXPECT_EQ(block00.rows(), 2);
    EXPECT_EQ(block00.cols(), 2);
    EXPECT_EQ(block01.rows(), 2);
    EXPECT_EQ(block01.cols(), 1);
    
    block00(0, 0) = 10.0;
    block11(0, 0) = 20.0;
    
    EXPECT_EQ(block00(0, 0), 10.0);
    EXPECT_EQ(block11(0, 0), 20.0);
    
    // Test out of range
    EXPECT_THROW(bm.block(2, 0), std::out_of_range);
    EXPECT_THROW(bm.block(0, 2), std::out_of_range);
}

TEST_F(BlockMatrixTest, ConstBlockAccess) {
    BlockMatrix<double> bm({"test"}, {2});
    bm.block("test", "test")(0, 0) = 42.0;
    
    const auto& const_bm = bm;
    
    EXPECT_NO_THROW({
        const auto& const_block = const_bm.block("test", "test");
        EXPECT_EQ(const_block(0, 0), 42.0);
    });
    
    // Accessing non-existent block should throw
    EXPECT_THROW(const_bm.block("nonexistent", "test"), std::runtime_error);
}

TEST_F(BlockMatrixTest, HasBlock) {
    BlockMatrix<double> bm({"u", "p"}, {2, 1});
    
    EXPECT_FALSE(bm.has_block("u", "u"));  // Not created yet
    EXPECT_FALSE(bm.has_block("nonexistent", "u"));
    
    // Create a block
    bm.block("u", "u")(0, 0) = 1.0;
    EXPECT_TRUE(bm.has_block("u", "u"));
    EXPECT_FALSE(bm.has_block("u", "p"));  // Still not created
    
    // Test by index
    EXPECT_TRUE(bm.has_block(0, 0));
    EXPECT_FALSE(bm.has_block(0, 1));
    EXPECT_FALSE(bm.has_block(10, 10));
}

// === Matrix Operations Tests ===

TEST_F(BlockMatrixTest, Fill) {
    BlockMatrix<double> bm({"a", "b"}, {2, 2});
    
    bm.fill(7.5);
    
    // All blocks should be created and filled
    EXPECT_TRUE(bm.has_block("a", "a"));
    EXPECT_TRUE(bm.has_block("a", "b"));
    EXPECT_TRUE(bm.has_block("b", "a"));
    EXPECT_TRUE(bm.has_block("b", "b"));
    
    EXPECT_EQ(bm.block("a", "a")(0, 0), 7.5);
    EXPECT_EQ(bm.block("a", "a")(1, 1), 7.5);
    EXPECT_EQ(bm.block("b", "b")(0, 1), 7.5);
}

TEST_F(BlockMatrixTest, Zero) {
    BlockMatrix<double> bm({"test"}, {2});
    bm.fill(42.0);
    
    bm.zero();
    
    EXPECT_EQ(bm.block("test", "test")(0, 0), 0.0);
    EXPECT_EQ(bm.block("test", "test")(1, 1), 0.0);
}

TEST_F(BlockMatrixTest, SetIdentity) {
    BlockMatrix<double> bm({"u", "p"}, {2, 1});
    
    bm.set_identity();
    
    // Diagonal blocks should have identity pattern
    EXPECT_EQ(bm.block("u", "u")(0, 0), 1.0);
    EXPECT_EQ(bm.block("u", "u")(0, 1), 0.0);
    EXPECT_EQ(bm.block("u", "u")(1, 0), 0.0);
    EXPECT_EQ(bm.block("u", "u")(1, 1), 1.0);
    EXPECT_EQ(bm.block("p", "p")(0, 0), 1.0);
    
    // Off-diagonal blocks should be zero
    EXPECT_EQ(bm.block("u", "p")(0, 0), 0.0);
    EXPECT_EQ(bm.block("u", "p")(1, 0), 0.0);
    EXPECT_EQ(bm.block("p", "u")(0, 0), 0.0);
    EXPECT_EQ(bm.block("p", "u")(0, 1), 0.0);
}

TEST_F(BlockMatrixTest, SetIdentityNonSquare) {
    std::vector<std::string> row_names = {"row"};
    std::vector<size_t> row_sizes = {2};
    std::vector<std::string> col_names = {"col1", "col2"};
    std::vector<size_t> col_sizes = {1, 1};
    
    BlockMatrix<double> bm(row_names, row_sizes, col_names, col_sizes);
    
    EXPECT_THROW(bm.set_identity(), std::logic_error);
}

// === Matrix-Vector Multiplication Tests ===

TEST_F(BlockMatrixTest, BlockMatrixVectorMultiplication) {
    BlockMatrix<double> A({"u", "p"}, {2, 1});
    BlockVector<double> x({{"u", 2}, {"p", 1}});
    
    // Set up matrix A
    A.block("u", "u")(0, 0) = 2; A.block("u", "u")(0, 1) = 1;
    A.block("u", "u")(1, 0) = 1; A.block("u", "u")(1, 1) = 3;
    A.block("u", "p")(0, 0) = 4;
    A.block("u", "p")(1, 0) = 2;
    A.block("p", "u")(0, 0) = 1; A.block("p", "u")(0, 1) = 2;
    A.block("p", "p")(0, 0) = 5;
    
    // Set up vector x
    x.block("u")[0] = 1.0;
    x.block("u")[1] = 2.0;
    x.block("p")[0] = 3.0;
    
    auto y = A * x;
    
    EXPECT_EQ(y.num_blocks(), 2);
    EXPECT_TRUE(y.has_block("u"));
    EXPECT_TRUE(y.has_block("p"));
    
    // Check results: y = A * x
    // y_u = A_uu * x_u + A_up * x_p = [2 1; 1 3] * [1; 2] + [4; 2] * [3] = [4; 7] + [12; 6] = [16; 13]
    // y_p = A_pu * x_u + A_pp * x_p = [1 2] * [1; 2] + [5] * [3] = [5] + [15] = [20]
    EXPECT_EQ(y.block("u")[0], 16.0);
    EXPECT_EQ(y.block("u")[1], 13.0);
    EXPECT_EQ(y.block("p")[0], 20.0);
}

TEST_F(BlockMatrixTest, RegularMatrixVectorMultiplication) {
    BlockMatrix<double> A({"a", "b"}, {2, 1});
    Vector<double> x(3);
    
    // Set up matrix A as a 3x3 matrix
    A.block("a", "a")(0, 0) = 1; A.block("a", "a")(0, 1) = 2;
    A.block("a", "a")(1, 0) = 3; A.block("a", "a")(1, 1) = 4;
    A.block("a", "b")(0, 0) = 5;
    A.block("a", "b")(1, 0) = 6;
    A.block("b", "a")(0, 0) = 7; A.block("b", "a")(0, 1) = 8;
    A.block("b", "b")(0, 0) = 9;
    
    // Set up vector x
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0;
    
    auto y = A * x;
    
    EXPECT_EQ(y.size(), 3);
    
    // Manual calculation:
    // Row 0: 1*1 + 2*2 + 5*3 = 1 + 4 + 15 = 20
    // Row 1: 3*1 + 4*2 + 6*3 = 3 + 8 + 18 = 29
    // Row 2: 7*1 + 8*2 + 9*3 = 7 + 16 + 27 = 50
    EXPECT_EQ(y[0], 20.0);
    EXPECT_EQ(y[1], 29.0);
    EXPECT_EQ(y[2], 50.0);
}

TEST_F(BlockMatrixTest, MatrixVectorIncompatibleSizes) {
    BlockMatrix<double> A({"u"}, {2});
    Vector<double> x(3);  // Wrong size
    
    EXPECT_THROW(A * x, std::invalid_argument);
    
    BlockVector<double> bx({{"u", 2}, {"p", 1}});  // Wrong number of blocks
    EXPECT_THROW(A * bx, std::invalid_argument);
}

// === Block Matrix-Matrix Multiplication Tests ===

TEST_F(BlockMatrixTest, BlockMatrixMatrixMultiplication) {
    BlockMatrix<double> A({"row"}, {2}, {"col1", "col2"}, {2, 1});
    BlockMatrix<double> B({"col1", "col2"}, {2, 1}, {"result"}, {2});
    
    // Set up A: 2x3 matrix
    A.block("row", "col1")(0, 0) = 1; A.block("row", "col1")(0, 1) = 2;
    A.block("row", "col1")(1, 0) = 3; A.block("row", "col1")(1, 1) = 4;
    A.block("row", "col2")(0, 0) = 5;
    A.block("row", "col2")(1, 0) = 6;
    
    // Set up B: 3x2 matrix
    B.block("col1", "result")(0, 0) = 7; B.block("col1", "result")(0, 1) = 8;
    B.block("col1", "result")(1, 0) = 9; B.block("col1", "result")(1, 1) = 10;
    B.block("col2", "result")(0, 0) = 11; B.block("col2", "result")(0, 1) = 12;
    
    auto C = A * B;
    
    EXPECT_EQ(C.num_block_rows(), 1);
    EXPECT_EQ(C.num_block_cols(), 1);
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    
    // Manual calculation of A*B:
    // C = A*B where A is 2x3 and B is 3x2
    // C(0,0) = 1*7 + 2*9 + 5*11 = 7 + 18 + 55 = 80
    // C(0,1) = 1*8 + 2*10 + 5*12 = 8 + 20 + 60 = 88
    // C(1,0) = 3*7 + 4*9 + 6*11 = 21 + 36 + 66 = 123
    // C(1,1) = 3*8 + 4*10 + 6*12 = 24 + 40 + 72 = 136
    
    EXPECT_EQ(C.block(0, 0)(0, 0), 80.0);
    EXPECT_EQ(C.block(0, 0)(0, 1), 88.0);
    EXPECT_EQ(C.block(0, 0)(1, 0), 123.0);
    EXPECT_EQ(C.block(0, 0)(1, 1), 136.0);
}

TEST_F(BlockMatrixTest, BlockMatrixMatrixIncompatibleSizes) {
    BlockMatrix<double> A({"row"}, {2}, {"col"}, {2});
    BlockMatrix<double> B({"wrongcol"}, {3}, {"result"}, {1});  // Wrong dimensions
    
    EXPECT_THROW(A * B, std::invalid_argument);
}

// === Arithmetic Operations Tests ===

TEST_F(BlockMatrixTest, Addition) {
    BlockMatrix<double> A({"u", "p"}, {2, 1});
    BlockMatrix<double> B({"u", "p"}, {2, 1});
    
    // Set up A
    A.block("u", "u")(0, 0) = 1; A.block("u", "u")(0, 1) = 2;
    A.block("u", "u")(1, 0) = 3; A.block("u", "u")(1, 1) = 4;
    A.block("u", "p")(0, 0) = 5; A.block("u", "p")(1, 0) = 6;
    A.block("p", "u")(0, 0) = 7; A.block("p", "u")(0, 1) = 8;
    A.block("p", "p")(0, 0) = 9;
    
    // Set up B
    B.block("u", "u")(0, 0) = 1; B.block("u", "u")(0, 1) = 1;
    B.block("u", "u")(1, 0) = 1; B.block("u", "u")(1, 1) = 1;
    B.block("u", "p")(0, 0) = 1; B.block("u", "p")(1, 0) = 1;
    B.block("p", "u")(0, 0) = 1; B.block("p", "u")(0, 1) = 1;
    B.block("p", "p")(0, 0) = 1;
    
    A += B;
    
    EXPECT_EQ(A.block("u", "u")(0, 0), 2);
    EXPECT_EQ(A.block("u", "u")(0, 1), 3);
    EXPECT_EQ(A.block("u", "u")(1, 0), 4);
    EXPECT_EQ(A.block("u", "u")(1, 1), 5);
    EXPECT_EQ(A.block("u", "p")(0, 0), 6);
    EXPECT_EQ(A.block("p", "p")(0, 0), 10);
}

TEST_F(BlockMatrixTest, AdditionIncompatibleStructure) {
    BlockMatrix<double> A({"u", "p"}, {2, 1});
    BlockMatrix<double> B({"u", "p", "extra"}, {2, 1, 1});  // Different structure
    
    EXPECT_THROW(A += B, std::invalid_argument);
}

TEST_F(BlockMatrixTest, Subtraction) {
    BlockMatrix<double> A({"test"}, {2});
    BlockMatrix<double> B({"test"}, {2});
    
    A.block("test", "test")(0, 0) = 10; A.block("test", "test")(0, 1) = 8;
    A.block("test", "test")(1, 0) = 6; A.block("test", "test")(1, 1) = 4;
    
    B.block("test", "test")(0, 0) = 1; B.block("test", "test")(0, 1) = 2;
    B.block("test", "test")(1, 0) = 3; B.block("test", "test")(1, 1) = 4;
    
    A -= B;
    
    EXPECT_EQ(A.block("test", "test")(0, 0), 9);
    EXPECT_EQ(A.block("test", "test")(0, 1), 6);
    EXPECT_EQ(A.block("test", "test")(1, 0), 3);
    EXPECT_EQ(A.block("test", "test")(1, 1), 0);
}

TEST_F(BlockMatrixTest, ScalarMultiplication) {
    BlockMatrix<double> bm({"test"}, {2});
    
    bm.block("test", "test")(0, 0) = 2; bm.block("test", "test")(0, 1) = 4;
    bm.block("test", "test")(1, 0) = 6; bm.block("test", "test")(1, 1) = 8;
    
    bm *= 2.5;
    
    EXPECT_EQ(bm.block("test", "test")(0, 0), 5.0);
    EXPECT_EQ(bm.block("test", "test")(0, 1), 10.0);
    EXPECT_EQ(bm.block("test", "test")(1, 0), 15.0);
    EXPECT_EQ(bm.block("test", "test")(1, 1), 20.0);
}

TEST_F(BlockMatrixTest, ScalarDivision) {
    BlockMatrix<double> bm({"test"}, {2});
    
    bm.block("test", "test")(0, 0) = 10; bm.block("test", "test")(0, 1) = 20;
    bm.block("test", "test")(1, 0) = 30; bm.block("test", "test")(1, 1) = 40;
    
    bm /= 10.0;
    
    EXPECT_EQ(bm.block("test", "test")(0, 0), 1.0);
    EXPECT_EQ(bm.block("test", "test")(0, 1), 2.0);
    EXPECT_EQ(bm.block("test", "test")(1, 0), 3.0);
    EXPECT_EQ(bm.block("test", "test")(1, 1), 4.0);
}

// === Norms Tests ===

TEST_F(BlockMatrixTest, FrobeniusNorm) {
    BlockMatrix<double> bm({"test"}, {2});
    
    bm.block("test", "test")(0, 0) = 3;
    bm.block("test", "test")(0, 1) = 4;
    bm.block("test", "test")(1, 0) = 0;
    bm.block("test", "test")(1, 1) = 0;
    
    double norm = bm.frobenius_norm();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(BlockMatrixTest, MaxNorm) {
    BlockMatrix<double> bm({"a", "b"}, {1, 1});
    
    bm.block("a", "a")(0, 0) = -7.0;
    bm.block("a", "b")(0, 0) = 3.0;
    bm.block("b", "a")(0, 0) = 5.0;
    bm.block("b", "b")(0, 0) = -2.0;
    
    double norm = bm.max_norm();
    EXPECT_EQ(norm, 7.0);
}

// === Properties Tests ===

TEST_F(BlockMatrixTest, SymmetryProperty) {
    BlockMatrix<double> bm({"test"}, {2}, BlockMatrix<double>::SymmetryType::Symmetric);
    
    EXPECT_EQ(bm.symmetry(), BlockMatrix<double>::SymmetryType::Symmetric);
    
    bm.set_symmetry(BlockMatrix<double>::SymmetryType::General);
    EXPECT_EQ(bm.symmetry(), BlockMatrix<double>::SymmetryType::General);
}

// === Non-member Operations Tests ===

TEST_F(BlockMatrixTest, NonMemberAddition) {
    BlockMatrix<double> A({"test"}, {2});
    BlockMatrix<int> B({"test"}, {2});
    
    A.block("test", "test")(0, 0) = 1.5; A.block("test", "test")(0, 1) = 2.5;
    A.block("test", "test")(1, 0) = 3.5; A.block("test", "test")(1, 1) = 4.5;
    
    B.block("test", "test")(0, 0) = 1; B.block("test", "test")(0, 1) = 2;
    B.block("test", "test")(1, 0) = 3; B.block("test", "test")(1, 1) = 4;
    
    auto result = A + B;
    
    // Check type promotion
    static_assert(std::is_same_v<decltype(result), BlockMatrix<double>>);
    
    EXPECT_NEAR(result.block("test", "test")(0, 0), 2.5, 1e-10);
    EXPECT_NEAR(result.block("test", "test")(0, 1), 4.5, 1e-10);
    EXPECT_NEAR(result.block("test", "test")(1, 0), 6.5, 1e-10);
    EXPECT_NEAR(result.block("test", "test")(1, 1), 8.5, 1e-10);
}

TEST_F(BlockMatrixTest, NonMemberSubtraction) {
    BlockMatrix<double> A({"test"}, {2});
    BlockMatrix<float> B({"test"}, {2});
    
    A.block("test", "test")(0, 0) = 5.0; A.block("test", "test")(1, 1) = 10.0;
    B.block("test", "test")(0, 0) = 1.0f; B.block("test", "test")(1, 1) = 3.0f;
    
    auto result = A - B;
    
    EXPECT_NEAR(result.block("test", "test")(0, 0), 4.0, 1e-6);
    EXPECT_NEAR(result.block("test", "test")(1, 1), 7.0, 1e-6);
}

TEST_F(BlockMatrixTest, ScalarMultiplicationNonMember) {
    BlockMatrix<double> bm({"test"}, {2});
    
    bm.block("test", "test")(0, 0) = 3.0;
    bm.block("test", "test")(1, 1) = 4.0;
    
    auto result1 = 2.0 * bm;
    auto result2 = bm * 2.0;
    
    EXPECT_EQ(result1.block("test", "test")(0, 0), 6.0);
    EXPECT_EQ(result1.block("test", "test")(1, 1), 8.0);
    EXPECT_EQ(result2.block("test", "test")(0, 0), 6.0);
    EXPECT_EQ(result2.block("test", "test")(1, 1), 8.0);
}

// === Complex Numbers Tests ===

TEST_F(BlockMatrixTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    BlockMatrix<Complex> bm({"test"}, {2});
    
    bm.block("test", "test")(0, 0) = Complex(1, 2);
    bm.block("test", "test")(0, 1) = Complex(3, -1);
    bm.block("test", "test")(1, 0) = Complex(-2, 1);
    bm.block("test", "test")(1, 1) = Complex(0, 4);
    
    bm *= Complex(2, 0);
    
    EXPECT_EQ(bm.block("test", "test")(0, 0), Complex(2, 4));
    EXPECT_EQ(bm.block("test", "test")(0, 1), Complex(6, -2));
    EXPECT_EQ(bm.block("test", "test")(1, 0), Complex(-4, 2));
    EXPECT_EQ(bm.block("test", "test")(1, 1), Complex(0, 8));
}

// === Multi-Physics Example Tests ===

TEST_F(BlockMatrixTest, FluidStructureInteractionExample) {
    // Example: 2D FSI problem with velocity (u,v), pressure (p), and displacement (d_x, d_y)
    std::vector<std::string> fields = {"u", "v", "p", "d_x", "d_y"};
    std::vector<size_t> field_sizes = {100, 100, 50, 25, 25};
    
    BlockMatrix<double> K(fields, field_sizes);
    
    EXPECT_EQ(K.num_block_rows(), 5);
    EXPECT_EQ(K.num_block_cols(), 5);
    EXPECT_EQ(K.rows(), 300);
    EXPECT_EQ(K.cols(), 300);
    EXPECT_TRUE(K.is_square());
    
    // Set up some coupling terms
    auto& K_uu = K.block("u", "u");  // u-velocity to u-velocity
    auto& K_uv = K.block("u", "v");  // u-velocity to v-velocity
    auto& K_up = K.block("u", "p");  // u-velocity to pressure
    
    EXPECT_EQ(K_uu.rows(), 100);
    EXPECT_EQ(K_uu.cols(), 100);
    EXPECT_EQ(K_uv.rows(), 100);
    EXPECT_EQ(K_uv.cols(), 100);
    EXPECT_EQ(K_up.rows(), 100);
    EXPECT_EQ(K_up.cols(), 50);
    
    // Set some sample values
    K_uu(0, 0) = 1.0;  // Diagonal dominance for stability
    K_up(0, 0) = 0.1;  // Pressure coupling
    
    EXPECT_EQ(K_uu(0, 0), 1.0);
    EXPECT_EQ(K_up(0, 0), 0.1);
}

TEST_F(BlockMatrixTest, ThermomechanicalExample) {
    // Coupled thermal-mechanical problem
    BlockMatrix<double> K({{"displacement", "temperature"}, {6, 3}});
    
    // Structural stiffness
    auto& K_uu = K.block("displacement", "displacement");
    K_uu.set_identity();  // Simplified
    
    // Thermal stiffness
    auto& K_TT = K.block("temperature", "temperature");
    K_TT.set_identity();  // Simplified
    
    // Thermal expansion coupling
    auto& K_uT = K.block("displacement", "temperature");
    K_uT.fill(0.01);  // Thermal expansion coefficient
    
    // Verify structure
    EXPECT_EQ(K.rows(), 9);
    EXPECT_EQ(K.cols(), 9);
    EXPECT_EQ(K_uu(0, 0), 1.0);
    EXPECT_EQ(K_TT(0, 0), 1.0);
    EXPECT_EQ(K_uT(0, 0), 0.01);
}

// === Stream Output Tests ===

TEST_F(BlockMatrixTest, StreamOutput) {
    BlockMatrix<double> bm({{"u", "p"}, {2, 1}});
    
    bm.block("u", "u")(0, 0) = 1.5;
    bm.block("u", "p")(0, 0) = 2.5;
    
    std::ostringstream oss;
    oss << bm;
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("BlockMatrix") != std::string::npos);
    EXPECT_TRUE(output.find("2x2 blocks") != std::string::npos);
    EXPECT_TRUE(output.find("3x3 total") != std::string::npos);
    EXPECT_TRUE(output.find("u") != std::string::npos);
    EXPECT_TRUE(output.find("p") != std::string::npos);
}

// === Error Handling Tests ===

TEST_F(BlockMatrixTest, ErrorCases) {
    BlockMatrix<double> bm;
    
    // Access non-existent block
    EXPECT_THROW(bm.block("nonexistent", "test"), std::invalid_argument);
    EXPECT_THROW(bm.row_block_size("nonexistent"), std::invalid_argument);
    EXPECT_THROW(bm.col_block_size("nonexistent"), std::invalid_argument);
    
    // Out of range block index
    EXPECT_THROW(bm.block(0, 0), std::out_of_range);
    
    // Invalid const access
    const BlockMatrix<double> const_bm({"test"}, {2});
    EXPECT_THROW(const_bm.block("nonexistent", "test"), std::runtime_error);
}

// === Performance Tests ===

TEST_F(BlockMatrixTest, LargeBlockMatrix) {
    std::vector<std::string> fields;
    std::vector<size_t> sizes;
    
    for (int i = 0; i < 10; ++i) {
        fields.push_back("field" + std::to_string(i));
        sizes.push_back(100);
    }
    
    EXPECT_NO_THROW({
        BlockMatrix<double> large_bm(fields, sizes);
        
        EXPECT_EQ(large_bm.rows(), 1000);
        EXPECT_EQ(large_bm.cols(), 1000);
        EXPECT_EQ(large_bm.num_block_rows(), 10);
        
        // Operations should work efficiently
        large_bm.block("field0", "field0")(0, 0) = 1.0;
        large_bm.block("field9", "field9")(99, 99) = 2.0;
        
        EXPECT_EQ(large_bm.block("field0", "field0")(0, 0), 1.0);
        EXPECT_EQ(large_bm.block("field9", "field9")(99, 99), 2.0);
    });
}

// === Edge Cases ===

TEST_F(BlockMatrixTest, SingleBlockMatrix) {
    BlockMatrix<double> bm({"single"}, {3});
    
    auto& block = bm.block("single", "single");
    block(0, 0) = 1; block(0, 1) = 2; block(0, 2) = 3;
    block(1, 0) = 4; block(1, 1) = 5; block(1, 2) = 6;
    block(2, 0) = 7; block(2, 1) = 8; block(2, 2) = 9;
    
    EXPECT_EQ(bm.num_block_rows(), 1);
    EXPECT_EQ(bm.num_block_cols(), 1);
    EXPECT_EQ(bm.rows(), 3);
    EXPECT_EQ(bm.cols(), 3);
    
    EXPECT_EQ(block(0, 0), 1);
    EXPECT_EQ(block(2, 2), 9);
}

TEST_F(BlockMatrixTest, EmptyBlocks) {
    BlockMatrix<double> bm({"empty"}, {0});
    
    EXPECT_EQ(bm.rows(), 0);
    EXPECT_EQ(bm.cols(), 0);
    EXPECT_TRUE(bm.empty());
    
    auto& block = bm.block("empty", "empty");
    EXPECT_EQ(block.rows(), 0);
    EXPECT_EQ(block.cols(), 0);
}

} // namespace

/**
 * @file test_small_matrix.cpp
 * @brief Comprehensive unit tests for the SmallMatrix class
 */

#include <gtest/gtest.h>
#include <sstream>
#include <complex>
#include <chrono>
#include <core/small_matrix.h>

using namespace fem::numeric;

namespace {

// Test fixture for small matrix tests
class SmallMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// === Constructor Tests ===

TEST_F(SmallMatrixTest, DefaultConstructor) {
    SmallMatrix<double, 3, 3> m;
    
    // Should be zero-initialized
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m(i, j), 0.0);
        }
    }
    
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.size(), 9);
    EXPECT_FALSE(m.empty());
}

TEST_F(SmallMatrixTest, ValueConstructor) {
    SmallMatrix<double, 2, 3> m(5.5);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m(i, j), 5.5);
        }
    }
}

TEST_F(SmallMatrixTest, InitializerListConstructor) {
    SmallMatrix<double, 2, 3> m{1, 2, 3, 4, 5, 6};
    
    EXPECT_EQ(m(0, 0), 1); EXPECT_EQ(m(0, 1), 2); EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4); EXPECT_EQ(m(1, 1), 5); EXPECT_EQ(m(1, 2), 6);
}

TEST_F(SmallMatrixTest, InitializerList2D) {
    SmallMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
    
    EXPECT_EQ(m(0, 0), 1); EXPECT_EQ(m(0, 1), 2); EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4); EXPECT_EQ(m(1, 1), 5); EXPECT_EQ(m(1, 2), 6);
}

TEST_F(SmallMatrixTest, InvalidInitializerList2D) {
    // Wrong number of rows
    EXPECT_THROW((SmallMatrix<double, 2, 2>{{1, 2}, {3, 4}, {5, 6}}), std::invalid_argument);
    
    // Wrong number of columns
    EXPECT_THROW((SmallMatrix<double, 2, 2>{{1, 2, 3}, {4, 5}}), std::invalid_argument);
}

TEST_F(SmallMatrixTest, ArrayConstructor) {
    std::array<double, 6> arr = {1, 2, 3, 4, 5, 6};
    SmallMatrix<double, 2, 3> m(arr);
    
    EXPECT_EQ(m[0], 1); EXPECT_EQ(m[1], 2); EXPECT_EQ(m[2], 3);
    EXPECT_EQ(m[3], 4); EXPECT_EQ(m[4], 5); EXPECT_EQ(m[5], 6);
}

TEST_F(SmallMatrixTest, CopyConstructorDifferentSize) {
    SmallMatrix<double, 3, 3> large;
    large.fill(2.0);
    large(2, 2) = 9.0;  // Corner element
    
    // Copy to smaller matrix (should truncate)
    SmallMatrix<double, 2, 2> small(large);
    
    EXPECT_EQ(small(0, 0), 2.0);
    EXPECT_EQ(small(0, 1), 2.0);
    EXPECT_EQ(small(1, 0), 2.0);
    EXPECT_EQ(small(1, 1), 2.0);
    
    // Copy to larger matrix (should zero-pad)
    SmallMatrix<double, 4, 4> larger(large);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(larger(i, j), large(i, j));
        }
    }
    
    // Zero-padded regions
    EXPECT_EQ(larger(3, 0), 0.0);
    EXPECT_EQ(larger(0, 3), 0.0);
    EXPECT_EQ(larger(3, 3), 0.0);
}

TEST_F(SmallMatrixTest, CopyAndMoveSemantics) {
    SmallMatrix<double, 2, 2> original{{1, 2}, {3, 4}};
    
    // Copy constructor
    SmallMatrix<double, 2, 2> copied(original);
    EXPECT_EQ(copied(0, 0), 1);
    EXPECT_EQ(copied(1, 1), 4);
    
    // Copy assignment
    SmallMatrix<double, 2, 2> assigned;
    assigned = original;
    EXPECT_EQ(assigned(0, 1), 2);
    EXPECT_EQ(assigned(1, 0), 3);
    
    // Move constructor
    SmallMatrix<double, 2, 2> moved(std::move(original));
    EXPECT_EQ(moved(0, 0), 1);
    EXPECT_EQ(moved(1, 1), 4);
    
    // Move assignment
    SmallMatrix<double, 2, 2> move_assigned;
    move_assigned = std::move(copied);
    EXPECT_EQ(move_assigned(0, 0), 1);
    EXPECT_EQ(move_assigned(1, 1), 4);
}

// === Size and Properties Tests ===

TEST_F(SmallMatrixTest, SizeProperties) {
    SmallMatrix<double, 4, 5> m;
    
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.size(), 20);
    EXPECT_FALSE(m.empty());  // SmallMatrix is never empty
    EXPECT_FALSE(m.is_square_matrix());
    EXPECT_FALSE(m.is_vector_matrix());
    
    SmallMatrix<int, 3, 3> square;
    EXPECT_TRUE(square.is_square_matrix());
    
    SmallMatrix<float, 5, 1> vector;
    EXPECT_TRUE(vector.is_vector_matrix());
}

TEST_F(SmallMatrixTest, CompileTimeConstants) {
    SmallMatrix<double, 3, 4> m;
    
    EXPECT_EQ(m.rows_c, 3);
    EXPECT_EQ(m.cols_c, 4);
    EXPECT_EQ(m.size_c, 12);
    EXPECT_EQ(m.tensor_rank, 2);  // Matrices are rank-2 tensors
    
    static_assert(SmallMatrix<double, 3, 4>::rows_c == 3);
    static_assert(SmallMatrix<double, 3, 4>::cols_c == 4);
    static_assert(SmallMatrix<double, 3, 4>::size_c == 12);
}

// === Element Access Tests ===

TEST_F(SmallMatrixTest, ElementAccess) {
    SmallMatrix<double, 3, 3> m;
    
    // Test bounds checking
    EXPECT_NO_THROW(m.at(0, 0));
    EXPECT_NO_THROW(m.at(2, 2));
    EXPECT_THROW(m.at(3, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 3), std::out_of_range);
    
    // Test element setting and getting
    m(1, 2) = 42.0;
    EXPECT_EQ(m(1, 2), 42.0);
    EXPECT_EQ(m.at(1, 2), 42.0);
    
    // Test linear indexing
    m[5] = 99.0;
    EXPECT_EQ(m[5], 99.0);
}

TEST_F(SmallMatrixTest, DataAccess) {
    SmallMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
    
    auto* data = m.data();
    EXPECT_NE(data, nullptr);
    
    // Check row-major storage
    EXPECT_EQ(data[0], 1);  // m(0,0)
    EXPECT_EQ(data[1], 2);  // m(0,1)
    EXPECT_EQ(data[2], 3);  // m(1,0)
    EXPECT_EQ(data[3], 4);  // m(1,1)
    
    // Modify through data pointer
    data[0] = 99.0;
    EXPECT_EQ(m(0, 0), 99.0);
    
    // Const data access
    const auto& const_m = m;
    const auto* const_data = const_m.data();
    EXPECT_EQ(const_data[0], 99.0);
}

// === Iterator Tests ===

TEST_F(SmallMatrixTest, Iterators) {
    SmallMatrix<int, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
    
    // Forward iteration
    std::vector<int> forward_values;
    for (auto it = m.begin(); it != m.end(); ++it) {
        forward_values.push_back(*it);
    }
    std::vector<int> expected = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(forward_values, expected);
    
    // Range-based for loop
    std::vector<int> range_values;
    for (const auto& val : m) {
        range_values.push_back(val);
    }
    EXPECT_EQ(range_values, expected);
    
    // Reverse iteration
    std::vector<int> reverse_values;
    for (auto it = m.rbegin(); it != m.rend(); ++it) {
        reverse_values.push_back(*it);
    }
    std::reverse(expected.begin(), expected.end());
    EXPECT_EQ(reverse_values, expected);
    
    // Const iterators
    const auto& const_m = m;
    auto const_it = const_m.begin();
    EXPECT_EQ(*const_it, 1);
    
    auto const_cit = const_m.cbegin();
    EXPECT_EQ(*const_cit, 1);
}

// === Matrix Operations Tests ===

TEST_F(SmallMatrixTest, Fill) {
    SmallMatrix<double, 2, 3> m;
    
    m.fill(7.5);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m(i, j), 7.5);
        }
    }
}

TEST_F(SmallMatrixTest, Zero) {
    SmallMatrix<double, 2, 2> m(5.0);
    
    m.zero();
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(m(i, j), 0.0);
        }
    }
}

TEST_F(SmallMatrixTest, SetIdentity) {
    SmallMatrix<double, 3, 3> m;
    
    m.set_identity();
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_EQ(m(i, j), 1.0);
            } else {
                EXPECT_EQ(m(i, j), 0.0);
            }
        }
    }
    
    // Test identity creation
    auto identity = SmallMatrix<double, 4, 4>::identity();
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_EQ(identity(i, j), 1.0);
            } else {
                EXPECT_EQ(identity(i, j), 0.0);
            }
        }
    }
}

TEST_F(SmallMatrixTest, SetIdentityNonSquare) {
    // Should only compile for square matrices
    SmallMatrix<double, 2, 3> non_square;
    // non_square.set_identity(); // This should not compile due to requires clause
}

TEST_F(SmallMatrixTest, Transpose) {
    SmallMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
    
    auto transposed = m.transpose();
    
    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 2);
    
    EXPECT_EQ(transposed(0, 0), 1);
    EXPECT_EQ(transposed(1, 0), 2);
    EXPECT_EQ(transposed(2, 0), 3);
    EXPECT_EQ(transposed(0, 1), 4);
    EXPECT_EQ(transposed(1, 1), 5);
    EXPECT_EQ(transposed(2, 1), 6);
}

TEST_F(SmallMatrixTest, TransposeInPlace) {
    SmallMatrix<double, 3, 3> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    m.transpose_inplace();
    
    EXPECT_EQ(m(0, 0), 1); EXPECT_EQ(m(0, 1), 4); EXPECT_EQ(m(0, 2), 7);
    EXPECT_EQ(m(1, 0), 2); EXPECT_EQ(m(1, 1), 5); EXPECT_EQ(m(1, 2), 8);
    EXPECT_EQ(m(2, 0), 3); EXPECT_EQ(m(2, 1), 6); EXPECT_EQ(m(2, 2), 9);
}

// === Arithmetic Operations Tests ===

TEST_F(SmallMatrixTest, Addition) {
    SmallMatrix<double, 2, 2> a{{1, 2}, {3, 4}};
    SmallMatrix<double, 2, 2> b{{5, 6}, {7, 8}};
    
    a += b;
    
    EXPECT_EQ(a(0, 0), 6);
    EXPECT_EQ(a(0, 1), 8);
    EXPECT_EQ(a(1, 0), 10);
    EXPECT_EQ(a(1, 1), 12);
}

TEST_F(SmallMatrixTest, AdditionMixedTypes) {
    SmallMatrix<double, 2, 2> a{{1.5, 2.5}, {3.5, 4.5}};
    SmallMatrix<int, 2, 2> b{{1, 2}, {3, 4}};
    
    a += b;
    
    EXPECT_EQ(a(0, 0), 2.5);
    EXPECT_EQ(a(0, 1), 4.5);
    EXPECT_EQ(a(1, 0), 6.5);
    EXPECT_EQ(a(1, 1), 8.5);
}

TEST_F(SmallMatrixTest, Subtraction) {
    SmallMatrix<double, 2, 2> a{{10, 9}, {8, 7}};
    SmallMatrix<double, 2, 2> b{{1, 2}, {3, 4}};
    
    a -= b;
    
    EXPECT_EQ(a(0, 0), 9);
    EXPECT_EQ(a(0, 1), 7);
    EXPECT_EQ(a(1, 0), 5);
    EXPECT_EQ(a(1, 1), 3);
}

TEST_F(SmallMatrixTest, ScalarMultiplication) {
    SmallMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
    
    m *= 2.5;
    
    EXPECT_EQ(m(0, 0), 2.5);
    EXPECT_EQ(m(0, 1), 5.0);
    EXPECT_EQ(m(1, 0), 7.5);
    EXPECT_EQ(m(1, 1), 10.0);
}

TEST_F(SmallMatrixTest, ScalarDivision) {
    SmallMatrix<double, 2, 2> m{{10, 20}, {30, 40}};
    
    m /= 10.0;
    
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 1), 2.0);
    EXPECT_EQ(m(1, 0), 3.0);
    EXPECT_EQ(m(1, 1), 4.0);
}

TEST_F(SmallMatrixTest, HadamardProduct) {
    SmallMatrix<double, 2, 2> a{{2, 3}, {4, 5}};
    SmallMatrix<double, 2, 2> b{{1, 2}, {3, 4}};
    
    a.hadamard_product(b);
    
    EXPECT_EQ(a(0, 0), 2);
    EXPECT_EQ(a(0, 1), 6);
    EXPECT_EQ(a(1, 0), 12);
    EXPECT_EQ(a(1, 1), 20);
}

// === Matrix-Vector Multiplication Tests ===

TEST_F(SmallMatrixTest, MatrixVectorMultiplication) {
    SmallMatrix<double, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
    SmallMatrix<double, 3, 1> x{{1}, {2}, {3}};
    
    auto y = A * x;
    
    EXPECT_EQ(y.rows(), 2);
    EXPECT_EQ(y.cols(), 1);
    
    // y = A * x = [1*1 + 2*2 + 3*3; 4*1 + 5*2 + 6*3] = [14; 32]
    EXPECT_EQ(y(0, 0), 14);
    EXPECT_EQ(y(1, 0), 32);
}

TEST_F(SmallMatrixTest, MatrixVectorMultiplicationMixedTypes) {
    SmallMatrix<double, 2, 2> A{{1.5, 2.5}, {3.5, 4.5}};
    SmallMatrix<int, 2, 1> x{{2}, {3}};
    
    auto y = A * x;
    
    // Check type promotion
    static_assert(std::is_same_v<decltype(y), SmallMatrix<double, 2, 1>>);
    
    // y = [1.5*2 + 2.5*3; 3.5*2 + 4.5*3] = [10.5; 20.5]
    EXPECT_EQ(y(0, 0), 10.5);
    EXPECT_EQ(y(1, 0), 20.5);
}

// === Matrix-Matrix Multiplication Tests ===

TEST_F(SmallMatrixTest, MatrixMatrixMultiplication) {
    SmallMatrix<double, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
    SmallMatrix<double, 3, 2> B{{7, 8}, {9, 10}, {11, 12}};
    
    auto C = A * B;
    
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    
    // C = A * B
    // C(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C(0,1) = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C(1,0) = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C(1,1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    EXPECT_EQ(C(0, 0), 58);
    EXPECT_EQ(C(0, 1), 64);
    EXPECT_EQ(C(1, 0), 139);
    EXPECT_EQ(C(1, 1), 154);
}

TEST_F(SmallMatrixTest, MatrixMatrixMultiplicationSquare) {
    SmallMatrix<double, 2, 2> A{{1, 2}, {3, 4}};
    SmallMatrix<double, 2, 2> B{{5, 6}, {7, 8}};
    
    auto C = A * B;
    
    // C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    EXPECT_EQ(C(0, 0), 19);
    EXPECT_EQ(C(0, 1), 22);
    EXPECT_EQ(C(1, 0), 43);
    EXPECT_EQ(C(1, 1), 50);
}

// === Specialized Operations Tests ===

TEST_F(SmallMatrixTest, Determinant1x1) {
    SmallMatrix<double, 1, 1> m{{5}};
    
    auto det = m.determinant();
    EXPECT_EQ(det, 5.0);
}

TEST_F(SmallMatrixTest, Determinant2x2) {
    SmallMatrix<double, 2, 2> m{{3, 2}, {1, 4}};
    
    auto det = m.determinant();
    EXPECT_EQ(det, 10.0);  // 3*4 - 2*1 = 10
}

TEST_F(SmallMatrixTest, Determinant3x3) {
    SmallMatrix<double, 3, 3> m{{2, 1, 3}, {1, 0, 1}, {1, 1, 1}};
    
    auto det = m.determinant();
    // Manual calculation: 2*(0*1 - 1*1) - 1*(1*1 - 1*1) + 3*(1*1 - 0*1) = 2*(-1) - 1*0 + 3*1 = -2 + 3 = 1
    EXPECT_EQ(det, 1.0);
}

TEST_F(SmallMatrixTest, Determinant4x4) {
    SmallMatrix<double, 4, 4> m{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}};
    
    auto det = m.determinant();
    EXPECT_EQ(det, 24.0);  // Product of diagonal elements for diagonal matrix
}

TEST_F(SmallMatrixTest, Trace) {
    SmallMatrix<double, 3, 3> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    auto tr = m.trace();
    EXPECT_EQ(tr, 15.0);  // 1 + 5 + 9
}

// === Norms Tests ===

TEST_F(SmallMatrixTest, FrobeniusNorm) {
    SmallMatrix<double, 2, 2> m{{3, 4}, {0, 0}};
    
    auto norm = m.frobenius_norm();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(SmallMatrixTest, MaxNorm) {
    SmallMatrix<double, 2, 2> m{{-7, 3}, {5, -2}};
    
    auto norm = m.max_norm();
    EXPECT_EQ(norm, 7.0);
}

TEST_F(SmallMatrixTest, Norm1) {
    SmallMatrix<double, 2, 2> m{{1, -2}, {3, -4}};
    
    auto norm = m.norm1();
    // Column sums: |1| + |3| = 4, |-2| + |-4| = 6; max = 6
    EXPECT_EQ(norm, 6.0);
}

TEST_F(SmallMatrixTest, NormInf) {
    SmallMatrix<double, 2, 2> m{{1, -2}, {3, -4}};
    
    auto norm = m.norm_inf();
    // Row sums: |1| + |-2| = 3, |3| + |-4| = 7; max = 7
    EXPECT_EQ(norm, 7.0);
}

// === Utility Functions Tests ===

TEST_F(SmallMatrixTest, SwapRows) {
    SmallMatrix<double, 3, 2> m{{1, 2}, {3, 4}, {5, 6}};
    
    m.swap_rows(0, 2);
    
    EXPECT_EQ(m(0, 0), 5); EXPECT_EQ(m(0, 1), 6);
    EXPECT_EQ(m(1, 0), 3); EXPECT_EQ(m(1, 1), 4);
    EXPECT_EQ(m(2, 0), 1); EXPECT_EQ(m(2, 1), 2);
    
    // Swap with itself should do nothing
    SmallMatrix<double, 2, 2> m2{{1, 2}, {3, 4}};
    m2.swap_rows(0, 0);
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 2);
}

TEST_F(SmallMatrixTest, SwapCols) {
    SmallMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
    
    m.swap_cols(0, 2);
    
    EXPECT_EQ(m(0, 0), 3); EXPECT_EQ(m(0, 1), 2); EXPECT_EQ(m(0, 2), 1);
    EXPECT_EQ(m(1, 0), 6); EXPECT_EQ(m(1, 1), 5); EXPECT_EQ(m(1, 2), 4);
}

TEST_F(SmallMatrixTest, StorageAccess) {
    SmallMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
    
    const auto& storage = m.storage();
    EXPECT_EQ(storage.size(), 4);
    EXPECT_EQ(storage[0], 1);
    EXPECT_EQ(storage[3], 4);
    
    auto& mutable_storage = m.storage();
    mutable_storage[0] = 99;
    EXPECT_EQ(m(0, 0), 99);
}

// === Comparison Tests ===

TEST_F(SmallMatrixTest, Equality) {
    SmallMatrix<double, 2, 2> a{{1, 2}, {3, 4}};
    SmallMatrix<double, 2, 2> b{{1, 2}, {3, 4}};
    SmallMatrix<double, 2, 2> c{{1, 2}, {3, 5}};
    
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(SmallMatrixTest, EqualityMixedTypes) {
    SmallMatrix<double, 2, 2> a{{1.0, 2.0}, {3.0, 4.0}};
    SmallMatrix<int, 2, 2> b{{1, 2}, {3, 4}};
    
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

// === Non-member Operations Tests ===

TEST_F(SmallMatrixTest, NonMemberAddition) {
    SmallMatrix<double, 2, 2> a{{1, 2}, {3, 4}};
    SmallMatrix<int, 2, 2> b{{5, 6}, {7, 8}};
    
    auto result = a + b;
    
    // Check type promotion
    static_assert(std::is_same_v<decltype(result), SmallMatrix<double, 2, 2>>);
    
    EXPECT_EQ(result(0, 0), 6);
    EXPECT_EQ(result(0, 1), 8);
    EXPECT_EQ(result(1, 0), 10);
    EXPECT_EQ(result(1, 1), 12);
}

TEST_F(SmallMatrixTest, NonMemberSubtraction) {
    SmallMatrix<double, 2, 2> a{{5, 6}, {7, 8}};
    SmallMatrix<float, 2, 2> b{{1, 2}, {3, 4}};
    
    auto result = a - b;
    
    EXPECT_NEAR(result(0, 0), 4, 1e-6);
    EXPECT_NEAR(result(0, 1), 4, 1e-6);
    EXPECT_NEAR(result(1, 0), 4, 1e-6);
    EXPECT_NEAR(result(1, 1), 4, 1e-6);
}

TEST_F(SmallMatrixTest, ScalarMultiplicationNonMember) {
    SmallMatrix<double, 2, 2> m{{3, 4}, {5, 6}};
    
    auto result1 = 2.0 * m;
    auto result2 = m * 2.0;
    
    EXPECT_EQ(result1(0, 0), 6);
    EXPECT_EQ(result1(1, 1), 12);
    EXPECT_EQ(result2(0, 1), 8);
    EXPECT_EQ(result2(1, 0), 10);
}

// === Complex Numbers Tests ===

TEST_F(SmallMatrixTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    SmallMatrix<Complex, 2, 2> m;
    
    m(0, 0) = Complex(1, 2);
    m(0, 1) = Complex(3, -1);
    m(1, 0) = Complex(-2, 1);
    m(1, 1) = Complex(0, 4);
    
    m *= Complex(2, 0);
    
    EXPECT_EQ(m(0, 0), Complex(2, 4));
    EXPECT_EQ(m(0, 1), Complex(6, -2));
    EXPECT_EQ(m(1, 0), Complex(-4, 2));
    EXPECT_EQ(m(1, 1), Complex(0, 8));
    
    // Test norms with complex numbers
    SmallMatrix<Complex, 1, 1> single{{Complex(3, 4)}};
    auto norm = single.frobenius_norm();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // |3+4i| = sqrt(9+16) = 5
}

// === Type Aliases Tests ===

TEST_F(SmallMatrixTest, TypeAliases) {
    // Test vector aliases
    SmallVector2<double> v2;
    v2(0, 0) = 1.0;
    v2(1, 0) = 2.0;
    EXPECT_EQ(v2.rows(), 2);
    EXPECT_EQ(v2.cols(), 1);
    EXPECT_EQ(v2(0, 0), 1.0);
    EXPECT_EQ(v2(1, 0), 2.0);
    
    // Test matrix aliases
    SmallMatrix3x3<int> m3x3;
    m3x3.set_identity();
    EXPECT_EQ(m3x3(0, 0), 1);
    EXPECT_EQ(m3x3(1, 1), 1);
    EXPECT_EQ(m3x3(2, 2), 1);
    EXPECT_EQ(m3x3(0, 1), 0);
    
    // Test element matrix
    ElementMatrix2D<double> elem;
    EXPECT_EQ(elem.rows(), 8);
    EXPECT_EQ(elem.cols(), 8);
}

// === Stream Output Tests ===

TEST_F(SmallMatrixTest, StreamOutput) {
    SmallMatrix<double, 2, 2> m{{1.5, 2.5}, {3.5, 4.5}};
    
    std::ostringstream oss;
    oss << m;
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("SmallMatrix<2x2>") != std::string::npos);
    EXPECT_TRUE(output.find("1.5") != std::string::npos);
    EXPECT_TRUE(output.find("2.5") != std::string::npos);
    EXPECT_TRUE(output.find("3.5") != std::string::npos);
    EXPECT_TRUE(output.find("4.5") != std::string::npos);
}

// === FEM Application Tests ===

TEST_F(SmallMatrixTest, ElementStiffnessMatrix) {
    // Typical 2D 4-node element stiffness matrix (8x8)
    ElementMatrix2D<double> K_e;
    K_e.zero();
    
    // Set some typical stiffness values
    K_e(0, 0) = 1.0;  // Diagonal terms
    K_e(1, 1) = 1.0;
    K_e(0, 2) = -0.5; // Coupling terms
    K_e(2, 0) = -0.5;
    
    EXPECT_EQ(K_e(0, 0), 1.0);
    EXPECT_EQ(K_e(0, 2), -0.5);
    EXPECT_EQ(K_e(7, 7), 0.0);  // Still zero
    
    // Test that operations are efficient
    K_e *= 2.0;
    EXPECT_EQ(K_e(0, 0), 2.0);
    EXPECT_EQ(K_e(0, 2), -1.0);
}

TEST_F(SmallMatrixTest, StrainDisplacementMatrix) {
    // B matrix relating strain to displacements
    SmallMatrix<double, 3, 8> B;  // 3 strain components, 8 DOFs
    B.zero();
    
    // Set some shape function derivatives
    B(0, 0) = 0.25;  // dN1/dx
    B(1, 1) = 0.25;  // dN1/dy
    B(2, 0) = 0.25;  // dN1/dy (for shear)
    B(2, 1) = 0.25;  // dN1/dx (for shear)
    
    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 8);
    EXPECT_EQ(B(0, 0), 0.25);
    EXPECT_EQ(B(2, 1), 0.25);
}

TEST_F(SmallMatrixTest, ConstitutiveMatrix) {
    // D matrix for plane stress
    SmallMatrix<double, 3, 3> D;
    D.zero();
    
    double E = 200e9;     // Young's modulus
    double nu = 0.3;      // Poisson's ratio
    double factor = E / (1 - nu * nu);
    
    D(0, 0) = factor;
    D(1, 1) = factor;
    D(0, 1) = factor * nu;
    D(1, 0) = factor * nu;
    D(2, 2) = factor * (1 - nu) / 2;
    
    EXPECT_EQ(D(0, 0), factor);
    EXPECT_EQ(D(0, 1), factor * nu);
    EXPECT_EQ(D(2, 2), factor * (1 - nu) / 2);
}

// === Performance Tests ===

TEST_F(SmallMatrixTest, PerformanceComparison) {
    constexpr size_t N = 1000;
    
    // Test small matrix operations are fast
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < N; ++i) {
        SmallMatrix<double, 4, 4> A, B;
        A.fill(1.0);
        B.fill(2.0);
        
        auto C = A * B;  // Should be very fast due to compile-time optimization
        auto det = C.determinant();
        (void)det;  // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Small matrices should be very fast - this is more of a sanity check
    // that the operations complete without hanging
    EXPECT_LT(duration.count(), 100000);  // Less than 100ms for 1000 operations
}

// === Edge Cases ===

TEST_F(SmallMatrixTest, SingleElement) {
    SmallMatrix<double, 1, 1> m{{42.0}};
    
    EXPECT_EQ(m.size(), 1);
    EXPECT_EQ(m(0, 0), 42.0);
    EXPECT_EQ(m[0], 42.0);
    
    m *= 2.0;
    EXPECT_EQ(m(0, 0), 84.0);
    
    auto det = m.determinant();
    EXPECT_EQ(det, 84.0);
    
    auto trace = m.trace();
    EXPECT_EQ(trace, 84.0);
}

TEST_F(SmallMatrixTest, LargestSupportedMatrix) {
    // Test with maximum allowed size (32x32)
    SmallMatrix<double, 32, 32> large;
    
    EXPECT_EQ(large.rows(), 32);
    EXPECT_EQ(large.cols(), 32);
    EXPECT_EQ(large.size(), 1024);
    
    // Operations should still work
    large.fill(1.0);
    large.set_identity();
    
    EXPECT_EQ(large(0, 0), 1.0);
    EXPECT_EQ(large(0, 1), 0.0);
    EXPECT_EQ(large(31, 31), 1.0);
}

// === Constexpr Tests ===

TEST_F(SmallMatrixTest, ConstexprOperations) {
    // Test that operations can be constexpr where possible
    constexpr SmallMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
    
    static_assert(m.rows() == 2);
    static_assert(m.cols() == 2);
    static_assert(m.size() == 4);
    static_assert(!m.empty());
    
    // Element access should be constexpr
    static_assert(m(0, 0) == 1.0);
    static_assert(m(1, 1) == 4.0);
}

} // namespace

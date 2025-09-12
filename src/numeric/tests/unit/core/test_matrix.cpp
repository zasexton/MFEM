#include <gtest/gtest.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <random>
#include <chrono>

using namespace fem::numeric;

// Test fixture for Matrix tests
template<typename T>
class MatrixTest : public ::testing::Test {
protected:
    using value_type = T;
    using MatrixType = Matrix<T>;
    using VectorType = Vector<T>;
    
    void SetUp() override {
        // Initialize test matrices
        mat_3x3_ = MatrixType{{T{1}, T{2}, T{3}},
                              {T{4}, T{5}, T{6}},
                              {T{7}, T{8}, T{9}}};
        
        mat_2x3_ = MatrixType{{T{1}, T{2}, T{3}},
                              {T{4}, T{5}, T{6}}};
        
        mat_3x2_ = MatrixType{{T{1}, T{2}},
                              {T{3}, T{4}},
                              {T{5}, T{6}}};
        
        identity_3_ = MatrixType::identity(3);
    }
    
    MatrixType mat_3x3_;
    MatrixType mat_2x3_;
    MatrixType mat_3x2_;
    MatrixType identity_3_;
    
    // Helper to check matrix equality with tolerance
    bool matrices_equal(const MatrixType& a, const MatrixType& b, 
                        double tol = 1e-10) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < a.cols(); ++j) {
                if (std::abs(a(i,j) - b(i,j)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
};

// Instantiate tests for different types
using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixTest, TestTypes);

// ============================================================================
// Construction and Basic Properties
// ============================================================================

TEST(MatrixConstruction, DefaultConstructor) {
    Matrix<double> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixConstruction, SizeConstructor) {
    Matrix<double> m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_FALSE(m.empty());
}

TEST(MatrixConstruction, ValueConstructor) {
    Matrix<double> m(2, 3, 5.0);
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            EXPECT_EQ(m(i, j), 5.0);
        }
    }
}

TEST(MatrixConstruction, InitializerListConstructor) {
    Matrix<double> m = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}};
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 1), 2.0);
    EXPECT_EQ(m(0, 2), 3.0);
    EXPECT_EQ(m(1, 0), 4.0);
    EXPECT_EQ(m(1, 1), 5.0);
    EXPECT_EQ(m(1, 2), 6.0);
}

TEST(MatrixConstruction, InitializerListInvalidDimensions) {
    // Rows with different column counts should throw
    EXPECT_THROW((Matrix<double>{{1.0, 2.0}, {3.0, 4.0, 5.0}}), 
                 std::invalid_argument);
}

// ============================================================================
// Element Access
// ============================================================================

TYPED_TEST(MatrixTest, ElementAccess) {
    using T = typename TestFixture::value_type;
    Matrix<T> m(3, 3);
    
    // Set and get elements
    using real_t = decltype(std::real(T{}));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = T(static_cast<real_t>(i * 3 + j));
        }
    }
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m(i, j), T(static_cast<real_t>(i * 3 + j)));
        }
    }
}

TYPED_TEST(MatrixTest, AtWithBoundsChecking) {
    using T = typename TestFixture::value_type;
    Matrix<T> m(2, 3);
    
    // Valid access
    m.at(1, 2) = T{42};
    EXPECT_EQ(m.at(1, 2), T{42});
    
    // Invalid access should throw
    EXPECT_THROW(m.at(2, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 3), std::out_of_range);
}

// ============================================================================
// Row and Column Access
// ============================================================================

TEST(MatrixRowCol, RowAccess) {
    Matrix<double> m = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    
    auto row1 = m.row(1);
    EXPECT_EQ(row1.size(), 3);
    EXPECT_EQ(row1[0], 4.0);
    EXPECT_EQ(row1[1], 5.0);
    EXPECT_EQ(row1[2], 6.0);
    
    // Modify through row view
    row1[1] = 50.0;
    EXPECT_EQ(m(1, 1), 50.0);
}

TEST(MatrixRowCol, ColumnAccess) {
    Matrix<double> m = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    
    auto col2 = m.col(2);
    EXPECT_EQ(col2.size(), 3);
    EXPECT_EQ(col2[0], 3.0);
    EXPECT_EQ(col2[1], 6.0);
    EXPECT_EQ(col2[2], 9.0);
    
    // Modify through column view
    col2[1] = 60.0;
    EXPECT_EQ(m(1, 2), 60.0);
}

TEST(MatrixRowCol, DiagonalAccess) {
    Matrix<double> m = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    
    auto diag = m.diag();
    EXPECT_EQ(diag.size(), 3);
    EXPECT_EQ(diag[0], 1.0);
    EXPECT_EQ(diag[1], 5.0);
    EXPECT_EQ(diag[2], 9.0);
    
    // Modify diagonal
    diag[1] = 50.0;
    EXPECT_EQ(m(1, 1), 50.0);
}

// ============================================================================
// Identity and Diagonal Matrices
// ============================================================================

TEST(MatrixSpecial, IdentityMatrix) {
    auto I = Matrix<double>::identity(4);
    EXPECT_EQ(I.rows(), 4);
    EXPECT_EQ(I.cols(), 4);
    
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_EQ(I(i, j), 1.0);
            } else {
                EXPECT_EQ(I(i, j), 0.0);
            }
        }
    }
}

TEST(MatrixSpecial, DiagonalMatrix) {
    Vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto D = Matrix<double>::diag(v);
    
    EXPECT_EQ(D.rows(), 4);
    EXPECT_EQ(D.cols(), 4);
    
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_EQ(D(i, j), v[i]);
            } else {
                EXPECT_EQ(D(i, j), 0.0);
            }
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

TYPED_TEST(MatrixTest, Addition) {
    using T = typename TestFixture::value_type;
    Matrix<T> a = {{T{1}, T{2}}, 
                   {T{3}, T{4}}};
    Matrix<T> b = {{T{5}, T{6}}, 
                   {T{7}, T{8}}};
    
    auto c = a + b;
    EXPECT_EQ(c(0, 0), T{6});
    EXPECT_EQ(c(0, 1), T{8});
    EXPECT_EQ(c(1, 0), T{10});
    EXPECT_EQ(c(1, 1), T{12});
    
    // In-place addition
    a += b;
    EXPECT_EQ(a(0, 0), T{6});
    EXPECT_EQ(a(1, 1), T{12});
}

TYPED_TEST(MatrixTest, Subtraction) {
    using T = typename TestFixture::value_type;
    Matrix<T> a = {{T{5}, T{6}}, 
                   {T{7}, T{8}}};
    Matrix<T> b = {{T{1}, T{2}}, 
                   {T{3}, T{4}}};
    
    auto c = a - b;
    EXPECT_EQ(c(0, 0), T{4});
    EXPECT_EQ(c(0, 1), T{4});
    EXPECT_EQ(c(1, 0), T{4});
    EXPECT_EQ(c(1, 1), T{4});
}

TYPED_TEST(MatrixTest, ScalarMultiplication) {
    using T = typename TestFixture::value_type;
    Matrix<T> a = {{T{1}, T{2}}, 
                   {T{3}, T{4}}};
    
    auto b = a * T{2};
    EXPECT_EQ(b(0, 0), T{2});
    EXPECT_EQ(b(0, 1), T{4});
    EXPECT_EQ(b(1, 0), T{6});
    EXPECT_EQ(b(1, 1), T{8});
    
    // Scalar on left
    auto c = T{3} * a;
    EXPECT_EQ(c(0, 0), T{3});
    EXPECT_EQ(c(1, 1), T{12});
}

// ============================================================================
// Matrix-Vector Multiplication
// ============================================================================

TEST(MatrixVector, MatrixVectorMultiplication) {
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}};
    Vector<double> x = {1.0, 2.0, 3.0};
    
    auto y = A * x;
    EXPECT_EQ(y.size(), 2);
    EXPECT_DOUBLE_EQ(y[0], 14.0);  // 1*1 + 2*2 + 3*3 = 14
    EXPECT_DOUBLE_EQ(y[1], 32.0);  // 4*1 + 5*2 + 6*3 = 32
}

TEST(MatrixVector, MatrixVectorDimensionMismatch) {
    Matrix<double> A(3, 4);
    Vector<double> x(3);  // Wrong size
    
    EXPECT_THROW(A * x, std::invalid_argument);
}

// ============================================================================
// Matrix-Matrix Multiplication
// ============================================================================

TEST(MatrixMatrix, MatrixMatrixMultiplication) {
    Matrix<double> A = {{1.0, 2.0},
                        {3.0, 4.0}};
    Matrix<double> B = {{5.0, 6.0},
                        {7.0, 8.0}};
    
    auto C = A * B;
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_DOUBLE_EQ(C(0, 0), 19.0);  // 1*5 + 2*7 = 19
    EXPECT_DOUBLE_EQ(C(0, 1), 22.0);  // 1*6 + 2*8 = 22
    EXPECT_DOUBLE_EQ(C(1, 0), 43.0);  // 3*5 + 4*7 = 43
    EXPECT_DOUBLE_EQ(C(1, 1), 50.0);  // 3*6 + 4*8 = 50
}

TEST(MatrixMatrix, NonSquareMultiplication) {
    Matrix<double> A(2, 3);  // 2x3
    Matrix<double> B(3, 4);  // 3x4
    
    // Initialize with some values
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            A(i, j) = static_cast<double>(i) * static_cast<double>(A.cols()) + static_cast<double>(j) + 1.0;
        }
    }
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            B(i, j) = static_cast<double>(i) * static_cast<double>(B.cols()) + static_cast<double>(j) + 1.0;
        }
    }
    
    auto C = A * B;  // Should be 2x4
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 4);
}

TEST(MatrixMatrix, IdentityMultiplication) {
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    auto I = Matrix<double>::identity(3);
    
    auto B = A * I;
    auto C = I * A;
    
    // A * I = A
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_DOUBLE_EQ(B(i, j), A(i, j));
            EXPECT_DOUBLE_EQ(C(i, j), A(i, j));
        }
    }
}

// ============================================================================
// Norms
// ============================================================================

TEST(MatrixNorms, FrobeniusNorm) {
    Matrix<double> A = {{3.0, 4.0},
                        {0.0, 0.0}};
    
    // Frobenius norm = sqrt(9 + 16) = 5
    EXPECT_DOUBLE_EQ(A.frobenius_norm(), 5.0);
    
    // Test with larger matrix
    Matrix<double> B = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}};
    double expected = std::sqrt(1 + 4 + 9 + 16 + 25 + 36);
    EXPECT_NEAR(B.frobenius_norm(), expected, 1e-10);
}

TEST(MatrixNorms, MaxNorm) {
    Matrix<double> A = {{-5.0, 2.0},
                        {3.0, -4.0}};
    
    EXPECT_DOUBLE_EQ(A.max_norm(), 5.0);
}

// ============================================================================
// Utility Operations
// ============================================================================

TEST(MatrixUtility, SwapRows) {
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    
    A.swap_rows(0, 2);
    
    EXPECT_DOUBLE_EQ(A(0, 0), 7.0);
    EXPECT_DOUBLE_EQ(A(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(A(0, 2), 9.0);
    EXPECT_DOUBLE_EQ(A(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(2, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(2, 2), 3.0);
}

TEST(MatrixUtility, SwapColumns) {
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                        {7.0, 8.0, 9.0}};
    
    A.swap_cols(0, 2);
    
    EXPECT_DOUBLE_EQ(A(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(A(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(A(1, 2), 4.0);
}

TEST(MatrixUtility, Reshape) {
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}};
    
    A.reshape(3, 2);
    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 2);
    
    // Check that data is preserved (row-major order)
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(A(2, 0), 5.0);
    EXPECT_DOUBLE_EQ(A(2, 1), 6.0);
    
    // Invalid reshape should throw
    EXPECT_THROW(A.reshape(2, 4), std::invalid_argument);
}

// ============================================================================
// Complex Number Support
// ============================================================================

TEST(MatrixComplex, ComplexMatrixOperations) {
    using Complex = std::complex<double>;
    Matrix<Complex> A = {{Complex(1, 2), Complex(3, 4)},
                         {Complex(5, 6), Complex(7, 8)}};
    
    Matrix<Complex> B = {{Complex(1, 0), Complex(0, 1)},
                         {Complex(0, -1), Complex(1, 0)}};
    
    auto C = A + B;
    EXPECT_EQ(C(0, 0), Complex(2, 2));
    EXPECT_EQ(C(0, 1), Complex(3, 5));
    
    auto D = A * B;
    // Verify a few elements of the product
    // (1+2i)(1) + (3+4i)(0-i) = (1+2i) + (4-3i) = (5 - i)
    EXPECT_EQ(D(0, 0), Complex(5, -1));
}

// ============================================================================
// Performance and Edge Cases
// ============================================================================

TEST(MatrixPerformance, LargeMatrixMultiplication) {
    const size_t n = 100;
    Matrix<double> A(n, n);
    Matrix<double> B(n, n);
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = dis(gen);
            B(i, j) = dis(gen);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    
    // Just verify it completes and has correct dimensions
    EXPECT_EQ(C.rows(), n);
    EXPECT_EQ(C.cols(), n);
    
    // Optional: print timing
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "100x100 matrix multiplication took: " << duration.count() << " ms\n";
}

TEST(MatrixEdgeCases, EmptyMatrix) {
    Matrix<double> A;
    Matrix<double> B;
    
    EXPECT_TRUE(A.empty());
    EXPECT_EQ(A.size(), 0);
    
    // Operations on empty matrices
    EXPECT_NO_THROW(A.frobenius_norm());
    EXPECT_EQ(A.frobenius_norm(), 0.0);
}

TEST(MatrixEdgeCases, SingleElement) {
    Matrix<double> A(1, 1, 42.0);
    EXPECT_EQ(A.rows(), 1);
    EXPECT_EQ(A.cols(), 1);
    EXPECT_EQ(A(0, 0), 42.0);
    EXPECT_TRUE(A.is_square());
    
    Vector<double> x = {2.0};
    auto y = A * x;
    EXPECT_EQ(y.size(), 1);
    EXPECT_EQ(y[0], 84.0);
}

// ============================================================================
// Storage Order Tests
// ============================================================================

TEST(MatrixStorage, ColumnMajorStorage) {
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> A(2, 3);
    
    // Set values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(i * 3 + j);
        }
    }
    
    // Verify values are correctly stored/retrieved
    EXPECT_EQ(A(0, 0), 0.0);
    EXPECT_EQ(A(0, 1), 1.0);
    EXPECT_EQ(A(0, 2), 2.0);
    EXPECT_EQ(A(1, 0), 3.0);
    EXPECT_EQ(A(1, 1), 4.0);
    EXPECT_EQ(A(1, 2), 5.0);
    
    // Test operations work correctly with column-major
    Vector<double> x = {1.0, 2.0, 3.0};
    auto y = A * x;
    EXPECT_DOUBLE_EQ(y[0], 8.0);   // 0*1 + 1*2 + 2*3 = 8
    EXPECT_DOUBLE_EQ(y[1], 26.0);  // 3*1 + 4*2 + 5*3 = 26
}

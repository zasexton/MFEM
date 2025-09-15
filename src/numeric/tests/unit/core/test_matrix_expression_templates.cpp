#include <gtest/gtest.h>
#include <core/matrix.h>
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

using namespace fem::numeric;

class MatrixExpressionTemplateTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    std::mt19937 gen;
    const double tolerance = 1e-12;
};

// === Expression Template Interface Tests ===

TEST_F(MatrixExpressionTemplateTest, ExpressionTemplateInterface) {
    Matrix<double> mat(3, 4);
    // Fill with test data
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat(i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }
    
    // Test shape method
    auto shape = mat.shape();
    EXPECT_EQ(shape.rank(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape.size(), 12);
    
    // Test eval method (linear indexing)
    for (size_t i = 0; i < 12; ++i) {
        double expected = static_cast<double>(i + 1);
        EXPECT_EQ(mat.eval<double>(i), expected);
    }
    
    // Test eval_at method
    EXPECT_EQ(mat.eval_at<double>(0, 0), 1.0);
    EXPECT_EQ(mat.eval_at<double>(1, 2), 7.0); // Row 1, Col 2 = 1*4 + 2 + 1 = 7
    EXPECT_EQ(mat.eval_at<double>(2, 3), 12.0); // Row 2, Col 3 = 2*4 + 3 + 1 = 12
    
    // Test eval_to method
    Matrix<float> result(3, 4);
    mat.eval_to(result);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), static_cast<float>(mat(i, j)));
        }
    }
    
    // Test interface properties
    EXPECT_TRUE(mat.is_parallelizable());
    EXPECT_TRUE(mat.is_vectorizable());
    EXPECT_EQ(mat.complexity(), 12);
}

// === Lazy Evaluation Tests ===

TEST_F(MatrixExpressionTemplateTest, LazyEvaluationBasic) {
    Matrix<double> m1(2, 2);
    Matrix<double> m2(2, 2);
    
    m1(0, 0) = 1; m1(0, 1) = 2;
    m1(1, 0) = 3; m1(1, 1) = 4;
    
    m2(0, 0) = 5; m2(0, 1) = 6;
    m2(1, 0) = 7; m2(1, 1) = 8;
    
    // Create expression - should not evaluate yet
    auto expr = m1 + m2;
    
    // Expression should have correct shape
    EXPECT_EQ(expr.shape().rank(), 2);
    EXPECT_EQ(expr.shape()[0], 2);
    EXPECT_EQ(expr.shape()[1], 2);
    
    // Test lazy evaluation
    EXPECT_EQ(expr.template eval<double>(0), 6.0);  // m1(0,0) + m2(0,0) = 1 + 5
    EXPECT_EQ(expr.template eval<double>(1), 8.0);  // m1(0,1) + m2(0,1) = 2 + 6
    EXPECT_EQ(expr.template eval<double>(2), 10.0); // m1(1,0) + m2(1,0) = 3 + 7
    EXPECT_EQ(expr.template eval<double>(3), 12.0); // m1(1,1) + m2(1,1) = 4 + 8
    
    // Assignment triggers evaluation
    Matrix<double> result = expr;
    EXPECT_EQ(result(0, 0), 6.0);
    EXPECT_EQ(result(0, 1), 8.0);
    EXPECT_EQ(result(1, 0), 10.0);
    EXPECT_EQ(result(1, 1), 12.0);
}

TEST_F(MatrixExpressionTemplateTest, ComplexExpressionChain) {
    Matrix<double> m1(2, 3);
    Matrix<double> m2(2, 3);
    Matrix<double> m3(2, 3);
    
    // Initialize matrices
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1(i, j) = 1.0;
            m2(i, j) = 2.0;
            m3(i, j) = 0.5;
        }
    }
    
    // Complex expression: (m1 + m2) * 3.0 - m3
    auto expr = (m1 + m2) * 3.0 - m3;
    
    Matrix<double> result = expr;
    
    // Expected: ((1+2)*3-0.5) = (9-0.5) = 8.5 for all elements
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(result(i, j), 8.5);
        }
    }
}

// === Broadcasting Tests ===

TEST_F(MatrixExpressionTemplateTest, ScalarBroadcasting) {
    Matrix<double> mat(3, 2);
    
    // Initialize matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            mat(i, j) = static_cast<double>(i * 2 + j + 1);
        }
    }
    
    // Matrix + Scalar
    {
        auto expr = mat + 10.0;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), mat(i, j) + 10.0);
            }
        }
    }
    
    // Scalar + Matrix
    {
        auto expr = 20.0 + mat;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), 20.0 + mat(i, j));
            }
        }
    }
    
    // Matrix * Scalar
    {
        auto expr = mat * 3.0;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), mat(i, j) * 3.0);
            }
        }
    }
    
    // Scalar * Matrix
    {
        auto expr = 4.0 * mat;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), 4.0 * mat(i, j));
            }
        }
    }
    
    // Matrix / Scalar
    {
        auto expr = mat / 2.0;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), mat(i, j) / 2.0);
            }
        }
    }
}

// === Matrix-Matrix Operations ===

TEST_F(MatrixExpressionTemplateTest, MatrixMatrixArithmetic) {
    Matrix<double> m1(2, 3);
    Matrix<double> m2(2, 3);
    
    // Initialize matrices
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1(i, j) = static_cast<double>(i * 3 + j + 1);
            m2(i, j) = static_cast<double>((i * 3 + j) * 2 + 1);
        }
    }
    
    // Addition
    {
        auto expr = m1 + m2;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(result(i, j), m1(i, j) + m2(i, j));
            }
        }
    }
    
    // Subtraction
    {
        auto expr = m2 - m1;
        Matrix<double> result = expr;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(result(i, j), m2(i, j) - m1(i, j));
            }
        }
    }
}

// === Unary Operations ===

TEST_F(MatrixExpressionTemplateTest, UnaryOperations) {
    Matrix<double> mat(2, 2);
    mat(0, 0) = 1.0;  mat(0, 1) = -2.0;
    mat(1, 0) = 3.0;  mat(1, 1) = -4.0;
    
    // Unary negation
    auto expr = -mat;
    Matrix<double> result = expr;
    
    EXPECT_EQ(result(0, 0), -1.0);
    EXPECT_EQ(result(0, 1), 2.0);
    EXPECT_EQ(result(1, 0), -3.0);
    EXPECT_EQ(result(1, 1), 4.0);
}

// === Storage Order Tests ===

TEST_F(MatrixExpressionTemplateTest, StorageOrderCompatibility) {
    Matrix<double, DynamicStorage<double>, StorageOrder::RowMajor> row_major(2, 3);
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> col_major(2, 3);
    
    // Fill matrices
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double val = static_cast<double>(i * 3 + j + 1);
            row_major(i, j) = val;
            col_major(i, j) = val;
        }
    }
    
    // Expression with different storage orders
    auto expr = row_major + col_major;
    Matrix<double> result = expr;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double expected = 2.0 * static_cast<double>(i * 3 + j + 1);
            EXPECT_EQ(result(i, j), expected);
        }
    }
}

// === Type Deduction Tests ===

TEST_F(MatrixExpressionTemplateTest, TypeDeduction) {
    Matrix<int> mi(2, 2);
    Matrix<double> md(2, 2);
    
    mi(0, 0) = 1; mi(0, 1) = 2;
    mi(1, 0) = 3; mi(1, 1) = 4;
    
    md(0, 0) = 1.5; md(0, 1) = 2.5;
    md(1, 0) = 3.5; md(1, 1) = 4.5;
    
    // int + double should give double
    auto expr = mi + md;
    Matrix<double> result = expr;
    
    EXPECT_NEAR(result(0, 0), 2.5, tolerance);
    EXPECT_NEAR(result(0, 1), 4.5, tolerance);
    EXPECT_NEAR(result(1, 0), 6.5, tolerance);
    EXPECT_NEAR(result(1, 1), 8.5, tolerance);
}

// === Complex Numbers ===

TEST_F(MatrixExpressionTemplateTest, ComplexNumberSupport) {
    using Complex = std::complex<double>;
    
    Matrix<Complex> m1(2, 2);
    Matrix<Complex> m2(2, 2);
    
    m1(0, 0) = Complex(1, 2); m1(0, 1) = Complex(3, 4);
    m1(1, 0) = Complex(5, 6); m1(1, 1) = Complex(7, 8);
    
    m2(0, 0) = Complex(1, 1); m2(0, 1) = Complex(2, 2);
    m2(1, 0) = Complex(3, 3); m2(1, 1) = Complex(4, 4);
    
    auto expr = m1 + m2;
    Matrix<Complex> result = expr;
    
    EXPECT_EQ(result(0, 0), Complex(2, 3));   // (1+2i) + (1+1i) = (2+3i)
    EXPECT_EQ(result(0, 1), Complex(5, 6));   // (3+4i) + (2+2i) = (5+6i)
    EXPECT_EQ(result(1, 0), Complex(8, 9));   // (5+6i) + (3+3i) = (8+9i)
    EXPECT_EQ(result(1, 1), Complex(11, 12)); // (7+8i) + (4+4i) = (11+12i)
}

// === Performance Tests ===

TEST_F(MatrixExpressionTemplateTest, NoTemporaryCreation) {
    const size_t rows = 100, cols = 100;
    Matrix<double> m1(rows, cols, 1.0);
    Matrix<double> m2(rows, cols, 2.0);
    Matrix<double> m3(rows, cols, 3.0);
    Matrix<double> m4(rows, cols, 4.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Complex expression should not create temporaries
    auto expr = (m1 + m2) * (m3 - m4) + m1 * 2.5;
    Matrix<double> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify correctness for a sample of elements
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            double expected = (1.0 + 2.0) * (3.0 - 4.0) + 1.0 * 2.5; // -0.5
            EXPECT_NEAR(result(i, j), expected, tolerance);
        }
    }
    
    EXPECT_LT(duration.count(), 50000); // Less than 50ms
}

// === Expression Assignment Tests ===

TEST_F(MatrixExpressionTemplateTest, ExpressionAssignment) {
    Matrix<double> m1(3, 3);
    Matrix<double> m2(3, 3);
    
    // Fill matrices
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1(i, j) = static_cast<double>(i * 3 + j + 1);
            m2(i, j) = static_cast<double>((i * 3 + j) * 2);
        }
    }
    
    Matrix<double> result(3, 3);
    
    // Assign expression to existing matrix
    auto expr = m1 * 2.0 + m2;
    result = expr;
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double expected = m1(i, j) * 2.0 + m2(i, j);
            EXPECT_EQ(result(i, j), expected);
        }
    }
}

// === Error Handling ===

TEST_F(MatrixExpressionTemplateTest, SizeCompatibilityCheck) {
    Matrix<double> m1(2, 3);
    Matrix<double> m2(3, 2); // Different dimensions
    
    m1(0, 0) = 1.0; m2(0, 0) = 2.0;
    
    // Expression creation should succeed
    auto expr = m1 + m2;
    
    // But evaluation should throw due to dimension mismatch
    EXPECT_THROW({
        Matrix<double> result = expr;
    }, std::exception);
}

// === Chained Assignment ===

TEST_F(MatrixExpressionTemplateTest, ChainedAssignment) {
    Matrix<double> m1(2, 2, 1.0);
    Matrix<double> m2(2, 2, 2.0);
    Matrix<double> result1(2, 2), result2(2, 2), result3(2, 2);
    
    // Chained assignment from same expression
    auto expr = m1 * 3.0 + m2;
    
    result1 = expr;
    result2 = expr;
    result3 = expr;
    
    // All should have same values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double expected = 1.0 * 3.0 + 2.0; // 5.0
            EXPECT_EQ(result1(i, j), expected);
            EXPECT_EQ(result2(i, j), expected);
            EXPECT_EQ(result3(i, j), expected);
        }
    }
}

// === Matrix-Vector Multiplication with Expressions ===

TEST_F(MatrixExpressionTemplateTest, MatrixVectorMultiplication) {
    Matrix<double> mat(3, 3);
    Vector<double> vec{1.0, 2.0, 3.0};
    
    // Initialize matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            mat(i, j) = static_cast<double>(i * 3 + j + 1);
        }
    }
    
    // Test that we can still use matrix-vector multiplication
    // even with expression template integrated matrices
    auto result = mat * vec;
    
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 14.0);  // 1*1 + 2*2 + 3*3 = 14
    EXPECT_EQ(result[1], 32.0);  // 4*1 + 5*2 + 6*3 = 32
    EXPECT_EQ(result[2], 50.0);  // 7*1 + 8*2 + 9*3 = 50
}

// === Large Matrix Performance ===

TEST_F(MatrixExpressionTemplateTest, LargeMatrixPerformance) {
    const size_t N = 200;
    Matrix<double> m1(N, N), m2(N, N), m3(N, N);
    
    // Fill with data
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m1(i, j) = dist(gen);
            m2(i, j) = dist(gen);
            m3(i, j) = dist(gen);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Complex expression
    auto expr = (m1 + m2) * 0.5 - m3 / 2.0;
    Matrix<double> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify a few elements
    for (size_t i = 0; i < 10; i += 2) {
        for (size_t j = 0; j < 10; j += 2) {
            double expected = (m1(i, j) + m2(i, j)) * 0.5 - m3(i, j) / 2.0;
            EXPECT_NEAR(result(i, j), expected, tolerance);
        }
    }
    
    EXPECT_LT(duration.count(), 500); // Should complete in less than 500ms
}

// === Mixed Type Operations ===

TEST_F(MatrixExpressionTemplateTest, MixedTypeOperations) {
    Matrix<float> mf(2, 2);
    Matrix<double> md(2, 2);
    
    mf(0, 0) = 1.0f; mf(0, 1) = 2.0f;
    mf(1, 0) = 3.0f; mf(1, 1) = 4.0f;
    
    md(0, 0) = 1.5; md(0, 1) = 2.5;
    md(1, 0) = 3.5; md(1, 1) = 4.5;
    
    // float + double -> double
    auto expr = mf + md;
    Matrix<double> result = expr;
    
    EXPECT_NEAR(result(0, 0), 2.5, tolerance);
    EXPECT_NEAR(result(0, 1), 4.5, tolerance);
    EXPECT_NEAR(result(1, 0), 6.5, tolerance);
    EXPECT_NEAR(result(1, 1), 8.5, tolerance);
    
    // Also test with scalars
    auto expr2 = mf * 2.5;  // float * double -> double
    Matrix<double> result2 = expr2;
    
    EXPECT_NEAR(result2(0, 0), 2.5, tolerance);
    EXPECT_NEAR(result2(0, 1), 5.0, tolerance);
    EXPECT_NEAR(result2(1, 0), 7.5, tolerance);
    EXPECT_NEAR(result2(1, 1), 10.0, tolerance);
}

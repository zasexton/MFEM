#include <gtest/gtest.h>
#include <core/vector.h>
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

using namespace fem::numeric;

class VectorExpressionTemplateTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    std::mt19937 gen;
    const double tolerance = 1e-12;
};

// === Expression Template Interface Tests ===

TEST_F(VectorExpressionTemplateTest, ExpressionTemplateInterface) {
    Vector<double> vec{1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Test shape method
    auto shape = vec.shape();
    EXPECT_EQ(shape.rank(), 1);
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape.size(), 5);
    
    // Test eval method
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(vec.eval<double>(i), vec[i]);
    }
    
    // Test eval_at method
    EXPECT_EQ(vec.eval_at<double>(2), 3.0);
    EXPECT_EQ(vec.eval_at<double>(0), 1.0);
    EXPECT_EQ(vec.eval_at<double>(4), 5.0);
    
    // Test eval_to method
    Vector<float> result(5);
    vec.eval_to(result);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(vec[i]));
    }
    
    // Test interface properties
    EXPECT_TRUE(vec.is_parallelizable());
    EXPECT_TRUE(vec.is_vectorizable());
    EXPECT_EQ(vec.complexity(), 5);
}

// === Lazy Evaluation Tests ===

TEST_F(VectorExpressionTemplateTest, LazyEvaluationBasic) {
    Vector<double> v1{1.0, 2.0, 3.0};
    Vector<double> v2{4.0, 5.0, 6.0};
    
    // Create expression - should not evaluate yet
    auto expr = v1 + v2;
    
    // Expression should have correct shape
    EXPECT_EQ(expr.shape().rank(), 1);
    EXPECT_EQ(expr.shape()[0], 3);
    
    // Test lazy evaluation
    EXPECT_EQ(expr.template eval<double>(0), 5.0);  // 1 + 4
    EXPECT_EQ(expr.template eval<double>(1), 7.0);  // 2 + 5
    EXPECT_EQ(expr.template eval<double>(2), 9.0);  // 3 + 6
    
    // Assignment triggers evaluation
    Vector<double> result = expr;
    EXPECT_EQ(result[0], 5.0);
    EXPECT_EQ(result[1], 7.0);
    EXPECT_EQ(result[2], 9.0);
}

TEST_F(VectorExpressionTemplateTest, ComplexExpressionChain) {
    Vector<double> v1{1.0, 2.0, 3.0, 4.0};
    Vector<double> v2{2.0, 3.0, 4.0, 5.0};
    Vector<double> v3{1.0, 1.0, 1.0, 1.0};
    
    // Complex expression: (v1 + v2) * 2.0 - v3
    auto expr = (v1 + v2) * 2.0 - v3;
    
    Vector<double> result = expr;
    
    // Expected: ((1+2)*2-1, (2+3)*2-1, (3+4)*2-1, (4+5)*2-1) = (5, 9, 13, 17)
    EXPECT_EQ(result[0], 5.0);
    EXPECT_EQ(result[1], 9.0);
    EXPECT_EQ(result[2], 13.0);
    EXPECT_EQ(result[3], 17.0);
}

// === Broadcasting Tests ===

TEST_F(VectorExpressionTemplateTest, ScalarBroadcasting) {
    Vector<double> vec{1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Vector + Scalar
    {
        auto expr = vec + 10.0;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], vec[i] + 10.0);
        }
    }
    
    // Scalar + Vector
    {
        auto expr = 20.0 + vec;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], 20.0 + vec[i]);
        }
    }
    
    // Vector - Scalar
    {
        auto expr = vec - 2.5;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], vec[i] - 2.5);
        }
    }
    
    // Scalar - Vector
    {
        auto expr = 10.0 - vec;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], 10.0 - vec[i]);
        }
    }
    
    // Vector * Scalar
    {
        auto expr = vec * 3.0;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], vec[i] * 3.0);
        }
    }
    
    // Scalar * Vector  
    {
        auto expr = 4.0 * vec;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], 4.0 * vec[i]);
        }
    }
    
    // Vector / Scalar
    {
        auto expr = vec / 2.0;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result[i], vec[i] / 2.0);
        }
    }
}

// === Vector-Vector Operations ===

TEST_F(VectorExpressionTemplateTest, VectorVectorArithmetic) {
    Vector<double> v1{1.0, 2.0, 3.0, 4.0};
    Vector<double> v2{5.0, 6.0, 7.0, 8.0};
    
    // Addition
    {
        auto expr = v1 + v2;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(result[i], v1[i] + v2[i]);
        }
    }
    
    // Subtraction
    {
        auto expr = v2 - v1;
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(result[i], v2[i] - v1[i]);
        }
    }
    
    // Element-wise multiplication
    {
        auto expr = multiply(v1, v2);
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(result[i], v1[i] * v2[i]);
        }
    }
    
    // Element-wise division
    {
        auto expr = divide(v2, v1);
        Vector<double> result = expr;
        
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(result[i], v2[i] / v1[i]);
        }
    }
}

// === Unary Operations ===

TEST_F(VectorExpressionTemplateTest, UnaryOperations) {
    Vector<double> vec{1.0, -2.0, 3.0, -4.0};
    
    // Unary negation
    auto expr = -vec;
    Vector<double> result = expr;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], -vec[i]);
    }
}

// === Type Deduction Tests ===

TEST_F(VectorExpressionTemplateTest, TypeDeduction) {
    Vector<int> vi{1, 2, 3};
    Vector<double> vd{1.5, 2.5, 3.5};
    
    // int + double should give double
    auto expr = vi + vd;
    Vector<double> result = expr;
    
    EXPECT_NEAR(result[0], 2.5, tolerance);
    EXPECT_NEAR(result[1], 4.5, tolerance);
    EXPECT_NEAR(result[2], 6.5, tolerance);
}

// === Complex Numbers ===

TEST_F(VectorExpressionTemplateTest, ComplexNumberSupport) {
    using Complex = std::complex<double>;
    
    Vector<Complex> v1{Complex(1, 2), Complex(3, 4)};
    Vector<Complex> v2{Complex(5, 6), Complex(7, 8)};
    
    auto expr = v1 + v2;
    Vector<Complex> result = expr;
    
    EXPECT_EQ(result[0], Complex(6, 8));   // (1+2i) + (5+6i) = (6+8i)
    EXPECT_EQ(result[1], Complex(10, 12)); // (3+4i) + (7+8i) = (10+12i)
}

// === Performance and Memory Tests ===

TEST_F(VectorExpressionTemplateTest, NoTemporaryCreation) {
    const size_t N = 1000;
    Vector<double> v1(N, 1.0);
    Vector<double> v2(N, 2.0);
    Vector<double> v3(N, 3.0);
    Vector<double> v4(N, 4.0);
    
    // Complex expression should not create temporaries
    auto start = std::chrono::high_resolution_clock::now();
    
    auto expr = (v1 + v2) * (v3 - v4) + v1 * 2.0;
    Vector<double> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify correctness
    for (size_t i = 0; i < N; ++i) {
        double expected = (v1[i] + v2[i]) * (v3[i] - v4[i]) + v1[i] * 2.0;
        EXPECT_NEAR(result[i], expected, tolerance);
    }
    
    // Performance should be reasonable
    EXPECT_LT(duration.count(), 10000); // Less than 10ms
}

// === Error Handling ===

TEST_F(VectorExpressionTemplateTest, SizeCompatibilityCheck) {
    Vector<double> v1{1.0, 2.0, 3.0};
    Vector<double> v2{4.0, 5.0}; // Different size
    
    // This should work at expression creation
    auto expr = v1 + v2;
    
    // But should throw during evaluation/assignment due to size mismatch
    EXPECT_THROW({
        Vector<double> result = expr;
    }, std::exception);
}

// === Chained Assignment ===

TEST_F(VectorExpressionTemplateTest, ChainedAssignment) {
    Vector<double> v1{1.0, 2.0, 3.0};
    Vector<double> v2{4.0, 5.0, 6.0};
    Vector<double> result1(3), result2(3), result3(3);
    
    // Chained assignment from same expression
    auto expr = v1 * 2.0 + v2;
    
    result1 = expr;
    result2 = expr; 
    result3 = expr;
    
    // All should have same values
    for (size_t i = 0; i < 3; ++i) {
        double expected = v1[i] * 2.0 + v2[i];
        EXPECT_EQ(result1[i], expected);
        EXPECT_EQ(result2[i], expected);
        EXPECT_EQ(result3[i], expected);
    }
}

// === Expression Properties ===

TEST_F(VectorExpressionTemplateTest, ExpressionProperties) {
    Vector<double> v1{1.0, 2.0, 3.0};
    Vector<double> v2{4.0, 5.0, 6.0};
    
    auto expr = v1 + v2 * 2.0;
    
    EXPECT_TRUE(expr.is_parallelizable());
    EXPECT_TRUE(expr.is_vectorizable());
    EXPECT_GT(expr.complexity(), 0);
}

// === Move Semantics ===

TEST_F(VectorExpressionTemplateTest, MoveSemanticsSupport) {
    Vector<double> v1{1.0, 2.0, 3.0};
    
    // Test with temporary vector
    auto expr = std::move(v1) + Vector<double>{4.0, 5.0, 6.0};
    Vector<double> result = expr;
    
    EXPECT_EQ(result[0], 5.0);
    EXPECT_EQ(result[1], 7.0);
    EXPECT_EQ(result[2], 9.0);
}

// === Large Vector Performance ===

TEST_F(VectorExpressionTemplateTest, LargeVectorPerformance) {
    const size_t N = 100000;
    Vector<double> v1(N), v2(N), v3(N);
    
    // Fill with random data
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (size_t i = 0; i < N; ++i) {
        v1[i] = dist(gen);
        v2[i] = dist(gen);
        v3[i] = dist(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Complex expression
    auto expr = (v1 + v2) * 0.5 - v3 / 2.0;
    Vector<double> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify a few elements
    for (size_t i = 0; i < 100; i += 10) {
        double expected = (v1[i] + v2[i]) * 0.5 - v3[i] / 2.0;
        EXPECT_NEAR(result[i], expected, tolerance);
    }
    
    EXPECT_LT(duration.count(), 100); // Should complete in less than 100ms
}

// === Mixed Type Operations ===

TEST_F(VectorExpressionTemplateTest, MixedTypeOperations) {
    Vector<float> vf{1.0f, 2.0f, 3.0f};
    Vector<double> vd{1.5, 2.5, 3.5};
    
    // float + double -> double
    auto expr = vf + vd;
    Vector<double> result = expr;
    
    EXPECT_NEAR(result[0], 2.5, tolerance);
    EXPECT_NEAR(result[1], 4.5, tolerance);
    EXPECT_NEAR(result[2], 6.5, tolerance);
    
    // Also test with scalars
    auto expr2 = vf * 2.5;  // float * double -> double
    Vector<double> result2 = expr2;
    
    EXPECT_NEAR(result2[0], 2.5, tolerance);
    EXPECT_NEAR(result2[1], 5.0, tolerance);
    EXPECT_NEAR(result2[2], 7.5, tolerance);
}

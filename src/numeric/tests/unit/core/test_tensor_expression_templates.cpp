#include <gtest/gtest.h>
#include <core/tensor.h>
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

using namespace fem::numeric;

class TensorExpressionTemplateTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    std::mt19937 gen;
    const double tolerance = 1e-12;
};

// === Expression Template Interface Tests ===

TEST_F(TensorExpressionTemplateTest, ExpressionTemplateInterface1D) {
    Tensor<double, 1> tensor({5});
    
    // Fill with test data
    for (size_t i = 0; i < 5; ++i) {
        tensor(i) = static_cast<double>(i + 1);
    }
    
    // Test shape method
    auto shape = tensor.shape();
    EXPECT_EQ(shape.rank(), 1);
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape.size(), 5);
    
    // Test eval method
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(tensor.eval<double>(i), static_cast<double>(i + 1));
    }
    
    // Test eval_at method
    EXPECT_EQ(tensor.eval_at<double>(0), 1.0);
    EXPECT_EQ(tensor.eval_at<double>(2), 3.0);
    EXPECT_EQ(tensor.eval_at<double>(4), 5.0);
    
    // Test interface properties
    EXPECT_TRUE(tensor.is_parallelizable());
    EXPECT_TRUE(tensor.is_vectorizable());
    EXPECT_EQ(tensor.complexity(), 5);
}

TEST_F(TensorExpressionTemplateTest, ExpressionTemplateInterface2D) {
    Tensor<double, 2> tensor({3, 4});
    
    // Fill with test data
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            tensor(i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }
    
    // Test shape method
    auto shape = tensor.shape();
    EXPECT_EQ(shape.rank(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape.size(), 12);
    
    // Test eval method (linear indexing)
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(tensor.eval<double>(i), static_cast<double>(i + 1));
    }
    
    // Test eval_at method
    EXPECT_EQ(tensor.eval_at<double>(0, 0), 1.0);
    EXPECT_EQ(tensor.eval_at<double>(1, 2), 7.0); // 1*4 + 2 + 1 = 7
    EXPECT_EQ(tensor.eval_at<double>(2, 3), 12.0); // 2*4 + 3 + 1 = 12
}

TEST_F(TensorExpressionTemplateTest, ExpressionTemplateInterface3D) {
    Tensor<double, 3> tensor({2, 3, 4});
    
    // Fill with test data
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                tensor(i, j, k) = static_cast<double>(i * 12 + j * 4 + k + 1);
            }
        }
    }
    
    // Test shape method for higher rank tensors (flattened size)
    auto shape = tensor.shape();
    EXPECT_EQ(shape.size(), 24); // Total elements
    
    // Test eval method
    EXPECT_EQ(tensor.eval<double>(0), 1.0);   // (0,0,0)
    EXPECT_EQ(tensor.eval<double>(4), 5.0);   // (0,1,0)
    EXPECT_EQ(tensor.eval<double>(12), 13.0); // (1,0,0)
    
    // Test eval_at method
    EXPECT_EQ(tensor.eval_at<double>(0, 0, 0), 1.0);
    EXPECT_EQ(tensor.eval_at<double>(0, 1, 2), 7.0); // 0*12 + 1*4 + 2 + 1 = 7
    EXPECT_EQ(tensor.eval_at<double>(1, 2, 3), 24.0); // 1*12 + 2*4 + 3 + 1 = 24
}

// === Lazy Evaluation Tests ===

TEST_F(TensorExpressionTemplateTest, LazyEvaluation1D) {
    Tensor<double, 1> t1({3});
    Tensor<double, 1> t2({3});
    
    t1(0) = 1.0; t1(1) = 2.0; t1(2) = 3.0;
    t2(0) = 4.0; t2(1) = 5.0; t2(2) = 6.0;
    
    // Create expression - should not evaluate yet
    auto expr = t1 + t2;
    
    // Expression should have correct shape
    EXPECT_EQ(expr.shape().rank(), 1);
    EXPECT_EQ(expr.shape()[0], 3);
    
    // Test lazy evaluation
    EXPECT_EQ(expr.template eval<double>(0), 5.0);  // 1 + 4
    EXPECT_EQ(expr.template eval<double>(1), 7.0);  // 2 + 5
    EXPECT_EQ(expr.template eval<double>(2), 9.0);  // 3 + 6
    
    // Assignment triggers evaluation
    Tensor<double, 1> result = expr;
    EXPECT_EQ(result(0), 5.0);
    EXPECT_EQ(result(1), 7.0);
    EXPECT_EQ(result(2), 9.0);
}

TEST_F(TensorExpressionTemplateTest, LazyEvaluation2D) {
    Tensor<double, 2> t1({2, 2});
    Tensor<double, 2> t2({2, 2});
    
    t1(0, 0) = 1; t1(0, 1) = 2;
    t1(1, 0) = 3; t1(1, 1) = 4;
    
    t2(0, 0) = 5; t2(0, 1) = 6;
    t2(1, 0) = 7; t2(1, 1) = 8;
    
    // Create expression - should not evaluate yet
    auto expr = t1 + t2;
    
    // Expression should have correct shape
    EXPECT_EQ(expr.shape().rank(), 2);
    EXPECT_EQ(expr.shape()[0], 2);
    EXPECT_EQ(expr.shape()[1], 2);
    
    // Test lazy evaluation
    EXPECT_EQ(expr.template eval<double>(0), 6.0);   // 1 + 5
    EXPECT_EQ(expr.template eval<double>(1), 8.0);   // 2 + 6
    EXPECT_EQ(expr.template eval<double>(2), 10.0);  // 3 + 7
    EXPECT_EQ(expr.template eval<double>(3), 12.0);  // 4 + 8
    
    // Assignment triggers evaluation
    Tensor<double, 2> result = expr;
    EXPECT_EQ(result(0, 0), 6.0);
    EXPECT_EQ(result(0, 1), 8.0);
    EXPECT_EQ(result(1, 0), 10.0);
    EXPECT_EQ(result(1, 1), 12.0);
}

TEST_F(TensorExpressionTemplateTest, ComplexExpressionChain) {
    Tensor<double, 2> t1({2, 3});
    Tensor<double, 2> t2({2, 3});
    Tensor<double, 2> t3({2, 3});
    
    // Initialize tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t1(i, j) = 1.0;
            t2(i, j) = 2.0;
            t3(i, j) = 0.5;
        }
    }
    
    // Complex expression: (t1 + t2) * 3.0 - t3
    auto expr = (t1 + t2) * 3.0 - t3;
    
    Tensor<double, 2> result = expr;
    
    // Expected: ((1+2)*3-0.5) = (9-0.5) = 8.5 for all elements
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(result(i, j), 8.5);
        }
    }
}

// === Broadcasting Tests ===

TEST_F(TensorExpressionTemplateTest, ScalarBroadcasting1D) {
    Tensor<double, 1> tensor({5});
    
    // Initialize tensor
    for (size_t i = 0; i < 5; ++i) {
        tensor(i) = static_cast<double>(i + 1);
    }
    
    // Tensor + Scalar
    {
        auto expr = tensor + 10.0;
        Tensor<double, 1> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result(i), tensor(i) + 10.0);
        }
    }
    
    // Scalar * Tensor
    {
        auto expr = 3.0 * tensor;
        Tensor<double, 1> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result(i), 3.0 * tensor(i));
        }
    }
    
    // Tensor / Scalar
    {
        auto expr = tensor / 2.0;
        Tensor<double, 1> result = expr;
        
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(result(i), tensor(i) / 2.0);
        }
    }
}

TEST_F(TensorExpressionTemplateTest, ScalarBroadcasting2D) {
    Tensor<double, 2> tensor({3, 2});
    
    // Initialize tensor
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            tensor(i, j) = static_cast<double>(i * 2 + j + 1);
        }
    }
    
    // Tensor + Scalar
    {
        auto expr = tensor + 5.0;
        Tensor<double, 2> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), tensor(i, j) + 5.0);
            }
        }
    }
    
    // Scalar * Tensor
    {
        auto expr = 2.5 * tensor;
        Tensor<double, 2> result = expr;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_EQ(result(i, j), 2.5 * tensor(i, j));
            }
        }
    }
}

TEST_F(TensorExpressionTemplateTest, ScalarBroadcasting3D) {
    Tensor<double, 3> tensor({2, 2, 2});
    
    // Initialize tensor
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                tensor(i, j, k) = static_cast<double>(i * 4 + j * 2 + k + 1);
            }
        }
    }
    
    // Tensor * Scalar
    {
        auto expr = tensor * 0.5;
        Tensor<double, 3> result = expr;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 2; ++k) {
                    EXPECT_EQ(result(i, j, k), tensor(i, j, k) * 0.5);
                }
            }
        }
    }
}

// === Tensor-Tensor Operations ===

TEST_F(TensorExpressionTemplateTest, TensorTensorArithmetic) {
    Tensor<double, 2> t1({2, 3});
    Tensor<double, 2> t2({2, 3});
    
    // Initialize tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t1(i, j) = static_cast<double>(i * 3 + j + 1);
            t2(i, j) = static_cast<double>((i * 3 + j) * 2 + 1);
        }
    }
    
    // Addition
    {
        auto expr = t1 + t2;
        Tensor<double, 2> result = expr;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(result(i, j), t1(i, j) + t2(i, j));
            }
        }
    }
    
    // Subtraction
    {
        auto expr = t2 - t1;
        Tensor<double, 2> result = expr;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(result(i, j), t2(i, j) - t1(i, j));
            }
        }
    }
}

// === Unary Operations ===

TEST_F(TensorExpressionTemplateTest, UnaryOperations) {
    Tensor<double, 2> tensor({2, 2});
    tensor(0, 0) = 1.0;  tensor(0, 1) = -2.0;
    tensor(1, 0) = 3.0;  tensor(1, 1) = -4.0;
    
    // Unary negation
    auto expr = -tensor;
    Tensor<double, 2> result = expr;
    
    EXPECT_EQ(result(0, 0), -1.0);
    EXPECT_EQ(result(0, 1), 2.0);
    EXPECT_EQ(result(1, 0), -3.0);
    EXPECT_EQ(result(1, 1), 4.0);
}

// === Type Deduction Tests ===

TEST_F(TensorExpressionTemplateTest, TypeDeduction) {
    Tensor<int, 2> ti({2, 2});
    Tensor<double, 2> td({2, 2});
    
    ti(0, 0) = 1; ti(0, 1) = 2;
    ti(1, 0) = 3; ti(1, 1) = 4;
    
    td(0, 0) = 1.5; td(0, 1) = 2.5;
    td(1, 0) = 3.5; td(1, 1) = 4.5;
    
    // int + double should give double
    auto expr = ti + td;
    Tensor<double, 2> result = expr;
    
    EXPECT_NEAR(result(0, 0), 2.5, tolerance);
    EXPECT_NEAR(result(0, 1), 4.5, tolerance);
    EXPECT_NEAR(result(1, 0), 6.5, tolerance);
    EXPECT_NEAR(result(1, 1), 8.5, tolerance);
}

// === Complex Numbers ===

TEST_F(TensorExpressionTemplateTest, ComplexNumberSupport) {
    using Complex = std::complex<double>;
    
    Tensor<Complex, 2> t1({2, 2});
    Tensor<Complex, 2> t2({2, 2});
    
    t1(0, 0) = Complex(1, 2); t1(0, 1) = Complex(3, 4);
    t1(1, 0) = Complex(5, 6); t1(1, 1) = Complex(7, 8);
    
    t2(0, 0) = Complex(1, 1); t2(0, 1) = Complex(2, 2);
    t2(1, 0) = Complex(3, 3); t2(1, 1) = Complex(4, 4);
    
    auto expr = t1 + t2;
    Tensor<Complex, 2> result = expr;
    
    EXPECT_EQ(result(0, 0), Complex(2, 3));   // (1+2i) + (1+1i) = (2+3i)
    EXPECT_EQ(result(0, 1), Complex(5, 6));   // (3+4i) + (2+2i) = (5+6i)
    EXPECT_EQ(result(1, 0), Complex(8, 9));   // (5+6i) + (3+3i) = (8+9i)
    EXPECT_EQ(result(1, 1), Complex(11, 12)); // (7+8i) + (4+4i) = (11+12i)
}

// === Rank-0 Tensor (Scalar) Tests ===

TEST_F(TensorExpressionTemplateTest, ScalarTensorOperations) {
    Tensor<double, 0> s1(5.0);
    Tensor<double, 0> s2(3.0);
    
    // Addition
    {
        auto expr = s1 + s2;
        Tensor<double, 0> result = expr;
        // Access scalar tensor value (implementation specific)
        EXPECT_EQ(result[0], 8.0); // 5 + 3
    }
    
    // Multiplication with scalar
    {
        auto expr = s1 * 2.0;
        Tensor<double, 0> result = expr;
        EXPECT_EQ(result[0], 10.0); // 5 * 2
    }
}

// === Performance Tests ===

TEST_F(TensorExpressionTemplateTest, NoTemporaryCreation) {
    const size_t N = 50;
    Tensor<double, 3> t1({N, N, 2}, 1.0);
    Tensor<double, 3> t2({N, N, 2}, 2.0);
    Tensor<double, 3> t3({N, N, 2}, 3.0);
    Tensor<double, 3> t4({N, N, 2}, 4.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Complex expression should not create temporaries
    auto expr = (t1 + t2) * (t3 - t4) + t1 * 2.5;
    Tensor<double, 3> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify correctness for a sample of elements
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                double expected = (1.0 + 2.0) * (3.0 - 4.0) + 1.0 * 2.5; // -0.5
                EXPECT_NEAR(result(i, j, k), expected, tolerance);
            }
        }
    }
    
    EXPECT_LT(duration.count(), 100000); // Less than 100ms
}

// === Expression Assignment Tests ===

TEST_F(TensorExpressionTemplateTest, ExpressionAssignment) {
    Tensor<double, 3> t1({2, 2, 2});
    Tensor<double, 3> t2({2, 2, 2});
    
    // Fill tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                t1(i, j, k) = static_cast<double>(i * 4 + j * 2 + k + 1);
                t2(i, j, k) = static_cast<double>((i * 4 + j * 2 + k) * 2);
            }
        }
    }
    
    Tensor<double, 3> result({2, 2, 2});
    
    // Assign expression to existing tensor
    auto expr = t1 * 2.0 + t2;
    result = expr;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                double expected = t1(i, j, k) * 2.0 + t2(i, j, k);
                EXPECT_EQ(result(i, j, k), expected);
            }
        }
    }
}

// === Error Handling ===

TEST_F(TensorExpressionTemplateTest, SizeCompatibilityCheck) {
    Tensor<double, 2> t1({2, 3});
    Tensor<double, 2> t2({3, 2}); // Different dimensions
    
    t1(0, 0) = 1.0;
    t2(0, 0) = 2.0;
    
    // Expression creation should succeed
    auto expr = t1 + t2;
    
    // But evaluation should throw due to shape mismatch
    EXPECT_THROW(([&](){ Tensor<double, 2> result = expr; }()), std::exception);
}

// === Different Rank Operations ===

TEST_F(TensorExpressionTemplateTest, DifferentRankTensors) {
    Tensor<double, 1> t1d({4});
    Tensor<double, 2> t2d({2, 2});
    Tensor<double, 3> t3d({2, 1, 2});
    
    // Fill tensors
    for (size_t i = 0; i < 4; ++i) {
        t1d(i) = static_cast<double>(i + 1);
    }
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            t2d(i, j) = static_cast<double>(i * 2 + j + 1);
        }
    }
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 1; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                t3d(i, j, k) = static_cast<double>(i * 2 + j * 2 + k + 1);
            }
        }
    }
    
    // Test that tensors of different ranks exist and work independently
    EXPECT_EQ(t1d.rank(), 1);
    EXPECT_EQ(t2d.rank(), 2);
    EXPECT_EQ(t3d.rank(), 3);
}

// === Large Tensor Performance ===

TEST_F(TensorExpressionTemplateTest, LargeTensorPerformance) {
    const size_t N = 100;
    Tensor<double, 2> t1({N, N}), t2({N, N}), t3({N, N});
    
    // Fill with data
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            t1(i, j) = dist(gen);
            t2(i, j) = dist(gen);
            t3(i, j) = dist(gen);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Complex expression
    auto expr = (t1 + t2) * 0.5 - t3 / 2.0;
    Tensor<double, 2> result = expr;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify a few elements
    for (size_t i = 0; i < 10; i += 2) {
        for (size_t j = 0; j < 10; j += 2) {
            double expected = (t1(i, j) + t2(i, j)) * 0.5 - t3(i, j) / 2.0;
            EXPECT_NEAR(result(i, j), expected, tolerance);
        }
    }
    
    EXPECT_LT(duration.count(), 200); // Should complete in less than 200ms
}

// === Mixed Type Operations ===

TEST_F(TensorExpressionTemplateTest, MixedTypeOperations) {
    Tensor<float, 2> tf({2, 2});
    Tensor<double, 2> td({2, 2});
    
    tf(0, 0) = 1.0f; tf(0, 1) = 2.0f;
    tf(1, 0) = 3.0f; tf(1, 1) = 4.0f;
    
    td(0, 0) = 1.5; td(0, 1) = 2.5;
    td(1, 0) = 3.5; td(1, 1) = 4.5;
    
    // float + double -> double
    auto expr = tf + td;
    Tensor<double, 2> result = expr;
    
    EXPECT_NEAR(result(0, 0), 2.5, tolerance);
    EXPECT_NEAR(result(0, 1), 4.5, tolerance);
    EXPECT_NEAR(result(1, 0), 6.5, tolerance);
    EXPECT_NEAR(result(1, 1), 8.5, tolerance);
}

// === Chained Assignment ===

TEST_F(TensorExpressionTemplateTest, ChainedAssignment) {
    Tensor<double, 2> t1({2, 2}, 1.0);
    Tensor<double, 2> t2({2, 2}, 2.0);
    Tensor<double, 2> result1({2, 2}), result2({2, 2}), result3({2, 2});
    
    // Chained assignment from same expression
    auto expr = t1 * 3.0 + t2;
    
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

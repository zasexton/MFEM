/**
 * @file test_tensor.cpp
 * @brief Comprehensive unit tests for the Tensor class
 */

#include <gtest/gtest.h>
#include <sstream>
#include <complex>
#include <type_traits>
#include <core/tensor.h>

using namespace fem::numeric;

namespace {

// Test fixture for tensor tests
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// === Constructor Tests ===

TEST_F(TensorTest, DefaultConstructor) {
    Tensor<double, 3> t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
    
    auto shape = t.shape();
    EXPECT_EQ(shape[0], 0);
    EXPECT_EQ(shape[1], 0); 
    EXPECT_EQ(shape[2], 0);
}

TEST_F(TensorTest, ShapeConstructor) {
    std::array<size_t, 3> shape = {2, 3, 4};
    Tensor<double, 3> t(shape);
    
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.size(2), 4);
    EXPECT_EQ(t.shape(), shape);
}

TEST_F(TensorTest, ShapeAndValueConstructor) {
    std::array<size_t, 2> shape = {3, 3};
    Tensor<double, 2> t(shape, 2.5);
    
    EXPECT_EQ(t.size(), 9);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(t(i, j), 2.5);
        }
    }
}

TEST_F(TensorTest, ScalarConstructor) {
    Tensor<double, 0> scalar(42.0);
    EXPECT_EQ(scalar.size(), 1);
    EXPECT_EQ(scalar(), 42.0);
}

TEST_F(TensorTest, InitializerListConstructor1D) {
    Tensor<double, 1> t{1.0, 2.0, 3.0, 4.0};
    
    EXPECT_EQ(t.size(), 4);
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t(0), 1.0);
    EXPECT_EQ(t(1), 2.0);
    EXPECT_EQ(t(2), 3.0);
    EXPECT_EQ(t(3), 4.0);
}

TEST_F(TensorTest, CopyConstructor) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> original(shape, 3.14);
    Tensor<double, 2> copy(original);
    
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(copy(i, j), original(i, j));
        }
    }
}

TEST_F(TensorTest, MoveConstructor) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> original(shape, 3.14);
    auto original_shape = original.shape();
    
    Tensor<double, 2> moved(std::move(original));
    
    EXPECT_EQ(moved.shape(), original_shape);
    EXPECT_EQ(moved.size(), 4);
    EXPECT_EQ(moved(0, 0), 3.14);
}

// === Element Access Tests ===

TEST_F(TensorTest, ElementAccess) {
    std::array<size_t, 3> shape = {2, 2, 2};
    Tensor<double, 3> t(shape);
    
    // Test bounds checking
    EXPECT_NO_THROW(t.at(0, 0, 0));
    EXPECT_NO_THROW(t.at(1, 1, 1));
    EXPECT_THROW(t.at(2, 0, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 2, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 0, 2), std::out_of_range);
    
    // Test element setting and getting
    t(1, 0, 1) = 42.0;
    EXPECT_EQ(t(1, 0, 1), 42.0);
    EXPECT_EQ(t.at(1, 0, 1), 42.0);
}

TEST_F(TensorTest, LinearIndexing) {
    std::array<size_t, 2> shape = {2, 3};
    Tensor<double, 2> t(shape);
    
    // Fill with linear indexing
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i);
    }
    
    // Check values
    EXPECT_EQ(t[0], 0.0);
    EXPECT_EQ(t[1], 1.0);
    EXPECT_EQ(t[5], 5.0);
    
    // Check correspondence with multi-indexing
    EXPECT_EQ(t(0, 0), t[0]);
    EXPECT_EQ(t(0, 1), t[1]);
    EXPECT_EQ(t(1, 2), t[5]);
}

TEST_F(TensorTest, DataAccess) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape, 1.5);
    
    auto* data = t.data();
    EXPECT_NE(data, nullptr);
    
    // Modify through data pointer
    data[0] = 99.0;
    EXPECT_EQ(t(0, 0), 99.0);
    
    // Const data access
    const auto& const_t = t;
    const auto* const_data = const_t.data();
    EXPECT_EQ(const_data[0], 99.0);
}

TEST_F(TensorTest, ViewReflectsData) {
    Tensor<double, 3> t({2, 2, 2}, 0.0);
    auto v = t.view();
    v(0, 0, 0) = 42.0;
    EXPECT_EQ(t(0, 0, 0), 42.0);

    const Tensor<double, 3> ct = t;
    auto cv = ct.view();
    EXPECT_EQ(cv(0, 0, 0), 42.0);
}

TEST_F(TensorTest, MultiIndexSlicingView) {
    Tensor<double, 3> t({2, 3, 4});
    size_t counter = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t(i, j, k) = static_cast<double>(++counter);
            }
        }
    }

    auto sub = t(idx(1, Slice(0, 3, 2), all));
    Tensor<double, 2> sliced(sub);
    EXPECT_EQ(sliced.shape()[0], 2u);
    EXPECT_EQ(sliced.shape()[1], 4u);

    for (size_t j = 0; j < 2; ++j) {
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_EQ(sliced(j, k), t(1, j * 2, k));
        }
    }

    auto trailing = t(idx(Slice(0, 1)));
    Tensor<double, 3> trailing_tensor(trailing);
    EXPECT_EQ(trailing_tensor.shape()[0], 1u);
    EXPECT_EQ(trailing_tensor.shape()[1], 3u);
    EXPECT_EQ(trailing_tensor.shape()[2], 4u);
}

TEST_F(TensorTest, NewAxisInsertionView) {
    Tensor<double, 3> t({2, 3, 4});
    size_t counter = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t(i, j, k) = static_cast<double>(++counter);
            }
        }
    }

    auto expanded = t(idx(newaxis, all, all, all));
    Tensor<double, 4> expanded_tensor(expanded);
    EXPECT_EQ(expanded_tensor.shape()[0], 1u);
    EXPECT_EQ(expanded_tensor.shape()[1], 2u);
    EXPECT_EQ(expanded_tensor.shape()[2], 3u);
    EXPECT_EQ(expanded_tensor.shape()[3], 4u);

    auto inserted = t(idx(all, newaxis, Slice(1, 3), 0));
    Tensor<double, 3> inserted_tensor(inserted);
    EXPECT_EQ(inserted_tensor.shape()[0], 2u);
    EXPECT_EQ(inserted_tensor.shape()[1], 1u);
    EXPECT_EQ(inserted_tensor.shape()[2], 2u);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 2; ++k) {
            EXPECT_EQ(inserted_tensor(i, 0, k), t(i, k + 1, 0));
        }
    }
}

// === Tensor Operations Tests ===

TEST_F(TensorTest, Fill) {
    std::array<size_t, 2> shape = {3, 2};
    Tensor<double, 2> t(shape);
    
    t.fill(7.5);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(t(i, j), 7.5);
        }
    }
}

TEST_F(TensorTest, Zero) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape, 5.0);
    
    t.zero();
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(t(i, j), 0.0);
        }
    }
}

TEST_F(TensorTest, Reshape) {
    std::array<size_t, 2> shape = {2, 6};
    Tensor<double, 2> t(shape);
    
    // Fill with sequential values
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i);
    }
    
    // Reshape to 3x4
    std::array<size_t, 2> new_shape = {3, 4};
    t.reshape(new_shape);
    
    EXPECT_EQ(t.shape(), new_shape);
    EXPECT_EQ(t.size(), 12);
    
    // Values should be preserved in linear order
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(t[i], static_cast<double>(i));
    }
    
    // Test invalid reshape
    std::array<size_t, 2> invalid_shape = {2, 5};  // Different total size
    EXPECT_THROW(t.reshape(invalid_shape), std::invalid_argument);
}

TEST_F(TensorTest, PermuteAxes2D) {
    std::array<size_t, 2> shape = {2, 3};
    Tensor<double, 2> t(shape);
    
    // Fill matrix
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;
    
    // Transpose (swap axes)
    std::array<size_t, 2> axes = {1, 0};
    auto transposed = t.permute(axes);
    
    EXPECT_EQ(transposed.size(0), 3);
    EXPECT_EQ(transposed.size(1), 2);
    
    EXPECT_EQ(transposed(0, 0), 1);
    EXPECT_EQ(transposed(1, 0), 2);
    EXPECT_EQ(transposed(2, 0), 3);
    EXPECT_EQ(transposed(0, 1), 4);
    EXPECT_EQ(transposed(1, 1), 5);
    EXPECT_EQ(transposed(2, 1), 6);
}

TEST_F(TensorTest, Transpose2D) {
    std::array<size_t, 2> shape = {2, 3};
    Tensor<double, 2> t(shape);
    
    // Fill matrix
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;
    
    auto transposed = t.transpose();
    
    EXPECT_EQ(transposed.size(0), 3);
    EXPECT_EQ(transposed.size(1), 2);
    EXPECT_EQ(transposed(0, 0), 1);
    EXPECT_EQ(transposed(1, 0), 2);
    EXPECT_EQ(transposed(0, 1), 4);
    EXPECT_EQ(transposed(1, 1), 5);
}

TEST_F(TensorTest, PermuteAxes3DGeneral) {
    Tensor<int, 3> t({2, 3, 4});
    int value = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t(i, j, k) = value++;
            }
        }
    }

    auto permuted = t.permute({2, 0, 1});
    EXPECT_EQ(permuted.size(0), 4u);
    EXPECT_EQ(permuted.size(1), 2u);
    EXPECT_EQ(permuted.size(2), 3u);

    for (size_t a = 0; a < 4; ++a) {
        for (size_t b = 0; b < 2; ++b) {
            for (size_t c = 0; c < 3; ++c) {
                EXPECT_EQ(permuted(a, b, c), t(b, c, a));
            }
        }
    }
}

// === Arithmetic Operations Tests ===

TEST_F(TensorTest, Addition) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> a(shape);
    Tensor<double, 2> b(shape);
    
    a(0, 0) = 1; a(0, 1) = 2; a(1, 0) = 3; a(1, 1) = 4;
    b(0, 0) = 5; b(0, 1) = 6; b(1, 0) = 7; b(1, 1) = 8;
    
    a += b;
    
    EXPECT_EQ(a(0, 0), 6);
    EXPECT_EQ(a(0, 1), 8);
    EXPECT_EQ(a(1, 0), 10);
    EXPECT_EQ(a(1, 1), 12);
    
    // Test dimension mismatch
    std::array<size_t, 2> wrong_shape = {3, 2};
    Tensor<double, 2> c(wrong_shape);
    EXPECT_THROW(a += c, std::invalid_argument);
}

TEST_F(TensorTest, Subtraction) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> a(shape);
    Tensor<double, 2> b(shape);
    
    a(0, 0) = 10; a(0, 1) = 9; a(1, 0) = 8; a(1, 1) = 7;
    b(0, 0) = 1; b(0, 1) = 2; b(1, 0) = 3; b(1, 1) = 4;
    
    a -= b;
    
    EXPECT_EQ(a(0, 0), 9);
    EXPECT_EQ(a(0, 1), 7);
    EXPECT_EQ(a(1, 0), 5);
    EXPECT_EQ(a(1, 1), 3);
}

TEST_F(TensorTest, ScalarMultiplication) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape);
    
    t(0, 0) = 1; t(0, 1) = 2; t(1, 0) = 3; t(1, 1) = 4;
    
    t *= 2.5;
    
    EXPECT_EQ(t(0, 0), 2.5);
    EXPECT_EQ(t(0, 1), 5.0);
    EXPECT_EQ(t(1, 0), 7.5);
    EXPECT_EQ(t(1, 1), 10.0);
}

TEST_F(TensorTest, ScalarDivision) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape);
    
    t(0, 0) = 10; t(0, 1) = 20; t(1, 0) = 30; t(1, 1) = 40;
    
    t /= 10.0;
    
    EXPECT_EQ(t(0, 0), 1.0);
    EXPECT_EQ(t(0, 1), 2.0);
    EXPECT_EQ(t(1, 0), 3.0);
    EXPECT_EQ(t(1, 1), 4.0);
}

TEST_F(TensorTest, HadamardProduct) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> a(shape);
    Tensor<double, 2> b(shape);
    
    a(0, 0) = 2; a(0, 1) = 3; a(1, 0) = 4; a(1, 1) = 5;
    b(0, 0) = 1; b(0, 1) = 2; b(1, 0) = 3; b(1, 1) = 4;
    
    a.hadamard_product(b);
    
    EXPECT_EQ(a(0, 0), 2);
    EXPECT_EQ(a(0, 1), 6);
    EXPECT_EQ(a(1, 0), 12);
    EXPECT_EQ(a(1, 1), 20);
}

// === Norms Tests ===

TEST_F(TensorTest, FrobeniusNorm) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape);
    
    t(0, 0) = 3; t(0, 1) = 4; t(1, 0) = 0; t(1, 1) = 0;
    
    double norm = t.frobenius_norm();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(TensorTest, MaxNorm) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape);
    
    t(0, 0) = -7; t(0, 1) = 3; t(1, 0) = 5; t(1, 1) = -2;
    
    double norm = t.max_norm();
    EXPECT_EQ(norm, 7.0);
}

// === Utility Tests ===

TEST_F(TensorTest, Flatten) {
    std::array<size_t, 3> shape = {2, 2, 2};
    Tensor<double, 3> t(shape);
    
    // Fill with sequential values
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i);
    }
    
    auto flat = t.flatten();
    EXPECT_EQ(flat.size(0), 8);
    
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(flat(i), static_cast<double>(i));
    }
}

// === Non-member Operation Tests ===

TEST_F(TensorTest, NonMemberAddition) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> a(shape, 1.0);
    Tensor<int, 2> b(shape, 2);
    
    auto expr = a + b;

    // Ensure we returned an expression template
    using expr_type = std::decay_t<decltype(expr)>;
    static_assert(std::is_base_of_v<ExpressionBase<expr_type>, expr_type>);

    Tensor<double, 2> result(expr);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(result(i, j), 3.0);
        }
    }
}

TEST_F(TensorTest, NonMemberSubtraction) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> a(shape, 5.0);
    Tensor<float, 2> b(shape, 2.0f);
    
    auto expr = a - b;
    Tensor<double, 2> result(expr);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(result(i, j), 3.0, 1e-6);
        }
    }
}

TEST_F(TensorTest, ScalarMultiplicationNonMember) {
    std::array<size_t, 2> shape = {2, 2};
    Tensor<double, 2> t(shape, 3.0);
    
    auto expr1 = 2.0 * t;
    auto expr2 = t * 2.0;

    Tensor<double, 2> result1(expr1);
    Tensor<double, 2> result2(expr2);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(result1(i, j), 6.0);
            EXPECT_EQ(result2(i, j), 6.0);
        }
    }
}

// === Complex Number Tests ===

TEST_F(TensorTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    std::array<size_t, 2> shape = {2, 2};
    Tensor<Complex, 2> t(shape);
    
    t(0, 0) = Complex(1, 2);
    t(0, 1) = Complex(3, -1);
    t(1, 0) = Complex(-2, 1);
    t(1, 1) = Complex(0, 4);
    
    // Test complex operations
    t *= Complex(2, 0);
    
    EXPECT_EQ(t(0, 0), Complex(2, 4));
    EXPECT_EQ(t(0, 1), Complex(6, -2));
    EXPECT_EQ(t(1, 0), Complex(-4, 2));
    EXPECT_EQ(t(1, 1), Complex(0, 8));
}

// === Type Alias Tests ===

TEST_F(TensorTest, TypeAliases) {
    // Test scalar alias
    Scalar<double> s(42.0);
    EXPECT_EQ(s(), 42.0);
    
    // Test vector alias
    Vector1D<int> v({5});
    v(0) = 10;
    v(1) = 20;
    v(2) = 30;
    EXPECT_EQ(v(0), 10);
    EXPECT_EQ(v(2), 30);
    
    // Test matrix alias
    Matrix2D<float> m({2, 2});
    m(0, 0) = 1.5f;
    m(1, 1) = 2.5f;
    EXPECT_EQ(m(0, 0), 1.5f);
    EXPECT_EQ(m(1, 1), 2.5f);
}

// === Error Handling Tests ===

TEST_F(TensorTest, OutOfBoundsAccess) {
    std::array<size_t, 2> shape = {2, 3};
    Tensor<double, 2> t(shape);
    
    EXPECT_THROW(t.at(2, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 3), std::out_of_range);
    EXPECT_THROW(t.at(-1, 0), std::out_of_range);  // size_t wraps, becomes large
}

TEST_F(TensorTest, InvalidPermutationAxes) {
    std::array<size_t, 3> shape = {2, 3, 4};
    Tensor<double, 3> t(shape);
    
    // Invalid axes (out of range)
    std::array<size_t, 3> invalid_axes1 = {0, 1, 3};
    EXPECT_THROW(t.permute(invalid_axes1), std::invalid_argument);
    
    // Invalid axes (duplicate)
    std::array<size_t, 3> invalid_axes2 = {0, 1, 1};
    EXPECT_THROW(t.permute(invalid_axes2), std::invalid_argument);
}

// === Stream Output Tests ===

TEST_F(TensorTest, StreamOutput) {
    // Test scalar output
    Scalar<double> s(3.14);
    std::ostringstream oss1;
    oss1 << s;
    EXPECT_TRUE(oss1.str().find("3.14") != std::string::npos);
    
    // Test vector output
    Vector1D<int> v({3});
    v(0) = 1; v(1) = 2; v(2) = 3;
    std::ostringstream oss2;
    oss2 << v;
    std::string output = oss2.str();
    EXPECT_TRUE(output.find("1") != std::string::npos);
    EXPECT_TRUE(output.find("2") != std::string::npos);
    EXPECT_TRUE(output.find("3") != std::string::npos);
    
    // Test matrix output
    Matrix2D<double> m({2, 2});
    m(0, 0) = 1.1; m(0, 1) = 1.2;
    m(1, 0) = 2.1; m(1, 1) = 2.2;
    std::ostringstream oss3;
    oss3 << m;
    std::string matrix_output = oss3.str();
    EXPECT_TRUE(matrix_output.find("1.1") != std::string::npos);
    EXPECT_TRUE(matrix_output.find("2.2") != std::string::npos);
}

// === Performance Tests ===

TEST_F(TensorTest, LargeOperations) {
    std::array<size_t, 2> shape = {100, 100};
    Tensor<double, 2> a(shape, 1.0);
    Tensor<double, 2> b(shape, 2.0);
    
    // Large tensor operations should not crash
    EXPECT_NO_THROW(a += b);
    EXPECT_NO_THROW(a *= 0.5);
    EXPECT_NO_THROW(auto norm = a.frobenius_norm());
    
    // Verify a sample of results
    EXPECT_EQ(a(0, 0), 1.5);
    EXPECT_EQ(a(50, 50), 1.5);
    EXPECT_EQ(a(99, 99), 1.5);
}

// === Edge Cases ===

TEST_F(TensorTest, SingleElementTensor) {
    std::array<size_t, 3> shape = {1, 1, 1};
    Tensor<double, 3> t(shape, 42.0);
    
    EXPECT_EQ(t.size(), 1);
    EXPECT_EQ(t(0, 0, 0), 42.0);
    
    t *= 2.0;
    EXPECT_EQ(t(0, 0, 0), 84.0);
    
    auto norm = t.frobenius_norm();
    EXPECT_EQ(norm, 84.0);
}

TEST_F(TensorTest, ZeroSizeDimensions) {
    std::array<size_t, 2> shape = {0, 5};
    Tensor<double, 2> t(shape);
    
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.size(0), 0);
    EXPECT_EQ(t.size(1), 5);
}

} // namespace

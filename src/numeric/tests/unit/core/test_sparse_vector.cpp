/**
 * @file test_sparse_vector.cpp
 * @brief Comprehensive unit tests for the SparseVector class
 */

#include <gtest/gtest.h>
#include <sstream>
#include <complex>
#include <core/sparse_vector.h>

using namespace fem::numeric;

namespace {

// Test fixture for sparse vector tests
class SparseVectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// === Constructor Tests ===

TEST_F(SparseVectorTest, DefaultConstructor) {
    SparseVector<double> sv;
    
    EXPECT_EQ(sv.size(), 0);
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_TRUE(sv.empty());
    EXPECT_EQ(sv.sparsity(), 0.0);
    EXPECT_EQ(sv.format(), SparseVector<double>::StorageFormat::COO);
}

TEST_F(SparseVectorTest, SizeConstructor) {
    SparseVector<double> sv(10);
    
    EXPECT_EQ(sv.size(), 10);
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_FALSE(sv.empty());
    EXPECT_EQ(sv.sparsity(), 0.0);
}

TEST_F(SparseVectorTest, ConstructorWithFormat) {
    SparseVector<double> sv_coo(10, SparseVector<double>::StorageFormat::COO);
    SparseVector<double> sv_sorted(10, SparseVector<double>::StorageFormat::Sorted);
    SparseVector<double> sv_hash(10, SparseVector<double>::StorageFormat::HashMap);
    
    EXPECT_EQ(sv_coo.format(), SparseVector<double>::StorageFormat::COO);
    EXPECT_EQ(sv_sorted.format(), SparseVector<double>::StorageFormat::Sorted);
    EXPECT_EQ(sv_hash.format(), SparseVector<double>::StorageFormat::HashMap);
    
    // All should behave the same externally
    sv_coo.set(5, 1.0);
    sv_sorted.set(5, 1.0);
    sv_hash.set(5, 1.0);
    
    EXPECT_EQ(sv_coo[5], 1.0);
    EXPECT_EQ(sv_sorted[5], 1.0);
    EXPECT_EQ(sv_hash[5], 1.0);
}

TEST_F(SparseVectorTest, ConstructorFromDenseVector) {
    Vector<double> dense(5);
    dense[0] = 1.0;
    dense[1] = 0.0;  // Should be omitted
    dense[2] = 3.0;
    dense[3] = 0.0;  // Should be omitted
    dense[4] = 5.0;
    
    SparseVector<double> sv(dense);
    
    EXPECT_EQ(sv.size(), 5);
    EXPECT_EQ(sv.nnz(), 3);
    EXPECT_NEAR(sv.sparsity(), 0.6, 1e-10);
    
    EXPECT_EQ(sv[0], 1.0);
    EXPECT_EQ(sv[1], 0.0);
    EXPECT_EQ(sv[2], 3.0);
    EXPECT_EQ(sv[3], 0.0);
    EXPECT_EQ(sv[4], 5.0);
}

TEST_F(SparseVectorTest, ConstructorFromDenseVectorWithTolerance) {
    Vector<double> dense(3);
    dense[0] = 1.0;
    dense[1] = 1e-15;  // Should be considered zero with default tolerance
    dense[2] = 1e-10;  // Should be kept with custom tolerance
    
    // Default tolerance
    SparseVector<double> sv1(dense);
    EXPECT_EQ(sv1.nnz(), 2);  // 1.0 and 1e-10
    
    // Custom tolerance
    SparseVector<double> sv2(dense, SparseVector<double>::StorageFormat::COO, 1e-12);
    EXPECT_EQ(sv2.nnz(), 1);  // Only 1.0
}

TEST_F(SparseVectorTest, ConstructorFromPairs) {
    std::vector<std::pair<size_t, double>> pairs = {{0, 1.0}, {2, 3.0}, {4, 5.0}};
    
    SparseVector<double> sv(5, pairs);
    
    EXPECT_EQ(sv.size(), 5);
    EXPECT_EQ(sv.nnz(), 3);
    
    EXPECT_EQ(sv[0], 1.0);
    EXPECT_EQ(sv[1], 0.0);
    EXPECT_EQ(sv[2], 3.0);
    EXPECT_EQ(sv[3], 0.0);
    EXPECT_EQ(sv[4], 5.0);
}

TEST_F(SparseVectorTest, ConstructorFromPairsInvalidIndex) {
    std::vector<std::pair<size_t, double>> pairs = {{0, 1.0}, {5, 3.0}};  // Index 5 out of range for size 5
    
    EXPECT_THROW(SparseVector<double> sv(5, pairs), std::out_of_range);
}

TEST_F(SparseVectorTest, CopyConstructor) {
    SparseVector<double> original(5);
    original.set(1, 2.0);
    original.set(3, 4.0);
    
    SparseVector<double> copy(original);
    
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.nnz(), original.nnz());
    EXPECT_EQ(copy[1], 2.0);
    EXPECT_EQ(copy[3], 4.0);
    
    // Modify original to ensure deep copy
    original.set(1, 999.0);
    EXPECT_EQ(copy[1], 2.0);
}

TEST_F(SparseVectorTest, MoveConstructor) {
    SparseVector<double> original(3);
    original.set(0, 42.0);
    
    SparseVector<double> moved(std::move(original));
    
    EXPECT_EQ(moved.size(), 3);
    EXPECT_EQ(moved.nnz(), 1);
    EXPECT_EQ(moved[0], 42.0);
}

// === Element Access Tests ===

TEST_F(SparseVectorTest, ElementAccess) {
    SparseVector<double> sv(5);
    
    // Initially all zeros
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(sv[i], 0.0);
    }
    
    // Set some values
    sv.set(1, 10.0);
    sv.set(3, 30.0);
    
    EXPECT_EQ(sv[0], 0.0);
    EXPECT_EQ(sv[1], 10.0);
    EXPECT_EQ(sv[2], 0.0);
    EXPECT_EQ(sv[3], 30.0);
    EXPECT_EQ(sv[4], 0.0);
    
    EXPECT_EQ(sv.nnz(), 2);
}

TEST_F(SparseVectorTest, ElementAccessBoundsChecking) {
    SparseVector<double> sv(3);
    
    EXPECT_NO_THROW(sv.at(0));
    EXPECT_NO_THROW(sv.at(2));
    EXPECT_THROW(sv.at(3), std::out_of_range);
    
    // Out of bounds access with operator[] should return 0
    EXPECT_EQ(sv[100], 0.0);
}

TEST_F(SparseVectorTest, SetAndGet) {
    SparseVector<double> sv(5);
    
    sv.set(0, 1.0);
    sv.set(2, 3.0);
    sv.set(4, 5.0);
    
    EXPECT_EQ(sv.nnz(), 3);
    EXPECT_EQ(sv[0], 1.0);
    EXPECT_EQ(sv[2], 3.0);
    EXPECT_EQ(sv[4], 5.0);
    
    // Setting to zero should remove element
    sv.set(2, 0.0);
    EXPECT_EQ(sv.nnz(), 2);
    EXPECT_EQ(sv[2], 0.0);
}

TEST_F(SparseVectorTest, SetOutOfRange) {
    SparseVector<double> sv(3);
    
    EXPECT_THROW(sv.set(3, 1.0), std::out_of_range);
}

TEST_F(SparseVectorTest, AddOperation) {
    SparseVector<double> sv(5);
    
    sv.set(1, 2.0);
    sv.add(1, 3.0);  // Should become 5.0
    sv.add(3, 4.0);  // Should set to 4.0
    
    EXPECT_EQ(sv[1], 5.0);
    EXPECT_EQ(sv[3], 4.0);
    EXPECT_EQ(sv.nnz(), 2);
}

// === Vector Operations Tests ===

TEST_F(SparseVectorTest, Resize) {
    SparseVector<double> sv(5);
    sv.set(0, 1.0);
    sv.set(4, 5.0);
    
    // Resize smaller - should lose element at index 4
    sv.resize(3);
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv.nnz(), 1);
    EXPECT_EQ(sv[0], 1.0);
    EXPECT_EQ(sv[4], 0.0);  // Out of bounds now
    
    // Resize larger
    sv.resize(10);
    EXPECT_EQ(sv.size(), 10);
    EXPECT_EQ(sv.nnz(), 1);
    EXPECT_EQ(sv[0], 1.0);
}

TEST_F(SparseVectorTest, Clear) {
    SparseVector<double> sv(5);
    sv.set(1, 2.0);
    sv.set(3, 4.0);
    
    sv.clear();
    
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_EQ(sv[1], 0.0);
    EXPECT_EQ(sv[3], 0.0);
}

TEST_F(SparseVectorTest, Reserve) {
    SparseVector<double> sv(100);
    
    // Reserve should not change size or nnz
    sv.reserve(50);
    EXPECT_EQ(sv.size(), 100);
    EXPECT_EQ(sv.nnz(), 0);
}

TEST_F(SparseVectorTest, Compress) {
    SparseVector<double> sv(5);
    sv.set(1, 2.0);
    sv.set(3, 0.0);  // Should be removed by compress
    sv.set(4, 5.0);
    
    sv.compress();
    
    EXPECT_EQ(sv.nnz(), 2);  // Only non-zeros remaining
    EXPECT_EQ(sv[1], 2.0);
    EXPECT_EQ(sv[3], 0.0);
    EXPECT_EQ(sv[4], 5.0);
}

// === Arithmetic Operations Tests ===

TEST_F(SparseVectorTest, Addition) {
    SparseVector<double> a(5);
    SparseVector<double> b(5);
    
    a.set(0, 1.0);
    a.set(2, 3.0);
    
    b.set(1, 2.0);
    b.set(2, 4.0);
    
    a += b;
    
    EXPECT_EQ(a[0], 1.0);
    EXPECT_EQ(a[1], 2.0);
    EXPECT_EQ(a[2], 7.0);
    EXPECT_EQ(a[3], 0.0);
    EXPECT_EQ(a[4], 0.0);
    
    EXPECT_EQ(a.nnz(), 3);
}

TEST_F(SparseVectorTest, AdditionIncompatibleSize) {
    SparseVector<double> a(5);
    SparseVector<double> b(3);
    
    EXPECT_THROW(a += b, std::invalid_argument);
}

TEST_F(SparseVectorTest, Subtraction) {
    SparseVector<double> a(5);
    SparseVector<double> b(5);
    
    a.set(0, 5.0);
    a.set(2, 10.0);
    
    b.set(0, 2.0);
    b.set(1, 3.0);
    b.set(2, 4.0);
    
    a -= b;
    
    EXPECT_EQ(a[0], 3.0);
    EXPECT_EQ(a[1], -3.0);
    EXPECT_EQ(a[2], 6.0);
    
    EXPECT_EQ(a.nnz(), 3);
}

TEST_F(SparseVectorTest, ScalarMultiplication) {
    SparseVector<double> sv(3);
    sv.set(0, 2.0);
    sv.set(2, 4.0);
    
    sv *= 2.5;
    
    EXPECT_EQ(sv[0], 5.0);
    EXPECT_EQ(sv[1], 0.0);
    EXPECT_EQ(sv[2], 10.0);
    EXPECT_EQ(sv.nnz(), 2);
}

TEST_F(SparseVectorTest, ScalarMultiplicationByZero) {
    SparseVector<double> sv(3);
    sv.set(0, 2.0);
    sv.set(2, 4.0);
    
    sv *= 0.0;
    
    EXPECT_EQ(sv.nnz(), 0);  // All elements should be removed
    EXPECT_EQ(sv[0], 0.0);
    EXPECT_EQ(sv[2], 0.0);
}

TEST_F(SparseVectorTest, ScalarDivision) {
    SparseVector<double> sv(3);
    sv.set(0, 10.0);
    sv.set(2, 20.0);
    
    sv /= 2.0;
    
    EXPECT_EQ(sv[0], 5.0);
    EXPECT_EQ(sv[2], 10.0);
    EXPECT_EQ(sv.nnz(), 2);
}

TEST_F(SparseVectorTest, ScalarDivisionByZero) {
    SparseVector<double> sv(3);
    sv.set(0, 2.0);
    
    EXPECT_THROW(sv /= 0.0, std::invalid_argument);
}

// === Dot Product Tests ===

TEST_F(SparseVectorTest, DotProductSparse) {
    SparseVector<double> a(5);
    SparseVector<double> b(5);
    
    a.set(0, 2.0);
    a.set(2, 3.0);
    a.set(4, 5.0);
    
    b.set(0, 1.0);
    b.set(1, 4.0);  // No overlap with a
    b.set(2, 2.0);
    
    auto result = a.dot(b);
    
    // result = 2*1 + 0*4 + 3*2 + 0*0 + 5*0 = 2 + 6 = 8
    EXPECT_EQ(result, 8.0);
}

TEST_F(SparseVectorTest, DotProductSparseIncompatibleSize) {
    SparseVector<double> a(5);
    SparseVector<double> b(3);
    
    EXPECT_THROW(a.dot(b), std::invalid_argument);
}

TEST_F(SparseVectorTest, DotProductDense) {
    SparseVector<double> sv(3);
    Vector<double> dense(3);
    
    sv.set(0, 2.0);
    sv.set(2, 4.0);
    
    dense[0] = 1.0;
    dense[1] = 3.0;
    dense[2] = 2.0;
    
    auto result = sv.dot(dense);
    
    // result = 2*1 + 0*3 + 4*2 = 2 + 8 = 10
    EXPECT_EQ(result, 10.0);
}

TEST_F(SparseVectorTest, DotProductDenseIncompatibleSize) {
    SparseVector<double> sv(5);
    Vector<double> dense(3);
    
    EXPECT_THROW(sv.dot(dense), std::invalid_argument);
}

// === Norms Tests ===

TEST_F(SparseVectorTest, Norm2) {
    SparseVector<double> sv(5);
    sv.set(0, 3.0);
    sv.set(2, 4.0);
    
    auto norm = sv.norm2();
    EXPECT_NEAR(norm, 5.0, 1e-10);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(SparseVectorTest, Norm1) {
    SparseVector<double> sv(4);
    sv.set(0, -2.0);
    sv.set(1, 3.0);
    sv.set(3, -1.0);
    
    auto norm = sv.norm1();
    EXPECT_EQ(norm, 6.0);  // |−2| + |3| + |−1| = 6
}

TEST_F(SparseVectorTest, NormInf) {
    SparseVector<double> sv(4);
    sv.set(0, -7.0);
    sv.set(1, 3.0);
    sv.set(3, 5.0);
    
    auto norm = sv.norm_inf();
    EXPECT_EQ(norm, 7.0);  // max(|−7|, |3|, |5|) = 7
}

// === Conversion Tests ===

TEST_F(SparseVectorTest, ToDense) {
    SparseVector<double> sv(5);
    sv.set(1, 2.0);
    sv.set(3, 4.0);
    
    auto dense = sv.to_dense();
    
    EXPECT_EQ(dense.size(), 5);
    EXPECT_EQ(dense[0], 0.0);
    EXPECT_EQ(dense[1], 2.0);
    EXPECT_EQ(dense[2], 0.0);
    EXPECT_EQ(dense[3], 4.0);
    EXPECT_EQ(dense[4], 0.0);
}

TEST_F(SparseVectorTest, ChangeFormat) {
    SparseVector<double> sv(5, SparseVector<double>::StorageFormat::COO);
    sv.set(1, 2.0);
    sv.set(3, 4.0);
    
    EXPECT_EQ(sv.format(), SparseVector<double>::StorageFormat::COO);
    
    // Change to HashMap
    sv.change_format(SparseVector<double>::StorageFormat::HashMap);
    EXPECT_EQ(sv.format(), SparseVector<double>::StorageFormat::HashMap);
    
    // Values should be preserved
    EXPECT_EQ(sv[1], 2.0);
    EXPECT_EQ(sv[3], 4.0);
    EXPECT_EQ(sv.nnz(), 2);
    
    // Change to Sorted
    sv.change_format(SparseVector<double>::StorageFormat::Sorted);
    EXPECT_EQ(sv.format(), SparseVector<double>::StorageFormat::Sorted);
    
    // Values should still be preserved
    EXPECT_EQ(sv[1], 2.0);
    EXPECT_EQ(sv[3], 4.0);
    EXPECT_EQ(sv.nnz(), 2);
    
    // Change back to COO
    sv.change_format(SparseVector<double>::StorageFormat::COO);
    EXPECT_EQ(sv.format(), SparseVector<double>::StorageFormat::COO);
    
    EXPECT_EQ(sv[1], 2.0);
    EXPECT_EQ(sv[3], 4.0);
    EXPECT_EQ(sv.nnz(), 2);
}

// === Iterator Tests ===

TEST_F(SparseVectorTest, Iterator) {
    SparseVector<double> sv(5);
    sv.set(1, 2.0);
    sv.set(3, 4.0);
    
    std::vector<std::pair<size_t, double>> expected = {{1, 2.0}, {3, 4.0}};
    std::vector<std::pair<size_t, double>> actual;
    
    for (auto it = sv.begin(); it != sv.end(); ++it) {
        auto [index, value] = *it;
        actual.emplace_back(index, value);
    }
    
    // Note: Order may vary depending on internal storage
    EXPECT_EQ(actual.size(), 2);
    
    // Check that both expected pairs are present
    for (const auto& expected_pair : expected) {
        bool found = false;
        for (const auto& actual_pair : actual) {
            if (actual_pair.first == expected_pair.first && 
                actual_pair.second == expected_pair.second) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(SparseVectorTest, RangeBasedFor) {
    SparseVector<double> sv(4);
    sv.set(0, 1.0);
    sv.set(2, 3.0);
    
    size_t count = 0;
    double sum = 0.0;
    
    for (const auto& entry : sv) {
        auto [index, value] = entry;
        count++;
        sum += value;
    }
    
    EXPECT_EQ(count, 2);
    EXPECT_EQ(sum, 4.0);
}

// === Non-member Operations Tests ===

TEST_F(SparseVectorTest, NonMemberAddition) {
    SparseVector<double> a(3);
    SparseVector<int> b(3);
    
    a.set(0, 1.5);
    a.set(2, 2.5);
    
    b.set(0, 1);
    b.set(1, 2);
    
    auto result = a + b;
    
    // Check type promotion
    static_assert(std::is_same_v<decltype(result), SparseVector<double>>);
    
    EXPECT_NEAR(result[0], 2.5, 1e-10);
    EXPECT_NEAR(result[1], 2.0, 1e-10);
    EXPECT_NEAR(result[2], 2.5, 1e-10);
}

TEST_F(SparseVectorTest, NonMemberSubtraction) {
    SparseVector<double> a(3);
    SparseVector<float> b(3);
    
    a.set(0, 5.0);
    a.set(1, 10.0);
    
    b.set(0, 1.0f);
    b.set(1, 3.0f);
    
    auto result = a - b;
    
    EXPECT_NEAR(result[0], 4.0, 1e-6);
    EXPECT_NEAR(result[1], 7.0, 1e-6);
    EXPECT_EQ(result[2], 0.0);
}

TEST_F(SparseVectorTest, ScalarMultiplicationNonMember) {
    SparseVector<double> sv(3);
    sv.set(0, 3.0);
    sv.set(2, 4.0);
    
    auto result1 = 2.0 * sv;
    auto result2 = sv * 2.0;
    
    EXPECT_EQ(result1[0], 6.0);
    EXPECT_EQ(result1[2], 8.0);
    EXPECT_EQ(result2[0], 6.0);
    EXPECT_EQ(result2[2], 8.0);
}

// === Complex Numbers Tests ===

TEST_F(SparseVectorTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    SparseVector<Complex> sv(3);
    
    sv.set(0, Complex(1, 2));
    sv.set(2, Complex(3, -1));
    
    EXPECT_EQ(sv[0], Complex(1, 2));
    EXPECT_EQ(sv[1], Complex(0, 0));
    EXPECT_EQ(sv[2], Complex(3, -1));
    
    sv *= Complex(2, 0);
    
    EXPECT_EQ(sv[0], Complex(2, 4));
    EXPECT_EQ(sv[2], Complex(6, -2));
    
    // Test norms with complex numbers
    auto norm2 = sv.norm2();
    // |2+4i|^2 + |6-2i|^2 = (4+16) + (36+4) = 20 + 40 = 60
    EXPECT_NEAR(norm2, std::sqrt(60.0), 1e-10);
}

// === Storage Format Comparison Tests ===

TEST_F(SparseVectorTest, StorageFormatConsistency) {
    // Test that all storage formats behave consistently
    std::vector<std::pair<size_t, double>> test_data = {{0, 1.0}, {3, 4.0}, {7, 7.0}};
    
    SparseVector<double> coo_sv(10, test_data, SparseVector<double>::StorageFormat::COO);
    SparseVector<double> sorted_sv(10, test_data, SparseVector<double>::StorageFormat::Sorted);
    SparseVector<double> hash_sv(10, test_data, SparseVector<double>::StorageFormat::HashMap);
    
    // All should have same size and nnz
    EXPECT_EQ(coo_sv.size(), 10);
    EXPECT_EQ(sorted_sv.size(), 10);
    EXPECT_EQ(hash_sv.size(), 10);
    
    EXPECT_EQ(coo_sv.nnz(), 3);
    EXPECT_EQ(sorted_sv.nnz(), 3);
    EXPECT_EQ(hash_sv.nnz(), 3);
    
    // All should have same values
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(coo_sv[i], sorted_sv[i]);
        EXPECT_EQ(coo_sv[i], hash_sv[i]);
    }
    
    // All should behave the same for operations
    coo_sv.set(5, 5.0);
    sorted_sv.set(5, 5.0);
    hash_sv.set(5, 5.0);
    
    EXPECT_EQ(coo_sv.nnz(), 4);
    EXPECT_EQ(sorted_sv.nnz(), 4);
    EXPECT_EQ(hash_sv.nnz(), 4);
    
    EXPECT_EQ(coo_sv[5], 5.0);
    EXPECT_EQ(sorted_sv[5], 5.0);
    EXPECT_EQ(hash_sv[5], 5.0);
}

// === Stream Output Tests ===

TEST_F(SparseVectorTest, StreamOutput) {
    SparseVector<double> sv(5);
    sv.set(1, 1.5);
    sv.set(3, 3.5);
    
    std::ostringstream oss;
    oss << sv;
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("SparseVector") != std::string::npos);
    EXPECT_TRUE(output.find("size=5") != std::string::npos);
    EXPECT_TRUE(output.find("nnz=2") != std::string::npos);
    EXPECT_TRUE(output.find("1.5") != std::string::npos);
    EXPECT_TRUE(output.find("3.5") != std::string::npos);
}

// === Error Handling Tests ===

TEST_F(SparseVectorTest, ErrorCases) {
    SparseVector<double> sv(3);
    
    // Out of range access
    EXPECT_THROW(sv.at(3), std::out_of_range);
    EXPECT_THROW(sv.set(3, 1.0), std::out_of_range);
    EXPECT_THROW(sv.add(3, 1.0), std::out_of_range);
    
    // Division by zero
    sv.set(0, 1.0);
    EXPECT_THROW(sv /= 0.0, std::invalid_argument);
}

// === Performance Tests ===

TEST_F(SparseVectorTest, LargeSparseVector) {
    const size_t size = 100000;
    const size_t nnz_count = 100;  // Very sparse
    
    SparseVector<double> sv(size);
    
    // Set sparse elements
    for (size_t i = 0; i < nnz_count; ++i) {
        size_t index = i * (size / nnz_count);
        sv.set(index, static_cast<double>(i + 1));
    }
    
    EXPECT_EQ(sv.size(), size);
    EXPECT_EQ(sv.nnz(), nnz_count);
    EXPECT_NEAR(sv.sparsity(), static_cast<double>(nnz_count) / size, 1e-10);
    
    // Operations should still work efficiently
    sv *= 2.0;
    
    EXPECT_EQ(sv[0], 2.0);
    EXPECT_EQ(sv[size / nnz_count], 4.0);
    EXPECT_EQ(sv.nnz(), nnz_count);
}

// === Real-world Application Tests ===

TEST_F(SparseVectorTest, FiniteElementVector) {
    // Simulate a finite element load vector where only boundary nodes have loads
    const size_t num_nodes = 1000;
    const size_t num_dofs = 3 * num_nodes;  // 3 DOFs per node
    
    SparseVector<double> load_vector(num_dofs);
    
    // Apply loads to boundary nodes (first and last 10 nodes)
    for (size_t node = 0; node < 10; ++node) {
        size_t dof_x = 3 * node + 0;
        size_t dof_y = 3 * node + 1;
        size_t dof_z = 3 * node + 2;
        
        load_vector.set(dof_x, 100.0);  // X force
        load_vector.set(dof_y, 200.0);  // Y force
        // Z force remains zero
    }
    
    for (size_t node = num_nodes - 10; node < num_nodes; ++node) {
        size_t dof_x = 3 * node + 0;
        size_t dof_y = 3 * node + 1;
        
        load_vector.set(dof_x, -50.0);   // Reaction force
        load_vector.set(dof_y, -100.0);  // Reaction force
    }
    
    EXPECT_EQ(load_vector.size(), num_dofs);
    EXPECT_EQ(load_vector.nnz(), 40);  // 20 nodes * 2 DOFs each
    EXPECT_LT(load_vector.sparsity(), 0.02);  // Very sparse
    
    // Check some values
    EXPECT_EQ(load_vector[0], 100.0);
    EXPECT_EQ(load_vector[1], 200.0);
    EXPECT_EQ(load_vector[2], 0.0);
    EXPECT_EQ(load_vector[num_dofs - 3], -50.0);
    EXPECT_EQ(load_vector[num_dofs - 2], -100.0);
}

// === Edge Cases ===

TEST_F(SparseVectorTest, SingleElementVector) {
    SparseVector<double> sv(1);
    
    EXPECT_EQ(sv.size(), 1);
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_EQ(sv[0], 0.0);
    
    sv.set(0, 42.0);
    
    EXPECT_EQ(sv.nnz(), 1);
    EXPECT_EQ(sv[0], 42.0);
    EXPECT_EQ(sv.sparsity(), 1.0);
    
    sv.set(0, 0.0);  // Remove element
    
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_EQ(sv[0], 0.0);
}

TEST_F(SparseVectorTest, EmptyVector) {
    SparseVector<double> sv(0);
    
    EXPECT_TRUE(sv.empty());
    EXPECT_EQ(sv.size(), 0);
    EXPECT_EQ(sv.nnz(), 0);
    EXPECT_EQ(sv.sparsity(), 0.0);
}

TEST_F(SparseVectorTest, DenseVector) {
    // Test sparse vector that becomes dense
    SparseVector<double> sv(5);
    
    // Fill all elements
    for (size_t i = 0; i < 5; ++i) {
        sv.set(i, static_cast<double>(i + 1));
    }
    
    EXPECT_EQ(sv.nnz(), 5);
    EXPECT_EQ(sv.sparsity(), 1.0);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(sv[i], static_cast<double>(i + 1));
    }
}

TEST_F(SparseVectorTest, VectorWithDuplicateInsertions) {
    // Test behavior when same index is set multiple times
    SparseVector<double> sv(5);
    
    sv.set(2, 1.0);
    sv.set(2, 2.0);  // Should overwrite
    sv.set(2, 3.0);  // Should overwrite again
    
    EXPECT_EQ(sv.nnz(), 1);
    EXPECT_EQ(sv[2], 3.0);
}

} // namespace

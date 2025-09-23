#include <gtest/gtest.h>

#include <decompositions/lu.h>
#include <core/matrix.h>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

#if defined(FEM_NUMERIC_ENABLE_LAPACK) || defined(FEM_NUMERIC_ENABLE_LAPACKE)

namespace {

template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j) {
            T s{};
            for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

} // namespace

TEST(Decompositions_LAPACK, LU_ColumnMajor_Reconstruct)
{
    using T = double;
    Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor> A(3, 3, T{});
    // General matrix
    T a[3][3] = {{2, 1, 1}, {4, -6, 0}, {-2, 7, 2}};
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) A(i, j) = a[i][j];

    std::vector<int> piv;
    int info = lu_factor(A, piv);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(piv.size(), 3u);

    // Build P*A0 by applying recorded row swaps to a copy of A0
    Matrix<T> PA(3, 3, T{});
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) PA(i, j) = a[i][j];
    for (size_t k = 0; k < piv.size(); ++k) {
        size_t p = static_cast<size_t>(piv[k]);
        if (p != k) {
            for (size_t j = 0; j < 3; ++j) std::swap(PA(k, j), PA(p, j));
        }
    }

    // Extract L and U from packed A
    Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor> L(3, 3, T{}), U(3, 3, T{});
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) {
        if (i > j) L(i, j) = A(i, j);
        else if (i == j) { L(i, j) = T{1}; U(i, j) = A(i, j); }
        else U(i, j) = A(i, j);
    }
    auto LU = matmul(L, U);
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j)
        EXPECT_NEAR(LU(i, j), PA(i, j), 1e-10);
}

#endif // FEM_NUMERIC_ENABLE_LAPACK || FEM_NUMERIC_ENABLE_LAPACKE


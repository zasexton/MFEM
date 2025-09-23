#include <gtest/gtest.h>

#include <decompositions/cholesky.h>
#include <core/matrix.h>
#include <core/vector.h>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Uplo;

#if defined(FEM_NUMERIC_ENABLE_LAPACK) || defined(FEM_NUMERIC_ENABLE_LAPACKE)

namespace {

template <typename T>
Matrix<T> conj_transpose(const Matrix<T>& A)
{
    Matrix<T> AH(A.cols(), A.rows(), T{});
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            if constexpr (fem::numeric::is_complex_number_v<T>) AH(j, i) = std::conj(A(i, j));
            else AH(j, i) = A(i, j);
    return AH;
}

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

TEST(Decompositions_LAPACK, Cholesky_ColumnMajor_Reconstruct)
{
    using T = double;
    Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor> A(3, 3, T{});
    // SPD matrix
    T a[3][3] = {{25, 15, -5}, {15, 18, 0}, {-5, 0, 11}};
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) A(i, j) = a[i][j];

    int info = cholesky_factor(A, Uplo::Lower);
    ASSERT_EQ(info, 0);

    Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor> L(3, 3, T{});
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j <= i; ++j) L(i, j) = A(i, j);

    auto At = conj_transpose(L);
    auto LLt = matmul(L, At);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(LLt(i, j), a[i][j], 1e-10);
}

#endif // FEM_NUMERIC_ENABLE_LAPACK || FEM_NUMERIC_ENABLE_LAPACKE


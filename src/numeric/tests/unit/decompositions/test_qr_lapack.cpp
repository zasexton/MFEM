#include <gtest/gtest.h>

#include <decompositions/qr.h>
#include <core/matrix.h>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Side;
using fem::numeric::linear_algebra::Trans;

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

TEST(Decompositions_LAPACK, QR_ColumnMajor_Reconstruct)
{
    using T = double;
    const size_t m = 4, n = 3;
    Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor> A(m, n, T{});
    // Fill with deterministic values
    T v = 0.5;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) { A(i, j) = std::sin(v); v += 0.37; }

    auto A0 = A; // keep a copy

    std::vector<T> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);

    // Form R and reconstruct Q(:,1:r)*R
    Matrix<T> R = form_R(A);
    const size_t r = std::min(m, n);
    Matrix<T> B(m, n, T{});
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j)
            B(i, j) = R(i, j);

    apply_Q_inplace(Side::Left, Trans::NoTrans, A, tau, B);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_NEAR(B(i, j), A0(i, j), 1e-9);
}

#endif // FEM_NUMERIC_ENABLE_LAPACK || FEM_NUMERIC_ENABLE_LAPACKE


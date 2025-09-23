#include <gtest/gtest.h>

#include <decompositions/qr.h>
#include <core/matrix.h>
#include <vector>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Side;
using fem::numeric::linear_algebra::Trans;

#if defined(FEM_NUMERIC_QR_TEST_WY)

namespace {

template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i=0;i<A.rows();++i)
        for (size_t j=0;j<B.cols();++j) {
            T s{}; for (size_t k=0;k<A.cols();++k) s += A(i,k) * B(k,j);
            C(i,j) = s;
        }
    return C;
}

} // namespace

TEST(Decompositions_WY_Dev, RowMajor_Wide_Equals_Unblocked)
{
    // Random-ish wide matrices; compare blocked WY vs unblocked QR
    for (auto [m,n] : { std::pair<size_t,size_t>{3,5}, std::pair<size_t,size_t>{4,7} }) {
        Matrix<double> A0(m,n, 0.0);
        double v=0.3; for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A0(i,j) = std::sin(v) + 0.2*std::cos(2*v); v += 0.37; }

        // Unblocked reference
        Matrix<double> Au = A0; std::vector<double> tau_u; ASSERT_EQ(qr_factor_unblocked(Au, tau_u), 0);
        Matrix<double> Qu(m,m, 0.0); for (size_t i=0;i<m;++i) Qu(i,i) = 1.0;
        apply_Q_inplace(Side::Left, Trans::NoTrans, Au, tau_u, Qu);
        Matrix<double> Ru = form_R(Au);
        Matrix<double> QRu = matmul(Qu, Ru);

        // Blocked WY under row-major (dev path): call blocked directly
        Matrix<double> Ab = A0; std::vector<double> tau_b; ASSERT_EQ(qr_factor_blocked(Ab, tau_b, 32), 0);
        Matrix<double> Qb(m,m, 0.0); for (size_t i=0;i<m;++i) Qb(i,i) = 1.0;
        apply_Q_inplace(Side::Left, Trans::NoTrans, Ab, tau_b, Qb);
        Matrix<double> Rb = form_R(Ab);
        Matrix<double> QRb = matmul(Qb, Rb);

        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(QRb(i,j), QRu(i,j), 1e-10);
    }
}

#endif // FEM_NUMERIC_QR_TEST_WY


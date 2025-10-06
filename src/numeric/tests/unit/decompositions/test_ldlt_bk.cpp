#include <gtest/gtest.h>

#include <decompositions/ldlt.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Uplo;

namespace {

template <typename T>
Vector<T> matvec(const Matrix<T>& A, const Vector<T>& x)
{
    Vector<T> y(A.rows(), T{});
    for (std::size_t i = 0; i < A.rows(); ++i) {
        T s{};
        for (std::size_t j = 0; j < A.cols(); ++j) s += A(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

} // namespace

TEST(Decompositions_LDLT_BK, Real_Indefinite_2x2)
{
    // Indefinite symmetric with zero diagonal -> expect 2x2 pivot
    Matrix<double> A0 = {{0.0, 1.0},
                         {1.0, 0.0}};
    Matrix<double> Af = A0;
    std::vector<int> ipiv;
    ASSERT_EQ(ldlt_factor(Af, ipiv, Uplo::Lower), 0);
    ASSERT_EQ(ipiv.size(), 2u);
    // At least one 2x2 negative entry
    EXPECT_LT(std::min(ipiv[0], ipiv[1]), 0);

    Vector<double> x_ref = {3.0, -2.0};
    Vector<double> b = matvec(A0, x_ref);
    Vector<double> x = b;
    ldlt_solve_inplace(Af, ipiv, x, Uplo::Lower);
    ASSERT_EQ(x.size(), x_ref.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], x_ref[i], 1e-12);
}

TEST(Decompositions_LDLT_BK, Real_Indefinite_3x3)
{
    Matrix<double> A0 = {{ 0.0,  2.0,  3.0},
                         { 2.0, -1.0,  1.0},
                         { 3.0,  1.0,  0.5}}; // indefinite
    Matrix<double> Af = A0;
    std::vector<int> ipiv;
    ASSERT_EQ(ldlt_factor(Af, ipiv, Uplo::Lower), 0);
    Vector<double> x_ref = {1.0, -2.0, 3.0};
    Vector<double> b = matvec(A0, x_ref);
    Vector<double> x = b;
    ldlt_solve_inplace(Af, ipiv, x, Uplo::Lower);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], x_ref[i], 1e-10);
}

TEST(Decompositions_LDLT_BK, Complex_Hermitian_2x2_Indefinite)
{
    using Z = std::complex<double>;
    Matrix<Z> A0(2, 2, Z{});
    A0(0,0) = Z(0.0, 0.0);
    A0(0,1) = Z(1.0, 1.0);
    A0(1,0) = std::conj(A0(0,1));
    A0(1,1) = Z(0.5, 0.0);

    Matrix<Z> Af = A0;
    std::vector<int> ipiv;
    ASSERT_EQ(ldlt_factor(Af, ipiv, Uplo::Lower), 0);
    ASSERT_EQ(ipiv.size(), 2u);
    EXPECT_LT(std::min(ipiv[0], ipiv[1]), 0);
    // Check that D diag entries are real (imag ~ 0)
    EXPECT_NEAR(std::abs(std::imag(Af(0,0))), 0.0, 1e-14);
    EXPECT_NEAR(std::abs(std::imag(Af(1,1))), 0.0, 1e-14);

    Vector<Z> x_ref = {Z(1.0,-1.0), Z(2.0,0.5)};
    Vector<Z> b = matvec(A0, x_ref);
    Vector<Z> x = b;
    ldlt_solve_inplace(Af, ipiv, x, Uplo::Lower);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(std::abs(x[i] - x_ref[i]), 0.0, 1e-10);
}

TEST(Decompositions_LDLT_BK, MultiRHS_InPlace)
{
    Matrix<double> A0 = {{ 0.0,  2.0,  3.0},
                         { 2.0, -1.0,  1.0},
                         { 3.0,  1.0,  0.5}};
    Matrix<double> Af = A0;
    std::vector<int> ipiv;
    ASSERT_EQ(ldlt_factor(Af, ipiv, Uplo::Lower), 0);

    Vector<double> e1 = {1.0, 0.0, 0.0};
    Vector<double> e2 = {0.0, 1.0, 0.0};
    Vector<double> b1 = matvec(A0, e1);
    Vector<double> b2 = matvec(A0, e2);
    Matrix<double> B(3, 2, 0.0);
    for (std::size_t i = 0; i < 3; ++i) { B(i,0) = b1[i]; B(i,1) = b2[i]; }
    ldlt_solve_inplace(Af, ipiv, B, Uplo::Lower);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(B(i,0), e1[i], 1e-10);
        EXPECT_NEAR(B(i,1), e2[i], 1e-10);
    }
}


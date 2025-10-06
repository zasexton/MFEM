#include <gtest/gtest.h>

#include <decompositions/ldlt.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <random>
#include <vector>

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

template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < B.cols(); ++j) {
            T s{};
            for (std::size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

} // namespace

TEST(Decompositions, LDLT_Solve_SPD_Real_VectorAndMatrix)
{
    // Symmetric positive definite 3x3
    Matrix<double> A0 = {{4.0, 1.0, 2.0},
                         {1.0, 3.0, 0.0},
                         {2.0, 0.0, 5.0}};
    Matrix<double> Af = A0;
    std::vector<int> piv;
    int info = ldlt_factor(Af, piv, Uplo::Lower);
    EXPECT_EQ(info, 0);

    // Known solution x_ref; build b = A0 * x_ref
    Vector<double> x_ref = {1.0, -2.0, 3.0};
    Vector<double> b = matvec(A0, x_ref);

    // Solve single RHS
    Vector<double> x = b;
    ldlt_solve_inplace(Af, piv, x, Uplo::Lower);
    ASSERT_EQ(x.size(), x_ref.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], x_ref[i], 1e-12);

    // Solve multiple RHS (columns)
    Matrix<double> B(A0.rows(), 2, 0.0);
    // RHS1 = A0 * e1, RHS2 = A0 * e2
    Vector<double> e1 = {1.0, 0.0, 0.0};
    Vector<double> e2 = {0.0, 1.0, 0.0};
    Vector<double> b1 = matvec(A0, e1);
    Vector<double> b2 = matvec(A0, e2);
    for (std::size_t i = 0; i < 3; ++i) { B(i, 0) = b1[i]; B(i, 1) = b2[i]; }

    Matrix<double> X = B;
    ldlt_solve_inplace(Af, piv, X, Uplo::Lower);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(X(i, 0), e1[i], 1e-12);
        EXPECT_NEAR(X(i, 1), e2[i], 1e-12);
    }
}

TEST(Decompositions, LDLT_Singular_Detection)
{
    // Rank-1 symmetric matrix -> should report failure (info > 0)
    Matrix<double> A = {{1.0, 1.0},
                        {1.0, 1.0}};
    std::vector<int> piv;
    int info = ldlt_factor(A, piv, Uplo::Lower);
    EXPECT_NE(info, 0);
}

TEST(Decompositions, LDLT_Solve_Indefinite_Real)
{
    // Symmetric indefinite but nonsingular
    Matrix<double> A0 = {{ 2.0,  3.0},
                         { 3.0, -1.0}};
    Matrix<double> Af = A0;
    std::vector<int> piv;
    int info = ldlt_factor(Af, piv, Uplo::Lower);
    EXPECT_EQ(info, 0);

    Vector<double> x_ref = {0.5, -1.5};
    Vector<double> b = matvec(A0, x_ref);
    Vector<double> x = b;
    ldlt_solve_inplace(Af, piv, x, Uplo::Lower);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], x_ref[i], 1e-12);
}

TEST(Decompositions, LDLT_Solve_Complex_Hermitian)
{
    using Z = std::complex<double>;
    Matrix<Z> H0(2, 2, Z{});
    H0(0,0) = Z(4.0, 0.0);
    H0(0,1) = Z(1.0, 1.0);
    H0(1,0) = std::conj(H0(0,1));
    H0(1,1) = Z(3.0, 0.0);

    Matrix<Z> Hf = H0;
    std::vector<int> piv;
    int info = ldlt_factor(Hf, piv, Uplo::Lower);
    EXPECT_EQ(info, 0);

    Vector<Z> x_ref = {Z(1.0, -1.0), Z(2.0, 0.5)};
    Vector<Z> b = matvec(H0, x_ref);
    Vector<Z> x = b;
    ldlt_solve_inplace(Hf, piv, x, Uplo::Lower);
    ASSERT_EQ(x.size(), x_ref.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(std::abs(x[i] - x_ref[i]), 0.0, 1e-10);
}

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
TEST(Decompositions, LDLT_Solve_Upper_WithBackend)
{
    // Exercise Upper path when LAPACK backend is enabled
    Matrix<double> A0 = {{5.0, 2.0, 1.0},
                         {2.0, 4.0, 0.5},
                         {1.0, 0.5, 3.0}};
    Matrix<double> Af = A0;
    std::vector<int> piv;
    int info = ldlt_factor(Af, piv, Uplo::Upper);
    EXPECT_EQ(info, 0);

    Vector<double> x_ref = {1.0, -2.0, 3.0};
    Vector<double> b = matvec(A0, x_ref);
    Vector<double> x = b;
    ldlt_solve_inplace(Af, piv, x, Uplo::Upper);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], x_ref[i], 1e-12);
}
#endif

TEST(Decompositions, LDLT_DimensionMismatch_Throws)
{
    Matrix<double> A0 = {{2.0, 0.5},
                         {0.5, 1.5}};
    Matrix<double> Af = A0;
    std::vector<int> piv;
    ASSERT_EQ(ldlt_factor(Af, piv, Uplo::Lower), 0);

    // Vector size mismatch
    Vector<double> b_bad = {1.0, 2.0, 3.0};
    EXPECT_THROW({ ldlt_solve_inplace(Af, piv, b_bad, Uplo::Lower); }, std::invalid_argument);

    // Matrix rows mismatch
    Matrix<double> B_bad(3, 1, 0.0);
    EXPECT_THROW({ ldlt_solve_inplace(Af, piv, B_bad, Uplo::Lower); }, std::invalid_argument);
}

TEST(Decompositions, LDLT_ZeroDimension)
{
    Matrix<double> A; // 0x0
    std::vector<int> piv;
    int info = ldlt_factor(A, piv, Uplo::Lower);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(piv.size(), 0u);
}

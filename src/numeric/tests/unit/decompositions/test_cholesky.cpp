#include <gtest/gtest.h>

#include <decompositions/cholesky.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Uplo;

namespace {

template <typename T>
Matrix<T> matmulH(const Matrix<T>& A, const Matrix<T>& B)
{
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i=0;i<A.rows();++i)
        for (size_t j=0;j<B.cols();++j) {
            T s{};
            for (size_t k=0;k<A.cols();++k) s += A(i,k) * B(k,j);
            C(i,j) = s;
        }
    return C;
}

template <typename T>
Matrix<T> conj_transpose(const Matrix<T>& A)
{
    Matrix<T> AH(A.cols(), A.rows(), T{});
    for (size_t i=0;i<A.rows();++i)
        for (size_t j=0;j<A.cols();++j) {
            if constexpr (fem::numeric::is_complex_number_v<T>) AH(j,i) = std::conj(A(i,j));
            else AH(j,i) = A(i,j);
        }
    return AH;
}

template <typename T>
Vector<T> matvec(const Matrix<T>& A, const Vector<T>& x)
{
    Vector<T> y(A.rows(), T{});
    for (size_t i=0;i<A.rows();++i) {
        T s{};
        for (size_t j=0;j<A.cols();++j) s += A(i,j) * x[j];
        y[i] = s;
    }
    return y;
}

}

TEST(Decompositions, Cholesky_Lower_ReconstructAndSolve)
{
    Matrix<double> A0 = {{4,1},
                         {1,3}}; // SPD
    Matrix<double> A = A0; // factor in-place
    int info = cholesky_factor(A, Uplo::Lower);
    EXPECT_EQ(info, 0);

    // A now holds L in lower triangle; reconstruct L*L^T
    Matrix<double> L(2,2, 0.0);
    L(0,0) = A(0,0); L(1,0) = A(1,0); L(1,1) = A(1,1);
    Matrix<double> LLt = matmulH(L, conj_transpose(L));
    EXPECT_NEAR(LLt(0,0), A0(0,0), 1e-12);
    EXPECT_NEAR(LLt(0,1), A0(0,1), 1e-12);
    EXPECT_NEAR(LLt(1,0), A0(1,0), 1e-12);
    EXPECT_NEAR(LLt(1,1), A0(1,1), 1e-12);

    // Solve A0 x = b for known solution
    Vector<double> x_ref = {2.0, -1.0};
    Vector<double> b = matvec(A0, x_ref);
    Matrix<double> chol = A; // factor already in A
    Vector<double> x = b;
    cholesky_solve_inplace(chol, x, Uplo::Lower);
    EXPECT_NEAR(x[0], x_ref[0], 1e-12);
    EXPECT_NEAR(x[1], x_ref[1], 1e-12);

    // Determinant check
    double det = cholesky_determinant(chol);
    double det_ref = A0(0,0)*A0(1,1) - A0(0,1)*A0(1,0);
    EXPECT_NEAR(det, det_ref, 1e-12);
}

TEST(Decompositions, Cholesky_Upper_Reconstruct)
{
    Matrix<double> A0 = {{25, 15, -5},
                         {15, 18,  0},
                         {-5,  0, 11}}; // SPD
    Matrix<double> A = A0;
    int info = cholesky_factor(A, Uplo::Upper);
    EXPECT_EQ(info, 0);

    // A now holds U in upper triangle; reconstruct U^T * U
    Matrix<double> U(3,3, 0.0);
    for (size_t i=0;i<3;++i) {
        for (size_t j=i;j<3;++j) U(i,j) = A(i,j);
    }
    Matrix<double> UtU = matmulH(conj_transpose(U), U);
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j) {
        EXPECT_NEAR(UtU(i,j), A0(i,j), 1e-10);
    }
}

TEST(Decompositions, Cholesky_ComplexHPD)
{
    using Z = std::complex<double>;
    Matrix<Z> H0(2,2, Z(0,0));
    H0(0,0) = Z(4,0);
    H0(0,1) = Z(1, 1);
    H0(1,0) = std::conj(H0(0,1));
    H0(1,1) = Z(3,0);

    Matrix<Z> H = H0;
    int info = cholesky_factor(H, Uplo::Lower);
    EXPECT_EQ(info, 0);

    // Reconstruct L*L^H
    Matrix<Z> L(2,2, Z(0,0));
    L(0,0) = H(0,0); L(1,0) = H(1,0); L(1,1) = H(1,1);
    Matrix<Z> LLH = matmulH(L, conj_transpose(L));
    for (size_t i=0;i<2;++i) for (size_t j=0;j<2;++j) {
        EXPECT_NEAR(std::abs(LLH(i,j) - H0(i,j)), 0.0, 1e-12);
    }
}

TEST(Decompositions, Cholesky_NonSPDDetection)
{
    Matrix<double> A = {{1,2},
                        {2,1}}; // indefinite (det < 0)
    int info = cholesky_factor(A, Uplo::Lower);
    EXPECT_EQ(info, 2); // failure at i=1 (return i+1)
}


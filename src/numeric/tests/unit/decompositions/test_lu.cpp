#include <gtest/gtest.h>

#include <decompositions/lu.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

namespace {

template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
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

TEST(Decompositions, LU_Factor_Reconstruct)
{
    // A with partial pivoting
    Matrix<double> A0 = {{2, 1, 1},
                         {4,-6, 0},
                         {-2, 7, 2}};
    Matrix<double> A = A0; // factor in-place

    std::vector<int> piv;
    int info = lu_factor(A, piv);
    EXPECT_EQ(info, 0);
    ASSERT_EQ(piv.size(), 3u);

    // Build P*A0 by applying recorded row swaps to a copy of A0
    Matrix<double> PA = A0;
    for (size_t k=0;k<piv.size();++k) {
        size_t p = static_cast<size_t>(piv[k]);
        if (p != k) {
            for (size_t j=0;j<PA.cols();++j) {
                auto tmp = PA(k,j); PA(k,j) = PA(p,j); PA(p,j) = tmp;
            }
        }
    }

    // Extract L and U from packed A
    Matrix<double> L(3,3, 0.0), U(3,3, 0.0);
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j) {
        if (i>j) L(i,j) = A(i,j);
        else if (i==j) { L(i,j) = 1.0; U(i,j) = A(i,j); }
        else U(i,j) = A(i,j);
    }
    Matrix<double> LU = matmul(L, U);
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j) {
        EXPECT_NEAR(LU(i,j), PA(i,j), 1e-12);
    }
}

TEST(Decompositions, LU_Solve_VectorAndMatrix)
{
    Matrix<double> A0 = {{3,1,6},
                         {2,1,3},
                         {1,1,1}};
    // Build RHS from known solution x = [1,2,3]^T
    Vector<double> x_ref = {1.0, 2.0, 3.0};
    Vector<double> b = matvec(A0, x_ref);

    // Factor and solve single RHS
    Matrix<double> LU = A0;
    std::vector<int> piv;
    int info = lu_factor(LU, piv);
    EXPECT_EQ(info, 0);

    Vector<double> x = b;
    lu_solve(LU, piv, x);
    ASSERT_EQ(x.size(), x_ref.size());
    for (size_t i=0;i<x.size();++i) EXPECT_NEAR(x[i], x_ref[i], 1e-12);

    // Multiple RHS (columns are different RHS)
    Matrix<double> B(A0.rows(), 2, 0.0);
    // RHS1 = A0*[1,0,0], RHS2 = A0*[0,1,0]
    Vector<double> e1 = {1.0,0.0,0.0};
    Vector<double> e2 = {0.0,1.0,0.0};
    Vector<double> b1 = matvec(A0, e1);
    Vector<double> b2 = matvec(A0, e2);
    for (size_t i=0;i<3;++i) { B(i,0) = b1[i]; B(i,1) = b2[i]; }

    Matrix<double> X = B; // in-place solve
    lu_solve_inplace(LU, piv, X);
    // Expect X columns equal to e1 and e2 (solutions to A x = RHS)
    for (size_t i=0;i<3;++i) {
        EXPECT_NEAR(X(i,0), e1[i], 1e-12);
        EXPECT_NEAR(X(i,1), e2[i], 1e-12);
    }
}

TEST(Decompositions, LU_Determinant)
{
    Matrix<double> A = {{1,2,3},
                        {0,1,4},
                        {5,6,0}};
    Matrix<double> LU = A;
    std::vector<int> piv;
    int info = lu_factor(LU, piv);
    EXPECT_EQ(info, 0);
    double det_lu = lu_determinant(LU, piv);
    // Direct determinant for 3x3
    double det_ref = A(0,0)*(A(1,1)*A(2,2)-A(1,2)*A(2,1))
                   - A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                   + A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
    EXPECT_NEAR(det_lu, det_ref, 1e-12);
}

TEST(Decompositions, LU_SingularDetection)
{
    Matrix<double> A = {{1,2},
                        {2,4}}; // rank-1
    std::vector<int> piv;
    int info = lu_factor(A, piv);
    EXPECT_NE(info, 0); // should flag singular (zero pivot)
}


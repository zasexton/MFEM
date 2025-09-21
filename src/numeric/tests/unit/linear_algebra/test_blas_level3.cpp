#include <gtest/gtest.h>

#include <linear_algebra/blas_level3.h>
#include <core/matrix.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::linear_algebra;

TEST(BLAS3, GemmBasic)
{
    Matrix<double> A = {{1,2},{3,4}};
    Matrix<double> B = {{5,6},{7,8}};
    Matrix<double> C(2,2, 1.0);

    gemm(Trans::NoTrans, Trans::NoTrans, 1.0, A, B, 0.0, C);
    // C = A*B
    EXPECT_DOUBLE_EQ(C(0,0), 19);
    EXPECT_DOUBLE_EQ(C(0,1), 22);
    EXPECT_DOUBLE_EQ(C(1,0), 43);
    EXPECT_DOUBLE_EQ(C(1,1), 50);
}

TEST(BLAS3, SyrkHerk)
{
    Matrix<double> A = {{1,2,3},{4,5,6}}; // 2x3
    Matrix<double> C(2,2, 0.0);
    syrk(Uplo::Upper, Trans::NoTrans, 1.0, A, 0.0, C); // C = A*A^T
    Matrix<double> ref(2,2,0.0);
    for (size_t i=0;i<2;++i) for(size_t j=i;j<2;++j) {
        double s=0; for(size_t k=0;k<3;++k) s+=A(i,k)*A(j,k); ref(i,j)=s;
    }
    EXPECT_NEAR(C(0,0), ref(0,0), 1e-12);
    EXPECT_NEAR(C(0,1), ref(0,1), 1e-12);
    EXPECT_NEAR(C(1,1), ref(1,1), 1e-12);

    using Cx = std::complex<double>;
    Matrix<Cx> Ac(2,2); Ac(0,0)=Cx(1,1); Ac(0,1)=Cx(2,-1); Ac(1,0)=Cx(0.5,2); Ac(1,1)=Cx(3,0);
    Matrix<Cx> H(2,2, Cx(0,0));
    herk(Uplo::Lower, Trans::NoTrans, 1.0, Ac, 0.0, H);
    // Check lower entries against op(A)*op(A)^H
    for(size_t i=0;i<2;++i) for(size_t j=0;j<=i;++j) {
        Cx s(0,0); for(size_t k=0;k<2;++k) s += Ac(i,k)*std::conj(Ac(j,k));
        EXPECT_NEAR(std::abs(H(i,j)-s), 0.0, 1e-12);
    }
}

TEST(BLAS3, SymmHemm)
{
    Matrix<double> A = {{1,2},{2,3}}; // symmetric
    Matrix<double> B = {{1,0},{0,1}}; // identity
    Matrix<double> C(2,2, 0.0);
    symm(Side::Left, Uplo::Upper, 2.0, A, B, 0.0, C); // C = 2*A
    EXPECT_DOUBLE_EQ(C(0,0), 2*1);
    EXPECT_DOUBLE_EQ(C(0,1), 2*2);
    EXPECT_DOUBLE_EQ(C(1,0), 2*2);
    EXPECT_DOUBLE_EQ(C(1,1), 2*3);

    using Z = std::complex<double>;
    Matrix<Z> H(2,2);
    H(0,0)=Z(2,0); H(0,1)=Z(1,2); H(1,0)=std::conj(H(0,1)); H(1,1)=Z(5,0);
    Matrix<Z> I(2,2, Z(0,0)); I(0,0)=I(1,1)=Z(1,0);
    Matrix<Z> Cc(2,2, Z(0,0));
    hemm(Side::Right, Uplo::Upper, Z(1,0), H, I, Z(0,0), Cc);
    // Cc = I*H = H
    EXPECT_NEAR(std::abs(Cc(0,1)-H(0,1)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(Cc(1,0)-H(1,0)), 0.0, 1e-12);
}

TEST(BLAS3, TrmmTrsm)
{
    // Triangular A
    Matrix<double> A = {{2,1},{0,3}}; // upper
    Matrix<double> B = {{1,2},{3,4}};
    Matrix<double> Btrmm = B;
    trmm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, 1.0, A, Btrmm);
    // Btrmm = A*B
    Matrix<double> ref(2,2,0.0);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) for(size_t k=0;k<2;++k) ref(i,j)+=A(i,k)*B(k,j);
    EXPECT_NEAR(Btrmm(0,1), ref(0,1), 1e-12);
    EXPECT_NEAR(Btrmm(1,1), ref(1,1), 1e-12);

    // Solve A*X = B  (Left, Upper)
    Matrix<double> X = B;
    trsm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, 1.0, A, X);
    // Check A*X equals B
    // Compute product properly
    Matrix<double> AX2(2,2,0.0);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) for(size_t k=0;k<2;++k) AX2(i,j)+=A(i,k)*X(k,j);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) EXPECT_NEAR(AX2(i,j), B(i,j), 1e-10);
}

TEST(BLAS3, GemmTransposeConjAndPointer)
{
    using C = std::complex<double>;
    Matrix<C> A = {{C(1,1), C(2,0)},
                   {C(0,-1), C(3,2)}}; // 2x2
    Matrix<C> B = {{C(1,0), C(0,1)},
                   {C(2,-1), C(1,0)}};
    Matrix<C> Cmat(2,2, C(0,0));
    // C = A^H * B
    gemm(Trans::ConjTranspose, Trans::NoTrans, C(1,0), A, B, C(0,0), Cmat);
    Matrix<C> ref(2,2, C(0,0));
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) for(size_t k=0;k<2;++k) ref(i,j)+=std::conj(A(k,i))*B(k,j);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) EXPECT_NEAR(std::abs(Cmat(i,j)-ref(i,j)), 0.0, 1e-12);

    // Pointer path RowMajor
    double Ar[6] = {1,2,3,4,5,6}; // 2x3 row-major
    double Br[6] = {1,0,0,1,1,1}; // 3x2 row-major
    double Cr[4] = {0,0,0,0};     // 2x2 row-major
    gemm<double,double,double,double,double>(fem::numeric::linear_algebra::Layout::RowMajor, Trans::NoTrans, Trans::NoTrans, 2, 2, 3,
        1.0, Ar, 3, Br, 2, 0.0, Cr, 2);
    // Reference
    double cref[4] = {0,0,0,0};
    auto idx = [&](double* P, size_t i, size_t j, size_t ld)->double&{ return P[i*ld+j]; };
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) for(size_t k=0;k<3;++k) idx(cref,i,j,2)+=Ar[i*3+k]*Br[k*2+j];
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) EXPECT_NEAR(idx(Cr,i,j,2), idx(cref,i,j,2), 1e-12);
}

TEST(BLAS3, SyrkTransposeAndPointer)
{
    Matrix<double> A = {{1,2},{3,4},{5,6}}; // 3x2
    Matrix<double> C(2,2, 0.0);
    syrk(Uplo::Lower, Trans::Transpose, 1.0, A, 0.0, C); // C = A^T * A lower
    Matrix<double> ref(2,2, 0.0);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<=i;++j) { double s=0; for(size_t k=0;k<3;++k) s+=A(k,i)*A(k,j); ref(i,j)=s; }
    for(size_t i=0;i<2;++i) for(size_t j=0;j<=i;++j) EXPECT_NEAR(C(i,j), ref(i,j), 1e-12);
}

TEST(BLAS3, GemmBetaFastPaths)
{
    Matrix<double> A = {{1,2},{3,4}};
    Matrix<double> B = {{5,6},{7,8}};
    // beta = 0
    Matrix<double> C0(2,2, 9.0);
    gemm(Trans::NoTrans, Trans::NoTrans, 1.0, A, B, 0.0, C0);
    EXPECT_DOUBLE_EQ(C0(0,0), 19);
    EXPECT_DOUBLE_EQ(C0(1,1), 50);
    // beta = 1
    Matrix<double> C1(2,2, 1.0);
    gemm(Trans::NoTrans, Trans::NoTrans, 1.0, A, B, 1.0, C1);
    EXPECT_DOUBLE_EQ(C1(0,0), 1.0 + 19);
    EXPECT_DOUBLE_EQ(C1(1,1), 1.0 + 50);
}

TEST(BLAS3, PointerSymmHemmTrmmTrsmVariants)
{
    // symm pointer (Lower, Left)
    double As[4] = {1,2, 2,3}; // 2x2 row-major symmetric
    double I2[4] = {1,0, 0,1};
    double C[4]  = {0,0, 0,0};
    symm<double,double,double,double,double>(fem::numeric::linear_algebra::Layout::RowMajor, Side::Left, Uplo::Lower,
        2,2, 3.0, As, 2, I2, 2, 0.0, C, 2);
    EXPECT_DOUBLE_EQ(C[0], 3*1); EXPECT_DOUBLE_EQ(C[1], 3*2);
    EXPECT_DOUBLE_EQ(C[2], 3*2); EXPECT_DOUBLE_EQ(C[3], 3*3);

    // hemm pointer (Lower, Right), H times identity -> H
    using Z = std::complex<double>;
    Z H[4] = {Z(2,0), Z(1,2), Z(1,-2), Z(5,0)}; // Hermitian
    Z I[4] = {Z(1,0), Z(0,0), Z(0,0), Z(1,0)};
    Z CH[4] = {Z(0,0),Z(0,0),Z(0,0),Z(0,0)};
    hemm<Z,Z,Z,Z,Z>(fem::numeric::linear_algebra::Layout::RowMajor, Side::Right, Uplo::Lower,
        2,2, Z(1,0), H, 2, I, 2, Z(0,0), CH, 2);
    EXPECT_NEAR(std::abs(CH[1]-H[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(CH[2]-H[2]), 0.0, 1e-12);

    // trmm pointer Right, Lower, Transpose, Unit
    double Al[4] = {9,0, 2,7}; // lower triangular; diag ignored (unit)
    double Bm[4] = {1,0, 0,1};
    trmm<double,double,double>(fem::numeric::linear_algebra::Layout::RowMajor, Side::Right, Uplo::Lower, Trans::Transpose, Diag::Unit,
        2,2, 1.0, Al, 2, Bm, 2);
    // Expect B := B * op(A) with op(A)=A^T (upper unit with u12=2)
    EXPECT_DOUBLE_EQ(Bm[0], 1); EXPECT_DOUBLE_EQ(Bm[1], 2);
    EXPECT_DOUBLE_EQ(Bm[2], 0); EXPECT_DOUBLE_EQ(Bm[3], 1);

    // trsm pointer Right, Lower, Transpose, Unit; Solve X * op(A) = I
    double Al2[4] = {9,0, 2,7};
    double X[4] = {1,0, 0,1};
    trsm<double,double,double>(fem::numeric::linear_algebra::Layout::RowMajor, Side::Right, Uplo::Lower, Trans::Transpose, Diag::Unit,
        2,2, 1.0, Al2, 2, X, 2);
    // X should be inverse of unit upper with u12=2 -> [[1,-2],[0,1]]
    EXPECT_DOUBLE_EQ(X[0], 1); EXPECT_DOUBLE_EQ(X[1], -2);
    EXPECT_DOUBLE_EQ(X[2], 0); EXPECT_DOUBLE_EQ(X[3], 1);
}

TEST(BLAS3, PointerSymmHemmTrmmTrsmVariants_ColMajor)
{
    // symm pointer (Lower, Left) ColumnMajor
    double As_col[4] = {1, 2, 2, 3}; // col0: [1,2], col1: [2,3]
    double I2_col[4] = {1, 0, 0, 1};
    double C_col[4]  = {0, 0, 0, 0};
    symm<double,double,double,double,double>(fem::numeric::linear_algebra::Layout::ColMajor, Side::Left, Uplo::Lower,
        2, 2, 4.0, As_col, 2, I2_col, 2, 0.0, C_col, 2);
    // Expect C = 4*A
    EXPECT_DOUBLE_EQ(C_col[0], 4*1); EXPECT_DOUBLE_EQ(C_col[1], 4*2);
    EXPECT_DOUBLE_EQ(C_col[2], 4*2); EXPECT_DOUBLE_EQ(C_col[3], 4*3);

    // hemm pointer (Lower, Right) ColumnMajor: C = I*H = H
    using Z = std::complex<double>;
    Z H_col[4] = {Z(2,0), Z(1,-2), Z(1,2), Z(5,0)}; // col0: [2, 1-2i], col1: [1+2i, 5]
    Z I_col[4] = {Z(1,0), Z(0,0), Z(0,0), Z(1,0)};
    Z CH_col[4] = {Z(0,0), Z(0,0), Z(0,0), Z(0,0)};
    hemm<Z,Z,Z,Z,Z>(fem::numeric::linear_algebra::Layout::ColMajor, Side::Right, Uplo::Lower,
        2, 2, Z(1,0), H_col, 2, I_col, 2, Z(0,0), CH_col, 2);
    EXPECT_NEAR(std::abs(CH_col[2] - H_col[2]), 0.0, 1e-12); // (0,1)
    EXPECT_NEAR(std::abs(CH_col[1] - H_col[1]), 0.0, 1e-12); // (1,0)

    // trmm pointer Right, Lower, Transpose, Unit, ColumnMajor
    double A_low_col[4] = {9, 2, 0, 7}; // lower tri, col-major
    double B_col[4] = {1,0, 0,1};
    trmm<double,double,double>(fem::numeric::linear_algebra::Layout::ColMajor, Side::Right, Uplo::Lower, Trans::Transpose, Diag::Unit,
        2, 2, 1.0, A_low_col, 2, B_col, 2);
    // B := B * op(A), op(A) = A^T -> upper unit with u01=2
    // Column-major B expected [[1,2],[0,1]] => [1,0, 2,1]
    EXPECT_DOUBLE_EQ(B_col[0], 1);
    EXPECT_DOUBLE_EQ(B_col[1], 0);
    EXPECT_DOUBLE_EQ(B_col[2], 2);
    EXPECT_DOUBLE_EQ(B_col[3], 1);

    // trsm pointer Right, Lower, Transpose, Unit, ColumnMajor: solve X*op(A)=I
    double A_low_col2[4] = {9, 2, 0, 7};
    double X_col[4] = {1,0, 0,1};
    trsm<double,double,double>(fem::numeric::linear_algebra::Layout::ColMajor, Side::Right, Uplo::Lower, Trans::Transpose, Diag::Unit,
        2, 2, 1.0, A_low_col2, 2, X_col, 2);
    // inverse of upper unit [[1,2],[0,1]] -> [[1,-2],[0,1]] -> col-major [1,0,-2,1]
    EXPECT_DOUBLE_EQ(X_col[0], 1);
    EXPECT_DOUBLE_EQ(X_col[1], 0);
    EXPECT_DOUBLE_EQ(X_col[2], -2);
    EXPECT_DOUBLE_EQ(X_col[3], 1);
}

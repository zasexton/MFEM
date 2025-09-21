#include <gtest/gtest.h>

#include <linear_algebra/blas_level2.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::linear_algebra;

TEST(BLAS2, GemvNoTrans)
{
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}}; // 2x3
    Vector<double> x = {1.0, 2.0, 3.0};
    Vector<double> y = {10.0, 20.0};

    gemv(Trans::NoTrans, 2.0, A, x, 0.5, y);
    // y = 0.5*y + 2*A*x; A*x = [14, 32]
    EXPECT_DOUBLE_EQ(y[0], 0.5*10 + 2*14);
    EXPECT_DOUBLE_EQ(y[1], 0.5*20 + 2*32);
}

TEST(BLAS2, GemvTranspose)
{
    Matrix<double> A = {{1.0, 2.0},
                        {3.0, 4.0},
                        {5.0, 6.0}}; // 3x2
    Vector<double> x = {1.0, 2.0, 3.0};
    Vector<double> y = {0.0, 0.0};

    gemv(Trans::Transpose, 1.0, A, x, 0.0, y); // y = A^T x
    // A^T x = [1*1+3*2+5*3, 2*1+4*2+6*3] = [22, 28]
    EXPECT_DOUBLE_EQ(y[0], 22.0);
    EXPECT_DOUBLE_EQ(y[1], 28.0);
}

TEST(BLAS2, GerSyrHemv)
{
    Vector<double> x = {1.0, 2.0, 3.0};
    Vector<double> y = {4.0, 5.0, 6.0};
    Matrix<double> A(3, 3); A.fill(0.0);

    ger(2.0, x, y, A); // A += 2*x*y^T
    // Check a couple entries
    EXPECT_DOUBLE_EQ(A(0,0), 2*1*4);
    EXPECT_DOUBLE_EQ(A(2,1), 2*3*5);

    // symv with upper triangle (reuse A as dense symmetric by mirroring)
    // Build symmetric S = (A + A^T)/2
    Matrix<double> S = A;
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j) S(i,j) = 0.5*(A(i,j)+A(j,i));
    Vector<double> z(3, 0.0);
    symv(Uplo::Upper, 1.0, S, x, 0.0, z);
    // z should equal S*x
    Vector<double> zx(3, 0.0);
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j) zx[i]+=S(i,j)*x[j];
    for (size_t i=0;i<3;++i) EXPECT_NEAR(z[i], zx[i], 1e-12);

    // hemv with complex
    using C = std::complex<double>;
    Matrix<C> H(2,2);
    H(0,0)=C(2,0); H(0,1)=C(1,2);
    H(1,0)=std::conj(H(0,1)); H(1,1)=C(3,0);
    Vector<C> xc = {C(1,0), C(2,-1)};
    Vector<C> yc = {C(0,0), C(0,0)};
    hemv(Uplo::Upper, C(1,0), H, xc, C(0,0), yc); // yc=H*xc
    Vector<C> yc_ref(2, C(0,0));
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) yc_ref[i]+=H(i,j)*xc[j];
    for(size_t i=0;i<2;++i) EXPECT_NEAR(std::abs(yc[i]-yc_ref[i]), 0.0, 1e-12);
}

TEST(BLAS2, GemvConjTransposeComplexAndPointer)
{
    using C = std::complex<double>;
    Matrix<C> A = {{C(1,1), C(2,-1)},
                   {C(3,2), C(4,0)}}; // 2x2
    Vector<C> x = {C(1,-1), C(2,1)};
    Vector<C> y(2, C(0,0));

    gemv(Trans::ConjTranspose, C(1,0), A, x, C(0,0), y); // y = A^H x
    Vector<C> ref(2, C(0,0));
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) ref[i]+=std::conj(A(j,i))*x[j];
    for(size_t i=0;i<2;++i) EXPECT_NEAR(std::abs(y[i]-ref[i]), 0.0, 1e-12);

    // Pointer path RowMajor
    double Ar[6] = {1,2,3,4,5,6}; // 2x3 row-major
    double xv[3] = {1,2,3};
    double yv[2] = {0,0};
    gemv<double,double,double,double,double>(fem::numeric::linear_algebra::Layout::RowMajor, Trans::NoTrans, 2, 3, 1.0, Ar, 3, xv, 1, 0.0, yv, 1);
    EXPECT_DOUBLE_EQ(yv[0], 14);
    EXPECT_DOUBLE_EQ(yv[1], 32);
}

TEST(BLAS2, SyrHerAndSymvLower)
{
    // syr lower update
    Vector<double> x = {1,2,3};
    Matrix<double> A(3,3, 0.0);
    syr(Uplo::Lower, 1.0, x, A);
    for(size_t i=0;i<3;++i) for(size_t j=0;j<=i;++j) EXPECT_DOUBLE_EQ(A(i,j), x[i]*x[j]);
    // Upper untouched
    for(size_t i=0;i<3;++i) for(size_t j=i+1;j<3;++j) EXPECT_DOUBLE_EQ(A(i,j), 0.0);

    // her lower update with complex
    using C = std::complex<double>;
    Vector<C> xc = {C(1,1), C(2,-1)};
    Matrix<C> H(2,2, C(0,0));
    her(Uplo::Lower, 1.0, xc, H);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<=i;++j)
        EXPECT_NEAR(std::abs(H(i,j) - xc[i]*std::conj(xc[j])), 0.0, 1e-12);

    // symv lower path check
    Matrix<double> S = {{2,1,0},{1,3,4},{0,4,5}}; // symmetric assumed
    Vector<double> vx = {1,2,3};
    Vector<double> vy(3,0.0);
    symv(Uplo::Lower, 1.0, S, vx, 0.0, vy);
    Vector<double> vy_ref(3,0.0);
    for(size_t i=0;i<3;++i) for(size_t j=0;j<3;++j) vy_ref[i]+=S(i,j)*vx[j];
    for(size_t i=0;i<3;++i) EXPECT_NEAR(vy[i], vy_ref[i], 1e-12);
}

TEST(BLAS2, GemvBetaFastPaths)
{
    Matrix<double> A = {{1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0}}; // 2x3
    Vector<double> x = {1.0, 2.0, 3.0};

    // beta = 0 => y = alpha*A*x
    Vector<double> y0 = {10.0, 20.0};
    gemv(Trans::NoTrans, 2.0, A, x, 0.0, y0);
    EXPECT_DOUBLE_EQ(y0[0], 2.0*14.0);
    EXPECT_DOUBLE_EQ(y0[1], 2.0*32.0);

    // beta = 1 => y = y + alpha*A*x
    Vector<double> y1 = {10.0, 20.0};
    gemv(Trans::NoTrans, 2.0, A, x, 1.0, y1);
    EXPECT_DOUBLE_EQ(y1[0], 10.0 + 2.0*14.0);
    EXPECT_DOUBLE_EQ(y1[1], 20.0 + 2.0*32.0);
}

TEST(BLAS2, GemvDimensionMismatchThrows)
{
    Matrix<double> A(2,3);
    Vector<double> x(2);
    Vector<double> y(2);
    EXPECT_THROW(gemv(Trans::NoTrans, 1.0, A, x, 0.0, y), std::invalid_argument);
}

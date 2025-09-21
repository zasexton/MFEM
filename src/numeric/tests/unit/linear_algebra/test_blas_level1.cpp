#include <gtest/gtest.h>

#include <linear_algebra/blas_level1.h>
#include <core/vector.h>
#include <complex>

using namespace fem::numeric;
using namespace fem::numeric::linear_algebra;

TEST(BLAS1, ScalAxpyCopySwapReal)
{
    Vector<double> x = {1.0, 2.0, 3.0};
    Vector<double> y = {4.0, 5.0, 6.0};

    scal(2.0, x); // x = [2,4,6]
    EXPECT_DOUBLE_EQ(x[0], 2.0);
    EXPECT_DOUBLE_EQ(x[1], 4.0);
    EXPECT_DOUBLE_EQ(x[2], 6.0);

    axpy(3.0, x, y); // y = 3*x + y = [10,17,24]
    EXPECT_DOUBLE_EQ(y[0], 10.0);
    EXPECT_DOUBLE_EQ(y[1], 17.0);
    EXPECT_DOUBLE_EQ(y[2], 24.0);

    Vector<double> z(3);
    copy(x, z);
    EXPECT_DOUBLE_EQ(z[0], 2.0);
    EXPECT_DOUBLE_EQ(z[1], 4.0);
    EXPECT_DOUBLE_EQ(z[2], 6.0);

    swap(x, z); // swap back
    EXPECT_DOUBLE_EQ(x[1], 4.0);
}

TEST(BLAS1, DotAndNorms)
{
    Vector<double> a = {1.0, -2.0, 3.0};
    Vector<double> b = {-4.0, 5.0, -6.0};

    auto d = dotu(a, b);
    EXPECT_DOUBLE_EQ(d, 1*(-4) + (-2)*5 + 3*(-6));

    auto n2 = nrm2(a);
    EXPECT_NEAR(n2, std::sqrt(1 + 4 + 9), 1e-12);

    auto a1 = asum(b);
    EXPECT_DOUBLE_EQ(a1, 4.0 + 5.0 + 6.0);

    auto idx = iamax(a);
    EXPECT_EQ(idx, 2u);
}

TEST(BLAS1, ComplexDotc)
{
    using C = std::complex<double>;
    Vector<C> x = {C(1,2), C(3,4)}; // [1+2i, 3+4i]
    Vector<C> y = {C(5,6), C(7,8)}; // [5+6i, 7+8i]

    auto du = dotu(x, y);   // no conjugation
    auto dc = dotc(x, y);   // conj(x) * y

    // du = (1+2i)(5+6i) + (3+4i)(7+8i)
    C du_expected = (C(1,2)*C(5,6)) + (C(3,4)*C(7,8));
    EXPECT_EQ(du, du_expected);

    // dc = (1-2i)(5+6i) + (3-4i)(7+8i)
    C dc_expected = std::conj(C(1,2))*C(5,6) + std::conj(C(3,4))*C(7,8);
    EXPECT_EQ(dc, dc_expected);
}

TEST(BLAS1, EdgeCasesAndPointerPaths)
{
    // Zero-length vectors
    Vector<double> empty;
    EXPECT_NO_THROW(scal(2.0, empty));
    EXPECT_NO_THROW(asum(empty));
    EXPECT_NO_THROW(nrm2(empty));

    // alpha = 0 for scal/axpy
    Vector<double> x = {1,2,3};
    Vector<double> y = {7,8,9};
    scal(0.0, x);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
    EXPECT_DOUBLE_EQ(x[2], 0.0);
    axpy(0.0, x, y); // y unchanged
    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 8.0);
    EXPECT_DOUBLE_EQ(y[2], 9.0);

    // iamax/iamin tie-break (first index)
    Vector<double> t = {5.0, -5.0, 3.0};
    EXPECT_EQ(iamax(t), 0u);
    EXPECT_EQ(iamin(t), 2u); // min abs is 3 at index 2

    // asum complex semantics: |Re|+|Im|
    using C = std::complex<double>;
    Vector<C> cx = {C(3,-4), C(-1,2)};
    auto asum_c = asum(cx);
    EXPECT_DOUBLE_EQ(asum_c, 3 + 4 + 1 + 2);

    // rot: apply a simple rotation
    Vector<double> xr = {1.0, 0.0};
    Vector<double> yr = {0.0, 1.0};
    double c = std::sqrt(0.5);
    double s = c;
    rot(xr, yr, c, s);
    // First element becomes [c*1 + s*0, c*0 + s*1] = [c, s]
    EXPECT_NEAR(xr[0], c, 1e-12);
    EXPECT_NEAR(yr[0], -s*1 + c*0, 1e-12); // equals -s

    // Pointer paths with increments
    double xp[5] = {1, 2, 3, 4, 5};
    scal<std::double_t,double>(3, 2.0, xp, 2); // scale elements at 0,2,4
    EXPECT_DOUBLE_EQ(xp[0], 2.0);
    EXPECT_DOUBLE_EQ(xp[2], 6.0);
    EXPECT_DOUBLE_EQ(xp[4], 10.0);

    double yp[5] = {10, 20, 30, 40, 50};
    axpy(3, 0.5, xp, 2, yp, 2); // y[0]+=1.0, y[2]+=3.0, y[4]+=5.0 => 11,23,55
    EXPECT_DOUBLE_EQ(yp[0], 11.0);
    EXPECT_DOUBLE_EQ(yp[2], 33.0);
    EXPECT_DOUBLE_EQ(yp[4], 55.0);

    // dot alias equals dotu (real and complex)
    Vector<double> r1 = {1,2}, r2 = {3,4};
    EXPECT_DOUBLE_EQ(dotu(r1,r2), dot(r1,r2));
    Vector<C> c1 = {C(1,1), C(2,-1)};
    Vector<C> c2 = {C(3,0), C(0,4)};
    auto d1 = dotu(c1,c2);
    auto d2 = dot(c1,c2);
    EXPECT_NEAR(std::abs(d1-d2), 0.0, 1e-12);

    // pointer rot path
    double xr2[4] = {1,0, 2,0};
    double yr2[4] = {0,1, 0,2};
    rot(2, xr2, 2, yr2, 2, c, s);
    // First pair rotates (1,0) with (0,1), second pair rotates (2,0) with (0,2)
    EXPECT_NEAR(xr2[0], c*1 + s*0, 1e-12);
    EXPECT_NEAR(yr2[0], c*0 - s*1, 1e-12);
}

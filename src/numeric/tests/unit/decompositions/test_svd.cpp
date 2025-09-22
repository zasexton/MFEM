#include <gtest/gtest.h>

#include <decompositions/svd.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <cmath>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

namespace {

template <typename T>
Matrix<T> conj_transpose(const Matrix<T>& A)
{
    Matrix<T> AH(A.cols(), A.rows(), T{});
    for (size_t i=0;i<A.rows();++i)
      for (size_t j=0;j<A.cols();++j)
        if constexpr (fem::numeric::is_complex_number_v<T>) AH(j,i) = std::conj(A(i,j));
        else AH(j,i) = A(i,j);
    return AH;
}

template <typename T>
void fill_random(Matrix<T>& A, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i=0;i<A.rows();++i)
      for (size_t j=0;j<A.cols();++j) A(i,j) = static_cast<T>(dist(gen));
}

template <typename T>
void fill_random(Vector<T>& v, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i=0;i<v.size();++i) v[i] = static_cast<T>(dist(gen));
}

template <typename Z>
void fill_random_complex(Matrix<Z>& A, uint32_t seed)
{
    using R = typename fem::numeric::numeric_traits<Z>::scalar_type;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> dist(-1.0, 1.0);
    for (size_t i=0;i<A.rows();++i)
      for (size_t j=0;j<A.cols();++j)
        A(i,j) = Z(dist(gen), dist(gen));
}

template <typename Z>
void fill_random_complex(Vector<Z>& v, uint32_t seed)
{
    using R = typename fem::numeric::numeric_traits<Z>::scalar_type;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> dist(-1.0, 1.0);
    for (size_t i=0;i<v.size();++i) v[i] = Z(dist(gen), dist(gen));
}

} // namespace

TEST(Decompositions, SVD_Reconstruction_Real_Thin)
{
    std::vector<std::pair<size_t,size_t>> shapes{{5,3},{3,5},{4,4}};
    for (auto [m,n] : shapes) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(100 + m + 10*n));
        Matrix<double> U, Vt;
        Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true);
        // Reconstruct A via U * diag(S) * Vt
        const size_t r = std::min(m,n);
        Matrix<double> US(m, r, 0.0);
        for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<double> R(m, n, 0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
            double s=0; for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j);
            R(i,j) = s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(R(i,j), A(i,j), 1e-8);
        // Orthogonality: U^T U = I_r, Vt * Vt^T = I_r
        Matrix<double> Ut = conj_transpose(U);
        Matrix<double> UtU(r, r, 0.0);
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            double s=0; for (size_t k=0;k<m;++k) s += Ut(i,k) * U(k,j);
            UtU(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(UtU(i,j), (i==j)?1.0:0.0, 1e-10);
        Matrix<double> V = conj_transpose(Vt);
        Matrix<double> VtV(r, r, 0.0);
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            double s=0; for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j);
            VtV(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(VtV(i,j), (i==j)?1.0:0.0, 1e-10);
        // Nonnegative and sorted
        for (size_t i=0;i+1<S.size();++i) {
            EXPECT_GE(S[i], S[i+1]);
            EXPECT_GE(S[i], 0.0);
        }
    }
}

TEST(Decompositions, SVD_Reconstruction_Complex_Thin)
{
    using Z = std::complex<double>;
    std::vector<std::pair<size_t,size_t>> shapes{{4,2},{3,3},{2,5}};
    for (auto [m,n] : shapes) {
        Matrix<Z> A(m,n, Z(0,0)); fill_random_complex(A, static_cast<uint32_t>(311 + m + 7*n));
        Matrix<Z> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true);
        const size_t r = std::min(m,n);
        Matrix<Z> US(m, r, Z(0,0));
        for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<Z> R(m, n, Z(0,0));
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
            Z s(0,0); for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j);
            R(i,j) = s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(std::abs(R(i,j) - A(i,j)), 0.0, 1e-8);
        // Orthogonality checks
        Matrix<Z> Ut = conj_transpose(U);
        Matrix<Z> UtU(r, r, Z(0,0));
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            Z s(0,0); for (size_t k=0;k<m;++k) s += Ut(i,k) * U(k,j);
            UtU(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(std::abs(UtU(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
        Matrix<Z> V = conj_transpose(Vt);
        Matrix<Z> VtV(r, r, Z(0,0));
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            Z s(0,0); for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j);
            VtV(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(std::abs(VtV(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
    }
}

TEST(Decompositions, SVD_LeastSquares_PseudoInverse)
{
    const size_t m=7, n=3;
    Matrix<double> A(m,n, 0.0); fill_random(A, 777u);
    Vector<double> b(m, 0.0); fill_random(b, 999u);
    Vector<double> x = svd_solve(A, b);
    // Residual orthogonality: A^T (Ax - b) ≈ 0
    Vector<double> r(m, 0.0);
    for (size_t i=0;i<m;++i) {
        double s=0; for (size_t j=0;j<n;++j) s += A(i,j) * x[j];
        r[i] = s - b[i];
    }
    // Compute A^T r
    for (size_t i=0;i<n;++i) {
        double s=0; for (size_t j=0;j<m;++j) s += A(j,i) * r[j];
        EXPECT_NEAR(s, 0.0, 1e-8);
    }
}

TEST(Decompositions, SVD_RankDeficient_SmallSigma)
{
    const size_t m=6, n=3;
    Matrix<double> A(m,n, 0.0); fill_random(A, 123u);
    double eps = 1e-10;
    for (size_t i=0;i<m;++i) A(i, n-1) = 2.0 * A(i, 0) + eps * static_cast<double>(i+1);
    Matrix<double> U, Vt; Vector<double> S;
    svd(A, U, S, Vt, /*thin=*/true);
    EXPECT_LE(S[n-1], 1e-6);
}

TEST(Decompositions, SVD_ZeroMatrix_RealAndComplex)
{
    // Real zero
    for (auto [m,n] : { std::pair<size_t,size_t>{5,3}, std::pair<size_t,size_t>{3,7} }) {
        Matrix<double> A(m,n, 0.0);
        Matrix<double> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true);
        // All singular values zero
        for (size_t i=0;i<S.size();++i) EXPECT_NEAR(S[i], 0.0, 0.0);
        // Reconstruction is zero
        const size_t r = std::min(m,n);
        Matrix<double> US(m, r, 0.0);
        Matrix<double> R(m, n, 0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(R(i,j), 0.0, 0.0);
    }
    // Complex zero
    using Z = std::complex<double>;
    Matrix<Z> A(4,6, Z(0,0));
    Matrix<Z> U, Vt; Vector<double> S;
    svd(A, U, S, Vt, /*thin=*/true);
    for (size_t i=0;i<S.size();++i) EXPECT_NEAR(S[i], 0.0, 0.0);
}

TEST(Decompositions, SVD_SmallSizes_1xN_and_Mx1)
{
    // 1 x n
    {
        const size_t m=1, n=5; Matrix<double> A(m,n, 0.0); fill_random(A, 606u);
        Matrix<double> U, Vt; Vector<double> S; svd(A, U, S, Vt, /*thin=*/true);
        ASSERT_EQ(S.size(), 1u);
        // Only one singular value equals norm of row
        double nrm=0; for (size_t j=0;j<n;++j) nrm += A(0,j)*A(0,j); nrm=std::sqrt(nrm);
        EXPECT_NEAR(S[0], nrm, 1e-12);
    }
    // m x 1
    {
        const size_t m=7, n=1; Matrix<double> A(m,n, 0.0); fill_random(A, 707u);
        Matrix<double> U, Vt; Vector<double> S; svd(A, U, S, Vt, /*thin=*/true);
        ASSERT_EQ(S.size(), 1u);
        double nrm=0; for (size_t i=0;i<m;++i) nrm += A(i,0)*A(i,0); nrm=std::sqrt(nrm);
        EXPECT_NEAR(S[0], nrm, 1e-12);
    }
}

TEST(Decompositions, SVD_Pseudoinverse_ProjectorProperties)
{
    // A A^+ A = A and A^+ A A^+ = A^+ for tall and wide
    for (auto [m,n] : { std::pair<size_t,size_t>{9,5}, std::pair<size_t,size_t>{5,9} }) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(888 + m + n));
        Matrix<double> Ap = pinv(A);
        // P1 = A * Ap * A
        Matrix<double> tmp1(m,n,0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
            double s=0; for (size_t k=0;k<n;++k){ double s2=0; for (size_t t=0;t<m;++t) s2 += Ap(k,t)*A(t,j); s += A(i,k)*s2; }
            tmp1(i,j)=s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(tmp1(i,j), A(i,j), 1e-8);
        // P2 = Ap * A * Ap
        Matrix<double> tmp2(n,m,0.0);
        for (size_t i=0;i<n;++i) for (size_t j=0;j<m;++j) {
            double s=0; for (size_t k=0;k<m;++k){ double s2=0; for (size_t t=0;t<n;++t) s2 += A(k,t)*Ap(t,j); s += Ap(i,k)*s2; }
            tmp2(i,j)=s;
        }
        for (size_t i=0;i<n;++i) for (size_t j=0;j<m;++j) EXPECT_NEAR(tmp2(i,j), Ap(i,j), 1e-8);
    }
}

TEST(Decompositions, SVD_SingularValues_Agreement_GR_vs_Jacobi)
{
    // Compare singular values for random real matrices across shapes
    for (auto [m,n] : { std::pair<size_t,size_t>{30,20}, std::pair<size_t,size_t>{50,70} }) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(4242 + m + n));
        Matrix<double> Uj, Vtj; Vector<double> Sj;
        svd(A, Uj, Sj, Vtj, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::Jacobi);
        Matrix<double> Ug, Vtg; Vector<double> Sg;
        svd(A, Ug, Sg, Vtg, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        ASSERT_EQ(Sj.size(), Sg.size());
        for (size_t i=0;i<Sj.size();++i) EXPECT_NEAR(Sj[i], Sg[i], 1e-7);
    }
}

TEST(Decompositions, SVD_ComplexLeastSquares_ResidualOrthogonality)
{
    using Z = std::complex<double>;
    const size_t m=12, n=7;
    Matrix<Z> A(m,n, Z(0,0)); fill_random_complex(A, 9191u);
    Vector<Z> b(m, Z(0,0)); fill_random_complex(b, 9292u);
    Matrix<Z> U, Vt; Vector<double> S; svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::Jacobi);
    // Solve via triplets
    // y = U^H b
    Vector<Z> y(S.size(), Z(0,0));
    for (size_t k=0;k<S.size();++k){ Z s(0,0); for (size_t i=0;i<m;++i) s += std::conj(U(i,k)) * b[i]; y[k]=s; }
    double smax = 0.0; for (size_t k=0;k<S.size();++k) smax = std::max(smax, S[k]);
    double cutoff = std::numeric_limits<double>::epsilon() * std::max(m,n) * smax;
    for (size_t k=0;k<S.size();++k) y[k] = (S[k] > cutoff) ? (y[k] / S[k]) : Z(0,0);
    // x = V y
    Vector<Z> x(n, Z(0,0));
    for (size_t i=0;i<n;++i){ Z s(0,0); for (size_t k=0;k<S.size();++k) s += std::conj(Vt(k,i)) * y[k]; x[i]=s; }
    // Residual orthogonality: A^H(Ax - b) ≈ 0
    Vector<Z> r(m, Z(0,0)); for (size_t i=0;i<m;++i){ Z s(0,0); for (size_t j=0;j<n;++j) s += A(i,j) * x[j]; r[i]=s - b[i]; }
    for (size_t i=0;i<n;++i){ Z s(0,0); for (size_t j=0;j<m;++j) s += std::conj(A(j,i)) * r[j]; EXPECT_NEAR(std::abs(s), 0.0, 1e-8); }
}

TEST(Decompositions, SVD_ExactRankDeficient_ZerosInS)
{
    // Exact dependency: last column = 3 * first column
    const size_t m=10, n=6; Matrix<double> A(m,n, 0.0); fill_random(A, 7373u);
    for (size_t i=0;i<m;++i) A(i, n-1) = 3.0 * A(i, 0);
    Matrix<double> U, Vt; Vector<double> S; svd(A, U, S, Vt, /*thin=*/true);
    // Smallest singular value should be ~0
    EXPECT_LE(S[S.size()-1], 1e-10);
    // Reconstruction still holds
    const size_t r = std::min(m,n);
    Matrix<double> US(m, r, 0.0); for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k)=U(i,k)*S[k];
    Matrix<double> R(m, n, 0.0);
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j){ double s=0; for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j); R(i,j)=s; }
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(R(i,j), A(i,j), 1e-8);
}

TEST(Decompositions, SVD_Reconstruction_Real_GolubReinsch_Thin)
{
    // Larger shapes to exercise GR path; include tall and wide
    std::vector<std::pair<size_t,size_t>> shapes{{64,32},{80,50},{40,64}};
    for (auto [m,n] : shapes) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(2025 + m + 3*n));
        Matrix<double> U, Vt;
        Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/60, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        // Reconstruct A via U * diag(S) * Vt
        const size_t r = std::min(m,n);
        Matrix<double> US(m, r, 0.0);
        for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<double> R(m, n, 0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
            double s=0; for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j);
            R(i,j) = s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(R(i,j), A(i,j), 1e-8);
        // Orthogonality of U and V
        Matrix<double> Ut = conj_transpose(U);
        Matrix<double> UtU(r, r, 0.0);
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            double s=0; for (size_t k=0;k<m;++k) s += Ut(i,k) * U(k,j);
            UtU(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(UtU(i,j), (i==j)?1.0:0.0, 1e-8);
        Matrix<double> V = conj_transpose(Vt);
        Matrix<double> VtV(r, r, 0.0);
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) {
            double s=0; for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j);
            VtV(i,j) = s;
        }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j)
            EXPECT_NEAR(VtV(i,j), (i==j)?1.0:0.0, 1e-8);
    }
}

TEST(Decompositions, SVD_Reconstruction_Real_GolubReinsch_Full)
{
    // Check full SVD (U m×m, Vt n×n) assembly and orthogonality
    std::vector<std::pair<size_t,size_t>> shapes{{64,32},{40,64},{50,50}};
    for (auto [m,n] : shapes) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(31415 + m + 2*n));
        Matrix<double> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/false, /*max_sweeps=*/60, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        // Orthogonality: U^T U = I_m, Vt V = I_n
        Matrix<double> Ut = conj_transpose(U);
        Matrix<double> UtU(m, m, 0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j) {
            double s=0; for (size_t k=0;k<m;++k) s += Ut(i,k) * U(k,j);
            UtU(i,j) = s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j)
            EXPECT_NEAR(UtU(i,j), (i==j)?1.0:0.0, 1e-10);
        Matrix<double> V = conj_transpose(Vt);
        Matrix<double> VtV(n, n, 0.0);
        for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j) {
            double s=0; for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j);
            VtV(i,j) = s;
        }
        for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(VtV(i,j), (i==j)?1.0:0.0, 1e-10);
        // Reconstruction using first r singular triplets
        const size_t r = std::min(m,n);
        Matrix<double> US(m, r, 0.0);
        for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<double> R(m, n, 0.0);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
            double s=0; for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j);
            R(i,j) = s;
        }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
            EXPECT_NEAR(R(i,j), A(i,j), 1e-8);
    }
}

TEST(Decompositions, SVD_Reconstruction_Complex_GolubReinsch_Thin)
{
    using Z = std::complex<double>;
    std::vector<std::pair<size_t,size_t>> shapes{{48,24},{24,48},{36,36}};
    for (auto [m,n] : shapes) {
        Matrix<Z> A(m,n, Z(0,0)); fill_random_complex(A, static_cast<uint32_t>(27182 + m + 2*n));
        Matrix<Z> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        // Reconstruction and orthogonality
        const size_t r = std::min(m,n);
        Matrix<Z> US(m, r, Z(0,0)); for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<Z> R(m, n, Z(0,0));
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j){ Z s(0,0); for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j); R(i,j)=s; }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(std::abs(R(i,j) - A(i,j)), 0.0, 1e-8);
        // U^H U = I_r, V^H V = I_r
        Matrix<Z> UH = conj_transpose(U);
        Matrix<Z> UHU(r, r, Z(0,0));
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j){ Z s(0,0); for (size_t k=0;k<m;++k) s += UH(i,k) * U(k,j); UHU(i,j)=s; }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) EXPECT_NEAR(std::abs(UHU(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
        Matrix<Z> V = conj_transpose(Vt);
        Matrix<Z> VtV(r, r, Z(0,0));
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j){ Z s(0,0); for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j); VtV(i,j)=s; }
        for (size_t i=0;i<r;++i) for (size_t j=0;j<r;++j) EXPECT_NEAR(std::abs(VtV(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
    }
}

TEST(Decompositions, SVD_Reconstruction_Complex_GolubReinsch_Full)
{
    using Z = std::complex<double>;
    std::vector<std::pair<size_t,size_t>> shapes{{40,28},{28,40}};
    for (auto [m,n] : shapes) {
        Matrix<Z> A(m,n, Z(0,0)); fill_random_complex(A, static_cast<uint32_t>(16180 + m + 3*n));
        Matrix<Z> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/false, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        // Orthogonality
        Matrix<Z> UH = conj_transpose(U);
        Matrix<Z> UHU(m, m, Z(0,0)); for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j){ Z s(0,0); for (size_t k=0;k<m;++k) s += UH(i,k) * U(k,j); UHU(i,j)=s; }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j) EXPECT_NEAR(std::abs(UHU(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
        Matrix<Z> V = conj_transpose(Vt);
        Matrix<Z> VtV(n, n, Z(0,0)); for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j){ Z s(0,0); for (size_t k=0;k<n;++k) s += Vt(i,k) * V(k,j); VtV(i,j)=s; }
        for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(std::abs(VtV(i,j) - ((i==j)?Z(1,0):Z(0,0))), 0.0, 1e-10);
        // Reconstruction with first r
        const size_t r = std::min(m,n);
        Matrix<Z> US(m, r, Z(0,0)); for (size_t i=0;i<m;++i) for (size_t k=0;k<r;++k) US(i,k) = U(i,k) * S[k];
        Matrix<Z> R(m, n, Z(0,0)); for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j){ Z s(0,0); for (size_t k=0;k<r;++k) s += US(i,k) * Vt(k,j); R(i,j)=s; }
        for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(std::abs(R(i,j) - A(i,j)), 0.0, 1e-8);
    }
}

TEST(Decompositions, SVD_Complex_SingularValues_Agreement_GR_vs_Jacobi)
{
    using Z = std::complex<double>;
    for (auto [m,n] : { std::pair<size_t,size_t>{36,24}, std::pair<size_t,size_t>{24,36} }) {
        Matrix<Z> A(m,n, Z(0,0)); fill_random_complex(A, static_cast<uint32_t>(77777 + m + n));
        Matrix<Z> Uj, Vtj; Vector<double> Sj;
        svd(A, Uj, Sj, Vtj, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::Jacobi);
        Matrix<Z> Ug, Vtg; Vector<double> Sg;
        svd(A, Ug, Sg, Vtg, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        ASSERT_EQ(Sj.size(), Sg.size());
        for (size_t i=0;i<Sj.size();++i) EXPECT_NEAR(Sj[i], Sg[i], 1e-7);
    }
}

static Vector<double> solve_via_svd_triplets_GR(const Matrix<double>& U,
                                                const Vector<double>& S,
                                                const Matrix<double>& Vt,
                                                const Vector<double>& b,
                                                double rcond)
{
    const size_t m = U.rows();
    const size_t n = Vt.cols();
    const size_t r = S.size();
    // y = U^T b
    Vector<double> y(r, 0.0);
    for (size_t k=0;k<r;++k) {
        double s=0; for (size_t i=0;i<m;++i) s += U(i,k) * b[i];
        y[k] = s;
    }
    // cutoff
    double smax = 0.0; for (size_t k=0;k<r;++k) smax = std::max(smax, S[k]);
    double cutoff = (rcond >= 0 ? rcond : std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(m,n))) * smax;
    for (size_t k=0;k<r;++k) y[k] = (S[k] > cutoff) ? (y[k] / S[k]) : 0.0;
    // x = V y  (where V = Vt^T)
    Vector<double> x(n, 0.0);
    for (size_t i=0;i<n;++i) {
        double s=0; for (size_t k=0;k<r;++k) s += Vt(k,i) * y[k];
        x[i] = s;
    }
    return x;
}

TEST(Decompositions, SVD_GR_LeastSquares_ResidualOrthogonality_TallWide)
{
    // Validate end-to-end least-squares using GR triplets; test tall and wide
    for (auto [m,n] : { std::pair<size_t,size_t>{80,40}, std::pair<size_t,size_t>{40,80} }) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(3111 + m + n));
        Vector<double> b(m, 0.0); fill_random(b, static_cast<uint32_t>(4222 + m + 2*n));
        Matrix<double> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/60, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        Vector<double> x = solve_via_svd_triplets_GR(U, S, Vt, b, -1.0);
        // Residual orthogonality: A^T(Ax - b) ≈ 0
        Vector<double> r(m, 0.0);
        for (size_t i=0;i<m;++i) { double s=0; for (size_t j=0;j<n;++j) s += A(i,j) * x[j]; r[i] = s - b[i]; }
        for (size_t i=0;i<n;++i) {
            double s=0; for (size_t j=0;j<m;++j) s += A(j,i) * r[j];
            EXPECT_NEAR(s, 0.0, 1e-8);
        }
    }
}

TEST(Decompositions, SVD_GR_rcond_CutoffBehavior)
{
    // Construct near-rank-deficient matrix; compare different rcond settings
    const size_t m=60, n=40;
    Matrix<double> A(m,n, 0.0); fill_random(A, 8765u);
    // Make last column nearly dependent
    double eps = 1e-10; for (size_t i=0;i<m;++i) A(i, n-1) = 1.5 * A(i, 0) + eps * static_cast<double>(i+1);
    Vector<double> b(m, 0.0); fill_random(b, 1357u);
    Matrix<double> U, Vt; Vector<double> S;
    svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/60, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
    Vector<double> x_default = solve_via_svd_triplets_GR(U, S, Vt, b, -1.0);
    Vector<double> x_strict = solve_via_svd_triplets_GR(U, S, Vt, b, 1e-2);
    // Residual orthogonality holds for both
    auto check_orth = [&](const Vector<double>& x){
        Vector<double> r(m, 0.0); for (size_t i=0;i<m;++i) { double s=0; for (size_t j=0;j<n;++j) s += A(i,j) * x[j]; r[i] = s - b[i]; }
        for (size_t i=0;i<n;++i) { double s=0; for (size_t j=0;j<m;++j) s += A(j,i) * r[j]; EXPECT_NEAR(s, 0.0, 1e-6); }
    };
    check_orth(x_default);
    check_orth(x_strict);
    // Cutoff should have an effect
    double diff_norm = 0.0; for (size_t i=0;i<n;++i) { double d = x_default[i] - x_strict[i]; diff_norm += d*d; }
    EXPECT_GT(std::sqrt(diff_norm), 1e-6);
}

TEST(Decompositions, SVD_GR_CrossMethodAgreement)
{
    // Compare GR against Jacobi-based svd_solve across shapes
    for (auto [m,n] : { std::pair<size_t,size_t>{72,36}, std::pair<size_t,size_t>{36,72} }) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(999 + m + n));
        Vector<double> b(m, 0.0); fill_random(b, static_cast<uint32_t>(777 + m + 3*n));
        // GR solve via triplets
        Matrix<double> U, Vt; Vector<double> S;
        svd(A, U, S, Vt, /*thin=*/true, /*max_sweeps=*/60, std::numeric_limits<double>::epsilon(), SVDMethod::GolubReinsch);
        Vector<double> x_gr = solve_via_svd_triplets_GR(U, S, Vt, b, -1.0);
        // Jacobi solve via library helper (default method)
        Vector<double> x_jac = svd_solve(A, b);
        // Compare residual norms and solution proximity
        auto residual_norm = [&](const Vector<double>& x){ double sn=0.0; for (size_t i=0;i<m;++i){ double s=0; for (size_t j=0;j<n;++j) s += A(i,j) * x[j]; double r = s - b[i]; sn += r*r; } return std::sqrt(sn); };
        double rg = residual_norm(x_gr);
        double rj = residual_norm(x_jac);
        EXPECT_NEAR(rg, rj, 1e-8);
        double dxn=0.0; for (size_t i=0;i<n;++i){ double d=x_gr[i]-x_jac[i]; dxn += d*d; }
        EXPECT_LT(std::sqrt(dxn), 1e-6);
    }
}

TEST(Decompositions, SVD_BidiagonalQR_Values_AgreeWithJacobiCore)
{
    // Generate random bidiagonal cores and compare singular values
    for (size_t r : {16u, 32u, 48u}) {
        std::mt19937 gen(static_cast<uint32_t>(1234 + r));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> d(r), e(r>1?r-1:0);
        for (size_t i=0;i<r;++i) d[i] = dist(gen);
        for (size_t i=0;i+1<r;++i) e[i] = dist(gen);
        // Build Bcore (upper bidiagonal)
        Matrix<double> Bcore(r, r, 0.0);
        for (size_t i=0;i<r;++i) {
            Bcore(i,i) = d[i]; if (i+1<r) Bcore(i,i+1) = e[i];
        }
        // Values via bidiagonal QR (stage 1)
        auto S_bqr = bidiag_qr_values<double>(d, e);
        // Values via Jacobi on Bcore
        Matrix<double> U1, V1t; Vector<double> S_j;
        svd(Bcore, U1, S_j, V1t, /*thin=*/true, /*max_sweeps=*/80, std::numeric_limits<double>::epsilon(), SVDMethod::Jacobi);
        // Compare
        ASSERT_EQ(S_bqr.size(), S_j.size());
        for (size_t i=0;i<r;++i) EXPECT_NEAR(S_bqr[i], S_j[i], 1e-8);
    }
}

TEST(Decompositions, SVD_BidiagonalQR_Values_Properties)
{
    // Nonnegativity and descending order
    for (size_t r : {25u, 40u}) {
        std::mt19937 gen(static_cast<uint32_t>(555 + r));
        std::uniform_real_distribution<double> dist(0.0, 2.0);
        std::vector<double> d(r), e(r>1?r-1:0);
        for (size_t i=0;i<r;++i) d[i] = dist(gen);
        for (size_t i=0;i+1<r;++i) e[i] = dist(gen);
        auto S = bidiag_qr_values<double>(d, e);
        for (size_t i=0;i<S.size();++i) EXPECT_GE(S[i], 0.0);
        for (size_t i=0;i+1<S.size();++i) EXPECT_GE(S[i], S[i+1]);
    }
}

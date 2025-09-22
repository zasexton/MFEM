#include <gtest/gtest.h>

#include <decompositions/qr.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <cmath>
#include <utility>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Side;
using fem::numeric::linear_algebra::Trans;

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
        T s{}; for (size_t j=0;j<A.cols();++j) s += A(i,j) * x[j];
        y[i] = s;
    }
    return y;
}

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

// Random helpers (deterministic via seed)
inline double urand(std::mt19937& gen)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
}

template <typename T>
void fill_random(Matrix<T>& A, uint32_t seed)
{
    std::mt19937 gen(seed);
    for (size_t i=0;i<A.rows();++i)
      for (size_t j=0;j<A.cols();++j) A(i,j) = static_cast<T>(urand(gen));
}

template <typename T>
void fill_random(Vector<T>& v, uint32_t seed)
{
    std::mt19937 gen(seed);
    for (size_t i=0;i<v.size();++i) v[i] = static_cast<T>(urand(gen));
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

}

TEST(Decompositions, QR_Factor_Reconstruct_Tall)
{
    // A is 4x3 (tall), full rank
    Matrix<double> A0 = {{1, 2, 3},
                         {4, 5, 6},
                         {7, 8, 10},
                         {2, 1, 0}};
    Matrix<double> A = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);
    ASSERT_EQ(tau.size(), std::min(A.rows(), A.cols()));

    // Build R (economy) and stack zeros to m x n, then apply Q to recover A0
    Matrix<double> R = form_R(A); // size min(m,n) x n = 3x3
    Matrix<double> B(A0.rows(), A0.cols(), 0.0);
    for (size_t i=0;i<R.rows();++i)
      for (size_t j=0;j<R.cols();++j)
        B(i,j) = R(i,j);

    apply_Q_inplace(Side::Left, Trans::NoTrans, A, tau, B);
    for (size_t i=0;i<A0.rows();++i)
      for (size_t j=0;j<A0.cols();++j)
        EXPECT_NEAR(B(i,j), A0(i,j), 1e-10);
}

TEST(Decompositions, QR_Factor_Reconstruct_Square_Complex)
{
    using Z = std::complex<double>;
    Matrix<Z> A0(3,3, Z(0,0));
    A0(0,0)=Z(1,1); A0(0,1)=Z(2,-1); A0(0,2)=Z(0.5, 0.3);
    A0(1,0)=Z(-1,2); A0(1,1)=Z(0,1); A0(1,2)=Z(3,-2);
    A0(2,0)=Z(2,0); A0(2,1)=Z(-1,1); A0(2,2)=Z(1,4);

    Matrix<Z> A = A0;
    std::vector<Z> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);

    // Reconstruct A via Q * R
    Matrix<Z> R = form_R(A); // 3x3
    Matrix<Z> B = R; // already m==n
    apply_Q_inplace(Side::Left, Trans::NoTrans, A, tau, B);
    for (size_t i=0;i<3;++i)
      for (size_t j=0;j<3;++j)
        EXPECT_NEAR(std::abs(B(i,j) - A0(i,j)), 0.0, 1e-10);
}

TEST(Decompositions, QR_Solve_LeastSquares)
{
    // Tall system m>n, exact solution
    const size_t m=5, n=3;
    Matrix<double> A(m,n, 0.0);
    // Simple structured A to avoid ill-conditioning
    A(0,0)=2; A(0,1)=-1; A(0,2)=0;
    A(1,0)=0; A(1,1)= 3; A(1,2)=1;
    A(2,0)=1; A(2,1)= 1; A(2,2)=4;
    A(3,0)=2; A(3,1)= 0; A(3,2)=1;
    A(4,0)=-1;A(4,1)= 2; A(4,2)=3;

    Vector<double> x_ref = {1.0, -2.0, 0.5};
    Vector<double> b = matvec(A, x_ref);

    Matrix<double> Af = A;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(Af, tau), 0);
    Vector<double> x = qr_solve(Af, tau, b);
    ASSERT_EQ(x.size(), n);
    for (size_t i=0;i<n;++i) EXPECT_NEAR(x[i], x_ref[i], 1e-10);
}

TEST(Decompositions, QR_ApplyQ_LeftOrthogonality)
{
    // Verify Q^H * Q = I by applying reflectors twice
    Matrix<double> A0 = {{1,2,3},
                         {2,0,1},
                         {0,1,4}}; // 3x3
    Matrix<double> A = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);

    // Start with identity and apply Q then Q^H
    Matrix<double> B(3,3, 0.0);
    for (size_t i=0;i<3;++i) B(i,i)=1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, A, tau, B);
    apply_Q_inplace(Side::Left, Trans::ConjTranspose, A, tau, B);
    for (size_t i=0;i<3;++i) for (size_t j=0;j<3;++j)
      EXPECT_NEAR(B(i,j), (i==j)?1.0:0.0, 1e-12);
}

TEST(Decompositions, QR_ApplyQ_RightVectorRoundTrip)
{
    // Right-apply on a vector of length n, then apply inverse (ConjTranspose) and recover
    Matrix<double> A0 = {{1,2,0},
                         {0,3,1},
                         {2,1,1},
                         {1,0,1}}; // 4x3
    Matrix<double> A = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);

    Vector<double> v = {1.0, -1.0, 2.0};
    Vector<double> v0 = v;
    apply_Q_inplace(Side::Right, Trans::NoTrans, A, tau, v);
    apply_Q_inplace(Side::Right, Trans::ConjTranspose, A, tau, v);
    for (size_t i=0;i<v.size();++i) EXPECT_NEAR(v[i], v0[i], 1e-12);
}

TEST(Decompositions, QR_RankDeficient_DetectInSolve)
{
    // Columns dependent -> rank deficiency -> R(ii) == 0 for some i
    Matrix<double> A = {{1,2},
                        {2,4},
                        {-1,-2}}; // col2 = 2*col1
    Matrix<double> Af = A;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(Af, tau), 0);

    Matrix<double> R = form_R(Af);
    // R(1,1) ~ 0
    EXPECT_NEAR(R(1,1), 0.0, 1e-10);

    // qr_solve should throw on zero diagonal
    Vector<double> b = {0.0, 0.0, 0.0};
    EXPECT_THROW({
        auto x = qr_solve(Af, tau, b);
        (void)x;
    }, std::runtime_error);
}

TEST(Decompositions, QR_Degenerate_Small)
{
    // 1x1
    Matrix<double> A = {{3.5}};
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(A, tau), 0);
    ASSERT_EQ(tau.size(), 1u);
    Matrix<double> R = form_R(A);
    Matrix<double> B(1,1, 0.0); B(0,0) = R(0,0);
    apply_Q_inplace(Side::Left, Trans::NoTrans, A, tau, B);
    EXPECT_NEAR(B(0,0), 3.5, 1e-12);
}

TEST(Decompositions, QR_Random_Reconstruction_SmallSweep)
{
    // Sweep a few seeds and shapes; check Q(:,1:r)*R ≈ A
    std::vector<std::pair<size_t,size_t>> shapes{{2,2},{3,2},{4,3},{5,5}};
    std::vector<uint32_t> seeds{1u, 7u, 13u};
    for (auto [m,n] : shapes) {
        for (auto s : seeds) {
            Matrix<double> A0(m,n, 0.0); fill_random(A0, s);
            Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
            Matrix<double> Q(m,m, 0.0); for (size_t i=0;i<m;++i) Q(i,i)=1.0;
            apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
            Matrix<double> R = form_R(Af);
            const size_t r = std::min(m,n);
            Matrix<double> QR(m,n, 0.0);
            for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
                double ss=0; for (size_t k=0;k<r;++k) ss += Q(i,k) * R(k,j);
                QR(i,j) = ss;
            }
            for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
                EXPECT_NEAR(QR(i,j), A0(i,j), 1e-10);
        }
    }
}

TEST(Decompositions, QR_Random_Reconstruction_Complex_SmallSweep)
{
    using Z = std::complex<double>;
    std::vector<std::pair<size_t,size_t>> shapes{{4,2},{3,3}};
    std::vector<uint32_t> seeds{11u, 23u};
    for (auto [m,n] : shapes) {
        for (auto s : seeds) {
            Matrix<Z> A0(m,n, Z(0,0)); fill_random_complex(A0, s);
            Matrix<Z> Af = A0; std::vector<Z> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
            Matrix<Z> Q(m,m, Z(0,0)); for (size_t i=0;i<m;++i) Q(i,i)=Z(1,0);
            apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
            Matrix<Z> R = form_R(Af);
            const size_t r = std::min(m,n);
            Matrix<Z> QR(m,n, Z(0,0));
            for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
                Z ss(0,0); for (size_t k=0;k<r;++k) ss += Q(i,k) * R(k,j);
                QR(i,j) = ss;
            }
            for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
                EXPECT_NEAR(std::abs(QR(i,j) - A0(i,j)), 0.0, 1e-10);
        }
    }
}

TEST(Decompositions, QR_Random_LeastSquares_Residuals)
{
    // Random A (tall), random b; check residual orthogonality
    std::vector<std::pair<size_t,size_t>> shapes{{6,3},{7,4}};
    for (auto [m,n] : shapes) {
        Matrix<double> A(m,n, 0.0); fill_random(A, static_cast<uint32_t>(101u + m + n));
        Vector<double> b(m, 0.0); fill_random(b, static_cast<uint32_t>(211u + m*3 + n));
        Matrix<double> Af = A; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
        Vector<double> x = qr_solve(Af, tau, b);
        Vector<double> r = matvec(A, x);
        for (size_t i=0;i<m;++i) r[i] -= b[i];
        // A^T r ≈ 0
        Matrix<double> At = conj_transpose(A);
        for (size_t i=0;i<n;++i) {
            double s=0; for (size_t j=0;j<m;++j) s += At(i,j) * r[j];
            EXPECT_NEAR(s, 0.0, 1e-8);
        }
    }
}

TEST(Decompositions, QR_ApplyQ_RightOnMatrix_RoundTrip)
{
    const size_t m=5, n=3, p=4; // B is p x n for right-apply
    Matrix<double> A0(m,n, 0.0); fill_random(A0, 77u);
    Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    Matrix<double> B0(p,n, 0.0); fill_random(B0, 99u);
    Matrix<double> B = B0;
    apply_Q_inplace(Side::Right, Trans::NoTrans, Af, tau, B);
    apply_Q_inplace(Side::Right, Trans::ConjTranspose, Af, tau, B);
    for (size_t i=0;i<p;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(B(i,j), B0(i,j), 1e-12);
}

TEST(Decompositions, QR_ZeroMatrix_EdgeCases)
{
    const size_t m=4, n=3;
    Matrix<double> A0(m,n, 0.0);
    Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    // All tau should be 0 or undefined; reconstruct Q·R should be zero
    Matrix<double> Q(m,m, 0.0); for (size_t i=0;i<m;++i) Q(i,i)=1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
    Matrix<double> R = form_R(Af);
    Matrix<double> QR = matmul(Q, R);
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) EXPECT_NEAR(QR(i,j), 0.0, 1e-12);
    // Solve should throw (rank deficient)
    Vector<double> b(m, 0.0);
    EXPECT_THROW((void)qr_solve(Af, tau, b), std::runtime_error);
}

TEST(Decompositions, QR_NearRankDeficient_Stability)
{
    // Make last column nearly a multiple of the first
    const size_t m=6, n=3;
    Matrix<double> A(m,n, 0.0); fill_random(A, 123u);
    double eps = 1e-10;
    for (size_t i=0;i<m;++i) A(i, n-1) = 2.0 * A(i, 0) + eps * static_cast<double>(i+1);
    Matrix<double> Af = A; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    Matrix<double> R = form_R(Af);
    EXPECT_LE(std::abs(R(n-1,n-1)), 1e-6); // small diagonal element indicates near rank deficiency
    // Least squares residual orthogonality still holds to a modest tolerance
    Vector<double> b(m, 0.0); fill_random(b, 321u);
    Vector<double> x = qr_solve(Af, tau, b);
    Vector<double> r = matvec(A, x);
    for (size_t i=0;i<m;++i) r[i] -= b[i];
    Matrix<double> At = conj_transpose(A);
    for (size_t i=0;i<n;++i) {
        double s=0; for (size_t j=0;j<m;++j) s += At(i,j) * r[j];
        EXPECT_NEAR(s, 0.0, 1e-6);
    }
}

TEST(Decompositions, QR_SingleColumn_Vector)
{
    const size_t m=5, n=1;
    Matrix<double> A0(m,n, 0.0); fill_random(A0, 808u);
    Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    Matrix<double> R = form_R(Af);
    Matrix<double> B(m,n, 0.0); for (size_t i=0;i<R.rows();++i) B(i,0)=R(i,0);
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, B);
    for (size_t i=0;i<m;++i) EXPECT_NEAR(B(i,0), A0(i,0), 1e-10);
}

TEST(Decompositions, QR_SingleRow_WideEdge)
{
    const size_t m=1, n=4;
    Matrix<double> A0(m,n, 0.0); fill_random(A0, 909u);
    Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    Matrix<double> Q(m,m, 0.0); Q(0,0)=1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
    Matrix<double> R = form_R(Af);
    Matrix<double> QR = matmul(Q, R);
    for (size_t j=0;j<n;++j) EXPECT_NEAR(QR(0,j), A0(0,j), 1e-12);
}

TEST(Decompositions, QR_FormExplicitQ_ReconstructTall)
{
    // Build explicit Q from reflectors and verify Q * R = A0 (tall m>n)
    const size_t m=6, n=3;
    Matrix<double> A0(m,n, 0.0);
    // Deterministic fill
    double v=1.0;
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A0(i,j) = v; v += 0.5; }

    Matrix<double> Af = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(Af, tau), 0);

    // Form explicit Q by applying reflectors to I_m
    Matrix<double> Q(m,m, 0.0); for (size_t i=0;i<m;++i) Q(i,i) = 1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);

    Matrix<double> R = form_R(Af); // r x n, r=min(m,n)=n
    // Compute Q(:,1:r) * R (m x r times r x n -> m x n)
    Matrix<double> QR(m,n, 0.0);
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
        double s=0; for (size_t k=0;k<n;++k) s += Q(i,k) * R(k,j);
        QR(i,j) = s;
    }
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
        EXPECT_NEAR(QR(i,j), A0(i,j), 1e-10);
}

TEST(Decompositions, QR_Wide_Reconstruct)
{
    // m<n: Q is m x m, R is m x n; Check Q * R == A0
    const size_t m=3, n=5;
    Matrix<double> A0(m,n, 0.0);
    double v=0.3;
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A0(i,j) = std::sin(v); v += 0.7; }

    Matrix<double> Af = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(Af, tau), 0);

    Matrix<double> Q(m,m, 0.0); for (size_t i=0;i<m;++i) Q(i,i) = 1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
    Matrix<double> R = form_R(Af); // m x n here
    Matrix<double> QR = matmul(Q, R);
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j)
        EXPECT_NEAR(QR(i,j), A0(i,j), 1e-10);
}

TEST(Decompositions, QR_Orthogonality_MultipleSizes)
{
    // Test Q^T Q ≈ I for varying shapes
    for (auto [m,n] : { std::pair<size_t,size_t>{4,2}, std::pair<size_t,size_t>{5,3}, std::pair<size_t,size_t>{6,6} }) {
        Matrix<double> A0(m,n, 0.0);
        double v=0.1; for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A0(i,j) = std::cos(v); v += 0.31; }
        Matrix<double> Af = A0; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
        Matrix<double> Q(m,m, 0.0); for (size_t i=0;i<m;++i) Q(i,i)=1.0;
        apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
        // Compute QtQ = Q^T * Q
        Matrix<double> Qt = conj_transpose(Q);
        Matrix<double> QtQ = matmul(Qt, Q);
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j)
            EXPECT_NEAR(QtQ(i,j), (i==j)?1.0:0.0, 1e-10);
    }
}

TEST(Decompositions, QR_LeastSquares_ResidualOrthogonality)
{
    // Randomish tall A, random b; check A^T r ≈ 0 for r = Ax - b
    const size_t m=7, n=3;
    Matrix<double> A(m,n, 0.0);
    double v=0.2; for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A(i,j) = std::sin(v) + 0.1*std::cos(2*v); v += 0.37; }
    Vector<double> b(m, 0.0); v = -0.5; for (size_t i=0;i<m;++i) { b[i] = std::cos(v) + 0.05*std::sin(3*v); v += 0.29; }

    Matrix<double> Af = A; std::vector<double> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    Vector<double> x = qr_solve(Af, tau, b);
    // r = A x - b
    Vector<double> Ax = matvec(A, x);
    Vector<double> r(m, 0.0); for (size_t i=0;i<m;++i) r[i] = Ax[i] - b[i];
    // Check A^T r ≈ 0
    Matrix<double> At = conj_transpose(A);
    Vector<double> Atr(n, 0.0);
    for (size_t i=0;i<n;++i) {
        double s=0; for (size_t j=0;j<m;++j) s += At(i,j) * r[j];
        Atr[i] = s;
    }
    for (size_t i=0;i<n;++i) EXPECT_NEAR(Atr[i], 0.0, 1e-8);
}

TEST(Decompositions, QR_ComplexLeastSquares_ResidualOrthogonality)
{
    using Z = std::complex<double>;
    const size_t m=5, n=3;
    Matrix<Z> A(m,n, Z(0,0));
    double v=0.4; for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) { A(i,j) = Z(std::cos(v), std::sin(2*v)); v += 0.5; }
    Vector<Z> b(m, Z(0,0)); v = -0.3; for (size_t i=0;i<m;++i) { b[i] = Z(std::sin(v), std::cos(3*v)); v += 0.7; }

    Matrix<Z> Af = A; std::vector<Z> tau; ASSERT_EQ(qr_factor(Af, tau), 0);
    // Solve
    Vector<Z> x = qr_solve(Af, tau, b);
    // Residual orthogonality: A^H (A x - b) ≈ 0
    Vector<Z> Ax = matvec(A, x);
    Vector<Z> r(m, Z(0,0)); for (size_t i=0;i<m;++i) r[i] = Ax[i] - b[i];
    Matrix<Z> AH = conj_transpose(A);
    Vector<Z> AHr(n, Z(0,0));
    for (size_t i=0;i<n;++i) {
        Z s(0,0); for (size_t j=0;j<m;++j) s += AH(i,j) * r[j]; AHr[i] = s;
    }
    for (size_t i=0;i<n;++i) EXPECT_NEAR(std::abs(AHr[i]), 0.0, 1e-8);
}

TEST(Decompositions, QR_TauZeroPath_ZeroColumn)
{
    // Construct A where a column segment is zero so tau[j]==0 branch is exercised
    Matrix<double> A0 = {{1, 0, 2},
                         {0, 0, 0},
                         {0, 0, 3},
                         {0, 0, 0}}; // Column 1 is identically zero
    Matrix<double> Af = A0;
    std::vector<double> tau;
    ASSERT_EQ(qr_factor(Af, tau), 0);

    // Reconstruct via explicit Q and R
    Matrix<double> Q(Af.rows(), Af.rows(), 0.0); for (size_t i=0;i<Q.rows();++i) Q(i,i)=1.0;
    apply_Q_inplace(Side::Left, Trans::NoTrans, Af, tau, Q);
    Matrix<double> R = form_R(Af);
    Matrix<double> QR = matmul(Q, R);
    for (size_t i=0;i<A0.rows();++i) for (size_t j=0;j<A0.cols();++j)
        EXPECT_NEAR(QR(i,j), A0(i,j), 1e-10);
}

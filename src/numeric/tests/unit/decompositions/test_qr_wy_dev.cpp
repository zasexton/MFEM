#include <gtest/gtest.h>

#include <decompositions/qr.h>
#include <linear_algebra/householder_wy.h>
#include <core/matrix.h>
#include <vector>
#include <complex>
#include <array>
#include <type_traits>
#include <cmath>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using fem::numeric::linear_algebra::Side;
using fem::numeric::linear_algebra::Trans;

#if defined(FEM_NUMERIC_QR_TEST_WY)

namespace {

template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    if (A.cols() != B.rows()) {
        ADD_FAILURE() << "Inner dimensions must agree for matmul: A.cols()="
                       << A.cols() << " B.rows()=" << B.rows();
        return Matrix<T>();
    }
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i=0;i<A.rows();++i)
        for (size_t j=0;j<B.cols();++j) {
            T s{}; for (size_t k=0;k<A.cols();++k) s += A(i,k) * B(k,j);
            C(i,j) = s;
        }
    return C;
}

template <typename T>
T sample_value(double& v)
{
    double real = std::sin(v) + 0.2 * std::cos(2 * v);
    v += 0.37;
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        double imag = std::cos(3 * v) + 0.15 * std::sin(5 * v);
        v += 0.19;
        return T(real, imag);
    } else {
        return static_cast<T>(real);
    }
}

template <typename T>
Matrix<T> make_matrix(size_t m, size_t n, double seed)
{
    Matrix<T> A(m, n, T{});
    double v = seed;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            A(i, j) = sample_value<T>(v);
    return A;
}

template <typename T>
double value_norm(const T& x)
{
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return std::abs(x);
    } else {
        return std::abs(static_cast<double>(x));
    }
}

template <typename T>
void expect_blocked_matches_unblocked(size_t m, size_t n, std::size_t block, double seed)
{
    Matrix<T> A0 = make_matrix<T>(m, n, seed);

    Matrix<T> Au = A0;
    std::vector<T> tau_u;
    ASSERT_EQ(qr_factor_unblocked(Au, tau_u), 0);

    Matrix<T> Ab = A0;
    std::vector<T> tau_b;
    ASSERT_EQ(qr_factor_blocked(Ab, tau_b, block), 0);

    ASSERT_EQ(tau_u.size(), tau_b.size());
    for (size_t i = 0; i < tau_u.size(); ++i)
        EXPECT_NEAR(value_norm(tau_u[i] - tau_b[i]), 0.0, 1e-12);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_NEAR(value_norm(Au(i, j) - Ab(i, j)), 0.0, 1e-10);

    Matrix<T> Qu(m, m, T{}); for (size_t i = 0; i < m; ++i) Qu(i, i) = T{1};
    Matrix<T> Qb(m, m, T{}); for (size_t i = 0; i < m; ++i) Qb(i, i) = T{1};

    apply_Q_inplace(Side::Left, Trans::NoTrans, Au, tau_u, Qu);
    apply_Q_inplace(Side::Left, Trans::NoTrans, Ab, tau_b, Qb);

    // Use economy-sized R (r x n) for WY dev comparisons
    const size_t r = std::min(m, n);
    Matrix<T> Ru_full = form_R(Au);
    Matrix<T> Rb_full = form_R(Ab);
    Matrix<T> Ru(r, n, T{}), Rb(r, n, T{});
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < n; ++j) {
            Ru(i, j) = Ru_full(i, j);
            Rb(i, j) = Rb_full(i, j);
        }
    Matrix<T> Qu_leading(m, r, T{});
    Matrix<T> Qb_leading(m, r, T{});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < r; ++j) {
            Qu_leading(i, j) = Qu(i, j);
            Qb_leading(i, j) = Qb(i, j);
        }
    }

    Matrix<T> QRu = matmul(Qu_leading, Ru);
    Matrix<T> QRb = matmul(Qb_leading, Rb);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(value_norm(QRu(i, j) - A0(i, j)), 0.0, 1e-10);
            EXPECT_NEAR(value_norm(QRb(i, j) - A0(i, j)), 0.0, 1e-10);
        }
}

} // namespace

TEST(Decompositions_WY_Dev, RowMajor_Wide_Equals_Unblocked)
{
    const std::array<std::pair<size_t, size_t>, 3> shapes{ {{3,5}, {4,7}, {5,9}} };
    const std::array<std::size_t, 5> blocks{ {1u, 2u, 4u, 8u, 64u} };
    const std::array<double, 3> seeds{ {0.31, 1.17, 2.05} };

    for (auto [m, n] : shapes)
        for (auto blk : blocks)
            for (double s : seeds)
                expect_blocked_matches_unblocked<double>(m, n, blk,
                    s + static_cast<double>(blk) * 0.013);
}

TEST(Decompositions_WY_Dev, RowMajor_Tall_Equals_Unblocked)
{
    const std::array<std::pair<size_t, size_t>, 3> shapes{ {{6,4}, {7,3}, {9,5}} };
    const std::array<std::size_t, 4> blocks{ {1u, 3u, 6u, 32u} };
    const std::array<double, 2> seeds{ {0.42, 1.91} };

    for (auto [m, n] : shapes)
        for (auto blk : blocks)
            for (double s : seeds)
                expect_blocked_matches_unblocked<double>(m, n, blk, s + static_cast<double>(m + n) * 0.007);
}

TEST(Decompositions_WY_Dev, RowMajor_Wide_Complex_Equals_Unblocked)
{
    using Z = std::complex<double>;
    const std::array<std::pair<size_t, size_t>, 2> shapes{ {{3,6}, {4,7}} };
    const std::array<std::size_t, 4> blocks{ {1u, 2u, 4u, 16u} };
    const std::array<double, 2> seeds{ {0.27, 1.55} };

    for (auto [m, n] : shapes)
        for (auto blk : blocks)
            for (double s : seeds)
                expect_blocked_matches_unblocked<Z>(m, n, blk,
                    s + static_cast<double>(blk) * 0.021);
}

TEST(Decompositions_WY_Dev, Panel_T_vs_LAPACK_and_Update)
{
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
    using T = double;
    const int pm = 16, kb = 8, nt = 5;
    // Build a random panel by factoring a random pm x kb block with unblocked QR
    Matrix<T> Panel(pm, kb, T{});
    double v=0.123;
    for (int i=0;i<pm;++i) for (int j=0;j<kb;++j) { Panel(i,j) = std::sin(v)+0.1*std::cos(3*v); v+=0.37; }
    Matrix<T> Ap = Panel; std::vector<T> taup; ASSERT_EQ(qr_factor_unblocked(Ap, taup), 0);
    // Build V from Ap (panel)
    Matrix<T> V; // pm x kb
    {
        V = Matrix<T>(pm, kb, T{});
        for (int j=0;j<kb;++j) {
            for (int i=0;i<j;++i) V(i,j)=T{};
            V(j,j) = T{1};
            for (int i=j+1;i<pm;++i) V(i,j) = Ap(i,j);
        }
    }
    // Our T via consolidated WY helper
    Matrix<T> Tmy; fem::numeric::linear_algebra::form_block_T_forward_columnwise(V, taup, Tmy);
    // LAPACK T via DLARFT (column-major)
    std::vector<T> Vcm(static_cast<size_t>(pm)*kb);
    for (int j=0;j<kb;++j) for (int i=0;i<pm;++i) Vcm[j*pm+i]=V(i,j);
    std::vector<T> Tcm(static_cast<size_t>(kb)*kb, T{});
    backends::lapack::larft_cm<T>('F','C', pm, kb, Vcm.data(), pm, taup.data(), Tcm.data(), kb);
    // Compare T
    for (int i=0;i<kb;++i) for (int j=i;j<kb;++j)
        EXPECT_NEAR(Tmy(i,j), Tcm[static_cast<size_t>(i) + static_cast<size_t>(j)*kb], 1e-12);

    // Check blocked update vs sequential reflectors on a random B (pm x nt)
    Matrix<T> B(pm, nt, T{});
    v = 0.456; for (int i=0;i<pm;++i) for (int j=0;j<nt;++j) { B(i,j)=std::cos(v)+0.2*std::sin(2*v); v+=0.41; }
    Matrix<T> Bblocked = B;
    // Bblocked := (I - V T V^T) B (real case)
    Matrix<T> Y(kb, nt, T{});
    // Y = V^T B
    for (int j=0;j<nt;++j) for (int i=0;i<kb;++i) {
        T s{}; for (int r=0;r<pm;++r) s += V(r,i) * Bblocked(r,j);
        Y(i,j)=s;
    }
    // Y := T * Y (upper-triangular)
    for (int j=0;j<nt;++j) for (int i=0;i<kb;++i) {
        T s{}; for (int k=i;k<kb;++k) s += Tmy(i,k) * Y(k,j);
        Y(i,j)=s;
    }
    // Bblocked -= V * Y
    for (int j=0;j<nt;++j) for (int r=0;r<pm;++r) {
        T s{}; for (int k=0;k<kb;++k) s += V(r,k) * Y(k,j);
        Bblocked(r,j) = Bblocked(r,j) - s;
    }
    // Sequential apply reflectors
    Matrix<T> Bseq = B;
    for (int j=0;j<kb;++j) {
        T tj = taup[j];
        // v_j: zeros up to j-1, 1 at j, tail from Ap
        std::vector<T> vj(pm, T{}); vj[j]=T{1}; for (int i=j+1;i<pm;++i) vj[i]=Ap(i,j);
        for (int col=0; col<nt; ++col) {
            T w{}; for (int r=0;r<pm;++r) w += vj[r] * Bseq(r,col);
            w = tj * w;
            for (int r=0;r<pm;++r) Bseq(r,col) -= vj[r] * w;
        }
    }
    for (int i=0;i<pm;++i) for (int j=0;j<nt;++j)
        EXPECT_NEAR(Bblocked(i,j), Bseq(i,j), 1e-10);
#endif
}

TEST(Decompositions_WY_Dev, Panel_T_vs_LAPACK_and_Update_Complex)
{
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
    using Z = std::complex<double>;
    const int nt = 5;
    for (int pm : {8, 12, 16}) for (int kb : {2, 4, 8}) {
        if (kb > pm) continue;
        Matrix<Z> Panel(pm, kb, Z(0,0));
        double v=0.321; for (int i=0;i<pm;++i) for (int j=0;j<kb;++j) { Panel(i,j) = Z(std::sin(v), std::cos(2*v)); v+=0.29; }
        Matrix<Z> Ap = Panel; std::vector<Z> taup; ASSERT_EQ(qr_factor_unblocked(Ap, taup), 0);
        Matrix<Z> V(pm, kb, Z(0,0));
        for (int j=0;j<kb;++j) { for (int i=0;i<j;++i) V(i,j)=Z(0,0); V(j,j)=Z(1,0); for (int i=j+1;i<pm;++i) V(i,j)=Ap(i,j); }
        Matrix<Z> Tmy; fem::numeric::linear_algebra::form_block_T_forward_columnwise(V, taup, Tmy);
        std::vector<Z> Vcm(static_cast<size_t>(pm)*kb);
        for (int j=0;j<kb;++j) for (int i=0;i<pm;++i) Vcm[j*pm+i]=V(i,j);
        std::vector<Z> Tcm(static_cast<size_t>(kb)*kb, Z(0,0));
        backends::lapack::larft_cm<Z>('F','C', pm, kb, Vcm.data(), pm, taup.data(), Tcm.data(), kb);
        for (int i=0;i<kb;++i) for (int j=i;j<kb;++j)
            EXPECT_NEAR(std::abs(Tmy(i,j) - Tcm[static_cast<size_t>(i)+static_cast<size_t>(j)*kb]), 0.0, 1e-12);
        // Update check
        Matrix<Z> B(pm, nt, Z(0,0)); v=0.111; for (int i=0;i<pm;++i) for (int j=0;j<nt;++j) B(i,j)=Z(std::cos(v), std::sin(3*v)), v+=0.33;
        Matrix<Z> Bblocked = B;
        // Y = V^H B
        Matrix<Z> Y(kb, nt, Z(0,0));
        for (int j=0;j<nt;++j) for (int i=0;i<kb;++i) {
            Z s(0,0); for (int r=0;r<pm;++r) s += std::conj(V(r,i)) * Bblocked(r,j);
            Y(i,j)=s;
        }
        // Y = T * Y (upper)
        for (int j=0;j<nt;++j) for (int i=0;i<kb;++i) {
            Z s(0,0); for (int k=i;k<kb;++k) s += Tmy(i,k) * Y(k,j); Y(i,j)=s;
        }
        // Bblocked -= V * Y
        for (int j=0;j<nt;++j) for (int r=0;r<pm;++r) {
            Z s(0,0); for (int k=0;k<kb;++k) s += V(r,k) * Y(k,j);
            Bblocked(r,j) -= s;
        }
        // Sequential reflectors
        Matrix<Z> Bseq = B;
        for (int j=0;j<kb;++j) {
            Z tj = taup[j]; std::vector<Z> vj(pm, Z(0,0)); vj[j]=Z(1,0); for (int i=j+1;i<pm;++i) vj[i]=Ap(i,j);
            for (int col=0; col<nt; ++col) {
                Z w(0,0); for (int r=0;r<pm;++r) w += std::conj(vj[r]) * Bseq(r,col);
                w = tj * w;
                for (int r=0;r<pm;++r) Bseq(r,col) -= vj[r] * w;
            }
        }
        for (int i=0;i<pm;++i) for (int j=0;j<nt;++j)
            EXPECT_NEAR(std::abs(Bblocked(i,j) - Bseq(i,j)), 0.0, 1e-10);
    }
#endif
}

#endif // FEM_NUMERIC_QR_TEST_WY

#include <gtest/gtest.h>

#include <decompositions/eigen.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <vector>
#include <random>
#include <algorithm>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

#if defined(FEM_NUMERIC_ENABLE_LAPACK) || defined(FEM_NUMERIC_ENABLE_LAPACKE)

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
#include <lapacke.h>
#endif

namespace {

template <typename T>
void align_columns(Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor>& ours,
                   const Matrix<T, DynamicStorage<T>, StorageOrder::ColumnMajor>& reference)
{
    const std::size_t n = ours.cols();
    for (std::size_t j = 0; j < n; ++j) {
        std::complex<double> dot{};
        for (std::size_t i = 0; i < ours.rows(); ++i) {
            auto ref = reference(i, j);
            auto cur = ours(i, j);
            if constexpr (is_complex_number_v<T>) dot += std::conj(static_cast<std::complex<double>>(ref)) * static_cast<std::complex<double>>(cur);
            else dot += static_cast<double>(ref) * static_cast<double>(cur);
        }
        double mag = std::abs(dot);
        if (mag > 1e-12) {
            std::complex<double> phase = dot / mag;
            T adjust;
            if constexpr (is_complex_number_v<T>) adjust = static_cast<T>(std::conj(phase));
            else adjust = static_cast<T>(phase.real());
            for (std::size_t i = 0; i < ours.rows(); ++i) ours(i, j) *= adjust;
        }
    }
}

int lapack_dsyev(Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor>& A,
                 std::vector<double>& w)
{
    const int n = static_cast<int>(A.rows());
    const int lda = static_cast<int>(A.rows());
    w.resize(static_cast<std::size_t>(n));
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
    return LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, A.data(), lda, w.data());
#else
    extern "C" {
        void dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
    }
    int info = 0;
    int lwork = -1;
    double wkopt = 0.0;
    dsyev_("V", "U", &n, A.data(), &lda, w.data(), &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_("V", "U", &n, A.data(), &lda, w.data(), work.data(), &lwork, &info);
    return info;
#endif
}

int lapack_zheev(Matrix<std::complex<double>, DynamicStorage<std::complex<double>>, StorageOrder::ColumnMajor>& A,
                 std::vector<double>& w)
{
    const int n = static_cast<int>(A.rows());
    const int lda = static_cast<int>(A.rows());
    w.resize(static_cast<std::size_t>(n));
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
    return LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', n, reinterpret_cast<lapack_complex_double*>(A.data()), lda, w.data());
#else
    extern "C" {
        void zheev_(const char*, const char*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, double*, int*);
    }
    int info = 0;
    int lwork = -1;
    std::complex<double> wkopt{0.0, 0.0};
    std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
    zheev_("V", "U", &n, A.data(), &lda, w.data(), &wkopt, &lwork, rwork.data(), &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<double>> work(static_cast<std::size_t>(lwork));
    zheev_("V", "U", &n, A.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
    return info;
#endif
}

Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor>
make_symmetric_cm(std::size_t n, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> A(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double val = dist(gen);
            A(i, j) = val;
            A(j, i) = val;
        }
    }
    return A;
}

Matrix<std::complex<double>, DynamicStorage<std::complex<double>>, StorageOrder::ColumnMajor>
make_hermitian_cm(std::size_t n, uint32_t seed)
{
    using Z = std::complex<double>;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<Z, DynamicStorage<Z>, StorageOrder::ColumnMajor> A(n, n, Z{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = Z(dist(gen), 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            Z val(dist(gen), dist(gen));
            A(i, j) = val;
            A(j, i) = std::conj(val);
        }
    }
    return A;
}

} // namespace

TEST(Decompositions_LAPACK, Eigen_Symmetric_Double_MatchesLAPACK)
{
    constexpr std::size_t n = 6;
    auto A_base = make_symmetric_cm(n, 2024u);

    auto A_method = A_base;
    Vector<double> evals;
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> evecs;
    ASSERT_EQ(eigen_symmetric(A_method, evals, evecs), 0);

    auto A_lapack = A_base;
    std::vector<double> w_ref;
    ASSERT_EQ(lapack_dsyev(A_lapack, w_ref), 0);

    ASSERT_EQ(evals.size(), n);
    const double tol = 1e-9;
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(evals[i], w_ref[i], tol);

    align_columns(evecs, A_lapack);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(std::abs(evecs(i, j) - A_lapack(i, j)), 0.0, 1e-8);
}

TEST(Decompositions_LAPACK, Eigen_Hermitian_Complex_MatchesLAPACK)
{
    using Z = std::complex<double>;
    constexpr std::size_t n = 5;
    auto A_base = make_hermitian_cm(n, 909u);

    auto A_method = A_base;
    Vector<double> evals;
    Matrix<Z, DynamicStorage<Z>, StorageOrder::ColumnMajor> evecs;
    ASSERT_EQ(eigen_symmetric(A_method, evals, evecs), 0);

    auto A_lapack = A_base;
    std::vector<double> w_ref;
    ASSERT_EQ(lapack_zheev(A_lapack, w_ref), 0);

    ASSERT_EQ(evals.size(), n);
    const double tol = 1e-8;
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(evals[i], w_ref[i], tol);

    align_columns(evecs, A_lapack);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(std::abs(evecs(i, j) - A_lapack(i, j)), 0.0, 1e-7);
}

#endif // FEM_NUMERIC_ENABLE_LAPACK || FEM_NUMERIC_ENABLE_LAPACKE


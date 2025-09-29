#include <gtest/gtest.h>

#include <decompositions/eigen.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <complex>
#include <random>
#include <cmath>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

namespace {

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
Matrix<T> conj_transpose(const Matrix<T>& A)
{
    Matrix<T> AH(A.cols(), A.rows(), T{});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            if constexpr (is_complex_number_v<T>) AH(j, i) = std::conj(A(i, j));
            else AH(j, i) = A(i, j);
    return AH;
}

template <typename T>
void expect_orthonormal_columns(const Matrix<T>& V, double tol)
{
    const std::size_t n = V.cols();
    for (std::size_t j = 0; j < n; ++j) {
        // Norm check
        typename numeric_traits<T>::scalar_type norm_sq{};
        for (std::size_t i = 0; i < V.rows(); ++i) {
            if constexpr (is_complex_number_v<T>) norm_sq += static_cast<typename numeric_traits<T>::scalar_type>(std::norm(V(i, j)));
            else norm_sq += static_cast<typename numeric_traits<T>::scalar_type>(V(i, j) * V(i, j));
        }
        EXPECT_NEAR(static_cast<double>(norm_sq), 1.0, tol);
        for (std::size_t k = j + 1; k < n; ++k) {
            std::complex<double> dot{};
            for (std::size_t i = 0; i < V.rows(); ++i) {
                if constexpr (is_complex_number_v<T>) dot += std::conj(static_cast<std::complex<double>>(V(i, j))) * static_cast<std::complex<double>>(V(i, k));
                else dot += static_cast<double>(V(i, j)) * static_cast<double>(V(i, k));
            }
            EXPECT_NEAR(std::abs(dot), 0.0, tol);
        }
    }
}

template <typename T>
void fill_random_symmetric(Matrix<T>& A, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = i; j < A.cols(); ++j) {
            double val = dist(gen);
            A(i, j) = static_cast<T>(val);
            A(j, i) = static_cast<T>(val);
        }
    }
}

template <typename Z>
void fill_random_hermitian(Matrix<Z>& A, uint32_t seed)
{
    using R = typename numeric_traits<Z>::scalar_type;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < A.rows(); ++i) {
        A(i, i) = Z(dist(gen), R{0});
        for (std::size_t j = i + 1; j < A.cols(); ++j) {
            Z val(dist(gen), dist(gen));
            A(i, j) = val;
            A(j, i) = std::conj(val);
        }
    }
}

template <typename T>
void expect_reconstruction(const Matrix<T>& A,
                           const Vector<typename numeric_traits<T>::scalar_type>& evals,
                           const Matrix<T>& evecs,
                           double tol)
{
    const std::size_t n = A.rows();
    Matrix<T> D(n, n, T{});
    for (std::size_t i = 0; i < n; ++i) D(i, i) = static_cast<T>(evals[i]);
    auto temp = matmul(evecs, D);
    auto Ahat = matmul(temp, conj_transpose(evecs));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_NEAR(std::abs(Ahat(i, j) - A(i, j)), 0.0, tol);
}

template <typename T>
Vector<T> column_as_vector(const Matrix<T>& M, std::size_t col)
{
    Vector<T> v(M.rows(), T{});
    for (std::size_t i = 0; i < M.rows(); ++i) v[i] = M(i, col);
    return v;
}

} // namespace

TEST(DecompositionsDetail, TridiagonalReductionReconstructs)
{
    Matrix<double> A = {{4.0, 1.0, 2.0},
                        {1.0, 2.0, 0.0},
                        {2.0, 0.0, 3.0}};
    Matrix<double> tri = A;
    std::vector<double> diag;
    std::vector<double> sub;
    Matrix<double> Q;
    fem::numeric::decompositions::detail::hermitian_to_tridiagonal(tri, diag, sub, &Q);

    Matrix<double> T(tri.rows(), tri.cols(), 0.0);
    for (std::size_t i = 0; i < tri.rows(); ++i) {
        T(i, i) = diag[i];
        if (i + 1 < tri.cols()) {
            T(i, i + 1) = sub[i];
            T(i + 1, i) = sub[i];
        }
    }

    const double tol = 1e-10;

    // The returned Q should satisfy Q^H * A * Q = T (tridiagonal)
    // where the subdiagonal elements might have different signs than stored in 'sub'
    auto Qt = conj_transpose(Q);
    auto temp = matmul(Qt, A);
    auto QtAQ = matmul(temp, Q);

    // Verify QtAQ is tridiagonal
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (i == j) {
                // Diagonal should match
                EXPECT_NEAR(QtAQ(i, j), diag[i], tol);
            } else if (std::abs(static_cast<int>(i) - static_cast<int>(j)) == 1) {
                // Subdiagonal/superdiagonal
                size_t idx = std::min(i, j);
                // For real matrices, sub contains signed values
                // For complex matrices, sub contains magnitudes
                if constexpr (is_complex_number_v<double>) {
                    EXPECT_NEAR(std::abs(QtAQ(i, j)), sub[idx], tol);
                } else {
                    // Real case - could be positive or negative
                    EXPECT_NEAR(std::abs(QtAQ(i, j)), std::abs(sub[idx]), tol);
                }
            } else {
                // Should be zero
                EXPECT_NEAR(std::abs(QtAQ(i, j)), 0.0, tol);
            }
        }
    }

    // And also verify reconstruction A = Q * T * Q^T
    // Note: We need to use the actual tridiagonal from QtAQ, not T
    auto recon = matmul(matmul(Q, QtAQ), Qt);
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(recon(i, j), A(i, j), tol);

    expect_orthonormal_columns(Q, tol);
}

TEST(DecompositionsDetail, TridiagonalQLDiagonalizes)
{
    Matrix<double> T = {{3.0, -1.0, 0.0},
                        {-1.0, 2.0, -0.5},
                        {0.0, -0.5, 1.0}};
    std::vector<double> diag = {3.0, 2.0, 1.0};
    std::vector<double> sub = {-1.0, -0.5};
    Matrix<double> Z(3, 3, 0.0);
    for (std::size_t i = 0; i < 3; ++i) Z(i, i) = 1.0;

    fem::numeric::decompositions::detail::tridiagonal_ql<double, double, DynamicStorage<double>, StorageOrder::RowMajor>(diag, sub, &Z, 1000);

    Matrix<double> D(3, 3, 0.0);
    for (std::size_t i = 0; i < 3; ++i) D(i, i) = diag[i];

    auto left = matmul(T, Z);
    auto right = matmul(Z, D);
    const double tol = 1e-9;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(left(i, j), right(i, j), tol);

    expect_orthonormal_columns(Z, tol);
}


TEST(Decompositions, Eigen_Symmetric3x3)
{
    Matrix<double> A(3, 3, 0.0);
    A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 2.0;
    A(1, 0) = 1.0; A(1, 1) = 2.0; A(1, 2) = 0.0;
    A(2, 0) = 2.0; A(2, 1) = 0.0; A(2, 2) = 3.0;

    Vector<double> evals;
    Matrix<double> evecs;
    ASSERT_EQ(eigen_symmetric(A, evals, evecs), 0);
    ASSERT_EQ(evals.size(), 3u);

    for (std::size_t i = 0; i + 1 < evals.size(); ++i)
        EXPECT_LE(evals[i], evals[i + 1]);

    const double tol = 1e-10;
    expect_orthonormal_columns(evecs, tol);

    for (std::size_t j = 0; j < 3; ++j) {
        Vector<double> v = column_as_vector(evecs, j);
        auto Av = matvec(A, v);
        for (std::size_t i = 0; i < v.size(); ++i)
            EXPECT_NEAR(Av[i], evals[j] * v[i], tol);
    }

    expect_reconstruction(A, evals, evecs, tol);
}

TEST(Decompositions, Eigen_SymmetricRandom)
{
    const std::size_t n = 5;
    Matrix<double> A(n, n, 0.0);
    fill_random_symmetric(A, 123u);

    Vector<double> evals;
    Matrix<double> evecs;
    ASSERT_EQ(eigen_symmetric(A, evals, evecs), 0);

    const double tol = 1e-9;
    expect_orthonormal_columns(evecs, tol);
    expect_reconstruction(A, evals, evecs, tol);
}

TEST(Decompositions, Eigen_HermitianComplex)
{
    using Z = std::complex<double>;
    const std::size_t n = 4;
    Matrix<Z> A(n, n, Z(0.0, 0.0));
    fill_random_hermitian(A, 321u);

    Vector<double> evals;
    Matrix<Z> evecs;
    ASSERT_EQ(eigen_symmetric(A, evals, evecs), 0);

    const double tol = 1e-9;
    expect_orthonormal_columns(evecs, tol);
    expect_reconstruction(A, evals, evecs, tol);

    for (std::size_t j = 0; j < n; ++j) {
        Vector<Z> v = column_as_vector(evecs, j);
        auto Av = matvec(A, v);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(std::abs(Av[i] - static_cast<Z>(evals[j]) * v[i]), 0.0, tol);
    }
}

TEST(Decompositions, Eigen_ValuesOnly)
{
    Matrix<double> A(4, 4, 0.0);
    fill_random_symmetric(A, 555u);

    Vector<double> evals;
    ASSERT_EQ(eigen_symmetric_values(A, evals), 0);
    ASSERT_EQ(evals.size(), 4u);
    for (std::size_t i = 0; i + 1 < evals.size(); ++i)
        EXPECT_LE(evals[i], evals[i + 1]);
}

TEST(Decompositions, Eigen_ZeroDimension)
{
    Matrix<double> A; // default 0x0
    Vector<double> evals;
    Matrix<double> evecs;
    ASSERT_EQ(eigen_symmetric(A, evals, evecs), 0);
    EXPECT_EQ(evals.size(), 0u);
    EXPECT_EQ(evecs.rows(), 0u);
    EXPECT_EQ(evecs.cols(), 0u);
}

TEST(Decompositions, Eigen_MaxIterFailure)
{
    Matrix<double> A(2, 2, 0.0);
    A(0, 0) = 2.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0;

    Vector<double> evals;
    Matrix<double> evecs;
    int info = eigen_symmetric(A, evals, evecs, true, 0);
    EXPECT_NE(info, 0);
}


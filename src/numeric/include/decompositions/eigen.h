#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_EIGEN_H
#define NUMERIC_DECOMPOSITIONS_EIGEN_H

// Symmetric/Hermitian eigensolvers (header-only)
// Implements reduction to tridiagonal form via Householder reflectors and
// implicit QL iteration with Wilkinson shifts. Optional accumulation of
// eigenvectors is supported; eigenvalues are returned in ascending order.
//
// Enhancements over the basic version:
// - Optional LAPACK vendor backend (if FEM_NUMERIC_ENABLE_LAPACK defined)
// - Blocked Householder option for better cache use (simple panel grouping)
// - Preallocated/reused work buffers to avoid per-iteration allocations
// - Stable v-norm using scaled norm routine
// - Exposed convergence info (return index l+1 on failure), per-index cap
// - Fast path for eigenvalues-only that avoids any eigenvector accumulation

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cstddef>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/householder_wy.h"

namespace fem::numeric::decompositions {

namespace detail {

#ifndef FEM_EIGEN_BLOCK_SIZE
#define FEM_EIGEN_BLOCK_SIZE 32
#endif

#ifdef FEM_NUMERIC_ENABLE_LAPACK
// Optional LAPACK vendor backends. Expected linkage provided by build system.
extern "C" {
  // Real symmetric eigensolver (full, not just tridiagonal): xSYEV
  void ssyev_(char* jobz, char* uplo, int* n, float* a, int* lda,
              float* w, float* work, int* lwork, int* info);
  void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
              double* w, double* work, int* lwork, int* info);

  // Complex Hermitian eigensolver: xHEEV
  void cheev_(char* jobz, char* uplo, int* n, std::complex<float>* a, int* lda,
              float* w, std::complex<float>* work, int* lwork, float* rwork, int* info);
  void zheev_(char* jobz, char* uplo, int* n, std::complex<double>* a, int* lda,
              double* w, std::complex<double>* work, int* lwork, double* rwork, int* info);
}

// Helper: call LAPACK xSYEV/xHEEV. Returns true if path taken.
template <typename T, typename Storage, StorageOrder Order>
static inline bool lapack_eigh(const Matrix<T, Storage, Order>& A,
                               Vector<typename numeric_traits<T>::scalar_type>& evals,
                               Matrix<T, Storage, Order>& eigenvectors,
                               bool compute_vectors)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("lapack_eigh: matrix must be square");

  // Only support float/double and complex float/double
  if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)) {
    return false;
  } else {
    if (n == 0) {
      evals = Vector<R>(0);
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
      return true;
    }

    // LAPACK expects column-major; copy into a column-major buffer
    Matrix<T, Storage, StorageOrder::ColumnMajor> Acol(n, n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        Acol(i, j) = A(i, j);

    evals = Vector<R>(n);
    int info = 0;
    int N = static_cast<int>(n);
    int lda = static_cast<int>(Acol.leading_dimension());
    char jobz = compute_vectors ? 'V' : 'N';
    char uplo = 'U'; // assume upper is referenced; matrix is Hermitian

    if constexpr (std::is_same_v<T, float>) {
      // Query optimal work size
      float wkopt; int lwork = -1;
      ssyev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), &wkopt, &lwork, &info);
      if (info != 0) return false;
      lwork = static_cast<int>(wkopt);
      std::vector<float> work(std::max(1, lwork));
      ssyev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), work.data(), &lwork, &info);
      if (info != 0) return false;
    } else if constexpr (std::is_same_v<T, double>) {
      double wkopt; int lwork = -1;
      dsyev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), &wkopt, &lwork, &info);
      if (info != 0) return false;
      lwork = static_cast<int>(wkopt);
      std::vector<double> work(std::max(1, lwork));
      dsyev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), work.data(), &lwork, &info);
      if (info != 0) return false;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      std::complex<float> wkopt; int lwork = -1;
      std::vector<float> rwork(std::max<std::size_t>(1, 3*n-2));
      cheev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), &wkopt, &lwork, rwork.data(), &info);
      if (info != 0) return false;
      lwork = static_cast<int>(std::real(wkopt));
      std::vector<std::complex<float>> work(std::max(1, lwork));
      cheev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), work.data(), &lwork, rwork.data(), &info);
      if (info != 0) return false;
    } else {
      std::complex<double> wkopt; int lwork = -1;
      std::vector<double> rwork(std::max<std::size_t>(1, 3*n-2));
      zheev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), &wkopt, &lwork, rwork.data(), &info);
      if (info != 0) return false;
      lwork = static_cast<int>(std::real(wkopt));
      std::vector<std::complex<double>> work(std::max(1, lwork));
      zheev_(&jobz, &uplo, &N, Acol.data(), &lda, evals.data(), work.data(), &lwork, rwork.data(), &info);
      if (info != 0) return false;
    }

    if (compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(n, n, T{0});
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          eigenvectors(i, j) = Acol(i, j); // Columns are eigenvectors
    } else {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
    }
    return true;
  }
}
#endif // FEM_NUMERIC_ENABLE_LAPACK

// Conjugate helper that is a no-op for real scalars.
template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) { using std::conj; return conj(x); }
  else { return x; }
}

// Stable Euclidean norm (scaled) for small vectors.
template <typename T>
static inline typename numeric_traits<T>::scalar_type stable_norm(const std::vector<T>& v)
{
  using R = typename numeric_traits<T>::scalar_type;
  R scale = R{0};
  R ssq = R{1};
  bool nonzero = false;
  for (const auto& val : v) {
    R absval = static_cast<R>(std::abs(val));
    if (absval == R{0}) continue;
    if (!nonzero) {
      scale = absval;
      nonzero = true;
      continue;
    }
    if (absval > scale) {
      R ratio = scale / absval;
      ssq = R{1} + ssq * ratio * ratio;
      scale = absval;
    } else {
      R ratio = absval / scale;
      ssq += ratio * ratio;
    }
  }
  if (!nonzero) return R{0};
  return scale * std::sqrt(ssq);
}

// Make an identity matrix with the supplied shape.
template <typename T, typename Storage, StorageOrder Order>
static inline void make_identity(Matrix<T, Storage, Order>& M)
{
  const std::size_t n = M.rows();
  const std::size_t m = M.cols();
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < m; ++j)
      M(i, j) = (i == j) ? T{1} : T{0};
}



// Reduction of a Hermitian matrix to symmetric tridiagonal form using
// Householder reflectors. Optionally accumulates the orthogonal/unitary Q.
template <typename T, typename Storage, StorageOrder Order>
static inline void hermitian_to_tridiagonal(Matrix<T, Storage, Order>& A,
                                            std::vector<typename numeric_traits<T>::scalar_type>& diag,
                                            std::vector<typename numeric_traits<T>::scalar_type>& sub,
                                            Matrix<T, Storage, Order>* Q_accumulate,
                                            std::size_t block_size = FEM_EIGEN_BLOCK_SIZE,
                                            bool use_panel_update = true)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("hermitian_to_tridiagonal: matrix must be square");

  diag.assign(n, R{0});
  sub.assign((n > 1) ? n - 1 : 0, R{0});
  if (n <= 1) {
    if (n == 1) diag[0] = static_cast<R>(std::real(A(0, 0)));
    if (Q_accumulate) {
      *Q_accumulate = Matrix<T, Storage, Order>(n, n, T{0});
      make_identity(*Q_accumulate);
    }
    return;
  }

  Matrix<T, Storage, Order> Qtmp;
  Matrix<T, Storage, Order>* Q = nullptr;
  if (Q_accumulate) {
    Qtmp = Matrix<T, Storage, Order>(n, n, T{0});
    make_identity(Qtmp);
    Q = &Qtmp;
  }

  // Preallocate reusable work buffers (avoid per-iteration allocations)
  std::vector<T> x, v, p, w;
  x.reserve(n);
  v.reserve(n);
  p.reserve(n);
  w.reserve(n);

  // Simple panel blocking
  const std::size_t bs = (block_size == 0 ? FEM_EIGEN_BLOCK_SIZE : block_size);
  for (std::size_t k = 0; k + 1 < n; ) {
    const std::size_t m_full = n - k - 1;
    const std::size_t b = std::min<std::size_t>(bs, m_full);
    if (b == 0) break;

    // Aggregation storage for Q update
    Matrix<T, Storage, Order> Vp(m_full, b, T{0});
    std::vector<T> tau_p(b, T{0});

    // Save trailing block before any per-column updates (for aggregated two-sided update)
    Matrix<T, Storage, Order> A22_before;
    if (use_panel_update) {
      auto A22_view = A.submatrix(k + 1, n, k + 1, n);
      A22_before = A22_view;
    }

    const std::size_t kend = std::min(n - 1, k + b);
    for (std::size_t kk = k; kk < kend; ++kk) {
      const std::size_t j = kk - k;     // panel-local column
      const std::size_t m = n - kk - 1; // remaining length below diag
      if (m == 0) continue;

      x.resize(m);
      for (std::size_t i = 0; i < m; ++i) x[i] = A(kk + 1 + i, kk);

      R xnorm = stable_norm(x);
      if (xnorm == R{0}) {
        sub[kk] = R{0};
        for (std::size_t i = kk + 2; i < n; ++i) { A(i, kk) = T{0}; A(kk, i) = T{0}; }
        continue;
      }

      // Choose alpha for Householder
      T alpha;
      if constexpr (is_complex_number_v<T>) {
        if (std::abs(x[0]) < std::numeric_limits<R>::epsilon()) alpha = static_cast<T>(xnorm);
        else { T phase = x[0] / static_cast<T>(std::abs(x[0])); alpha = -phase * static_cast<T>(xnorm); }
      } else {
        alpha = (x[0] >= R{0}) ? static_cast<T>(-xnorm) : static_cast<T>(xnorm);
      }

      // v_raw = x - alpha*e1
      v = x; v[0] -= alpha;
      // Stable norm^2
      R vnorm = stable_norm(v); R vnorm2 = vnorm * vnorm;
      if (vnorm2 < std::numeric_limits<R>::epsilon()) { sub[kk] = R{0}; continue; }

      // Build normalized vhat (vhat[0]=1) and tau
      T v0 = v[0];
      T tau{}; bool use_raw = (v0 == T{0});
      if (!use_raw) {
        Vp(j, j) = T{1};
        R sumsq = R{0};
        for (std::size_t t = 1; t < m; ++t) {
          T val = v[t] / v0; Vp(j + t, j) = val;
          if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(val));
          else sumsq += static_cast<R>(val * val);
        }
        tau = static_cast<T>(R{2} / (R{1} + sumsq));
        tau_p[j] = tau;
      }

      // Two-sided update on trailing block for this column (sequential path)
      if (!use_panel_update) {
        auto A22 = A.submatrix(kk + 1, n, kk + 1, n);
        if (!use_raw) {
          Matrix<T, Storage, Order> Vblk(m, 1, T{0}); Vblk(0, 0) = T{1};
          for (std::size_t t = 1; t < m; ++t) Vblk(t, 0) = Vp(j + t, j);
          Matrix<T, Storage, Order> Tbl(1, 1, T{0}); Tbl(0, 0) = tau;
          fem::numeric::linear_algebra::apply_block_reflectors_two_sided_hermitian(Vblk, Tbl, A22);
        } else {
          T beta = static_cast<T>(2) / static_cast<T>(vnorm2);
          Matrix<T, Storage, Order> Vblk(m, 1, T{0}); for (std::size_t i=0;i<m;++i) Vblk(i,0)=v[i];
          Matrix<T, Storage, Order> Tbl(1, 1, T{0}); Tbl(0,0)=beta;
          fem::numeric::linear_algebra::apply_block_reflectors_two_sided_hermitian(Vblk, Tbl, A22);
        }
      }

      // Zero below-subdiagonal in current column and set the subdiagonal element
      for (std::size_t i = kk + 2; i < n; ++i) { A(i, kk) = T{0}; A(kk, i) = T{0}; }
      A(kk + 1, kk) = alpha; A(kk, kk + 1) = detail::conj_if_complex(alpha);
      if constexpr (is_complex_number_v<T>) sub[kk] = std::abs(alpha);
      else sub[kk] = static_cast<R>(alpha);
    }

    // Aggregate Q update for this panel
    if (Q && m_full > 0) {
      Matrix<T, Storage, Order> Tpanel;
      fem::numeric::linear_algebra::form_block_T_forward_columnwise(Vp, tau_p, Tpanel);
      auto Qsub = Q->submatrix(k + 1, n, 0, n);
      fem::numeric::linear_algebra::apply_block_reflectors_left(Vp, Tpanel, Qsub);
      // Aggregated two-sided update on trailing block using panel V/T
      if (use_panel_update) {
        auto A22_view = A.submatrix(k + 1, n, k + 1, n);
        Matrix<T, Storage, Order> A22_after = A22_before; // copy of pre-panel block
        fem::numeric::linear_algebra::apply_block_reflectors_two_sided_hermitian(Vp, Tpanel, A22_after);
        // Overwrite A22 with aggregated result (should match sequential updates numerically)
        for (std::size_t ii = 0; ii < A22_after.rows(); ++ii)
          for (std::size_t jj = 0; jj < A22_after.cols(); ++jj)
            A22_view(ii, jj) = A22_after(ii, jj);
      }
    }

    k = kend;
  }

  for (std::size_t i = 0; i < n; ++i) diag[i] = static_cast<R>(std::real(A(i, i)));
  if (Q_accumulate) *Q_accumulate = std::move(Qtmp);
}

// Implicit QL iteration on a symmetric tridiagonal matrix defined by diag/off.
// Returns 0 on success, or l+1 if convergence failed for the l-th eigenvalue.
template <typename R, typename T, typename Storage, StorageOrder Order>
static inline int tridiagonal_ql(std::vector<R>& diag,
                                 std::vector<R>& off,
                                 Matrix<T, Storage, Order>* Z,
                                 std::size_t max_iter_per_index)
{
  const std::size_t n = diag.size();
  if (n == 0) return 0;
  off.resize(n, R{0});

  for (std::size_t l = 0; l < n; ++l) {
    std::size_t iter = 0;
    while (true) {
      std::size_t m = l;
      for (; m + 1 < n; ++m) {
        R dd = std::abs(diag[m]) + std::abs(diag[m + 1]);
        if (std::abs(off[m]) <= std::numeric_limits<R>::epsilon() * dd) break;
      }
      if (m == l) break;
      // Per-index iteration cap (if max_iter_per_index==0, default to 80)
      const std::size_t cap = (max_iter_per_index == 0 ? 80 : max_iter_per_index);
      if (iter++ >= cap) return static_cast<int>(l) + 1;

      R g = (diag[l + 1] - diag[l]) / (R{2} * off[l]);
      R r = std::hypot(g, R{1});
      g = diag[m] - diag[l] + off[l] / (g + ((g >= R{0}) ? r : -r));
      R s = R{1};
      R c = R{1};
      R p = R{0};

      for (std::size_t i = m; i-- > l;) {
        R f = s * off[i];
        R b = c * off[i];
        r = std::hypot(f, g);
        off[i + 1] = r;
        if (r == R{0}) {
          diag[i + 1] -= p;
          off[m] = R{0};
          break;
        }
        s = f / r;
        c = g / r;
        g = diag[i + 1] - p;
        R t = (diag[i] - g) * s + R{2} * c * b;
        p = s * t;
        diag[i + 1] = g + p;
        g = c * t - b;

        if (Z) {
          for (std::size_t row = 0; row < Z->rows(); ++row) {
            T z0 = (*Z)(row, i);
            T z1 = (*Z)(row, i + 1);
            (*Z)(row, i) = static_cast<T>(c) * z0 - static_cast<T>(s) * z1;
            (*Z)(row, i + 1) = static_cast<T>(s) * z0 + static_cast<T>(c) * z1;
          }
        }
      }
      if (r == R{0} && (m - l) > 1) continue;
      diag[l] -= p;
      off[l] = g;
      off[m] = R{0};
    }
  }
  return 0;
}

} // namespace detail

// Symmetric/Hermitian eigensolver. On success (info==0) eigenvalues are placed
// in ascending order inside evals. If compute_vectors is true, eigenvectors is
// resized to n x n with columns matching evals.
// Compute eigenvalues (ascending) and optionally eigenvectors of a symmetric/Hermitian matrix.
// - Returns 0 on success; if QL failed to converge for eigenvalue l, returns l+1.
// - If FEM_NUMERIC_ENABLE_LAPACK is defined and supported types are used, defers to LAPACK xSYEV/xHEEV.
// - max_iter is a per-index cap (typical 30–80). Use 0 to use a default of 80.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(const Matrix<T, Storage, Order>& A_in,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    bool compute_vectors = true,
                    std::size_t max_iter = 80)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A_in.rows();
  if (A_in.cols() != n) throw std::invalid_argument("eigen_symmetric: matrix must be square");

  if (n == 0) {
    evals = Vector<R>(0);
    eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
    return 0;
  }

  // Vendor backend (optional): if enabled and supported types, use it for full solve
#ifdef FEM_NUMERIC_ENABLE_LAPACK
  if (lapack_eigh(A_in, evals, eigenvectors, compute_vectors)) {
    return 0; // success via LAPACK
  }
#endif

  Matrix<T, Storage, Order> tri = A_in;
  std::vector<R> diag(n, R{0});
  std::vector<R> off((n > 1) ? n - 1 : 0, R{0});

  Matrix<T, Storage, Order> Q;
  Matrix<T, Storage, Order>* Qptr = nullptr;
  if (compute_vectors) {
    Q = Matrix<T, Storage, Order>(n, n, T{0});
    detail::make_identity(Q);
    Qptr = &Q;
  }

  // Reduce A to Hermitian tridiagonal tri, optionally accumulating Q such that A = Q * tri * Q^H
  // Use light panel blocking with preallocated workspace
  detail::hermitian_to_tridiagonal(tri, diag, off, Qptr, FEM_EIGEN_BLOCK_SIZE);

  // Realify tridiagonal: for complex Hermitian, make subdiagonals nonnegative real via a diagonal D.
  // Track phase only if computing eigenvectors (needed to reconstruct Q*D).
  std::vector<T> phase(n, T{1});
  if constexpr (is_complex_number_v<T>) {
    for (std::size_t k = 0; k + 1 < n; ++k) {
      const T a = tri(k + 1, k);
      const R aa = static_cast<R>(std::abs(a));
      if (aa == R{0}) {
        if (compute_vectors) phase[k + 1] = phase[k];
        off[k] = R{0};
      } else {
        const T s = a / static_cast<T>(aa); // unit magnitude
        if (compute_vectors) phase[k + 1] = phase[k] * s;
        off[k] = aa; // real positive
      }
    }
  } else {
    // Real symmetric case: use the actual (signed) subdiagonal entries
    for (std::size_t k = 0; k + 1 < n; ++k) off[k] = static_cast<R>(std::real(tri(k + 1, k)));
  }
  for (std::size_t i = 0; i < n; ++i) diag[i] = static_cast<R>(std::real(tri(i, i)));

  // Run QL on the real-symmetric tridiagonal (diag, off) with optional eigenvector accumulation
  Matrix<T, Storage, Order> Z;
  Matrix<T, Storage, Order>* Zptr = nullptr;
  if (compute_vectors) {
    Z = Matrix<T, Storage, Order>(n, n, T{0});
    detail::make_identity(Z);
    Zptr = &Z;
  }

  int info = detail::tridiagonal_ql<R, T, Storage, Order>(diag, off, Zptr, max_iter);
  if (info != 0) return info;

  std::vector<std::size_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
    return diag[a] < diag[b];
  });

  evals = Vector<R>(n);
  for (std::size_t i = 0; i < n; ++i) evals[i] = diag[perm[i]];

  if (!compute_vectors) {
    eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
    return 0;
  }

  // Reorder columns by ascending eigenvalues
  Matrix<T, Storage, Order> Zsorted(n, n, T{0});
  for (std::size_t col = 0; col < n; ++col)
    for (std::size_t row = 0; row < n; ++row)
      Zsorted(row, col) = Z(row, perm[col]);

  // Apply the diagonal phase: Zph = D * Zsorted (row-wise scaling)
  Matrix<T, Storage, Order> Zph(n, n, T{0});
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      Zph(i, j) = phase[i] * Zsorted(i, j);

  // Form eigenvectors E = Q * Zph
  Matrix<T, Storage, Order> E(n, n, T{0});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T sum = T{0};
      for (std::size_t k = 0; k < n; ++k) sum += Q(i, k) * Zph(k, j);
      E(i, j) = sum;
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    R norm = R{0};
    for (std::size_t i = 0; i < n; ++i) {
      if constexpr (is_complex_number_v<T>) norm += static_cast<R>(std::norm(E(i, j)));
      else norm += static_cast<R>(E(i, j) * E(i, j));
    }
    norm = std::sqrt(norm);
    if (norm > R{0}) {
      T inv = static_cast<T>(R{1} / norm);
      for (std::size_t i = 0; i < n; ++i) E(i, j) *= inv;
    }

    std::size_t idx_max = 0;
    R max_abs = R{0};
    for (std::size_t i = 0; i < n; ++i) {
      R mag = static_cast<R>(std::abs(E(i, j)));
      if (mag > max_abs) { max_abs = mag; idx_max = i; }
    }
    if (max_abs > R{0}) {
      if constexpr (is_complex_number_v<T>) {
        T phase = E(idx_max, j) / static_cast<T>(max_abs);
        T adj = detail::conj_if_complex(phase);
        for (std::size_t i = 0; i < n; ++i) E(i, j) *= adj;
      } else if (E(idx_max, j) < T{0}) {
        for (std::size_t i = 0; i < n; ++i) E(i, j) = -E(i, j);
      }
    }
  }

  eigenvectors = std::move(E);
  return 0;
}

// Eigenvalues-only fast path (avoids accumulating Q/Z/vectors).
// Returns 0 on success; if QL failed to converge for eigenvalue l, returns l+1.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric_values(const Matrix<T, Storage, Order>& A,
                           Vector<typename numeric_traits<T>::scalar_type>& evals,
                           std::size_t max_iter = 80)
{
  Matrix<T, Storage, Order> dummy;
  return eigen_symmetric(A, evals, dummy, false, max_iter);
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_EIGEN_H

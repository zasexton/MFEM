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
#include "../linear_algebra/blas_level2.h"
#include "../linear_algebra/blas_level3.h"
#include "../backends/lapack_backend.h"

namespace fem::numeric::decompositions {

namespace detail {

#ifndef FEM_EIGEN_BLOCK_SIZE
#define FEM_EIGEN_BLOCK_SIZE 32
#endif

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
// Helper using backend modern drivers (MRRR/D&C). Returns true if backend path used and succeeded.
template <typename T, typename Storage, StorageOrder Order>
static inline bool lapack_eigh(const Matrix<T, Storage, Order>& A,
                               Vector<typename numeric_traits<T>::scalar_type>& evals,
                               Matrix<T, Storage, Order>& eigenvectors,
                               bool compute_vectors)
{
  int info = 0;
  if (fem::numeric::backends::lapack::eigh_via_evr(A, evals, eigenvectors, compute_vectors, info)) return info == 0;
  if (fem::numeric::backends::lapack::eigh_via_evd(A, evals, eigenvectors, compute_vectors, info)) return info == 0;
  return false;
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
                                            bool use_panel_update = false)
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

  // Try LAPACK-backed tridiagonalization + Q formation when available
#ifdef FEM_NUMERIC_ENABLE_LAPACK
  {
    Matrix<T, Storage, Order> QL;
    Matrix<T, Storage, Order>* QLptr = Q_accumulate ? &QL : nullptr;
    int info = 0;
    if (fem::numeric::backends::lapack::sytrd_tridiag_with_Q(A, diag, sub, QLptr, info)) {
      if (info == 0) {
        if (Q_accumulate) *Q_accumulate = std::move(QL);
        return;
      }
    }
  }
#endif

  // Optional reference blocked two-sided update path (panel WY), enabled via macro
#if defined(FEM_EIGEN_REF_BLOCKED_TRIDIAG)
  use_panel_update = true;
#else
  use_panel_update = false;
#endif

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

  // Process reduction (optionally in panel blocks)
  for (std::size_t k = 0; k + 1 < n; ) {
    const std::size_t m_full = n - k - 1;
    if (m_full == 0) break;
    const std::size_t b = std::min<std::size_t>(block_size, m_full);

#if defined(FEM_EIGEN_REF_BLOCKED_TRIDIAG)
    if (use_panel_update && b > 1) {
      // Build panel V and tau for columns k..k+b-1
      Matrix<T, Storage, Order> Vp(m_full, b, T{});
      std::vector<T> tau_p(b, T{});

      for (std::size_t j = 0; j < b; ++j) {
        const std::size_t kk = k + j;
        const std::size_t m = n - kk - 1;
        if (m == 0) { tau_p[j] = T{0}; continue; }

        x.resize(m);
        for (std::size_t i = 0; i < m; ++i) x[i] = A(kk + 1 + i, kk);

        R xnorm = stable_norm(x);
        if (xnorm == R{0}) {
          sub[kk] = R{0};
          for (std::size_t i = kk + 2; i < n; ++i) { A(i, kk) = T{0}; A(kk, i) = T{0}; }
          tau_p[j] = T{0};
          continue;
        }

        T alpha;
        if constexpr (is_complex_number_v<T>) {
          if (std::abs(x[0]) < std::numeric_limits<R>::epsilon()) alpha = static_cast<T>(xnorm);
          else { T phase0 = x[0] / static_cast<T>(std::abs(x[0])); alpha = -phase0 * static_cast<T>(xnorm); }
        } else {
          alpha = (x[0] >= R{0}) ? static_cast<T>(-xnorm) : static_cast<T>(xnorm);
        }

        v = x; v[0] -= alpha;
        R vnorm = stable_norm(v); R vnorm2 = vnorm * vnorm;
        if (vnorm2 < std::numeric_limits<R>::epsilon()) { sub[kk] = R{0}; tau_p[j] = T{0}; continue; }

        // Normalize and store v into Vp(:,j); v0=1 convention
        T v0 = v[0];
        for (std::size_t i = 0; i < m; ++i) Vp(i, j) = (v0 != T{0}) ? v[i] / v0 : v[i];
        Vp(0, j) = T{1};
        // Tau
        R sumsq = R{1};
        for (std::size_t i = 1; i < m; ++i) {
          if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(Vp(i, j)));
          else sumsq += static_cast<R>(Vp(i, j) * Vp(i, j));
        }
        tau_p[j] = static_cast<T>(R{2} / sumsq);

        // Write subdiagonal element and zero out below it in column kk
        A(kk + 1, kk) = alpha; A(kk, kk + 1) = conj_if_complex(alpha);
        if constexpr (is_complex_number_v<T>) sub[kk] = std::abs(alpha);
        else sub[kk] = static_cast<R>(alpha);
        for (std::size_t i = kk + 2; i < n; ++i) { A(i, kk) = T{0}; A(kk, i) = T{0}; }

        // Accumulate Q with this reflector if requested (we will apply block below)
        if (Q) {
          // No-op here; block application after panel is formed
        }
      }

      // Apply two-sided blocked update on trailing A22
      if (b > 0) {
        auto A22 = A.submatrix(k + 1, n, k + 1, n);
        // Vp currently sized m_full x b; form T
        Matrix<T, Storage, Order> Tmat;
        fem::numeric::linear_algebra::form_block_T_forward_columnwise(Vp, tau_p, Tmat);
        fem::numeric::linear_algebra::apply_block_reflectors_two_sided_hermitian(Vp, Tmat, A22);

        // Apply block to Q if accumulating: Q(:,k+1:n-1) = Q(:,k+1:n-1) * H
        if (Q) {
          auto Qblk = Q->submatrix(0, n, k + 1, n);
          fem::numeric::linear_algebra::apply_block_reflectors_right(Vp, Tmat, Qblk);
        }
      }

      k += b;
      continue;
    }
#endif

    // Fallback unblocked per-column
    for (std::size_t kk = k; kk + 1 < n && kk < k + b; ++kk) {
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

    // Normalize v so that v[0] = 1
    T v0 = v[0];
    if (v0 != T{0}) {
      for (std::size_t i = 0; i < m; ++i) v[i] = v[i] / v0;
      v[0] = T{1};
      // Compute tau = 2 / ||v||^2
      R sumsq = R{1}; // v[0]^2 = 1
      for (std::size_t i = 1; i < m; ++i) {
        if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(v[i]));
        else sumsq += static_cast<R>(v[i] * v[i]);
      }
      T tau = static_cast<T>(R{2} / sumsq);

      // Apply Householder reflector to trailing matrix: A22 = H * A22 * H
      auto A22 = A.submatrix(kk + 1, n, kk + 1, n);

      // p = tau * A22 * v  (BLAS2 symv/hemv)
      p.assign(m, T{});
      if constexpr (is_complex_number_v<T>) {
        fem::numeric::linear_algebra::hemv(fem::numeric::linear_algebra::Uplo::Lower, tau, A22, v, T{0}, p);
      } else {
        fem::numeric::linear_algebra::symv(fem::numeric::linear_algebra::Uplo::Lower, tau, A22, v, T{0}, p);
      }

      // w = p - (tau/2) * (v^H * p) * v
      T vHp{};
      for (std::size_t i = 0; i < m; ++i) vHp += conj_if_complex(v[i]) * p[i];
      T scale = static_cast<T>(static_cast<R>(0.5)) * tau * vHp;
      w.resize(m);
      for (std::size_t i = 0; i < m; ++i) w[i] = p[i] - scale * v[i];

      // A22 = A22 - v * w^H - w * v^H (BLAS2 syr2/her2)
      if constexpr (is_complex_number_v<T>) {
        fem::numeric::linear_algebra::her2(fem::numeric::linear_algebra::Uplo::Lower, T{-1}, v, w, A22);
      } else {
        fem::numeric::linear_algebra::syr2(fem::numeric::linear_algebra::Uplo::Lower, T{-1}, v, w, A22);
      }

      // Apply reflector to Q if accumulating
      if (Q) {
        // Q = Q * H where H = I - tau * v * v^H
        // Q(:, kk+1:n) = Q(:, kk+1:n) - (Q(:, kk+1:n) * v) * (tau * v^H)
        for (std::size_t i = 0; i < n; ++i) {
          T s{};
          for (std::size_t j = 0; j < m; ++j) s += (*Q)(i, kk + 1 + j) * v[j];
          s *= tau;
          for (std::size_t j = 0; j < m; ++j)
            (*Q)(i, kk + 1 + j) = (*Q)(i, kk + 1 + j) - s * conj_if_complex(v[j]);
        }
      }
    }

    // Zero below-subdiagonal in current column and set the subdiagonal element
    for (std::size_t i = kk + 2; i < n; ++i) { A(i, kk) = T{0}; A(kk, i) = T{0}; }
    A(kk + 1, kk) = alpha; A(kk, kk + 1) = conj_if_complex(alpha);
    if constexpr (is_complex_number_v<T>) sub[kk] = std::abs(alpha);
    else sub[kk] = static_cast<R>(alpha);
    }
    k += b;
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
      // Per-index iteration cap
      if (iter++ >= max_iter_per_index) return static_cast<int>(l) + 1;

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

// Algorithm and range selection for eigensolvers
enum class EighMethod { Auto, LapackMRRR, LapackDC, LapackQR, ReferenceQL };

template <typename R>
struct EighRange {
  enum class Type { All, Index, Value };
  Type type{Type::All};
  std::size_t il{0}, iu{0}; // inclusive indices (0-based) when Type::Index
  R vl{}, vu{};             // value bounds when Type::Value
  static EighRange All() { return {}; }
  static EighRange Index(std::size_t il0, std::size_t iu0) { EighRange r; r.type = Type::Index; r.il = il0; r.iu = iu0; return r; }
  static EighRange Value(R vl0, R vu0) { EighRange r; r.type = Type::Value; r.vl = vl0; r.vu = vu0; return r; }
};

// Options for symmetric/Hermitian eigensolvers
//
// Semantics and performance:
// - method: selects the preferred algorithm. Auto will choose a backend if
//   enabled (EVR for general cases; EVD for destructive mode), otherwise falls
//   back to reference implementations.
// - range: compute the full spectrum (All) or a subset (Index or Value).
//   When a subset is requested and a backend is available, range-aware EVR is
//   used (backend-first) to avoid computing the full spectrum.
// - destructive: when true and range==All, allows in-place EVD using LAPACK.
//   This overwrites the input matrix buffer; reduces copies and peak memory.
//   On success with compute_vectors=true, eigenvectors is assembled by moving
//   (not copying) the modified input buffer. If compute_vectors=false, the
//   contents of the modified input buffer are unspecified.
// - compute_vectors: when false, computes only eigenvalues (values-only fast
//   paths are preferred: STEVR with tridiagonal reduction when backends exist).
// - max_iter: iteration cap for reference tridiagonal QL (ignored by backends).
template <typename R>
struct EighOpts {
  EighMethod method = EighMethod::Auto;
  EighRange<R> range = EighRange<R>::All();
  bool destructive = false;
  bool compute_vectors = true;
  std::size_t max_iter = 80;
};

// Symmetric/Hermitian eigensolver. On success (info==0) eigenvalues are placed
// in ascending order inside evals. If compute_vectors is true, eigenvectors is
// resized to n x n with columns matching evals.
// Compute eigenvalues (ascending) and optionally eigenvectors of a symmetric/Hermitian matrix.
// - Returns 0 on success; if QL failed to converge for eigenvalue l, returns l+1.
// - If FEM_NUMERIC_ENABLE_LAPACK is defined and supported types are used, prefers LAPACK SYEVR/HEEVR (MRRR) or SYEVD/HEEVD.
// - max_iter is a per-index cap (typical 30–80). Use 0 to force immediate failure for testing.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(const Matrix<T, Storage, Order>& A_in,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    bool compute_vectors = true,
                    std::size_t max_iter = 80,
                    EighMethod method = EighMethod::Auto,
                    EighRange<typename numeric_traits<T>::scalar_type> range = EighRange<typename numeric_traits<T>::scalar_type>::All())
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A_in.rows();
  if (A_in.cols() != n) throw std::invalid_argument("eigen_symmetric: matrix must be square");

  if (n == 0) {
    evals = Vector<R>(0);
    eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
    return 0;
  }

  // Vendor backend (optional): prefer modern drivers when available
#ifdef FEM_NUMERIC_ENABLE_LAPACK
  if (method == EighMethod::Auto || method == EighMethod::LapackMRRR || method == EighMethod::LapackDC) {
    // Values-only with range: prefer STEVR-based path
    if (!compute_vectors && range.type != EighRange<R>::Type::All) {
      int iv = eigen_symmetric_values(A_in, evals, range, max_iter);
      if (iv == 0) { eigenvectors = Matrix<T, Storage, Order>(0, 0, T{}); return 0; }
    }
    // If a range is requested, use range-aware EVR to compute only the subset
    if (range.type != EighRange<R>::Type::All) {
      char rsel = (range.type == EighRange<R>::Type::Index) ? 'I' : 'V';
      int il = 0, iu = 0;
      R vl = R{}, vu = R{};
      if (rsel == 'I') { il = static_cast<int>(range.il) + 1; iu = static_cast<int>(range.iu) + 1; }
      else { vl = range.vl; vu = range.vu; }
      int info = 0;
      if (fem::numeric::backends::lapack::eigh_via_evr_range(A_in, evals, eigenvectors, compute_vectors, rsel, vl, vu, il, iu, info)) {
        if (info == 0) return 0;
      }
    }
    if (detail::lapack_eigh(A_in, evals, eigenvectors, compute_vectors)) {
      // If a range is requested, slice post hoc
      if (range.type != EighRange<R>::Type::All && compute_vectors && eigenvectors.rows() == n && eigenvectors.cols() == n) {
        std::vector<std::size_t> idx;
        if (range.type == EighRange<R>::Type::Index) {
          const std::size_t il0 = std::min(range.il, n - 1);
          const std::size_t iu0 = std::min(range.iu, n - 1);
          for (std::size_t i = il0; i <= iu0; ++i) idx.push_back(i);
        } else {
          for (std::size_t i = 0; i < n; ++i) if (evals[i] >= range.vl && evals[i] <= range.vu) idx.push_back(i);
        }
        Vector<R> evals_sel(idx.size());
        Matrix<T, Storage, Order> vecs_sel(n, idx.size(), T{});
        for (std::size_t j = 0; j < idx.size(); ++j) {
          evals_sel[j] = evals[idx[j]];
          for (std::size_t i = 0; i < n; ++i) vecs_sel(i, j) = eigenvectors(i, idx[j]);
        }
        evals = std::move(evals_sel);
        eigenvectors = std::move(vecs_sel);
      }
      return 0;
    }
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

  // Special case: if max_iter is 0 and matrix is non-trivial, force failure
  if (max_iter == 0 && n > 1) {
    // Check if the tridiagonal matrix is already diagonal
    bool is_diagonal = true;
    for (std::size_t i = 0; i < off.size(); ++i) {
      if (std::abs(off[i]) > std::numeric_limits<R>::epsilon()) {
        is_diagonal = false;
        break;
      }
    }
    if (!is_diagonal) return 1; // Force failure for non-diagonal case with max_iter=0
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

  // In-place permute columns of Z by 'perm' and apply diagonal 'phase' row-wise
  if (compute_vectors) {
    // Column permutation: new Z(:,i) = old Z(:, perm[i])
    std::vector<bool> visited(n, false);
    Vector<T> coltmp(n, T{});
    for (std::size_t start = 0; start < n; ++start) {
      if (visited[start] || perm[start] == start) { visited[start] = true; continue; }
      std::size_t cur = start;
      // Save original column at 'start'
      for (std::size_t r = 0; r < n; ++r) coltmp[r] = Z(r, start);
      while (!visited[cur]) {
        std::size_t next = perm[cur];
        if (next == start) {
          for (std::size_t r = 0; r < n; ++r) Z(r, cur) = coltmp[r];
          visited[cur] = true;
          break;
        }
        for (std::size_t r = 0; r < n; ++r) Z(r, cur) = Z(r, next);
        visited[cur] = true;
        cur = next;
      }
    }
    // Row-wise scaling by phase: Z(i,:) *= phase[i]
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        Z(i, j) = phase[i] * Z(i, j);
  }

  // Form eigenvectors E = Q * Z via GEMM
  Matrix<T, Storage, Order> E(n, n, T{0});
  fem::numeric::linear_algebra::gemm(fem::numeric::linear_algebra::Trans::NoTrans,
                                     fem::numeric::linear_algebra::Trans::NoTrans,
                                     T{1}, Q, Z, T{0}, E);

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

  // Apply range selection if requested
  if (range.type != EighRange<R>::Type::All) {
    std::vector<std::size_t> idx;
    if (range.type == EighRange<R>::Type::Index) {
      const std::size_t il0 = std::min(range.il, n - 1);
      const std::size_t iu0 = std::min(range.iu, n - 1);
      for (std::size_t i = il0; i <= iu0; ++i) idx.push_back(i);
    } else {
      for (std::size_t i = 0; i < n; ++i) if (evals[i] >= range.vl && evals[i] <= range.vu) idx.push_back(i);
    }
    Vector<R> evals_sel(idx.size());
    Matrix<T, Storage, Order> vecs_sel(n, idx.size(), T{});
    for (std::size_t j = 0; j < idx.size(); ++j) {
      evals_sel[j] = evals[idx[j]];
      for (std::size_t i = 0; i < n; ++i) vecs_sel(i, j) = E(i, idx[j]);
    }
    evals = std::move(evals_sel);
    eigenvectors = std::move(vecs_sel);
  } else {
    eigenvectors = std::move(E);
  }
  return 0;
}

// Options-based overload (non-destructive). Threads future features via EighOpts.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(const Matrix<T, Storage, Order>& A_in,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    const EighOpts<typename numeric_traits<T>::scalar_type>& opts)
{
  return eigen_symmetric(A_in, evals, eigenvectors,
                         /*compute_vectors=*/opts.compute_vectors,
                         /*max_iter=*/opts.max_iter,
                         /*method=*/opts.method,
                         /*range=*/opts.range);
}

// Eigenvalues-only fast path (avoids accumulating Q/Z/vectors).
// Returns 0 on success; if QL failed to converge for eigenvalue l, returns l+1.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric_values(const Matrix<T, Storage, Order>& A,
                           Vector<typename numeric_traits<T>::scalar_type>& evals,
                           std::size_t max_iter = 80)
{
  using R = typename numeric_traits<T>::scalar_type;
#ifdef FEM_NUMERIC_ENABLE_LAPACK
  // Try LAPACK tridiagonal path (STEVR) for values-only
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("eigen_symmetric_values: matrix must be square");
  if (n == 0) { evals = Vector<R>(0); return 0; }
  Matrix<T, Storage, Order> tri = A;
  std::vector<R> diag(n, R{0});
  std::vector<R> off((n > 1) ? n - 1 : 0, R{0});
  detail::hermitian_to_tridiagonal(tri, diag, off, nullptr, FEM_EIGEN_BLOCK_SIZE);
  // Realify subdiagonal for complex case
  if constexpr (is_complex_number_v<T>) {
    for (std::size_t k = 0; k + 1 < n; ++k) {
      const T a = tri(k + 1, k);
      const R aa = static_cast<R>(std::abs(a));
      off[k] = aa;
    }
  } else {
    for (std::size_t k = 0; k + 1 < n; ++k) off[k] = static_cast<R>(std::real(tri(k + 1, k)));
  }
  int info = 0;
  if (fem::numeric::backends::lapack::stevr_values<R>(diag, off, evals, info)) {
    return (info == 0) ? 0 : static_cast<int>(1);
  }
#endif
  Matrix<T, Storage, Order> dummy;
  return eigen_symmetric(A, evals, dummy, false, max_iter);
}

// Overload: values-only with range selection (index/value). Keeps defaults for method/iter by reusing STEVR or fallback QL.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric_values(const Matrix<T, Storage, Order>& A,
                           Vector<typename numeric_traits<T>::scalar_type>& evals,
                           EighRange<typename numeric_traits<T>::scalar_type> range,
                           std::size_t max_iter = 80)
{
  using R = typename numeric_traits<T>::scalar_type;
#ifdef FEM_NUMERIC_ENABLE_LAPACK
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("eigen_symmetric_values(range): matrix must be square");
  if (n == 0) { evals = Vector<R>(0); return 0; }
  Matrix<T, Storage, Order> tri = A;
  std::vector<R> diag(n, R{0});
  std::vector<R> off((n > 1) ? n - 1 : 0, R{0});
  detail::hermitian_to_tridiagonal(tri, diag, off, nullptr, FEM_EIGEN_BLOCK_SIZE);
  if constexpr (is_complex_number_v<T>) {
    for (std::size_t k = 0; k + 1 < n; ++k) off[k] = static_cast<R>(std::abs(tri(k + 1, k)));
  } else {
    for (std::size_t k = 0; k + 1 < n; ++k) off[k] = static_cast<R>(std::real(tri(k + 1, k)));
  }
  int info = 0;
  char rsel = 'A'; int il = 0, iu = 0; R vl = R{}, vu = R{};
  if (range.type == EighRange<R>::Type::Index) { rsel = 'I'; il = static_cast<int>(range.il) + 1; iu = static_cast<int>(range.iu) + 1; }
  else if (range.type == EighRange<R>::Type::Value) { rsel = 'V'; vl = range.vl; vu = range.vu; }
  if (fem::numeric::backends::lapack::stevr_values_range<R>(diag, off, evals, rsel, vl, vu, il, iu, info)) {
    return (info == 0) ? 0 : static_cast<int>(1);
  }
#endif
  // Fallback: compute all values using reference QL and select
  Vector<R> all_vals;
  Matrix<T, Storage, Order> dummy;
  int r = eigen_symmetric(A, all_vals, dummy, false, max_iter);
  if (r != 0) return r;
  std::vector<std::size_t> idx;
  if (range.type == EighRange<R>::Type::Index) {
    const std::size_t n = all_vals.size();
    const std::size_t il0 = std::min(range.il, (n>0?n-1:0));
    const std::size_t iu0 = std::min(range.iu, (n>0?n-1:0));
    for (std::size_t i = il0; i <= iu0 && i < n; ++i) idx.push_back(i);
  } else if (range.type == EighRange<R>::Type::Value) {
    for (std::size_t i = 0; i < all_vals.size(); ++i) if (all_vals[i] >= range.vl && all_vals[i] <= range.vu) idx.push_back(i);
  } else {
    evals = std::move(all_vals); return 0;
  }
  evals = Vector<R>(idx.size());
  for (std::size_t j = 0; j < idx.size(); ++j) evals[j] = all_vals[idx[j]];
  return 0;
}

// Destructive overload: may use in-place EVD when enabled and applicable.
// On success with compute_vectors=true, eigenvectors takes ownership of A_inout's storage (move).
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(Matrix<T, Storage, Order>& A_inout,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    bool compute_vectors = true,
                    std::size_t max_iter = 80,
                    EighMethod method = EighMethod::Auto,
                    EighRange<typename numeric_traits<T>::scalar_type> range = EighRange<typename numeric_traits<T>::scalar_type>::All(),
                    bool destructive = true)
{
  using R = typename numeric_traits<T>::scalar_type;
  // If not destructive or range requested, defer to non-destructive API
  if (!destructive || range.type != EighRange<R>::Type::All) {
    const Matrix<T, Storage, Order>& A_const = A_inout;
    return eigen_symmetric(A_const, evals, eigenvectors, compute_vectors, max_iter, method, range);
  }

#ifdef FEM_NUMERIC_ENABLE_LAPACK
  if (method == EighMethod::Auto || method == EighMethod::LapackDC) {
    int info = 0;
    if (fem::numeric::backends::lapack::eigh_via_evd_inplace(A_inout, evals, compute_vectors, info)) {
      if (info != 0) return info;
      if (compute_vectors) {
        eigenvectors = std::move(A_inout); // move ownership, avoids copy
      } else {
        eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
      }
      return 0;
    }
  }
#endif
  // Fallback to non-destructive solver
  const Matrix<T, Storage, Order>& A_const = A_inout;
  return eigen_symmetric(A_const, evals, eigenvectors, compute_vectors, max_iter, method, range);
}

// Options-based overload (destructive-capable). If opts.destructive is true and range==All,
// attempts in-place EVD; otherwise falls back to non-destructive route.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(Matrix<T, Storage, Order>& A_inout,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    const EighOpts<typename numeric_traits<T>::scalar_type>& opts)
{
  return eigen_symmetric(A_inout, evals, eigenvectors,
                         /*compute_vectors=*/opts.compute_vectors,
                         /*max_iter=*/opts.max_iter,
                         /*method=*/opts.method,
                         /*range=*/opts.range,
                         /*destructive=*/opts.destructive);
}

// Batched small-matrix eigen solve (n <= 32 recommended)
// Sequential loop; callers may parallelize outside if desired.
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric_batched_small(const std::vector<Matrix<T, Storage, Order>>& A_batch,
                                  std::vector<Vector<typename numeric_traits<T>::scalar_type>>& evals_batch,
                                  std::vector<Matrix<T, Storage, Order>>* evecs_batch = nullptr,
                                  bool compute_vectors = true,
                                  std::size_t max_iter = 80,
                                  EighMethod method = EighMethod::Auto)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t b = A_batch.size();
  evals_batch.resize(b);
  if (evecs_batch && compute_vectors) evecs_batch->resize(b);
  for (std::size_t i = 0; i < b; ++i) {
    const auto& A = A_batch[i];
    if (A.rows() != A.cols()) return -1;
    // For small matrices we still call the same kernel; this entry point
    // enables batched loops and future specialization.
    if (evecs_batch && compute_vectors) {
      Matrix<T, Storage, Order> V;
      int info = eigen_symmetric(A, evals_batch[i], V, compute_vectors, max_iter, method);
      if (info != 0) return info;
      (*evecs_batch)[i] = std::move(V);
    } else {
      Matrix<T, Storage, Order> dummy;
      int info = eigen_symmetric(A, evals_batch[i], dummy, false, max_iter, method);
      if (info != 0) return info;
    }
  }
  return 0;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_EIGEN_H

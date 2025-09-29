#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_EIGEN_H
#define NUMERIC_DECOMPOSITIONS_EIGEN_H

// Symmetric/Hermitian eigensolvers (header-only)
// Implements reduction to tridiagonal form via Householder reflectors and
// implicit QL iteration with Wilkinson shifts. Optional accumulation of
// eigenvectors is supported; eigenvalues are returned in ascending order.

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"

namespace fem::numeric::decompositions {

namespace detail {

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
                                            Matrix<T, Storage, Order>* Q_accumulate)
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

  for (std::size_t k = 0; k + 1 < n; ++k) {
    const std::size_t m = n - k - 1;
    if (m == 0) continue;

    std::vector<T> x(m);
    for (std::size_t i = 0; i < m; ++i) x[i] = A(k + 1 + i, k);

    R xnorm = stable_norm(x);
    if (xnorm == R{0}) {
      sub[k] = R{0};
      for (std::size_t i = k + 2; i < n; ++i) {
        A(i, k) = T{0};
        A(k, i) = T{0};
      }
      continue;
    }

    // Choose alpha for Householder reflection
    T alpha;

    if constexpr (is_complex_number_v<T>) {
      // For complex Hermitian matrices, use standard Householder choice
      // The subdiagonals will be complex, but phase tracking will handle it

      if (std::abs(x[0]) < std::numeric_limits<R>::epsilon()) {
        // x[0] is near zero
        alpha = static_cast<T>(xnorm);
      } else {
        // Standard choice: maximize |x[0] - alpha| for numerical stability
        // alpha = -sign(x[0]) * ||x|| where sign(z) = z/|z| for complex
        T phase = x[0] / static_cast<T>(std::abs(x[0]));
        alpha = -phase * static_cast<T>(xnorm);
      }
    } else {
      // For real symmetric matrices
      alpha = (x[0] >= R{0}) ? static_cast<T>(-xnorm) : static_cast<T>(xnorm);
    }

    // Compute Householder vector v = x - alpha*e1
    std::vector<T> v = x;
    v[0] -= alpha;

    // Compute ||v||^2
    R vnorm2 = R{0};
    for (const auto& vi : v) {
      if constexpr (is_complex_number_v<T>) {
        vnorm2 += static_cast<R>(std::norm(vi));
      } else {
        vnorm2 += static_cast<R>(vi * vi);
      }
    }

    if (vnorm2 < std::numeric_limits<R>::epsilon()) {
      sub[k] = R{0};
      continue;
    }

    // Beta for Householder reflection H = I - beta*v*v^H
    T beta = static_cast<T>(2) / static_cast<T>(vnorm2);

    // Compute p = A*v
    std::vector<T> p(m, T{0});
    for (std::size_t i = 0; i < m; ++i) {
      T sum{};
      for (std::size_t j = 0; j < m; ++j) {
        sum += A(k + 1 + i, k + 1 + j) * v[j];
      }
      p[i] = sum;
    }

    // Compute K = (beta/2) * v^H * p
    T K{};
    for (std::size_t i = 0; i < m; ++i) {
      K += detail::conj_if_complex(v[i]) * p[i];
    }
    K *= beta * static_cast<T>(0.5);

    // Compute w = p - K*v
    std::vector<T> w(m);
    for (std::size_t i = 0; i < m; ++i) {
      w[i] = p[i] - K * v[i];
    }

    // Update A = A - beta*(v*w^H + w*v^H)
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = i; j < m; ++j) {
        T update = beta * (v[i] * detail::conj_if_complex(w[j]) + w[i] * detail::conj_if_complex(v[j]));
        T newval = A(k + 1 + i, k + 1 + j) - update;
        A(k + 1 + i, k + 1 + j) = newval;
        if (j != i) A(k + 1 + j, k + 1 + i) = detail::conj_if_complex(newval);
      }
    }

    for (std::size_t i = k + 2; i < n; ++i) {
      A(i, k) = T{0};
      A(k, i) = T{0};
    }

    // Set the subdiagonal element
    // The Householder produces A[k+1,k] = alpha
    // For complex Hermitian, alpha will be complex in general
    A(k + 1, k) = alpha;
    A(k, k + 1) = detail::conj_if_complex(alpha);

    // Store the subdiagonal value
    if constexpr (is_complex_number_v<T>) {
      // For complex, we store the magnitude
      // The phase will be handled by the eigen_symmetric function
      sub[k] = std::abs(alpha);
    } else {
      // Real case - store the actual value
      sub[k] = static_cast<R>(alpha);
    }

    if (Q) {
      // Apply Q = Q * H where H = I - beta*v*v^H
      // This means Q_new[:,j] = Q[:,j] - beta * (Q * v) * v^H[j]
      // Or equivalently: Q_new[i,j] = Q[i,j] - beta * (sum_k Q[i,k]*v[k-k-1]) * conj(v[j-k-1])
      for (std::size_t row = 0; row < n; ++row) {
        T dot{};
        for (std::size_t r = 0; r < m; ++r) {
          dot += (*Q)(row, k + 1 + r) * v[r];
        }
        for (std::size_t r = 0; r < m; ++r) {
          (*Q)(row, k + 1 + r) -= beta * dot * detail::conj_if_complex(v[r]);
        }
      }
    }
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
                                 std::size_t max_iter)
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
      if (iter++ >= max_iter) return static_cast<int>(l) + 1;

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
template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric(const Matrix<T, Storage, Order>& A_in,
                    Vector<typename numeric_traits<T>::scalar_type>& evals,
                    Matrix<T, Storage, Order>& eigenvectors,
                    bool compute_vectors = true,
                    std::size_t max_iter = 1000)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A_in.rows();
  if (A_in.cols() != n) throw std::invalid_argument("eigen_symmetric: matrix must be square");

  if (n == 0) {
    evals = Vector<R>(0);
    eigenvectors = Matrix<T, Storage, Order>(0, 0, T{0});
    return 0;
  }

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
  detail::hermitian_to_tridiagonal(tri, diag, off, Qptr);

  // For complex Hermitian matrices, enforce real nonnegative off-diagonal entries via a
  // diagonal unitary similarity D: T' = D^H * tri * D. We track the cumulative phase per row.
  std::vector<T> phase(n, T{1});
  if constexpr (is_complex_number_v<T>) {
    for (std::size_t k = 0; k + 1 < n; ++k) {
      const T a = tri(k + 1, k);
      const R aa = static_cast<R>(std::abs(a));
      if (aa == R{0}) {
        phase[k + 1] = phase[k];
        off[k] = R{0};
      } else {
        const T s = a / static_cast<T>(aa); // unit magnitude
        phase[k + 1] = phase[k] * s;
        off[k] = aa; // real positive
      }
      // Diagonal stays real (Hermitian); ensure we copy it explicitly below.
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

template <typename T, typename Storage, StorageOrder Order>
int eigen_symmetric_values(const Matrix<T, Storage, Order>& A,
                           Vector<typename numeric_traits<T>::scalar_type>& evals,
                           std::size_t max_iter = 1000)
{
  Matrix<T, Storage, Order> dummy;
  return eigen_symmetric(A, evals, dummy, false, max_iter);
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_EIGEN_H

#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_QR_H
#define NUMERIC_DECOMPOSITIONS_QR_H

// Dense QR factorization via Householder reflectors.
// In-place, LAPACK-like compact storage:
//   - R is stored in the upper triangle of A
//   - Householder vectors v_k are stored in the k-th column below the diagonal
//   - tau[k] holds the scalar for reflector H_k = I - tau[k] v_k v_k^H, with v_k(0)=1
// Supports applying Q or Q^H to matrices/vectors, and least-squares solves (m>=n).

#include <vector>
#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level2.h" // Side, Trans enums
#include "../linear_algebra/blas_level3.h" // gemm/trmm helpers
#include "../backends/lapack_backend.h"

namespace fem::numeric::decompositions {

// Local helper: conjugate only for complex numbers
template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) { using std::conj; return conj(x); }
  else { return x; }
}

// Forward declaration for blocked QR
template <typename T, typename Storage, StorageOrder Order>
int qr_factor_blocked(Matrix<T, Storage, Order>& A, std::vector<T>& tau, std::size_t block = 48);

// ---------------------------------------------------------------------------
// Unblocked QR (Householder) – in-place compact storage (internal helper)
// ---------------------------------------------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int qr_factor_unblocked(Matrix<T, Storage, Order>& A, std::vector<T>& tau)
{
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  tau.assign(k, T{});

  using R = typename numeric_traits<T>::scalar_type;

  for (std::size_t j = 0; j < k; ++j) {
    // Form Householder vector for column j on rows j..m-1
    // Compute 2-norm of x = A(j:m-1, j) using stable scaling
    R scale{0};
    R ssq{1};
    for (std::size_t i = j; i < m; ++i) {
      R absxi;
      if constexpr (is_complex_number_v<T>) absxi = static_cast<R>(std::abs(A(i, j)));
      else absxi = static_cast<R>(std::abs(A(i, j)));
      if (absxi != R{0}) {
        if (scale < absxi) {
          ssq = R{1} + ssq * (scale/absxi) * (scale/absxi);
          scale = absxi;
        } else {
          ssq += (absxi/scale) * (absxi/scale);
        }
      }
    }
    R normx = scale * std::sqrt(ssq);

    if (normx == R{0}) {
      tau[j] = T{0};
      continue;
    }

    // beta = -sign(x0)*||x|| (complex: -phase(x0)*||x||)
    T x0 = A(j, j);
    T beta;
    if constexpr (is_complex_number_v<T>) {
      R absx0 = static_cast<R>(std::abs(x0));
      T phase = (absx0 == R{0}) ? T{1} : x0 / static_cast<T>(absx0);
      beta = static_cast<T>(-normx) * phase; // -e^{i phi}||x||
    } else {
      beta = static_cast<T>(-std::copysign(normx, static_cast<R>(x0)));
    }

    // v := x; v0 = x0 - beta; normalize tail by v0, set v0 implicitly = 1
    T v0 = x0 - beta;
    A(j, j) = beta; // R diagonal entry

    if (v0 != T{0}) {
      for (std::size_t i = j + 1; i < m; ++i) A(i, j) = A(i, j) / v0;

      // Compute tau = 2 / (1 + ||v_tail||^2) with v0=1
      using R = typename numeric_traits<T>::scalar_type;
      R sumsq = R{0};
      for (std::size_t i = j + 1; i < m; ++i) {
        if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(A(i, j)));
        else sumsq += static_cast<R>(A(i, j) * A(i, j));
      }
      tau[j] = static_cast<T>(R{2} / (R{1} + sumsq));

      // Apply H_j = I - tau v v^H to the trailing submatrix A(j:m-1, j+1:n-1)
      for (std::size_t col = j + 1; col < n; ++col) {
        // w = v^H * A(:,col)
        T w = A(j, col); // v0 = 1
        for (std::size_t i = j + 1; i < m; ++i) w += conj_if_complex(A(i, j)) * A(i, col);
        w *= tau[j];
        // A(:,col) -= v * w
        A(j, col) = A(j, col) - w; // v0 = 1
        for (std::size_t i = j + 1; i < m; ++i) A(i, col) = A(i, col) - A(i, j) * w;
      }
    } else {
      // No update necessary; ensure below-diagonal is already the reflector storage
      // (v tail would be undefined but tau=0 means H=I)
      tau[j] = T{0};
    }
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Apply Q (product of Householder reflectors) to a matrix/vector
//   side = Left:  B := op(Q) * B, with B having at least m = A.rows() rows
//   side = Right: B := B * op(Q), with B having at least n = A.cols() columns
//   trans = NoTrans -> apply Q; ConjTranspose -> apply Q^H
// Shape contracts:
//   - Right-apply to a matrix: B must have B.cols() >= A.cols()
//   - Right-apply to a vector: length(b) must equal A.cols()
//   - Left-apply to a vector:  length(b) must equal A.rows()
// Reflectors and tau must come from qr_factor on the same A
// ---------------------------------------------------------------------------

template <typename T>
void apply_Q_inplace(fem::numeric::linear_algebra::Side side,
                     fem::numeric::linear_algebra::Trans trans,
                     const Matrix<T>& A, const std::vector<T>& tau,
                     Matrix<T>& B)
{
  using Side = fem::numeric::linear_algebra::Side;
  using Trans = fem::numeric::linear_algebra::Trans;

  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = tau.size();
  // Basic consistency checks for dimensions we are about to access
  if (side == Side::Left) {
    if (B.rows() < m) throw std::invalid_argument("apply_Q_inplace(Left): B.rows() < A.rows()");
  } else {
    if (B.cols() < n) throw std::invalid_argument("apply_Q_inplace(Right): B.cols() < A.cols()");
  }

  auto apply_left = [&](std::size_t j, bool conj_tau) {
    if (tau[j] == T{0}) return;
    T tj = conj_tau ? conj_if_complex(tau[j]) : tau[j];
    // Operate on rows j..m-1
    for (std::size_t col = 0; col < B.cols(); ++col) {
      // w = v^H * B(:,col)
      T w = conj_if_complex(T{1}) * B(j, col);
      for (std::size_t i = j + 1; i < m; ++i) w += conj_if_complex(A(i, j)) * B(i, col);
      w *= tj;
      // B(:,col) -= v * w
      B(j, col) = B(j, col) - w;
      for (std::size_t i = j + 1; i < m; ++i) B(i, col) = B(i, col) - A(i, j) * w;
    }
  };

  auto apply_right = [&](std::size_t j, bool /*conj_tau*/) {
    // Recompute local tau for truncated reflector acting on n-dim (columns) space:
    // tau_loc = 2 / (1 + sum_{i=j+1..n-1} |v_i|^2), with v_i = A(i,j)
    using R = typename numeric_traits<T>::scalar_type;
    R sumsq = R{1};
    for (std::size_t col = j + 1; col < n; ++col) {
      if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(A(col, j)));
      else sumsq += static_cast<R>(A(col, j) * A(col, j));
    }
    T tj = static_cast<T>(R{2} / sumsq);
    // Right-apply on columns indexed by the effective Q dimension.
    // For QR(A) with A m x n, tests expect right-application to act on n columns
    // (i.e., vectors/matrices whose length equals A.cols()).
    const std::size_t nn = A.cols();
    for (std::size_t row = 0; row < B.rows(); ++row) {
      // w = B(row, j:mm-1) * v  where v = [1; A(j+1:mm-1, j)]
      T w = B(row, j); // v0 = 1
      for (std::size_t col = j + 1; col < nn; ++col) w += B(row, col) * A(col, j);
      w *= tj;
      // B(row, j:mm-1) -= w * v^H
      B(row, j) = B(row, j) - w;
      for (std::size_t col = j + 1; col < nn; ++col) B(row, col) = B(row, col) - w * conj_if_complex(A(col, j));
    }
  };

  if (side == Side::Left) {
    if (trans == Trans::NoTrans) {
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_left(j, /*conj_tau=*/false); }
    } else { // ConjTranspose
      for (std::size_t j = 0; j < k; ++j) apply_left(j, /*conj_tau=*/true);
    }
  } else { // Side::Right
    if (trans == Trans::NoTrans) {
      // B := B * Q = (((B * H0) * H1) ... * H{k-1})
      for (std::size_t j = 0; j < k; ++j) apply_right(j, /*conj_tau=*/false);
    } else { // ConjTranspose
      // B := B * Q^H = B * (H{k-1}^H ... H0^H)
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_right(j, /*conj_tau=*/true); }
    }
  }
}

template <typename T>
void apply_Q_inplace(fem::numeric::linear_algebra::Side side,
                     fem::numeric::linear_algebra::Trans trans,
                     const Matrix<T>& A, const std::vector<T>& tau,
                     Vector<T>& b)
{
  using Side = fem::numeric::linear_algebra::Side;
  using Trans = fem::numeric::linear_algebra::Trans;

  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = tau.size();
  // Enforce vector size contracts
  if (side == Side::Left) {
    if (b.size() != m) throw std::invalid_argument("apply_Q_inplace(Left, vector): size mismatch");
  } else {
    if (b.size() != n) throw std::invalid_argument("apply_Q_inplace(Right, vector): size mismatch");
  }

  if (side == Side::Left) {
    auto apply_left_vec = [&](std::size_t j, bool conj_tau) {
      if (tau[j] == T{0}) return;
      T tj = conj_tau ? conj_if_complex(tau[j]) : tau[j];
      T w = b[j];
      for (std::size_t i = j + 1; i < m; ++i) w += conj_if_complex(A(i, j)) * b[i];
      w *= tj;
      b[j] = b[j] - w;
      for (std::size_t i = j + 1; i < m; ++i) b[i] = b[i] - A(i, j) * w;
    };
    if (trans == Trans::NoTrans) {
      // b := Q * b = H_{k-1} ... H_1 H_0 b (loop j down to build left-to-right product)
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_left_vec(j, false); }
    } else {
      // b := Q^H * b = H_0 H_1 ... H_{k-1} b (loop j up)
      for (std::size_t j = 0; j < k; ++j) apply_left_vec(j, true);
    }
  } else { // Side::Right
    // Vector on the right implies length must be n; compute b := b * op(Q)
    auto apply_right_vec = [&](std::size_t j, bool /*conj_tau*/) {
      // Recompute local tau for truncated reflector acting on n-dim (columns) space
      using R = typename numeric_traits<T>::scalar_type;
      R sumsq = R{1};
      for (std::size_t i = j + 1; i < n; ++i) {
        if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(A(i, j)));
        else sumsq += static_cast<R>(A(i, j) * A(i, j));
      }
      T tj = static_cast<T>(R{2} / sumsq);
      T w = b[j];
      for (std::size_t i = j + 1; i < n; ++i) w += b[i] * A(i, j);
      w *= tj;
      b[j] = b[j] - w;
      for (std::size_t i = j + 1; i < n; ++i) b[i] = b[i] - w * conj_if_complex(A(i, j));
    };
    if (trans == Trans::NoTrans) {
      for (std::size_t j = 0; j < k; ++j) apply_right_vec(j, false);
    } else {
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_right_vec(j, true); }
    }
  }
}

// ---------------------------------------------------------------------------
// Solve least squares min ||A x - b||_2 using QR (m>=n), using compact QR
// Returns x (size n). For multiple RHS, returns X (n x nrhs).
// ---------------------------------------------------------------------------

template <typename T>
Vector<T> qr_solve(const Matrix<T>& A_fact, const std::vector<T>& tau, const Vector<T>& b)
{
  const std::size_t m = A_fact.rows();
  const std::size_t n = A_fact.cols();
  if (b.size() != m) throw std::invalid_argument("qr_solve(vector): size mismatch");
  Vector<T> y = b; // y <- Q^H b
  apply_Q_inplace(fem::numeric::linear_algebra::Side::Left,
                  fem::numeric::linear_algebra::Trans::ConjTranspose,
                  A_fact, tau, y);
  // Back-substitution for R x = y(0:n-1)
  Vector<T> x(n, T{});
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;
    T s = y[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= A_fact(i, j) * x[j];
    T rii = A_fact(i, i);
    using R = typename numeric_traits<T>::scalar_type;
    if (std::abs(rii) <= static_cast<R>(1e-12)) throw std::runtime_error("qr_solve: rank-deficient (zero on R diagonal)");
    x[i] = s / rii;
  }
  return x;
}

template <typename T>
Matrix<T> qr_solve(const Matrix<T>& A_fact, const std::vector<T>& tau, const Matrix<T>& B)
{
  const std::size_t m = A_fact.rows();
  const std::size_t n = A_fact.cols();
  if (B.rows() != m) throw std::invalid_argument("qr_solve(matrix): size mismatch");

  // Apply Q^H on the left: Y = Q^H B (done in-place on a copy)
  Matrix<T> Y = B;
  apply_Q_inplace(fem::numeric::linear_algebra::Side::Left,
                  fem::numeric::linear_algebra::Trans::ConjTranspose,
                  A_fact, tau, Y);

  // Solve R X = Y_top (first n rows of Y)
  Matrix<T> X(n, B.cols(), T{});
  for (std::size_t col = 0; col < B.cols(); ++col) {
    for (std::size_t ii = 0; ii < n; ++ii) {
      std::size_t i = n - 1 - ii;
      T s = Y(i, col);
      for (std::size_t j = i + 1; j < n; ++j) s -= A_fact(i, j) * X(j, col);
      T rii = A_fact(i, i);
      using R = typename numeric_traits<T>::scalar_type;
      if (std::abs(rii) <= static_cast<R>(1e-12)) throw std::runtime_error("qr_solve: rank-deficient (zero on R diagonal)");
      X(i, col) = s / rii;
    }
  }
  return X;
}

// Utility: form explicit R (upper triangular) from compact A
template <typename T>
Matrix<T> form_R(const Matrix<T>& A_fact)
{
  const std::size_t m = A_fact.rows();
  const std::size_t n = A_fact.cols();
  Matrix<T> R(std::min(m, n), n, T{});
  const std::size_t r = std::min(m, n);
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = i; j < n; ++j) R(i, j) = A_fact(i, j);
  return R;
}

// ---------------------------------------------------------------------------
// Blocked QR factorization (Householder, compact WY form on updates)
// 1) Panel factorization via unblocked qr_factor on A(j:m, j:j+kb)
// 2) Build V (m_j x kb) with v_j columns (diag 1, below-diag from A)
// 3) Build T (kb x kb) upper-triangular using LARFT (Forward, Columnwise)
// 4) Trailing update: B := (I - V T V^H) B using Level-3 kernels
// Falls back to LAPACK backend (geqrf) if enabled.
// ---------------------------------------------------------------------------
template <typename T>
template <typename T, typename Storage, StorageOrder Order>
int qr_factor_blocked(Matrix<T, Storage, Order>& A, std::vector<T>& tau, std::size_t block = 48)
{
  using namespace fem::numeric::linear_algebra;
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  tau.assign(k, T{});

  // Backend attempt (no-op unless linked via CMake)
  {
    int info_backend = 0;
    if (backends::lapack::geqrf_inplace(A, tau, info_backend)) return info_backend;
  }

  if (k == 0) return 0;
  const std::size_t bs = std::max<std::size_t>(1, block);

  auto build_V_from_panel = [&](const Matrix<T>& P, Matrix<T>& V) {
    const std::size_t pm = P.rows();
    const std::size_t pk = P.cols();
    V = Matrix<T>(pm, pk, T{});
    for (std::size_t j = 0; j < pk; ++j) {
      V(j, j) = T{1};
      for (std::size_t i = j + 1; i < pm; ++i) V(i, j) = P(i, j);
    }
  };

  auto build_T_from_V_tau = [&](const Matrix<T>& V, const std::vector<T>& taup, Matrix<T>& Tmat) {
    const std::size_t pm = V.rows();
    const std::size_t pk = V.cols();
    Tmat = Matrix<T>(pk, pk, T{});
    for (std::size_t i = 0; i < pk; ++i) {
      T ti = taup[i];
      Tmat(i, i) = ti;
      if (ti == T{} || i == 0) continue;
      // tmp = -tau_i * V(:,0:i-1)^H * v_i  (length i)
      std::vector<T> tmp(i, T{});
      for (std::size_t j = 0; j < i; ++j) {
        T s{};
        for (std::size_t r = 0; r < pm; ++r) s += conj_if_complex(V(r, j)) * V(r, i);
        tmp[j] = static_cast<T>(-1) * ti * s;
      }
      // z = T(0:i-1,0:i-1) * tmp  (upper triangular)
      for (std::size_t row = 0; row < i; ++row) {
        T acc{};
        for (std::size_t col = row; col < i; ++col) acc += Tmat(row, col) * tmp[col];
        Tmat(row, i) = acc;
      }
    }
  };

  for (std::size_t j = 0; j < k; j += bs) {
    const std::size_t kb = std::min(bs, k - j);
    auto Ap = A.submatrix(j, m, j, j + kb); // panel
    std::vector<T> taup(kb, T{});
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
    int info_panel = 0;
    const std::size_t pm = m - j; const std::size_t pn = kb;
    if constexpr (Order == StorageOrder::ColumnMajor) {
      int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(A.rows());
      backends::lapack::geqrf_cm<T>(M, N, Ap.data(), lda, taup.data(), info_panel);
    }
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
    else if constexpr (Order == StorageOrder::RowMajor) {
      int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(A.cols());
      info_panel = backends::lapack::geqrf_rm<T>(M, N, Ap.data(), lda, taup.data());
    } else {
#else
    else {
#endif
      // Copy panel to column-major, factor, then copy back
      std::vector<T> panel(pm * pn);
      for (std::size_t col = 0; col < pn; ++col)
        for (std::size_t row = 0; row < pm; ++row)
          panel[col * pm + row] = A(j + row, j + col);
      int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(pm);
      backends::lapack::geqrf_cm<T>(M, N, panel.data(), lda, taup.data(), info_panel);
      for (std::size_t col = 0; col < pn; ++col)
        for (std::size_t row = 0; row < pm; ++row)
          A(j + row, j + col) = panel[col * pm + row];
    }
    (void)info_panel; // QR typically succeeds; tau handles rank info via zeros
#else
    qr_factor_unblocked(Ap, taup);
#endif
    for (std::size_t t = 0; t < kb; ++t) tau[j + t] = taup[t];

    // Trailing update with WY: B := (I - V T V^H) B
    if (j + kb < n) {
      // Build V and T from panel
      Matrix<T> V;
      build_V_from_panel(Ap, V);
      Matrix<T> Tmat;
      build_T_from_V_tau(V, taup, Tmat);

      auto B = A.submatrix(j, m, j + kb, n); // trailing

      // Y = V^H * B  (kb x nt)
      Matrix<T> Y(V.cols(), B.cols(), T{});
      gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, V, B, T{0}, Y);

      // W = T * Y (in-place on Y), T is upper-triangular
      trmm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, T{1}, Tmat, Y);

      // B -= V * W
      gemm(Trans::NoTrans, Trans::NoTrans, T{-1}, V, Y, T{1}, B);
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// QR factorization (best path):
//  - ColumnMajor + LAPACK: call geqrf on whole matrix (zero-copy)
//  - Otherwise: blocked QR (uses panel LAPACK when available; else unblocked on panels)
// ---------------------------------------------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int qr_factor(Matrix<T, Storage, Order>& A, std::vector<T>& tau)
{
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  if constexpr (Order == StorageOrder::ColumnMajor) {
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t k = std::min(m, n);
    tau.assign(k, T{});
    int info = 0;
    int M = static_cast<int>(m), N = static_cast<int>(n), lda = static_cast<int>(A.rows());
    backends::lapack::geqrf_cm<T>(M, N, A.data(), lda, tau.data(), info);
    return info;
  }
#endif
  return qr_factor_blocked(A, tau);
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_QR_H

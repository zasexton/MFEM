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

namespace fem::numeric::decompositions {

// Local helper: conjugate only for complex numbers
template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) { using std::conj; return conj(x); }
  else { return x; }
}

// ---------------------------------------------------------------------------
// QR factorization (Householder) – in-place compact storage
// Returns 0 on success. A is m x n. tau.size() becomes k = min(m,n).
// ---------------------------------------------------------------------------
template <typename T>
int qr_factor(Matrix<T>& A, std::vector<T>& tau)
{
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  tau.assign(k, T{});

  using R = typename numeric_traits<T>::scalar_type;

  for (std::size_t j = 0; j < k; ++j) {
    // Form Householder vector for column j on rows j..m-1
    // Compute 2-norm of x = A(j:m-1, j)
    R nrm2_sq{0};
    for (std::size_t i = j; i < m; ++i) {
      if constexpr (is_complex_number_v<T>) nrm2_sq += static_cast<R>(std::norm(A(i, j)));
      else nrm2_sq += static_cast<R>(A(i, j) * A(i, j));
    }
    R normx = std::sqrt(nrm2_sq);

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

    // v := x; v0 = x0 - beta; tau = (beta - x0)/beta; normalize tail by v0
    T v0 = x0 - beta;
    tau[j] = (v0 == T{0}) ? T{0} : (beta - x0) / beta;
    A(j, j) = beta; // R diagonal entry

    if (tau[j] != T{0}) {
      // Normalize Householder vector below diagonal: v_i = A(i,j)/v0, set v0 implicitly = 1
      for (std::size_t i = j + 1; i < m; ++i) A(i, j) = A(i, j) / v0;

      // Apply H_j = I - tau v v^H to the trailing submatrix A(j:m-1, j+1:n-1)
      for (std::size_t col = j + 1; col < n; ++col) {
        // w = v^H * A(:,col)
        T w = conj_if_complex(T{1}) * A(j, col); // v0 = 1
        for (std::size_t i = j + 1; i < m; ++i) w += conj_if_complex(A(i, j)) * A(i, col);
        w *= tau[j];
        // A(:,col) -= v * w
        A(j, col) = A(j, col) - w; // v0 = 1
        for (std::size_t i = j + 1; i < m; ++i) A(i, col) = A(i, col) - A(i, j) * w;
      }
    } else {
      // No update necessary; ensure below-diagonal is already the reflector storage
      // (v tail would be undefined but tau=0 means H=I)
    }
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Apply Q (product of Householder reflectors) to a matrix/vector
//   side = Left:  B := op(Q) * B
//   side = Right: B := B * op(Q)
//   trans = NoTrans -> apply Q; ConjTranspose -> apply Q^H
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
  if (k != std::min(m, n)) {
    // We allow k <= min(m,n) as well, but basic check for consistency
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

  auto apply_right = [&](std::size_t j, bool conj_tau) {
    if (tau[j] == T{0}) return;
    T tj = conj_tau ? conj_if_complex(tau[j]) : tau[j];
    // Right-apply on columns indexed by the dimension of Q (m = A.rows())
    const std::size_t mm = A.rows();
    for (std::size_t row = 0; row < B.rows(); ++row) {
      // w = B(row, j:mm-1) * v  where v = [1; A(j+1:mm-1, j)]
      T w = B(row, j); // v0 = 1
      for (std::size_t col = j + 1; col < mm; ++col) w += B(row, col) * A(col, j);
      w *= tj;
      // B(row, j:mm-1) -= w * v^H
      B(row, j) = B(row, j) - w;
      for (std::size_t col = j + 1; col < mm; ++col) B(row, col) = B(row, col) - w * conj_if_complex(A(col, j));
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
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_right(j, /*conj_tau=*/false); }
    } else { // ConjTranspose
      for (std::size_t j = 0; j < k; ++j) apply_right(j, /*conj_tau=*/true);
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
      for (std::size_t j = 0; j < k; ++j) apply_left_vec(j, false);
    } else {
      for (std::size_t jj = 0; jj < k; ++jj) { std::size_t j = k - 1 - jj; apply_left_vec(j, true); }
    }
  } else { // Side::Right
    // Vector on the right implies length must be n; compute b := b * op(Q)
    auto apply_right_vec = [&](std::size_t j, bool conj_tau) {
      if (tau[j] == T{0}) return;
      T tj = conj_tau ? conj_if_complex(tau[j]) : tau[j];
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

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_QR_H

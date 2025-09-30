#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_QR_PIVOTED_H
#define NUMERIC_DECOMPOSITIONS_QR_PIVOTED_H

// Column-pivoted QR (CPQR) / Rank-Revealing QR scaffold.
// Provides LAPACK-backed path via xGEQP3 when enabled and a simple unblocked
// reference fallback with greedy column pivoting and norm downdates.
//
// Interface (dense, in-place):
//   - qr_factor_pivoted(A, tau, jpiv, rank, tol)
//       A: on input, m x n; on output, R in upper triangle and Householder
//          reflectors in lower triangle, compact form compatible with apply_Q.
//       tau: scalars for Householder reflectors (size k = min(m, n)).
//       jpiv: column permutation as 0-based indices, length n. A * P = Q R.
//       rank: estimated numerical rank based on R diagonal and tol.
//       tol: if <=0, a default tol = eps * (m + n) * max|R_ii| is used.
//
// Notes:
//   - For multiple RHS or LS solves, use the same interfaces as in qr.h,
//     but apply column permutation to the RHS or solution as appropriate.

#include <vector>
#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <limits>
#include <algorithm>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"

namespace fem::numeric::decompositions {

// Conjugate helper that is a no-op for real scalars.
template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) { using std::conj; return conj(x); }
  else { return x; }
}

// ----------------------------
// LAPACK prototypes (optional)
// ----------------------------
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
extern "C" {
  void sgeqp3_(const int* m, const int* n, float* a, const int* lda,
               int* jpvt, float* tau, float* work, const int* lwork, int* info);
  void dgeqp3_(const int* m, const int* n, double* a, const int* lda,
               int* jpvt, double* tau, double* work, const int* lwork, int* info);
  void cgeqp3_(const int* m, const int* n, std::complex<float>* a, const int* lda,
               int* jpvt, std::complex<float>* tau, std::complex<float>* work, const int* lwork, int* info);
  void zgeqp3_(const int* m, const int* n, std::complex<double>* a, const int* lda,
               int* jpvt, std::complex<double>* tau, std::complex<double>* work, const int* lwork, int* info);
}
#endif

// ----------------------------
// Reference unblocked CPQR
// ----------------------------
template <typename T, typename Storage, StorageOrder Order>
static inline int qr_factor_pivoted_unblocked(Matrix<T, Storage, Order>& A,
                                              std::vector<T>& tau,
                                              std::vector<int>& jpiv,
                                              std::size_t& rank_out,
                                              typename numeric_traits<T>::scalar_type tol)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  tau.assign(k, T{});
  jpiv.resize(n);
  for (std::size_t j = 0; j < n; ++j) jpiv[j] = static_cast<int>(j);
  if (k == 0) { rank_out = 0; return 0; }

  // Column norm estimates and originals
  std::vector<R> cnorm(n, R{}), cnorm0(n, R{});
  auto col_norm2_from = [&](std::size_t col, std::size_t start_row) -> R {
    R s{};
    for (std::size_t i = start_row; i < m; ++i) {
      if constexpr (is_complex_number_v<T>) s += static_cast<R>(std::norm(A(i, col)));
      else s += static_cast<R>(A(i, col) * A(i, col));
    }
    return s;
  };
  for (std::size_t j = 0; j < n; ++j) cnorm0[j] = cnorm[j] = col_norm2_from(j, 0);

  const R eps = std::numeric_limits<R>::epsilon();
  for (std::size_t j = 0; j < k; ++j) {
    // Select pivot by largest remaining column norm
    std::size_t p = j;
    R best = cnorm[j];
    for (std::size_t q = j + 1; q < n; ++q) if (cnorm[q] > best) { best = cnorm[q]; p = q; }
    if (p != j) {
      for (std::size_t i = 0; i < m; ++i) std::swap(A(i, j), A(i, p));
      std::swap(cnorm[j], cnorm[p]);
      std::swap(cnorm0[j], cnorm0[p]);
      std::swap(jpiv[j], jpiv[p]);
    }

    // Householder on column j (rows j..m-1)
    // Compute norm of x = A(j:m-1, j)
    R scale{0}, ssq{1}; bool nz = false;
    auto accum_norm = [&](const T& val) {
      R av = static_cast<R>(std::abs(val));
      if (av == R{0}) return;
      if (!nz) { scale = av; nz = true; return; }
      if (av > scale) { R r = scale / av; ssq = R{1} + ssq * r * r; scale = av; }
      else { R r = av / scale; ssq += r * r; }
    };
    for (std::size_t i = j; i < m; ++i) accum_norm(A(i, j));
    R normx = nz ? scale * std::sqrt(ssq) : R{0};

    if (normx == R{0}) { tau[j] = T{0}; continue; }

    T x0 = A(j, j);
    T beta;
    if constexpr (is_complex_number_v<T>) {
      R a0 = static_cast<R>(std::abs(x0));
      T phase = (a0 == R{0}) ? T{1} : x0 / static_cast<T>(a0);
      beta = static_cast<T>(-normx) * phase;
    } else {
      beta = static_cast<T>(-std::copysign(normx, static_cast<R>(x0)));
    }

    T v0 = x0 - beta;
    A(j, j) = beta; // R(j,j)
    if (v0 != T{0}) {
      for (std::size_t i = j + 1; i < m; ++i) A(i, j) = A(i, j) / v0;
      // tau = 2/(1 + ||v_tail||^2)
      R sumsq = R{0};
      for (std::size_t i = j + 1; i < m; ++i) {
        if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(A(i, j)));
        else sumsq += static_cast<R>(A(i, j) * A(i, j));
      }
      tau[j] = static_cast<T>(R{2} / (R{1} + sumsq));

      // Apply H_j to trailing columns j+1..n-1
      for (std::size_t col = j + 1; col < n; ++col) {
        T w = A(j, col);
        for (std::size_t i = j + 1; i < m; ++i) w += conj_if_complex(A(i, j)) * A(i, col);
        w *= tau[j];
        A(j, col) = A(j, col) - w;
        for (std::size_t i = j + 1; i < m; ++i) A(i, col) = A(i, col) - A(i, j) * w;
      }
    } else {
      tau[j] = T{0};
    }

    // Downdate column norms and occasionally recompute
    for (std::size_t col = j + 1; col < n; ++col) {
      R a = static_cast<R>(std::abs(A(j, col)));
      R a2 = a * a;
      R newv = std::max<R>(R{0}, cnorm[col] - a2);
      cnorm[col] = newv;
      if (newv <= (R{0.001}) * cnorm0[col]) {
        cnorm[col] = col_norm2_from(col, j + 1);
        cnorm0[col] = cnorm[col];
      }
    }
  }

  // Rank estimate
  R rmax = R{0};
  for (std::size_t i = 0; i < k; ++i) {
    R di = static_cast<R>(std::abs(A(i, i)));
    if (di > rmax) rmax = di;
  }
  R thresh = (tol > R{0}) ? tol : (std::numeric_limits<R>::epsilon() * static_cast<R>(m + n) * (rmax > R{0} ? rmax : R{1}));
  std::size_t r = 0; for (; r < k; ++r) { if (static_cast<R>(std::abs(A(r, r))) <= thresh) break; }
  rank_out = r;
  return 0;
}

// -----------------------------------------
// Public API: qr_factor_pivoted (best path)
// -----------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int qr_factor_pivoted(Matrix<T, Storage, Order>& A,
                      std::vector<T>& tau,
                      std::vector<int>& jpiv,
                      std::size_t& rank_out,
                      typename numeric_traits<T>::scalar_type tol = typename numeric_traits<T>::scalar_type(0))
{
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  tau.assign(k, T{});
  jpiv.resize(n);
  for (std::size_t j = 0; j < n; ++j) jpiv[j] = static_cast<int>(j);
  if (k == 0) { rank_out = 0; return 0; }

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Pack to column-major and call xGEQP3
  std::vector<T> a_cm(m * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      a_cm[j * m + i] = A(i, j);

  int M = static_cast<int>(m), N = static_cast<int>(n), LDA = static_cast<int>(m);
  std::vector<int> jpvt_lapack(n, 0); // 0 => free column
  int info = 0;

  // Workspace query
  int lwork = -1;
  if constexpr (std::is_same_v<T, float>) {
    float wkopt; sgeqp3_(&M, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<float*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    std::vector<float> work(std::max(1, lwork));
    sgeqp3_(&M, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<float*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    double wkopt; dgeqp3_(&M, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<double*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(std::max(1, lwork));
    dgeqp3_(&M, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<double*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> wkopt; cgeqp3_(&M, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<std::complex<float>*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    cgeqp3_(&M, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<std::complex<float>*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> wkopt; zgeqp3_(&M, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<std::complex<double>*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    zgeqp3_(&M, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, jpvt_lapack.data(), reinterpret_cast<std::complex<double>*>(tau.data()), work.data(), &lwork, &info);
  } else {
    // Unsupported type by backend -> fallback
    return qr_factor_pivoted_unblocked(A, tau, jpiv, rank_out, tol);
  }
  if (info != 0) return info;

  // Copy result back to row-major A
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      A(i, j) = a_cm[j * m + i];

  // Convert LAPACK 1-based jpvt to 0-based permutation
  jpiv.resize(n);
  for (std::size_t j = 0; j < n; ++j) jpiv[j] = jpvt_lapack[j] - 1;

  // Rank estimate from R diagonal
  using R = typename numeric_traits<T>::scalar_type;
  R rmax = R{0};
  const std::size_t kk = k;
  for (std::size_t i = 0; i < kk; ++i) {
    R di = static_cast<R>(std::abs(A(i, i)));
    if (di > rmax) rmax = di;
  }
  R thresh = (tol > R{0}) ? tol : (std::numeric_limits<R>::epsilon() * static_cast<R>(m + n) * (rmax > R{0} ? rmax : R{1}));
  std::size_t r = 0; for (; r < kk; ++r) { if (static_cast<R>(std::abs(A(r, r))) <= thresh) break; }
  rank_out = r;
  return 0;
#else
  return qr_factor_pivoted_unblocked(A, tau, jpiv, rank_out, tol);
#endif
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_QR_PIVOTED_H


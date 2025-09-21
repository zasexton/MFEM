#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_SPARSE_OPS_H
#define NUMERIC_LINEAR_ALGEBRA_SPARSE_OPS_H

// Sparse linear algebra operations (CSR/CSC SpMV and SpMM) in a header-only form.
// The routines are minimal and portable, suitable as defaults before delegating
// to optimized backends.

#include <cstddef>
#include <type_traits>
#include <stdexcept>

namespace fem::numeric::linear_algebra {

// ---------------------------------------------------------------------------
// CSR SpMV: y := alpha * A * x + beta * y
// A in CSR: row_ptr (size m+1), col_ind (size nnz), val (size nnz)
// ---------------------------------------------------------------------------

template <typename Alpha, typename Index, typename TA, typename TX, typename Beta, typename TY>
inline void spmv_csr(std::size_t m, std::size_t n, const Alpha& alpha,
                     const Index* row_ptr, const Index* col_ind, const TA* val,
                     const TX* x, const Beta& beta, TY* y)
{
  if (!row_ptr || !col_ind || !val || !x || !y) return;
  // Scale y
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < m; ++i) y[i] = TY{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < m; ++i) y[i] = static_cast<TY>(beta * y[i]);
  }
  if (alpha == Alpha{}) return;
  for (std::size_t i = 0; i < m; ++i) {
    auto sum = decltype(alpha * val[0] * x[0]){};
    Index p0 = row_ptr[i];
    Index p1 = row_ptr[i + 1];
    for (Index p = p0; p < p1; ++p) {
      const auto j = static_cast<std::size_t>(col_ind[p]);
      sum += static_cast<decltype(sum)>(val[p]) * static_cast<decltype(sum)>(x[j]);
    }
    y[i] = static_cast<TY>(y[i] + alpha * sum);
  }
}

// Transposed CSR SpMV (y := alpha * A^T * x + beta * y)
template <typename Alpha, typename Index, typename TA, typename TX, typename Beta, typename TY>
inline void spmv_csr_transpose(std::size_t m, std::size_t n, const Alpha& alpha,
                               const Index* row_ptr, const Index* col_ind, const TA* val,
                               const TX* x, const Beta& beta, TY* y)
{
  if (!row_ptr || !col_ind || !val || !x || !y) return;
  // Scale y
  if (beta == Beta{}) {
    for (std::size_t j = 0; j < n; ++j) y[j] = TY{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t j = 0; j < n; ++j) y[j] = static_cast<TY>(beta * y[j]);
  }
  if (alpha == Alpha{}) return;
  for (std::size_t i = 0; i < m; ++i) {
    auto xi = x[i];
    if (xi == TX{}) continue;
    Index p0 = row_ptr[i];
    Index p1 = row_ptr[i + 1];
    for (Index p = p0; p < p1; ++p) {
      const auto j = static_cast<std::size_t>(col_ind[p]);
      y[j] = static_cast<TY>(y[j] + alpha * static_cast<decltype(alpha * val[0])>(val[p]) * xi);
    }
  }
}

// ---------------------------------------------------------------------------
// CSC SpMV: y := alpha * A * x + beta * y
// A in CSC: col_ptr (size n+1), row_ind (size nnz), val (size nnz)
// ---------------------------------------------------------------------------

template <typename Alpha, typename Index, typename TA, typename TX, typename Beta, typename TY>
inline void spmv_csc(std::size_t m, std::size_t n, const Alpha& alpha,
                     const Index* col_ptr, const Index* row_ind, const TA* val,
                     const TX* x, const Beta& beta, TY* y)
{
  if (!col_ptr || !row_ind || !val || !x || !y) return;
  // Scale y
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < m; ++i) y[i] = TY{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < m; ++i) y[i] = static_cast<TY>(beta * y[i]);
  }
  if (alpha == Alpha{}) return;
  for (std::size_t j = 0; j < n; ++j) {
    auto xj = x[j];
    if (xj == TX{}) continue;
    Index p0 = col_ptr[j];
    Index p1 = col_ptr[j + 1];
    for (Index p = p0; p < p1; ++p) {
      const auto i = static_cast<std::size_t>(row_ind[p]);
      y[i] = static_cast<TY>(y[i] + alpha * static_cast<decltype(alpha * val[0])>(val[p]) * xj);
    }
  }
}

// ---------------------------------------------------------------------------
// CSR SpMM: C := alpha * A * B + beta * C
// A (m x k) CSR, B (k x n) dense (row-major or col-major handled by leading dim)
// C (m x n) dense
// ---------------------------------------------------------------------------

enum class DenseLayout { RowMajor, ColMajor };

template <typename Alpha, typename Index, typename TA, typename TB, typename Beta, typename TC>
inline void spmm_csr(std::size_t m, std::size_t k, std::size_t n,
                     const Alpha& alpha,
                     const Index* row_ptr, const Index* col_ind, const TA* aval,
                     const TB* B, std::size_t ldb, DenseLayout layoutB,
                     const Beta& beta,
                     TC* C, std::size_t ldc, DenseLayout layoutC)
{
  if (!row_ptr || !col_ind || !aval || !B || !C) return;

  auto idxB = [&](std::size_t i, std::size_t j) -> TB {
    return (layoutB == DenseLayout::RowMajor) ? B[i * ldb + j] : B[i + j * ldb];
  };
  auto refC = [&](std::size_t i, std::size_t j) -> TC& {
    return (layoutC == DenseLayout::RowMajor) ? C[i * ldc + j] : C[i + j * ldc];
  };

  // Scale C
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) refC(i, j) = TC{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) refC(i, j) = static_cast<TC>(beta * refC(i, j));
  }
  if (alpha == Alpha{}) return;

  for (std::size_t i = 0; i < m; ++i) {
    const Index p0 = row_ptr[i];
    const Index p1 = row_ptr[i + 1];
    for (Index p = p0; p < p1; ++p) {
      const auto kcol = static_cast<std::size_t>(col_ind[p]);
      const auto a = static_cast<decltype(alpha * aval[0])>(alpha * aval[p]);
      for (std::size_t j = 0; j < n; ++j) {
        refC(i, j) = static_cast<TC>(refC(i, j) + a * idxB(kcol, j));
      }
    }
  }
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_SPARSE_OPS_H


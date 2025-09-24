// Lightweight LAPACK/MKL/cuSOLVER backend shims for optional acceleration.
// These compile to no-ops unless FEM_NUMERIC_ENABLE_LAPACK (or CUDA) is defined
// and CMake links the appropriate libraries.

#pragma once

#ifndef NUMERIC_BACKENDS_LAPACK_BACKEND_H
#define NUMERIC_BACKENDS_LAPACK_BACKEND_H

#include <vector>
#include <type_traits>

#include "../core/matrix.h"
#include "../linear_algebra/blas_level2.h"
#include <complex>
#include <vector>

namespace fem::numeric::backends::lapack {

// Low-level column-major wrappers (used for tile/column-major fast paths)
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
extern "C" {
  void spotrf_(const char*, const int*, float*, const int*, int*);
  void dpotrf_(const char*, const int*, double*, const int*, int*);
  void cpotrf_(const char*, const int*, std::complex<float>*, const int*, int*);
  void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*);

  void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
  void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
  void cgetrf_(const int*, const int*, std::complex<float>*, const int*, int*, int*);
  void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);

  void sgeqrf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
  void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
  void cgeqrf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*, std::complex<float>*, const int*, int*);
  void zgeqrf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*, std::complex<double>*, const int*, int*);
  // DLARFT (form T for block of reflectors): Forward, Columnwise variants used.
  void slarft_(const char*, const char*, const int*, const int*, const float*, const int*, const float*, float*, const int*);
  void dlarft_(const char*, const char*, const int*, const int*, const double*, const int*, const double*, double*, const int*);
  void clarft_(const char*, const char*, const int*, const int*, const std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*);
  void zlarft_(const char*, const char*, const int*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*);
}

template <typename T>
inline void potrf_cm(char uplo, int n, T* a, int lda, int& info)
{
  if constexpr (std::is_same_v<T, float>)      spotrf_(&uplo, &n, reinterpret_cast<float*>(a), &lda, &info);
  else if constexpr (std::is_same_v<T, double>) dpotrf_(&uplo, &n, reinterpret_cast<double*>(a), &lda, &info);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  cpotrf_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, &info);
  else if constexpr (std::is_same_v<T, std::complex<double>>) zpotrf_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, &info);
}

template <typename T>
inline void getrf_cm(int m, int n, T* a, int lda, int* ipiv, int& info)
{
  if constexpr (std::is_same_v<T, float>)      sgetrf_(&m, &n, reinterpret_cast<float*>(a), &lda, ipiv, &info);
  else if constexpr (std::is_same_v<T, double>) dgetrf_(&m, &n, reinterpret_cast<double*>(a), &lda, ipiv, &info);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  cgetrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, &info);
  else if constexpr (std::is_same_v<T, std::complex<double>>) zgetrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, &info);
}

template <typename T>
inline void geqrf_cm(int m, int n, T* a, int lda, T* tau, int& info)
{
  int lwork = -1;
  T wkopt{};
  if constexpr (std::is_same_v<T, float>)      sgeqrf_(&m, &n, reinterpret_cast<float*>(a), &lda, reinterpret_cast<float*>(tau), reinterpret_cast<float*>(&wkopt), &lwork, &info);
  else if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, reinterpret_cast<double*>(a), &lda, reinterpret_cast<double*>(tau), reinterpret_cast<double*>(&wkopt), &lwork, &info);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(&wkopt), &lwork, &info);
  else if constexpr (std::is_same_v<T, std::complex<double>>) zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(&wkopt), &lwork, &info);
  if (info != 0) return;
  int opt = 0;
  if constexpr (!is_complex_number_v<T>) opt = static_cast<int>(wkopt);
  else opt = static_cast<int>(wkopt.real());
  lwork = (opt > 0) ? opt : std::max(32, n) * 4;
  std::vector<T> work(static_cast<std::size_t>(lwork));
  if constexpr (std::is_same_v<T, float>)      sgeqrf_(&m, &n, reinterpret_cast<float*>(a), &lda, reinterpret_cast<float*>(tau), reinterpret_cast<float*>(work.data()), &lwork, &info);
  else if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, reinterpret_cast<double*>(a), &lda, reinterpret_cast<double*>(tau), reinterpret_cast<double*>(work.data()), &lwork, &info);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work.data()), &lwork, &info);
  else if constexpr (std::is_same_v<T, std::complex<double>>) zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work.data()), &lwork, &info);
}

// Lightweight wrapper to form T from V and tau for a block of reflectors
template <typename T>
inline void larft_cm(char direct, char storev, int n, int k,
                     const T* V, int ldv,
                     const T* tau,
                     T* Tm, int ldt)
{
  if constexpr (std::is_same_v<T, float>)      slarft_(&direct, &storev, &n, &k, reinterpret_cast<const float*>(V), &ldv, reinterpret_cast<const float*>(tau), reinterpret_cast<float*>(Tm), &ldt);
  else if constexpr (std::is_same_v<T, double>) dlarft_(&direct, &storev, &n, &k, reinterpret_cast<const double*>(V), &ldv, reinterpret_cast<const double*>(tau), reinterpret_cast<double*>(Tm), &ldt);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  clarft_(&direct, &storev, &n, &k, reinterpret_cast<const std::complex<float>*>(V), &ldv, reinterpret_cast<const std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(Tm), &ldt);
  else if constexpr (std::is_same_v<T, std::complex<double>>) zlarft_(&direct, &storev, &n, &k, reinterpret_cast<const std::complex<double>*>(V), &ldv, reinterpret_cast<const std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(Tm), &ldt);
}
#endif // FEM_NUMERIC_ENABLE_LAPACK

// LAPACKE row-major wrappers (optional)
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
#include <lapacke.h>

template <typename T>
inline int potrf_rm(char uplo, int n, T* a, int lda)
{
  if constexpr (std::is_same_v<T, float>)      return LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, reinterpret_cast<float*>(a), lda);
  else if constexpr (std::is_same_v<T, double>) return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, reinterpret_cast<double*>(a), lda);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  return LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n, reinterpret_cast<lapack_complex_float*>(a), lda);
  else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zpotrf(LAPACK_ROW_MAJOR, uplo, n, reinterpret_cast<lapack_complex_double*>(a), lda);
  else return -1;
}

template <typename T>
inline int getrf_rm(int m, int n, T* a, int lda, int* ipiv)
{
  if constexpr (std::is_same_v<T, float>)      return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<float*>(a), lda, ipiv);
  else if constexpr (std::is_same_v<T, double>) return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<double*>(a), lda, ipiv);
  else if constexpr (std::is_same_v<T, std::complex<float>>)  return LAPACKE_cgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda, ipiv);
  else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda, ipiv);
  else return -1;
}

template <typename T>
inline int geqrf_rm(int m, int n, T* a, int lda, T* tau)
{
  if constexpr (std::is_same_v<T, float>)      return LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<float*>(a), lda, reinterpret_cast<float*>(tau));
  else if constexpr (std::is_same_v<T, double>) return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<double*>(a), lda, reinterpret_cast<double*>(tau));
  else if constexpr (std::is_same_v<T, std::complex<float>>)  return LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda, reinterpret_cast<lapack_complex_float*>(tau));
  else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda, reinterpret_cast<lapack_complex_double*>(tau));
  else return -1;
}

#endif // FEM_NUMERIC_ENABLE_LAPACKE

// Returns true if backend handled the call; false => caller should fallback.
// When true is returned, info_out holds the LAPACK info result.

// -----------------
// Cholesky: POTRF
// -----------------
template <typename T, typename Storage, StorageOrder Order>
inline bool potrf_inplace(Matrix<T, Storage, Order>& A,
                          fem::numeric::linear_algebra::Uplo uplo,
                          int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)uplo; info_out = 0; return false;
#else
  using namespace fem::numeric::linear_algebra;
  const std::size_t n = A.rows();
  if (A.cols() != n) return false;
  if (n == 0) { info_out = 0; return true; }

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    char u = (uplo == Uplo::Lower) ? 'L' : 'U';
    int info = potrf_rm<T>(u, static_cast<int>(n), A.data(), static_cast<int>(A.cols()));
    info_out = info;
    return true;
  }
#endif

  // Pack to column-major buffer
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);

  char u = (uplo == Uplo::Lower) ? 'L' : 'U';
  int N = static_cast<int>(n);
  int LDA = static_cast<int>(n);
  int info = 0;

  extern "C" {
    void spotrf_(const char*, const int*, float*, const int*, int*);
    void dpotrf_(const char*, const int*, double*, const int*, int*);
    void cpotrf_(const char*, const int*, std::complex<float>*, const int*, int*);
    void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*);
  }

  if constexpr (std::is_same_v<T, float>)      { spotrf_(&u, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, &info); }
  else if constexpr (std::is_same_v<T, double>) { dpotrf_(&u, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, &info); }
  else if constexpr (std::is_same_v<T, std::complex<float>>)  { cpotrf_(&u, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, &info); }
  else if constexpr (std::is_same_v<T, std::complex<double>>) { zpotrf_(&u, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, &info); }
  else { return false; }

  // Copy back (zero the opposite triangle for consistency)
  if (info >= 0) {
    if (uplo == Uplo::Lower) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          if (j <= i) A(i, j) = a_cm[j * n + i];
          else A(i, j) = T{};
        }
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          if (j >= i) A(i, j) = a_cm[j * n + i];
          else A(i, j) = T{};
        }
      }
    }
  }
  info_out = info;
  return true;
#endif
}

// ---------
// LU: GETRF
// ---------
template <typename T, typename Storage, StorageOrder Order>
inline bool getrf_inplace(Matrix<T, Storage, Order>& A, std::vector<int>& piv, int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)piv; info_out = 0; return false;
#else
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  if (m == 0 || n == 0) { piv.clear(); info_out = 0; return true; }

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    int M = static_cast<int>(m), N = static_cast<int>(n), lda = static_cast<int>(A.cols());
    const std::size_t k = std::min(m, n);
    piv.resize(k);
    int info = getrf_rm<T>(M, N, A.data(), lda, piv.data());
    // LAPACKE returns 1-based pivots; convert to 0-based
    for (std::size_t i = 0; i < k; ++i) piv[i] -= 1;
    info_out = info;
    return true;
  }
#endif

  std::vector<T> a_cm(m * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      a_cm[j * m + i] = A(i, j);

  int M = static_cast<int>(m), N = static_cast<int>(n), LDA = static_cast<int>(m);
  int k = static_cast<int>(std::min(m, n));
  std::vector<int> ipiv(std::max(1, k));
  int info = 0;

  extern "C" {
    void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
    void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
    void cgetrf_(const int*, const int*, std::complex<float>*, const int*, int*, int*);
    void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);
  }

  if constexpr (std::is_same_v<T, float>)      { sgetrf_(&M, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, ipiv.data(), &info); }
  else if constexpr (std::is_same_v<T, double>) { dgetrf_(&M, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, ipiv.data(), &info); }
  else if constexpr (std::is_same_v<T, std::complex<float>>)  { cgetrf_(&M, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, ipiv.data(), &info); }
  else if constexpr (std::is_same_v<T, std::complex<double>>) { zgetrf_(&M, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, ipiv.data(), &info); }
  else { return false; }

  // Copy back factorized LU
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      A(i, j) = a_cm[j * m + i];

  // Convert 1-based pivot rows to 0-based
  piv.resize(k);
  for (int i = 0; i < k; ++i) piv[static_cast<std::size_t>(i)] = ipiv[i] - 1;

  info_out = info;
  return true;
#endif
}

// ---------
// QR: GEQRF
// ---------
template <typename T, typename Storage, StorageOrder Order>
inline bool geqrf_inplace(Matrix<T, Storage, Order>& A, std::vector<T>& tau, int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)tau; info_out = 0; return false;
#else
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t k = std::min(m, n);
  if (k == 0) { tau.clear(); info_out = 0; return true; }

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    int M = static_cast<int>(m), N = static_cast<int>(n), lda = static_cast<int>(A.cols());
    tau.assign(k, T{});
    int info = geqrf_rm<T>(M, N, A.data(), lda, tau.data());
    info_out = info;
    return true;
  }
#endif

  std::vector<T> a_cm(m * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      a_cm[j * m + i] = A(i, j);

  int M = static_cast<int>(m), N = static_cast<int>(n), LDA = static_cast<int>(m);
  tau.assign(k, T{});
  int info = 0;

  // Workspace query
  int lwork = -1;
  // Work scalar of appropriate type
  T wkopt{};

  extern "C" {
    void sgeqrf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
    void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
    void cgeqrf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*, std::complex<float>*, const int*, int*);
    void zgeqrf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*, std::complex<double>*, const int*, int*);
  }

  if constexpr (std::is_same_v<T, float>) {
    sgeqrf_(&M, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(tau.data()), reinterpret_cast<float*>(&wkopt), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    dgeqrf_(&M, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(tau.data()), reinterpret_cast<double*>(&wkopt), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    cgeqrf_(&M, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<std::complex<float>*>(tau.data()), reinterpret_cast<std::complex<float>*>(&wkopt), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    zgeqrf_(&M, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<std::complex<double>*>(tau.data()), reinterpret_cast<std::complex<double>*>(&wkopt), &lwork, &info);
  } else {
    return false;
  }
  if (info != 0) { info_out = info; return true; }

  // Allocate optimal work
  // For real, wkopt’s real part holds lwork; for complex, real part as well
  int opt = 0;
  if constexpr (!is_complex_number_v<T>) {
    opt = static_cast<int>(wkopt);
  } else {
    opt = static_cast<int>(wkopt.real());
  }
  lwork = std::max(opt, 32 * static_cast<int>(k));
  std::vector<T> work(static_cast<std::size_t>(lwork));

  // Actual factorization
  if constexpr (std::is_same_v<T, float>) {
    sgeqrf_(&M, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(tau.data()), reinterpret_cast<float*>(work.data()), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    dgeqrf_(&M, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(tau.data()), reinterpret_cast<double*>(work.data()), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    cgeqrf_(&M, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<std::complex<float>*>(tau.data()), reinterpret_cast<std::complex<float>*>(work.data()), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    zgeqrf_(&M, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<std::complex<double>*>(tau.data()), reinterpret_cast<std::complex<double>*>(work.data()), &lwork, &info);
  }

  // Copy back to row-major A
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      A(i, j) = a_cm[j * m + i];

  info_out = info;
  return true;
#endif
}

} // namespace fem::numeric::backends::lapack

#endif // NUMERIC_BACKENDS_LAPACK_BACKEND_H

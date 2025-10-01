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

  // --- Modern symmetric/Hermitian eigensolvers ---
  // Real symmetric: SYEVR (MRRR) and SYEVD (divide & conquer)
  void ssyevr_(const char*, const char*, const char*, const int*, float*, const int*, const float*, const float*, const int*, const int*, const float*, int*, float*, float*, const int*, int*, float*, const int*, int*, const int*, int*);
  void dsyevr_(const char*, const char*, const char*, const int*, double*, const int*, const double*, const double*, const int*, const int*, const double*, int*, double*, double*, const int*, int*, double*, const int*, int*, const int*, int*);
  void ssyevd_(const char*, const char*, const int*, float*, const int*, float*, float*, const int*, int*, const int*, int*);
  void dsyevd_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*, const int*, int*);

  // Complex Hermitian: HEEVR and HEEVD
  void cheevr_(const char*, const char*, const char*, const int*, std::complex<float>*, const int*, const float*, const float*, const int*, const int*, const float*, int*, float*, std::complex<float>*, const int*, int*, std::complex<float>*, const int*, float*, const int*, int*, const int*, int*);
  void zheevr_(const char*, const char*, const char*, const int*, std::complex<double>*, const int*, const double*, const double*, const int*, const int*, const double*, int*, double*, std::complex<double>*, const int*, int*, std::complex<double>*, const int*, double*, const int*, int*, const int*, int*);
  void cheevd_(const char*, const char*, const int*, std::complex<float>*, const int*, float*, std::complex<float>*, const int*, float*, const int*, int*, const int*, int*);
  void zheevd_(const char*, const char*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, double*, const int*, int*, const int*, int*);

  // Symmetric tridiagonal: STEVR (MRRR) and STEVD (D&C), real only
  void sstevr_(const char*, const char*, const int*, float*, float*, const float*, const float*, const int*, const int*, const float*, int*, float*, float*, const int*, int*, float*, const int*, int*, const int*, int*);
  void dstevr_(const char*, const char*, const int*, double*, double*, const double*, const double*, const int*, const int*, const double*, int*, double*, double*, const int*, int*, double*, const int*, int*, const int*, int*);
  void sstevd_(const char*, const int*, float*, float*, float*, const int*, float*, const int*, int*, const int*, int*);
  void dstevd_(const char*, const int*, double*, double*, double*, const int*, double*, const int*, int*, const int*, int*);

  // Symmetric/Hermitian tridiagonalization and Q formation
  void ssytrd_(const char*, const int*, float*, const int*, float*, float*, float*, float*, const int*, int*);
  void dsytrd_(const char*, const int*, double*, const int*, double*, double*, double*, double*, const int*, int*);
  void chetrd_(const char*, const int*, std::complex<float>*, const int*, float*, float*, std::complex<float>*, std::complex<float>*, const int*, int*);
  void zhetrd_(const char*, const int*, std::complex<double>*, const int*, double*, double*, std::complex<double>*, std::complex<double>*, const int*, int*);

  void sorgtr_(const char*, const int*, float*, const int*, const float*, float*, const int*, int*);
  void dorgtr_(const char*, const int*, double*, const int*, const double*, double*, const int*, int*);
  void cungtr_(const char*, const int*, std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*, int*);
  void zungtr_(const char*, const int*, std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*, int*);
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

// Backward-compatible overload defaulting to Upper for SYTRD + Q
template <typename T, typename Storage, StorageOrder Order>
inline bool sytrd_tridiag_with_Q(const Matrix<T, Storage, Order>& A,
                                 std::vector<typename fem::numeric::numeric_traits<T>::scalar_type>& d,
                                 std::vector<typename fem::numeric::numeric_traits<T>::scalar_type>& e,
                                 Matrix<T, Storage, Order>* Qopt,
                                 int& info_out)
{
  return sytrd_tridiag_with_Q(A, d, e, Qopt, fem::numeric::linear_algebra::Uplo::Upper, info_out);
}

// Backward-compatible overload defaulting to Upper triangle for EVD (non-destructive)
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evd(const Matrix<T, Storage, Order>& A,
                         Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                         Matrix<T, Storage, Order>& eigenvectors,
                         bool compute_vectors,
                         int& info_out)
{
  return eigh_via_evd(A, evals, eigenvectors, compute_vectors, fem::numeric::linear_algebra::Uplo::Upper, info_out);
}

// Backward-compatible overload defaulting to Upper triangle for range-select EVR
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evr_range(const Matrix<T, Storage, Order>& A,
                               Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                               Matrix<T, Storage, Order>& eigenvectors,
                               bool compute_vectors,
                               char range_sel,
                               typename fem::numeric::numeric_traits<T>::scalar_type vl,
                               typename fem::numeric::numeric_traits<T>::scalar_type vu,
                               int il, int iu,
                               int& info_out,
                               int* m_out)
{
  return eigh_via_evr_range(A, evals, eigenvectors, compute_vectors, fem::numeric::linear_algebra::Uplo::Upper, range_sel, vl, vu, il, iu, info_out, m_out);
}

// Backward-compatible overloads defaulting to Upper triangle
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evr(const Matrix<T, Storage, Order>& A,
                         Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                         Matrix<T, Storage, Order>& eigenvectors,
                         bool compute_vectors,
                         int& info_out,
                         int* m_out)
{
  return eigh_via_evr(A, evals, eigenvectors, compute_vectors, fem::numeric::linear_algebra::Uplo::Upper, info_out, m_out);
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

// -----------------------------
// Eigen (symmetric/Hermitian)
// -----------------------------

// Prefer MRRR (SYEVR/HEEVR) when available; fall back to divide & conquer (SYEVD/HEEVD).
// Returns true if a backend path was attempted. info_out holds LAPACK info if true.
// uplo_sel selects which triangle of A is read (Upper or Lower).
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evr(const Matrix<T, Storage, Order>& A,
                         Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                         Matrix<T, Storage, Order>& eigenvectors,
                         bool compute_vectors,
                         fem::numeric::linear_algebra::Uplo uplo_sel,
                         int& info_out,
                         int* m_out = nullptr)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)evals; (void)eigenvectors; (void)compute_vectors; info_out = 0; return false;
#else
  using R = typename fem::numeric::numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) return false;
  if (n == 0) { evals = Vector<R>(0); eigenvectors = Matrix<T, Storage, Order>(0,0,T{}); info_out = 0; return true; }

  // RowMajor fast path via LAPACKE (avoid pack) when available
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    char jobz = compute_vectors ? 'V' : 'N';
    char range = 'A';
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
    int N = static_cast<int>(n);
    evals = Vector<R>(n);
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    int M = 0;
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_ssyevr(LAPACK_ROW_MAJOR, jobz, range, uplo, N,
                            reinterpret_cast<float*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            0.0f, 0.0f, 0, 0, 0.0f, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<float*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, double>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, jobz, range, uplo, N,
                            reinterpret_cast<double*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            0.0, 0.0, 0, 0, 0.0, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<double*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_cheevr(LAPACK_ROW_MAJOR, jobz, range, uplo, N,
                            reinterpret_cast<lapack_complex_float*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            0.0f, 0.0f, 0, 0, 0.0f, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<lapack_complex_float*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_zheevr(LAPACK_ROW_MAJOR, jobz, range, uplo, N,
                            reinterpret_cast<lapack_complex_double*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            0.0, 0.0, 0, 0, 0.0, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<lapack_complex_double*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else {
      info_out = -1; return true;
    }
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (!compute_vectors) {
        eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
      } else if (M < static_cast<int>(n)) {
        eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
      }
    }
    return true;
  }
#endif

  // Pack to column-major
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);

  char jobz = compute_vectors ? 'V' : 'N';
  char range = 'A';
  char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
  int N = static_cast<int>(n);
  int LDA = static_cast<int>(n);
  R vl = R{}; R vu = R{}; int il = 0, iu = 0;
  R abstol = R{}; // default
  int M = 0;
  evals = Vector<R>(n);
  int info = 0;

  if constexpr (std::is_same_v<T, float>) {
    // Workspace query
    int lwork = -1, liwork = -1;
    float wkopt; int iwkopt;
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    ssyevr_(&jobz, &range, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu,
            reinterpret_cast<float*>(&abstol), &M, evals.data(), nullptr, &N, isuppz.data(),
            &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<float> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    ssyevr_(&jobz, &range, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu,
            reinterpret_cast<float*>(&abstol), &M, evals.data(), compute_vectors ? reinterpret_cast<float*>(eigenvectors.data()) : nullptr, &N,
            isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0 && compute_vectors && M < static_cast<int>(n)) {
      eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, double>) {
    int lwork = -1, liwork = -1;
    double wkopt; int iwkopt;
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    dsyevr_(&jobz, &range, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), nullptr, &N, isuppz.data(),
            &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<double> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    dsyevr_(&jobz, &range, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), compute_vectors ? reinterpret_cast<double*>(eigenvectors.data()) : nullptr, &N,
            isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0 && compute_vectors && M < static_cast<int>(n)) {
      eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    int lwork = -1, liwork = -1, lrwork = -1;
    std::complex<float> wkopt; int iwkopt; float rwkopt;
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    cheevr_(&jobz, &range, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real()); liwork = iwkopt; lrwork = static_cast<int>(rwkopt);
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    std::vector<float> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    cheevr_(&jobz, &range, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), compute_vectors ? reinterpret_cast<std::complex<float>*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0 && compute_vectors && M < static_cast<int>(n)) {
      eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    int lwork = -1, liwork = -1, lrwork = -1;
    std::complex<double> wkopt; int iwkopt; double rwkopt;
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    zheevr_(&jobz, &range, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), nullptr, &N, isuppz.data(),
            &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real()); liwork = iwkopt; lrwork = static_cast<int>(rwkopt);
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    std::vector<double> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    zheevr_(&jobz, &range, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), compute_vectors ? reinterpret_cast<std::complex<double>*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (info == 0 && compute_vectors && M < static_cast<int>(n)) {
      eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else {
    return false;
  }
#endif
}

// Range-aware EVR (MRRR): select by index or value
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evr_range(const Matrix<T, Storage, Order>& A,
                               Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                               Matrix<T, Storage, Order>& eigenvectors,
                               bool compute_vectors,
                               fem::numeric::linear_algebra::Uplo uplo_sel,
                               char range_sel, // 'A', 'I', or 'V'
                               typename fem::numeric::numeric_traits<T>::scalar_type vl,
                               typename fem::numeric::numeric_traits<T>::scalar_type vu,
                               int il, int iu, // 1-based for LAPACK when range='I'
                               int& info_out,
                               int* m_out = nullptr)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)evals; (void)eigenvectors; (void)compute_vectors; (void)range_sel; (void)vl; (void)vu; (void)il; (void)iu; info_out = 0; return false;
#else
  using R = typename fem::numeric::numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) return false;
  if (n == 0) { evals = Vector<R>(0); eigenvectors = Matrix<T, Storage, Order>(0,0,T{}); info_out = 0; return true; }

  // RowMajor fast path via LAPACKE to avoid packing, when available
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    char jobz = compute_vectors ? 'V' : 'N';
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
    int N = static_cast<int>(n);
    evals = Vector<R>(n);
    std::vector<int> isuppz(2 * std::max<int>(1, N));
    int M = 0; int info = 0;
    if constexpr (std::is_same_v<T, float>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_ssyevr(LAPACK_ROW_MAJOR, jobz, range_sel, uplo, N,
                            reinterpret_cast<float*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            static_cast<float>(vl), static_cast<float>(vu), il, iu, 0.0f, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<float*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, double>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, jobz, range_sel, uplo, N,
                            reinterpret_cast<double*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            static_cast<double>(vl), static_cast<double>(vu), il, iu, 0.0, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<double*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_cheevr(LAPACK_ROW_MAJOR, jobz, range_sel, uplo, N,
                            reinterpret_cast<lapack_complex_float*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            static_cast<float>(vl), static_cast<float>(vu), il, iu, 0.0f, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<lapack_complex_float*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      info = LAPACKE_zheevr(LAPACK_ROW_MAJOR, jobz, range_sel, uplo, N,
                            reinterpret_cast<lapack_complex_double*>(const_cast<T*>(A.data())), static_cast<int>(A.cols()),
                            static_cast<double>(vl), static_cast<double>(vu), il, iu, 0.0, &M,
                            evals.data(), compute_vectors ? reinterpret_cast<lapack_complex_double*>(eigenvectors.data()) : nullptr, static_cast<int>(n),
                            isuppz.data());
    } else {
      info_out = -1; return true;
    }
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (!compute_vectors) {
        eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
      } else if (M < static_cast<int>(n)) {
        eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
      }
    }
    return true;
  }
#endif

  // Pack to column-major
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);

  char jobz = compute_vectors ? 'V' : 'N';
  char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
  int N = static_cast<int>(n);
  int LDA = static_cast<int>(n);
  R abstol = R{};
  int M = 0;
  evals = Vector<R>(n);
  int info = 0;

  if constexpr (std::is_same_v<T, float>) {
    int lwork = -1, liwork = -1; float wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max<int>(1, N));
    ssyevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<float> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    ssyevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), compute_vectors ? reinterpret_cast<float*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (compute_vectors && M < static_cast<int>(n)) eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    }
    if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, double>) {
    int lwork = -1, liwork = -1; double wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max<int>(1, N));
    dsyevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<double> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    dsyevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), compute_vectors ? reinterpret_cast<double*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (compute_vectors && M < static_cast<int>(n)) eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    }
    if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    int lwork = -1, liwork = -1, lrwork = -1; std::complex<float> wkopt; int iwkopt; float rwkopt; std::vector<int> isuppz(2 * std::max<int>(1, N));
    cheevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    std::vector<float> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    cheevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA,
            reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &iu, reinterpret_cast<float*>(&abstol),
            &M, evals.data(), compute_vectors ? reinterpret_cast<std::complex<float>*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (m_out) *m_out = M;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (compute_vectors && M < static_cast<int>(n)) eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    }
    if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    int lwork = -1, liwork = -1, lrwork = -1; std::complex<double> wkopt; int iwkopt; double rwkopt; std::vector<int> isuppz(2 * std::max<int>(1, N));
    zheevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    std::vector<double> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
    std::vector<int> iwork(std::max(1, liwork));
    if (compute_vectors) eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
    zheevr_(&jobz, &range_sel, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA,
            &vl, &vu, &il, &iu, &abstol, &M, evals.data(), compute_vectors ? reinterpret_cast<std::complex<double>*>(eigenvectors.data()) : nullptr, &N, isuppz.data(),
            work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    info_out = info;
    if (info == 0) {
      evals.resize(static_cast<std::size_t>(M));
      if (compute_vectors && M < static_cast<int>(n)) eigenvectors.shrink_to_cols(static_cast<std::size_t>(M));
    }
    if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  } else {
    return false;
  }
#endif
}

// Tridiagonalization + optional Q formation via SYTRD/ORGTR (or HETRD/UNGTR)
template <typename T, typename Storage, StorageOrder Order>
inline bool sytrd_tridiag_with_Q(const Matrix<T, Storage, Order>& A,
                                 std::vector<typename fem::numeric::numeric_traits<T>::scalar_type>& d,
                                 std::vector<typename fem::numeric::numeric_traits<T>::scalar_type>& e,
                                 Matrix<T, Storage, Order>* Qopt,
                                 fem::numeric::linear_algebra::Uplo uplo_sel,
                                 int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)d; (void)e; (void)Qopt; info_out = 0; return false;
#else
  using R = typename fem::numeric::numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) return false;
  if (n == 0) { d.clear(); e.clear(); if (Qopt) *Qopt = Matrix<T, Storage, Order>(0,0,T{}); info_out = 0; return true; }

  // RowMajor fast path via LAPACKE to avoid packing
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    d.assign(n, R{});
    e.assign((n>1)? n-1 : 0, R{});
    std::vector<T> tau(std::max<std::size_t>(1, n));
    // Work on a row-major copy to avoid mutating A
    std::vector<T> a_rm(A.rows() * A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) for (std::size_t j = 0; j < A.cols(); ++j) a_rm[i * A.cols() + j] = A(i, j);
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L'; int N = static_cast<int>(n);
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_ssytrd(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<float*>(a_rm.data()), static_cast<int>(A.cols()),
                            reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<float*>(tau.data()));
    } else if constexpr (std::is_same_v<T, double>) {
      info = LAPACKE_dsytrd(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<double*>(a_rm.data()), static_cast<int>(A.cols()),
                            reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<double*>(tau.data()));
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      info = LAPACKE_chetrd(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<lapack_complex_float*>(a_rm.data()), static_cast<int>(A.cols()),
                            reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<lapack_complex_float*>(tau.data()));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      info = LAPACKE_zhetrd(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<lapack_complex_double*>(a_rm.data()), static_cast<int>(A.cols()),
                            reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<lapack_complex_double*>(tau.data()));
    }
    info_out = info;
    if (info != 0) return true;
    if (Qopt) {
      // Form Q in-place on the same row-major buffer
      if constexpr (std::is_same_v<T, float>) {
        int rc = LAPACKE_sorgtr(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<float*>(a_rm.data()), static_cast<int>(A.cols()), reinterpret_cast<float*>(tau.data()));
        info_out = rc;
      } else if constexpr (std::is_same_v<T, double>) {
        int rc = LAPACKE_dorgtr(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<double*>(a_rm.data()), static_cast<int>(A.cols()), reinterpret_cast<double*>(tau.data()));
        info_out = rc;
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        int rc = LAPACKE_cungtr(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<lapack_complex_float*>(a_rm.data()), static_cast<int>(A.cols()), reinterpret_cast<lapack_complex_float*>(tau.data()));
        info_out = rc;
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        int rc = LAPACKE_zungtr(LAPACK_ROW_MAJOR, uplo, N, reinterpret_cast<lapack_complex_double*>(a_rm.data()), static_cast<int>(A.cols()), reinterpret_cast<lapack_complex_double*>(tau.data()));
        info_out = rc;
      }
      if (info_out != 0) return true;
      *Qopt = Matrix<T, Storage, Order>(n, n, T{});
      for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) (*Qopt)(i, j) = a_rm[i * A.cols() + j];
    }
    return true;
  }
#endif

  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);

  d.assign(n, R{});
  e.assign((n>1)? n-1 : 0, R{});
  std::vector<T> tau(std::max<std::size_t>(1, n));
  char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
  int N = static_cast<int>(n), LDA = static_cast<int>(n);
  int info = 0;

  // Workspace query
  int lwork = -1;
  if constexpr (std::is_same_v<T, float>) {
    float wkopt; ssytrd_(&uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<float*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt);
    std::vector<float> work(std::max(1, lwork));
    ssytrd_(&uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<float*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    double wkopt; dsytrd_(&uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<double*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(std::max(1, lwork));
    dsytrd_(&uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<double*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> wkopt; chetrd_(&uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<std::complex<float>*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    chetrd_(&uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<float*>(d.data()), reinterpret_cast<float*>(e.data()), reinterpret_cast<std::complex<float>*>(tau.data()), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> wkopt; zhetrd_(&uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<std::complex<double>*>(tau.data()), &wkopt, &lwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    zhetrd_(&uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<double*>(d.data()), reinterpret_cast<double*>(e.data()), reinterpret_cast<std::complex<double>*>(tau.data()), work.data(), &lwork, &info);
  } else {
    return false;
  }
  info_out = info;
  if (info != 0) return true;

  if (Qopt) {
    // Form Q from reflectors
    if constexpr (std::is_same_v<T, float>) {
      int lwork2 = -1; float wk2; sorgtr_(&uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(tau.data()), &wk2, &lwork2, &info);
      if (info != 0) { info_out = info; return true; }
      lwork2 = static_cast<int>(wk2);
      std::vector<float> work2(std::max(1, lwork2));
      sorgtr_(&uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, reinterpret_cast<float*>(tau.data()), work2.data(), &lwork2, &info);
    } else if constexpr (std::is_same_v<T, double>) {
      int lwork2 = -1; double wk2; dorgtr_(&uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(tau.data()), &wk2, &lwork2, &info);
      if (info != 0) { info_out = info; return true; }
      lwork2 = static_cast<int>(wk2);
      std::vector<double> work2(std::max(1, lwork2));
      dorgtr_(&uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, reinterpret_cast<double*>(tau.data()), work2.data(), &lwork2, &info);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      int lwork2 = -1; std::complex<float> wk2; cungtr_(&uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<const std::complex<float>*>(tau.data()), &wk2, &lwork2, &info);
      if (info != 0) { info_out = info; return true; }
      lwork2 = static_cast<int>(wk2.real());
      std::vector<std::complex<float>> work2(static_cast<std::size_t>(std::max(1, lwork2)));
      cungtr_(&uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, reinterpret_cast<const std::complex<float>*>(tau.data()), work2.data(), &lwork2, &info);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      int lwork2 = -1; std::complex<double> wk2; zungtr_(&uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<const std::complex<double>*>(tau.data()), &wk2, &lwork2, &info);
      if (info != 0) { info_out = info; return true; }
      lwork2 = static_cast<int>(wk2.real());
      std::vector<std::complex<double>> work2(static_cast<std::size_t>(std::max(1, lwork2)));
      zungtr_(&uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, reinterpret_cast<const std::complex<double>*>(tau.data()), work2.data(), &lwork2, &info);
    }
    if (info != 0) { info_out = info; return true; }
    *Qopt = Matrix<T, Storage, Order>(n, n, T{});
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i)
        (*Qopt)(i, j) = a_cm[j * n + i];
  }
  return true;
#endif
}
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evd(const Matrix<T, Storage, Order>& A,
                         Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                         Matrix<T, Storage, Order>& eigenvectors,
                         bool compute_vectors,
                         fem::numeric::linear_algebra::Uplo uplo_sel,
                         int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A; (void)evals; (void)eigenvectors; (void)compute_vectors; info_out = 0; return false;
#else
  using R = typename fem::numeric::numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) return false;
  if (n == 0) { evals = Vector<R>(0); eigenvectors = Matrix<T, Storage, Order>(0,0,T{}); info_out = 0; return true; }
  // RowMajor fast path via LAPACKE to avoid packing
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    Matrix<T, Storage, Order> W = A; // work on a mutable row-major copy
    char jobz = compute_vectors ? 'V' : 'N';
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
    int N = static_cast<int>(A.rows());
    evals = Vector<R>(N);
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<float*>(W.data()), static_cast<int>(W.cols()), evals.data());
    } else if constexpr (std::is_same_v<T, double>) {
      info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<double*>(W.data()), static_cast<int>(W.cols()), evals.data());
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      info = LAPACKE_cheevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<lapack_complex_float*>(W.data()), static_cast<int>(W.cols()), evals.data());
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<lapack_complex_double*>(W.data()), static_cast<int>(W.cols()), evals.data());
    } else {
      info_out = -1; return true;
    }
    info_out = info;
    if (info == 0) {
      if (compute_vectors) eigenvectors = std::move(W);
      else eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  }
#endif

  // ColumnMajor direct-write path: operate in-place on a column-major matrix buffer
  if constexpr (Order == StorageOrder::ColumnMajor) {
    Matrix<T, Storage, Order> W(n, n, T{});
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i)
        W(i, j) = A(i, j);

    char jobz = compute_vectors ? 'V' : 'N';
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
    int N = static_cast<int>(n);
    int LDA = static_cast<int>(W.leading_dimension());
    evals = Vector<R>(n);
    int info = 0;

    if constexpr (std::is_same_v<T, float>) {
      int lwork = -1, liwork = -1; float wkopt; int iwkopt;
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(W.data()), &LDA, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<float> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(W.data()), &LDA, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, double>) {
      int lwork = -1, liwork = -1; double wkopt; int iwkopt;
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(W.data()), &LDA, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<double> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(W.data()), &LDA, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<float> wkopt; int iwkopt; float rwkopt;
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(W.data()), &LDA, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<float> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(W.data()), &LDA, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<double> wkopt; int iwkopt; double rwkopt;
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(W.data()), &LDA, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<double> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(W.data()), &LDA, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else {
      return false;
    }

    if (info_out == 0 && compute_vectors) {
      eigenvectors = std::move(W);
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  }

  // Fallback: pack to column-major buffer then copy back
  {
    std::vector<T> a_cm(n * n);
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i)
        a_cm[j * n + i] = A(i, j);

    char jobz = compute_vectors ? 'V' : 'N';
    char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
    int N = static_cast<int>(n);
    int LDA = static_cast<int>(n);
    evals = Vector<R>(n);
    int info = 0;

    if constexpr (std::is_same_v<T, float>) {
      int lwork = -1, liwork = -1; float wkopt; int iwkopt;
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<float> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, double>) {
      int lwork = -1, liwork = -1; double wkopt; int iwkopt;
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<double> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<float> wkopt; int iwkopt; float rwkopt;
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<float> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<double> wkopt; int iwkopt; double rwkopt;
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<double> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else {
      return false;
    }

    if (info_out == 0 && compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(n, n, T{});
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
          eigenvectors(i, j) = a_cm[j * n + i];
    } else if (!compute_vectors) {
      eigenvectors = Matrix<T, Storage, Order>(0, 0, T{});
    }
    return true;
  }
#endif
}

// Tridiagonal values via STEVR (prefer) or STEVD as fallback. Returns true if attempted.
template <typename R>
inline bool stevr_values(std::vector<R>& d, std::vector<R>& e,
                         Vector<R>& w_out, int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)d; (void)e; (void)w_out; info_out = 0; return false;
#else
  const std::size_t n = d.size();
  if ((n == 0) || (e.size() + 1 != n)) { info_out = -1; return true; }
  int N = static_cast<int>(n);
  char jobz = 'N'; char range = 'A';
  R vl{}, vu{}, abstol{}; int il = 0, iu = 0;
  int M = 0; int info = 0;
  w_out = Vector<R>(n);

  if constexpr (std::is_same_v<R, float>) {
    int lwork = -1, liwork = -1; float wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max(1, N));
    sstevr_(&jobz, &range, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<float> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    sstevr_(&jobz, &range, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info; return true;
  } else if constexpr (std::is_same_v<R, double>) {
    int lwork = -1, liwork = -1; double wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max(1, N));
    dstevr_(&jobz, &range, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<double> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    dstevr_(&jobz, &range, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info; return true;
  } else {
    // Complex scalar type not valid here (should pass real R)
    info_out = -1; return true;
  }
#endif
}

// Range-aware STEVR for tridiagonal values
template <typename R>
inline bool stevr_values_range(std::vector<R>& d, std::vector<R>& e,
                               Vector<R>& w_out,
                               char range_sel, // 'A','I','V'
                               R vl, R vu, int il, int iu,
                               int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)d; (void)e; (void)w_out; (void)range_sel; (void)vl; (void)vu; (void)il; (void)iu; info_out = 0; return false;
#else
  const std::size_t n = d.size();
  if ((n == 0) || (e.size() + 1 != n)) { info_out = -1; return true; }
  int N = static_cast<int>(n);
  char jobz = 'N';
  R abstol{};
  int M = 0; int info = 0;
  w_out = Vector<R>(n);
  if constexpr (std::is_same_v<R, float>) {
    int lwork = -1, liwork = -1; float wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max(1, N));
    sstevr_(&jobz, &range_sel, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<float> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    sstevr_(&jobz, &range_sel, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info; return true;
  } else if constexpr (std::is_same_v<R, double>) {
    int lwork = -1, liwork = -1; double wkopt; int iwkopt; std::vector<int> isuppz(2 * std::max(1, N));
    dstevr_(&jobz, &range_sel, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) { info_out = info; return true; }
    lwork = static_cast<int>(wkopt); liwork = iwkopt;
    std::vector<double> work(std::max(1, lwork));
    std::vector<int> iwork(std::max(1, liwork));
    dstevr_(&jobz, &range_sel, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &abstol, &M,
            w_out.data(), nullptr, &N, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    info_out = info; return true;
  } else {
    info_out = -1; return true;
  }
#endif
}

// In-place EVD (destructive). On success, A_inout holds eigenvectors (columns) when compute_vectors is true.
// Works for RowMajor via LAPACKE and ColumnMajor via Fortran EVD, without extra allocations for the matrix data.
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evd_inplace(Matrix<T, Storage, Order>& A_inout,
                                 Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                                 bool compute_vectors,
                                 fem::numeric::linear_algebra::Uplo uplo_sel,
                                 int& info_out)
{
#if !defined(FEM_NUMERIC_ENABLE_LAPACK)
  (void)A_inout; (void)evals; (void)compute_vectors; info_out = 0; return false;
#else
  using R = typename fem::numeric::numeric_traits<T>::scalar_type;
  const std::size_t n = A_inout.rows();
  if (A_inout.cols() != n) return false;
  if (n == 0) { evals = Vector<R>(0); info_out = 0; return true; }

  char jobz = compute_vectors ? 'V' : 'N';
  char uplo = (uplo_sel == fem::numeric::linear_algebra::Uplo::Upper) ? 'U' : 'L';
  evals = Vector<R>(n);

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    int N = static_cast<int>(n);
    int lda = static_cast<int>(A_inout.cols());
    int info = 0;
    if constexpr (std::is_same_v<T, float>)      info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<float*>(A_inout.data()), lda, evals.data());
    else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<double*>(A_inout.data()), lda, evals.data());
    else if constexpr (std::is_same_v<T, std::complex<float>>)  info = LAPACKE_cheevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<lapack_complex_float*>(A_inout.data()), lda, evals.data());
    else if constexpr (std::is_same_v<T, std::complex<double>>) info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, jobz, uplo, N, reinterpret_cast<lapack_complex_double*>(A_inout.data()), lda, evals.data());
    else { info_out = -1; return true; }
    info_out = info;
    return true;
  }
#endif

  if constexpr (Order == StorageOrder::ColumnMajor) {
    int N = static_cast<int>(n);
    int lda = static_cast<int>(A_inout.leading_dimension());
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
      int lwork = -1, liwork = -1; float wkopt; int iwkopt;
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(A_inout.data()), &lda, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<float> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      ssyevd_(&jobz, &uplo, &N, reinterpret_cast<float*>(A_inout.data()), &lda, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, double>) {
      int lwork = -1, liwork = -1; double wkopt; int iwkopt;
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(A_inout.data()), &lda, evals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt); liwork = iwkopt;
      std::vector<double> work(std::max(1, lwork));
      std::vector<int> iwork(std::max(1, liwork));
      dsyevd_(&jobz, &uplo, &N, reinterpret_cast<double*>(A_inout.data()), &lda, evals.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<float> wkopt; int iwkopt; float rwkopt;
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(A_inout.data()), &lda, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<float> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      cheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<float>*>(A_inout.data()), &lda, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      int lwork = -1, liwork = -1, lrwork = -1; std::complex<double> wkopt; int iwkopt; double rwkopt;
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(A_inout.data()), &lda, evals.data(), &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
      if (info != 0) { info_out = info; return true; }
      lwork = static_cast<int>(wkopt.real()); lrwork = static_cast<int>(rwkopt); liwork = iwkopt;
      std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
      std::vector<double> rwork(static_cast<std::size_t>(std::max(1, lrwork)));
      std::vector<int> iwork(std::max(1, liwork));
      zheevd_(&jobz, &uplo, &N, reinterpret_cast<std::complex<double>*>(A_inout.data()), &lda, evals.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      info_out = info;
    } else {
      return false;
    }
    return true;
  }

  return false;
#endif
}

// Backward-compatible overload defaulting to Upper for in-place EVD
template <typename T, typename Storage, StorageOrder Order>
inline bool eigh_via_evd_inplace(Matrix<T, Storage, Order>& A_inout,
                                 Vector<typename fem::numeric::numeric_traits<T>::scalar_type>& evals,
                                 bool compute_vectors,
                                 int& info_out)
{
  return eigh_via_evd_inplace(A_inout, evals, compute_vectors, fem::numeric::linear_algebra::Uplo::Upper, info_out);
}

} // namespace fem::numeric::backends::lapack

#endif // NUMERIC_BACKENDS_LAPACK_BACKEND_H

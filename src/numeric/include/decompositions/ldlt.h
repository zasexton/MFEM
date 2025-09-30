#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_LDLT_H
#define NUMERIC_DECOMPOSITIONS_LDLT_H

// Symmetric/Hermitian LDL^T (LDL^H) factorization scaffolding.
// Provides a LAPACK-backed path (xSYTRF/xHETRF + xSYTRS/xHETRS) when enabled,
// and a minimal, unpivoted reference fallback for development/testing.
//
// Interface (dense, in-place):
//   - ldlt_factor(A, piv, uplo) -> stores L (unit diag) and D in A per LAPACK layout
//     when backend is used. Fallback stores L (strict lower/upper) and diagonal D
//     on the main diagonal (no 2x2 blocks). Pivot vector follows LAPACK semantics
//     in backend mode; in fallback it is identity (no pivoting).
//   - ldlt_solve_inplace(A_fact, piv, b, uplo) -> solves A x = b using the factor.
//
// Notes:
//   - The fallback is unpivoted and intended for SPD or mildly indefinite cases.
//     For robust indefinite handling, prefer the LAPACK path.
//   - For complex T, the factorization is Hermitian (LDL^H).

#include <vector>
#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level2.h" // Uplo
#include "../linear_algebra/blas_level3.h" // (not required but consistent with others)

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
  // Real symmetric BK (LDLT)
  void ssytrf_(char* uplo, int* n, float* a, int* lda, int* ipiv,
               float* work, int* lwork, int* info);
  void dsytrf_(char* uplo, int* n, double* a, int* lda, int* ipiv,
               double* work, int* lwork, int* info);
  void ssytrs_(char* uplo, int* n, int* nrhs, const float* a, int* lda,
               const int* ipiv, float* b, int* ldb, int* info);
  void dsytrs_(char* uplo, int* n, int* nrhs, const double* a, int* lda,
               const int* ipiv, double* b, int* ldb, int* info);

  // Complex Hermitian BK (LDL^H)
  void chetrf_(char* uplo, int* n, std::complex<float>* a, int* lda, int* ipiv,
               std::complex<float>* work, int* lwork, int* info);
  void zhetrf_(char* uplo, int* n, std::complex<double>* a, int* lda, int* ipiv,
               std::complex<double>* work, int* lwork, int* info);
  void chetrs_(char* uplo, int* n, int* nrhs, const std::complex<float>* a, int* lda,
               const int* ipiv, std::complex<float>* b, int* ldb, int* info);
  void zhetrs_(char* uplo, int* n, int* nrhs, const std::complex<double>* a, int* lda,
               const int* ipiv, std::complex<double>* b, int* ldb, int* info);
}
#endif

// ------------------------------
// Reference unpivoted LDL^T/LDL^H
// ------------------------------
template <typename T, typename Storage, StorageOrder Order>
static inline int ldlt_factor_unpivoted(Matrix<T, Storage, Order>& A,
                                        fem::numeric::linear_algebra::Uplo uplo)
{
  using R = typename numeric_traits<T>::scalar_type;
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor_unpivoted: matrix must be square");
  if (n == 0) return 0;

  // We store L (unit diagonal) in the requested triangle and D on the main diagonal.
  if (uplo == Uplo::Lower) {
    for (std::size_t i = 0; i < n; ++i) {
      // Compute D(i)
      T dii = A(i, i);
      for (std::size_t k = 0; k < i; ++k) {
        T lik = A(i, k); // L(i,k)
        R dkk;
        if constexpr (is_complex_number_v<T>) dkk = static_cast<R>(std::real(A(k, k)));
        else dkk = static_cast<R>(A(k, k));
        if constexpr (is_complex_number_v<T>) dii -= lik * static_cast<T>(dkk) * conj_if_complex(lik);
        else dii -= lik * static_cast<T>(dkk) * lik;
      }
      // Basic singularity check
      R absd = static_cast<R>(std::abs(dii));
      if (!(absd > R{0})) return static_cast<int>(i) + 1;
      A(i, i) = dii;
      // Compute L(j,i) for j>i
      for (std::size_t j = i + 1; j < n; ++j) {
        T aji = (uplo == Uplo::Lower) ? A(j, i) : A(i, j);
        T sum = aji;
        for (std::size_t k = 0; k < i; ++k) {
          T ljk = A(j, k);
          T lik = A(i, k);
          R dkk;
          if constexpr (is_complex_number_v<T>) dkk = static_cast<R>(std::real(A(k, k)));
          else dkk = static_cast<R>(A(k, k));
          if constexpr (is_complex_number_v<T>) sum -= ljk * static_cast<T>(dkk) * conj_if_complex(lik);
          else sum -= ljk * static_cast<T>(dkk) * lik;
        }
        // L(j,i) = sum / D(i)
        A(j, i) = sum / A(i, i);
      }
      // Zero out the strict upper for cleanliness (lower storage)
      for (std::size_t j = i + 1; j < n; ++j) A(i, j) = T{};
    }
  } else { // Uplo::Upper, symmetric version writing into upper triangle
    for (std::size_t i = 0; i < n; ++i) {
      T dii = A(i, i);
      for (std::size_t k = 0; k < i; ++k) {
        T uki = A(k, i);
        R dkk;
        if constexpr (is_complex_number_v<T>) dkk = static_cast<R>(std::real(A(k, k)));
        else dkk = static_cast<R>(A(k, k));
        if constexpr (is_complex_number_v<T>) dii -= conj_if_complex(uki) * static_cast<T>(dkk) * uki;
        else dii -= uki * static_cast<T>(dkk) * uki;
      }
      R absd = static_cast<R>(std::abs(dii));
      if (!(absd > R{0})) return static_cast<int>(i) + 1;
      A(i, i) = dii;
    
      for (std::size_t j = i + 1; j < n; ++j) {
        T aij = A(i, j);
        T sum = aij;
        for (std::size_t k = 0; k < i; ++k) {
          T uki = A(k, i);
          T ukj = A(k, j);
          R dkk;
          if constexpr (is_complex_number_v<T>) dkk = static_cast<R>(std::real(A(k, k)));
          else dkk = static_cast<R>(A(k, k));
          if constexpr (is_complex_number_v<T>) sum -= conj_if_complex(uki) * static_cast<T>(dkk) * ukj;
          else sum -= uki * static_cast<T>(dkk) * ukj;
        }
        A(i, j) = sum / A(i, i);
      }
      for (std::size_t j = 0; j < i; ++j) A(i, j) = T{};
    }
  }
  return 0;
}

// -------------------------------------
// Public API: ldlt_factor (best backend)
// -------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int ldlt_factor(Matrix<T, Storage, Order>& A,
                std::vector<int>& piv,
                fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor: matrix must be square");
  piv.clear();
  if (n == 0) return 0;

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Always pack to column-major and use xSYTRF/xHETRF for robustness.
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);

  char u = (uplo == Uplo::Lower) ? 'L' : 'U';
  int N = static_cast<int>(n);
  int LDA = static_cast<int>(n);
  int info = 0;

  // Workspace query
  int lwork = -1;
  // Note: real and complex have different APIs; use T to branch.
  if constexpr (std::is_same_v<T, float>) {
    float wkopt; ssytrf_(&u, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    piv.resize(n);
    std::vector<float> work(std::max(1, lwork));
    ssytrf_(&u, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    double wkopt; dsytrf_(&u, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    piv.resize(n);
    std::vector<double> work(std::max(1, lwork));
    dsytrf_(&u, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> wkopt; chetrf_(&u, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    piv.resize(n);
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    chetrf_(&u, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> wkopt; zhetrf_(&u, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    piv.resize(n);
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    zhetrf_(&u, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else {
    // Unsupported type by backend; fallback
    return ldlt_factor_unpivoted(A, uplo);
  }

  if (info != 0) return info;

  // Copy factored data back to row-major storage (keep LAPACK layout in triangle)
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      A(i, j) = a_cm[j * n + i];

  // LAPACK ipiv is 1-based and encodes 2x2 blocks with negative values.
  // Keep ipiv as-is to be compatible with xSYTRS/xHETRS in ldlt_solve.
  return 0;
#else
  // Fallback: unpivoted LDL^T/LDL^H
  piv.resize(n);
  for (std::size_t i = 0; i < n; ++i) piv[i] = static_cast<int>(i + 1); // 1-based identity for solve helper
  return ldlt_factor_unpivoted(A, uplo);
#endif
}

// ---------------------------------------
// Solve using LDLT factor (vector/matrix)
// ---------------------------------------
template <typename T>
void ldlt_solve_inplace(const Matrix<T>& A_fact,
                        const std::vector<int>& piv,
                        Vector<T>& b,
                        fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || b.size() != n) throw std::invalid_argument("ldlt_solve_inplace(vector): dimension mismatch");

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Use xSYTRS/xHETRS by repacking to column-major
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A_fact(i, j);
  char u = (uplo == Uplo::Lower) ? 'L' : 'U';
  int N = static_cast<int>(n), NRHS = 1, LDA = static_cast<int>(n), LDB = static_cast<int>(n);
  int info = 0;

  if constexpr (std::is_same_v<T, float>) {
    ssytrs_(&u, &N, &NRHS, reinterpret_cast<const float*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<float*>(b.data()), &LDB, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    dsytrs_(&u, &N, &NRHS, reinterpret_cast<const double*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<double*>(b.data()), &LDB, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    chetrs_(&u, &N, &NRHS, reinterpret_cast<const std::complex<float>*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<std::complex<float>*>(b.data()), &LDB, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    zhetrs_(&u, &N, &NRHS, reinterpret_cast<const std::complex<double>*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<std::complex<double>*>(b.data()), &LDB, &info);
  } else {
    // No backend: fall through to reference
    info = -1;
  }
  if (info == 0) return;
#endif

  // Reference solve (unpivoted LDL^T). Assumes A_fact contains L (unit diag) and D on diag
  // as produced by ldlt_factor_unpivoted.
  // 1) Forward: L y = b
  for (std::size_t i = 0; i < n; ++i) {
    T sum = b[i];
    for (std::size_t k = 0; k < i; ++k) sum -= A_fact(i, k) * b[k];
    b[i] = sum; // unit diag
  }
  // 2) Diagonal solve: D z = y (D is diagonal here)
  for (std::size_t i = 0; i < n; ++i) b[i] = b[i] / A_fact(i, i);
  // 3) Backward: L^H x = z
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;
    T sum = b[i];
    for (std::size_t k = i + 1; k < n; ++k) sum -= conj_if_complex(A_fact(k, i)) * b[k];
    b[i] = sum; // unit diag
  }
}

template <typename T>
void ldlt_solve_inplace(const Matrix<T>& A_fact,
                        const std::vector<int>& piv,
                        Matrix<T>& B,
                        fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || B.rows() != n) throw std::invalid_argument("ldlt_solve_inplace(matrix): dimension mismatch");
  for (std::size_t j = 0; j < B.cols(); ++j) {
    Vector<T> bj(n);
    for (std::size_t i = 0; i < n; ++i) bj[i] = B(i, j);
    ldlt_solve_inplace(A_fact, piv, bj, uplo);
    for (std::size_t i = 0; i < n; ++i) B(i, j) = bj[i];
  }
}

template <typename T>
Vector<T> ldlt_solve(const Matrix<T>& A_fact,
                     const std::vector<int>& piv,
                     const Vector<T>& b,
                     fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  Vector<T> x = b;
  ldlt_solve_inplace(A_fact, piv, x, uplo);
  return x;
}

template <typename T>
Matrix<T> ldlt_solve(const Matrix<T>& A_fact,
                     const std::vector<int>& piv,
                     const Matrix<T>& B,
                     fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  Matrix<T> X = B;
  ldlt_solve_inplace(A_fact, piv, X, uplo);
  return X;
}

#endif // NUMERIC_DECOMPOSITIONS_LDLT_H


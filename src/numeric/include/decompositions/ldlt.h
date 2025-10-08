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
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor_unpivoted: matrix must be square");
  if (n == 0) return 0;

  // We store L (unit diagonal) in the requested triangle and D on the main diagonal.
  if (uplo == fem::numeric::linear_algebra::Uplo::Lower) {
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
      // Store D(i) with zero imaginary part for Hermitian inputs
      if constexpr (is_complex_number_v<T>) {
        A(i, i) = static_cast<T>(std::real(dii));
      } else {
        A(i, i) = dii;
      }
      // Compute L(j,i) for j>i
      for (std::size_t j = i + 1; j < n; ++j) {
        T aji = (uplo == fem::numeric::linear_algebra::Uplo::Lower) ? A(j, i) : A(i, j);
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
      if constexpr (is_complex_number_v<T>) {
        A(i, i) = static_cast<T>(std::real(dii));
      } else {
        A(i, i) = dii;
      }
    
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
// Blocked unpivoted LDL^T/LDL^H using BLAS3 updates (Lower only)
// -------------------------------------
template <typename T, typename Storage, StorageOrder Order>
static inline int ldlt_factor_unpivoted_blocked(Matrix<T, Storage, Order>& A,
                                                fem::numeric::linear_algebra::Uplo uplo,
                                                std::size_t block = 64)
{
  using namespace fem::numeric::linear_algebra;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor_unpivoted_blocked: matrix must be square");
  if (n == 0) return 0;
  if (uplo != Uplo::Lower) {
    // For Upper: mirror to Lower, factor, then mirror back
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = j + 1; i < n; ++i) A(i, j) = conj_if_complex(A(j, i));
    int info = ldlt_factor_unpivoted_blocked(A, Uplo::Lower, block);
    // Zero strictly lower and mirror back to upper triangle for cleanliness
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) A(i, j) = conj_if_complex(A(j, i));
      for (std::size_t j = 0; j < i; ++j) A(i, j) = T{};
    }
    return info;
  }

  const std::size_t bs = std::max<std::size_t>(1, block);
  for (std::size_t k = 0; k < n; k += bs) {
    const std::size_t kb = std::min(bs, n - k);
    // Factor diagonal block Akk in-place (unblocked)
    {
      auto Akk = A.submatrix(k, k + kb, k, k + kb);
      int info_blk = ldlt_factor_unpivoted(Akk, Uplo::Lower);
      if (info_blk != 0) {
        // Return LAPACK-style absolute index
        return static_cast<int>(k) + info_blk;
      }
    }

    // Panel + trailing update
    if (k + kb < n) {
      auto L21 = A.submatrix(k + kb, n, k, k + kb);   // (n-k-kb) x kb, will become L21 after scaling by D^-1 and solve
      auto Akk = A.submatrix(k, k + kb, k, k + kb);   // holds Lkk (unit) and Dkk on diag

      // Right solve: L21 := L21 * inv(Lkk^H)
      trsm(Side::Right, Uplo::Lower, Trans::ConjTranspose, Diag::Unit, T{1}, Akk, L21);

      // Build W = L21 * D (scale columns of L21 by Dkk)
      Matrix<T> W = L21;
      for (std::size_t j = 0; j < kb; ++j) {
        T djj = Akk(j, j);
        for (std::size_t i = 0; i < W.rows(); ++i) W(i, j) = static_cast<T>(W(i, j) * djj);
      }
      // Scale L21 by D^-1 to finalize L21 (columns j /= djj)
      for (std::size_t j = 0; j < kb; ++j) {
        T djj = Akk(j, j);
        for (std::size_t i = 0; i < L21.rows(); ++i) L21(i, j) = static_cast<T>(L21(i, j) / djj);
      }

      // Trailing update: A22 := A22 - W * L21^H (Hermitian rank-k update)
      auto A22 = A.submatrix(k + kb, n, k + kb, n);
      gemm(Trans::NoTrans, Trans::ConjTranspose, T{-1}, W, L21, T{1}, A22);
    }
  }
  return 0;
}

// -------------------------------------
// Unblocked Bunch–Kaufman (rook) LDL^T/LDL^H factorization (robust fallback)
// Stores unit-lower L in strict lower triangle and block-diagonal D in the
// diagonal (1x1) and subdiagonal (2x2 off-diagonal at (k+1,k)).
// ipiv semantics (LAPACK-compatible subset):
//  - ipiv[k] > 0: 1x1 pivot. If ipiv[k] != k+1, row/col k was swapped with ipiv[k]-1.
//  - ipiv[k] = ipiv[k+1] < 0: 2x2 pivot block on rows/cols k and k+1. If -ipiv[k] != k+1,
//    row/col k+1 was swapped with -ipiv[k]-1 prior to forming the 2x2 block.
// Lower variant implemented; Upper mirrors to Lower.
// -------------------------------------
template <typename T, typename Storage, StorageOrder Order>
static inline int ldlt_factor_bk_rook(Matrix<T, Storage, Order>& A,
                                      std::vector<int>& ipiv,
                                      fem::numeric::linear_algebra::Uplo uplo)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor_bk_rook: matrix must be square");
  ipiv.assign(n, 0);
  if (n == 0) return 0;

  if (uplo != Uplo::Lower) {
    // Mirror to lower, factor, and mirror result back
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = j + 1; i < n; ++i) A(i, j) = conj_if_complex(A(j, i));
    int info = ldlt_factor_bk_rook(A, ipiv, Uplo::Lower);
    // Reflect lower to upper and zero lower strictly for cleanliness
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) A(i, j) = conj_if_complex(A(j, i));
      for (std::size_t j = 0; j < i; ++j) A(i, j) = T{};
    }
    return info;
  }

  const R alpha = static_cast<R>((1.0 + std::sqrt(17.0)) / 8.0);

  std::size_t k = 0;
  while (k < n) {
    // Find pivot
    std::size_t kp = k; int kstep = 1;
    R absakk = static_cast<R>(std::abs(A(k, k)));
    // colmax: max |A(i,k)| for i>k
    std::size_t imax = k; R colmax = R{0};
    if (k + 1 < n) {
      for (std::size_t i = k + 1; i < n; ++i) {
        R v = static_cast<R>(std::abs(A(i, k)));
        if (v > colmax) { colmax = v; imax = i; }
      }
    }
    if (absakk >= alpha * colmax || (k + 1) == n) {
      // 1x1 pivot at k
      kp = k; kstep = 1;
    } else {
      // rook test on row imax
      R rowmax = R{0};
      // scan left part (k..imax-1) as row entries
      for (std::size_t j = k; j < imax; ++j) {
        R v = static_cast<R>(std::abs(A(imax, j)));
        if (v > rowmax) rowmax = v;
      }
      // scan below (imax+1..n-1) via column values
      for (std::size_t i2 = imax + 1; i2 < n; ++i2) {
        R v = static_cast<R>(std::abs(A(i2, imax)));
        if (v > rowmax) rowmax = v;
      }
      R absaimax = static_cast<R>(std::abs(A(imax, imax)));
      if (absaimax >= alpha * rowmax) {
        // 1x1 pivot with interchange k <-> imax
        kp = imax; kstep = 1;
      } else {
        // 2x2 pivot: will pivot on (k, p) after moving p to k+1
        kp = imax; kstep = 2;
      }
    }

    if (kstep == 1) {
      // Interchange k and kp if needed
      if (kp != k) { A.swap_rows(k, kp); A.swap_cols(k, kp); }
      // D(k,k) with real diag for complex
      if constexpr (is_complex_number_v<T>) A(k, k) = static_cast<T>(std::real(A(k, k)));
      // Singular check for 1x1 pivot
      if (static_cast<R>(std::abs(A(k, k))) == R{0}) return static_cast<int>(k) + 1;
      // Compute multipliers and update trailing
      if (k + 1 < n) {
        T dkk = A(k, k);
        // Singular pivot check
        if (static_cast<R>(std::abs(dkk)) == R{0}) return static_cast<int>(k) + 1;
        // L(i,k) = A(i,k) / dkk
        for (std::size_t i = k + 1; i < n; ++i) A(i, k) = static_cast<T>(A(i, k) / dkk);
        // A22 := A22 - dkk * v * v^H, v = column k below diag
        for (std::size_t i = k + 1; i < n; ++i) {
          T vik = A(i, k);
          for (std::size_t j = i; j < n; ++j) {
            A(j, i) = static_cast<T>(A(j, i) - dkk * vik * conj_if_complex(A(j, k)));
          }
        }
        // Clean upper part of column k
        for (std::size_t j = k + 1; j < n; ++j) A(k, j) = T{};
      }
      ipiv[k] = static_cast<int>(kp) + 1;
      k += 1;
    } else {
      // 2x2 block pivot on k and k+1. Move kp to k+1 if needed
      if (kp != k + 1) { A.swap_rows(k + 1, kp); A.swap_cols(k + 1, kp); }
      // Extract 2x2 block
      T D11 = A(k, k);
      T D21 = A(k + 1, k);
      T D22 = A(k + 1, k + 1);
      // Make diagonals explicitly real
      if constexpr (is_complex_number_v<T>) {
        D11 = static_cast<T>(std::real(D11));
        D22 = static_cast<T>(std::real(D22));
      }
      A(k, k) = D11; A(k + 1, k) = D21; A(k + 1, k + 1) = D22; if (k + 2 <= n) for (std::size_t j = k; j < k + 2; ++j) for (std::size_t i = 0; i < j; ++i) (void)0;
      // Pre-save A21 block below for update
      const std::size_t m = (k + 2 < n) ? (n - (k + 2)) : 0;
      std::vector<T> W1(m), W2(m);
      for (std::size_t i = 0; i < m; ++i) {
        W1[i] = A(k + 2 + i, k);
        W2[i] = A(k + 2 + i, k + 1);
      }
      // Compute inv(D) for the 2x2 Hermitian block
      // Let D = [[D11, D12]; [conj(D12), D22]] with D12 = conj(D21) stored implicitly.
      // inv(D) = 1/den * [[ D22,      -D12        ],
      //                   [ -conj(D12),  D11       ]],
      // where den = D11*D22 - |D12|^2 = D11*D22 - D21*conj(D21).
      T denom = static_cast<T>(D11 * D22 - D21 * conj_if_complex(D21));
      if (static_cast<R>(std::abs(denom)) == R{0}) return static_cast<int>(k) + 1;
      // Compute L(i,[k,k+1]) = [W1 W2] * invD for i>k+1
      for (std::size_t i = 0; i < m; ++i) {
        T a1 = W1[i], a2 = W2[i];
        // Use D12 = conj(D21) for upper-right element of D
        T l1 = static_cast<T>(( a1 * D22 - a2 * conj_if_complex(D21)) / denom);
        T l2 = static_cast<T>((-a1 * D21 + a2 * D11) / denom);
        A(k + 2 + i, k)     = l1;
        A(k + 2 + i, k + 1) = l2;
      }
      // Trailing update: A22 -= W * L^H where W=[W1 W2], L=[l1 l2]
      for (std::size_t ii = 0; ii < m; ++ii) {
        std::size_t irow = k + 2 + ii;
        T w1 = W1[ii], w2 = W2[ii];
        for (std::size_t jj = ii; jj < m; ++jj) {
          std::size_t jrow = k + 2 + jj;
          T lj1 = A(jrow, k);
          T lj2 = A(jrow, k + 1);
          A(jrow, irow) = static_cast<T>(A(jrow, irow) - (w1 * conj_if_complex(lj1) + w2 * conj_if_complex(lj2)));
        }
      }
      // Zero out upper parts
      for (std::size_t j = k + 1; j < n; ++j) A(k, j) = T{};
      for (std::size_t j = k + 2; j < n; ++j) A(k + 1, j) = T{};
      ipiv[k] = -static_cast<int>(k + 1 + (kp != k + 1 ? 0 : 0) + (kp != k + 1 ? (kp - (k + 1)) : 0));
      // LAPACK-compatible: ipiv[k] = ipiv[k+1] = -(p), where p is (index swapped with k+1)+1
      int p1 = static_cast<int>((kp) + 1);
      ipiv[k] = -p1;
      ipiv[k + 1] = -p1;
      k += 2;
    }
  }
  return 0;
}

// Pivoted solve using BK ipiv (vector)
template <typename T>
static inline void ldlt_solve_bk_inplace(const Matrix<T>& A_fact,
                                         const std::vector<int>& ipiv,
                                         Vector<T>& b,
                                         fem::numeric::linear_algebra::Uplo uplo)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || b.size() != n) throw std::invalid_argument("ldlt_solve_bk(vector): dimension mismatch");
  if (uplo != Uplo::Lower) {
    // Mirror to lower and reuse
    Matrix<T> L = A_fact;
    for (std::size_t j = 0; j < n; ++j) for (std::size_t i = j + 1; i < n; ++i) L(i, j) = conj_if_complex(A_fact(j, i));
    ldlt_solve_bk_inplace(L, ipiv, b, Uplo::Lower);
    return;
  }
  // Apply P to RHS (P is symmetric; same as P^T, but use standard LAPACK order)
  // For a 2x2 pivot (ipiv[k] < 0), LAPACK encodes ipiv[k] = ipiv[k+1] = -p,
  // meaning that row/col k+1 was swapped with p-1 prior to forming the block.
  // Therefore we must swap b[k+1] with b[p-1] (not b[k]).
  for (std::size_t k = 0; k < n; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      std::size_t p = static_cast<std::size_t>(piv - 1);
      if (p != k) std::swap(b[k], b[p]);
      k += 1;
    } else {
      std::size_t p = static_cast<std::size_t>(-piv - 1);
      std::size_t kp1 = k + 1; // partner row in the 2x2 block
      if (p != kp1) std::swap(b[kp1], b[p]);
      k += 2; // skip partner index (ipiv[k] == ipiv[k+1])
    }
  }
  // Forward solve: L y = b
  for (std::size_t i = 0; i < n; ++i) {
    T sum = b[i];
    for (std::size_t k = 0; k < i; ++k) {
      // For a 2x2 pivot starting at k (ipiv[k] < 0 and ipiv[k+1]==ipiv[k]),
      // the entry L(k+1,k) is structurally zero (A_fact(k+1,k) stores D21).
      if (k + 1 < n && ipiv[k] < 0 && ipiv[k + 1] == ipiv[k] && i == k + 1) continue;
      sum -= A_fact(i, k) * b[k];
    }
    b[i] = sum;
  }
  // Diagonal block solve
  for (std::size_t k = 0; k < n; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      b[k] = static_cast<T>(b[k] / A_fact(k, k));
      k += 1;
    } else {
      // 2x2 block at k,k+1. Form D12 = conj(D21) for clarity.
      T D11 = A_fact(k, k);
      T D21 = A_fact(k + 1, k);
      T D12 = conj_if_complex(D21);
      T D22 = A_fact(k + 1, k + 1);
      T rhs1 = b[k], rhs2 = b[k + 1];
      T denom = static_cast<T>(D11 * D22 - D12 * conj_if_complex(D12));
      T xk  = static_cast<T>(( D22 * rhs1 - D12 * rhs2) / denom);
      T xk1 = static_cast<T>((-conj_if_complex(D12) * rhs1 + D11 * rhs2) / denom);
      b[k] = xk; b[k + 1] = xk1;
      k += 2;
    }
  }
  // Backward solve: L^H x = z
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;
    T sum = b[i];
    for (std::size_t k = i + 1; k < n; ++k) {
      // Skip the non-existent L(i+1,i) when i is the start of a 2x2 block
      if (i + 1 < n && ipiv[i] < 0 && ipiv[i + 1] == ipiv[i] && k == i + 1) continue;
      sum -= conj_if_complex(A_fact(k, i)) * b[k];
    }
    b[i] = sum;
  }
  // Apply P^T to result (reverse order). For a 2x2 pivot stored at indices
  // k-1 and k (ipiv[k] < 0), the swap was between row k and p-1, so we must
  // undo that by swapping b[k] with b[p-1]. Then skip the partner index.
  for (std::size_t k = n; k-- > 0; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      std::size_t p = static_cast<std::size_t>(piv - 1);
      if (p != k) std::swap(b[k], b[p]);
    } else {
      std::size_t p = static_cast<std::size_t>(-piv - 1);
      // k is the second index of the 2x2 block; undo swap on row k
      if (p != k) std::swap(b[k], b[p]);
      if (k == 0) break; // safety
      k -= 1; // skip partner (k-1)
    }
  }
}

// Pivoted solve using BK ipiv (matrix)
template <typename T>
static inline void ldlt_solve_bk_inplace(const Matrix<T>& A_fact,
                                         const std::vector<int>& ipiv,
                                         Matrix<T>& B,
                                         fem::numeric::linear_algebra::Uplo uplo)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || B.rows() != n) throw std::invalid_argument("ldlt_solve_bk(matrix): dimension mismatch");
  if (uplo != Uplo::Lower) {
    Matrix<T> L = A_fact;
    for (std::size_t j = 0; j < n; ++j) for (std::size_t i = j + 1; i < n; ++i) L(i, j) = conj_if_complex(A_fact(j, i));
    ldlt_solve_bk_inplace(L, ipiv, B, Uplo::Lower);
    return;
  }
  const std::size_t nrhs = B.cols();
  // Apply P to RHS (row swaps)
  for (std::size_t k = 0; k < n; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      std::size_t p = static_cast<std::size_t>(piv - 1);
      if (p != k) for (std::size_t j = 0; j < nrhs; ++j) std::swap(B(k, j), B(p, j));
      k += 1;
    } else {
      std::size_t p = static_cast<std::size_t>(-piv - 1);
      std::size_t kp1 = k + 1;
      if (p != kp1) for (std::size_t j = 0; j < nrhs; ++j) std::swap(B(kp1, j), B(p, j));
      k += 2;
    }
  }
  // Forward: L Y = B
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < nrhs; ++j) {
      T sum = B(i, j);
      for (std::size_t k = 0; k < i; ++k) {
        if (k + 1 < n && ipiv[k] < 0 && ipiv[k + 1] == ipiv[k] && i == k + 1) continue;
        sum -= A_fact(i, k) * B(k, j);
      }
      B(i, j) = sum;
    }
  }
  // Diagonal block solve
  for (std::size_t k = 0; k < n; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      T dkk = A_fact(k, k);
      for (std::size_t j = 0; j < nrhs; ++j) B(k, j) = static_cast<T>(B(k, j) / dkk);
      k += 1;
    } else {
      // 2x2 block. Use D12 = conj(D21) for correct Hermitian handling.
      T D11 = A_fact(k, k);
      T D21 = A_fact(k + 1, k);
      T D12 = conj_if_complex(D21);
      T D22 = A_fact(k + 1, k + 1);
      T denom = static_cast<T>(D11 * D22 - D12 * conj_if_complex(D12));
      for (std::size_t j = 0; j < nrhs; ++j) {
        T rhs1 = B(k, j), rhs2 = B(k + 1, j);
        T xk  = static_cast<T>(( D22 * rhs1 - D12 * rhs2) / denom);
        T xk1 = static_cast<T>((-conj_if_complex(D12) * rhs1 + D11 * rhs2) / denom);
        B(k, j) = xk; B(k + 1, j) = xk1;
      }
      k += 2;
    }
  }
  // Backward: L^H X = Z
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;
    for (std::size_t j = 0; j < nrhs; ++j) {
      T sum = B(i, j);
      for (std::size_t k = i + 1; k < n; ++k) {
        if (i + 1 < n && ipiv[i] < 0 && ipiv[i + 1] == ipiv[i] && k == i + 1) continue;
        sum -= conj_if_complex(A_fact(k, i)) * B(k, j);
      }
      B(i, j) = sum;
    }
  }
  // Apply P^T to result in reverse
  for (std::size_t k = n; k-- > 0; ) {
    int piv = ipiv[k];
    if (piv > 0) {
      std::size_t p = static_cast<std::size_t>(piv - 1);
      if (p != k) for (std::size_t j = 0; j < nrhs; ++j) std::swap(B(k, j), B(p, j));
    } else {
      std::size_t p = static_cast<std::size_t>(-piv - 1);
      if (p != k) for (std::size_t j = 0; j < nrhs; ++j) std::swap(B(k, j), B(p, j));
      if (k == 0) break;
      k -= 1; // skip partner
    }
  }
}

// -------------------------------------
// Public API: ldlt_factor (best backend)
// -------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int ldlt_factor(Matrix<T, Storage, Order>& A,
                std::vector<int>& piv,
                fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("ldlt_factor: matrix must be square");
  piv.clear();
  if (n == 0) return 0;

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  char u = (uplo == fem::numeric::linear_algebra::Uplo::Lower) ? 'L' : 'U';
  int N = static_cast<int>(n);
  piv.resize(n);
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  if constexpr (Order == StorageOrder::RowMajor) {
    // In-place row-major factorization via LAPACKE
    int info_rm = backends::lapack::sytrf_rm<T>(u, N, A.data(), static_cast<int>(A.cols()), piv.data());
    return info_rm;
  }
#endif
  // Pack to column-major and use xSYTRF/xHETRF
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A(i, j);
  int LDA = static_cast<int>(n);
  int info = 0;

  int lwork = -1;
  if constexpr (std::is_same_v<T, float>) {
    float wkopt; ssytrf_(&u, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    std::vector<float> work(std::max(1, lwork));
    ssytrf_(&u, &N, reinterpret_cast<float*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    double wkopt; dsytrf_(&u, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(std::max(1, lwork));
    dsytrf_(&u, &N, reinterpret_cast<double*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> wkopt; chetrf_(&u, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<float>> work(static_cast<std::size_t>(std::max(1, lwork)));
    chetrf_(&u, &N, reinterpret_cast<std::complex<float>*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> wkopt; zhetrf_(&u, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, nullptr, &wkopt, &lwork, &info);
    if (info != 0) return info;
    lwork = static_cast<int>(wkopt.real());
    std::vector<std::complex<double>> work(static_cast<std::size_t>(std::max(1, lwork)));
    zhetrf_(&u, &N, reinterpret_cast<std::complex<double>*>(a_cm.data()), &LDA, piv.data(), work.data(), &lwork, &info);
  } else {
    // Unsupported type by backend; fallback
    piv.resize(n);
    for (std::size_t i = 0; i < n; ++i) piv[i] = static_cast<int>(i + 1);
    return ldlt_factor_unpivoted_blocked(A, uplo);
  }
  if (info != 0) return info;
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      A(i, j) = a_cm[j * n + i];
  return 0;
#else
  // Fallback: robust unblocked BK-rook LDLT with pivoting
  return ldlt_factor_bk_rook(A, piv, uplo);
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
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || b.size() != n) throw std::invalid_argument("ldlt_solve_inplace(vector): dimension mismatch");

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Use xSYTRS/xHETRS by repacking to column-major
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A_fact(i, j);
  char u = (uplo == fem::numeric::linear_algebra::Uplo::Lower) ? 'L' : 'U';
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
  // Fallback pivoted solve using BK ipiv
  ldlt_solve_bk_inplace(A_fact, piv, b, uplo);
}

template <typename T>
void ldlt_solve_inplace(const Matrix<T>& A_fact,
                        const std::vector<int>& piv,
                        Matrix<T>& B,
                        fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  const std::size_t n = A_fact.rows();
  if (A_fact.cols() != n || B.rows() != n) throw std::invalid_argument("ldlt_solve_inplace(matrix): dimension mismatch");

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Prefer backend when available
  std::vector<T> a_cm(n * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      a_cm[j * n + i] = A_fact(i, j);
  char u = (uplo == Uplo::Lower) ? 'L' : 'U';
  int N = static_cast<int>(n), NRHS = static_cast<int>(B.cols()), LDA = static_cast<int>(n), LDB = static_cast<int>(B.cols());
  int info = 0;
  // Pack B (row-major) to column-major with leading dim n
  std::vector<T> b_cm(n * B.cols());
  for (std::size_t j = 0; j < B.cols(); ++j)
    for (std::size_t i = 0; i < n; ++i)
      b_cm[j * n + i] = B(i, j);
  if constexpr (std::is_same_v<T, float>) {
    ssytrs_(&u, &N, &NRHS, reinterpret_cast<const float*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<float*>(b_cm.data()), &LDA, &info);
  } else if constexpr (std::is_same_v<T, double>) {
    dsytrs_(&u, &N, &NRHS, reinterpret_cast<const double*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<double*>(b_cm.data()), &LDA, &info);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    chetrs_(&u, &N, &NRHS, reinterpret_cast<const std::complex<float>*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<std::complex<float>*>(b_cm.data()), &LDA, &info);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    zhetrs_(&u, &N, &NRHS, reinterpret_cast<const std::complex<double>*>(a_cm.data()), &LDA, piv.data(), reinterpret_cast<std::complex<double>*>(b_cm.data()), &LDA, &info);
  } else {
    info = -1;
  }
  if (info == 0) {
    // Unpack to B
    for (std::size_t j = 0; j < B.cols(); ++j)
      for (std::size_t i = 0; i < n; ++i)
        B(i, j) = b_cm[j * n + i];
    return;
  }
#endif

  // Fallback pivoted solve using BK ipiv
  ldlt_solve_bk_inplace(A_fact, piv, B, uplo);
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

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_LDLT_H

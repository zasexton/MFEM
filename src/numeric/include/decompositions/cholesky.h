#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_CHOLESKY_H
#define NUMERIC_DECOMPOSITIONS_CHOLESKY_H

// Dense Cholesky factorization (LL^T / U^T U) for SPD/HPD matrices.
// In-place factorization matching common BLAS/LAPACK semantics:
//  - If uplo == Lower: A becomes L with A = L * L^H (L is lower, diag real+)
//  - If uplo == Upper: A becomes U with A = U^H * U (U is upper, diag real+)
// Includes solves for vector/matrix RHS using the packed factor.

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level2.h" // for Uplo and conj helper pattern
#include "../linear_algebra/blas_level3.h" // for syrk/trsm/gemm
#include "../backends/lapack_backend.h"

namespace fem::numeric::decompositions {

// Small helper: conjugate only for complex types
template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) {
    using std::conj;
    return conj(x);
  } else {
    return x;
  }
}

// Forward declaration (blocked Cholesky)
template <typename T, typename Storage, StorageOrder Order>
int cholesky_factor_blocked(Matrix<T, Storage, Order>& A,
                            fem::numeric::linear_algebra::Uplo uplo,
                            std::size_t block = 64);

// ---------------------------------------------------------------------------
// Factorization: Cholesky (best path, in-place)
// - ColumnMajor + LAPACK: full POTRF (zero-copy)
// - Otherwise: blocked Cholesky (tile-level LAPACK for row-major if enabled)
// Returns 0 on success; returns i+1 on failure like LAPACK.
// ---------------------------------------------------------------------------

template <typename T, typename Storage, StorageOrder Order>
int cholesky_factor(Matrix<T, Storage, Order>& A,
                    fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  if constexpr (Order == StorageOrder::ColumnMajor) {
    const std::size_t n = A.rows();
    if (A.cols() != n) throw std::invalid_argument("cholesky_factor: matrix must be square");
    int N = static_cast<int>(n), lda = static_cast<int>(A.rows());
    int info = 0;
    char u = (uplo == fem::numeric::linear_algebra::Uplo::Lower) ? 'L' : 'U';
    backends::lapack::potrf_cm<T>(u, N, A.data(), lda, info);
    return info;
  }
#endif
  return cholesky_factor_blocked(A, uplo);
}

// ---------------------------------------------------------------------------
// Blocked Cholesky factorization (LL^H / U^H U), right-looking
// Uses Level-3 BLAS updates (TRSM + SYRK) on trailing matrix.
// Falls back to LAPACK backend if enabled (stubbed unless linked in CMake).
// Returns 0 on success; k+1 if leading minor of order k+1 is not SPD.
// ---------------------------------------------------------------------------
template <typename T, typename Storage, StorageOrder Order>
int cholesky_factor_blocked(Matrix<T, Storage, Order>& A,
                            fem::numeric::linear_algebra::Uplo uplo,
                            std::size_t block)
{
  using namespace fem::numeric::linear_algebra;

  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("cholesky_factor_blocked: matrix must be square");

  // Local helper: unblocked Cholesky on a view (Lower/Upper)
  auto chol_unblocked_view = [&](auto&& V, Uplo u) -> int {
    using R = typename numeric_traits<T>::scalar_type;
    const std::size_t m = V.rows();
    if (V.cols() != m) throw std::invalid_argument("chol_unblocked_view: square view required");
    if (u == Uplo::Lower) {
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
          auto sum = V(i, j);
          for (std::size_t k = 0; k < j; ++k) sum -= V(i, k) * conj_if_complex(V(j, k));
          V(i, j) = sum / V(j, j);
        }
        using std::real;
        R diag = static_cast<R>(real(V(i, i)));
        for (std::size_t k = 0; k < i; ++k) {
          auto lik = V(i, k);
          R mag2;
          if constexpr (is_complex_number_v<T>) {
            mag2 = static_cast<R>(std::norm(lik));
          } else {
            mag2 = static_cast<R>(lik * lik);
          }
          diag -= mag2;
        }
        if (!(diag > R{0})) return static_cast<int>(i) + 1;
        auto rii = static_cast<R>(std::sqrt(diag));
        V(i, i) = static_cast<T>(rii);
        for (std::size_t j = i + 1; j < m; ++j) V(i, j) = T{};
      }
    } else { // Upper
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
          auto sum = V(j, i);
          for (std::size_t k = 0; k < j; ++k) sum -= conj_if_complex(V(k, j)) * V(k, i);
          V(j, i) = sum / V(j, j);
        }
        using std::real;
        R diag = static_cast<R>(real(V(i, i)));
        for (std::size_t k = 0; k < i; ++k) {
          auto uki = V(k, i);
          R mag2;
          if constexpr (is_complex_number_v<T>) {
            mag2 = static_cast<R>(std::norm(uki));
          } else {
            mag2 = static_cast<R>(uki * uki);
          }
          diag -= mag2;
        }
        if (!(diag > R{0})) return static_cast<int>(i) + 1;
        auto rii = static_cast<R>(std::sqrt(diag));
        V(i, i) = static_cast<T>(rii);
        for (std::size_t j = 0; j < i; ++j) V(i, j) = T{};
      }
    }
    return 0;
  };

  if (n == 0) return 0;
  const std::size_t bs = std::max<std::size_t>(1, block);

  if (uplo == Uplo::Lower) {
    for (std::size_t k = 0; k < n; k += bs) {
      const std::size_t kb = std::min(bs, n - k);
      // Factor diagonal block (prefer LAPACK on tile if available)
      auto Akk = A.submatrix(k, k + kb, k, k + kb);
      int info_blk = 0;
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
      if constexpr (Order == StorageOrder::ColumnMajor) {
        char u = 'L'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(A.rows());
        backends::lapack::potrf_cm<T>(u, Nkb, Akk.data(), lda, info_blk);
      }
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
      else if constexpr (Order == StorageOrder::RowMajor) {
        char u = 'L'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(A.cols());
        // Akk.data() points at &A(k,k)
        info_blk = backends::lapack::potrf_rm<T>(u, Nkb, Akk.data(), lda);
      } else {
#else
      else {
#endif
        // Copy kb x kb tile to column-major buffer
        std::vector<T> tile(kb * kb);
        for (std::size_t j = 0; j < kb; ++j)
          for (std::size_t i = 0; i < kb; ++i)
            tile[j * kb + i] = Akk(i, j);
        char u = 'L'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(kb);
        backends::lapack::potrf_cm<T>(u, Nkb, tile.data(), lda, info_blk);
        if (info_blk == 0) {
          for (std::size_t i = 0; i < kb; ++i) {
            for (std::size_t j = 0; j < kb; ++j) Akk(i, j) = (j <= i) ? tile[j * kb + i] : T{};
          }
        }
      }
      if (info_blk != 0) {
        int ret = (info_blk > 0) ? static_cast<int>(k) + info_blk : info_blk;
        return ret;
      }
#else
      info_blk = chol_unblocked_view(Akk, Uplo::Lower);
      if (info_blk != 0) {
        int ret = (info_blk > 0) ? static_cast<int>(k) + info_blk : info_blk;
        return ret;
      }
#endif

      // Panel solve: L_ik = A_ik * inv(L_kk^H)
      if (k + kb < n) {
        auto Aik = A.submatrix(k + kb, n, k, k + kb); // (n-kb-k) x kb
        trsm(Side::Right, Uplo::Lower,
             Trans::ConjTranspose, Diag::NonUnit,
             T{1}, Akk, Aik);

        // Trailing update: A22 := A22 - Aik * Aik^H (SYRK on Lower)
        auto A22 = A.submatrix(k + kb, n, k + kb, n);
        // C = alpha*A*A^H + beta*C with alpha = -1, beta = 1
        syrk(Uplo::Lower, Trans::NoTrans, T{-1}, Aik, T{1}, A22);
      }
    }
  } else { // Upper
    for (std::size_t k = 0; k < n; k += bs) {
      const std::size_t kb = std::min(bs, n - k);
      // Factor diagonal block (Upper) with LAPACK if available
      auto Akk = A.submatrix(k, k + kb, k, k + kb);
      int info_blk = 0;
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
      if constexpr (Order == StorageOrder::ColumnMajor) {
        char u = 'U'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(A.rows());
        backends::lapack::potrf_cm<T>(u, Nkb, Akk.data(), lda, info_blk);
      }
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
      else if constexpr (Order == StorageOrder::RowMajor) {
        char u = 'U'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(A.cols());
        info_blk = backends::lapack::potrf_rm<T>(u, Nkb, Akk.data(), lda);
      } else {
#else
      else {
#endif
        std::vector<T> tile(kb * kb);
        for (std::size_t j = 0; j < kb; ++j)
          for (std::size_t i = 0; i < kb; ++i)
            tile[j * kb + i] = Akk(i, j);
        char u = 'U'; int Nkb = static_cast<int>(kb); int lda = static_cast<int>(kb);
        backends::lapack::potrf_cm<T>(u, Nkb, tile.data(), lda, info_blk);
        if (info_blk == 0) {
          for (std::size_t i = 0; i < kb; ++i) {
            for (std::size_t j = 0; j < kb; ++j) Akk(i, j) = (j >= i) ? tile[j * kb + i] : T{};
          }
        }
      }
      if (info_blk != 0) {
        int ret = (info_blk > 0) ? static_cast<int>(k) + info_blk : info_blk;
        return ret;
      }
#else
      info_blk = chol_unblocked_view(Akk, Uplo::Upper);
      if (info_blk != 0) {
        int ret = (info_blk > 0) ? static_cast<int>(k) + info_blk : info_blk;
        return ret;
      }
#endif

      if (k + kb < n) {
        // Panel: U_jk solve: U_ki = inv(U_kk^H) * A_ki^H (equivalently right with Upper)
        auto Aki = A.submatrix(k, k + kb, k + kb, n); // kb x (n-kb-k)
        trsm(Side::Left, Uplo::Upper,
             Trans::ConjTranspose, Diag::NonUnit,
             T{1}, Akk, Aki);

        // Update trailing: A22 := A22 - Aki^H * Aki (Upper)
        auto A22 = A.submatrix(k + kb, n, k + kb, n);
        // Using syrk on Upper with transA = ConjTranspose so A^H*A
        syrk(Uplo::Upper, Trans::ConjTranspose, T{-1}, Aki, T{1}, A22);
      }
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Solve using packed Cholesky factor (in place on b/B)
// For Lower: solve L y = b, then L^H x = y
// For Upper: solve U^H y = b, then U x = y
// ---------------------------------------------------------------------------

template <typename T>
void cholesky_solve_inplace(const Matrix<T>& chol,
                            Vector<T>& b,
                            fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  if (chol.rows() != chol.cols() || b.size() != chol.rows()) {
    throw std::invalid_argument("cholesky_solve_inplace(vector): dimension mismatch");
  }
  const std::size_t n = chol.rows();

  if (uplo == Uplo::Lower) {
    // Forward solve: L y = b (overwrite b with y)
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = b[i];
      for (std::size_t k = 0; k < i; ++k) sum -= chol(i, k) * b[k];
      b[i] = sum / chol(i, i);
    }
    // Backward solve: L^H x = y
    for (std::size_t ii = 0; ii < n; ++ii) {
      std::size_t i = n - 1 - ii;
      auto sum = b[i];
      for (std::size_t k = i + 1; k < n; ++k) sum -= conj_if_complex(chol(k, i)) * b[k];
      b[i] = sum / chol(i, i);
    }
  } else { // Upper
    // Forward solve: U^H y = b (U^H is lower)
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = b[i];
      for (std::size_t k = 0; k < i; ++k) sum -= conj_if_complex(chol(k, i)) * b[k];
      b[i] = sum / chol(i, i);
    }
    // Backward solve: U x = y (U is upper)
    for (std::size_t ii = 0; ii < n; ++ii) {
      std::size_t i = n - 1 - ii;
      auto sum = b[i];
      for (std::size_t k = i + 1; k < n; ++k) sum -= chol(i, k) * b[k];
      b[i] = sum / chol(i, i);
    }
  }
}

template <typename T>
void cholesky_solve_inplace(const Matrix<T>& chol,
                            Matrix<T>& B,
                            fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  if (chol.rows() != chol.cols() || B.rows() != chol.rows()) {
    throw std::invalid_argument("cholesky_solve_inplace(matrix): dimension mismatch");
  }
  const std::size_t n = chol.rows();
  const std::size_t nrhs = B.cols();

  if (uplo == Uplo::Lower) {
    // L y = B
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < nrhs; ++j) {
        auto sum = B(i, j);
        for (std::size_t k = 0; k < i; ++k) sum -= chol(i, k) * B(k, j);
        B(i, j) = sum / chol(i, i);
      }
    }
    // L^H x = y
    for (std::size_t ii = 0; ii < n; ++ii) {
      std::size_t i = n - 1 - ii;
      for (std::size_t j = 0; j < nrhs; ++j) {
        auto sum = B(i, j);
        for (std::size_t k = i + 1; k < n; ++k) sum -= conj_if_complex(chol(k, i)) * B(k, j);
        B(i, j) = sum / chol(i, i);
      }
    }
  } else { // Upper
    // U^H y = B
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < nrhs; ++j) {
        auto sum = B(i, j);
        for (std::size_t k = 0; k < i; ++k) sum -= conj_if_complex(chol(k, i)) * B(k, j);
        B(i, j) = sum / chol(i, i);
      }
    }
    // U x = y
    for (std::size_t ii = 0; ii < n; ++ii) {
      std::size_t i = n - 1 - ii;
      for (std::size_t j = 0; j < nrhs; ++j) {
        auto sum = B(i, j);
        for (std::size_t k = i + 1; k < n; ++k) sum -= chol(i, k) * B(k, j);
        B(i, j) = sum / chol(i, i);
      }
    }
  }
}

template <typename T>
Vector<T> cholesky_solve(const Matrix<T>& chol,
                         const Vector<T>& b,
                         fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  Vector<T> x = b;
  cholesky_solve_inplace(chol, x, uplo);
  return x;
}

template <typename T>
Matrix<T> cholesky_solve(const Matrix<T>& chol,
                         const Matrix<T>& B,
                         fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  Matrix<T> X = B;
  cholesky_solve_inplace(chol, X, uplo);
  return X;
}

// ---------------------------------------------------------------------------
// Determinant from Cholesky factor: det(A) = (prod diag)^2
// For complex T, returns a complex with zero imaginary (positive real value)
// ---------------------------------------------------------------------------

template <typename T>
T cholesky_determinant(const Matrix<T>& chol)
{
  const std::size_t n = chol.rows();
  if (chol.cols() != n) throw std::invalid_argument("cholesky_determinant: matrix must be square");
  T det = T{1};
  for (std::size_t i = 0; i < n; ++i) det *= chol(i, i) * chol(i, i);
  return det;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_CHOLESKY_H

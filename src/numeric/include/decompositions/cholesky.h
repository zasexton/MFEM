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

// ---------------------------------------------------------------------------
// Factorization: Cholesky (in-place)
// Returns 0 on success; returns i+1 if the leading i-by-i minor is not
// positive-definite (i.e., a non-positive pivot encountered at step i).
// ---------------------------------------------------------------------------

template <typename T>
int cholesky_factor(Matrix<T>& A,
                    fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower)
{
  using Uplo = fem::numeric::linear_algebra::Uplo;
  using R = typename numeric_traits<T>::scalar_type;

  const std::size_t n = A.rows();
  if (A.cols() != n) {
    throw std::invalid_argument("cholesky_factor: matrix must be square");
  }

  if (uplo == Uplo::Lower) {
    // Compute lower-triangular L such that A = L * L^H
    for (std::size_t i = 0; i < n; ++i) {
      // Off-diagonal elements in row i (columns j < i)
      for (std::size_t j = 0; j < i; ++j) {
        // A(i,j) := (A(i,j) - sum_{k<j} A(i,k) * conj(A(j,k))) / A(j,j)
        auto sum = A(i, j);
        for (std::size_t k = 0; k < j; ++k) {
          sum -= A(i, k) * conj_if_complex(A(j, k));
        }
        A(i, j) = sum / A(j, j);
      }
      // Diagonal: A(i,i) := sqrt( Re(A(i,i) - sum_{k<i} |A(i,k)|^2) )
      using std::real;
      R diag = static_cast<R>(real(A(i, i)));
      for (std::size_t k = 0; k < i; ++k) {
        auto lik = A(i, k);
        R mag2;
        if constexpr (is_complex_number_v<T>) {
          mag2 = static_cast<R>(std::norm(lik));
        } else {
          mag2 = static_cast<R>(lik * lik);
        }
        diag -= mag2;
      }
      if (!(diag > R{0})) {
        // Not strictly positive definite
        return static_cast<int>(i) + 1;
      }
      auto rii = static_cast<R>(std::sqrt(diag));
      A(i, i) = static_cast<T>(rii); // Diagonal is real-positive

      // Zero out the upper part to maintain a clean packed L (optional)
      for (std::size_t j = i + 1; j < n; ++j) A(i, j) = T{};
    }
  } else { // Uplo::Upper
    // Compute upper-triangular U such that A = U^H * U
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        // A(j,i) := (A(j,i) - sum_{k<j} conj(A(k,j)) * A(k,i)) / A(j,j)
        auto sum = A(j, i);
        for (std::size_t k = 0; k < j; ++k) {
          sum -= conj_if_complex(A(k, j)) * A(k, i);
        }
        A(j, i) = sum / A(j, j);
      }
      // Diagonal: A(i,i) := sqrt( Re(A(i,i) - sum_{k<i} |A(k,i)|^2) )
      using std::real;
      R diag = static_cast<R>(real(A(i, i)));
      for (std::size_t k = 0; k < i; ++k) {
        auto uki = A(k, i);
        R mag2;
        if constexpr (is_complex_number_v<T>) {
          mag2 = static_cast<R>(std::norm(uki));
        } else {
          mag2 = static_cast<R>(uki * uki);
        }
        diag -= mag2;
      }
      if (!(diag > R{0})) {
        return static_cast<int>(i) + 1;
      }
      auto rii = static_cast<R>(std::sqrt(diag));
      A(i, i) = static_cast<T>(rii);

      // Zero the lower part for a clean packed U (optional)
      for (std::size_t j = 0; j < i; ++j) A(i, j) = T{};
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

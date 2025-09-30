#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL2_H
#define NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL2_H

// BLAS Level-2 style routines: matrix–vector operations and rank-1 updates.
// Header-only, templated, and compatible with fem::numeric containers and views.

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <limits>
#include <algorithm>

#include "../base/traits_base.h"

namespace fem::numeric::linear_algebra {

// ---------------------------------------------------------------------------
// Concepts and helpers
// ---------------------------------------------------------------------------

template <typename V>
concept VectorLike = requires(V v) {
  { v.size() } -> std::convertible_to<std::size_t>;
  { v[std::size_t{0}] };
};

template <typename M>
concept MatrixLike = requires(M a) {
  { a.rows() } -> std::convertible_to<std::size_t>;
  { a.cols() } -> std::convertible_to<std::size_t>;
  { a(std::size_t{0}, std::size_t{0}) };
};

template <typename V>
using vec_elem_t = std::remove_reference_t<decltype(std::declval<V&>()[std::declval<std::size_t>()])>;

template <typename M>
using mat_elem_t = std::remove_reference_t<decltype(std::declval<M&>()(std::declval<std::size_t>(), std::declval<std::size_t>()))>;

template <typename T>
constexpr auto conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) {
    using std::conj;
    return conj(x);
  } else {
    return x;
  }
}

// Operation specifiers to match BLAS conventions
enum class Trans { NoTrans, Transpose, ConjTranspose };
enum class Uplo  { Upper, Lower };
enum class Layout { RowMajor, ColMajor };
enum class Side  { Left, Right };
enum class Diag  { Unit, NonUnit };

// ---------------------------------------------------------------------------
// GEMV: y := alpha * op(A) * x + beta * y
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename X, typename Beta, typename Y>
  requires (MatrixLike<A> && VectorLike<X> && VectorLike<Y>)
inline void gemv(Trans trans, const Alpha& alpha, const A& A_, const X& x,
                 const Beta& beta, Y& y)
{
  using TY = vec_elem_t<Y>;

  const std::size_t m = A_.rows();
  const std::size_t n = A_.cols();
  const bool do_trans = (trans != Trans::NoTrans);
  const bool do_conj  = (trans == Trans::ConjTranspose);

  const std::size_t len_y = do_trans ? n : m;
  const std::size_t len_x = do_trans ? m : n;

  if (x.size() != len_x || y.size() != len_y) {
    throw std::invalid_argument("gemv: size mismatch");
  }

  // y := beta * y
  if constexpr (std::is_same_v<Beta, int> || std::is_floating_point_v<Beta> || is_complex_number_v<Beta>) {
    if (beta == Beta{}) {
      for (std::size_t i = 0; i < len_y; ++i) y[i] = TY{};
    } else if (!(beta == Beta{1})) {
      for (std::size_t i = 0; i < len_y; ++i) y[i] = static_cast<TY>(beta * y[i]);
    }
  } else {
    for (std::size_t i = 0; i < len_y; ++i) y[i] = static_cast<TY>(beta * y[i]);
  }

  if (!do_trans) {
    // y_i += alpha * sum_j A(i,j) * x_j
    for (std::size_t i = 0; i < m; ++i) {
      auto sum = decltype(alpha * A_(i, std::size_t{0}) * x[std::size_t{0}]){};
      for (std::size_t j = 0; j < n; ++j) {
        sum += static_cast<decltype(sum)>(A_(i, j)) * static_cast<decltype(sum)>(x[j]);
      }
      y[i] = static_cast<TY>(y[i] + alpha * sum);
    }
  } else {
    // y_i += alpha * sum_j op(A)(i,j) * x_j = alpha * sum_j A(j,i) * x_j (with optional conj)
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = decltype(alpha * A_(std::size_t{0}, i) * x[std::size_t{0}]){};
      for (std::size_t j = 0; j < m; ++j) {
        auto aji = A_(j, i);
        if (do_conj) aji = conj_if_complex(aji);
        sum += static_cast<decltype(sum)>(aji) * static_cast<decltype(sum)>(x[j]);
      }
      y[i] = static_cast<TY>(y[i] + alpha * sum);
    }
  }
}

// Raw-pointer version (A in row-major or column-major with leading dimension)
template <typename Alpha, typename TA, typename TX, typename Beta, typename TY>
inline void gemv(Layout layout, Trans trans, std::size_t m, std::size_t n,
                 const Alpha& alpha, const TA* A, std::size_t lda,
                 const TX* x, std::ptrdiff_t incx,
                 const Beta& beta, TY* y, std::ptrdiff_t incy)
{
  if (!A || !x || !y) return;
  const bool do_trans = (trans != Trans::NoTrans);
  const bool do_conj  = (trans == Trans::ConjTranspose);

  const std::size_t len_y = do_trans ? n : m;
  const std::size_t len_x = do_trans ? m : n;

  // y := beta * y
  if (beta == Beta{}) {
    std::ptrdiff_t iy = 0;
    for (std::size_t i = 0; i < len_y; ++i, iy += incy) y[iy] = TY{};
  } else if (!(beta == Beta{1})) {
    std::ptrdiff_t iy = 0;
    for (std::size_t i = 0; i < len_y; ++i, iy += incy) y[iy] = static_cast<TY>(beta * y[iy]);
  }

  if (layout == Layout::RowMajor) {
    if (!do_trans) {
      // A(i,j) at A[i*lda + j]
      std::ptrdiff_t iy = 0;
      for (std::size_t i = 0; i < m; ++i, iy += incy) {
        auto sum = decltype(alpha * A[0] * x[0]){};
        std::ptrdiff_t ix = 0;
        const std::size_t row_off = i * lda;
        for (std::size_t j = 0; j < n; ++j, ix += incx) {
          sum += static_cast<decltype(sum)>(A[row_off + j]) * static_cast<decltype(sum)>(x[ix]);
        }
        y[iy] = static_cast<TY>(y[iy] + alpha * sum);
      }
    } else {
      // op(A) = A^T or conj(A)^T -> y size n
      std::ptrdiff_t iy = 0;
      for (std::size_t i = 0; i < n; ++i, iy += incy) {
        auto sum = decltype(alpha * A[0] * x[0]){};
        std::ptrdiff_t ix = 0;
        for (std::size_t j = 0; j < m; ++j, ix += incx) {
          auto aji = A[j * lda + i];
          if (do_conj) aji = conj_if_complex(aji);
          sum += static_cast<decltype(sum)>(aji) * static_cast<decltype(sum)>(x[ix]);
        }
        y[iy] = static_cast<TY>(y[iy] + alpha * sum);
      }
    }
  } else { // ColMajor
    if (!do_trans) {
      // A(i,j) at A[i + j*lda] -> y m-length
      // y += alpha * sum_j A(:,j) * x_j
      std::ptrdiff_t ix = 0;
      for (std::size_t j = 0; j < n; ++j, ix += incx) {
        auto xj = x[ix];
        if (xj != TX{}) {
          std::ptrdiff_t iy = 0;
          const std::size_t col_off = j * lda;
          for (std::size_t i = 0; i < m; ++i, iy += incy) {
            y[iy] = static_cast<TY>(y[iy] + alpha * static_cast<decltype(alpha * A[0])>(A[col_off + i]) * xj);
          }
        }
      }
    } else {
      // y length n
      std::ptrdiff_t iy = 0;
      for (std::size_t i = 0; i < n; ++i, iy += incy) {
        auto sum = decltype(alpha * A[0] * x[0]){};
        std::ptrdiff_t ix = 0;
        for (std::size_t j = 0; j < m; ++j, ix += incx) {
          auto aji = A[i * lda + j]; // since transposing column-major -> treat rows
          if (do_conj) aji = conj_if_complex(aji);
          sum += static_cast<decltype(sum)>(aji) * static_cast<decltype(sum)>(x[ix]);
        }
        y[iy] = static_cast<TY>(y[iy] + alpha * sum);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// GER/GERC: A := alpha * x * y^T (+ conj on y for gerc) + A
// ---------------------------------------------------------------------------

template <typename Alpha, typename X, typename Y, typename A>
  requires (VectorLike<X> && VectorLike<Y> && MatrixLike<A>)
inline void ger(const Alpha& alpha, const X& x, const Y& y, A& A_)
{
  const std::size_t m = A_.rows();
  const std::size_t n = A_.cols();
  if (x.size() != m || y.size() != n) {
    throw std::invalid_argument("ger: size mismatch");
  }
  for (std::size_t i = 0; i < m; ++i) {
    auto xi = x[i];
    if (xi == decltype(xi){}) continue;
    for (std::size_t j = 0; j < n; ++j) {
      A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * y[j]);
    }
  }
}

template <typename Alpha, typename X, typename Y, typename A>
  requires (VectorLike<X> && VectorLike<Y> && MatrixLike<A>)
inline void gerc(const Alpha& alpha, const X& x, const Y& y, A& A_)
{
  const std::size_t m = A_.rows();
  const std::size_t n = A_.cols();
  if (x.size() != m || y.size() != n) {
    throw std::invalid_argument("gerc: size mismatch");
  }
  for (std::size_t i = 0; i < m; ++i) {
    auto xi = x[i];
    if (xi == decltype(xi){}) continue;
    for (std::size_t j = 0; j < n; ++j) {
      auto yj = y[j];
      yj = conj_if_complex(yj);
      A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * yj);
    }
  }
}

// Raw-pointer GER/GERC (row- or column-major)
template <typename Alpha, typename TX, typename TY, typename TA>
inline void ger(Layout layout, std::size_t m, std::size_t n, const Alpha& alpha,
                const TX* x, std::ptrdiff_t incx, const TY* y, std::ptrdiff_t incy,
                TA* A, std::size_t lda)
{
  if (!A || !x || !y) return;
  if (layout == Layout::RowMajor) {
    std::ptrdiff_t ix = 0;
    for (std::size_t i = 0; i < m; ++i, ix += incx) {
      auto xi = x[ix];
      if (xi == TX{}) continue;
      const std::size_t row_off = i * lda;
      std::ptrdiff_t iy = 0;
      for (std::size_t j = 0; j < n; ++j, iy += incy) {
        A[row_off + j] = static_cast<TA>(A[row_off + j] + alpha * xi * y[iy]);
      }
    }
  } else { // ColMajor
    std::ptrdiff_t iy = 0;
    for (std::size_t j = 0; j < n; ++j, iy += incy) {
      auto yj = y[iy];
      if (yj == TY{}) continue;
      const std::size_t col_off = j * lda;
      std::ptrdiff_t ix = 0;
      for (std::size_t i = 0; i < m; ++i, ix += incx) {
        A[col_off + i] = static_cast<TA>(A[col_off + i] + alpha * x[ix] * yj);
      }
    }
  }
}

template <typename Alpha, typename TX, typename TY, typename TA>
inline void gerc(Layout layout, std::size_t m, std::size_t n, const Alpha& alpha,
                 const TX* x, std::ptrdiff_t incx, const TY* y, std::ptrdiff_t incy,
                 TA* A, std::size_t lda)
{
  if (!A || !x || !y) return;
  if (layout == Layout::RowMajor) {
    std::ptrdiff_t ix = 0;
    for (std::size_t i = 0; i < m; ++i, ix += incx) {
      auto xi = x[ix];
      if (xi == TX{}) continue;
      const std::size_t row_off = i * lda;
      std::ptrdiff_t iy = 0;
      for (std::size_t j = 0; j < n; ++j, iy += incy) {
        auto yj = conj_if_complex(y[iy]);
        A[row_off + j] = static_cast<TA>(A[row_off + j] + alpha * xi * yj);
      }
    }
  } else { // ColMajor
    std::ptrdiff_t iy = 0;
    for (std::size_t j = 0; j < n; ++j, iy += incy) {
      auto yj = conj_if_complex(y[iy]);
      if (yj == TY{}) continue;
      const std::size_t col_off = j * lda;
      std::ptrdiff_t ix = 0;
      for (std::size_t i = 0; i < m; ++i, ix += incx) {
        A[col_off + i] = static_cast<TA>(A[col_off + i] + alpha * x[ix] * yj);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYMV/HEMV: y := alpha * A * x + beta * y  (symmetric/Hermitian)
// ---------------------------------------------------------------------------

// Symmetric (real / non-conjugate)
template <typename Alpha, typename A, typename X, typename Beta, typename Y>
  requires (MatrixLike<A> && VectorLike<X> && VectorLike<Y>)
inline void symv(Uplo uplo, const Alpha& alpha, const A& A_, const X& x,
                 const Beta& beta, Y& y)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n || y.size() != n) {
    throw std::invalid_argument("symv: size mismatch");
  }
  // y := beta*y
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < n; ++i) y[i] = vec_elem_t<Y>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < n; ++i) y[i] = static_cast<vec_elem_t<Y>>(beta * y[i]);
  }

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = decltype(alpha * A_(i, i) * x[i]){};
      sum += static_cast<decltype(sum)>(A_(i, i)) * static_cast<decltype(sum)>(x[i]);
      for (std::size_t j = i + 1; j < n; ++j) {
        auto aij = A_(i, j);
        sum += static_cast<decltype(sum)>(aij) * static_cast<decltype(sum)>(x[j]);
        y[j] = static_cast<vec_elem_t<Y>>(y[j] + alpha * aij * x[i]);
      }
      y[i] = static_cast<vec_elem_t<Y>>(y[i] + alpha * sum);
    }
  } else { // Lower
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = decltype(alpha * A_(i, i) * x[i]){};
      sum += static_cast<decltype(sum)>(A_(i, i)) * static_cast<decltype(sum)>(x[i]);
      for (std::size_t j = 0; j < i; ++j) {
        auto aij = A_(i, j);
        sum += static_cast<decltype(sum)>(aij) * static_cast<decltype(sum)>(x[j]);
        y[j] = static_cast<vec_elem_t<Y>>(y[j] + alpha * aij * x[i]);
      }
      y[i] = static_cast<vec_elem_t<Y>>(y[i] + alpha * sum);
    }
  }
}

// Hermitian (complex): uses conjugate on the symmetric counterpart
template <typename Alpha, typename A, typename X, typename Beta, typename Y>
  requires (MatrixLike<A> && VectorLike<X> && VectorLike<Y>)
inline void hemv(Uplo uplo, const Alpha& alpha, const A& A_, const X& x,
                 const Beta& beta, Y& y)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n || y.size() != n) {
    throw std::invalid_argument("hemv: size mismatch");
  }
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < n; ++i) y[i] = vec_elem_t<Y>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < n; ++i) y[i] = static_cast<vec_elem_t<Y>>(beta * y[i]);
  }

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = decltype(alpha * A_(i, i) * x[i]){};
      // Diagonal of Hermitian is real in exact math; we just use stored value
      sum += static_cast<decltype(sum)>(A_(i, i)) * static_cast<decltype(sum)>(x[i]);
      for (std::size_t j = i + 1; j < n; ++j) {
        auto aij = A_(i, j);
        sum += static_cast<decltype(sum)>(aij) * static_cast<decltype(sum)>(x[j]);
        y[j] = static_cast<vec_elem_t<Y>>(y[j] + alpha * conj_if_complex(aij) * x[i]);
      }
      y[i] = static_cast<vec_elem_t<Y>>(y[i] + alpha * sum);
    }
  } else { // Lower
    for (std::size_t i = 0; i < n; ++i) {
      auto sum = decltype(alpha * A_(i, i) * x[i]){};
      sum += static_cast<decltype(sum)>(A_(i, i)) * static_cast<decltype(sum)>(x[i]);
      for (std::size_t j = 0; j < i; ++j) {
        auto aij = A_(i, j);
        sum += static_cast<decltype(sum)>(aij) * static_cast<decltype(sum)>(x[j]);
        y[j] = static_cast<vec_elem_t<Y>>(y[j] + alpha * conj_if_complex(aij) * x[i]);
      }
      y[i] = static_cast<vec_elem_t<Y>>(y[i] + alpha * sum);
    }
  }
}

// ---------------------------------------------------------------------------
// SYR/HER: A := alpha * x * x^T (+ conj on second x for HER) + A (triangular part)
// ---------------------------------------------------------------------------

template <typename Alpha, typename X, typename A>
  requires (VectorLike<X> && MatrixLike<A>)
inline void syr(Uplo uplo, const Alpha& alpha, const X& x, A& A_)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n) {
    throw std::invalid_argument("syr: size mismatch");
  }
  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = i; j < n; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * x[j]);
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = 0; j <= i; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * x[j]);
      }
    }
  }
}

template <typename Alpha, typename X, typename A>
  requires (VectorLike<X> && MatrixLike<A>)
inline void her(Uplo uplo, const Alpha& alpha, const X& x, A& A_)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n) {
    throw std::invalid_argument("her: size mismatch");
  }
  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = i; j < n; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * conj_if_complex(x[j]));
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = 0; j <= i; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * xi * conj_if_complex(x[j]));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYR2/HER2: A := alpha * x * y^T + alpha * y * x^T (+ conj on second vec for HER2) + A
// ---------------------------------------------------------------------------

template <typename Alpha, typename X, typename Y, typename A>
  requires (VectorLike<X> && VectorLike<Y> && MatrixLike<A>)
inline void syr2(Uplo uplo, const Alpha& alpha, const X& x, const Y& y, A& A_)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n || y.size() != n) {
    throw std::invalid_argument("syr2: size mismatch");
  }
  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = i; j < n; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * (xi * y[j] + y[i] * x[j]));
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = 0; j <= i; ++j) {
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * (xi * y[j] + y[i] * x[j]));
      }
    }
  }
}

template <typename Alpha, typename X, typename Y, typename A>
  requires (VectorLike<X> && VectorLike<Y> && MatrixLike<A>)
inline void her2(Uplo uplo, const Alpha& alpha, const X& x, const Y& y, A& A_)
{
  const std::size_t n = A_.rows();
  if (A_.cols() != n || x.size() != n || y.size() != n) {
    throw std::invalid_argument("her2: size mismatch");
  }
  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = i; j < n; ++j) {
        auto yj = y[j];
        auto xj = x[j];
        auto yi = y[i];
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * (xi * conj_if_complex(yj) + yi * conj_if_complex(xj)));
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      auto xi = x[i];
      for (std::size_t j = 0; j <= i; ++j) {
        auto yj = y[j];
        auto xj = x[j];
        auto yi = y[i];
        A_(i, j) = static_cast<mat_elem_t<A>>(A_(i, j) + alpha * (xi * conj_if_complex(yj) + yi * conj_if_complex(xj)));
      }
    }
  }
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL2_H

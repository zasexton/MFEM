#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_NORMS_H
#define NUMERIC_LINEAR_ALGEBRA_NORMS_H

#include <type_traits>
#include <cstddef>
#include <cmath>
#include <limits>

#include "../base/traits_base.h"

namespace fem::numeric::linear_algebra {

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

template <typename T>
constexpr auto abs_val(const T& x) {
  if constexpr (is_complex_number_v<T>) return std::abs(x);
  else return (x < T{0} ? -x : x);
}

// Vector norms ----------------------------------------------------------------

template <VectorLike X>
inline auto nrm2(const X& x) {
  using T = std::remove_reference_t<decltype(x[std::size_t{0}])>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  R scale{0};
  R ssq{1};
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) {
    R a = static_cast<R>(abs_val(x[i]));
    if (a != R{0}) {
      if (scale < a) {
        ssq = R{1} + ssq * (scale / a) * (scale / a);
        scale = a;
      } else {
        ssq += (a / scale) * (a / scale);
      }
    }
  }
  return scale * std::sqrt(ssq);
}

template <VectorLike X>
inline auto norm1(const X& x) {
  using T = std::remove_reference_t<decltype(x[std::size_t{0}])>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  R sum{};
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<R>(abs_val(x[i]));
  return sum;
}

template <VectorLike X>
inline auto norm_inf(const X& x) {
  using T = std::remove_reference_t<decltype(x[std::size_t{0}])>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  R mx{};
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) {
    R v = static_cast<R>(abs_val(x[i]));
    if (v > mx) mx = v;
  }
  return mx;
}

// Matrix norms ----------------------------------------------------------------

template <MatrixLike A>
inline auto frobenius(const A& A_) {
  using T = std::remove_reference_t<decltype(A_(std::size_t{0}, std::size_t{0}))>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  R scale{0};
  R ssq{1};
  const std::size_t m = A_.rows(), n = A_.cols();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      R a = static_cast<R>(abs_val(A_(i, j)));
      if (a != R{0}) {
        if (scale < a) {
          ssq = R{1} + ssq * (scale / a) * (scale / a);
          scale = a;
        } else {
          ssq += (a / scale) * (a / scale);
        }
      }
    }
  }
  return scale * std::sqrt(ssq);
}

template <MatrixLike A>
inline auto one_norm(const A& A_) {
  using T = std::remove_reference_t<decltype(A_(std::size_t{0}, std::size_t{0}))>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  const std::size_t m = A_.rows(), n = A_.cols();
  R best{};
  for (std::size_t j = 0; j < n; ++j) {
    R col{};
    for (std::size_t i = 0; i < m; ++i) col += static_cast<R>(abs_val(A_(i, j)));
    if (col > best) best = col;
  }
  return best;
}

template <MatrixLike A>
inline auto inf_norm(const A& A_) {
  using T = std::remove_reference_t<decltype(A_(std::size_t{0}, std::size_t{0}))>;
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  const std::size_t m = A_.rows(), n = A_.cols();
  R best{};
  for (std::size_t i = 0; i < m; ++i) {
    R row{};
    for (std::size_t j = 0; j < n; ++j) row += static_cast<R>(abs_val(A_(i, j)));
    if (row > best) best = row;
  }
  return best;
}

// Pointer-based matrix norms -------------------------------------------------

enum class Layout { RowMajor, ColMajor };

template <typename T>
inline auto frobenius(std::size_t m, std::size_t n, const T* A, std::size_t lda, Layout layout)
{
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  auto at = [&](std::size_t i, std::size_t j) -> T {
    return (layout == Layout::RowMajor) ? A[i * lda + j] : A[i + j * lda];
  };
  R scale{0};
  R ssq{1};
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      R a = static_cast<R>(abs_val(at(i, j)));
      if (a != R{0}) {
        if (scale < a) { ssq = R{1} + ssq * (scale / a) * (scale / a); scale = a; }
        else { ssq += (a / scale) * (a / scale); }
      }
    }
  }
  return scale * std::sqrt(ssq);
}

template <typename T>
inline auto one_norm(std::size_t m, std::size_t n, const T* A, std::size_t lda, Layout layout)
{
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  auto at = [&](std::size_t i, std::size_t j) -> T {
    return (layout == Layout::RowMajor) ? A[i * lda + j] : A[i + j * lda];
  };
  R best{};
  for (std::size_t j = 0; j < n; ++j) {
    R col{};
    for (std::size_t i = 0; i < m; ++i) col += static_cast<R>(abs_val(at(i, j)));
    if (col > best) best = col;
  }
  return best;
}

template <typename T>
inline auto inf_norm(std::size_t m, std::size_t n, const T* A, std::size_t lda, Layout layout)
{
  using R = std::conditional_t<is_complex_number_v<T>, typename T::value_type, T>;
  auto at = [&](std::size_t i, std::size_t j) -> T {
    return (layout == Layout::RowMajor) ? A[i * lda + j] : A[i + j * lda];
  };
  R best{};
  for (std::size_t i = 0; i < m; ++i) {
    R row{};
    for (std::size_t j = 0; j < n; ++j) row += static_cast<R>(abs_val(at(i, j)));
    if (row > best) best = row;
  }
  return best;
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_NORMS_H


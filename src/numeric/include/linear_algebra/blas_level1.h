#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL1_H
#define NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL1_H

// BLAS Level-1 style routines for vectors and vector-like views.
// The API is header-only, templated, and integrates with fem::numeric containers.
// Functions operate on any type that provides size() and operator[] access.

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <limits>
#include <algorithm>
#include <numeric>
#include <execution>

#include "../base/traits_base.h"

namespace fem::numeric::linear_algebra {

// Small utilities -----------------------------------------------------------

template <typename V>
concept VectorLike = requires(V v) {
  { v.size() } -> std::convertible_to<std::size_t>;
  { v[std::size_t{0}] };
};

template <typename V>
using element_t = std::remove_reference_t<decltype(std::declval<V&>()[std::declval<std::size_t>()])>;

template <typename T>
constexpr auto conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) {
    using std::conj;
    return conj(x);
  } else {
    return x;
  }
}

template <typename T>
constexpr auto abs_blas1(const T& x) {
  if constexpr (is_complex_number_v<T>) {
    // BLAS asum/iamax for complex uses |Re| + |Im|
    using R = typename T::value_type;
    return static_cast<R>(std::abs(x.real())) + static_cast<R>(std::abs(x.imag()));
  } else {
    using std::abs;
    return abs(x);
  }
}

// 1) scal: x := alpha * x ---------------------------------------------------

template <typename Alpha, typename X>
  requires VectorLike<X>
inline void scal(const Alpha& alpha, X& x)
{
  using T = element_t<X>;
  const std::size_t n = x.size();

  if (n == 0) return;
  if constexpr (std::is_arithmetic_v<Alpha> && std::is_same_v<Alpha, T>) {
    // Fast path: same type, allow vectorization
    std::transform(std::execution::par_unseq, x.begin(), x.end(), x.begin(),
                   [&alpha](const T& v) { return v * alpha; });
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = static_cast<T>(alpha * x[i]);
    }
  }
}

template <typename Alpha, typename TX>
inline void scal(std::size_t n, const Alpha& alpha, TX* x, std::ptrdiff_t incx = 1)
{
  if (!x || n == 0) return;
  if (incx == 1) {
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<TX>(alpha * x[i]);
  } else {
    std::ptrdiff_t idx = 0;
    for (std::size_t i = 0; i < n; ++i, idx += incx) x[idx] = static_cast<TX>(alpha * x[idx]);
  }
}

// 2) axpy: y := alpha * x + y ----------------------------------------------

template <typename Alpha, typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline void axpy(const Alpha& alpha, const X& x, Y& y)
{
  using TY = element_t<Y>;
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("axpy: size mismatch");
  if (n == 0) return;

  if constexpr (std::is_same_v<element_t<X>, element_t<Y>> && std::is_arithmetic_v<Alpha>) {
    std::transform(std::execution::par_unseq,
                   x.begin(), x.end(), y.begin(), y.begin(),
                   [&alpha](const auto& xv, const auto& yv) { return static_cast<TY>(alpha * xv + yv); });
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = static_cast<TY>(alpha * x[i] + y[i]);
    }
  }
}

template <typename Alpha, typename TX, typename TY>
inline void axpy(std::size_t n, const Alpha& alpha, const TX* x, std::ptrdiff_t incx,
                 TY* y, std::ptrdiff_t incy)
{
  if (!x || !y || n == 0) return;
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) {
    y[iy] = static_cast<TY>(alpha * x[ix] + y[iy]);
  }
}

// 3) copy: y := x -----------------------------------------------------------

template <typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline void copy(const X& x, Y& y)
{
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("copy: size mismatch");
  if (n == 0) return;
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i];
}

template <typename TX, typename TY>
inline void copy(std::size_t n, const TX* x, std::ptrdiff_t incx, TY* y, std::ptrdiff_t incy)
{
  if (!x || !y || n == 0) return;
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) y[iy] = static_cast<TY>(x[ix]);
}

// 4) swap: swap x and y -----------------------------------------------------

template <typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline void swap(X& x, Y& y)
{
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("swap: size mismatch");
  for (std::size_t i = 0; i < n; ++i) std::swap(x[i], y[i]);
}

template <typename TX, typename TY>
inline void swap(std::size_t n, TX* x, std::ptrdiff_t incx, TY* y, std::ptrdiff_t incy)
{
  if (!x || !y || n == 0) return;
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) std::swap(x[ix], y[iy]);
}

// 5) dot products -----------------------------------------------------------

// dotu: unconjugated dot product (matches default behavior in this library)
template <typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline auto dotu(const X& x, const Y& y)
{
  using TX = element_t<X>;
  using TY = element_t<Y>;
  using R = std::common_type_t<TX, TY>;
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("dotu: size mismatch");
  if (n == 0) return R{};

  if constexpr (std::is_same_v<TX, TY>) {
    return std::transform_reduce(std::execution::par_unseq,
                                 x.begin(), x.end(), y.begin(), R{}, std::plus<>{},
                                 [](const TX& a, const TY& b) { return static_cast<R>(a * b); });
  } else {
    R acc{};
    for (std::size_t i = 0; i < n; ++i) acc += static_cast<R>(x[i]) * static_cast<R>(y[i]);
    return acc;
  }
}

// dotc: conjugated dot product (conj(x) * y for complex types; same as dotu for real)
template <typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline auto dotc(const X& x, const Y& y)
{
  using TX = element_t<X>;
  using TY = element_t<Y>;
  using R = std::common_type_t<TX, TY>;
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("dotc: size mismatch");
  if (n == 0) return R{};

  R acc{};
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<R>(conj_if_complex(static_cast<R>(x[i])) * static_cast<R>(y[i]));
  }
  return acc;
}

// Default dot aligns with unconjugated behavior used elsewhere in the library
template <typename X, typename Y>
  requires (VectorLike<X> && VectorLike<Y>)
inline auto dot(const X& x, const Y& y) { return dotu(x, y); }

template <typename TX, typename TY>
inline auto dotu(std::size_t n, const TX* x, std::ptrdiff_t incx, const TY* y, std::ptrdiff_t incy)
{
  using R = std::common_type_t<TX, TY>;
  if (!x || !y || n == 0) return R{};
  R acc{};
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) acc += static_cast<R>(x[ix]) * static_cast<R>(y[iy]);
  return acc;
}

template <typename TX, typename TY>
inline auto dotc(std::size_t n, const TX* x, std::ptrdiff_t incx, const TY* y, std::ptrdiff_t incy)
{
  using R = std::common_type_t<TX, TY>;
  if (!x || !y || n == 0) return R{};
  R acc{};
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) {
    acc += static_cast<R>(conj_if_complex(static_cast<R>(x[ix])) * static_cast<R>(y[iy]));
  }
  return acc;
}

// 6) nrm2: Euclidean (2-) norm ---------------------------------------------

template <typename X>
  requires VectorLike<X>
inline auto nrm2(const X& x)
{
  using T = element_t<X>;
  using R = typename numeric_traits<T>::scalar_type;
  R scale{0};
  R ssq{1};
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) {
    R a = is_complex_number_v<T> ? static_cast<R>(std::abs(x[i])) : static_cast<R>(std::abs(x[i]));
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

template <typename TX>
inline auto nrm2(std::size_t n, const TX* x, std::ptrdiff_t incx)
{
  using R = typename numeric_traits<TX>::scalar_type;
  if (!x || n == 0) return R{0};
  R scale{0};
  R ssq{1};
  std::ptrdiff_t ix = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx) {
    R a = is_complex_number_v<TX> ? static_cast<R>(std::abs(x[ix])) : static_cast<R>(std::abs(x[ix]));
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

// 7) asum: sum of absolute values (BLAS semantics for complex) --------------

template <typename X>
  requires VectorLike<X>
inline auto asum(const X& x)
{
  using T = element_t<X>;
  using R = typename numeric_traits<T>::scalar_type;
  R acc{};
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) acc += static_cast<R>(abs_blas1(x[i]));
  return acc;
}

template <typename TX>
inline auto asum(std::size_t n, const TX* x, std::ptrdiff_t incx)
{
  using R = typename numeric_traits<TX>::scalar_type;
  if (!x || n == 0) return R{0};
  R acc{};
  std::ptrdiff_t ix = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx) acc += static_cast<R>(abs_blas1(x[ix]));
  return acc;
}

// 8) iamax / iamin: index of max/min absolute value -------------------------

template <typename X>
  requires VectorLike<X>
inline std::size_t iamax(const X& x)
{
  // using T = element_t<X>; // unused
  const std::size_t n = x.size();
  if (n == 0) return 0;
  std::size_t idx = 0;
  auto best = abs_blas1(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    auto v = abs_blas1(x[i]);
    if (v > best) { best = v; idx = i; }
  }
  return idx; // 0-based
}

template <typename X>
  requires VectorLike<X>
inline std::size_t iamin(const X& x)
{
  // using T = element_t<X>; // unused
  const std::size_t n = x.size();
  if (n == 0) return 0;
  std::size_t idx = 0;
  auto best = abs_blas1(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    auto v = abs_blas1(x[i]);
    if (v < best) { best = v; idx = i; }
  }
  return idx; // 0-based
}

template <typename TX>
inline std::size_t iamax(std::size_t n, const TX* x, std::ptrdiff_t incx)
{
  if (!x || n == 0) return 0;
  std::size_t idx = 0;
  std::ptrdiff_t ix = 0;
  auto best = abs_blas1(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    ix += incx;
    auto v = abs_blas1(x[ix]);
    if (v > best) { best = v; idx = i; }
  }
  return idx;
}

template <typename TX>
inline std::size_t iamin(std::size_t n, const TX* x, std::ptrdiff_t incx)
{
  if (!x || n == 0) return 0;
  std::size_t idx = 0;
  std::ptrdiff_t ix = 0;
  auto best = abs_blas1(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    ix += incx;
    auto v = abs_blas1(x[ix]);
    if (v < best) { best = v; idx = i; }
  }
  return idx;
}

// 9) rot: apply Givens rotation to two vectors ------------------------------

// Given c and s, apply: x' = c*x + s*y; y' = c*y - s*x
template <typename X, typename Y, typename C, typename S>
  requires (VectorLike<X> && VectorLike<Y>)
inline void rot(X& x, Y& y, const C& c, const S& s)
{
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("rot: size mismatch");
  for (std::size_t i = 0; i < n; ++i) {
    auto xi = x[i];
    auto yi = y[i];
    x[i] = static_cast<element_t<X>>(c * xi + s * yi);
    y[i] = static_cast<element_t<Y>>(c * yi - s * xi);
  }
}

template <typename TX, typename TY, typename C, typename S>
inline void rot(std::size_t n, TX* x, std::ptrdiff_t incx, TY* y, std::ptrdiff_t incy,
                const C& c, const S& s)
{
  if (!x || !y || n == 0) return;
  std::ptrdiff_t ix = 0, iy = 0;
  for (std::size_t i = 0; i < n; ++i, ix += incx, iy += incy) {
    auto xi = x[ix];
    auto yi = y[iy];
    x[ix] = static_cast<TX>(c * xi + s * yi);
    y[iy] = static_cast<TY>(c * yi - s * xi);
  }
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL1_H

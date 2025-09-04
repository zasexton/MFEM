#pragma once

#ifndef NUMERIC_DUAL_MATH_H
#define NUMERIC_DUAL_MATH_H

#include "dual_base.h"
#include <cmath>

namespace fem::numeric::autodiff {

// ============================================================================
// Extended trigonometric functions
// ============================================================================

// Tangent: d/dx[tan(x)] = sec²(x)
template<typename T, std::size_t N>
DualBase<T, N> tan(const DualBase<T, N>& x) {
    using std::tan;
    using std::cos;
    const T val = tan(x.value());
    const T cos_x = cos(x.value());
    const T dval = T(1) / (cos_x * cos_x);

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Arcsine: d/dx[asin(x)] = 1/√(1-x²)
template<typename T, std::size_t N>
DualBase<T, N> asin(const DualBase<T, N>& x) {
    using std::asin;
    using std::sqrt;
    const T val = asin(x.value());
    const T dval = T(1) / sqrt(T(1) - x.value() * x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Arccosine: d/dx[acos(x)] = -1/√(1-x²)
template<typename T, std::size_t N>
DualBase<T, N> acos(const DualBase<T, N>& x) {
    using std::acos;
    using std::sqrt;
    const T val = acos(x.value());
    const T dval = -T(1) / sqrt(T(1) - x.value() * x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Arctangent: d/dx[atan(x)] = 1/(1+x²)
template<typename T, std::size_t N>
DualBase<T, N> atan(const DualBase<T, N>& x) {
    using std::atan;
    const T val = atan(x.value());
    const T dval = T(1) / (T(1) + x.value() * x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Two-argument arctangent
template<typename T, std::size_t N>
DualBase<T, N> atan2(const DualBase<T, N>& y, const DualBase<T, N>& x) {
    using std::atan2;
    const T val = atan2(y.value(), x.value());
    const T denom = x.value() * x.value() + y.value() * y.value();

    DualBase<T, N> result(val);
    if (denom != T(0)) {
        for (std::size_t i = 0; i < N; ++i) {
            result.derivative(i) = (x.value() * y.derivative(i) -
                                    y.value() * x.derivative(i)) / denom;
        }
    }
    return result;
}

// ============================================================================
// Hyperbolic functions
// ============================================================================

// Hyperbolic sine: d/dx[sinh(x)] = cosh(x)
template<typename T, std::size_t N>
DualBase<T, N> sinh(const DualBase<T, N>& x) {
    using std::sinh;
    using std::cosh;
    const T val = sinh(x.value());
    const T dval = cosh(x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Hyperbolic cosine: d/dx[cosh(x)] = sinh(x)
template<typename T, std::size_t N>
DualBase<T, N> cosh(const DualBase<T, N>& x) {
    using std::cosh;
    using std::sinh;
    const T val = cosh(x.value());
    const T dval = sinh(x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Hyperbolic tangent: d/dx[tanh(x)] = sech²(x)
template<typename T, std::size_t N>
DualBase<T, N> tanh(const DualBase<T, N>& x) {
    using std::tanh;
    using std::cosh;
    const T val = tanh(x.value());
    const T cosh_x = cosh(x.value());
    const T dval = T(1) / (cosh_x * cosh_x);

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// ============================================================================
// Additional logarithmic functions
// ============================================================================

// Base-10 logarithm
template<typename T, std::size_t N>
DualBase<T, N> log10(const DualBase<T, N>& x) {
    using std::log10;
    const T val = log10(x.value());
    const T dval = T(1) / (x.value() * std::log(T(10)));

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// Base-2 logarithm
template<typename T, std::size_t N>
DualBase<T, N> log2(const DualBase<T, N>& x) {
    using std::log2;
    const T val = log2(x.value());
    const T dval = T(1) / (x.value() * std::log(T(2)));

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// ============================================================================
// Extended power functions
// ============================================================================

// Power function for dual exponent: x^y = exp(y * ln(x))
template<typename T, std::size_t N>
DualBase<T, N> pow(const DualBase<T, N>& x, const DualBase<T, N>& y) {
    using std::pow;
    using std::log;
    const T val = pow(x.value(), y.value());

    DualBase<T, N> result(val);
    if (x.value() > T(0)) {
        const T log_x = log(x.value());
        for (std::size_t i = 0; i < N; ++i) {
            result.derivative(i) = val * (y.derivative(i) * log_x +
                                          y.value() * x.derivative(i) / x.value());
        }
    }
    return result;
}

// Cube root: d/dx[∛x] = 1/(3x^(2/3))
template<typename T, std::size_t N>
DualBase<T, N> cbrt(const DualBase<T, N>& x) {
    using std::cbrt;
    using std::pow;
    const T val = cbrt(x.value());
    const T dval = T(1) / (T(3) * pow(x.value(), T(2)/T(3)));

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

// ============================================================================
// Special functions with careful derivative handling
// ============================================================================

/**
 * @brief Absolute value function with subgradient at zero
 *
 * WARNING: Non-differentiable at x=0. Uses subgradient convention.
 * The derivative is undefined at x=0, but we return 0 for stability.
 */
template<typename T, std::size_t N>
DualBase<T, N> abs(const DualBase<T, N>& x) {
    using std::abs;
    const T val = abs(x.value());

    DualBase<T, N> result(val);
    if (x.value() != T(0)) {
        const T dval = x.value() >= T(0) ? T(1) : T(-1);
        for (std::size_t i = 0; i < N; ++i) {
            result.derivative(i) = dval * x.derivative(i);
        }
    }
    // else: derivatives remain zero (subgradient convention)
    return result;
}

/**
 * @brief Sign function
 *
 * WARNING: Derivative is zero everywhere except at x=0 where it's undefined.
 */
template<typename T, std::size_t N>
DualBase<T, N> sign(const DualBase<T, N>& x) {
    const T val = (x.value() > T(0)) ? T(1) :
                  (x.value() < T(0)) ? T(-1) : T(0);

    // Derivative is zero everywhere (except at discontinuity)
    return DualBase<T, N>(val);
}

/**
 * @brief Maximum of two dual numbers
 *
 * WARNING: Non-differentiable when a.value() == b.value().
 * Uses convention of choosing the first argument's derivative at equality.
 */
template<typename T, std::size_t N>
DualBase<T, N> max(const DualBase<T, N>& a, const DualBase<T, N>& b) {
    return (a.value() >= b.value()) ? a : b;
}

/**
 * @brief Minimum of two dual numbers
 *
 * WARNING: Non-differentiable when a.value() == b.value().
 * Uses convention of choosing the first argument's derivative at equality.
 */
template<typename T, std::size_t N>
DualBase<T, N> min(const DualBase<T, N>& a, const DualBase<T, N>& b) {
    return (a.value() <= b.value()) ? a : b;
}

/**
 * @brief Smoothed maximum using softmax approximation
 *
 * Differentiable approximation: max(a,b) ≈ log(exp(k*a) + exp(k*b))/k
 * As k → ∞, this approaches the true maximum.
 *
 * @param k Smoothing parameter (larger = closer to true max, less smooth)
 */
template<typename T, std::size_t N>
DualBase<T, N> smooth_max(const DualBase<T, N>& a, const DualBase<T, N>& b,
                          const T& k = T(10)) {
    using std::exp;
    using std::log;

    const T exp_ka = exp(k * a.value());
    const T exp_kb = exp(k * b.value());
    const T sum_exp = exp_ka + exp_kb;
    const T val = log(sum_exp) / k;

    DualBase<T, N> result(val);
    const T wa = exp_ka / sum_exp;  // Weight for a
    const T wb = exp_kb / sum_exp;  // Weight for b

    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = wa * a.derivative(i) + wb * b.derivative(i);
    }
    return result;
}

/**
 * @brief Smoothed minimum using softmin approximation
 */
template<typename T, std::size_t N>
DualBase<T, N> smooth_min(const DualBase<T, N>& a, const DualBase<T, N>& b,
                          const T& k = T(10)) {
    return -smooth_max(-a, -b, k);
}

// ============================================================================
// Utility functions
// ============================================================================

// Clamp value between min and max
template<typename T, std::size_t N>
DualBase<T, N> clamp(const DualBase<T, N>& x, const T& min_val, const T& max_val) {
    if (x.value() < min_val) {
        return DualBase<T, N>(min_val);  // Derivative is zero at boundary
    } else if (x.value() > max_val) {
        return DualBase<T, N>(max_val);  // Derivative is zero at boundary
    } else {
        return x;  // Pass through with derivatives
    }
}

} // namespace fem::numeric::autodiff

#endif //NUMERIC_DUAL_MATH_H
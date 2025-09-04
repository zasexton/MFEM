#pragma once

#ifndef NUMERIC_DUAL_COMPARISON_H
#define NUMERIC_DUAL_COMPARISON_H

#include "dual_base.h"

namespace fem::numeric::autodiff {

/**
 * @brief Comparison operators for dual numbers
 *
 * WARNING: These operators compare ONLY the primal values, ignoring derivatives.
 * This is mathematically questionable but sometimes necessary for algorithms.
 *
 * For derivative-aware comparisons, explicitly compare the values:
 *   if (a.value() < b.value()) { ... }
 *
 * This header is separated from the base to make the design decision explicit.
 */

// Less than operators
template<typename T, std::size_t N>
bool operator<(const DualBase<T, N>& lhs, const DualBase<T, N>& rhs) {
    return lhs.value() < rhs.value();
}

template<typename T, std::size_t N>
bool operator<(const DualBase<T, N>& lhs, const T& rhs) {
    return lhs.value() < rhs;
}

template<typename T, std::size_t N>
bool operator<(const T& lhs, const DualBase<T, N>& rhs) {
    return lhs < rhs.value();
}

// Less than or equal operators
template<typename T, std::size_t N>
bool operator<=(const DualBase<T, N>& lhs, const DualBase<T, N>& rhs) {
    return lhs.value() <= rhs.value();
}

template<typename T, std::size_t N>
bool operator<=(const DualBase<T, N>& lhs, const T& rhs) {
    return lhs.value() <= rhs;
}

template<typename T, std::size_t N>
bool operator<=(const T& lhs, const DualBase<T, N>& rhs) {
    return lhs <= rhs.value();
}

// Greater than operators
template<typename T, std::size_t N>
bool operator>(const DualBase<T, N>& lhs, const DualBase<T, N>& rhs) {
    return lhs.value() > rhs.value();
}

template<typename T, std::size_t N>
bool operator>(const DualBase<T, N>& lhs, const T& rhs) {
    return lhs.value() > rhs;
}

template<typename T, std::size_t N>
bool operator>(const T& lhs, const DualBase<T, N>& rhs) {
    return lhs > rhs.value();
}

// Greater than or equal operators
template<typename T, std::size_t N>
bool operator>=(const DualBase<T, N>& lhs, const DualBase<T, N>& rhs) {
    return lhs.value() >= rhs.value();
}

template<typename T, std::size_t N>
bool operator>=(const DualBase<T, N>& lhs, const T& rhs) {
    return lhs.value() >= rhs;
}

template<typename T, std::size_t N>
bool operator>=(const T& lhs, const DualBase<T, N>& rhs) {
    return lhs >= rhs.value();
}

// ============================================================================
// Comparison utilities
// ============================================================================

/**
 * @brief Compare two dual numbers including derivatives
 *
 * Returns true if both value and all derivatives are equal within tolerance
 */
template<typename T, std::size_t N>
bool approx_equal(const DualBase<T, N>& a, const DualBase<T, N>& b,
                  const T& tol = std::numeric_limits<T>::epsilon() * T(100)) {
    using std::abs;

    // Check value
    if (abs(a.value() - b.value()) > tol) {
        return false;
    }

    // Check derivatives
    for (std::size_t i = 0; i < N; ++i) {
        if (abs(a.derivative(i) - b.derivative(i)) > tol) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Lexicographic comparison of dual numbers
 *
 * First compares values, then derivatives if values are equal.
 * This provides a total ordering for use in sorted containers.
 */
template<typename T, std::size_t N>
struct DualLexicographicLess {
    bool operator()(const DualBase<T, N>& a, const DualBase<T, N>& b) const {
        // First compare values
        if (a.value() < b.value()) return true;
        if (b.value() < a.value()) return false;

        // Values are equal, compare derivatives
        for (std::size_t i = 0; i < N; ++i) {
            if (a.derivative(i) < b.derivative(i)) return true;
            if (b.derivative(i) < a.derivative(i)) return false;
        }

        return false;  // Completely equal
    }
};

/**
 * @brief Check if a dual number has finite value and derivatives
 */
template<typename T, std::size_t N>
bool is_finite(const DualBase<T, N>& x) {
    using std::isfinite;

    if (!isfinite(x.value())) {
        return false;
    }

    for (std::size_t i = 0; i < N; ++i) {
        if (!isfinite(x.derivative(i))) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Check if a dual number has any NaN components
 */
template<typename T, std::size_t N>
bool has_nan(const DualBase<T, N>& x) {
    using std::isnan;

    if (isnan(x.value())) {
        return true;
    }

    for (std::size_t i = 0; i < N; ++i) {
        if (isnan(x.derivative(i))) {
            return true;
        }
    }

    return false;
}

} // namespace fem::numeric::autodiff

#endif //NUMERIC_DUAL_COMPARISON_H
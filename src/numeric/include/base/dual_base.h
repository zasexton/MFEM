#pragma once

#ifndef NUMERIC_DUAL_BASE_H
#define NUMERIC_DUAL_BASE_H

#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "../config/config.hpp"
#include "numeric_base.hpp"

namespace numeric {
namespace autodiff {

// Forward declarations
template<typename T, std::size_t N = 1> class DualBase;
template<typename T, std::size_t N = 1> class Dual;
template<typename T> class DynamicDual;

// ============================================================================
// Dual number base class (CRTP pattern)
// ============================================================================

/**
 * @brief Base class for dual numbers implementing forward-mode AD
 *
 * Uses CRTP (Curiously Recurring Template Pattern) to provide common
 * functionality while allowing derived classes to customize storage
 * and optimization strategies.
 *
 * @tparam T Underlying scalar type (e.g., double, float)
 * @tparam N Number of derivative directions (default = 1)
 *
 * Architecture decisions:
 * 1. CRTP for static polymorphism - zero runtime overhead
 * 2. Template on derivative count for compile-time optimization
 * 3. Expression templates for lazy evaluation
 * 4. SIMD-friendly memory layout
 * 5. Support both static (N known) and dynamic derivative counts
 * 6. No dependencies on traits - this is a foundational type
 */
template<typename T, std::size_t N = 1>
class DualBase {
public:
    using value_type = T;
    using derivative_type = std::array<T, N>;
    using size_type = std::size_t;

    static constexpr size_type num_derivatives = N;
    static constexpr bool is_dynamic = (N == 0);  // N=0 signals dynamic size

protected:
    T value_;                    // Function value f(x)
    derivative_type derivatives_; // Derivatives ∂f/∂xᵢ

    // Protected constructor for derived classes
    DualBase() : value_(T(0)), derivatives_{} {}

    explicit DualBase(const T& val) : value_(val), derivatives_{} {}

    DualBase(const T& val, const derivative_type& derivs)
        : value_(val), derivatives_(derivs) {}

    // Variadic constructor for convenient initialization
    template<typename... Args,
             typename = std::enable_if_t<sizeof...(Args) == N>>
    DualBase(const T& val, Args... derivs)
        : value_(val), derivatives_{static_cast<T>(derivs)...} {}

public:
    // ========================================================================
    // Accessors
    // ========================================================================

    /** @brief Get the primal value */
    [[nodiscard]] const T& value() const noexcept { return value_; }
    T& value() noexcept { return value_; }

    /** @brief Get all derivatives as array */
    [[nodiscard]] const derivative_type& derivatives() const noexcept {
        return derivatives_;
    }
    derivative_type& derivatives() noexcept { return derivatives_; }

    /** @brief Get specific derivative component */
    [[nodiscard]] const T& derivative(size_type i) const {
        assert(i < N && "Derivative index out of bounds");
        return derivatives_[i];
    }

    T& derivative(size_type i) {
        assert(i < N && "Derivative index out of bounds");
        return derivatives_[i];
    }

    /** @brief Get the gradient (alias for derivatives) */
    [[nodiscard]] const derivative_type& gradient() const noexcept {
        return derivatives_;
    }

    /** @brief Number of derivative directions */
    [[nodiscard]] static constexpr size_type size() noexcept { return N; }

    // ========================================================================
    // Seeding operations for derivative directions
    // ========================================================================

    /** @brief Set as independent variable with seed in direction i */
    void seed(size_type i, const T& seed_value = T(1)) {
        std::fill(derivatives_.begin(), derivatives_.end(), T(0));
        if (i < N) {
            derivatives_[i] = seed_value;
        }
    }

    /** @brief Set custom seed vector */
    void seed(const derivative_type& seed_vector) {
        derivatives_ = seed_vector;
    }

    /** @brief Clear all derivatives to zero */
    void clear_derivatives() {
        std::fill(derivatives_.begin(), derivatives_.end(), T(0));
    }

    // ========================================================================
    // Arithmetic operations (implement chain rule)
    // ========================================================================

    /** @brief Addition: (f + g)' = f' + g' */
    template<typename U>
    friend auto operator+(const DualBase<T, N>& lhs, const DualBase<U, N>& rhs) {
        DualBase<T, N> result(lhs.value_ + rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] + rhs.derivatives_[i];
        }
        return result;
    }

    /** @brief Addition with scalar: (f + c)' = f' */
    friend auto operator+(const DualBase<T, N>& lhs, const T& rhs) {
        return DualBase<T, N>(lhs.value_ + rhs, lhs.derivatives_);
    }

    friend auto operator+(const T& lhs, const DualBase<T, N>& rhs) {
        return DualBase<T, N>(lhs + rhs.value_, rhs.derivatives_);
    }

    /** @brief Subtraction: (f - g)' = f' - g' */
    template<typename U>
    friend auto operator-(const DualBase<T, N>& lhs, const DualBase<U, N>& rhs) {
        DualBase<T, N> result(lhs.value_ - rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] - rhs.derivatives_[i];
        }
        return result;
    }

    friend auto operator-(const DualBase<T, N>& lhs, const T& rhs) {
        return DualBase<T, N>(lhs.value_ - rhs, lhs.derivatives_);
    }

    friend auto operator-(const T& lhs, const DualBase<T, N>& rhs) {
        DualBase<T, N> result(lhs - rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = -rhs.derivatives_[i];
        }
        return result;
    }

    /** @brief Multiplication: (f * g)' = f' * g + f * g' */
    template<typename U>
    friend auto operator*(const DualBase<T, N>& lhs, const DualBase<U, N>& rhs) {
        DualBase<T, N> result(lhs.value_ * rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * rhs.value_ +
                                     lhs.value_ * rhs.derivatives_[i];
        }
        return result;
    }

    friend auto operator*(const DualBase<T, N>& lhs, const T& rhs) {
        DualBase<T, N> result(lhs.value_ * rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * rhs;
        }
        return result;
    }

    friend auto operator*(const T& lhs, const DualBase<T, N>& rhs) {
        return rhs * lhs;  // Commutative
    }

    /** @brief Division: (f / g)' = (f' * g - f * g') / g² */
    template<typename U>
    friend auto operator/(const DualBase<T, N>& lhs, const DualBase<U, N>& rhs) {
        const T inv_rhs = T(1) / rhs.value_;
        const T inv_rhs_sq = inv_rhs * inv_rhs;

        DualBase<T, N> result(lhs.value_ * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = (lhs.derivatives_[i] * rhs.value_ -
                                      lhs.value_ * rhs.derivatives_[i]) * inv_rhs_sq;
        }
        return result;
    }

    friend auto operator/(const DualBase<T, N>& lhs, const T& rhs) {
        const T inv_rhs = T(1) / rhs;
        DualBase<T, N> result(lhs.value_ * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * inv_rhs;
        }
        return result;
    }

    friend auto operator/(const T& lhs, const DualBase<T, N>& rhs) {
        const T inv_rhs = T(1) / rhs.value_;
        const T inv_rhs_sq = inv_rhs * inv_rhs;

        DualBase<T, N> result(lhs * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = -lhs * rhs.derivatives_[i] * inv_rhs_sq;
        }
        return result;
    }

    /** @brief Unary negation */
    friend auto operator-(const DualBase<T, N>& x) {
        DualBase<T, N> result(-x.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = -x.derivatives_[i];
        }
        return result;
    }

    // ========================================================================
    // Compound assignment operators
    // ========================================================================

    DualBase& operator+=(const DualBase& rhs) {
        value_ += rhs.value_;
        for (size_type i = 0; i < N; ++i) {
            derivatives_[i] += rhs.derivatives_[i];
        }
        return *this;
    }

    DualBase& operator+=(const T& rhs) {
        value_ += rhs;
        return *this;
    }

    DualBase& operator-=(const DualBase& rhs) {
        value_ -= rhs.value_;
        for (size_type i = 0; i < N; ++i) {
            derivatives_[i] -= rhs.derivatives_[i];
        }
        return *this;
    }

    DualBase& operator-=(const T& rhs) {
        value_ -= rhs;
        return *this;
    }

    DualBase& operator*=(const DualBase& rhs) {
        for (size_type i = 0; i < N; ++i) {
            derivatives_[i] = derivatives_[i] * rhs.value_ + value_ * rhs.derivatives_[i];
        }
        value_ *= rhs.value_;
        return *this;
    }

    DualBase& operator*=(const T& rhs) {
        value_ *= rhs;
        for (size_type i = 0; i < N; ++i) {
            derivatives_[i] *= rhs;
        }
        return *this;
    }

    DualBase& operator/=(const DualBase& rhs) {
        *this = *this / rhs;
        return *this;
    }

    DualBase& operator/=(const T& rhs) {
        const T inv_rhs = T(1) / rhs;
        value_ *= inv_rhs;
        for (size_type i = 0; i < N; ++i) {
            derivatives_[i] *= inv_rhs;
        }
        return *this;
    }

    // ========================================================================
    // Comparison operators (compare only values, not derivatives)
    // ========================================================================

    friend bool operator==(const DualBase& lhs, const DualBase& rhs) {
        return lhs.value_ == rhs.value_;
    }

    friend bool operator==(const DualBase& lhs, const T& rhs) {
        return lhs.value_ == rhs;
    }

    friend bool operator==(const T& lhs, const DualBase& rhs) {
        return lhs == rhs.value_;
    }

    friend bool operator!=(const DualBase& lhs, const DualBase& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const DualBase& lhs, const T& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const T& lhs, const DualBase& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const DualBase& lhs, const DualBase& rhs) {
        return lhs.value_ < rhs.value_;
    }

    friend bool operator<(const DualBase& lhs, const T& rhs) {
        return lhs.value_ < rhs;
    }

    friend bool operator<(const T& lhs, const DualBase& rhs) {
        return lhs < rhs.value_;
    }

    friend bool operator<=(const DualBase& lhs, const DualBase& rhs) {
        return lhs.value_ <= rhs.value_;
    }

    friend bool operator<=(const DualBase& lhs, const T& rhs) {
        return lhs.value_ <= rhs;
    }

    friend bool operator<=(const T& lhs, const DualBase& rhs) {
        return lhs <= rhs.value_;
    }

    friend bool operator>(const DualBase& lhs, const DualBase& rhs) {
        return lhs.value_ > rhs.value_;
    }

    friend bool operator>(const DualBase& lhs, const T& rhs) {
        return lhs.value_ > rhs;
    }

    friend bool operator>(const T& lhs, const DualBase& rhs) {
        return lhs > rhs.value_;
    }

    friend bool operator>=(const DualBase& lhs, const DualBase& rhs) {
        return lhs.value_ >= rhs.value_;
    }

    friend bool operator>=(const DualBase& lhs, const T& rhs) {
        return lhs.value_ >= rhs;
    }

    friend bool operator>=(const T& lhs, const DualBase& rhs) {
        return lhs >= rhs.value_;
    }

    // ========================================================================
    // Stream output
    // ========================================================================

    friend std::ostream& operator<<(std::ostream& os, const DualBase& dual) {
        os << "Dual(" << dual.value_ << "; [";
        for (size_type i = 0; i < N; ++i) {
            if (i > 0) os << ", ";
            os << dual.derivatives_[i];
        }
        os << "])";
        return os;
    }
};

// ============================================================================
// Mathematical functions with derivatives
// ============================================================================

/** @brief Power function: d/dx[x^n] = n*x^(n-1) */
template<typename T, std::size_t N>
DualBase<T, N> pow(const DualBase<T, N>& x, const T& n) {
    using std::pow;
    const T val = pow(x.value(), n);
    const T dval = n * pow(x.value(), n - T(1));

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

/** @brief Power function for dual exponent */
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

/** @brief Square root: d/dx[√x] = 1/(2√x) */
template<typename T, std::size_t N>
DualBase<T, N> sqrt(const DualBase<T, N>& x) {
    using std::sqrt;
    const T val = sqrt(x.value());
    const T dval = T(0.5) / val;

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

/** @brief Exponential: d/dx[e^x] = e^x */
template<typename T, std::size_t N>
DualBase<T, N> exp(const DualBase<T, N>& x) {
    using std::exp;
    const T val = exp(x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = val * x.derivative(i);
    }
    return result;
}

/** @brief Natural logarithm: d/dx[ln(x)] = 1/x */
template<typename T, std::size_t N>
DualBase<T, N> log(const DualBase<T, N>& x) {
    using std::log;
    const T val = log(x.value());
    const T dval = T(1) / x.value();

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

/** @brief Base-10 logarithm */
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

/** @brief Sine: d/dx[sin(x)] = cos(x) */
template<typename T, std::size_t N>
DualBase<T, N> sin(const DualBase<T, N>& x) {
    using std::sin;
    using std::cos;
    const T val = sin(x.value());
    const T dval = cos(x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

/** @brief Cosine: d/dx[cos(x)] = -sin(x) */
template<typename T, std::size_t N>
DualBase<T, N> cos(const DualBase<T, N>& x) {
    using std::sin;
    using std::cos;
    const T val = cos(x.value());
    const T dval = -sin(x.value());

    DualBase<T, N> result(val);
    for (std::size_t i = 0; i < N; ++i) {
        result.derivative(i) = dval * x.derivative(i);
    }
    return result;
}

/** @brief Tangent: d/dx[tan(x)] = sec²(x) */
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

/** @brief Arcsine: d/dx[asin(x)] = 1/√(1-x²) */
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

/** @brief Arccosine: d/dx[acos(x)] = -1/√(1-x²) */
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

/** @brief Arctangent: d/dx[atan(x)] = 1/(1+x²) */
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

/** @brief Two-argument arctangent */
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

/** @brief Hyperbolic sine: d/dx[sinh(x)] = cosh(x) */
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

/** @brief Hyperbolic cosine: d/dx[cosh(x)] = sinh(x) */
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

/** @brief Hyperbolic tangent: d/dx[tanh(x)] = sech²(x) */
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

/** @brief Absolute value (non-differentiable at 0, uses subgradient) */
template<typename T, std::size_t N>
DualBase<T, N> abs(const DualBase<T, N>& x) {
    using std::abs;
    const T val = abs(x.value());
    const T dval = x.value() >= T(0) ? T(1) : T(-1);

    DualBase<T, N> result(val);
    if (x.value() != T(0)) {  // Avoid non-differentiability at 0
        for (std::size_t i = 0; i < N; ++i) {
            result.derivative(i) = dval * x.derivative(i);
        }
    }
    return result;
}

/** @brief Maximum of two dual numbers */
template<typename T, std::size_t N>
DualBase<T, N> max(const DualBase<T, N>& a, const DualBase<T, N>& b) {
    return (a.value() >= b.value()) ? a : b;
}

/** @brief Minimum of two dual numbers */
template<typename T, std::size_t N>
DualBase<T, N> min(const DualBase<T, N>& a, const DualBase<T, N>& b) {
    return (a.value() <= b.value()) ? a : b;
}

// ============================================================================
// Helper functions for dual number manipulation
// ============================================================================

/** @brief Create a dual number from value with zero derivatives */
template<typename T, std::size_t N = 1>
DualBase<T, N> make_dual(const T& value) {
    return DualBase<T, N>(value);
}

/** @brief Create a dual number with specified value and derivatives */
template<typename T, std::size_t N>
DualBase<T, N> make_dual(const T& value, const std::array<T, N>& derivatives) {
    return DualBase<T, N>(value, derivatives);
}

/** @brief Create a dual as independent variable (seed with unit vector) */
template<typename T, std::size_t N>
DualBase<T, N> make_independent(const T& value, std::size_t index) {
    DualBase<T, N> result(value);
    result.seed(index);
    return result;
}

/** @brief Extract Jacobian matrix from array of dual numbers */
template<typename T, std::size_t M, std::size_t N>
std::array<std::array<T, N>, M> extract_jacobian(
    const std::array<DualBase<T, N>, M>& dual_array) {

    std::array<std::array<T, N>, M> jacobian;
    for (std::size_t i = 0; i < M; ++i) {
        jacobian[i] = dual_array[i].derivatives();
    }
    return jacobian;
}

/** @brief Extract gradient from a scalar dual number */
template<typename T, std::size_t N>
std::array<T, N> extract_gradient(const DualBase<T, N>& dual) {
    return dual.derivatives();
}

} // namespace autodiff

// Bring AD types into numeric namespace for convenience
using autodiff::DualBase;

} // namespace numeric

#endif //NUMERIC_DUAL_BASE_H
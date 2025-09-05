#pragma once

#ifndef NUMERIC_DUAL_BASE_H
#define NUMERIC_DUAL_BASE_H

#include <cmath>
#include <type_traits>
#include <array>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "numeric_base.h"

namespace fem::numeric::autodiff {

// Forward declarations
template<typename T, std::size_t N> class DualBase;
template<typename T, std::size_t N> class Dual;
template<typename T> class DynamicDual;

// ============================================================================
// Dual number base class (CRTP pattern)
// ============================================================================

/**
 * @brief Base class for dual numbers implementing forward-mode AD
 *
 * This class provides the core dual number functionality for automatic
 * differentiation. It tracks both function values and their derivatives
 * using the chain rule.
 *
 * @tparam T Underlying scalar type (e.g., double, float)
 * @tparam N Number of derivative directions (default = 1)
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

public:
    // ========================================================================
    // Constructors - explicitly defaulted for clarity
    // ========================================================================

    DualBase() : value_(T(0)), derivatives_{} {}

    explicit DualBase(const T& val) : value_(val), derivatives_{} {}

    DualBase(const T& val, const derivative_type& derivs)
        : value_(val), derivatives_(derivs) {}

    // Variadic constructor for convenient initialization
    template<typename... Args,
             typename = std::enable_if_t<sizeof...(Args) == N>>
    DualBase(const T& val, Args... derivs)
        : value_(val), derivatives_{static_cast<T>(derivs)...} {}

    // Explicitly defaulted special members
    DualBase(const DualBase&) = default;
    DualBase(DualBase&&) = default;
    DualBase& operator=(const DualBase&) = default;
    DualBase& operator=(DualBase&&) = default;
    ~DualBase() = default;

    // ========================================================================
    // Core accessors
    // ========================================================================

    [[nodiscard]] const T& value() const noexcept { return value_; }
    T& value() noexcept { return value_; }

    [[nodiscard]] const derivative_type& derivatives() const noexcept {
        return derivatives_;
    }
    derivative_type& derivatives() noexcept { return derivatives_; }

    [[nodiscard]] const T& derivative(size_type i) const {
        assert(i < N && "Derivative index out of bounds");
        return derivatives_[i];
    }

    T& derivative(size_type i) {
        assert(i < N && "Derivative index out of bounds");
        return derivatives_[i];
    }

    [[nodiscard]] const derivative_type& gradient() const noexcept {
        return derivatives_;
    }

    [[nodiscard]] static constexpr size_type size() noexcept { return N; }

    // ========================================================================
    // Seeding operations for derivative directions
    // ========================================================================

    void seed(size_type i, const T& seed_value = T(1)) {
        std::fill(derivatives_.begin(), derivatives_.end(), T(0));
        if (i < N) {
            derivatives_[i] = seed_value;
        }
    }

    void seed(const derivative_type& seed_vector) {
        derivatives_ = seed_vector;
    }

    void clear_derivatives() {
        std::fill(derivatives_.begin(), derivatives_.end(), T(0));
    }

    // ========================================================================
    // Core arithmetic operations implementing chain rule
    // ========================================================================

    // Addition: (f + g)' = f' + g'
    friend DualBase operator+(const DualBase& lhs, const DualBase& rhs) {
        DualBase result(lhs.value_ + rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] + rhs.derivatives_[i];
        }
        return result;
    }

    friend DualBase operator+(const DualBase& lhs, const T& rhs) {
        return DualBase(lhs.value_ + rhs, lhs.derivatives_);
    }

    friend DualBase operator+(const T& lhs, const DualBase& rhs) {
        return DualBase(lhs + rhs.value_, rhs.derivatives_);
    }

    // Subtraction: (f - g)' = f' - g'
    friend DualBase operator-(const DualBase& lhs, const DualBase& rhs) {
        DualBase result(lhs.value_ - rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] - rhs.derivatives_[i];
        }
        return result;
    }

    friend DualBase operator-(const DualBase& lhs, const T& rhs) {
        return DualBase(lhs.value_ - rhs, lhs.derivatives_);
    }

    friend DualBase operator-(const T& lhs, const DualBase& rhs) {
        DualBase result(lhs - rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = -rhs.derivatives_[i];
        }
        return result;
    }

    // Multiplication: (f * g)' = f' * g + f * g'
    friend DualBase operator*(const DualBase& lhs, const DualBase& rhs) {
        DualBase result(lhs.value_ * rhs.value_);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * rhs.value_ +
                                     lhs.value_ * rhs.derivatives_[i];
        }
        return result;
    }

    friend DualBase operator*(const DualBase& lhs, const T& rhs) {
        DualBase result(lhs.value_ * rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * rhs;
        }
        return result;
    }

    friend DualBase operator*(const T& lhs, const DualBase& rhs) {
        return rhs * lhs;  // Commutative
    }

    // Division: (f / g)' = (f' * g - f * g') / g²
    friend DualBase operator/(const DualBase& lhs, const DualBase& rhs) {
        const T inv_rhs = T(1) / rhs.value_;
        const T inv_rhs_sq = inv_rhs * inv_rhs;

        DualBase result(lhs.value_ * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = (lhs.derivatives_[i] * rhs.value_ -
                                      lhs.value_ * rhs.derivatives_[i]) * inv_rhs_sq;
        }
        return result;
    }

    friend DualBase operator/(const DualBase& lhs, const T& rhs) {
        const T inv_rhs = T(1) / rhs;
        DualBase result(lhs.value_ * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = lhs.derivatives_[i] * inv_rhs;
        }
        return result;
    }

    friend DualBase operator/(const T& lhs, const DualBase& rhs) {
        const T inv_rhs = T(1) / rhs.value_;
        const T inv_rhs_sq = inv_rhs * inv_rhs;

        DualBase result(lhs * inv_rhs);
        for (size_type i = 0; i < N; ++i) {
            result.derivatives_[i] = -lhs * rhs.derivatives_[i] * inv_rhs_sq;
        }
        return result;
    }

    // Unary negation
    friend DualBase operator-(const DualBase& x) {
        DualBase result(-x.value_);
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
    // Essential comparison operators (only equality)
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
// Core mathematical functions with derivatives
// ============================================================================

// Power function: d/dx[x^n] = n*x^(n-1)
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

// Square root: d/dx[√x] = 1/(2√x)
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

// Exponential: d/dx[e^x] = e^x
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

// Natural logarithm: d/dx[ln(x)] = 1/x
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

// Sine: d/dx[sin(x)] = cos(x)
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

// Cosine: d/dx[cos(x)] = -sin(x)
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

// ============================================================================
// Helper functions
// ============================================================================

// Create a dual number from value with zero derivatives
template<typename T, std::size_t N = 1>
DualBase<T, N> make_dual(const T& value) {
    return DualBase<T, N>(value);
}

// Create a dual number with specified value and derivatives
template<typename T, std::size_t N>
DualBase<T, N> make_dual(const T& value, const std::array<T, N>& derivatives) {
    return DualBase<T, N>(value, derivatives);
}

// Create a dual as independent variable (seed with unit vector)
template<typename T, std::size_t N>
DualBase<T, N> make_independent(const T& value, std::size_t index) {
    DualBase<T, N> result(value);
    result.seed(index);
    return result;
}

// Extract Jacobian matrix from array of dual numbers
template<typename T, std::size_t M, std::size_t N>
std::array<std::array<T, N>, M> extract_jacobian(
    const std::array<DualBase<T, N>, M>& dual_array) {

    std::array<std::array<T, N>, M> jacobian;
    for (std::size_t i = 0; i < M; ++i) {
        jacobian[i] = dual_array[i].derivatives();
    }
    return jacobian;
}

// Extract gradient from a scalar dual number
template<typename T, std::size_t N>
std::array<T, N> extract_gradient(const DualBase<T, N>& dual) {
    return dual.derivatives();
}

} // namespace fem::numeric::autodiff

namespace fem::numeric {
    // Bring AD types into numeric namespace for convenience
    using autodiff::DualBase;
} // namespace fem::numeric

#endif // NUMERIC_DUAL_BASE_H
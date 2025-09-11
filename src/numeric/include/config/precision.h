/**
 * @file precision.h
 * @brief Floating-point precision settings and numerical tolerances
 *
 * This header defines precision-related types, tolerances, and comparison
 * functions for floating-point operations in the FEM Numeric Library.
 */

#ifndef FEM_NUMERIC_PRECISION_H
#define FEM_NUMERIC_PRECISION_H

#include <cmath>
#include <limits>
#include <type_traits>
#include <complex>

#include "compiler.h"
#include "platform.h"

FEM_NUMERIC_BEGIN_NAMESPACE

// ============================================================================
// Precision Types
// ============================================================================

// Default real type for the library
#ifndef FEM_NUMERIC_REAL_TYPE
  #ifdef FEM_NUMERIC_USE_SINGLE_PRECISION
    using Real = float;
  #elif defined(FEM_NUMERIC_USE_LONG_DOUBLE)
    using Real = long double;
  #else
    using Real = double;  // Default
  #endif
#else
  using Real = FEM_NUMERIC_REAL_TYPE;
#endif

// Complex type based on Real
using Complex = std::complex<Real>;

// Size type for indexing
using size_type = FEM_NUMERIC_DEFAULT_INDEX;

// Signed size type
using ssize_type = std::make_signed_t<size_type>;

// ============================================================================
// Machine Epsilon and Tolerances
// ============================================================================

namespace precision {

// Machine epsilon for different types
template<typename T>
struct machine_epsilon {
    static constexpr T value = std::numeric_limits<T>::epsilon();
};

// Specialized for common types
template<> struct machine_epsilon<float> {
    static constexpr float value = 1.192092896e-07f;  // FLT_EPSILON
};

template<> struct machine_epsilon<double> {
    static constexpr double value = 2.2204460492503131e-16;  // DBL_EPSILON
};

template<> struct machine_epsilon<long double> {
    static constexpr long double value = 1.0842021724855044e-19L;  // LDBL_EPSILON
};

// Default tolerances based on machine epsilon
template<typename T>
struct default_tolerance {
    static constexpr T value = T(100) * machine_epsilon<T>::value;
};

// Tolerance for different operations
template<typename T>
struct tolerance {
    // Absolute tolerance for comparisons
    static constexpr T absolute = default_tolerance<T>::value;

    // Relative tolerance for comparisons
    static constexpr T relative = T(10) * machine_epsilon<T>::value;

    // Tolerance for matrix operations
    static constexpr T matrix = T(1000) * machine_epsilon<T>::value;

    // Tolerance for iterative solvers
    static constexpr T solver = T(1e-10);

    // Tolerance for Newton convergence
    static constexpr T newton = T(1e-12);

    // Tolerance for eigenvalue computations
    static constexpr T eigen = T(100) * machine_epsilon<T>::value;

    // Tolerance for rank determination
    static constexpr T rank = T(1e-14);

    // Tolerance for singularity detection
    static constexpr T singular = T(1e-15);
};

// Specialized tolerances for float
template<>
struct tolerance<float> {
    static constexpr float absolute = 1e-5f;
    static constexpr float relative = 1e-4f;
    static constexpr float matrix = 1e-4f;
    static constexpr float solver = 1e-6f;
    static constexpr float newton = 1e-7f;
    static constexpr float eigen = 1e-5f;
    static constexpr float rank = 1e-6f;
    static constexpr float singular = 1e-7f;
};

} // namespace precision

// ============================================================================
// Floating-Point Comparison Functions
// ============================================================================

namespace detail {

// Type trait to check if a type is floating-point or complex
template<typename T>
struct is_floating_or_complex : std::is_floating_point<T> {};

template<typename T>
struct is_floating_or_complex<std::complex<T>> : std::is_floating_point<T> {};

template<typename T>
inline constexpr bool is_floating_or_complex_v = is_floating_or_complex<T>::value;

} // namespace detail

// Absolute value that works for real and complex numbers
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
auto abs_value(const T& x) -> decltype(std::abs(x)) {
    return std::abs(x);
}

// Check if a value is NaN
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
is_nan(T x) {
    return std::isnan(x);
}

template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
bool is_nan(const std::complex<T>& x) {
    return std::isnan(x.real()) || std::isnan(x.imag());
}

// Check if a value is infinite
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
is_inf(T x) {
    return std::isinf(x);
}

template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
bool is_inf(const std::complex<T>& x) {
    return std::isinf(x.real()) || std::isinf(x.imag());
}

// Check if a value is finite
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
is_finite(T x) {
    return std::isfinite(x);
}

template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
bool is_finite(const std::complex<T>& x) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
}

// Fuzzy equality with absolute tolerance
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<detail::is_floating_or_complex_v<T>, bool>
near_zero(const T& x,
          typename precision::tolerance<typename std::decay_t<T>>::absolute tol =
          precision::tolerance<typename std::decay_t<T>>::absolute) {
    return abs_value(x) <= tol;
}

// Fuzzy equality with both absolute and relative tolerance
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
nearly_equal(T a, T b,
             T rel_tol = precision::tolerance<T>::relative,
             T abs_tol = precision::tolerance<T>::absolute) {
    // Handle exact equality (also handles infinities)
    if (a == b) return true;

    // Handle NaN
    if (is_nan(a) || is_nan(b)) return false;

    const T diff = std::abs(a - b);

    // Absolute tolerance check
    if (diff <= abs_tol) return true;

    // Relative tolerance check
    const T largest = std::max(std::abs(a), std::abs(b));
    return diff <= largest * rel_tol;
}

// Complex number equality
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
bool nearly_equal(const std::complex<T>& a, const std::complex<T>& b,
                  T rel_tol = precision::tolerance<T>::relative,
                  T abs_tol = precision::tolerance<T>::absolute) {
    return nearly_equal(a.real(), b.real(), rel_tol, abs_tol) &&
           nearly_equal(a.imag(), b.imag(), rel_tol, abs_tol);
}

// Check if a < b with tolerance
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
definitely_less_than(T a, T b, T tol = precision::tolerance<T>::absolute) {
    return a < b - tol;
}

// Check if a > b with tolerance
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
definitely_greater_than(T a, T b, T tol = precision::tolerance<T>::absolute) {
    return a > b + tol;
}

// ============================================================================
// Rounding and Truncation
// ============================================================================

// Round to n decimal places
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
round_to_decimals(T value, int n) {
    const T multiplier = std::pow(T(10), n);
    return std::round(value * multiplier) / multiplier;
}

// Round to n significant figures
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
round_to_significant(T value, int n) {
    if (value == T(0)) return T(0);

    const T d = std::ceil(std::log10(std::abs(value)));
    const T power = n - d;
    const T magnitude = std::pow(T(10), power);

    return std::round(value * magnitude) / magnitude;
}

// Truncate small values to zero
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
chop(T value, T tol = precision::tolerance<T>::absolute) {
    return (std::abs(value) < tol) ? T(0) : value;
}

template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
std::complex<T> chop(const std::complex<T>& value,
                     T tol = precision::tolerance<T>::absolute) {
    return std::complex<T>(chop(value.real(), tol), chop(value.imag(), tol));
}

// ============================================================================
// Special Values
// ============================================================================

namespace constants {

// Mathematical constants with appropriate precision
template<typename T>
struct math {
    static constexpr T pi = T(3.141592653589793238462643383279502884L);
    static constexpr T e = T(2.718281828459045235360287471352662498L);
    static constexpr T sqrt2 = T(1.414213562373095048801688724209698079L);
    static constexpr T sqrt3 = T(1.732050807568877293527446341505872367L);
    static constexpr T golden_ratio = T(1.618033988749894848204586834365638118L);
    static constexpr T euler_gamma = T(0.577215664901532860606512090082402431L);
};

// Special floating-point values
template<typename T>
struct special {
    static constexpr T inf = std::numeric_limits<T>::infinity();
    static constexpr T nan = std::numeric_limits<T>::quiet_NaN();
    static constexpr T max = std::numeric_limits<T>::max();
    static constexpr T min = std::numeric_limits<T>::min();
    static constexpr T lowest = std::numeric_limits<T>::lowest();
    static constexpr T denorm_min = std::numeric_limits<T>::denorm_min();
};

} // namespace constants

// ============================================================================
// Floating-Point Environment Control
// ============================================================================

#if FEM_NUMERIC_HAS_FENV

// RAII class for floating-point environment
class FloatingPointEnvironment {
private:
    std::fenv_t env_backup;
    bool saved;

public:
    FloatingPointEnvironment() : saved(false) {}

    // Save current environment
    void save() {
        if (std::fegetenv(&env_backup) == 0) {
            saved = true;
        }
    }

    // Restore saved environment
    void restore() {
        if (saved) {
            std::fesetenv(&env_backup);
            saved = false;
        }
    }

    // Set rounding mode
    static bool set_rounding(int mode) {
        return std::fesetround(mode) == 0;
    }

    // Get current rounding mode
    static int get_rounding() {
        return std::fegetround();
    }

    // Clear floating-point exceptions
    static void clear_exceptions() {
        std::feclearexcept(FE_ALL_EXCEPT);
    }

    // Check for floating-point exceptions
    static bool has_exception(int excepts) {
        return std::fetestexcept(excepts) != 0;
    }

    ~FloatingPointEnvironment() {
        restore();
    }
};

// RAII class to temporarily set rounding mode
class ScopedRoundingMode {
private:
    int old_mode;

public:
    explicit ScopedRoundingMode(int new_mode)
        : old_mode(std::fegetround()) {
        std::fesetround(new_mode);
    }

    ~ScopedRoundingMode() {
        std::fesetround(old_mode);
    }

    // Disable copy and move
    ScopedRoundingMode(const ScopedRoundingMode&) = delete;
    ScopedRoundingMode& operator=(const ScopedRoundingMode&) = delete;
};

#endif // FEM_NUMERIC_HAS_FENV

// ============================================================================
// Numerical Stability Helpers
// ============================================================================

// Safe division with zero check
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
safe_divide(T numerator, T denominator,
            T zero_result = T(0),
            T tol = precision::tolerance<T>::singular) {
    return (std::abs(denominator) > tol) ? (numerator / denominator) : zero_result;
}

// Safe reciprocal
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
safe_reciprocal(T x, T tol = precision::tolerance<T>::singular) {
    return safe_divide(T(1), x, constants::special<T>::inf, tol);
}

// Safe square root (returns 0 for negative values)
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
safe_sqrt(T x) {
    return (x > T(0)) ? std::sqrt(x) : T(0);
}

// Numerically stable norm computation (avoids overflow)
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
stable_norm(T x, T y) {
    const T ax = std::abs(x);
    const T ay = std::abs(y);

    if (ax > ay) {
        const T r = ay / ax;
        return ax * std::sqrt(T(1) + r * r);
    } else if (ay > T(0)) {
        const T r = ax / ay;
        return ay * std::sqrt(T(1) + r * r);
    } else {
        return T(0);
    }
}

// Numerically stable norm for three components
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, T>
stable_norm(T x, T y, T z) {
    const T ax = std::abs(x);
    const T ay = std::abs(y);
    const T az = std::abs(z);

    const T max_val = std::max({ax, ay, az});

    if (max_val > T(0)) {
        const T x_scaled = x / max_val;
        const T y_scaled = y / max_val;
        const T z_scaled = z / max_val;
        return max_val * std::sqrt(x_scaled * x_scaled +
                                   y_scaled * y_scaled +
                                   z_scaled * z_scaled);
    } else {
        return T(0);
    }
}

// ============================================================================
// Unit in Last Place (ULP) Comparisons
// ============================================================================

// Get the number of ULPs between two floating-point numbers
template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, std::int64_t>
ulp_distance(T a, T b) {
    // Handle special cases
    if (a == b) return 0;
    if (is_nan(a) || is_nan(b)) return std::numeric_limits<std::int64_t>::max();
    if (is_inf(a) || is_inf(b)) return std::numeric_limits<std::int64_t>::max();

    // Different signs means large distance
    if ((a < 0) != (b < 0)) return std::numeric_limits<std::int64_t>::max();

    // Reinterpret as integers for ULP calculation
    std::int64_t ia, ib;
    if constexpr (sizeof(T) == sizeof(float)) {
        ia = *reinterpret_cast<const std::int32_t*>(&a);
        ib = *reinterpret_cast<const std::int32_t*>(&b);
    } else {
        ia = *reinterpret_cast<const std::int64_t*>(&a);
        ib = *reinterpret_cast<const std::int64_t*>(&b);
    }

    // Make lexicographically ordered as twos-complement
    if (ia < 0) ia = 0x8000000000000000LL - ia;
    if (ib < 0) ib = 0x8000000000000000LL - ib;

    return std::abs(ia - ib);
}

// Check if two values are within n ULPs
template<typename T>
FEM_NUMERIC_ALWAYS_INLINE
typename std::enable_if_t<std::is_floating_point_v<T>, bool>
within_ulps(T a, T b, std::int64_t max_ulps = 4) {
    return ulp_distance(a, b) <= max_ulps;
}

FEM_NUMERIC_END_NAMESPACE

#endif // FEM_NUMERIC_PRECISION_H
#pragma once

#ifndef NUMERIC_TRAITS_H
#define NUMERIC_TRAITS_H

#include <limits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>

#include "../base/numeric_base.h"
#include "../base/traits_base.h"

#include "type_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Extended numeric limits providing additional properties
     * Builds on std::numeric_limits and integrates with base types
     */
    template<typename T>
    struct numeric_limits : std::numeric_limits<T> {
        using base = std::numeric_limits<T>;

        // Additional properties not in std::numeric_limits
        static constexpr bool is_ieee754 = IEEECompliant<T>;

        static constexpr bool supports_nan = base::has_quiet_NaN ||
                                             base::has_signaling_NaN;

        static constexpr bool supports_infinity = base::has_infinity;

        // Machine epsilon for comparisons
        static constexpr T machine_epsilon() noexcept {
            if constexpr (std::is_floating_point_v<T>) {
                return base::epsilon();
            } else {
                return T{1};
            }
        }

        // Safe comparison tolerance
        static constexpr T comparison_tolerance() noexcept {
            if constexpr (std::is_floating_point_v<T>) {
                return base::epsilon() * T{100};  // 100 * machine epsilon
            } else {
                return T{0};
            }
        }

        // Get a "zero" value
        static constexpr T zero() noexcept {
            return T{0};
        }

        // Get a "one" value
        static constexpr T one() noexcept {
            return T{1};
        }

        // Get the additive identity
        static constexpr T additive_identity() noexcept {
            return zero();
        }

        // Get the multiplicative identity
        static constexpr T multiplicative_identity() noexcept {
            return one();
        }
    };

    /**
     * @brief Specialization for complex numbers
     */
    template<typename T>
    struct numeric_limits<std::complex<T>> {
        using value_type = std::complex<T>;
        using real_type = T;

        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = std::numeric_limits<T>::is_exact;
        static constexpr bool is_ieee754 = IEEECompliant<std::complex<T>>;
        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = std::numeric_limits<T>::has_signaling_NaN;
        static constexpr bool supports_nan = has_quiet_NaN || has_signaling_NaN;
        static constexpr bool supports_infinity = has_infinity;

        static constexpr int digits = std::numeric_limits<T>::digits;
        static constexpr int digits10 = std::numeric_limits<T>::digits10;
        static constexpr int max_digits10 = std::numeric_limits<T>::max_digits10;

        static constexpr std::complex<T> zero() noexcept {
            return {T{0}, T{0}};
        }

        static constexpr std::complex<T> one() noexcept {
            return {T{1}, T{0}};
        }

        static constexpr std::complex<T> i() noexcept {
            return {T{0}, T{1}};
        }

        static constexpr std::complex<T> epsilon() noexcept {
            return {std::numeric_limits<T>::epsilon(), T{0}};
        }

        static constexpr std::complex<T> machine_epsilon() noexcept {
            return epsilon();
        }

        static constexpr std::complex<T> comparison_tolerance() noexcept {
            return {std::numeric_limits<T>::epsilon() * T{100}, T{0}};
        }

        static constexpr std::complex<T> quiet_NaN() noexcept {
            if constexpr (has_quiet_NaN) {
                const T nan = std::numeric_limits<T>::quiet_NaN();
                return {nan, nan};
            } else {
                return zero();
            }
        }

        static constexpr std::complex<T> infinity() noexcept {
            if constexpr (has_infinity) {
                return {std::numeric_limits<T>::infinity(), T{0}};
            } else {
                return {std::numeric_limits<T>::max(), T{0}};
            }
        }

        static constexpr std::complex<T> min() noexcept {
            return {std::numeric_limits<T>::min(), T{0}};
        }

        static constexpr std::complex<T> max() noexcept {
            return {std::numeric_limits<T>::max(), T{0}};
        }

        static constexpr std::complex<T> lowest() noexcept {
            return {std::numeric_limits<T>::lowest(), T{0}};
        }
    };

    /**
     * @brief Precision categories for floating-point types
     */
    enum class PrecisionCategory {
        Half,        // 16-bit
        Single,      // 32-bit
        Double,      // 64-bit
        Extended,    // 80-bit
        Quad,        // 128-bit
        Unknown
    };

    /**
     * @brief Get precision category for a type
     */
    template<typename T>
    struct precision_category {
        static constexpr PrecisionCategory value = [] {
            if constexpr (std::is_floating_point_v<T>) {
                if constexpr (sizeof(T) == 2) return PrecisionCategory::Half;
                else if constexpr (sizeof(T) == 4) return PrecisionCategory::Single;
                else if constexpr (sizeof(T) == 8) return PrecisionCategory::Double;
                else if constexpr (sizeof(T) == 10) return PrecisionCategory::Extended;
                else if constexpr (sizeof(T) == 16) return PrecisionCategory::Quad;
                else return PrecisionCategory::Unknown;
            } else {
                return PrecisionCategory::Unknown;
            }
        }();
    };

    template<typename T>
    inline constexpr PrecisionCategory precision_category_v = precision_category<T>::value;

    /**
     * @brief Use promote_traits from traits_base.h and extend it
     */
    using fem::numeric::promote_traits;
    using fem::numeric::promote_t;

    /**
     * @brief Variadic type promotion (promote multiple types)
     */
    template<typename... Types>
    struct promote_types;

    template<typename T>
    struct promote_types<T> {
        using type = T;
    };

    template<typename T1, typename T2>
    struct promote_types<T1, T2> {
        using type = promote_t<T1, T2>;
    };

    template<typename T1, typename T2, typename... Rest>
    struct promote_types<T1, T2, Rest...> {
        using type = typename promote_types<
                promote_t<T1, T2>, Rest...>::type;
    };

    template<typename... Types>
    using promote_types_t = typename promote_types<Types...>::type;

    /**
     * @brief Get the number of bytes needed to store N elements
     */
    template<typename T>
    struct storage_requirements {
        static constexpr size_t bytes_per_element = sizeof(T);
        static constexpr size_t alignment = alignof(T);  // Use natural alignment of T

        static constexpr size_t bytes_needed(size_t n) noexcept {
            return n * bytes_per_element;
        }

        static constexpr size_t aligned_bytes_needed(size_t n, size_t align = alignment) noexcept {
            size_t bytes = bytes_needed(n);
            size_t remainder = bytes % align;
            return remainder == 0 ? bytes : bytes + (align - remainder);
        }
    };

    /**
     * @brief Numeric properties for a type (integrates with base traits)
     */
    template<typename T>
    struct numeric_properties {
        using value_type = T;
        using limits = numeric_limits<T>;
        using base_traits = numeric_traits<T>;  // From traits_base.h

        // Type classification
        static constexpr bool is_integral = std::is_integral_v<T>;
        static constexpr bool is_floating = std::is_floating_point_v<T>;
        static constexpr bool is_complex = base_traits::is_complex;
        static constexpr bool is_signed = base_traits::is_signed;
        static constexpr bool is_exact = std::numeric_limits<T>::is_exact;

        // IEEE compliance (using base concepts)
        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee = IEEECompliant<T>;
        static constexpr bool has_nan = base_traits::has_quiet_nan;
        static constexpr bool has_inf = base_traits::has_infinity;

        // Precision info
        static constexpr int mantissa_bits = std::numeric_limits<T>::digits;
        static constexpr int decimal_digits = std::numeric_limits<T>::digits10;
        static constexpr PrecisionCategory precision = precision_category_v<T>;

        // Memory layout
        static constexpr size_t size = sizeof(T);
        static constexpr size_t alignment = alignof(T);
        static constexpr bool is_pod = is_pod_v<T>;

        // Special values (using base traits)
        static constexpr T zero() noexcept { return base_traits::zero(); }
        static constexpr T one() noexcept { return base_traits::one(); }
        static constexpr T eps() noexcept { return base_traits::epsilon(); }
        static constexpr T min() noexcept { return base_traits::min(); }
        static constexpr T max() noexcept { return base_traits::max(); }
    };

    /**
     * @brief Check if a value is finite (uses IEEEComplianceChecker from base)
     */
    template<typename T>
    inline bool is_finite(T value) noexcept {
    return IEEEComplianceChecker::is_finite(value);
    }

    /**
     * @brief Check if a value is NaN (uses IEEEComplianceChecker from base)
     */
    template<typename T>
    inline bool is_nan(T value) noexcept {
    return IEEEComplianceChecker::is_nan(value);
    }

    /**
     * @brief Check if a value is infinite (uses IEEEComplianceChecker from base)
     */
    template<typename T>
    inline bool is_inf(T value) noexcept {
    return IEEEComplianceChecker::is_inf(value);
    }

    /**
     * @brief Safe comparison with tolerance for floating-point types
     */
    template<typename T>
    inline bool approximately_equal(T a, T b,
                                   T tolerance = numeric_limits<T>::comparison_tolerance()) noexcept {
        if constexpr (std::is_floating_point_v<T>) {
            // Handle special cases using base functions
            if (is_nan(a) || is_nan(b)) return false;
            if (is_inf(a) || is_inf(b)) return a == b;

            // Check for exact equality first (handles 0.0 == 0.0)
            if (a == b) return true;

            T diff = std::abs(a - b);
            T abs_a = std::abs(a);
            T abs_b = std::abs(b);

            // Get the larger of the two absolute values
            T largest = (abs_b > abs_a) ? abs_b : abs_a;

            // If the largest value is smaller than the tolerance,
            // we're dealing with values very close to zero where
            // relative tolerance doesn't make sense.
            // Use absolute tolerance instead.
            if (largest <= tolerance) {
                return diff <= tolerance;
            }

            // For normal-sized values, use relative tolerance
            return diff <= largest * tolerance;

        } else if constexpr (is_complex_v<T>) {
            return approximately_equal(a.real(), b.real(), tolerance.real()) &&
                   approximately_equal(a.imag(), b.imag(), tolerance.real());
        } else {
            return a == b;  // Exact comparison for integers
        }
    }

    /**
     * @brief Get the appropriate zero value for a type (using base traits)
     */
    template<typename T>
    inline constexpr T zero_value() noexcept {
        return numeric_traits<T>::zero();
    }

    /**
     * @brief Get the appropriate one value for a type (using base traits)
     */
    template<typename T>
    inline constexpr T one_value() noexcept {
        return numeric_traits<T>::one();
    }

    /**
     * @brief Check compatibility between types (using base helper)
     */
    template<typename T1, typename T2>
    inline constexpr bool are_compatible_types = are_compatible_v<T1, T2>;

} // namespace fem::numeric::traits

#endif //NUMERIC_TRAITS_H

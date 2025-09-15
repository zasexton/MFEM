#pragma once

#ifndef NUMERIC_TRAITS_BASE_H
#define NUMERIC_TRAITS_BASE_H

#include <type_traits>
#include <complex>
#include <limits>

#include "numeric_base.h"
#include "dual_base.h"

namespace fem::numeric {

    // ============================================================================
    // Type detection traits
    // ============================================================================

    /**
     * @brief Check if a type is a dual number
     * This will be specialized when dual_base.h is included
     */
    template<typename T>
    struct is_dual_number : std::false_type {};

    template<typename T>
    inline constexpr bool is_dual_number_v = is_dual_number<T>::value;

    /**
     * @brief Check if a type is a complex number
     */
    template<typename T>
    struct is_complex_number : std::false_type {};

    template<typename T>
    struct is_complex_number<std::complex<T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_complex_number_v = is_complex_number<T>::value;

    /**
     * @brief Concept for types that can be stored in containers
     * Expanded from NumberLike to support composite types
     */
    template<typename T>
    concept StorableType = NumberLike<T> ||
                          is_complex_number_v<T> ||
                          is_dual_number_v<T>;

    /**
     * @brief Extract the underlying scalar type from composite types
     */
    template<typename T>
    struct scalar_type {
        using type = T;
    };

    template<typename T>
    struct scalar_type<std::complex<T>> {
        using type = T;
    };

    // This will be specialized when dual_base.h is included
    template<typename T>
    using scalar_type_t = typename scalar_type<T>::type;

    // ============================================================================
    // Storage optimization traits
    // ============================================================================

    /**
     * @brief Traits to determine storage optimization strategies
     */
    template<typename T>
    struct storage_optimization_traits {
        // Can we use memcpy/memmove for this type?
        static constexpr bool is_trivially_relocatable =
            std::is_trivially_copyable_v<T> && !is_dual_number_v<T>;

        // Can we use SIMD operations?
        // Allow SIMD for arithmetic scalars and complex numbers of arithmetic scalars.
        static constexpr bool supports_simd =
            !is_dual_number_v<T> && (
                std::is_arithmetic_v<T> || is_complex_number_v<T>
            );

        // Should we use aligned allocation for better performance?
        static constexpr bool prefers_alignment =
            sizeof(T) >= 16 || is_dual_number_v<T>;

        // Can we use fast fill operations?
        static constexpr bool supports_fast_fill =
            std::is_trivially_copyable_v<T> && sizeof(T) <= 16 && !is_dual_number_v<T>;
    };

    /**
     * @brief Base traits for numeric types
     *
     * Provides compile-time information about IEEE-compliant numeric types
     */
    template<typename T>
    struct numeric_traits {
        using value_type = T;
        using real_type = T;
        using scalar_type = T;  // For real types, scalar is the type itself
        using complex_type = std::complex<T>;

        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee_compliant = IEEECompliant<T>;
        static constexpr bool is_floating_point = std::is_floating_point_v<T>;
        static constexpr bool is_integral = std::is_integral_v<T>;
        static constexpr bool is_signed = std::is_signed_v<T>;
        static constexpr bool is_complex = false;
        static constexpr bool is_dual = false;
        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_nan = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_nan = std::numeric_limits<T>::has_signaling_NaN;

        static constexpr size_t size = sizeof(T);
        static constexpr size_t alignment = alignof(T);

        static constexpr T zero() noexcept { return T{0}; }
        static constexpr T one() noexcept { return T{1}; }
        static constexpr T min() noexcept { return std::numeric_limits<T>::min(); }
        static constexpr T max() noexcept { return std::numeric_limits<T>::max(); }
        static constexpr T lowest() noexcept { return std::numeric_limits<T>::lowest(); }
        static constexpr T epsilon() noexcept { return std::numeric_limits<T>::epsilon(); }

        static constexpr T quiet_nan() noexcept {
            if constexpr (has_quiet_nan) {
                return std::numeric_limits<T>::quiet_NaN();
            } else {
                return T{};
            }
        }

        static constexpr T infinity() noexcept {
            if constexpr (has_infinity) {
                return std::numeric_limits<T>::infinity();
            } else {
                return max();
            }
        }

        static constexpr T neg_infinity() noexcept {
            if constexpr (has_infinity) {
                return -std::numeric_limits<T>::infinity();
            } else {
                return lowest();
            }
        }
    };

    /**
     * @brief Specialization for complex numbers
     */
    template<typename T>
    struct numeric_traits<std::complex<T>> {
        using value_type = std::complex<T>;
        using real_type = T;
        using scalar_type = T;  // Underlying scalar type
        using complex_type = std::complex<T>;

        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee_compliant = IEEECompliant<T>;
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_signed = true;
        static constexpr bool is_complex = true;
        static constexpr bool is_dual = false;
        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_nan = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_nan = std::numeric_limits<T>::has_signaling_NaN;

        static constexpr size_t size = sizeof(std::complex<T>);
        static constexpr size_t alignment = alignof(std::complex<T>);

        static constexpr std::complex<T> zero() noexcept { return {T{0}, T{0}}; }
        static constexpr std::complex<T> one() noexcept { return {T{1}, T{0}}; }
        static constexpr std::complex<T> i() noexcept { return {T{0}, T{1}}; }

        static constexpr std::complex<T> quiet_nan() noexcept {
            if constexpr (has_quiet_nan) {
                return {std::numeric_limits<T>::quiet_NaN(),
                        std::numeric_limits<T>::quiet_NaN()};
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
    };

    // (DualBase traits specialization moved to end of file, alongside other dual helpers)

    /**
     * @brief Type promotion rules (similar to NumPy)
     */
    template<typename T1, typename T2>
    struct promote_traits {
        using type = decltype(std::declval<T1>() + std::declval<T2>());
    };

    // Specializations for same-type promotions to preserve type
    template<> struct promote_traits<int8_t, int8_t> { using type = int8_t; };
    template<> struct promote_traits<int16_t, int16_t> { using type = int16_t; };
    template<> struct promote_traits<int32_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int64_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<uint8_t, uint8_t> { using type = uint8_t; };
    template<> struct promote_traits<uint16_t, uint16_t> { using type = uint16_t; };
    template<> struct promote_traits<uint32_t, uint32_t> { using type = uint32_t; };
    template<> struct promote_traits<uint64_t, uint64_t> { using type = uint64_t; };
    template<> struct promote_traits<float, float> { using type = float; };
    template<> struct promote_traits<double, double> { using type = double; };

    // Unsigned widening promotions to the wider unsigned type (common cases)
    template<> struct promote_traits<uint8_t, uint16_t> { using type = uint16_t; };
    template<> struct promote_traits<uint16_t, uint8_t> { using type = uint16_t; };
    template<> struct promote_traits<uint8_t, uint32_t> { using type = uint32_t; };
    template<> struct promote_traits<uint32_t, uint8_t> { using type = uint32_t; };
    template<> struct promote_traits<uint8_t, uint64_t> { using type = uint64_t; };
    template<> struct promote_traits<uint64_t, uint8_t> { using type = uint64_t; };
    template<> struct promote_traits<uint16_t, uint32_t> { using type = uint32_t; };
    template<> struct promote_traits<uint32_t, uint16_t> { using type = uint32_t; };
    template<> struct promote_traits<uint16_t, uint64_t> { using type = uint64_t; };
    template<> struct promote_traits<uint64_t, uint16_t> { using type = uint64_t; };
    template<> struct promote_traits<uint32_t, uint64_t> { using type = uint64_t; };
    template<> struct promote_traits<uint64_t, uint32_t> { using type = uint64_t; };

    // Signed integer widening promotions
    template<> struct promote_traits<int8_t, int16_t> { using type = int16_t; };
    template<> struct promote_traits<int16_t, int8_t> { using type = int16_t; };
    template<> struct promote_traits<int8_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int32_t, int8_t> { using type = int32_t; };
    template<> struct promote_traits<int8_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int64_t, int8_t> { using type = int64_t; };
    template<> struct promote_traits<int16_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int32_t, int16_t> { using type = int32_t; };
    template<> struct promote_traits<int16_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int64_t, int16_t> { using type = int64_t; };
    template<> struct promote_traits<int32_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int64_t, int32_t> { using type = int64_t; };

    // Mixed sign common promotions where signed is wider
    template<> struct promote_traits<int16_t, uint8_t> { using type = int16_t; };
    template<> struct promote_traits<uint8_t, int16_t> { using type = int16_t; };
    template<> struct promote_traits<int32_t, uint16_t> { using type = int32_t; };
    template<> struct promote_traits<uint16_t, int32_t> { using type = int32_t; };

    // Integer to floating promotions (defer to decltype for mixed; explicit same-type handled above)
    template<> struct promote_traits<float, double> { using type = double; };

    // Complex promotions
    template<typename T1, typename T2>
    struct promote_traits<std::complex<T1>, T2> {
        using real_promoted = typename promote_traits<T1, T2>::type;
        using type = std::complex<real_promoted>;
    };

    template<typename T1, typename T2>
    struct promote_traits<T1, std::complex<T2>> {
        using real_promoted = typename promote_traits<T1, T2>::type;
        using type = std::complex<real_promoted>;
    };

    template<typename T1, typename T2>
    struct promote_traits<std::complex<T1>, std::complex<T2>> {
        using real_promoted = typename promote_traits<T1, T2>::type;
        using type = std::complex<real_promoted>;
    };

    template<typename T1, typename T2>
    using promote_t = typename promote_traits<T1, T2>::type;

    /**
     * @brief Container traits
     */
    template<typename Container>
    struct container_traits {
        using container_type = Container;
        using value_type = typename Container::value_type;
        using size_type = typename Container::size_type;

        static constexpr bool is_container = requires(Container c) {
            typename Container::value_type;
            { c.size() } -> std::convertible_to<size_t>;
            { c.data() };
        };

        static constexpr bool is_numeric_container =
                is_container && StorableType<value_type>;

        static constexpr bool is_ieee_container =
                is_numeric_container && IEEECompliant<value_type>;

        static constexpr bool has_shape = requires(Container c) {
            { c.shape() } -> std::convertible_to<Shape>;
        };

        static constexpr bool is_resizable = requires(Container c) {
            { c.resize(std::declval<Shape>()) };
        };

        static constexpr bool is_view = requires(Container c) {
            { c.is_view() } -> std::convertible_to<bool>;
        };
    };

    /**
     * @brief Storage traits
     */
    template<typename Storage>
    struct storage_traits {
        using storage_type = Storage;
        using value_type = typename Storage::value_type;

        static constexpr bool is_dynamic = requires(Storage s) {
            { s.resize(size_t{}) };
        };

        static constexpr bool is_static = !is_dynamic;

        static constexpr bool is_contiguous = requires(Storage s) {
            { s.is_contiguous() } -> std::convertible_to<bool>;
        };

        static constexpr bool is_aligned = requires {
            Storage::alignment;
        };

        static constexpr size_t alignment = [] {
            if constexpr (is_aligned) {
            return Storage::alignment;
        } else {
            return alignof(value_type);
        }
        }();

        static constexpr bool supports_simd =
            storage_optimization_traits<value_type>::supports_simd;

        static constexpr bool is_trivially_relocatable =
            storage_optimization_traits<value_type>::is_trivially_relocatable;
    };

    /**
     * @brief Operation traits for compile-time operation validation
     */
    template<typename Op, typename T>
    struct operation_traits {
        using operation_type = Op;
        using value_type = T;

        static constexpr bool is_binary = requires(Op op, T a, T b) {
            { op(a, b) } -> std::convertible_to<T>;
        };

        static constexpr bool is_unary = requires(Op op, T a) {
            { op(a) } -> std::convertible_to<T>;
        };

        static constexpr bool preserves_type =
                std::is_same_v<decltype(std::declval<Op>()(std::declval<T>())), T>;

        static constexpr bool is_comparison = requires(Op op, T a, T b) {
            { op(a, b) } -> std::convertible_to<bool>;
        };

        static constexpr bool is_ieee_safe = [] {
            if constexpr (std::is_floating_point_v<T>) {
            // Check if operation handles NaN/Inf correctly
            return true;  // Most standard operations are IEEE-compliant
        }
            return true;
        }();
    };

    /**
     * @brief SIMD traits for vectorization support
     */
    template<typename T>
    struct simd_traits {
        using value_type = T;

        static constexpr bool is_vectorizable =
                std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && !is_dual_number_v<T>;

        static constexpr size_t vector_size = [] {
            if constexpr (is_vectorizable) {
            // Determine SIMD vector size based on type and architecture
            if constexpr (sizeof(T) == 4) {
                return 8;  // 256-bit AVX for 32-bit types
            } else if constexpr (sizeof(T) == 8) {
                return 4;  // 256-bit AVX for 64-bit types
            } else {
                return 16 / sizeof(T);  // 128-bit SSE fallback
            }
        }
            return 1;
        }();

        static constexpr size_t alignment = [] {
            if constexpr (is_vectorizable) {
            return vector_size * sizeof(T);
        }
            return alignof(T);
        }();
    };

    /**
     * @brief Helper to check if types are compatible for operations
     */
    template<typename T1, typename T2>
    struct are_compatible {
        static constexpr bool value =
                NumberLike<T1> && NumberLike<T2> &&
                std::is_convertible_v<T1, typename promote_traits<T1, T2>::type> &&
                std::is_convertible_v<T2, typename promote_traits<T1, T2>::type>;
    };

    template<typename T1, typename T2>
    inline constexpr bool are_compatible_v = are_compatible<T1, T2>::value;

} // namespace fem::numeric


namespace fem::numeric {
    // Specialization to recognize DualBase as a dual number
    template<typename T, std::size_t N>
    struct is_dual_number<DualBase<T, N>> : std::true_type {};

    // Preserve cv-qualifiers in detection
    template<typename T>
    struct is_dual_number<const T> : is_dual_number<T> {};
    template<typename T>
    struct is_dual_number<volatile T> : is_dual_number<T> {};
    template<typename T>
    struct is_dual_number<const volatile T> : is_dual_number<T> {};

    // Specialization for scalar_type extraction
    template<typename T, std::size_t N>
    struct scalar_type<DualBase<T, N>> {
        using type = T;
    };

    // Numeric traits specialization for DualBase
    template<typename T, std::size_t N>
    struct numeric_traits<DualBase<T, N>> {
        using value_type = DualBase<T, N>;
        using real_type = T;
        using scalar_type = T;  // Underlying scalar type
        using complex_type = std::complex<T>;

        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee_compliant = IEEECompliant<T>;
        static constexpr bool is_floating_point = false;  // It's composite
        static constexpr bool is_integral = false;
        static constexpr bool is_signed = std::is_signed_v<T>;
        static constexpr bool is_complex = false;
        static constexpr bool is_dual = true;  // Mark as dual
        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_nan = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_nan = false; // signaling NaN not modeled for Dual

        static constexpr size_t size = sizeof(value_type);
        static constexpr size_t alignment = alignof(value_type);
        static constexpr size_t num_derivatives = N;

        // Factory methods
        static value_type zero() noexcept {
            return value_type(T{0});
        }

        static value_type one() noexcept {
            return value_type(T{1});
        }

        static value_type quiet_nan() noexcept {
            if constexpr (has_quiet_nan) {
                return value_type(std::numeric_limits<T>::quiet_NaN());
            } else {
                return zero();
            }
        }

        static value_type infinity() noexcept {
            if constexpr (std::numeric_limits<T>::has_infinity) {
                return value_type(std::numeric_limits<T>::infinity());
            } else {
                return value_type(std::numeric_limits<T>::max());
            }
        }

        static value_type neg_infinity() noexcept {
            if constexpr (std::numeric_limits<T>::has_infinity) {
                return value_type(-std::numeric_limits<T>::infinity());
            } else {
                return value_type(std::numeric_limits<T>::lowest());
            }
        }

        static value_type make_independent(const T& val, std::size_t index) {
            value_type result(val);
            result.seed(index);
            return result;
        }
    };
}

#endif //NUMERIC_TRAITS_BASE_H

#pragma once

#ifndef NUMERIC_TRAITS_BASE_H
#define NUMERIC_TRAITS_BASE_H

#include <type_traits>
#include <complex>
#include <limits>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base traits for numeric types
     *
     * Provides compile-time information about IEEE-compliant numeric types
     */
    template<typename T>
    struct numeric_traits {
        using value_type = T;
        using real_type = T;
        using complex_type = std::complex<T>;

        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee_compliant = IEEECompliant<T>;
        static constexpr bool is_floating_point = std::is_floating_point_v<T>;
        static constexpr bool is_integral = std::is_integral_v<T>;
        static constexpr bool is_signed = std::is_signed_v<T>;
        static constexpr bool is_complex = false;
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
        using complex_type = std::complex<T>;

        static constexpr bool is_number_like = NumberLike<T>;
        static constexpr bool is_ieee_compliant = IEEECompliant<T>;
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_signed = true;
        static constexpr bool is_complex = true;
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

    /**
     * @brief Type promotion rules (similar to NumPy)
     */
    template<typename T1, typename T2>
    struct promote_traits {
        using type = decltype(std::declval<T1>() + std::declval<T2>());
    };

    // Specializations for common promotions
    template<> struct promote_traits<int8_t, int8_t> { using type = int8_t; };
    template<> struct promote_traits<int8_t, int16_t> { using type = int16_t; };
    template<> struct promote_traits<int8_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int8_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int8_t, float> { using type = float; };
    template<> struct promote_traits<int8_t, double> { using type = double; };

    template<> struct promote_traits<int16_t, int16_t> { using type = int16_t; };
    template<> struct promote_traits<int16_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int16_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int16_t, float> { using type = float; };
    template<> struct promote_traits<int16_t, double> { using type = double; };

    template<> struct promote_traits<int32_t, int32_t> { using type = int32_t; };
    template<> struct promote_traits<int32_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int32_t, float> { using type = float; };
    template<> struct promote_traits<int32_t, double> { using type = double; };

    template<> struct promote_traits<int64_t, int64_t> { using type = int64_t; };
    template<> struct promote_traits<int64_t, float> { using type = float; };
    template<> struct promote_traits<int64_t, double> { using type = double; };

    template<> struct promote_traits<float, float> { using type = float; };
    template<> struct promote_traits<float, double> { using type = double; };
    template<> struct promote_traits<double, double> { using type = double; };

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
                is_container && NumberLike<value_type>;

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
                typename Storage::alignment;
        };

        static constexpr size_t alignment = [] {
            if constexpr (is_aligned) {
            return Storage::alignment;
        } else {
            return alignof(value_type);
        }
        }();
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
                std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

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

#endif //NUMERIC_TRAITS_BASE_H

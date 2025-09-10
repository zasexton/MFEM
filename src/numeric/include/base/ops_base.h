#pragma once

#ifndef NUMERIC_OPS_BASE_H
#define NUMERIC_OPS_BASE_H

#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base operation functors with IEEE compliance checking
     */
    namespace ops {

        /**
         * @brief Base class for unary operations
         */
        template<typename T>
        struct UnaryOp {
            using value_type = T;
            using result_type = T;

            // Check for IEEE compliance if enabled
            void check_input(T val) const {
                if (NumericOptions::defaults().check_finite) {
                    if (IEEEComplianceChecker::is_nan(val)) {
                        // NaN propagation is allowed
                    }
                }
            }
        };

        /**
         * @brief Base class for binary operations
         */
        template<typename T>
        struct BinaryOp {
            using value_type = T;
            using result_type = T;

            void check_inputs(T lhs, T rhs) const {
                if (NumericOptions::defaults().check_finite) {
                    if constexpr (std::is_floating_point_v<T>) {
                        if (IEEEComplianceChecker::is_nan(lhs) ||
                            IEEEComplianceChecker::is_nan(rhs)) {
                            // NaN propagation is allowed
                        }
                    }
                }
            }
        };

        template<>
        struct BinaryOp<void> {
            using value_type = void;
            using result_type = void;

            template<typename L, typename R>
            void check_inputs(const L& lhs, const R& rhs) const {
                if (NumericOptions::defaults().check_finite) {
                    if constexpr (std::is_floating_point_v<std::decay_t<L>>) {
                        if (IEEEComplianceChecker::is_nan(lhs)) {
                            // NaN propagation is allowed
                        }
                    }
                    if constexpr (std::is_floating_point_v<std::decay_t<R>>) {
                        if (IEEEComplianceChecker::is_nan(rhs)) {
                            // NaN propagation is allowed
                        }
                    }
                }
            }
        };

        /**
         * @brief Base class for reduction operations
         */
        template<typename T>
        struct ReductionOp {
            using value_type = T;
            using result_type = T;

            void check_range() const {
                // Range validation if needed
            }
        };

        // ============================================================
        // Arithmetic operations
        // ============================================================

        template<typename T = void>
        struct plus : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                return lhs + rhs;
            }
        };

        template<>
        struct plus<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) + std::forward<U>(rhs)) {
                using result_type = decltype(std::forward<T>(lhs) + std::forward<U>(rhs));
                if constexpr (std::is_integral_v<std::decay_t<T>> && std::is_floating_point_v<result_type>) {
                    return static_cast<result_type>(lhs) + std::forward<U>(rhs);
                } else if constexpr (std::is_integral_v<std::decay_t<U>> && std::is_floating_point_v<result_type>) {
                    return std::forward<T>(lhs) + static_cast<result_type>(rhs);
                } else {
                    return std::forward<T>(lhs) + std::forward<U>(rhs);
                }
            }
        };

        template<typename T = void>
        struct minus : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                return lhs - rhs;
            }
        };

        template<>
        struct minus<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) - std::forward<U>(rhs)) {
                return std::forward<T>(lhs) - std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct multiplies : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                return lhs * rhs;
            }
        };

        template<>
        struct multiplies<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) * std::forward<U>(rhs)) {
                return std::forward<T>(lhs) * std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct divides : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                if (rhs == T{0}) {
                    // IEEE 754: division by zero produces ±inf or NaN
                    if constexpr (std::is_floating_point_v<T>) {
                        return lhs / rhs;  // Let IEEE handle it
                    } else {
                        throw ComputationError("Division by zero");
                    }
                }
                return lhs / rhs;
            }
        };

        template<>
        struct divides<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) / std::forward<U>(rhs)) {
                return std::forward<T>(lhs) / std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct modulus : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                if (rhs == T{0}) {
                    throw ComputationError("Modulo by zero");
                }
                if constexpr (std::is_floating_point_v<T>) {
                    return std::fmod(lhs, rhs);
                } else {
                    return lhs % rhs;
                }
            }
        };

        template<>
        struct modulus<void> {
            template<typename T, typename U>
            auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) % std::forward<U>(rhs)) {
                return std::forward<T>(lhs) % std::forward<U>(rhs);
            }
        };

        // Alias for backward compatibility
        template<typename T = void>
        using mod_op = modulus<T>;

        // ============================================================
        // Assignment operations (for in-place modifications)
        // ============================================================

        template<typename T = void>
        struct plus_assign : BinaryOp<T> {
            template<typename L, typename R>
            constexpr auto operator()(L&& lhs, R&& rhs) const -> decltype(lhs) {
                this->check_inputs(lhs, rhs);
                lhs += std::forward<R>(rhs);
                return lhs;
            }
        };

        template<typename T = void>
        struct minus_assign : BinaryOp<T> {
            template<typename L, typename R>
            constexpr auto operator()(L&& lhs, R&& rhs) const -> decltype(lhs) {
                this->check_inputs(lhs, rhs);
                lhs -= std::forward<R>(rhs);
                return lhs;
            }
        };

        template<typename T = void>
        struct multiplies_assign : BinaryOp<T> {
            template<typename L, typename R>
            constexpr auto operator()(L&& lhs, R&& rhs) const -> decltype(lhs) {
                this->check_inputs(lhs, rhs);
                lhs *= std::forward<R>(rhs);
                return lhs;
            }
        };

        template<typename T = void>
        struct divides_assign : BinaryOp<T> {
            template<typename L, typename R>
            constexpr auto operator()(L&& lhs, R&& rhs) const -> decltype(lhs) {
                this->check_inputs(lhs, rhs);
                using RVal = std::decay_t<R>;
                if (rhs == RVal{0}) {
                    if constexpr (std::is_floating_point_v<RVal>) {
                        lhs /= std::forward<R>(rhs);  // IEEE handles div by zero
                        return lhs;
                    } else {
                        throw ComputationError("Division by zero");
                    }
                }
                lhs /= std::forward<R>(rhs);
                return lhs;
            }
        };

        template<typename T = void>
        struct modulus_assign : BinaryOp<T> {
            template<typename L, typename R>
            constexpr auto operator()(L&& lhs, R&& rhs) const -> decltype(lhs) {
                this->check_inputs(lhs, rhs);
                using RVal = std::decay_t<R>;
                if (rhs == RVal{0}) {
                    throw ComputationError("Modulo by zero");
                }
                if constexpr (std::is_floating_point_v<std::decay_t<L>> ||
                              std::is_floating_point_v<RVal>) {
                    lhs = std::fmod(lhs, rhs);
                } else {
                    lhs %= std::forward<R>(rhs);
                }
                return lhs;
            }
        };

        // ============================================================
        // Unary operations
        // ============================================================

        template<typename T = void>
        struct negate : UnaryOp<T> {
            constexpr T operator()(const T& val) const {
                this->check_input(val);
                return -val;
            }
        };

        template<>
        struct negate<void> {
            template<typename T>
            constexpr auto operator()(T&& val) const
            -> decltype(-std::forward<T>(val)) {
                return -std::forward<T>(val);
            }
        };

        template<typename T = void>
        struct abs_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_unsigned_v<T>) {
                    return val;
                } else {
                    return std::abs(val);
                }
            }
        };

        template<>
        struct abs_op<void> {
            template<typename T>
            auto operator()(T&& val) const
            -> decltype(std::abs(std::forward<T>(val))) {
                return std::abs(std::forward<T>(val));
            }
        };

        template<typename T = void>
        struct sign_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if (val > T{0}) return T{1};
                if (val < T{0}) return T{-1};
                return T{0};
            }
        };

        template<>
        struct sign_op<void> {
            template<typename T>
            auto operator()(T&& val) const {
                using value_type = std::decay_t<T>;
                if (val > value_type{0}) return value_type{1};
                if (val < value_type{0}) return value_type{-1};
                return value_type{0};
            }
        };

        template<typename T = void>
        struct sqrt_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    // IEEE 754: sqrt of negative produces NaN
                    return std::sqrt(val);
                } else {
                    if (val < T{0}) {
                        throw ComputationError("Square root of negative number");
                    }
                    return static_cast<T>(std::sqrt(static_cast<double>(val)));
                }
            }
        };

        // ============================================================
        // Transcendental functions
        // ============================================================

        template<typename T = void>
        struct sin_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::sin(val);
                } else {
                    return static_cast<T>(std::sin(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct cos_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::cos(val);
                } else {
                    return static_cast<T>(std::cos(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct tan_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::tan(val);
                } else {
                    return static_cast<T>(std::tan(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct asin_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::asin(val);
                } else {
                    if (val < T{-1} || val > T{1}) {
                        throw ComputationError("asin domain error");
                    }
                    return static_cast<T>(std::asin(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct acos_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::acos(val);
                } else {
                    if (val < T{-1} || val > T{1}) {
                        throw ComputationError("acos domain error");
                    }
                    return static_cast<T>(std::acos(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct atan_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::atan(val);
                } else {
                    return static_cast<T>(std::atan(static_cast<double>(val)));
                }
            }
        };

        // Hyperbolic functions
        template<typename T = void>
        struct sinh_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::sinh(val);
                } else {
                    return static_cast<T>(std::sinh(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct cosh_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::cosh(val);
                } else {
                    return static_cast<T>(std::cosh(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct tanh_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::tanh(val);
                } else {
                    return static_cast<T>(std::tanh(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct exp_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::exp(val);
                } else {
                    return static_cast<T>(std::exp(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct log_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if (val <= T{0}) {
                    if constexpr (std::is_floating_point_v<T>) {
                        // IEEE 754: log of negative produces NaN, log(0) produces -inf
                        return std::log(val);
                    } else {
                        throw ComputationError("Logarithm of non-positive number");
                    }
                }
                if constexpr (std::is_floating_point_v<T>) {
                    return std::log(val);
                } else {
                    return static_cast<T>(std::log(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct log10_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if (val <= T{0}) {
                    if constexpr (std::is_floating_point_v<T>) {
                        return std::log10(val);
                    } else {
                        throw ComputationError("Log10 of non-positive number");
                    }
                }
                if constexpr (std::is_floating_point_v<T>) {
                    return std::log10(val);
                } else {
                    return static_cast<T>(std::log10(static_cast<double>(val)));
                }
            }
        };

        template<typename T = void>
        struct log2_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if (val <= T{0}) {
                    if constexpr (std::is_floating_point_v<T>) {
                        return std::log2(val);
                    } else {
                        throw ComputationError("Log2 of non-positive number");
                    }
                }
                if constexpr (std::is_floating_point_v<T>) {
                    return std::log2(val);
                } else {
                    return static_cast<T>(std::log2(static_cast<double>(val)));
                }
            }
        };

        // ============================================================
        // Comparison operations with IEEE 754 compliance
        // ============================================================

        template<typename T = void>
        struct equal_to : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                // IEEE 754: NaN != NaN
                if constexpr (std::is_floating_point_v<T>) {
                    if (IEEEComplianceChecker::is_nan(lhs) ||
                        IEEEComplianceChecker::is_nan(rhs)) {
                        return false;
                    }
                }
                return lhs == rhs;
            }
        };

        template<>
        struct equal_to<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) == std::forward<U>(rhs)) {
                return std::forward<T>(lhs) == std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct not_equal_to : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                return !equal_to<T>{}(lhs, rhs);
            }
        };

        template<>
        struct not_equal_to<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) != std::forward<U>(rhs)) {
                return std::forward<T>(lhs) != std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct less : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                // IEEE 754: comparisons with NaN return false
                if constexpr (std::is_floating_point_v<T>) {
                    if (IEEEComplianceChecker::is_nan(lhs) ||
                        IEEEComplianceChecker::is_nan(rhs)) {
                        return false;
                    }
                }
                return lhs < rhs;
            }
        };

        template<>
        struct less<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) < std::forward<U>(rhs)) {
                return std::forward<T>(lhs) < std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct greater : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                return less<T>{}(rhs, lhs);
            }
        };

        template<>
        struct greater<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) > std::forward<U>(rhs)) {
                return std::forward<T>(lhs) > std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct less_equal : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                return !less<T>{}(rhs, lhs);
            }
        };

        template<>
        struct less_equal<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) <= std::forward<U>(rhs)) {
                return std::forward<T>(lhs) <= std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct greater_equal : BinaryOp<bool> {
            bool operator()(const T& lhs, const T& rhs) const {
                return !less<T>{}(lhs, rhs);
            }
        };

        template<>
        struct greater_equal<void> {
            template<typename T, typename U>
            constexpr auto operator()(T&& lhs, U&& rhs) const
            -> decltype(std::forward<T>(lhs) >= std::forward<U>(rhs)) {
                return std::forward<T>(lhs) >= std::forward<U>(rhs);
            }
        };

        // ============================================================
        // Special operations
        // ============================================================

        template<typename T>
        struct power_op : BinaryOp<T> {
            T operator()(const T& base, const T& exp) const {
                this->check_inputs(base, exp);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::pow(base, exp);
                } else {
                    return static_cast<T>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
                }
            }
        };

        // Alias for consistency
        template<typename T>
        using pow_op = power_op<T>;

        template<typename T>
        struct atan2_op : BinaryOp<T> {
            T operator()(const T& y, const T& x) const {
                this->check_inputs(y, x);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::atan2(y, x);
                } else {
                    return static_cast<T>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
                }
            }
        };

        template<typename T>
        struct hypot_op : BinaryOp<T> {
            T operator()(const T& x, const T& y) const {
                this->check_inputs(x, y);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::hypot(x, y);
                } else {
                    return static_cast<T>(std::hypot(static_cast<double>(x), static_cast<double>(y)));
                }
            }
        };

        // Min/max operations with IEEE compliance
        template<typename T>
        struct min_op : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                // IEEE 754: min(NaN, x) = min(x, NaN) = x
                if constexpr (std::is_floating_point_v<T>) {
                    if (IEEEComplianceChecker::is_nan(lhs)) return rhs;
                    if (IEEEComplianceChecker::is_nan(rhs)) return lhs;
                }
                return std::min(lhs, rhs);
            }
        };

        template<typename T>
        struct max_op : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                // IEEE 754: max(NaN, x) = max(x, NaN) = x
                if constexpr (std::is_floating_point_v<T>) {
                    if (IEEEComplianceChecker::is_nan(lhs)) return rhs;
                    if (IEEEComplianceChecker::is_nan(rhs)) return lhs;
                }
                return std::max(lhs, rhs);
            }
        };

        // ============================================================
        // Rounding operations
        // ============================================================

        template<typename T>
        struct round_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::round(val);
                } else {
                    return val;
                }
            }
        };

        template<typename T>
        struct floor_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::floor(val);
                } else {
                    return val;
                }
            }
        };

        template<typename T>
        struct ceil_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::ceil(val);
                } else {
                    return val;
                }
            }
        };

        template<typename T>
        struct trunc_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                if constexpr (std::is_floating_point_v<T>) {
                    return std::trunc(val);
                } else {
                    return val;
                }
            }
        };

        // ============================================================
        // Logical operations
        // ============================================================

        struct logical_and {
            template<typename T, typename U>
            bool operator()(const T& lhs, const U& rhs) const {
                return static_cast<bool>(lhs) && static_cast<bool>(rhs);
            }
        };

        struct logical_or {
            template<typename T, typename U>
            bool operator()(const T& lhs, const U& rhs) const {
                return static_cast<bool>(lhs) || static_cast<bool>(rhs);
            }
        };

        struct logical_not {
            template<typename T>
            bool operator()(const T& val) const {
                return !static_cast<bool>(val);
            }
        };

        struct logical_xor {
            template<typename T, typename U>
            bool operator()(const T& lhs, const U& rhs) const {
                return static_cast<bool>(lhs) != static_cast<bool>(rhs);
            }
        };

        // ============================================================
        // Bitwise operations (for integral types)
        // ============================================================

        template<typename T = void>
        struct bit_and : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return lhs & rhs;
            }
        };

        template<typename T = void>
        struct bit_or : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return lhs | rhs;
            }
        };

        template<typename T = void>
        struct bit_xor : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return lhs ^ rhs;
            }
        };

        template<typename T = void>
        struct bit_not : UnaryOp<T> {
            T operator()(const T& val) const {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return ~val;
            }
        };

        template<typename T = void>
        struct left_shift : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                static_assert(std::is_integral_v<T>, "Shift operations require integral types");
                return lhs << rhs;
            }
        };

        template<typename T = void>
        struct right_shift : BinaryOp<T> {
            T operator()(const T& lhs, const T& rhs) const {
                static_assert(std::is_integral_v<T>, "Shift operations require integral types");
                return lhs >> rhs;
            }
        };

        // ============================================================
        // Reduction operations
        // ============================================================

        template<typename T = void>
        struct sum_op : ReductionOp<T> {
            template<typename Iterator>
            T operator()(Iterator first, Iterator last) const {
                this->check_range();
                return std::accumulate(first, last, T{0});
            }

            template<typename Container>
            T operator()(const Container& c) const {
                return (*this)(std::begin(c), std::end(c));
            }
        };

        template<typename T = void>
        struct product_op : ReductionOp<T> {
            template<typename Iterator>
            T operator()(Iterator first, Iterator last) const {
                this->check_range();
                return std::accumulate(first, last, T{1}, std::multiplies<T>{});
            }

            template<typename Container>
            T operator()(const Container& c) const {
                return (*this)(std::begin(c), std::end(c));
            }
        };

        template<typename T = void>
        struct mean_op : ReductionOp<T> {
            template<typename Iterator>
            T operator()(Iterator first, Iterator last) const {
                this->check_range();
                auto n = std::distance(first, last);
                if (n == 0) {
                    throw ComputationError("Mean of empty sequence");
                }
                T sum = std::accumulate(first, last, T{0});
                return sum / static_cast<T>(n);
            }

            template<typename Container>
            T operator()(const Container& c) const {
                return (*this)(std::begin(c), std::end(c));
            }
        };

        template<typename T = void>
        struct variance_op : ReductionOp<T> {
            template<typename Iterator>
            T operator()(Iterator first, Iterator last, bool sample = false) const {
                this->check_range();
                auto n = std::distance(first, last);
                if (n == 0) {
                    throw ComputationError("Variance of empty sequence");
                }
                if (sample && n == 1) {
                    throw ComputationError("Sample variance requires at least 2 elements");
                }

                T mean = mean_op<T>{}(first, last);
                T sum_sq = T{0};

                for (auto it = first; it != last; ++it) {
                    T diff = *it - mean;
                    sum_sq += diff * diff;
                }

                return sum_sq / static_cast<T>(sample ? n - 1 : n);
            }

            template<typename Container>
            T operator()(const Container& c, bool sample = false) const {
                return (*this)(std::begin(c), std::end(c), sample);
            }
        };

        template<typename T = void>
        struct stddev_op : ReductionOp<T> {
            template<typename Iterator>
            T operator()(Iterator first, Iterator last, bool sample = false) const {
                T var = variance_op<T>{}(first, last, sample);
                return std::sqrt(var);
            }

            template<typename Container>
            T operator()(const Container& c, bool sample = false) const {
                return (*this)(std::begin(c), std::end(c), sample);
            }
        };

    } // namespace ops

    /**
     * @brief Operation dispatcher for runtime selection
     */
    class OperationDispatcher {
    public:
        enum class OpType {
            // Binary arithmetic
            ADD, SUB, MUL, DIV, POW, MOD,
            // Binary special
            MIN, MAX, ATAN2, HYPOT,
            // Comparisons
            EQ, NE, LT, LE, GT, GE,
            // Unary basic
            NEG, ABS, SIGN, SQRT,
            // Transcendental
            EXP, LOG, LOG10, LOG2,
            SIN, COS, TAN,
            ASIN, ACOS, ATAN,
            SINH, COSH, TANH,
            // Rounding
            ROUND, FLOOR, CEIL, TRUNC,
            // Logical
            AND, OR, NOT, XOR,
            // Bitwise
            BIT_AND, BIT_OR, BIT_XOR, BIT_NOT,
            LEFT_SHIFT, RIGHT_SHIFT,
            // Reductions
            SUM, PRODUCT, MEAN, VARIANCE, STDDEV
        };

        template<typename T>
        static std::function<T(T, T)> get_binary_op(OpType type) {
            switch (type) {
                case OpType::ADD: return ops::plus<T>{};
                case OpType::SUB: return ops::minus<T>{};
                case OpType::MUL: return ops::multiplies<T>{};
                case OpType::DIV: return ops::divides<T>{};
                case OpType::POW: return ops::power_op<T>{};
                case OpType::MOD: return ops::modulus<T>{};
                case OpType::MIN: return ops::min_op<T>{};
                case OpType::MAX: return ops::max_op<T>{};
                case OpType::ATAN2: return ops::atan2_op<T>{};
                case OpType::HYPOT: return ops::hypot_op<T>{};
                default:
                    throw std::invalid_argument("Invalid binary operation type");
            }
        }

        template<typename T>
        static std::function<T(T)> get_unary_op(OpType type) {
            switch (type) {
                case OpType::NEG: return ops::negate<T>{};
                case OpType::ABS: return ops::abs_op<T>{};
                case OpType::SIGN: return ops::sign_op<T>{};
                case OpType::SQRT: return ops::sqrt_op<T>{};
                case OpType::EXP: return ops::exp_op<T>{};
                case OpType::LOG: return ops::log_op<T>{};
                case OpType::LOG10: return ops::log10_op<T>{};
                case OpType::LOG2: return ops::log2_op<T>{};
                case OpType::SIN: return ops::sin_op<T>{};
                case OpType::COS: return ops::cos_op<T>{};
                case OpType::TAN: return ops::tan_op<T>{};
                case OpType::ASIN: return ops::asin_op<T>{};
                case OpType::ACOS: return ops::acos_op<T>{};
                case OpType::ATAN: return ops::atan_op<T>{};
                case OpType::SINH: return ops::sinh_op<T>{};
                case OpType::COSH: return ops::cosh_op<T>{};
                case OpType::TANH: return ops::tanh_op<T>{};
                case OpType::ROUND: return ops::round_op<T>{};
                case OpType::FLOOR: return ops::floor_op<T>{};
                case OpType::CEIL: return ops::ceil_op<T>{};
                case OpType::TRUNC: return ops::trunc_op<T>{};
                default:
                    throw std::invalid_argument("Invalid unary operation type");
            }
        }

        template<typename T>
        static std::function<bool(T, T)> get_comparison_op(OpType type) {
            switch (type) {
                case OpType::EQ: return ops::equal_to<T>{};
                case OpType::NE: return ops::not_equal_to<T>{};
                case OpType::LT: return ops::less<T>{};
                case OpType::LE: return ops::less_equal<T>{};
                case OpType::GT: return ops::greater<T>{};
                case OpType::GE: return ops::greater_equal<T>{};
                default:
                    throw std::invalid_argument("Invalid comparison operation type");
            }
        }

        // Helper to determine if operation is unary
        static bool is_unary(OpType type) {
            return type == OpType::NEG || type == OpType::ABS ||
                   type == OpType::SIGN || type == OpType::SQRT ||
                   type == OpType::EXP || type == OpType::LOG ||
                   type == OpType::LOG10 || type == OpType::LOG2 ||
                   type == OpType::SIN || type == OpType::COS ||
                   type == OpType::TAN || type == OpType::ASIN ||
                   type == OpType::ACOS || type == OpType::ATAN ||
                   type == OpType::SINH || type == OpType::COSH ||
                   type == OpType::TANH || type == OpType::ROUND ||
                   type == OpType::FLOOR || type == OpType::CEIL ||
                   type == OpType::TRUNC || type == OpType::BIT_NOT;
        }

        // Helper to determine if operation is a reduction
        static bool is_reduction(OpType type) {
            return type == OpType::SUM || type == OpType::PRODUCT ||
                   type == OpType::MEAN || type == OpType::VARIANCE ||
                   type == OpType::STDDEV;
        }

        // Helper to determine if operation is a comparison
        static bool is_comparison(OpType type) {
            return type == OpType::EQ || type == OpType::NE ||
                   type == OpType::LT || type == OpType::LE ||
                   type == OpType::GT || type == OpType::GE;
        }
    };

} // namespace fem::numeric

#endif //NUMERIC_OPS_BASE_H
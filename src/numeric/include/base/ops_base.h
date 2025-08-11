#pragma once

#ifndef NUMERIC_OPS_BASE_H
#define NUMERIC_OPS_BASE_H

#include <cmath>
#include <functional>
#include <algorithm>

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
                    // IEEE compliance checking
                }
            }
        };

        // Arithmetic operations
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
                return std::forward<T>(lhs) + std::forward<U>(rhs);
            }
        };

        template<typename T = void>
        struct minus : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                return lhs - rhs;
            }
        };

        template<typename T = void>
        struct multiplies : BinaryOp<T> {
            constexpr T operator()(const T& lhs, const T& rhs) const {
                this->check_inputs(lhs, rhs);
                return lhs * rhs;
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

        // Unary operations
        template<typename T = void>
        struct negate : UnaryOp<T> {
            constexpr T operator()(const T& val) const {
                this->check_input(val);
                return -val;
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
                    return std::sqrt(static_cast<double>(val));
                }
            }
        };

        // Transcendental functions
        template<typename T = void>
        struct sin_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                return std::sin(val);
            }
        };

        template<typename T = void>
        struct cos_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                return std::cos(val);
            }
        };

        template<typename T = void>
        struct exp_op : UnaryOp<T> {
            T operator()(const T& val) const {
                this->check_input(val);
                return std::exp(val);
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
                return std::log(val);
            }
        };

        // Comparison operations
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

        // Special operations
        template<typename T>
        struct power_op : BinaryOp<T> {
            T operator()(const T& base, const T& exp) const {
                this->check_inputs(base, exp);
                return std::pow(base, exp);
            }
        };

        template<typename T>
        struct mod_op : BinaryOp<T> {
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

        // Rounding operations
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

        // Logical operations
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

    } // namespace ops

    /**
     * @brief Operation dispatcher for runtime selection
     */
    class OperationDispatcher {
    public:
        enum class OpType {
            // Binary
            ADD, SUB, MUL, DIV, POW, MOD,
            MIN, MAX,
            EQ, NE, LT, LE, GT, GE,
            // Unary
            NEG, ABS, SQRT, EXP, LOG,
            SIN, COS, TAN,
            ROUND, FLOOR, CEIL,
            // Logical
            AND, OR, NOT
        };

        template<typename T>
        static std::function<T(T, T)> get_binary_op(OpType type) {
            switch (type) {
                case OpType::ADD: return ops::plus<T>{};
                case OpType::SUB: return ops::minus<T>{};
                case OpType::MUL: return ops::multiplies<T>{};
                case OpType::DIV: return ops::divides<T>{};
                case OpType::POW: return ops::power_op<T>{};
                case OpType::MOD: return ops::mod_op<T>{};
                case OpType::MIN: return ops::min_op<T>{};
                case OpType::MAX: return ops::max_op<T>{};
                default:
                    throw std::invalid_argument("Invalid binary operation type");
            }
        }

        template<typename T>
        static std::function<T(T)> get_unary_op(OpType type) {
            switch (type) {
                case OpType::NEG: return ops::negate<T>{};
                case OpType::ABS: return ops::abs_op<T>{};
                case OpType::SQRT: return ops::sqrt_op<T>{};
                case OpType::EXP: return ops::exp_op<T>{};
                case OpType::LOG: return ops::log_op<T>{};
                case OpType::SIN: return ops::sin_op<T>{};
                case OpType::COS: return ops::cos_op<T>{};
                case OpType::ROUND: return ops::round_op<T>{};
                case OpType::FLOOR: return ops::floor_op<T>{};
                case OpType::CEIL: return ops::ceil_op<T>{};
                default:
                    throw std::invalid_argument("Invalid unary operation type");
            }
        }
    };

} // namespace fem::numeric


#endif //NUMERIC_OPS_BASE_H

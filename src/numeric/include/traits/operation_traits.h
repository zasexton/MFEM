#pragma once

#ifndef NUMERIC_OPERATION_TRAITS_H
#define NUMERIC_OPERATION_TRAITS_H

#include <type_traits>
#include <functional>
#include <cmath>
#include <tuple>
#include <limits>

#include "../base/ops_base.h"
#include "../base/numeric_base.h"
#include "../base/traits_base.h"

#include "type_traits.h"
#include "numeric_traits.h"
#include "SFINAE.h"

namespace fem::numeric::traits {

    /**
     * @brief Operation categories for dispatch
     */
    enum class OperationCategory {
        Arithmetic,     // +, -, *, /
        Comparison,     // <, >, ==, !=, <=, >=
        Logical,        // &&, ||, !
        Bitwise,        // &, |, ^, ~, <<, >>
        Transcendental, // sin, cos, exp, log, etc.
        Reduction,      // sum, product, min, max
        LinearAlgebra,  // dot, matmul, solve
        Special,        // erf, gamma, bessel, etc.
        Assignment,     // =, +=, -=, *=, /=
        ElementWise,    // Operations that work element-by-element
        Unknown
    };

    /**
     * @brief Algebraic properties of operations
     */
    struct AlgebraicProperties {
        bool commutative = false;      // a op b = b op a
        bool associative = false;       // (a op b) op c = a op (b op c)
        bool distributive = false;      // a * (b + c) = a*b + a*c
        bool has_identity = false;      // exists e: a op e = a
        bool has_inverse = false;       // exists inv(a): a op inv(a) = e
        bool idempotent = false;        // a op a = a
        bool nilpotent = false;         // exists n: a^n = 0
        bool involutive = false;        // f(f(a)) = a
    };

    // Forward declarations
    template<typename Op>
    struct operation_category;

    template<typename Op>
    inline constexpr OperationCategory operation_category_v = operation_category<Op>::value;

    template<typename Op>
    struct is_reduction_operation;

    template<typename Op>
    inline constexpr bool is_reduction_operation_v = is_reduction_operation<Op>::value;

    template<typename Op>
    struct is_elementwise_operation;

    template<typename Op>
    inline constexpr bool is_elementwise_operation_v = is_elementwise_operation<Op>::value;

    /**
     * @brief Detect if operation is in-place (modifies first argument)
     */
    template<typename Op>
    struct is_inplace_operation : std::false_type {};

    // Specializations for in-place operations
    template<> struct is_inplace_operation<ops::plus_assign<>> : std::true_type {};
    template<> struct is_inplace_operation<ops::minus_assign<>> : std::true_type {};
    template<> struct is_inplace_operation<ops::multiplies_assign<>> : std::true_type {};
    template<> struct is_inplace_operation<ops::divides_assign<>> : std::true_type {};
    template<> struct is_inplace_operation<ops::modulus_assign<>> : std::true_type {};

    template<typename Op>
    inline constexpr bool is_inplace_operation_v = is_inplace_operation<Op>::value;

    /**
     * @brief Check if operation is valid for given types
     */
    template<typename Op, typename... Args>
    struct is_valid_operation {
        static constexpr bool value = std::is_invocable_v<Op, Args...>;
    };

    template<typename Op, typename... Args>
    inline constexpr bool is_valid_operation_v = is_valid_operation<Op, Args...>::value;

    /**
     * @brief Detect if operation is element-wise (for arrays/matrices)
     */
    template<typename Op>
    struct is_elementwise_operation {
        static constexpr bool value =
            operation_category_v<Op> == OperationCategory::Arithmetic ||
            operation_category_v<Op> == OperationCategory::Transcendental ||
            operation_category_v<Op> == OperationCategory::Comparison ||
            std::is_same_v<Op, ops::abs_op<>> ||
            std::is_same_v<Op, ops::sign_op<>>;
    };

    /**
     * @brief Detect if operation requires special domain handling
     */
    template<typename Op>
    struct has_domain_restrictions {
        static constexpr bool value =
                std::is_same_v<Op, ops::sqrt_op<>> ||    // x >= 0
                std::is_same_v<Op, ops::log_op<>> ||     // x > 0
                std::is_same_v<Op, ops::asin_op<>> ||    // |x| <= 1
                std::is_same_v<Op, ops::acos_op<>> ||    // |x| <= 1
                std::is_same_v<Op, ops::tanh_op<>>;      // Changed from atanh_op
    };

    template<typename Op>
    inline constexpr bool has_domain_restrictions_v = has_domain_restrictions<Op>::value;

    /**
     * @brief Get the corresponding in-place operation
     */
    template<typename Op>
    struct make_inplace {
        using type = void;
    };

    template<> struct make_inplace<ops::plus<>> { using type = ops::plus_assign<>; };
    template<> struct make_inplace<ops::minus<>> { using type = ops::minus_assign<>; };
    template<> struct make_inplace<ops::multiplies<>> { using type = ops::multiplies_assign<>; };
    template<> struct make_inplace<ops::divides<>> { using type = ops::divides_assign<>; };

    template<typename Op>
    using make_inplace_t = typename make_inplace<Op>::type;

    /**
     * @brief Get the inverse operation
     */
    template<typename Op>
    struct inverse_operation {
        using type = void;
    };

    template<> struct inverse_operation<ops::plus<>> { using type = ops::minus<>; };
    template<> struct inverse_operation<ops::minus<>> { using type = ops::plus<>; };
    template<> struct inverse_operation<ops::multiplies<>> { using type = ops::divides<>; };
    template<> struct inverse_operation<ops::divides<>> { using type = ops::multiplies<>; };
    template<> struct inverse_operation<ops::exp_op<>> { using type = ops::log_op<>; };
    template<> struct inverse_operation<ops::log_op<>> { using type = ops::exp_op<>; };
    template<> struct inverse_operation<ops::sin_op<>> { using type = ops::asin_op<>; };
    template<> struct inverse_operation<ops::cos_op<>> { using type = ops::acos_op<>; };
    template<> struct inverse_operation<ops::tan_op<>> { using type = ops::atan_op<>; };

    template<typename Op>
    using inverse_operation_t = typename inverse_operation<Op>::type;

    /**
     * @brief Detect if operation is unary
     */
    template<typename Op, typename T = double>
    struct is_unary_operation {
        static constexpr bool value =
                std::is_invocable_v<Op, T> &&
                !std::is_invocable_v<Op, T, T>;
    };

    template<typename Op, typename T = double>
    inline constexpr bool is_unary_operation_v = is_unary_operation<Op, T>::value;

    /**
     * @brief Detect if operation is binary
     */
    template<typename Op, typename T = double>
    struct is_binary_operation {
        static constexpr bool value = std::is_invocable_v<Op, T, T>;
    };

    template<typename Op, typename T = double>
    inline constexpr bool is_binary_operation_v = is_binary_operation<Op, T>::value;

    /**
     * @brief Get operation result type
     */
    template<typename Op, typename... Args>
    struct operation_result {
        using type = std::invoke_result_t<Op, Args...>;
    };

    template<typename Op, typename... Args>
    using operation_result_t = typename operation_result<Op, Args...>::type;

    // Helper to compute result type based on operation arity
    template<bool IsUnary, bool IsBinary, typename Op, typename T>
    struct operation_result_helper { using type = void; };

    template<bool IsBinary, typename Op, typename T>
    struct operation_result_helper<true, IsBinary, Op, T> {
        using type = operation_result_t<Op, T>;
    };

    template<typename Op, typename T>
    struct operation_result_helper<false, true, Op, T> {
        using type = operation_result_t<Op, T, T>;
    };

    /**
     * @brief Operation category detection
     */
    template<typename Op>
    struct operation_category {
        static constexpr OperationCategory value = [] {
            // Check arithmetic operations
            if constexpr (is_specialization_v<Op, ops::plus> ||
                          is_specialization_v<Op, ops::minus> ||
                          is_specialization_v<Op, ops::multiplies> ||
                          is_specialization_v<Op, ops::divides> ||
                          std::is_same_v<Op, ops::modulus<>> ||
                          std::is_same_v<Op, ops::negate<>>) {
                return OperationCategory::Arithmetic;
            }
            // Check comparison operations
            else if constexpr (std::is_same_v<Op, ops::less<>> ||
                               std::is_same_v<Op, ops::greater<>> ||
                               std::is_same_v<Op, ops::less_equal<>> ||
                               std::is_same_v<Op, ops::greater_equal<>> ||
                               std::is_same_v<Op, ops::equal_to<>> ||
                               std::is_same_v<Op, ops::not_equal_to<>>) {
                return OperationCategory::Comparison;
            }
            // Check transcendental operations - Fixed template parameters
            else if constexpr (std::is_same_v<Op, ops::sin_op<>> ||
                               std::is_same_v<Op, ops::cos_op<>> ||
                               std::is_same_v<Op, ops::tan_op<>> ||
                               std::is_same_v<Op, ops::exp_op<>> ||
                               std::is_same_v<Op, ops::log_op<>> ||
                               std::is_same_v<Op, ops::sqrt_op<>>) {
                return OperationCategory::Transcendental;
            }
            // Handle power_op separately as it requires a template parameter
            else if constexpr (requires { typename Op::result_type; }) {
                // Check if it's a power operation by other means
                return OperationCategory::Transcendental;
            }
            else {
                return OperationCategory::Unknown;
            }
        }();
    };

    // Specialization for power_op
    template<typename T>
    struct operation_category<ops::power_op<T>> {
        static constexpr OperationCategory value = OperationCategory::Transcendental;
    };

    /**
     * @brief Get algebraic properties of operation
     */
    template<typename Op>
    struct algebraic_properties {
        static constexpr AlgebraicProperties value = [] {
            AlgebraicProperties props;

            // Addition
            if constexpr (is_specialization_v<Op, ops::plus>) {
                props.commutative = true;
                props.associative = true;
                props.has_identity = true;  // identity = 0
                props.has_inverse = true;   // inverse = negation
            }
            // Subtraction
            else if constexpr (is_specialization_v<Op, ops::minus>) {
                props.commutative = false;  // a - b ≠ b - a
                props.associative = false;  // (a - b) - c ≠ a - (b - c)
                props.has_identity = true;  // identity = 0 (for right identity: a - 0 = a)
                props.has_inverse = false;  // no general inverse
            }
            // Multiplication
            else if constexpr (is_specialization_v<Op, ops::multiplies>) {
                props.commutative = true;
                props.associative = true;
                props.distributive = true;  // over addition
                props.has_identity = true;  // identity = 1
                props.has_inverse = true;   // inverse = reciprocal (except 0)
            }
            // Division
            else if constexpr (is_specialization_v<Op, ops::divides>) {
                props.commutative = false;  // a / b ≠ b / a
                props.associative = false;  // (a / b) / c ≠ a / (b / c)
                props.has_identity = true;  // identity = 1 (for right identity: a / 1 = a)
                props.has_inverse = false;  // no general inverse operation
                props.distributive = false; // division doesn't distribute over addition
            }
            // Modulus
            else if constexpr (std::is_same_v<Op, ops::modulus<>>) {
                props.commutative = false;  // a % b ≠ b % a
                props.associative = false;  // (a % b) % c ≠ a % (b % c)
                props.has_identity = false; // no identity element
                props.has_inverse = false;  // no inverse operation
            }
            // Logical AND
            else if constexpr (std::is_same_v<Op, ops::logical_and>) {
                props.commutative = true;
                props.associative = true;
                props.idempotent = true;    // a && a = a
                props.has_identity = true;  // identity = true
                props.distributive = true;  // over OR
            }
            // Logical OR
            else if constexpr (std::is_same_v<Op, ops::logical_or>) {
                props.commutative = true;
                props.associative = true;
                props.idempotent = true;    // a || a = a
                props.has_identity = true;  // identity = false
                props.distributive = true;  // over AND
            }
            // Bitwise AND
            else if constexpr (std::is_same_v<Op, std::bit_and<>>) {
                props.commutative = true;
                props.associative = true;
                props.idempotent = true;    // a & a = a
                props.has_identity = true;  // identity = all bits set (~0)
                props.distributive = true;  // over OR
            }
            // Bitwise OR
            else if constexpr (std::is_same_v<Op, std::bit_or<>>) {
                props.commutative = true;
                props.associative = true;
                props.idempotent = true;    // a | a = a
                props.has_identity = true;  // identity = 0
                props.distributive = true;  // over AND
            }
            // Bitwise XOR
            else if constexpr (std::is_same_v<Op, std::bit_xor<>>) {
                props.commutative = true;
                props.associative = true;
                props.idempotent = false;   // a ^ a = 0, not a
                props.has_identity = true;  // identity = 0
                props.has_inverse = true;   // self-inverse: a ^ b ^ b = a
            }

            return props;
        }();
    };

    // Specializations for min/max with template parameter
    template<typename T>
    struct algebraic_properties<ops::min_op<T>> {
        static constexpr AlgebraicProperties value = [] {
            AlgebraicProperties props;
            props.commutative = true;
            props.associative = true;
            props.idempotent = true;    // min(a,a) = a
            props.has_identity = false; // no universal identity (depends on type)
            return props;
        }();
    };

    template<typename T>
    struct algebraic_properties<ops::max_op<T>> {
        static constexpr AlgebraicProperties value = [] {
            AlgebraicProperties props;
            props.commutative = true;
            props.associative = true;
            props.idempotent = true;    // max(a,a) = a
            props.has_identity = false; // no universal identity (depends on type)
            return props;
        }();
    };

    // Specialization for power_op
    template<typename T>
    struct algebraic_properties<ops::power_op<T>> {
        static constexpr AlgebraicProperties value = [] {
            AlgebraicProperties props;
            props.commutative = false;  // a^b ≠ b^a (in general)
            props.associative = false;  // (a^b)^c ≠ a^(b^c)
            props.has_identity = true;  // identity = 1 (for right identity: a^1 = a)
            props.has_inverse = false;  // logarithm is inverse but different operation
            return props;
        }();
    };

    /**
     * @brief Check if operation preserves type
     */
    template<typename Op, typename T>
    struct preserves_type {
        static constexpr bool value = [] {
            if constexpr (is_unary_operation_v<Op, T>) {
                return std::is_same_v<operation_result_t<Op, T>, T>;
            } else if constexpr (is_binary_operation_v<Op, T>) {
                return std::is_same_v<operation_result_t<Op, T, T>, T>;
            } else {
                return false;
            }
        }();
    };

    template<typename Op, typename T>
    inline constexpr bool preserves_type_v = preserves_type<Op, T>::value;

    /**
     * @brief Check if operation is IEEE-safe (handles NaN/Inf correctly)
     */
    template<typename Op, typename T>
    struct is_ieee_safe {
        static constexpr bool value = [] {
            if constexpr (!IEEECompliant<T>) {
                return true;  // Non-IEEE types are "safe" by default
            }

            // Most standard operations are IEEE-compliant
            if constexpr (operation_category_v<Op> == OperationCategory::Arithmetic ||
                          operation_category_v<Op> == OperationCategory::Transcendental) {
                return true;
            }

            // Comparisons need special handling for NaN
            if constexpr (operation_category_v<Op> == OperationCategory::Comparison) {
                return std::is_same_v<Op, ops::not_equal_to<>>;  // != is the only one that works with NaN
            }

            return false;
        }();
    };

    template<typename Op, typename T>
    inline constexpr bool is_ieee_safe_v = is_ieee_safe<Op, T>::value;

    /**
     * @brief Check if operation might overflow
     */
    template<typename Op, typename T>
    struct can_overflow {
        static constexpr bool value = [] {
            if constexpr (std::is_floating_point_v<T>) {
                // Floating point "overflows" to infinity
                if constexpr (operation_category_v<Op> == OperationCategory::Arithmetic) {
                    return true;
                }
                else if constexpr (std::is_same_v<Op, ops::exp_op<>>) {
                    return true;
                }
                return false;
            } else if constexpr (std::is_integral_v<T>) {
                // Integer overflow is undefined behavior
                return is_specialization_v<Op, ops::plus> ||
                       is_specialization_v<Op, ops::minus> ||
                       is_specialization_v<Op, ops::multiplies> ||
                       std::is_same_v<Op, ops::negate<>>;
            }
            return false;
        }();
    };

    // Specialization for power_op
    template<typename T>
    struct can_overflow<ops::power_op<T>, T> {
        static constexpr bool value = std::is_floating_point_v<T>;
    };

    template<typename Op, typename T>
    inline constexpr bool can_overflow_v = can_overflow<Op, T>::value;

    /**
     * @brief Check if operation might produce NaN
     */
    template<typename Op, typename T>
    struct can_produce_nan {
        static constexpr bool value = [] {
            if constexpr (!std::is_floating_point_v<T> && !is_complex_v<T>) {
                return false;
            }

            // Operations that can produce NaN
            return is_specialization_v<Op, ops::divides> ||    // 0/0, inf/inf
                   std::is_same_v<Op, ops::sqrt_op<>> ||     // sqrt(negative)
                   std::is_same_v<Op, ops::log_op<>> ||      // log(negative)
                   std::is_same_v<Op, ops::asin_op<>> ||     // asin(|x| > 1)
                   std::is_same_v<Op, ops::acos_op<>>;       // acos(|x| > 1)
        }();
    };

    // Specialization for power_op
    template<typename T>
    struct can_produce_nan<ops::power_op<T>, T> {
        static constexpr bool value = std::is_floating_point_v<T> || is_complex_v<T>;
    };

    template<typename Op, typename T>
    inline constexpr bool can_produce_nan_v = can_produce_nan<Op, T>::value;

    /**
     * @brief Operation complexity for optimization decisions
     */
    template<typename Op>
    struct operation_complexity {
        enum Level { Trivial, Simple, Moderate, Complex, VeryComplex };

        static constexpr Level value = [] {
            // Arithmetic operations
            if constexpr (is_specialization_v<Op, ops::plus> ||
                         is_specialization_v<Op, ops::minus>) {
                return Trivial;
            }
            else if constexpr (is_specialization_v<Op, ops::multiplies>) {
                return Simple;
            }
            else if constexpr (is_specialization_v<Op, ops::divides>) {
                return Moderate;
            }
            // Transcendental operations
            else if constexpr (std::is_same_v<Op, ops::sqrt_op<>>) {
                return Moderate;
            }
            else if constexpr (std::is_same_v<Op, ops::sin_op<>> ||
                             std::is_same_v<Op, ops::cos_op<>> ||
                             std::is_same_v<Op, ops::exp_op<>> ||
                             std::is_same_v<Op, ops::log_op<>>) {
                return Complex;
            }
            else if constexpr (std::is_same_v<Op, ops::tan_op<>>) {
                return VeryComplex;
            }
            else {
                return Simple;
            }
        }();
    };

    // Specialization for power_op
    template<typename T>
    struct operation_complexity<ops::power_op<T>> {
        enum Level { Trivial, Simple, Moderate, Complex, VeryComplex };
        static constexpr Level value = VeryComplex;
    };

    template<typename Op>
    inline constexpr auto operation_complexity_v = operation_complexity<Op>::value;

    /**
     * @brief Check if operation benefits from vectorization
     */
    template<typename Op, typename T>
    struct benefits_from_vectorization {
        static constexpr bool value =
                preserves_type_v<Op, T> &&
                (operation_complexity_v<Op> <= operation_complexity<Op>::Moderate) &&
                std::is_arithmetic_v<T> &&
                !can_produce_nan_v<Op, T>;  // NaN handling complicates vectorization
    };

    template<typename Op, typename T>
    inline constexpr bool benefits_from_vectorization_v =
            benefits_from_vectorization<Op, T>::value;

    /**
     * @brief Operation properties aggregator
     */
    template<typename Op, typename T = double>
    struct operation_properties {
        // Classification
        static constexpr bool is_unary = is_unary_operation_v<Op, T>;
        static constexpr bool is_binary = is_binary_operation_v<Op, T>;
        static constexpr OperationCategory category = operation_category_v<Op>;

        // Type behavior
        static constexpr bool preserves_type = preserves_type_v<Op, T>;
        using result_type = typename operation_result_helper<is_unary, is_binary, Op, T>::type;

        // Algebraic properties
        static constexpr auto algebra = algebraic_properties<Op>::value;

        // Safety and correctness
        static constexpr bool ieee_safe = is_ieee_safe_v<Op, T>;
        static constexpr bool might_overflow = can_overflow_v<Op, T>;
        static constexpr bool might_produce_nan = can_produce_nan_v<Op, T>;

        // Performance hints
        static constexpr auto complexity = operation_complexity_v<Op>;
        static constexpr bool vectorizable = benefits_from_vectorization_v<Op, T>;

        // Identity and inverse elements (if applicable)
        static constexpr T identity() {
            if constexpr (is_specialization_v<Op, ops::plus>) {
                return T{0};
            } else if constexpr (is_specialization_v<Op, ops::multiplies>) {
                return T{1};
            } else if constexpr (is_specialization_v<Op, ops::divides>) {
                return T{1};  // Right identity: a / 1 = a
            } else if constexpr (is_specialization_v<Op, ops::minus>) {
                return T{0};  // Right identity: a - 0 = a
            } else if constexpr (std::is_same_v<Op, std::bit_and<>>) {
                return ~T{0};  // All bits set
            } else if constexpr (std::is_same_v<Op, std::bit_or<>> ||
                                std::is_same_v<Op, std::bit_xor<>>) {
                return T{0};
            } else if constexpr (std::is_same_v<Op, ops::logical_and>) {
                return true;
            } else if constexpr (std::is_same_v<Op, ops::logical_or>) {
                return false;
            } else {
                return T{};  // No identity element
            }
        }
    };

    // Specialization for power_op identity
    template<typename T>
    struct operation_properties<ops::power_op<T>, T> {
        static constexpr bool is_unary = false;
        static constexpr bool is_binary = true;
        static constexpr OperationCategory category = OperationCategory::Transcendental;
        static constexpr bool preserves_type = preserves_type_v<ops::power_op<T>, T>;
        using result_type = T;
        static constexpr auto algebra = algebraic_properties<ops::power_op<T>>::value;
        static constexpr bool ieee_safe = is_ieee_safe_v<ops::power_op<T>, T>;
        static constexpr bool might_overflow = can_overflow_v<ops::power_op<T>, T>;
        static constexpr bool might_produce_nan = can_produce_nan_v<ops::power_op<T>, T>;
        static constexpr auto complexity = operation_complexity_v<ops::power_op<T>>;
        static constexpr bool vectorizable = benefits_from_vectorization_v<ops::power_op<T>, T>;

        static constexpr T identity() {
            return T{1};  // Right identity: a^1 = a
        }
    };

    /**
     * @brief Check if two operations can be fused
     */
    template<typename Op1, typename Op2>
    struct can_fuse_operations {
        static constexpr bool value =
                (operation_complexity_v<Op1> <= operation_complexity<Op1>::Simple) &&
                (operation_complexity_v<Op2> <= operation_complexity<Op2>::Simple);
    };

    template<typename Op1, typename Op2>
    inline constexpr bool can_fuse_operations_v = can_fuse_operations<Op1, Op2>::value;

    /**
     * @brief Determine if operation should use specialized implementation
     */
    template<typename Op, typename T>
    struct use_specialized_impl {
        // These would need actual matrix operations to be defined
        static constexpr bool use_blas = false;
        static constexpr bool use_simd = benefits_from_vectorization_v<Op, T>;
        static constexpr bool use_parallel =
                operation_complexity_v<Op> >= operation_complexity<Op>::Complex;
    };

    /**
     * @brief Error propagation analysis
     */
    template<typename Op, typename T>
    struct error_propagation {
        // Condition number estimation for the operation
        static constexpr double condition_number() {
            if constexpr (is_specialization_v<Op, ops::plus> ||
                         is_specialization_v<Op, ops::minus>) {
                return 1.0;  // Well-conditioned
            } else if constexpr (is_specialization_v<Op, ops::multiplies>) {
                return 1.0;  // Well-conditioned for moderate values
            } else if constexpr (is_specialization_v<Op, ops::divides>) {
                return 2.0;  // Can amplify errors
            } else if constexpr (std::is_same_v<Op, ops::exp_op<>>) {
                return 10.0; // Exponential error growth
            } else {
                return 1.0;
            }
        }

        static constexpr bool is_numerically_stable = condition_number() < 10.0;
    };

    /**
     * @brief Check if operation requires specific memory alignment
     */
    template<typename Op, typename T>
    struct requires_alignment {
        static constexpr size_t cache_line = 64;  // Define cache_line constant

        static constexpr size_t value = [] {
            if constexpr (benefits_from_vectorization_v<Op, T>) {
                // SIMD operations need aligned memory
                if constexpr (sizeof(T) == 4) {
                    return size_t{16};  // SSE alignment for float
                } else if constexpr (sizeof(T) == 8) {
                    return size_t{32};  // AVX alignment for double
                } else {
                    return alignof(T);
                }
            }
            return alignof(T);  // Default alignment
        }();
    };

    template<typename Op, typename T>
    inline constexpr size_t requires_alignment_v = requires_alignment<Op, T>::value;

    /**
     * @brief Detect if operation can be done in parallel without synchronization
     */
    template<typename Op>
    struct is_thread_safe {
        static constexpr bool value =
                !is_inplace_operation_v<Op> &&  // In-place ops modify state
                is_elementwise_operation_v<Op>;  // Element-wise ops are independent
    };

    template<typename Op>
    inline constexpr bool is_thread_safe_v = is_thread_safe<Op>::value;

    /**
     * @brief Memory access pattern for optimization
     */
    enum class MemoryAccessPattern {
        Sequential,      // Access elements in order
        Strided,        // Access with fixed stride
        Random,         // Random access pattern
        Broadcast,      // One element accessed multiple times
        Reduction       // Many-to-one pattern
    };

    /**
     * @brief Reduction operation traits
     */
    template<typename Op>
    struct is_reduction_operation {
        static constexpr bool value =
                std::is_same_v<Op, ops::sum_op<>> ||
                std::is_same_v<Op, ops::product_op<>> ||
                std::is_same_v<Op, ops::mean_op<>>;
    };

    // Specializations for min/max with template parameter
    template<typename T>
    struct is_reduction_operation<ops::min_op<T>> : std::true_type {};

    template<typename T>
    struct is_reduction_operation<ops::max_op<T>> : std::true_type {};

    /**
     * @brief Determine memory access pattern for operation
     */
    template<typename Op>
    struct memory_access_pattern {
        static constexpr MemoryAccessPattern value = [] {
            if constexpr (is_reduction_operation_v<Op>) {
                return MemoryAccessPattern::Reduction;
            } else if constexpr (is_elementwise_operation_v<Op>) {
                return MemoryAccessPattern::Sequential;
            } else {
                return MemoryAccessPattern::Random;
            }
        }();
    };

    template<typename Op>
    inline constexpr MemoryAccessPattern memory_access_pattern_v =
        memory_access_pattern<Op>::value;

    /**
     * @brief Batch processing hints
     */
    template<typename Op, typename T>
    struct batch_processing_hints {
        static constexpr size_t cache_line = 64;  // Typical cache line size
        static constexpr size_t l1_cache = 32768; // Typical L1 cache size

        // Optimal batch size for cache efficiency
        static constexpr size_t optimal_batch_size = [] {
            if constexpr (operation_complexity_v<Op> >= operation_complexity<Op>::Complex) {
                // Complex operations: smaller batches to fit in L1
                return l1_cache / (sizeof(T) * 4);
            } else {
                // Simple operations: larger batches
                return l1_cache / sizeof(T);
            }
        }();

        // Minimum batch size to amortize overhead
        static constexpr size_t minimum_batch_size = cache_line / sizeof(T);

        // Whether to prefetch data
        static constexpr bool use_prefetch =
                operation_complexity_v<Op> >= operation_complexity<Op>::Moderate;
    };

    /**
     * @brief Operation fusion compatibility
     */
    template<typename Op1, typename Op2>
    struct fusion_compatibility {
        // Can fuse without changing results
        static constexpr bool is_safe = [] {
            constexpr auto props1 = algebraic_properties<Op1>::value;
            constexpr auto props2 = algebraic_properties<Op2>::value;

            // Safe if both are element-wise and have compatible properties
            return is_elementwise_operation_v<Op1> &&
                   is_elementwise_operation_v<Op2> &&
                   (props1.associative || !props2.associative);
        }();

        // Expected performance gain from fusion
        static constexpr double speedup_factor = [] {
            if constexpr (is_safe) {
                // Estimate based on complexity
                auto c1 = operation_complexity_v<Op1>;
                auto c2 = operation_complexity_v<Op2>;

                if (c1 <= operation_complexity<Op1>::Simple &&
                    c2 <= operation_complexity<Op2>::Simple) {
                    return 1.5;  // Good fusion opportunity
                } else {
                    return 1.1;  // Modest gain
                }
            }
            return 1.0;  // No gain
        }();
    };

    /**
     * @brief Special value handling requirements
     */
    template<typename Op, typename T>
    struct special_value_handling {
        // Needs special handling for zero
        static constexpr bool handle_zero =
                is_specialization_v<Op, ops::divides> ||
                std::is_same_v<Op, ops::modulus<>> ||
                std::is_same_v<Op, ops::log_op<>>;

        // Needs special handling for negative values
        static constexpr bool handle_negative =
                std::is_same_v<Op, ops::sqrt_op<>> ||
                std::is_same_v<Op, ops::log_op<>>;

        // Needs special handling for infinity
        static constexpr bool handle_infinity =
                IEEECompliant<T> && (
                        operation_category_v<Op> == OperationCategory::Arithmetic ||
                        operation_category_v<Op> == OperationCategory::Transcendental
                );

        // Needs special handling for NaN
        static constexpr bool handle_nan =
                IEEECompliant<T> &&
                operation_category_v<Op> != OperationCategory::Comparison;
    };

    // Specialization for power_op special value handling
    template<typename T>
    struct special_value_handling<ops::power_op<T>, T> {
        static constexpr bool handle_zero = false;
        static constexpr bool handle_negative = std::is_integral_v<T>;
        static constexpr bool handle_infinity = IEEECompliant<T>;
        static constexpr bool handle_nan = IEEECompliant<T>;
    };

    /**
     * @brief Get initial value for reduction
     */
    template<typename Op, typename T>
    struct reduction_identity {
        static constexpr T value() {
            if constexpr (std::is_same_v<Op, ops::sum_op<>> ||
                         std::is_same_v<Op, ops::mean_op<>>) {
                return T{0};
            } else if constexpr (std::is_same_v<Op, ops::product_op<>>) {
                return T{1};
            } else {
                return T{};
            }
        }
    };

    // Specializations for min/max
    template<typename T>
    struct reduction_identity<ops::min_op<T>, T> {
        static constexpr T value() {
            return std::numeric_limits<T>::max();
        }
    };

    template<typename T>
    struct reduction_identity<ops::max_op<T>, T> {
        static constexpr T value() {
            return std::numeric_limits<T>::lowest();
        }
    };

    /**
     * @brief Accuracy and precision requirements
     */
    template<typename Op, typename T>
    struct accuracy_requirements {
        // ULP (Units in Last Place) error tolerance
        static constexpr int max_ulp_error = [] {
            if constexpr (is_specialization_v<Op, ops::plus> ||
                          is_specialization_v<Op, ops::minus>) {
                return 1;  // Exact for IEEE
            } else if constexpr (is_specialization_v<Op, ops::multiplies>) {
                return 1;  // Exact for IEEE
            } else if constexpr (is_specialization_v<Op, ops::divides>) {
                return 1;  // Correctly rounded
            } else if constexpr (std::is_same_v<Op, ops::sqrt_op<>>) {
                return 1;  // Correctly rounded
            } else if constexpr (operation_category_v<Op> == OperationCategory::Transcendental) {
                return 4;  // Typical for transcendental functions
            } else {
                return 2;
            }
        }();

        // Whether operation requires extended precision internally
        static constexpr bool needs_extended_precision =
                (std::is_same_v<Op, ops::sum_op<>> && std::is_floating_point_v<T>);
    };

    // Specialization for power_op
    template<typename T>
    struct accuracy_requirements<ops::power_op<T>, T> {
        static constexpr int max_ulp_error = 4;
        static constexpr bool needs_extended_precision = true;
    };

    enum class DispatchStrategy {
        Scalar,             // Simple scalar loop
        Vectorized,         // SIMD vectorization
        Parallel,           // Multithreaded
        ParallelVectorized, // Both parallel and SIMD
        Specialized,        // Use specialized library (BLAS, etc.)
        Lazy                // Defer evaluation
    };

    /**
     * @brief Operation dispatch strategy selector
     */
    template<typename Op, typename T, size_t Size = 0>
    struct dispatch_strategy {
        using Strategy = DispatchStrategy;

        static constexpr Strategy select() {
            // Size-based heuristics
            if constexpr (Size > 0 && Size < 16) {
                return Strategy::Scalar;  // Too small for optimization overhead
            }

            // Check for specialized implementations
            if constexpr (use_specialized_impl<Op, T>::use_blas) {
                return Strategy::Specialized;
            }

            // Check vectorization
            constexpr bool can_vectorize = benefits_from_vectorization_v<Op, T>;
            constexpr bool is_large = (Size == 0 || Size > 1000);
            constexpr bool is_complex_op =
                    operation_complexity_v<Op> >= operation_complexity<Op>::Complex;

            if constexpr (can_vectorize && is_large && is_thread_safe_v<Op>) {
                return Strategy::ParallelVectorized;
            } else if constexpr (can_vectorize) {
                return Strategy::Vectorized;
            } else if constexpr (is_large && is_complex_op && is_thread_safe_v<Op>) {
                return Strategy::Parallel;
            } else {
                return Strategy::Scalar;
            }
        }

        static constexpr Strategy value = select();
    };

} // namespace fem::numeric::traits

#endif //NUMERIC_OPERATION_TRAITS_H
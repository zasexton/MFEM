#pragma once

#ifndef NUMERIC_EXPRESSION_TRAITS_H
#define NUMERIC_EXPRESSION_TRAITS_H

#include <type_traits>
#include <tuple>

#include "../base/expression_base.h"
#include "../base/container_base.h"
#include "../base/ops_base.h"

#include "type_traits.h"
#include "numeric_traits.h"
#include "operation_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Expression categories for dispatch and optimization
     */
    enum class ExpressionCategory {
        Terminal,       // Leaf node (actual data)
        Unary,         // Unary operation
        Binary,        // Binary operation
        Scalar,        // Scalar-array operation
        Reduction,     // Reduction operation (sum, mean, etc.)
        Broadcast,     // Broadcasting operation
        MatMul,        // Matrix multiplication
        Composite,     // Complex composite expression
        Unknown
    };

    /**
     * @brief Evaluation strategies for expressions
     */
    enum class EvaluationStrategy {
        Immediate,     // Evaluate immediately
        Lazy,          // Keep as expression
        Chunked,       // Evaluate in chunks for cache efficiency
        Parallel,      // Parallel evaluation
        Vectorized,    // SIMD vectorization
        Fused          // Fused operations
    };

    /**
     * @brief Detect if type is derived from ExpressionBase
     */
    template<typename T>
    struct is_expression {
    private:
        template<typename E>
        static std::true_type test(const ExpressionBase<E>*);
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    };

    template<typename T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

    /**
     * @brief Detect if type is a terminal expression
     */
    template<typename T>
    struct is_terminal_expression : std::false_type {};

    template<typename Container>
    struct is_terminal_expression<TerminalExpression<Container>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_terminal_expression_v = is_terminal_expression<T>::value;

    /**
     * @brief Detect if type is a binary expression
     */
    template<typename T>
    struct is_binary_expression : std::false_type {};

    template<typename Op, typename Left, typename Right>
    struct is_binary_expression<BinaryExpression<Op, Left, Right>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_binary_expression_v = is_binary_expression<T>::value;

    /**
     * @brief Detect if type is a unary expression
     */
    template<typename T>
    struct is_unary_expression : std::false_type {};

    template<typename Op, typename Arg>
    struct is_unary_expression<UnaryExpression<Op, Arg>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_unary_expression_v = is_unary_expression<T>::value;

    /**
     * @brief Detect if type is a scalar expression
     */
    template<typename T>
    struct is_scalar_expression : std::false_type {};

    template<typename T>
    struct is_scalar_expression<ScalarExpression<T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_scalar_expression_v = is_scalar_expression<T>::value;

    /**
     * @brief Detect if type is a broadcast expression
     */
    template<typename T>
    struct is_broadcast_expression : std::false_type {};

  
    // TODO: Define BroadcastExpression and re-enable specialization
    // template<typename Container>
    // struct is_broadcast_expression<BroadcastExpression<Container>> : std::true_type {};


    template<typename T>
    inline constexpr bool is_broadcast_expression_v = is_broadcast_expression<T>::value;

    /**
     * @brief Extract value type from expression
     */
    template<typename Expr, typename = void>
    struct expression_value_type {
        using type = void;
    };

    template<typename Expr>
    struct expression_value_type<Expr, std::void_t<typename Expr::value_type>> {
        using type = typename Expr::value_type;
    };

    template<typename Expr>
    using expression_value_type_t = typename expression_value_type<Expr>::type;

    /**
     * @brief Get expression category
     */
    template<typename Expr>
    struct expression_category {
        static constexpr ExpressionCategory value = [] {
            if constexpr (is_terminal_expression_v<Expr>) {
            return ExpressionCategory::Terminal;
        } else if constexpr (is_unary_expression_v<Expr>) {
            return ExpressionCategory::Unary;
        } else if constexpr (is_binary_expression_v<Expr>) {
            return ExpressionCategory::Binary;
        } else if constexpr (is_scalar_expression_v<Expr>) {
            return ExpressionCategory::Scalar;
        } else if constexpr (is_broadcast_expression_v<Expr>) {
            return ExpressionCategory::Broadcast;
        } else if constexpr (is_expression_v<Expr>) {
            return ExpressionCategory::Composite;
        } else {
            return ExpressionCategory::Unknown;
        }
        }();
    };

    template<typename Expr>
    inline constexpr ExpressionCategory expression_category_v = expression_category<Expr>::value;

    /**
     * @brief Calculate expression depth (for optimization decisions)
     */
    template<typename Expr>
    struct expression_depth {
        static constexpr size_t value = 0;
    };

    template<typename Container>
    struct expression_depth<TerminalExpression<Container>> {
    static constexpr size_t value = 1;
    };

    template<typename Op, typename Arg>
    struct expression_depth<UnaryExpression<Op, Arg>> {
    static constexpr size_t value = 1 + expression_depth<Arg>::value;
    };

    template<typename Op, typename Left, typename Right>
    struct expression_depth<BinaryExpression<Op, Left, Right>> {
    static constexpr size_t value = 1 + std::max(
            expression_depth<Left>::value,
            expression_depth<Right>::value
    );
    };

    template<typename Expr>
    inline constexpr size_t expression_depth_v = expression_depth<Expr>::value;

    /**
     * @brief Count operations in expression tree
     */
    template<typename Expr>
    struct operation_count {
        static constexpr size_t value = 0;
    };

    template<typename Container>
    struct operation_count<TerminalExpression<Container>> {
    static constexpr size_t value = 0;
    };

    template<typename Op, typename Arg>
    struct operation_count<UnaryExpression<Op, Arg>> {
    static constexpr size_t value = 1 + operation_count<Arg>::value;
    };

    template<typename Op, typename Left, typename Right>
    struct operation_count<BinaryExpression<Op, Left, Right>> {
    static constexpr size_t value = 1 +
                                    operation_count<Left>::value +
                                    operation_count<Right>::value;
    };

    template<typename Expr>
    inline constexpr size_t operation_count_v = operation_count<Expr>::value;

    /**
     * @brief Check if expression contains broadcasts
     */
    template<typename Expr>
    struct has_broadcast {
        static constexpr bool value = is_broadcast_expression_v<Expr>;
    };

    template<typename Op, typename Arg>
    struct has_broadcast<UnaryExpression<Op, Arg>> {
    static constexpr bool value = has_broadcast<Arg>::value;
    };

    template<typename Op, typename Left, typename Right>
    struct has_broadcast<BinaryExpression<Op, Left, Right>> {
    static constexpr bool value = has_broadcast<Left>::value ||
                                  has_broadcast<Right>::value;
    };

    template<typename Expr>
    inline constexpr bool has_broadcast_v = has_broadcast<Expr>::value;

    /**
     * @brief Determine if expression can be vectorized
     */
    template<typename Expr>
    struct is_vectorizable {
        using value_type = expression_value_type_t<Expr>;

        static constexpr bool value =
                !std::is_void_v<value_type> &&
                std::is_arithmetic_v<value_type> &&
                !has_broadcast_v<Expr> &&  // Broadcasting complicates vectorization
                expression_depth_v<Expr> <= 10;  // Too deep expressions may not vectorize well
    };

    template<typename Expr>
    inline constexpr bool is_vectorizable_v = is_vectorizable<Expr>::value;

    /**
     * @brief Expression properties aggregator
     */
    template<typename Expr>
    struct expression_properties {
        // Basic properties
        static constexpr bool is_expression = is_expression_v<Expr>;
        static constexpr ExpressionCategory category = expression_category_v<Expr>;

        // Structure
        static constexpr size_t depth = expression_depth_v<Expr>;
        static constexpr size_t operation_count = operation_count_v<Expr>;
        static constexpr bool has_broadcasts = has_broadcast_v<Expr>;

        // Types
        using value_type = expression_value_type_t<Expr>;

        // Optimization hints
        static constexpr bool can_vectorize = is_vectorizable_v<Expr>;
        static constexpr bool should_parallelize = operation_count > 1000;  // Heuristic
        static constexpr bool should_materialize = depth > 5;  // Deep expressions might benefit from materialization

        // Evaluation strategy recommendation
        static constexpr EvaluationStrategy recommended_strategy = [] {
            if constexpr (!is_expression) {
            return EvaluationStrategy::Immediate;
        } else if constexpr (can_vectorize) {
            return EvaluationStrategy::Vectorized;
        } else if constexpr (should_parallelize) {
            return EvaluationStrategy::Parallel;
        } else if constexpr (depth <= 3) {
            return EvaluationStrategy::Lazy;
        } else {
            return EvaluationStrategy::Chunked;
        }
        }();
    };

    /**
     * @brief Extract operation type from expression
     */
    template<typename Expr>
    struct expression_operation {
        using type = void;
    };

    template<typename Op, typename Arg>
    struct expression_operation<UnaryExpression<Op, Arg>> {
        using type = Op;
    };

    template<typename Op, typename Left, typename Right>
    struct expression_operation<BinaryExpression<Op, Left, Right>> {
        using type = Op;
    };

    template<typename Expr>
    using expression_operation_t = typename expression_operation<Expr>::type;

    /**
     * @brief Check if expression operations are commutative
     */
    template<typename Expr>
    struct has_commutative_operations {
        static constexpr bool value = [] {
            if constexpr (is_binary_expression_v<Expr>) {
                using Op = expression_operation_t<Expr>;
                return algebraic_properties<Op>::value.commutative;
            } else {
                return false;
            }
        }();
    };

    template<typename Expr>
    inline constexpr bool has_commutative_operations_v = has_commutative_operations<Expr>::value;

    /**
     * @brief Check if expression can be evaluated in parallel
     */
    template<typename Expr>
    struct is_parallel_safe {
        static constexpr bool value =
                is_expression_v<Expr> &&
                !std::is_void_v<expression_value_type_t<Expr>> &&
        NumberLike<expression_value_type_t<Expr>>;
    };

    template<typename Expr>
    inline constexpr bool is_parallel_safe_v = is_parallel_safe<Expr>::value;

    /**
     * @brief Determine result shape from expression (compile-time if possible)
     */
    template<typename Expr>
    struct expression_shape {
        static Shape compute(const Expr& expr) {
            if constexpr (requires { expr.shape(); }) {
                return expr.shape();
            } else {
                return Shape{};  // Unknown shape
            }
        }

        static constexpr bool is_static = false;
    };

    /**
     * @brief Check if two expressions are compatible for operations
     */
    template<typename Expr1, typename Expr2>
    struct are_expressions_compatible {
        using value_type1 = expression_value_type_t<Expr1>;
        using value_type2 = expression_value_type_t<Expr2>;

        static constexpr bool value =
                !std::is_void_v<value_type1> &&
                !std::is_void_v<value_type2> &&
                are_compatible_v<value_type1, value_type2>;
    };

    template<typename Expr1, typename Expr2>
    inline constexpr bool are_expressions_compatible_v =
            are_expressions_compatible<Expr1, Expr2>::value;

    /**
     * @brief Expression tree visitor helper
     */
    template<typename Expr, typename Visitor>
    struct expression_visitor {
        static void visit(const Expr& expr, Visitor&& visitor) {
            visitor(expr);

            if constexpr (is_unary_expression_v<Expr>) {
                expression_visitor<typename Expr::argument_type, Visitor>::visit(
                        expr.argument(), std::forward<Visitor>(visitor));
            } else if constexpr (is_binary_expression_v<Expr>) {
                expression_visitor<typename Expr::left_type, Visitor>::visit(
                        expr.left(), std::forward<Visitor>(visitor));
                expression_visitor<typename Expr::right_type, Visitor>::visit(
                        expr.right(), std::forward<Visitor>(visitor));
            }
        }
    };

    /**
     * @brief Count terminals in expression tree
     */
    template<typename Expr>
    struct terminal_count {
        static constexpr size_t value = is_terminal_expression_v<Expr> ? 1 : 0;
    };

    template<typename Op, typename Arg>
    struct terminal_count<UnaryExpression<Op, Arg>> {
        static constexpr size_t value = terminal_count<Arg>::value;
    };

    template<typename Op, typename Left, typename Right>
    struct terminal_count<BinaryExpression<Op, Left, Right>> {
        static constexpr size_t value = terminal_count<Left>::value +
                                        terminal_count<Right>::value;
    };

    template<typename Expr>
    inline constexpr size_t terminal_count_v = terminal_count<Expr>::value;

    /**
     * @brief Check if expression should be materialized
     */
    template<typename Expr>
    struct should_materialize {
        static constexpr bool value =
                expression_depth_v<Expr> > 5 ||           // Deep expression trees
                operation_count_v<Expr> > 10 ||           // Many operations
                terminal_count_v<Expr> > 4 ||             // Many data sources
                has_broadcast_v<Expr>;                    // Broadcasting complexity
    };

    template<typename Expr>
    inline constexpr bool should_materialize_v = should_materialize<Expr>::value;

    /**
     * @brief Expression optimization hints
     */
    template<typename Expr>
    struct optimization_hints {
        static constexpr bool use_simd = is_vectorizable_v<Expr>;
        static constexpr bool use_parallel =
                is_parallel_safe_v<Expr> &&
                operation_count_v<Expr> > 1000;
        static constexpr bool materialize = should_materialize_v<Expr>;
        static constexpr bool can_reorder = has_commutative_operations_v<Expr>;
        static constexpr bool fuse_operations =
                expression_depth_v<Expr> <= 3 &&
                !has_broadcast_v<Expr>;
    };

    /**
     * @brief Helper to determine evaluation order
     */
    template<typename Expr>
    struct evaluation_order {
        enum Order { LeftToRight, RightToLeft, Optimal };

        static constexpr Order value = [] {
            if constexpr (is_binary_expression_v<Expr>) {
                // Simple heuristic: evaluate side with fewer operations first
                constexpr auto left_ops = operation_count<typename Expr::left_type>::value;
                constexpr auto right_ops = operation_count<typename Expr::right_type>::value;

                if (left_ops < right_ops) {
                    return LeftToRight;
                } else if (right_ops < left_ops) {
                    return RightToLeft;
                } else {
                    return Optimal;
                }
            }
            return LeftToRight;
        }();
    };

} // namespace fem::numeric::traits

#endif //NUMERIC_EXPRESSION_TRAITS_H

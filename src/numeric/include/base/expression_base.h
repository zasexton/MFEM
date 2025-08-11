#pragma once

#ifndef NUMERIC_EXPRESSION_BASE_H
#define NUMERIC_EXPRESSION_BASE_H

#include <type_traits>
#include <utility>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for expression templates
     *
     * Enables lazy evaluation and eliminates temporaries for IEEE-compliant operations
     */
    template<typename Derived>
    class ExpressionBase {
    public:
        using derived_type = Derived;

        /**
         * @brief CRTP helper to get derived class
         */
        const Derived& derived() const noexcept {
            return static_cast<const Derived&>(*this);
        }

        /**
         * @brief Get the shape of the expression result
         */
        Shape shape() const {
            return derived().shape();
        }

        /**
         * @brief Get total number of elements
         */
        size_t size() const {
            return shape().size();
        }

        /**
         * @brief Evaluate expression at linear index
         */
        template<typename T>
        auto operator[](size_t i) const {
            return derived().template eval<T>(i);
        }

        /**
         * @brief Evaluate expression at multi-dimensional index
         */
        template<typename T, typename... Indices>
        auto at(Indices... indices) const {
            return derived().template eval_at<T>(indices...);
        }

        /**
         * @brief Force evaluation into a container
         */
        template<template<typename> class Container, typename T>
        Container<T> eval() const {
            Container<T> result(shape());
            derived().template eval_to(result);
            return result;
        }

        /**
         * @brief Evaluate into existing container
         */
        template<typename Container>
        void eval_to(Container& result) const {
            derived().eval_to(result);
        }

        /**
         * @brief Check if expression can be evaluated in parallel
         */
        bool is_parallelizable() const noexcept {
            return derived().is_parallelizable();
        }

        /**
         * @brief Get computational complexity estimate
         */
        size_t complexity() const noexcept {
            return derived().complexity();
        }

    protected:
        ExpressionBase() = default;
        ~ExpressionBase() = default;
        ExpressionBase(const ExpressionBase&) = default;
        ExpressionBase(ExpressionBase&&) = default;
        ExpressionBase& operator=(const ExpressionBase&) = default;
        ExpressionBase& operator=(ExpressionBase&&) = default;
    };

    /**
     * @brief Terminal expression (leaf node in expression tree)
     */
    template<typename Container>
    class TerminalExpression : public ExpressionBase<TerminalExpression<Container>> {
    public:
        using value_type = typename Container::value_type;
        using const_reference = const Container&;

        explicit TerminalExpression(const_reference data)
                : data_(data) {}

        Shape shape() const { return data_.shape(); }

        template<typename T>
        auto eval(size_t i) const {
            return static_cast<T>(data_[i]);
        }

        template<typename T, typename... Indices>
        auto eval_at(Indices... indices) const {
            return static_cast<T>(data_.at(indices...));
        }

        template<typename ResultContainer>
        void eval_to(ResultContainer& result) const {
            using result_type = typename ResultContainer::value_type;
            if (result.shape() != shape()) {
                result.resize(shape());
            }
            for (size_t i = 0; i < data_.size(); ++i) {
                result[i] = static_cast<result_type>(data_[i]);
            }
        }

        bool is_parallelizable() const noexcept { return true; }
        size_t complexity() const noexcept { return size(); }

        const_reference data() const { return data_; }

    private:
        const_reference data_;
    };

    /**
     * @brief Scalar expression (broadcasts scalar to shape)
     */
    template<typename T>
    class ScalarExpression : public ExpressionBase<ScalarExpression<T>> {
    public:
        using value_type = T;

        ScalarExpression(const T& value, const Shape& shape)
                : value_(value), shape_(shape) {}

        Shape shape() const { return shape_; }

        template<typename U>
        auto eval(size_t) const {
            return static_cast<U>(value_);
        }

        template<typename U, typename... Indices>
        auto eval_at(Indices...) const {
            return static_cast<U>(value_);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using result_type = typename Container::value_type;
            if (result.shape() != shape_) {
                result.resize(shape_);
            }
            result.fill(static_cast<result_type>(value_));
        }

        bool is_parallelizable() const noexcept { return true; }
        size_t complexity() const noexcept { return shape_.size(); }

        const T& value() const { return value_; }

    private:
        T value_;
        Shape shape_;
    };

    /**
     * @brief Binary expression template
     */
    template<typename LHS, typename RHS, typename Op>
    class BinaryExpression : public ExpressionBase<BinaryExpression<LHS, RHS, Op>> {
    public:
        using lhs_type = LHS;
        using rhs_type = RHS;
        using operation_type = Op;

        BinaryExpression(const LHS& lhs, const RHS& rhs, Op op = Op{})
                : lhs_(lhs), rhs_(rhs), op_(op) {
            // Validate shapes for broadcasting
            shape_ = compute_broadcast_shape(lhs.shape(), rhs.shape());
        }

        Shape shape() const { return shape_; }

        template<typename T>
        auto eval(size_t i) const {
            // Handle broadcasting
            size_t lhs_idx = broadcast_index(i, lhs_.shape(), shape_);
            size_t rhs_idx = broadcast_index(i, rhs_.shape(), shape_);

            auto lhs_val = lhs_.template eval<T>(lhs_idx);
            auto rhs_val = rhs_.template eval<T>(rhs_idx);

            // Check for IEEE compliance issues
            if constexpr (std::is_floating_point_v<T>) {
                if (NumericOptions::defaults().check_finite) {
                    check_operation_validity(lhs_val, rhs_val);
                }
            }

            return op_(lhs_val, rhs_val);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using T = typename Container::value_type;
            if (result.shape() != shape_) {
                result.resize(shape_);
            }

#pragma omp parallel for if(is_parallelizable() && shape_.size() > 1000)
            for (size_t i = 0; i < shape_.size(); ++i) {
                result[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return lhs_.is_parallelizable() && rhs_.is_parallelizable();
        }

        size_t complexity() const noexcept {
            return lhs_.complexity() + rhs_.complexity() + shape_.size();
        }

    private:
        const LHS& lhs_;
        const RHS& rhs_;
        Op op_;
        Shape shape_;

        static Shape compute_broadcast_shape(const Shape& s1, const Shape& s2) {
            // NumPy-style broadcasting rules
            size_t max_rank = std::max(s1.rank(), s2.rank());
            Shape result;

            for (size_t i = 0; i < max_rank; ++i) {
                size_t dim1 = i < s1.rank() ? s1[s1.rank() - 1 - i] : 1;
                size_t dim2 = i < s2.rank() ? s2[s2.rank() - 1 - i] : 1;

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw DimensionError("Incompatible shapes for broadcasting");
                }

                result[max_rank - 1 - i] = std::max(dim1, dim2);
            }

            return result;
        }

        static size_t broadcast_index(size_t idx, const Shape& from_shape, const Shape& to_shape) {
            // Convert broadcasted index back to original shape index
            if (from_shape == to_shape) {
                return idx;
            }

            // Implementation of index mapping for broadcasting
            // This is simplified - full implementation would handle all cases
            return idx % from_shape.size();
        }

        template<typename T>
        void check_operation_validity(T lhs_val, T rhs_val) const {
            if (IEEEComplianceChecker::is_nan(lhs_val) || IEEEComplianceChecker::is_nan(rhs_val)) {
                // NaN propagation is allowed by IEEE 754
                return;
            }

            // Check for operations that might produce invalid results
            if constexpr (std::is_same_v<Op, std::divides<>>) {
                if (rhs_val == T{0}) {
                    // Division by zero - IEEE 754 defines this behavior
                    // Result will be ±inf or NaN depending on numerator
                }
            }
        }
    };

    /**
     * @brief Unary expression template
     */
    template<typename Expr, typename Op>
    class UnaryExpression : public ExpressionBase<UnaryExpression<Expr, Op>> {
    public:
        using expression_type = Expr;
        using operation_type = Op;

        UnaryExpression(const Expr& expr, Op op = Op{})
                : expr_(expr), op_(op) {}

        Shape shape() const { return expr_.shape(); }

        template<typename T>
        auto eval(size_t i) const {
            auto val = expr_.template eval<T>(i);

            // Check for IEEE compliance
            if constexpr (std::is_floating_point_v<T>) {
                if (NumericOptions::defaults().check_finite) {
                    check_operation_validity(val);
                }
            }

            return op_(val);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using T = typename Container::value_type;
            if (result.shape() != shape()) {
                result.resize(shape());
            }

#pragma omp parallel for if(is_parallelizable() && shape().size() > 1000)
            for (size_t i = 0; i < shape().size(); ++i) {
                result[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return expr_.is_parallelizable();
        }

        size_t complexity() const noexcept {
            return expr_.complexity() + shape().size();
        }

    private:
        const Expr& expr_;
        Op op_;

        template<typename T>
        void check_operation_validity(T val) const {
            // Check for operations that might produce invalid results
            if constexpr (std::is_same_v<Op, struct sqrt_op>) {
                if (val < T{0}) {
                    // Square root of negative number
                    // Result will be NaN for real types
                }
            }
            // Add other operation-specific checks
        }
    };

/**
 * @brief Expression traits for type deduction
 */
    template<typename T>
    struct is_expression : std::false_type {};

    template<typename Derived>
    struct is_expression<ExpressionBase<Derived>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

/**
 * @brief Helper to create terminal expressions
 */
    template<typename Container>
    auto make_expression(const Container& c) {
        return TerminalExpression<Container>(c);
    }

/**
 * @brief Helper to create scalar expressions
 */
    template<typename T>
    auto make_scalar_expression(const T& value, const Shape& shape) {
        return ScalarExpression<T>(value, shape);
    }

} // namespace fem::numeric

#endif //NUMERIC_EXPRESSION_BASE_H

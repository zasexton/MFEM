#pragma once

#ifndef NUMERIC_EXPRESSION_BASE_H
#define NUMERIC_EXPRESSION_BASE_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <execution>
#include <functional>
#include <unordered_map>
#include <any>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "numeric_base.h"
#include "ops_base.h"
#include "container_base.h"
#include "storage_base.h"
#include "iterator_base.h"
#include "slice_base.h"
#include "view_base.h"

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

        /**
         * @brief Check if expression can be vectorized
         */
        bool is_vectorizable() const noexcept {
            return derived().is_vectorizable();
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
     * @brief Expression traits for optimization
     */
    template<typename Expr>
    struct expression_traits {
        static constexpr bool is_temporary = false;
        static constexpr bool can_vectorize = false;
        static constexpr size_t alignment = alignof(typename Expr::value_type);
    };

    /**
     * @brief Terminal expression (leaf node in expression tree)
     */
    template<typename Container>
    class TerminalExpression : public ExpressionBase<TerminalExpression<Container>> {
    public:
        using value_type = typename Container::value_type;
        using container_type = Container;

        // Constructor for lvalue references - stores reference
        explicit TerminalExpression(const Container& data)
            : data_ptr_(&data), owns_data_(false) {}

        // Constructor for rvalue references - moves/copies data
        explicit TerminalExpression(Container&& data)
            : owned_data_(std::move(data)), data_ptr_(&owned_data_), owns_data_(true) {}

        Shape shape() const { return data().shape(); }

        template<typename T>
        auto eval(size_t i) const {
            return static_cast<T>(data()[i]);
        }

        template<typename T, typename... Indices>
        auto eval_at(Indices... indices) const {
            return static_cast<T>(data().at(indices...));
        }

        template<typename ResultContainer>
        void eval_to(ResultContainer& result) const {
            using result_type = typename ResultContainer::value_type;
            if (result.shape() != shape()) {
                result.resize(shape());
            }

#ifdef _OPENMP
            bool use_parallel = NumericOptions::defaults().allow_parallel &&
                              data().size() > 1000;
#pragma omp parallel for if(use_parallel)
#endif
            for (size_t i = 0; i < data().size(); ++i) {
                result[i] = static_cast<result_type>(data()[i]);
            }
        }

        bool is_parallelizable() const noexcept { return true; }
        bool is_vectorizable() const noexcept { return data().is_contiguous(); }
        size_t complexity() const noexcept { return data().size(); }

        const Container& data() const {
            return *data_ptr_;
        }

    private:
        Container owned_data_;  // Storage for moved/copied data
        const Container* data_ptr_;  // Pointer to either owned_data_ or external data
        bool owns_data_;  // Whether we own the data
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
        bool is_vectorizable() const noexcept { return true; }
        size_t complexity() const noexcept { return shape_.size(); }

        const T& value() const { return value_; }

    private:
        T value_;
        Shape shape_;
    };

    /**
     * @brief View expression for non-owning references
     */
    template<typename T>
    class ViewExpression : public ExpressionBase<ViewExpression<T>> {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;

        ViewExpression(pointer data, const Shape& shape)
            : data_(data), shape_(shape) {}

        Shape shape() const { return shape_; }

        template<typename U>
        auto eval(size_t i) const {
            return static_cast<U>(data_[i]);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using result_type = typename Container::value_type;
            if (result.shape() != shape_) {
                result.resize(shape_);
            }

            #ifdef _OPENMP
            bool use_parallel = NumericOptions::defaults().allow_parallel &&
                              shape_.size() > 1000;
            #pragma omp parallel for if(use_parallel)
            #endif
            for (size_t i = 0; i < shape_.size(); ++i) {
                result[i] = static_cast<result_type>(data_[i]);
            }
        }

        bool is_parallelizable() const noexcept { return true; }
        bool is_vectorizable() const noexcept { return true; }
        size_t complexity() const noexcept { return shape_.size(); }

    private:
        pointer data_;
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

            #ifdef _OPENMP
            bool use_parallel = is_parallelizable() &&
                              shape_.size() > 1000 &&
                              NumericOptions::defaults().allow_parallel;
            #pragma omp parallel for if(use_parallel)
            #endif
            for (size_t i = 0; i < shape_.size(); ++i) {
                result[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return lhs_.is_parallelizable() && rhs_.is_parallelizable();
        }

        bool is_vectorizable() const noexcept {
            return lhs_.is_vectorizable() && rhs_.is_vectorizable();
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
            if (!s1.is_broadcastable_with(s2)) {
                throw DimensionError("Incompatible shapes for broadcasting");
            }
            return s1.broadcast_to(s2);
        }

        static size_t broadcast_index(size_t idx, const Shape& from_shape, const Shape& to_shape) {
            if (from_shape == to_shape) {
                return idx;
            }

            // Proper multi-dimensional index calculation
            std::vector<size_t> indices(to_shape.rank());
            size_t temp = idx;

            // Convert linear index to multi-dimensional indices
            for (size_t i = to_shape.rank(); i > 0; --i) {
                size_t dim_idx = i - 1;
                indices[dim_idx] = temp % to_shape[dim_idx];
                temp /= to_shape[dim_idx];
            }

            // Map to source shape considering broadcasting rules
            size_t result = 0;
            size_t stride = 1;

            size_t offset = to_shape.rank() - from_shape.rank();
            for (size_t i = from_shape.rank(); i > 0; --i) {
                size_t from_idx = i - 1;
                size_t to_idx = from_idx + offset;
                size_t coord = (from_shape[from_idx] == 1) ? 0 : indices[to_idx];
                result += coord * stride;
                stride *= from_shape[from_idx];
            }

            return result;
        }

        template<typename T>
        void check_operation_validity(T lhs_val, T rhs_val) const {
            if (IEEEComplianceChecker::is_nan(lhs_val) || IEEEComplianceChecker::is_nan(rhs_val)) {
                // NaN propagation is allowed by IEEE 754
                return;
            }

            // Check for operations that might produce invalid results
            if constexpr (std::is_same_v<Op, ops::divides<>> ||
                         std::is_same_v<Op, ops::divides<T>>) {
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

            #ifdef _OPENMP
            bool use_parallel = is_parallelizable() &&
                              shape().size() > 1000 &&
                              NumericOptions::defaults().allow_parallel;
            #pragma omp parallel for if(use_parallel)
            #endif
            for (size_t i = 0; i < shape().size(); ++i) {
                result[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return expr_.is_parallelizable();
        }

        bool is_vectorizable() const noexcept {
            return expr_.is_vectorizable();
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
            if constexpr (std::is_same_v<Op, ops::sqrt_op<T>>) {
                if (val < T{0}) {
                    // Square root of negative number
                    // Result will be NaN for real types
                }
            }
            else if constexpr (std::is_same_v<Op, ops::log_op<T>>) {
                if (val <= T{0}) {
                    // Log of non-positive number
                    // Result will be NaN or -inf
                }
            }
            else if constexpr (std::is_same_v<Op, ops::asin_op<T>> ||
                              std::is_same_v<Op, ops::acos_op<T>>) {
                if (val < T{-1} || val > T{1}) {
                    // Domain error for inverse trig functions
                }
            }
        }
    };

    /**
     * @brief Slice expression for lazy evaluation of slices
     */
    template<typename Expr>
    class SliceExpression : public ExpressionBase<SliceExpression<Expr>> {
    public:
        using expression_type = Expr;
        using value_type = typename Expr::value_type;

        SliceExpression(const Expr& expr, const MultiIndex& indices)
            : expr_(expr), indices_(indices) {
            shape_ = indices_.result_shape(expr.shape());
        }

        Shape shape() const { return shape_; }

        template<typename T>
        auto eval(size_t i) const {
            // Map slice index to original expression index
            size_t orig_idx = map_slice_index(i);
            return expr_.template eval<T>(orig_idx);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using T = typename Container::value_type;
            if (result.shape() != shape_) {
                result.resize(shape_);
            }

            #ifdef _OPENMP
            bool use_parallel = is_parallelizable() &&
                              shape_.size() > 1000 &&
                              NumericOptions::defaults().allow_parallel;
            #pragma omp parallel for if(use_parallel)
            #endif
            for (size_t i = 0; i < shape_.size(); ++i) {
                result[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return expr_.is_parallelizable();
        }

        bool is_vectorizable() const noexcept {
            // Slicing may break contiguity
            return false;
        }

        size_t complexity() const noexcept {
            return expr_.complexity() + shape_.size();
        }

    private:
        const Expr& expr_;
        MultiIndex indices_;
        Shape shape_;

        size_t map_slice_index(size_t slice_idx) const {
            // Implementation would map from slice coordinates to original coordinates
            // This is a simplified placeholder
            return slice_idx;
        }
    };

    /**
     * @brief Reduction expression for lazy evaluation of reductions
     */
    template<typename Expr, typename Op>
    class ReductionExpression : public ExpressionBase<ReductionExpression<Expr, Op>> {
    public:
        using expression_type = Expr;
        using operation_type = Op;
        using value_type = typename Expr::value_type;

        ReductionExpression(const Expr& expr, Op op, int axis = -1)
            : expr_(expr), op_(op), axis_(axis) {
            if (axis_ < 0) {
                // Full reduction to scalar
                shape_ = Shape({1});
            } else {
                // Reduction along specific axis
                auto orig_shape = expr.shape();
                std::vector<size_t> new_dims;
                for (size_t i = 0; i < orig_shape.rank(); ++i) {
                    if (static_cast<int>(i) != axis_) {
                        new_dims.push_back(orig_shape[i]);
                    }
                }
                shape_ = Shape(new_dims);
            }
        }

        Shape shape() const { return shape_; }

        template<typename T>
        auto eval(size_t i) const {
            // For simplicity, this would aggregate values along the reduction axis
            // Full implementation would handle axis-specific reduction
            if (axis_ < 0) {
                // Full reduction
                T result = T{};
                for (size_t j = 0; j < expr_.shape().size(); ++j) {
                    result = op_(result, expr_.template eval<T>(j));
                }
                return result;
            }
            // Axis-specific reduction would go here
            return expr_.template eval<T>(i);
        }

        template<typename Container>
        void eval_to(Container& result) const {
            using T = typename Container::value_type;
            if (result.shape() != shape_) {
                result.resize(shape_);
            }

            if (axis_ < 0) {
                // Full reduction
                result[0] = eval<T>(0);
            } else {
                // Axis-specific reduction
                #ifdef _OPENMP
                bool use_parallel = is_parallelizable() &&
                                  shape_.size() > 1000 &&
                                  NumericOptions::defaults().allow_parallel;
                #pragma omp parallel for if(use_parallel)
                #endif
                for (size_t i = 0; i < shape_.size(); ++i) {
                    result[i] = eval<T>(i);
                }
            }
        }

        bool is_parallelizable() const noexcept {
            return expr_.is_parallelizable();
        }

        bool is_vectorizable() const noexcept {
            return false; // Reductions typically break vectorization
        }

        size_t complexity() const noexcept {
            return expr_.complexity() * expr_.shape().size() / shape_.size();
        }

    private:
        const Expr& expr_;
        Op op_;
        int axis_;
        Shape shape_;
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

    /**
     * @brief Helper to create view expressions
     */
    template<typename T>
    auto make_view_expression(T* data, const Shape& shape) {
        return ViewExpression<T>(data, shape);
    }

    // ============================================================
    // Expression builder operators
    // ============================================================

    // ============================================================
    // Operation wrappers for expression templates
    // These wrappers delegate to ops_base.h operations with proper type deduction
    // ============================================================

    namespace expr_ops {

        /**
         * @brief Generic wrapper that uses ops_base operations with runtime type deduction
         * This allows us to use ops_base.h operations without void specialization issues
         */
        template<template<typename> class OpTemplate>
        struct op_wrapper {
            template<typename T>
            auto operator()(const T& val) const {
                using op_type = OpTemplate<T>;
                return op_type{}(val);
            }
        };

        template<template<typename> class OpTemplate>
        struct binary_op_wrapper {
            template<typename T>
            auto operator()(const T& lhs, const T& rhs) const {
                using op_type = OpTemplate<T>;
                return op_type{}(lhs, rhs);
            }
        };

        // Unary operation wrappers using ops_base.h implementations
        using negate_wrapper = op_wrapper<ops::negate>;
        using abs_wrapper = op_wrapper<ops::abs_op>;
        using sign_wrapper = op_wrapper<ops::sign_op>;
        using sqrt_wrapper = op_wrapper<ops::sqrt_op>;
        using exp_wrapper = op_wrapper<ops::exp_op>;
        using log_wrapper = op_wrapper<ops::log_op>;
        using log10_wrapper = op_wrapper<ops::log10_op>;
        using log2_wrapper = op_wrapper<ops::log2_op>;
        using sin_wrapper = op_wrapper<ops::sin_op>;
        using cos_wrapper = op_wrapper<ops::cos_op>;
        using tan_wrapper = op_wrapper<ops::tan_op>;
        using asin_wrapper = op_wrapper<ops::asin_op>;
        using acos_wrapper = op_wrapper<ops::acos_op>;
        using atan_wrapper = op_wrapper<ops::atan_op>;
        using sinh_wrapper = op_wrapper<ops::sinh_op>;
        using cosh_wrapper = op_wrapper<ops::cosh_op>;
        using tanh_wrapper = op_wrapper<ops::tanh_op>;
        using round_wrapper = op_wrapper<ops::round_op>;
        using floor_wrapper = op_wrapper<ops::floor_op>;
        using ceil_wrapper = op_wrapper<ops::ceil_op>;
        using trunc_wrapper = op_wrapper<ops::trunc_op>;

        // Binary operation wrappers
        using pow_wrapper = binary_op_wrapper<ops::power_op>;
        using atan2_wrapper = binary_op_wrapper<ops::atan2_op>;
        using hypot_wrapper = binary_op_wrapper<ops::hypot_op>;
        using min_wrapper = binary_op_wrapper<ops::min_op>;
        using max_wrapper = binary_op_wrapper<ops::max_op>;
    }

    // ============================================================
    // Expression builder operators - updated to use wrappers
    // ============================================================

    // Binary arithmetic operations - these can use ops directly with void specialization
    template<typename LHS, typename RHS>
    auto operator+(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, ops::plus<>>(
            lhs.derived(), rhs.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto operator-(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, ops::minus<>>(
            lhs.derived(), rhs.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto operator*(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, ops::multiplies<>>(
            lhs.derived(), rhs.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto operator/(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, ops::divides<>>(
            lhs.derived(), rhs.derived()
        );
    }

    // Unary operations
    template<typename Expr>
    auto operator-(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::negate_wrapper>(expr.derived());
    }

    // Mathematical functions - using wrappers that delegate to ops_base.h
    template<typename Expr>
    auto abs(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::abs_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto sign(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::sign_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto sqrt(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::sqrt_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto exp(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::exp_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto log(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::log_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto log10(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::log10_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto log2(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::log2_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto sin(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::sin_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto cos(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::cos_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto tan(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::tan_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto asin(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::asin_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto acos(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::acos_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto atan(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::atan_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto sinh(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::sinh_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto cosh(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::cosh_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto tanh(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::tanh_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto round(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::round_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto floor(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::floor_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto ceil(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::ceil_wrapper>(expr.derived());
    }

    template<typename Expr>
    auto trunc(const ExpressionBase<Expr>& expr) {
        return UnaryExpression<Expr, expr_ops::trunc_wrapper>(expr.derived());
    }

    // Binary mathematical functions
    template<typename LHS, typename RHS>
    auto pow(const ExpressionBase<LHS>& base, const ExpressionBase<RHS>& exp) {
        return BinaryExpression<LHS, RHS, expr_ops::pow_wrapper>(
            base.derived(), exp.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto atan2(const ExpressionBase<LHS>& y, const ExpressionBase<RHS>& x) {
        return BinaryExpression<LHS, RHS, expr_ops::atan2_wrapper>(
            y.derived(), x.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto hypot(const ExpressionBase<LHS>& x, const ExpressionBase<RHS>& y) {
        return BinaryExpression<LHS, RHS, expr_ops::hypot_wrapper>(
            x.derived(), y.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto min(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, expr_ops::min_wrapper>(
            lhs.derived(), rhs.derived()
        );
    }

    template<typename LHS, typename RHS>
    auto max(const ExpressionBase<LHS>& lhs, const ExpressionBase<RHS>& rhs) {
        return BinaryExpression<LHS, RHS, expr_ops::max_wrapper>(
            lhs.derived(), rhs.derived()
        );
    }

} // namespace fem::numeric

#endif //NUMERIC_EXPRESSION_BASE_H
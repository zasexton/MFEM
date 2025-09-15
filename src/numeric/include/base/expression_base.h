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
#include <memory>
#include <iostream>
#include <array>
#include <stdexcept>
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
         * @brief Get total number of elements
         */
        size_t size() const {
            return derived().shape().size();
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
            Container<T> result(derived().shape());
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
            : data_ptr_(&data) {}

        // Constructor for rvalue references - moves data
        explicit TerminalExpression(Container&& data)
            : owned_data_(std::make_unique<Container>(std::move(data))),
              data_ptr_(owned_data_.get()) {}

        TerminalExpression(const TerminalExpression& other) {
            if (other.owned_data_) {
                owned_data_ = std::make_unique<Container>(*other.owned_data_);
                data_ptr_ = owned_data_.get();
            } else {
                data_ptr_ = other.data_ptr_;
            }
        }

        TerminalExpression(TerminalExpression&& other) noexcept
            : owned_data_(std::move(other.owned_data_)),
              data_ptr_(owned_data_ ? owned_data_.get() : other.data_ptr_) {}

        TerminalExpression& operator=(const TerminalExpression& other) {
            if (this != &other) {
                if (other.owned_data_) {
                    owned_data_ = std::make_unique<Container>(*other.owned_data_);
                    data_ptr_ = owned_data_.get();
                } else {
                    owned_data_.reset();
                    data_ptr_ = other.data_ptr_;
                }
            }
            return *this;
        }

        TerminalExpression& operator=(TerminalExpression&& other) noexcept {
            if (this != &other) {
                owned_data_ = std::move(other.owned_data_);
                data_ptr_ = owned_data_ ? owned_data_.get() : other.data_ptr_;
            }
            return *this;
        }

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
            return owned_data_ ? *owned_data_ : *data_ptr_;
        }

    private:
        std::unique_ptr<Container> owned_data_{};  // Storage when owning data
        const Container* data_ptr_{};              // Pointer to either owned or external data
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

    template<typename Op, typename Left, typename Right>
    class BinaryExpression : public ExpressionBase<BinaryExpression<Op, Left, Right>> {
    public:
        using operation_type = Op;
        using left_type  = Left;
        using right_type = Right;
        using lhs_type  = Left;
        using rhs_type  = Right;
        using value_type = std::common_type_t<typename Left::value_type,
                                              typename Right::value_type>;

        template<typename LeftArg, typename RightArg>
        BinaryExpression(LeftArg&& left, RightArg&& right, Op op = Op{})
            : left_(std::forward<LeftArg>(left)),
              right_(std::forward<RightArg>(right)),
              op_(op) {
            shape_ = compute_broadcast_shape(left_.shape(), right_.shape());
            init_broadcast(left_.shape(), right_.shape());
        }

        Shape shape() const { return shape_; }

        template<typename T>
        auto eval(size_t i) const {
            // Debug shapes
            // std::cout << "Debug shapes: lhs.shape=" << lhs().shape().to_string()
            //          << " rhs.shape=" << rhs().shape().to_string()
            //          << " broadcast_shape=" << shape_.to_string() << std::endl;

            // Handle broadcasting
            size_t left_idx = left_broadcast_ ? map_index(i, left_strides_.get()) : i;
            size_t right_idx = right_broadcast_ ? map_index(i, right_strides_.get()) : i;

            auto left_val = left_.template eval<T>(left_idx);
            auto right_val = right_.template eval<T>(right_idx);

            // Add debugging output
            // std::cout << "Debug: i=" << i
            //          << " lhs_idx=" << lhs_idx << " rhs_idx=" << rhs_idx
            //          << " lhs_val=" << lhs_val << " rhs_val=" << rhs_val;

            // Check for IEEE compliance issues
            if constexpr (std::is_floating_point_v<T>) {
                if (NumericOptions::defaults().check_finite) {
                    check_operation_validity(left_val, right_val);
                }
            }

            auto result = op_(left_val, right_val);
            // std::cout << " result=" << result << std::endl;

            return result;
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
                result.data()[i] = eval<T>(i);
            }
        }

        bool is_parallelizable() const noexcept {
            return left_.is_parallelizable() && right_.is_parallelizable();
        }

        bool is_vectorizable() const noexcept {
            return left_.is_vectorizable() && right_.is_vectorizable();
        }

        size_t complexity() const noexcept {
            return left_.complexity() + right_.complexity() + shape_.size();
        }

        const Left& left()  const noexcept { return left_; }
        const Right& right() const noexcept { return right_; }

    private:
        Left left_;
        Right right_;

        Op op_;
        Shape shape_;

        // Precomputed broadcasting information
        size_t rank_{};
        bool left_broadcast_{};
        bool right_broadcast_{};
        std::unique_ptr<size_t[]> left_strides_;
        std::unique_ptr<size_t[]> right_strides_;

        static Shape compute_broadcast_shape(const Shape& s1, const Shape& s2) {
            // Get the maximum rank
            size_t max_rank = std::max(s1.rank(), s2.rank());

            // Build the result shape
            std::vector<size_t> result_dims(max_rank);

            for (size_t i = 0; i < max_rank; ++i) {
                // Work from the rightmost dimension
                size_t dim1 = (i < s1.rank()) ? s1[s1.rank() - 1 - i] : 1;
                size_t dim2 = (i < s2.rank()) ? s2[s2.rank() - 1 - i] : 1;

                // Check if dimensions are compatible
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw DimensionError("Incompatible shapes for broadcasting");
                }

                // Take the maximum (non-1) dimension
                result_dims[max_rank - 1 - i] = std::max(dim1, dim2);
            }

            return Shape(result_dims);
        }

        void init_broadcast(const Shape& lhs_shape, const Shape& rhs_shape) {
            rank_ = shape_.rank();

            left_broadcast_ = (lhs_shape != shape_);
            right_broadcast_ = (rhs_shape != shape_);

            if (left_broadcast_) {
                left_strides_ = std::make_unique<size_t[]>(rank_);
                compute_broadcast_strides(lhs_shape, left_strides_.get());
            }

            if (right_broadcast_) {
                right_strides_ = std::make_unique<size_t[]>(rank_);
                compute_broadcast_strides(rhs_shape, right_strides_.get());
            }
        }

        void compute_broadcast_strides(const Shape& from_shape, size_t* out) {
            std::fill(out, out + rank_, size_t{0});

            if (from_shape.size() == 1 || from_shape.rank() == 0) {
                return;
            }

            size_t from_rank = from_shape.rank();
            size_t offset = rank_ - from_rank;

            std::vector<size_t> temp(from_rank);
            size_t stride = 1;
            for (size_t i = from_rank; i > 0; --i) {
                size_t dim = i - 1;
                temp[dim] = stride;
                stride *= from_shape[dim];
            }

            for (size_t i = 0; i < from_rank; ++i) {
                size_t to_dim = i + offset;
                if (from_shape[i] != 1) {
                    out[to_dim] = temp[i];
                }
            }
        }

        size_t map_index(size_t idx, const size_t* strides) const {
            if (!rank_) { return 0; }
            size_t temp = idx;
            size_t result = 0;
            for (size_t i = rank_; i > 0; --i) {
                size_t dim = i - 1;
                size_t coord = temp % shape_[dim];
                temp /= shape_[dim];
                result += coord * strides[dim];
            }
            return result;
        }

        template<typename L, typename R>
        void check_operation_validity(L left_val, R right_val) const {
            using CT = std::common_type_t<L,R>;
            if (IEEEComplianceChecker::is_nan(static_cast<CT>(left_val)) || IEEEComplianceChecker::is_nan(static_cast<CT>(right_val))) {
                // NaN propagation is allowed by IEEE 754
                return;
            }

            // Check for operations that might produce invalid results
            if constexpr (std::is_same_v<Op, ops::divides<>> ||
                          std::is_same_v<Op, ops::divides<CT>>) {
                if (static_cast<CT>(right_val) == CT{0}) {
                    // Division by zero - IEEE 754 defines this behavior
                    // Result will be ±inf or NaN depending on numerator
                }
            }
        }
    };

    // Backward compatibility alias for legacy template parameter order <Left, Right, Op>
    template<typename Left, typename Right, typename Op>
    using BinaryExpressionLegacy = BinaryExpression<Op, Left, Right>;

    /**
     * @brief Unary expression template
     */
    template<typename Expr, typename Op>
    class UnaryExpression : public ExpressionBase<UnaryExpression<Expr, Op>> {
    public:
        using expression_type = Expr;
        using operation_type = Op;
        using value_type = typename Expr::value_type;

        template<typename ExprArg>
        UnaryExpression(ExprArg&& expr, Op op = Op{})
            : expr_(std::forward<ExprArg>(expr)), op_(op) {}

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
                result.data()[i] = eval<T>(i);
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
        Expr expr_;
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
    struct is_expression : std::bool_constant<
        std::is_base_of_v<ExpressionBase<std::decay_t<T>>, std::decay_t<T>>
    > {};

    // Forward declarations to selectively disable expression participation
    template<typename T> class BlockMatrix;
    template<typename T> class BlockVector;
    template<typename T, size_t Rows, size_t Cols> class SmallMatrix;

    // Block containers are not treated as expression operands in generic ops
    template<typename T>
    struct is_expression<BlockMatrix<T>> : std::false_type {};
    template<typename T>
    struct is_expression<BlockVector<T>> : std::false_type {};
    template<typename T, size_t Rows, size_t Cols>
    struct is_expression<SmallMatrix<T, Rows, Cols>> : std::false_type {};

    // Specialize for all expression types
    template<typename Derived>
    struct is_expression<ExpressionBase<Derived>> : std::true_type {};

    template<typename Container>
    struct is_expression<TerminalExpression<Container>> : std::true_type {};

    template<typename T>
    struct is_expression<ScalarExpression<T>> : std::true_type {};

    template<typename T>
    struct is_expression<ViewExpression<T>> : std::true_type {};

    template<typename Op, typename Left, typename Right>
    struct is_expression<BinaryExpression<Op, Left, Right>> : std::true_type {};

    template<typename Expr, typename Op>
    struct is_expression<UnaryExpression<Expr, Op>> : std::true_type {};

    template<typename Expr>
    struct is_expression<SliceExpression<Expr>> : std::true_type {};

    template<typename Expr, typename Op>
    struct is_expression<ReductionExpression<Expr, Op>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_expression_v = is_expression<std::decay_t<T>>::value;

    /**
     * @brief Helper to create terminal expressions
     */
    template<typename Container>
    auto make_expression(Container&& c) {
        return TerminalExpression<std::decay_t<Container>>(std::forward<Container>(c));
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
    // Generic expression builder infrastructure with perfect forwarding
    // ============================================================

    /**
     * @brief Generic unary expression builder with perfect forwarding
     */
    template<typename Op, typename Expr>
    auto make_unary_expression(Expr&& expr, Op op = Op{}) {
        return UnaryExpression<std::decay_t<Expr>, Op>(
            std::forward<Expr>(expr), op
        );
    }

    /**
     * @brief Generic binary expression builder with perfect forwarding
     */
    template<typename Op, typename Left, typename Right>
    auto make_binary_expression(Left&& left, Right&& right, Op op = Op{}) {
        return BinaryExpression<Op, std::decay_t<Left>, std::decay_t<Right>>(
            std::forward<Left>(left), std::forward<Right>(right), op
        );
    }


    template<template<typename> class OpTemplate>
    struct TypedOpWrapper {
        template<typename L, typename R>
        auto operator()(const L& lhs, const R& rhs) const {
            using T = std::common_type_t<L, R>;
            using Op = OpTemplate<T>;
            Op op{};
            return op(static_cast<T>(lhs), static_cast<T>(rhs));
        }
    };

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
    // Container detection helpers for operator SFINAE
    // ============================================================

    enum class StorageOrder : int;
    template<typename T, typename Storage, StorageOrder O> class Matrix;
    template<typename T, typename Storage> class Vector;

    template<typename T>
    struct is_matrix_type : std::false_type {};
    template<typename T, typename S, StorageOrder O>
    struct is_matrix_type<Matrix<T, S, O>> : std::true_type {};

    template<typename T>
    struct is_vector_type : std::false_type {};
    template<typename T, typename S>
    struct is_vector_type<Vector<T, S>> : std::true_type {};

    template<typename T>
    struct is_block_matrix_type : std::false_type {};
    template<typename T>
    struct is_block_matrix_type<BlockMatrix<T>> : std::true_type {};

    template<typename T>
    struct is_block_vector_type : std::false_type {};
    template<typename T>
    struct is_block_vector_type<BlockVector<T>> : std::true_type {};

    // Detect SparseVector to avoid conflicting with its eager operators
    template<typename T>
    class SparseVector; // forward decl
    template<typename T>
    struct is_sparse_vector_type : std::false_type {};
    template<typename T>
    struct is_sparse_vector_type<SparseVector<T>> : std::true_type {};

    // Detect Tensor to avoid conflicting with its eager operators
    template<typename T, size_t Rank, typename Storage> class Tensor; // forward decl
    template<typename T>
    struct is_tensor_type : std::false_type {};
    template<typename T, size_t Rank, typename S>
    struct is_tensor_type<Tensor<T, Rank, S>> : std::true_type {};

    // ============================================================
    // Binary arithmetic operators with perfect forwarding
    // ============================================================

    template<typename LHS, typename RHS>
    auto operator+(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<
            (is_expression_v<LHS> || is_expression_v<RHS>) &&
            // Defer to specialized overloads for SparseVector and Tensor (tensor-tensor only)
            !(is_sparse_vector_type<std::decay_t<LHS>>::value && is_sparse_vector_type<std::decay_t<RHS>>::value) &&
            !(is_tensor_type<std::decay_t<LHS>>::value && is_tensor_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && is_matrix_type<std::decay_t<RHS>>::value) &&
            !(is_vector_type<std::decay_t<LHS>>::value && is_vector_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && !is_expression_v<RHS>) &&
            !(is_matrix_type<std::decay_t<RHS>>::value && !is_expression_v<LHS>),
            BinaryExpression<TypedOpWrapper<ops::plus>, std::decay_t<LHS>, std::decay_t<RHS>>
        > {
        return make_binary_expression<TypedOpWrapper<ops::plus>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto operator-(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<
            (is_expression_v<LHS> || is_expression_v<RHS>) &&
            // Defer to specialized overloads for SparseVector and Tensor (tensor-tensor only)
            !(is_sparse_vector_type<std::decay_t<LHS>>::value && is_sparse_vector_type<std::decay_t<RHS>>::value) &&
            !(is_tensor_type<std::decay_t<LHS>>::value && is_tensor_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && is_matrix_type<std::decay_t<RHS>>::value) &&
            !(is_vector_type<std::decay_t<LHS>>::value && is_vector_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && !is_expression_v<RHS>) &&
            !(is_matrix_type<std::decay_t<RHS>>::value && !is_expression_v<LHS>),
            BinaryExpression<TypedOpWrapper<ops::minus>, std::decay_t<LHS>, std::decay_t<RHS>>
        > {
        return make_binary_expression<TypedOpWrapper<ops::minus>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto operator*(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<
            (is_expression_v<LHS> || is_expression_v<RHS>) &&
            // Exclude container multiplications handled by dedicated overloads
            !(is_matrix_type<std::decay_t<LHS>>::value && is_matrix_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && is_vector_type<std::decay_t<RHS>>::value) &&
            !(is_vector_type<std::decay_t<LHS>>::value && is_matrix_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && !is_expression_v<RHS>) &&
            !(is_matrix_type<std::decay_t<RHS>>::value && !is_expression_v<LHS>) &&
            !(is_block_matrix_type<std::decay_t<LHS>>::value ||
              is_block_matrix_type<std::decay_t<RHS>>::value ||
              is_block_vector_type<std::decay_t<LHS>>::value ||
              is_block_vector_type<std::decay_t<RHS>>::value) &&
            // Defer to SparseVector/Tensor scalar overloads
            !(((std::is_arithmetic_v<std::decay_t<LHS>> || is_complex_number_v<std::decay_t<LHS>> || is_dual_number_v<std::decay_t<LHS>>) && (is_sparse_vector_type<std::decay_t<RHS>>::value || is_tensor_type<std::decay_t<RHS>>::value)) ||
              ((std::is_arithmetic_v<std::decay_t<RHS>> || is_complex_number_v<std::decay_t<RHS>> || is_dual_number_v<std::decay_t<RHS>>) && (is_sparse_vector_type<std::decay_t<LHS>>::value || is_tensor_type<std::decay_t<LHS>>::value))),
            BinaryExpression<TypedOpWrapper<ops::multiplies>, std::decay_t<LHS>, std::decay_t<RHS>>
        > {
            return make_binary_expression<TypedOpWrapper<ops::multiplies>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
        }

    template<typename LHS, typename RHS>
    auto operator/(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<
            (is_expression_v<LHS> || is_expression_v<RHS>) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && is_matrix_type<std::decay_t<RHS>>::value) &&
            !(is_matrix_type<std::decay_t<LHS>>::value && !is_expression_v<RHS>) &&
            !(is_matrix_type<std::decay_t<RHS>>::value && !is_expression_v<LHS>),
            BinaryExpression<TypedOpWrapper<ops::divides>, std::decay_t<LHS>, std::decay_t<RHS>>
        > {
        return make_binary_expression<TypedOpWrapper<ops::divides>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto operator%(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<TypedOpWrapper<ops::modulus>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<TypedOpWrapper<ops::modulus>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS>
    auto operator+(LHS&& lhs, double rhs) ->
        std::enable_if_t<is_expression_v<LHS> && !is_matrix_type<std::decay_t<LHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::plus>, std::decay_t<LHS>, ScalarExpression<double>>> {
        auto scalar_expr = make_scalar_expression(rhs, lhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::plus>>(std::forward<LHS>(lhs), std::move(scalar_expr));
    }

    template<typename LHS>
    auto operator-(LHS&& lhs, double rhs) ->
        std::enable_if_t<is_expression_v<LHS> && !is_matrix_type<std::decay_t<LHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::minus>, std::decay_t<LHS>, ScalarExpression<double>>> {
        auto scalar_expr = make_scalar_expression(rhs, lhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::minus>>(std::forward<LHS>(lhs), std::move(scalar_expr));
    }

    template<typename LHS>
    auto operator*(LHS&& lhs, double rhs) ->
        std::enable_if_t<is_expression_v<LHS> && !is_matrix_type<std::decay_t<LHS>>::value &&
                         !is_sparse_vector_type<std::decay_t<LHS>>::value &&
                         !is_tensor_type<std::decay_t<LHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::multiplies>, std::decay_t<LHS>, ScalarExpression<double>>> {
        auto scalar_expr = make_scalar_expression(rhs, lhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::multiplies>>(std::forward<LHS>(lhs), std::move(scalar_expr));
    }

    template<typename LHS>
    auto operator/(LHS&& lhs, double rhs) ->
        std::enable_if_t<is_expression_v<LHS> && !is_matrix_type<std::decay_t<LHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::divides>, std::decay_t<LHS>, ScalarExpression<double>>> {
        auto scalar_expr = make_scalar_expression(rhs, lhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::divides>>(std::forward<LHS>(lhs), std::move(scalar_expr));
    }

    // Scalar-Expression operations (scalar on left)
    template<typename RHS>
    auto operator+(double lhs, RHS&& rhs) ->
        std::enable_if_t<is_expression_v<RHS> && !is_matrix_type<std::decay_t<RHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::plus>, ScalarExpression<double>, std::decay_t<RHS>>> {
        auto scalar_expr = make_scalar_expression(lhs, rhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::plus>>(std::move(scalar_expr), std::forward<RHS>(rhs));
    }

    template<typename RHS>
    auto operator-(double lhs, RHS&& rhs) ->
        std::enable_if_t<is_expression_v<RHS> && !is_matrix_type<std::decay_t<RHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::minus>, ScalarExpression<double>, std::decay_t<RHS>>> {
        auto scalar_expr = make_scalar_expression(lhs, rhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::minus>>(std::move(scalar_expr), std::forward<RHS>(rhs));
    }

    template<typename RHS>
    auto operator*(double lhs, RHS&& rhs) ->
        std::enable_if_t<is_expression_v<RHS> && !is_matrix_type<std::decay_t<RHS>>::value &&
                         !is_sparse_vector_type<std::decay_t<RHS>>::value &&
                         !is_tensor_type<std::decay_t<RHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::multiplies>, ScalarExpression<double>, std::decay_t<RHS>>> {
        auto scalar_expr = make_scalar_expression(lhs, rhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::multiplies>>(std::move(scalar_expr), std::forward<RHS>(rhs));
    }

    template<typename RHS>
    auto operator/(double lhs, RHS&& rhs) ->
        std::enable_if_t<is_expression_v<RHS> && !is_matrix_type<std::decay_t<RHS>>::value,
                         BinaryExpression<TypedOpWrapper<ops::divides>, ScalarExpression<double>, std::decay_t<RHS>>> {
        auto scalar_expr = make_scalar_expression(lhs, rhs.shape());
        return make_binary_expression<TypedOpWrapper<ops::divides>>(std::move(scalar_expr), std::forward<RHS>(rhs));
    }
    // ============================================================
    // Unary operators with perfect forwarding
    // ============================================================

    template<typename Expr>
    auto operator-(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::negate_wrapper>> {
        return make_unary_expression<expr_ops::negate_wrapper>(std::forward<Expr>(expr));
    }

    // ============================================================
    // Unary mathematical functions with perfect forwarding
    // ============================================================

    template<typename Expr>
    auto abs(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::abs_wrapper>> {
        return make_unary_expression<expr_ops::abs_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto sign(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::sign_wrapper>> {
        return make_unary_expression<expr_ops::sign_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto sqrt(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::sqrt_wrapper>> {
        return make_unary_expression<expr_ops::sqrt_wrapper>(std::forward<Expr>(expr));
    }

    // Exponential and logarithmic functions
    template<typename Expr>
    auto exp(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::exp_wrapper>> {
        return make_unary_expression<expr_ops::exp_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto log(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::log_wrapper>> {
        return make_unary_expression<expr_ops::log_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto log10(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::log10_wrapper>> {
        return make_unary_expression<expr_ops::log10_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto log2(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::log2_wrapper>> {
        return make_unary_expression<expr_ops::log2_wrapper>(std::forward<Expr>(expr));
    }

    // Trigonometric functions
    template<typename Expr>
    auto sin(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::sin_wrapper>> {
        return make_unary_expression<expr_ops::sin_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto cos(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::cos_wrapper>> {
        return make_unary_expression<expr_ops::cos_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto tan(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::tan_wrapper>> {
        return make_unary_expression<expr_ops::tan_wrapper>(std::forward<Expr>(expr));
    }

    // Inverse trigonometric functions
    template<typename Expr>
    auto asin(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::asin_wrapper>> {
        return make_unary_expression<expr_ops::asin_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto acos(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::acos_wrapper>> {
        return make_unary_expression<expr_ops::acos_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto atan(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::atan_wrapper>> {
        return make_unary_expression<expr_ops::atan_wrapper>(std::forward<Expr>(expr));
    }

    // Hyperbolic functions
    template<typename Expr>
    auto sinh(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::sinh_wrapper>> {
        return make_unary_expression<expr_ops::sinh_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto cosh(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::cosh_wrapper>> {
        return make_unary_expression<expr_ops::cosh_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto tanh(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::tanh_wrapper>> {
        return make_unary_expression<expr_ops::tanh_wrapper>(std::forward<Expr>(expr));
    }

    // Rounding functions
    template<typename Expr>
    auto round(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::round_wrapper>> {
        return make_unary_expression<expr_ops::round_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto floor(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::floor_wrapper>> {
        return make_unary_expression<expr_ops::floor_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto ceil(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::ceil_wrapper>> {
        return make_unary_expression<expr_ops::ceil_wrapper>(std::forward<Expr>(expr));
    }

    template<typename Expr>
    auto trunc(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, expr_ops::trunc_wrapper>> {
        return make_unary_expression<expr_ops::trunc_wrapper>(std::forward<Expr>(expr));
    }

    // ============================================================
    // Binary mathematical functions with perfect forwarding
    // ============================================================

    template<typename LHS, typename RHS>
    auto pow(LHS&& base, RHS&& exp) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<expr_ops::pow_wrapper, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<expr_ops::pow_wrapper>(std::forward<LHS>(base), std::forward<RHS>(exp));
    }

    template<typename LHS, typename RHS>
    auto atan2(LHS&& y, RHS&& x) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<expr_ops::atan2_wrapper, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<expr_ops::atan2_wrapper>(std::forward<LHS>(y), std::forward<RHS>(x));
    }

    template<typename LHS, typename RHS>
    auto hypot(LHS&& x, RHS&& y) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<expr_ops::hypot_wrapper, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<expr_ops::hypot_wrapper>(std::forward<LHS>(x), std::forward<RHS>(y));
    }

    template<typename LHS, typename RHS>
    auto min(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<expr_ops::min_wrapper, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<expr_ops::min_wrapper>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto max(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<expr_ops::max_wrapper, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<expr_ops::max_wrapper>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    // ============================================================
    // Comparison operations with perfect forwarding
    // ============================================================

    template<typename LHS, typename RHS>
    auto equal(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::equal_to<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::equal_to<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto not_equal(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::not_equal_to<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::not_equal_to<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto less(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::less<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::less<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto less_equal(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::less_equal<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::less_equal<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto greater(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::greater<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::greater<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto greater_equal(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::greater_equal<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::greater_equal<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    // ============================================================
    // Logical operations with perfect forwarding
    // ============================================================


    template<typename Expr>
    auto logical_not(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, ops::logical_not>> {
        return make_unary_expression<ops::logical_not>(std::forward<Expr>(expr));
    }

    template<typename LHS, typename RHS>
    auto logical_and(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::logical_and, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::logical_and>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto logical_or(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::logical_or, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::logical_or>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    // ============================================================
    // Bitwise operations with perfect forwarding (for integral types)
    // ============================================================

    template<typename LHS, typename RHS>
    auto logical_xor(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::logical_xor, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::logical_xor>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto bit_and(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::bit_and<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::bit_and<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto bit_or(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::bit_or<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::bit_or<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto bit_xor(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::bit_xor<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::bit_xor<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename Expr>
    auto bit_not(Expr&& expr) ->
        std::enable_if_t<is_expression_v<Expr>,
                         UnaryExpression<std::decay_t<Expr>, ops::bit_not<>>> {
        return make_unary_expression<ops::bit_not<>>(std::forward<Expr>(expr));
    }

    template<typename LHS, typename RHS>
    auto left_shift(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::left_shift<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::left_shift<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

    template<typename LHS, typename RHS>
    auto right_shift(LHS&& lhs, RHS&& rhs) ->
        std::enable_if_t<(is_expression_v<LHS> || is_expression_v<RHS>),
                         BinaryExpression<ops::right_shift<>, std::decay_t<LHS>, std::decay_t<RHS>>> {
        return make_binary_expression<ops::right_shift<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    }

} // namespace fem::numeric

#endif //NUMERIC_EXPRESSION_BASE_H

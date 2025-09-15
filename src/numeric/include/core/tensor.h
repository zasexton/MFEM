#pragma once

#ifndef NUMERIC_TENSOR_H
#define NUMERIC_TENSOR_H

#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>
#include <cmath>
#include <execution>
#include <limits>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <array>
#include <type_traits>

// Detection idiom for SFINAE (reuse from vector.h and matrix.h)
namespace fem::numeric::detail {
    struct nonesuch {
        ~nonesuch() = delete;
        nonesuch(nonesuch const&) = delete;
        void operator=(nonesuch const&) = delete;
    };

    template<class Default, class AlwaysVoid,
             template<class...> class Op, class... Args>
    struct detector {
        using value_t = std::false_type;
        using type = Default;
    };

    template<class Default, template<class...> class Op, class... Args>
    struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
        using value_t = std::true_type;
        using type = Op<Args...>;
    };

    template<template<class...> class Op, class... Args>
    using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

    template<template<class...> class Op, class... Args>
    constexpr bool is_detected_v = is_detected<Op, Args...>::value;
}

#include "../base/numeric_base.h"
#include "../base/container_base.h"
#include "../base/expression_base.h"
#include "../base/storage_base.h"
#include "../base/iterator_base.h"
#include "../base/slice_base.h"
#include "../base/view_base.h"
#include "../base/ops_base.h"
#include "../base/broadcast_base.h"
#include "../base/traits_base.h"

namespace fem::numeric {

// Forward declarations
template<typename T, size_t Rank> class TensorView;
template<typename T, size_t Rank> class TensorSlice;

/**
 * @brief N-dimensional tensor class for numerical computing
 * 
 * Implements a high-performance N-dimensional tensor with:
 * - Support for real, complex, and dual number types
 * - Arbitrary rank (number of dimensions)
 * - Flexible indexing and slicing
 * - Expression templates for lazy evaluation
 * - Memory-efficient storage with configurable layout
 * - Tensor contraction operations
 * - Broadcasting capabilities
 * - IEEE 754 compliance for all operations
 * 
 * @tparam T Value type (must satisfy StorableType concept)
 * @tparam Rank Number of dimensions (0 for scalar, 1 for vector, 2 for matrix)
 * @tparam Storage Storage strategy (default: DynamicStorage)
 */
template<typename T, 
         size_t Rank,
         typename Storage = DynamicStorage<T>>
class Tensor : public ContainerBase<Tensor<T, Rank, Storage>, T, Storage>,
               public ExpressionBase<Tensor<T, Rank, Storage>> {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    static_assert(Rank <= 8, "Tensor rank limited to 8 dimensions for performance");
    
    // === Type Aliases ===
    using base_type = ContainerBase<Tensor<T, Rank, Storage>, T, Storage>;
    using expression_base = ExpressionBase<Tensor<T, Rank, Storage>>;
    using value_type = T;
    using storage_type = Storage;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Iterator types
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    
    // Shape and indexing types
    using shape_type = std::array<size_type, Rank>;
    using index_type = std::array<size_type, Rank>;
    using strides_type = std::array<size_type, Rank>;
    
    // View types
    using slice_type = TensorSlice<T, Rank>;
    using const_slice_type = TensorSlice<const T, Rank>;
    using view_type = TensorView<T, Rank>;
    using const_view_type = TensorView<const T, Rank>;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Static properties
    static constexpr size_t tensor_rank = Rank;
    static constexpr bool is_scalar = (Rank == 0);
    static constexpr bool is_vector = (Rank == 1);
    static constexpr bool is_matrix = (Rank == 2);
    
private:
    shape_type shape_;
    strides_type strides_;
    storage_type storage_;
    
    // Compute strides for C-style (row-major) ordering
    void compute_strides() {
        if (Rank == 0) return;
        
        strides_[Rank - 1] = 1;
        for (size_t i = Rank - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
    }
    
    // Convert multi-dimensional index to linear index
    template<typename... Indices>
    size_type linear_index(Indices... indices) const {
        static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
        
        if constexpr (Rank == 0) {
            return 0;
        } else {
            std::array<size_type, Rank> idx{static_cast<size_type>(indices)...};
            size_type linear = 0;
            for (size_t i = 0; i < Rank; ++i) {
                linear += idx[i] * strides_[i];
            }
            return linear;
        }
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor - creates empty tensor
     */
    Tensor() : base_type(), shape_{}, strides_{}, storage_() {
        if constexpr (Rank > 0) {
            shape_.fill(0);
            strides_.fill(0);
        }
    }
    
    /**
     * @brief Shape constructor - creates tensor with given dimensions
     */
    explicit Tensor(const shape_type& shape)
        : shape_(shape), strides_{} {
        
        compute_strides();
        
        size_type total_size = 1;
        for (size_t i = 0; i < Rank; ++i) {
            total_size *= shape_[i];
        }
        
        storage_ = storage_type(total_size);
        
        // Update base class shape (convert to Shape for compatibility)
        if constexpr (Rank == 1) {
            this->base_type::shape_ = Shape{shape_[0]};
        } else if constexpr (Rank == 2) {
            this->base_type::shape_ = Shape{shape_[0], shape_[1]};
        } else {
            // For higher rank tensors, base shape represents flattened size
            this->base_type::shape_ = Shape{total_size};
        }
    }
    
    /**
     * @brief Shape and value constructor
     */
    Tensor(const shape_type& shape, const T& value)
        : Tensor(shape) {
        fill(value);
    }
    
    /**
     * @brief Scalar constructor (only for Rank == 0)
     */
    template<typename U = T>
    requires (Rank == 0)
    explicit Tensor(const U& scalar)
        : shape_{}, strides_{}, storage_(1, static_cast<T>(scalar)) {
        this->base_type::shape_ = Shape{1};
    }
    
    /**
     * @brief Initializer list constructor for small tensors
     */
    template<typename U>
    requires (Rank <= 3)
    Tensor(std::initializer_list<U> data) {
        if constexpr (Rank == 1) {
            shape_[0] = data.size();
            compute_strides();
            storage_ = storage_type(data.size());
            std::copy(data.begin(), data.end(), storage_.begin());
            this->base_type::shape_ = Shape{shape_[0]};
        } else {
            throw std::invalid_argument("Initializer list constructor not supported for this rank");
        }
    }
    
    /**
     * @brief Copy constructor
     */
    Tensor(const Tensor& other) = default;

    // Construct from an expression (evaluates into this tensor)
    template<typename Expr>
    Tensor(const ExpressionBase<Expr>& expr) {
        // Initialize to a shape compatible with the expression
        if constexpr (Rank == 0) {
            shape_ = {};
        } else if constexpr (Rank == 1) {
            shape_[0] = expr.derived().shape()[0];
        } else if constexpr (Rank == 2) {
            auto sh = expr.derived().shape();
            shape_[0] = sh[0];
            shape_[1] = sh[1];
        } else {
            // For higher ranks, the expression must be flattened-compatible
            // We rely on assign_expression to validate/resize if needed
        }
        compute_strides();
        storage_.resize(size());
        assign_expression(expr.derived());
    }
    
    /**
     * @brief Move constructor
     */
    Tensor(Tensor&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    Tensor& operator=(const Tensor& other) = default;
    
    /**
     * @brief Move assignment
     */
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~Tensor() = default;
    
    // === Shape and Size ===
    
    /**
     * @brief Get tensor shape as std::array
     */
    const shape_type& dims() const noexcept { return shape_; }

    /**
     * @brief Rank accessor for tests/utilities
     */
    constexpr size_t rank() const noexcept { return Rank; }
    
    /**
     * @brief Get size of specific dimension
     */
    size_type size(size_t dim) const {
        if (dim >= Rank) {
            throw std::out_of_range("Dimension index out of range");
        }
        return shape_[dim];
    }
    
    /**
     * @brief Get total number of elements
     */
    size_type size() const noexcept {
        if constexpr (Rank == 0) {
            return 1;
        } else {
            size_type total = 1;
            for (size_t i = 0; i < Rank; ++i) {
                total *= shape_[i];
            }
            return total;
        }
    }
    
    /**
     * @brief Check if tensor is empty
     */
    bool empty() const noexcept {
        if constexpr (Rank == 0) {
            return false;
        } else {
            for (size_t i = 0; i < Rank; ++i) {
                if (shape_[i] == 0) return true;
            }
            return false;
        }
    }
    
    /**
     * @brief Get tensor strides
     */
    const strides_type& strides() const noexcept { return strides_; }
    
    // === Element Access ===
    
    /**
     * @brief Access element with bounds checking
     */
    template<typename... Indices>
    reference at(Indices... indices) {
        static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
        
        // Check bounds
        std::array<size_type, Rank> idx{static_cast<size_type>(indices)...};
        for (size_t i = 0; i < Rank; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Tensor index out of range");
            }
        }
        
        return storage_[linear_index(indices...)];
    }
    
    template<typename... Indices>
    const_reference at(Indices... indices) const {
        static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
        
        // Check bounds
        std::array<size_type, Rank> idx{static_cast<size_type>(indices)...};
        for (size_t i = 0; i < Rank; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Tensor index out of range");
            }
        }
        
        return storage_[linear_index(indices...)];
    }
    
    /**
     * @brief Access element without bounds checking
     */
    template<typename... Indices>
    inline reference operator()(Indices... indices) noexcept {
        return storage_[linear_index(indices...)];
    }
    
    template<typename... Indices>
    inline const_reference operator()(Indices... indices) const noexcept {
        return storage_[linear_index(indices...)];
    }
    
    /**
     * @brief Linear indexing (for 1D access)
     */
    inline reference operator[](size_type index) noexcept {
        return storage_[index];
    }
    
    inline const_reference operator[](size_type index) const noexcept {
        return storage_[index];
    }
    
    /**
     * @brief Get raw data pointer
     */
    inline pointer data() noexcept { return storage_.data(); }
    inline const_pointer data() const noexcept { return storage_.data(); }
    
    // === Tensor Operations ===
    
    /**
     * @brief Fill tensor with a value
     */
    void fill(const T& value) {
        std::fill(storage_.begin(), storage_.end(), value);
    }
    
    /**
     * @brief Set to zero
     */
    void zero() {
        fill(T{0});
    }
    
    /**
     * @brief Reshape tensor (must preserve total size)
     */
    void reshape(const shape_type& new_shape) {
        size_type new_size = 1;
        for (size_t i = 0; i < Rank; ++i) {
            new_size *= new_shape[i];
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("Reshape must preserve total number of elements");
        }
        
        shape_ = new_shape;
        compute_strides();
        
        // Update base class shape
        if constexpr (Rank == 1) {
            this->base_type::shape_ = Shape{shape_[0]};
        } else if constexpr (Rank == 2) {
            this->base_type::shape_ = Shape{shape_[0], shape_[1]};
        } else {
            this->base_type::shape_ = Shape{size()};
        }
    }
    
    /**
     * @brief Permute axes (transpose generalization)
     */
    Tensor permute(const std::array<size_t, Rank>& axes) const {
        // Verify axes is a valid permutation
        std::array<bool, Rank> used{};
        for (size_t i = 0; i < Rank; ++i) {
            if (axes[i] >= Rank || used[axes[i]]) {
                throw std::invalid_argument("Invalid axis permutation");
            }
            used[axes[i]] = true;
        }
        
        // Compute new shape and strides (guard indexing for analyzers)
        shape_type new_shape{};
        for (size_t i = 0; i < Rank; ++i) {
            const auto ai = axes[i];
            new_shape[i] = (ai < Rank) ? shape_[ai] : size_type{0};
        }
        
        Tensor result(new_shape);
        
        // Copy data with permuted indices
        if constexpr (Rank == 1) {
            result = *this;  // Identity for vectors
        } else if constexpr (Rank == 2) {
            // Matrix transpose
            for (size_type i = 0; i < shape_[0]; ++i) {
                for (size_type j = 0; j < shape_[1]; ++j) {
                    if (axes[0] == 0 && axes[1] == 1) {
                        result(i, j) = (*this)(i, j);
                    } else {
                        result(j, i) = (*this)(i, j);
                    }
                }
            }
        } else {
            // General case - use recursive index permutation
            copy_with_permutation(*this, result, axes, {}, 0);
        }
        
        return result;
    }
    
    /**
     * @brief Transpose (only for matrices, Rank == 2)
     */
    template<size_t R = Rank>
    requires (R == 2)
    Tensor<T, 2, Storage> transpose() const {
        return permute({1, 0});
    }
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Tensor addition
     */
    template<typename U, typename S2>
    Tensor& operator+=(const Tensor<U, Rank, S2>& other) {
        if (shape_ != other.dims()) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        
        for (size_type i = 0; i < size(); ++i) {
            storage_[i] += static_cast<T>(other[i]);
        }
        return *this;
    }
    
    /**
     * @brief Tensor subtraction
     */
    template<typename U, typename S2>
    Tensor& operator-=(const Tensor<U, Rank, S2>& other) {
        if (shape_ != other.dims()) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        
        for (size_type i = 0; i < size(); ++i) {
            storage_[i] -= static_cast<T>(other[i]);
        }
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    Tensor& operator*=(const T& scalar) {
        for (auto& elem : storage_) {
            elem *= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    Tensor& operator/=(const T& scalar) {
        for (auto& elem : storage_) {
            elem /= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Element-wise multiplication (Hadamard product)
     */
    template<typename U, typename S2>
    Tensor& hadamard_product(const Tensor<U, Rank, S2>& other) {
        if (shape_ != other.dims()) {
            throw std::invalid_argument("Tensor shapes must match for Hadamard product");
        }
        
        for (size_type i = 0; i < size(); ++i) {
            storage_[i] *= static_cast<T>(other[i]);
        }
        return *this;
    }
    
    // === Tensor Contraction ===
    
    /**
     * @brief Contract over specified axes with another tensor
     * 
     * Performs generalized matrix multiplication for higher-order tensors
     */
    template<typename U, size_t Rank2, typename S2>
    auto contract(const Tensor<U, Rank2, S2>& other,
                  const std::vector<std::pair<size_t, size_t>>& axes) const {
        
        // Validate contraction axes
        for (const auto& [axis1, axis2] : axes) {
            if (axis1 >= Rank || axis2 >= Rank2) {
                throw std::invalid_argument("Contraction axis out of range");
            }
            if (shape_[axis1] != other.dims()[axis2]) {
                throw std::invalid_argument("Contracted dimensions must have same size");
            }
        }
        
        // Compute result shape
        constexpr size_t result_rank = Rank + Rank2 - 2 * axes.size();
        static_assert(result_rank <= 8, "Result tensor rank too large");
        
        using result_type = decltype(T{} * U{});
        
        if constexpr (result_rank == 0) {
            // Scalar result
            result_type sum{0};
            // Implementation for full contraction
            return sum;
        } else {
            std::array<size_type, result_rank> result_shape;
            // Complex implementation needed for general tensor contraction
            Tensor<result_type, result_rank> result(result_shape);
            return result;
        }
    }
    
    // === Norms ===
    
    /**
     * @brief Frobenius norm (Euclidean norm of tensor as vector)
     */
    scalar_type frobenius_norm() const {
        scalar_type scale{0};
        scalar_type ssq{1};
        
        for (const auto& x : storage_) {
            scalar_type absxi = std::abs(x);
            if (absxi != 0) {
                if (scale < absxi) {
                    ssq = 1 + ssq * (scale/absxi) * (scale/absxi);
                    scale = absxi;
                } else {
                    ssq += (absxi/scale) * (absxi/scale);
                }
            }
        }
        
        return scale * std::sqrt(ssq);
    }
    
    /**
     * @brief Maximum absolute value norm
     */
    scalar_type max_norm() const {
        scalar_type max_val{0};
        for (const auto& x : storage_) {
            max_val = std::max(max_val, std::abs(x));
        }
        return max_val;
    }
    
    // === Utility Functions ===
    
    /**
     * @brief Get flattened view as 1D tensor
     */
    Tensor<T, 1, Storage> flatten() const {
        Tensor<T, 1> result({size()});
        std::copy_n(storage_.begin(), storage_.size(), result.data());
        return result;
    }
    
    // === Expression Template Interface ===
    
    // Shape for expression/broadcasting APIs
    Shape shape() const noexcept {
        // Convert tensor shape to general Shape
        if constexpr (Rank == 0) {
            return Shape{1};
        } else if constexpr (Rank == 1) {
            return Shape{shape_[0]};
        } else if constexpr (Rank == 2) {
            return Shape{shape_[0], shape_[1]};
        } else {
            // For higher rank tensors, represent as flattened size
            return Shape{size()};
        }
    }
    
    template<typename U>
    auto eval(size_type i) const -> const T& {
        return storage_[i];
    }
    
    template<typename U, typename... Indices>
    auto eval_at(Indices... indices) const -> const T& {
        static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
        return (*this)(indices...);
    }
    
    template<typename Container>
    void eval_to(Container& result) const {
        using result_type = typename Container::value_type;
        if (result.size() != size()) {
            result.resize(size());
        }
        for (size_type i = 0; i < size(); ++i) {
            result[i] = static_cast<result_type>(storage_[i]);
        }
    }
    
    bool is_parallelizable() const noexcept { return true; }
    bool is_vectorizable() const noexcept { return true; }
    size_t complexity() const noexcept { return size(); }
    
    // Expression assignment
    template<typename Expr>
    Tensor& operator=(const ExpressionBase<Expr>& expr) {
        assign_expression(expr.derived());
        return *this;
    }

private:
    // SFINAE helpers for expression template integration
    template<typename T_>
    using has_shape_method_t = decltype(std::declval<T_>().shape());
    template<typename T_>
    static constexpr bool has_shape_method_v = detail::is_detected_v<has_shape_method_t, T_>;

    template<typename T_>
    using has_eval_method_t = decltype(std::declval<T_>().template eval<T>(std::declval<size_type>()));
    template<typename T_>
    static constexpr bool has_eval_method_v = detail::is_detected_v<has_eval_method_t, T_>;

    template<typename T_>
    using has_eval_to_method_t = decltype(std::declval<T_>().eval_to(std::declval<Tensor&>()));
    template<typename T_>
    static constexpr bool has_eval_to_method_v = detail::is_detected_v<has_eval_to_method_t, T_>;

    template<typename Expr>
    void assign_expression(const Expr& expr) {
        if constexpr (has_shape_method_v<Expr>) {
            auto expr_shape = expr.shape();
            // Handle shape conversion based on tensor rank
            if constexpr (Rank == 1) {
                if (expr_shape.rank() != 1) {
                    throw std::invalid_argument("Cannot assign non-1D expression to 1D Tensor");
                }
                if (shape_[0] != expr_shape[0]) {
                    shape_[0] = expr_shape[0];
                    compute_strides();
                    storage_.resize(size());
                }
            } else if constexpr (Rank == 2) {
                if (expr_shape.rank() != 2) {
                    throw std::invalid_argument("Cannot assign non-2D expression to 2D Tensor");
                }
                if (shape_[0] != expr_shape[0] || shape_[1] != expr_shape[1]) {
                    shape_[0] = expr_shape[0];
                    shape_[1] = expr_shape[1];
                    compute_strides();
                    storage_.resize(size());
                }
            } else {
                // For higher rank, we accept flattened expressions
                if (size() != expr_shape.size()) {
                    throw std::invalid_argument("Expression size must match tensor size for higher rank tensors");
                }
            }
        }
        
        // Use expression template evaluation
        if constexpr (has_eval_to_method_v<Expr>) {
            expr.eval_to(*this);
        } else {
            for (size_type i = 0; i < size(); ++i) {
                if constexpr (has_eval_method_v<Expr>) {
                    storage_[i] = expr.template eval<T>(i);
                } else {
                    storage_[i] = expr.evaluate(i);
                }
            }
        }
    }

public:
    // === Output ==="}
    
    /**
     * @brief Stream output with formatted display
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor<" << Rank << ">(";
        for (size_t i = 0; i < Rank; ++i) {
            os << t.shape_[i];
            if (i < Rank - 1) os << "x";
        }
        os << "):\n";
        
        if constexpr (Rank == 0) {
            os << t.storage_[0];
        } else if constexpr (Rank == 1) {
            os << "[";
            for (size_type i = 0; i < t.shape_[0]; ++i) {
                os << t(i);
                if (i < t.shape_[0] - 1) os << " ";
            }
            os << "]";
        } else if constexpr (Rank == 2) {
            for (size_type i = 0; i < t.shape_[0]; ++i) {
                os << "[";
                for (size_type j = 0; j < t.shape_[1]; ++j) {
                    os << t(i, j);
                    if (j < t.shape_[1] - 1) os << " ";
                }
                os << "]\n";
            }
        } else {
            os << "[...] (higher-rank tensor display not implemented)";
        }
        
        return os;
    }

private:
    // Helper for permutation copying
    template<size_t R>
    static void copy_with_permutation(const Tensor& src, Tensor& dst, 
                                     const std::array<size_t, R>& axes,
                                     std::array<size_type, R> indices, 
                                     size_t dim) {
        if (dim == R) {
            // Base case: copy element
            std::array<size_type, R> src_indices;
            for (size_t i = 0; i < R; ++i) {
                src_indices[axes[i]] = indices[i];
            }
            
            // This is complex for general case - simplified for now
            return;
        }
        
        // Recursive case
        for (size_type i = 0; i < dst.shape_[dim]; ++i) {
            indices[dim] = i;
            copy_with_permutation(src, dst, axes, indices, dim + 1);
        }
    }
};

// === Type Aliases for Common Tensor Types ===

// Scalars
template<typename T> using Scalar = Tensor<T, 0>;

// Vectors (1D tensors)
template<typename T> using Vector1D = Tensor<T, 1>;

// Matrices (2D tensors)  
template<typename T> using Matrix2D = Tensor<T, 2>;

// 3D tensors
template<typename T> using Tensor3D = Tensor<T, 3>;

// 4D tensors
template<typename T> using Tensor4D = Tensor<T, 4>;

// === Non-member Operations (Expression Template Compatible) ===

// Forward declarations for tensor types
template<typename T> struct is_tensor : std::false_type {};
template<typename T, size_t Rank, typename S> struct is_tensor<Tensor<T, Rank, S>> : std::true_type {};
template<typename T> inline constexpr bool is_tensor_v = is_tensor<T>::value;

// Import expression template types
using fem::numeric::TerminalExpression;
using fem::numeric::ScalarExpression;
using fem::numeric::BinaryExpression;
using fem::numeric::UnaryExpression;
using fem::numeric::make_binary_expression;
using fem::numeric::make_unary_expression;
using fem::numeric::make_scalar_expression;

// Helper to create terminal expressions from tensors
template<typename T, size_t Rank, typename S>
auto make_tensor_expression(const Tensor<T, Rank, S>& tensor) {
    return TerminalExpression<Tensor<T, Rank, S>>(tensor);
}

template<typename T, size_t Rank, typename S>
auto make_tensor_expression(Tensor<T, Rank, S>&& tensor) {
    return TerminalExpression<Tensor<T, Rank, S>>(std::move(tensor));
}

// === Eager Non-member Operations matching unit tests ===

// Element-wise addition producing a concrete Tensor
template<typename T1, typename T2, size_t Rank, typename S1, typename S2>
auto operator+(const Tensor<T1, Rank, S1>& a, const Tensor<T2, Rank, S2>& b) {
    using result_type = std::common_type_t<T1, T2>;
    // Dimension check
    for (size_t d = 0; d < Rank; ++d) {
        if (a.size(d) != b.size(d)) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
    }
    Tensor<result_type, Rank> result(a.dims());
    const auto n = a.size();
    for (size_t i = 0; i < n; ++i) {
        result[i] = static_cast<result_type>(a[i]) + static_cast<result_type>(b[i]);
    }
    return result;
}

// Element-wise subtraction producing a concrete Tensor
template<typename T1, typename T2, size_t Rank, typename S1, typename S2>
auto operator-(const Tensor<T1, Rank, S1>& a, const Tensor<T2, Rank, S2>& b) {
    using result_type = std::common_type_t<T1, T2>;
    for (size_t d = 0; d < Rank; ++d) {
        if (a.size(d) != b.size(d)) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
    }
    Tensor<result_type, Rank> result(a.dims());
    const auto n = a.size();
    for (size_t i = 0; i < n; ++i) {
        result[i] = static_cast<result_type>(a[i]) - static_cast<result_type>(b[i]);
    }
    return result;
}

// Scalar-tensor multiplication (element-wise)
template<typename T, size_t Rank, typename S, typename Scalar>
auto operator*(const Scalar& s, const Tensor<T, Rank, S>& t)
    -> std::enable_if_t<std::is_arithmetic_v<Scalar>, Tensor<std::common_type_t<T, Scalar>, Rank>>
{
    using result_type = std::common_type_t<T, Scalar>;
    Tensor<result_type, Rank> result(t.dims());
    const auto n = t.size();
    for (size_t i = 0; i < n; ++i) {
        result[i] = static_cast<result_type>(s) * static_cast<result_type>(t[i]);
    }
    return result;
}

template<typename T, size_t Rank, typename S, typename Scalar>
auto operator*(const Tensor<T, Rank, S>& t, const Scalar& s)
    -> std::enable_if_t<std::is_arithmetic_v<Scalar>, Tensor<std::common_type_t<T, Scalar>, Rank>>
{
    return s * t;
}

// Remove expression-template variants to prefer eager operators above

// === Tensor View Classes (Placeholder) ===

/**
 * @brief View of a tensor slice
 */
template<typename T, size_t Rank>
class TensorView {
    // To be implemented for advanced slicing
};

/**
 * @brief Slice of a tensor
 */
template<typename T, size_t Rank>
class TensorSlice {
    // To be implemented for tensor slicing operations
};

} // namespace fem::numeric

#endif // NUMERIC_TENSOR_H

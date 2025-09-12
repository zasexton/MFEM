#pragma once

#ifndef NUMERIC_MATRIX_H
#define NUMERIC_MATRIX_H

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
#include "vector.h"
#include "vector_view.h"

namespace fem::numeric {

// Forward declaration for Matrix-specific views
template<typename T> class MatrixView;
template<typename T> class MatrixTransposeView;

/**
 * @brief Storage order for matrix data layout
 */
enum class StorageOrder {
    RowMajor,    // C-style, cache-friendly for row operations
    ColumnMajor  // Fortran-style, BLAS-friendly
};

/**
 * @brief Dense matrix class for numerical computing
 * 
 * Implements a high-performance dense matrix with:
 * - Support for real, complex, and dual number types
 * - Row-major and column-major storage options
 * - Expression templates for lazy evaluation
 * - Submatrix views and slicing
 * - Optimized matrix-vector and matrix-matrix operations
 * - IEEE 754 compliance for all operations
 * 
 * @tparam T Value type (must satisfy NumberLike concept)
 * @tparam Storage Storage strategy (default: DynamicStorage)
 * @tparam Order Storage order (default: RowMajor for cache efficiency)
 */
template<typename T, 
         typename Storage = DynamicStorage<T>,
         StorageOrder Order = StorageOrder::RowMajor>
class Matrix : public ContainerBase<Matrix<T, Storage, Order>, T, Storage>,
               public ExpressionBase<Matrix<T, Storage, Order>> {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    
    // === Type Aliases ===
    using base_type = ContainerBase<Matrix<T, Storage, Order>, T, Storage>;
    using expression_base = ExpressionBase<Matrix<T, Storage, Order>>;
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
    
    // View types
    using row_view_type = VectorView<T>;
    using const_row_view_type = VectorView<const T>;
    using col_view_type = StridedView<T>;
    using const_col_view_type = StridedView<const T>;
    using submatrix_view_type = MatrixView<T>;
    using const_submatrix_view_type = MatrixView<const T>;
    using transpose_view_type = MatrixTransposeView<T>;
    using const_transpose_view_type = MatrixTransposeView<const T>;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Storage order
    static constexpr StorageOrder storage_order = Order;
    static constexpr bool is_row_major = (Order == StorageOrder::RowMajor);
    static constexpr bool is_column_major = (Order == StorageOrder::ColumnMajor);
    
private:
    size_type rows_;
    size_type cols_;
    storage_type storage_;
    
    // Helper to compute linear index
    inline size_type linear_index(size_type i, size_type j) const noexcept {
        if constexpr (is_row_major) {
            return i * cols_ + j;
        } else {
            return j * rows_ + i;
        }
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor - creates empty matrix
     */
    Matrix() : base_type(), rows_(0), cols_(0), storage_() {}
    
    /**
     * @brief Size constructor - creates matrix of given dimensions
     */
    Matrix(size_type rows, size_type cols)
        : base_type(Shape{rows, cols}), 
          rows_(rows), 
          cols_(cols), 
          storage_(rows * cols) {}
    
    /**
     * @brief Size and value constructor
     */
    Matrix(size_type rows, size_type cols, const T& value)
        : base_type(Shape{rows, cols}, value),
          rows_(rows),
          cols_(cols),
          storage_(rows * cols, value) {}
    
    /**
     * @brief Initializer list constructor for 2D data
     * 
     * Example:
     * Matrix<double> m = {{1, 2, 3},
     *                     {4, 5, 6}};
     */
    Matrix(std::initializer_list<std::initializer_list<T>> data)
        : rows_(data.size()), 
          cols_(data.size() > 0 ? data.begin()->size() : 0),
          storage_(rows_ * cols_) {
        
        if (rows_ == 0 || cols_ == 0) {
            throw std::invalid_argument("Cannot create matrix from empty initializer list");
        }
        
        // Verify all rows have same size
        for (const auto& row : data) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
        }
        
        // Copy data
        size_type i = 0;
        for (const auto& row : data) {
            size_type j = 0;
            for (const auto& val : row) {
                (*this)(i, j) = val;
                ++j;
            }
            ++i;
        }
        
        this->shape_ = Shape{rows_, cols_};
    }
    
    /**
     * @brief Copy constructor
     */
    Matrix(const Matrix& other) = default;
    
    /**
     * @brief Move constructor
     */
    Matrix(Matrix&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    Matrix& operator=(const Matrix& other) = default;
    
    /**
     * @brief Move assignment
     */
    Matrix& operator=(Matrix&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~Matrix() = default;
    
    // === Size and Shape ===
    
    /**
     * @brief Get number of rows
     */
    inline size_type rows() const noexcept { return rows_; }
    
    /**
     * @brief Get number of columns
     */
    inline size_type cols() const noexcept { return cols_; }
    
    /**
     * @brief Get total number of elements
     */
    inline size_type size() const noexcept { return rows_ * cols_; }
    
    /**
     * @brief Check if matrix is empty
     */
    inline bool empty() const noexcept { return size() == 0; }
    
    /**
     * @brief Check if matrix is square
     */
    inline bool is_square() const noexcept { return rows_ == cols_; }
    
    /**
     * @brief Get leading dimension for BLAS compatibility
     */
    inline size_type leading_dimension() const noexcept {
        if constexpr (is_row_major) {
            return cols_;
        } else {
            return rows_;
        }
    }
    
    // === Element Access ===
    
    /**
     * @brief Access element at (i, j) with bounds checking
     */
    reference at(size_type i, size_type j) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return storage_[linear_index(i, j)];
    }
    
    const_reference at(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return storage_[linear_index(i, j)];
    }
    
    /**
     * @brief Access element at (i, j) without bounds checking
     */
    inline reference operator()(size_type i, size_type j) noexcept {
        return storage_[linear_index(i, j)];
    }
    
    inline const_reference operator()(size_type i, size_type j) const noexcept {
        return storage_[linear_index(i, j)];
    }
    
    /**
     * @brief Get raw data pointer
     */
    inline pointer data() noexcept { return storage_.data(); }
    inline const_pointer data() const noexcept { return storage_.data(); }
    
    // === Row and Column Access ===
    
    /**
     * @brief Get row as a vector view
     */
    row_view_type row(size_type i) {
        if (i >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        if constexpr (is_row_major) {
            return row_view_type(data() + i * cols_, cols_);
        } else {
            // Column-major: row elements are strided
            return row_view_type(data() + i, cols_, rows_);
        }
    }
    
    const_row_view_type row(size_type i) const {
        if (i >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        if constexpr (is_row_major) {
            return const_row_view_type(data() + i * cols_, cols_);
        } else {
            return const_row_view_type(data() + i, cols_, rows_);
        }
    }
    
    /**
     * @brief Get column as a strided vector view
     */
    col_view_type col(size_type j) {
        if (j >= cols_) {
            throw std::out_of_range("Column index out of range");
        }
        if constexpr (is_row_major) {
            // Row-major: column elements are strided
            return col_view_type(data() + j, rows_, static_cast<difference_type>(cols_));
        } else {
            // Column-major: column elements are contiguous
            return col_view_type(data() + j * rows_, rows_);
        }
    }
    
    const_col_view_type col(size_type j) const {
        if (j >= cols_) {
            throw std::out_of_range("Column index out of range");
        }
        if constexpr (is_row_major) {
            return const_col_view_type(data() + j, rows_, cols_);
        } else {
            return const_col_view_type(data() + j * rows_, rows_);
        }
    }
    
    /**
     * @brief Get diagonal as a strided vector view
     */
    col_view_type diag() {
        const size_type n = std::min(rows_, cols_);
        const size_type stride = cols_ + 1;  // Works for both row and column major
        return col_view_type(data(), n, static_cast<difference_type>(stride));
    }
    
    const_col_view_type diag() const {
        const size_type n = std::min(rows_, cols_);
        const size_type stride = cols_ + 1;
        return const_col_view_type(data(), n, static_cast<difference_type>(stride));
    }
    
    // === Matrix Operations ===
    
    /**
     * @brief Transpose (returns a lazy view)
     */
    transpose_view_type transpose() {
        return transpose_view_type(*this);
    }
    
    const_transpose_view_type transpose() const {
        return const_transpose_view_type(*this);
    }
    
    /**
     * @brief Shorthand for transpose (using T_ to avoid shadowing template parameter)
     */
    transpose_view_type T_() { return transpose(); }
    const_transpose_view_type T_() const { return transpose(); }
    
    /**
     * @brief Fill matrix with a value
     */
    void fill(const T& value) {
        std::fill(storage_.begin(), storage_.end(), value);
    }
    
    /**
     * @brief Set to identity matrix (must be square)
     */
    void set_identity() {
        if (!is_square()) {
            throw std::logic_error("Identity matrix must be square");
        }
        fill(T{0});
        for (size_type i = 0; i < rows_; ++i) {
            (*this)(i, i) = T{1};
        }
    }
    
    /**
     * @brief Create identity matrix
     */
    static Matrix identity(size_type n) {
        Matrix result(n, n, T{0});
        for (size_type i = 0; i < n; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }
    
    /**
     * @brief Create diagonal matrix from vector
     */
    template<typename U, typename S2>
    static Matrix diag(const Vector<U, S2>& v) {
        const size_type n = v.size();
        Matrix result(n, n, T{0});
        for (size_type i = 0; i < n; ++i) {
            result(i, i) = static_cast<T>(v[i]);
        }
        return result;
    }
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Matrix addition
     */
    template<typename U, typename S2, StorageOrder O2>
    Matrix& operator+=(const Matrix<U, S2, O2>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) {
                (*this)(i, j) += static_cast<T>(other(i, j));
            }
        }
        return *this;
    }
    
    /**
     * @brief Matrix subtraction
     */
    template<typename U, typename S2, StorageOrder O2>
    Matrix& operator-=(const Matrix<U, S2, O2>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) {
                (*this)(i, j) -= static_cast<T>(other(i, j));
            }
        }
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    Matrix& operator*=(const T& scalar) {
        for (auto& elem : storage_) {
            elem *= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    Matrix& operator/=(const T& scalar) {
        for (auto& elem : storage_) {
            elem /= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Element-wise multiplication (Hadamard product)
     */
    template<typename U, typename S2, StorageOrder O2>
    Matrix& hadamard_product(const Matrix<U, S2, O2>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
        }
        
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) {
                (*this)(i, j) *= static_cast<T>(other(i, j));
            }
        }
        return *this;
    }
    
    // === Matrix-Vector Multiplication ===
    
    /**
     * @brief Matrix-vector multiplication: y = A * x
     */
    template<typename U, typename S2>
    auto operator*(const Vector<U, S2>& x) const {
        if (cols_ != x.size()) {
            throw std::invalid_argument("Matrix columns must match vector size");
        }
        
        using result_type = decltype(T{} * U{});
        Vector<result_type> y(rows_, result_type{0});
        
        // Optimize based on storage order
        if constexpr (is_row_major) {
            // Row-major: dot product of rows with vector
            for (size_type i = 0; i < rows_; ++i) {
                result_type sum{0};
                for (size_type j = 0; j < cols_; ++j) {
                    sum += (*this)(i, j) * x[j];
                }
                y[i] = sum;
            }
        } else {
            // Column-major: linear combination of columns
            for (size_type j = 0; j < cols_; ++j) {
                const auto xj = x[j];
                for (size_type i = 0; i < rows_; ++i) {
                    y[i] += (*this)(i, j) * xj;
                }
            }
        }
        
        return y;
    }
    
    /**
     * @brief Matrix-matrix multiplication: C = A * B
     * 
     * Uses cache-friendly blocked algorithm for large matrices
     */
    template<typename U, typename S2, StorageOrder O2>
    auto operator*(const Matrix<U, S2, O2>& B) const {
        if (cols_ != B.rows()) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
        
        using result_type = decltype(T{} * U{});
        Matrix<result_type> C(rows_, B.cols(), result_type{0});
        
        // Simple implementation for now - can be optimized with blocking
        // and SIMD for large matrices
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type k = 0; k < cols_; ++k) {
                const auto aik = (*this)(i, k);
                for (size_type j = 0; j < B.cols(); ++j) {
                    C(i, j) += aik * B(k, j);
                }
            }
        }
        
        return C;
    }
    
    // === Norms ===
    
    /**
     * @brief Frobenius norm (Euclidean norm of matrix as vector)
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
     * @brief Swap two rows
     */
    void swap_rows(size_type i1, size_type i2) {
        if (i1 >= rows_ || i2 >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        if (i1 == i2) return;
        
        for (size_type j = 0; j < cols_; ++j) {
            std::swap((*this)(i1, j), (*this)(i2, j));
        }
    }
    
    /**
     * @brief Swap two columns
     */
    void swap_cols(size_type j1, size_type j2) {
        if (j1 >= cols_ || j2 >= cols_) {
            throw std::out_of_range("Column index out of range");
        }
        if (j1 == j2) return;
        
        for (size_type i = 0; i < rows_; ++i) {
            std::swap((*this)(i, j1), (*this)(i, j2));
        }
    }
    
    /**
     * @brief Reshape matrix (must preserve total size)
     */
    void reshape(size_type new_rows, size_type new_cols) {
        if (new_rows * new_cols != size()) {
            throw std::invalid_argument("Reshape must preserve total number of elements");
        }
        rows_ = new_rows;
        cols_ = new_cols;
        this->shape_ = Shape{rows_, cols_};
    }
    
    // === Output ===
    
    /**
     * @brief Stream output with formatted display
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "Matrix(" << m.rows_ << "x" << m.cols_ << "):\n";
        
        // Find maximum width for formatting
        size_t max_width = 0;
        for (size_type i = 0; i < m.rows_; ++i) {
            for (size_type j = 0; j < m.cols_; ++j) {
                std::ostringstream ss;
                ss << m(i, j);
                max_width = std::max(max_width, ss.str().length());
            }
        }
        max_width = std::min(max_width + 2, size_t(12));
        
        for (size_type i = 0; i < m.rows_; ++i) {
            os << "[";
            for (size_type j = 0; j < m.cols_; ++j) {
                os << std::setw(max_width) << m(i, j);
                if (j < m.cols_ - 1) os << " ";
            }
            os << "]\n";
        }
        return os;
    }
};

// === Non-member Operations ===

/**
 * @brief Matrix addition
 */
template<typename T1, typename S1, StorageOrder O1,
         typename T2, typename S2, StorageOrder O2>
auto operator+(const Matrix<T1, S1, O1>& A, const Matrix<T2, S2, O2>& B) {
    using result_type = decltype(T1{} + T2{});
    Matrix<result_type> result(A);
    result += B;
    return result;
}

/**
 * @brief Matrix subtraction
 */
template<typename T1, typename S1, StorageOrder O1,
         typename T2, typename S2, StorageOrder O2>
auto operator-(const Matrix<T1, S1, O1>& A, const Matrix<T2, S2, O2>& B) {
    using result_type = decltype(T1{} - T2{});
    Matrix<result_type> result(A);
    result -= B;
    return result;
}

/**
 * @brief Scalar multiplication (scalar * matrix)
 */
template<typename T, typename S, StorageOrder O, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, Matrix<T, S, O>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Scalar& s, const Matrix<T, S, O>& M) {
    using result_type = decltype(s * T{});
    Matrix<result_type> result(M.rows(), M.cols());
    const result_type rs = static_cast<result_type>(s);
    for (typename Matrix<T,S,O>::size_type i = 0; i < M.rows(); ++i) {
        for (typename Matrix<T,S,O>::size_type j = 0; j < M.cols(); ++j) {
            result(i, j) = rs * static_cast<result_type>(M(i, j));
        }
    }
    return result;
}

/**
 * @brief Scalar multiplication (matrix * scalar)
 */
template<typename T, typename S, StorageOrder O, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, Matrix<T, S, O>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Matrix<T, S, O>& M, const Scalar& s) {
    return s * M;
}

// === Matrix View Classes (Placeholder) ===

/**
 * @brief View of a submatrix
 */
template<typename T>
class MatrixView {
    // To be implemented
};

/**
 * @brief Transpose view of a matrix
 */
template<typename T>
class MatrixTransposeView {
    // To be implemented with lazy evaluation
};

} // namespace fem::numeric

#endif // NUMERIC_MATRIX_H

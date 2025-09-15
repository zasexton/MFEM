#pragma once

#ifndef NUMERIC_SMALL_MATRIX_H
#define NUMERIC_SMALL_MATRIX_H

#include <array>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iomanip>

#include "../base/numeric_base.h"
#include "../base/storage_base.h"
#include "../base/iterator_base.h"
#include "../base/slice_base.h"
#include "../base/view_base.h"
#include "../base/ops_base.h"
#include "../base/broadcast_base.h"
#include "../base/traits_base.h"

namespace fem::numeric {

// Forward declarations
template<typename T, size_t Rows, size_t Cols> class SmallMatrixView;

/**
 * @brief Small matrix class optimized for compile-time known dimensions
 * 
 * Implements a high-performance small matrix with stack allocation:
 * - Compile-time dimensions for maximum optimization
 * - Stack allocation with no dynamic memory
 * - Vectorized operations for small sizes
 * - Optimized for FEM element matrices (typical sizes 3x3, 4x4, 8x8, 16x16)
 * - Cache-friendly contiguous storage
 * - Specializations for common operations
 * - Full compatibility with larger matrix operations
 * 
 * Typical usage in FEM:
 * ```cpp
 * SmallMatrix<double, 8, 8> K_e;  // Element stiffness matrix
 * SmallMatrix<double, 8, 3> B;    // Strain-displacement matrix
 * SmallMatrix<double, 3, 1> strain;  // Strain vector
 * ```
 * 
 * @tparam T Value type (must satisfy StorableType concept)
 * @tparam Rows Number of rows (compile-time constant)
 * @tparam Cols Number of columns (compile-time constant)
 */
template<typename T, size_t Rows, size_t Cols>
class SmallMatrix {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be positive");
    static_assert(Rows <= 64 && Cols <= 64, "Small matrix dimensions limited to 64x64");
    
    // === Type Aliases ===
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using storage_type = std::array<T, Rows * Cols>;
    
    // Iterator types
    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;
    using reverse_iterator = typename storage_type::reverse_iterator;
    using const_reverse_iterator = typename storage_type::const_reverse_iterator;
    
    // View types (placeholder for now)
    using view_type = SmallMatrixView<T, Rows, Cols>;
    using const_view_type = SmallMatrixView<const T, Rows, Cols>;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Compile-time constants
    static constexpr size_t rows_c = Rows;
    static constexpr size_t cols_c = Cols;
    static constexpr size_t size_c = Rows * Cols;
    static constexpr bool is_vector = (Cols == 1);
    static constexpr bool is_square = (Rows == Cols);
    static constexpr size_t tensor_rank = 2;
    
private:
    alignas(32) storage_type data_;  // 32-byte aligned for SIMD
    
    // Convert (i,j) to linear index (row-major order)
    static constexpr size_type linear_index(size_type i, size_type j) noexcept {
        return i * Cols + j;
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor - zero initialization
     */
    constexpr SmallMatrix() : data_{} {}
    
    /**
     * @brief Value constructor - initialize all elements to value
     */
    explicit constexpr SmallMatrix(const T& value) : data_{} {
        data_.fill(value);
    }
    
    /**
     * @brief Initializer list constructor (row-major order)
     */
    constexpr SmallMatrix(std::initializer_list<T> values) requires (Rows != 1 && Cols != 1) : data_{} {
        if (values.size() > size_c) {
            throw std::invalid_argument("Too many initializer values for small matrix");
        }
        
        auto it = values.begin();
        for (size_type i = 0; i < std::min(values.size(), size_c); ++i) {
            data_[i] = *it++;
        }
    }
    
    /**
     * @brief 2D initializer list constructor
     */
    constexpr SmallMatrix(std::initializer_list<std::initializer_list<T>> rows) : data_{} {
        if (rows.size() != Rows) {
            throw std::invalid_argument("Initializer list row count must match matrix rows");
        }
        
        size_type i = 0;
        for (const auto& row : rows) {
            if (row.size() != Cols) {
                throw std::invalid_argument("Initializer list column count must match matrix columns");
            }
            
            size_type j = 0;
            for (const auto& val : row) {
                data_[linear_index(i, j)] = val;
                ++j;
            }
            ++i;
        }
    }
    
    /**
     * @brief Array constructor
     */
    explicit constexpr SmallMatrix(const storage_type& arr) : data_(arr) {}
    
    /**
     * @brief Copy constructor from different size (truncate or pad with zeros)
     */
    template<size_t R2, size_t C2>
    explicit constexpr SmallMatrix(const SmallMatrix<T, R2, C2>& other) : data_{} {
        const size_t min_rows = std::min(Rows, R2);
        const size_t min_cols = std::min(Cols, C2);
        
        for (size_t i = 0; i < min_rows; ++i) {
            for (size_t j = 0; j < min_cols; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    // Cross-type copy (same dimensions)
    template<typename U>
    explicit constexpr SmallMatrix(const SmallMatrix<U, Rows, Cols>& other) : data_{} {
        for (size_type i = 0; i < Rows; ++i) {
            for (size_type j = 0; j < Cols; ++j) {
                (*this)(i, j) = static_cast<T>(other(i, j));
            }
        }
    }
    
    /**
     * @brief Copy constructor
     */
    constexpr SmallMatrix(const SmallMatrix& other) = default;
    
    /**
     * @brief Move constructor
     */
    constexpr SmallMatrix(SmallMatrix&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    SmallMatrix& operator=(const SmallMatrix& other) = default;
    
    /**
     * @brief Move assignment
     */
    SmallMatrix& operator=(SmallMatrix&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~SmallMatrix() = default;
    
    // === Size and Shape ===
    
    /**
     * @brief Get number of rows
     */
    static constexpr size_type rows() noexcept { return Rows; }
    
    /**
     * @brief Get number of columns
     */
    static constexpr size_type cols() noexcept { return Cols; }
    
    /**
     * @brief Get total number of elements
     */
    static constexpr size_type size() noexcept { return size_c; }
    
    /**
     * @brief Always non-empty for small matrices
     */
    static constexpr bool empty() noexcept { return false; }
    
    /**
     * @brief Check if matrix is square
     */
    static constexpr bool is_square_matrix() noexcept { return is_square; }
    
    /**
     * @brief Check if matrix is vector
     */
    static constexpr bool is_vector_matrix() noexcept { return is_vector; }
    
    // === Element Access ===
    
    /**
     * @brief Access element at (i, j) with bounds checking
     */
    constexpr reference at(size_type i, size_type j) {
        if (i >= Rows || j >= Cols) {
            throw std::out_of_range("Small matrix index out of range");
        }
        return data_[linear_index(i, j)];
    }
    
    constexpr const_reference at(size_type i, size_type j) const {
        if (i >= Rows || j >= Cols) {
            throw std::out_of_range("Small matrix index out of range");
        }
        return data_[linear_index(i, j)];
    }
    
    /**
     * @brief Access element at (i, j) without bounds checking
     */
    constexpr reference operator()(size_type i, size_type j) noexcept {
        return data_[linear_index(i, j)];
    }
    
    constexpr const_reference operator()(size_type i, size_type j) const noexcept {
        return data_[linear_index(i, j)];
    }
    
    /**
     * @brief Linear indexing
     */
    constexpr reference operator[](size_type index) noexcept {
        return data_[index];
    }
    
    constexpr const_reference operator[](size_type index) const noexcept {
        return data_[index];
    }
    
    /**
     * @brief Get raw data pointer
     */
    constexpr pointer data() noexcept { return data_.data(); }
    constexpr const_pointer data() const noexcept { return data_.data(); }
    
    // === Iterators ===
    
    constexpr iterator begin() noexcept { return data_.begin(); }
    constexpr const_iterator begin() const noexcept { return data_.begin(); }
    constexpr const_iterator cbegin() const noexcept { return data_.cbegin(); }
    
    constexpr iterator end() noexcept { return data_.end(); }
    constexpr const_iterator end() const noexcept { return data_.end(); }
    constexpr const_iterator cend() const noexcept { return data_.cend(); }
    
    constexpr reverse_iterator rbegin() noexcept { return data_.rbegin(); }
    constexpr const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
    constexpr const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
    
    constexpr reverse_iterator rend() noexcept { return data_.rend(); }
    constexpr const_reverse_iterator rend() const noexcept { return data_.rend(); }
    constexpr const_reverse_iterator crend() const noexcept { return data_.crend(); }
    
    // === Matrix Operations ===
    
    /**
     * @brief Fill matrix with a value
     */
    void fill(const T& value) noexcept {
        data_.fill(value);
    }
    
    /**
     * @brief Set matrix to zero
     */
    void zero() noexcept {
        fill(T{0});
    }
    
    /**
     * @brief Set to identity matrix (square matrices only)
     */
    template<bool Enable = is_square>
    requires Enable
    void set_identity() noexcept {
        zero();
        for (size_type i = 0; i < Rows; ++i) {
            (*this)(i, i) = T{1};
        }
    }
    
    /**
     * @brief Create identity matrix (square matrices only)
     */
    template<bool Enable = is_square>
    requires Enable
    static SmallMatrix identity() noexcept {
        SmallMatrix result;
        result.set_identity();
        return result;
    }
    
    /**
     * @brief Transpose (creates new matrix with swapped dimensions)
     */
    SmallMatrix<T, Cols, Rows> transpose() const noexcept {
        SmallMatrix<T, Cols, Rows> result;
        for (size_type i = 0; i < Rows; ++i) {
            for (size_type j = 0; j < Cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    /**
     * @brief In-place transpose (square matrices only)
     */
    template<bool Enable = is_square>
    requires Enable
    void transpose_inplace() noexcept {
        for (size_type i = 0; i < Rows; ++i) {
            for (size_type j = i + 1; j < Cols; ++j) {
                std::swap((*this)(i, j), (*this)(j, i));
            }
        }
    }
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Matrix addition
     */
    template<typename U>
    SmallMatrix& operator+=(const SmallMatrix<U, Rows, Cols>& other) noexcept {
        for (size_type i = 0; i < size_c; ++i) {
            data_[i] += static_cast<T>(other[i]);
        }
        return *this;
    }
    
    /**
     * @brief Matrix subtraction
     */
    template<typename U>
    SmallMatrix& operator-=(const SmallMatrix<U, Rows, Cols>& other) noexcept {
        for (size_type i = 0; i < size_c; ++i) {
            data_[i] -= static_cast<T>(other[i]);
        }
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    SmallMatrix& operator*=(const T& scalar) noexcept {
        for (auto& elem : data_) {
            elem *= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    SmallMatrix& operator/=(const T& scalar) noexcept {
        for (auto& elem : data_) {
            elem /= scalar;
        }
        return *this;
    }
    
    /**
     * @brief Element-wise multiplication (Hadamard product)
     */
    template<typename U>
    SmallMatrix& hadamard_product(const SmallMatrix<U, Rows, Cols>& other) noexcept {
        for (size_type i = 0; i < size_c; ++i) {
            data_[i] *= static_cast<T>(other[i]);
        }
        return *this;
    }
    
    // === Matrix-Vector Multiplication ===
    
    /**
     * @brief Matrix-vector multiplication: y = A * x
     */
    template<typename U>
    SmallMatrix<decltype(T{} * U{}), Rows, 1> 
    operator*(const SmallMatrix<U, Cols, 1>& x) const noexcept {
        using result_type = decltype(T{} * U{});
        SmallMatrix<result_type, Rows, 1> y;
        
        for (size_type i = 0; i < Rows; ++i) {
            result_type sum{0};
            for (size_type j = 0; j < Cols; ++j) {
                sum += (*this)(i, j) * x(j, 0);
            }
            y(i, 0) = sum;
        }
        return y;
    }
    
    // === Matrix-Matrix Multiplication ===
    
    /**
     * @brief Matrix-matrix multiplication: C = A * B
     */
    template<typename U, size_t K>
    SmallMatrix<decltype(T{} * U{}), Rows, K> 
    operator*(const SmallMatrix<U, Cols, K>& B) const noexcept {
        using result_type = decltype(T{} * U{});
        SmallMatrix<result_type, Rows, K> C;
        
        // Initialize to zero
        C.zero();
        
        // Optimized multiplication with loop reordering for cache efficiency
        for (size_type i = 0; i < Rows; ++i) {
            for (size_type k = 0; k < Cols; ++k) {
                const result_type aik = static_cast<result_type>((*this)(i, k));
                for (size_type j = 0; j < K; ++j) {
                    C(i, j) += aik * static_cast<result_type>(B(k, j));
                }
            }
        }
        
        return C;
    }
    
    // === Specialized Operations ===
    
    /**
     * @brief Determinant (up to 4x4 with explicit formulas)
     */
    template<bool Enable = is_square>
    requires Enable
    T determinant() const {
        if constexpr (Rows == 1) {
            return (*this)(0, 0);
        } else if constexpr (Rows == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        } else if constexpr (Rows == 3) {
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
                 - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
                 + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        } else if constexpr (Rows == 4) {
            // Optimized 4x4 determinant using cofactor expansion
            T det = T{0};
            
            // Expand along first row
            for (size_type j = 0; j < 4; ++j) {
                T cofactor = ((j % 2 == 0) ? T{1} : T{-1}) * (*this)(0, j);
                
                // 3x3 minor
                T minor = T{0};
                size_type rows[3], cols[3];
                size_type r_idx = 0, c_idx = 0;
                
                for (size_type r = 1; r < 4; ++r) rows[r_idx++] = r;
                for (size_type c = 0; c < 4; ++c) {
                    if (c != j) cols[c_idx++] = c;
                }
                
                minor = (*this)(rows[0], cols[0]) * ((*this)(rows[1], cols[1]) * (*this)(rows[2], cols[2]) - 
                                                    (*this)(rows[1], cols[2]) * (*this)(rows[2], cols[1]))
                      - (*this)(rows[0], cols[1]) * ((*this)(rows[1], cols[0]) * (*this)(rows[2], cols[2]) - 
                                                    (*this)(rows[1], cols[2]) * (*this)(rows[2], cols[0]))
                      + (*this)(rows[0], cols[2]) * ((*this)(rows[1], cols[0]) * (*this)(rows[2], cols[1]) - 
                                                    (*this)(rows[1], cols[1]) * (*this)(rows[2], cols[0]));
                
                det += cofactor * minor;
            }
            return det;
        } else {
            // For larger matrices, use LU decomposition
            // This would require a more complex implementation
            static_assert(Rows <= 4, "Determinant only implemented up to 4x4 matrices");
            return T{0};
        }
    }
    
    /**
     * @brief Trace (sum of diagonal elements, square matrices only)
     */
    template<bool Enable = is_square>
    requires Enable
    T trace() const noexcept {
        T sum{0};
        for (size_type i = 0; i < Rows; ++i) {
            sum += (*this)(i, i);
        }
        return sum;
    }
    
    // === Norms ===
    
    /**
     * @brief Frobenius norm
     */
    scalar_type frobenius_norm() const noexcept {
        scalar_type sum{0};
        for (const auto& elem : data_) {
            if constexpr (is_complex_number_v<T>) {
                sum += std::norm(elem);
            } else {
                sum += static_cast<scalar_type>(elem) * static_cast<scalar_type>(elem);
            }
        }
        return std::sqrt(sum);
    }
    
    /**
     * @brief Maximum absolute value norm
     */
    scalar_type max_norm() const noexcept {
        scalar_type max_val{0};
        for (const auto& elem : data_) {
            max_val = std::max(max_val, std::abs(elem));
        }
        return max_val;
    }
    
    /**
     * @brief 1-norm (maximum column sum)
     */
    scalar_type norm1() const noexcept {
        scalar_type max_col_sum{0};
        for (size_type j = 0; j < Cols; ++j) {
            scalar_type col_sum{0};
            for (size_type i = 0; i < Rows; ++i) {
                col_sum += std::abs((*this)(i, j));
            }
            max_col_sum = std::max(max_col_sum, col_sum);
        }
        return max_col_sum;
    }
    
    /**
     * @brief Infinity norm (maximum row sum)
     */
    scalar_type norm_inf() const noexcept {
        scalar_type max_row_sum{0};
        for (size_type i = 0; i < Rows; ++i) {
            scalar_type row_sum{0};
            for (size_type j = 0; j < Cols; ++j) {
                row_sum += std::abs((*this)(i, j));
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }
        return max_row_sum;
    }
    
    // === Utility Functions ===
    
    /**
     * @brief Swap two rows
     */
    void swap_rows(size_type i1, size_type i2) noexcept {
        if (i1 >= Rows || i2 >= Rows || i1 == i2) return;
        
        for (size_type j = 0; j < Cols; ++j) {
            std::swap((*this)(i1, j), (*this)(i2, j));
        }
    }
    
    /**
     * @brief Swap two columns
     */
    void swap_cols(size_type j1, size_type j2) noexcept {
        if (j1 >= Cols || j2 >= Cols || j1 == j2) return;
        
        for (size_type i = 0; i < Rows; ++i) {
            std::swap((*this)(i, j1), (*this)(i, j2));
        }
    }
    
    /**
     * @brief Get underlying storage array
     */
    const storage_type& storage() const noexcept { return data_; }
    storage_type& storage() noexcept { return data_; }
    
    // === Comparison ===
    
    /**
     * @brief Element-wise equality comparison
     */
    template<typename U>
    bool operator==(const SmallMatrix<U, Rows, Cols>& other) const noexcept {
        for (size_type i = 0; i < size_c; ++i) {
            if (data_[i] != static_cast<T>(other[i])) {
                return false;
            }
        }
        return true;
    }
    
    template<typename U>
    bool operator!=(const SmallMatrix<U, Rows, Cols>& other) const noexcept {
        return !(*this == other);
    }
    
    // === Output ===
    
    /**
     * @brief Stream output with formatted display
     */
    friend std::ostream& operator<<(std::ostream& os, const SmallMatrix& m) {
        os << "SmallMatrix<" << Rows << "x" << Cols << ">:\n";
        
        // Find maximum width for formatting
        size_t max_width = 0;
        for (size_type i = 0; i < Rows; ++i) {
            for (size_type j = 0; j < Cols; ++j) {
                std::ostringstream ss;
                ss << m(i, j);
                max_width = std::max(max_width, ss.str().length());
            }
        }
        max_width = std::min(max_width + 2, size_t(12));
        
        for (size_type i = 0; i < Rows; ++i) {
            os << "[";
            for (size_type j = 0; j < Cols; ++j) {
                os << std::setw(static_cast<int>(max_width)) << m(i, j);
                if (j < Cols - 1) os << " ";
            }
            os << "]\n";
        }
        return os;
    }
};

// === Type Aliases for Common Small Matrices ===

// Vectors
template<typename T> using SmallVector1 = SmallMatrix<T, 1, 1>;
template<typename T> using SmallVector2 = SmallMatrix<T, 2, 1>;
template<typename T> using SmallVector3 = SmallMatrix<T, 3, 1>;
template<typename T> using SmallVector4 = SmallMatrix<T, 4, 1>;
template<typename T> using SmallVector8 = SmallMatrix<T, 8, 1>;
template<typename T> using SmallVector16 = SmallMatrix<T, 16, 1>;

// Square matrices
template<typename T> using SmallMatrix2x2 = SmallMatrix<T, 2, 2>;
template<typename T> using SmallMatrix3x3 = SmallMatrix<T, 3, 3>;
template<typename T> using SmallMatrix4x4 = SmallMatrix<T, 4, 4>;
template<typename T> using SmallMatrix8x8 = SmallMatrix<T, 8, 8>;
template<typename T> using SmallMatrix16x16 = SmallMatrix<T, 16, 16>;

// Common FEM element matrices
template<typename T> using ElementMatrix3D = SmallMatrix<T, 24, 24>;  // 8-node hex
template<typename T> using ElementMatrix2D = SmallMatrix<T, 8, 8>;    // 4-node quad
template<typename T> using ElementStiffness = SmallMatrix<T, 6, 6>;   // General element

// === Non-member Operations ===

/**
 * @brief Small matrix addition
 */
template<typename T1, typename T2, size_t R, size_t C>
auto operator+(const SmallMatrix<T1, R, C>& A, const SmallMatrix<T2, R, C>& B) noexcept {
    using result_type = decltype(T1{} + T2{});
    SmallMatrix<result_type, R, C> result(A);
    result += B;
    return result;
}

/**
 * @brief Small matrix subtraction
 */
template<typename T1, typename T2, size_t R, size_t C>
auto operator-(const SmallMatrix<T1, R, C>& A, const SmallMatrix<T2, R, C>& B) noexcept {
    using result_type = decltype(T1{} - T2{});
    SmallMatrix<result_type, R, C> result(A);
    result -= B;
    return result;
}

/**
 * @brief Scalar multiplication (scalar * matrix)
 */
template<typename T, size_t R, size_t C, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, SmallMatrix<T, R, C>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Scalar& s, const SmallMatrix<T, R, C>& M) noexcept {
    using result_type = decltype(s * T{});
    SmallMatrix<result_type, R, C> result(M);
    result *= static_cast<result_type>(s);
    return result;
}

/**
 * @brief Scalar multiplication (matrix * scalar)
 */
template<typename T, size_t R, size_t C, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, SmallMatrix<T, R, C>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const SmallMatrix<T, R, C>& M, const Scalar& s) noexcept {
    return s * M;
}

// === Small Matrix View Class (Placeholder) ===

/**
 * @brief View of a small matrix
 */
template<typename T, size_t Rows, size_t Cols>
class SmallMatrixView {
    // To be implemented for small matrix views and submatrices
};

} // namespace fem::numeric

#endif // NUMERIC_SMALL_MATRIX_H

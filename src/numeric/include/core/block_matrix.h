#pragma once

#ifndef NUMERIC_BLOCK_MATRIX_H
#define NUMERIC_BLOCK_MATRIX_H

#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <string>
#include <map>
#include <utility>

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
#include "matrix.h"
#include "vector.h"
#include "block_vector.h"

namespace fem::numeric {

// Forward declarations
template<typename T> class BlockMatrixView;

/**
 * @brief Block matrix class for multi-field problems
 * 
 * Implements a matrix composed of multiple blocks, each representing
 * interactions between different fields in a multi-physics problem:
 * - Hierarchical storage with individual matrices as blocks
 * - Named blocks for intuitive access (e.g., ("velocity", "pressure"))
 * - Automatic differentiation support through element types
 * - Block-aware operations including matrix-vector multiplication
 * - Efficient assembly and solvers for block systems
 * - Memory layout compatible with block preconditioners
 * 
 * Example usage:
 * ```cpp
 * BlockMatrix<double> K;
 * K.set_block_structure({"u", "v", "p"}, {3*N, 3*N, N});
 * 
 * auto& K_uu = K.block("u", "u");  // Velocity-velocity coupling
 * auto& K_up = K.block("u", "p");  // Velocity-pressure coupling
 * ```
 * 
 * @tparam T Value type (must satisfy StorableType concept)
 */
template<typename T>
class BlockMatrix : public ContainerBase<BlockMatrix<T>, T, DynamicStorage<T>>,
                    public ExpressionBase<BlockMatrix<T>> {
public:
    template<typename> friend class BlockMatrix;
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    
    // === Type Aliases ===
    using base_type = ContainerBase<BlockMatrix<T>, T, DynamicStorage<T>>;
    using expression_base = ExpressionBase<BlockMatrix<T>>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Block types
    using block_type = Matrix<T>;
    using block_reference = block_type&;
    using const_block_reference = const block_type&;
    using block_pointer = std::unique_ptr<block_type>;
    
    // Vector types for multiplication
    using vector_type = Vector<T>;
    using block_vector_type = BlockVector<T>;
    
    // Storage types
    using block_info = std::pair<std::string, size_type>;  // name, size
    using block_structure = std::vector<block_info>;
    using block_grid = std::vector<std::vector<block_pointer>>;
    using block_map = std::map<std::pair<std::string, std::string>, std::pair<size_type, size_type>>;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Matrix properties
    enum class SymmetryType {
        General,     // No symmetry assumption
        Symmetric,   // A = A^T
        Skew,        // A = -A^T
        Hermitian    // A = A^H (for complex matrices)
    };
    
private:
    // Block structure
    block_structure row_structure_;
    block_structure col_structure_;
    size_type num_block_rows_;
    size_type num_block_cols_;
    size_type total_rows_;
    size_type total_cols_;
    
    // Block storage
    block_grid blocks_;  // [block_row][block_col]
    block_map block_map_;  // (row_name, col_name) -> (block_row, block_col)
    
    // Matrix properties
    SymmetryType symmetry_;
    
    // Helper to find block index by name
    size_type find_row_block_index(const std::string& name) const {
        for (size_type i = 0; i < row_structure_.size(); ++i) {
            if (row_structure_[i].first == name) {
                return i;
            }
        }
        throw std::invalid_argument("Row block '" + name + "' not found");
    }
    
    size_type find_col_block_index(const std::string& name) const {
        for (size_type i = 0; i < col_structure_.size(); ++i) {
            if (col_structure_[i].first == name) {
                return i;
            }
        }
        throw std::invalid_argument("Column block '" + name + "' not found");
    }
    
    // Update total dimensions
    void update_dimensions() {
        total_rows_ = 0;
        for (const auto& [name, size] : row_structure_) {
            total_rows_ += size;
        }
        
        total_cols_ = 0;
        for (const auto& [name, size] : col_structure_) {
            total_cols_ += size;
        }
        
        this->shape_ = Shape{total_rows_, total_cols_};
    }
    
    // Initialize block grid
    void initialize_blocks() {
        blocks_.clear();
        blocks_.resize(num_block_rows_);
        for (size_type i = 0; i < num_block_rows_; ++i) {
            blocks_[i].resize(num_block_cols_);
        }
        
        // Update block map
        block_map_.clear();
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                const std::string& row_name = row_structure_[i].first;
                const std::string& col_name = col_structure_[j].first;
                block_map_[{row_name, col_name}] = {i, j};
            }
        }
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor - creates empty block matrix
     */
    explicit BlockMatrix(SymmetryType symmetry = SymmetryType::General)
        : base_type(), row_structure_(), col_structure_(),
          num_block_rows_(0), num_block_cols_(0),
          total_rows_(0), total_cols_(0),
          blocks_(), block_map_(), symmetry_(symmetry) {
        this->shape_ = Shape{0, 0};
    }
    
    /**
     * @brief Constructor with symmetric block structure
     */
    BlockMatrix(const std::vector<std::string>& block_names,
                const std::vector<size_type>& block_sizes,
                SymmetryType symmetry = SymmetryType::General)
        : BlockMatrix(symmetry) {
        set_block_structure(block_names, block_sizes, block_names, block_sizes);
    }
    
    /**
     * @brief Constructor with general block structure
     */
    BlockMatrix(const std::vector<std::string>& row_names,
                const std::vector<size_type>& row_sizes,
                const std::vector<std::string>& col_names,
                const std::vector<size_type>& col_sizes,
                SymmetryType symmetry = SymmetryType::General)
        : BlockMatrix(symmetry) {
        set_block_structure(row_names, row_sizes, col_names, col_sizes);
    }
    
    /**
     * @brief Copy constructor
     */
    BlockMatrix(const BlockMatrix& other)
        : base_type(other), row_structure_(other.row_structure_), 
          col_structure_(other.col_structure_),
          num_block_rows_(other.num_block_rows_), num_block_cols_(other.num_block_cols_),
          total_rows_(other.total_rows_), total_cols_(other.total_cols_),
          block_map_(other.block_map_), symmetry_(other.symmetry_) {
        
        // Deep copy blocks
        blocks_.resize(num_block_rows_);
        for (size_type i = 0; i < num_block_rows_; ++i) {
            blocks_[i].resize(num_block_cols_);
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (other.blocks_[i][j]) {
                    blocks_[i][j] = std::make_unique<block_type>(*other.blocks_[i][j]);
                }
            }
        }
    }
    
    /**
     * @brief Move constructor
     */
    BlockMatrix(BlockMatrix&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    BlockMatrix& operator=(const BlockMatrix& other) {
        if (this != &other) {
            base_type::operator=(other);
            row_structure_ = other.row_structure_;
            col_structure_ = other.col_structure_;
            num_block_rows_ = other.num_block_rows_;
            num_block_cols_ = other.num_block_cols_;
            total_rows_ = other.total_rows_;
            total_cols_ = other.total_cols_;
            block_map_ = other.block_map_;
            symmetry_ = other.symmetry_;
            
            // Deep copy blocks
            blocks_.clear();
            blocks_.resize(num_block_rows_);
            for (size_type i = 0; i < num_block_rows_; ++i) {
                blocks_[i].resize(num_block_cols_);
                for (size_type j = 0; j < num_block_cols_; ++j) {
                    if (other.blocks_[i][j]) {
                        blocks_[i][j] = std::make_unique<block_type>(*other.blocks_[i][j]);
                    }
                }
            }
        }
        return *this;
    }
    
    /**
     * @brief Move assignment
     */
    BlockMatrix& operator=(BlockMatrix&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~BlockMatrix() = default;
    
    // === Block Structure Management ===
    
    /**
     * @brief Set symmetric block structure
     */
    void set_block_structure(const std::vector<std::string>& block_names,
                           const std::vector<size_type>& block_sizes) {
        set_block_structure(block_names, block_sizes, block_names, block_sizes);
    }
    
    /**
     * @brief Set general block structure
     */
    void set_block_structure(const std::vector<std::string>& row_names,
                           const std::vector<size_type>& row_sizes,
                           const std::vector<std::string>& col_names,
                           const std::vector<size_type>& col_sizes) {
        
        if (row_names.size() != row_sizes.size()) {
            throw std::invalid_argument("Row names and sizes must have same length");
        }
        if (col_names.size() != col_sizes.size()) {
            throw std::invalid_argument("Column names and sizes must have same length");
        }
        
        // Build row structure
        row_structure_.clear();
        for (size_type i = 0; i < row_names.size(); ++i) {
            row_structure_.emplace_back(row_names[i], row_sizes[i]);
        }
        
        // Build column structure
        col_structure_.clear();
        for (size_type i = 0; i < col_names.size(); ++i) {
            col_structure_.emplace_back(col_names[i], col_sizes[i]);
        }
        
        num_block_rows_ = row_structure_.size();
        num_block_cols_ = col_structure_.size();
        
        update_dimensions();
        initialize_blocks();
    }
    
    /**
     * @brief Get number of block rows
     */
    size_type num_block_rows() const noexcept { return num_block_rows_; }
    
    /**
     * @brief Get number of block columns
     */
    size_type num_block_cols() const noexcept { return num_block_cols_; }
    
    /**
     * @brief Get total matrix rows
     */
    size_type rows() const noexcept { return total_rows_; }
    
    /**
     * @brief Get total matrix columns
     */
    size_type cols() const noexcept { return total_cols_; }
    
    /**
     * @brief Get total matrix size (number of elements)
     */
    size_type size() const noexcept { return total_rows_ * total_cols_; }

    /**
     * @brief Expression shape for compatibility
     */
    Shape shape() const noexcept { return Shape{total_rows_, total_cols_}; }
    
    /**
     * @brief Check if matrix is empty
     */
    bool empty() const noexcept { return size() == 0; }
    
    /**
     * @brief Check if matrix is square
     */
    bool is_square() const noexcept { return total_rows_ == total_cols_; }
    
    /**
     * @brief Get row block names
     */
    std::vector<std::string> row_block_names() const {
        std::vector<std::string> names;
        names.reserve(row_structure_.size());
        for (const auto& [name, _] : row_structure_) {
            names.push_back(name);
        }
        return names;
    }
    
    /**
     * @brief Get column block names
     */
    std::vector<std::string> col_block_names() const {
        std::vector<std::string> names;
        names.reserve(col_structure_.size());
        for (const auto& [name, _] : col_structure_) {
            names.push_back(name);
        }
        return names;
    }
    
    /**
     * @brief Get row block size
     */
    size_type row_block_size(const std::string& name) const {
        size_type index = find_row_block_index(name);
        return row_structure_[index].second;
    }
    
    /**
     * @brief Get column block size
     */
    size_type col_block_size(const std::string& name) const {
        size_type index = find_col_block_index(name);
        return col_structure_[index].second;
    }
    
    // === Block Access ===
    
    /**
     * @brief Get block by name (creates if doesn't exist)
     */
    block_reference block(const std::string& row_name, const std::string& col_name) {
        size_type i = find_row_block_index(row_name);
        size_type j = find_col_block_index(col_name);
        
        if (!blocks_[i][j]) {
            size_type rows = row_structure_[i].second;
            size_type cols = col_structure_[j].second;
            blocks_[i][j] = std::make_unique<block_type>(rows, cols, T{0});
        }
        
        return *blocks_[i][j];
    }
    
    const_block_reference block(const std::string& row_name, const std::string& col_name) const {
        size_type i = find_row_block_index(row_name);
        size_type j = find_col_block_index(col_name);
        
        if (!blocks_[i][j]) {
            throw std::runtime_error("Block (" + row_name + ", " + col_name + ") does not exist");
        }
        
        return *blocks_[i][j];
    }
    
    /**
     * @brief Get block by indices (creates if doesn't exist)
     */
    block_reference block(size_type i, size_type j) {
        if (i >= num_block_rows_ || j >= num_block_cols_) {
            throw std::out_of_range("Block index out of range");
        }
        
        if (!blocks_[i][j]) {
            size_type rows = row_structure_[i].second;
            size_type cols = col_structure_[j].second;
            blocks_[i][j] = std::make_unique<block_type>(rows, cols, T{0});
        }
        
        return *blocks_[i][j];
    }
    
    const_block_reference block(size_type i, size_type j) const {
        if (i >= num_block_rows_ || j >= num_block_cols_) {
            throw std::out_of_range("Block index out of range");
        }
        
        if (!blocks_[i][j]) {
            throw std::runtime_error("Block (" + std::to_string(i) + ", " + 
                                   std::to_string(j) + ") does not exist");
        }
        
        return *blocks_[i][j];
    }
    
    /**
     * @brief Check if block exists
     */
    bool has_block(const std::string& row_name, const std::string& col_name) const {
        try {
            size_type i = find_row_block_index(row_name);
            size_type j = find_col_block_index(col_name);
            return blocks_[i][j] != nullptr;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }
    
    bool has_block(size_type i, size_type j) const {
        if (i >= num_block_rows_ || j >= num_block_cols_) {
            return false;
        }
        return blocks_[i][j] != nullptr;
    }
    
    // === Matrix Operations ===
    
    /**
     * @brief Fill entire matrix with a value
     */
    void fill(const T& value) {
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (!blocks_[i][j]) {
                    size_type rows = row_structure_[i].second;
                    size_type cols = col_structure_[j].second;
                    blocks_[i][j] = std::make_unique<block_type>(rows, cols, value);
                } else {
                    blocks_[i][j]->fill(value);
                }
            }
        }
    }
    
    /**
     * @brief Set all elements to zero
     */
    void zero() {
        fill(T{0});
    }
    
    /**
     * @brief Set to identity matrix (must be square)
     */
    void set_identity() {
        if (!is_square()) {
            throw std::logic_error("Identity matrix must be square");
        }
        
        // Zero all blocks
        zero();
        
        // Set diagonal blocks to identity
        for (size_type i = 0; i < num_block_rows_; ++i) {
            if (i < num_block_cols_) {
                size_type block_size = std::min(row_structure_[i].second, col_structure_[i].second);
                for (size_type k = 0; k < block_size; ++k) {
                    block(i, i)(k, k) = T{1};
                }
            }
        }
    }
    
    // === Block Matrix-Vector Multiplication ===
    
    /**
     * @brief Block matrix-vector multiplication: y = A * x
     */
    block_vector_type operator*(const block_vector_type& x) const {
        if (num_block_cols_ != x.num_blocks()) {
            throw std::invalid_argument("Number of column blocks must match vector blocks");
        }
        
        // Create result block vector with same structure as matrix rows
        block_vector_type y;
        for (size_type i = 0; i < num_block_rows_; ++i) {
            const std::string& row_name = row_structure_[i].first;
            size_type row_size = row_structure_[i].second;
            y.add_block(row_name, row_size);
        }
        y.zero();
        
        // Perform block matrix-vector multiplication
        for (size_type i = 0; i < num_block_rows_; ++i) {
            auto y_block = y.block(i);
            
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (blocks_[i][j]) {
                    auto x_block = x.block(j);
                    auto temp = (*blocks_[i][j]) * Vector<T>(x_block.data(), x_block.size());
                    
                    // Add contribution to result block
                    for (size_type k = 0; k < y_block.size(); ++k) {
                        y_block[k] += temp[k];
                    }
                }
            }
        }
        
        return y;
    }
    
    /**
     * @brief Regular matrix-vector multiplication: y = A * x
     */
    vector_type operator*(const vector_type& x) const {
        if (total_cols_ != x.size()) {
            throw std::invalid_argument("Matrix columns must match vector size");
        }
        
        vector_type y(total_rows_, T{0});
        
        size_type row_offset = 0;
        for (size_type i = 0; i < num_block_rows_; ++i) {
            size_type row_size = row_structure_[i].second;
            
            size_type col_offset = 0;
            for (size_type j = 0; j < num_block_cols_; ++j) {
                size_type col_size = col_structure_[j].second;
                
                if (blocks_[i][j]) {
                    // Extract subvector
                    vector_type x_sub(col_size);
                    for (size_type k = 0; k < col_size; ++k) {
                        x_sub[k] = x[col_offset + k];
                    }
                    
                    // Multiply block with subvector
                    auto temp = (*blocks_[i][j]) * x_sub;
                    
                    // Add to result
                    for (size_type k = 0; k < row_size; ++k) {
                        y[row_offset + k] += temp[k];
                    }
                }
                
                col_offset += col_size;
            }
            
            row_offset += row_size;
        }
        
        return y;
    }
    
    // === Block Matrix-Matrix Multiplication ===
    
    /**
     * @brief Block matrix-matrix multiplication: C = A * B
     */
    template<typename U>
    auto operator*(const BlockMatrix<U>& B) const {
        if (num_block_cols_ != B.num_block_rows()) {
            throw std::invalid_argument("Incompatible block dimensions for multiplication");
        }
        
        using result_type = decltype(T{} * U{});
        BlockMatrix<result_type> C;
        
        // Set up result structure
        std::vector<std::string> result_row_names = row_block_names();
        std::vector<size_type> result_row_sizes;
        for (const auto& [name, size] : row_structure_) {
            result_row_sizes.push_back(size);
        }
        
        std::vector<std::string> result_col_names = B.col_block_names();
        std::vector<size_type> result_col_sizes;
        for (const auto& [name, size] : B.col_structure_) {
            result_col_sizes.push_back(size);
        }
        
        C.set_block_structure(result_row_names, result_row_sizes, 
                             result_col_names, result_col_sizes);
        
        // Perform block multiplication
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < B.num_block_cols_; ++j) {
                
                // Initialize result block to zero
                size_type rows = row_structure_[i].second;
                size_type cols = B.col_structure_[j].second;
                Matrix<result_type> sum(rows, cols, result_type{0});
                
                // Sum over intermediate blocks
                for (size_type k = 0; k < num_block_cols_; ++k) {
                    if (blocks_[i][k] && B.blocks_[k][j]) {
                        auto product = (*blocks_[i][k]) * (*B.blocks_[k][j]);
                        sum += product;
                    }
                }
                
                // Set result block
                C.blocks_[i][j] = std::make_unique<Matrix<result_type>>(std::move(sum));
            }
        }
        
        return C;
    }
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Block matrix addition
     */
    template<typename U>
    BlockMatrix& operator+=(const BlockMatrix<U>& other) {
        if (num_block_rows_ != other.num_block_rows_ || 
            num_block_cols_ != other.num_block_cols_) {
            throw std::invalid_argument("Block matrices must have same structure");
        }
        
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (other.blocks_[i][j]) {
                    if (!blocks_[i][j]) {
                        size_type rows = row_structure_[i].second;
                        size_type cols = col_structure_[j].second;
                        blocks_[i][j] = std::make_unique<block_type>(rows, cols, T{0});
                    }
                    *blocks_[i][j] += *other.blocks_[i][j];
                }
            }
        }
        
        return *this;
    }
    
    /**
     * @brief Block matrix subtraction
     */
    template<typename U>
    BlockMatrix& operator-=(const BlockMatrix<U>& other) {
        if (num_block_rows_ != other.num_block_rows_ || 
            num_block_cols_ != other.num_block_cols_) {
            throw std::invalid_argument("Block matrices must have same structure");
        }
        
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (other.blocks_[i][j]) {
                    if (!blocks_[i][j]) {
                        size_type rows = row_structure_[i].second;
                        size_type cols = col_structure_[j].second;
                        blocks_[i][j] = std::make_unique<block_type>(rows, cols, T{0});
                    }
                    *blocks_[i][j] -= *other.blocks_[i][j];
                }
            }
        }
        
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    BlockMatrix& operator*=(const T& scalar) {
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (blocks_[i][j]) {
                    *blocks_[i][j] *= scalar;
                }
            }
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    BlockMatrix& operator/=(const T& scalar) {
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (blocks_[i][j]) {
                    *blocks_[i][j] /= scalar;
                }
            }
        }
        return *this;
    }
    
    // === Norms ===
    
    /**
     * @brief Frobenius norm across all blocks
     */
    scalar_type frobenius_norm() const {
        scalar_type sum{0};
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (blocks_[i][j]) {
                    scalar_type block_norm = blocks_[i][j]->frobenius_norm();
                    sum += block_norm * block_norm;
                }
            }
        }
        return std::sqrt(sum);
    }
    
    /**
     * @brief Maximum absolute value across all blocks
     */
    scalar_type max_norm() const {
        scalar_type max_val{0};
        for (size_type i = 0; i < num_block_rows_; ++i) {
            for (size_type j = 0; j < num_block_cols_; ++j) {
                if (blocks_[i][j]) {
                    max_val = std::max(max_val, blocks_[i][j]->max_norm());
                }
            }
        }
        return max_val;
    }
    
    // === Properties ===
    
    /**
     * @brief Get/set symmetry type
     */
    SymmetryType symmetry() const noexcept { return symmetry_; }
    void set_symmetry(SymmetryType sym) { symmetry_ = sym; }
    
    // === Output ===
    
    /**
     * @brief Stream output with block structure display
     */
    friend std::ostream& operator<<(std::ostream& os, const BlockMatrix& bm) {
        os << "BlockMatrix(" << bm.num_block_rows_ << "x" << bm.num_block_cols_ 
           << " blocks, " << bm.total_rows_ << "x" << bm.total_cols_ << " total):\n";
        
        // Display block structure
        os << "Row blocks: ";
        for (size_type i = 0; i < bm.row_structure_.size(); ++i) {
            os << bm.row_structure_[i].first << "(" << bm.row_structure_[i].second << ")";
            if (i < bm.row_structure_.size() - 1) os << ", ";
        }
        os << "\n";
        
        os << "Col blocks: ";
        for (size_type i = 0; i < bm.col_structure_.size(); ++i) {
            os << bm.col_structure_[i].first << "(" << bm.col_structure_[i].second << ")";
            if (i < bm.col_structure_.size() - 1) os << ", ";
        }
        os << "\n";
        
        // Display existing blocks
        os << "Existing blocks:\n";
        for (size_type i = 0; i < bm.num_block_rows_; ++i) {
            for (size_type j = 0; j < bm.num_block_cols_; ++j) {
                if (bm.blocks_[i][j]) {
                    os << "  [" << bm.row_structure_[i].first << ", " 
                       << bm.col_structure_[j].first << "]: " 
                       << bm.blocks_[i][j]->rows() << "x" << bm.blocks_[i][j]->cols() << "\n";
                }
            }
        }
        
        return os;
    }
};

// === Non-member Operations ===

/**
 * @brief Block matrix addition
 */
template<typename T1, typename T2>
auto operator+(const BlockMatrix<T1>& A, const BlockMatrix<T2>& B) {
    using result_type = decltype(T1{} + T2{});
    BlockMatrix<result_type> result(A);
    result += B;
    return result;
}

/**
 * @brief Block matrix subtraction
 */
template<typename T1, typename T2>
auto operator-(const BlockMatrix<T1>& A, const BlockMatrix<T2>& B) {
    using result_type = decltype(T1{} - T2{});
    BlockMatrix<result_type> result(A);
    result -= B;
    return result;
}

/**
 * @brief Scalar multiplication (scalar * block_matrix)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockMatrix<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Scalar& s, const BlockMatrix<T>& bm) {
    using result_type = decltype(s * T{});
    BlockMatrix<result_type> result(bm);
    result *= static_cast<result_type>(s);
    return result;
}

/**
 * @brief Scalar multiplication (block_matrix * scalar)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockMatrix<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const BlockMatrix<T>& bm, const Scalar& s) {
    return s * bm;
}

// === Block Matrix View Class (Placeholder) ===

/**
 * @brief View of a block matrix
 */
template<typename T>
class BlockMatrixView {
    // To be implemented for advanced block matrix views
};

} // namespace fem::numeric

#endif // NUMERIC_BLOCK_MATRIX_H

#pragma once

#ifndef NUMERIC_BLOCK_VECTOR_H
#define NUMERIC_BLOCK_VECTOR_H

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

// Forward declarations
template<typename T> class BlockVectorView;

/**
 * @brief Block vector class for multi-field problems
 * 
 * Implements a vector composed of multiple blocks, each representing
 * a different field or component in a multi-physics problem:
 * - Efficient storage with contiguous memory or separate blocks
 * - Named blocks for intuitive access (e.g., "velocity", "pressure")
 * - Automatic differentiation support through element types
 * - Consistent interface with regular Vector operations
 * - Block-aware operations and iterators
 * - Memory layout compatible with block solvers
 * 
 * Example usage:
 * ```cpp
 * BlockVector<double> u;
 * u.add_block("velocity", 3 * num_nodes);    // 3D velocity
 * u.add_block("pressure", num_nodes);        // Scalar pressure
 * 
 * auto& vel = u.block("velocity");
 * auto& p = u.block("pressure");
 * ```
 * 
 * @tparam T Value type (must satisfy StorableType concept)
 */
template<typename T>
class BlockVector : public ContainerBase<BlockVector<T>, T, DynamicStorage<T>>,
                    public ExpressionBase<BlockVector<T>> {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    
    // === Type Aliases ===
    using base_type = ContainerBase<BlockVector<T>, T, DynamicStorage<T>>;
    using expression_base = ExpressionBase<BlockVector<T>>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Block types
    using block_type = Vector<T>;
    using block_reference = block_type&;
    using const_block_reference = const block_type&;
    using block_view_type = VectorView<T>;
    using const_block_view_type = VectorView<const T>;
    
    // Storage types
    using block_info = std::pair<std::string, size_type>;  // name, size
    using block_container = std::vector<block_info>;
    using offset_container = std::vector<size_type>;
    
    // Iterator types (iterate over elements, not blocks)
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Storage modes
    enum class StorageMode {
        Contiguous,  // Single contiguous memory block
        Separate     // Separate storage for each block
    };
    
private:
    StorageMode storage_mode_;
    
    // Contiguous storage mode
    DynamicStorage<T> storage_;
    block_container block_info_;
    offset_container block_offsets_;
    
    // Separate storage mode (alternative implementation)
    std::vector<std::unique_ptr<Vector<T>>> separate_blocks_;
    std::vector<std::string> block_names_;
    
    size_type total_size_;
    
    // Helper to find block index by name
    size_type find_block_index(const std::string& name) const {
        if (storage_mode_ == StorageMode::Contiguous) {
            for (size_type i = 0; i < block_info_.size(); ++i) {
                if (block_info_[i].first == name) {
                    return i;
                }
            }
        } else {
            for (size_type i = 0; i < block_names_.size(); ++i) {
                if (block_names_[i] == name) {
                    return i;
                }
            }
        }
        throw std::invalid_argument("Block '" + name + "' not found");
    }
    
    // Update offsets for contiguous storage
    void update_offsets() {
        if (storage_mode_ != StorageMode::Contiguous) return;
        
        block_offsets_.clear();
        block_offsets_.reserve(block_info_.size() + 1);
        
        size_type offset = 0;
        for (const auto& [name, size] : block_info_) {
            block_offsets_.push_back(offset);
            offset += size;
        }
        block_offsets_.push_back(offset);  // Total size
        
        total_size_ = offset;
        
        // Resize storage if necessary
        if (storage_.size() < total_size_) {
            storage_.resize(total_size_);
        }
    }
    
    // Update total size for separate storage
    void update_total_size() {
        if (storage_mode_ != StorageMode::Separate) return;
        
        total_size_ = 0;
        for (const auto& block : separate_blocks_) {
            if (block) {
                total_size_ += block->size();
            }
        }
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor - creates empty block vector
     */
    explicit BlockVector(StorageMode mode = StorageMode::Contiguous)
        : base_type(), storage_mode_(mode), storage_(), block_info_(), 
          block_offsets_(), separate_blocks_(), block_names_(), total_size_(0) {
        this->shape_ = Shape{0};
    }
    
    /**
     * @brief Constructor with block sizes
     */
    BlockVector(std::initializer_list<size_type> block_sizes, 
                StorageMode mode = StorageMode::Contiguous)
        : BlockVector(mode) {
        
        size_type block_index = 0;
        for (size_type size : block_sizes) {
            std::string name = "block_" + std::to_string(block_index);
            add_block(name, size);
            ++block_index;
        }
    }
    
    /**
     * @brief Constructor with named blocks
     */
    BlockVector(std::initializer_list<std::pair<std::string, size_type>> named_blocks,
                StorageMode mode = StorageMode::Contiguous)
        : BlockVector(mode) {
        
        for (const auto& [name, size] : named_blocks) {
            add_block(name, size);
        }
    }

    /**
     * @brief Constructor from a vector of named blocks
     */
    explicit BlockVector(const block_container& named_blocks,
                         StorageMode mode = StorageMode::Contiguous)
        : BlockVector(mode) {
        for (const auto& [name, size] : named_blocks) {
            add_block(name, size);
        }
    }

    /**
     * @brief Constructor from a movable vector of named blocks
     */
    explicit BlockVector(block_container&& named_blocks,
                         StorageMode mode = StorageMode::Contiguous)
        : BlockVector(mode) {
        for (const auto& [name, size] : named_blocks) {
            add_block(name, size);
        }
    }
    
    /**
     * @brief Copy constructor
     */
    BlockVector(const BlockVector& other)
        : base_type(other), storage_mode_(other.storage_mode_),
          block_info_(other.block_info_), block_offsets_(other.block_offsets_),
          block_names_(other.block_names_), total_size_(other.total_size_) {
        
        if (storage_mode_ == StorageMode::Contiguous) {
            storage_ = other.storage_;
        } else {
            separate_blocks_.reserve(other.separate_blocks_.size());
            for (const auto& block : other.separate_blocks_) {
                if (block) {
                    separate_blocks_.push_back(std::make_unique<Vector<T>>(*block));
                } else {
                    separate_blocks_.push_back(nullptr);
                }
            }
        }
    }
    
    /**
     * @brief Move constructor
     */
    BlockVector(BlockVector&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    BlockVector& operator=(const BlockVector& other) {
        if (this != &other) {
            base_type::operator=(other);
            storage_mode_ = other.storage_mode_;
            block_info_ = other.block_info_;
            block_offsets_ = other.block_offsets_;
            block_names_ = other.block_names_;
            total_size_ = other.total_size_;
            
            if (storage_mode_ == StorageMode::Contiguous) {
                storage_ = other.storage_;
                separate_blocks_.clear();
            } else {
                storage_ = DynamicStorage<T>();
                separate_blocks_.clear();
                separate_blocks_.reserve(other.separate_blocks_.size());
                for (const auto& block : other.separate_blocks_) {
                    if (block) {
                        separate_blocks_.push_back(std::make_unique<Vector<T>>(*block));
                    } else {
                        separate_blocks_.push_back(nullptr);
                    }
                }
            }
        }
        return *this;
    }
    
    /**
     * @brief Move assignment
     */
    BlockVector& operator=(BlockVector&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~BlockVector() = default;
    
    // === Block Management ===
    
    /**
     * @brief Add a new block with given name and size
     */
    void add_block(const std::string& name, size_type size) {
        // Check for duplicate names
        if (storage_mode_ == StorageMode::Contiguous) {
            for (const auto& [existing_name, _] : block_info_) {
                if (existing_name == name) {
                    throw std::invalid_argument("Block name '" + name + "' already exists");
                }
            }
            block_info_.emplace_back(name, size);
            update_offsets();
        } else {
            for (const auto& existing_name : block_names_) {
                if (existing_name == name) {
                    throw std::invalid_argument("Block name '" + name + "' already exists");
                }
            }
            block_names_.push_back(name);
            separate_blocks_.push_back(std::make_unique<Vector<T>>(size));
            update_total_size();
        }
        
        this->shape_ = Shape{total_size_};
    }
    
    /**
     * @brief Remove a block by name
     */
    void remove_block(const std::string& name) {
        if (storage_mode_ == StorageMode::Contiguous) {
            auto it = std::find_if(block_info_.begin(), block_info_.end(),
                [&name](const block_info& info) { return info.first == name; });
            
            if (it == block_info_.end()) {
                throw std::invalid_argument("Block '" + name + "' not found");
            }
            
            block_info_.erase(it);
            update_offsets();
        } else {
            size_type index = find_block_index(name);
            block_names_.erase(block_names_.begin() + static_cast<difference_type>(index));
            separate_blocks_.erase(separate_blocks_.begin() + static_cast<difference_type>(index));
            update_total_size();
        }
        
        this->shape_ = Shape{total_size_};
    }
    
    /**
     * @brief Get number of blocks
     */
    size_type num_blocks() const noexcept {
        if (storage_mode_ == StorageMode::Contiguous) {
            return block_info_.size();
        } else {
            return separate_blocks_.size();
        }
    }
    
    /**
     * @brief Check if block exists
     */
    bool has_block(const std::string& name) const {
        try {
            find_block_index(name);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }
    
    /**
     * @brief Get block names
     */
    std::vector<std::string> block_names() const {
        if (storage_mode_ == StorageMode::Contiguous) {
            std::vector<std::string> names;
            names.reserve(block_info_.size());
            for (const auto& [name, _] : block_info_) {
                names.push_back(name);
            }
            return names;
        } else {
            return block_names_;
        }
    }
    
    /**
     * @brief Get block size by name
     */
    size_type block_size(const std::string& name) const {
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type index = find_block_index(name);
            return block_info_[index].second;
        } else {
            size_type index = find_block_index(name);
            return separate_blocks_[index] ? separate_blocks_[index]->size() : 0;
        }
    }
    
    /**
     * @brief Get block offset (starting index) by name
     */
    size_type block_offset(const std::string& name) const {
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type index = find_block_index(name);
            return block_offsets_[index];
        } else {
            size_type index = find_block_index(name);
            size_type offset = 0;
            for (size_type i = 0; i < index; ++i) {
                if (separate_blocks_[i]) {
                    offset += separate_blocks_[i]->size();
                }
            }
            return offset;
        }
    }
    
    // === Size and Shape ===
    
    /**
     * @brief Get total size across all blocks
     */
    size_type size() const noexcept { return total_size_; }
    
    /**
     * @brief Check if block vector is empty
     */
    bool empty() const noexcept { return total_size_ == 0; }
    
    // === Block Access ===
    
    /**
     * @brief Get block by name as view
     */
    block_view_type block(const std::string& name) {
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type index = find_block_index(name);
            size_type offset = block_offsets_[index];
            size_type size = block_info_[index].second;
            return block_view_type(storage_.data() + offset, size);
        } else {
            size_type index = find_block_index(name);
            if (!separate_blocks_[index]) {
                throw std::runtime_error("Block '" + name + "' is null");
            }
            return block_view_type(separate_blocks_[index]->data(), 
                                   separate_blocks_[index]->size());
        }
    }
    
    const_block_view_type block(const std::string& name) const {
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type index = find_block_index(name);
            size_type offset = block_offsets_[index];
            size_type size = block_info_[index].second;
            return const_block_view_type(storage_.data() + offset, size);
        } else {
            size_type index = find_block_index(name);
            if (!separate_blocks_[index]) {
                throw std::runtime_error("Block '" + name + "' is null");
            }
            return const_block_view_type(separate_blocks_[index]->data(), 
                                         separate_blocks_[index]->size());
        }
    }
    
    /**
     * @brief Get block by index as view
     */
    block_view_type block(size_type index) {
        if (index >= num_blocks()) {
            throw std::out_of_range("Block index out of range");
        }
        
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type offset = block_offsets_[index];
            size_type size = block_info_[index].second;
            return block_view_type(storage_.data() + offset, size);
        } else {
            if (!separate_blocks_[index]) {
                throw std::runtime_error("Block at index " + std::to_string(index) + " is null");
            }
            return block_view_type(separate_blocks_[index]->data(), 
                                   separate_blocks_[index]->size());
        }
    }
    
    const_block_view_type block(size_type index) const {
        if (index >= num_blocks()) {
            throw std::out_of_range("Block index out of range");
        }
        
        if (storage_mode_ == StorageMode::Contiguous) {
            size_type offset = block_offsets_[index];
            size_type size = block_info_[index].second;
            return const_block_view_type(storage_.data() + offset, size);
        } else {
            if (!separate_blocks_[index]) {
                throw std::runtime_error("Block at index " + std::to_string(index) + " is null");
            }
            return const_block_view_type(separate_blocks_[index]->data(), 
                                         separate_blocks_[index]->size());
        }
    }
    
    // === Element Access ===
    
    /**
     * @brief Access element by global index with bounds checking
     */
    reference at(size_type index) {
        if (index >= total_size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    const_reference at(size_type index) const {
        if (index >= total_size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    /**
     * @brief Access element by global index without bounds checking
     */
    reference operator[](size_type index) {
        if (storage_mode_ == StorageMode::Contiguous) {
            return storage_[index];
        } else {
            // Find which block contains this index
            size_type current_offset = 0;
            for (size_type i = 0; i < separate_blocks_.size(); ++i) {
                if (!separate_blocks_[i]) continue;
                
                size_type block_size = separate_blocks_[i]->size();
                if (index < current_offset + block_size) {
                    return (*separate_blocks_[i])[index - current_offset];
                }
                current_offset += block_size;
            }
            // Should not reach here if index is valid
            throw std::out_of_range("Index out of range in separate storage");
        }
    }
    
    const_reference operator[](size_type index) const {
        if (storage_mode_ == StorageMode::Contiguous) {
            return storage_[index];
        } else {
            // Find which block contains this index
            size_type current_offset = 0;
            for (size_type i = 0; i < separate_blocks_.size(); ++i) {
                if (!separate_blocks_[i]) continue;
                
                size_type block_size = separate_blocks_[i]->size();
                if (index < current_offset + block_size) {
                    return (*separate_blocks_[i])[index - current_offset];
                }
                current_offset += block_size;
            }
            // Should not reach here if index is valid
            throw std::out_of_range("Index out of range in separate storage");
        }
    }
    
    /**
     * @brief Get raw data pointer (only for contiguous storage)
     */
    pointer data() {
        if (storage_mode_ != StorageMode::Contiguous) {
            throw std::logic_error("Raw data access only available for contiguous storage");
        }
        return storage_.data();
    }
    
    const_pointer data() const {
        if (storage_mode_ != StorageMode::Contiguous) {
            throw std::logic_error("Raw data access only available for contiguous storage");
        }
        return storage_.data();
    }
    
    // === Vector Operations ===
    
    /**
     * @brief Fill all blocks with a value
     */
    void fill(const T& value) {
        if (storage_mode_ == StorageMode::Contiguous) {
            std::fill(storage_.begin(), storage_.end(), value);
        } else {
            for (auto& block : separate_blocks_) {
                if (block) {
                    block->fill(value);
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
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Block vector addition
     */
    template<typename U>
    BlockVector& operator+=(const BlockVector<U>& other) {
        if (num_blocks() != other.num_blocks()) {
            throw std::invalid_argument("Block vectors must have same number of blocks");
        }
        
        // Check block compatibility
        for (size_type i = 0; i < num_blocks(); ++i) {
            if (block(i).size() != other.block(i).size()) {
                throw std::invalid_argument("Corresponding blocks must have same size");
            }
        }
        
        for (size_type i = 0; i < total_size_; ++i) {
            (*this)[i] += static_cast<T>(other[i]);
        }
        
        return *this;
    }
    
    /**
     * @brief Block vector subtraction
     */
    template<typename U>
    BlockVector& operator-=(const BlockVector<U>& other) {
        if (num_blocks() != other.num_blocks()) {
            throw std::invalid_argument("Block vectors must have same number of blocks");
        }
        
        // Check block compatibility
        for (size_type i = 0; i < num_blocks(); ++i) {
            if (block(i).size() != other.block(i).size()) {
                throw std::invalid_argument("Corresponding blocks must have same size");
            }
        }
        
        for (size_type i = 0; i < total_size_; ++i) {
            (*this)[i] -= static_cast<T>(other[i]);
        }
        
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    BlockVector& operator*=(const T& scalar) {
        if (storage_mode_ == StorageMode::Contiguous) {
            for (auto& elem : storage_) {
                elem *= scalar;
            }
        } else {
            for (auto& block : separate_blocks_) {
                if (block) {
                    *block *= scalar;
                }
            }
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    BlockVector& operator/=(const T& scalar) {
        if (storage_mode_ == StorageMode::Contiguous) {
            for (auto& elem : storage_) {
                elem /= scalar;
            }
        } else {
            for (auto& block : separate_blocks_) {
                if (block) {
                    *block /= scalar;
                }
            }
        }
        return *this;
    }
    
    // === Norms ===
    
    /**
     * @brief Euclidean norm across all blocks
     */
    scalar_type norm2() const {
        scalar_type sum{0};
        for (size_type i = 0; i < total_size_; ++i) {
            sum += std::norm((*this)[i]);
        }
        return std::sqrt(sum);
    }
    
    /**
     * @brief Maximum absolute value across all blocks
     */
    scalar_type max_norm() const {
        scalar_type max_val{0};
        for (size_type i = 0; i < total_size_; ++i) {
            max_val = std::max(max_val, std::abs((*this)[i]));
        }
        return max_val;
    }
    
    // === Conversion ===
    
    /**
     * @brief Convert to regular Vector (contiguous storage only)
     */
    Vector<T> to_vector() const {
        if (storage_mode_ != StorageMode::Contiguous) {
            throw std::logic_error("to_vector() only available for contiguous storage");
        }
        
        Vector<T> result(total_size_);
        std::copy_n(storage_.begin(), total_size_, result.data());
        return result;
    }
    
    // === Output ===
    
    /**
     * @brief Stream output with block structure display
     */
    friend std::ostream& operator<<(std::ostream& os, const BlockVector& bv) {
        os << "BlockVector(" << bv.num_blocks() << " blocks, " << bv.size() << " total):\n";
        
        for (size_type i = 0; i < bv.num_blocks(); ++i) {
            std::string name;
            if (bv.storage_mode_ == StorageMode::Contiguous) {
                name = bv.block_info_[i].first;
            } else {
                name = bv.block_names_[i];
            }
            
            os << "  " << name << " (" << bv.block(i).size() << "): ";
            
            auto block_view = bv.block(i);
            os << "[";
            size_type display_limit = std::min(block_view.size(), size_type(5));
            for (size_type j = 0; j < display_limit; ++j) {
                os << block_view[j];
                if (j < display_limit - 1) os << " ";
            }
            if (block_view.size() > 5) {
                os << " ...";
            }
            os << "]\n";
        }
        
        return os;
    }
};

// === Non-member Operations ===

/**
 * @brief Block vector addition
 */
template<typename T1, typename T2>
auto operator+(const BlockVector<T1>& a, const BlockVector<T2>& b) {
    using result_type = decltype(T1{} + T2{});
    BlockVector<result_type> result(a);
    result += b;
    return result;
}

/**
 * @brief Block vector subtraction
 */
template<typename T1, typename T2>
auto operator-(const BlockVector<T1>& a, const BlockVector<T2>& b) {
    using result_type = decltype(T1{} - T2{});
    BlockVector<result_type> result(a);
    result -= b;
    return result;
}

/**
 * @brief Scalar multiplication (scalar * block_vector)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockVector<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Scalar& s, const BlockVector<T>& bv) {
    using result_type = decltype(s * T{});
    BlockVector<result_type> result(bv);
    result *= static_cast<result_type>(s);
    return result;
}

/**
 * @brief Scalar multiplication (block_vector * scalar)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockVector<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const BlockVector<T>& bv, const Scalar& s) {
    return s * bv;
}

// === Block Vector View Class (Placeholder) ===

/**
 * @brief View of a block vector
 */
template<typename T>
class BlockVectorView {
    // To be implemented for advanced block vector views
};

} // namespace fem::numeric

#endif // NUMERIC_BLOCK_VECTOR_H

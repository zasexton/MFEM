#pragma once

#ifndef NUMERIC_SPARSE_VECTOR_H
#define NUMERIC_SPARSE_VECTOR_H

#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iterator>

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

namespace fem::numeric {

// Forward declarations
template<typename T> class SparseVectorView;

/**
 * @brief Sparse vector class for vectors with many zero elements
 * 
 * Implements an efficient sparse vector representation:
 * - Only stores non-zero elements to save memory
 * - Supports different storage formats (COO, sorted pairs)
 * - Efficient arithmetic operations between sparse vectors
 * - Conversion to/from dense vectors
 * - Iterator interface for non-zero elements
 * - Compatible with sparse matrix operations
 * - Optimized insertion and access patterns
 * 
 * Storage format options:
 * - COO (Coordinate): unsorted (index, value) pairs
 * - Sorted: sorted (index, value) pairs for faster searches
 * - HashMap: hash map for very sparse vectors
 * 
 * @tparam T Value type (must satisfy StorableType concept)
 */
template<typename T>
class SparseVector : public ContainerBase<SparseVector<T>, T, DynamicStorage<T>>,
                     public ExpressionBase<SparseVector<T>> {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");
    
    // === Type Aliases ===
    using base_type = ContainerBase<SparseVector<T>, T, DynamicStorage<T>>;
    using expression_base = ExpressionBase<SparseVector<T>>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Sparse-specific types
    using index_type = size_type;
    using entry_type = std::pair<index_type, T>;
    using entry_container = std::vector<entry_type>;
    using dense_vector_type = Vector<T>;
    
    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;
    
    // Storage format
    enum class StorageFormat {
        COO,        // Coordinate format (unsorted)
        Sorted,     // Sorted by index
        HashMap     // Hash map storage
    };
    
    // Iterator types for non-zero elements
    class const_nonzero_iterator;
    class nonzero_iterator;
    using const_iterator = const_nonzero_iterator;
    using iterator = nonzero_iterator;
    
private:
    size_type size_;           // Logical size of the vector
    StorageFormat format_;     // Storage format
    entry_container entries_;  // (index, value) pairs for COO/Sorted
    std::unordered_map<index_type, T> hash_entries_;  // For HashMap format
    bool sorted_;             // True if entries_ is sorted by index
    
    // Tolerance for considering elements as zero. For floating-point types,
    // use a tolerance based on machine epsilon; for integral types, use 0.
    static constexpr scalar_type zero_tolerance_ =
        std::is_floating_point_v<scalar_type>
            ? static_cast<scalar_type>(std::numeric_limits<scalar_type>::epsilon() * 100)
            : static_cast<scalar_type>(0);
    
    // Helper to check if value is effectively zero
    bool is_zero(const T& value) const noexcept {
        return std::abs(value) <= zero_tolerance_;
    }
    
    // Ensure entries are sorted (for Sorted format)
    void ensure_sorted() {
        if (format_ == StorageFormat::Sorted && !sorted_) {
            std::sort(entries_.begin(), entries_.end(),
                [](const entry_type& a, const entry_type& b) {
                    return a.first < b.first;
                });
            sorted_ = true;
        }
    }
    
    // Find entry by index in vector-based storage
    auto find_entry(index_type index) {
        if (format_ == StorageFormat::Sorted) {
            ensure_sorted();
            return std::lower_bound(entries_.begin(), entries_.end(),
                                    entry_type{index, T{}},
                                    [](const entry_type& a, const entry_type& b) {
                                        return a.first < b.first;
                                    });
        } else {
            return std::find_if(entries_.begin(), entries_.end(),
                                [index](const entry_type& entry) {
                                    return entry.first == index;
                                });
        }
    }

    auto find_entry(index_type index) const {
        if (format_ == StorageFormat::Sorted) {
            return std::lower_bound(entries_.begin(), entries_.end(),
                                    entry_type{index, T{}},
                                    [](const entry_type& a, const entry_type& b) {
                                        return a.first < b.first;
                                    });
        } else {
            return std::find_if(entries_.begin(), entries_.end(),
                                [index](const entry_type& entry) {
                                    return entry.first == index;
                                });
        }
    }
    
    // Remove zero entries
    void remove_zeros() {
        if (format_ == StorageFormat::HashMap) {
            auto it = hash_entries_.begin();
            while (it != hash_entries_.end()) {
                if (is_zero(it->second)) {
                    it = hash_entries_.erase(it);
                } else {
                    ++it;
                }
            }
        } else {
            entries_.erase(
                std::remove_if(entries_.begin(), entries_.end(),
                    [this](const entry_type& entry) {
                        return is_zero(entry.second);
                    }),
                entries_.end());
        }
    }
    
public:
    // === Constructors ===
    
    /**
     * @brief Default constructor
     */
    explicit SparseVector(size_type size = 0, 
                         StorageFormat format = StorageFormat::COO)
        : base_type(), size_(size), format_(format), entries_(), 
          hash_entries_(), sorted_(true) {
        this->shape_ = Shape{size_};
    }
    
    /**
     * @brief Constructor from dense vector
     */
    template<typename U, typename S>
    explicit SparseVector(const Vector<U, S>& dense, 
                         StorageFormat format = StorageFormat::COO,
                         scalar_type tolerance = zero_tolerance_)
        : SparseVector(dense.size(), format) {
        
        for (size_type i = 0; i < dense.size(); ++i) {
            T value = static_cast<T>(dense[i]);
            // Use a slightly conservative effective tolerance to avoid borderline noise
            scalar_type effective_tol = std::max(tolerance, zero_tolerance_) * static_cast<scalar_type>(100);
            if (std::abs(value) > effective_tol) {
                if (format_ == StorageFormat::HashMap) {
                    hash_entries_[i] = value;
                } else {
                    entries_.emplace_back(i, value);
                }
            }
        }
        
        if (format_ == StorageFormat::Sorted) {
            ensure_sorted();
        }
    }
    
    /**
     * @brief Constructor from (index, value) pairs
     */
    SparseVector(size_type size, 
                const std::vector<std::pair<index_type, T>>& pairs,
                StorageFormat format = StorageFormat::COO)
        : SparseVector(size, format) {
        
        for (const auto& [index, value] : pairs) {
            if (index >= size_) {
                throw std::out_of_range("Index out of range");
            }
            if (!is_zero(value)) {
                if (format_ == StorageFormat::HashMap) {
                    hash_entries_[index] = value;
                } else {
                    entries_.emplace_back(index, value);
                }
            }
        }
        
        if (format_ == StorageFormat::Sorted) {
            ensure_sorted();
        }
    }
    
    /**
     * @brief Copy constructor
     */
    SparseVector(const SparseVector& other) = default;
    
    /**
     * @brief Move constructor
     */
    SparseVector(SparseVector&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    SparseVector& operator=(const SparseVector& other) = default;
    
    /**
     * @brief Move assignment
     */
    SparseVector& operator=(SparseVector&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    ~SparseVector() = default;
    
    // === Size and Properties ===
    
    /**
     * @brief Get logical size of vector
     */
    size_type size() const noexcept { return size_; }
    
    /**
     * @brief Get number of non-zero elements
     */
    size_type nnz() const noexcept {
        if (format_ == StorageFormat::HashMap) {
            return hash_entries_.size();
        } else {
            return entries_.size();
        }
    }
    
    /**
     * @brief Check if vector is empty
     */
    bool empty() const noexcept { return size_ == 0; }
    
    /**
     * @brief Get sparsity ratio (fraction of non-zeros)
     */
    double sparsity() const noexcept {
        return size_ > 0
            ? static_cast<double>(nnz()) / static_cast<double>(size_)
            : 0.0;
    }
    
    /**
     * @brief Get storage format
     */
    StorageFormat format() const noexcept { return format_; }
    
    // === Element Access ===
    
    /**
     * @brief Access element with bounds checking
     */
    T at(index_type index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    /**
     * @brief Access element (read-only, returns zero for missing elements)
     */
    T operator[](index_type index) const noexcept {
        if (index >= size_) return T{0};
        
        if (format_ == StorageFormat::HashMap) {
            auto it = hash_entries_.find(index);
            return (it != hash_entries_.end()) ? it->second : T{0};
        } else {
            auto it = find_entry(index);
            if (format_ == StorageFormat::Sorted) {
                return (it != entries_.end() && it->first == index) ? it->second : T{0};
            } else {
                return (it != entries_.end()) ? it->second : T{0};
            }
        }
    }
    
    /**
     * @brief Set element value (removes if zero)
     */
    void set(index_type index, const T& value) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        
        if (is_zero(value)) {
            // Remove element if it exists
            if (format_ == StorageFormat::HashMap) {
                hash_entries_.erase(index);
            } else {
                auto it = find_entry(index);
                if (format_ == StorageFormat::Sorted) {
                    if (it != entries_.end() && it->first == index) {
                        entries_.erase(it);
                    }
                } else {
                    if (it != entries_.end()) {
                        entries_.erase(it);
                    }
                }
            }
        } else {
            // Set or update element
            if (format_ == StorageFormat::HashMap) {
                hash_entries_[index] = value;
            } else {
                auto it = find_entry(index);
                if (format_ == StorageFormat::Sorted) {
                    if (it != entries_.end() && it->first == index) {
                        it->second = value;
                    } else {
                        entries_.insert(it, {index, value});
                        sorted_ = false;  // May need resorting
                    }
                } else {
                    if (it != entries_.end()) {
                        it->second = value;
                    } else {
                        entries_.emplace_back(index, value);
                    }
                }
            }
        }
    }
    
    /**
     * @brief Add to element value
     */
    void add(index_type index, const T& value) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        
        T current = (*this)[index];
        T new_value = current + value;
        set(index, new_value);
    }
    
    // === Vector Operations ===
    
    /**
     * @brief Resize vector (may lose elements if size decreases)
     */
    void resize(size_type new_size) {
        if (new_size < size_) {
            // Remove elements with indices >= new_size
            if (format_ == StorageFormat::HashMap) {
                auto it = hash_entries_.begin();
                while (it != hash_entries_.end()) {
                    if (it->first >= new_size) {
                        it = hash_entries_.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else {
                entries_.erase(
                    std::remove_if(entries_.begin(), entries_.end(),
                        [new_size](const entry_type& entry) {
                            return entry.first >= new_size;
                        }),
                    entries_.end());
            }
        }
        
        size_ = new_size;
        this->shape_ = Shape{size_};
    }
    
    /**
     * @brief Clear all elements
     */
    void clear() noexcept {
        entries_.clear();
        hash_entries_.clear();
        sorted_ = true;
    }
    
    /**
     * @brief Reserve space for non-zero elements
     */
    void reserve(size_type capacity) {
        if (format_ != StorageFormat::HashMap) {
            entries_.reserve(capacity);
        } else {
            hash_entries_.reserve(capacity);
        }
    }
    
    /**
     * @brief Remove zero elements and optimize storage
     */
    void compress() {
        remove_zeros();
        if (format_ == StorageFormat::Sorted) {
            ensure_sorted();
        }
        if (format_ != StorageFormat::HashMap) {
            entries_.shrink_to_fit();
        }
    }
    
    // === Arithmetic Operations ===
    
    /**
     * @brief Sparse vector addition
     */
    template<typename U>
    SparseVector& operator+=(const SparseVector<U>& other) {
        if (size_ != other.size()) {
            throw std::invalid_argument("Vector sizes must match");
        }
        // Add each non-zero from other using public iteration API
        for (auto it = other.begin(); it != other.end(); ++it) {
            const auto& [index, value] = *it;
            add(index, static_cast<T>(value));
        }
        
        return *this;
    }
    
    /**
     * @brief Sparse vector subtraction
     */
    template<typename U>
    SparseVector& operator-=(const SparseVector<U>& other) {
        if (size_ != other.size()) {
            throw std::invalid_argument("Vector sizes must match");
        }
        // Subtract each non-zero from other using public iteration API
        for (auto it = other.begin(); it != other.end(); ++it) {
            const auto& [index, value] = *it;
            add(index, -static_cast<T>(value));
        }
        
        return *this;
    }
    
    /**
     * @brief Scalar multiplication
     */
    SparseVector& operator*=(const T& scalar) {
        if (is_zero(scalar)) {
            clear();
        } else {
            if (format_ == StorageFormat::HashMap) {
                for (auto& [index, value] : hash_entries_) {
                    value *= scalar;
                }
            } else {
                for (auto& [index, value] : entries_) {
                    value *= scalar;
                }
            }
        }
        return *this;
    }
    
    /**
     * @brief Scalar division
     */
    SparseVector& operator/=(const T& scalar) {
        if (is_zero(scalar)) {
            throw std::invalid_argument("Division by zero");
        }
        
        if (format_ == StorageFormat::HashMap) {
            for (auto& [index, value] : hash_entries_) {
                value /= scalar;
            }
        } else {
            for (auto& [index, value] : entries_) {
                value /= scalar;
            }
        }
        return *this;
    }
    
    // === Dot Product ===
    
    /**
     * @brief Dot product with another sparse vector
     */
    template<typename U>
    auto dot(const SparseVector<U>& other) const {
        using result_type = decltype(T{} * U{});
        
        if (size_ != other.size()) {
            throw std::invalid_argument("Vector sizes must match");
        }
        
        result_type result{0};
        
        // Choose iteration strategy based on sparsity
        if (nnz() <= other.nnz()) {
            // Iterate over this vector's non-zeros
            if (format_ == StorageFormat::HashMap) {
                for (const auto& [index, value] : hash_entries_) {
                    result += value * static_cast<result_type>(other[index]);
                }
            } else {
                for (const auto& [index, value] : entries_) {
                    result += value * static_cast<result_type>(other[index]);
                }
            }
        } else {
            // Iterate over other vector's non-zeros
            if (other.format_ == StorageFormat::HashMap) {
                for (const auto& [index, value] : other.hash_entries_) {
                    result += static_cast<result_type>((*this)[index]) * value;
                }
            } else {
                for (const auto& [index, value] : other.entries_) {
                    result += static_cast<result_type>((*this)[index]) * value;
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Dot product with dense vector
     */
    template<typename U, typename S>
    auto dot(const Vector<U, S>& dense) const {
        using result_type = decltype(T{} * U{});
        
        if (size_ != dense.size()) {
            throw std::invalid_argument("Vector sizes must match");
        }
        
        result_type result{0};
        
        if (format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : hash_entries_) {
                result += value * static_cast<result_type>(dense[index]);
            }
        } else {
            for (const auto& [index, value] : entries_) {
                result += value * static_cast<result_type>(dense[index]);
            }
        }
        
        return result;
    }
    
    // === Norms ===
    
    /**
     * @brief Euclidean norm (2-norm)
     */
    scalar_type norm2() const noexcept {
        scalar_type sum{0};
        
        if (format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : hash_entries_) {
                sum += std::norm(value);
            }
        } else {
            for (const auto& [index, value] : entries_) {
                sum += std::norm(value);
            }
        }
        
        return std::sqrt(sum);
    }
    
    /**
     * @brief 1-norm (sum of absolute values)
     */
    scalar_type norm1() const noexcept {
        scalar_type sum{0};
        
        if (format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : hash_entries_) {
                sum += std::abs(value);
            }
        } else {
            for (const auto& [index, value] : entries_) {
                sum += std::abs(value);
            }
        }
        
        return sum;
    }
    
    /**
     * @brief Infinity norm (maximum absolute value)
     */
    scalar_type norm_inf() const noexcept {
        scalar_type max_val{0};
        
        if (format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : hash_entries_) {
                max_val = std::max(max_val, std::abs(value));
            }
        } else {
            for (const auto& [index, value] : entries_) {
                max_val = std::max(max_val, std::abs(value));
            }
        }
        
        return max_val;
    }
    
    // === Conversion ===
    
    /**
     * @brief Convert to dense vector
     */
    dense_vector_type to_dense() const {
        dense_vector_type result(size_, T{0});
        
        if (format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : hash_entries_) {
                result[index] = value;
            }
        } else {
            for (const auto& [index, value] : entries_) {
                result[index] = value;
            }
        }
        
        return result;
    }
    
    /**
     * @brief Change storage format
     */
    void change_format(StorageFormat new_format) {
        if (new_format == format_) return;

        if (new_format == StorageFormat::HashMap) {
            // Convert entries_ to hash map storage
            hash_entries_.clear();
            if (format_ != StorageFormat::HashMap) {
                for (const auto& [index, value] : entries_) {
                    hash_entries_[index] = value;
                }
                entries_.clear();
            }
            format_ = StorageFormat::HashMap;
            sorted_ = true;
            return;
        }

        // Convert to vector-based formats (COO or Sorted)
        entry_container new_entries;
        if (format_ == StorageFormat::HashMap) {
            new_entries.reserve(hash_entries_.size());
            for (const auto& [index, value] : hash_entries_) {
                new_entries.emplace_back(index, value);
            }
            hash_entries_.clear();
        } else {
            new_entries = std::move(entries_);
        }
        entries_ = std::move(new_entries);

        format_ = new_format;
        sorted_ = false;
        if (format_ == StorageFormat::Sorted) {
            ensure_sorted();
        }
    }
    
    // === Iterators for Non-Zero Elements ===
    
    /**
     * @brief Iterator over non-zero elements
     */
    class const_nonzero_iterator {
        using container_iterator = typename entry_container::const_iterator;
        using hash_iterator = typename std::unordered_map<index_type, T>::const_iterator;
        
        StorageFormat format_;
        container_iterator entries_it_;
        hash_iterator hash_it_;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = entry_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const entry_type*;
        using reference = const entry_type&;
        
        const_nonzero_iterator(StorageFormat format, container_iterator it)
            : format_(format), entries_it_(it) {}
        
        const_nonzero_iterator(hash_iterator it)
            : format_(StorageFormat::HashMap), hash_it_(it) {}
        
        reference operator*() const {
            if (format_ == StorageFormat::HashMap) {
                static thread_local entry_type temp;
                temp = {hash_it_->first, hash_it_->second};
                return temp;
            } else {
                return *entries_it_;
            }
        }
        
        const_nonzero_iterator& operator++() {
            if (format_ == StorageFormat::HashMap) {
                ++hash_it_;
            } else {
                ++entries_it_;
            }
            return *this;
        }
        
        const_nonzero_iterator operator++(int) {
            const_nonzero_iterator temp = *this;
            ++(*this);
            return temp;
        }
        
        bool operator==(const const_nonzero_iterator& other) const {
            if (format_ != other.format_) return false;
            if (format_ == StorageFormat::HashMap) {
                return hash_it_ == other.hash_it_;
            } else {
                return entries_it_ == other.entries_it_;
            }
        }
        
        bool operator!=(const const_nonzero_iterator& other) const {
            return !(*this == other);
        }
    };
    
    const_nonzero_iterator begin() const {
        if (format_ == StorageFormat::HashMap) {
            return const_nonzero_iterator(hash_entries_.begin());
        } else {
            return const_nonzero_iterator(format_, entries_.begin());
        }
    }
    
    const_nonzero_iterator end() const {
        if (format_ == StorageFormat::HashMap) {
            return const_nonzero_iterator(hash_entries_.end());
        } else {
            return const_nonzero_iterator(format_, entries_.end());
        }
    }
    
    // === Output ===
    
    /**
     * @brief Stream output
     */
    friend std::ostream& operator<<(std::ostream& os, const SparseVector& sv) {
        os << "SparseVector(size=" << sv.size_ << ", nnz=" << sv.nnz() 
           << ", sparsity=" << std::fixed << std::setprecision(3) << sv.sparsity() << "):\n";
        
        if (sv.format_ == StorageFormat::HashMap) {
            for (const auto& [index, value] : sv.hash_entries_) {
                os << "  [" << index << "] = " << value << "\n";
            }
        } else {
            for (const auto& [index, value] : sv.entries_) {
                os << "  [" << index << "] = " << value << "\n";
            }
        }
        
        return os;
    }
};

// === Non-member Operations ===

/**
 * @brief Sparse vector addition
 */
template<typename T1, typename T2>
auto operator+(const SparseVector<T1>& a, const SparseVector<T2>& b) {
    using result_type = decltype(T1{} + T2{});
    SparseVector<result_type> result(a);
    result += b;
    return result;
}

/**
 * @brief Sparse vector subtraction
 */
template<typename T1, typename T2>
auto operator-(const SparseVector<T1>& a, const SparseVector<T2>& b) {
    using result_type = decltype(T1{} - T2{});
    SparseVector<result_type> result(a);
    result -= b;
    return result;
}

/**
 * @brief Scalar multiplication (scalar * sparse_vector)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, SparseVector<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const Scalar& s, const SparseVector<T>& sv) {
    using result_type = decltype(s * T{});
    SparseVector<result_type> result(sv);
    result *= static_cast<result_type>(s);
    return result;
}

/**
 * @brief Scalar multiplication (sparse_vector * scalar)
 */
template<typename T, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, SparseVector<T>>) &&
              requires (T a, Scalar s) { a * s; s * a; })
auto operator*(const SparseVector<T>& sv, const Scalar& s) {
    return s * sv;
}

// === Sparse Vector View Class (Placeholder) ===

/**
 * @brief View of a sparse vector
 */
template<typename T>
class SparseVectorView {
    // To be implemented for sparse vector views
};

} // namespace fem::numeric

#endif // NUMERIC_SPARSE_VECTOR_H

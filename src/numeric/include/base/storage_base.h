#pragma once

#ifndef NUMERIC_STORAGE_BASE_H
#define NUMERIC_STORAGE_BASE_H

#include <memory>
#include <vector>
#include <cstring>
#include <span>
#include <array>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <cstddef>

#include "numeric_base.h"
#include "traits_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for all storage implementations
     *
     * Provides shared implementations for common operations while
     * maintaining a clean interface for memory management.
     * Supports both simple numeric types and composite types like DualBase.
     */
    template<typename T>
    class StorageBase {
    public:
        using value_type = T;
        using size_type = size_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;

        static_assert(StorableType<T>, "Storage type must satisfy StorableType concept");

        StorageBase() = default;
        virtual ~StorageBase() = default;

        // === Pure Virtual Interface ===
        // These must be implemented by derived classes
        virtual size_type size() const noexcept = 0;
        virtual size_type capacity() const noexcept = 0;
        virtual pointer data() noexcept = 0;
        virtual const_pointer data() const noexcept = 0;

        virtual void resize(size_type new_size) = 0;
        virtual void resize(size_type new_size, const T& value) = 0;
        virtual void reserve(size_type new_capacity) = 0;
        virtual void clear() = 0;
        virtual std::unique_ptr<StorageBase<T>> clone() const = 0;

        // === Virtual with Default Implementation ===
        // Override only if behavior differs
        virtual Layout layout() const noexcept { return Layout::RowMajor; }
        virtual Device device() const noexcept { return Device::CPU; }
        virtual bool is_contiguous() const noexcept { return true; }

        virtual bool supports_simd() const noexcept {
            return storage_optimization_traits<T>::supports_simd;
        }

        // === Non-Virtual Shared Methods ===
        // These use the virtual interface and provide consistent behavior

        // Empty check - same for all storage types
        bool empty() const noexcept { return size() == 0; }

        // Element access with bounds checking
        reference operator[](size_type i) {
            assert_bounds(i);
            return data()[i];
        }

        const_reference operator[](size_type i) const {
            assert_bounds(i);
            return data()[i];
        }

        // Safe element access with exception
        reference at(size_type i) {
            check_bounds(i);
            return data()[i];
        }

        const_reference at(size_type i) const {
            check_bounds(i);
            return data()[i];
        }

        // Span views - valid for all contiguous storage
        std::span<T> span() noexcept {
            assert(is_contiguous() && "span() requires contiguous storage");
            return { data(), size() };
        }

        std::span<const T> span() const noexcept {
            assert(is_contiguous() && "span() requires contiguous storage");
            return { data(), size() };
        }

        // Byte views - only available for trivially copyable types
        template<typename U = T>
        std::enable_if_t<std::is_trivially_copyable_v<U>, std::span<const std::byte>>
        as_bytes() const noexcept {
            auto s = span();
            return std::as_bytes(s);
        }

        template<typename U = T>
        std::enable_if_t<std::is_trivially_copyable_v<U>, std::span<std::byte>>
        as_writable_bytes() noexcept {
            auto s = span();
            return std::as_writable_bytes(s);
        }

        // Fill operation - uses common implementation
        void fill(const T& value) {
            if constexpr (storage_optimization_traits<T>::supports_fast_fill) {
                std::fill_n(data(), size(), value);
            } else {
                // Safer for complex types like DualBase
                T* ptr = data();
                size_type n = size();
                for (size_type i = 0; i < n; ++i) {
                    ptr[i] = value;
                }
            }
        }

        // Type-safe swap with better error handling
        void swap(StorageBase& other) {
            if (typeid(*this) != typeid(other)) {
                throw std::logic_error(
                    std::string("Cannot swap incompatible storage types: ") +
                    typeid(*this).name() + " and " + typeid(other).name()
                );
            }
            do_swap(other);
        }

    protected:
        // === Protected Virtual Interface ===
        // Derived classes implement type-specific swap
        virtual void do_swap(StorageBase& other) = 0;

        // === Protected Helpers ===
        // Shared utilities for derived classes

        // Debug-mode bounds checking
        void assert_bounds(size_type i) const noexcept {
            assert(i < size() && "Storage index out of bounds");
        }

        // Release-mode bounds checking with exception
        void check_bounds(size_type i) const {
            if (i >= size()) {
                throw std::out_of_range(
                    "Storage index " + std::to_string(i) +
                    " out of range [0, " + std::to_string(size()) + ")"
                );
            }
        }
    };

    /**
     * @brief Dynamic storage using std::vector
     *
     * Standard heap-allocated storage with automatic growth
     */
    template<typename T, typename Allocator = std::allocator<T>>
    class DynamicStorage : public StorageBase<T> {
    public:
        using typename StorageBase<T>::value_type;
        using typename StorageBase<T>::size_type;
        using typename StorageBase<T>::pointer;
        using typename StorageBase<T>::const_pointer;
        using allocator_type = Allocator;

        // === Constructors ===
        DynamicStorage() = default;

        explicit DynamicStorage(size_type n) : data_(n) {}

        DynamicStorage(size_type n, const T& value) : data_(n, value) {}

        template<typename InputIt>
        requires std::input_iterator<InputIt> &&
                 std::convertible_to<typename std::iterator_traits<InputIt>::value_type, T>
        DynamicStorage(InputIt first, InputIt last) : data_(first, last) {}

        DynamicStorage(std::initializer_list<T> init) : data_(init) {}


        DynamicStorage(const DynamicStorage&) = default;
        DynamicStorage(DynamicStorage&&) noexcept = default;
        DynamicStorage& operator=(const DynamicStorage&) = default;
        DynamicStorage& operator=(DynamicStorage&&) noexcept = default;

        // StorageBase interface
        // === Required Virtual Interface ===
        size_type size() const noexcept override {
            return data_.size();
        }

        size_type capacity() const noexcept override {
            return data_.capacity();
        }

        pointer data() noexcept override {
            return data_.data();
        }

        const_pointer data() const noexcept override {
            return data_.data();
        }

        void resize(size_type new_size) override {
            data_.resize(new_size);
        }

        void resize(size_type new_size, const T& value) override {
            data_.resize(new_size, value);
        }

        void reserve(size_type new_capacity) override {
            data_.reserve(new_capacity);
        }

        void clear() override {
            data_.clear();
        }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<DynamicStorage>(*this);
        }
        
        // === Iterator Support ===
        using iterator = typename std::vector<T, Allocator>::iterator;
        using const_iterator = typename std::vector<T, Allocator>::const_iterator;
        
        iterator begin() noexcept { return data_.begin(); }
        const_iterator begin() const noexcept { return data_.begin(); }
        iterator end() noexcept { return data_.end(); }
        const_iterator end() const noexcept { return data_.end(); }

    protected:

        void do_swap(StorageBase<T>& other) override {
            auto& other_dynamic = static_cast<DynamicStorage&>(other);
            data_.swap(other_dynamic.data_);
        }

    private:
        std::vector<T, Allocator> data_;
    };

    /**
     * @brief Static storage with compile-time size
     *
     * Stack-allocated storage for small, fixed-size containers
     */
    template<typename T, size_t N>
    class StaticStorage : public StorageBase<T> {
    public:
        using typename StorageBase<T>::value_type;
        using typename StorageBase<T>::size_type;
        using typename StorageBase<T>::pointer;
        using typename StorageBase<T>::const_pointer;

        static constexpr size_type max_size() noexcept { return N; }

        // === Constructors ===
        StaticStorage() : data_{}, size_(0) {}

        explicit StaticStorage(size_type n) : data_{}, size_(n) {
            if (n > N) {
                throw std::length_error(
                    "Size " + std::to_string(n) +
                    " exceeds static storage capacity " + std::to_string(N)
                );
            }
            // Initialize elements
            for (size_type i = 0; i < n; ++i) {
                data_[i] = T{};
            }
        }

        StaticStorage(size_type n, const T& value) : data_{}, size_(n) {
            if (n > N) {
                throw std::length_error(
                    "Size " + std::to_string(n) +
                    " exceeds static storage capacity " + std::to_string(N)
                );
            }
            std::fill_n(data_.begin(), n, value);
        }

        // === Required Virtual Interface ===
        size_type size() const noexcept override {
            return size_;
        }

        size_type capacity() const noexcept override {
            return N;
        }

        pointer data() noexcept override {
            return data_.data();
        }

        const_pointer data() const noexcept override {
            return data_.data();
        }

        void resize(size_type new_size) override {
            if (new_size > N) {
                throw std::length_error("Cannot resize beyond static capacity");
            }
            if (new_size > size_) {
                std::fill(data_.begin() + size_, data_.begin() + new_size, T{});
            }
            size_ = new_size;
        }

        void resize(size_type new_size, const T& value) override {
            if (new_size > N) {
                throw std::length_error("Cannot resize beyond static capacity");
            }
            if (new_size > size_) {
                std::fill(data_.begin() + size_, data_.begin() + new_size, value);
            }
            size_ = new_size;
        }

        void reserve(size_type new_capacity) override {
            if (new_capacity > N) {
                throw std::length_error("Cannot reserve beyond static capacity");
            }
        }

        void clear() override {
            size_ = 0;
        }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<StaticStorage>(*this);
        }
    protected:
        void do_swap(StorageBase<T>& other) override {
            auto& other_static = dynamic_cast<StaticStorage&>(other);
            //if (!other_static) {
            //    throw std::logic_error("Cannot swap StaticStorage with different storage type");
            //}
            std::swap(data_, other_static.data_);
            std::swap(size_, other_static.size_);
        }
    private:
        std::array<T, N> data_;
        size_type size_;
    };

    /**
     * @brief Aligned storage for SIMD operations
     *
     * Ensures proper memory alignment for vectorized operations
     */
    template<typename T, size_t Alignment = 32>
    class AlignedStorage : public StorageBase<T> {
    public:
        using typename StorageBase<T>::value_type;
        using typename StorageBase<T>::size_type;
        using typename StorageBase<T>::pointer; // should be guarunteeded to be aligned
        using typename StorageBase<T>::const_pointer; // const version of the aligned memory pointer

        // Use storage_optimization_traits to determine best alignment
        static constexpr size_t alignment =
            storage_optimization_traits<T>::prefers_alignment ?
            std::max(Alignment, alignof(std::max_align_t)) : Alignment;

        // === Constructors and Destructor ===
        AlignedStorage() : data_(nullptr), size_(0), capacity_(0) {}

        explicit AlignedStorage(size_type n)
            : data_(nullptr), size_(n), capacity_(n) {
            allocate(n);
            construct_range(data_, data_ + n);
        }

        AlignedStorage(size_type n, const T& value)
            : data_(nullptr), size_(n), capacity_(n) {
            allocate(n);
            construct_range(data_, data_ + n, value);
        }

        ~AlignedStorage() {
            if (data_) {
                destroy_range(data_, data_ + size_);
                deallocate();
            }
        }

        AlignedStorage(const AlignedStorage& other)
            : data_(nullptr), size_(other.size_), capacity_(other.capacity_) {
            if (other.data_) {
                allocate(capacity_);
                copy_construct_range(other.data_, other.data_ + size_, data_);
            }
        }

        AlignedStorage(AlignedStorage&& other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }

        AlignedStorage& operator=(const AlignedStorage& other) {
            if (this != &other) {
                if (data_) {
                    destroy_range(data_, data_ + size_);
                    deallocate();
                }
                size_ = other.size_;
                capacity_ = other.capacity_;
                if (other.data_) {
                    allocate(capacity_);
                    copy_construct_range(other.data_, other.data_ + size_, data_);
                } else {
                    data_ = nullptr;
                }
            }
            return *this;
        }

        AlignedStorage& operator=(AlignedStorage&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    destroy_range(data_, data_ + size_);
                    deallocate();
                }
                data_ = other.data_;
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        size_type size() const noexcept override { return size_; }
        size_type capacity() const noexcept override { return capacity_; }
        pointer data() noexcept override { return data_; }
        const_pointer data() const noexcept override { return data_; }

        void resize(size_type new_size) override {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            if (new_size > size_) {
                construct_range(data_ + size_, data_ + new_size);
            } else if (new_size < size_) {
                destroy_range(data_ + new_size, data_ + size_);
            }
            size_ = new_size;
        }

        void resize(size_type new_size, const T& value) override {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            if (new_size > size_) {
                construct_range(data_ + size_, data_ + new_size, value);
            } else if (new_size < size_) {
                destroy_range(data_ + new_size, data_ + size_);
            }
            size_ = new_size;
        }

        void reserve(size_type new_capacity) override {
            if (new_capacity > capacity_) {
                reallocate(new_capacity);
            }
        }

        void clear() override {
            destroy_range(data_, data_ + size_);
            size_ = 0;
        }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<AlignedStorage>(*this);
        }

        bool supports_simd() const noexcept override {
            return storage_optimization_traits<T>::supports_simd &&
                   alignment >= 32;
        }

    protected:
        void do_swap(StorageBase<T>& other) override {
            auto& other_aligned = static_cast<AlignedStorage&>(other);
            std::swap(data_, other_aligned.data_);
            std::swap(size_, other_aligned.size_);
            std::swap(capacity_, other_aligned.capacity_);
        }

    private:
        pointer data_;
        size_type size_;
        size_type capacity_;

        void allocate(size_type n) {
            if (n > 0) {
                // Check for overflow before multiplication
                constexpr size_type max_n = std::numeric_limits<size_type>::max() / sizeof(value_type);
                if (n > max_n) {
                    throw std::bad_alloc();  // Fail early on overflow
                }

                size_type bytes = n * sizeof(value_type);

                // Check alignment calculation won't overflow
                if (bytes > std::numeric_limits<size_type>::max() - alignment + 1) {
                    throw std::bad_alloc();
                }

                size_type aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;

                // Additional sanity check
                if (aligned_bytes < bytes) {  // Overflow occurred
                    throw std::bad_alloc();
                }

                void* ptr = std::aligned_alloc(alignment, aligned_bytes);
                if (!ptr) {
                    throw std::bad_alloc();
                }
                data_ = static_cast<pointer>(ptr);
            }
        }

        void deallocate() {
            if (data_) {
                std::free(data_);
                data_ = nullptr;
            }
        }

        // Safe construction using traits to optimize
        void construct_range(pointer first, pointer last) {
            if constexpr (std::is_trivially_default_constructible_v<T>) {
                std::uninitialized_default_construct(first, last);
            } else {
                pointer current = first;
                try {
                    for (; current != last; ++current) {
                        ::new(static_cast<void*>(current)) T();
                    }
                } catch (...) {
                    destroy_range(first, current);
                    throw;
                }
            }
        }

        void construct_range(pointer first, pointer last, const T& value) {
            if constexpr (storage_optimization_traits<T>::is_trivially_relocatable) {
                std::uninitialized_fill(first, last, value);
            } else {
                pointer current = first;
                try {
                    for (; current != last; ++current) {
                        ::new(static_cast<void*>(current)) T(value);
                    }
                } catch (...) {
                    destroy_range(first, current);
                    throw;
                }
            }
        }

        void copy_construct_range(const_pointer first, const_pointer last, pointer dest) {
            if constexpr (storage_optimization_traits<T>::is_trivially_relocatable) {
                std::uninitialized_copy(first, last, dest);
            } else {
                pointer current = dest;
                try {
                    for (const_pointer src = first; src != last; ++src, ++current) {
                        ::new(static_cast<void*>(current)) T(*src);
                    }
                } catch (...) {
                    destroy_range(dest, current);
                    throw;
                }
            }
        }

        void destroy_range(pointer first, pointer last) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy(first, last);
            }
        }

        void reallocate(size_type new_capacity) {
            pointer new_data = nullptr;

            if (new_capacity > 0) {
                // Check for overflow
                constexpr size_type max_n = std::numeric_limits<size_type>::max() / sizeof(value_type);
                if (new_capacity > max_n) {
                    throw std::bad_alloc();
                }

                size_type bytes = new_capacity * sizeof(value_type);
                if (bytes > std::numeric_limits<size_type>::max() - alignment + 1) {
                    throw std::bad_alloc();
                }

                size_type aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;
                if (aligned_bytes < bytes) {
                    throw std::bad_alloc();
                }

                void* ptr = std::aligned_alloc(alignment, aligned_bytes);
                if (!ptr) {
                    throw std::bad_alloc();
                }
                new_data = static_cast<pointer>(ptr);
            }

            if (data_ && new_data) {
                // Use traits to determine best move/copy strategy
                if constexpr (std::is_nothrow_move_constructible_v<T> &&
                             storage_optimization_traits<T>::is_trivially_relocatable) {
                    try {
                        std::uninitialized_move_n(data_, size_, new_data);
                    } catch (...) {
                        std::free(new_data);
                        throw;
                    }
                } else {
                    try {
                        copy_construct_range(data_, data_ + size_, new_data);
                    } catch (...) {
                        std::free(new_data);
                        throw;
                    }
                }
                destroy_range(data_, data_ + size_);
            }

            deallocate();
            data_ = new_data;
            capacity_ = new_capacity;
        }
    };
}
#endif //NUMERIC_STORAGE_BASE_H

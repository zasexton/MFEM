#pragma once

#ifndef NUMERIC_STORAGE_BASE_H
#define NUMERIC_STORAGE_BASE_H

#include <memory>
#include <vector>
#include <cstring>
#include <span>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for all storage implementations
     *
     * Handles memory management for numeric containers with IEEE-compliant types
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

        static_assert(NumberLike<T>, "Storage type must satisfy NumberLike concept");

        StorageBase() = default;
        virtual ~StorageBase() = default;

        // Pure virtual interface
        virtual size_type size() const noexcept = 0;
        virtual size_type capacity() const noexcept = 0;
        virtual bool empty() const noexcept = 0;

        virtual pointer data() noexcept = 0;
        virtual const_pointer data() const noexcept = 0;

        virtual reference operator[](size_type i) = 0;
        virtual const_reference operator[](size_type i) const = 0;

        virtual void resize(size_type new_size) = 0;
        virtual void resize(size_type new_size, const T& value) = 0;
        virtual void reserve(size_type new_capacity) = 0;
        virtual void clear() = 0;

        virtual Layout layout() const noexcept = 0;
        virtual Device device() const noexcept = 0;
        virtual bool is_contiguous() const noexcept = 0;

        // Clone the storage
        virtual std::unique_ptr<StorageBase> clone() const = 0;

        // Memory operations
        virtual void fill(const T& value) = 0;
        virtual void swap(StorageBase& other) = 0;
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
        using typename StorageBase<T>::reference;
        using typename StorageBase<T>::const_reference;
        using allocator_type = Allocator;

        DynamicStorage() = default;

        explicit DynamicStorage(size_type n)
                : data_(n) {}

        DynamicStorage(size_type n, const T& value)
                : data_(n, value) {}

        template<typename InputIt>
        DynamicStorage(InputIt first, InputIt last)
                : data_(first, last) {}

        DynamicStorage(std::initializer_list<T> init)
                : data_(init) {}

        DynamicStorage(const DynamicStorage&) = default;
        DynamicStorage(DynamicStorage&&) noexcept = default;
        DynamicStorage& operator=(const DynamicStorage&) = default;
        DynamicStorage& operator=(DynamicStorage&&) noexcept = default;

        // StorageBase interface
        size_type size() const noexcept override { return data_.size(); }
        size_type capacity() const noexcept override { return data_.capacity(); }
        bool empty() const noexcept override { return data_.empty(); }

        pointer data() noexcept override { return data_.data(); }
        const_pointer data() const noexcept override { return data_.data(); }

        reference operator[](size_type i) override { return data_[i]; }
        const_reference operator[](size_type i) const override { return data_[i]; }

        void resize(size_type new_size) override { data_.resize(new_size); }
        void resize(size_type new_size, const T& value) override {
            data_.resize(new_size, value);
        }
        void reserve(size_type new_capacity) override { data_.reserve(new_capacity); }
        void clear() override { data_.clear(); }

        Layout layout() const noexcept override { return Layout::RowMajor; }
        Device device() const noexcept override { return Device::CPU; }
        bool is_contiguous() const noexcept override { return true; }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<DynamicStorage>(*this);
        }

        void fill(const T& value) override {
            std::fill(data_.begin(), data_.end(), value);
        }

        void swap(StorageBase<T>& other) override {
            if (auto* other_dynamic = dynamic_cast<DynamicStorage*>(&other)) {
                data_.swap(other_dynamic->data_);
            } else {
                throw std::runtime_error("Cannot swap different storage types");
            }
        }

        // Additional vector-like operations
        reference front() { return data_.front(); }
        const_reference front() const { return data_.front(); }
        reference back() { return data_.back(); }
        const_reference back() const { return data_.back(); }

        void push_back(const T& value) { data_.push_back(value); }
        void push_back(T&& value) { data_.push_back(std::move(value)); }
        void pop_back() { data_.pop_back(); }

        auto begin() noexcept { return data_.begin(); }
        auto begin() const noexcept { return data_.begin(); }
        auto end() noexcept { return data_.end(); }
        auto end() const noexcept { return data_.end(); }

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
        using typename StorageBase<T>::reference;
        using typename StorageBase<T>::const_reference;

        StaticStorage() : size_(0) {
            // Initialize with default values for IEEE compliance
            std::fill(data_.begin(), data_.end(), T{});
        }

        explicit StaticStorage(size_type n) : size_(n) {
            if (n > N) {
                throw std::length_error("Size exceeds static storage capacity");
            }
            std::fill(data_.begin(), data_.begin() + n, T{});
        }

        StaticStorage(size_type n, const T& value) : size_(n) {
            if (n > N) {
                throw std::length_error("Size exceeds static storage capacity");
            }
            std::fill(data_.begin(), data_.begin() + n, value);
        }

        // StorageBase interface
        size_type size() const noexcept override { return size_; }
        size_type capacity() const noexcept override { return N; }
        bool empty() const noexcept override { return size_ == 0; }

        pointer data() noexcept override { return data_.data(); }
        const_pointer data() const noexcept override { return data_.data(); }

        reference operator[](size_type i) override { return data_[i]; }
        const_reference operator[](size_type i) const override { return data_[i]; }

        void resize(size_type new_size) override {
            if (new_size > N) {
                throw std::length_error("Cannot resize beyond static capacity");
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

        void clear() override { size_ = 0; }

        Layout layout() const noexcept override { return Layout::RowMajor; }
        Device device() const noexcept override { return Device::CPU; }
        bool is_contiguous() const noexcept override { return true; }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<StaticStorage>(*this);
        }

        void fill(const T& value) override {
            std::fill(data_.begin(), data_.begin() + size_, value);
        }

        void swap(StorageBase<T>& other) override {
            if (auto* other_static = dynamic_cast<StaticStorage*>(&other)) {
                std::swap(data_, other_static->data_);
                std::swap(size_, other_static->size_);
            } else {
                throw std::runtime_error("Cannot swap different storage types");
            }
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
        using typename StorageBase<T>::pointer;
        using typename StorageBase<T>::const_pointer;
        using typename StorageBase<T>::reference;
        using typename StorageBase<T>::const_reference;

        static constexpr size_t alignment = Alignment;

        AlignedStorage() : data_(nullptr), size_(0), capacity_(0) {}

        explicit AlignedStorage(size_type n)
                : data_(nullptr), size_(n), capacity_(n) {
            allocate(n);
            std::uninitialized_fill_n(data_, n, T{});
        }

        AlignedStorage(size_type n, const T& value)
                : data_(nullptr), size_(n), capacity_(n) {
            allocate(n);
            std::uninitialized_fill_n(data_, n, value);
        }

        ~AlignedStorage() {
            if (data_) {
                std::destroy_n(data_, size_);
                deallocate();
            }
        }

        AlignedStorage(const AlignedStorage& other)
                : data_(nullptr), size_(other.size_), capacity_(other.capacity_) {
            if (other.data_) {
                allocate(capacity_);
                std::uninitialized_copy_n(other.data_, size_, data_);
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
                    std::destroy_n(data_, size_);
                    deallocate();
                }
                size_ = other.size_;
                capacity_ = other.capacity_;
                if (other.data_) {
                    allocate(capacity_);
                    std::uninitialized_copy_n(other.data_, size_, data_);
                }
            }
            return *this;
        }

        AlignedStorage& operator=(AlignedStorage&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    std::destroy_n(data_, size_);
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

        // StorageBase interface
        size_type size() const noexcept override { return size_; }
        size_type capacity() const noexcept override { return capacity_; }
        bool empty() const noexcept override { return size_ == 0; }

        pointer data() noexcept override { return data_; }
        const_pointer data() const noexcept override { return data_; }

        reference operator[](size_type i) override { return data_[i]; }
        const_reference operator[](size_type i) const override { return data_[i]; }

        void resize(size_type new_size) override {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            if (new_size > size_) {
                std::uninitialized_fill(data_ + size_, data_ + new_size, T{});
            } else if (new_size < size_) {
                std::destroy(data_ + new_size, data_ + size_);
            }
            size_ = new_size;
        }

        void resize(size_type new_size, const T& value) override {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            if (new_size > size_) {
                std::uninitialized_fill(data_ + size_, data_ + new_size, value);
            } else if (new_size < size_) {
                std::destroy(data_ + new_size, data_ + size_);
            }
            size_ = new_size;
        }

        void reserve(size_type new_capacity) override {
            if (new_capacity > capacity_) {
                reallocate(new_capacity);
            }
        }

        void clear() override {
            std::destroy_n(data_, size_);
            size_ = 0;
        }

        Layout layout() const noexcept override { return Layout::RowMajor; }
        Device device() const noexcept override { return Device::CPU; }
        bool is_contiguous() const noexcept override { return true; }

        std::unique_ptr<StorageBase<T>> clone() const override {
            return std::make_unique<AlignedStorage>(*this);
        }

        void fill(const T& value) override {
            std::fill_n(data_, size_, value);
        }

        void swap(StorageBase<T>& other) override {
            if (auto* other_aligned = dynamic_cast<AlignedStorage*>(&other)) {
                std::swap(data_, other_aligned->data_);
                std::swap(size_, other_aligned->size_);
                std::swap(capacity_, other_aligned->capacity_);
            } else {
                throw std::runtime_error("Cannot swap different storage types");
            }
        }

    private:
        pointer data_;
        size_type size_;
        size_type capacity_;

        void allocate(size_type n) {
            if (n > 0) {
                void* ptr = std::aligned_alloc(alignment, n * sizeof(T));
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

        void reallocate(size_type new_capacity) {
            pointer new_data = nullptr;
            if (new_capacity > 0) {
                void* ptr = std::aligned_alloc(alignment, new_capacity * sizeof(T));
                if (!ptr) {
                    throw std::bad_alloc();
                }
                new_data = static_cast<pointer>(ptr);
            }

            if (data_ && new_data) {
                std::uninitialized_move_n(data_, size_, new_data);
                std::destroy_n(data_, size_);
            }

            deallocate();
            data_ = new_data;
            capacity_ = new_capacity;
        }
    };

#endif //NUMERIC_STORAGE_BASE_H

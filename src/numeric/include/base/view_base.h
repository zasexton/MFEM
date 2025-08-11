#pragma once

#ifndef NUMERIC_VIEW_BASE_H
#define NUMERIC_VIEW_BASE_H

#include <span>

#include "numeric_base.h"
#include "storage_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for views (non-owning references to data)
     *
     * Provides interface for creating views of IEEE-compliant numeric containers
     */
    template<typename T>
    class ViewBase {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        ViewBase() noexcept : data_(nullptr), size_(0) {}

        ViewBase(pointer data, size_type size) noexcept
                : data_(data), size_(size) {}

        // Basic accessors
        pointer data() noexcept { return data_; }
        const_pointer data() const noexcept { return data_; }
        size_type size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }

        // Element access
        reference operator[](size_type i) { return data_[i]; }
        const_reference operator[](size_type i) const { return data_[i]; }

        reference at(size_type i) {
            if (i >= size_) {
                throw std::out_of_range("View index out of range");
            }
            return data_[i];
        }

        const_reference at(size_type i) const {
            if (i >= size_) {
                throw std::out_of_range("View index out of range");
            }
            return data_[i];
        }

        reference front() { return data_[0]; }
        const_reference front() const { return data_[0]; }
        reference back() { return data_[size_ - 1]; }
        const_reference back() const { return data_[size_ - 1]; }

        // View creation
        ViewBase subview(size_type offset, size_type count) const {
            if (offset + count > size_) {
                throw std::out_of_range("Subview out of range");
            }
            return ViewBase(data_ + offset, count);
        }

        // Conversion to std::span (C++20)
        operator std::span<T>() { return std::span<T>(data_, size_); }
        operator std::span<const T>() const { return std::span<const T>(data_, size_); }

        // Check if view is valid
        bool is_valid() const noexcept { return data_ != nullptr; }

    protected:
        pointer data_;
        size_type size_;
    };

    /**
     * @brief Strided view for non-contiguous data access
     */
    template<typename T>
    class StridedView : public ViewBase<T> {
    public:
        using typename ViewBase<T>::value_type;
        using typename ViewBase<T>::pointer;
        using typename ViewBase<T>::size_type;
        using typename ViewBase<T>::reference;
        using typename ViewBase<T>::const_reference;
        using typename ViewBase<T>::difference_type;

        StridedView() noexcept : ViewBase<T>(), stride_(0) {}

        StridedView(pointer data, size_type size, difference_type stride) noexcept
                : ViewBase<T>(data, size), stride_(stride) {}

        // Element access with stride
        reference operator[](size_type i) {
            return this->data_[i * stride_];
        }

        const_reference operator[](size_type i) const {
            return this->data_[i * stride_];
        }

        reference at(size_type i) {
            if (i >= this->size_) {
                throw std::out_of_range("Strided view index out of range");
            }
            return this->data_[i * stride_];
        }

        const_reference at(size_type i) const {
            if (i >= this->size_) {
                throw std::out_of_range("Strided view index out of range");
            }
            return this->data_[i * stride_];
        }

        difference_type stride() const noexcept { return stride_; }

        bool is_contiguous() const noexcept { return stride_ == 1; }

        // Create subview with same stride
        StridedView subview(size_type offset, size_type count) const {
            if (offset + count > this->size_) {
                throw std::out_of_range("Subview out of range");
            }
            return StridedView(this->data_ + offset * stride_, count, stride_);
        }

    private:
        difference_type stride_;
    };

    /**
     * @brief Multi-dimensional view
     */
    template<typename T, size_t Rank>
    class MultiDimView {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = size_t;
        using shape_type = std::array<size_type, Rank>;
        using stride_type = std::array<std::ptrdiff_t, Rank>;

        MultiDimView() noexcept : data_(nullptr), shape_{}, strides_{} {}

        MultiDimView(pointer data, const shape_type& shape, const stride_type& strides)
                : data_(data), shape_(shape), strides_(strides) {}

        // Static factory for contiguous data
        static MultiDimView from_contiguous(pointer data, const shape_type& shape) {
            stride_type strides;
            std::ptrdiff_t stride = 1;
            for (size_t i = Rank; i > 0; --i) {
                strides[i-1] = stride;
                stride *= shape[i-1];
            }
            return MultiDimView(data, shape, strides);
        }

        // Multi-dimensional indexing
        template<typename... Indices>
        reference operator()(Indices... indices) {
            static_assert(sizeof...(indices) == Rank, "Wrong number of indices");
            return data_[linear_index(indices...)];
        }

        template<typename... Indices>
        const_reference operator()(Indices... indices) const {
            static_assert(sizeof...(indices) == Rank, "Wrong number of indices");
            return data_[linear_index(indices...)];
        }

        // Shape and size
        const shape_type& shape() const noexcept { return shape_; }
        const stride_type& strides() const noexcept { return strides_; }

        size_type size() const noexcept {
            size_type s = 1;
            for (size_t dim : shape_) {
                s *= dim;
            }
            return s;
        }

        size_type size(size_t dim) const {
            return shape_[dim];
        }

        std::ptrdiff_t stride(size_t dim) const {
            return strides_[dim];
        }

        pointer data() noexcept { return data_; }
        const_pointer data() const noexcept { return data_; }

        bool is_contiguous() const noexcept {
            std::ptrdiff_t expected = 1;
            for (size_t i = Rank; i > 0; --i) {
                if (strides_[i-1] != expected) return false;
                expected *= shape_[i-1];
            }
            return true;
        }

        // Create slice along dimension
        MultiDimView<T, Rank> slice(size_t dim, size_t index) const {
            if (dim >= Rank || index >= shape_[dim]) {
                throw std::out_of_range("Slice index out of range");
            }

            shape_type new_shape = shape_;
            new_shape[dim] = 1;

            pointer new_data = data_ + index * strides_[dim];

            return MultiDimView(new_data, new_shape, strides_);
        }

    private:
        pointer data_;
        shape_type shape_;
        stride_type strides_;

        template<typename... Indices>
        size_type linear_index(Indices... indices) const {
            std::array<size_type, Rank> idx_array{static_cast<size_type>(indices)...};
            size_type linear_idx = 0;

            for (size_t i = 0; i < Rank; ++i) {
                if (idx_array[i] >= shape_[i]) {
                    throw std::out_of_range("Index out of bounds");
                }
                linear_idx += idx_array[i] * strides_[i];
            }

            return linear_idx;
        }
    };

    /**
     * @brief View factory for creating different view types
     */
    class ViewFactory {
    public:
        template<typename T>
        static ViewBase<T> create_view(T* data, size_t size) {
            return ViewBase<T>(data, size);
        }

        template<typename T>
        static StridedView<T> create_strided_view(T* data, size_t size, std::ptrdiff_t stride) {
            return StridedView<T>(data, size, stride);
        }

        template<typename T, size_t Rank>
        static MultiDimView<T, Rank> create_multidim_view(
                T* data,
                const std::array<size_t, Rank>& shape,
                const std::array<std::ptrdiff_t, Rank>& strides) {
            return MultiDimView<T, Rank>(data, shape, strides);
        }

        // Create a transposed view (2D only)
        template<typename T>
        static MultiDimView<T, 2> create_transposed_view(
                T* data,
                size_t rows,
                size_t cols) {
            return MultiDimView<T, 2>(
                    data,
                    {cols, rows},  // Swap dimensions
                    {1, static_cast<std::ptrdiff_t>(cols)}  // Swap strides
            );
        }
    };

} // namespace fem::numeric

#endif //MFEM_VIEW_BASE_H

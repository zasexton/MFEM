#pragma once

#ifndef NUMERIC_ITERATOR_BASE_H
#define NUMERIC_ITERATOR_BASE_H

#include <iterator>
#include <type_traits>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base iterator for numeric containers
     *
     * Provides standard iterator interface with IEEE-compliant value checking
     */
    template<typename T>
    class ContainerIterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        ContainerIterator() noexcept : ptr_(nullptr) {}
        explicit ContainerIterator(pointer ptr) noexcept : ptr_(ptr) {}

        reference operator*() const noexcept { return *ptr_; }
        pointer operator->() const noexcept { return ptr_; }

        reference operator[](difference_type n) const noexcept {
            return ptr_[n];
        }

        ContainerIterator& operator++() noexcept {
            ++ptr_;
            return *this;
        }

        ContainerIterator operator++(int) noexcept {
            ContainerIterator tmp = *this;
            ++ptr_;
            return tmp;
        }

        ContainerIterator& operator--() noexcept {
            --ptr_;
            return *this;
        }

        ContainerIterator operator--(int) noexcept {
            ContainerIterator tmp = *this;
            --ptr_;
            return tmp;
        }

        ContainerIterator& operator+=(difference_type n) noexcept {
            ptr_ += n;
            return *this;
        }

        ContainerIterator& operator-=(difference_type n) noexcept {
            ptr_ -= n;
            return *this;
        }

        friend ContainerIterator operator+(ContainerIterator it, difference_type n) noexcept {
            return ContainerIterator(it.ptr_ + n);
        }

        friend ContainerIterator operator+(difference_type n, ContainerIterator it) noexcept {
            return ContainerIterator(it.ptr_ + n);
        }

        friend ContainerIterator operator-(ContainerIterator it, difference_type n) noexcept {
            return ContainerIterator(it.ptr_ - n);
        }

        friend difference_type operator-(const ContainerIterator& lhs,
                                         const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ - rhs.ptr_;
        }

        friend bool operator==(const ContainerIterator& lhs,
                               const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ == rhs.ptr_;
        }

        friend bool operator!=(const ContainerIterator& lhs,
                               const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ != rhs.ptr_;
        }

        friend bool operator<(const ContainerIterator& lhs,
                              const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ < rhs.ptr_;
        }

        friend bool operator>(const ContainerIterator& lhs,
                              const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ > rhs.ptr_;
        }

        friend bool operator<=(const ContainerIterator& lhs,
                               const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ <= rhs.ptr_;
        }

        friend bool operator>=(const ContainerIterator& lhs,
                               const ContainerIterator& rhs) noexcept {
            return lhs.ptr_ >= rhs.ptr_;
        }

        pointer base() const noexcept { return ptr_; }

    private:
        pointer ptr_;
    };

    /**
     * @brief Strided iterator for non-contiguous data access
     *
     * Supports views with custom strides for IEEE-compliant numeric types
     */
    template<typename T>
    class StridedIterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        StridedIterator() noexcept
                : ptr_(nullptr), stride_(0) {}

        StridedIterator(pointer ptr, difference_type stride) noexcept
                : ptr_(ptr), stride_(stride) {}

        reference operator*() const noexcept { return *ptr_; }
        pointer operator->() const noexcept { return ptr_; }

        reference operator[](difference_type n) const noexcept {
            return ptr_[n * stride_];
        }

        StridedIterator& operator++() noexcept {
            ptr_ += stride_;
            return *this;
        }

        StridedIterator operator++(int) noexcept {
            StridedIterator tmp = *this;
            ptr_ += stride_;
            return tmp;
        }

        StridedIterator& operator--() noexcept {
            ptr_ -= stride_;
            return *this;
        }

        StridedIterator operator--(int) noexcept {
            StridedIterator tmp = *this;
            ptr_ -= stride_;
            return tmp;
        }

        StridedIterator& operator+=(difference_type n) noexcept {
            ptr_ += n * stride_;
            return *this;
        }

        StridedIterator& operator-=(difference_type n) noexcept {
            ptr_ -= n * stride_;
            return *this;
        }

        friend StridedIterator operator+(StridedIterator it, difference_type n) noexcept {
            return StridedIterator(it.ptr_ + n * it.stride_, it.stride_);
        }

        friend StridedIterator operator+(difference_type n, StridedIterator it) noexcept {
            return StridedIterator(it.ptr_ + n * it.stride_, it.stride_);
        }

        friend StridedIterator operator-(StridedIterator it, difference_type n) noexcept {
            return StridedIterator(it.ptr_ - n * it.stride_, it.stride_);
        }

        friend difference_type operator-(const StridedIterator& lhs,
                                         const StridedIterator& rhs) noexcept {
            return (lhs.ptr_ - rhs.ptr_) / lhs.stride_;
        }

        friend bool operator==(const StridedIterator& lhs,
                               const StridedIterator& rhs) noexcept {
            return lhs.ptr_ == rhs.ptr_;
        }

        friend bool operator!=(const StridedIterator& lhs,
                               const StridedIterator& rhs) noexcept {
            return lhs.ptr_ != rhs.ptr_;
        }

        friend bool operator<(const StridedIterator& lhs,
                              const StridedIterator& rhs) noexcept {
            return lhs.ptr_ < rhs.ptr_;
        }

        friend bool operator>(const StridedIterator& lhs,
                              const StridedIterator& rhs) noexcept {
            return lhs.ptr_ > rhs.ptr_;
        }

        friend bool operator<=(const StridedIterator& lhs,
                               const StridedIterator& rhs) noexcept {
            return lhs.ptr_ <= rhs.ptr_;
        }

        friend bool operator>=(const StridedIterator& lhs,
                               const StridedIterator& rhs) noexcept {
            return lhs.ptr_ >= rhs.ptr_;
        }

        pointer base() const noexcept { return ptr_; }
        difference_type stride() const noexcept { return stride_; }

    private:
        pointer ptr_;
        difference_type stride_;
    };

    /**
     * @brief Multi-dimensional iterator for tensors
     *
     * Iterates over multi-dimensional arrays with proper index calculation
     */
    template<typename T, size_t Rank>
    class MultiDimIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        using index_type = std::array<size_t, Rank>;

        MultiDimIterator() noexcept
                : ptr_(nullptr), strides_{}, shape_{}, indices_{} {}

        MultiDimIterator(pointer ptr, const index_type& strides,
                         const index_type& shape) noexcept
                : ptr_(ptr), strides_(strides), shape_(shape), indices_{} {}

        reference operator*() const noexcept {
            return ptr_[linear_index()];
        }

        pointer operator->() const noexcept {
            return &ptr_[linear_index()];
        }

        MultiDimIterator& operator++() noexcept {
            // Increment indices with carry
            for (size_t i = Rank; i > 0; --i) {
                if (++indices_[i-1] < shape_[i-1]) {
                    break;
                }
                if (i > 1) {
                    indices_[i-1] = 0;
                }
            }
            return *this;
        }

        MultiDimIterator operator++(int) noexcept {
            MultiDimIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const MultiDimIterator& lhs,
                               const MultiDimIterator& rhs) noexcept {
            return lhs.indices_ == rhs.indices_;
        }

        friend bool operator!=(const MultiDimIterator& lhs,
                               const MultiDimIterator& rhs) noexcept {
            return !(lhs == rhs);
        }

        const index_type& indices() const noexcept { return indices_; }

        bool is_end() const noexcept {
            return indices_[0] >= shape_[0];
        }

    private:
        pointer ptr_;
        index_type strides_;
        index_type shape_;
        index_type indices_;

        size_t linear_index() const noexcept {
            size_t idx = 0;
            for (size_t i = 0; i < Rank; ++i) {
                idx += indices_[i] * strides_[i];
            }
            return idx;
        }
    };

    /**
     * @brief Checked iterator that validates IEEE compliance
     *
     * Wraps another iterator and checks values for NaN/Inf when configured
     */
    template<typename Iterator>
    class CheckedIterator {
    public:
        using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
        using value_type = typename std::iterator_traits<Iterator>::value_type;
        using difference_type = typename std::iterator_traits<Iterator>::difference_type;
        using pointer = typename std::iterator_traits<Iterator>::pointer;
        using reference = typename std::iterator_traits<Iterator>::reference;

        CheckedIterator() = default;
        explicit CheckedIterator(Iterator it) : it_(it) {}

        reference operator*() const {
            if (NumericOptions::defaults().check_finite) {
                check_value(*it_);
            }
            return *it_;
        }

        pointer operator->() const {
            if (NumericOptions::defaults().check_finite) {
                check_value(*it_);
            }
            return it_.operator->();
        }

        CheckedIterator& operator++() {
            ++it_;
            return *this;
        }

        CheckedIterator operator++(int) {
            CheckedIterator tmp = *this;
            ++it_;
            return tmp;
        }

        // Forward other operations to underlying iterator
        template<typename... Args>
        auto operator[](Args&&... args) const
        -> decltype(std::declval<Iterator>().operator[](std::forward<Args>(args)...)) {

            if constexpr (IEEECompliant<value_type>) {
                if (NumericOptions::defaults().check_finite) {
                    // Use function call syntax instead of subscript
                    decltype(auto) val = it_.operator[](std::forward<Args>(args)...);

                    // Check if finite
                    if (!IEEEComplianceChecker::is_finite(val)) {
                        throw ComputationError("Non-finite value detected in iterator access");
                    }

                    // Return the value
                    return val;
                }
            }

            // Pass through without checking
            return it_.operator[](std::forward<Args>(args)...);
        }

        friend bool operator==(const CheckedIterator& lhs, const CheckedIterator& rhs) {
            return lhs.it_ == rhs.it_;
        }

        friend bool operator!=(const CheckedIterator& lhs, const CheckedIterator& rhs) {
            return lhs.it_ != rhs.it_;
        }

        Iterator base() const { return it_; }

    private:
        Iterator it_;

        void check_value(const value_type& val) const {
            if (IEEEComplianceChecker::is_nan(val)) {
                throw ComputationError("NaN value encountered during iteration");
            }
            if (IEEEComplianceChecker::is_inf(val)) {
                throw ComputationError("Infinite value encountered during iteration");
            }
        }
    };

    /**
     * @brief Iterator traits for numeric iterators
     */
    template<typename Iterator>
    struct numeric_iterator_traits {
        using value_type = typename std::iterator_traits<Iterator>::value_type;
        static constexpr bool is_numeric = NumberLike<value_type>;
        static constexpr bool is_ieee_compliant = IEEECompliant<value_type>;
        static constexpr bool is_contiguous =
                std::is_same_v<Iterator, ContainerIterator<value_type>>;
        static constexpr bool is_strided =
                std::is_same_v<Iterator, StridedIterator<value_type>>;
    };

} // namespace fem::numeric

#endif //NUMERIC_ITERATOR_BASE_H

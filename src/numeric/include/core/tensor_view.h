/*
 * Tensor view types extracted from tensor.h to compartmentalize code.
 */

#pragma once

#ifndef NUMERIC_TENSOR_VIEW_H
#define NUMERIC_TENSOR_VIEW_H

#include <cstddef>
#include <type_traits>
#include <optional>
#include <vector>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <array>
#include <utility>

#include "../base/numeric_base.h"
#include "../base/expression_base.h"

namespace fem::numeric {

template<typename Derived, typename ValueType>
class TensorViewBase : public ExpressionBase<Derived> {
public:
    using value_type = ValueType;

    Shape shape() const { return static_cast<const Derived&>(*this).shape_impl(); }
    size_t size() const noexcept { return static_cast<const Derived&>(*this).size_impl(); }
    size_t rank() const noexcept { return static_cast<const Derived&>(*this).rank_impl(); }
    bool is_parallelizable() const noexcept { return true; }
    bool is_vectorizable() const noexcept { return static_cast<const Derived&>(*this).is_vectorizable_impl(); }
    size_t complexity() const noexcept { return size(); }
};

template<typename T, size_t Rank>
class TensorView : public TensorViewBase<TensorView<T, Rank>, std::remove_const_t<T>> {
public:
    using value_type = std::remove_const_t<T>;
    using pointer = std::conditional_t<std::is_const_v<T>, const value_type*, value_type*>;
    using reference = std::conditional_t<std::is_const_v<T>, const value_type&, value_type&>;
    using const_reference = const value_type&;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    TensorView() = default;

    using index_map_type = std::optional<std::vector<size_type>>;

    TensorView(pointer data,
               std::vector<size_type> shape,
               std::vector<difference_type> strides,
               std::vector<index_map_type> index_maps = {})
        : data_(data),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          index_maps_(index_maps.empty()
                          ? std::vector<index_map_type>(shape_.size())
                          : std::move(index_maps)) {
        if (index_maps_.size() != shape_.size()) {
            throw std::invalid_argument("TensorView index map size must match shape rank");
        }
    }

    pointer data() const noexcept { return data_; }

    size_t rank_impl() const noexcept { return shape_.size(); }

    size_type size_impl() const noexcept {
        if (shape_.empty()) {
            return 1;
        }
        return std::accumulate(shape_.begin(), shape_.end(), size_type{1}, std::multiplies<size_type>());
    }

    Shape shape_impl() const { return Shape(shape_); }

    bool empty() const noexcept { return size_impl() == 0; }

    reference operator[](size_type index) {
        if (size_impl() == 0) {
            throw std::out_of_range("TensorView index out of range");
        }
        return data_[offset_from_linear(index)];
    }

    const_reference operator[](size_type index) const {
        if (size_impl() == 0) {
            throw std::out_of_range("TensorView index out of range");
        }
        return data_[offset_from_linear(index)];
    }

    template<typename... Indices>
    reference operator()(Indices... indices) {
        static_assert(sizeof...(indices) >= 0, "Invalid number of indices");
        std::array<size_type, sizeof...(indices)> idx{static_cast<size_type>(indices)...};
        if (idx.size() != shape_.size()) {
            throw std::out_of_range("Incorrect number of indices for TensorView");
        }
        return data_[offset_from_indices(idx.data(), idx.size())];
    }

    template<typename... Indices>
    const_reference operator()(Indices... indices) const {
        static_assert(sizeof...(indices) >= 0, "Invalid number of indices");
        std::array<size_type, sizeof...(indices)> idx{static_cast<size_type>(indices)...};
        if (idx.size() != shape_.size()) {
            throw std::out_of_range("Incorrect number of indices for TensorView");
        }
        return data_[offset_from_indices(idx.data(), idx.size())];
    }

    template<typename U>
    auto eval(size_type i) const {
        return static_cast<U>(data_[offset_from_linear(i)]);
    }

    template<typename Container>
    void eval_to(Container& result) const {
        if (result.shape() != shape_impl()) {
            result.resize(shape_impl());
        }
        using result_type = typename Container::value_type;
        const size_type total = size_impl();
        for (size_type i = 0; i < total; ++i) {
            result.data()[i] = static_cast<result_type>(data_[offset_from_linear(i)]);
        }
    }

    bool is_vectorizable_impl() const noexcept {
        if (shape_.empty()) { return true; }
        // Vectorizable if last stride is 1 and all others are non-negative,
        // and there are no gather maps
        if (strides_.empty()) { return true; }
        for (const auto& map : index_maps_) {
            if (map.has_value()) {
                return false;
            }
        }
        if (strides_.back() != 1) { return false; }
        for (size_t i = 0; i + 1 < strides_.size(); ++i) {
            if (strides_[i] <= 0) { return false; }
        }
        return true;
    }

    const std::vector<size_type>& dims() const noexcept { return shape_; }
    const std::vector<difference_type>& strides() const noexcept { return strides_; }

private:
    difference_type offset_from_linear(size_type index) const {
        if (shape_.empty()) {
            return 0;
        }

        std::vector<size_type> coords(shape_.size(), 0);
        size_type remainder = index;
        for (size_t i = shape_.size(); i-- > 0;) {
            size_type dim = shape_[i];
            if (dim == 0) {
                return 0;
            }
            coords[i] = remainder % dim;
            remainder /= dim;
        }
        return offset_from_indices(coords.data(), coords.size());
    }

    difference_type offset_from_indices(const size_type* idx, size_t count) const {
        difference_type offset = 0;
        for (size_t i = 0; i < count; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("TensorView index out of range");
            }
            size_type actual_index = idx[i];
            if (i < index_maps_.size() && index_maps_[i]) {
                const auto& mapped = index_maps_[i].value();
                if (actual_index >= mapped.size()) {
                    throw std::out_of_range("TensorView gather index out of range");
                }
                actual_index = mapped[actual_index];
            }
            offset += static_cast<difference_type>(actual_index) * strides_[i];
        }
        return offset;
    }

    pointer data_ = nullptr;
    std::vector<size_type> shape_;
    std::vector<difference_type> strides_;
    std::vector<index_map_type> index_maps_;
};

} // namespace fem::numeric

#endif // NUMERIC_TENSOR_VIEW_H

#pragma once

#ifndef NUMERIC_CONTAINER_BASE_H
#define NUMERIC_CONTAINER_BASE_H

#include <initializer_list>
#include <algorithm>
#include <functional>
#include <variant>
#include <execution>
#include <vector>

#include "numeric_base.h"
#include "storage_base.h"
#include "iterator_base.h"
#include "slice_base.h"
#include "view_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for numeric containers (Vector, Matrix, Tensor)
     *
     * Provides core container functionality for owned data
     */
    template<typename Derived, typename T, typename Storage = DynamicStorage<T>>
    class ContainerBase : public NumericBase<Derived> {
    public:
        using value_type = T;
        using storage_type = Storage;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

        // Iterator types
        using iterator = ContainerIterator<T>;
        using const_iterator = ContainerIterator<const T>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        static_assert(NumberLike<T>,
            "Container value_type must satisfy NumberLike concept");

        // === Constructors and Destructor ===

        ContainerBase() = default;

        explicit ContainerBase(const Shape& shape)
            : shape_(shape), storage_(shape.size()) {
            check_ieee_compliance();
        }

        ContainerBase(const Shape& shape, const T& value)
            : shape_(shape), storage_(shape.size(), value) {
            check_ieee_compliance();
        }

        ContainerBase(const Shape& shape, std::initializer_list<T> values)
            : shape_(shape), storage_(values) {
            if (storage_.size() != shape.size()) {
                throw DimensionError("Initializer list size doesn't match shape");
            }
            check_ieee_compliance();
        }

        ContainerBase(const ContainerBase&) = default;
        ContainerBase(ContainerBase&&) noexcept = default;
        ContainerBase& operator=(const ContainerBase&) = default;
        ContainerBase& operator=(ContainerBase&&) noexcept = default;
        ~ContainerBase() = default;

        // === Core Accessors ===

        const Shape& shape() const noexcept { return shape_; }
        size_type size() const noexcept { return storage_.size(); }
        bool empty() const noexcept { return storage_.empty(); }
        size_type ndim() const noexcept { return shape_.ndim(); }
        bool owns_data() const noexcept { return true; }

        // === Storage Access ===

        Storage& storage() noexcept { return storage_; }
        const Storage& storage() const noexcept { return storage_; }
        pointer data() noexcept { return storage_.data(); }
        const_pointer data() const noexcept { return storage_.data(); }

        // === Element Access ===

        reference operator[](size_type i) {
            check_bounds(i);
            return storage_[i];
        }

        const_reference operator[](size_type i) const {
            check_bounds(i);
            return storage_[i];
        }

        reference at(size_type i) {
            if (i >= size()) {
                throw std::out_of_range("Index out of range");
            }
            return storage_[i];
        }

        const_reference at(size_type i) const {
            if (i >= size()) {
                throw std::out_of_range("Index out of range");
            }
            return storage_[i];
        }

        reference front() {
            if (empty()) throw std::out_of_range("Container is empty");
            return storage_[0];
        }

        const_reference front() const {
            if (empty()) throw std::out_of_range("Container is empty");
            return storage_[0];
        }

        reference back() {
            if (empty()) throw std::out_of_range("Container is empty");
            return storage_[size() - 1];
        }

        const_reference back() const {
            if (empty()) throw std::out_of_range("Container is empty");
            return storage_[size() - 1];
        }

        // === Iterators ===

        iterator begin() noexcept { return iterator(data()); }
        const_iterator begin() const noexcept { return const_iterator(data()); }
        const_iterator cbegin() const noexcept { return begin(); }

        iterator end() noexcept { return iterator(data() + size()); }
        const_iterator end() const noexcept { return const_iterator(data() + size()); }
        const_iterator cend() const noexcept { return end(); }

        reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
        const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
        const_reverse_iterator crbegin() const noexcept { return rbegin(); }

        reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
        const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
        const_reverse_iterator crend() const noexcept { return rend(); }

        // === Modifiers ===

        void fill(const T& value) {
            std::fill(begin(), end(), value);
        }

        void swap(ContainerBase& other) noexcept {
            std::swap(shape_, other.shape_);
            storage_.swap(other.storage_);
        }

        void clear() {
            shape_ = Shape();
            storage_.clear();
        }

        // === Shape Operations ===

        void resize(const Shape& new_shape) {
            shape_ = new_shape;
            storage_.resize(shape_.size());
        }

        void resize(const Shape& new_shape, const T& value) {
            shape_ = new_shape;
            storage_.resize(shape_.size(), value);
        }

        void reshape(const Shape& new_shape) {
            if (new_shape.size() != size()) {
                throw DimensionError(
                    "Cannot reshape from size " + std::to_string(size()) +
                    " to size " + std::to_string(new_shape.size())
                );
            }
            shape_ = new_shape;
        }

        // === Memory Information ===

        Layout layout() const noexcept { return storage_.layout(); }
        Device device() const noexcept { return storage_.device(); }
        bool is_contiguous() const noexcept { return storage_.is_contiguous(); }
        size_t nbytes() const noexcept { return size() * sizeof(T); }
        const std::type_info& dtype() const noexcept { return typeid(T); }

        // === Copy Operations ===

        Derived copy() const {
            return Derived(static_cast<const Derived&>(*this));
        }

        Derived deep_copy() const {
            Derived result(shape_);
            std::copy(begin(), end(), result.begin());
            return result;
        }

        // === Validation Operations ===

        bool all_finite() const {
            return std::all_of(begin(), end(), [](const T& val) {
                return IEEEComplianceChecker::is_finite(val);
            });
        }

        bool has_nan() const {
            return std::any_of(begin(), end(), [](const T& val) {
                return IEEEComplianceChecker::is_nan(val);
            });
        }

        bool has_inf() const {
            return std::any_of(begin(), end(), [](const T& val) {
                return IEEEComplianceChecker::is_inf(val);
            });
        }

        // === Transformation Operations ===

        template<typename UnaryOp>
        void apply(UnaryOp op) {
            std::transform(begin(), end(), begin(), op);
        }

        template<typename UnaryOp>
        Derived apply_copy(UnaryOp op) const {
            Derived result(shape_);
            std::transform(begin(), end(), result.begin(), op);
            return result;
        }

        template<typename BinaryOp>
        void apply(const ContainerBase& other, BinaryOp op) {
            if (shape_ != other.shape_) {
                throw DimensionError("Shapes must match for element-wise operation");
            }
            std::transform(begin(), end(), other.begin(), begin(), op);
        }

        // === Parallel Transformation ===

        template<typename UnaryOp>
        void parallel_apply(UnaryOp op) {
#ifdef __cpp_lib_execution
            if (size() > 1000) {
                std::transform(std::execution::par_unseq, begin(), end(), begin(), op);
            } else {
                apply(op);
            }
#else
            apply(op);
#endif
        }

        // === Reduction Operations ===

        template<typename BinaryOp>
        T reduce(BinaryOp op, T init = T{}) const {
            return std::reduce(begin(), end(), init, op);
        }

        T sum() const {
            return reduce(std::plus<T>(), T{0});
        }

        T product() const {
            return reduce(std::multiplies<T>(), T{1});
        }

        T min() const {
            if (empty()) {
                throw std::runtime_error("Cannot compute min of empty container");
            }
            return *std::min_element(begin(), end());
        }

        T max() const {
            if (empty()) {
                throw std::runtime_error("Cannot compute max of empty container");
            }
            return *std::max_element(begin(), end());
        }

        // === Comparison Operations ===

        bool operator==(const ContainerBase& other) const {
            return shape_ == other.shape_ &&
                   std::equal(begin(), end(), other.begin());
        }

        bool operator!=(const ContainerBase& other) const {
            return !(*this == other);
        }

        // === Broadcasting Support ===

        bool is_broadcastable_with(const ContainerBase& other) const {
            return shape_.is_broadcastable_with(other.shape_);
        }

        Shape broadcast_shape(const ContainerBase& other) const {
            return shape_.broadcast_to(other.shape_);
        }

        // === Element-wise Scalar Operations ===

        Derived& operator+=(const T& scalar) {
            apply([scalar](const T& x) { return x + scalar; });
            return static_cast<Derived&>(*this);
        }

        Derived& operator-=(const T& scalar) {
            apply([scalar](const T& x) { return x - scalar; });
            return static_cast<Derived&>(*this);
        }

        Derived& operator*=(const T& scalar) {
            apply([scalar](const T& x) { return x * scalar; });
            return static_cast<Derived&>(*this);
        }

        Derived& operator/=(const T& scalar) {
            if (scalar == T{0}) {
                throw ComputationError("Division by zero");
            }
            apply([scalar](const T& x) { return x / scalar; });
            return static_cast<Derived&>(*this);
        }

        // === Element-wise Container Operations ===

        Derived& operator+=(const ContainerBase& other) {
            apply(other, std::plus<T>());
            return static_cast<Derived&>(*this);
        }

        Derived& operator-=(const ContainerBase& other) {
            apply(other, std::minus<T>());
            return static_cast<Derived&>(*this);
        }

        Derived& operator*=(const ContainerBase& other) {
            apply(other, std::multiplies<T>());
            return static_cast<Derived&>(*this);
        }

        Derived& operator/=(const ContainerBase& other) {
            apply(other, [](const T& a, const T& b) {
                if (b == T{0}) {
                    throw ComputationError("Division by zero");
                }
                return a / b;
            });
            return static_cast<Derived&>(*this);
        }

    protected:
        Shape shape_;
        Storage storage_;

        void check_ieee_compliance() const {
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::numeric_limits<T>::is_iec559) {
                    // Log warning if logging is enabled
                }
            }
        }

        void check_bounds(size_type i) const {
#ifndef NDEBUG
            if (NumericOptions::defaults().check_bounds) {
                if (i >= size()) {
                    throw std::out_of_range(
                        "Index " + std::to_string(i) +
                        " out of range [0, " + std::to_string(size()) + ")"
                    );
                }
            }
#endif
        }

        void check_finite() const {
            if (NumericOptions::defaults().check_finite) {
                if (has_nan()) {
                    throw ComputationError("Container contains NaN values");
                }
                if (has_inf()) {
                    throw ComputationError("Container contains infinite values");
                }
            }
        }
    };

    /**
     * @brief Container that references data without owning it
     */
    template<typename Derived, typename T>
    class ViewContainer : public NumericBase<Derived> {
    public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

        ViewContainer() : data_(nullptr), shape_() {}

        ViewContainer(pointer data, const Shape& shape)
            : data_(data), shape_(shape) {}

        const Shape& shape() const noexcept { return shape_; }
        size_type size() const noexcept { return shape_.size(); }
        bool empty() const noexcept { return size() == 0; }
        bool owns_data() const noexcept { return false; }

        pointer data() noexcept { return data_; }
        const_pointer data() const noexcept { return data_; }

        reference operator[](size_type i) { return data_[i]; }
        const_reference operator[](size_type i) const { return data_[i]; }

        bool is_valid() const noexcept { return data_ != nullptr; }

        bool overlaps(const ViewContainer& other) const noexcept {
            if (!data_ || !other.data_) return false;
            const T* this_end = data_ + size();
            const T* other_end = other.data_ + other.size();
            return !(this_end <= other.data_ || other_end <= data_);
        }

    protected:
        pointer data_;
        Shape shape_;
    };

    /**
     * @brief Type-erased multi-dimensional view wrapper
     */
    template<typename T>
    class AnyMultiDimView {
    public:
        AnyMultiDimView() : rank_(0), data_(nullptr) {}

        template<size_t Rank>
        AnyMultiDimView(const MultiDimView<T, Rank>& view)
            : rank_(Rank), data_(view.data()) {
            shape_.resize(Rank);
            strides_.resize(Rank);
            for (size_t i = 0; i < Rank; ++i) {
                shape_[i] = view.shape()[i];
                strides_[i] = view.strides()[i];
            }
        }

        size_t rank() const { return rank_; }
        T* data() { return data_; }
        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<std::ptrdiff_t>& strides() const { return strides_; }

        T& operator[](size_t i) { return data_[i]; }
        const T& operator[](size_t i) const { return data_[i]; }

    private:
        size_t rank_;
        T* data_;
        std::vector<size_t> shape_;
        std::vector<std::ptrdiff_t> strides_;
    };

    /**
     * @brief Mixin for containers that support slicing
     */
    template<typename Derived, typename T>
    class SliceableContainer {
    public:
        using slice_result = std::variant<std::reference_wrapper<T>, ViewBase<T>, StridedView<T>, AnyMultiDimView<T>>;

        slice_result operator[](const MultiIndex& idx) {
            auto& self = static_cast<Derived&>(*this);
            auto result_shape = idx.result_shape(self.shape());

            if (result_shape.rank() == 0) {
                size_t linear_idx = self.compute_index(idx);
                return self.data()[linear_idx];
            }
            else if (result_shape.rank() == 1) {
                auto [offset, stride, count] = self.compute_1d_slice(idx);
                if (stride == 1) {
                    return ViewBase<T>(self.data() + offset, count);
                } else {
                    return StridedView<T>(self.data() + offset, count, stride);
                }
            }
            else {
                return self.create_multidim_view_impl(idx, result_shape);
            }
        }

        slice_result operator[](const char* idx_str) {
            return (*this)[IndexParser::parse(idx_str)];
        }
    };

    /**
     * @brief Mixin for containers that support broadcasting
     */
    template<typename Derived, typename T>
    class BroadcastableContainer {
    public:
        template<typename Other>
        bool is_broadcastable_with(const Other& other) const {
            auto& self = static_cast<const Derived&>(*this);
            return self.shape().is_broadcastable_with(other.shape());
        }

        template<typename Other>
        Shape broadcast_shape(const Other& other) const {
            auto& self = static_cast<const Derived&>(*this);
            return self.shape().broadcast_to(other.shape());
        }
    };

    /**
     * @brief Mixin for containers that support reduction along axes
     */
    template<typename Derived, typename T>
    class AxisReducible {
    public:
        template<typename BinaryOp>
        auto reduce_along_axis(int axis, BinaryOp op, T init = T{}) {
            return static_cast<Derived*>(this)->reduce_axis_impl(axis, op, init);
        }

        auto sum_axis(int axis) {
            return reduce_along_axis(axis, std::plus<T>(), T{0});
        }

        auto max_axis(int axis) {
            return reduce_along_axis(axis,
                [](const T& a, const T& b) { return std::max(a, b); });
        }

        auto min_axis(int axis) {
            return reduce_along_axis(axis,
                [](const T& a, const T& b) { return std::min(a, b); });
        }
    };

    // === Concepts ===

    template<typename C>
    concept Container = requires(C c) {
        typename C::value_type;
        typename C::size_type;
        { c.size() } -> std::convertible_to<size_t>;
        { c.empty() } -> std::convertible_to<bool>;
        { c.begin() };
        { c.end() };
    };

    template<typename C>
    concept NumericContainer = Container<C> && requires(C c) {
        { c.shape() } -> std::convertible_to<Shape>;
        { c.data() } -> std::convertible_to<typename C::pointer>;
        typename C::value_type;
        requires NumberLike<typename C::value_type>;
    };

} // namespace fem::numeric

#endif //NUMERIC_CONTAINER_BASE_H
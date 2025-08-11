#pragma once

#ifndef NUMERIC_CONTAINER_BASE_H
#define NUMERIC_CONTAINER_BASE_H

#include <initializer_list>
#include <algorithm>
#include <functional>

#include "numeric_base.h"
#include "storage_base.h"
#include "iterator_base.h"

namespace fem::numeric {

    /**
     * @brief Base class for numeric containers (Vector, Matrix, Tensor)
     *
     * Provides common interface for all containers that hold number-like types
     * Ensures IEEE compliance and proper memory management
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

        /**
         * @brief Default constructor
         */
        ContainerBase() = default;

        /**
         * @brief Constructor with shape
         */
        explicit ContainerBase(const Shape& shape)
                : shape_(shape), storage_(shape.size()) {
            check_ieee_compliance();
        }

        /**
         * @brief Constructor with shape and initial value
         */
        ContainerBase(const Shape& shape, const T& value)
                : shape_(shape), storage_(shape.size(), value) {
            check_ieee_compliance();
        }

        /**
         * @brief Constructor from initializer list
         */
        ContainerBase(const Shape& shape, std::initializer_list<T> values)
                : shape_(shape), storage_(values) {
            if (storage_.size() != shape.size()) {
                throw DimensionError("Initializer list size doesn't match shape");
            }
            check_ieee_compliance();
        }

        /**
         * @brief Copy constructor
         */
        ContainerBase(const ContainerBase& other) = default;

        /**
         * @brief Move constructor
         */
        ContainerBase(ContainerBase&& other) noexcept = default;

        /**
         * @brief Copy assignment
         */
        ContainerBase& operator=(const ContainerBase& other) = default;

        /**
         * @brief Move assignment
         */
        ContainerBase& operator=(ContainerBase&& other) noexcept = default;

        /**
         * @brief Destructor
         */
        ~ContainerBase() = default;

        // Shape and size
        const Shape& shape() const noexcept { return shape_; }
        size_type size() const noexcept { return storage_.size(); }
        bool empty() const noexcept { return storage_.empty(); }

        // Storage access
        Storage& storage() noexcept { return storage_; }
        const Storage& storage() const noexcept { return storage_; }

        // Data access
        pointer data() noexcept { return storage_.data(); }
        const_pointer data() const noexcept { return storage_.data(); }

        // Element access
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

        reference front() { return storage_.front(); }
        const_reference front() const { return storage_.front(); }
        reference back() { return storage_.back(); }
        const_reference back() const { return storage_.back(); }

        // Iterators
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

        // Modifiers
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

        // Resize operations
        void resize(const Shape& new_shape) {
            shape_ = new_shape;
            storage_.resize(shape_.size());
        }

        void resize(const Shape& new_shape, const T& value) {
            shape_ = new_shape;
            storage_.resize(shape_.size(), value);
        }

        // Memory operations
        Layout layout() const noexcept { return storage_.layout(); }
        Device device() const noexcept { return storage_.device(); }
        bool is_contiguous() const noexcept { return storage_.is_contiguous(); }
        size_t nbytes() const noexcept { return size() * sizeof(T); }
        const std::type_info& dtype() const noexcept { return typeid(T); }

        // Copy operations
        Derived copy() const {
            return Derived(static_cast<const Derived&>(*this));
        }

        Derived deep_copy() const {
            Derived result(shape_);
            std::copy(begin(), end(), result.begin());
            return result;
        }

        // Validation operations
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

        // Apply operations
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

        // Reduction operations
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

        // Statistics
        T mean() const {
            if (empty()) return T{};
            return sum() / static_cast<T>(size());
        }

        // Comparison operations
        bool operator==(const ContainerBase& other) const {
            return shape_ == other.shape_ &&
                   std::equal(begin(), end(), other.begin());
        }

        bool operator!=(const ContainerBase& other) const {
            return !(*this == other);
        }

        // Element-wise operations
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

        /**
         * @brief Check IEEE compliance if enabled
         */
        void check_ieee_compliance() const {
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::numeric_limits<T>::is_iec559) {
                    // Warning: Type is not IEEE 754 compliant
                    // Could log or handle this based on options
                }
            }
        }

        /**
         * @brief Check bounds if enabled
         */
        void check_bounds(size_type i) const {
#ifndef NDEBUG
            if (NumericOptions::defaults().check_bounds) {
                if (i >= size()) {
                    throw std::out_of_range("Index " + std::to_string(i) +
                                            " out of range [0, " + std::to_string(size()) + ")");
                }
            }
#endif
        }

        /**
         * @brief Check for finite values if enabled
         */
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
     * @brief Concept for container types
     */
    template<typename C>
    concept Container = requires(C c) {
        typename C::value_type;
        typename C::size_type;
    { c.size() } -> std::convertible_to<size_t>;
    { c.empty() } -> std::convertible_to<bool>;
    { c.begin() };
    { c.end() };
    };

    /**
     * @brief Concept for numeric containers
     */
    template<typename C>
    concept NumericContainer = Container<C> && requires(C c) {
        { c.shape() } -> std::convertible_to<Shape>;
        { c.data() } -> std::convertible_to<typename C::pointer>;
        typename C::value_type;
        requires NumberLike<typename C::value_type>;
    };

} // namespace fem::numeric

#endif //NUMERIC_CONTAINER_BASE_H

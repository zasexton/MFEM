#pragma once

#ifndef NUMERIC_BASE_H
#define NUMERIC_BASE_H

#if __cplusplus < 202002L
#  error "MFEM numeric requires C++20 or later"
#endif

#include <cstddef>
#include <type_traits>
#include <concepts>
#include <limits>
#include <complex>
#include <memory>
#include <string>
#include <typeinfo>
#include <functional>
#include <string>
#include <stdexcept>

namespace fem::numeric {

    // Forward declarations
    template<typename T> class Vector;
    template<typename T> class Matrix;
    template<typename T, size_t Rank> class Tensor;

    /**
     * @brief Concept for number-like types that can be used in numeric containers
     *
     * Ensures IEEE compliance and standard arithmetic operations
     */
    template<typename T>
    concept NumberLike = requires(T a, T b) {
        // Arithmetic operations
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;

        // Comparison operations
        { a == b } -> std::convertible_to<bool>;
        { a != b } -> std::convertible_to<bool>;
        { a < b } -> std::convertible_to<bool>;
        { a > b } -> std::convertible_to<bool>;

        // Assignment
        { a = b } -> std::same_as<T&>;

        // Zero and one
        { T{} } -> std::convertible_to<T>;
        { T{0} } -> std::convertible_to<T>;
        { T{1} } -> std::convertible_to<T>;
    };

    // Primary template - defaults to false
    template<typename T>
    inline constexpr bool is_complex_v = false;

    // Specialization for std::complex types
    template<typename T>
    inline constexpr bool is_complex_v<std::complex<T>> = true;

    // Alternative trait-based approach (more compatible)
    template<typename T>
    struct is_complex : std::false_type {};

    template<typename T>
    struct is_complex<std::complex<T>> : std::true_type {};

    /**
     * @brief Extended concept for IEEE floating-point compliant types
     */
    template<typename T>
    concept IEEECompliant = NumberLike<T> && std::numeric_limits<T>::is_iec559;

    /**
    * @brief Shape class for multi-dimensional containers
    */
    class Shape {
    public:

        Shape() = default;
        Shape(std::initializer_list<size_t> dims) : dims_(dims) {};
        explicit Shape(const std::vector<size_t>& dims) : dims_(dims) {} ;

        // Accessors
        inline size_t rank() const noexcept { return dims_.size(); }
        inline size_t operator[](size_t i) const { return dims_[i]; }
        inline size_t& operator[](size_t i) { return dims_[i]; }

        size_t size() const noexcept {
            if (dims_.empty()) {
                return 1;
            }
            size_t result = 1;
            for (auto d : dims_) {
                result *= d;
            }
            return result;
        }

        bool empty() const noexcept { return dims_.empty(); }

        // Get number of dimensions
        size_t ndim() const noexcept {
            return dims_.size();
        }

        // Shape operations
        Shape squeeze() const {
            std::vector<size_t> new_dims;
            for (auto d : dims_) {
                if (d != 1) {
                    new_dims.push_back(d);
                }
            }
            return Shape(new_dims);
        }

        size_t normalize_axis(int axis) const {
            if (axis < 0) {
                axis += static_cast<int>(dims_.size());
            }
            if (axis < 0 || axis >= static_cast<int>(dims_.size())) {
                throw std::out_of_range("Axis out of range");
            }
            return static_cast<size_t>(axis);
        }

        // Add dimension of size 1 at specified axis
        Shape unsqueeze(int axis) const {
            size_t norm_axis;
            if (axis < 0) {
                // Handle negative axis
                int adjusted = axis + static_cast<int>(dims_.size()) + 1;
                if (adjusted < 0) {
                    throw std::out_of_range("Axis out of range for unsqueeze");
                }
                norm_axis = static_cast<size_t>(adjusted);
            } else {
                norm_axis = static_cast<size_t>(axis);
            }

            if (norm_axis > dims_.size()) {
                throw std::out_of_range("Axis out of range for unsqueeze");
            }

            std::vector<size_t> new_dims = dims_;
            new_dims.insert(new_dims.begin() + static_cast<std::ptrdiff_t>(norm_axis), 1);
            return Shape(new_dims);
        }

        // Flatten to 1D
        Shape flatten() const {
            return Shape{size()};
        }

        // Reverse dimensions (like NumPy's transpose with no args)
        Shape reverse() const {
            std::vector<size_t> reversed(dims_.rbegin(), dims_.rend());
            return Shape(reversed);
        }

        // Permute dimensions (for general transpose)
        Shape permute(const std::vector<size_t>& axes) const {
            if (axes.size() != dims_.size()) {
                throw std::invalid_argument("Permutation must have same length as dimensions");
            }
            std::vector<size_t> new_dims(dims_.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                if (axes[i] >= dims_.size()) {
                    throw std::out_of_range("Permutation index out of range");
                }
                new_dims[i] = dims_[axes[i]];
            }
            return Shape(new_dims);
        }

        Shape broadcast_to(const Shape& target) const {
            if (!is_broadcastable_with(target)) {
                throw std::invalid_argument("Cannot broadcast to target shape");
            }

            std::vector<size_t> new_dims(target.dims_.size());

            // Fill from the right, aligning dimensions
            size_t offset = target.dims_.size() - dims_.size();

            for (size_t i = 0; i < target.dims_.size(); ++i) {
                if (i < offset) {
                    // This dimension doesn't exist in source, take from target
                    new_dims[i] = target.dims_[i];
                } else {
                    // Both shapes have this dimension (when right-aligned)
                    size_t src_idx = i - offset;
                    if (dims_[src_idx] == 1) {
                        new_dims[i] = target.dims_[i];
                    } else {
                        new_dims[i] = dims_[src_idx];
                    }
                }
            }
            return Shape(new_dims);
        }

        // Reshape with -1 for inferred dimension (NumPy-like)
        Shape reshape_infer(const std::vector<int>& new_shape) const {
            size_t total = size();
            int infer_idx = -1;
            size_t known_size = 1;

            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (new_shape[i] == -1) {
                    if (infer_idx != -1) {
                        throw std::invalid_argument("Only one dimension can be -1");
                    }
                    infer_idx = static_cast<int>(i);  // Explicit cast
                } else if (new_shape[i] < 0) {
                    throw std::invalid_argument("Negative dimensions not allowed except -1");
                } else {
                    known_size *= static_cast<size_t>(new_shape[i]);  // Explicit cast
                }
            }

            std::vector<size_t> result_dims;
            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (static_cast<int>(i) == infer_idx) {
                    if (total % known_size != 0) {
                        throw std::invalid_argument("Cannot infer dimension: size mismatch");
                    }
                    result_dims.push_back(total / known_size);
                } else {
                    result_dims.push_back(static_cast<size_t>(new_shape[i]));  // Explicit cast
                }
            }

            return Shape(result_dims);
        }

        // Comparisons
        bool operator==(const Shape& other) const = default;
        bool is_compatible_with(const Shape& other) const {
            return dims_ == other.dims_;
        }
        bool is_broadcastable_with(const Shape& other) const {
            // Compare dimensions from right to left
            size_t min_dims = std::min(dims_.size(), other.dims_.size());

            for (size_t i = 0; i < min_dims; ++i) {
                size_t this_idx = dims_.size() - 1 - i;
                size_t other_idx = other.dims_.size() - 1 - i;

                if (dims_[this_idx] != other.dims_[other_idx] &&
                    dims_[this_idx] != 1 &&
                    other.dims_[other_idx] != 1) {
                    return false;
                    }
            }
            return true;
        }

        // Iterators
        auto begin() const { return dims_.begin(); }
        auto end() const { return dims_.end(); }
        auto begin() { return dims_.begin(); }
        auto end() { return dims_.end(); }

        // String representation
        std::string to_string() const {
            if (dims_.empty()) {
                return "()";
            }
            std::string result = "(";
            for (size_t i = 0; i < dims_.size(); ++i) {
                result += std::to_string(dims_[i]);
                if (i < dims_.size() - 1) {
                    result += ", ";
                }
            }
            result += ")";
            return result;
        }

        // Get underlying dimensions vector (const)
        const std::vector<size_t>& dims() const noexcept {
            return dims_;
        }
    protected:
        std::vector<size_t>& mutable_dims() { return dims_; }
        const std::vector<size_t>& dims_internal() const { return dims_; }
    private:
        std::vector<size_t> dims_;
    };

    class StridedShape : public Shape {
        std::vector<size_t> strides_;

    public:
        using Shape::Shape;  // Inherit constructors

        // Compute default strides for row-major (C-style) layout
        void compute_strides_row_major() {
            const auto& shape_dims = dims();  // Renamed to avoid collision
            strides_.resize(shape_dims.size());
            size_t stride = 1;
            for (int i = static_cast<int>(shape_dims.size()) - 1; i >= 0; --i) {
                strides_[static_cast<size_t>(i)] = stride;
                stride *= shape_dims[static_cast<size_t>(i)];
            }
        }

        // Compute default strides for column-major (Fortran-style) layout
        void compute_strides_col_major() {
            const auto& shape_dims = dims();  // Renamed to avoid collision
            strides_.resize(shape_dims.size());
            size_t stride = 1;
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                strides_[i] = stride;
                stride *= shape_dims[i];
            }
        }

        const std::vector<size_t>& strides() const { return strides_; }
    };

    /**
     * @brief Memory layout for multi-dimensional arrays
     */
    enum class Layout {
        RowMajor,     // C-style (last index varies fastest)
        ColumnMajor,  // Fortran-style (first index varies fastest)
        Strided       // Custom strides
    };

    /**
     * @brief Device location for computation
     */
    enum class Device {
        CPU,
        GPU,
        AUTO  // Automatically select best device
    };

    /**
     * @brief Base class for all numeric objects
     *
     * Provides common interface for vectors, matrices, and tensors
     */
    template<typename Derived>
    class NumericBase {
    public:
        using derived_type = Derived;

        /**
         * @brief CRTP helper to get derived class
         */
        Derived& derived() noexcept {
            return static_cast<Derived&>(*this);
        }

        const Derived& derived() const noexcept {
            return static_cast<const Derived&>(*this);
        }

        /**
         * @brief Get the shape of the numeric object
         */
        const Shape& shape() const noexcept {
            return derived().shape();
        }

        /**
         * @brief Get total number of elements
         */
        size_t size() const noexcept {
            return shape().size();
        }

        /**
         * @brief Check if empty
         */
        bool empty() const noexcept {
            return size() == 0;
        }

        /**
         * @brief Get memory layout
         */
        Layout layout() const noexcept {
            return derived().layout();
        }

        /**
         * @brief Get device location
         */
        Device device() const noexcept {
            return derived().device();
        }

        /**
         * @brief Check if data is contiguous in memory
         */
        bool is_contiguous() const noexcept {
            return derived().is_contiguous();
        }

        /**
         * @brief Get raw data pointer (if available)
         */
        template<typename T>
        T* data() noexcept {
            return derived().template data<T>();
        }

        template<typename T>
        const T* data() const noexcept {
            return derived().template data<T>();
        }

        /**
         * @brief Type information
         */
        const std::type_info& dtype() const noexcept {
            return derived().dtype();
        }

        /**
         * @brief Memory usage in bytes
         */
        size_t nbytes() const noexcept {
            return derived().nbytes();
        }

        /*
         *  @brief Get number of dimensions
         */
        size_t ndim() const noexcept {
            return shape().rank();
        }
        /**
         * @brief Create a copy
         */
        Derived copy() const {
            return derived().copy();
        }

        /**
         * @brief Create a view (non-owning reference)
         */
        auto view() {
            return derived().view();
        }

        auto view() const {
            return derived().view();
        }

        /**
         * @brief Reshape to new dimensions
         */
        auto reshape(const Shape& new_shape) {
            return derived().reshape(new_shape);
        }

        /**
         * @brief Transpose (reverse dimensions)
         */
        auto transpose() const {
            return derived().transpose();
        }

        /**
         * @brief String representation
         */
        std::string to_string() const {
            return derived().to_string();
        }

    protected:
        NumericBase() = default;
        ~NumericBase() = default;
        NumericBase(const NumericBase&) = default;
        NumericBase(NumericBase&&) = default;
        NumericBase& operator=(const NumericBase&) = default;
        NumericBase& operator=(NumericBase&&) = default;
    };

    /**
     * @brief IEEE 754 compliance checker
     */
    class IEEEComplianceChecker {
    public:
        template<typename T>
        static constexpr bool is_compliant() {
            if constexpr (std::is_floating_point_v<T>) {
                return std::numeric_limits<T>::is_iec559;
            } else if constexpr (is_complex_v<T>) {
                return std::numeric_limits<typename T::value_type>::is_iec559;
            }
            return false;
        }

        template<typename T>
        static bool is_finite(T value) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isfinite(value);
            } else if constexpr (is_complex_v<T>) {
                return std::isfinite(value.real()) && std::isfinite(value.imag());
            }
            return true;
        }

        template<typename T>
        static bool is_nan(T value) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isnan(value);
            } else if constexpr (is_complex_v<T>) {
                return std::isnan(value.real()) || std::isnan(value.imag());
            }
            return false;
        }

        template<typename T>
        static bool is_inf(T value) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isinf(value);
            } else if constexpr (is_complex_v<T>) {
                return std::isinf(value.real()) || std::isinf(value.imag());
            }
            return false;
        }

        template<typename T>
        static T quiet_nan() {
            if constexpr (std::is_floating_point_v<T>) {
                return std::numeric_limits<T>::quiet_NaN();
            } else if constexpr (is_complex_v<T>) {
                using value_type = typename T::value_type;
                return T(std::numeric_limits<value_type>::quiet_NaN(),
                         std::numeric_limits<value_type>::quiet_NaN());
            }
            return T{};
        }

        template<typename T>
        static T infinity() {
            if constexpr (std::is_floating_point_v<T>) {
                return std::numeric_limits<T>::infinity();
            } else if constexpr (is_complex_v<T>) {
                using value_type = typename T::value_type;
                return T(std::numeric_limits<value_type>::infinity(), 0);
            }
            return T{};
        }
    };

    /**
     * @brief Numeric object metadata
     */
    template<typename T = double>
    class NumericMetadata {
    public:
        NumericMetadata()
            : is_floating_point_(std::is_floating_point_v<T>)
            , is_integer_(std::is_integral_v<T>)
            , is_complex_(is_complex_v<T>)
            , is_ieee_compliant_(std::numeric_limits<T>::is_iec559) {};

        NumericMetadata(const std::type_info& dtype,
                        const Shape& shape,
                        Layout layout = Layout::RowMajor,
                        Device device = Device::CPU)
                : dtype_(&dtype), shape_(shape), layout_(layout), device_(device) {}

        const std::type_info& dtype() const { return *dtype_; }
        const Shape& shape() const { return shape_; }
        Layout layout() const { return layout_; }
        Device device() const { return device_; }

        size_t element_size() const { return element_size_; }
        size_t total_size() const { return shape_.size(); }
        size_t nbytes() const { return total_size() * element_size_; }

        bool is_floating_point() const { return is_floating_point_; }
        bool is_integer() const { return is_integer_; }
        bool is_complex() const { return is_complex_; }
        bool is_ieee_compliant() const { return is_ieee_compliant_; }

        // Metadata operations
        bool is_compatible_with(const NumericMetadata& other) const;
        NumericMetadata broadcast_with(const NumericMetadata& other) const;

    private:
        const std::type_info* dtype_ = &typeid(void);
        Shape shape_;
        Layout layout_ = Layout::RowMajor;
        Device device_ = Device::CPU;
        size_t element_size_ = 0;
        bool is_floating_point_;
        bool is_integer_;
        bool is_complex_;
        bool is_ieee_compliant_;
    };

    /**
     * @brief Options for numeric operations
     */
    struct NumericOptions {
        // Error handling
        bool check_finite = false;  // Check for NaN/Inf
        bool check_alignment = true;  // Check memory alignment
        bool check_bounds = true;  // Bounds checking (debug mode)

        // Performance
        bool allow_parallel = true;  // Enable parallel execution
        bool allow_simd = true;  // Enable SIMD optimizations
        bool force_contiguous = false;  // Force contiguous memory

        // Precision
        double tolerance = 1e-10;  // Numerical tolerance
        bool use_high_precision = false;  // Use extended precision

        // Memory
        size_t alignment = 0;  // Memory alignment (bytes)
        bool use_memory_pool = false;  // Use memory pooling

        static NumericOptions& defaults() {
            static NumericOptions opts;
            return opts;
        }
    };

    /**
     * @brief Base error class for numeric operations
     */
    class NumericError : public std::runtime_error {
    public:
        explicit NumericError(const std::string& msg)
                : std::runtime_error(msg) {}  // Just pass the message directly
    };

    class DimensionError : public NumericError {
    public:
        explicit DimensionError(const std::string& msg)
                : NumericError(msg) {}  // Don't add "Dimension mismatch: " prefix
    };

    class ComputationError : public NumericError {
    public:
        explicit ComputationError(const std::string& msg)
                : NumericError(msg) {}  // Don't add "Computation failed: " prefix
    };

    class ConvergenceError : public NumericError {
    public:
        explicit ConvergenceError(const std::string& msg)
                : NumericError(msg) {}
    };

} // namespace fem::numeric

#endif //NUMERIC_BASE_H

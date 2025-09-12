#pragma once

#ifndef NUMERIC_VECTOR_H
#define NUMERIC_VECTOR_H

#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>
#include <cmath>
#include <execution>
#include <limits>
#include <cassert>
#include <stdexcept>
// #include <format> // C++20 format not available everywhere

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
// Note: VectorView will be implemented in a separate file later

namespace fem::numeric {

// Vector class is forward declared in numeric_base.h

/**
 * @brief Dense vector class for numerical computing
 *
 * Implements a high-performance dense vector with:
 * - Support for real, complex, and dual number types
 * - Expression templates for lazy evaluation
 * - Optimized storage strategies
 * - Intuitive element access and slicing
 * - IEEE 754 compliance for all operations
 */
template<typename T, typename Storage = DynamicStorage<T>>
class Vector : public ContainerBase<Vector<T, Storage>, T, Storage>,
               public SliceableContainer<Vector<T, Storage>, T>,
               public BroadcastableContainer<Vector<T, Storage>, T>,
               public ExpressionBase<Vector<T, Storage>> {
public:
    // === Static Assertions ===
    static_assert(StorableType<T>, "T must satisfy StorableType concept");

    // === Type Aliases ===
    using base_type = ContainerBase<Vector<T, Storage>, T, Storage>;
    using expression_base = ExpressionBase<Vector<T, Storage>>;
    using value_type = T;
    using storage_type = Storage;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    // Iterator types
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using reverse_iterator = typename base_type::reverse_iterator;
    using const_reverse_iterator = typename base_type::const_reverse_iterator;

    // Expression and view types (simplified for now)
    using view_type = ViewBase<T>;
    using const_view_type = ViewBase<const T>;
    using strided_view_type = StridedView<T>;
    using const_strided_view_type = StridedView<const T>;

    // Scalar type for operations
    using scalar_type = typename numeric_traits<T>::scalar_type;

    // === Constructors ===

    /**
     * @brief Default constructor - creates empty vector
     */
    Vector() : base_type(), storage_() {}

    /**
     * @brief Size constructor - creates vector of given size
     */
    explicit Vector(size_type size)
        : base_type(Shape{size}), storage_(size) {}

    /**
     * @brief Size and value constructor
     */
    Vector(size_type size, const T& value)
        : base_type(Shape{size}, value), storage_(size, value) {}

    /**
     * @brief Initializer list constructor
     */
    Vector(std::initializer_list<T> init)
        : base_type(Shape{init.size()}, init), storage_(init) {}

    /**
     * @brief Iterator range constructor
     */
    template<typename InputIt>
    Vector(InputIt first, InputIt last)
        : base_type(Shape{static_cast<size_type>(std::distance(first, last))}), 
          storage_(first, last) {}

    /**
     * @brief Copy constructor
     */
    Vector(const Vector& other)
        : base_type(other), storage_(other.storage_) {}

    /**
     * @brief Move constructor
     */
    Vector(Vector&& other) noexcept
        : base_type(std::move(other)), storage_(std::move(other.storage_)) {}

    /**
     * @brief Expression constructor for lazy evaluation
     */
    template<typename Expr>
    Vector(const ExpressionBase<Expr>& expr)
        : Vector(expr.derived().size()) {
        assign_expression(expr.derived());
    }

    /**
     * @brief View constructor
     */
    explicit Vector(const view_type& view)
        : Vector(view.size()) {
        std::copy(view.begin(), view.end(), begin());
    }

    // === Assignment Operators ===

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            resize(other.size());
            std::copy(other.begin(), other.end(), begin());
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            base_type::operator=(std::move(other));
        }
        return *this;
    }

    Vector& operator=(std::initializer_list<T> init) {
        resize(init.size());
        std::copy(init.begin(), init.end(), begin());
        return *this;
    }

    template<typename Expr>
    Vector& operator=(const ExpressionBase<Expr>& expr) {
        assign_expression(expr.derived());
        return *this;
    }

    Vector& operator=(const T& value) {
        std::fill(begin(), end(), value);
        return *this;
    }

    // === Element Access ===

    reference operator[](size_type index) {
        assert_valid_index(index);
        return storage_[index];
    }

    const_reference operator[](size_type index) const {
        assert_valid_index(index);
        return storage_[index];
    }

    reference at(size_type index) {
        check_bounds(index);
        return storage_[index];
    }

    const_reference at(size_type index) const {
        check_bounds(index);
        return storage_[index];
    }

    reference front() {
        assert(!empty());
        return storage_[0];
    }

    const_reference front() const {
        assert(!empty());
        return storage_[0];
    }

    reference back() {
        assert(!empty());
        return storage_[size() - 1];
    }

    const_reference back() const {
        assert(!empty());
        return storage_[size() - 1];
    }

    // === Data Access ===

    T* data() noexcept { return storage_.data(); }
    const T* data() const noexcept { return storage_.data(); }

    // === Iterators ===

    iterator begin() { return iterator(data()); }
    const_iterator begin() const { return const_iterator(data()); }
    const_iterator cbegin() const { return const_iterator(data()); }

    iterator end() { return iterator(data() + size()); }
    const_iterator end() const { return const_iterator(data() + size()); }
    const_iterator cend() const { return const_iterator(data() + size()); }

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }

    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }

    // === Size and Capacity ===

    size_type size() const noexcept { return storage_.size(); }
    size_type capacity() const noexcept { return storage_.capacity(); }
    bool empty() const noexcept { return storage_.empty(); }

    void resize(size_type new_size) {
        storage_.resize(new_size);
        base_type::shape_ = Shape{new_size};
    }

    void resize(size_type new_size, const T& value) {
        storage_.resize(new_size, value);
        base_type::shape_ = Shape{new_size};
    }

    void reserve(size_type new_capacity) {
        storage_.reserve(new_capacity);
    }

    void shrink_to_fit() {
        storage_.shrink_to_fit();
    }

    void clear() {
        storage_.clear();
        base_type::shape_ = Shape{0};
    }

    // === Shape Information ===

    size_t rank() const noexcept { return 1; }

    // === Slicing and Views ===

    view_type view() { 
        return view_type(data(), size());
    }

    const_view_type view() const {
        return const_view_type(data(), size());
    }

    view_type view(size_type start, size_type end) {
        check_slice_bounds(start, end);
        return view_type(data() + start, end - start);
    }

    const_view_type view(size_type start, size_type end) const {
        check_slice_bounds(start, end);
        return const_view_type(data() + start, end - start);
    }

    // Create a strided view from a Slice (NumPy-like)
    strided_view_type slice(const Slice& s) {
        const auto n = size();
        const auto norm = s.normalize(n);
        const auto count = s.count(n);

        if (count == 0) {
            return strided_view_type(data(), 0, 1);
        }

        const std::ptrdiff_t step = norm.step();
        const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(norm.start());
        return strided_view_type(data() + start, count, step);
    }

    const_strided_view_type slice(const Slice& s) const {
        const auto n = size();
        const auto norm = s.normalize(n);
        const auto count = s.count(n);

        if (count == 0) {
            return const_strided_view_type(data(), 0, 1);
        }

        const std::ptrdiff_t step = norm.step();
        const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(norm.start());
        return const_strided_view_type(data() + start, count, step);
    }

    // Shorthand call operator for slicing
    const_strided_view_type operator()(const Slice& s) const { return slice(s); }
    strided_view_type operator()(const Slice& s) { return slice(s); }

    // as_strided: expose a 1D strided view with the given logical shape/stride
    // Only 1D shape is supported for Vector; throws otherwise.
    strided_view_type as_strided(const Shape& shape, const Shape& strides) {
        if (shape.ndim() != 1 || strides.ndim() != 1) {
            throw std::invalid_argument("Vector::as_strided expects 1D shape and strides");
        }
        const auto len = shape[0];
        const auto stride = static_cast<std::ptrdiff_t>(strides[0]);
        if (len == 0) { return strided_view_type(data(), 0, 1); }
        return strided_view_type(data(), len, stride);
    }

    const_strided_view_type as_strided(const Shape& shape, const Shape& strides) const {
        if (shape.ndim() != 1 || strides.ndim() != 1) {
            throw std::invalid_argument("Vector::as_strided expects 1D shape and strides");
        }
        const auto len = shape[0];
        const auto stride = static_cast<std::ptrdiff_t>(strides[0]);
        if (len == 0) { return const_strided_view_type(data(), 0, 1); }
        return const_strided_view_type(data(), len, stride);
    }

    // === Mathematical Operations ===

    // Dot product
    template<typename U, typename S2>
    auto dot(const Vector<U, S2>& other) const {
        check_compatible_size(other);
        using result_type = decltype(T{} * U{});
        result_type result{};

        if constexpr (std::is_same_v<T, U>) {
            // Optimized path for same types
            result = std::transform_reduce(
                std::execution::par_unseq,
                begin(), end(), other.begin(),
                result_type{},
                std::plus<>{},
                std::multiplies<>{}
            );
        } else {
            // General path with type promotion
            for (size_type i = 0; i < size(); ++i) {
                result += (*this)[i] * other[i];
            }
        }

        return result;
    }

    // Norm calculations with numerical stability
    scalar_type norm() const {
        // Use stable algorithm to avoid overflow
        scalar_type scale{0};
        scalar_type ssq{1};
        
        for (const auto& x : *this) {
            if constexpr (is_complex_number_v<T>) {
                scalar_type absxi = std::abs(x);
                if (absxi != 0) {
                    if (scale < absxi) {
                        ssq = 1 + ssq * (scale/absxi) * (scale/absxi);
                        scale = absxi;
                    } else {
                        ssq += (absxi/scale) * (absxi/scale);
                    }
                }
            } else {
                scalar_type absxi = std::abs(x);
                if (absxi != 0) {
                    if (scale < absxi) {
                        ssq = 1 + ssq * (scale/absxi) * (scale/absxi);
                        scale = absxi;
                    } else {
                        ssq += (absxi/scale) * (absxi/scale);
                    }
                }
            }
        }
        return scale * std::sqrt(ssq);
    }

    scalar_type norm2() const {
        scalar_type result{0};
        if constexpr (is_complex_number_v<T>) {
            for (const auto& x : *this) {
                result += std::real(x * std::conj(x));
            }
        } else {
            for (const auto& x : *this) {
                result += x * x;
            }
        }
        return result;
    }

    scalar_type norm1() const {
        scalar_type result{};
        if constexpr (is_complex_number_v<T>) {
            for (const auto& x : *this) {
                result += std::abs(x);
            }
        } else {
            for (const auto& x : *this) {
                result += std::abs(x);
            }
        }
        return result;
    }

    scalar_type norm_inf() const {
        scalar_type result{};
        if constexpr (is_complex_number_v<T>) {
            for (const auto& x : *this) {
                result = std::max(result, std::abs(x));
            }
        } else {
            for (const auto& x : *this) {
                result = std::max(result, std::abs(x));
            }
        }
        return result;
    }

    // Normalization with numerical stability
    Vector& normalize() {
        scalar_type n = norm();
        // Check for both zero and subnormal numbers
        if (n > std::numeric_limits<scalar_type>::min()) {
            *this /= n;
        } else if (n != scalar_type{0}) {
            // Handle subnormal case by scaling up first
            scalar_type scale = scalar_type{1} / std::numeric_limits<scalar_type>::min();
            *this *= scale;
            n = norm();
            if (n > scalar_type{0}) {
                *this /= n;
            }
        }
        return *this;
    }

    Vector normalized() const {
        Vector result = *this;
        return result.normalize();
    }

    // Sum and products
    T sum() const {
        if (empty()) return T{0};
        // Only use parallel execution for large vectors
        if (size() > 10000) {
            return std::reduce(std::execution::par_unseq, begin(), end(), T{});
        } else {
            return std::reduce(begin(), end(), T{});
        }
    }

    T product() const {
        if (empty()) return T{1};
        // Only use parallel execution for large vectors
        if (size() > 10000) {
            return std::reduce(std::execution::par_unseq, begin(), end(), T{1}, std::multiplies<>{});
        } else {
            return std::reduce(begin(), end(), T{1}, std::multiplies<>{});
        }
    }

    // Min/Max
    T min() const {
        assert(!empty());
        if constexpr (is_complex_number_v<T>) {
            return *std::min_element(begin(), end(), 
                [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
        } else {
            return *std::min_element(begin(), end());
        }
    }

    T max() const {
        assert(!empty());
        if constexpr (is_complex_number_v<T>) {
            return *std::max_element(begin(), end(), 
                [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
        } else {
            return *std::max_element(begin(), end());
        }
    }

    // === In-place Arithmetic Operators ===

    Vector& operator+=(const Vector& other) {
        check_compatible_size(other);
        // Only use parallel execution for large vectors
        if (size() > 10000) {
            std::transform(std::execution::par_unseq, 
                          begin(), end(), other.begin(), begin(), std::plus<>{});
        } else {
            std::transform(begin(), end(), other.begin(), begin(), std::plus<>{});
        }
        return *this;
    }

    Vector& operator-=(const Vector& other) {
        check_compatible_size(other);
        // Only use parallel execution for large vectors
        if (size() > 10000) {
            std::transform(std::execution::par_unseq, 
                          begin(), end(), other.begin(), begin(), std::minus<>{});
        } else {
            std::transform(begin(), end(), other.begin(), begin(), std::minus<>{});
        }
        return *this;
    }

    Vector& operator*=(const T& scalar) {
        std::transform(std::execution::par_unseq, 
                      begin(), end(), begin(), 
                      [scalar](const T& x) { return x * scalar; });
        return *this;
    }

    Vector& operator/=(const T& scalar) {
        assert(scalar != T{0});
        std::transform(std::execution::par_unseq, 
                      begin(), end(), begin(), 
                      [scalar](const T& x) { return x / scalar; });
        return *this;
    }

    // Element-wise multiplication and division
    Vector& operator*=(const Vector& other) {
        check_compatible_size(other);
        std::transform(std::execution::par_unseq, 
                      begin(), end(), other.begin(), begin(), std::multiplies<>{});
        return *this;
    }

    Vector& operator/=(const Vector& other) {
        check_compatible_size(other);
        std::transform(std::execution::par_unseq, 
                      begin(), end(), other.begin(), begin(), std::divides<>{});
        return *this;
    }

    // === Unary Operators ===

    Vector operator+() const {
        return *this;
    }

    Vector operator-() const {
        Vector result(size());
        std::transform(std::execution::par_unseq, 
                      begin(), end(), result.begin(), std::negate<>{});
        return result;
    }

    // === Expression Template Interface ===

    size_type expression_size() const noexcept {
        return size();
    }

    template<typename Index>
    const T& evaluate(Index i) const noexcept {
        return (*this)[i];
    }

    // === Utility Methods ===

    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }

    void swap(Vector& other) noexcept {
        storage_.swap(other.storage_);
    }

    // === Comparison ===

    bool operator==(const Vector& other) const {
        return size() == other.size() && 
               std::equal(begin(), end(), other.begin());
    }

    bool operator!=(const Vector& other) const {
        return !(*this == other);
    }

    // Approximate equality for floating point
    bool approx_equal(const Vector& other, scalar_type tolerance = numeric_traits<T>::epsilon()) const {
        if (size() != other.size()) return false;
        
        for (size_type i = 0; i < size(); ++i) {
            if constexpr (is_complex_number_v<T>) {
                if (std::abs((*this)[i] - other[i]) > tolerance) return false;
            } else {
                if (std::abs((*this)[i] - other[i]) > tolerance) return false;
            }
        }
        return true;
    }

    // === Debug and Information ===

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Vector<" << typeid(T).name() << ">(" << size() << ")[";
        for (size_type i = 0; i < size(); ++i) {
            if (i > 0) oss << ", ";
            oss << (*this)[i];
            if (i >= 10) { oss << ", ..."; break; }
        }
        oss << "]";
        return oss.str();
    }

    void print(std::ostream& os = std::cout) const {
        os << to_string() << std::endl;
    }

private:
    storage_type storage_;

    // === Helper Methods ===

    void assert_valid_index(size_type index) const {
        assert(index < size() && "Vector index out of bounds");
    }

    void check_bounds(size_type index) const {
        if (index >= size()) {
            std::ostringstream oss;
            oss << "Vector index " << index << " out of bounds (size: " << size() << ")";
            throw std::out_of_range(oss.str());
        }
    }

    void check_slice_bounds(size_type start, size_type end) const {
        if (start > end || end > size()) {
            std::ostringstream oss;
            oss << "Invalid slice bounds [" << start << ", " << end 
                << ") for vector of size " << size();
            throw std::out_of_range(oss.str());
        }
    }

    template<typename OtherVector>
    void check_compatible_size(const OtherVector& other) const {
        if (size() != other.size()) {
            std::ostringstream oss;
            oss << "Vector size mismatch: " << size() << " vs " << other.size();
            throw std::invalid_argument(oss.str());
        }
    }

    template<typename Expr>
    void assign_expression(const Expr& expr) {
        resize(expr.expression_size());
        for (size_type i = 0; i < size(); ++i) {
            (*this)[i] = expr.evaluate(i);
        }
    }
};

// === Type Aliases ===

// Common vector types with default storage
template<typename T>
using VectorDefault = Vector<T, DynamicStorage<T>>;

using VectorF = VectorDefault<float>;
using VectorD = VectorDefault<double>;
using VectorLD = VectorDefault<long double>;
using VectorI = VectorDefault<int>;
using VectorL = VectorDefault<long>;
using VectorLL = VectorDefault<long long>;

// Complex vector types
using VectorCF = VectorDefault<std::complex<float>>;
using VectorCD = VectorDefault<std::complex<double>>;
using VectorCLD = VectorDefault<std::complex<long double>>;

// === Non-member Operations ===

// Binary arithmetic operators
template<StorableType T, typename S1, typename S2>
auto operator+(const Vector<T, S1>& lhs, const Vector<T, S2>& rhs) {
    Vector<T, S1> result(lhs);
    return result += rhs;
}

template<StorableType T, typename S>
auto operator+(const Vector<T, S>& vec, const T& scalar) {
    Vector<T, S> result(vec);
    return result += Vector<T, S>(vec.size(), scalar);
}

template<StorableType T, typename S>
auto operator+(const T& scalar, const Vector<T, S>& vec) {
    return vec + scalar;
}

template<StorableType T, typename S1, typename S2>
auto operator-(const Vector<T, S1>& lhs, const Vector<T, S2>& rhs) {
    Vector<T, S1> result(lhs);
    return result -= rhs;
}

template<StorableType T, typename S>
auto operator-(const Vector<T, S>& vec, const T& scalar) {
    Vector<T> result(vec);
    return result -= Vector<T>(vec.size(), scalar);
}

template<StorableType T, typename S>
auto operator-(const T& scalar, const Vector<T, S>& vec) {
    Vector<T> result(vec.size(), scalar);
    return result -= vec;
}

template<StorableType T, typename S>
auto operator*(const Vector<T, S>& vec, const T& scalar) {
    Vector<T> result(vec);
    return result *= scalar;
}

template<StorableType T, typename S>
auto operator*(const T& scalar, const Vector<T, S>& vec) {
    return vec * scalar;
}

template<StorableType T, typename S>
auto operator/(const Vector<T, S>& vec, const T& scalar) {
    Vector<T> result(vec);
    return result /= scalar;
}

// Element-wise operations
template<typename T, typename S1, typename S2>
auto multiply(const Vector<T, S1>& lhs, const Vector<T, S2>& rhs) {
    Vector<T> result(lhs);
    return result *= rhs;
}

template<typename T, typename S1, typename S2>
auto divide(const Vector<T, S1>& lhs, const Vector<T, S2>& rhs) {
    Vector<T> result(lhs);
    return result /= rhs;
}

// === Stream Operators ===

template<StorableType T, typename S>
std::ostream& operator<<(std::ostream& os, const Vector<T, S>& vec) {
    vec.print(os);
    return os;
}

template<StorableType T, typename S>
std::istream& operator>>(std::istream& is, Vector<T, S>& vec) {
    for (auto& element : vec) {
        is >> element;
    }
    return is;
}

// === Utility Functions ===

template<StorableType T, typename S>
void swap(Vector<T, S>& lhs, Vector<T, S>& rhs) noexcept {
    lhs.swap(rhs);
}

// === Factory Functions ===

template<StorableType T>
VectorDefault<T> zeros(size_t size) {
    return VectorDefault<T>(size, T{0});
}

template<StorableType T>
VectorDefault<T> ones(size_t size) {
    return VectorDefault<T>(size, T{1});
}

template<StorableType T>
VectorDefault<T> linspace(T start, T end, size_t num_points) {
    VectorDefault<T> result(num_points);
    if (num_points <= 1) {
        if (num_points == 1) result[0] = start;
        return result;
    }
    
    T step = (end - start) / T(num_points - 1);
    for (size_t i = 0; i < num_points; ++i) {
        result[i] = start + T(i) * step;
    }
    return result;
}

template<StorableType T>
VectorDefault<T> arange(T start, T end, T step = T{1}) {
    size_t size = static_cast<size_t>((end - start) / step);
    VectorDefault<T> result(size);
    T current = start;
    for (size_t i = 0; i < size; ++i, current += step) {
        result[i] = current;
    }
    return result;
}

} // namespace fem::numeric

// === Hash Support ===
namespace std {
    template<typename T, typename S>
    struct hash<fem::numeric::Vector<T, S>> {
        std::size_t operator()(const fem::numeric::Vector<T, S>& vec) const {
            std::size_t seed = vec.size();
            for (const auto& element : vec) {
                seed ^= std::hash<T>{}(element) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

#endif // NUMERIC_VECTOR_H

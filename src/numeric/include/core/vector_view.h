#pragma once

#ifndef NUMERIC_VECTOR_VIEW_H
#define NUMERIC_VECTOR_VIEW_H

#include "../base/view_base.h"
#include "../base/traits_base.h"

namespace fem::numeric {

// Forward declaration
template<typename T, typename Storage> class Vector;

/**
 * @brief Lightweight view of vector data
 * 
 * Provides non-owning access to contiguous vector data with
 * minimal overhead and full mathematical operations support.
 */
template<typename T>
class VectorView : public ViewBase<T> {
public:
    using base_type = ViewBase<T>;
    using value_type = T;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using scalar_type = typename scalar_type<T>::type;
    
    // Inherit constructors
    using ViewBase<T>::ViewBase;
    
    // Mathematical operations
    scalar_type norm() const {
        scalar_type scale{0};
        scalar_type ssq{1};
        
        for (size_type i = 0; i < this->size(); ++i) {
            scalar_type absxi = std::abs((*this)[i]);
            if (absxi != 0) {
                if (scale < absxi) {
                    ssq = 1 + ssq * (scale/absxi) * (scale/absxi);
                    scale = absxi;
                } else {
                    ssq += (absxi/scale) * (absxi/scale);
                }
            }
        }
        return scale * std::sqrt(ssq);
    }
    
    scalar_type norm2() const {
        scalar_type result{0};
        for (size_type i = 0; i < this->size(); ++i) {
            const auto& x = (*this)[i];
            if constexpr (is_complex_number_v<T>) {
                result += std::real(x * std::conj(x));
            } else {
                result += x * x;
            }
        }
        return result;
    }
    
    T sum() const {
        T result{0};
        for (size_type i = 0; i < this->size(); ++i) {
            result += (*this)[i];
        }
        return result;
    }
    
    // Dot product with another view or vector
    template<typename Other>
    auto dot(const Other& other) const {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Size mismatch in dot product");
        }
        
        using result_type = decltype(T{} * typename Other::value_type{});
        result_type result{};
        
        for (size_type i = 0; i < this->size(); ++i) {
            result += (*this)[i] * other[i];
        }
        
        return result;
    }
};

/**
 * @brief Slice view for vectors with stride support
 * 
 * Provides strided access to vector data for advanced slicing operations.
 */
template<typename T>
class VectorSlice : public StridedView<T> {
public:
    using base_type = StridedView<T>;
    using value_type = T;
    using size_type = size_t;
    using scalar_type = typename scalar_type<T>::type;
    
    // Constructor from vector and slice
    template<typename VectorType>
    VectorSlice(VectorType& vec, const Slice& slice)
        : base_type(nullptr, 0, 1) {
        const auto n = vec.size();
        const auto norm = slice.normalize(n);
        const auto count = slice.count(n);
        if (count == 0) {
            // leave as empty view
            return;
        }
        const std::ptrdiff_t step = norm.step();
        const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(norm.start());
        // Construct base with computed pointer, count, and stride
        *static_cast<base_type*>(this) = base_type(vec.data() + start, count, step);
    }
    
    // Mathematical operations (inherit from base or implement as needed)
    scalar_type norm() const {
        scalar_type scale{0};
        scalar_type ssq{1};
        
        for (size_type i = 0; i < this->size(); ++i) {
            scalar_type absxi = std::abs((*this)[i]);
            if (absxi != 0) {
                if (scale < absxi) {
                    ssq = 1 + ssq * (scale/absxi) * (scale/absxi);
                    scale = absxi;
                } else {
                    ssq += (absxi/scale) * (absxi/scale);
                }
            }
        }
        return scale * std::sqrt(ssq);
    }
};

} // namespace fem::numeric

#endif // NUMERIC_VECTOR_VIEW_H

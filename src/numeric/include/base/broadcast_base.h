#pragma once

#ifndef NUMERIC_BROADCAST_BASE_H
#define NUMERIC_BROADCAST_BASE_H

#include <algorithm>
#include <numeric>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Broadcasting support for operations
     *
     * Implements broadcasting rules for IEEE-compliant numeric operations
     */
    class BroadcastHelper {
    public:
        /**
         * @brief Check if two shapes are compatible for broadcasting
         */
        static bool are_broadcastable(const Shape& shape1, const Shape& shape2) {
            size_t ndim1 = shape1.rank();
            size_t ndim2 = shape2.rank();
            size_t max_ndim = std::max(ndim1, ndim2);

            // Compare dimensions from right to left
            for (size_t i = 0; i < max_ndim; ++i) {
                size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
                size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

                // Dimensions are compatible if equal or one is 1
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Compute broadcasted shape from two shapes
         */
        static Shape broadcast_shape(const Shape& shape1, const Shape& shape2) {
            if (!are_broadcastable(shape1, shape2)) {
                throw DimensionError("Shapes are not compatible for broadcasting");
            }

            size_t ndim1 = shape1.rank();
            size_t ndim2 = shape2.rank();
            size_t max_ndim = std::max(ndim1, ndim2);

            std::vector<size_t> result_dims(max_ndim);

            for (size_t i = 0; i < max_ndim; ++i) {
                size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
                size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

                result_dims[max_ndim - 1 - i] = std::max(dim1, dim2);
            }

            return Shape(result_dims);
        }

        /**
         * @brief Compute broadcasted shape from multiple shapes
         */
        template<typename... Shapes>
        static Shape broadcast_shapes(const Shape& first, const Shapes&... rest) {
            if constexpr (sizeof...(rest) == 0) {
                return first;
            } else {
                Shape result = first;
                ((result = broadcast_shape(result, rest)), ...);
                return result;
            }
        }

        /**
         * @brief Convert flat index to multi-dimensional indices
         */
        static std::vector<size_t> unravel_index(size_t flat_idx, const Shape& shape) {
            std::vector<size_t> indices(shape.rank());

            for (size_t i = shape.rank(); i > 0; --i) {
                indices[i - 1] = flat_idx % shape[i - 1];
                flat_idx /= shape[i - 1];
            }

            return indices;
        }

        /**
         * @brief Convert multi-dimensional indices to flat index
         */
        static size_t ravel_index(const std::vector<size_t>& indices, const Shape& shape) {
            size_t flat_idx = 0;
            size_t stride = 1;

            for (size_t i = shape.rank(); i > 0; --i) {
                flat_idx += indices[i - 1] * stride;
                stride *= shape[i - 1];
            }

            return flat_idx;
        }

        /**
         * @brief Map broadcasted index back to original shape index
         */
        static size_t broadcast_index(size_t broadcast_flat_idx,
                                      const Shape& broadcast_shape,
                                      const Shape& original_shape) {
            // Unravel broadcasted index
            auto broadcast_indices = unravel_index(broadcast_flat_idx, broadcast_shape);

            // Map to original indices
            std::vector<size_t> original_indices(original_shape.rank());
            size_t broadcast_offset = broadcast_shape.rank() - original_shape.rank();

            for (size_t i = 0; i < original_shape.rank(); ++i) {
                size_t broadcast_dim = broadcast_indices[i + broadcast_offset];
                size_t original_dim = original_shape[i];

                // If dimension is 1 in original, index is always 0
                original_indices[i] = (original_dim == 1) ? 0 : broadcast_dim;
            }

            return ravel_index(original_indices, original_shape);
        }

        /**
         * @brief Iterator for broadcasting
         */
        template<typename T>
        class BroadcastIterator {
        public:
            BroadcastIterator(const T* data, const Shape& data_shape,
                              const Shape& broadcast_shape, size_t position = 0)
                    : data_(data), data_shape_(data_shape),
                      broadcast_shape_(broadcast_shape), position_(position) {}

            T operator*() const {
                size_t idx = BroadcastHelper::broadcast_index(
                        position_, broadcast_shape_, data_shape_);
                return data_[idx];
            }

            BroadcastIterator& operator++() {
                ++position_;
                return *this;
            }

            BroadcastIterator operator++(int) {
                BroadcastIterator tmp = *this;
                ++position_;
                return tmp;
            }

            bool operator==(const BroadcastIterator& other) const {
                return position_ == other.position_;
            }

            bool operator!=(const BroadcastIterator& other) const {
                return position_ != other.position_;
            }

        private:
            const T* data_;
            Shape data_shape_;
            Shape broadcast_shape_;
            size_t position_;
        };

        /**
         * @brief Create broadcast iterator
         */
        template<typename T>
        static BroadcastIterator<T> make_broadcast_iterator(
                const T* data, const Shape& data_shape, const Shape& broadcast_shape) {
            return BroadcastIterator<T>(data, data_shape, broadcast_shape);
        }
    };

    /**
     * @brief Broadcast expression for lazy evaluation
     */
    template<typename Container>
    class BroadcastExpression {
    public:
        using value_type = typename Container::value_type;

        BroadcastExpression(const Container& container, const Shape& target_shape)
                : container_(container), target_shape_(target_shape) {
            if (!BroadcastHelper::are_broadcastable(container.shape(), target_shape)) {
                throw DimensionError("Cannot broadcast to target shape");
            }
        }

        Shape shape() const { return target_shape_; }
        size_t size() const { return target_shape_.size(); }

        value_type operator[](size_t i) const {
            size_t original_idx = BroadcastHelper::broadcast_index(
                    i, target_shape_, container_.shape());
            return container_[original_idx];
        }

        template<typename ResultContainer>
        void eval_to(ResultContainer& result) const {
            if (result.shape() != target_shape_) {
                result.resize(target_shape_);
            }

            for (size_t i = 0; i < target_shape_.size(); ++i) {
                result[i] = (*this)[i];
            }
        }

    private:
        const Container& container_;
        Shape target_shape_;
    };

    /**
     * @brief Axis reducer for reduction operations
     */
    class AxisReducer {
    public:
        /**
         * @brief Compute shape after reduction along axes
         */
        static Shape reduced_shape(const Shape& shape,
                                   const std::vector<int>& axes,
                                   bool keepdims = false) {
            std::vector<size_t> result_dims;
            std::vector<bool> reduce_axis(shape.rank(), false);

            // Mark axes to reduce
            for (int axis : axes) {
                int normalized_axis = axis < 0 ? shape.rank() + axis : axis;
                if (normalized_axis < 0 || normalized_axis >= static_cast<int>(shape.rank())) {
                    throw std::out_of_range("Axis out of range");
                }
                reduce_axis[normalized_axis] = true;
            }

            // Build result shape
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (reduce_axis[i]) {
                    if (keepdims) {
                        result_dims.push_back(1);
                    }
                } else {
                    result_dims.push_back(shape[i]);
                }
            }

            // If all axes reduced and not keepdims, result is scalar (rank 0)
            if (result_dims.empty() && !keepdims) {
                result_dims.push_back(1);  // Scalar as 1-element array
            }

            return Shape(result_dims);
        }

        /**
         * @brief Get iteration pattern for reduction
         */
        static std::vector<std::vector<size_t>> get_reduction_groups(
                const Shape& shape, const std::vector<int>& axes) {

            // This is simplified - full implementation would handle all cases
            std::vector<std::vector<size_t>> groups;

            // For now, just create groups for reduction
            size_t total_size = shape.size();
            Shape reduced = reduced_shape(shape, axes, false);
            size_t n_groups = reduced.size();

            groups.resize(n_groups);

            // Distribute indices into groups
            // (Simplified - actual implementation needs proper index mapping)
            for (size_t i = 0; i < total_size; ++i) {
                size_t group_idx = i % n_groups;
                groups[group_idx].push_back(i);
            }

            return groups;
        }
    };

    /**
     * @brief Helper for einsum-style operations
     */
    class EinsumHelper {
    public:
        // Simplified einsum parser and executor
        // Full implementation would parse Einstein notation strings
        // and generate optimized contraction code

        static Shape parse_einsum_shape(const std::string& subscripts,
                                        const std::vector<Shape>& input_shapes) {
            // Simplified - would parse subscripts and compute output shape
            return Shape();
        }

        template<typename T>
        static void execute_einsum(const std::string& subscripts,
                                   const std::vector<const T*>& inputs,
                                   const std::vector<Shape>& input_shapes,
                                   T* output,
                                   const Shape& output_shape) {
            // Simplified - would execute the einsum operation
            // Full implementation requires parsing and optimization
        }
    };

} // namespace fem::numeric

#endif //NUMERIC_BROADCAST_BASE_H

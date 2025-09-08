#pragma once

#ifndef NUMERIC_BROADCAST_BASE_H
#define NUMERIC_BROADCAST_BASE_H

#include <algorithm>
#include <vector>

#include "numeric_base.h"

namespace fem::numeric {
    // Forward declaration
    template<typename T>
    class BroadcastIterator;
    /**
     * @brief Broadcasting support for element-wise operations
     *
     * Implements NumPy-style broadcasting rules for numeric operations.
     * This is a lightweight utility that only depends on Shape from numeric_base.h
     */
    class BroadcastHelper {
    public:
        /**
         * @brief Check if two shapes are compatible for broadcasting
         *
         * Two shapes are compatible when, comparing from right to left:
         * - Dimensions are equal, OR
         * - One dimension is 1, OR
         * - One shape has fewer dimensions (implicitly prepended with 1s)
         */
        static bool are_broadcastable(const Shape& shape1, const Shape& shape2) {
            size_t ndim1 = shape1.rank();
            size_t ndim2 = shape2.rank();

            // Check if either shape represents an empty array (has a 0 dimension)
            bool shape1_empty = false;
            bool shape2_empty = false;

            for (size_t i = 0; i < ndim1; ++i) {
                if (shape1[i] == 0) {
                    shape1_empty = true;
                    break;
                }
            }

            for (size_t i = 0; i < ndim2; ++i) {
                if (shape2[i] == 0) {
                    shape2_empty = true;
                    break;
                }
            }

            // If both shapes are empty, they're always broadcastable
            // (the result will also be empty)
            if (shape1_empty && shape2_empty) {
                return true;
            }

            // If only one is empty, they're not broadcastable
            if (shape1_empty || shape2_empty) {
                return false;
            }

            // Normal broadcasting rules for non-empty arrays
            size_t min_ndim = std::min(ndim1, ndim2);
            for (size_t i = 0; i < min_ndim; ++i) {
                size_t dim1 = shape1[ndim1 - 1 - i];
                size_t dim2 = shape2[ndim2 - 1 - i];

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Compute the result shape after broadcasting
         */
        static Shape broadcast_shape(const Shape& shape1, const Shape& shape2) {
            if (!are_broadcastable(shape1, shape2)) {
                throw DimensionError(
                    "Shapes " + shape1.to_string() + " and " +
                    shape2.to_string() + " are not compatible for broadcasting"
                );
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
         * @brief Map a broadcasted flat index to the original flat index
         *
         * Given an index in the broadcasted shape, returns the corresponding
         * index in the original shape (handling dimension replication)
         */
        static size_t map_broadcast_index(
            size_t broadcast_flat_idx,
            const Shape& broadcast_shape,
            const Shape& original_shape
        ) {
            // Convert flat index to multi-dimensional coordinates
            std::vector<size_t> broadcast_coords(broadcast_shape.rank());
            size_t temp_idx = broadcast_flat_idx;

            for (size_t i = broadcast_shape.rank(); i > 0; --i) {
                broadcast_coords[i - 1] = temp_idx % broadcast_shape[i - 1];
                temp_idx /= broadcast_shape[i - 1];
            }

            // Map to original coordinates (handling broadcasting)
            std::vector<size_t> original_coords(original_shape.rank());
            size_t broadcast_offset = broadcast_shape.rank() - original_shape.rank();

            for (size_t i = 0; i < original_shape.rank(); ++i) {
                size_t broadcast_coord = broadcast_coords[i + broadcast_offset];
                size_t original_dim = original_shape[i];

                // If dimension is 1 in original, always use index 0
                original_coords[i] = (original_dim == 1) ? 0 : broadcast_coord;
            }

            // Convert back to flat index
            size_t flat_idx = 0;
            size_t stride = 1;

            for (size_t i = original_shape.rank(); i > 0; --i) {
                flat_idx += original_coords[i - 1] * stride;
                stride *= original_shape[i - 1];
            }

            return flat_idx;
        }

        /**
         * @brief Compute strides for broadcasting iteration
         *
         * Returns stride values that can be used for efficient iteration
         * over broadcasted arrays without copying data
         */
        static std::vector<std::ptrdiff_t> compute_broadcast_strides(
            const Shape& original_shape,
            const Shape& broadcast_shape
        ) {
            size_t original_rank = original_shape.rank();
            size_t broadcast_rank = broadcast_shape.rank();
            std::vector<std::ptrdiff_t> strides(broadcast_rank);

            // Start from the rightmost dimension
            std::ptrdiff_t stride = 1;

            for (size_t i = 0; i < broadcast_rank; ++i) {
                size_t broadcast_idx = broadcast_rank - 1 - i;

                if (i < original_rank) {
                    size_t original_idx = original_rank - 1 - i;

                    if (original_shape[original_idx] == 1) {
                        // Broadcasted dimension - stride is 0
                        strides[broadcast_idx] = 0;
                    } else {
                        // Normal dimension
                        strides[broadcast_idx] = stride;
                        stride *= static_cast<std::ptrdiff_t>(original_shape[original_idx]);
                    }
                } else {
                    // Dimension doesn't exist in original (implicit 1)
                    strides[broadcast_idx] = 0;
                }
            }

            return strides;
        }

        /**
         * @brief Create a broadcast iterator for convenient iteration
         *
         * @param data Pointer to the original data
         * @param data_shape Shape of the original data
         * @param broadcast_shape Target shape to broadcast to
         * @return BroadcastIterator configured for the broadcast operation
         */
        template<typename T>
        static BroadcastIterator<T> make_broadcast_iterator(
            const T* data,
            const Shape& data_shape,
            const Shape& broadcast_shape,
            size_t position = 0) {

            // Verify shapes are broadcastable
            if (!are_broadcastable(data_shape, broadcast_shape)) {
                throw DimensionError(
                    "Cannot create broadcast iterator: shapes " +
                    data_shape.to_string() + " and " +
                    broadcast_shape.to_string() + " are not broadcastable"
                );
            }

            return BroadcastIterator<T>(data, data_shape, broadcast_shape, position);
        }

        /**
         * @brief Create begin and end iterators for range-based iteration
         *
         * @param data Pointer to the original data
         * @param data_shape Shape of the original data
         * @param broadcast_shape Target shape to broadcast to
         * @return Pair of begin and end iterators
         */
        template<typename T>
        static std::pair<BroadcastIterator<T>, BroadcastIterator<T>>
        make_broadcast_range(
            const T* data,
            const Shape& data_shape,
            const Shape& broadcast_shape) {

            auto begin = make_broadcast_iterator(data, data_shape, broadcast_shape, 0);
            auto end = make_broadcast_iterator(data, data_shape, broadcast_shape,
                                               broadcast_shape.size());
            return {begin, end};
        }
    };

    /**
     * @brief Iterator for efficient broadcasting without copying data
     */
    template<typename T>
    class BroadcastIterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using reference = const T&;
        using pointer = const T*;
        using iterator_category = std::input_iterator_tag;

        BroadcastIterator(const T* data,
                          const Shape& original_shape,
                          const Shape& broadcast_shape,
                          size_t position = 0)
            : data_(data)
            , original_shape_(original_shape)
            , broadcast_shape_(broadcast_shape)
            , position_(position)
            , strides_(BroadcastHelper::compute_broadcast_strides(
                  original_shape, broadcast_shape)) {}

        T operator*() const {
            // Use strides for efficient index calculation
            size_t original_idx = 0;
            size_t temp = position_;

            for (size_t i = broadcast_shape_.rank(); i > 0; --i) {
                size_t coord = temp % broadcast_shape_[i-1];
                temp /= broadcast_shape_[i-1];

                // Handle stride multiplication safely
                std::ptrdiff_t stride = strides_[i-1];
                if (stride > 0) {
                    original_idx += coord * static_cast<size_t>(stride);
                }
                // stride == 0 means broadcasted dimension, adds nothing
            }

            return data_[original_idx];
        }

        BroadcastIterator& operator++() {
            ++position_;
            return *this;
        }

        bool operator!=(const BroadcastIterator& other) const {
            return position_ != other.position_;
        }

    private:
        const T* data_;
        Shape original_shape_;
        Shape broadcast_shape_;
        size_t position_;
        std::vector<std::ptrdiff_t> strides_;
    };
} // namespace fem::numeric

#endif //NUMERIC_BROADCAST_BASE_H
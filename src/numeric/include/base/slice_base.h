#pragma once

#ifndef NUMERIC_SLICE_BASE_H
#define NUMERIC_SLICE_BASE_H

#include <optional>
#include <variant>
#include <vector>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Slice object for NumPy-like slicing
     */
    class Slice {
    public:
        using index_type = std::ptrdiff_t;
        static constexpr index_type none = std::numeric_limits<index_type>::max();

        // Constructors for different slice types
        Slice() : start_(0), stop_(none), step_(1) {}

        explicit Slice(index_type stop)
                : start_(0), stop_(stop), step_(1) {}

        Slice(index_type start, index_type stop)
                : start_(start), stop_(stop), step_(1) {}

        Slice(index_type start, index_type stop, index_type step)
                : start_(start), stop_(stop), step_(step) {
            if (step == 0) {
                throw std::invalid_argument("Slice step cannot be zero");
            }
        }

        // Static factory methods for common slices
        static Slice all() { return Slice(); }
        static Slice from(index_type start) { return Slice(start, none, 1); }
        static Slice to(index_type stop) { return Slice(0, stop, 1); }
        static Slice single(index_type index) { return Slice(index, index + 1, 1); }

        // Getters
        index_type start() const noexcept { return start_; }
        index_type stop() const noexcept { return stop_; }
        index_type step() const noexcept { return step_; }

        bool is_all() const noexcept {
            return start_ == 0 && stop_ == none && step_ == 1;
        }

        // Normalize slice for given size
        Slice normalize(size_t size) const {
            index_type norm_start = start_;
            index_type norm_stop = stop_;

            // Handle negative indices
            if (norm_start < 0) {
                norm_start += static_cast<index_type>(size);
            }
            if (norm_stop < 0) {
                norm_stop += static_cast<index_type>(size);
            }

            // Handle none/max values
            if (norm_stop == none || norm_stop > static_cast<index_type>(size)) {
                norm_stop = static_cast<index_type>(size);
            }

            // Clamp to valid range
            norm_start = std::max<index_type>(0, std::min<index_type>(norm_start, size));
            norm_stop = std::max<index_type>(0, std::min<index_type>(norm_stop, size));

            return Slice(norm_start, norm_stop, step_);
        }

        // Calculate number of elements in slice
        size_t count(size_t size) const {
            Slice norm = normalize(size);

            if (norm.start_ >= norm.stop_) {
                return 0;
            }

            if (step_ > 0) {
                return (norm.stop_ - norm.start_ + step_ - 1) / step_;
            } else {
                return (norm.start_ - norm.stop_ - step_ - 1) / (-step_);
            }
        }

        // Get indices for iteration
        std::vector<size_t> indices(size_t size) const {
            Slice norm = normalize(size);
            std::vector<size_t> result;
            result.reserve(count(size));

            if (step_ > 0) {
                for (index_type i = norm.start_; i < norm.stop_; i += step_) {
                    result.push_back(static_cast<size_t>(i));
                }
            } else {
                for (index_type i = norm.start_; i > norm.stop_; i += step_) {
                    result.push_back(static_cast<size_t>(i));
                }
            }

            return result;
        }

    private:
        index_type start_;
        index_type stop_;
        index_type step_;
    };

    /**
     * @brief All sentinel for selecting all elements
     */
        struct All {};
        inline constexpr All all{};

    /**
     * @brief NewAxis sentinel for adding dimensions
     */
    struct NewAxis {};
    inline constexpr NewAxis newaxis{};

    /**
     * @brief Ellipsis for selecting multiple dimensions
     */
    struct Ellipsis {};
    inline constexpr Ellipsis ellipsis{};

    /**
     * @brief Index type that can be integer, slice, or special value
     */
    using IndexVariant = std::variant<
    std::ptrdiff_t,           // Single index
    Slice,                    // Slice object
    std::vector<std::ptrdiff_t>, // Integer array indexing
    std::vector<bool>,        // Boolean mask indexing
    All,                      // Select all
    NewAxis,                  // Add dimension
    Ellipsis                  // Multiple dimensions
    >;

    /**
     * @brief Multi-dimensional index for advanced indexing
     */
    class MultiIndex {
    public:
        MultiIndex() = default;

        MultiIndex(std::initializer_list<IndexVariant> indices)
                : indices_(indices) {}

        explicit MultiIndex(const std::vector<IndexVariant>& indices)
                : indices_(indices) {}

        // Add index component
        void append(const IndexVariant& index) {
            indices_.push_back(index);
        }

        // Get number of index components
        size_t size() const noexcept { return indices_.size(); }

        // Access index component
        const IndexVariant& operator[](size_t i) const { return indices_[i]; }
        IndexVariant& operator[](size_t i) { return indices_[i]; }

        // Check if contains ellipsis
        bool has_ellipsis() const {
            for (const auto& idx : indices_) {
                if (std::holds_alternative<Ellipsis>(idx)) {
                    return true;
                }
            }
            return false;
        }

        // Count newaxis occurrences
        size_t newaxis_count() const {
            size_t count = 0;
            for (const auto& idx : indices_) {
                if (std::holds_alternative<NewAxis>(idx)) {
                    count++;
                }
            }
            return count;
        }

        // Normalize for given shape
        MultiIndex normalize(const Shape& shape) const {
            MultiIndex result;
            size_t shape_idx = 0;

            for (const auto& idx : indices_) {
                if (std::holds_alternative<Ellipsis>(idx)) {
                    // Expand ellipsis to match remaining dimensions
                    size_t remaining = shape.rank() - shape_idx - (indices_.size() - result.size() - 1);
                    for (size_t i = 0; i < remaining; ++i) {
                        result.append(All{});
                        shape_idx++;
                    }
                } else if (std::holds_alternative<NewAxis>(idx)) {
                    result.append(idx);
                    // NewAxis doesn't consume a dimension
                } else {
                    result.append(idx);
                    shape_idx++;
                }
            }

            return result;
        }

        // Calculate result shape after indexing
        Shape result_shape(const Shape& input_shape) const {
            std::vector<size_t> dims;
            MultiIndex norm = normalize(input_shape);
            size_t shape_idx = 0;

            for (const auto& idx : norm.indices_) {
                if (std::holds_alternative<std::ptrdiff_t>(idx)) {
                    // Single index reduces dimension
                    shape_idx++;
                } else if (std::holds_alternative<Slice>(idx)) {
                    // Slice preserves dimension with new size
                    const auto& slice = std::get<Slice>(idx);
                    dims.push_back(slice.count(input_shape[shape_idx]));
                    shape_idx++;
                } else if (std::holds_alternative<All>(idx)) {
                    // All preserves dimension
                    dims.push_back(input_shape[shape_idx]);
                    shape_idx++;
                } else if (std::holds_alternative<NewAxis>(idx)) {
                    // NewAxis adds dimension of size 1
                    dims.push_back(1);
                } else if (std::holds_alternative<std::vector<std::ptrdiff_t>>(idx)) {
                    // Integer array indexing
                    const auto& vec = std::get<std::vector<std::ptrdiff_t>>(idx);
                    dims.push_back(vec.size());
                    shape_idx++;
                } else if (std::holds_alternative<std::vector<bool>>(idx)) {
                    // Boolean mask indexing
                    const auto& mask = std::get<std::vector<bool>>(idx);
                    dims.push_back(std::count(mask.begin(), mask.end(), true));
                    shape_idx++;
                }
            }

            return Shape(dims);
        }

    private:
        std::vector<IndexVariant> indices_;
    };

    /**
     * @brief Slice parser for string-based slicing (e.g., "1:5:2")
     */
    class SliceParser {
    public:
        static Slice parse(const std::string& str) {
            if (str.empty() || str == ":") {
                return Slice::all();
            }

            std::vector<std::optional<std::ptrdiff_t>> parts;
            size_t start = 0;

            // Split by ':'
            for (size_t i = 0; i <= str.length(); ++i) {
                if (i == str.length() || str[i] == ':') {
                    std::string part = str.substr(start, i - start);
                    if (part.empty()) {
                        parts.push_back(std::nullopt);
                    } else {
                        parts.push_back(std::stoi(part));
                    }
                    start = i + 1;
                }
            }

            // Create slice based on number of parts
            if (parts.size() == 1) {
                return parts[0] ? Slice(*parts[0]) : Slice();
            } else if (parts.size() == 2) {
                auto start = parts[0].value_or(0);
                auto stop = parts[1].value_or(Slice::none);
                return Slice(start, stop);
            } else if (parts.size() == 3) {
                auto start = parts[0].value_or(0);
                auto stop = parts[1].value_or(Slice::none);
                auto step = parts[2].value_or(1);
                return Slice(start, stop, step);
            }

            throw std::invalid_argument("Invalid slice format: " + str);
        }
    };

    /**
     * @brief User-defined literal for slice creation
     */
    inline namespace literals {
        inline Slice operator""_s(const char* str, size_t) {
            return SliceParser::parse(str);
        }
    }

} // namespace fem::numeric

#endif //NUMERIC_SLICE_BASE_H

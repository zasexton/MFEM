#pragma once

#ifndef NUMERIC_SLICE_BASE_H
#define NUMERIC_SLICE_BASE_H

#include <algorithm>
#include <numeric>
#include <string>
#include <optional>
#include <variant>
#include <vector>
#include <limits>

#include "numeric_base.h"

namespace fem::numeric {

    // Forward declarations
    class MultiIndex;
    class SliceParser;

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
            auto ssize = static_cast<index_type>(size);

            // Handle negative indices
            if (norm_start != none && norm_start < 0) {
                norm_start += ssize;
            }
            if (norm_stop != none && norm_stop < 0) {
                norm_stop += ssize;
            }

            if (step_ > 0) {
                // Forward iteration
                if (norm_start == none) norm_start = 0;
                if (norm_stop == none) norm_stop = ssize;

                // Clamp to valid range
                norm_start = std::max<index_type>(0, std::min<index_type>(norm_start, ssize));
                norm_stop = std::max<index_type>(0, std::min<index_type>(norm_stop, ssize));
            } else {
                // Backward iteration
                if (norm_start == none) norm_start = ssize - 1;
                if (norm_stop == none) {
                    norm_stop = -(ssize + 1); // Special marker for "include 0"
                }

                // Clamp start only (stop can go negative for reverse)
                norm_start = std::max<index_type>(-1, std::min<index_type>(norm_start, ssize - 1));
                // Don't clamp stop - it indicates where to stop
            }

            return Slice(norm_start, norm_stop, step_);
        }

        size_t count(size_t size) const {
            Slice norm = normalize(size);
            auto ssize = static_cast<index_type>(size);

            if (step_ > 0) {
                if (norm.start_ >= norm.stop_) {
                    return 0;
                }
                return static_cast<size_t>((norm.stop_ - norm.start_ + step_ - 1) / step_);
            } else {
                // Check for special sentinel value
                index_type actual_stop = norm.stop_;
                if (actual_stop <= -ssize) {
                    actual_stop = -1;  // Include index 0
                }

                if (norm.start_ <= actual_stop) {
                    return 0;
                }
                return static_cast<size_t>((norm.start_ - actual_stop - step_ - 1) / (-step_));
            }
        }

        std::vector<size_t> indices(size_t size) const {
            Slice norm = normalize(size);
            auto ssize = static_cast<index_type>(size);
            std::vector<size_t> result;

            if (step_ > 0) {
                if (norm.start_ >= norm.stop_) {
                    return result;
                }
                result.reserve(count(size));
                for (index_type i = norm.start_; i < norm.stop_; i += step_) {
                    result.push_back(static_cast<size_t>(i));
                }
            } else {
                // Check for special sentinel value
                index_type actual_stop = norm.stop_;
                if (actual_stop <= -ssize) {
                    actual_stop = -1;  // Include index 0
                }

                if (norm.start_ <= actual_stop) {
                    return result;
                }
                result.reserve(count(size));
                for (index_type i = norm.start_; i > actual_stop; i += step_) {
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

        const std::vector<IndexVariant>& components() const noexcept { return indices_; }
        auto begin() const noexcept { return indices_.begin(); }
        auto end() const noexcept { return indices_.end(); }

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

            // First, check for multiple ellipses
            int ellipsis_count = 0;
            for (const auto& idx : indices_) {
                if (std::holds_alternative<Ellipsis>(idx)) {
                    ellipsis_count++;
                }
            }
            if (ellipsis_count > 1) {
                throw std::invalid_argument("Only one ellipsis allowed in index");
            }

            // Count how many dimensions we're indexing (excluding NewAxis and Ellipsis)
            size_t index_dims = 0;
            for (const auto& idx : indices_) {
                if (!std::holds_alternative<NewAxis>(idx) &&
                    !std::holds_alternative<Ellipsis>(idx)) {
                    index_dims++;
                    }
            }

            // If no ellipsis and we have more indices than dimensions, error
            if (ellipsis_count == 0 && index_dims > shape.rank()) {
                throw std::out_of_range("Too many indices for shape");
            }

            // Now do the actual normalization
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

            while (shape_idx < shape.rank()) {
                result.append(All{});
                shape_idx++;
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
                    dims.push_back(static_cast<size_t>(std::count(mask.begin(), mask.end(), true)));
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

            // Count colons
            size_t colon_count = static_cast<size_t>(std::count(str.begin(), str.end(), ':'));
            if (colon_count > 2) {
                throw std::invalid_argument("Too many colons in slice: " + str);
            }

            // "::0" should still be invalid (zero step)
            if (str == "::0") {
                throw std::invalid_argument("Slice step cannot be zero");
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
                        // Validate that the string is a valid integer
                        // Check for invalid characters (including '.')
                        bool is_negative = (part[0] == '-');
                        size_t digit_start = is_negative ? 1 : 0;

                        for (size_t j = digit_start; j < part.length(); ++j) {
                            if (!std::isdigit(part[j])) {
                                throw std::invalid_argument("Invalid number in slice: " + part);
                            }
                        }

                        if (digit_start == part.length()) {
                            // Just a '-' with no digits
                            throw std::invalid_argument("Invalid number in slice: " + part);
                        }

                        try {
                            parts.push_back(std::stoi(part));
                        } catch (const std::exception&) {
                            throw std::invalid_argument("Invalid number in slice: " + part);
                        }
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
     * @brief Parser for NumPy-like multi-dimensional indexing strings
     * Supports syntax like ":,2,:", "3::,5:", "None,3", "...,2", etc.
     */
    class IndexParser {
    public:
        static MultiIndex parse(const std::string& str);  // Declaration only

    private:
        static IndexVariant parse_single_index(const std::string& part);
        static IndexVariant parse_array_index(const std::string& content);
    };

    // Implementation of IndexParser methods after all classes are defined
    inline MultiIndex IndexParser::parse(const std::string& str) {
        MultiIndex result;

        // Remove outer brackets if present
        std::string s = str;
        if (!s.empty() && s.front() == '[' && s.back() == ']') {
            s = s.substr(1, s.length() - 2);
        }

        size_t start = 0;
        int bracket_depth = 0;

        // Split by commas, but respect bracket nesting
        for (size_t i = 0; i <= s.length(); ++i) {
            if (i < s.length()) {
                if (s[i] == '[') bracket_depth++;
                else if (s[i] == ']') bracket_depth--;
            }

            if ((i == s.length() || (s[i] == ',' && bracket_depth == 0))) {
                std::string part = s.substr(start, i - start);

                // Trim whitespace
                part.erase(0, part.find_first_not_of(" \t\n\r"));
                part.erase(part.find_last_not_of(" \t\n\r") + 1);

                result.append(parse_single_index(part));
                start = i + 1;
            }
        }

        return result;
    }

    inline IndexVariant IndexParser::parse_single_index(const std::string& part) {
        // Empty or just ":" means all
        if (part.empty() || part == ":") {
            return all;
        }

        // Ellipsis
        if (part == "...") {
            return ellipsis;
        }

        // NewAxis (Python's None)
        if (part == "None" || part == "np.newaxis") {
            return newaxis;
        }

        // Check if it's a slice (contains ':')
        if (part.find(':') != std::string::npos) {
            return SliceParser::parse(part);
        }

        // Check for array indexing [1,2,3]
        if (part.front() == '[' && part.back() == ']') {
            return parse_array_index(part.substr(1, part.length() - 2));
        }

        // Otherwise, try to parse as integer
        bool is_negative = !part.empty() && part[0] == '-';
        size_t digit_start = is_negative ? 1 : 0;

        for (size_t j = digit_start; j < part.length(); ++j) {
            if (!std::isdigit(part[j])) {
                throw std::invalid_argument("Invalid index: " + part);
            }
        }

        if (digit_start == part.length()) {
            throw std::invalid_argument("Invalid index: " + part);
        }

        try {
            return static_cast<std::ptrdiff_t>(std::stoll(part));
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid index: " + part);
        }
    }

    inline IndexVariant IndexParser::parse_array_index(const std::string& content) {
        std::vector<std::ptrdiff_t> indices;
        size_t start = 0;

        for (size_t i = 0; i <= content.length(); ++i) {
            if (i == content.length() || content[i] == ',') {
                std::string num = content.substr(start, i - start);

                // Trim whitespace
                num.erase(0, num.find_first_not_of(" \t"));
                num.erase(num.find_last_not_of(" \t") + 1);

                if (num.empty()) {
                    throw std::invalid_argument("Empty array index element");
                }

                try {
                    indices.push_back(std::stoll(num));
                } catch (const std::exception&) {
                    throw std::invalid_argument("Invalid array index: " + num);
                }
                start = i + 1;
            }
        }

        if (indices.empty()) {
            throw std::invalid_argument("Empty array index");
        }

        return indices;
    }

    /**
     * @brief User-defined literal for slice creation
     */
    inline namespace literals {
        inline Slice operator""_s(const char* str, size_t) {
            return SliceParser::parse(str);
        }

        /**
         * @brief Create MultiIndex from NumPy-like string syntax
         * Examples:
         *   ":,2,:"_idx      -> all, index 2, all
         *   "3::,5:"_idx     -> slice from 3 with default step, slice from 5
         *   "None,3"_idx     -> newaxis, index 3
         *   "...,2"_idx      -> ellipsis, index 2
         *   "[1,2,3],::-1"_idx -> array indexing, reverse slice
         */
        inline MultiIndex operator""_idx(const char* str, size_t) {
            return IndexParser::parse(str);
        }
    }

    inline constexpr All _{};
    inline constexpr NewAxis N{};
    inline constexpr Ellipsis E{};

    template<typename... Args>
    inline MultiIndex idx(Args... args) {
        return MultiIndex{args...};
    }
} // namespace fem::numeric

#endif //NUMERIC_SLICE_BASE_H

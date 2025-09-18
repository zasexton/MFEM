/*
 * Matrix view types extracted from matrix.h to compartmentalize code.
 */

#pragma once

#ifndef NUMERIC_MATRIX_VIEW_H
#define NUMERIC_MATRIX_VIEW_H

#include <cstddef>
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include "../base/numeric_base.h"
#include "../base/expression_base.h"
#include "../base/slice_base.h"

namespace fem::numeric {

template<typename T> class MatrixTransposeView;

/**
 * Lightweight 2D view over matrix data (non-owning)
 */
template<typename T>
class MatrixView : public ExpressionBase<MatrixView<T>> {
public:
    using value_type = std::remove_const_t<T>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<T>, const value_type*, value_type*>;
    using reference = std::conditional_t<std::is_const_v<T>, const value_type&, value_type&>;
    using const_reference = const value_type&;

    MatrixView() noexcept
        : data_(nullptr), rows_(0), cols_(0), row_stride_(0), col_stride_(0) {}

    MatrixView(pointer data,
               size_type rows,
               size_type cols,
               difference_type row_stride,
               difference_type col_stride) noexcept
        : data_(data),
          rows_(rows),
          cols_(cols),
          row_stride_(row_stride),
          col_stride_(col_stride) {}

    pointer data() const noexcept { return data_; }
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    difference_type row_stride() const noexcept { return row_stride_; }
    difference_type col_stride() const noexcept { return col_stride_; }
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }
    size_type size() const noexcept { return rows_ * cols_; }
    Shape shape() const noexcept { return Shape{rows_, cols_}; }

    template<typename U = T>
    std::enable_if_t<!std::is_const_v<U>, value_type&>
    operator()(size_type i, size_type j) {
        assert(i < rows_ && j < cols_);
        return data_[static_cast<difference_type>(i) * row_stride_ +
                     static_cast<difference_type>(j) * col_stride_];
    }

    const_reference operator()(size_type i, size_type j) const {
        assert(i < rows_ && j < cols_);
        return data_[static_cast<difference_type>(i) * row_stride_ +
                     static_cast<difference_type>(j) * col_stride_];
    }

    template<typename U = T>
    std::enable_if_t<!std::is_const_v<U>, value_type&>
    at(size_type i, size_type j) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("MatrixView index out of range");
        }
        return (*this)(i, j);
    }

    const_reference at(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("MatrixView index out of range");
        }
        return (*this)(i, j);
    }

    MatrixView submatrix(size_type row_start, size_type row_end,
                         size_type col_start, size_type col_end) const {
        if (row_start > row_end || col_start > col_end ||
            row_end > rows_ || col_end > cols_) {
            throw std::out_of_range("MatrixView submatrix range out of bounds");
        }
        const size_type row_count = row_end - row_start;
        const size_type col_count = col_end - col_start;
        if (row_count == 0 || col_count == 0) {
            return MatrixView(data_, 0, 0, row_stride_, col_stride_);
        }
        pointer base_ptr = data_ + static_cast<difference_type>(row_start) * row_stride_ +
                           static_cast<difference_type>(col_start) * col_stride_;
        return MatrixView(base_ptr, row_count, col_count, row_stride_, col_stride_);
    }

    MatrixView operator()(const Slice& row_slice, const Slice& col_slice) const {
        const auto row_norm = row_slice.normalize(rows_);
        const auto col_norm = col_slice.normalize(cols_);
        const auto row_count = row_slice.count(rows_);
        const auto col_count = col_slice.count(cols_);

        if (row_count == 0 || col_count == 0) {
            return MatrixView(data_, 0, 0,
                              row_stride_ * row_norm.step(),
                              col_stride_ * col_norm.step());
        }

        const auto row_offset = static_cast<difference_type>(row_norm.start());
        const auto col_offset = static_cast<difference_type>(col_norm.start());

        pointer base_ptr = data_ + row_offset * row_stride_ + col_offset * col_stride_;
        return MatrixView(base_ptr,
                          row_count,
                          col_count,
                          row_stride_ * row_norm.step(),
                          col_stride_ * col_norm.step());
    }

    MatrixTransposeView<T> transpose() const;

    bool is_parallelizable() const noexcept { return true; }
    bool is_vectorizable() const noexcept { return col_stride_ == 1; }
    size_t complexity() const noexcept { return size(); }

    template<typename U>
    auto eval(size_type index) const {
        if (cols_ == 0) {
            return U{};
        }
        const size_type row = index / cols_;
        const size_type col = index % cols_;
        return static_cast<U>((*this)(row, col));
    }

    template<typename Container>
    void eval_to(Container& result) const {
        using result_type = typename Container::value_type;
        if (result.shape() != shape()) {
            result.resize(shape());
        }
        for (size_type i = 0; i < size(); ++i) {
            result.data()[i] = static_cast<result_type>(eval<result_type>(i));
        }
    }

private:
    pointer data_;
    size_type rows_;
    size_type cols_;
    difference_type row_stride_;
    difference_type col_stride_;
};

/**
 * Transposed matrix view implemented lazily
 */
template<typename T>
class MatrixTransposeView : public ExpressionBase<MatrixTransposeView<T>> {
public:
    using view_type = MatrixView<T>;
    using value_type = typename view_type::value_type;
    using size_type = typename view_type::size_type;
    using difference_type = typename view_type::difference_type;
    using pointer = typename view_type::pointer;
    using reference = typename view_type::reference;
    using const_reference = typename view_type::const_reference;

    MatrixTransposeView() = default;

    explicit MatrixTransposeView(view_type base) : base_(std::move(base)) {}

    MatrixTransposeView(pointer data,
                        size_type rows,
                        size_type cols,
                        difference_type row_stride,
                        difference_type col_stride)
        : base_(data, cols, rows, col_stride, row_stride) {}

    pointer data() const noexcept { return base_.data(); }
    size_type rows() const noexcept { return base_.cols(); }
    size_type cols() const noexcept { return base_.rows(); }
    difference_type row_stride() const noexcept { return base_.col_stride(); }
    difference_type col_stride() const noexcept { return base_.row_stride(); }
    bool empty() const noexcept { return base_.empty(); }
    size_type size() const noexcept { return rows() * cols(); }
    Shape shape() const noexcept { return Shape{rows(), cols()}; }

    template<typename U = T>
    std::enable_if_t<!std::is_const_v<U>, value_type&>
    operator()(size_type i, size_type j) {
        assert(i < rows() && j < cols());
        return base_(j, i);
    }

    const_reference operator()(size_type i, size_type j) const {
        assert(i < rows() && j < cols());
        return base_(j, i);
    }

    template<typename U = T>
    std::enable_if_t<!std::is_const_v<U>, value_type&>
    at(size_type i, size_type j) {
        if (i >= rows() || j >= cols()) {
            throw std::out_of_range("MatrixTransposeView index out of range");
        }
        return (*this)(i, j);
    }

    const_reference at(size_type i, size_type j) const {
        if (i >= rows() || j >= cols()) {
            throw std::out_of_range("MatrixTransposeView index out of range");
        }
        return (*this)(i, j);
    }

    MatrixTransposeView operator()(const Slice& row_slice, const Slice& col_slice) const {
        auto sliced = base_.operator()(col_slice, row_slice);
        return MatrixTransposeView(std::move(sliced));
    }

    view_type transpose() const { return base_; }

    bool is_parallelizable() const noexcept { return true; }
    bool is_vectorizable() const noexcept { return base_.col_stride() == 1; }
    size_t complexity() const noexcept { return size(); }

    template<typename U>
    auto eval(size_type index) const {
        if (cols() == 0) {
            return U{};
        }
        const size_type row = index / cols();
        const size_type col = index % cols();
        return static_cast<U>((*this)(row, col));
    }

    template<typename Container>
    void eval_to(Container& result) const {
        using result_type = typename Container::value_type;
        if (result.shape() != shape()) {
            result.resize(shape());
        }
        for (size_type i = 0; i < size(); ++i) {
            result.data()[i] = static_cast<result_type>(eval<result_type>(i));
        }
    }

private:
    view_type base_;
};

template<typename T>
inline MatrixTransposeView<T> MatrixView<T>::transpose() const {
    return MatrixTransposeView<T>(*this);
}

} // namespace fem::numeric

#endif // NUMERIC_MATRIX_VIEW_H

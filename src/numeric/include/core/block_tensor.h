#pragma once

#ifndef NUMERIC_BLOCK_TENSOR_H
#define NUMERIC_BLOCK_TENSOR_H

#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <string>
#include <map>
#include <array>

#include "../base/numeric_base.h"
#include "../base/container_base.h"
#include "../base/expression_base.h"
#include "../base/traits_base.h"
#include "tensor.h"

namespace fem::numeric {

template<typename T, size_t Rank>
class BlockTensor : public ContainerBase<BlockTensor<T, Rank>, T, DynamicStorage<T>>,
                    public ExpressionBase<BlockTensor<T, Rank>> {
public:
    static_assert(Rank > 0, "BlockTensor rank must be >= 1");
    static_assert(StorableType<T>, "T must satisfy StorableType concept");

    using base_type = ContainerBase<BlockTensor<T, Rank>, T, DynamicStorage<T>>;
    using expression_base = ExpressionBase<BlockTensor<T, Rank>>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using tensor_type = Tensor<T, Rank>;
    using tensor_ptr = std::unique_ptr<tensor_type>;
    using shape_type = std::array<size_type, Rank>;
    using axis_block_info = std::pair<std::string, size_type>; // name, size
    using block_structure = std::vector<axis_block_info>;

private:
    std::array<block_structure, Rank> axes_{};
    std::array<std::map<std::string, size_type>, Rank> name_to_index_{};
    std::array<size_type, Rank> axis_block_counts_{};
    shape_type total_dims_{};

    std::vector<tensor_ptr> blocks_;

    size_type linear_block_index(const std::array<size_type, Rank>& bidx) const {
        size_type idx = 0;
        size_type stride = 1;
        for (size_t ax = Rank; ax-- > 0;) {
            if (bidx[ax] >= axis_block_counts_[ax]) {
                throw std::out_of_range("BlockTensor block index out of range");
            }
            idx += bidx[ax] * stride;
            stride *= axis_block_counts_[ax];
        }
        return idx;
    }

    void rebuild_from_axes() {
        for (size_t ax = 0; ax < Rank; ++ax) {
            name_to_index_[ax].clear();
            axis_block_counts_[ax] = axes_[ax].size();
            for (size_type i = 0; i < axes_[ax].size(); ++i) {
                const auto& name = axes_[ax][i].first;
                if (!name_to_index_[ax].emplace(name, i).second) {
                    throw std::invalid_argument("Duplicate block name on axis " + std::to_string(ax) + ": " + name);
                }
            }
        }

        for (size_t ax = 0; ax < Rank; ++ax) {
            size_type sum = 0;
            for (const auto& bi : axes_[ax]) sum += bi.second;
            total_dims_[ax] = sum;
        }

        size_type total_blocks = 1;
        for (size_t ax = 0; ax < Rank; ++ax) {
            total_blocks *= axis_block_counts_[ax] == 0 ? 1 : axis_block_counts_[ax];
        }
        blocks_.clear();
        blocks_.resize(total_blocks);

        std::vector<size_t> dims_vec(Rank);
        for (size_t i = 0; i < Rank; ++i) dims_vec[i] = total_dims_[i];
        this->shape_ = Shape(dims_vec);
    }

    shape_type block_dims_at(const std::array<size_type, Rank>& bidx) const {
        shape_type dims{};
        for (size_t ax = 0; ax < Rank; ++ax) {
            dims[ax] = axes_[ax][bidx[ax]].second;
        }
        return dims;
    }

    tensor_type& ensure_block(const std::array<size_type, Rank>& bidx) {
        auto linear = linear_block_index(bidx);
        auto& ptr = blocks_[linear];
        if (!ptr) {
            auto dims = block_dims_at(bidx);
            ptr = std::make_unique<tensor_type>(dims);
            ptr->zero();
        }
        return *ptr;
    }

public:
    BlockTensor() { this->shape_ = Shape(std::vector<size_t>(Rank, 0)); }

    BlockTensor(const std::array<std::vector<std::string>, Rank>& axis_names,
                const std::array<std::vector<size_type>, Rank>& axis_sizes) {
        set_block_structure(axis_names, axis_sizes);
    }

    BlockTensor(const BlockTensor& other)
        : base_type(other), axes_(other.axes_), name_to_index_(other.name_to_index_),
          axis_block_counts_(other.axis_block_counts_), total_dims_(other.total_dims_),
          blocks_() {
        blocks_.resize(other.blocks_.size());
        for (size_t i = 0; i < other.blocks_.size(); ++i) {
            if (other.blocks_[i]) {
                blocks_[i] = std::make_unique<tensor_type>(*other.blocks_[i]);
            }
        }
    }

    BlockTensor(BlockTensor&&) noexcept = default;
    BlockTensor& operator=(BlockTensor&&) noexcept = default;

    BlockTensor& operator=(const BlockTensor& other) {
        if (this != &other) {
            base_type::operator=(other);
            axes_ = other.axes_;
            name_to_index_ = other.name_to_index_;
            axis_block_counts_ = other.axis_block_counts_;
            total_dims_ = other.total_dims_;
            blocks_.clear();
            blocks_.resize(other.blocks_.size());
            for (size_t i = 0; i < other.blocks_.size(); ++i) {
                if (other.blocks_[i]) {
                    blocks_[i] = std::make_unique<tensor_type>(*other.blocks_[i]);
                }
            }
        }
        return *this;
    }

    ~BlockTensor() = default;

    void set_block_structure(const std::array<std::vector<std::string>, Rank>& axis_names,
                             const std::array<std::vector<size_type>, Rank>& axis_sizes) {
        for (size_t ax = 0; ax < Rank; ++ax) {
            if (axis_names[ax].size() != axis_sizes[ax].size()) {
                throw std::invalid_argument("Axis names and sizes length mismatch at axis " + std::to_string(ax));
            }
            axes_[ax].clear();
            axes_[ax].reserve(axis_names[ax].size());
            for (size_t i = 0; i < axis_names[ax].size(); ++i) {
                axes_[ax].emplace_back(axis_names[ax][i], axis_sizes[ax][i]);
            }
        }
        rebuild_from_axes();
    }

    void set_block_structure(const std::vector<std::string>& names,
                             const std::vector<size_type>& sizes) {
        std::array<std::vector<std::string>, Rank> axis_names;
        std::array<std::vector<size_type>, Rank> axis_sizes;
        for (size_t ax = 0; ax < Rank; ++ax) {
            axis_names[ax] = names;
            axis_sizes[ax] = sizes;
        }
        set_block_structure(axis_names, axis_sizes);
    }

    size_type num_axes() const noexcept { return Rank; }
    size_type num_blocks(size_t axis) const noexcept { return axis_block_counts_[axis]; }
    const block_structure& axis(size_t ax) const { return axes_[ax]; }

    const shape_type& dims() const noexcept { return total_dims_; }

    Shape shape() const noexcept {
        std::vector<size_t> dims_vec(Rank);
        for (size_t i = 0; i < Rank; ++i) dims_vec[i] = total_dims_[i];
        return Shape(dims_vec);
    }

    bool empty() const noexcept {
        for (auto d : total_dims_) if (d != 0) return false;
        return true;
    }

    tensor_type& block(const std::array<size_type, Rank>& bidx) {
        return ensure_block(bidx);
    }

    const tensor_type& block(const std::array<size_type, Rank>& bidx) const {
        auto linear = linear_block_index(bidx);
        const auto& ptr = blocks_[linear];
        if (!ptr) {
            throw std::runtime_error("Requested BlockTensor block is not allocated");
        }
        return *ptr;
    }

    tensor_type& block_by_names(const std::array<std::string, Rank>& names) {
        std::array<size_type, Rank> bidx{};
        for (size_t ax = 0; ax < Rank; ++ax) {
            auto it = name_to_index_[ax].find(names[ax]);
            if (it == name_to_index_[ax].end()) {
                throw std::invalid_argument("Unknown block name on axis " + std::to_string(ax) + ": " + names[ax]);
            }
            bidx[ax] = it->second;
        }
        return ensure_block(bidx);
    }

    const tensor_type& block_by_names(const std::array<std::string, Rank>& names) const {
        std::array<size_type, Rank> bidx{};
        for (size_t ax = 0; ax < Rank; ++ax) {
            auto it = name_to_index_[ax].find(names[ax]);
            if (it == name_to_index_[ax].end()) {
                throw std::invalid_argument("Unknown block name on axis " + std::to_string(ax) + ": " + names[ax]);
            }
            bidx[ax] = it->second;
        }
        return block(bidx);
    }

    bool has_block(const std::array<size_type, Rank>& bidx) const {
        auto linear = linear_block_index(bidx);
        return blocks_[linear] != nullptr;
    }

    bool has_block_by_names(const std::array<std::string, Rank>& names) const {
        std::array<size_type, Rank> bidx{};
        for (size_t ax = 0; ax < Rank; ++ax) {
            auto it = name_to_index_[ax].find(names[ax]);
            if (it == name_to_index_[ax].end()) return false;
            bidx[ax] = it->second;
        }
        return has_block(bidx);
    }

    using scalar_type = typename numeric_traits<T>::scalar_type;

    BlockTensor& operator+=(const BlockTensor& other) {
        ensure_same_structure(other);
        for (size_type lin = 0; lin < blocks_.size(); ++lin) {
            if (other.blocks_[lin]) {
                if (!blocks_[lin]) {
                    blocks_[lin] = std::make_unique<tensor_type>(*other.blocks_[lin]);
                } else {
                    add_in_place(*blocks_[lin], *other.blocks_[lin]);
                }
            }
        }
        return *this;
    }

    BlockTensor& operator-=(const BlockTensor& other) {
        ensure_same_structure(other);
        for (size_type lin = 0; lin < blocks_.size(); ++lin) {
            if (other.blocks_[lin]) {
                if (!blocks_[lin]) {
                    auto tmp = std::make_unique<tensor_type>(*other.blocks_[lin]);
                    negate_in_place(*tmp);
                    blocks_[lin] = std::move(tmp);
                } else {
                    sub_in_place(*blocks_[lin], *other.blocks_[lin]);
                }
            }
        }
        return *this;
    }

    BlockTensor& operator*=(const T& scalar) {
        for (auto& blk : blocks_) {
            if (blk) {
                scale_in_place(*blk, scalar);
            }
        }
        return *this;
    }

    BlockTensor& operator/=(const T& scalar) {
        for (auto& blk : blocks_) {
            if (blk) {
                scale_in_place(*blk, T{1} / scalar);
            }
        }
        return *this;
    }

    scalar_type norm2() const {
        scalar_type sum{0};
        for (const auto& blk : blocks_) {
            if (blk) {
                const auto& b = *blk;
                const size_type n = b.size();
                for (size_type i = 0; i < n; ++i) {
                    if constexpr (is_complex_number_v<T>) {
                        auto v = b.data()[i];
                        sum += std::real(v * std::conj(v));
                    } else {
                        auto v = b.data()[i];
                        sum += v * v;
                    }
                }
            }
        }
        return std::sqrt(sum);
    }

    scalar_type max_norm() const {
        scalar_type maxv{0};
        for (const auto& blk : blocks_) {
            if (blk) {
                const auto& b = *blk;
                const size_type n = b.size();
                for (size_type i = 0; i < n; ++i) {
                    maxv = std::max(maxv, static_cast<scalar_type>(std::abs(b.data()[i])));
                }
            }
        }
        return maxv;
    }

    friend std::ostream& operator<<(std::ostream& os, const BlockTensor& bt) {
        os << "BlockTensor<Rank=" << Rank << "> (";
        for (size_t ax = 0; ax < Rank; ++ax) {
            os << bt.total_dims_[ax];
            if (ax + 1 < Rank) os << "x";
        }
        os << ") with per-axis blocks:\n";

        for (size_t ax = 0; ax < Rank; ++ax) {
            os << " axis " << ax << ": ";
            for (size_t i = 0; i < bt.axes_[ax].size(); ++i) {
                os << bt.axes_[ax][i].first << "(" << bt.axes_[ax][i].second << ")";
                if (i + 1 < bt.axes_[ax].size()) os << ", ";
            }
            os << "\n";
        }

        os << " existing blocks: ";
        size_type existing = 0;
        for (const auto& p : bt.blocks_) existing += (p != nullptr);
        os << existing << "/" << bt.blocks_.size() << "\n";
        return os;
    }

private:
    void ensure_same_structure(const BlockTensor& other) const {
        for (size_t ax = 0; ax < Rank; ++ax) {
            if (axes_[ax].size() != other.axes_[ax].size()) {
                throw std::invalid_argument("Mismatched block counts on axis " + std::to_string(ax));
            }
            for (size_t i = 0; i < axes_[ax].size(); ++i) {
                if (axes_[ax][i].second != other.axes_[ax][i].second ||
                    axes_[ax][i].first  != other.axes_[ax][i].first) {
                    throw std::invalid_argument("Mismatched block structure on axis " + std::to_string(ax));
                }
            }
        }
    }

    static void add_in_place(tensor_type& a, const tensor_type& b) {
        if (a.dims() != b.dims()) {
            throw std::invalid_argument("BlockTensor add: block shape mismatch");
        }
        const size_type n = a.size();
        for (size_type i = 0; i < n; ++i) {
            a.data()[i] += b.data()[i];
        }
    }

    static void sub_in_place(tensor_type& a, const tensor_type& b) {
        if (a.dims() != b.dims()) {
            throw std::invalid_argument("BlockTensor sub: block shape mismatch");
        }
        const size_type n = a.size();
        for (size_type i = 0; i < n; ++i) {
            a.data()[i] -= b.data()[i];
        }
    }

    static void scale_in_place(tensor_type& a, const T& s) {
        const size_type n = a.size();
        for (size_type i = 0; i < n; ++i) {
            a.data()[i] *= s;
        }
    }

    static void negate_in_place(tensor_type& a) {
        const size_type n = a.size();
        for (size_type i = 0; i < n; ++i) {
            a.data()[i] = -a.data()[i];
        }
    }
};

template<typename T, size_t Rank>
auto operator+(const BlockTensor<T, Rank>& A, const BlockTensor<T, Rank>& B) {
    BlockTensor<T, Rank> R(A);
    R += B;
    return R;
}

template<typename T, size_t Rank>
auto operator-(const BlockTensor<T, Rank>& A, const BlockTensor<T, Rank>& B) {
    BlockTensor<T, Rank> R(A);
    R -= B;
    return R;
}

template<typename T, size_t Rank, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockTensor<T, Rank>>))
auto operator*(const Scalar& s, const BlockTensor<T, Rank>& A) {
    BlockTensor<T, Rank> R(A);
    R *= static_cast<T>(s);
    return R;
}

template<typename T, size_t Rank, typename Scalar>
    requires ((std::is_arithmetic_v<Scalar> || is_complex_number_v<Scalar> || is_dual_number_v<Scalar>) &&
              (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, BlockTensor<T, Rank>>))
auto operator*(const BlockTensor<T, Rank>& A, const Scalar& s) {
    return s * A;
}

} // namespace fem::numeric

#endif // NUMERIC_BLOCK_TENSOR_H


#pragma once

#ifndef NUMERIC_CORE_SPARSE_MATRIX_H
#define NUMERIC_CORE_SPARSE_MATRIX_H

#include <vector>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace numeric::core {

template<typename T>
class SparseMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;

    enum class StorageFormat { COO, CSR, CSC };

    struct Entry { size_type row, col; T value; };

    // Iterator over nonzeros
    using iterator = typename std::vector<Entry>::iterator;
    using const_iterator = typename std::vector<Entry>::const_iterator;

    SparseMatrix() : rows_(0), cols_(0) {}
    SparseMatrix(size_type r, size_type c) : rows_(r), cols_(c) {}

    SparseMatrix(size_type r, size_type c,
                 const std::vector<std::tuple<size_type,size_type,T>>& triplets,
                 StorageFormat /*fmt*/ = StorageFormat::COO)
        : rows_(r), cols_(c) {
        for (auto& t : triplets) {
            auto [i,j,v] = t;
            if (v != T{}) insert_or_accumulate(i,j,v);
        }
        rebuild_rows();
    }

    // Basic properties
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type nnz() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }

    // Element access (const)
    T operator()(size_type i, size_type j) const {
        bounds_check(i,j);
        auto it = index_.find(key(i,j));
        if (it == index_.end()) return T{};
        return entries_[it->second].value;
    }

    // Set value (insert/update, remove on zero)
    void set(size_type i, size_type j, const T& v) {
        bounds_check(i,j);
        auto k = key(i,j);
        auto it = index_.find(k);
        if (v == T{}) {
            if (it != index_.end()) {
                remove_at(it->second);
            }
            return;
        }
        if (it == index_.end()) {
            Entry e{i,j,v};
            index_[k] = entries_.size();
            entries_.push_back(e);
            rows_adjacency_dirty_ = true;
        } else {
            entries_[it->second].value = v;
        }
    }

    // Iteration over nonzeros
    iterator begin() { return entries_.begin(); }
    iterator end() { return entries_.end(); }
    const_iterator begin() const { return entries_.begin(); }
    const_iterator end() const { return entries_.end(); }

    // Arithmetic
    SparseMatrix operator+(const SparseMatrix& other) const {
        ensure_same_shape(other);
        SparseMatrix out(rows_, cols_);
        out.entries_.reserve(entries_.size() + other.entries_.size());
        for (const auto& e : entries_) out.set(e.row, e.col, e.value);
        for (const auto& e : other.entries_) out.set(e.row, e.col, out(e.row,e.col) + e.value);
        out.rebuild_rows();
        return out;
    }

    SparseMatrix operator-(const SparseMatrix& other) const {
        ensure_same_shape(other);
        SparseMatrix out(rows_, cols_);
        for (const auto& e : entries_) out.set(e.row, e.col, e.value);
        for (const auto& e : other.entries_) out.set(e.row, e.col, out(e.row,e.col) - e.value);
        out.rebuild_rows();
        return out;
    }

    friend SparseMatrix operator*(const SparseMatrix& A, const T& s) {
        SparseMatrix out(A.rows_, A.cols_);
        out.entries_.reserve(A.entries_.size());
        for (auto e : A.entries_) if (e.value != T{}) out.entries_.push_back({e.row,e.col,e.value*s});
        out.rebuild_index();
        out.rebuild_rows();
        return out;
    }
    friend SparseMatrix operator*(const T& s, const SparseMatrix& A) { return A * s; }

    // Conjugate (for complex); for real types returns copy
    SparseMatrix conjugate() const {
        SparseMatrix out(rows_, cols_);
        for (const auto& e : entries_) {
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<long double>>) {
                out.set(e.row, e.col, std::conj(e.value));
            } else {
                out.set(e.row, e.col, e.value);
            }
        }
        out.rebuild_rows();
        return out;
    }

    // Approximate memory usage (bytes)
    std::size_t memory_usage() const noexcept {
        return entries_.capacity() * sizeof(Entry) + index_.size() * (sizeof(std::uint64_t) + sizeof(size_type));
    }

    // Extract submatrix by selected rows/cols
    SparseMatrix submatrix(const std::vector<size_type>& rows_sel,
                           const std::vector<size_type>& cols_sel) const {
        SparseMatrix out(rows_sel.size(), cols_sel.size());
        std::unordered_map<size_type,size_type> rmap, cmap;
        for (size_type i=0;i<rows_sel.size();++i) rmap[rows_sel[i]]=i;
        for (size_type j=0;j<cols_sel.size();++j) cmap[cols_sel[j]]=j;
        for (const auto& e : entries_) {
            auto rit = rmap.find(e.row); if (rit==rmap.end()) continue;
            auto cit = cmap.find(e.col); if (cit==cmap.end()) continue;
            out.set(rit->second, cit->second, e.value);
        }
        out.rebuild_rows();
        return out;
    }

    // Row sums
    std::vector<T> row_sum() const {
        std::vector<T> sums(rows_, T{});
        for (const auto& e : entries_) sums[e.row] += e.value;
        return sums;
    }

    // Multiply by dense std::vector
    template<typename U>
    auto operator*(const std::vector<U>& x) const {
        using R = decltype(T{} * U{});
        if (cols_ != x.size()) throw std::invalid_argument("Vector size mismatch");
        std::vector<R> y(rows_, R{});
        for (const auto& e : entries_) y[e.row] += static_cast<R>(e.value) * static_cast<R>(x[e.col]);
        return y;
    }

    // Sparse matrix-matrix multiplication
    SparseMatrix operator*(const SparseMatrix& B) const {
        if (cols_ != B.rows_) throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        SparseMatrix C(rows_, B.cols_);
        B.ensure_rows_adjacency();
        ensure_rows_adjacency();
        for (size_type i = 0; i < rows_; ++i) {
            for (auto [k, aik] : rows_adj_[i]) {
                if (k >= B.rows_) continue;
                for (auto [j, bkj] : B.rows_adj_[k]) {
                    auto cur = C(i,j);
                    C.set(i,j, cur + aik * bkj);
                }
            }
        }
        C.rebuild_rows();
        return C;
    }

    // Transpose
    SparseMatrix transpose() const {
        SparseMatrix Tm(cols_, rows_);
        for (const auto& e : entries_) Tm.set(e.col, e.row, e.value);
        Tm.rebuild_rows();
        return Tm;
    }

    // Norms
    auto frobenius_norm() const {
        using R = decltype(std::abs(T{}));
        long double sum = 0;
        for (const auto& e : entries_) {
            auto a = static_cast<long double>(std::abs(e.value));
            sum += a * a;
        }
        return static_cast<R>(std::sqrt(sum));
    }
    // One-norm (max column sum)
    auto one_norm() const {
        using R = decltype(std::abs(T{}));
        std::vector<long double> colsum(cols_, 0);
        for (const auto& e : entries_) colsum[e.col] += std::abs(e.value);
        long double m = 0; for (auto v : colsum) m = std::max(m,v);
        return static_cast<R>(m);
    }
    // Infinity-norm (max row sum)
    auto infinity_norm() const {
        using R = decltype(std::abs(T{}));
        std::vector<long double> rowsum(rows_, 0);
        for (const auto& e : entries_) rowsum[e.row] += std::abs(e.value);
        long double m = 0; for (auto v : rowsum) m = std::max(m,v);
        return static_cast<R>(m);
    }

private:
    size_type rows_, cols_;
    std::vector<Entry> entries_;
    std::unordered_map<std::uint64_t, size_type> index_;

    // adjacency by rows for faster ops
    mutable bool rows_adjacency_dirty_{true};
    mutable std::vector<std::vector<std::pair<size_type,T>>> rows_adj_; 

    static std::uint64_t key(size_type i, size_type j) {
        return (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(j);
    }
    void bounds_check(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("SparseMatrix index out of range");
    }
    void insert_or_accumulate(size_type i, size_type j, const T& v) {
        auto k = key(i,j);
        auto it = index_.find(k);
        if (it == index_.end()) {
            index_[k] = entries_.size();
            entries_.push_back({i,j,v});
        } else {
            entries_[it->second].value += v;
        }
        rows_adjacency_dirty_ = true;
    }
    void remove_at(size_type idx) {
        auto e = entries_[idx];
        auto last_idx = entries_.size()-1;
        if (idx != last_idx) {
            entries_[idx] = entries_[last_idx];
            index_[key(entries_[idx].row, entries_[idx].col)] = idx;
        }
        entries_.pop_back();
        index_.erase(key(e.row,e.col));
        rows_adjacency_dirty_ = true;
    }
    void rebuild_index() {
        index_.clear();
        for (size_type p=0;p<entries_.size();++p) index_[key(entries_[p].row, entries_[p].col)] = p;
    }
    void ensure_same_shape(const SparseMatrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument("SparseMatrix shape mismatch");
    }
    void ensure_rows_adjacency() const {
        if (!rows_adjacency_dirty_) return;
        rows_adj_.clear(); rows_adj_.resize(rows_);
        for (const auto& e : entries_) rows_adj_[e.row].emplace_back(e.col, e.value);
        rows_adjacency_dirty_ = false;
    }
    void rebuild_rows() const { ensure_rows_adjacency(); }
};

} // namespace numeric::core

#endif // NUMERIC_CORE_SPARSE_MATRIX_H

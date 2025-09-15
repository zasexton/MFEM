#pragma once

#ifndef NUMERIC_CORE_SPARSE_TENSOR_H
#define NUMERIC_CORE_SPARSE_TENSOR_H

#include <array>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>
#include <complex>
#include <type_traits>

namespace numeric::core {

template<typename T, std::size_t Rank>
class SparseTensor {
public:
    using value_type = T;
    using size_type = std::size_t;
    using index_type = std::array<size_t, Rank>;

    struct Entry { index_type index; T value; };
    using iterator = typename std::vector<Entry>::iterator;
    using const_iterator = typename std::vector<Entry>::const_iterator;

    SparseTensor() { shape_.fill(0); }
    explicit SparseTensor(const index_type& shape) : shape_(shape) {}
    SparseTensor(const index_type& shape, const std::vector<std::pair<index_type,T>>& entries)
        : shape_(shape) {
        for (auto& p : entries) if (p.second != T{}) set(p.first, p.second);
    }

    // basic properties
    constexpr size_type rank() const noexcept { return Rank; }
    size_type size(size_t dim) const { return shape_.at(dim); }
    bool empty() const noexcept { return entries_.empty(); }
    size_type nnz() const noexcept { return entries_.size(); }

    // Approximate memory usage (bytes)
    std::size_t memory_usage() const noexcept {
        // Rough estimate: entries vector storage + map node storage
        return entries_.capacity() * sizeof(Entry)
             + map_.size() * (sizeof(std::string) + sizeof(size_type));
    }

    // access
    template<typename... Indices>
    T operator()(Indices... idxs) const {
        static_assert(sizeof...(idxs) == Rank, "wrong number of indices");
        index_type idx{static_cast<size_t>(idxs)...};
        bounds_check(idx);
        auto it = map_.find(key(idx));
        if (it == map_.end()) return T{};
        return entries_[it->second].value;
    }

    // Direct indexed access using full index type
    T value_at(const index_type& idx) const {
        bounds_check(idx);
        auto it = map_.find(key(idx));
        if (it == map_.end()) return T{};
        return entries_[it->second].value;
    }

    void set(const index_type& idx, const T& v) {
        bounds_check(idx);
        auto k = key(idx);
        auto it = map_.find(k);
        if (v == T{}) {
            if (it != map_.end()) remove_at(it->second);
            return;
        }
        if (it == map_.end()) {
            map_[k] = entries_.size();
            entries_.push_back({idx, v});
        } else {
            entries_[it->second].value = v;
        }
    }

    // iterators
    iterator begin() { return entries_.begin(); }
    iterator end() { return entries_.end(); }
    const_iterator begin() const { return entries_.begin(); }
    const_iterator end() const { return entries_.end(); }

    // arithmetic
    SparseTensor operator+(const SparseTensor& other) const {
        ensure_same_shape(other);
        SparseTensor out(shape_);
        for (auto& e : entries_) out.set(e.index, e.value);
        for (auto& e : other.entries_) out.set(e.index, out.value_at(e.index) + e.value);
        return out;
    }
    SparseTensor operator-(const SparseTensor& other) const {
        ensure_same_shape(other);
        SparseTensor out(shape_);
        for (auto& e : entries_) out.set(e.index, e.value);
        for (auto& e : other.entries_) out.set(e.index, out.value_at(e.index) - e.value);
        return out;
    }
    friend SparseTensor operator*(const SparseTensor& A, const T& s) {
        SparseTensor out(A.shape_);
        for (auto& e : A.entries_) if (e.value != T{}) out.set(e.index, e.value * s);
        return out;
    }
    friend SparseTensor operator*(const T& s, const SparseTensor& A) { return A * s; }

    // slice along dimension dim at fixed index -> rank-1 tensor
    auto slice(size_t dim, size_t fixed) const {
        using OutIndex = std::array<size_t, Rank-1>;
        SparseTensor<T, Rank-1> out(reduced_shape(dim));
        for (const auto& e : entries_) {
            if (e.index[dim] == fixed) {
                OutIndex idx{}; size_t p=0; for (size_t i=0;i<Rank;++i) if (i!=dim) idx[p++] = e.index[i];
                out.set(idx, e.value);
            }
        }
        return out;
    }

    // permute dimensions
    SparseTensor permute(const index_type& order) const {
        SparseTensor out(permute_shape(order));
        for (const auto& e : entries_) {
            index_type idx; for (size_t i=0;i<Rank;++i) idx[i] = e.index[order[i]];
            out.set(idx, e.value);
        }
        return out;
    }

    // conjugate (for complex)
    SparseTensor conjugate() const {
        SparseTensor out(shape_);
        for (const auto& e : entries_) {
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<long double>>) {
                out.set(e.index, std::conj(e.value));
            } else {
                out.set(e.index, e.value);
            }
        }
        return out;
    }

    // contraction along A.dimA and B.dimB -> rank A+B-2
    template<std::size_t R2>
    auto contract(const SparseTensor<T, R2>& B, size_t dimA, size_t dimB) const {
        constexpr std::size_t ROut = Rank + R2 - 2;
        std::array<size_t, ROut> out_shape{}; size_t p=0;
        for (size_t i=0;i<Rank;++i) if (i!=dimA) out_shape[p++] = shape_[i];
        for (size_t j=0;j<R2;++j) if (j!=dimB) out_shape[p++] = B.size(j);
        SparseTensor<T, ROut> out(out_shape);
        for (const auto& ea : entries_) {
            for (const auto& eb : B.entries_) {
                if (ea.index[dimA] == eb.index[dimB]) {
                    std::array<size_t, ROut> idx{}; size_t q=0;
                    for (size_t i=0;i<Rank;++i) if (i!=dimA) idx[q++] = ea.index[i];
                    for (size_t j=0;j<R2;++j) if (j!=dimB) idx[q++] = eb.index[j];
                    out.set(idx, out.value_at(idx) + ea.value * eb.value);
                }
            }
        }
        return out;
    }

    // outer product -> rank A+B
    template<std::size_t R2>
    auto outer_product(const SparseTensor<T, R2>& B) const {
        std::array<size_t, Rank+R2> out_shape{};
        for (size_t i=0;i<Rank;++i) out_shape[i] = shape_[i];
        for (size_t j=0;j<R2;++j) out_shape[Rank+j] = B.size(j);
        SparseTensor<T, Rank+R2> out(out_shape);
        for (const auto& ea : entries_) {
            for (const auto& eb : B.entries_) {
                std::array<size_t, Rank+R2> idx{};
                for (size_t i=0;i<Rank;++i) idx[i] = ea.index[i];
                for (size_t j=0;j<R2;++j) idx[Rank+j] = eb.index[j];
                out.set(idx, out.value_at(idx) + ea.value * eb.value);
            }
        }
        return out;
    }

    // Norms: Frobenius, L1 (sum abs), Linf (max abs)
    auto frobenius_norm() const {
        long double sum = 0;
        for (const auto& e : entries_) {
            auto a = static_cast<long double>(std::abs(e.value));
            sum += a * a;
        }
        return static_cast<decltype(std::abs(T{}))>(std::sqrt(sum));
    }
    auto one_norm() const {
        long double s = 0; for (const auto& e : entries_) s += std::abs(e.value); return static_cast<decltype(std::abs(T{}))>(s);
    }
    auto infinity_norm() const {
        long double m = 0; for (const auto& e : entries_) m = std::max<long double>(m, std::abs(e.value)); return static_cast<decltype(std::abs(T{}))>(m);
    }

private:
    index_type shape_{};
    std::vector<Entry> entries_;
    std::unordered_map<std::string, size_type> map_;

    static std::string key(const index_type& idx) {
        std::string s; s.reserve(Rank*12);
        for (size_t i=0;i<Rank;++i) { if (i) s.push_back(','); s += std::to_string(idx[i]); }
        return s;
    }
    T operator()(const index_type& idx) const {
        auto it = map_.find(key(idx));
        return (it==map_.end()) ? T{} : entries_[it->second].value;
    }
    void remove_at(size_type pos) {
        auto old = entries_[pos].index;
        auto last = entries_.size()-1;
        if (pos != last) {
            entries_[pos] = entries_[last];
            map_[key(entries_[pos].index)] = pos;
        }
        entries_.pop_back();
        map_.erase(key(old));
    }
    void bounds_check(const index_type& idx) const {
        for (size_t i=0;i<Rank;++i) if (idx[i] >= shape_[i]) throw std::out_of_range("SparseTensor index out of range");
    }
    void ensure_same_shape(const SparseTensor& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("SparseTensor shape mismatch");
    }
    template<std::size_t R=Rank>
    std::array<size_t, R-1> reduced_shape(size_t dim) const {
        std::array<size_t, R-1> sh{}; size_t p=0; for (size_t i=0;i<R;++i) if (i!=dim) sh[p++]=shape_[i]; return sh;
    }
    index_type permute_shape(const index_type& order) const {
        index_type sh{}; for (size_t i=0;i<Rank;++i) sh[i]=shape_[order[i]]; return sh;
    }
};

} // namespace numeric::core

#endif // NUMERIC_CORE_SPARSE_TENSOR_H

#pragma once

#ifndef CORE_MEMORY_FLAT_MAP_H
#define CORE_MEMORY_FLAT_MAP_H

#include <vector>
#include <algorithm>
#include <utility>
#include <functional>

#include <config/config.h>

#include "memory_resource.h"

namespace fem::core::memory {

// A simple sorted-vector-based associative container.
// Provides a subset of std::map interface with much better cache locality for
// small to medium sizes. Keys must be strictly totally ordered.
template<class Key, class T, class Compare = std::less<Key>, class Alloc = polymorphic_allocator<std::pair<Key, T>>>
class flat_map {
public:
    using value_type = std::pair<Key, T>;
    using container_type = std::vector<value_type, Alloc>;
    using size_type = typename container_type::size_type;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    explicit flat_map(memory_resource* mr = default_resource(), Compare comp = {})
        : data_(Alloc(mr)), comp_(std::move(comp)) {}

    explicit flat_map(const Alloc& alloc, Compare comp = {})
        : data_(alloc), comp_(std::move(comp)) {}

    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] iterator begin() noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data_.begin(); }
    [[nodiscard]] iterator end() noexcept { return data_.end(); }
    [[nodiscard]] const_iterator end() const noexcept { return data_.end(); }

    T& operator[](const Key& k) { return insert_or_assign(k, T{}).first->second; }
    T& operator[](Key&& k) { return insert_or_assign(std::move(k), T{}).first->second; }

    std::pair<iterator, bool> insert(value_type v) {
        auto it = lower_bound(v.first);
        if (it != data_.end() && equal_key(it->first, v.first)) return {it, false};
        return {data_.insert(it, std::move(v)), true};
    }

    template<class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        value_type v(std::forward<Args>(args)...);
        return insert(std::move(v));
    }

    std::pair<iterator, bool> insert_or_assign(Key key, T value) {
        auto it = lower_bound(key);
        if (it != data_.end() && equal_key(it->first, key)) {
            it->second = std::move(value);
            return {it, false};
        }
        return {data_.insert(it, value_type(std::move(key), std::move(value))), true};
    }

    iterator find(const Key& k) { auto it = lower_bound(k); return (it != data_.end() && equal_key(it->first, k)) ? it : data_.end(); }
    const_iterator find(const Key& k) const { auto it = lower_bound(k); return (it != data_.end() && equal_key(it->first, k)) ? it : data_.end(); }

    size_type erase(const Key& k) {
        auto it = find(k);
        if (it == data_.end()) return 0;
        data_.erase(it);
        return 1;
    }

    void clear() noexcept { data_.clear(); }
    void reserve(size_type n) { data_.reserve(n); }

private:
    container_type data_;
    Compare comp_{};

    iterator lower_bound(const Key& k) {
        return std::lower_bound(data_.begin(), data_.end(), k, [this](const value_type& a, const Key& b) { return comp_(a.first, b); });
    }
    const_iterator lower_bound(const Key& k) const {
        return std::lower_bound(data_.begin(), data_.end(), k, [this](const value_type& a, const Key& b) { return comp_(a.first, b); });
    }
    bool equal_key(const Key& a, const Key& b) const { return !comp_(a, b) && !comp_(b, a); }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_FLAT_MAP_H


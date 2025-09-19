#pragma once

#ifndef CORE_MEMORY_FLAT_SET_H
#define CORE_MEMORY_FLAT_SET_H

#include <vector>
#include <algorithm>
#include <functional>

#include <config/config.h>

#include "memory_resource.h"

namespace fem::core::memory {

// A simple sorted-vector-based set with good cache locality for small/medium sizes.
template<class Key, class Compare = std::less<Key>, class Alloc = polymorphic_allocator<Key>>
class flat_set {
public:
    using value_type = Key;
    using container_type = std::vector<Key, Alloc>;
    using size_type = typename container_type::size_type;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    explicit flat_set(memory_resource* mr = default_resource(), Compare comp = {})
        : data_(Alloc(mr)), comp_(std::move(comp)) {}

    explicit flat_set(const Alloc& alloc, Compare comp = {})
        : data_(alloc), comp_(std::move(comp)) {}

    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] iterator begin() noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data_.begin(); }
    [[nodiscard]] iterator end() noexcept { return data_.end(); }
    [[nodiscard]] const_iterator end() const noexcept { return data_.end(); }

    std::pair<iterator, bool> insert(Key key) {
        auto it = lower_bound(key);
        if (it != data_.end() && equal_key(*it, key)) return {it, false};
        return {data_.insert(it, std::move(key)), true};
    }

    template<class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        Key k(std::forward<Args>(args)...);
        return insert(std::move(k));
    }

    iterator find(const Key& k) { auto it = lower_bound(k); return (it != data_.end() && equal_key(*it, k)) ? it : data_.end(); }
    const_iterator find(const Key& k) const { auto it = lower_bound(k); return (it != data_.end() && equal_key(*it, k)) ? it : data_.end(); }

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
        return std::lower_bound(data_.begin(), data_.end(), k, [this](const Key& a, const Key& b) { return comp_(a, b); });
    }
    const_iterator lower_bound(const Key& k) const {
        return std::lower_bound(data_.begin(), data_.end(), k, [this](const Key& a, const Key& b) { return comp_(a, b); });
    }
    bool equal_key(const Key& a, const Key& b) const { return !comp_(a, b) && !comp_(b, a); }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_FLAT_SET_H


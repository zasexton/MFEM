#pragma once

#ifndef CORE_MEMORY_BUMP_VECTOR_H
#define CORE_MEMORY_BUMP_VECTOR_H

#include <cstddef>
#include <type_traits>
#include <new>
#include <utility>

#include <config/config.h>

#include "memory_resource.h"

namespace fem::core::memory {

// bump_vector: append-only vector optimized for fast growth and bulk reset.
// Does not support erase in the middle; only push_back/emplace_back/clear.
template<class T, class Alloc = polymorphic_allocator<T>>
class bump_vector {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using allocator_type = Alloc;

    explicit bump_vector(memory_resource* mr = default_resource()) : alloc_(mr) {}
    explicit bump_vector(const Alloc& alloc) : alloc_(alloc) {}

    bump_vector(const bump_vector&) = delete;
    bump_vector& operator=(const bump_vector&) = delete;

    bump_vector(bump_vector&& other) noexcept : alloc_(std::move(other.alloc_)) { move_from(std::move(other)); }
    bump_vector& operator=(bump_vector&& other) noexcept {
        if (this != &other) { destroy_all(); deallocate(); alloc_ = std::move(other.alloc_); move_from(std::move(other)); }
        return *this;
    }
    ~bump_vector() { destroy_all(); deallocate(); }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type capacity() const noexcept { return cap_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] reference operator[](size_type i) { return data_[i]; }
    [[nodiscard]] const_reference operator[](size_type i) const { return data_[i]; }

    void reserve(size_type n) { if (n > cap_) reallocate(grow_to(n)); }

    template<class... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == cap_) reallocate(grow_to(size_ + 1));
        ::new ((void*)(data_ + size_)) T(std::forward<Args>(args)...);
        return data_[size_++];
    }
    void push_back(const T& v) { (void)emplace_back(v); }
    void push_back(T&& v) { (void)emplace_back(std::move(v)); }

    void clear() noexcept { destroy_all(); size_ = 0; }

private:
    allocator_type alloc_;
    pointer data_{nullptr};
    size_type size_{0};
    size_type cap_{0};

    size_type grow_to(size_type min_needed) const {
        size_type cap = (cap_ == 0 ? 8 : cap_);
        while (cap < min_needed) cap = cap + (cap >> 1) + 1;
        return cap;
    }

    void reallocate(size_type new_cap) {
        pointer new_mem = std::allocator_traits<Alloc>::allocate(alloc_, new_cap);
        // move-construct
        for (size_type i = 0; i < size_; ++i) ::new ((void*)(new_mem + i)) T(std::move_if_noexcept(data_[i]));
        destroy_all();
        deallocate();
        data_ = new_mem; cap_ = new_cap;
    }
    void destroy_all() noexcept { if constexpr (!std::is_trivially_destructible_v<T>) for (size_type i = 0; i < size_; ++i) data_[i].~T(); }
    void deallocate() noexcept { if (data_) { std::allocator_traits<Alloc>::deallocate(alloc_, data_, cap_); data_ = nullptr; cap_ = 0; } }
    void move_from(bump_vector&& o) noexcept { data_ = o.data_; size_ = o.size_; cap_ = o.cap_; o.data_ = nullptr; o.size_ = o.cap_ = 0; }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_BUMP_VECTOR_H

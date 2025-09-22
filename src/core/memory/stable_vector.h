#pragma once

#ifndef CORE_MEMORY_STABLE_VECTOR_H
#define CORE_MEMORY_STABLE_VECTOR_H

#include <cstddef>
#include <vector>
#include <type_traits>
#include <utility>
#include <new>
#include <functional>
#include <iterator>

#include <config/config.h>
#include <config/debug.h>

#include "memory_resource.h"

namespace fem::core::memory {

#if CORE_ENABLE_PROFILING || CORE_ENABLE_LOGGING
    #ifndef CORE_MEMORY_ENABLE_TELEMETRY
        #define CORE_MEMORY_ENABLE_TELEMETRY 1
    #endif
#else
    #ifndef CORE_MEMORY_ENABLE_TELEMETRY
        #define CORE_MEMORY_ENABLE_TELEMETRY 0
    #endif
#endif

// stable_vector: pointer/reference-stable growth using fixed-size blocks.
// Not contiguous; indices remain valid, and addresses of elements never change after creation.
template<class T, std::size_t BlockBytes = fem::config::PAGE_SIZE, class Alloc = polymorphic_allocator<T>>
class stable_vector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using allocator_type = Alloc;

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t blocks_allocated = 0;
        std::size_t constructed_elements = 0;
        std::size_t destroyed_elements = 0;
        std::size_t peak_size = 0;
        std::size_t capacity = 0; // in elements
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    static constexpr size_type elems_per_block() noexcept {
        return BlockBytes / (sizeof(T) == 0 ? 1 : sizeof(T)) ? (BlockBytes / (sizeof(T) == 0 ? 1 : sizeof(T))) : 1;
    }

    explicit stable_vector(memory_resource* mr = default_resource())
        : alloc_(mr) {}

    explicit stable_vector(const allocator_type& alloc)
        : alloc_(alloc) {}

    stable_vector(const stable_vector& other)
        : alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)) {
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i) emplace_back(other[i]);
    }

    stable_vector(stable_vector&& other) noexcept
        : alloc_(std::move(other.alloc_)), blocks_(std::move(other.blocks_)), size_(other.size_) {
        other.size_ = 0;
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = other.telemetry_;
#endif
    }

    ~stable_vector() {
        clear();
        deallocate_all_blocks();
    }

    stable_vector& operator=(const stable_vector& other) {
        if (this == &other) return *this;
        clear();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            alloc_ = other.alloc_;
        }
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i) emplace_back(other[i]);
        return *this;
    }

    stable_vector& operator=(stable_vector&& other) noexcept {
        if (this == &other) return *this;
        clear();
        deallocate_all_blocks();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
        }
        blocks_ = std::move(other.blocks_);
        size_ = other.size_;
        other.size_ = 0;
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = other.telemetry_;
#endif
        return *this;
    }

    // Capacity and access
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_type capacity() const noexcept { return blocks_.size() * elems_per_block(); }

    [[nodiscard]] reference operator[](size_type i) { FEM_ASSERT(i < size_); return *ptr_at(i); }
    [[nodiscard]] const_reference operator[](size_type i) const { FEM_ASSERT(i < size_); return *ptr_at(i); }

    [[nodiscard]] reference back() { FEM_ASSERT(size_ > 0); return (*this)[size_ - 1]; }
    [[nodiscard]] const_reference back() const { FEM_ASSERT(size_ > 0); return (*this)[size_ - 1]; }

    void clear() noexcept {
        // Destroy all constructed elements
        for (size_type i = size_; i > 0; --i) {
            destroy_at(i - 1);
        }
        size_ = 0;
    }

    void reserve(size_type n) {
        if (n <= capacity()) return;
        size_type need_blocks = (n + elems_per_block() - 1) / elems_per_block();
        while (blocks_.size() < need_blocks) allocate_block();
    }

    void shrink_to_fit() {
        const size_type needed_blocks = (size_ + elems_per_block() - 1) / elems_per_block();
        while (blocks_.size() > needed_blocks) deallocate_last_block();
    }

    template<class... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity()) allocate_block();
        T* p = ptr_at(size_);
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
        ++size_;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.constructed_elements;
        if (size_ > telemetry_.peak_size) telemetry_.peak_size = size_;
#endif
        return *p;
    }

    void push_back(const T& v) { (void)emplace_back(v); }
    void push_back(T&& v) { (void)emplace_back(std::move(v)); }

    void pop_back() noexcept {
        FEM_ASSERT(size_ > 0);
        destroy_at(size_ - 1);
        --size_;
    }

    // Simple index iterator that preserves pointer stability semantics
    class iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::forward_iterator_tag;

        iterator(stable_vector* v, size_type i) : v_(v), i_(i) {}
        reference operator*() const { return (*v_)[i_]; }
        pointer operator->() const { return &(*v_)[i_]; }
        iterator& operator++() { ++i_; return *this; }
        iterator operator++(int) { iterator tmp(*this); ++(*this); return tmp; }
        bool operator==(const iterator& o) const { return v_ == o.v_ && i_ == o.i_; }
        bool operator!=(const iterator& o) const { return !(*this == o); }

    private:
        stable_vector* v_{};
        size_type i_{};
    };

    class const_iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator(const stable_vector* v, size_type i) : v_(v), i_(i) {}
        reference operator*() const { return (*v_)[i_]; }
        pointer operator->() const { return &(*v_)[i_]; }
        const_iterator& operator++() { ++i_; return *this; }
        const_iterator operator++(int) { const_iterator tmp(*this); ++(*this); return tmp; }
        bool operator==(const const_iterator& o) const { return v_ == o.v_ && i_ == o.i_; }
        bool operator!=(const const_iterator& o) const { return !(*this == o); }

    private:
        const stable_vector* v_{};
        size_type i_{};
    };

    [[nodiscard]] iterator begin() { return iterator(this, 0); }
    [[nodiscard]] iterator end() { return iterator(this, size_); }
    [[nodiscard]] const_iterator begin() const { return const_iterator(this, 0); }
    [[nodiscard]] const_iterator end() const { return const_iterator(this, size_); }

#if CORE_MEMORY_ENABLE_TELEMETRY
    [[nodiscard]] const telemetry_t& telemetry() const noexcept { return telemetry_; }
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    allocator_type alloc_;
    std::vector<T*> blocks_{}; // each block has elems_per_block() elements
    size_type size_{0};

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif

    void allocate_block() {
        T* block = std::allocator_traits<Alloc>::allocate(alloc_, elems_per_block());
        blocks_.push_back(block);
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.blocks_allocated;
        telemetry_.capacity = capacity();
        if (telemetry_cb_) telemetry_cb_("allocate_block", telemetry_);
#endif
    }

    void deallocate_last_block() noexcept {
        if (blocks_.empty()) return;
        T* block = blocks_.back();
        blocks_.pop_back();
        std::allocator_traits<Alloc>::deallocate(alloc_, block, elems_per_block());
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_.capacity = capacity();
        if (telemetry_cb_) telemetry_cb_("deallocate_block", telemetry_);
#endif
    }

    void deallocate_all_blocks() noexcept {
        for (T* block : blocks_) {
            std::allocator_traits<Alloc>::deallocate(alloc_, block, elems_per_block());
        }
        blocks_.clear();
    }

    [[nodiscard]] T* ptr_at(size_type index) const noexcept {
        size_type bi = index / elems_per_block();
        size_type off = index % elems_per_block();
        return blocks_[bi] + off;
    }

    void destroy_at(size_type index) noexcept {
        T* p = ptr_at(index);
        if constexpr (!std::is_trivially_destructible_v<T>) p->~T();
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.destroyed_elements;
#endif
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_STABLE_VECTOR_H

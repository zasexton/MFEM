#pragma once

#ifndef CORE_MEMORY_SMALL_VECTOR_H
#define CORE_MEMORY_SMALL_VECTOR_H

#include <cstddef>
#include <type_traits>
#include <new>
#include <utility>
#include <initializer_list>
#include <functional>

#include <config/config.h>
#include <config/debug.h>

#include "aligned_storage.h"
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

// SmallVector: stores up to InlineCapacity elements in-place, spills to heap beyond.
template<class T, std::size_t InlineCapacity = 8, class Alloc = polymorphic_allocator<T>>
class small_vector {
public:
    using value_type = T;
    using allocator_type = Alloc;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t heap_allocs = 0;
        std::size_t reallocs = 0;
        std::size_t moved_elements = 0;
        std::size_t constructed_elements = 0;
        std::size_t destroyed_elements = 0;
        std::size_t spills = 0;     // inline -> heap transitions
        std::size_t peak_size = 0;
        std::size_t peak_capacity = InlineCapacity;
    };

    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    small_vector() noexcept : alloc_(default_resource()) {}

    explicit small_vector(memory_resource* mr) noexcept : alloc_(mr) {}

    explicit small_vector(const allocator_type& alloc) noexcept : alloc_(alloc) {}

    small_vector(std::initializer_list<T> init, memory_resource* mr = default_resource())
        : alloc_(mr) {
        reserve(init.size());
        for (const auto& v : init) emplace_back(v);
    }

    small_vector(const small_vector& other)
        : alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)) {
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i) emplace_back(other[i]);
    }

    small_vector(small_vector&& other) noexcept
        : alloc_(std::move(other.alloc_)) {
        move_from(std::move(other));
    }

    small_vector& operator=(const small_vector& other) {
        if (this == &other) return *this;
        clear();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            alloc_ = other.alloc_;
        }
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i) emplace_back(other[i]);
        return *this;
    }

    small_vector& operator=(small_vector&& other) noexcept {
        if (this == &other) return *this;
        clear();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
        }
        destroy_heap();
        move_from(std::move(other));
        return *this;
    }

    ~small_vector() { clear(); destroy_heap(); }

    // Capacity & access
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type capacity() const noexcept { return is_heap() ? heap_cap_ : InlineCapacity; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] pointer data() noexcept { return is_heap() ? heap_ : inline_ptr(); }
    [[nodiscard]] const_pointer data() const noexcept { return is_heap() ? heap_ : inline_ptr(); }
    [[nodiscard]] reference operator[](size_type i) { FEM_ASSERT(i < size_); return data()[i]; }
    [[nodiscard]] const_reference operator[](size_type i) const { FEM_ASSERT(i < size_); return data()[i]; }

    [[nodiscard]] reference back() { FEM_ASSERT(size_ > 0); return (*this)[size_ - 1]; }
    [[nodiscard]] const_reference back() const { FEM_ASSERT(size_ > 0); return (*this)[size_ - 1]; }

    // Modifiers
    void clear() noexcept {
        // Destroy elements in reverse order
        while (size_ > 0) {
            pop_back();
        }
    }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity()) return;
        reallocate(grow_to(new_cap));
    }

    void shrink_to_fit() {
        if (!is_heap()) return; // already minimal
        if (size_ <= InlineCapacity) {
            // move back to inline
            T* src = heap_;
            T* dst = inline_ptr();
            move_range(src, dst, size_);
            dealloc_heap();
            heap_ = nullptr;
            heap_cap_ = 0;
        } else {
            // reallocate to exact fit
            reallocate(size_);
        }
    }

    template<class... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity()) reallocate(grow_to(size_ + 1));
        T* p = data() + size_;
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
        T* p = data() + (size_ - 1);
        if constexpr (!std::is_trivially_destructible_v<T>) p->~T();
        --size_;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.destroyed_elements;
#endif
    }

    // Iterators
    [[nodiscard]] pointer begin() noexcept { return data(); }
    [[nodiscard]] const_pointer begin() const noexcept { return data(); }
    [[nodiscard]] pointer end() noexcept { return data() + size_; }
    [[nodiscard]] const_pointer end() const noexcept { return data() + size_; }

#if CORE_MEMORY_ENABLE_TELEMETRY
    [[nodiscard]] const telemetry_t& telemetry() const noexcept { return telemetry_; }
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    using InlineBuffer = AlignedBuffer<InlineCapacity * sizeof(T), alignof(T)>;

    allocator_type alloc_;
    InlineBuffer inline_storage_{};
    T* heap_{nullptr};
    size_type heap_cap_{0};
    size_type size_{0};

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif

    [[nodiscard]] bool is_heap() const noexcept { return heap_ != nullptr; }
    [[nodiscard]] T* inline_ptr() noexcept { return std::launder(reinterpret_cast<T*>(inline_storage_.data)); }
    [[nodiscard]] const T* inline_ptr() const noexcept { return std::launder(reinterpret_cast<const T*>(inline_storage_.data)); }

    static size_type grow_to(size_type min_needed) {
        // Growth policy: double capacity until >= min_needed.
        size_type cap = InlineCapacity;
        if (min_needed <= cap) return cap;
        // Start from current power-of-two >= InlineCapacity
        cap = InlineCapacity ? InlineCapacity : 1;
        while (cap < min_needed) cap = cap + (cap >> 1) + 1; // ~1.5x growth
        return cap;
    }

    void reallocate(size_type new_cap) {
        FEM_ASSERT(new_cap >= size_);
        T* new_mem = std::allocator_traits<Alloc>::allocate(alloc_, new_cap);
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.heap_allocs;
        ++telemetry_.reallocs;
        if (!is_heap()) ++telemetry_.spills;
        if (new_cap > telemetry_.peak_capacity) telemetry_.peak_capacity = new_cap;
        if (telemetry_cb_) telemetry_cb_("reallocate", telemetry_);
#endif
        // Move-construct existing elements into new storage
        move_range(data(), new_mem, size_);

        // Destroy old and swap in
        destroy_heap();
        heap_ = new_mem;
        heap_cap_ = new_cap;
    }

    void destroy_heap() noexcept {
        if (is_heap()) {
            dealloc_heap();
            heap_ = nullptr;
            heap_cap_ = 0;
        }
    }

    void dealloc_heap() noexcept {
        if (heap_) {
            std::allocator_traits<Alloc>::deallocate(alloc_, heap_, heap_cap_);
        }
    }

    void move_from(small_vector&& other) noexcept {
        if (other.is_heap()) {
            heap_ = other.heap_;
            heap_cap_ = other.heap_cap_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.heap_cap_ = 0;
            other.size_ = 0;
        } else {
            // move inline elements
            for (size_type i = 0; i < other.size_; ++i) {
                ::new (static_cast<void*>(inline_ptr() + i)) T(std::move(other[i]));
            }
            size_ = other.size_;
            other.clear();
        }
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = other.telemetry_;
#endif
    }

    void move_range(T* src, T* dst, size_type count) {
        if constexpr (std::is_nothrow_move_constructible_v<T> || !std::is_copy_constructible_v<T>) {
            for (size_type i = 0; i < count; ++i) {
                ::new (static_cast<void*>(dst + i)) T(std::move(src[i]));
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                ::new (static_cast<void*>(dst + i)) T(src[i]);
            }
        }
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_.moved_elements += count;
#endif
        // Destroy source if it was inline or temporary; caller handles deallocation
        if (src == inline_ptr()) {
            for (size_type i = 0; i < count; ++i) {
                if constexpr (!std::is_trivially_destructible_v<T>) (src + i)->~T();
            }
        }
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_SMALL_VECTOR_H


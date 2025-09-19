#pragma once

#ifndef CORE_MEMORY_RING_BUFFER_H
#define CORE_MEMORY_RING_BUFFER_H

#include <cstddef>
#include <utility>
#include <type_traits>
#include <new>
#include <initializer_list>
#include <functional>

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

// A bounded, non-thread-safe ring buffer with allocator support.
// Provides strong exception safety for push/emplace operations.
template<class T, class Alloc = polymorphic_allocator<T>>
class ring_buffer {
public:
    using value_type = T;
    using allocator_type = Alloc;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t pushes = 0;
        std::size_t pops = 0;
        std::size_t drops = 0;      // push attempts when full
        std::size_t heap_allocs = 0; // buffer allocations
        std::size_t peak_size = 0;
        std::size_t capacity = 0;
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    explicit ring_buffer(size_type capacity,
                         memory_resource* mr = default_resource())
        : alloc_(mr), cap_(capacity) {
        allocate_storage(capacity);
    }

    ring_buffer(size_type capacity, const allocator_type& alloc)
        : alloc_(alloc), cap_(capacity) {
        allocate_storage(capacity);
    }

    ring_buffer(const ring_buffer& other)
        : alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)) {
        allocate_storage(other.cap_);
        for (size_type i = 0; i < other.size_; ++i) {
            new (ptr_at(i)) T(other[i]);
            ++size_;
        }
    }

    ring_buffer(ring_buffer&& other) noexcept
        : alloc_(std::move(other.alloc_)) {
        move_from(std::move(other));
    }

    ring_buffer& operator=(const ring_buffer& other) {
        if (this == &other) return *this;
        clear();
        if constexpr (!std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            // keep alloc_
        } else {
            alloc_ = other.alloc_;
        }
        if (cap_ != other.cap_) {
            deallocate_storage();
            allocate_storage(other.cap_);
        }
        for (size_type i = 0; i < other.size_; ++i) {
            new (ptr_at(i)) T(other[i]);
            ++size_;
        }
        return *this;
    }

    ring_buffer& operator=(ring_buffer&& other) noexcept {
        if (this == &other) return *this;
        clear();
        deallocate_storage();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
        }
        move_from(std::move(other));
        return *this;
    }

    ~ring_buffer() { clear(); deallocate_storage(); }

    // Capacity
    [[nodiscard]] size_type capacity() const noexcept { return cap_; }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool full() const noexcept { return size_ == cap_; }

    // Element access (front only; this structure is queue-like)
    [[nodiscard]] reference front() { FEM_ASSERT(size_ > 0); return *ptr_at(0); }
    [[nodiscard]] const_reference front() const { FEM_ASSERT(size_ > 0); return *ptr_at(0); }

    // Operations
    bool push(const T& v) { return emplace(v); }
    bool push(T&& v) { return emplace(std::move(v)); }

    template<class... Args>
    bool emplace(Args&&... args) {
        if (full()) {
#if CORE_MEMORY_ENABLE_TELEMETRY
            ++telemetry_.drops;
            if (telemetry_cb_) telemetry_cb_("drop", telemetry_);
#endif
            return false;
        }
        new (ptr_at(size_)) T(std::forward<Args>(args)...);
        ++size_;
        if (++tail_ == cap_) tail_ = 0;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.pushes;
        if (size_ > telemetry_.peak_size) telemetry_.peak_size = size_;
#endif
        return true;
    }

    bool pop() {
        if (empty()) return false;
        // Destroy front element
        ptr_at(0)->~T();
        --size_;
        if (++head_ == cap_) head_ = 0;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.pops;
#endif
        return true;
    }

    bool pop(T& out) {
        if (empty()) return false;
        out = std::move(front());
        return pop();
    }

    // Clear without releasing storage
    void clear() noexcept {
        while (!empty()) { pop(); }
        head_ = tail_ = 0;
    }

    // Iteration support (logical order from front to back)
    template<class F>
    void for_each(F&& f) {
        for (size_type i = 0; i < size_; ++i) f((*this)[i]);
    }

    // Indexing (0 = front)
    [[nodiscard]] reference operator[](size_type i) { FEM_ASSERT(i < size_); return *ptr_at(i); }
    [[nodiscard]] const_reference operator[](size_type i) const { FEM_ASSERT(i < size_); return *ptr_at(i); }

#if CORE_MEMORY_ENABLE_TELEMETRY
    [[nodiscard]] const telemetry_t& telemetry() const noexcept { return telemetry_; }
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    allocator_type alloc_;
    T* storage_{nullptr}; // raw storage for cap_ Ts
    size_type cap_{0};
    size_type head_{0};
    size_type tail_{0};
    size_type size_{0};

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif

    void allocate_storage(size_type capacity) {
        FEM_ASSERT(capacity > 0);
        storage_ = std::allocator_traits<Alloc>::allocate(alloc_, capacity);
        cap_ = capacity;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.heap_allocs;
        telemetry_.capacity = capacity;
        if (telemetry_cb_) telemetry_cb_("allocate", telemetry_);
#endif
        head_ = tail_ = size_ = 0;
    }

    void deallocate_storage() noexcept {
        if (!storage_) return;
        if constexpr (true) {
#if CORE_MEMORY_ENABLE_TELEMETRY
            if (telemetry_cb_) telemetry_cb_("deallocate", telemetry_);
#endif
        }
        std::allocator_traits<Alloc>::deallocate(alloc_, storage_, cap_);
        storage_ = nullptr;
        cap_ = head_ = tail_ = size_ = 0;
    }

    // pointer to i-th logical element (0 = front)
    T* ptr_at(size_type logical_index) const noexcept {
        size_type physical = head_ + logical_index;
        if (physical >= cap_) physical -= cap_;
        return storage_ + physical;
    }

    void move_from(ring_buffer&& other) noexcept {
        storage_ = other.storage_;
        cap_ = other.cap_;
        head_ = other.head_;
        tail_ = other.tail_;
        size_ = other.size_;
        other.storage_ = nullptr;
        other.cap_ = other.head_ = other.tail_ = other.size_ = 0;
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = other.telemetry_;
#endif
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_RING_BUFFER_H

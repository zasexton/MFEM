#pragma once

#ifndef CORE_MEMORY_STACK_ALLOCATOR_H
#define CORE_MEMORY_STACK_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <new>

#include <config/config.h>
#include <config/debug.h>

#include "aligned_storage.h"

namespace fem::core::memory {

// stack_storage: simple bump allocator over a fixed-size buffer.
// Not thread-safe. Deallocation is LIFO-only (optional); typically you reset.
template<std::size_t CapacityBytes, std::size_t Alignment = alignof(std::max_align_t)>
class stack_storage {
public:
    stack_storage() noexcept : begin_(buffer_.data), cur_(buffer_.data), end_(buffer_.data + CapacityBytes) {}

    stack_storage(const stack_storage&) = delete;
    stack_storage& operator=(const stack_storage&) = delete;

    [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment = Alignment) {
        FEM_ASSERT(bytes > 0);
        FEM_ASSERT(is_power_of_two(alignment));
        std::byte* aligned = static_cast<std::byte*>(align_up(cur_, alignment));
        if (aligned + bytes > end_) {
            // out of local storage
            throw std::bad_alloc{};
        }
        cur_ = aligned + bytes;
        return aligned;
    }

    // Optional LIFO deallocate; only frees if p is the top-most allocation.
    void deallocate(void* p, std::size_t bytes, std::size_t alignment = Alignment) noexcept {
        auto* ptr = static_cast<std::byte*>(p);
        std::byte* aligned_end = static_cast<std::byte*>(align_up(ptr, alignment)) + bytes;
        if (aligned_end == cur_) cur_ = static_cast<std::byte*>(p);
    }

    void reset() noexcept { cur_ = begin_; }

    [[nodiscard]] std::size_t used() const noexcept { return static_cast<std::size_t>(cur_ - begin_); }
    [[nodiscard]] std::size_t capacity() const noexcept { return CapacityBytes; }

    struct Marker { std::byte* cur; };
    [[nodiscard]] Marker mark() const noexcept { return Marker{cur_}; }
    void rewind(Marker m) noexcept { if (m.cur >= begin_ && m.cur <= end_) cur_ = m.cur; }

private:
    AlignedBuffer<CapacityBytes, Alignment> buffer_{};
    std::byte* begin_{nullptr};
    std::byte* cur_{nullptr};
    std::byte* end_{nullptr};
};

// StackAllocator: std::allocator-compatible adapter over stack_storage.
template<class T, std::size_t CapacityBytes, std::size_t Alignment = alignof(T)>
class StackAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = StackAllocator<U, CapacityBytes, (Alignment > alignof(U) ? Alignment : alignof(U))>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    explicit StackAllocator(stack_storage<CapacityBytes, Alignment>* storage = nullptr) noexcept : storage_(storage) {}

    template<class U, std::size_t C, std::size_t A>
    explicit StackAllocator(const StackAllocator<U, C, A>& other) noexcept : storage_(other.storage()) {}

    [[nodiscard]] pointer allocate(size_type n) {
        FEM_ASSERT(storage_ != nullptr);
        void* p = storage_->allocate(n * sizeof(T), Alignment);
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (!p || !storage_) return;
        storage_->deallocate(p, n * sizeof(T), Alignment);
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    [[nodiscard]] auto storage() const noexcept { return storage_; }

    template<class U, std::size_t C, std::size_t A>
    bool operator==(const StackAllocator<U, C, A>& other) const noexcept { return storage_ == other.storage(); }
    template<class U, std::size_t C, std::size_t A>
    bool operator!=(const StackAllocator<U, C, A>& other) const noexcept { return !(*this == other); }

private:
    template<class, std::size_t, std::size_t> friend class StackAllocator;
    stack_storage<CapacityBytes, Alignment>* storage_{nullptr};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_STACK_ALLOCATOR_H


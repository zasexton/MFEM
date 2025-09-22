#pragma once

#ifndef CORE_MEMORY_STACK_ALLOCATOR_H
#define CORE_MEMORY_STACK_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <new>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "aligned_storage.h"

namespace fem::core::memory {

// stack_storage: simple bump allocator over a fixed-size buffer.
// Not thread-safe. Deallocation is LIFO-only (optional); typically you reset.
template<std::size_t CapacityBytes, std::size_t Alignment = alignof(std::max_align_t)>
class stack_storage {
public:
    stack_storage() noexcept : begin_(buffer_.data), cur_(buffer_.data), end_(buffer_.data + CapacityBytes), last_alloc_ptr_(nullptr), last_alloc_size_(0) {}

    stack_storage(const stack_storage&) = delete;
    stack_storage& operator=(const stack_storage&) = delete;

    [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment = Alignment) {
        FEM_ASSERT(is_power_of_two(alignment));

        // Handle zero-size allocation - return unique non-null pointer
        if (bytes == 0) {
            bytes = 1;
        }

        std::byte* aligned = static_cast<std::byte*>(align_up(cur_, alignment));
        if (aligned + bytes > end_) {
            // out of local storage
            throw std::bad_alloc{};
        }
        last_alloc_ptr_ = aligned;
        last_alloc_size_ = bytes;
        cur_ = aligned + bytes;
        return aligned;
    }

    // Optional LIFO deallocate; only frees if p is the top-most allocation.
    void deallocate(void* p, std::size_t bytes, std::size_t alignment = Alignment) noexcept {
        CORE_UNUSED(alignment);  // Alignment not used in this simple implementation
        if (!p) return;

        // Handle zero-size allocation case
        if (bytes == 0) {
            bytes = 1;
        }

        auto* ptr = static_cast<std::byte*>(p);
        // For LIFO deallocation, check if this is the most recent allocation
        if (ptr == last_alloc_ptr_ && bytes == last_alloc_size_) {
            cur_ = ptr;
            last_alloc_ptr_ = nullptr;
            last_alloc_size_ = 0;
        }
    }

    void reset() noexcept {
        cur_ = begin_;
        last_alloc_ptr_ = nullptr;
        last_alloc_size_ = 0;
    }

    [[nodiscard]] std::size_t used() const noexcept { return static_cast<std::size_t>(cur_ - begin_); }
    [[nodiscard]] std::size_t capacity() const noexcept { return CapacityBytes; }

    struct Marker { std::byte* cur; };
    [[nodiscard]] Marker mark() const noexcept { return Marker{cur_}; }
    void rewind(Marker m) noexcept {
        if (m.cur >= begin_ && m.cur <= end_) {
            cur_ = m.cur;
            // Clear last allocation tracking after rewind
            last_alloc_ptr_ = nullptr;
            last_alloc_size_ = 0;
        }
    }

private:
    AlignedBuffer<CapacityBytes, Alignment> buffer_{};
    std::byte* begin_{nullptr};
    std::byte* cur_{nullptr};
    std::byte* end_{nullptr};
    std::byte* last_alloc_ptr_{nullptr};
    std::size_t last_alloc_size_{0};
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

    template<class U> struct rebind { using other = StackAllocator<U, CapacityBytes, alignof(U)>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    // Constructor from raw storage pointer - storage must have sufficient alignment
    explicit StackAllocator(void* storage = nullptr) noexcept : storage_(storage) {}

    // Constructor from typed stack_storage
    template<std::size_t StorageAlignment>
    explicit StackAllocator(stack_storage<CapacityBytes, StorageAlignment>* storage) noexcept
        : storage_(static_cast<void*>(storage)) {
        static_assert(StorageAlignment >= Alignment,
                      "Storage alignment must be at least as strict as allocator alignment");
    }

    // Rebinding constructor - accepts allocators with different alignments
    template<class U, std::size_t OtherAlignment>
    explicit StackAllocator(const StackAllocator<U, CapacityBytes, OtherAlignment>& other) noexcept
        : storage_(other.storage_) {
        // Can share storage between different alignments
    }

    [[nodiscard]] pointer allocate(size_type n) {
        FEM_ASSERT(storage_ != nullptr);
        // Cast back to the appropriate stack_storage type for allocation
        // We use the maximum alignment to ensure compatibility
        auto* typed_storage = static_cast<stack_storage<CapacityBytes, alignof(std::max_align_t)>*>(storage_);
        void* p = typed_storage->allocate(n * sizeof(T), Alignment);
        return static_cast<pointer>(p);
    }

    [[nodiscard]] fem::core::error::Result<pointer, fem::core::error::ErrorCode>
    try_allocate(size_type n) {
        using fem::core::error::ErrorCode;
        try {
            return allocate(n);
        } catch (const std::bad_alloc&) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::OutOfMemory};
        } catch (...) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::SystemError};
        }
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (!p || !storage_) return;
        // Cast back to the appropriate stack_storage type for deallocation
        auto* typed_storage = static_cast<stack_storage<CapacityBytes, alignof(std::max_align_t)>*>(storage_);
        typed_storage->deallocate(p, n * sizeof(T), Alignment);
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    [[nodiscard]] void* storage() const noexcept { return storage_; }

    template<class U, std::size_t OtherAlignment>
    bool operator==(const StackAllocator<U, CapacityBytes, OtherAlignment>& other) const noexcept {
        return storage_ == other.storage_;
    }
    template<class U, std::size_t OtherAlignment>
    bool operator!=(const StackAllocator<U, CapacityBytes, OtherAlignment>& other) const noexcept {
        return !(*this == other);
    }

private:
    template<class, std::size_t, std::size_t> friend class StackAllocator;
    void* storage_{nullptr};  // Type-erased storage pointer for cross-alignment sharing
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_STACK_ALLOCATOR_H

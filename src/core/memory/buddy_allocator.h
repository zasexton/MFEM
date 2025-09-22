#pragma once

#ifndef CORE_MEMORY_BUDDY_ALLOCATOR_H
#define CORE_MEMORY_BUDDY_ALLOCATOR_H

#include <cstddef>
#include <type_traits>

#include <config/config.h>

#include "memory_resource.h"
#include <core/error/result.h>
#include <core/error/error_code.h>

namespace fem::core::memory {

// Simplified BuddyAllocator<T> interface that rounds allocation size up to the
// next power of two and requests that size from the underlying memory_resource.
// This is a minimal, portable approximation suitable for typical container
// allocations; it does not maintain a full buddy tree (no coalescing). It
// exists to provide the expected API surface from AGENT.md.
template<class T>
class BuddyAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = BuddyAllocator<U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    explicit BuddyAllocator(memory_resource* mr = default_resource()) noexcept : mr_(mr) {}
    template<class U>
    BuddyAllocator(const BuddyAllocator<U>& other) noexcept : mr_(other.resource()) {}

    [[nodiscard]] pointer allocate(size_type n) {
        const std::size_t raw = n * sizeof(T);
        const std::size_t bytes = next_pow2(raw == 0 ? 1 : raw);
        return static_cast<pointer>(mr_->allocate(bytes, alignof(T)));
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (!p) return;
        const std::size_t raw = n * sizeof(T);
        const std::size_t bytes = next_pow2(raw == 0 ? 1 : raw);
        mr_->deallocate(p, bytes, alignof(T));
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

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    [[nodiscard]] memory_resource* resource() const noexcept { return mr_; }

    template<class U>
    bool operator==(const BuddyAllocator<U>& other) const noexcept { return mr_ == other.resource(); }
    template<class U>
    bool operator!=(const BuddyAllocator<U>& other) const noexcept { return !(*this == other); }

private:
    template<class> friend class BuddyAllocator;
    memory_resource* mr_;

    static std::size_t next_pow2(std::size_t x) noexcept {
        if (x <= 1) return 1;
        --x;
        x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
        if constexpr (sizeof(std::size_t) == 8) x |= x >> 32;
        return x + 1;
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_BUDDY_ALLOCATOR_H

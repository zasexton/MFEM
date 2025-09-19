#pragma once

#ifndef CORE_MEMORY_ALIGNED_ALLOCATOR_H
#define CORE_MEMORY_ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <type_traits>

#include <config/config.h>
#include <config/debug.h>

#include "memory_resource.h"

namespace fem::core::memory {

// AlignedAllocator: std::allocator-compatible allocator that requests a
// specific alignment from the underlying memory_resource. The alignment
// defaults to alignof(T) but can be increased via the template parameter.
template<class T, std::size_t Alignment = alignof(T)>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = AlignedAllocator<U, (Alignment > alignof(U) ? Alignment : alignof(U))>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    static constexpr std::size_t alignment = Alignment;

    explicit AlignedAllocator(memory_resource* mr = default_resource()) noexcept : mr_(mr) {
        FEM_ASSERT(mr_ != nullptr);
    }

    template<class U>
    explicit AlignedAllocator(const AlignedAllocator<U, (Alignment > alignof(U) ? Alignment : alignof(U))>& other) noexcept
        : mr_(other.resource()) {}

    [[nodiscard]] pointer allocate(size_type n) {
        const std::size_t bytes = n * sizeof(T);
        return static_cast<pointer>(mr_->allocate(bytes, Alignment));
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (!p) return;
        mr_->deallocate(p, n * sizeof(T), Alignment);
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    [[nodiscard]] memory_resource* resource() const noexcept { return mr_; }

    template<class U, std::size_t A>
    bool operator==(const AlignedAllocator<U, A>& other) const noexcept { return mr_ == other.resource() && Alignment == A; }
    template<class U, std::size_t A>
    bool operator!=(const AlignedAllocator<U, A>& other) const noexcept { return !(*this == other); }

private:
    template<class, std::size_t> friend class AlignedAllocator;
    memory_resource* mr_{nullptr};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_ALIGNED_ALLOCATOR_H


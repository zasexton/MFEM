#pragma once

#ifndef CORE_MEMORY_MALLOC_ALLOCATOR_H
#define CORE_MEMORY_MALLOC_ALLOCATOR_H

#include <cstddef>
#include <new>
#include <type_traits>

#include <config/config.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

namespace fem::core::memory {

// Minimal std::allocator-compatible allocator over global new/delete.
// Provided for completeness and as a baseline when PMR is not desired.
template<class T>
class MallocAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = MallocAllocator<U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type; // stateless

    MallocAllocator() noexcept = default;
    template<class U>
    MallocAllocator(const MallocAllocator<U>&) noexcept {}

    [[nodiscard]] pointer allocate(size_type n) {
#if defined(__cpp_aligned_new)
        return static_cast<pointer>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
#else
        return static_cast<pointer>(::operator new(n * sizeof(T)));
#endif
    }

    [[nodiscard]] fem::core::error::Result<pointer, fem::core::error::ErrorCode>
    try_allocate(size_type n) {
        using fem::core::error::ErrorCode;
        try {
            return allocate(n);
        } catch (const std::bad_alloc&) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::OutOfMemory);
        } catch (...) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::SystemError);
        }
    }

    void deallocate(pointer p, size_type n) noexcept {
#if defined(__cpp_aligned_new)
        ::operator delete(p, n * sizeof(T), std::align_val_t(alignof(T)));
#else
        ::operator delete(p);
#endif
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    template<class U>
    bool operator==(const MallocAllocator<U>&) const noexcept { return true; }
    template<class U>
    bool operator!=(const MallocAllocator<U>&) const noexcept { return false; }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_MALLOC_ALLOCATOR_H

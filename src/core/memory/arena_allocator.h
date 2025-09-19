#pragma once

#ifndef CORE_MEMORY_ARENA_ALLOCATOR_H
#define CORE_MEMORY_ARENA_ALLOCATOR_H

#include <cstddef>
#include <type_traits>
#include <new>
#include <utility>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "arena.h"

namespace fem::core::memory {

// std::allocator-compatible adapter over Arena. Not owning; deallocate is no-op.
template<class T>
class ArenaAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    template<class U> struct rebind { using other = ArenaAllocator<U>; };

    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    explicit ArenaAllocator(Arena* arena = nullptr) noexcept : arena_(arena) {}

    template<class U>
    explicit ArenaAllocator(const ArenaAllocator<U>& other) noexcept : arena_(other.arena()) {}

    [[nodiscard]] pointer allocate(size_type n) {
        FEM_ASSERT(arena_ != nullptr);
        void* p = arena_->allocate(n * sizeof(T), alignof(T));
        return static_cast<pointer>(p);
    }

    [[nodiscard]] fem::core::error::Result<pointer, fem::core::error::ErrorCode>
    try_allocate(size_type n) {
        using fem::core::error::ErrorCode;
        if (arena_ == nullptr) return fem::core::error::Err<ErrorCode>(ErrorCode::InvalidState);
        try {
            return allocate(n);
        } catch (const std::bad_alloc&) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::OutOfMemory);
        } catch (...) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::SystemError);
        }
    }

    void deallocate(pointer /*p*/, size_type /*n*/) noexcept {
        // No-op. Memory reclaimed when the arena rewinds/resets/destroys.
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }

    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    [[nodiscard]] Arena* arena() const noexcept { return arena_; }

    // Equality: same arena pointer
    template<class U>
    bool operator==(const ArenaAllocator<U>& other) const noexcept { return arena_ == other.arena(); }
    template<class U>
    bool operator!=(const ArenaAllocator<U>& other) const noexcept { return !(*this == other); }

private:
    template<class> friend class ArenaAllocator;
    Arena* arena_{nullptr};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_ARENA_ALLOCATOR_H

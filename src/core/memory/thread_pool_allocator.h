#pragma once

#ifndef CORE_MEMORY_THREAD_POOL_ALLOCATOR_H
#define CORE_MEMORY_THREAD_POOL_ALLOCATOR_H

#include <cstddef>
#include <type_traits>

#include <config/config.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "pool_allocator.h"

namespace fem::core::memory {

// ThreadPoolAllocator<T>: std::allocator-compatible allocator that uses a
// thread-local PoolAllocator<T> to reduce contention in multi-threaded code.
// Each thread owns its own small-object free list.
template<class T, std::size_t BlockBytes = fem::config::PAGE_SIZE>
class ThreadPoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = ThreadPoolAllocator<U, BlockBytes>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type; // per-thread state only

    ThreadPoolAllocator() noexcept = default;
    template<class U>
    ThreadPoolAllocator(const ThreadPoolAllocator<U, BlockBytes>&) noexcept {}

    [[nodiscard]] pointer allocate(size_type n) {
        if (n == 1) return tls_pool().allocate(1);
        // bulk fallback
        return static_cast<pointer>(default_resource()->allocate(n * sizeof(T), alignof(T)));
    }
    void deallocate(pointer p, size_type n) noexcept {
        if (!p) return;
        if (n == 1) tls_pool().deallocate(p, 1);
        else default_resource()->deallocate(p, n * sizeof(T), alignof(T));
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

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

private:
    using Pool = PoolAllocator<T, BlockBytes>;
    static Pool& tls_pool() {
        thread_local Pool pool(default_resource());
        return pool;
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_THREAD_POOL_ALLOCATOR_H

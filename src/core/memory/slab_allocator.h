#pragma once

#ifndef CORE_MEMORY_SLAB_ALLOCATOR_H
#define CORE_MEMORY_SLAB_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <type_traits>

#include <config/config.h>

#include "memory_resource.h"
#include "memory_pool.h"
#include <core/error/result.h>
#include <core/error/error_code.h>

namespace fem::core::memory {

// SlabAllocator<T>: std::allocator-compatible allocator using a per-type
// MemoryPool with nodes grouped into slabs (blocks) of configurable size.
// This mirrors a typical slab allocator pattern at a high level while relying
// on MemoryPool for the underlying block management.
template<class T>
class SlabAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = SlabAllocator<U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type; // stateful shared pool

    explicit SlabAllocator(std::shared_ptr<MemoryPool> pool) : pool_(std::move(pool)) {}
    explicit SlabAllocator(memory_resource* upstream = default_resource(), std::size_t nodes_per_slab = 256)
        : pool_(std::make_shared<MemoryPool>(MemoryPool::Config{sizeof(T), alignof(T), nodes_per_slab}, upstream)) {}

    template<class U>
    SlabAllocator(const SlabAllocator<U>& other) : pool_() {
        // When rebinding to a different type, we cannot share the pool
        // because the pool is configured for a specific object size.
        // Create a new pool for the rebound type.
        if (other.pool_) {
            if constexpr (sizeof(T) != sizeof(U) || alignof(T) != alignof(U)) {
                // Different size/alignment - need a new pool
                pool_ = std::make_shared<MemoryPool>(
                    MemoryPool::Config{sizeof(T), alignof(T), 256},
                    other.pool_->get_upstream());
            } else {
                // Same size/alignment - can share the pool
                pool_ = other.pool_;
            }
        }
    }

    [[nodiscard]] pointer allocate(size_type n) {
        if (n == 1) return static_cast<pointer>(pool_->allocate());
        return static_cast<pointer>(default_resource()->allocate(n * sizeof(T), alignof(T))); // bulk fallback
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
        if (!p) return;
        if (n == 1) pool_->deallocate(p);
        else default_resource()->deallocate(p, n * sizeof(T), alignof(T));
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    template<class U>
    bool operator==(const SlabAllocator<U>& other) const noexcept { return pool_.get() == other.pool_.get(); }
    template<class U>
    bool operator!=(const SlabAllocator<U>& other) const noexcept { return !(*this == other); }

private:
    template<class> friend class SlabAllocator;
    std::shared_ptr<MemoryPool> pool_{};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_SLAB_ALLOCATOR_H

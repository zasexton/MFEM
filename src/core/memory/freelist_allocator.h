#pragma once

#ifndef CORE_MEMORY_FREELIST_ALLOCATOR_H
#define CORE_MEMORY_FREELIST_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <type_traits>

#include <config/config.h>
#include <config/debug.h>

#include "memory_resource.h"
#include "memory_pool.h"

namespace fem::core::memory {

// FreeListAllocator<T>: std::allocator-compatible allocator backed by a
// generic MemoryPool of fixed-size nodes (size = sizeof(T)). This is similar
// in spirit to PoolAllocator but uses the untyped MemoryPool primitive and a
// shared ownership model so multiple containers can share a pool.
template<class T>
class FreeListAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = FreeListAllocator<U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type; // stateful (shared pool)

    // Construct with shared pool or by creating one
    explicit FreeListAllocator(std::shared_ptr<MemoryPool> pool)
        : pool_(std::move(pool)) { FEM_ASSERT(pool_ != nullptr); }

    explicit FreeListAllocator(memory_resource* upstream = default_resource(),
                               std::size_t nodes_per_block = 256)
        : pool_(std::make_shared<MemoryPool>(MemoryPool::Config{sizeof(T), alignof(T), nodes_per_block}, upstream)) {}

    template<class U>
    FreeListAllocator(const FreeListAllocator<U>& other) noexcept : pool_(other.pool_) {}

    [[nodiscard]] pointer allocate(size_type n) {
        FEM_ASSERT(n > 0);
        if (n == 1) {
            return static_cast<pointer>(pool_->allocate());
        }
        // bulk allocation: fall back to upstream through a temporary allocator
        auto* mr = pool_upstream();
        return static_cast<pointer>(mr->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (!p) return;
        if (n == 1) pool_->deallocate(p);
        else pool_upstream()->deallocate(p, n * sizeof(T), alignof(T));
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }
    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    template<class U>
    bool operator==(const FreeListAllocator<U>& other) const noexcept { return pool_.get() == other.pool_.get(); }
    template<class U>
    bool operator!=(const FreeListAllocator<U>& other) const noexcept { return !(*this == other); }

private:
    template<class> friend class FreeListAllocator;
    std::shared_ptr<MemoryPool> pool_{};

    memory_resource* pool_upstream() const noexcept { return default_resource(); }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_FREELIST_ALLOCATOR_H

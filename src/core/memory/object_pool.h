#pragma once

#ifndef CORE_MEMORY_OBJECT_POOL_H
#define CORE_MEMORY_OBJECT_POOL_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <functional>

#include <config/config.h>
#include <config/debug.h>

#include "pool_allocator.h"
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

// ObjectPool: acquires single objects of T with stable addresses, returning them to the pool on destruction.
// Not thread-safe (use a concurrent variant in core/concurrency for MPMC use cases).
template<class T, std::size_t BlockBytes = fem::config::PAGE_SIZE>
class ObjectPool {
public:
    using value_type = T;
    using allocator_type = PoolAllocator<T, BlockBytes>;

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t outstanding = 0;
        std::size_t acquired = 0;
        std::size_t released = 0;
        std::size_t prewarmed = 0;
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    struct Releaser {
        ObjectPool* pool;
        void operator()(T* p) const noexcept {
            if (!pool || !p) return;
            if constexpr (!std::is_trivially_destructible_v<T>) p->~T();
            pool->alloc_.deallocate(p, 1);
            --pool->outstanding_;
#if CORE_MEMORY_ENABLE_TELEMETRY
            ++pool->telemetry_.released;
            pool->telemetry_.outstanding = pool->outstanding_;
            if (pool->telemetry_cb_) pool->telemetry_cb_("release", pool->telemetry_);
#endif
        }
    };

    using handle = std::unique_ptr<T, Releaser>;

    explicit ObjectPool(memory_resource* upstream = default_resource())
        : alloc_(upstream) {}

    // Disallow copy/move to avoid dangling deleters. Pools are typically singletons or long-lived.
    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    ~ObjectPool() {
        // All outstanding objects must be returned prior to pool destruction.
        FEM_DEBUG_ASSERT(outstanding_ == 0);
    }

    template<class... Args>
    [[nodiscard]] handle acquire(Args&&... args) {
        T* p = alloc_.allocate(1); // raw storage from pool (uninitialized)
        ::new ((void*)p) T(std::forward<Args>(args)...);
        ++outstanding_;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.acquired;
        telemetry_.outstanding = outstanding_;
        if (telemetry_cb_) telemetry_cb_("acquire", telemetry_);
#endif
        return handle(p, Releaser{this});
    }

    // Pre-allocate nodes into the free list without constructing T
    void reserve_nodes(std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            T* p = alloc_.allocate(1);
            alloc_.deallocate(p, 1); // return immediately to free list
        }
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_.prewarmed += n;
        if (telemetry_cb_) telemetry_cb_("reserve", telemetry_);
#endif
    }

    // Manual release in case you don't use handle
    void release(T* p) noexcept { Releaser{this}(p); }

    [[nodiscard]] std::size_t outstanding() const noexcept { return outstanding_; }

#if CORE_MEMORY_ENABLE_TELEMETRY
    [[nodiscard]] const telemetry_t& telemetry() const noexcept { return telemetry_; }
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    allocator_type alloc_;
    std::size_t outstanding_{0};

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_OBJECT_POOL_H

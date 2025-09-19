#pragma once

#ifndef CORE_MEMORY_CONCURRENT_POOL_H
#define CORE_MEMORY_CONCURRENT_POOL_H

#include <cstddef>
#include <mutex>
#include <functional>

#include <config/config.h>

#include "memory_pool.h"

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

// Thread-safe wrapper around MemoryPool using a mutex for simplicity.
class ConcurrentPool {
public:
    using Config = MemoryPool::Config;

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t alloc_calls = 0;
        std::size_t dealloc_calls = 0;
        std::size_t refills = 0;
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    explicit ConcurrentPool(Config cfg, memory_resource* upstream = default_resource())
        : pool_(cfg, upstream) {}

    void* allocate() {
        std::lock_guard lk(mutex_);
        if (pool_.free_count() == 0) {
            pool_.reserve_nodes(pool_.nodes_per_block());
#if CORE_MEMORY_ENABLE_TELEMETRY
            ++telemetry_.refills;
            if (telemetry_cb_) telemetry_cb_("refill", telemetry_);
#endif
        }
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.alloc_calls;
#endif
        return pool_.allocate();
    }

    void deallocate(void* p) {
        std::lock_guard lk(mutex_);
        pool_.deallocate(p);
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.dealloc_calls;
#endif
    }

    void reserve_nodes(std::size_t n) {
        std::lock_guard lk(mutex_);
        pool_.reserve_nodes(n);
    }

    std::size_t free_count() const {
        std::lock_guard lk(mutex_);
        return pool_.free_count();
    }

    std::size_t block_count() const {
        std::lock_guard lk(mutex_);
        return pool_.block_count();
    }

#if CORE_MEMORY_ENABLE_TELEMETRY
    void set_telemetry_callback(telemetry_callback_t cb) { std::lock_guard lk(mutex_); telemetry_cb_ = std::move(cb); }
#endif

private:
    mutable std::mutex mutex_;
    MemoryPool pool_;

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_CONCURRENT_POOL_H


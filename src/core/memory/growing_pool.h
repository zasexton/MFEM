#pragma once

#ifndef CORE_MEMORY_GROWING_POOL_H
#define CORE_MEMORY_GROWING_POOL_H

#include <cstddef>
#include <functional>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "memory_resource.h"
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

// A non-thread-safe pool that grows geometrically: each new block allocates
// more nodes than the previous, bounded by max_nodes_per_block. Useful when the
// eventual scale is unknown and you want fewer upstream allocations over time.
class GrowingPool {
public:
    struct Config {
        std::size_t object_size;
        std::size_t alignment = alignof(std::max_align_t);
        std::size_t initial_nodes_per_block = 256;
        double growth_factor = 2.0;             // multiply nodes_per_block by this on each refill
        std::size_t max_nodes_per_block = 8192; // upper bound per block
    };

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t refills = 0;
        std::size_t current_nodes_per_block = 0;
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    explicit GrowingPool(Config cfg, memory_resource* upstream = default_resource())
        : upstream_(upstream)
        , cfg_(cfg)
        , current_nodes_per_block_(cfg.initial_nodes_per_block)
        , pool_(MemoryPool::Config{cfg.object_size, cfg.alignment, cfg.initial_nodes_per_block}, upstream) {
        FEM_ASSERT(upstream_ != nullptr);
        FEM_ASSERT(cfg_.object_size > 0);
        FEM_ASSERT(cfg_.initial_nodes_per_block > 0);
        FEM_ASSERT(cfg_.growth_factor >= 1.0);
    }

    GrowingPool(const GrowingPool&) = delete;
    GrowingPool& operator=(const GrowingPool&) = delete;

    GrowingPool(GrowingPool&&) noexcept = default;
    GrowingPool& operator=(GrowingPool&&) noexcept = default;

    [[nodiscard]] void* allocate() {
        if (pool_.free_count() == 0) refill();
        return pool_allocate();
    }

    void deallocate(void* p) noexcept { pool_.deallocate(p); }

    [[nodiscard]] fem::core::error::Result<void*, fem::core::error::ErrorCode>
    try_allocate() {
        using fem::core::error::ErrorCode;
        if (pool_.free_count() == 0) {
            // Attempt a growth refill; map failures
            try {
                refill();
            } catch (const std::bad_alloc&) {
                return fem::core::error::Err<ErrorCode>(ErrorCode::OutOfMemory);
            } catch (...) {
                return fem::core::error::Err<ErrorCode>(ErrorCode::SystemError);
            }
        }
        auto r = pool_.try_allocate();
        if (!r) return fem::core::error::Err(r.error());
        return r;
    }

    void reserve_nodes(std::size_t n) {
        while (pool_.free_count() < n) refill();
    }

    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    try_reserve_nodes(std::size_t n) {
        using fem::core::error::ErrorCode;
        try {
            reserve_nodes(n);
            return {};
        } catch (const std::bad_alloc&) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::OutOfMemory);
        } catch (...) {
            return fem::core::error::Err<ErrorCode>(ErrorCode::SystemError);
        }
    }

    [[nodiscard]] std::size_t free_count() const noexcept { return pool_.free_count(); }
    [[nodiscard]] std::size_t block_count() const noexcept { return pool_.block_count(); }
    [[nodiscard]] std::size_t current_nodes_per_block() const noexcept { return current_nodes_per_block_; }

#if CORE_MEMORY_ENABLE_TELEMETRY
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    memory_resource* upstream_{};
    Config cfg_{};
    std::size_t current_nodes_per_block_{};
    MemoryPool pool_;

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif

    void refill() {
        // Create a temp MemoryPool with the new block size and move-merge its free list into ours
        std::size_t next_np = static_cast<std::size_t>(static_cast<double>(current_nodes_per_block_) * cfg_.growth_factor);
        if (next_np < current_nodes_per_block_) next_np = current_nodes_per_block_; // guard overflow
        if (next_np > cfg_.max_nodes_per_block) next_np = cfg_.max_nodes_per_block;

        // Build a temporary pool to allocate one block of the new size, then merge by deallocating into ours
        MemoryPool temp(MemoryPool::Config{cfg_.object_size, cfg_.alignment, next_np}, upstream_);
        temp.reserve_nodes(next_np);
        // Consume temp's nodes into our pool
        for (std::size_t i = 0; i < next_np; ++i) {
            void* node = temp.allocate();
            pool_.deallocate(node); // push onto our free list
        }
        current_nodes_per_block_ = next_np;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.refills;
        telemetry_.current_nodes_per_block = current_nodes_per_block_;
        if (telemetry_cb_) telemetry_cb_("refill", telemetry_);
#endif
    }

    void* pool_allocate() { return pool_.allocate(); }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_GROWING_POOL_H

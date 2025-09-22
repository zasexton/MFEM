#pragma once

#ifndef CORE_MEMORY_MEMORY_POOL_H
#define CORE_MEMORY_MEMORY_POOL_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

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

// A generic, non-thread-safe fixed-size memory pool for arbitrary object sizes.
// Allocates blocks from an upstream memory_resource and carves them into nodes.
class MemoryPool {
public:
    struct Config {
        std::size_t object_size;               // size of each node (bytes)
        std::size_t alignment = alignof(std::max_align_t);
        std::size_t nodes_per_block = 256;     // default block granularity
    };

#if CORE_MEMORY_ENABLE_TELEMETRY
    struct telemetry_t {
        std::size_t blocks_allocated = 0;
        std::size_t nodes_total = 0;     // total nodes ever added to the pool
        std::size_t nodes_free = 0;      // free-list length
        std::size_t alloc_calls = 0;     // allocate() calls served by pool
        std::size_t dealloc_calls = 0;   // deallocate() calls
        std::size_t peak_in_use = 0;     // max (nodes_total - nodes_free)
    };
    using telemetry_callback_t = std::function<void(const char* event, const telemetry_t&)>;
#endif

    explicit MemoryPool(Config cfg,
                        memory_resource* upstream = default_resource())
        : upstream_(upstream)
        , object_size_(cfg.object_size < sizeof(void*) ? sizeof(void*) : cfg.object_size)
        , alignment_(cfg.alignment)
        , nodes_per_block_(cfg.nodes_per_block) {
        FEM_ASSERT(upstream_ != nullptr);
        FEM_ASSERT(object_size_ > 0);
        FEM_ASSERT(nodes_per_block_ > 0);
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    MemoryPool(MemoryPool&& other) noexcept { move_from(std::move(other)); }
    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this != &other) { release_all(); move_from(std::move(other)); }
        return *this;
    }

    ~MemoryPool() { release_all(); }

    // Allocate one node; undefined behavior to request more than one at a time.
    [[nodiscard]] void* allocate() {
        if (!free_list_) refill();
        Node* n = free_list_;
        free_list_ = free_list_->next;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.alloc_calls;
        --telemetry_.nodes_free;
        const std::size_t in_use = telemetry_.nodes_total - telemetry_.nodes_free;
        if (in_use > telemetry_.peak_in_use) telemetry_.peak_in_use = in_use;
#endif
        return reinterpret_cast<void*>(n);
    }

    [[nodiscard]] fem::core::error::Result<void*, fem::core::error::ErrorCode>
    try_allocate() {
        using fem::core::error::ErrorCode;
        try {
            return allocate();
        } catch (const std::bad_alloc&) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::OutOfMemory};
        } catch (...) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::SystemError};
        }
    }

    void deallocate(void* p) noexcept {
        if (!p) return;
        auto* n = static_cast<Node*>(p);
        n->next = free_list_;
        free_list_ = n;
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.dealloc_calls;
        ++telemetry_.nodes_free;
#endif
    }

    // Reserve at least n nodes in the free list
    void reserve_nodes(std::size_t n) {
        while (free_count() < n) refill();
    }

    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    try_reserve_nodes(std::size_t n) {
        using fem::core::error::ErrorCode;
        try {
            reserve_nodes(n);
            return {};
        } catch (const std::bad_alloc&) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::OutOfMemory};
        } catch (...) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::SystemError};
        }
    }

    void shrink_to_fit() noexcept {
        // Release all fully-free blocks? For simplicity, keep blocks; full reclamation via release_all.
        // Intentionally no-op to guarantee pointer stability of remaining nodes.
    }

    void release_all() noexcept {
        for (void* b : blocks_) upstream_->deallocate(b, block_bytes(), alignment_);
        blocks_.clear();
        free_list_ = nullptr;
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = {};
#endif
    }

    [[nodiscard]] std::size_t object_size() const noexcept { return object_size_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] std::size_t nodes_per_block() const noexcept { return nodes_per_block_; }
    [[nodiscard]] std::size_t block_bytes() const noexcept { return object_size_ * nodes_per_block_; }

    [[nodiscard]] std::size_t free_count() const noexcept {
        std::size_t c = 0; for (Node* n = free_list_; n; n = n->next) ++c; return c;
    }

    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }

    [[nodiscard]] memory_resource* get_upstream() const noexcept { return upstream_; }

#if CORE_MEMORY_ENABLE_TELEMETRY
    [[nodiscard]] const telemetry_t& telemetry() const noexcept { return telemetry_; }
    void set_telemetry_callback(telemetry_callback_t cb) { telemetry_cb_ = std::move(cb); }
#endif

private:
    struct Node { Node* next; };

    memory_resource* upstream_{};
    std::vector<void*> blocks_{};
    Node* free_list_{nullptr};
    std::size_t object_size_{};
    std::size_t alignment_{};
    std::size_t nodes_per_block_{};

#if CORE_MEMORY_ENABLE_TELEMETRY
    telemetry_t telemetry_{};
    telemetry_callback_t telemetry_cb_{};
#endif

    void refill() {
        void* raw = upstream_->allocate(block_bytes(), alignment_);
        blocks_.push_back(raw);
        auto* base = static_cast<std::byte*>(raw);
        for (std::size_t i = 0; i < nodes_per_block_; ++i) {
            auto* node = reinterpret_cast<Node*>(base + i * object_size_);
            node->next = free_list_;
            free_list_ = node;
        }
#if CORE_MEMORY_ENABLE_TELEMETRY
        ++telemetry_.blocks_allocated;
        telemetry_.nodes_total += nodes_per_block_;
        telemetry_.nodes_free += nodes_per_block_;
        if (telemetry_cb_) telemetry_cb_("allocate_block", telemetry_);
#endif
    }

    void move_from(MemoryPool&& o) noexcept {
        upstream_ = o.upstream_;
        blocks_ = std::move(o.blocks_);
        free_list_ = o.free_list_; o.free_list_ = nullptr;
        object_size_ = o.object_size_;
        alignment_ = o.alignment_;
        nodes_per_block_ = o.nodes_per_block_;
#if CORE_MEMORY_ENABLE_TELEMETRY
        telemetry_ = o.telemetry_;
        telemetry_cb_ = std::move(o.telemetry_cb_);
#endif
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_POOL_H

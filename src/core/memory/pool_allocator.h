#pragma once

#ifndef CORE_MEMORY_POOL_ALLOCATOR_H
#define CORE_MEMORY_POOL_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <type_traits>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "memory_resource.h"

namespace fem::core::memory {

// Fixed-size node pool optimized for single-object allocations of T.
// Not thread-safe. For n>1 allocations, falls back to upstream resource.
template<class T, std::size_t BlockBytes = fem::config::PAGE_SIZE>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    template<class U> struct rebind { using other = PoolAllocator<U, BlockBytes>; };

    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    explicit PoolAllocator(memory_resource* upstream = default_resource()) noexcept
        : upstream_(upstream) {
        FEM_ASSERT(upstream_ != nullptr);
        init_block_layout();
    }

    template<class U>
    explicit PoolAllocator(const PoolAllocator<U, BlockBytes>& other) noexcept
        : upstream_(other.upstream_) {
        init_block_layout();
    }

    ~PoolAllocator() {
        release_all();
    }

    [[nodiscard]] pointer allocate(size_type n) {
        if (n == 1) {
            if (!free_list_) refill();
            auto* node = free_list_;
            free_list_ = free_list_->next;
            return reinterpret_cast<pointer>(node);
        }
        // Fallback for multi-object allocation
        return static_cast<pointer>(upstream_->allocate(n * sizeof(T), alignof(T)));
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
        if (!p) return;
        if (n == 1) {
            auto* node = reinterpret_cast<Node*>(p);
            node->next = free_list_;
            free_list_ = node;
        } else {
            upstream_->deallocate(p, n * sizeof(T), alignof(T));
        }
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) { ::new ((void*)p) U(std::forward<Args>(args)...); }

    template<class U>
    void destroy(U* p) { if constexpr (!std::is_trivially_destructible_v<U>) p->~U(); }

    template<class U>
    bool operator==(const PoolAllocator<U, BlockBytes>& other) const noexcept { return upstream_ == other.upstream_; }
    template<class U>
    bool operator!=(const PoolAllocator<U, BlockBytes>& other) const noexcept { return !(*this == other); }

private:
    struct Node { Node* next; };

    memory_resource* upstream_{nullptr};
    Node* free_list_{nullptr};
    std::vector<void*> blocks_{};
    std::size_t nodes_per_block_{0};
    std::size_t node_size_{0};

    void init_block_layout() noexcept {
        node_size_ = std::max(sizeof(Node), sizeof(T));
        nodes_per_block_ = BlockBytes / node_size_;
        if (nodes_per_block_ == 0) nodes_per_block_ = 1;
    }

    void release_all() noexcept {
        for (void* b : blocks_) {
            upstream_->deallocate(b, BlockBytes, fem::config::DEFAULT_ALIGNMENT);
        }
        blocks_.clear();
        free_list_ = nullptr;
    }

    void refill() {
        // Allocate a new block and thread nodes into free list
        void* raw = upstream_->allocate(BlockBytes, fem::config::DEFAULT_ALIGNMENT);
        blocks_.push_back(raw);
        std::byte* base = static_cast<std::byte*>(raw);

        // Carve nodes
        for (std::size_t i = 0; i < nodes_per_block_; ++i) {
            auto* node = reinterpret_cast<Node*>(base + i * node_size_);
            node->next = free_list_;
            free_list_ = node;
        }
    }

    template<class, std::size_t> friend class PoolAllocator;
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_POOL_ALLOCATOR_H

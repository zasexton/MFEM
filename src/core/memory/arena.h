#pragma once

#ifndef CORE_MEMORY_ARENA_H
#define CORE_MEMORY_ARENA_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <new>
#include <type_traits>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#include "aligned_storage.h"
#include "memory_resource.h"

namespace fem::core::memory {

// A simple bump/linear arena with RAII scopes. Not thread-safe.
class Arena {
public:
    struct Marker {
        std::size_t block_index{0};
        std::size_t offset{0};
    };

    class Scope {
    public:
        explicit Scope(Arena& a) noexcept : arena_(a), mark_(a.mark()) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { arena_.rewind(mark_); }
    private:
        Arena& arena_;
        Marker mark_{};
    };

    Arena() : Arena(default_resource()) {}

    explicit Arena(std::size_t initial_block_bytes)
        : Arena(default_resource(), initial_block_bytes) {}

    explicit Arena(memory_resource* upstream,
                   std::size_t initial_block_bytes = fem::config::PAGE_SIZE)
        : upstream_(upstream),
          initial_block_bytes_(initial_block_bytes == 0 ? fem::config::PAGE_SIZE : initial_block_bytes) {
        FEM_ASSERT(upstream_ != nullptr);
    }

    // Move-only
    Arena(Arena&& other) noexcept { move_from(std::move(other)); }
    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            release_all();
            move_from(std::move(other));
        }
        return *this;
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    ~Arena() { release_all(); }

    // Allocate raw bytes with alignment
    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        // Handle zero-size allocations
        if (bytes == 0) bytes = 1;
        FEM_ASSERT(is_power_of_two(alignment));

        if (blocks_.empty()) { add_block(std::max(grow_size(bytes, alignment), initial_block_bytes_)); }

        // try current block
        for (;;) {
            Block& b = blocks_.back();
            std::uintptr_t base = reinterpret_cast<std::uintptr_t>(b.data);
            std::uintptr_t aligned = align_up(base + b.offset, alignment);
            std::size_t padding = static_cast<std::size_t>(aligned - (base + b.offset));
            if (b.offset + padding + bytes <= b.size) {
                b.offset += padding;
                void* p = b.data + b.offset;
                b.offset += bytes;
                FEM_ASSERT(is_aligned(p, alignment));
                return p;
            }
            // need a new block
            const std::size_t next_sz = std::max(grow_size(bytes, alignment), b.size * 2);
            add_block(next_sz);
        }
    }

    // Result-returning allocation (non-throwing)
    [[nodiscard]] fem::core::error::Result<void*, fem::core::error::ErrorCode>
    try_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        using fem::core::error::ErrorCode;
        if (!is_power_of_two(alignment) || bytes == 0) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::InvalidArgument};
        }
        try {
            return allocate(bytes, alignment);
        } catch (const std::bad_alloc&) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::OutOfMemory};
        } catch (...) {
            return fem::core::error::Error<ErrorCode>{ErrorCode::SystemError};
        }
    }

    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* p = allocate(sizeof(T), alignof(T));
        return ::new (p) T(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    fem::core::error::Result<T*, fem::core::error::ErrorCode>
    try_create(Args&&... args) {
        auto r = try_allocate(sizeof(T), alignof(T));
        if (!r) return fem::core::error::Error<fem::core::error::ErrorCode>{r.error()};
        T* p = static_cast<T*>(r.value());
        try {
            return ::new (p) T(std::forward<Args>(args)...);
        } catch (...) {
            return fem::core::error::Error<fem::core::error::ErrorCode>{fem::core::error::ErrorCode::SystemError};
        }
    }

    template<typename T>
    void destroy(T* ptr) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (ptr) ptr->~T();
        }
        // memory is reclaimed on rewind/reset
    }

    // Rewind to a previous position (must be a marker returned by this arena)
    void rewind(const Marker& m) noexcept {
        if (blocks_.empty()) return;
        // Free all blocks after m.block_index
        while (blocks_.size() > m.block_index + 1) {
            pop_block();
        }
        // reset offset
        blocks_.back().offset = m.offset;
    }

    // Mark current position
    [[nodiscard]] Marker mark() const noexcept {
        if (blocks_.empty()) return Marker{0, 0};
        return Marker{blocks_.size() - 1, blocks_.back().offset};
    }

    // Reset the entire arena, releasing all memory
    void reset() noexcept { release_all(); }

    [[nodiscard]] std::size_t used_bytes() const noexcept {
        std::size_t sum = 0;
        for (const auto& b : blocks_) sum += b.offset;
        return sum;
    }

    [[nodiscard]] std::size_t capacity_bytes() const noexcept {
        std::size_t sum = 0;
        for (const auto& b : blocks_) sum += b.size;
        return sum;
    }

    [[nodiscard]] bool empty() const noexcept { return used_bytes() == 0; }

    [[nodiscard]] memory_resource* upstream() const noexcept { return upstream_; }

    // Convenience aliases
    [[nodiscard]] std::size_t used() const noexcept { return used_bytes(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_bytes(); }
    [[nodiscard]] memory_resource* get_memory_resource() const noexcept { return upstream_; }
    [[nodiscard]] Scope scope() { return Scope(*this); }

private:
    struct Block {
        std::byte* data{nullptr};
        std::size_t size{0};
        std::size_t offset{0};
    };

    memory_resource* upstream_{nullptr};
    std::vector<Block> blocks_{};
    std::size_t initial_block_bytes_{0};

    static std::size_t grow_size(std::size_t bytes, std::size_t align) noexcept {
        // Account for potential alignment padding
        return align_up(bytes + align, fem::config::CACHE_LINE_SIZE);
    }

    void add_block(std::size_t bytes) {
        FEM_ASSERT(upstream_ != nullptr);
        auto* mem = static_cast<std::byte*>(upstream_->allocate(bytes, fem::config::DEFAULT_ALIGNMENT));
        blocks_.push_back(Block{mem, bytes, 0});
    }

    void pop_block() noexcept {
        if (blocks_.empty()) return;
        Block b = blocks_.back();
        blocks_.pop_back();
        upstream_->deallocate(b.data, b.size, fem::config::DEFAULT_ALIGNMENT);
    }

    void release_all() noexcept {
        for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
            upstream_->deallocate(it->data, it->size, fem::config::DEFAULT_ALIGNMENT);
        }
        blocks_.clear();
    }

    void move_from(Arena&& other) noexcept {
        upstream_ = other.upstream_;
        initial_block_bytes_ = other.initial_block_bytes_;
        blocks_ = std::move(other.blocks_);
        other.upstream_ = nullptr;
        other.initial_block_bytes_ = 0;
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_ARENA_H

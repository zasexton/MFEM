#pragma once

#ifndef NUMERIC_ALLOCATOR_BASE_H
#define NUMERIC_ALLOCATOR_BASE_H

#include <memory>
#include <cstdlib>
#include <new>
#include <atomic>
#include <vector>

#include "numeric_base.h"

namespace fem::numeric {

    /**
     * @brief Base allocator interface for numeric containers
     *
     * Provides memory allocation strategies for IEEE-compliant numeric types
     */
    template<typename T>
    class AllocatorBase {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        virtual ~AllocatorBase() = default;

        [[nodiscard]] virtual pointer allocate(size_type n) = 0;
        virtual void deallocate(pointer p, size_type n) = 0;

        virtual size_type max_size() const noexcept {
            return std::numeric_limits<size_type>::max() / sizeof(T);
        }

        // Optional: memory usage tracking
        virtual size_type allocated_size() const noexcept { return 0; }
        virtual size_type allocation_count() const noexcept { return 0; }
    };

    /**
     * @brief Aligned allocator for SIMD operations
     */
    template<typename T, size_t Alignment = 32>
    class AlignedAllocator {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        static constexpr size_t alignment = Alignment;

        template<typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() noexcept = default;

        template<typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        [[nodiscard]] pointer allocate(size_type n) {
            if (n == 0) return nullptr;

            // Check if n exceeds max_size
            if (n > max_size()) {
                throw std::bad_alloc();
            }

            size_type bytes = n * sizeof(T);
            void* p = std::aligned_alloc(alignment, bytes);

            if (!p) {
                throw std::bad_alloc();
            }

            return static_cast<pointer>(p);
        }

        void deallocate(pointer p, size_type) noexcept {
            std::free(p);
        }

        size_type max_size() const noexcept {
            return std::numeric_limits<size_type>::max() / sizeof(T);
        }

        template<typename U, typename... Args>
        void construct(U* p, Args&&... args) {
            if (p) {
                ::new(static_cast<void *>(p)) U(std::forward<Args>(args)...);
            }
        }

        template<typename U>
        void destroy(U* p) {
            if (p) {
                p->~U();
            }
        }

        bool operator==(const AlignedAllocator&) const noexcept { return true; }
        bool operator!=(const AlignedAllocator&) const noexcept { return false; }
    };

    /**
     * @brief Memory pool allocator for frequent allocations
     */
    template<typename T>
    class PoolAllocator {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        explicit PoolAllocator(size_type pool_size = 1024)
                : pool_size_(pool_size), current_block_(0) {
            allocate_block();
        }

        ~PoolAllocator() {
            for (auto* block : blocks_) {
                std::free(block);
            }
        }

        [[nodiscard]] pointer allocate(size_type n) {
            if (n > pool_size_) {
                // Large allocation - bypass pool
                return static_cast<pointer>(std::malloc(n * sizeof(T)));
            }

            std::lock_guard lock(mutex_);

            if (current_offset_ + n > pool_size_) {
                allocate_block();
            }

            pointer result = blocks_[current_block_] + current_offset_;
            current_offset_ += n;

            return result;
        }

        void deallocate(pointer p, size_type n) noexcept {
            // Pool allocator doesn't deallocate individual allocations
            // Memory is freed when the allocator is destroyed
            //TODO: check the need for specialized allocator/destructor

        }

        void reset() {
            std::lock_guard lock(mutex_);
            current_block_ = 0;
            current_offset_ = 0;
        }

    private:
        size_type pool_size_;
        size_type current_block_;
        size_type current_offset_ = 0;
        std::vector<pointer> blocks_;
        std::mutex mutex_;

        void allocate_block() {
            void* block = std::aligned_alloc(32, pool_size_ * sizeof(T));
            if (!block) {
                throw std::bad_alloc();
            }
            blocks_.push_back(static_cast<pointer>(block));
            current_block_ = blocks_.size() - 1;
            current_offset_ = 0;
        }
    };

    /**
     * @brief Tracking allocator for memory profiling
     */
    template<typename T, typename BaseAllocator = std::allocator<T>>
    class TrackingAllocator : public BaseAllocator {
    public:
        using value_type = T;
        using pointer = T*;
        using size_type = size_t;

        using BaseAllocator::BaseAllocator;

        [[nodiscard]] pointer allocate(size_type n) {
            pointer p = BaseAllocator::allocate(n);
            bytes_allocated_ += n * sizeof(T);
            allocation_count_++;

            if (bytes_allocated_ > peak_bytes_) {
                peak_bytes_ = bytes_allocated_.load();
            }

            return p;
        }

        void deallocate(pointer p, size_type n) noexcept {
            BaseAllocator::deallocate(p, n);
            bytes_allocated_ -= n * sizeof(T);
            deallocation_count_++;
        }

        static size_type get_bytes_allocated() noexcept {
            return bytes_allocated_.load();
        }

        static size_type get_peak_bytes() noexcept {
            return peak_bytes_.load();
        }

        static size_type get_allocation_count() noexcept {
            return allocation_count_.load();
        }

        static void reset_stats() noexcept {
            bytes_allocated_ = 0;
            peak_bytes_ = 0;
            allocation_count_ = 0;
            deallocation_count_ = 0;
        }

    private:
        static inline std::atomic<size_type> bytes_allocated_{0};
        static inline std::atomic<size_type> peak_bytes_{0};
        static inline std::atomic<size_type> allocation_count_{0};
        static inline std::atomic<size_type> deallocation_count_{0};
    };

    /**
     * @brief Stack allocator for temporary allocations
     */
    template<typename T, size_t StackSize = 1024>
    class StackAllocator {
    public:
        using value_type = T;
        using pointer = T*;
        using size_type = size_t;

        StackAllocator() noexcept : offset_(0) {}

        [[nodiscard]] pointer allocate(size_type n) {
            if (offset_ + n > StackSize) {
                // Fallback to heap allocation
                return static_cast<pointer>(std::malloc(n * sizeof(T)));
            }

            pointer result = reinterpret_cast<pointer>(&buffer_[offset_ * sizeof(T)]);
            offset_ += n;
            return result;
        }

        void deallocate(pointer p, size_type n) noexcept {
            // Check if pointer is from stack
            if (p >= reinterpret_cast<pointer>(buffer_) &&
                p < reinterpret_cast<pointer>(buffer_ + sizeof(buffer_))) {
                // Stack memory - can't individually deallocate
                return;
            }

            // Heap allocated
            std::free(p);
        }

        void reset() noexcept {
            offset_ = 0;
        }

    private:
        alignas(32) char buffer_[StackSize * sizeof(T)];
        size_type offset_;
    };

} // namespace fem::numeric

#endif //NUMERIC_ALLOCATOR_BASE_H

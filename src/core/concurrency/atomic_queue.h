#pragma once

#ifndef CORE_CONCURRENCY_ATOMIC_QUEUE_H
#define CORE_CONCURRENCY_ATOMIC_QUEUE_H

#include <atomic>
#include <memory>
#include <optional>
#include <array>
#include <cstddef>
#include <type_traits>
#include <functional>

#include <core/config/config.h>
#include <core/config/debug.h>

namespace fem::core::concurrency {

/**
 * @brief Lock-free multi-producer multi-consumer (MPMC) queue
 *
 * A high-performance, lock-free queue implementation suitable for concurrent
 * producer-consumer scenarios. Uses memory ordering and cache-line alignment
 * to minimize contention and false sharing.
 *
 * Features:
 * - Lock-free operations (enqueue/dequeue)
 * - MPMC (multi-producer, multi-consumer) support
 * - Memory-efficient node allocation
 * - ABA-safe using shared_ptr hazard pointers
 * - Proper memory ordering for correctness
 *
 * Example usage:
 * @code
 * AtomicQueue<int> queue;
 *
 * // Producer thread
 * queue.enqueue(42);
 * queue.enqueue(99);
 *
 * // Consumer thread
 * if (auto value = queue.dequeue()) {
 *     process(*value);
 * }
 * @endcode
 */
template<typename T>
class AtomicQueue {
private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<std::shared_ptr<Node>> next{nullptr};

        Node() = default;

        // Disable copy and move to prevent accidental misuse
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = delete;
        Node& operator=(Node&&) = delete;
    };

    // Cache-line aligned atomic pointers to prevent false sharing
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::shared_ptr<Node>> head_;
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::shared_ptr<Node>> tail_;

public:
    /**
     * @brief Construct an empty atomic queue
     */
    AtomicQueue() {
        auto dummy = std::make_shared<Node>();
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    /**
     * @brief Destructor - cleans up remaining nodes
     */
    ~AtomicQueue() {
        // Drain any remaining items
        while (dequeue()) {
            // Items are automatically destroyed
        }
    }

    // Disable copy operations
    AtomicQueue(const AtomicQueue&) = delete;
    AtomicQueue& operator=(const AtomicQueue&) = delete;

    // Enable move operations
    AtomicQueue(AtomicQueue&& other) noexcept {
        head_.store(other.head_.exchange(nullptr, std::memory_order_acq_rel), std::memory_order_relaxed);
        tail_.store(other.tail_.exchange(nullptr, std::memory_order_acq_rel), std::memory_order_relaxed);
    }

    AtomicQueue& operator=(AtomicQueue&& other) noexcept {
        if (this != &other) {
            // Clean up current queue
            while (dequeue()) {}

            head_.store(other.head_.exchange(nullptr, std::memory_order_acq_rel), std::memory_order_relaxed);
            tail_.store(other.tail_.exchange(nullptr, std::memory_order_acq_rel), std::memory_order_relaxed);
        }
        return *this;
    }

    /**
     * @brief Enqueue an item to the queue
     * @param item The item to enqueue (moved if possible)
     *
     * This operation is lock-free and safe for concurrent producers.
     * The item is moved into the queue if T supports move semantics.
     */
    void enqueue(T item) {
        auto new_node = std::make_shared<Node>();
        auto data = new T(std::move(item));

        new_node->data.store(data, std::memory_order_relaxed);

        // Get current tail and update it atomically
        auto prev_tail = tail_.exchange(new_node, std::memory_order_acq_rel);

        // Link the previous tail to the new node
        prev_tail->next.store(new_node, std::memory_order_release);
    }

    /**
     * @brief Attempt to dequeue an item from the queue
     * @return Optional containing the item if available, nullopt if empty
     *
     * This operation is lock-free and safe for concurrent consumers.
     * Returns nullopt if the queue is empty at the time of the call.
     */
    std::optional<T> dequeue() {
        auto head = head_.load(std::memory_order_acquire);
        auto next = head->next.load(std::memory_order_acquire);

        if (!next) {
            // Queue is empty
            return std::nullopt;
        }

        // Try to advance head to next
        if (head_.compare_exchange_weak(head, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Successfully advanced head, now extract data
            auto data_ptr = next->data.exchange(nullptr, std::memory_order_acq_rel);

            if (data_ptr) {
                T result = std::move(*data_ptr);
                delete data_ptr;
                return result;
            }
        }

        // Failed to dequeue (contention or empty)
        return std::nullopt;
    }

    /**
     * @brief Check if the queue appears empty
     * @return true if the queue appears empty, false otherwise
     *
     * Note: This is a snapshot view and may not reflect the exact state
     * in a concurrent environment. Use only for heuristics.
     */
    bool empty() const noexcept {
        auto head = head_.load(std::memory_order_acquire);
        return head->next.load(std::memory_order_acquire) == nullptr;
    }

    /**
     * @brief Get approximate size of the queue
     * @return Approximate number of items in the queue
     *
     * Note: This is expensive (O(n)) and provides only an approximate count
     * in a concurrent environment. Use sparingly and only for diagnostics.
     */
    std::size_t size() const noexcept {
        std::size_t count = 0;
        auto current = head_.load(std::memory_order_acquire);

        while (current && current->next.load(std::memory_order_acquire)) {
            current = current->next.load(std::memory_order_acquire);
            ++count;
        }

        return count;
    }
};

/**
 * @brief Bounded lock-free multi-producer multi-consumer queue
 *
 * A fixed-capacity queue that provides backpressure when full.
 * Uses a ring buffer approach with atomic operations for high performance.
 *
 * Features:
 * - Fixed capacity for predictable memory usage
 * - Lock-free try_enqueue/try_dequeue operations
 * - Backpressure handling when full
 * - Cache-line alignment to prevent false sharing
 *
 * Template parameters:
 * @tparam T The type of items stored in the queue
 * @tparam Capacity The maximum number of items (must be power of 2)
 */
template<typename T, std::size_t Capacity>
class BoundedAtomicQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(Capacity > 0, "Capacity must be greater than 0");

private:
    struct alignas(fem::config::CACHE_LINE_SIZE) Slot {
        std::atomic<T*> data{nullptr};
        std::atomic<std::size_t> sequence{0};
    };

    // Ring buffer of slots
    std::array<Slot, Capacity> buffer_;

    // Producer and consumer positions (cache-line aligned)
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::size_t> enqueue_pos_{0};
    alignas(fem::config::CACHE_LINE_SIZE) std::atomic<std::size_t> dequeue_pos_{0};

    static constexpr std::size_t index_mask = Capacity - 1;

public:
    /**
     * @brief Construct an empty bounded queue
     */
    BoundedAtomicQueue() {
        // Initialize sequence numbers
        for (std::size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Destructor - cleans up remaining items
     */
    ~BoundedAtomicQueue() {
        // Clean up any remaining items
        T item;
        while (try_dequeue(item)) {
            // Items are automatically destroyed
        }
    }

    // Disable copy operations
    BoundedAtomicQueue(const BoundedAtomicQueue&) = delete;
    BoundedAtomicQueue& operator=(const BoundedAtomicQueue&) = delete;

    // Disable move operations for simplicity (could be implemented if needed)
    BoundedAtomicQueue(BoundedAtomicQueue&&) = delete;
    BoundedAtomicQueue& operator=(BoundedAtomicQueue&&) = delete;

    /**
     * @brief Attempt to enqueue an item
     * @param item The item to enqueue
     * @return true if successfully enqueued, false if queue is full
     */
    bool try_enqueue(T item) {
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            Slot& slot = buffer_[pos & index_mask];
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);

            if (diff == 0) {
                // Slot is available for writing
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    // Successfully claimed the slot
                    auto data = new T(std::move(item));
                    slot.data.store(data, std::memory_order_relaxed);
                    slot.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // Queue is full
                return false;
            } else {
                // Another thread got this slot, try next position
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    /**
     * @brief Attempt to dequeue an item
     * @param[out] item Reference to store the dequeued item
     * @return true if successfully dequeued, false if queue is empty
     */
    bool try_dequeue(T& item) {
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            Slot& slot = buffer_[pos & index_mask];
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);

            if (diff == 0) {
                // Slot has data ready for reading
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    // Successfully claimed the slot
                    auto data_ptr = slot.data.exchange(nullptr, std::memory_order_relaxed);
                    if (data_ptr) {
                        item = std::move(*data_ptr);
                        delete data_ptr;
                        slot.sequence.store(pos + Capacity, std::memory_order_release);
                        return true;
                    }
                }
            } else if (diff < 0) {
                // Queue is empty
                return false;
            } else {
                // Another thread got this slot, try next position
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    /**
     * @brief Alternative dequeue that returns optional
     * @return Optional containing the item if available, nullopt if empty
     */
    std::optional<T> try_dequeue() {
        T item;
        if (try_dequeue(item)) {
            return std::move(item);
        }
        return std::nullopt;
    }

    /**
     * @brief Check if the queue appears empty
     * @return true if the queue appears empty, false otherwise
     */
    bool empty() const noexcept {
        std::size_t enqueue_pos = enqueue_pos_.load(std::memory_order_acquire);
        std::size_t dequeue_pos = dequeue_pos_.load(std::memory_order_acquire);
        return enqueue_pos == dequeue_pos;
    }

    /**
     * @brief Check if the queue appears full
     * @return true if the queue appears full, false otherwise
     */
    bool full() const noexcept {
        std::size_t enqueue_pos = enqueue_pos_.load(std::memory_order_acquire);
        std::size_t dequeue_pos = dequeue_pos_.load(std::memory_order_acquire);
        return (enqueue_pos - dequeue_pos) >= Capacity;
    }

    /**
     * @brief Get approximate size of the queue
     * @return Approximate number of items in the queue
     */
    std::size_t size() const noexcept {
        std::size_t enqueue_pos = enqueue_pos_.load(std::memory_order_acquire);
        std::size_t dequeue_pos = dequeue_pos_.load(std::memory_order_acquire);
        return enqueue_pos - dequeue_pos;
    }

    /**
     * @brief Get the maximum capacity of the queue
     * @return The maximum number of items the queue can hold
     */
    static constexpr std::size_t capacity() noexcept {
        return Capacity;
    }
};

// Common instantiations for convenience
using AtomicTaskQueue = AtomicQueue<std::function<void()>>;
using BoundedTaskQueue16 = BoundedAtomicQueue<std::function<void()>, 16>;
using BoundedTaskQueue64 = BoundedAtomicQueue<std::function<void()>, 64>;
using BoundedTaskQueue256 = BoundedAtomicQueue<std::function<void()>, 256>;

} // namespace fem::core::concurrency

#endif // CORE_CONCURRENCY_ATOMIC_QUEUE_H
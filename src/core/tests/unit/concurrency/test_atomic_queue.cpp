#include <gtest/gtest.h>
#include <core/concurrency/atomic_queue.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <set>
#include <algorithm>
#include <future>

namespace fcc = fem::core::concurrency;

class AtomicQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state
    }

    void TearDown() override {
        // Clean up after tests
    }
};

// ==================== Basic AtomicQueue Tests ====================

TEST_F(AtomicQueueTest, DefaultConstruction) {
    fcc::AtomicQueue<int> queue;
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(AtomicQueueTest, SingleEnqueueDequeue) {
    fcc::AtomicQueue<int> queue;

    queue.enqueue(42);
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 1);

    auto result = queue.dequeue();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
    EXPECT_TRUE(queue.empty());
}

TEST_F(AtomicQueueTest, MultipleEnqueueDequeue) {
    fcc::AtomicQueue<int> queue;

    // Enqueue multiple items
    for (int i = 1; i <= 5; ++i) {
        queue.enqueue(i);
    }

    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 5);

    // Dequeue in FIFO order
    for (int i = 1; i <= 5; ++i) {
        auto result = queue.dequeue();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, i);
    }

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(AtomicQueueTest, DequeueFromEmpty) {
    fcc::AtomicQueue<int> queue;

    auto result = queue.dequeue();
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(queue.empty());
}

TEST_F(AtomicQueueTest, MoveOnlyTypes) {
    fcc::AtomicQueue<std::unique_ptr<int>> queue;

    auto ptr = std::make_unique<int>(42);
    queue.enqueue(std::move(ptr));

    auto result = queue.dequeue();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(**result, 42);
}

TEST_F(AtomicQueueTest, StringTypes) {
    fcc::AtomicQueue<std::string> queue;

    queue.enqueue("hello");
    queue.enqueue("world");

    auto first = queue.dequeue();
    auto second = queue.dequeue();

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, "hello");
    EXPECT_EQ(*second, "world");
}

// ==================== Concurrent AtomicQueue Tests ====================

TEST_F(AtomicQueueTest, ConcurrentProducerConsumer) {
    fcc::AtomicQueue<int> queue;
    const int num_items = 1000;
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            queue.enqueue(i);
            produced++;
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int count = 0;
        while (count < num_items) {
            if (auto result = queue.dequeue()) {
                consumed++;
                count++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(produced.load(), num_items);
    EXPECT_EQ(consumed.load(), num_items);
    EXPECT_TRUE(queue.empty());
}

TEST_F(AtomicQueueTest, MultipleProducersConsumers) {
    fcc::AtomicQueue<int> queue;
    const int num_producers = 4;
    const int num_consumers = 3;
    const int items_per_producer = 250;
    const int total_items = num_producers * items_per_producer;

    std::atomic<int> total_produced{0};
    std::atomic<int> total_consumed{0};
    std::set<int> consumed_values;
    std::mutex consumed_mutex;

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                int value = p * items_per_producer + i;
                queue.enqueue(value);
                total_produced++;
            }
        });
    }

    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&]() {
            while (total_consumed.load() < total_items) {
                if (auto result = queue.dequeue()) {
                    {
                        std::lock_guard<std::mutex> lock(consumed_mutex);
                        consumed_values.insert(*result);
                    }
                    total_consumed++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for all producers
    for (auto& producer : producers) {
        producer.join();
    }

    // Wait for all consumers
    for (auto& consumer : consumers) {
        consumer.join();
    }

    EXPECT_EQ(total_produced.load(), total_items);
    EXPECT_EQ(total_consumed.load(), total_items);
    EXPECT_EQ(consumed_values.size(), total_items);
    EXPECT_TRUE(queue.empty());
}

TEST_F(AtomicQueueTest, StressTestLarge) {
    fcc::AtomicQueue<std::size_t> queue;
    const std::size_t num_operations = 10000;
    const int num_threads = 8;

    std::vector<std::thread> threads;
    std::atomic<std::size_t> enqueue_count{0};
    std::atomic<std::size_t> dequeue_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1);

            for (std::size_t i = 0; i < num_operations; ++i) {
                if (dis(gen) == 0) {
                    // Enqueue
                    std::size_t value = t * num_operations + i;
                    queue.enqueue(value);
                    enqueue_count++;
                } else {
                    // Dequeue
                    if (queue.dequeue()) {
                        dequeue_count++;
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Drain remaining items
    while (queue.dequeue()) {
        dequeue_count++;
    }

    EXPECT_EQ(enqueue_count.load(), dequeue_count.load());
    EXPECT_TRUE(queue.empty());
}

// ==================== BoundedAtomicQueue Tests ====================

TEST_F(AtomicQueueTest, BoundedQueueConstruction) {
    fcc::BoundedAtomicQueue<int, 16> queue;
    EXPECT_TRUE(queue.empty());
    EXPECT_FALSE(queue.full());
    EXPECT_EQ(queue.size(), 0);
    EXPECT_EQ(queue.capacity(), 16);
}

TEST_F(AtomicQueueTest, BoundedQueueBasicOperations) {
    fcc::BoundedAtomicQueue<int, 4> queue;

    // Fill the queue
    EXPECT_TRUE(queue.try_enqueue(1));
    EXPECT_TRUE(queue.try_enqueue(2));
    EXPECT_TRUE(queue.try_enqueue(3));
    EXPECT_TRUE(queue.try_enqueue(4));

    EXPECT_TRUE(queue.full());
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 4);

    // Queue should be full now
    EXPECT_FALSE(queue.try_enqueue(5));

    // Dequeue items
    int item;
    EXPECT_TRUE(queue.try_dequeue(item));
    EXPECT_EQ(item, 1);

    EXPECT_TRUE(queue.try_dequeue(item));
    EXPECT_EQ(item, 2);

    EXPECT_FALSE(queue.full());
    EXPECT_EQ(queue.size(), 2);

    // Can enqueue again
    EXPECT_TRUE(queue.try_enqueue(5));
    EXPECT_EQ(queue.size(), 3);
}

TEST_F(AtomicQueueTest, BoundedQueueAlternativeDequeue) {
    fcc::BoundedAtomicQueue<std::string, 8> queue;

    queue.try_enqueue("hello");
    queue.try_enqueue("world");

    auto result1 = queue.try_dequeue();
    auto result2 = queue.try_dequeue();
    auto result3 = queue.try_dequeue();

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    EXPECT_FALSE(result3.has_value());

    EXPECT_EQ(*result1, "hello");
    EXPECT_EQ(*result2, "world");
}

TEST_F(AtomicQueueTest, BoundedQueueBackpressure) {
    fcc::BoundedAtomicQueue<int, 2> queue;

    // Fill to capacity
    EXPECT_TRUE(queue.try_enqueue(1));
    EXPECT_TRUE(queue.try_enqueue(2));
    EXPECT_TRUE(queue.full());

    // Should reject additional items
    EXPECT_FALSE(queue.try_enqueue(3));
    EXPECT_FALSE(queue.try_enqueue(4));

    // Free up space
    int item;
    EXPECT_TRUE(queue.try_dequeue(item));
    EXPECT_EQ(item, 1);
    EXPECT_FALSE(queue.full());

    // Can enqueue again
    EXPECT_TRUE(queue.try_enqueue(3));
    EXPECT_TRUE(queue.full());
}

TEST_F(AtomicQueueTest, BoundedQueueConcurrent) {
    fcc::BoundedAtomicQueue<int, 64> queue;
    const int num_items = 500;
    std::atomic<int> successful_enqueues{0};
    std::atomic<int> successful_dequeues{0};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            while (!queue.try_enqueue(i)) {
                std::this_thread::yield();
            }
            successful_enqueues++;
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int consumed = 0;
        while (consumed < num_items) {
            int item;
            if (queue.try_dequeue(item)) {
                successful_dequeues++;
                consumed++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(successful_enqueues.load(), num_items);
    EXPECT_EQ(successful_dequeues.load(), num_items);
    EXPECT_TRUE(queue.empty());
}

TEST_F(AtomicQueueTest, BoundedQueueMultipleProducersConsumers) {
    fcc::BoundedAtomicQueue<std::size_t, 128> queue;
    const int num_producers = 3;
    const int num_consumers = 2;
    const int items_per_producer = 200;
    const int total_items = num_producers * items_per_producer;

    std::atomic<int> total_enqueued{0};
    std::atomic<int> total_dequeued{0};

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                std::size_t value = p * items_per_producer + i;
                while (!queue.try_enqueue(value)) {
                    std::this_thread::yield();
                }
                total_enqueued++;
            }
        });
    }

    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&]() {
            while (total_dequeued.load() < total_items) {
                std::size_t item;
                if (queue.try_dequeue(item)) {
                    total_dequeued++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for completion
    for (auto& producer : producers) {
        producer.join();
    }
    for (auto& consumer : consumers) {
        consumer.join();
    }

    EXPECT_EQ(total_enqueued.load(), total_items);
    EXPECT_EQ(total_dequeued.load(), total_items);
    EXPECT_TRUE(queue.empty());
}

// ==================== Performance and Edge Case Tests ====================

TEST_F(AtomicQueueTest, MoveSemantics) {
    fcc::AtomicQueue<std::vector<int>> queue;

    std::vector<int> large_vec(1000, 42);
    const void* original_data = large_vec.data();

    queue.enqueue(std::move(large_vec));
    EXPECT_TRUE(large_vec.empty()); // Should be moved from

    auto result = queue.dequeue();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 1000);
    EXPECT_EQ(result->data(), original_data); // Should be same memory
}

// Helper for exception safety testing
struct ThrowingType {
    int value;
    static std::atomic<int> construct_count;
    static std::atomic<int> destruct_count;

    ThrowingType(int v) : value(v) { construct_count++; }
    ~ThrowingType() { destruct_count++; }

    ThrowingType(const ThrowingType& other) : value(other.value) {
        construct_count++;
        if (value == 999) throw std::runtime_error("Copy failed");
    }

    ThrowingType(ThrowingType&& other) noexcept : value(other.value) {
        construct_count++;
        other.value = -1; // Mark as moved
    }

    ThrowingType& operator=(const ThrowingType& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    ThrowingType& operator=(ThrowingType&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = -1; // Mark as moved
        }
        return *this;
    }
};

// Static member definitions
std::atomic<int> ThrowingType::construct_count{0};
std::atomic<int> ThrowingType::destruct_count{0};

TEST_F(AtomicQueueTest, ExceptionSafety) {
    ThrowingType::construct_count = 0;
    ThrowingType::destruct_count = 0;

    {
        fcc::AtomicQueue<ThrowingType> queue;

        // Enqueue some items
        queue.enqueue(ThrowingType(1));
        queue.enqueue(ThrowingType(2));
        queue.enqueue(ThrowingType(3));

        // Dequeue them
        while (auto item = queue.dequeue()) {
            // Items are processed
        }
    }

    // All objects should be destroyed
    EXPECT_EQ(ThrowingType::construct_count.load(), ThrowingType::destruct_count.load());
}

TEST_F(AtomicQueueTest, HighContentionStress) {
    fcc::AtomicQueue<int> queue;
    const int num_threads = 16;
    const int operations_per_thread = 1000;

    std::atomic<int> total_enqueued{0};
    std::atomic<int> total_dequeued{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Mix of enqueue and dequeue operations
            for (int i = 0; i < operations_per_thread; ++i) {
                if (i % 2 == 0) {
                    queue.enqueue(t * operations_per_thread + i);
                    total_enqueued++;
                } else {
                    if (queue.dequeue()) {
                        total_dequeued++;
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Drain any remaining items
    while (queue.dequeue()) {
        total_dequeued++;
    }

    EXPECT_EQ(total_enqueued.load(), total_dequeued.load());
}

// ==================== Type Alias Tests ====================

TEST_F(AtomicQueueTest, TaskQueueAliases) {
    fcc::AtomicTaskQueue task_queue;
    bool executed = false;

    task_queue.enqueue([&executed]() { executed = true; });

    auto task = task_queue.dequeue();
    ASSERT_TRUE(task.has_value());
    (*task)();
    EXPECT_TRUE(executed);
}

TEST_F(AtomicQueueTest, BoundedTaskQueueAliases) {
    fcc::BoundedTaskQueue16 bounded_queue;
    std::atomic<int> counter{0};

    // Fill with tasks
    for (int i = 0; i < 16; ++i) {
        EXPECT_TRUE(bounded_queue.try_enqueue([&counter]() { counter++; }));
    }

    EXPECT_TRUE(bounded_queue.full());
    EXPECT_FALSE(bounded_queue.try_enqueue([]() {})); // Should fail

    // Execute all tasks
    while (auto task = bounded_queue.try_dequeue()) {
        (*task)();
    }

    EXPECT_EQ(counter.load(), 16);
    EXPECT_TRUE(bounded_queue.empty());
}
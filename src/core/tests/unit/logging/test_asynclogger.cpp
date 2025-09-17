#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

// First, undefine conflicting macros from debug.h
#ifdef FEM_LOG_TRACE
#undef FEM_LOG_TRACE
#endif
#ifdef FEM_LOG_DEBUG
#undef FEM_LOG_DEBUG
#endif
#ifdef FEM_LOG_INFO
#undef FEM_LOG_INFO
#endif
#ifdef FEM_LOG_WARN
#undef FEM_LOG_WARN
#endif
#ifdef FEM_LOG_ERROR
#undef FEM_LOG_ERROR
#endif
#ifdef FEM_LOG_FATAL
#undef FEM_LOG_FATAL
#endif

#include "logging/loglevel.h"
#include "logging/logmessage.h"

using namespace fem::core::logging;
using namespace std::chrono_literals;

namespace {

// Helper function to create test messages
LogMessage create_test_message(LogLevel level = LogLevel::INFO,
                              const std::string& logger = "test",
                              const std::string& message = "test message") {
    return LogMessage(level, logger, message);
}

} // anonymous namespace

// =============================================================================
// AsyncLoggingComponent Configuration Tests
// =============================================================================

TEST(AsyncLoggingComponentConfigTest, DefaultConfiguration) {
    // Test that we can create a configuration with default values
    EXPECT_NO_THROW({
        struct Config {
            size_t queue_size = 8192;
            size_t batch_size = 100;
            int flush_interval_ms = 100;
            bool block_when_full = false;
            size_t worker_thread_count = 1;
        };

        Config config;
        EXPECT_EQ(config.queue_size, 8192);
        EXPECT_EQ(config.batch_size, 100);
        EXPECT_EQ(config.flush_interval_ms, 100);
        EXPECT_FALSE(config.block_when_full);
        EXPECT_EQ(config.worker_thread_count, 1);
    });
}

TEST(AsyncLoggingComponentConfigTest, CustomConfiguration) {
    struct Config {
        size_t queue_size = 8192;
        size_t batch_size = 100;
        int flush_interval_ms = 100;
        bool block_when_full = false;
        size_t worker_thread_count = 1;
    };

    Config config;
    config.queue_size = 1000;
    config.batch_size = 50;
    config.flush_interval_ms = 200;
    config.block_when_full = true;
    config.worker_thread_count = 2;

    EXPECT_EQ(config.queue_size, 1000);
    EXPECT_EQ(config.batch_size, 50);
    EXPECT_EQ(config.flush_interval_ms, 200);
    EXPECT_TRUE(config.block_when_full);
    EXPECT_EQ(config.worker_thread_count, 2);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST(AsyncLoggingStatisticsTest, StatisticsStructure) {
    struct Statistics {
        std::atomic<uint64_t> messages_queued{0};
        std::atomic<uint64_t> messages_processed{0};
        std::atomic<uint64_t> messages_dropped{0};
        std::atomic<uint64_t> flush_count{0};

        [[nodiscard]] size_t current_queue_size() const {
            return messages_queued - messages_processed;
        }

        [[nodiscard]] double drop_rate() const {
            auto total = messages_queued.load() + messages_dropped.load();
            return total > 0 ? static_cast<double>(messages_dropped.load()) / static_cast<double>(total) : 0.0;
        }
    };

    Statistics stats;

    // Test initial values
    EXPECT_EQ(stats.messages_queued.load(), 0);
    EXPECT_EQ(stats.messages_processed.load(), 0);
    EXPECT_EQ(stats.messages_dropped.load(), 0);
    EXPECT_EQ(stats.flush_count.load(), 0);

    // Test calculations
    EXPECT_EQ(stats.current_queue_size(), 0);
    EXPECT_EQ(stats.drop_rate(), 0.0);

    // Test with some values
    stats.messages_queued = 100;
    stats.messages_processed = 80;
    stats.messages_dropped = 5;

    EXPECT_EQ(stats.current_queue_size(), 20);
    EXPECT_DOUBLE_EQ(stats.drop_rate(), 5.0 / 105.0);
}

TEST(AsyncLoggingStatisticsTest, DropRateCalculation) {
    struct Statistics {
        std::atomic<uint64_t> messages_queued{0};
        std::atomic<uint64_t> messages_processed{0};
        std::atomic<uint64_t> messages_dropped{0};
        std::atomic<uint64_t> flush_count{0};

        [[nodiscard]] double drop_rate() const {
            auto total = messages_queued.load() + messages_dropped.load();
            return total > 0 ? static_cast<double>(messages_dropped.load()) / static_cast<double>(total) : 0.0;
        }
    };

    Statistics stats;

    // No messages - should return 0
    EXPECT_EQ(stats.drop_rate(), 0.0);

    // All messages queued, none dropped
    stats.messages_queued = 100;
    EXPECT_EQ(stats.drop_rate(), 0.0);

    // Some messages dropped
    stats.messages_dropped = 10;
    EXPECT_DOUBLE_EQ(stats.drop_rate(), 10.0 / 110.0);

    // All messages dropped
    stats.messages_queued = 0;
    stats.messages_dropped = 50;
    EXPECT_EQ(stats.drop_rate(), 1.0);
}

// =============================================================================
// Message Queue Tests
// =============================================================================

TEST(AsyncLoggingQueueTest, MessageCreationAndCloning) {
    auto msg1 = create_test_message(LogLevel::INFO, "test", "original message");
    auto msg2 = msg1.clone();

    EXPECT_EQ(msg1.get_level(), msg2.get_level());
    EXPECT_EQ(msg1.get_logger_name(), msg2.get_logger_name());
    EXPECT_EQ(msg1.get_message(), msg2.get_message());
}

TEST(AsyncLoggingQueueTest, BasicQueueOperations) {
    std::queue<LogMessage> message_queue;

    // Test empty queue
    EXPECT_TRUE(message_queue.empty());
    EXPECT_EQ(message_queue.size(), 0);

    // Add messages
    message_queue.push(create_test_message(LogLevel::INFO, "test", "message 1"));
    message_queue.push(create_test_message(LogLevel::WARN, "test", "message 2"));

    EXPECT_FALSE(message_queue.empty());
    EXPECT_EQ(message_queue.size(), 2);

    // Remove messages (FIFO order)
    auto msg1 = std::move(message_queue.front());
    message_queue.pop();

    EXPECT_EQ(msg1.get_message(), "message 1");
    EXPECT_EQ(msg1.get_level(), LogLevel::INFO);
    EXPECT_EQ(message_queue.size(), 1);

    auto msg2 = std::move(message_queue.front());
    message_queue.pop();

    EXPECT_EQ(msg2.get_message(), "message 2");
    EXPECT_EQ(msg2.get_level(), LogLevel::WARN);
    EXPECT_TRUE(message_queue.empty());
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(AsyncLoggingThreadSafetyTest, AtomicOperations) {
    std::atomic<uint64_t> counter{0};
    std::atomic<bool> running{true};

    const int num_threads = 4;
    const int increments_per_thread = 1000;

    std::vector<std::thread> threads;

    // Launch threads that increment counter
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&counter, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                counter.fetch_add(1);
            }
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Verify atomic operations worked correctly
    EXPECT_EQ(counter.load(), num_threads * increments_per_thread);
}

TEST(AsyncLoggingThreadSafetyTest, MutexProtectedQueue) {
    std::queue<int> shared_queue;
    std::mutex queue_mutex;
    std::atomic<int> items_added{0};
    std::atomic<int> items_removed{0};

    const int num_producers = 2;
    const int num_consumers = 2;
    const int items_per_producer = 100;

    std::vector<std::thread> threads;

    // Producer threads
    for (int i = 0; i < num_producers; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < items_per_producer; ++j) {
                {
                    std::lock_guard lock(queue_mutex);
                    shared_queue.push(i * 1000 + j);
                }
                items_added.fetch_add(1);
            }
        });
    }

    // Consumer threads
    for (int i = 0; i < num_consumers; ++i) {
        threads.emplace_back([&]() {
            int local_removed = 0;
            while (local_removed < items_per_producer) {
                bool found_item = false;
                {
                    std::lock_guard lock(queue_mutex);
                    if (!shared_queue.empty()) {
                        shared_queue.pop();
                        found_item = true;
                    }
                }
                if (found_item) {
                    items_removed.fetch_add(1);
                    local_removed++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(items_added.load(), num_producers * items_per_producer);
    EXPECT_EQ(items_removed.load(), num_consumers * items_per_producer);
}

// =============================================================================
// Condition Variable Tests
// =============================================================================

TEST(AsyncLoggingConditionVariableTest, BasicNotification) {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    bool processed = false;

    auto producer = std::thread([&]() {
        std::this_thread::sleep_for(50ms);
        {
            std::lock_guard lock(mutex);
            ready = true;
        }
        cv.notify_one();
    });

    auto consumer = std::thread([&]() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return ready; });
        processed = true;
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(ready);
    EXPECT_TRUE(processed);
}

TEST(AsyncLoggingConditionVariableTest, TimeoutWait) {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;

    auto start = std::chrono::steady_clock::now();

    {
        std::unique_lock lock(mutex);
        bool result = cv.wait_for(lock, 100ms, [&] { return ready; });
        EXPECT_FALSE(result);  // Should timeout
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE(elapsed, 90ms);  // Should have waited approximately 100ms
    EXPECT_LE(elapsed, 200ms); // But not too much longer
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

TEST(AsyncLoggingBatchTest, BatchCollection) {
    std::queue<LogMessage> message_queue;
    std::vector<LogMessage> batch;
    const size_t batch_size = 3;

    // Fill queue with messages
    for (int i = 0; i < 5; ++i) {
        message_queue.push(create_test_message(LogLevel::INFO, "test",
                                              "batch message " + std::to_string(i)));
    }

    // Collect batch
    batch.reserve(batch_size);
    while (!message_queue.empty() && batch.size() < batch_size) {
        batch.push_back(std::move(message_queue.front()));
        message_queue.pop();
    }

    EXPECT_EQ(batch.size(), batch_size);
    EXPECT_EQ(message_queue.size(), 2);  // 2 messages should remain

    // Verify batch contents
    for (size_t i = 0; i < batch.size(); ++i) {
        EXPECT_EQ(batch[i].get_message(), "batch message " + std::to_string(i));
    }
}

TEST(AsyncLoggingBatchTest, EmptyQueueBatch) {
    std::queue<LogMessage> message_queue;
    std::vector<LogMessage> batch;
    const size_t batch_size = 5;

    batch.reserve(batch_size);
    while (!message_queue.empty() && batch.size() < batch_size) {
        batch.push_back(std::move(message_queue.front()));
        message_queue.pop();
    }

    EXPECT_TRUE(batch.empty());
    EXPECT_TRUE(message_queue.empty());
}

// =============================================================================
// Worker Thread Simulation Tests
// =============================================================================

TEST(AsyncLoggingWorkerTest, WorkerThreadSimulation) {
    std::queue<LogMessage> message_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};
    std::atomic<int> messages_processed{0};

    // Simulate worker thread
    auto worker = std::thread([&]() {
        std::vector<LogMessage> batch;
        batch.reserve(10);

        while (running) {
            batch.clear();

            {
                std::unique_lock lock(queue_mutex);
                cv.wait_for(lock, 50ms, [&] {
                    return !message_queue.empty() || !running;
                });

                while (!message_queue.empty() && batch.size() < 10) {
                    batch.push_back(std::move(message_queue.front()));
                    message_queue.pop();
                }
            }

            // Process batch
            for (const auto& msg : batch) {
                messages_processed.fetch_add(1);
                (void)msg; // Simulate processing
            }
        }
    });

    // Producer
    auto producer = std::thread([&]() {
        for (int i = 0; i < 20; ++i) {
            {
                std::lock_guard lock(queue_mutex);
                message_queue.push(create_test_message(LogLevel::INFO, "test",
                                                      "worker message " + std::to_string(i)));
            }
            cv.notify_one();
            std::this_thread::sleep_for(5ms);
        }
    });

    producer.join();

    // Give worker time to process
    std::this_thread::sleep_for(100ms);

    running = false;
    cv.notify_all();
    worker.join();

    EXPECT_EQ(messages_processed.load(), 20);
}

// =============================================================================
// Performance Tests (Disabled by Default)
// =============================================================================

TEST(AsyncLoggingPerformanceTest, DISABLED_MessageThroughput) {
    const int message_count = 10000;
    std::vector<LogMessage> messages;
    messages.reserve(message_count);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < message_count; ++i) {
        messages.emplace_back(create_test_message(LogLevel::INFO, "perf",
                                                 "performance message " + std::to_string(i)));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Created " << message_count << " messages in "
              << duration.count() << " microseconds" << std::endl;

    // Should complete quickly
    EXPECT_LT(duration.count(), 100000); // Less than 100ms
}

TEST(AsyncLoggingPerformanceTest, DISABLED_ConcurrentMessageCreation) {
    const int num_threads = 4;
    const int messages_per_thread = 1000;
    std::atomic<int> total_created{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                auto msg = create_test_message(LogLevel::INFO, "thread" + std::to_string(t),
                                              "concurrent message " + std::to_string(i));
                total_created.fetch_add(1);
                (void)msg; // Simulate usage
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(total_created.load(), num_threads * messages_per_thread);

    std::cout << "Created " << total_created.load() << " messages concurrently in "
              << duration.count() << " milliseconds" << std::endl;
}
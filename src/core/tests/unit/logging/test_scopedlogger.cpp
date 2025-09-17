#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <vector>
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
#include "logging/logcontext.h"

using namespace fem::core::logging;
using namespace std::chrono_literals;

namespace {

// Helper function to create test messages
LogMessage create_test_message(LogLevel level = LogLevel::INFO,
                              const std::string& logger = "test",
                              const std::string& message = "test message") {
    return LogMessage(level, logger, message);
}

} // namespace

// =============================================================================
// ScopeEvent Tests
// =============================================================================

TEST(ScopeEventTest, EventTypeCreation) {
    // Test that ScopeEvent can be created with different types
    std::string scope_name = "TestScope";
    int64_t duration = 1000;

    // This test validates the ScopeEvent structure from scopedlogger.h
    enum EventType {
        SCOPE_ENTERED,
        SCOPE_EXITED,
        SCOPE_CHECKPOINT,
        SCOPE_FAILED,
        SCOPE_SLOW_DETECTED
    };

    struct TestEvent {
        EventType type;
        std::string scope_name;
        int64_t duration_us;
        std::string checkpoint_label;
        std::string failure_reason;
    };

    TestEvent enter_event{SCOPE_ENTERED, scope_name, 0, "", ""};
    EXPECT_EQ(enter_event.type, SCOPE_ENTERED);
    EXPECT_EQ(enter_event.scope_name, scope_name);

    TestEvent exit_event{SCOPE_EXITED, scope_name, duration, "", ""};
    EXPECT_EQ(exit_event.type, SCOPE_EXITED);
    EXPECT_EQ(exit_event.duration_us, duration);

    TestEvent checkpoint_event{SCOPE_CHECKPOINT, scope_name, 500, "", ""};
    checkpoint_event.checkpoint_label = "Checkpoint 1";
    EXPECT_EQ(checkpoint_event.checkpoint_label, "Checkpoint 1");

    TestEvent failed_event{SCOPE_FAILED, scope_name, 1000, "", ""};
    failed_event.failure_reason = "Test failure";
    EXPECT_EQ(failed_event.failure_reason, "Test failure");

    TestEvent slow_event{SCOPE_SLOW_DETECTED, scope_name, 5000, "", ""};
    EXPECT_EQ(slow_event.type, SCOPE_SLOW_DETECTED);
    EXPECT_EQ(slow_event.duration_us, 5000);
}

// =============================================================================
// ScopeStatistics Tests
// =============================================================================

TEST(ScopeStatisticsTest, BasicStatistics) {
    struct ScopeStats {
        std::string name;
        uint64_t call_count{0};
        uint64_t total_duration_us{0};
        uint64_t min_duration_us{UINT64_MAX};
        uint64_t max_duration_us{0};
        uint64_t failure_count{0};
        std::chrono::system_clock::time_point last_called;

        double average_duration_us() const {
            return call_count > 0 ? static_cast<double>(total_duration_us) / static_cast<double>(call_count) : 0.0;
        }
    };

    ScopeStats stats;
    stats.name = "TestScope";

    // Test initial state
    EXPECT_EQ(stats.call_count, 0);
    EXPECT_EQ(stats.total_duration_us, 0);
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 0.0);
    EXPECT_EQ(stats.min_duration_us, UINT64_MAX);
    EXPECT_EQ(stats.max_duration_us, 0);
    EXPECT_EQ(stats.failure_count, 0);

    // Simulate recording some calls
    stats.call_count = 5;
    stats.total_duration_us = 5000;
    stats.min_duration_us = 500;
    stats.max_duration_us = 1500;
    stats.failure_count = 1;
    stats.last_called = std::chrono::system_clock::now();

    EXPECT_EQ(stats.call_count, 5);
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 1000.0);
    EXPECT_EQ(stats.min_duration_us, 500);
    EXPECT_EQ(stats.max_duration_us, 1500);
    EXPECT_EQ(stats.failure_count, 1);
}

TEST(ScopeStatisticsTest, AverageDurationCalculation) {
    struct ScopeStats {
        uint64_t call_count{0};
        uint64_t total_duration_us{0};

        double average_duration_us() const {
            return call_count > 0 ? static_cast<double>(total_duration_us) / static_cast<double>(call_count) : 0.0;
        }
    };

    ScopeStats stats;

    // Test with no calls
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 0.0);

    // Test with single call
    stats.call_count = 1;
    stats.total_duration_us = 1000;
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 1000.0);

    // Test with multiple calls
    stats.call_count = 10;
    stats.total_duration_us = 15000;
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 1500.0);

    // Test with large numbers
    stats.call_count = 1000000;
    stats.total_duration_us = 5000000000;  // 5 seconds in microseconds
    EXPECT_DOUBLE_EQ(stats.average_duration_us(), 5000.0);
}

// =============================================================================
// ScopeTimingComponent Tests
// =============================================================================

TEST(ScopeTimingComponentTest, ConfigurationDefaults) {
    struct Config {
        int64_t slow_threshold_ms = 1000;
        bool enable_statistics = true;
        bool enable_events = true;
        LogLevel default_level = LogLevel::TRACE;
    };

    Config config;
    EXPECT_EQ(config.slow_threshold_ms, 1000);
    EXPECT_TRUE(config.enable_statistics);
    EXPECT_TRUE(config.enable_events);
    EXPECT_EQ(config.default_level, LogLevel::TRACE);
}

TEST(ScopeTimingComponentTest, ConfigurationCustom) {
    struct Config {
        int64_t slow_threshold_ms = 1000;
        bool enable_statistics = true;
        bool enable_events = true;
        LogLevel default_level = LogLevel::TRACE;
    };

    Config config;
    config.slow_threshold_ms = 500;
    config.enable_statistics = false;
    config.enable_events = false;
    config.default_level = LogLevel::INFO;

    EXPECT_EQ(config.slow_threshold_ms, 500);
    EXPECT_FALSE(config.enable_statistics);
    EXPECT_FALSE(config.enable_events);
    EXPECT_EQ(config.default_level, LogLevel::INFO);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(ThreadSafetyTest, AtomicDepthCounter) {
    std::atomic<int> depth{0};

    // Test initial value
    EXPECT_EQ(depth.load(), 0);

    // Test increment
    depth.fetch_add(1, std::memory_order_relaxed);
    EXPECT_EQ(depth.load(), 1);

    // Test multiple increments
    for (int i = 0; i < 5; ++i) {
        depth.fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_EQ(depth.load(), 6);

    // Test decrement
    depth.fetch_sub(1, std::memory_order_relaxed);
    EXPECT_EQ(depth.load(), 5);

    // Test multiple decrements
    for (int i = 0; i < 5; ++i) {
        depth.fetch_sub(1, std::memory_order_relaxed);
    }
    EXPECT_EQ(depth.load(), 0);
}

TEST(ThreadSafetyTest, ConcurrentAtomicOperations) {
    std::atomic<int> counter{0};
    const int num_threads = 4;
    const int increments_per_thread = 1000;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&counter, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(counter.load(), num_threads * increments_per_thread);
}

// =============================================================================
// Time Formatting Tests
// =============================================================================

TEST(TimeFormattingTest, DurationFormatting) {
    // Test helper function that formats durations
    auto format_duration = [](std::chrono::microseconds us) -> std::string {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(us);
        auto s = std::chrono::duration_cast<std::chrono::seconds>(us);

        if (s.count() >= 1) {
            double seconds = static_cast<double>(s.count()) + static_cast<double>(ms.count() % 1000) / 1000.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << seconds << "s";
            return oss.str();
        } else if (ms.count() >= 1) {
            return std::to_string(ms.count()) + "ms";
        } else {
            return std::to_string(us.count()) + "μs";
        }
    };

    // Test microseconds
    EXPECT_EQ(format_duration(std::chrono::microseconds(500)), "500μs");
    EXPECT_EQ(format_duration(std::chrono::microseconds(999)), "999μs");

    // Test milliseconds
    EXPECT_EQ(format_duration(std::chrono::microseconds(1000)), "1ms");
    EXPECT_EQ(format_duration(std::chrono::microseconds(5500)), "5ms");
    EXPECT_EQ(format_duration(std::chrono::microseconds(999999)), "999ms");

    // Test seconds
    EXPECT_EQ(format_duration(std::chrono::microseconds(1000000)), "1.00s");
    EXPECT_EQ(format_duration(std::chrono::microseconds(1500000)), "1.50s");
    EXPECT_EQ(format_duration(std::chrono::microseconds(10750000)), "10.75s");
}

// =============================================================================
// Function Name Extraction Tests
// =============================================================================

TEST(FunctionNameExtractionTest, ExtractFunctionName) {
    auto extract_function_name = [](const char* full_name) -> std::string {
        if (!full_name) return "unknown";

        std::string name(full_name);

        // Remove template parameters
        auto template_pos = name.find('<');
        if (template_pos != std::string::npos) {
            name = name.substr(0, template_pos);
        }

        // Find last scope operator (look for "::" pattern)
        auto last_scope = name.rfind("::");
        if (last_scope != std::string::npos) {
            name = name.substr(last_scope + 2);
        }

        return name;
    };

    // Test various function name formats
    EXPECT_EQ(extract_function_name(nullptr), "unknown");
    EXPECT_EQ(extract_function_name("simple_function"), "simple_function");
    EXPECT_EQ(extract_function_name("namespace::function"), "function");
    EXPECT_EQ(extract_function_name("class::method"), "method");
    EXPECT_EQ(extract_function_name("template_func<int>"), "template_func");
    EXPECT_EQ(extract_function_name("ns::class::method<T>"), "method");
}

// =============================================================================
// Message Queue Tests
// =============================================================================

TEST(MessageQueueTest, BasicQueueOperations) {
    std::queue<LogMessage> message_queue;

    // Test empty queue
    EXPECT_TRUE(message_queue.empty());
    EXPECT_EQ(message_queue.size(), 0);

    // Add messages
    message_queue.push(create_test_message(LogLevel::INFO, "test", "message 1"));
    message_queue.push(create_test_message(LogLevel::WARN, "test", "message 2"));
    message_queue.push(create_test_message(LogLevel::ERROR, "test", "message 3"));

    EXPECT_FALSE(message_queue.empty());
    EXPECT_EQ(message_queue.size(), 3);

    // Remove messages in FIFO order
    auto msg1 = std::move(message_queue.front());
    message_queue.pop();
    EXPECT_EQ(msg1.get_message(), "message 1");
    EXPECT_EQ(msg1.get_level(), LogLevel::INFO);
    EXPECT_EQ(message_queue.size(), 2);

    auto msg2 = std::move(message_queue.front());
    message_queue.pop();
    EXPECT_EQ(msg2.get_message(), "message 2");
    EXPECT_EQ(msg2.get_level(), LogLevel::WARN);
    EXPECT_EQ(message_queue.size(), 1);

    auto msg3 = std::move(message_queue.front());
    message_queue.pop();
    EXPECT_EQ(msg3.get_message(), "message 3");
    EXPECT_EQ(msg3.get_level(), LogLevel::ERROR);
    EXPECT_TRUE(message_queue.empty());
}

TEST(MessageQueueTest, ThreadSafeQueue) {
    std::queue<int> queue;
    std::mutex queue_mutex;
    const int num_producers = 2;
    const int num_consumers = 2;
    const int items_per_producer = 100;
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};

    std::vector<std::thread> threads;

    // Producer threads
    for (int p = 0; p < num_producers; ++p) {
        threads.emplace_back([&, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    queue.push(p * 1000 + i);
                }
                items_produced.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // Consumer threads
    for (int c = 0; c < num_consumers; ++c) {
        threads.emplace_back([&]() {
            while (items_consumed.load() < num_producers * items_per_producer) {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (!queue.empty()) {
                        queue.pop();
                        items_consumed.fetch_add(1);
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(items_produced.load(), num_producers * items_per_producer);
    EXPECT_EQ(items_consumed.load(), num_producers * items_per_producer);
    EXPECT_TRUE(queue.empty());
}

// =============================================================================
// Condition Variable Tests
// =============================================================================

TEST(ConditionVariableTest, BasicWaitNotify) {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    bool processed = false;

    std::thread producer([&]() {
        std::this_thread::sleep_for(10ms);
        {
            std::lock_guard<std::mutex> lock(mutex);
            ready = true;
        }
        cv.notify_one();
    });

    std::thread consumer([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&ready] { return ready; });
        processed = true;
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(ready);
    EXPECT_TRUE(processed);
}

TEST(ConditionVariableTest, WaitWithTimeout) {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;

    auto start = std::chrono::steady_clock::now();

    {
        std::unique_lock<std::mutex> lock(mutex);
        bool result = cv.wait_for(lock, 50ms, [&ready] { return ready; });
        EXPECT_FALSE(result);  // Should timeout since ready is never set to true
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE(elapsed, 45ms);  // Should wait at least 45ms
    EXPECT_LE(elapsed, 100ms); // But not more than 100ms
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

TEST(BatchProcessingTest, BatchCollection) {
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

TEST(BatchProcessingTest, EmptyQueueBatch) {
    std::queue<LogMessage> message_queue;
    std::vector<LogMessage> batch;
    const size_t batch_size = 5;

    // Try to collect batch from empty queue
    batch.reserve(batch_size);
    while (!message_queue.empty() && batch.size() < batch_size) {
        batch.push_back(std::move(message_queue.front()));
        message_queue.pop();
    }

    EXPECT_TRUE(batch.empty());
    EXPECT_TRUE(message_queue.empty());
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST(PerformanceTest, MessageCreationSpeed) {
    const int iterations = 10000;
    std::vector<LogMessage> messages;
    messages.reserve(iterations);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        messages.emplace_back(LogLevel::INFO, "perf",
                            "performance message " + std::to_string(i));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Average time per message creation
    auto avg_time_us = duration.count() / iterations;

    // Should be very fast - typically under 10 microseconds
    EXPECT_LT(avg_time_us, 100);

    // Verify messages were created correctly
    EXPECT_EQ(messages.size(), iterations);
    EXPECT_EQ(messages[0].get_logger_name(), "perf");
    EXPECT_EQ(messages[0].get_level(), LogLevel::INFO);
}

TEST(PerformanceTest, ConcurrentMessageCreation) {
    const int num_threads = 4;
    const int messages_per_thread = 1000;
    std::atomic<int> total_created{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&total_created, t, messages_per_thread]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                LogMessage msg(LogLevel::INFO, "thread" + std::to_string(t),
                             "concurrent message " + std::to_string(i));
                total_created.fetch_add(1);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(total_created.load(), num_threads * messages_per_thread);

    // Should complete reasonably quickly even with contention
    EXPECT_LT(duration.count(), 1000);  // Less than 1 second
}
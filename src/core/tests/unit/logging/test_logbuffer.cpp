#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>

#include "logging/logbuffer.h"
#include "logging/logmessage.h"
#include "logging/loglevel.h"

using namespace fem::core::logging;

namespace {

// Helper function to create a test log message
LogMessage create_test_message(LogLevel level = LogLevel::INFO,
                              const std::string& logger = "test",
                              const std::string& message = "test message") {
    return LogMessage(level, logger, message);
}

// Helper function to create messages with different timestamps (unused in current tests)
// LogMessage create_message_with_delay(LogLevel level, const std::string& message, int delay_ms = 0) {
//     if (delay_ms > 0) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
//     }
//     return LogMessage(level, "test", message);
// }

} // anonymous namespace

// =============================================================================
// FifoBuffer Tests
// =============================================================================

class FifoBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer_ = std::make_unique<FifoBuffer>(3); // Small capacity for testing
    }

    std::unique_ptr<FifoBuffer> buffer_;
};

TEST_F(FifoBufferTest, InitialState) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_EQ(buffer_->capacity(), 3);
}

TEST_F(FifoBufferTest, PushSingleMessage) {
    auto msg = create_test_message();
    EXPECT_TRUE(buffer_->push(std::move(msg)));

    EXPECT_FALSE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 1);
}

TEST_F(FifoBufferTest, PushToCapacity) {
    // Fill buffer to capacity
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "message " + std::to_string(i));
        EXPECT_TRUE(buffer_->push(std::move(msg)));
    }

    EXPECT_FALSE(buffer_->empty());
    EXPECT_TRUE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 3);

    // Should reject additional messages
    auto extra_msg = create_test_message();
    EXPECT_FALSE(buffer_->push(std::move(extra_msg)));
    EXPECT_EQ(buffer_->size(), 3);
}

TEST_F(FifoBufferTest, PopMessage) {
    auto original_msg = create_test_message(LogLevel::WARN, "test", "pop test");
    buffer_->push(std::move(original_msg));

    auto popped = buffer_->pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->get_level(), LogLevel::WARN);
    EXPECT_EQ(popped->get_message(), "pop test");

    EXPECT_TRUE(buffer_->empty());
    EXPECT_EQ(buffer_->size(), 0);
}

TEST_F(FifoBufferTest, PopFromEmptyBuffer) {
    auto result = buffer_->pop();
    EXPECT_FALSE(result.has_value());
}

TEST_F(FifoBufferTest, FifoOrdering) {
    // Push messages in order
    std::vector<std::string> messages = {"first", "second", "third"};
    for (const auto& msg_text : messages) {
        auto msg = create_test_message(LogLevel::INFO, "test", msg_text);
        EXPECT_TRUE(buffer_->push(std::move(msg)));
    }

    // Pop should return messages in FIFO order
    for (const auto& expected : messages) {
        auto popped = buffer_->pop();
        ASSERT_TRUE(popped.has_value());
        EXPECT_EQ(popped->get_message(), expected);
    }
}

TEST_F(FifoBufferTest, DrainBuffer) {
    // Add multiple messages
    std::vector<std::string> messages = {"msg1", "msg2", "msg3"};
    for (const auto& msg_text : messages) {
        auto msg = create_test_message(LogLevel::INFO, "test", msg_text);
        buffer_->push(std::move(msg));
    }

    auto drained = buffer_->drain();
    EXPECT_EQ(drained.size(), 3);
    EXPECT_TRUE(buffer_->empty());

    // Check order is preserved
    for (size_t i = 0; i < messages.size(); ++i) {
        EXPECT_EQ(drained[i].get_message(), messages[i]);
    }
}

TEST_F(FifoBufferTest, Clear) {
    // Add messages
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message();
        buffer_->push(std::move(msg));
    }

    EXPECT_EQ(buffer_->size(), 3);
    buffer_->clear();
    EXPECT_TRUE(buffer_->empty());
    EXPECT_EQ(buffer_->size(), 0);
}

TEST_F(FifoBufferTest, ThreadSafety) {
    const int num_threads = 4;
    const int messages_per_thread = 10;
    std::vector<std::thread> threads;

    // Use larger buffer for thread safety test
    buffer_ = std::make_unique<FifoBuffer>(num_threads * messages_per_thread);

    // Launch producer threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, messages_per_thread]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                auto msg = create_test_message(LogLevel::INFO, "thread" + std::to_string(t),
                                             "message" + std::to_string(i));
                buffer_->push(std::move(msg));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(buffer_->size(), num_threads * messages_per_thread);
}

// =============================================================================
// CircularBuffer Tests
// =============================================================================

class CircularBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer_ = std::make_unique<CircularBuffer>(3); // Small capacity for testing
    }

    std::unique_ptr<CircularBuffer> buffer_;
};

TEST_F(CircularBufferTest, InitialState) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_EQ(buffer_->capacity(), 3);
    EXPECT_EQ(buffer_->get_overwrite_count(), 0);
}

TEST_F(CircularBufferTest, PushAlwaysSucceeds) {
    // Fill beyond capacity
    for (int i = 0; i < 5; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "message " + std::to_string(i));
        EXPECT_TRUE(buffer_->push(std::move(msg))); // Always succeeds
    }

    EXPECT_EQ(buffer_->size(), 3); // Size limited by capacity
    EXPECT_TRUE(buffer_->full());
    EXPECT_EQ(buffer_->get_overwrite_count(), 2); // 2 messages overwritten
}

TEST_F(CircularBufferTest, OverwriteOldestMessages) {
    // Fill buffer
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "old" + std::to_string(i));
        buffer_->push(std::move(msg));
    }

    // Add more messages, should overwrite oldest
    for (int i = 0; i < 2; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "new" + std::to_string(i));
        buffer_->push(std::move(msg));
    }

    // Should have newest 3 messages: old2, new0, new1
    auto drained = buffer_->drain();
    EXPECT_EQ(drained.size(), 3);
    EXPECT_EQ(drained[0].get_message(), "old2");
    EXPECT_EQ(drained[1].get_message(), "new0");
    EXPECT_EQ(drained[2].get_message(), "new1");
}

TEST_F(CircularBufferTest, PopMaintainsOrder) {
    // Add messages
    std::vector<std::string> messages = {"first", "second", "third"};
    for (const auto& msg_text : messages) {
        auto msg = create_test_message(LogLevel::INFO, "test", msg_text);
        buffer_->push(std::move(msg));
    }

    // Pop in order
    for (const auto& expected : messages) {
        auto popped = buffer_->pop();
        ASSERT_TRUE(popped.has_value());
        EXPECT_EQ(popped->get_message(), expected);
    }

    EXPECT_TRUE(buffer_->empty());
}

TEST_F(CircularBufferTest, Clear) {
    // Fill buffer
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message();
        buffer_->push(std::move(msg));
    }

    buffer_->clear();
    EXPECT_TRUE(buffer_->empty());
    EXPECT_EQ(buffer_->size(), 0);
    // Note: overwrite count is not reset by clear
}

// =============================================================================
// PriorityBuffer Tests
// =============================================================================

class PriorityBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer_ = std::make_unique<PriorityBuffer>(3); // Small capacity for testing
    }

    std::unique_ptr<PriorityBuffer> buffer_;
};

TEST_F(PriorityBufferTest, InitialState) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_EQ(buffer_->capacity(), 3);
}

TEST_F(PriorityBufferTest, SortsByPriority) {
    // Add messages in random order
    auto info_msg = create_test_message(LogLevel::INFO, "test", "info");
    auto error_msg = create_test_message(LogLevel::ERROR, "test", "error");
    auto debug_msg = create_test_message(LogLevel::DEBUG, "test", "debug");

    buffer_->push(std::move(info_msg));
    buffer_->push(std::move(error_msg));
    buffer_->push(std::move(debug_msg));

    // Pop should return highest priority first (ERROR > INFO > DEBUG)
    auto first = buffer_->pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->get_level(), LogLevel::ERROR);

    auto second = buffer_->pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->get_level(), LogLevel::INFO);

    auto third = buffer_->pop();
    ASSERT_TRUE(third.has_value());
    EXPECT_EQ(third->get_level(), LogLevel::DEBUG);
}

TEST_F(PriorityBufferTest, RejectsLowPriorityWhenFull) {
    // Fill with medium priority messages
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "info" + std::to_string(i));
        EXPECT_TRUE(buffer_->push(std::move(msg)));
    }

    // Should reject lower priority
    auto debug_msg = create_test_message(LogLevel::DEBUG, "test", "debug");
    EXPECT_FALSE(buffer_->push(std::move(debug_msg)));

    // Should accept higher priority (displacing lowest)
    auto error_msg = create_test_message(LogLevel::ERROR, "test", "error");
    EXPECT_TRUE(buffer_->push(std::move(error_msg)));

    EXPECT_EQ(buffer_->size(), 3);
}

TEST_F(PriorityBufferTest, DrainReturnsInPriorityOrder) {
    // Add messages with different priorities
    auto trace_msg = create_test_message(LogLevel::TRACE, "test", "trace");
    auto fatal_msg = create_test_message(LogLevel::FATAL, "test", "fatal");
    auto warn_msg = create_test_message(LogLevel::WARN, "test", "warn");

    buffer_->push(std::move(trace_msg));
    buffer_->push(std::move(fatal_msg));
    buffer_->push(std::move(warn_msg));

    auto drained = buffer_->drain();
    EXPECT_EQ(drained.size(), 3);

    // Should be in priority order: FATAL > WARN > TRACE
    EXPECT_EQ(drained[0].get_level(), LogLevel::FATAL);
    EXPECT_EQ(drained[1].get_level(), LogLevel::WARN);
    EXPECT_EQ(drained[2].get_level(), LogLevel::TRACE);

    EXPECT_TRUE(buffer_->empty());
}

// =============================================================================
// TimeWindowBuffer Tests
// =============================================================================

class TimeWindowBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Very short window for testing
        buffer_ = std::make_unique<TimeWindowBuffer>(std::chrono::seconds(1), 100);
    }

    std::unique_ptr<TimeWindowBuffer> buffer_;
};

TEST_F(TimeWindowBufferTest, InitialState) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_EQ(buffer_->capacity(), 100);
}

TEST_F(TimeWindowBufferTest, AcceptsRecentMessages) {
    auto msg = create_test_message();
    EXPECT_TRUE(buffer_->push(std::move(msg)));
    EXPECT_EQ(buffer_->size(), 1);
}

TEST_F(TimeWindowBufferTest, RemovesExpiredMessages) {
    // Add a message
    auto msg = create_test_message();
    buffer_->push(std::move(msg));
    EXPECT_EQ(buffer_->size(), 1);

    // Wait for expiration
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // Size check should trigger cleanup
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_TRUE(buffer_->empty());
}

TEST_F(TimeWindowBufferTest, PopRemovesExpiredFirst) {
    // Add a message
    auto old_msg = create_test_message(LogLevel::INFO, "test", "old");
    buffer_->push(std::move(old_msg));

    // Wait a bit, then add newer message
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto new_msg = create_test_message(LogLevel::INFO, "test", "new");
    buffer_->push(std::move(new_msg));

    // Pop should only return the new message (old one expired)
    auto popped = buffer_->pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->get_message(), "new");

    EXPECT_TRUE(buffer_->empty());
}

TEST_F(TimeWindowBufferTest, SetWindow) {
    buffer_->set_window(std::chrono::seconds(2));

    auto msg = create_test_message();
    buffer_->push(std::move(msg));

    // Should still be present after 1.5 seconds with 2-second window
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    EXPECT_EQ(buffer_->size(), 1);
}

TEST_F(TimeWindowBufferTest, DrainRemovesExpired) {
    // Add messages at different times
    auto old_msg = create_test_message(LogLevel::INFO, "test", "old");
    buffer_->push(std::move(old_msg));

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    auto new_msg = create_test_message(LogLevel::INFO, "test", "new");
    buffer_->push(std::move(new_msg));

    auto drained = buffer_->drain();
    EXPECT_EQ(drained.size(), 1); // Only new message should remain
    EXPECT_EQ(drained[0].get_message(), "new");
}

// =============================================================================
// CompressionBuffer Tests
// =============================================================================

class CompressionBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer_ = std::make_unique<CompressionBuffer>(10);
    }

    std::unique_ptr<CompressionBuffer> buffer_;
};

TEST_F(CompressionBufferTest, InitialState) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_FALSE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 0);
    EXPECT_EQ(buffer_->capacity(), 10);
}

TEST_F(CompressionBufferTest, CompressesDuplicateMessages) {
    // Add the same message multiple times
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "duplicate message");
        EXPECT_TRUE(buffer_->push(std::move(msg)));
    }

    // Should only have one unique entry
    EXPECT_EQ(buffer_->size(), 1);
}

TEST_F(CompressionBufferTest, PopNotSupported) {
    auto msg = create_test_message();
    buffer_->push(std::move(msg));

    // Pop should return nullopt for compression buffer
    auto result = buffer_->pop();
    EXPECT_FALSE(result.has_value());
}

TEST_F(CompressionBufferTest, DrainShowsCompressionInfo) {
    // Add duplicate messages
    for (int i = 0; i < 3; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "repeated message");
        buffer_->push(std::move(msg));
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Slight delay for timestamp
    }

    auto drained = buffer_->drain();
    EXPECT_EQ(drained.size(), 1);

    // Message should indicate repetition
    const auto& compressed_msg = drained[0];
    EXPECT_TRUE(compressed_msg.get_message().find("repeated 3 times") != std::string::npos);
}

TEST_F(CompressionBufferTest, DifferentMessagesNotCompressed) {
    auto msg1 = create_test_message(LogLevel::INFO, "test", "message 1");
    auto msg2 = create_test_message(LogLevel::INFO, "test", "message 2");
    auto msg3 = create_test_message(LogLevel::ERROR, "test", "message 1"); // Different level

    buffer_->push(std::move(msg1));
    buffer_->push(std::move(msg2));
    buffer_->push(std::move(msg3));

    EXPECT_EQ(buffer_->size(), 3); // All different
}

TEST_F(CompressionBufferTest, RespectsCapacity) {
    // Fill to capacity with unique messages
    for (int i = 0; i < 10; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "test", "message " + std::to_string(i));
        EXPECT_TRUE(buffer_->push(std::move(msg)));
    }

    // Should reject additional unique message
    auto extra_msg = create_test_message(LogLevel::INFO, "test", "extra message");
    EXPECT_FALSE(buffer_->push(std::move(extra_msg)));

    EXPECT_TRUE(buffer_->full());
    EXPECT_EQ(buffer_->size(), 10);
}

// =============================================================================
// LogBuffer Interface Tests
// =============================================================================

class LogBufferInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        fifo_buffer_ = std::make_unique<FifoBuffer>(5);
        circular_buffer_ = std::make_unique<CircularBuffer>(5);
        priority_buffer_ = std::make_unique<PriorityBuffer>(5);
    }

    std::unique_ptr<LogBuffer> fifo_buffer_;
    std::unique_ptr<LogBuffer> circular_buffer_;
    std::unique_ptr<LogBuffer> priority_buffer_;
};

TEST_F(LogBufferInterfaceTest, PolymorphicBehavior) {
    std::vector<std::unique_ptr<LogBuffer>> buffers;
    buffers.push_back(std::make_unique<FifoBuffer>(3));
    buffers.push_back(std::make_unique<CircularBuffer>(3));
    buffers.push_back(std::make_unique<PriorityBuffer>(3));

    for (auto& buffer : buffers) {
        EXPECT_TRUE(buffer->empty());
        EXPECT_EQ(buffer->capacity(), 3);

        auto msg = create_test_message();
        EXPECT_TRUE(buffer->push(std::move(msg)));
        EXPECT_FALSE(buffer->empty());
        EXPECT_EQ(buffer->size(), 1);

        buffer->clear();
        EXPECT_TRUE(buffer->empty());
    }
}

TEST_F(LogBufferInterfaceTest, MoveSemantics) {
    auto msg = create_test_message(LogLevel::INFO, "test", "move test");
    auto original_message = msg.get_message();

    EXPECT_TRUE(fifo_buffer_->push(std::move(msg)));

    auto popped = fifo_buffer_->pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->get_message(), original_message);
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

TEST(LogBufferEdgeCasesTest, EmptyBufferOperations) {
    FifoBuffer buffer(5);

    EXPECT_FALSE(buffer.pop().has_value());
    EXPECT_TRUE(buffer.drain().empty());

    // Clear on empty buffer should be safe
    buffer.clear();
    EXPECT_TRUE(buffer.empty());
}

TEST(LogBufferEdgeCasesTest, ZeroCapacityBuffer) {
    FifoBuffer buffer(0);

    EXPECT_TRUE(buffer.empty());
    EXPECT_TRUE(buffer.full());
    EXPECT_EQ(buffer.capacity(), 0);

    auto msg = create_test_message();
    EXPECT_FALSE(buffer.push(std::move(msg)));
}

TEST(LogBufferEdgeCasesTest, SingleCapacityBuffer) {
    FifoBuffer buffer(1);

    auto msg1 = create_test_message(LogLevel::INFO, "test", "first");
    EXPECT_TRUE(buffer.push(std::move(msg1)));
    EXPECT_TRUE(buffer.full());

    auto msg2 = create_test_message(LogLevel::INFO, "test", "second");
    EXPECT_FALSE(buffer.push(std::move(msg2))); // Should fail

    auto popped = buffer.pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->get_message(), "first");

    // Now should be able to add another
    auto msg3 = create_test_message(LogLevel::INFO, "test", "third");
    EXPECT_TRUE(buffer.push(std::move(msg3)));
}

// =============================================================================
// Performance and Stress Tests
// =============================================================================

// Keep performance test disabled by default for CI
TEST(LogBufferPerformanceTest, DISABLED_HighVolumeInserts) {
    const size_t num_messages = 100000;
    FifoBuffer buffer(num_messages);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_messages; ++i) {
        auto msg = create_test_message(LogLevel::INFO, "perf", "message " + std::to_string(i));
        (void)buffer.push(std::move(msg)); // Cast to void to ignore [[nodiscard]]
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(buffer.size(), num_messages);

    // Log performance result (adjust threshold as needed)
    std::cout << "Inserted " << num_messages << " messages in "
              << duration.count() << " microseconds" << std::endl;
}

TEST(LogBufferConcurrencyTest, MultiThreadedAccess) {
    const int num_threads = 4;  // Reduced for stability
    const int messages_per_thread = 500;  // Reduced for faster execution
    FifoBuffer buffer(num_threads * messages_per_thread);

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<int> consumed_count{0};

    // Launch producer threads
    for (int t = 0; t < num_threads; ++t) {
        producers.emplace_back([&buffer, t, messages_per_thread]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                auto msg = create_test_message(LogLevel::INFO,
                                             "thread" + std::to_string(t),
                                             "message" + std::to_string(i));
                while (!buffer.push(std::move(msg))) {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Launch consumer threads
    for (int t = 0; t < num_threads / 2; ++t) {
        consumers.emplace_back([&buffer, &consumed_count, num_threads, messages_per_thread]() {
            const int target_count = (num_threads * messages_per_thread) / 2;
            while (consumed_count.load() < target_count) {
                auto msg = buffer.pop();
                if (msg.has_value()) {
                    consumed_count.fetch_add(1);
                }
                std::this_thread::yield();
            }
        });
    }

    // Wait for all producers
    for (auto& t : producers) {
        t.join();
    }

    // Signal consumers to stop and wait
    const int target_count = (num_threads * messages_per_thread) / 2;
    consumed_count.store(target_count);
    for (auto& t : consumers) {
        t.join();
    }

    // Verify some messages remain (should be roughly half)
    const int expected_remaining = num_threads * messages_per_thread - target_count;
    EXPECT_GT(buffer.size(), 0);
    EXPECT_LE(buffer.size(), static_cast<size_t>(num_threads * messages_per_thread));

    // Log the results for analysis
    std::cout << "Concurrency test results:" << std::endl;
    std::cout << "  Total messages produced: " << num_threads * messages_per_thread << std::endl;
    std::cout << "  Messages consumed: " << consumed_count.load() << std::endl;
    std::cout << "  Messages remaining in buffer: " << buffer.size() << std::endl;
    std::cout << "  Expected remaining: ~" << expected_remaining << std::endl;
}

TEST(LogBufferConcurrencyTest, MultipleBufferTypesConcurrency) {
    const int num_threads = 2;
    const int messages_per_thread = 100;

    // Test different buffer types under concurrent access
    std::vector<std::unique_ptr<LogBuffer>> buffers;
    buffers.push_back(std::make_unique<FifoBuffer>(1000));
    buffers.push_back(std::make_unique<CircularBuffer>(50));  // Smaller to test overwrites
    buffers.push_back(std::make_unique<PriorityBuffer>(1000));

    for (size_t buffer_idx = 0; buffer_idx < buffers.size(); ++buffer_idx) {
        auto& buffer = buffers[buffer_idx];
        std::vector<std::thread> threads;
        std::atomic<int> error_count{0};

        // Launch producer threads
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&buffer, &error_count, t, messages_per_thread]() {
                try {
                    for (int i = 0; i < messages_per_thread; ++i) {
                        LogLevel level = static_cast<LogLevel>(i % 6);  // Vary log levels
                        auto msg = create_test_message(level,
                                                     "thread" + std::to_string(t),
                                                     "message" + std::to_string(i));
                        // For CircularBuffer, push always succeeds; for others, retry
                        while (!buffer->push(std::move(msg))) {
                            std::this_thread::sleep_for(std::chrono::microseconds(1));
                            msg = create_test_message(level,
                                                     "thread" + std::to_string(t),
                                                     "message" + std::to_string(i));
                        }
                    }
                } catch (const std::exception& e) {
                    error_count.fetch_add(1);
                }
            });
        }

        // Launch consumer thread
        threads.emplace_back([&buffer, &error_count]() {
            try {
                int consumed = 0;
                while (consumed < 50) {  // Consume some messages
                    auto msg = buffer->pop();
                    if (msg.has_value()) {
                        consumed++;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            } catch (const std::exception& e) {
                error_count.fetch_add(1);
            }
        });

        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }

        // Verify no errors occurred
        EXPECT_EQ(error_count.load(), 0) << "Buffer type " << buffer_idx << " had errors";

        // Buffer should still be functional
        EXPECT_LE(buffer->size(), buffer->capacity());

        std::cout << "Buffer type " << buffer_idx << " concurrency test passed. "
                  << "Final size: " << buffer->size() << "/" << buffer->capacity() << std::endl;
    }
}
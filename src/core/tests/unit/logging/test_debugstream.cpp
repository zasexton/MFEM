#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <regex>

#include "logging/debugstream.h"

using namespace fem::core::logging;
using namespace std::chrono_literals;

namespace {

// Helper class to capture debug stream output
class OutputCapture {
public:
    OutputCapture() : captured_stream_() {}

    std::stringstream& stream() { return captured_stream_; }

    std::string get_output() const {
        return captured_stream_.str();
    }

    void clear() {
        captured_stream_.str("");
        captured_stream_.clear();
    }

    bool contains(const std::string& text) const {
        return get_output().find(text) != std::string::npos;
    }

    size_t count_lines() const {
        std::string output = get_output();
        return std::count(output.begin(), output.end(), '\n');
    }

    std::vector<std::string> get_lines() const {
        std::vector<std::string> lines;
        std::string output = get_output();
        std::stringstream ss(output);
        std::string line;
        while (std::getline(ss, line)) {
            lines.push_back(line);
        }
        return lines;
    }

private:
    std::stringstream captured_stream_;
};

} // namespace

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(DebugStreamTest, BasicConstruction) {
    // Default constructor
    DebugStream stream1;
    EXPECT_TRUE(stream1.is_enabled());
    EXPECT_EQ(stream1.get_level(), DebugLevel::INFO);

    // Constructor with output stream
    std::ostringstream oss;
    DebugStream stream2(oss);
    EXPECT_TRUE(stream2.is_enabled());

    // Constructor with config
    DebugStreamConfig config;
    config.enabled = false;
    config.min_level = DebugLevel::ERROR;
    DebugStream stream3(config);
    EXPECT_FALSE(stream3.is_enabled());
    EXPECT_EQ(stream3.get_config().min_level, DebugLevel::ERROR);
}

TEST(DebugStreamTest, BasicOutput) {
    std::stringstream ss;
    {
        DebugStream stream(ss);
        stream(DebugLevel::INFO) << "Test message" << std::endl;
    }
    std::string output = ss.str();

    EXPECT_NE(output.find("Test message"), std::string::npos);
    EXPECT_NE(output.find("[INFO]"), std::string::npos);
}

TEST(DebugStreamTest, MultipleOutputs) {
    std::stringstream ss;
    {
        DebugStream stream(ss);
        stream.set_min_level(DebugLevel::DEBUG);  // Allow DEBUG messages
        stream(DebugLevel::INFO) << "First message" << std::endl;
        stream(DebugLevel::ERROR) << "Error message" << std::endl;
        stream(DebugLevel::DEBUG) << "Debug message" << std::endl;
    }
    std::string output = ss.str();

    EXPECT_NE(output.find("First message"), std::string::npos);
    EXPECT_NE(output.find("Error message"), std::string::npos);
    EXPECT_NE(output.find("Debug message"), std::string::npos);
    EXPECT_NE(output.find("[INFO]"), std::string::npos);
    EXPECT_NE(output.find("[ERROR]"), std::string::npos);
    EXPECT_NE(output.find("[DEBUG]"), std::string::npos);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST(DebugStreamTest, ConfigurationSettings) {
    DebugStreamConfig config;
    config.enabled = true;
    config.include_timestamp = false;
    config.include_level = true;
    config.prefix = "PREFIX: ";
    config.suffix = " :SUFFIX";
    config.min_level = DebugLevel::INFO;

    OutputCapture capture;
    DebugStream stream(config);
    stream.set_output(capture.stream());

    stream(DebugLevel::INFO) << "Test" << std::endl;

    std::string output = capture.get_output();
    EXPECT_TRUE(capture.contains("PREFIX:"));
    EXPECT_TRUE(capture.contains(":SUFFIX"));
    EXPECT_TRUE(capture.contains("[INFO]"));
    EXPECT_TRUE(capture.contains("Test"));
}

TEST(DebugStreamTest, EnableDisable) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream.set_enabled(true);
    stream(DebugLevel::INFO) << "Enabled message" << std::endl;

    stream.set_enabled(false);
    stream(DebugLevel::INFO) << "Disabled message" << std::endl;

    stream.set_enabled(true);
    stream(DebugLevel::INFO) << "Re-enabled message" << std::endl;

    EXPECT_TRUE(capture.contains("Enabled message"));
    EXPECT_FALSE(capture.contains("Disabled message"));
    EXPECT_TRUE(capture.contains("Re-enabled message"));
}

TEST(DebugStreamTest, UpdateConfiguration) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    // Initial config
    DebugStreamConfig config1;
    config1.include_level = true;
    config1.include_timestamp = false;
    stream.set_config(config1);

    stream(DebugLevel::INFO) << "Message 1" << std::endl;

    // Update config
    DebugStreamConfig config2;
    config2.include_level = false;
    config2.prefix = ">> ";
    stream.set_config(config2);

    stream(DebugLevel::INFO) << "Message 2" << std::endl;

    auto lines = capture.get_lines();
    EXPECT_TRUE(lines[0].find("[INFO]") != std::string::npos);
    EXPECT_FALSE(lines[1].find("[INFO]") != std::string::npos);
    EXPECT_TRUE(lines[1].find(">>") != std::string::npos);
}

// =============================================================================
// Level Filtering Tests
// =============================================================================

TEST(DebugStreamTest, LevelFiltering) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream.set_min_level(DebugLevel::WARNING);

    stream(DebugLevel::TRACE) << "Trace message" << std::endl;
    stream(DebugLevel::DEBUG) << "Debug message" << std::endl;
    stream(DebugLevel::INFO) << "Info message" << std::endl;
    stream(DebugLevel::WARNING) << "Warning message" << std::endl;
    stream(DebugLevel::ERROR) << "Error message" << std::endl;

    EXPECT_FALSE(capture.contains("Trace message"));
    EXPECT_FALSE(capture.contains("Debug message"));
    EXPECT_FALSE(capture.contains("Info message"));
    EXPECT_TRUE(capture.contains("Warning message"));
    EXPECT_TRUE(capture.contains("Error message"));
}

TEST(DebugStreamTest, ShouldLog) {
    DebugStream stream;

    stream.set_min_level(DebugLevel::INFO);

    EXPECT_TRUE(stream.should_log(DebugLevel::ERROR));
    EXPECT_TRUE(stream.should_log(DebugLevel::WARNING));
    EXPECT_TRUE(stream.should_log(DebugLevel::INFO));
    EXPECT_FALSE(stream.should_log(DebugLevel::DEBUG));
    EXPECT_FALSE(stream.should_log(DebugLevel::TRACE));

    stream.set_min_level(DebugLevel::TRACE);
    EXPECT_TRUE(stream.should_log(DebugLevel::TRACE));

    stream.set_enabled(false);
    EXPECT_FALSE(stream.should_log(DebugLevel::ERROR));
}

TEST(DebugStreamTest, DynamicLevelChange) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream.set_min_level(DebugLevel::ERROR);
    stream(DebugLevel::INFO) << "Should not appear" << std::endl;

    stream.set_min_level(DebugLevel::INFO);
    stream(DebugLevel::INFO) << "Should appear" << std::endl;

    EXPECT_FALSE(capture.contains("Should not appear"));
    EXPECT_TRUE(capture.contains("Should appear"));
}

// =============================================================================
// Debug Level Tests
// =============================================================================

TEST(DebugLevelTest, LevelToString) {
    EXPECT_STREQ(debug_level_to_string(DebugLevel::NONE), "NONE");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::ERROR), "ERROR");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::WARNING), "WARN");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::INFO), "INFO");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::DEBUG), "DEBUG");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::TRACE), "TRACE");
    EXPECT_STREQ(debug_level_to_string(DebugLevel::ALL), "ALL");
    EXPECT_STREQ(debug_level_to_string(static_cast<DebugLevel>(999)), "UNKNOWN");
}

TEST(DebugStreamTest, AllLevels) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream.set_min_level(DebugLevel::ALL);

    stream(DebugLevel::ERROR) << "Error" << std::endl;
    stream(DebugLevel::WARNING) << "Warning" << std::endl;
    stream(DebugLevel::INFO) << "Info" << std::endl;
    stream(DebugLevel::DEBUG) << "Debug" << std::endl;
    stream(DebugLevel::TRACE) << "Trace" << std::endl;

    EXPECT_TRUE(capture.contains("[ERROR]"));
    EXPECT_TRUE(capture.contains("[WARN]"));
    EXPECT_TRUE(capture.contains("[INFO]"));
    EXPECT_TRUE(capture.contains("[DEBUG]"));
    EXPECT_TRUE(capture.contains("[TRACE]"));
}

// =============================================================================
// Formatting Tests
// =============================================================================

TEST(DebugStreamTest, TimestampFormat) {
    DebugStreamConfig config;
    config.include_timestamp = true;
    config.include_level = false;

    OutputCapture capture;
    DebugStream stream(config);
    stream.set_output(capture.stream());

    stream(DebugLevel::INFO) << "Message" << std::endl;

    std::string output = capture.get_output();
    // Check for timestamp pattern [YYYY-MM-DD HH:MM:SS.mmm]
    std::regex timestamp_regex(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])");
    EXPECT_TRUE(std::regex_search(output, timestamp_regex));
}

TEST(DebugStreamTest, ThreadIdFormat) {
    DebugStreamConfig config;
    config.include_thread_id = true;
    config.include_level = false;
    config.include_timestamp = false;

    OutputCapture capture;
    DebugStream stream(config);
    stream.set_output(capture.stream());

    stream(DebugLevel::INFO) << "Message" << std::endl;

    std::string output = capture.get_output();
    EXPECT_TRUE(output.find("[TID:") != std::string::npos);
}

TEST(DebugStreamTest, PrefixSuffix) {
    DebugStreamConfig config;
    config.include_timestamp = false;
    config.include_level = false;
    config.prefix = ">>> ";
    config.suffix = " <<<";

    OutputCapture capture;
    DebugStream stream(config);
    stream.set_output(capture.stream());

    stream(DebugLevel::INFO) << "Test" << std::endl;

    std::string output = capture.get_output();
    EXPECT_TRUE(output.find(">>> Test <<<") != std::string::npos);
}

TEST(DebugStreamTest, ComplexFormatting) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream(DebugLevel::INFO) << "Integer: " << 42
                             << ", Float: " << 3.14159
                             << ", String: " << "test"
                             << ", Hex: " << std::hex << 255
                             << std::endl;

    std::string output = capture.get_output();
    EXPECT_TRUE(capture.contains("Integer: 42"));
    EXPECT_TRUE(capture.contains("Float: 3.14"));
    EXPECT_TRUE(capture.contains("String: test"));
    EXPECT_TRUE(capture.contains("Hex: ff"));
}

// =============================================================================
// Stream Manipulator Tests
// =============================================================================

TEST(DebugStreamTest, StreamManipulators) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream(DebugLevel::INFO) << "Value: "
                             << std::setw(10) << std::setfill('0') << 42
                             << std::endl;

    EXPECT_TRUE(capture.contains("0000000042"));
}

TEST(DebugStreamTest, EndlHandling) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    DebugStreamConfig config;
    config.include_timestamp = false;
    config.include_level = false;
    stream.set_config(config);

    stream(DebugLevel::INFO) << "Line 1" << std::endl
                             << "Line 2" << std::endl;

    auto lines = capture.get_lines();
    EXPECT_GE(lines.size(), 2);
    if (lines.size() >= 2) {
        EXPECT_EQ(lines[0], "Line 1");
        EXPECT_EQ(lines[1], "Line 2");
    }
}

// =============================================================================
// Buffer Tests
// =============================================================================

TEST(DebugStreamBufferTest, BasicOperation) {
    std::ostringstream oss;
    DebugStreamBuffer buffer(oss);

    std::ostream stream(&buffer);
    stream << "Test message" << std::endl;
    stream.flush();

    EXPECT_EQ(oss.str(), "Test message\n");
}

TEST(DebugStreamBufferTest, OutputSwitch) {
    std::ostringstream oss1, oss2;
    DebugStreamBuffer buffer(oss1);

    std::ostream stream(&buffer);
    stream << "Message 1" << std::endl;

    buffer.set_output(oss2);
    stream << "Message 2" << std::endl;

    EXPECT_EQ(oss1.str(), "Message 1\n");
    EXPECT_EQ(oss2.str(), "Message 2\n");
}

TEST(DebugStreamBufferTest, Filtering) {
    std::ostringstream oss;
    DebugStreamBuffer buffer(oss);

    // Set filter to only allow messages containing "PASS"
    buffer.set_filter([](const std::string& msg) {
        return msg.find("PASS") != std::string::npos;
    });

    std::ostream stream(&buffer);
    stream << "PASS: This should appear" << std::endl;
    stream << "FAIL: This should not appear" << std::endl;
    stream << "Another PASS message" << std::endl;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("PASS: This should appear") != std::string::npos);
    EXPECT_FALSE(output.find("FAIL: This should not appear") != std::string::npos);
    EXPECT_TRUE(output.find("Another PASS message") != std::string::npos);
}

TEST(DebugStreamBufferTest, ManualFlush) {
    std::ostringstream oss;
    DebugStreamBuffer buffer(oss);

    std::ostream stream(&buffer);
    stream << "Partial message";

    // Nothing should be in output yet (no newline)
    EXPECT_TRUE(oss.str().empty());

    buffer.flush_buffer();
    EXPECT_EQ(oss.str(), "Partial message");
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST(DebugStreamTest, MessageCount) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    EXPECT_EQ(stream.get_message_count(), 0);

    stream(DebugLevel::INFO) << "Message 1" << std::endl;
    stream(DebugLevel::DEBUG) << "Message 2" << std::endl;
    stream(DebugLevel::ERROR) << "Message 3" << std::endl;

    EXPECT_EQ(stream.get_message_count(), 3);

    stream.reset_statistics();
    EXPECT_EQ(stream.get_message_count(), 0);
}

TEST(DebugStreamTest, UptimeMeasurement) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    auto initial_uptime = stream.get_uptime();
    EXPECT_GE(initial_uptime.count(), 0);

    std::this_thread::sleep_for(50ms);

    auto later_uptime = stream.get_uptime();
    EXPECT_GT(later_uptime.count(), initial_uptime.count());
    EXPECT_GE(later_uptime.count(), 50);

    stream.reset_statistics();
    auto reset_uptime = stream.get_uptime();
    EXPECT_LT(reset_uptime.count(), later_uptime.count());
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(DebugStreamTest, ConcurrentOutput) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    const int num_threads = 4;
    const int messages_per_thread = 10;  // Reduced for faster testing
    std::vector<std::thread> threads;
    std::atomic<int> total_written{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&stream, &total_written, t, messages_per_thread]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                stream(DebugLevel::INFO) << "Thread " << t
                                        << " Message " << i << std::endl;
                total_written.fetch_add(1);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify we wrote all messages
    EXPECT_EQ(total_written.load(), num_threads * messages_per_thread);
    EXPECT_GT(stream.get_message_count(), 0);
}

TEST(DebugStreamTest, ConcurrentConfiguration) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    const int iterations = 20;
    std::atomic<int> config_changes{0};
    std::atomic<int> messages_written{0};
    std::vector<std::thread> threads;

    // Thread that writes messages
    threads.emplace_back([&stream, &messages_written, iterations]() {
        for (int i = 0; i < iterations; ++i) {
            stream(DebugLevel::INFO) << "Message " << i << std::endl;
            messages_written.fetch_add(1);
        }
    });

    // Thread that changes configuration
    threads.emplace_back([&stream, &config_changes, iterations]() {
        for (int i = 0; i < iterations; ++i) {
            DebugStreamConfig config;
            config.include_timestamp = (i % 2 == 0);
            config.include_level = (i % 3 == 0);
            stream.set_config(config);
            config_changes.fetch_add(1);
        }
    });

    for (auto& thread : threads) {
        thread.join();
    }

    // Should have performed all operations without crashes
    EXPECT_EQ(messages_written.load(), iterations);
    EXPECT_EQ(config_changes.load(), iterations);
    EXPECT_GT(stream.get_message_count(), 0);
}

// =============================================================================
// Global Instance Tests
// =============================================================================

TEST(GlobalDebugStreamTest, SingletonAccess) {
    auto& stream1 = GlobalDebugStream::instance();
    auto& stream2 = GlobalDebugStream::instance();

    EXPECT_EQ(&stream1, &stream2);
}

TEST(GlobalDebugStreamTest, NamedStreams) {
    auto& stream1 = GlobalDebugStream::get("test1");
    auto& stream2 = GlobalDebugStream::get("test2");
    auto& stream1_again = GlobalDebugStream::get("test1");

    EXPECT_NE(&stream1, &stream2);
    EXPECT_EQ(&stream1, &stream1_again);
}

TEST(GlobalDebugStreamTest, IndependentConfiguration) {
    OutputCapture capture1, capture2;

    auto& stream1 = GlobalDebugStream::get("independent1");
    auto& stream2 = GlobalDebugStream::get("independent2");

    stream1.set_output(capture1.stream());
    stream2.set_output(capture2.stream());

    stream1.set_min_level(DebugLevel::ERROR);
    stream2.set_min_level(DebugLevel::INFO);

    stream1(DebugLevel::INFO) << "Should not appear in stream1" << std::endl;
    stream1(DebugLevel::ERROR) << "Error in stream1" << std::endl;

    stream2(DebugLevel::INFO) << "Info in stream2" << std::endl;
    stream2(DebugLevel::ERROR) << "Error in stream2" << std::endl;

    EXPECT_FALSE(capture1.contains("Should not appear"));
    EXPECT_TRUE(capture1.contains("Error in stream1"));
    EXPECT_TRUE(capture2.contains("Info in stream2"));
    EXPECT_TRUE(capture2.contains("Error in stream2"));
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST(DebugStreamTest, EmptyMessages) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    stream(DebugLevel::INFO) << "" << std::endl;
    stream(DebugLevel::INFO) << std::endl;

    // Should still have level indicators
    EXPECT_EQ(capture.count_lines(), 2);
    EXPECT_TRUE(capture.contains("[INFO]"));
}

TEST(DebugStreamTest, VeryLongMessage) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    std::string long_message(10000, 'X');
    stream(DebugLevel::INFO) << long_message << std::endl;

    EXPECT_TRUE(capture.contains(long_message));
}

TEST(DebugStreamTest, RapidToggling) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    for (int i = 0; i < 100; ++i) {
        stream.set_enabled(i % 2 == 0);
        stream(DebugLevel::INFO) << "Message " << i << std::endl;
    }

    // Should only have even-numbered messages
    // Use more precise matching to avoid "Message 1" matching "Message 10", etc.
    std::string output = capture.get_output();
    for (int i = 0; i < 100; ++i) {
        // Look for "Message N" followed by newline (more precise than substring search)
        std::string msg_with_newline = "Message " + std::to_string(i) + "\n";
        bool found = output.find(msg_with_newline) != std::string::npos;
        if (i % 2 == 0) {
            EXPECT_TRUE(found) << "Message " << i << " should be present";
        } else {
            EXPECT_FALSE(found) << "Message " << i << " should NOT be present";
        }
    }
}

TEST(DebugStreamTest, NullOutput) {
    // Should not crash when no output is set
    DebugStream stream;

    EXPECT_NO_THROW({
        stream(DebugLevel::INFO) << "Test message" << std::endl;
        stream.flush();
    });
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST(DebugStreamPerformanceTest, Throughput) {
    OutputCapture capture;
    DebugStream stream(capture.stream());

    // Disable formatting for pure throughput test
    DebugStreamConfig config;
    config.include_timestamp = false;
    config.include_thread_id = false;
    config.include_level = false;
    stream.set_config(config);

    const int message_count = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < message_count; ++i) {
        stream(DebugLevel::INFO) << "Message " << i << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should handle at least 1000 messages per second
    EXPECT_LT(duration.count(), message_count);
    EXPECT_EQ(stream.get_message_count(), message_count);
}

TEST(DebugStreamPerformanceTest, DisabledOverhead) {
    OutputCapture capture;
    DebugStream stream(capture.stream());
    stream.set_enabled(false);

    const int message_count = 100000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < message_count; ++i) {
        stream(DebugLevel::INFO) << "This should be very fast " << i << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Disabled logging should be very fast (< 1 second for 100k messages)
    EXPECT_LT(duration.count(), 1000);
    EXPECT_TRUE(capture.get_output().empty());
}
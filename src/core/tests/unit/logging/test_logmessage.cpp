#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>

#include "logging/logmessage.h"
#include "logging/loglevel.h"

using namespace fem::core::logging;

// ============================================================================
// LogMessage Constructor Tests
// ============================================================================

TEST(LogMessageTest, BasicConstruction) {
    const std::string logger_name = "test.logger";
    const std::string message = "Test message";
    LogLevel level = LogLevel::INFO;

    LogMessage log_msg(level, logger_name, message);

    EXPECT_EQ(log_msg.get_level(), level);
    EXPECT_EQ(log_msg.get_logger_name(), logger_name);
    EXPECT_EQ(log_msg.get_message(), message);
    EXPECT_EQ(log_msg.get_sequence_number(), 0);
    EXPECT_FALSE(log_msg.has_exception());
    EXPECT_TRUE(log_msg.empty_context());
}

TEST(LogMessageTest, ConstructionWithSourceLocation) {
    LogMessage log_msg(LogLevel::ERROR, "test", "error message");

    // Source location should be captured
    EXPECT_NE(log_msg.get_file_name(), nullptr);
    EXPECT_NE(log_msg.get_function_name(), nullptr);
    EXPECT_GT(log_msg.get_line(), 0u);

    // Check that file name contains this test file
    std::string file_name(log_msg.get_file_name());
    EXPECT_TRUE(file_name.find("test_logmessage.cpp") != std::string::npos);
}

TEST(LogMessageTest, TimestampAndThreadId) {
    auto before = std::chrono::system_clock::now();
    LogMessage log_msg(LogLevel::DEBUG, "test", "debug message");
    auto after = std::chrono::system_clock::now();

    // Timestamp should be between before and after
    EXPECT_GE(log_msg.get_timestamp(), before);
    EXPECT_LE(log_msg.get_timestamp(), after);

    // Thread ID should match current thread
    EXPECT_EQ(log_msg.get_thread_id(), std::this_thread::get_id());
}

// ============================================================================
// LogMessage Copy/Move Semantics Tests
// ============================================================================

TEST(LogMessageTest, CopyConstructor) {
    LogMessage original(LogLevel::WARN, "original", "original message");
    original.set_sequence_number(42);
    original.set_context("key", "value");

    LogMessage copy(original);

    EXPECT_EQ(copy.get_level(), original.get_level());
    EXPECT_EQ(copy.get_logger_name(), original.get_logger_name());
    EXPECT_EQ(copy.get_message(), original.get_message());
    EXPECT_EQ(copy.get_timestamp(), original.get_timestamp());
    EXPECT_EQ(copy.get_thread_id(), original.get_thread_id());
    EXPECT_EQ(copy.get_sequence_number(), original.get_sequence_number());
    EXPECT_EQ(copy.get_context<std::string>("key"), "value");
}

TEST(LogMessageTest, CopyAssignment) {
    LogMessage original(LogLevel::FATAL, "original", "fatal error");
    original.set_context("test", "data");

    LogMessage assigned(LogLevel::TRACE, "different", "different message");
    assigned = original;

    EXPECT_EQ(assigned.get_level(), LogLevel::FATAL);
    EXPECT_EQ(assigned.get_logger_name(), "original");
    EXPECT_EQ(assigned.get_message(), "fatal error");
    EXPECT_EQ(assigned.get_context<std::string>("test"), "data");
}

TEST(LogMessageTest, MoveConstructor) {
    LogMessage original(LogLevel::INFO, "movable", "move test");
    original.set_context("key", "value");
    std::string original_logger = original.get_logger_name();
    std::string original_message = original.get_message();

    LogMessage moved(std::move(original));

    EXPECT_EQ(moved.get_level(), LogLevel::INFO);
    EXPECT_EQ(moved.get_logger_name(), original_logger);
    EXPECT_EQ(moved.get_message(), original_message);
    EXPECT_EQ(moved.get_context<std::string>("key"), "value");
}

TEST(LogMessageTest, MoveAssignment) {
    LogMessage original(LogLevel::ERROR, "source", "source message");
    original.set_context("data", "important");

    LogMessage target(LogLevel::TRACE, "target", "target message");
    target = std::move(original);

    EXPECT_EQ(target.get_level(), LogLevel::ERROR);
    EXPECT_EQ(target.get_logger_name(), "source");
    EXPECT_EQ(target.get_message(), "source message");
    EXPECT_EQ(target.get_context<std::string>("data"), "important");
}

// ============================================================================
// LogMessage Getter Tests
// ============================================================================

TEST(LogMessageTest, AllGetters) {
    const LogLevel level = LogLevel::WARN;
    const std::string logger = "test.logger.name";
    const std::string message = "Warning message content";

    LogMessage log_msg(level, logger, message);
    log_msg.set_sequence_number(12345);

    EXPECT_EQ(log_msg.get_level(), level);
    EXPECT_EQ(log_msg.get_logger_name(), logger);
    EXPECT_EQ(log_msg.get_message(), message);
    EXPECT_EQ(log_msg.get_sequence_number(), 12345u);

    // Test types
    EXPECT_TRUE((std::is_same_v<decltype(log_msg.get_level()), LogLevel>));
    EXPECT_TRUE((std::is_same_v<decltype(log_msg.get_logger_name()), const std::string&>));
    EXPECT_TRUE((std::is_same_v<decltype(log_msg.get_message()), const std::string&>));
    EXPECT_TRUE((std::is_same_v<decltype(log_msg.get_thread_id()), std::thread::id>));
    EXPECT_TRUE((std::is_same_v<decltype(log_msg.get_line()), std::uint_least32_t>));
}

// ============================================================================
// LogMessage Exception Support Tests
// ============================================================================

TEST(LogMessageTest, ExceptionSupport_NoException) {
    LogMessage log_msg(LogLevel::INFO, "test", "no exception");

    EXPECT_FALSE(log_msg.has_exception());
    EXPECT_EQ(log_msg.get_exception(), nullptr);
}

TEST(LogMessageTest, ExceptionSupport_WithException) {
    LogMessage log_msg(LogLevel::ERROR, "test", "with exception");

    std::exception_ptr ex;
    try {
        throw std::runtime_error("Test exception");
    } catch (...) {
        ex = std::current_exception();
    }

    log_msg.set_exception(ex);

    EXPECT_TRUE(log_msg.has_exception());
    EXPECT_NE(log_msg.get_exception(), nullptr);
    EXPECT_EQ(log_msg.get_exception(), ex);
}

TEST(LogMessageTest, ExceptionSupport_ClearException) {
    LogMessage log_msg(LogLevel::ERROR, "test", "clear exception");

    // Set exception
    try {
        throw std::logic_error("Test exception");
    } catch (...) {
        log_msg.set_exception(std::current_exception());
    }
    EXPECT_TRUE(log_msg.has_exception());

    // Clear exception
    log_msg.set_exception(nullptr);
    EXPECT_FALSE(log_msg.has_exception());
    EXPECT_EQ(log_msg.get_exception(), nullptr);
}

// ============================================================================
// LogMessage Sequence Number Tests
// ============================================================================

TEST(LogMessageTest, SequenceNumber_DefaultZero) {
    LogMessage log_msg(LogLevel::DEBUG, "test", "sequence test");
    EXPECT_EQ(log_msg.get_sequence_number(), 0u);
}

TEST(LogMessageTest, SequenceNumber_SetAndGet) {
    LogMessage log_msg(LogLevel::INFO, "test", "sequence test");

    log_msg.set_sequence_number(999999);
    EXPECT_EQ(log_msg.get_sequence_number(), 999999u);

    log_msg.set_sequence_number(0);
    EXPECT_EQ(log_msg.get_sequence_number(), 0u);

    log_msg.set_sequence_number(UINT64_MAX);
    EXPECT_EQ(log_msg.get_sequence_number(), UINT64_MAX);
}

// ============================================================================
// LogMessage Context Tests
// ============================================================================

TEST(LogMessageTest, Context_EmptyByDefault) {
    LogMessage log_msg(LogLevel::TRACE, "test", "context test");

    EXPECT_TRUE(log_msg.empty_context());
    EXPECT_EQ(log_msg.get_context().size(), 0u);
    EXPECT_TRUE(log_msg.get_context_keys().empty());
}

TEST(LogMessageTest, Context_SetAndGetString) {
    LogMessage log_msg(LogLevel::INFO, "test", "string context");

    log_msg.set_context("string_key", "string_value");

    EXPECT_FALSE(log_msg.empty_context());
    EXPECT_EQ(log_msg.get_context().size(), 1u);

    auto value = log_msg.get_context<std::string>("string_key");
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "string_value");
}

TEST(LogMessageTest, Context_SetAndGetCString) {
    LogMessage log_msg(LogLevel::WARN, "test", "cstring context");

    const char* test_value = "c_string_value";
    log_msg.set_context("cstring_key", test_value);

    auto value = log_msg.get_context<std::string>("cstring_key");
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "c_string_value");
}

TEST(LogMessageTest, Context_SetAndGetInt) {
    LogMessage log_msg(LogLevel::ERROR, "test", "int context");

    log_msg.set_context("int_key", 42);

    auto value = log_msg.get_context<int>("int_key");
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), 42);

    // Should also be retrievable as string
    auto str_value = log_msg.get_context<std::string>("int_key");
    ASSERT_TRUE(str_value.has_value());
    EXPECT_EQ(str_value.value(), "42");
}

TEST(LogMessageTest, Context_SetAndGetDouble) {
    LogMessage log_msg(LogLevel::FATAL, "test", "double context");

    log_msg.set_context("double_key", 3.14159);

    auto value = log_msg.get_context<double>("double_key");
    ASSERT_TRUE(value.has_value());
    EXPECT_NEAR(value.value(), 3.14159, 0.00001);
}

TEST(LogMessageTest, Context_SetAndGetBool) {
    LogMessage log_msg(LogLevel::DEBUG, "test", "bool context");

    log_msg.set_context("bool_true", true);
    log_msg.set_context("bool_false", false);

    auto true_value = log_msg.get_context<bool>("bool_true");
    ASSERT_TRUE(true_value.has_value());
    EXPECT_TRUE(true_value.value());

    auto false_value = log_msg.get_context<bool>("bool_false");
    ASSERT_TRUE(false_value.has_value());
    EXPECT_FALSE(false_value.value());
}

TEST(LogMessageTest, Context_BoolStringConversion) {
    LogMessage log_msg(LogLevel::INFO, "test", "bool string context");

    // Test various bool string representations
    log_msg.set_context("true_str", "true");
    log_msg.set_context("one_str", "1");
    log_msg.set_context("false_str", "false");
    log_msg.set_context("zero_str", "0");
    log_msg.set_context("other_str", "anything");

    EXPECT_TRUE(log_msg.get_context<bool>("true_str").value());
    EXPECT_TRUE(log_msg.get_context<bool>("one_str").value());
    EXPECT_FALSE(log_msg.get_context<bool>("false_str").value());
    EXPECT_FALSE(log_msg.get_context<bool>("zero_str").value());
    EXPECT_FALSE(log_msg.get_context<bool>("other_str").value());
}

TEST(LogMessageTest, Context_GetNonExistentKey) {
    LogMessage log_msg(LogLevel::TRACE, "test", "missing key test");

    auto value = log_msg.get_context<std::string>("nonexistent");
    EXPECT_FALSE(value.has_value());

    auto int_value = log_msg.get_context<int>("missing");
    EXPECT_FALSE(int_value.has_value());
}

TEST(LogMessageTest, Context_InvalidTypeConversion) {
    LogMessage log_msg(LogLevel::WARN, "test", "invalid conversion");

    log_msg.set_context("text", "not_a_number");

    auto int_value = log_msg.get_context<int>("text");
    EXPECT_FALSE(int_value.has_value());

    auto double_value = log_msg.get_context<double>("text");
    EXPECT_FALSE(double_value.has_value());
}

TEST(LogMessageTest, Context_GetKeys) {
    LogMessage log_msg(LogLevel::ERROR, "test", "keys test");

    log_msg.set_context("key1", "value1");
    log_msg.set_context("key2", 42);
    log_msg.set_context("key3", 3.14);

    auto keys = log_msg.get_context_keys();
    EXPECT_EQ(keys.size(), 3u);

    // Keys should contain all expected keys (order not guaranteed)
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key1") != keys.end());
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key2") != keys.end());
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key3") != keys.end());
}

TEST(LogMessageTest, Context_GetContextMap) {
    LogMessage log_msg(LogLevel::INFO, "test", "map test");

    log_msg.set_context("first", "one");
    log_msg.set_context("second", "two");

    const auto& context_map = log_msg.get_context();
    EXPECT_EQ(context_map.size(), 2u);
    EXPECT_EQ(context_map.at("first"), "one");
    EXPECT_EQ(context_map.at("second"), "two");
}

// ============================================================================
// LogMessage Clone Tests
// ============================================================================

TEST(LogMessageTest, Clone_CreatesIdenticalCopy) {
    LogMessage original(LogLevel::FATAL, "original.logger", "original message");
    original.set_sequence_number(123);
    original.set_context("key", "value");

    LogMessage cloned = original.clone();

    EXPECT_EQ(cloned.get_level(), original.get_level());
    EXPECT_EQ(cloned.get_logger_name(), original.get_logger_name());
    EXPECT_EQ(cloned.get_message(), original.get_message());
    EXPECT_EQ(cloned.get_timestamp(), original.get_timestamp());
    EXPECT_EQ(cloned.get_thread_id(), original.get_thread_id());
    EXPECT_EQ(cloned.get_sequence_number(), original.get_sequence_number());
    EXPECT_EQ(cloned.get_context<std::string>("key"), "value");
}

TEST(LogMessageTest, Clone_IndependentModification) {
    LogMessage original(LogLevel::DEBUG, "original", "original");
    original.set_context("shared", "original_value");

    LogMessage cloned = original.clone();

    // Modify original
    original.set_context("shared", "modified_value");
    original.set_context("new", "new_value");

    // Clone should be unchanged
    EXPECT_EQ(cloned.get_context<std::string>("shared"), "original_value");
    EXPECT_FALSE(cloned.get_context<std::string>("new").has_value());
}

// ============================================================================
// LogMessageBuilder Constructor Tests
// ============================================================================

TEST(LogMessageBuilderTest, BasicConstruction) {
    LogMessageBuilder builder(LogLevel::INFO);

    LogMessage msg = builder
        .logger("test.logger")
        .message("test message")
        .build();

    EXPECT_EQ(msg.get_level(), LogLevel::INFO);
    EXPECT_EQ(msg.get_logger_name(), "test.logger");
    EXPECT_EQ(msg.get_message(), "test message");
}

TEST(LogMessageBuilderTest, FluentInterface) {
    std::string value1 = "value1";
    LogMessage msg = LogMessageBuilder(LogLevel::WARN)
        .logger("fluent.test")
        .message("fluent interface test")
        .with("key1", value1)
        .with("key2", 42)
        .with("key3", 3.14)
        .build();

    EXPECT_EQ(msg.get_level(), LogLevel::WARN);
    EXPECT_EQ(msg.get_logger_name(), "fluent.test");
    EXPECT_EQ(msg.get_message(), "fluent interface test");
    EXPECT_EQ(msg.get_context<std::string>("key1"), "value1");
    EXPECT_EQ(msg.get_context<int>("key2"), 42);
    EXPECT_EQ(msg.get_context<double>("key3"), 3.14);
}

TEST(LogMessageBuilderTest, WithTemplates) {
    std::string string_value = "string_type";
    LogMessage msg = LogMessageBuilder(LogLevel::ERROR)
        .logger("template.test")
        .message("template test")
        .with("string_val", string_value)
        .with("int_val", 999)
        .with("double_val", 2.718)
        .with("bool_val", true)
        .build();

    EXPECT_EQ(msg.get_context<std::string>("string_val"), "string_type");
    EXPECT_EQ(msg.get_context<int>("int_val"), 999);
    EXPECT_EQ(msg.get_context<double>("double_val"), 2.718);
    EXPECT_EQ(msg.get_context<bool>("bool_val"), true);
}

TEST(LogMessageBuilderTest, EmptyBuilder) {
    LogMessage msg = LogMessageBuilder(LogLevel::TRACE).build();

    EXPECT_EQ(msg.get_level(), LogLevel::TRACE);
    EXPECT_TRUE(msg.get_logger_name().empty());
    EXPECT_TRUE(msg.get_message().empty());
    EXPECT_TRUE(msg.empty_context());
}

TEST(LogMessageBuilderTest, PartiallyConfigured) {
    LogMessage msg = LogMessageBuilder(LogLevel::DEBUG)
        .message("only message set")
        .build();

    EXPECT_EQ(msg.get_level(), LogLevel::DEBUG);
    EXPECT_TRUE(msg.get_logger_name().empty());
    EXPECT_EQ(msg.get_message(), "only message set");
}

TEST(LogMessageBuilderTest, SourceLocationCapture) {
    LogMessage msg = LogMessageBuilder(LogLevel::INFO)
        .logger("location.test")
        .message("source location test")
        .build();

    // Source location should be captured from build() call site
    EXPECT_NE(msg.get_file_name(), nullptr);
    EXPECT_NE(msg.get_function_name(), nullptr);
    EXPECT_GT(msg.get_line(), 0u);

    std::string file_name(msg.get_file_name());
    EXPECT_TRUE(file_name.find("test_logmessage.cpp") != std::string::npos);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(LogMessageIntegrationTest, CompleteWorkflow) {
    // Create message with builder
    std::string session_id = "abc-def-ghi";
    LogMessage msg = LogMessageBuilder(LogLevel::FATAL)
        .logger("integration.test")
        .message("Complete workflow test")
        .with("user_id", 12345)
        .with("session", session_id)
        .with("retry_count", 3)
        .build();

    // Add additional context
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        msg.get_timestamp().time_since_epoch()).count();
    msg.set_context("timestamp_ms", timestamp_ms);

    // Set sequence number
    msg.set_sequence_number(999);

    // Set exception
    std::exception_ptr ex;
    try {
        throw std::runtime_error("Integration test exception");
    } catch (...) {
        ex = std::current_exception();
    }
    msg.set_exception(ex);

    // Clone and verify
    LogMessage cloned = msg.clone();

    EXPECT_EQ(cloned.get_level(), LogLevel::FATAL);
    EXPECT_EQ(cloned.get_logger_name(), "integration.test");
    EXPECT_EQ(cloned.get_message(), "Complete workflow test");
    EXPECT_EQ(cloned.get_sequence_number(), 999u);
    EXPECT_TRUE(cloned.has_exception());
    EXPECT_EQ(cloned.get_context<int>("user_id"), 12345);
    EXPECT_EQ(cloned.get_context<std::string>("session"), "abc-def-ghi");
    EXPECT_EQ(cloned.get_context<int>("retry_count"), 3);
    // The timestamp should be stored as a string representation
    auto stored_timestamp = cloned.get_context<std::string>("timestamp_ms");
    EXPECT_TRUE(stored_timestamp.has_value());
    EXPECT_FALSE(stored_timestamp->empty());
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(LogMessageTest, ThreadIdCapture) {
    std::thread::id main_thread_id = std::this_thread::get_id();
    std::thread::id other_thread_id;
    LogMessage main_msg(LogLevel::INFO, "main", "main thread");
    LogMessage other_msg(LogLevel::INFO, "other", "other thread");

    std::thread other_thread([&]() {
        other_thread_id = std::this_thread::get_id();
        other_msg = LogMessage(LogLevel::INFO, "other", "other thread");
    });

    other_thread.join();

    EXPECT_EQ(main_msg.get_thread_id(), main_thread_id);
    EXPECT_EQ(other_msg.get_thread_id(), other_thread_id);
    EXPECT_NE(main_msg.get_thread_id(), other_msg.get_thread_id());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST(LogMessageTest, EmptyStrings) {
    LogMessage msg(LogLevel::OFF, "", "");

    EXPECT_EQ(msg.get_level(), LogLevel::OFF);
    EXPECT_TRUE(msg.get_logger_name().empty());
    EXPECT_TRUE(msg.get_message().empty());
}

TEST(LogMessageTest, VeryLongStrings) {
    std::string long_logger(10000, 'L');
    std::string long_message(50000, 'M');

    LogMessage msg(LogLevel::ERROR, long_logger, long_message);

    EXPECT_EQ(msg.get_logger_name().length(), 10000u);
    EXPECT_EQ(msg.get_message().length(), 50000u);
    EXPECT_EQ(msg.get_logger_name(), long_logger);
    EXPECT_EQ(msg.get_message(), long_message);
}

TEST(LogMessageTest, SpecialCharacters) {
    std::string special_logger = "logger.with.dots/and\\slashes:and:colons";
    std::string special_message = "Message with unicode: 你好世界 🌍 and newlines\n\r\t";

    LogMessage msg(LogLevel::WARN, special_logger, special_message);

    EXPECT_EQ(msg.get_logger_name(), special_logger);
    EXPECT_EQ(msg.get_message(), special_message);
}

TEST(LogMessageTest, Context_LargeNumberOfKeys) {
    LogMessage msg(LogLevel::DEBUG, "stress", "many keys test");

    // Add many context keys
    for (int i = 0; i < 1000; ++i) {
        msg.set_context("key_" + std::to_string(i), "value_" + std::to_string(i));
    }

    EXPECT_EQ(msg.get_context().size(), 1000u);
    EXPECT_EQ(msg.get_context_keys().size(), 1000u);
    EXPECT_EQ(msg.get_context<std::string>("key_500"), "value_500");
}

TEST(LogMessageTest, Context_KeyOverwrite) {
    LogMessage msg(LogLevel::INFO, "overwrite", "key overwrite test");

    msg.set_context("key", "original");
    EXPECT_EQ(msg.get_context<std::string>("key"), "original");

    msg.set_context("key", "overwritten");
    EXPECT_EQ(msg.get_context<std::string>("key"), "overwritten");

    // Context should still have only one entry
    EXPECT_EQ(msg.get_context().size(), 1u);
}

TEST(LogMessageTest, NumericLimits) {
    LogMessage msg(LogLevel::TRACE, "limits", "numeric limits test");

    msg.set_context("max_int", std::numeric_limits<int>::max());
    msg.set_context("min_int", std::numeric_limits<int>::min());
    msg.set_context("max_double", std::numeric_limits<double>::max());
    msg.set_context("min_double", std::numeric_limits<double>::lowest());

    EXPECT_EQ(msg.get_context<int>("max_int"), std::numeric_limits<int>::max());
    EXPECT_EQ(msg.get_context<int>("min_int"), std::numeric_limits<int>::min());
    EXPECT_EQ(msg.get_context<double>("max_double"), std::numeric_limits<double>::max());
    EXPECT_EQ(msg.get_context<double>("min_double"), std::numeric_limits<double>::lowest());
}
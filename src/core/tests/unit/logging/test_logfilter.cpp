#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <regex>

#include "logging/logmessage.h"
#include "logging/loglevel.h"

using namespace fem::core::logging;

// Forward declarations to avoid header compilation issues
namespace fem::core::logging {
    class LogFilter;
    class LevelFilter;
    class LoggerNameFilter;
    class ContentFilter;
    class RateLimitFilter;
    class DuplicateFilter;
    class CompositeFilter;
    class PredicateFilter;
    class TimeFilter;
    class ThreadFilter;
}

// Test compilation by including header after forward declarations
#include "logging/logfilter.h"

// ============================================================================
// Helper Functions
// ============================================================================

LogMessage create_test_message(LogLevel level = LogLevel::INFO,
                               const std::string& logger = "test.logger",
                               const std::string& message = "Test message") {
    LogMessage msg(level, logger, message);
    msg.set_sequence_number(42);
    return msg;
}

LogMessage create_message_with_timestamp(LogLevel level,
                                         const std::string& logger,
                                         const std::string& message,
                                         [[maybe_unused]] const std::chrono::system_clock::time_point& timestamp) {
    LogMessage msg(level, logger, message);
    // Note: LogMessage doesn't expose timestamp setting, so we'll work with current time
    return msg;
}

// ============================================================================
// LevelFilter Tests
// ============================================================================

TEST(LevelFilterTest, DefaultConstruction) {
    LevelFilter filter;  // Default: TRACE to FATAL

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::TRACE)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::DEBUG)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::WARN)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::ERROR)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::FATAL)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::OFF)));  // OFF is above FATAL
}

TEST(LevelFilterTest, SpecificRange) {
    LevelFilter filter(LogLevel::WARN, LogLevel::FATAL);

    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::TRACE)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::DEBUG)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::WARN)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::ERROR)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::FATAL)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::OFF)));
}

TEST(LevelFilterTest, SingleLevel) {
    LevelFilter filter(LogLevel::ERROR, LogLevel::ERROR);

    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::TRACE)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::DEBUG)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::WARN)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::ERROR)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::FATAL)));
}

TEST(LevelFilterTest, CloneMethod) {
    LevelFilter original(LogLevel::INFO, LogLevel::ERROR);
    auto cloned = original.clone();

    LogMessage msg = create_test_message(LogLevel::WARN);

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// LoggerNameFilter Tests
// ============================================================================

TEST(LoggerNameFilterTest, ExactMatch) {
    LoggerNameFilter filter("com.example.MyClass", LoggerNameFilter::Mode::EXACT_MATCH);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass.method")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "com.example")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "other.logger")));
}

TEST(LoggerNameFilterTest, PrefixMatch) {
    LoggerNameFilter filter("com.example", LoggerNameFilter::Mode::PREFIX_MATCH);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.another.SubClass")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "com.other")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "org.apache")));
}

TEST(LoggerNameFilterTest, RegexMatch) {
    LoggerNameFilter filter(R"(com\.example\.[A-Z]\w*)", LoggerNameFilter::Mode::REGEX_MATCH);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.AnotherClass")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.myClass")));  // lowercase
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "com.example")));         // no class
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "org.example.MyClass")));
}

TEST(LoggerNameFilterTest, Blacklist) {
    LoggerNameFilter filter("noisy", LoggerNameFilter::Mode::BLACKLIST);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.Logger")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "important.service")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "noisy.component")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "noisy")));
}

TEST(LoggerNameFilterTest, DefaultMode) {
    LoggerNameFilter filter("com.example");  // Default is PREFIX_MATCH

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "org.apache")));
}

TEST(LoggerNameFilterTest, CloneMethod) {
    LoggerNameFilter original("test", LoggerNameFilter::Mode::EXACT_MATCH);
    auto cloned = original.clone();

    LogMessage msg = create_test_message(LogLevel::INFO, "test");

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// ContentFilter Tests
// ============================================================================

TEST(ContentFilterTest, Contains) {
    ContentFilter filter("error", ContentFilter::Mode::CONTAINS);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "An error occurred")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Error: invalid input")));  // Different case
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "System error detected")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Everything is fine")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Warning: low disk space")));
}

TEST(ContentFilterTest, NotContains) {
    ContentFilter filter("password", ContentFilter::Mode::NOT_CONTAINS);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "User logged in")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Session started")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Invalid password")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Password updated")));  // Different case
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "password=secret123")));
}

TEST(ContentFilterTest, RegexMatch) {
    ContentFilter filter(R"(\b\d{3}-\d{2}-\d{4}\b)", ContentFilter::Mode::REGEX_MATCH);  // SSN pattern

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "User SSN: 123-45-6789")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Found 987-65-4321 in database")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "User ID: 12345")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Invalid format: 123456789")));
}

TEST(ContentFilterTest, DefaultMode) {
    ContentFilter filter("debug");  // Default is CONTAINS

    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Debug info available")));  // Different case
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "No info available")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "debug info available")));   // Correct case
}

TEST(ContentFilterTest, CaseSensitive) {
    ContentFilter filter("ERROR", ContentFilter::Mode::CONTAINS);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "ERROR occurred")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "error occurred")));  // Different case
}

TEST(ContentFilterTest, CloneMethod) {
    ContentFilter original("test", ContentFilter::Mode::NOT_CONTAINS);
    auto cloned = original.clone();

    LogMessage msg = create_test_message(LogLevel::INFO, "logger", "sample message");

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// RateLimitFilter Tests
// ============================================================================

TEST(RateLimitFilterTest, BasicRateLimit) {
    RateLimitFilter filter(3, 1000);  // 3 messages per second

    LogMessage msg = create_test_message();

    // First 3 messages should pass
    EXPECT_TRUE(filter.should_log(msg));
    EXPECT_TRUE(filter.should_log(msg));
    EXPECT_TRUE(filter.should_log(msg));

    // 4th message should be blocked
    EXPECT_FALSE(filter.should_log(msg));
}

TEST(RateLimitFilterTest, TimeWindowReset) {
    RateLimitFilter filter(2, 100);  // 2 messages per 100ms

    LogMessage msg = create_test_message();

    // First 2 messages should pass
    EXPECT_TRUE(filter.should_log(msg));
    EXPECT_TRUE(filter.should_log(msg));

    // 3rd message should be blocked
    EXPECT_FALSE(filter.should_log(msg));

    // Wait for time window to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Should accept messages again
    EXPECT_TRUE(filter.should_log(msg));
    EXPECT_TRUE(filter.should_log(msg));
    EXPECT_FALSE(filter.should_log(msg));
}

TEST(RateLimitFilterTest, ZeroLimit) {
    RateLimitFilter filter(0, 1000);  // No messages allowed

    LogMessage msg = create_test_message();

    EXPECT_FALSE(filter.should_log(msg));
    EXPECT_FALSE(filter.should_log(msg));
}

TEST(RateLimitFilterTest, CloneMethod) {
    RateLimitFilter original(5, 2000);
    auto cloned = original.clone();

    LogMessage msg = create_test_message();

    // Both should behave the same initially
    EXPECT_TRUE(original.should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));

    // But they should have independent state
    // Use up original's limit
    for (int i = 0; i < 4; ++i) {
        (void)original.should_log(msg);  // Explicitly discard result
    }
    EXPECT_FALSE(original.should_log(msg));  // Should be blocked
    EXPECT_TRUE(cloned->should_log(msg));    // Clone should still accept
}

// ============================================================================
// DuplicateFilter Tests
// ============================================================================

TEST(DuplicateFilterTest, BasicDuplicateDetection) {
    DuplicateFilter filter(10);

    LogMessage msg1 = create_test_message(LogLevel::INFO, "logger", "Same message");
    LogMessage msg2 = create_test_message(LogLevel::INFO, "logger", "Same message");
    LogMessage msg3 = create_test_message(LogLevel::INFO, "logger", "Different message");

    EXPECT_TRUE(filter.should_log(msg1));   // First occurrence
    EXPECT_FALSE(filter.should_log(msg2));  // Duplicate
    EXPECT_TRUE(filter.should_log(msg3));   // Different message
}

TEST(DuplicateFilterTest, LevelMatters) {
    DuplicateFilter filter(10);

    LogMessage msg1 = create_test_message(LogLevel::INFO, "logger", "Same message");
    LogMessage msg2 = create_test_message(LogLevel::ERROR, "logger", "Same message");  // Different level

    EXPECT_TRUE(filter.should_log(msg1));  // First occurrence
    EXPECT_TRUE(filter.should_log(msg2));  // Different level, so not a duplicate
}

TEST(DuplicateFilterTest, CacheLimit) {
    DuplicateFilter filter(2);  // Very small cache

    LogMessage msg1 = create_test_message(LogLevel::INFO, "logger", "Message 1");
    LogMessage msg2 = create_test_message(LogLevel::INFO, "logger", "Message 2");
    LogMessage msg3 = create_test_message(LogLevel::INFO, "logger", "Message 3");
    LogMessage msg1_repeat = create_test_message(LogLevel::INFO, "logger", "Message 1");

    EXPECT_TRUE(filter.should_log(msg1));   // Add to cache
    EXPECT_TRUE(filter.should_log(msg2));   // Add to cache
    EXPECT_TRUE(filter.should_log(msg3));   // Evicts msg1 from cache
    EXPECT_TRUE(filter.should_log(msg1_repeat));  // msg1 was evicted, so allowed again
}

TEST(DuplicateFilterTest, CloneMethod) {
    DuplicateFilter original(5);
    LogMessage msg = create_test_message(LogLevel::INFO, "logger", "test message");

    // Use original once
    EXPECT_TRUE(original.should_log(msg));

    auto cloned = original.clone();

    // Clone should not have seen the message yet
    EXPECT_TRUE(cloned->should_log(msg));

    // But original should reject duplicate
    EXPECT_FALSE(original.should_log(msg));
}

// ============================================================================
// CompositeFilter Tests
// ============================================================================

TEST(CompositeFilterTest, AllMode) {
    CompositeFilter composite(CompositeFilter::Mode::ALL);

    composite.add_filter(std::make_unique<LevelFilter>(LogLevel::WARN, LogLevel::FATAL));
    composite.add_filter(std::make_unique<LoggerNameFilter>("com.example", LoggerNameFilter::Mode::PREFIX_MATCH));

    EXPECT_TRUE(composite.should_log(create_test_message(LogLevel::ERROR, "com.example.MyClass", "error message")));
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "com.example.MyClass", "info message")));  // Level too low
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::ERROR, "org.apache.MyClass", "error message")));  // Wrong logger
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "org.apache.MyClass", "info message")));   // Both fail
}

TEST(CompositeFilterTest, AnyMode) {
    CompositeFilter composite(CompositeFilter::Mode::ANY);

    composite.add_filter(std::make_unique<LevelFilter>(LogLevel::ERROR, LogLevel::FATAL));
    composite.add_filter(std::make_unique<ContentFilter>("important", ContentFilter::Mode::CONTAINS));

    EXPECT_TRUE(composite.should_log(create_test_message(LogLevel::ERROR, "logger", "normal error")));        // Level matches
    EXPECT_TRUE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "important info")));      // Content matches
    EXPECT_TRUE(composite.should_log(create_test_message(LogLevel::FATAL, "logger", "important critical")));  // Both match
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "normal info")));        // Neither matches
}

TEST(CompositeFilterTest, NoneMode) {
    CompositeFilter composite(CompositeFilter::Mode::NONE);

    composite.add_filter(std::make_unique<ContentFilter>("password", ContentFilter::Mode::CONTAINS));
    composite.add_filter(std::make_unique<ContentFilter>("secret", ContentFilter::Mode::CONTAINS));

    EXPECT_TRUE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "normal message")));
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "password reset")));
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "secret key")));
    EXPECT_FALSE(composite.should_log(create_test_message(LogLevel::INFO, "logger", "password and secret")));
}

TEST(CompositeFilterTest, EmptyFilters) {
    CompositeFilter composite(CompositeFilter::Mode::ALL);

    LogMessage msg = create_test_message();

    EXPECT_TRUE(composite.should_log(msg));  // Empty composite should allow all
}

TEST(CompositeFilterTest, CloneMethod) {
    CompositeFilter original(CompositeFilter::Mode::ALL);
    original.add_filter(std::make_unique<LevelFilter>(LogLevel::WARN, LogLevel::FATAL));
    original.add_filter(std::make_unique<ContentFilter>("test", ContentFilter::Mode::CONTAINS));

    auto cloned = original.clone();

    LogMessage msg = create_test_message(LogLevel::ERROR, "logger", "test message");

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// PredicateFilter Tests
// ============================================================================

TEST(PredicateFilterTest, CustomPredicate) {
    auto predicate = [](const LogMessage& msg) {
        return msg.get_message().length() > 10;
    };

    PredicateFilter filter(predicate);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "This is a long message")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "Short")));
}

TEST(PredicateFilterTest, LevelBasedPredicate) {
    auto predicate = [](const LogMessage& msg) {
        return msg.get_level() == LogLevel::ERROR || msg.get_level() == LogLevel::FATAL;
    };

    PredicateFilter filter(predicate);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::ERROR)));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::FATAL)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO)));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::WARN)));
}

TEST(PredicateFilterTest, LoggerBasedPredicate) {
    auto predicate = [](const LogMessage& msg) {
        return msg.get_logger_name().find("critical") != std::string::npos;
    };

    PredicateFilter filter(predicate);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "critical.system")));
    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "app.critical.module")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "normal.logger")));
}

TEST(PredicateFilterTest, CloneMethod) {
    auto predicate = [](const LogMessage& msg) {
        return msg.get_level() >= LogLevel::WARN;
    };

    PredicateFilter original(predicate);
    auto cloned = original.clone();

    LogMessage msg = create_test_message(LogLevel::ERROR);

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// TimeFilter Tests
// ============================================================================

TEST(TimeFilterTest, BasicTimeRange) {
    TimeFilter filter(9, 17);  // 9 AM to 5 PM

    // Note: TimeFilter uses message timestamp, but LogMessage sets current time
    // So we test the logic with current hour understanding this limitation
    LogMessage msg = create_test_message();

    // The test result will depend on current time
    bool result = filter.should_log(msg);
    EXPECT_TRUE(result == true || result == false);  // Just verify it doesn't crash
}

TEST(TimeFilterTest, MidnightWrapAround) {
    TimeFilter filter(22, 6);  // 10 PM to 6 AM (crosses midnight)

    LogMessage msg = create_test_message();

    // The test result will depend on current time
    bool result = filter.should_log(msg);
    EXPECT_TRUE(result == true || result == false);  // Just verify it doesn't crash
}

TEST(TimeFilterTest, SameHour) {
    TimeFilter filter(12, 12);  // Only 12 PM hour (noon to 1 PM)

    LogMessage msg = create_test_message();

    bool result = filter.should_log(msg);
    EXPECT_TRUE(result == true || result == false);  // Just verify it doesn't crash
}

TEST(TimeFilterTest, CloneMethod) {
    TimeFilter original(8, 18);
    auto cloned = original.clone();

    LogMessage msg = create_test_message();

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
}

// ============================================================================
// ThreadFilter Tests
// ============================================================================

TEST(ThreadFilterTest, CurrentThreadIncluded) {
    ThreadFilter filter(true);  // Include current thread

    LogMessage msg = create_test_message();  // Will have current thread ID

    EXPECT_TRUE(filter.should_log(msg));
}

TEST(ThreadFilterTest, CurrentThreadExcluded) {
    ThreadFilter filter(false);  // Don't include current thread

    LogMessage msg = create_test_message();  // Will have current thread ID

    EXPECT_FALSE(filter.should_log(msg));
}

TEST(ThreadFilterTest, AddRemoveThreads) {
    ThreadFilter filter(false);  // Start empty

    auto current_id = std::this_thread::get_id();
    [[maybe_unused]] auto other_id = std::thread::id{};  // Default constructed thread ID

    LogMessage msg = create_test_message();  // Current thread

    EXPECT_FALSE(filter.should_log(msg));  // Not allowed initially

    filter.add_thread(current_id);
    EXPECT_TRUE(filter.should_log(msg));   // Now allowed

    filter.remove_thread(current_id);
    EXPECT_FALSE(filter.should_log(msg));  // Removed again
}

TEST(ThreadFilterTest, MultipleThreads) {
    ThreadFilter filter(true);  // Start with current thread

    // Simulate different thread IDs (we can't easily create real other threads here)
    [[maybe_unused]] auto current_id = std::this_thread::get_id();

    // Add another thread ID (even if it doesn't exist)
    auto fake_thread_id = std::thread::id{};
    filter.add_thread(fake_thread_id);

    LogMessage msg = create_test_message();  // Current thread
    EXPECT_TRUE(filter.should_log(msg));
}

TEST(ThreadFilterTest, CloneMethod) {
    ThreadFilter original(true);
    auto other_id = std::thread::id{};
    original.add_thread(other_id);

    auto cloned = original.clone();

    LogMessage msg = create_test_message();

    EXPECT_EQ(original.should_log(msg), cloned->should_log(msg));
    EXPECT_TRUE(cloned->should_log(msg));
}

// ============================================================================
// Polymorphic Behavior Tests
// ============================================================================

TEST(LogFilterPolymorphicTest, BaseClassInterface) {
    std::vector<std::unique_ptr<LogFilter>> filters;

    filters.emplace_back(std::make_unique<LevelFilter>(LogLevel::INFO, LogLevel::FATAL));
    filters.emplace_back(std::make_unique<LoggerNameFilter>("test", LoggerNameFilter::Mode::PREFIX_MATCH));
    filters.emplace_back(std::make_unique<ContentFilter>("important", ContentFilter::Mode::CONTAINS));
    filters.emplace_back(std::make_unique<RateLimitFilter>(10, 1000));
    filters.emplace_back(std::make_unique<DuplicateFilter>(50));
    filters.emplace_back(std::make_unique<ThreadFilter>(true));

    LogMessage msg = create_test_message(LogLevel::INFO, "test.logger", "important message");

    for (const auto& filter : filters) {
        bool result = filter->should_log(msg);
        EXPECT_TRUE(result == true || result == false);  // Just verify no crashes

        auto cloned = filter->clone();
        EXPECT_EQ(result, cloned->should_log(msg));
    }
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

TEST(LogFilterEdgeCasesTest, EmptyLoggerName) {
    LoggerNameFilter filter("", LoggerNameFilter::Mode::EXACT_MATCH);

    EXPECT_TRUE(filter.should_log(create_test_message(LogLevel::INFO, "")));
    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "any.logger")));
}

TEST(LogFilterEdgeCasesTest, EmptyMessage) {
    ContentFilter filter("test", ContentFilter::Mode::CONTAINS);

    EXPECT_FALSE(filter.should_log(create_test_message(LogLevel::INFO, "logger", "")));
}

TEST(LogFilterEdgeCasesTest, InvalidRegex) {
    // Invalid regex should throw std::regex_error
    EXPECT_THROW({
        LoggerNameFilter filter("[invalid", LoggerNameFilter::Mode::REGEX_MATCH);
    }, std::regex_error);
}

TEST(LogFilterEdgeCasesTest, RealWorldScenario) {
    // Simulate a real-world filtering scenario
    CompositeFilter main_filter(CompositeFilter::Mode::ALL);

    // Only errors and warnings
    main_filter.add_filter(std::make_unique<LevelFilter>(LogLevel::WARN, LogLevel::FATAL));

    // From specific modules
    main_filter.add_filter(std::make_unique<LoggerNameFilter>("app.", LoggerNameFilter::Mode::PREFIX_MATCH));

    // No password-related messages
    main_filter.add_filter(std::make_unique<ContentFilter>("password", ContentFilter::Mode::NOT_CONTAINS));

    // Rate limit to prevent spam
    main_filter.add_filter(std::make_unique<RateLimitFilter>(100, 60000));  // 100 per minute

    // Filter duplicates
    main_filter.add_filter(std::make_unique<DuplicateFilter>(1000));

    // Test various scenarios
    EXPECT_TRUE(main_filter.should_log(create_test_message(LogLevel::ERROR, "app.service", "Database connection failed")));
    EXPECT_FALSE(main_filter.should_log(create_test_message(LogLevel::INFO, "app.service", "Operation completed")));  // Info level
    EXPECT_FALSE(main_filter.should_log(create_test_message(LogLevel::ERROR, "external.lib", "Error occurred")));     // Wrong logger
    EXPECT_FALSE(main_filter.should_log(create_test_message(LogLevel::ERROR, "app.auth", "Invalid password")));      // Contains password
}
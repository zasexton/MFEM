#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>

#include "logging/loglevel.h"

using namespace fem::core::logging;

// ============================================================================
// LogLevel Enum Tests
// ============================================================================

TEST(LogLevelTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(LogLevel::TRACE), 0);
    EXPECT_EQ(static_cast<int>(LogLevel::DEBUG), 1);
    EXPECT_EQ(static_cast<int>(LogLevel::INFO), 2);
    EXPECT_EQ(static_cast<int>(LogLevel::WARN), 3);
    EXPECT_EQ(static_cast<int>(LogLevel::ERROR), 4);
    EXPECT_EQ(static_cast<int>(LogLevel::FATAL), 5);
    EXPECT_EQ(static_cast<int>(LogLevel::OFF), 6);
}

// ============================================================================
// to_string() Function Tests
// ============================================================================

TEST(LogLevelTest, ToStringValidLevels) {
    EXPECT_EQ(to_string(LogLevel::TRACE), "TRACE");
    EXPECT_EQ(to_string(LogLevel::DEBUG), "DEBUG");
    EXPECT_EQ(to_string(LogLevel::INFO), "INFO");
    EXPECT_EQ(to_string(LogLevel::WARN), "WARN");
    EXPECT_EQ(to_string(LogLevel::ERROR), "ERROR");
    EXPECT_EQ(to_string(LogLevel::FATAL), "FATAL");
    EXPECT_EQ(to_string(LogLevel::OFF), "OFF");
}

TEST(LogLevelTest, ToStringInvalidLevel) {
    // Test with invalid enum value
    LogLevel invalid_level = static_cast<LogLevel>(999);
    EXPECT_EQ(to_string(invalid_level), "UNKNOWN");
}

// ============================================================================
// to_short_string() Function Tests
// ============================================================================

TEST(LogLevelTest, ToShortStringValidLevels) {
    EXPECT_EQ(to_short_string(LogLevel::TRACE), 'T');
    EXPECT_EQ(to_short_string(LogLevel::DEBUG), 'D');
    EXPECT_EQ(to_short_string(LogLevel::INFO), 'I');
    EXPECT_EQ(to_short_string(LogLevel::WARN), 'W');
    EXPECT_EQ(to_short_string(LogLevel::ERROR), 'E');
    EXPECT_EQ(to_short_string(LogLevel::FATAL), 'F');
    EXPECT_EQ(to_short_string(LogLevel::OFF), 'O');
}

TEST(LogLevelTest, ToShortStringInvalidLevel) {
    // Test with invalid enum value
    LogLevel invalid_level = static_cast<LogLevel>(999);
    EXPECT_EQ(to_short_string(invalid_level), '?');
}

// ============================================================================
// from_string() Function Tests
// ============================================================================

TEST(LogLevelTest, FromStringValidUppercase) {
    EXPECT_EQ(from_string("TRACE"), LogLevel::TRACE);
    EXPECT_EQ(from_string("DEBUG"), LogLevel::DEBUG);
    EXPECT_EQ(from_string("INFO"), LogLevel::INFO);
    EXPECT_EQ(from_string("WARN"), LogLevel::WARN);
    EXPECT_EQ(from_string("ERROR"), LogLevel::ERROR);
    EXPECT_EQ(from_string("FATAL"), LogLevel::FATAL);
    EXPECT_EQ(from_string("OFF"), LogLevel::OFF);
}

TEST(LogLevelTest, FromStringValidLowercase) {
    EXPECT_EQ(from_string("trace"), LogLevel::TRACE);
    EXPECT_EQ(from_string("debug"), LogLevel::DEBUG);
    EXPECT_EQ(from_string("info"), LogLevel::INFO);
    EXPECT_EQ(from_string("warn"), LogLevel::WARN);
    EXPECT_EQ(from_string("error"), LogLevel::ERROR);
    EXPECT_EQ(from_string("fatal"), LogLevel::FATAL);
    EXPECT_EQ(from_string("off"), LogLevel::OFF);
}

TEST(LogLevelTest, FromStringInvalidValues) {
    EXPECT_THROW(from_string("invalid"), std::invalid_argument);
    EXPECT_THROW(from_string(""), std::invalid_argument);
    EXPECT_THROW(from_string("TRACE "), std::invalid_argument);  // Trailing space
    EXPECT_THROW(from_string(" TRACE"), std::invalid_argument);  // Leading space
    EXPECT_THROW(from_string("Trace"), std::invalid_argument);   // Mixed case
    EXPECT_THROW(from_string("trace123"), std::invalid_argument);
    EXPECT_THROW(from_string("999"), std::invalid_argument);
}

TEST(LogLevelTest, FromStringExceptionMessage) {
    try {
        from_string("invalid_level");
        FAIL() << "Expected std::invalid_argument to be thrown";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Invalid log level: invalid_level");
    }
}

// ============================================================================
// get_color_code() Function Tests
// ============================================================================

TEST(LogLevelTest, GetColorCodeValidLevels) {
    EXPECT_EQ(get_color_code(LogLevel::TRACE), "\033[37m");     // White
    EXPECT_EQ(get_color_code(LogLevel::DEBUG), "\033[36m");     // Cyan
    EXPECT_EQ(get_color_code(LogLevel::INFO), "\033[32m");      // Green
    EXPECT_EQ(get_color_code(LogLevel::WARN), "\033[33m");      // Yellow
    EXPECT_EQ(get_color_code(LogLevel::ERROR), "\033[31m");     // Red
    EXPECT_EQ(get_color_code(LogLevel::FATAL), "\033[35;1m");   // Bold Magenta
    EXPECT_EQ(get_color_code(LogLevel::OFF), "\033[0m");        // Reset
}

TEST(LogLevelTest, GetColorCodeInvalidLevel) {
    LogLevel invalid_level = static_cast<LogLevel>(999);
    EXPECT_EQ(get_color_code(invalid_level), "\033[0m");  // Reset
}

TEST(LogLevelTest, ColorResetConstant) {
    EXPECT_EQ(COLOR_RESET, "\033[0m");
}

// ============================================================================
// is_enabled() Function Tests
// ============================================================================

TEST(LogLevelTest, IsEnabledSameLevel) {
    EXPECT_TRUE(is_enabled(LogLevel::INFO, LogLevel::INFO));
    EXPECT_TRUE(is_enabled(LogLevel::ERROR, LogLevel::ERROR));
    EXPECT_TRUE(is_enabled(LogLevel::TRACE, LogLevel::TRACE));
}

TEST(LogLevelTest, IsEnabledHigherPriority) {
    // Higher priority messages should be enabled
    EXPECT_TRUE(is_enabled(LogLevel::ERROR, LogLevel::INFO));
    EXPECT_TRUE(is_enabled(LogLevel::FATAL, LogLevel::DEBUG));
    EXPECT_TRUE(is_enabled(LogLevel::WARN, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::INFO, LogLevel::DEBUG));
}

TEST(LogLevelTest, IsEnabledLowerPriority) {
    // Lower priority messages should be disabled
    EXPECT_FALSE(is_enabled(LogLevel::INFO, LogLevel::ERROR));
    EXPECT_FALSE(is_enabled(LogLevel::DEBUG, LogLevel::FATAL));
    EXPECT_FALSE(is_enabled(LogLevel::TRACE, LogLevel::WARN));
    EXPECT_FALSE(is_enabled(LogLevel::DEBUG, LogLevel::INFO));
}

TEST(LogLevelTest, IsEnabledOffLevel) {
    // When logger level is OFF, nothing should be enabled
    EXPECT_FALSE(is_enabled(LogLevel::TRACE, LogLevel::OFF));
    EXPECT_FALSE(is_enabled(LogLevel::DEBUG, LogLevel::OFF));
    EXPECT_FALSE(is_enabled(LogLevel::INFO, LogLevel::OFF));
    EXPECT_FALSE(is_enabled(LogLevel::WARN, LogLevel::OFF));
    EXPECT_FALSE(is_enabled(LogLevel::ERROR, LogLevel::OFF));
    EXPECT_FALSE(is_enabled(LogLevel::FATAL, LogLevel::OFF));
    EXPECT_TRUE(is_enabled(LogLevel::OFF, LogLevel::OFF));
}

TEST(LogLevelTest, IsEnabledTraceLevel) {
    // When logger level is TRACE, everything should be enabled
    EXPECT_TRUE(is_enabled(LogLevel::TRACE, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::DEBUG, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::INFO, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::WARN, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::ERROR, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::FATAL, LogLevel::TRACE));
    EXPECT_TRUE(is_enabled(LogLevel::OFF, LogLevel::TRACE));
}

// ============================================================================
// Stream Operator Tests
// ============================================================================

TEST(LogLevelTest, StreamOperatorValidLevels) {
    std::ostringstream oss;

    oss << LogLevel::TRACE;
    EXPECT_EQ(oss.str(), "TRACE");
    oss.str("");

    oss << LogLevel::DEBUG;
    EXPECT_EQ(oss.str(), "DEBUG");
    oss.str("");

    oss << LogLevel::INFO;
    EXPECT_EQ(oss.str(), "INFO");
    oss.str("");

    oss << LogLevel::WARN;
    EXPECT_EQ(oss.str(), "WARN");
    oss.str("");

    oss << LogLevel::ERROR;
    EXPECT_EQ(oss.str(), "ERROR");
    oss.str("");

    oss << LogLevel::FATAL;
    EXPECT_EQ(oss.str(), "FATAL");
    oss.str("");

    oss << LogLevel::OFF;
    EXPECT_EQ(oss.str(), "OFF");
}

TEST(LogLevelTest, StreamOperatorInvalidLevel) {
    std::ostringstream oss;
    LogLevel invalid_level = static_cast<LogLevel>(999);
    oss << invalid_level;
    EXPECT_EQ(oss.str(), "UNKNOWN");
}

// ============================================================================
// LogLevelConfig Tests
// ============================================================================

TEST(LogLevelConfigTest, DefaultLevelConstants) {
    EXPECT_EQ(LogLevelConfig::DEFAULT_LEVEL, LogLevel::INFO);
    EXPECT_EQ(LogLevelConfig::DEFAULT_FILE_LEVEL, LogLevel::DEBUG);
    EXPECT_EQ(LogLevelConfig::DEFAULT_CONSOLE_LEVEL, LogLevel::INFO);
}

TEST(LogLevelConfigTest, CompileTimeMinLevel) {
#ifdef NDEBUG
    EXPECT_EQ(LogLevelConfig::COMPILE_TIME_MIN_LEVEL, LogLevel::INFO);
#else
    EXPECT_EQ(LogLevelConfig::COMPILE_TIME_MIN_LEVEL, LogLevel::TRACE);
#endif
}

// ============================================================================
// LogLevelThreshold Tests
// ============================================================================

TEST(LogLevelThresholdTest, DefaultConstruction) {
    LogLevelThreshold threshold;
    EXPECT_EQ(threshold.get_level(), LogLevelConfig::DEFAULT_LEVEL);
}

TEST(LogLevelThresholdTest, ExplicitConstruction) {
    LogLevelThreshold threshold(LogLevel::ERROR);
    EXPECT_EQ(threshold.get_level(), LogLevel::ERROR);
}

TEST(LogLevelThresholdTest, ShouldLogMethod) {
    LogLevelThreshold threshold(LogLevel::WARN);

    // Messages at or above threshold should be logged
    EXPECT_TRUE(threshold.should_log(LogLevel::WARN));
    EXPECT_TRUE(threshold.should_log(LogLevel::ERROR));
    EXPECT_TRUE(threshold.should_log(LogLevel::FATAL));
    EXPECT_TRUE(threshold.should_log(LogLevel::OFF));

    // Messages below threshold should not be logged
    EXPECT_FALSE(threshold.should_log(LogLevel::TRACE));
    EXPECT_FALSE(threshold.should_log(LogLevel::DEBUG));
    EXPECT_FALSE(threshold.should_log(LogLevel::INFO));
}

TEST(LogLevelThresholdTest, SetLevel) {
    LogLevelThreshold threshold(LogLevel::INFO);
    EXPECT_EQ(threshold.get_level(), LogLevel::INFO);

    threshold.set_level(LogLevel::ERROR);
    EXPECT_EQ(threshold.get_level(), LogLevel::ERROR);

    threshold.set_level(LogLevel::TRACE);
    EXPECT_EQ(threshold.get_level(), LogLevel::TRACE);
}

TEST(LogLevelThresholdTest, IncreaseVerbosity) {
    LogLevelThreshold threshold(LogLevel::ERROR);

    // Should decrease level value (increase verbosity)
    threshold.increase_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::WARN);

    threshold.increase_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::INFO);

    threshold.increase_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::DEBUG);

    threshold.increase_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::TRACE);

    // Should not go below TRACE
    threshold.increase_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::TRACE);
}

TEST(LogLevelThresholdTest, DecreaseVerbosity) {
    LogLevelThreshold threshold(LogLevel::TRACE);

    // Should increase level value (decrease verbosity)
    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::DEBUG);

    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::INFO);

    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::WARN);

    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::ERROR);

    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::FATAL);

    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::OFF);

    // Should not go above OFF
    threshold.decrease_verbosity();
    EXPECT_EQ(threshold.get_level(), LogLevel::OFF);
}

TEST(LogLevelThresholdTest, VerbosityBoundaryConditions) {
    // Test increase verbosity at minimum level
    LogLevelThreshold min_threshold(LogLevel::TRACE);
    min_threshold.increase_verbosity();
    EXPECT_EQ(min_threshold.get_level(), LogLevel::TRACE);

    // Test decrease verbosity at maximum level
    LogLevelThreshold max_threshold(LogLevel::OFF);
    max_threshold.decrease_verbosity();
    EXPECT_EQ(max_threshold.get_level(), LogLevel::OFF);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(LogLevelIntegrationTest, RoundTripStringConversion) {
    // Test that converting to string and back preserves the value
    std::vector<LogLevel> levels = {
        LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
        LogLevel::WARN, LogLevel::ERROR, LogLevel::FATAL, LogLevel::OFF
    };

    for (LogLevel level : levels) {
        std::string str = std::string(to_string(level));
        LogLevel converted = from_string(str);
        EXPECT_EQ(level, converted);
    }
}

TEST(LogLevelIntegrationTest, ThresholdWithIsEnabled) {
    LogLevelThreshold threshold(LogLevel::INFO);

    // Test that threshold.should_log() and is_enabled() give same results
    std::vector<LogLevel> test_levels = {
        LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
        LogLevel::WARN, LogLevel::ERROR, LogLevel::FATAL, LogLevel::OFF
    };

    for (LogLevel level : test_levels) {
        bool threshold_result = threshold.should_log(level);
        bool is_enabled_result = is_enabled(level, threshold.get_level());
        EXPECT_EQ(threshold_result, is_enabled_result)
            << "Mismatch for level " << level << " with threshold " << threshold.get_level();
    }
}

TEST(LogLevelIntegrationTest, ColorCodeNonEmpty) {
    // Ensure all color codes are non-empty strings
    std::vector<LogLevel> levels = {
        LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
        LogLevel::WARN, LogLevel::ERROR, LogLevel::FATAL, LogLevel::OFF
    };

    for (LogLevel level : levels) {
        std::string_view color = get_color_code(level);
        EXPECT_FALSE(color.empty()) << "Color code for " << level << " is empty";
        EXPECT_TRUE(color.starts_with("\033[")) << "Color code for " << level << " doesn't start with ANSI escape";
    }
}
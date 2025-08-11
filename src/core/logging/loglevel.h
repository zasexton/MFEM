#pragma once

#ifndef LOGGING_LOGLEVEL_H
#define LOGGING_LOGLEVEL_H

#include <string>
#include <string_view>
#include <iostream>
#include <array>

namespace fem::core::logging {

/**
 * @brief Logging severity levels
 *
 * Defines the importance/severity of log messages. Higher values indicate
 * more severe conditions. The levels follow common logging conventions:
 * - TRACE: Detailed debugging information (function entry/exit, variable values)
 * - DEBUG: Debugging information for development
 * - INFO: General informational messages
 * - WARN: Warning conditions that might need attention
 * - ERROR: Error conditions that need handling
 * - FATAL: Critical errors that cause program termination
 * - OFF: Disable all logging
 */
    enum class LogLevel : int {
        TRACE = 0,
        DEBUG = 1,
        INFO  = 2,
        WARN  = 3,
        ERROR = 4,
        FATAL = 5,
        OFF   = 6
    };

/**
 * @brief Convert LogLevel to string representation
 */
    constexpr std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::TRACE: return "TRACE";
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO";
    case LogLevel::WARN:  return "WARN";
    case LogLevel::ERROR: return "ERROR";
    case LogLevel::FATAL: return "FATAL";
    case LogLevel::OFF:   return "OFF";
    default:              return "UNKNOWN";
}
}

/**
 * @brief Convert LogLevel to short string (single character)
 */
constexpr char to_short_string(LogLevel level) noexcept {
switch (level) {
case LogLevel::TRACE: return 'T';
case LogLevel::DEBUG: return 'D';
case LogLevel::INFO:  return 'I';
case LogLevel::WARN:  return 'W';
case LogLevel::ERROR: return 'E';
case LogLevel::FATAL: return 'F';
case LogLevel::OFF:   return 'O';
default:              return '?';
}
}

/**
 * @brief Convert string to LogLevel
 */
inline LogLevel from_string(std::string_view str) {
    if (str == "TRACE" || str == "trace") return LogLevel::TRACE;
    if (str == "DEBUG" || str == "debug") return LogLevel::DEBUG;
    if (str == "INFO"  || str == "info")  return LogLevel::INFO;
    if (str == "WARN"  || str == "warn")  return LogLevel::WARN;
    if (str == "ERROR" || str == "error") return LogLevel::ERROR;
    if (str == "FATAL" || str == "fatal") return LogLevel::FATAL;
    if (str == "OFF"   || str == "off")   return LogLevel::OFF;

    throw std::invalid_argument("Invalid log level: " + std::string(str));
}

/**
 * @brief Get color code for log level (for terminal output)
 */
constexpr std::string_view get_color_code(LogLevel level) noexcept {
switch (level) {
case LogLevel::TRACE: return "\033[37m";     // White
case LogLevel::DEBUG: return "\033[36m";     // Cyan
case LogLevel::INFO:  return "\033[32m";     // Green
case LogLevel::WARN:  return "\033[33m";     // Yellow
case LogLevel::ERROR: return "\033[31m";     // Red
case LogLevel::FATAL: return "\033[35;1m";   // Bold Magenta
default:              return "\033[0m";      // Reset
}
}

/**
 * @brief Reset color code for terminal
 */
constexpr std::string_view COLOR_RESET = "\033[0m";

/**
 * @brief Check if one log level is enabled by another
 */
constexpr bool is_enabled(LogLevel message_level, LogLevel logger_level) noexcept {
return static_cast<int>(message_level) >= static_cast<int>(logger_level);
}

/**
 * @brief Stream output operator for LogLevel
 */
inline std::ostream& operator<<(std::ostream& os, LogLevel level) {
    return os << to_string(level);
}

/**
 * @brief Configuration for log levels
 */
struct LogLevelConfig {
    static constexpr LogLevel DEFAULT_LEVEL = LogLevel::INFO;
    static constexpr LogLevel DEFAULT_FILE_LEVEL = LogLevel::DEBUG;
    static constexpr LogLevel DEFAULT_CONSOLE_LEVEL = LogLevel::INFO;

    // Compile-time minimum level (can optimize out lower-level logs)
#ifdef NDEBUG
    static constexpr LogLevel COMPILE_TIME_MIN_LEVEL = LogLevel::INFO;
#else
    static constexpr LogLevel COMPILE_TIME_MIN_LEVEL = LogLevel::TRACE;
#endif
};

/**
 * @brief Utility to work with log level thresholds
 */
class LogLevelThreshold {
public:
    explicit LogLevelThreshold(LogLevel level = LogLevelConfig::DEFAULT_LEVEL)
            : level_(level) {}

    /**
     * @brief Check if a message should be logged
     */
    [[nodiscard]] bool should_log(LogLevel message_level) const noexcept {
        return is_enabled(message_level, level_);
    }

    /**
     * @brief Set the threshold level
     */
    void set_level(LogLevel level) noexcept { level_ = level; }

    /**
     * @brief Get the current threshold level
     */
    [[nodiscard]] LogLevel get_level() const noexcept { return level_; }

    /**
     * @brief Increase verbosity (lower the threshold)
     */
    void increase_verbosity() noexcept {
        if (static_cast<int>(level_) > 0) {
            level_ = static_cast<LogLevel>(static_cast<int>(level_) - 1);
        }
    }

    /**
     * @brief Decrease verbosity (raise the threshold)
     */
    void decrease_verbosity() noexcept {
        if (static_cast<int>(level_) < static_cast<int>(LogLevel::OFF)) {
            level_ = static_cast<LogLevel>(static_cast<int>(level_) + 1);
        }
    }

private:
    LogLevel level_;
};

} // namespace fem::core::logging

#endif //LOGGING_LOGLEVEL_H

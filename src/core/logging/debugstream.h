#pragma once

#ifndef LOGGING_DEBUGSTREAM_H
#define LOGGING_DEBUGSTREAM_H

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <functional>
#include <vector>
#include <unordered_map>
#include <thread>

namespace fem::core::logging {

// Forward declarations
class DebugStream;
class DebugStreamBuffer;

/**
 * @brief Debug stream severity levels
 */
enum class DebugLevel {
    NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5,
    ALL = 6
};

/**
 * @brief Convert debug level to string
 */
inline const char* debug_level_to_string(DebugLevel level) {
    switch (level) {
        case DebugLevel::NONE:    return "NONE";
        case DebugLevel::ERROR:   return "ERROR";
        case DebugLevel::WARNING: return "WARN";
        case DebugLevel::INFO:    return "INFO";
        case DebugLevel::DEBUG:   return "DEBUG";
        case DebugLevel::TRACE:   return "TRACE";
        case DebugLevel::ALL:     return "ALL";
        default:                  return "UNKNOWN";
    }
}

/**
 * @brief Debug stream configuration
 */
struct DebugStreamConfig {
    bool enabled = true;
    bool include_timestamp = true;
    bool include_thread_id = false;
    bool include_level = true;
    bool include_location = false;
    bool auto_flush = true;
    bool colorize = false;
    DebugLevel min_level = DebugLevel::INFO;
    std::string prefix = "";
    std::string suffix = "";
};

/**
 * @brief Custom stream buffer for debug output
 */
class DebugStreamBuffer : public std::streambuf {
public:
    explicit DebugStreamBuffer(std::ostream& output = std::cerr)
        : output_stream_(output)
        , buffer_()
        , mutex_() {}

    void set_output(std::ostream& output) {
        std::lock_guard<std::mutex> lock(mutex_);
        output_stream_ = std::ref(output);
    }

    void set_filter(std::function<bool(const std::string&)> filter) {
        std::lock_guard<std::mutex> lock(mutex_);
        filter_ = filter;
    }

    void flush_buffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer_.empty()) {
            if (!filter_ || filter_(buffer_)) {
                output_stream_.get() << buffer_;
                output_stream_.get().flush();
            }
            buffer_.clear();
        }
    }

protected:
    int overflow(int c) override {
        if (c != EOF) {
            std::lock_guard<std::mutex> lock(mutex_);
            buffer_ += static_cast<char>(c);
            if (c == '\n') {
                if (!filter_ || filter_(buffer_)) {
                    output_stream_.get() << buffer_;
                    if (auto_flush_) {
                        output_stream_.get().flush();
                    }
                }
                buffer_.clear();
            }
        }
        return c;
    }

    int sync() override {
        flush_buffer();
        return 0;
    }

private:
    std::reference_wrapper<std::ostream> output_stream_;
    std::string buffer_;
    mutable std::mutex mutex_;
    std::function<bool(const std::string&)> filter_;
    bool auto_flush_ = true;
};

/**
 * @brief Thread-safe debug stream with filtering and formatting
 */
class DebugStream : public std::ostream {
public:
    /**
     * @brief Default constructor
     */
    DebugStream()
        : std::ostream(&buffer_)
        , buffer_()
        , config_()
        , current_level_(static_cast<int>(DebugLevel::INFO))
        , enabled_(true)
        , min_level_(static_cast<int>(DebugLevel::INFO))
        , message_count_(0)
        , start_time_(std::chrono::steady_clock::now()) {}

    /**
     * @brief Constructor with output stream
     */
    explicit DebugStream(std::ostream& output)
        : std::ostream(&buffer_)
        , buffer_(output)
        , config_()
        , current_level_(static_cast<int>(DebugLevel::INFO))
        , enabled_(true)
        , min_level_(static_cast<int>(DebugLevel::INFO))
        , message_count_(0)
        , start_time_(std::chrono::steady_clock::now()) {}

    /**
     * @brief Constructor with configuration
     */
    explicit DebugStream(const DebugStreamConfig& config)
        : std::ostream(&buffer_)
        , buffer_()
        , config_(config)
        , current_level_(static_cast<int>(DebugLevel::INFO))
        , enabled_(config.enabled)
        , min_level_(static_cast<int>(config.min_level))
        , message_count_(0)
        , start_time_(std::chrono::steady_clock::now()) {}

    // Configuration methods
    void set_config(const DebugStreamConfig& config) {
        config_ = config;
        enabled_ = config.enabled;
        min_level_.store(static_cast<int>(config.min_level));
    }

    DebugStreamConfig get_config() const {
        return config_;
    }

    void set_enabled(bool enabled) {
        enabled_.store(enabled);
        config_.enabled = enabled;  // Keep config in sync
    }

    bool is_enabled() const {
        return enabled_.load();
    }

    void set_level(DebugLevel level) {
        current_level_.store(static_cast<int>(level));
    }

    DebugLevel get_level() const {
        return static_cast<DebugLevel>(current_level_.load());
    }

    void set_min_level(DebugLevel level) {
        config_.min_level = level;
        min_level_.store(static_cast<int>(level));
    }

    bool should_log(DebugLevel level) const {
        // No mutex here to avoid deadlock - use atomics
        return enabled_.load() &&
               static_cast<int>(level) <= static_cast<int>(min_level_.load());
    }

    // Output methods
    DebugStream& operator()(DebugLevel level) {
        current_level_.store(static_cast<int>(level));
        // Always increment message count for any attempt
        message_count_.fetch_add(1);
        if (should_log(level)) {
            start_message();
        }
        return *this;
    }

    template<typename T>
    DebugStream& operator<<(const T& value) {
        auto level = static_cast<DebugLevel>(current_level_.load());
        if (should_log(level)) {
            static_cast<std::ostream&>(*this) << value;
        }
        return *this;
    }

    // Special handling for stream manipulators
    DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        auto level = static_cast<DebugLevel>(current_level_.load());
        if (should_log(level)) {
            if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
                // Write suffix before endl
                write_footer();
                manip(*this);
                // Flush after endl if needed
                if (config_.auto_flush) {
                    flush();
                }
            } else {
                manip(*this);
            }
        }
        return *this;
    }

    // Statistics
    uint64_t get_message_count() const {
        return message_count_.load();
    }

    std::chrono::milliseconds get_uptime() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    }

    void reset_statistics() {
        message_count_.store(0);
        start_time_ = std::chrono::steady_clock::now();
    }

    // Buffer control
    void flush() {
        buffer_.flush_buffer();
    }

    void set_output(std::ostream& output) {
        buffer_.set_output(output);
    }

    void set_filter(std::function<bool(const std::string&)> filter) {
        buffer_.set_filter(filter);
    }

    // Utility methods
    void write_header() {
        if (config_.include_timestamp) {
            write_timestamp();
        }
        if (config_.include_thread_id) {
            write_thread_id();
        }
        if (config_.include_level) {
            write_level();
        }
        if (!config_.prefix.empty()) {
            // Write directly to base stream to avoid recursion
            static_cast<std::ostream&>(*this) << config_.prefix;
        }
    }

    void write_footer() {
        if (!config_.suffix.empty()) {
            // Write directly to base stream to avoid recursion
            static_cast<std::ostream&>(*this) << config_.suffix;
        }
    }

private:
    void start_message() {
        write_header();
        // Message count is now incremented in operator(), not here
    }

    void end_message() {
        // Deprecated - footer is now written before endl in operator<<
        if (config_.auto_flush) {
            flush();
        }
    }

    void write_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        // Format: [YYYY-MM-DD HH:MM:SS.mmm]
        std::tm tm = *std::localtime(&time_t);
        static_cast<std::ostream&>(*this) << "["
            << std::setfill('0')
            << std::setw(4) << (tm.tm_year + 1900) << "-"
            << std::setw(2) << (tm.tm_mon + 1) << "-"
            << std::setw(2) << tm.tm_mday << " "
            << std::setw(2) << tm.tm_hour << ":"
            << std::setw(2) << tm.tm_min << ":"
            << std::setw(2) << tm.tm_sec << "."
            << std::setw(3) << ms.count()
            << "] ";
    }

    void write_thread_id() {
        // Write directly to base stream to avoid recursion
        static_cast<std::ostream&>(*this) << "[TID:" << std::this_thread::get_id() << "] ";
    }

    void write_level() {
        auto level = static_cast<DebugLevel>(current_level_.load());
        if (config_.colorize) {
            write_colored_level(level);
        } else {
            // Write directly to base stream to avoid recursion
            static_cast<std::ostream&>(*this) << "[" << debug_level_to_string(level) << "] ";
        }
    }

    void write_colored_level(DebugLevel level) {
        const char* color = "";
        switch (level) {
            case DebugLevel::ERROR:   color = "\033[31m"; break; // Red
            case DebugLevel::WARNING: color = "\033[33m"; break; // Yellow
            case DebugLevel::INFO:    color = "\033[32m"; break; // Green
            case DebugLevel::DEBUG:   color = "\033[36m"; break; // Cyan
            case DebugLevel::TRACE:   color = "\033[90m"; break; // Gray
            default: color = "\033[0m"; break; // Reset
        }
        // Write directly to base stream to avoid recursion
        static_cast<std::ostream&>(*this) << color << "[" << debug_level_to_string(level) << "]\033[0m ";
    }

private:
    DebugStreamBuffer buffer_;
    DebugStreamConfig config_;
    std::atomic<int> current_level_;
    std::atomic<bool> enabled_;
    std::atomic<int> min_level_;
    std::atomic<uint64_t> message_count_;
    std::chrono::steady_clock::time_point start_time_;
};

/**
 * @brief Global debug stream instance
 */
class GlobalDebugStream {
public:
    static DebugStream& instance() {
        static DebugStream stream(std::cerr);
        return stream;
    }

    static DebugStream& get(const std::string& name) {
        static std::unordered_map<std::string, std::unique_ptr<DebugStream>> streams;
        static std::mutex mutex;

        std::lock_guard<std::mutex> lock(mutex);
        auto it = streams.find(name);
        if (it == streams.end()) {
            streams[name] = std::make_unique<DebugStream>();
            return *streams[name];
        }
        return *it->second;
    }
};

// Convenience macros
#ifdef DEBUG
    #define DEBUG_STREAM fem::core::logging::GlobalDebugStream::instance()
    #define DEBUG_LOG(level) DEBUG_STREAM(fem::core::logging::DebugLevel::level)
    #define DEBUG_ERROR DEBUG_LOG(ERROR)
    #define DEBUG_WARN DEBUG_LOG(WARNING)
    #define DEBUG_INFO DEBUG_LOG(INFO)
    #define DEBUG_DEBUG DEBUG_LOG(DEBUG)
    #define DEBUG_TRACE DEBUG_LOG(TRACE)
#else
    #define DEBUG_STREAM if(false) fem::core::logging::GlobalDebugStream::instance()
    #define DEBUG_LOG(level) if(false) DEBUG_STREAM(fem::core::logging::DebugLevel::level)
    #define DEBUG_ERROR if(false) DEBUG_LOG(ERROR)
    #define DEBUG_WARN if(false) DEBUG_LOG(WARNING)
    #define DEBUG_INFO if(false) DEBUG_LOG(INFO)
    #define DEBUG_DEBUG if(false) DEBUG_LOG(DEBUG)
    #define DEBUG_TRACE if(false) DEBUG_LOG(TRACE)
#endif

} // namespace fem::core::logging

#endif // LOGGING_DEBUGSTREAM_H
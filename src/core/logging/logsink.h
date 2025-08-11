#pragma once

#ifndef LOGGING_LOGSINK_H
#define LOGGING_LOGSINK_H

#include <memory>
#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>
#include <sstream>

#include "logmessage.h"
#include "logformatter.h"
#include "logfilter.h"

#include "../base/object.h"
#include "../base/interface.h"
#include "../base/factory.h"
#include "../base/policy.h"

namespace fem::core::logging {

/**
 * @brief Interface for log sinks
 */
    class ILogSink : public TypedInterface<ILogSink> {
    public:
        virtual void write(const LogMessage& message) = 0;
        virtual void flush() = 0;
        virtual void set_formatter(std::unique_ptr<LogFormatter> formatter) = 0;
        virtual void set_level(LogLevel level) = 0;
        virtual bool should_log(const LogMessage& message) const = 0;
    };

/**
 * @brief Abstract base class for log output destinations
 *
 * Inherits from Object for lifecycle management and ILogSink for interface
 */
    class LogSink : public Object,
                    public ILogSink,
                    public IConfigurable {
    public:
        LogSink(const std::string& sink_type = "LogSink")
                : Object(sink_type) {}

        virtual ~LogSink() = default;

        /**
         * @brief Write a log message to the sink
         */
        virtual void write(const LogMessage& message) override = 0;

        /**
         * @brief Flush any buffered data
         */
        virtual void flush() override = 0;

        /**
         * @brief Set the formatter for this sink
         */
        void set_formatter(std::unique_ptr<LogFormatter> formatter) override {
            std::lock_guard<std::mutex> lock(formatter_mutex_);
            formatter_ = std::move(formatter);
        }

        /**
         * @brief Get the formatter (creates default if none set)
         */
        LogFormatter* get_formatter() {
            std::lock_guard<std::mutex> lock(formatter_mutex_);
            if (!formatter_) {
                formatter_ = std::make_unique<BasicLogFormatter>();
            }
            return formatter_.get();
        }

        /**
         * @brief Add a filter to this sink
         */
        void add_filter(std::unique_ptr<LogFilter> filter) {
            std::lock_guard<std::mutex> lock(filters_mutex_);
            filters_.push_back(std::move(filter));
        }

        /**
         * @brief Clear all filters
         */
        void clear_filters() {
            std::lock_guard<std::mutex> lock(filters_mutex_);
            filters_.clear();
        }

        /**
         * @brief Set minimum log level for this sink
         */
        void set_level(LogLevel level) override {
            min_level_.store(level, std::memory_order_relaxed);
        }

        /**
         * @brief Get minimum log level
         */
        [[nodiscard]] LogLevel get_level() const {
            return min_level_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Check if a message should be written to this sink
         */
        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            // Check level
            if (!is_enabled(message.get_level(), get_level())) {
                return false;
            }

            // Check filters
            std::lock_guard<std::mutex> lock(filters_mutex_);
            for (const auto& filter : filters_) {
                if (!filter->should_log(message)) {
                    return false;
                }
            }

            return is_enabled();
        }

        /**
         * @brief Enable/disable the sink
         */
        void set_enabled(bool enabled) {
            enabled_.store(enabled, std::memory_order_relaxed);
        }

        /**
         * @brief Check if sink is enabled
         */
        [[nodiscard]] bool is_enabled() const {
            return enabled_.load(std::memory_order_relaxed);
        }

        // IConfigurable interface
        bool configure(const std::unordered_map<std::string, std::string>& params) override {
            if (auto it = params.find("level"); it != params.end()) {
                try {
                    set_level(from_string(it->second));
                } catch (...) {
                    return false;
                }
            }

            if (auto it = params.find("enabled"); it != params.end()) {
                set_enabled(it->second == "true" || it->second == "1");
            }

            return true;
        }

        std::unordered_map<std::string, std::string> get_configuration() const override {
            return {
                    {"level", std::string(to_string(get_level()))},
                    {"enabled", is_enabled() ? "true" : "false"}
            };
        }

        std::vector<std::string> get_supported_keys() const override {
            return {"level", "enabled"};
        }

        void reset_configuration() override {
            set_level(LogLevel::TRACE);
            set_enabled(true);
        }

    protected:
        std::unique_ptr<LogFormatter> formatter_;
        mutable std::mutex formatter_mutex_;

        std::vector<std::unique_ptr<LogFilter>> filters_;
        mutable std::mutex filters_mutex_;

        std::atomic<LogLevel> min_level_{LogLevel::TRACE};
        std::atomic<bool> enabled_{true};
    };

/**
 * @brief Console sink for stdout/stderr output
 */
    class ConsoleSink : public LogSink,
                        public NonCopyable<ConsoleSink> {
    public:
        enum class OutputMode {
            STDOUT_ONLY,
            STDERR_ONLY,
            SPLIT_BY_LEVEL
        };

        explicit ConsoleSink(OutputMode mode = OutputMode::SPLIT_BY_LEVEL,
                             bool use_color = true)
                : LogSink("ConsoleSink")
                , mode_(mode)
                , use_color_(use_color) {}

        void write(const LogMessage& message) override {
            if (!should_log(message)) return;

            std::string formatted = get_formatter()->format(message);

            // Add color if enabled
            if (use_color_ && is_tty()) {
                formatted = add_color(message.get_level(), formatted);
            }

            // Choose output stream
            std::ostream* out = &std::cout;
            switch (mode_) {
                case OutputMode::STDOUT_ONLY:
                    out = &std::cout;
                    break;
                case OutputMode::STDERR_ONLY:
                    out = &std::cerr;
                    break;
                case OutputMode::SPLIT_BY_LEVEL:
                    out = (message.get_level() >= LogLevel::WARN) ? &std::cerr : &std::cout;
                    break;
            }

            // Thread-safe output
            std::lock_guard<std::mutex> lock(output_mutex_);
            *out << formatted << std::endl;
        }

        void flush() override {
            std::cout.flush();
            std::cerr.flush();
        }

        void set_color_enabled(bool enabled) { use_color_ = enabled; }
        [[nodiscard]] bool is_color_enabled() const { return use_color_; }

        // IConfigurable extensions
        bool configure(const std::unordered_map<std::string, std::string>& params) override {
            LogSink::configure(params);

            if (auto it = params.find("color"); it != params.end()) {
                use_color_ = (it->second == "true" || it->second == "1");
            }

            if (auto it = params.find("mode"); it != params.end()) {
                if (it->second == "stdout") mode_ = OutputMode::STDOUT_ONLY;
                else if (it->second == "stderr") mode_ = OutputMode::STDERR_ONLY;
                else if (it->second == "split") mode_ = OutputMode::SPLIT_BY_LEVEL;
            }

            return true;
        }

        std::vector<std::string> get_supported_keys() const override {
            auto keys = LogSink::get_supported_keys();
            keys.push_back("color");
            keys.push_back("mode");
            return keys;
        }

    private:
        OutputMode mode_;
        bool use_color_;
        mutable std::mutex output_mutex_;

        [[nodiscard]] bool is_tty() const {
            // Simple check - in production, use isatty() on Unix or GetConsoleMode() on Windows
            return true;
        }

        [[nodiscard]] std::string add_color(LogLevel level, const std::string& text) const {
            return std::string(get_color_code(level)) + text + std::string(COLOR_RESET);
        }
    };

/**
 * @brief File sink for logging to files
 */
    class FileSink : public LogSink,
                     public NonCopyable<FileSink> {
    public:
        explicit FileSink(const std::string& filename = "",
                          bool truncate = false,
                          bool auto_flush = false)
                : LogSink("FileSink")
                , filename_(filename)
                , auto_flush_(auto_flush) {

            if (!filename.empty()) {
                open_file(truncate);
            }
        }

        ~FileSink() override {
            if (file_.is_open()) {
                file_.close();
            }
        }

        void write(const LogMessage& message) override {
            if (!should_log(message)) return;

            std::string formatted = get_formatter()->format(message);

            std::lock_guard<std::mutex> lock(file_mutex_);
            if (file_.is_open()) {
                file_ << formatted << '\n';

                if (auto_flush_) {
                    file_.flush();
                }
            }
        }

        void flush() override {
            std::lock_guard<std::mutex> lock(file_mutex_);
            if (file_.is_open()) {
                file_.flush();
            }
        }

        [[nodiscard]] const std::string& get_filename() const { return filename_; }
        [[nodiscard]] bool is_open() const { return file_.is_open(); }

        void reopen() {
            std::lock_guard<std::mutex> lock(file_mutex_);
            file_.close();
            open_file(false);
        }

        // IConfigurable extensions
        bool configure(const std::unordered_map<std::string, std::string>& params) override {
            LogSink::configure(params);

            if (auto it = params.find("filename"); it != params.end()) {
                filename_ = it->second;
                open_file(false);
            }

            if (auto it = params.find("auto_flush"); it != params.end()) {
                auto_flush_ = (it->second == "true" || it->second == "1");
            }

            if (auto it = params.find("truncate"); it != params.end()) {
                bool truncate = (it->second == "true" || it->second == "1");
                if (truncate && file_.is_open()) {
                    file_.close();
                    open_file(true);
                }
            }

            return true;
        }

        std::vector<std::string> get_supported_keys() const override {
            auto keys = LogSink::get_supported_keys();
            keys.insert(keys.end(), {"filename", "auto_flush", "truncate"});
            return keys;
        }

    private:
        void open_file(bool truncate) {
            if (filename_.empty()) return;

            auto mode = truncate ? std::ios::trunc : std::ios::app;
            file_.open(filename_, mode);

            if (!file_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + filename_);
            }
        }

        std::string filename_;
        std::ofstream file_;
        bool auto_flush_;
        mutable std::mutex file_mutex_;
    };

/**
 * @brief Memory buffer sink for capturing logs in memory
 */
    class MemorySink : public LogSink {
    public:
        explicit MemorySink(size_t max_messages = 10000)
                : LogSink("MemorySink")
                , max_messages_(max_messages) {}

        void write(const LogMessage& message) override {
            if (!should_log(message)) return;

            std::lock_guard<std::mutex> lock(buffer_mutex_);

            if (messages_.size() >= max_messages_) {
                messages_.erase(messages_.begin());
            }

            messages_.push_back(message.clone());
            formatted_buffer_ << get_formatter()->format(message) << '\n';
        }

        void flush() override {
            // Memory sink doesn't need flushing
        }

        [[nodiscard]] std::vector<LogMessage> get_messages() const {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            return messages_;
        }

        [[nodiscard]] std::string get_formatted_content() const {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            return formatted_buffer_.str();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            messages_.clear();
            formatted_buffer_.str("");
            formatted_buffer_.clear();
        }

        [[nodiscard]] size_t size() const {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            return messages_.size();
        }

        // IConfigurable extensions
        bool configure(const std::unordered_map<std::string, std::string>& params) override {
            LogSink::configure(params);

            if (auto it = params.find("max_messages"); it != params.end()) {
                try {
                    max_messages_ = std::stoull(it->second);
                } catch (...) {
                    return false;
                }
            }

            return true;
        }

        std::vector<std::string> get_supported_keys() const override {
            auto keys = LogSink::get_supported_keys();
            keys.push_back("max_messages");
            return keys;
        }

    private:
        size_t max_messages_;
        mutable std::mutex buffer_mutex_;
        std::vector<LogMessage> messages_;
        std::stringstream formatted_buffer_;
    };

/**
 * @brief Multi-sink that forwards to multiple sinks
 */
    class MultiSink : public LogSink {
    public:
        MultiSink() : LogSink("MultiSink") {}

        void add_sink(std::shared_ptr<LogSink> sink) {
            std::lock_guard<std::mutex> lock(sinks_mutex_);
            sinks_.push_back(std::move(sink));
        }

        void remove_sink(const std::shared_ptr<LogSink>& sink) {
            std::lock_guard<std::mutex> lock(sinks_mutex_);
            sinks_.erase(
                    std::remove(sinks_.begin(), sinks_.end(), sink),
                    sinks_.end()
            );
        }

        void write(const LogMessage& message) override {
            if (!is_enabled()) return;

            std::lock_guard<std::mutex> lock(sinks_mutex_);
            for (auto& sink : sinks_) {
                sink->write(message);
            }
        }

        void flush() override {
            std::lock_guard<std::mutex> lock(sinks_mutex_);
            for (auto& sink : sinks_) {
                sink->flush();
            }
        }

        [[nodiscard]] size_t sink_count() const {
            std::lock_guard<std::mutex> lock(sinks_mutex_);
            return sinks_.size();
        }

    private:
        mutable std::mutex sinks_mutex_;
        std::vector<std::shared_ptr<LogSink>> sinks_;
    };

/**
 * @brief Null sink that discards all messages
 */
    class NullSink : public LogSink {
    public:
        NullSink() : LogSink("NullSink") {}

        void write(const LogMessage&) override {}
        void flush() override {}
    };

// Register sink types with factory
    inline void register_sink_types() {
        auto& factory = Factory<LogSink>::instance();
        factory.register_type<ConsoleSink>("console");
        factory.register_type<FileSink>("file");
        factory.register_type<MemorySink>("memory");
        factory.register_type<MultiSink>("multi");
        factory.register_type<NullSink>("null");
    }

} // namespace fem::core::logging

#endif //LOGGING_LOGSINK_H

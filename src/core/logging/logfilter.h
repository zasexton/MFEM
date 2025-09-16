#pragma once

#ifndef LOGGING_LOGFILTER_H
#define LOGGING_LOGFILTER_H

#include <regex>
#include <functional>
#include <unordered_set>
#include <queue>

#include "logmessage.h"

namespace fem::core::logging {

/**
 * @brief Abstract base class for log message filtering
 *
 * Filters determine whether a log message should be processed or discarded.
 * They can be attached to loggers or sinks for fine-grained control.
 *
 * Usage context:
 * - Filter out noisy debug messages from specific modules
 * - Only log errors from certain components
 * - Rate limiting to prevent log spam
 * - Content-based filtering (e.g., ignore messages containing passwords)
 */
    class LogFilter {
    public:
        virtual ~LogFilter() = default;

        /**
         * @brief Check if a message should be logged
         * @return true if message should be logged, false to filter out
         */
        [[nodiscard]] virtual bool should_log(const LogMessage& message) const = 0;

        /**
         * @brief Clone the filter
         */
        [[nodiscard]] virtual std::unique_ptr<LogFilter> clone() const = 0;
    };

/**
 * @brief Filter based on log level
 */
    class LevelFilter : public LogFilter {
    public:
        explicit LevelFilter(LogLevel min_level = LogLevel::TRACE,
                             LogLevel max_level = LogLevel::FATAL)
                : min_level_(min_level), max_level_(max_level) {}

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            auto level = message.get_level();
            return level >= min_level_ && level <= max_level_;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<LevelFilter>(min_level_, max_level_);
        }

    private:
        LogLevel min_level_;
        LogLevel max_level_;
    };

/**
 * @brief Filter based on logger name pattern
 */
    class LoggerNameFilter : public LogFilter {
    public:
        enum class Mode {
            EXACT_MATCH,      // Logger name must exactly match
            PREFIX_MATCH,     // Logger name must start with pattern
            REGEX_MATCH,      // Logger name must match regex
            BLACKLIST         // Exclude loggers matching pattern
        };

        LoggerNameFilter(const std::string& pattern, Mode mode = Mode::PREFIX_MATCH)
                : pattern_(pattern), mode_(mode) {
            if (mode_ == Mode::REGEX_MATCH) {
                regex_ = std::regex(pattern);
            }
        }

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            const auto& logger_name = message.get_logger_name();

            switch (mode_) {
                case Mode::EXACT_MATCH:
                    return logger_name == pattern_;

                case Mode::PREFIX_MATCH:
                    return logger_name.starts_with(pattern_);

                case Mode::REGEX_MATCH:
                    return std::regex_match(logger_name, regex_);

                case Mode::BLACKLIST:
                    return !logger_name.starts_with(pattern_);
            }

            return true;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<LoggerNameFilter>(pattern_, mode_);
        }

    private:
        std::string pattern_;
        Mode mode_;
        std::regex regex_;
    };

/**
 * @brief Filter based on message content
 */
    class ContentFilter : public LogFilter {
    public:
        enum class Mode {
            CONTAINS,         // Message must contain pattern
            NOT_CONTAINS,     // Message must not contain pattern
            REGEX_MATCH       // Message must match regex
        };

        ContentFilter(const std::string& pattern, Mode mode = Mode::CONTAINS)
                : pattern_(pattern), mode_(mode) {
            if (mode_ == Mode::REGEX_MATCH) {
                regex_ = std::regex(pattern);
            }
        }

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            const auto& content = message.get_message();

            switch (mode_) {
                case Mode::CONTAINS:
                    return content.find(pattern_) != std::string::npos;

                case Mode::NOT_CONTAINS:
                    return content.find(pattern_) == std::string::npos;

                case Mode::REGEX_MATCH:
                    return std::regex_search(content, regex_);
            }

            return true;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<ContentFilter>(pattern_, mode_);
        }

    private:
        std::string pattern_;
        Mode mode_;
        std::regex regex_;
    };

/**
 * @brief Rate limiting filter to prevent log spam
 */
    class RateLimitFilter : public LogFilter {
    public:
        /**
         * @param max_messages Maximum messages allowed in the time window
         * @param window_ms Time window in milliseconds
         */
        RateLimitFilter(size_t max_messages, int64_t window_ms)
                : max_messages_(max_messages), window_ms_(window_ms) {}

        [[nodiscard]] bool should_log(const LogMessage& /* message */) const override {
            auto now = std::chrono::steady_clock::now();

            // Clean old timestamps
            cleanup_old_timestamps(now);

            // Check rate limit
            if (timestamps_.size() >= max_messages_) {
                return false;
            }

            // Record timestamp
            timestamps_.push_back(now);
            return true;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<RateLimitFilter>(max_messages_, window_ms_);
        }

    private:
        void cleanup_old_timestamps(std::chrono::steady_clock::time_point now) const {
            auto cutoff = now - std::chrono::milliseconds(window_ms_);

            timestamps_.erase(
                    std::remove_if(timestamps_.begin(), timestamps_.end(),
                                   [cutoff](const auto& ts) { return ts < cutoff; }),
                    timestamps_.end()
            );
        }

        size_t max_messages_;
        int64_t window_ms_;
        mutable std::vector<std::chrono::steady_clock::time_point> timestamps_;
    };

/**
 * @brief Duplicate message filter
 */
    class DuplicateFilter : public LogFilter {
    public:
        explicit DuplicateFilter(size_t cache_size = 100)
                : cache_size_(cache_size) {}

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            // Create hash of message content and level
            size_t hash = std::hash<std::string>{}(message.get_message());
            hash ^= std::hash<int>{}(static_cast<int>(message.get_level()));

            // Check if we've seen this before
            if (seen_hashes_.contains(hash)) {
                return false;
            }

            // Add to cache
            seen_hashes_.insert(hash);
            hash_queue_.push(hash);

            // Maintain cache size
            if (hash_queue_.size() > cache_size_) {
                seen_hashes_.erase(hash_queue_.front());
                hash_queue_.pop();
            }

            return true;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<DuplicateFilter>(cache_size_);
        }

    private:
        size_t cache_size_;
        mutable std::unordered_set<size_t> seen_hashes_;
        mutable std::queue<size_t> hash_queue_;
    };

/**
 * @brief Composite filter that combines multiple filters
 */
    class CompositeFilter : public LogFilter {
    public:
        enum class Mode {
            ALL,    // All filters must pass (AND)
            ANY,    // Any filter must pass (OR)
            NONE    // No filter must pass (NOT)
        };

        explicit CompositeFilter(Mode mode = Mode::ALL) : mode_(mode) {}

        void add_filter(std::unique_ptr<LogFilter> filter) {
            filters_.push_back(std::move(filter));
        }

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            if (filters_.empty()) return true;

            switch (mode_) {
                case Mode::ALL:
                    return std::all_of(filters_.begin(), filters_.end(),
                                       [&](const auto& f) { return f->should_log(message); });

                case Mode::ANY:
                    return std::any_of(filters_.begin(), filters_.end(),
                                       [&](const auto& f) { return f->should_log(message); });

                case Mode::NONE:
                    return std::none_of(filters_.begin(), filters_.end(),
                                        [&](const auto& f) { return f->should_log(message); });
            }

            return true;
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            auto composite = std::make_unique<CompositeFilter>(mode_);
            for (const auto& filter : filters_) {
                composite->add_filter(filter->clone());
            }
            return composite;
        }

    private:
        Mode mode_;
        std::vector<std::unique_ptr<LogFilter>> filters_;
    };

/**
 * @brief Custom filter using a predicate function
 */
    class PredicateFilter : public LogFilter {
    public:
        using predicate_func = std::function<bool(const LogMessage&)>;

        explicit PredicateFilter(predicate_func predicate)
                : predicate_(std::move(predicate)) {}

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            return predicate_(message);
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<PredicateFilter>(predicate_);
        }

    private:
        predicate_func predicate_;
    };

/**
 * @brief Time-based filter
 */
    class TimeFilter : public LogFilter {
    public:
        /**
         * @brief Only log messages during specific hours
         */
        TimeFilter(int start_hour, int end_hour)
                : start_hour_(start_hour), end_hour_(end_hour) {}

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            auto time_t = std::chrono::system_clock::to_time_t(message.get_timestamp());
            std::tm tm{};

#ifdef _WIN32
            localtime_s(&tm, &time_t);
#else
            localtime_r(&time_t, &tm);
#endif

            int hour = tm.tm_hour;

            if (start_hour_ <= end_hour_) {
                return hour >= start_hour_ && hour < end_hour_;
            } else {
                // Wrap around midnight
                return hour >= start_hour_ || hour < end_hour_;
            }
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            return std::make_unique<TimeFilter>(start_hour_, end_hour_);
        }

    private:
        int start_hour_;
        int end_hour_;
    };

/**
 * @brief Filter based on thread ID
 */
    class ThreadFilter : public LogFilter {
    public:
        explicit ThreadFilter(bool include_current_thread = true) {
            if (include_current_thread) {
                allowed_threads_.insert(std::this_thread::get_id());
            }
        }

        void add_thread(std::thread::id thread_id) {
            allowed_threads_.insert(thread_id);
        }

        void remove_thread(std::thread::id thread_id) {
            allowed_threads_.erase(thread_id);
        }

        [[nodiscard]] bool should_log(const LogMessage& message) const override {
            return allowed_threads_.contains(message.get_thread_id());
        }

        [[nodiscard]] std::unique_ptr<LogFilter> clone() const override {
            auto filter = std::make_unique<ThreadFilter>(false);
            filter->allowed_threads_ = allowed_threads_;
            return filter;
        }

    private:
        std::unordered_set<std::thread::id> allowed_threads_;
    };

} // namespace fem::core::logging

#endif //LOGGING_LOGFILTER_H

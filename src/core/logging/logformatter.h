#pragma once

#ifndef LOGGING_LOGFORMATTER_H
#define LOGGING_LOGFORMATTER_H

#include <string>
#include <iomanip>
#include <sstream>
#include <format>

#include "logmessage.h"

namespace fem::core::logging {

/**
 * @brief Abstract base class for log message formatting
 *
 * Formatters control how log messages are converted to strings for output.
 * Different formatters can be used for different sinks (e.g., detailed for files,
 * concise for console, JSON for log aggregation systems).
 *
 * Usage context:
 * - Console output: Human-readable with optional colors
 * - File output: Detailed with full timestamps
 * - JSON output: Structured for log analysis tools
 * - CSV output: For data processing and analysis
 */
    class LogFormatter {
    public:
        virtual ~LogFormatter() = default;

        /**
         * @brief Format a log message into a string
         */
        [[nodiscard]] virtual std::string format(const LogMessage& message) const = 0;

        /**
         * @brief Clone the formatter
         */
        [[nodiscard]] virtual std::unique_ptr<LogFormatter> clone() const = 0;
    };

/**
 * @brief Basic formatter with customizable format
 *
 * Format: [TIMESTAMP] [LEVEL] [LOGGER] MESSAGE
 */
    class BasicLogFormatter : public LogFormatter {
    public:
        struct Options {
            bool include_timestamp = true;
            bool include_level = true;
            bool include_logger_name = true;
            bool include_thread_id = false;
            bool include_location = false;
            bool use_short_level = false;
            std::string timestamp_format = "%Y-%m-%d %H:%M:%S";
        };

        BasicLogFormatter();
        explicit BasicLogFormatter(const Options& options)
                : options_(options) {}

        [[nodiscard]] std::string format(const LogMessage& message) const override {
            std::ostringstream oss;

            if (options_.include_timestamp) {
                oss << '[' << format_timestamp(message.get_timestamp()) << "] ";
            }

            if (options_.include_level) {
                oss << '[';
                if (options_.use_short_level) {
                    oss << to_short_string(message.get_level());
                } else {
                    oss << std::left << std::setw(5) << to_string(message.get_level());
                }
                oss << "] ";
            }

            if (options_.include_logger_name && !message.get_logger_name().empty()) {
                oss << '[' << message.get_logger_name() << "] ";
            }

            if (options_.include_thread_id) {
                oss << "[T:" << message.get_thread_id() << "] ";
            }

            if (options_.include_location) {
                oss << '[' << extract_filename(message.get_file_name())
                    << ':' << message.get_line() << "] ";
            }

            oss << message.get_message();

            // Add exception info if present
            if (message.has_exception()) {
                oss << "\n" << format_exception(message.get_exception());
            }

            return oss.str();
        }

        [[nodiscard]] std::unique_ptr<LogFormatter> clone() const override {
            return std::make_unique<BasicLogFormatter>(options_);
        }

    private:
        Options options_;

        [[nodiscard]] std::string format_timestamp(const LogMessage::time_point& tp) const {
            auto time_t = std::chrono::system_clock::to_time_t(tp);
            std::tm tm{};

            // Thread-safe time conversion
#ifdef _WIN32
            localtime_s(&tm, &time_t);
#else
            localtime_r(&time_t, &tm);
#endif

            std::ostringstream oss;
            oss << std::put_time(&tm, options_.timestamp_format.c_str());

            // Add milliseconds
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    tp.time_since_epoch()) % 1000;
            oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

            return oss.str();
        }

        [[nodiscard]] std::string extract_filename(const char* path) const {
            if (!path) return "";
            std::string_view sv(path);
            auto pos = sv.find_last_of("/\\");
            return std::string(pos != std::string_view::npos ? sv.substr(pos + 1) : sv);
        }

        [[nodiscard]] std::string format_exception(std::exception_ptr eptr) const {
            try {
                if (eptr) {
                    std::rethrow_exception(eptr);
                }
            } catch (const std::exception& e) {
                return std::string("Exception: ") + e.what();
            } catch (...) {
                return "Unknown exception";
            }
            return "";
        }
    };

/**
 * @brief JSON formatter for structured logging
 *
 * Usage context:
 * - Log aggregation systems (ELK stack, Splunk)
 * - Machine-readable logs
 * - Cloud logging services
 */
    class JsonLogFormatter : public LogFormatter {
    public:
        struct Options {
            bool pretty_print = false;
            bool include_context = true;
            std::string timestamp_field = "timestamp";
            std::string level_field = "level";
            std::string message_field = "message";
            std::string logger_field = "logger";
        };

        JsonLogFormatter();
        explicit JsonLogFormatter(const Options& options)
                : options_(options) {}

        [[nodiscard]] std::string format(const LogMessage& message) const override {
            std::ostringstream json;

            json << '{';
            if (options_.pretty_print) json << '\n';

            // Timestamp
            add_field(json, options_.timestamp_field,
                      format_timestamp_iso8601(message.get_timestamp()), true);

            // Level
            add_field(json, options_.level_field,
                      std::string(to_string(message.get_level())));

            // Logger name
            if (!message.get_logger_name().empty()) {
                add_field(json, options_.logger_field, message.get_logger_name());
            }

            // Message
            add_field(json, options_.message_field, escape_json(message.get_message()));

            // Thread ID
            add_field(json, "thread_id", format_thread_id(message.get_thread_id()));

            // Location
            add_field(json, "file", extract_filename(message.get_file_name()));
            add_field(json, "line", std::to_string(message.get_line()));
            add_field(json, "function", message.get_function_name());

            // Context
            if (options_.include_context && !message.get_context().empty()) {
                json << ",";
                if (options_.pretty_print) json << "\n  ";
                json << "\"context\": {";

                bool first_context = true;
                for (const auto& key : message.get_context_keys()) {
                    if (!first_context) json << ",";
                    if (options_.pretty_print) json << "\n    ";
                    json << '"' << escape_json(key) << "\": ";
                    format_context_value(json, message, key);
                    first_context = false;
                }

                if (options_.pretty_print) json << "\n  ";
                json << "}";
            }

            // Exception
            if (message.has_exception()) {
                add_field(json, "exception", format_exception_json(message.get_exception()));
            }

            if (options_.pretty_print) json << '\n';
            json << '}';

            return json.str();
        }

        [[nodiscard]] std::unique_ptr<LogFormatter> clone() const override {
            return std::make_unique<JsonLogFormatter>(options_);
        }

    private:
        Options options_;

        void add_field(std::ostringstream& json, const std::string& name,
                       const std::string& value, bool first = false) const {
            if (!first) json << ",";
            if (options_.pretty_print) json << "\n  ";
            json << '"' << name << "\": \"" << value << '"';
        }

        [[nodiscard]] std::string format_timestamp_iso8601(const LogMessage::time_point& tp) const {
            auto time_t = std::chrono::system_clock::to_time_t(tp);
            std::tm tm{};

#ifdef _WIN32
            gmtime_s(&tm, &time_t);
#else
            gmtime_r(&time_t, &tm);
#endif

            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");

            // Add milliseconds
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    tp.time_since_epoch()) % 1000;
            oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';

            return oss.str();
        }

        [[nodiscard]] std::string escape_json(const std::string& str) const {
            std::string escaped;
            escaped.reserve(str.size() + str.size() / 5);  // ~20% extra space without float conversion

            for (char c : str) {
                switch (c) {
                    case '"':  escaped += "\\\""; break;
                    case '\\': escaped += "\\\\"; break;
                    case '\b': escaped += "\\b"; break;
                    case '\f': escaped += "\\f"; break;
                    case '\n': escaped += "\\n"; break;
                    case '\r': escaped += "\\r"; break;
                    case '\t': escaped += "\\t"; break;
                    default:
                        if (c >= 0x20) {
                            escaped += c;
                        } else {
                            // Unicode escape for control characters
                            char buf[7];
                            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                            escaped += buf;
                        }
                }
            }

            return escaped;
        }

        [[nodiscard]] std::string format_thread_id(std::thread::id id) const {
            std::ostringstream oss;
            oss << id;
            return oss.str();
        }

        [[nodiscard]] std::string extract_filename(const char* path) const {
            if (!path) return "";
            std::string_view sv(path);
            auto pos = sv.find_last_of("/\\");
            return std::string(pos != std::string_view::npos ? sv.substr(pos + 1) : sv);
        }

        [[nodiscard]] std::string format_exception_json(std::exception_ptr eptr) const {
            try {
                if (eptr) {
                    std::rethrow_exception(eptr);
                }
            } catch (const std::exception& e) {
                return escape_json(e.what());
            } catch (...) {
                return "Unknown exception";
            }
            return "";
        }

        void format_context_value(std::ostringstream& json,
                                  const LogMessage& message,
                                  const std::string& key) const {
            // Try common types
            if (auto int_val = message.get_context<int>(key)) {
                json << *int_val;
            } else if (auto double_val = message.get_context<double>(key)) {
                json << *double_val;
            } else if (auto bool_val = message.get_context<bool>(key)) {
                json << (*bool_val ? "true" : "false");
            } else if (auto string_val = message.get_context<std::string>(key)) {
                json << '"' << escape_json(*string_val) << '"';
            } else {
                json << "\"<unknown type>\"";
            }
        }
    };

/**
 * @brief Compact formatter for minimal output
 *
 * Format: LEVEL: MESSAGE
 */
    class CompactLogFormatter : public LogFormatter {
    public:
        [[nodiscard]] std::string format(const LogMessage& message) const override {
            return std::format("{}: {}",
                               to_short_string(message.get_level()),
                               message.get_message());
        }

        [[nodiscard]] std::unique_ptr<LogFormatter> clone() const override {
            return std::make_unique<CompactLogFormatter>();
        }
    };

/**
 * @brief CSV formatter for data analysis
 *
 * Usage context:
 * - Log analysis in spreadsheets
 * - Data processing pipelines
 * - Performance metrics logging
 */
    class CsvLogFormatter : public LogFormatter {
    public:
        [[nodiscard]] std::string format(const LogMessage& message) const override {
            std::ostringstream csv;

            // Timestamp
            csv << format_timestamp_iso8601(message.get_timestamp()) << ',';

            // Level
            csv << to_string(message.get_level()) << ',';

            // Logger
            csv << escape_csv(message.get_logger_name()) << ',';

            // Thread ID
            csv << message.get_thread_id() << ',';

            // File
            csv << escape_csv(extract_filename(message.get_file_name())) << ',';

            // Line
            csv << message.get_line() << ',';

            // Function
            csv << escape_csv(message.get_function_name()) << ',';

            // Message
            csv << escape_csv(message.get_message());

            return csv.str();
        }

        [[nodiscard]] std::unique_ptr<LogFormatter> clone() const override {
            return std::make_unique<CsvLogFormatter>();
        }

        [[nodiscard]] static std::string get_header() {
            return "timestamp,level,logger,thread_id,file,line,function,message";
        }

    private:
        [[nodiscard]] std::string escape_csv(const std::string& str) const {
            if (str.find_first_of(",\"\n\r") == std::string::npos) {
                return str;
            }

            std::string escaped = "\"";
            for (char c : str) {
                if (c == '"') {
                    escaped += "\"\"";
                } else {
                    escaped += c;
                }
            }
            escaped += "\"";
            return escaped;
        }

        [[nodiscard]] std::string format_timestamp_iso8601(const LogMessage::time_point& tp) const {
            auto time_t = std::chrono::system_clock::to_time_t(tp);
            std::tm tm{};

#ifdef _WIN32
            gmtime_s(&tm, &time_t);
#else
            gmtime_r(&time_t, &tm);
#endif

            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            return oss.str();
        }

        [[nodiscard]] std::string extract_filename(const char* path) const {
            if (!path) return "";
            std::string_view sv(path);
            auto pos = sv.find_last_of("/\\");
            return std::string(pos != std::string_view::npos ? sv.substr(pos + 1) : sv);
        }
    };

// ============================================================================
// Default Constructor Implementations
// ============================================================================

inline BasicLogFormatter::BasicLogFormatter() : BasicLogFormatter(Options{}) {}

inline JsonLogFormatter::JsonLogFormatter() : JsonLogFormatter(Options{}) {}

} // namespace fem::core::logging

#endif //LOGGING_LOGFORMATTER_H

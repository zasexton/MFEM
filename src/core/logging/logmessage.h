#pragma once

#ifndef LOGGING_LOGMESSAGE_H
#define LOGGING_LOGMESSAGE_H

#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <source_location>
#include <unordered_map>
#include <any>
#include <memory>

#include "loglevel.h"

namespace fem::core::logging {

/**
 * @brief Core log message structure
 *
 * Represents a single log entry with all associated metadata.
 * This is the fundamental data structure passed through the logging system.
 *
 * Usage context:
 * - Created by Logger when a log method is called
 * - Passed to LogSinks for output
 * - Used by LogFormatters to create formatted strings
 * - Can carry contextual information and custom data
 */
    class LogMessage {
    public:
        using time_point = std::chrono::system_clock::time_point;
        using thread_id = std::thread::id;
        using context_map = std::unordered_map<std::string, std::any>;

        /**
         * @brief Constructor with basic information
         */
        LogMessage(LogLevel level,
                   std::string_view message,
                   std::source_location location = std::source_location::current())
                : level_(level)
                , message_(message)
                , timestamp_(std::chrono::system_clock::now())
                , thread_id_(std::this_thread::get_id())
                , location_(location) {}

        /**
         * @brief Constructor with logger name
         */
        LogMessage(LogLevel level,
                   std::string_view logger_name,
                   std::string_view message,
                   std::source_location location = std::source_location::current())
                : level_(level)
                , logger_name_(logger_name)
                , message_(message)
                , timestamp_(std::chrono::system_clock::now())
                , thread_id_(std::this_thread::get_id())
                , location_(location) {}

        // Getters
        [[nodiscard]] LogLevel get_level() const noexcept { return level_; }
        [[nodiscard]] const std::string& get_logger_name() const noexcept { return logger_name_; }
        [[nodiscard]] const std::string& get_message() const noexcept { return message_; }
        [[nodiscard]] time_point get_timestamp() const noexcept { return timestamp_; }
        [[nodiscard]] thread_id get_thread_id() const noexcept { return thread_id_; }
        [[nodiscard]] const std::source_location& get_location() const noexcept { return location_; }

        // Source location helpers
        [[nodiscard]] const char* get_file_name() const noexcept { return location_.file_name(); }
        [[nodiscard]] std::uint_least32_t get_line() const noexcept { return location_.line(); }
        [[nodiscard]] std::uint_least32_t get_column() const noexcept { return location_.column(); }
        [[nodiscard]] const char* get_function_name() const noexcept { return location_.function_name(); }

        // Context management
        /**
         * @brief Add contextual information to the message
         */
        template<typename T>
        void add_context(const std::string& key, T&& value) {
            context_[key] = std::forward<T>(value);
        }

        /**
         * @brief Get contextual information
         */
        template<typename T>
        [[nodiscard]] std::optional<T> get_context(const std::string& key) const {
            auto it = context_.find(key);
            if (it != context_.end()) {
                try {
                    return std::any_cast<T>(it->second);
                } catch (const std::bad_any_cast&) {
                    return std::nullopt;
                }
            }
            return std::nullopt;
        }

        /**
         * @brief Check if context key exists
         */
        [[nodiscard]] bool has_context(const std::string& key) const {
            return context_.find(key) != context_.end();
        }

        /**
         * @brief Get all context keys
         */
        [[nodiscard]] std::vector<std::string> get_context_keys() const {
            std::vector<std::string> keys;
            keys.reserve(context_.size());
            for (const auto& [key, value] : context_) {
                keys.push_back(key);
            }
            return keys;
        }

        /**
         * @brief Get the entire context map
         */
        [[nodiscard]] const context_map& get_context() const noexcept { return context_; }

        // Additional metadata
        /**
         * @brief Set exception information
         */
        void set_exception(std::exception_ptr ex) { exception_ = ex; }

        /**
         * @brief Get exception information
         */
        [[nodiscard]] std::exception_ptr get_exception() const noexcept { return exception_; }

        /**
         * @brief Check if message has exception
         */
        [[nodiscard]] bool has_exception() const noexcept { return exception_ != nullptr; }

        /**
         * @brief Set sequence number (for ordering)
         */
        void set_sequence_number(std::uint64_t seq) noexcept { sequence_number_ = seq; }

        /**
         * @brief Get sequence number
         */
        [[nodiscard]] std::uint64_t get_sequence_number() const noexcept { return sequence_number_; }

        /**
         * @brief Set process ID
         */
        void set_process_id(int pid) noexcept { process_id_ = pid; }

        /**
         * @brief Get process ID
         */
        [[nodiscard]] int get_process_id() const noexcept { return process_id_; }

        /**
         * @brief Clone the message (deep copy)
         */
        [[nodiscard]] LogMessage clone() const {
            LogMessage msg(level_, logger_name_, message_, location_);
            msg.timestamp_ = timestamp_;
            msg.thread_id_ = thread_id_;
            msg.context_ = context_;
            msg.exception_ = exception_;
            msg.sequence_number_ = sequence_number_;
            msg.process_id_ = process_id_;
            return msg;
        }

    private:
        LogLevel level_;
        std::string logger_name_;
        std::string message_;
        time_point timestamp_;
        thread_id thread_id_;
        std::source_location location_;
        context_map context_;
        std::exception_ptr exception_;
        std::uint64_t sequence_number_{0};
        int process_id_{0};
    };

/**
 * @brief Structured logging builder for creating complex log messages
 *
 * Provides a fluent interface for building log messages with context
 */
    class LogMessageBuilder {
    public:
        explicit LogMessageBuilder(LogLevel level) : level_(level) {}

        LogMessageBuilder& logger(std::string_view name) {
            logger_name_ = name;
            return *this;
        }

        LogMessageBuilder& message(std::string_view msg) {
            message_ = msg;
            return *this;
        }

        template<typename T>
        LogMessageBuilder& with(const std::string& key, T&& value) {
            context_[key] = std::forward<T>(value);
            return *this;
        }

        LogMessageBuilder& exception(std::exception_ptr ex) {
            exception_ = ex;
            return *this;
        }

        LogMessageBuilder& location(std::source_location loc) {
            location_ = loc;
            return *this;
        }

        [[nodiscard]] LogMessage build() const {
            LogMessage msg(level_, logger_name_, message_, location_);

            // Add all context
            for (const auto& [key, value] : context_) {
                msg.add_context(key, value);
            }

            if (exception_) {
                msg.set_exception(exception_);
            }

            return msg;
        }

    private:
        LogLevel level_;
        std::string logger_name_;
        std::string message_;
        std::source_location location_{std::source_location::current()};
        std::unordered_map<std::string, std::any> context_;
        std::exception_ptr exception_;
    };

} // namespace fem::core::logging

#endif //LOGGING_LOGMESSAGE_H

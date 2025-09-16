#pragma once

#ifndef LOGGING_LOGMESSAGE_H
#define LOGGING_LOGMESSAGE_H

#include <string>
#include <chrono>
#include <thread>
#include <source_location>
#include <exception>
#include <optional>
#include <unordered_map>

#include "loglevel.h"

namespace fem::core::logging {

/**
 * @brief Log message containing all information about a log entry
 */
class LogMessage {
public:
    using time_point = std::chrono::system_clock::time_point;

    LogMessage(LogLevel level, const std::string& logger_name, const std::string& message,
               const std::source_location& loc = std::source_location::current())
        : level_(level)
        , logger_name_(logger_name)
        , message_(message)
        , timestamp_(std::chrono::system_clock::now())
        , thread_id_(std::this_thread::get_id())
        , file_name_(loc.file_name())
        , function_name_(loc.function_name())
        , line_(loc.line())
        , sequence_number_(0) {}

    // Copy constructor
    LogMessage(const LogMessage& other) = default;
    LogMessage& operator=(const LogMessage& other) = default;

    // Move constructor
    LogMessage(LogMessage&& other) noexcept = default;
    LogMessage& operator=(LogMessage&& other) noexcept = default;

    // Getters
    LogLevel get_level() const { return level_; }
    const std::string& get_logger_name() const { return logger_name_; }
    const std::string& get_message() const { return message_; }
    const time_point& get_timestamp() const { return timestamp_; }
    std::thread::id get_thread_id() const { return thread_id_; }
    const char* get_file_name() const { return file_name_; }
    const char* get_function_name() const { return function_name_; }
    std::uint_least32_t get_line() const { return line_; }
    uint64_t get_sequence_number() const { return sequence_number_; }

    // Exception support
    void set_exception(std::exception_ptr ex) { exception_ = ex; }
    bool has_exception() const { return exception_ != nullptr; }
    std::exception_ptr get_exception() const { return exception_; }

    // Sequence number
    void set_sequence_number(uint64_t seq) { sequence_number_ = seq; }

    // Context support (simplified)
    template<typename T>
    void set_context(const std::string& key, const T& value) {
        context_[key] = std::to_string(value);
    }

    void set_context(const std::string& key, const std::string& value) {
        context_[key] = value;
    }

    void set_context(const std::string& key, const char* value) {
        context_[key] = std::string(value);
    }

    template<typename T>
    std::optional<T> get_context(const std::string& key) const {
        auto it = context_.find(key);
        if (it != context_.end()) {
            if constexpr (std::is_same_v<T, std::string>) {
                return it->second;
            } else if constexpr (std::is_same_v<T, int>) {
                try {
                    return std::stoi(it->second);
                } catch (...) {
                    return std::nullopt;
                }
            } else if constexpr (std::is_same_v<T, double>) {
                try {
                    return std::stod(it->second);
                } catch (...) {
                    return std::nullopt;
                }
            } else if constexpr (std::is_same_v<T, bool>) {
                return it->second == "true" || it->second == "1";
            }
        }
        return std::nullopt;
    }

    const std::unordered_map<std::string, std::string>& get_context() const { return context_; }

    std::vector<std::string> get_context_keys() const {
        std::vector<std::string> keys;
        keys.reserve(context_.size());
        for (const auto& [key, value] : context_) {
            keys.push_back(key);
        }
        return keys;
    }

    bool empty_context() const { return context_.empty(); }

    // Clone method
    LogMessage clone() const {
        LogMessage copy = *this;
        return copy;
    }

private:
    LogLevel level_;
    std::string logger_name_;
    std::string message_;
    time_point timestamp_;
    std::thread::id thread_id_;
    const char* file_name_;
    const char* function_name_;
    std::uint_least32_t line_;
    uint64_t sequence_number_;
    std::exception_ptr exception_{nullptr};
    std::unordered_map<std::string, std::string> context_;
};

/**
 * @brief Builder for creating structured log messages
 */
class LogMessageBuilder {
public:
    explicit LogMessageBuilder(LogLevel level) : level_(level) {}

    LogMessageBuilder& logger(const std::string& logger_name) {
        logger_name_ = logger_name;
        return *this;
    }

    LogMessageBuilder& message(const std::string& message) {
        message_ = message;
        return *this;
    }

    template<typename T>
    LogMessageBuilder& with(const std::string& key, const T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            context_[key] = value;
        } else {
            context_[key] = std::to_string(value);
        }
        return *this;
    }

    LogMessage build(const std::source_location& loc = std::source_location::current()) {
        LogMessage msg(level_, logger_name_, message_, loc);
        for (const auto& [key, value] : context_) {
            msg.set_context(key, value);
        }
        return msg;
    }

private:
    LogLevel level_;
    std::string logger_name_;
    std::string message_;
    std::unordered_map<std::string, std::string> context_;
};

} // namespace fem::core::logging

#endif // LOGGING_LOGMESSAGE_H
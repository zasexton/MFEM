#pragma once

#ifndef LOGGING_LOGGER_H
#define LOGGING_LOGGER_H

#include <memory>
#include <vector>
#include <format>
#include <source_location>
#include <atomic>

#include "loglevel.h"
#include "logmessage.h"
#include "logsink.h"

#include "../base/object.h"
#include "../base/component.h"
#include "../base/observer.h"
#include "../base/interface.h"

// Forward declarations
class AsyncLoggingComponent;

/**
 * @brief Logger event for Observer pattern integration
 */
class LoggerEvent : public TypedEvent<LoggerEvent> {
public:
    enum Type {
        MESSAGE_LOGGED,
        LEVEL_CHANGED,
        SINK_ADDED,
        SINK_REMOVED,
        LOGGER_ENABLED,
        LOGGER_DISABLED
    };

    LoggerEvent(Type type, const std::string& logger_name)
            : TypedEvent("LoggerEvent")
            , type_(type)
            , logger_name_(logger_name) {}

    Type get_type() const { return type_; }
    const std::string& get_logger_name() const { return logger_name_; }

    void set_message(const LogMessage& msg) { message_ = msg.clone(); }
    const LogMessage& get_message() const { return message_; }

private:
    Type type_;
    std::string logger_name_;
    LogMessage message_{LogLevel::INFO, "", ""};
};

/**
 * @brief Interface for loggable objects
 */
class ILoggable : public TypedInterface<ILoggable> {
public:
    virtual void log_state(class Logger& logger) const = 0;
    virtual std::string get_log_name() const = 0;
};

/**
 * @brief Main logger class integrated with base infrastructure
 *
 * Inherits from:
 * - Object: For ID tracking, lifecycle management
 * - Entity: For component-based features (async logging, filtering)
 * - Subject: For observer pattern notifications
 */
class Logger : public Object,
               public Entity,
               public Subject<LoggerEvent>,
               public std::enable_shared_from_this<Logger> {
public:
    /**
     * @brief Constructor with logger name
     */
    explicit Logger(std::string name)
            : Object("Logger")
            , Entity(name)
            , logger_name_(std::move(name))
            , level_(LogLevelConfig::DEFAULT_LEVEL)
            , enabled_(true) {

        initialize();
    }

    ~Logger() override {
        on_destroy();
    }

    // Basic logging methods

    template<typename... Args>
    void trace(std::format_string<Args...> fmt, Args&&... args,
               std::source_location loc = std::source_location::current()) {
        log(LogLevel::TRACE, fmt, std::forward<Args>(args)..., loc);
    }

    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args,
               std::source_location loc = std::source_location::current()) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)..., loc);
    }

    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args,
              std::source_location loc = std::source_location::current()) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)..., loc);
    }

    template<typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args,
              std::source_location loc = std::source_location::current()) {
        log(LogLevel::WARN, fmt, std::forward<Args>(args)..., loc);
    }

    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args,
               std::source_location loc = std::source_location::current()) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)..., loc);
    }

    template<typename... Args>
    void fatal(std::format_string<Args...> fmt, Args&&... args,
               std::source_location loc = std::source_location::current()) {
        log(LogLevel::FATAL, fmt, std::forward<Args>(args)..., loc);
    }

    // Generic logging method
    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args,
             std::source_location loc = std::source_location::current()) {
        if constexpr (static_cast<int>(LogLevelConfig::COMPILE_TIME_MIN_LEVEL) >
                      static_cast<int>(LogLevel::TRACE)) {
            if (level < LogLevelConfig::COMPILE_TIME_MIN_LEVEL) return;
        }

        if (!should_log(level)) return;

        std::string message = std::format(fmt.get(), std::forward<Args>(args)...);
        LogMessage msg(level, logger_name_, message, loc);
        msg.set_sequence_number(next_sequence_number_++);

        log_message(msg);
    }

    // Log pre-constructed message
    virtual void log_message(const LogMessage& message) {
        if (!should_log(message.get_level())) return;

        // Apply filters from filter components
        if (has_component<FilterComponent>()) {
            auto* filter_comp = get_component<FilterComponent>();
            if (!filter_comp->should_log(message)) return;
        }

        // Check for async component
        if (auto* async = get_component<AsyncLoggingComponent>()) {
            async->queue_message(message);
        } else {
            write_to_sinks(message);
        }

        // Notify observers
        LoggerEvent event(LoggerEvent::MESSAGE_LOGGED, logger_name_);
        event.set_message(message);
        notify_observers(event);

        // Emit to global event bus
        emit_event<LoggerEvent>(event);
    }

    // Conditional logging
    template<typename... Args>
    void log_if(bool condition, LogLevel level,
                std::format_string<Args...> fmt, Args&&... args,
                std::source_location loc = std::source_location::current()) {
        if (condition) {
            log(level, fmt, std::forward<Args>(args)..., loc);
        }
    }

    // Exception logging
    void log_exception(LogLevel level, const std::string& message,
                       std::exception_ptr ex = std::current_exception(),
                       std::source_location loc = std::source_location::current()) {
        if (!should_log(level)) return;

        LogMessage msg(level, logger_name_, message, loc);
        msg.set_exception(ex);
        msg.set_sequence_number(next_sequence_number_++);

        log_message(msg);
    }

    // Structured logging
    [[nodiscard]] LogMessageBuilder log_structured(LogLevel level) {
        return LogMessageBuilder(level).logger(logger_name_);
    }

    // Configuration
    void set_level(LogLevel level) {
        level_.store(level, std::memory_order_relaxed);

        LoggerEvent event(LoggerEvent::LEVEL_CHANGED, logger_name_);
        notify_observers(event);
    }

    [[nodiscard]] LogLevel get_level() const {
        return level_.load(std::memory_order_relaxed);
    }

    void set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);

        LoggerEvent event(enabled ? LoggerEvent::LOGGER_ENABLED
                                  : LoggerEvent::LOGGER_DISABLED, logger_name_);
        notify_observers(event);
    }

    [[nodiscard]] bool is_enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] const std::string& name() const { return logger_name_; }

    // Sink management
    void add_sink(std::shared_ptr<LogSink> sink) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_.push_back(std::move(sink));

        LoggerEvent event(LoggerEvent::SINK_ADDED, logger_name_);
        notify_observers(event);
    }

    void remove_sink(const std::shared_ptr<LogSink>& sink) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_.erase(
                std::remove(sinks_.begin(), sinks_.end(), sink),
                sinks_.end()
        );

        LoggerEvent event(LoggerEvent::SINK_REMOVED, logger_name_);
        notify_observers(event);
    }

    void clear_sinks() {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_.clear();
    }

    [[nodiscard]] size_t sink_count() const {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        return sinks_.size();
    }

    void set_sinks(std::vector<std::shared_ptr<LogSink>> sinks) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_ = std::move(sinks);
    }

    [[nodiscard]] std::vector<std::shared_ptr<LogSink>> get_sinks() const {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        return sinks_;
    }

    virtual void flush() {
        // Flush async component if present
        if (auto* async = get_component<AsyncLoggingComponent>()) {
            async->flush();
        }

        // Flush all sinks
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            sink->flush();
        }
    }

    // Level checking
    [[nodiscard]] bool should_log(LogLevel level) const {
        return is_enabled() && is_enabled(level, get_level());
    }

    [[nodiscard]] bool is_trace_enabled() const { return should_log(LogLevel::TRACE); }
    [[nodiscard]] bool is_debug_enabled() const { return should_log(LogLevel::DEBUG); }
    [[nodiscard]] bool is_info_enabled() const { return should_log(LogLevel::INFO); }
    [[nodiscard]] bool is_warn_enabled() const { return should_log(LogLevel::WARN); }
    [[nodiscard]] bool is_error_enabled() const { return should_log(LogLevel::ERROR); }
    [[nodiscard]] bool is_fatal_enabled() const { return should_log(LogLevel::FATAL); }

    // Override from Object
    [[nodiscard]] std::string to_string() const override {
        return std::format("Logger(name={}, id={}, level={}, sinks={}, enabled={})",
                           logger_name_, id(), get_level(), sink_count(), is_enabled());
    }

    // Override from Entity
    void update(double dt) override {
        Entity::update(dt);  // Updates all components
    }

    void reset() override {
        Entity::reset();  // Resets all components
        next_sequence_number_ = 1;
    }

protected:
    // Override from Object
    void on_create() override {
        // Could register with global registry here
    }

    void on_destroy() override {
        flush();  // Ensure all messages are written
    }

    void write_to_sinks(const LogMessage& message) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            if (sink->should_log(message)) {
                sink->write(message);
            }
        }
    }

private:
    std::string logger_name_;
    std::atomic<LogLevel> level_;
    std::atomic<bool> enabled_;
    mutable std::mutex sinks_mutex_;
    std::vector<std::shared_ptr<LogSink>> sinks_;
    std::atomic<uint64_t> next_sequence_number_{1};

    friend class AsyncLoggingComponent;  // Allow access to write_to_sinks
};

/**
 * @brief Component for adding filters to a logger
 */
class FilterComponent : public TypedComponent<FilterComponent> {
public:
    FilterComponent() : TypedComponent("LogFilter") {}

    void add_filter(std::unique_ptr<class LogFilter> filter);
    bool should_log(const LogMessage& message) const;
    void clear_filters();

private:
    std::vector<std::unique_ptr<class LogFilter>> filters_;
    mutable std::mutex filters_mutex_;
};

// Convenience macros remain the same
#define FEM_LOG(logger, level, ...) \
    (logger)->log(level, __VA_ARGS__, std::source_location::current())

#define FEM_LOG_TRACE(logger, ...) FEM_LOG(logger, fem::core::logging::LogLevel::TRACE, __VA_ARGS__)
#define FEM_LOG_DEBUG(logger, ...) FEM_LOG(logger, fem::core::logging::LogLevel::DEBUG, __VA_ARGS__)
#define FEM_LOG_INFO(logger, ...)  FEM_LOG(logger, fem::core::logging::LogLevel::INFO, __VA_ARGS__)
#define FEM_LOG_WARN(logger, ...)  FEM_LOG(logger, fem::core::logging::LogLevel::WARN, __VA_ARGS__)
#define FEM_LOG_ERROR(logger, ...) FEM_LOG(logger, fem::core::logging::LogLevel::ERROR, __VA_ARGS__)
#define FEM_LOG_FATAL(logger, ...) FEM_LOG(logger, fem::core::logging::LogLevel::FATAL, __VA_ARGS__)

#define FEM_LOG_IF(condition, logger, level, ...) \
    do { \
        if (condition) { \
            FEM_LOG(logger, level, __VA_ARGS__); \
        } \
    } while(0)

#define FEM_LOG_EVERY_N(n, logger, level, ...) \
    do { \
        static std::atomic<int> FEM_LOG_OCCURRENCES{0}; \
        if (++FEM_LOG_OCCURRENCES % (n) == 0) { \
            FEM_LOG(logger, level, __VA_ARGS__); \
        } \
    } while(0)

#define FEM_LOG_FIRST_N(n, logger, level, ...) \
    do { \
        static std::atomic<int> FEM_LOG_OCCURRENCES{0}; \
        if (FEM_LOG_OCCURRENCES++ < (n)) { \
            FEM_LOG(logger, level, __VA_ARGS__); \
        } \
    } while(0)

} // namespace fem::core::logging

#endif //LOGGING_LOGGER_H

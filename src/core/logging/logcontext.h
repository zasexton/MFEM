#pragma once

#ifndef LOGGING_LOGCONTEXT_H
#define LOGGING_LOGCONTEXT_H

#include <unordered_map>
#include <any>
#include <memory>

#include "logmessage.h"

namespace fem::core::logging {

/**
 * @brief Thread-local logging context for automatic metadata attachment
 *
 * Allows setting contextual information that gets automatically added
 * to all log messages within a scope or thread.
 *
 * Usage context:
 * - Add user ID to all logs in a request handler
 * - Include simulation timestep in solver logs
 * - Track element/node ID during assembly
 * - Correlation IDs for distributed tracing
 *=
 * Example:
 * ```cpp
 * LogContext::set("element_id", elem_id);
 * LogContext::set("iteration", iter);
 * logger->info("Processing element"); // Automatically includes context
 * ```
 */
    class LogContext {
    public:
        using context_map = std::unordered_map<std::string, std::any>;

        /**
         * @brief Get the thread-local context instance
         */
        static LogContext& instance() {
            thread_local LogContext context;
            return context;
        }

        /**
         * @brief Set a context value
         */
        template<typename T>
        static void set(const std::string& key, T&& value) {
            instance().set_value(key, std::forward<T>(value));
        }

        /**
         * @brief Get a context value
         */
        template<typename T>
        static std::optional<T> get(const std::string& key) {
            return instance().get_value<T>(key);
        }

        /**
         * @brief Remove a context value
         */
        static void remove(const std::string& key) {
            instance().remove_value(key);
        }

        /**
         * @brief Clear all context
         */
        static void clear() {
            instance().clear_all();
        }

        /**
         * @brief Get all context as a map
         */
        static const context_map& get_all() {
            return instance().get_context();
        }

        /**
         * @brief Apply context to a log message
         */
        static void apply_to_message(LogMessage& message) {
            for (const auto& [key, value] : get_all()) {
                message.add_context(key, value);
            }
        }

        // Instance methods

        template<typename T>
        void set_value(const std::string& key, T&& value) {
            context_[key] = std::forward<T>(value);
        }

        template<typename T>
        std::optional<T> get_value(const std::string& key) const {
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

        void remove_value(const std::string& key) {
            context_.erase(key);
        }

        void clear_all() {
            context_.clear();
        }

        const context_map& get_context() const {
            return context_;
        }

    private:
        LogContext() = default;
        context_map context_;
    };

/**
 * @brief RAII helper for scoped context
 *
 * Automatically adds and removes context within a scope.
 */
    class ScopedLogContext {
    public:
        template<typename T>
        ScopedLogContext(const std::string& key, T&& value)
                : key_(key), had_previous_(false) {

            // Save previous value if it exists
            auto& context = LogContext::instance();
            if (context.get_context().contains(key)) {
                had_previous_ = true;
                previous_value_ = context.get_context().at(key);
            }

            // Set new value
            LogContext::set(key, std::forward<T>(value));
        }

        ~ScopedLogContext() {
            if (had_previous_) {
                // Restore previous value
                LogContext::instance().context_[key_] = previous_value_;
            } else {
                // Remove the key
                LogContext::remove(key_);
            }
        }

        // Non-copyable, non-movable
        ScopedLogContext(const ScopedLogContext&) = delete;
        ScopedLogContext& operator=(const ScopedLogContext&) = delete;
        ScopedLogContext(ScopedLogContext&&) = delete;
        ScopedLogContext& operator=(ScopedLogContext&&) = delete;

    private:
        std::string key_;
        bool had_previous_;
        std::any previous_value_;
    };

/**
 * @brief Multiple context values in a scope
 */
    class ScopedMultiContext {
    public:
        ScopedMultiContext() = default;

        template<typename T>
        ScopedMultiContext& add(const std::string& key, T&& value) {
            contexts_.emplace_back(
                    std::make_unique<ScopedLogContext>(key, std::forward<T>(value))
            );
            return *this;
        }

        ~ScopedMultiContext() = default;

    private:
        std::vector<std::unique_ptr<ScopedLogContext>> contexts_;
    };

/**
 * @brief Contextual logger that automatically includes context
 */
    class ContextualLogger {
    public:
        explicit ContextualLogger(std::shared_ptr<Logger> logger)
                : logger_(std::move(logger)) {}

        template<typename... Args>
        void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
            if (!logger_->should_log(level)) return;

            std::string message = std::format(fmt.get(), std::forward<Args>(args)...);
            LogMessage msg(level, logger_->name(), message);

            // Apply thread-local context
            LogContext::apply_to_message(msg);

            // Apply instance context
            for (const auto& [key, value] : instance_context_) {
                msg.add_context(key, value);
            }

            logger_->log_message(msg);
        }

        // Convenience methods
        template<typename... Args>
        void trace(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::TRACE, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void debug(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void info(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void warn(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void error(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
        }

        // Add instance-specific context
        template<typename T>
        void with_context(const std::string& key, T&& value) {
            instance_context_[key] = std::forward<T>(value);
        }

        void clear_context() {
            instance_context_.clear();
        }

    private:
        std::shared_ptr<Logger> logger_;
        std::unordered_map<std::string, std::any> instance_context_;
    };

/**
 * @brief MDC (Mapped Diagnostic Context) for structured logging
 *
 * Similar to LogContext but with string-only values for compatibility
 * with external logging frameworks.
 */
    class MDC {
    public:
        static MDC& instance() {
            thread_local MDC mdc;
            return mdc;
        }

        static void put(const std::string& key, const std::string& value) {
            instance().values_[key] = value;
        }

        static std::optional<std::string> get(const std::string& key) {
            auto& values = instance().values_;
            auto it = values.find(key);
            return it != values.end() ? std::make_optional(it->second) : std::nullopt;
        }

        static void remove(const std::string& key) {
            instance().values_.erase(key);
        }

        static void clear() {
            instance().values_.clear();
        }

        static std::unordered_map<std::string, std::string> get_context_map() {
            return instance().values_;
        }

    private:
        std::unordered_map<std::string, std::string> values_;
    };

/**
 * @brief NDC (Nested Diagnostic Context) for stack-based context
 */
    class NDC {
    public:
        static NDC& instance() {
            thread_local NDC ndc;
            return ndc;
        }

        static void push(const std::string& message) {
            instance().stack_.push_back(message);
        }

        static void pop() {
            auto& stack = instance().stack_;
            if (!stack.empty()) {
                stack.pop_back();
            }
        }

        static std::string peek() {
            auto& stack = instance().stack_;
            return stack.empty() ? "" : stack.back();
        }

        static std::string get_full_stack() {
            auto& stack = instance().stack_;
            std::string result;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) result += " > ";
                result += stack[i];
            }
            return result;
        }

        static void clear() {
            instance().stack_.clear();
        }

        static size_t depth() {
            return instance().stack_.size();
        }

    private:
        std::vector<std::string> stack_;
    };

/**
 * @brief RAII helper for NDC
 */
    class ScopedNDC {
    public:
        explicit ScopedNDC(const std::string& message) {
            NDC::push(message);
        }

        ~ScopedNDC() {
            NDC::pop();
        }

        // Non-copyable, non-movable
        ScopedNDC(const ScopedNDC&) = delete;
        ScopedNDC& operator=(const ScopedNDC&) = delete;
        ScopedNDC(ScopedNDC&&) = delete;
        ScopedNDC& operator=(ScopedNDC&&) = delete;
    };

// Convenience macros

/**
 * @brief Set context for current scope
 */
#define FEM_LOG_CONTEXT(key, value) \
    fem::core::logging::ScopedLogContext FEM_CONCAT(_ctx_, __LINE__)(key, value)

/**
 * @brief Push NDC context for current scope
 */
#define FEM_LOG_NDC(message) \
    fem::core::logging::ScopedNDC FEM_CONCAT(_ndc_, __LINE__)(message)

/**
 * @brief Set multiple context values
 */
#define FEM_LOG_MULTI_CONTEXT(...) \
    fem::core::logging::ScopedMultiContext FEM_CONCAT(_mctx_, __LINE__); \
    __VA_ARGS__

} // namespace fem::core::logging

#endif //LOGGING_LOGCONTEXT_H

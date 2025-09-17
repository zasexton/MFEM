#pragma once

#ifndef LOGGING_LOGCONTEXT_H
#define LOGGING_LOGCONTEXT_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <sstream>
#include <chrono>
#include <stack>
#include <functional>

namespace fem::core::logging {

// Forward declarations
class LogMessage;
class Logger;

/**
 * @brief Hierarchical context system for logging with automatic scope management
 *
 * LogContext provides a thread-safe, hierarchical context system that automatically
 * propagates contextual information to log messages. It supports:
 * - Thread-local context stacks for scope-based context management
 * - Global context shared across all threads
 * - Automatic context inheritance from parent scopes
 * - Type-safe context value storage and retrieval
 * - RAII-based scope management for automatic cleanup
 *
 * Context hierarchy:
 * 1. Global context (shared across all threads)
 * 2. Thread-local context stack (per-thread scoped contexts)
 * 3. Message-specific context (per-message overrides)
 */
class LogContext {
public:
    using ContextMap = std::unordered_map<std::string, std::string>;
    using ContextValidator = std::function<bool(const std::string& key, const std::string& value)>;

    /**
     * @brief Context scope types for different propagation behaviors
     */
    enum class ScopeType {
        /**
         * @brief Inherits from parent scope and can add/override values
         */
        INHERIT,

        /**
         * @brief Starts fresh without inheriting parent values
         */
        ISOLATED,

        /**
         * @brief Read-only view of parent scope (cannot modify)
         */
        READONLY
    };

    /**
     * @brief Priority levels for context value resolution
     */
    enum class Priority {
        GLOBAL = 0,      // Lowest priority (fallback values)
        THREAD = 1,      // Thread-local context
        SCOPE = 2,       // Current scope context
        MESSAGE = 3      // Highest priority (message-specific overrides)
    };

    /**
     * @brief RAII scope guard for automatic context management
     */
    class ContextScope {
    public:
        ContextScope(const std::string& scope_name,
                    ScopeType type = ScopeType::INHERIT,
                    ContextMap initial_context = {});

        ~ContextScope();

        // Non-copyable but movable
        ContextScope(const ContextScope&) = delete;
        ContextScope& operator=(const ContextScope&) = delete;
        ContextScope(ContextScope&&) noexcept;
        ContextScope& operator=(ContextScope&&) noexcept;

        /**
         * @brief Add or update context value in current scope
         */
        template<typename T>
        ContextScope& with(const std::string& key, const T& value) {
            set_context(key, value);
            return *this;
        }

        /**
         * @brief Set context value with type conversion
         */
        template<typename T>
        void set_context(const std::string& key, const T& value) {
            if (type_ == ScopeType::READONLY) {
                return; // Silently ignore modifications to readonly scopes
            }

            std::string str_value;
            if constexpr (std::is_same_v<T, std::string>) {
                str_value = value;
            } else if constexpr (std::is_arithmetic_v<T>) {
                str_value = std::to_string(value);
            } else if constexpr (std::is_same_v<T, const char*>) {
                str_value = std::string(value);
            } else {
                std::ostringstream oss;
                oss << value;
                str_value = oss.str();
            }

            LogContext::get_instance().set_scope_context(key, str_value);
        }

        /**
         * @brief Get context value from current resolution hierarchy
         */
        template<typename T = std::string>
        std::optional<T> get_context(const std::string& key) const {
            return LogContext::get_instance().get_context<T>(key);
        }

        /**
         * @brief Get scope name
         */
        const std::string& get_scope_name() const { return scope_name_; }

        /**
         * @brief Get scope type
         */
        ScopeType get_scope_type() const { return type_; }

        /**
         * @brief Get scope creation timestamp
         */
        std::chrono::steady_clock::time_point get_creation_time() const { return creation_time_; }

        /**
         * @brief Check if scope is valid (not moved from)
         */
        bool is_valid() const { return valid_; }

    private:
        std::string scope_name_;
        ScopeType type_;
        std::chrono::steady_clock::time_point creation_time_;
        bool valid_ = true;
    };

    /**
     * @brief Get the singleton LogContext instance
     */
    static LogContext& get_instance() {
        static LogContext instance;
        return instance;
    }

    // === Global Context Management ===

    /**
     * @brief Set global context value (affects all threads)
     */
    template<typename T>
    void set_global_context(const std::string& key, const T& value) {
        std::unique_lock lock(global_mutex_);
        global_context_[key] = convert_to_string(value);
    }

    /**
     * @brief Remove global context value
     */
    void remove_global_context(const std::string& key) {
        std::unique_lock lock(global_mutex_);
        global_context_.erase(key);
    }

    /**
     * @brief Clear all global context
     */
    void clear_global_context() {
        std::unique_lock lock(global_mutex_);
        global_context_.clear();
    }

    /**
     * @brief Get copy of global context
     */
    ContextMap get_global_context() const {
        std::shared_lock lock(global_mutex_);
        return global_context_;
    }

    // === Thread-Local Context Management ===

    /**
     * @brief Set context value in current thread scope
     */
    template<typename T>
    void set_thread_context(const std::string& key, const T& value) {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        stack.thread_context[key] = convert_to_string(value);
    }

    /**
     * @brief Remove thread context value
     */
    void remove_thread_context(const std::string& key) {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        stack.thread_context.erase(key);
    }

    /**
     * @brief Clear thread-local context
     */
    void clear_thread_context() {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        stack.thread_context.clear();
    }

    // === Scope Context Management ===

    /**
     * @brief Set context value in current scope
     */
    template<typename T>
    void set_scope_context(const std::string& key, const T& value) {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        if (!stack.scopes.empty()) {
            stack.scopes.top()[key] = convert_to_string(value);
        } else {
            stack.thread_context[key] = convert_to_string(value);
        }
    }

    // === Context Resolution ===

    /**
     * @brief Get context value with hierarchical resolution
     * Order: Message > Scope > Thread > Global
     */
    template<typename T = std::string>
    std::optional<T> get_context(const std::string& key,
                                const ContextMap* message_context = nullptr) const {
        // Check message-specific context first (highest priority)
        if (message_context) {
            if (auto it = message_context->find(key); it != message_context->end()) {
                return convert_from_string<T>(it->second);
            }
        }

        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);

        // Check current scope context
        if (!stack.scopes.empty()) {
            if (auto it = stack.scopes.top().find(key); it != stack.scopes.top().end()) {
                return convert_from_string<T>(it->second);
            }
        }

        // Check thread-local context
        if (auto it = stack.thread_context.find(key); it != stack.thread_context.end()) {
            return convert_from_string<T>(it->second);
        }

        // Check global context (lowest priority)
        {
            std::shared_lock global_lock(global_mutex_);
            if (auto it = global_context_.find(key); it != global_context_.end()) {
                return convert_from_string<T>(it->second);
            }
        }

        return std::nullopt;
    }

    /**
     * @brief Get all effective context (merged from all levels)
     */
    ContextMap get_effective_context(const ContextMap* message_context = nullptr) const {
        ContextMap result;

        // Start with global context (lowest priority)
        {
            std::shared_lock lock(global_mutex_);
            result = global_context_;
        }

        auto& stack = get_thread_stack();
        std::lock_guard stack_lock(stack.mutex);

        // Add thread-local context (overrides global)
        for (const auto& [key, value] : stack.thread_context) {
            result[key] = value;
        }

        // Add scope context (overrides thread)
        if (!stack.scopes.empty()) {
            for (const auto& [key, value] : stack.scopes.top()) {
                result[key] = value;
            }
        }

        // Add message context (highest priority)
        if (message_context) {
            for (const auto& [key, value] : *message_context) {
                result[key] = value;
            }
        }

        return result;
    }

    /**
     * @brief Apply context to a log message
     */
    void apply_context(LogMessage& message) const;

    // === Context Validation ===

    /**
     * @brief Register context validator for a key pattern
     */
    void register_validator(const std::string& key_pattern, ContextValidator validator) {
        std::unique_lock lock(validators_mutex_);
        validators_[key_pattern] = std::move(validator);
    }

    /**
     * @brief Validate context value
     */
    bool validate_context(const std::string& key, const std::string& value) const {
        std::shared_lock lock(validators_mutex_);
        for (const auto& [pattern, validator] : validators_) {
            if (key.find(pattern) != std::string::npos) {
                if (!validator(key, value)) {
                    return false;
                }
            }
        }
        return true;
    }

    // === Utility Methods ===

    /**
     * @brief Get current scope depth
     */
    size_t get_scope_depth() const {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        return stack.scopes.size();
    }

    /**
     * @brief Get context statistics
     */
    struct ContextStats {
        size_t global_keys = 0;
        size_t thread_keys = 0;
        size_t scope_keys = 0;
        size_t scope_depth = 0;
        std::thread::id thread_id;
    };

    ContextStats get_context_stats() const {
        ContextStats stats;
        stats.thread_id = std::this_thread::get_id();

        {
            std::shared_lock lock(global_mutex_);
            stats.global_keys = global_context_.size();
        }

        auto& stack = get_thread_stack();
        std::lock_guard stack_lock(stack.mutex);
        stats.thread_keys = stack.thread_context.size();
        stats.scope_depth = stack.scopes.size();
        stats.scope_keys = stack.scopes.empty() ? 0 : stack.scopes.top().size();

        return stats;
    }

    /**
     * @brief Create scoped context with RAII cleanup
     */
    ContextScope create_scope(const std::string& scope_name,
                             ScopeType type = ScopeType::INHERIT,
                             ContextMap initial_context = {}) {
        return ContextScope(scope_name, type, std::move(initial_context));
    }

private:
    LogContext() = default;
    LogContext(const LogContext&) = delete;
    LogContext& operator=(const LogContext&) = delete;

    // === Internal Scope Management ===

    struct ThreadContextStack {
        mutable std::mutex mutex;
        ContextMap thread_context;
        std::stack<ContextMap> scopes;
    };

    ThreadContextStack& get_thread_stack() const {
        thread_local ThreadContextStack stack;
        return stack;
    }

    void push_scope(const std::string& scope_name, ScopeType type, const ContextMap& initial) {
        (void)scope_name; // Currently unused, reserved for future debugging/logging
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);

        ContextMap scope_context;

        if (type == ScopeType::INHERIT && !stack.scopes.empty()) {
            // Inherit from current scope
            scope_context = stack.scopes.top();
        } else if (type == ScopeType::INHERIT && stack.scopes.empty()) {
            // Inherit from thread context
            scope_context = stack.thread_context;
        }
        // ISOLATED and READONLY start with empty context

        // Add initial context values
        for (const auto& [key, value] : initial) {
            scope_context[key] = value;
        }

        stack.scopes.push(std::move(scope_context));
    }

    void pop_scope() {
        auto& stack = get_thread_stack();
        std::lock_guard lock(stack.mutex);
        if (!stack.scopes.empty()) {
            stack.scopes.pop();
        }
    }

    // === Type Conversion Utilities ===

    template<typename T>
    std::string convert_to_string(const T& value) const {
        if constexpr (std::is_same_v<T, std::string>) {
            return value;
        } else if constexpr (std::is_arithmetic_v<T>) {
            return std::to_string(value);
        } else if constexpr (std::is_same_v<T, const char*>) {
            return std::string(value);
        } else {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }
    }

    template<typename T>
    std::optional<T> convert_from_string(const std::string& str) const {
        try {
            if constexpr (std::is_same_v<T, std::string>) {
                return str;
            } else if constexpr (std::is_same_v<T, int>) {
                return std::stoi(str);
            } else if constexpr (std::is_same_v<T, long>) {
                return std::stol(str);
            } else if constexpr (std::is_same_v<T, long long>) {
                return std::stoll(str);
            } else if constexpr (std::is_same_v<T, float>) {
                return std::stof(str);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(str);
            } else if constexpr (std::is_same_v<T, bool>) {
                return str == "true" || str == "1" || str == "yes" || str == "on";
            } else {
                static_assert(std::is_same_v<T, std::string>, "Unsupported type for context conversion");
                return std::nullopt;
            }
        } catch (...) {
            return std::nullopt;
        }
    }

    // === Member Variables ===

    // Global context shared across all threads
    mutable std::shared_mutex global_mutex_;
    ContextMap global_context_;

    // Context validators
    mutable std::shared_mutex validators_mutex_;
    std::unordered_map<std::string, ContextValidator> validators_;

    // Friend class for scope management
    friend class ContextScope;
};

// === ContextScope Implementation ===

inline LogContext::ContextScope::ContextScope(const std::string& scope_name,
                                               ScopeType type,
                                               ContextMap initial_context)
    : scope_name_(scope_name)
    , type_(type)
    , creation_time_(std::chrono::steady_clock::now())
    , valid_(true) {
    LogContext::get_instance().push_scope(scope_name, type, initial_context);
}

inline LogContext::ContextScope::~ContextScope() {
    if (valid_) {
        LogContext::get_instance().pop_scope();
    }
}

inline LogContext::ContextScope::ContextScope(ContextScope&& other) noexcept
    : scope_name_(std::move(other.scope_name_))
    , type_(other.type_)
    , creation_time_(other.creation_time_)
    , valid_(other.valid_) {
    other.valid_ = false;
}

inline LogContext::ContextScope& LogContext::ContextScope::operator=(ContextScope&& other) noexcept {
    if (this != &other) {
        if (valid_) {
            LogContext::get_instance().pop_scope();
        }
        scope_name_ = std::move(other.scope_name_);
        type_ = other.type_;
        creation_time_ = other.creation_time_;
        valid_ = other.valid_;
        other.valid_ = false;
    }
    return *this;
}

// === Convenience Macros for Scoped Context ===

/**
 * @brief Create a named scoped context that inherits from parent
 */
#define LOG_CONTEXT_SCOPE(name) \
    auto _log_ctx_scope = fem::core::logging::LogContext::get_instance().create_scope(name)

/**
 * @brief Create a scoped context with initial values
 */
#define LOG_CONTEXT_SCOPE_WITH(name, ...) \
    auto _log_ctx_scope = fem::core::logging::LogContext::get_instance().create_scope( \
        name, fem::core::logging::LogContext::ScopeType::INHERIT, __VA_ARGS__)

/**
 * @brief Create an isolated scoped context (doesn't inherit)
 */
#define LOG_CONTEXT_ISOLATED(name) \
    auto _log_ctx_scope = fem::core::logging::LogContext::get_instance().create_scope( \
        name, fem::core::logging::LogContext::ScopeType::ISOLATED)

/**
 * @brief Add context to current scope
 */
#define LOG_CONTEXT_SET(key, value) \
    if (auto* log_ctx_ptr = &fem::core::logging::LogContext::get_instance(); log_ctx_ptr) { \
        log_ctx_ptr->set_scope_context(key, value); \
    }

/**
 * @brief Set global context value
 */
#define LOG_CONTEXT_GLOBAL(key, value) \
    fem::core::logging::LogContext::get_instance().set_global_context(key, value)

/**
 * @brief Set thread-local context value
 */
#define LOG_CONTEXT_THREAD(key, value) \
    fem::core::logging::LogContext::get_instance().set_thread_context(key, value)

} // namespace fem::core::logging

#endif // LOGGING_LOGCONTEXT_H
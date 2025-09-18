#pragma once

#ifndef CORE_ERROR_ERROR_CONTEXT_H
#define CORE_ERROR_ERROR_CONTEXT_H

#include <string>
#include <map>
#include <vector>
#include <any>
#include <memory>
#include <thread>
#include <mutex>
#include <optional>
#include <chrono>
#include <format>
#include "source_location.h"
#include "stack_trace.h"

namespace fem::core::error {

/**
 * @brief Contextual information for error handling
 *
 * Provides thread-local and global context for error diagnosis:
 * - Key-value pairs for debugging information
 * - Hierarchical context scopes
 * - Automatic context capture on error
 * - Thread-safe context management
 */
class ErrorContext {
public:
    using ContextMap = std::map<std::string, std::any>;
    using StringMap = std::map<std::string, std::string>;

    /**
     * @brief Context scope for automatic cleanup
     */
    class Scope {
    public:
        explicit Scope(const std::string& name)
            : name_(name)
            , parent_(current_scope_)
            , start_time_(std::chrono::steady_clock::now()) {
            current_scope_ = this;
            ErrorContext::push_scope(name);
        }

        ~Scope() {
            if (current_scope_ == this) {
                current_scope_ = parent_;
                ErrorContext::pop_scope();
            }
        }

        const std::string& name() const { return name_; }
        
        std::chrono::milliseconds elapsed() const {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start_time_);
        }

        /**
         * @brief Add context value to this scope
         */
        template<typename T>
        Scope& add(const std::string& key, T&& value) {
            ErrorContext::set(key, std::forward<T>(value));
            return *this;
        }

    private:
        std::string name_;
        Scope* parent_;
        std::chrono::steady_clock::time_point start_time_;
        
        static thread_local Scope* current_scope_;
    };

    /**
     * @brief Get thread-local context instance
     */
    static ErrorContext& thread_local_context() {
        static thread_local ErrorContext context;
        return context;
    }

    /**
     * @brief Get global shared context
     */
    static ErrorContext& global_context() {
        static ErrorContext context;
        return context;
    }

    /**
     * @brief Set context value (thread-local)
     */
    template<typename T>
    static void set(const std::string& key, T&& value) {
        thread_local_context().set_value(key, std::forward<T>(value));
    }

    /**
     * @brief Get context value (thread-local)
     */
    template<typename T>
    static std::optional<T> get(const std::string& key) {
        return thread_local_context().get_value<T>(key);
    }

    /**
     * @brief Set global context value
     */
    template<typename T>
    static void set_global(const std::string& key, T&& value) {
        global_context().set_value(key, std::forward<T>(value));
    }

    /**
     * @brief Get global context value
     */
    template<typename T>
    static std::optional<T> get_global(const std::string& key) {
        return global_context().get_value<T>(key);
    }

    /**
     * @brief Clear thread-local context
     */
    static void clear() {
        thread_local_context().clear_all();
    }

    /**
     * @brief Clear global context
     */
    static void clear_global() {
        global_context().clear_all();
    }

    /**
     * @brief Push a new scope
     */
    static void push_scope(const std::string& name) {
        thread_local_context().push_scope_internal(name);
    }

    /**
     * @brief Pop the current scope
     */
    static void pop_scope() {
        thread_local_context().pop_scope_internal();
    }

    /**
     * @brief Get current scope path
     */
    static std::string scope_path() {
        return thread_local_context().get_scope_path();
    }

    /**
     * @brief Capture current context as string map
     */
    static StringMap capture() {
        return thread_local_context().capture_as_strings();
    }

    /**
     * @brief Capture global context as string map
     */
    static StringMap capture_global() {
        return global_context().capture_as_strings();
    }

    /**
     * @brief Format context for display
     */
    static std::string format(const StringMap& context,
                             const std::string& indent = "  ") {
        if (context.empty()) {
            return "";
        }

        std::ostringstream oss;
        for (const auto& [key, value] : context) {
            oss << indent << key << ": " << value << "\n";
        }
        return oss.str();
    }

private:
    template<typename T>
    void set_value(const std::string& key, T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[make_scoped_key(key)] = std::forward<T>(value);
    }

    template<typename T>
    std::optional<T> get_value(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = values_.find(make_scoped_key(key));
        if (it == values_.end()) {
            // Try without scope
            it = values_.find(key);
            if (it == values_.end()) {
                return std::nullopt;
            }
        }

        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return std::nullopt;
        }
    }

    void clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        values_.clear();
        scopes_.clear();
    }

    void push_scope_internal(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        scopes_.push_back(name);
    }

    void pop_scope_internal() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!scopes_.empty()) {
            // Clear values from this scope
            std::string scope_prefix = get_scope_path() + ".";
            auto it = values_.begin();
            while (it != values_.end()) {
                if (it->first.starts_with(scope_prefix)) {
                    it = values_.erase(it);
                } else {
                    ++it;
                }
            }
            
            scopes_.pop_back();
        }
    }

    std::string get_scope_path() const {
        std::ostringstream oss;
        for (size_t i = 0; i < scopes_.size(); ++i) {
            if (i > 0) oss << ".";
            oss << scopes_[i];
        }
        return oss.str();
    }

    std::string make_scoped_key(const std::string& key) const {
        if (scopes_.empty()) {
            return key;
        }
        return get_scope_path() + "." + key;
    }

    StringMap capture_as_strings() const {
        std::lock_guard<std::mutex> lock(mutex_);
        StringMap result;

        for (const auto& [key, value] : values_) {
            result[key] = any_to_string(value);
        }

        // Add metadata
        result["thread_id"] = std::format("{}",
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        
        if (!scopes_.empty()) {
            result["scope"] = get_scope_path();
        }

        return result;
    }

    static std::string any_to_string(const std::any& value) {
        if (value.type() == typeid(std::string)) {
            return std::any_cast<std::string>(value);
        } else if (value.type() == typeid(int)) {
            return std::to_string(std::any_cast<int>(value));
        } else if (value.type() == typeid(long)) {
            return std::to_string(std::any_cast<long>(value));
        } else if (value.type() == typeid(double)) {
            return std::format("{:.6f}", std::any_cast<double>(value));
        } else if (value.type() == typeid(bool)) {
            return std::any_cast<bool>(value) ? "true" : "false";
        } else if (value.type() == typeid(const char*)) {
            return std::any_cast<const char*>(value);
        } else {
            return std::format("<{}>" , value.type().name());
        }
    }

    mutable std::mutex mutex_;
    ContextMap values_;
    std::vector<std::string> scopes_;
};

// Static member definition
thread_local ErrorContext::Scope* ErrorContext::Scope::current_scope_ = nullptr;

/**
 * @brief Extended error context with additional debugging info
 */
class ExtendedContext {
public:
    struct Info {
        SourceLocation location;
        StackTrace stack_trace;
        ErrorContext::StringMap context;
        ErrorContext::StringMap global_context;
        std::chrono::system_clock::time_point timestamp;
        std::thread::id thread_id;
        std::string scope_path;
    };

    /**
     * @brief Capture complete context
     */
    static Info capture(const std::source_location& loc = std::source_location::current()) {
        Info info;
        info.location = SourceLocation(loc);
        info.stack_trace = StackTrace(1);  // Skip this function
        info.context = ErrorContext::capture();
        info.global_context = ErrorContext::capture_global();
        info.timestamp = std::chrono::system_clock::now();
        info.thread_id = std::this_thread::get_id();
        info.scope_path = ErrorContext::scope_path();
        return info;
    }

    /**
     * @brief Format extended context
     */
    static std::string format(const Info& info) {
        std::ostringstream oss;
        
        // Timestamp
        auto time_t = std::chrono::system_clock::to_time_t(info.timestamp);
        oss << "Timestamp: " << std::put_time(std::localtime(&time_t),
                                              "%Y-%m-%d %H:%M:%S") << "\n";
        
        // Location
        oss << "Location: " << info.location.to_string() << "\n";
        
        // Thread
        oss << "Thread: " << std::hash<std::thread::id>{}(info.thread_id) << "\n";
        
        // Scope
        if (!info.scope_path.empty()) {
            oss << "Scope: " << info.scope_path << "\n";
        }
        
        // Context
        if (!info.context.empty()) {
            oss << "Context:\n";
            oss << ErrorContext::format(info.context);
        }
        
        // Global context
        if (!info.global_context.empty()) {
            oss << "Global Context:\n";
            oss << ErrorContext::format(info.global_context);
        }
        
        // Stack trace
        oss << info.stack_trace.to_string(false);
        
        return oss.str();
    }
};

/**
 * @brief RAII context guard
 */
class ContextGuard {
public:
    template<typename... Args>
    explicit ContextGuard(Args&&... args) {
        add_values(std::forward<Args>(args)...);
    }

    ~ContextGuard() {
        for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
            remove_value(*it);
        }
    }

    template<typename T>
    ContextGuard& add(const std::string& key, T&& value) {
        ErrorContext::set(key, std::forward<T>(value));
        keys_.push_back(key);
        return *this;
    }

private:
    template<typename K, typename V, typename... Rest>
    void add_values(K&& key, V&& value, Rest&&... rest) {
        add(std::forward<K>(key), std::forward<V>(value));
        if constexpr (sizeof...(rest) > 0) {
            add_values(std::forward<Rest>(rest)...);
        }
    }

    void add_values() {}  // Base case

    void remove_value(const std::string& key) {
        // Would need a remove method in ErrorContext
        // For now, values persist until scope ends
    }

    std::vector<std::string> keys_;
};

/**
 * @brief Macro helpers for context management
 */

// Create a context scope
#define FEM_CONTEXT_SCOPE(name) \
    fem::core::error::ErrorContext::Scope _context_scope_##__LINE__(name)

// Add context value
#define FEM_CONTEXT_VALUE(key, value) \
    fem::core::error::ErrorContext::set(key, value)

// Create context guard with values
#define FEM_CONTEXT_GUARD(...) \
    fem::core::error::ContextGuard _context_guard_##__LINE__(__VA_ARGS__)

// Capture extended context
#define FEM_CAPTURE_CONTEXT() \
    fem::core::error::ExtendedContext::capture(std::source_location::current())

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_CONTEXT_H
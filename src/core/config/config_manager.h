#pragma once

#ifndef CORE_CONFIG_CONFIG_MANAGER_HPP
#define CORE_CONFIG_CONFIG_MANAGER_HPP

#include <variant>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <filesystem>
#include <shared_mutex>
#include <functional>
#include <any>

// ==============================================================================
// Runtime Configuration Management System
// ==============================================================================
// This is the runtime configuration system for managing application settings,
// loading from files, environment variables, etc. This is separate from the
// compile-time configuration in config.h
// ==============================================================================

namespace fem::core::config {

// Forward declarations
class ConfigSource;
class ConfigSchema;
class ConfigValidator;
enum class MergeStrategy;

// ==============================================================================
// Configuration Manager
// ==============================================================================

class ConfigManager {
public:
    // Configuration value types
    using Value = std::variant<
        std::monostate,                    // null
        bool,                              // boolean
        int64_t,                           // integer
        double,                            // floating point
        std::string,                       // string
        std::vector<Value>,                // array
        std::map<std::string, Value>       // object
    >;
    
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;
    
private:
    Object root_;
    std::vector<std::unique_ptr<ConfigSource>> sources_;
    std::vector<std::function<void(const std::string&)>> change_listeners_;
    mutable std::shared_mutex mutex_;
    
public:
    ConfigManager() = default;
    ~ConfigManager() = default;
    
    // Non-copyable but movable
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    ConfigManager(ConfigManager&&) = default;
    ConfigManager& operator=(ConfigManager&&) = default;
    
    // ========================================================================
    // Value Access
    // ========================================================================
    
    /**
     * @brief Get a configuration value by path
     * @param path Dot-separated path (e.g., "database.host")
     * @throws std::runtime_error if path doesn't exist or type mismatch
     */
    template<typename T>
    T get(const std::string& path) const {
        std::shared_lock lock(mutex_);
        return get_value<T>(resolve_path(path));
    }
    
    /**
     * @brief Get a configuration value with default
     * @param path Dot-separated path
     * @param default_value Value to return if path doesn't exist
     */
    template<typename T>
    T get_or(const std::string& path, T default_value) const {
        std::shared_lock lock(mutex_);
        try {
            return get<T>(path);
        } catch (...) {
            return default_value;
        }
    }
    
    /**
     * @brief Get an optional configuration value
     * @param path Dot-separated path
     * @return std::nullopt if path doesn't exist
     */
    template<typename T>
    std::optional<T> get_optional(const std::string& path) const {
        std::shared_lock lock(mutex_);
        if (has(path)) {
            try {
                return get<T>(path);
            } catch (...) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
    
    /**
     * @brief Check if a configuration path exists
     */
    bool has(const std::string& path) const;
    
    // ========================================================================
    // Value Modification
    // ========================================================================
    
    /**
     * @brief Set a configuration value
     * @param path Dot-separated path
     * @param value Value to set
     */
    template<typename T>
    void set(const std::string& path, T value) {
        std::unique_lock lock(mutex_);
        set_value(resolve_path(path), Value(std::move(value)));
        lock.unlock();
        notify_change(path);
    }
    
    /**
     * @brief Remove a configuration value
     */
    void remove(const std::string& path);
    
    /**
     * @brief Clear all configuration
     */
    void clear();
    
    // ========================================================================
    // Nested Configuration
    // ========================================================================
    
    /**
     * @brief Get a sub-configuration as a new ConfigManager
     */
    ConfigManager get_section(const std::string& path) const;
    
    /**
     * @brief Get all keys at a given path
     */
    std::vector<std::string> get_keys(const std::string& path = "") const;
    
    // ========================================================================
    // Array Operations
    // ========================================================================
    
    /**
     * @brief Get an array of values
     */
    template<typename T>
    std::vector<T> get_array(const std::string& path) const;
    
    /**
     * @brief Get the size of an array
     */
    size_t array_size(const std::string& path) const;
    
    /**
     * @brief Get array element by index
     */
    template<typename T>
    T get_array_element(const std::string& path, size_t index) const;
    
    // ========================================================================
    // Source Management
    // ========================================================================
    
    /**
     * @brief Add a configuration source
     */
    void add_source(std::unique_ptr<ConfigSource> source);
    
    /**
     * @brief Reload all configuration sources
     */
    void reload();
    
    /**
     * @brief Reload a specific source by name
     */
    void reload_source(const std::string& name);
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    /**
     * @brief Validate configuration against a schema
     * @throws validation_error if validation fails
     */
    void validate(const ConfigSchema& schema) const;
    
    /**
     * @brief Check if configuration is valid against schema
     */
    bool is_valid(const ConfigSchema& schema) const;
    
    // ========================================================================
    // Change Notification
    // ========================================================================
    
    /**
     * @brief Register a change listener
     */
    void on_change(std::function<void(const std::string&)> listener);
    
    /**
     * @brief Remove all change listeners
     */
    void clear_listeners();
    
    // ========================================================================
    // Serialization
    // ========================================================================
    
    /**
     * @brief Load configuration from file
     */
    void load_from_file(const std::filesystem::path& path);
    
    /**
     * @brief Save configuration to file
     */
    void save_to_file(const std::filesystem::path& path) const;
    
    /**
     * @brief Convert to JSON string
     */
    std::string to_json(bool pretty = true) const;
    
    /**
     * @brief Convert to YAML string
     */
    std::string to_yaml() const;
    
    /**
     * @brief Load from JSON string
     */
    void from_json(const std::string& json);
    
    /**
     * @brief Load from YAML string
     */
    void from_yaml(const std::string& yaml);
    
    // ========================================================================
    // Merging
    // ========================================================================
    
    enum class MergeStrategy {
        Override,    // New values override existing
        Combine,     // Merge objects, concatenate arrays
        KeepFirst,   // Keep existing values
        KeepLast,    // Take new values
        Deep         // Deep merge of nested objects
    };
    
    /**
     * @brief Merge another configuration into this one
     */
    void merge(const ConfigManager& other, MergeStrategy strategy = MergeStrategy::Override);
    
    // ========================================================================
    // Environment Expansion
    // ========================================================================
    
    /**
     * @brief Expand environment variables in string values
     * Replaces ${VAR} or $VAR with environment variable values
     */
    void expand_environment_variables();
    
    /**
     * @brief Resolve references between configuration values
     * Replaces ${config.path} with values from other paths
     */
    void resolve_references();
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * @brief Get configuration as a formatted string
     */
    std::string to_string() const;
    
    /**
     * @brief Compare with another configuration
     */
    bool equals(const ConfigManager& other) const;
    
    /**
     * @brief Get configuration diff
     */
    std::map<std::string, std::pair<Value, Value>> diff(const ConfigManager& other) const;
    
private:
    // Helper methods
    std::vector<std::string> resolve_path(const std::string& path) const;
    Value* find_value(const std::vector<std::string>& path);
    const Value* find_value(const std::vector<std::string>& path) const;
    void set_value(const std::vector<std::string>& path, Value value);
    void notify_change(const std::string& path);
    
    template<typename T>
    T get_value(const std::vector<std::string>& path) const;
    
    template<typename T>
    T convert_value(const Value& value) const;
};

// ==============================================================================
// Global Configuration Instance
// ==============================================================================

/**
 * @brief Get the global configuration instance
 */
ConfigManager& global_config();

/**
 * @brief Set the global configuration instance
 */
void set_global_config(ConfigManager config);

// ==============================================================================
// Convenience Functions
// ==============================================================================

/**
 * @brief Get a value from global configuration
 */
template<typename T>
T config_get(const std::string& path) {
    return global_config().get<T>(path);
}

/**
 * @brief Get a value from global configuration with default
 */
template<typename T>
T config_get_or(const std::string& path, T default_value) {
    return global_config().get_or(path, std::move(default_value));
}

/**
 * @brief Set a value in global configuration
 */
template<typename T>
void config_set(const std::string& path, T value) {
    global_config().set(path, std::move(value));
}

/**
 * @brief Check if path exists in global configuration
 */
inline bool config_has(const std::string& path) {
    return global_config().has(path);
}

} // namespace fem::core::config

#endif // CORE_CONFIG_CONFIG_MANAGER_HPP
# Core Configuration Infrastructure - AGENT.md

## Purpose
The `config/` layer provides both **compile-time configuration** (required by core library) and **runtime configuration management** (optional for applications). It enables type-safe, flexible configuration with validation, hot-reloading, and multi-source support.

## Architecture Philosophy
- **Dual-purpose**: Compile-time constants AND runtime settings
- **Type safety**: Strongly typed configuration values
- **Hierarchical structure**: Nested configuration with dot-notation
- **Multiple sources**: Files, environment, command line
- **Validation**: Schema-based validation with constraints
- **Zero dependencies**: Compile-time config has no dependencies

## Two-Part System

### Part 1: Compile-Time Configuration (Required)
Essential headers included by core library components:
- **config.h**: Basic types, constants, platform detection
- **debug.h**: Assertion macros and debug utilities

### Part 2: Runtime Configuration (Optional)
Sophisticated configuration management for applications:
- **config_manager.hpp**: Main configuration class
- **config_value.hpp**: Type-safe value wrapper
- **config_schema.hpp**: Validation schemas
- Various parsers and sources

## Files Overview

### Essential Compile-Time Headers
```cpp
config.h           // Core types and constants
debug.h            // Debug macros and assertions
```

### Runtime Configuration Core
```cpp
config_manager.hpp   // Main configuration manager
config_value.hpp     // Configuration value wrapper
config_schema.hpp    // Schema definition and validation
config_source.hpp    // Configuration source interface
```

### Parsers (Runtime)
```cpp
parsers/
├── json_parser.hpp    // JSON configuration parser
├── yaml_parser.hpp    // YAML configuration parser  
├── ini_parser.hpp     // INI file parser
└── toml_parser.hpp    // TOML configuration parser
```

### Sources (Runtime)
```cpp
sources/
├── file_source.hpp    // File-based configuration
├── env_source.hpp     // Environment variables
├── cmd_source.hpp     // Command-line arguments
└── memory_source.hpp  // In-memory configuration
```

### Utilities (Runtime)
```cpp
utils/
├── config_builder.hpp    // Fluent configuration builder
├── config_validator.hpp  // Advanced validation
├── config_watcher.hpp    // File change monitoring
└── config_merger.hpp     // Configuration merging
```

## Detailed Component Specifications

### Part 1: Compile-Time Configuration

#### `config.h`
```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace fem::config {

// ============================================================================
// Platform Detection
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
    #define FEM_PLATFORM_WINDOWS
#elif defined(__APPLE__) && defined(__MACH__)
    #define FEM_PLATFORM_MACOS
#elif defined(__linux__)
    #define FEM_PLATFORM_LINUX
#elif defined(__unix__)
    #define FEM_PLATFORM_UNIX
#endif

// ============================================================================
// Compiler Detection
// ============================================================================
#if defined(_MSC_VER)
    #define FEM_COMPILER_MSVC
    #define FEM_COMPILER_VERSION _MSC_VER
#elif defined(__clang__)
    #define FEM_COMPILER_CLANG
    #define FEM_COMPILER_VERSION (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
    #define FEM_COMPILER_GCC
    #define FEM_COMPILER_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

// ============================================================================
// Build Configuration
// ============================================================================
#ifdef NDEBUG
    #define FEM_RELEASE_BUILD
#else
    #define FEM_DEBUG_BUILD
#endif

// ============================================================================
// Core Types
// ============================================================================
using index_t = std::ptrdiff_t;  // Signed index type for safer arithmetic
using size_t = std::size_t;      // Standard size type
using id_type = std::uint64_t;   // Unique identifier type
using real_t = double;            // Default floating-point type

// ============================================================================
// Memory Configuration
// ============================================================================
inline constexpr std::size_t CACHE_LINE_SIZE = 64;
inline constexpr std::size_t DEFAULT_ALIGNMENT = alignof(std::max_align_t);
inline constexpr std::size_t SIMD_ALIGNMENT = 32;  // For AVX

// ============================================================================
// Numeric Tolerances
// ============================================================================
inline constexpr real_t EPSILON = std::numeric_limits<real_t>::epsilon();
inline constexpr real_t TOLERANCE = 1e-10;
inline constexpr real_t NEAR_ZERO = 1e-15;

// ============================================================================
// Feature Detection
// ============================================================================
#if __cplusplus >= 202002L
    #define FEM_HAS_CPP20
#endif

#ifdef __has_include
    #if __has_include(<execution>)
        #define FEM_HAS_PARALLEL_STL
    #endif
    #if __has_include(<format>)
        #define FEM_HAS_FORMAT
    #endif
#endif

} // namespace fem::config
```

#### `debug.h`
```cpp
#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <source_location>

namespace fem::debug {

// ============================================================================
// Debug Output Stream
// ============================================================================
#ifdef FEM_DEBUG_BUILD
    #define FEM_DEBUG_STREAM std::cerr
#else
    struct NullStream {
        template<typename T>
        NullStream& operator<<(const T&) { return *this; }
    };
    inline NullStream null_stream;
    #define FEM_DEBUG_STREAM fem::debug::null_stream
#endif

// ============================================================================
// Assertion Macros
// ============================================================================
#ifdef FEM_DEBUG_BUILD
    #define FEM_ASSERT(condition) \
        do { \
            if (!(condition)) { \
                std::cerr << "Assertion failed: " #condition \
                         << " at " << std::source_location::current().file_name() \
                         << ":" << std::source_location::current().line() \
                         << std::endl; \
                std::abort(); \
            } \
        } while(0)
    
    #define FEM_ASSERT_MSG(condition, message) \
        do { \
            if (!(condition)) { \
                std::cerr << "Assertion failed: " #condition \
                         << "\nMessage: " << message \
                         << "\nat " << std::source_location::current().file_name() \
                         << ":" << std::source_location::current().line() \
                         << std::endl; \
                std::abort(); \
            } \
        } while(0)
#else
    #define FEM_ASSERT(condition) ((void)0)
    #define FEM_ASSERT_MSG(condition, message) ((void)0)
#endif

// ============================================================================
// Precondition/Postcondition Checks
// ============================================================================
#define FEM_REQUIRES(condition) FEM_ASSERT_MSG(condition, "Precondition failed")
#define FEM_ENSURES(condition) FEM_ASSERT_MSG(condition, "Postcondition failed")

// ============================================================================
// Debug-Only Code Blocks
// ============================================================================
#ifdef FEM_DEBUG_BUILD
    #define FEM_DEBUG_ONLY(code) code
#else
    #define FEM_DEBUG_ONLY(code)
#endif

// ============================================================================
// Range Checking
// ============================================================================
template<typename T>
inline void check_range(T value, T min, T max, const char* name) {
    FEM_ASSERT_MSG(value >= min && value <= max,
                   std::string(name) + " out of range [" + 
                   std::to_string(min) + ", " + std::to_string(max) + "]");
}

} // namespace fem::debug
```

### Part 2: Runtime Configuration System

#### `config_manager.hpp`
```cpp
#pragma once

#include <variant>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <functional>
#include <shared_mutex>
#include <filesystem>

namespace fem::core::config {

class ConfigManager {
public:
    using Value = std::variant<
        std::monostate,  // null
        bool,
        int64_t,
        double,
        std::string,
        std::vector<Value>,
        std::map<std::string, Value>
    >;
    
private:
    std::map<std::string, Value> root_;
    std::vector<std::unique_ptr<ConfigSource>> sources_;
    std::vector<std::function<void(const std::string&)>> change_listeners_;
    mutable std::shared_mutex mutex_;
    
public:
    ConfigManager() = default;
    
    // Value access with type safety
    template<typename T>
    T get(const std::string& path) const {
        std::shared_lock lock(mutex_);
        return get_value<T>(resolve_path(path));
    }
    
    template<typename T>
    T get_or(const std::string& path, T default_value) const {
        try {
            return get<T>(path);
        } catch (...) {
            return default_value;
        }
    }
    
    template<typename T>
    std::optional<T> get_optional(const std::string& path) const {
        if (has(path)) {
            return get<T>(path);
        }
        return std::nullopt;
    }
    
    bool has(const std::string& path) const;
    
    // Value modification
    template<typename T>
    void set(const std::string& path, T value) {
        std::unique_lock lock(mutex_);
        set_value(resolve_path(path), Value(std::move(value)));
        notify_change(path);
    }
    
    void remove(const std::string& path);
    void clear();
    
    // Nested configuration
    ConfigManager get_section(const std::string& path) const;
    std::vector<std::string> get_keys(const std::string& path = "") const;
    
    // Source management
    void add_source(std::unique_ptr<ConfigSource> source);
    void reload();
    
    // Schema validation
    void validate(const ConfigSchema& schema) const;
    
    // Change notification
    void on_change(std::function<void(const std::string&)> listener);
    
    // File I/O
    void load_from_file(const std::filesystem::path& path);
    void save_to_file(const std::filesystem::path& path) const;
    
    // Merging
    enum class MergeStrategy { Override, Combine, Error };
    void merge(const ConfigManager& other, MergeStrategy strategy = MergeStrategy::Override);
    
private:
    std::vector<std::string> resolve_path(const std::string& path) const;
    
    template<typename T>
    T get_value(const std::vector<std::string>& path) const;
    
    void set_value(const std::vector<std::string>& path, Value value);
    void notify_change(const std::string& path);
};

// Global configuration singleton (optional)
ConfigManager& global_config();

} // namespace fem::core::config
```

#### `config_value.hpp`
```cpp
#pragma once

#include "config_manager.hpp"
#include <regex>

namespace fem::core::config {

class ConfigValue {
    ConfigManager::Value value_;
    std::optional<std::string> description_;
    std::optional<std::string> source_;
    bool is_secret_ = false;
    
public:
    template<typename T>
    explicit ConfigValue(T value) : value_(std::move(value)) {}
    
    // Type checking
    bool is_null() const { return std::holds_alternative<std::monostate>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_number() const { 
        return std::holds_alternative<int64_t>(value_) || 
               std::holds_alternative<double>(value_); 
    }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_array() const { return std::holds_alternative<std::vector<ConfigManager::Value>>(value_); }
    bool is_object() const { return std::holds_alternative<std::map<std::string, ConfigManager::Value>>(value_); }
    
    // Safe conversion
    template<typename T>
    T as() const {
        if constexpr (std::is_arithmetic_v<T>) {
            if (auto* i = std::get_if<int64_t>(&value_)) {
                return static_cast<T>(*i);
            }
            if (auto* d = std::get_if<double>(&value_)) {
                return static_cast<T>(*d);
            }
        }
        return std::get<T>(value_);
    }
    
    template<typename T>
    std::optional<T> try_as() const {
        try {
            return as<T>();
        } catch (...) {
            return std::nullopt;
        }
    }
    
    // Metadata
    void set_description(std::string desc) { description_ = std::move(desc); }
    void set_source(std::string src) { source_ = std::move(src); }
    void mark_secret() { is_secret_ = true; }
    
    const std::optional<std::string>& description() const { return description_; }
    const std::optional<std::string>& source() const { return source_; }
    bool is_secret() const { return is_secret_; }
    
    // Basic validation helpers
    void validate_range(double min, double max) const;
    void validate_regex(const std::string& pattern) const;
    void validate_enum(const std::vector<std::string>& values) const;
};

} // namespace fem::core::config
```

#### `config_schema.hpp`
```cpp
#pragma once

#include "config_manager.hpp"
#include "config_value.hpp"
#include <functional>

namespace fem::core::config {

class ConfigSchema {
public:
    struct Field {
        std::string name;
        std::string type;
        bool required = false;
        std::optional<ConfigValue> default_value;
        std::optional<std::string> description;
        std::vector<std::function<void(const ConfigValue&)>> validators;
        
        Field& set_required(bool req = true) {
            required = req;
            return *this;
        }
        
        template<typename T>
        Field& set_default(T value) {
            default_value = ConfigValue(std::move(value));
            return *this;
        }
        
        template<typename T>
        Field& set_type() {
            if constexpr (std::is_same_v<T, bool>) type = "boolean";
            else if constexpr (std::is_integral_v<T>) type = "integer";
            else if constexpr (std::is_floating_point_v<T>) type = "number";
            else if constexpr (std::is_same_v<T, std::string>) type = "string";
            return *this;
        }
        
        Field& add_validator(std::function<void(const ConfigValue&)> validator) {
            validators.push_back(validator);
            return *this;
        }
        
        Field& set_range(double min, double max) {
            return add_validator([min, max](const ConfigValue& v) {
                v.validate_range(min, max);
            });
        }
        
        Field& set_pattern(const std::string& regex) {
            return add_validator([regex](const ConfigValue& v) {
                v.validate_regex(regex);
            });
        }
        
        Field& one_of(const std::vector<std::string>& values) {
            return add_validator([values](const ConfigValue& v) {
                v.validate_enum(values);
            });
        }
    };
    
private:
    std::map<std::string, Field> fields_;
    std::map<std::string, ConfigSchema> nested_schemas_;
    
public:
    // Schema building
    Field& field(const std::string& name) {
        return fields_[name];
    }
    
    ConfigSchema& nested(const std::string& name) {
        return nested_schemas_[name];
    }
    
    // Validation
    void validate(const ConfigManager& config) const;
    std::vector<std::string> validate_with_errors(const ConfigManager& config) const;
    
    // Apply defaults
    void apply_defaults(ConfigManager& config) const;
    
    // Documentation
    std::string generate_documentation() const;
    std::string generate_example() const;
};

} // namespace fem::core::config
```

#### `config_source.hpp`
```cpp
#pragma once

#include "config_manager.hpp"

namespace fem::core::config {

class ConfigSource {
public:
    virtual ~ConfigSource() = default;
    
    // Load configuration from this source
    virtual ConfigManager::Value load() = 0;
    
    // Optional: Watch for changes
    virtual void watch(std::function<void()> callback) {}
    
    // Optional: Get source name for debugging
    virtual std::string name() const { return "unknown"; }
    
    // Optional: Priority for ordering multiple sources
    virtual int priority() const { return 0; }
};

} // namespace fem::core::config
```

## Key Implementation Files

### `sources/file_source.hpp`
```cpp
#pragma once

#include "../config_source.hpp"
#include <filesystem>

namespace fem::core::config {

class FileSource : public ConfigSource {
    std::filesystem::path path_;
    std::string format_;  // "json", "yaml", "ini", "toml"
    
public:
    explicit FileSource(const std::filesystem::path& path);
    
    ConfigManager::Value load() override;
    void watch(std::function<void()> callback) override;
    std::string name() const override { return path_.string(); }
    
private:
    std::string detect_format() const;
};

} // namespace fem::core::config
```

### `sources/env_source.hpp`
```cpp
#pragma once

#include "../config_source.hpp"

namespace fem::core::config {

class EnvironmentSource : public ConfigSource {
public:
    struct Options {
        std::string prefix = "";           // e.g., "APP_"
        std::string delimiter = "_";       // for nested keys
        bool lowercase = true;             // convert to lowercase
        std::map<std::string, std::string> mappings;  // custom mappings
    };
    
private:
    Options options_;
    
public:
    explicit EnvironmentSource(const Options& opts = {});
    
    ConfigManager::Value load() override;
    std::string name() const override { return "environment"; }
    int priority() const override { return 10; }  // Higher priority than files
    
private:
    std::string transform_key(const std::string& env_key) const;
    ConfigManager::Value parse_value(const std::string& value) const;
};

} // namespace fem::core::config
```

### `utils/config_builder.hpp`
```cpp
#pragma once

#include "../config_manager.hpp"

namespace fem::core::config {

class ConfigBuilder {
    ConfigManager config_;
    std::vector<std::string> current_path_;
    
public:
    ConfigBuilder() = default;
    
    // Fluent interface for building configuration
    template<typename T>
    ConfigBuilder& set(const std::string& key, T value) {
        std::string full_path = build_path(key);
        config_.set(full_path, std::move(value));
        return *this;
    }
    
    ConfigBuilder& section(const std::string& name) {
        current_path_.push_back(name);
        return *this;
    }
    
    ConfigBuilder& end_section() {
        if (!current_path_.empty()) {
            current_path_.pop_back();
        }
        return *this;
    }
    
    ConfigBuilder& from_file(const std::filesystem::path& path) {
        config_.load_from_file(path);
        return *this;
    }
    
    ConfigBuilder& from_env(const std::string& key, const std::string& env_var) {
        if (auto value = std::getenv(env_var.c_str())) {
            set(key, std::string(value));
        }
        return *this;
    }
    
    template<typename T>
    ConfigBuilder& set_default(const std::string& key, T value) {
        if (!config_.has(build_path(key))) {
            set(key, std::move(value));
        }
        return *this;
    }
    
    ConfigManager build() {
        return std::move(config_);
    }
    
private:
    std::string build_path(const std::string& key) const;
};

} // namespace fem::core::config
```

### `utils/config_watcher.hpp`
```cpp
#pragma once

#include "../config_manager.hpp"
#include <thread>
#include <atomic>
#include <chrono>

namespace fem::core::config {

class ConfigWatcher {
    ConfigManager& config_;
    std::vector<std::filesystem::path> watched_files_;
    std::thread watcher_thread_;
    std::atomic<bool> running_{false};
    std::chrono::milliseconds check_interval_{1000};
    std::map<std::filesystem::path, std::filesystem::file_time_type> last_modified_;
    std::vector<std::function<void(const std::filesystem::path&)>> callbacks_;
    
public:
    explicit ConfigWatcher(ConfigManager& config);
    ~ConfigWatcher();
    
    void watch_file(const std::filesystem::path& path);
    void set_check_interval(std::chrono::milliseconds interval);
    
    void start();
    void stop();
    
    void on_change(std::function<void(const std::filesystem::path&)> callback);
    
private:
    void watcher_loop();
    void check_files();
};

} // namespace fem::core::config
```

## Usage Examples

### Basic Configuration
```cpp
#include "core/config/config.h"
#include "core/config/debug.h"
#include "core/config/config_manager.hpp"

using namespace fem::config;
using namespace fem::core::config;

// Compile-time configuration
void process_data() {
    std::vector<real_t> data(1000);
    
    FEM_ASSERT(!data.empty());
    FEM_DEBUG_ONLY(
        std::cout << "Processing " << data.size() << " elements\n";
    );
    
    // Use compile-time constants
    if constexpr (sizeof(index_t) == 8) {
        // 64-bit index operations
    }
}

// Runtime configuration
void setup_application() {
    ConfigManager config;
    
    // Load from multiple sources
    config.load_from_file("config/default.json");
    config.add_source(std::make_unique<EnvironmentSource>(
        EnvironmentSource::Options{.prefix = "APP_"}
    ));
    
    // Access values
    auto port = config.get<int>("server.port");
    auto host = config.get_or<std::string>("server.host", "localhost");
    
    // Schema validation
    ConfigSchema schema;
    schema.field("server.port")
        .set_required()
        .set_type<int>()
        .set_range(1, 65535);
    
    schema.field("database.connections")
        .set_default(10)
        .set_type<int>()
        .set_range(1, 100);
    
    schema.validate(config);
}
```

### Hot Reloading
```cpp
void setup_hot_reload() {
    ConfigManager config;
    config.load_from_file("settings.json");
    
    ConfigWatcher watcher(config);
    watcher.watch_file("settings.json");
    
    watcher.on_change([&config](const std::filesystem::path& path) {
        std::cout << "Reloading configuration from " << path << "\n";
        config.reload();
    });
    
    watcher.start();
}
```

### Fluent Builder
```cpp
auto config = ConfigBuilder()
    .from_file("config/base.json")
    .from_env("port", "SERVER_PORT")
    .section("database")
        .set("host", "localhost")
        .set("port", 5432)
        .set_default("pool_size", 10)
    .end_section()
    .section("logging")
        .set("level", "INFO")
        .set("file", "/var/log/app.log")
    .end_section()
    .build();
```

## Design Rationale

### Two-Part System
- **Compile-time**: Zero-overhead configuration for core library
- **Runtime**: Full-featured configuration for applications
- Clear separation prevents unnecessary dependencies

### Type Safety
- Template-based getters ensure compile-time type checking
- std::variant ensures runtime type safety
- Optional types for nullable values

### Thread Safety
- Read-write locks for concurrent access
- Atomic operations for change notifications
- Thread-local storage for certain optimizations

### Extensibility
- Parser interface for new formats
- Source interface for new configuration sources
- Validator interface for custom validation

## Testing Requirements

### Compile-Time Tests
```cpp
TEST(ConfigTest, BasicTypes) {
    static_assert(sizeof(fem::config::index_t) == 8);
    static_assert(sizeof(fem::config::id_type) == 8);
    EXPECT_GE(fem::config::CACHE_LINE_SIZE, 64);
}

TEST(DebugTest, Assertions) {
#ifdef FEM_DEBUG_BUILD
    EXPECT_DEATH(FEM_ASSERT(false), "Assertion failed");
#endif
}
```

### Runtime Tests
```cpp
TEST(ConfigManagerTest, HierarchicalAccess) {
    ConfigManager config;
    config.set("database.primary.host", "db1.example.com");
    config.set("database.primary.port", 5432);
    
    EXPECT_EQ(config.get<std::string>("database.primary.host"), "db1.example.com");
    EXPECT_EQ(config.get<int>("database.primary.port"), 5432);
}

TEST(SchemaTest, Validation) {
    ConfigSchema schema;
    schema.field("port").set_required().set_range(1, 65535);
    
    ConfigManager config;
    EXPECT_THROW(schema.validate(config), std::runtime_error);
    
    config.set("port", 8080);
    EXPECT_NO_THROW(schema.validate(config));
}
```

## Performance Considerations

- Compile-time config has zero runtime overhead
- Runtime config uses lazy loading and caching
- Hot reload uses efficient file watching (inotify on Linux)
- Minimal allocations through move semantics

## Dependencies

### Compile-Time
- C++20 standard library only
- No external dependencies

### Runtime (Optional)
- nlohmann/json for JSON parsing (optional)
- yaml-cpp for YAML parsing (optional)
- toml++ for TOML parsing (optional)

## Anti-patterns to Avoid

- Don't use runtime config for compile-time constants
- Don't store passwords in plain text (use secret management)
- Don't create deeply nested configurations (>3 levels)
- Don't reload configuration in tight loops
- Don't ignore validation errors

## Integration with FEM Core

The configuration system integrates seamlessly with other FEM core components:
- Uses base/Object for configuration objects
- Integrates with logging/ for configuration logging
- Provides configuration for error/ handling
- Supports numeric/ type configurations
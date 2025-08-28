# Core Configuration - AGENT.md

## Purpose
The `config/` layer provides comprehensive configuration management including hierarchical settings, multiple format support, validation, hot-reloading, and environment-based configuration. It enables flexible, type-safe configuration with runtime modification capabilities.

## Architecture Philosophy
- **Type safety**: Strongly typed configuration values
- **Hierarchical structure**: Nested configuration with inheritance
- **Multiple sources**: Files, environment, command line, defaults
- **Live reloading**: Runtime configuration updates
- **Validation**: Schema-based validation and constraints

## Files Overview

### Core Components
```cpp
config.hpp           // Main configuration class
config_value.hpp     // Configuration value wrapper
config_node.hpp      // Configuration tree node
config_schema.hpp    // Schema definition and validation
config_source.hpp    // Configuration source interface
```

### Parsers and Formats
```cpp
json_parser.hpp      // JSON configuration parser
yaml_parser.hpp      // YAML configuration parser  
xml_parser.hpp       // XML configuration parser
toml_parser.hpp      // TOML configuration parser
ini_parser.hpp       // INI file parser
properties_parser.hpp // Java properties format
```

### Sources
```cpp
file_source.hpp      // File-based configuration
env_source.hpp       // Environment variables
cmd_source.hpp       // Command-line arguments
memory_source.hpp    // In-memory configuration
remote_source.hpp    // Remote configuration (HTTP, etc.)
```

### Advanced Features
```cpp
config_builder.hpp   // Fluent configuration builder
config_validator.hpp // Value validation
config_watcher.hpp   // Configuration change monitoring
config_merger.hpp    // Configuration merging strategies
config_resolver.hpp  // Variable resolution and interpolation
```

### Utilities
```cpp
config_macros.hpp    // Convenience macros
config_serialize.hpp // Serialization support
config_diff.hpp      // Configuration comparison
config_migration.hpp // Version migration
secret_config.hpp    // Secure/encrypted values
```

## Detailed Component Specifications

### `config.hpp`
```cpp
class Config {
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
    Config() = default;
    
    // Value access
    template<typename T>
    T get(const std::string& path) const {
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
        set_value(resolve_path(path), Value(std::move(value)));
        notify_change(path);
    }
    
    void remove(const std::string& path);
    void clear();
    
    // Nested configuration
    Config get_section(const std::string& path) const;
    std::vector<std::string> get_keys(const std::string& path = "") const;
    
    // Array operations
    template<typename T>
    std::vector<T> get_array(const std::string& path) const;
    
    size_t array_size(const std::string& path) const;
    
    // Source management
    void add_source(std::unique_ptr<ConfigSource> source);
    void reload();
    void reload_source(const std::string& name);
    
    // Validation
    void validate(const ConfigSchema& schema) const;
    
    // Change notification
    void on_change(std::function<void(const std::string&)> listener);
    
    // Serialization
    void load_from_file(const std::filesystem::path& path);
    void save_to_file(const std::filesystem::path& path) const;
    std::string to_json() const;
    std::string to_yaml() const;
    
    // Merging
    void merge(const Config& other, MergeStrategy strategy = MergeStrategy::Override);
    
    // Environment expansion
    void expand_environment_variables();
    void resolve_references();
    
private:
    std::vector<std::string> resolve_path(const std::string& path) const;
    void set_value(const std::vector<std::string>& path, Value value);
    void notify_change(const std::string& path);
};

// Global configuration
Config& global_config();
```
**Why necessary**: Centralized configuration management, type-safe access, hierarchical settings.
**Usage**: Application settings, feature flags, runtime configuration.

### `config_value.hpp`
```cpp
class ConfigValue {
    Config::Value value_;
    std::optional<std::string> description_;
    std::optional<std::string> source_;
    bool is_secret_ = false;
    
public:
    template<typename T>
    explicit ConfigValue(T value) : value_(std::move(value)) {}
    
    // Type checking
    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;
    
    // Conversion
    template<typename T>
    T as() const {
        return std::get<T>(value_);
    }
    
    template<typename T>
    std::optional<T> try_as() const {
        if (auto* ptr = std::get_if<T>(&value_)) {
            return *ptr;
        }
        return std::nullopt;
    }
    
    // Operators
    ConfigValue& operator[](const std::string& key);
    ConfigValue& operator[](size_t index);
    
    // Metadata
    void set_description(std::string desc) { description_ = std::move(desc); }
    void set_source(std::string src) { source_ = std::move(src); }
    void mark_secret() { is_secret_ = true; }
    
    // Validation
    void validate_type(const std::type_info& expected) const;
    void validate_range(double min, double max) const;
    void validate_regex(const std::string& pattern) const;
    void validate_enum(const std::vector<std::string>& values) const;
};
```
**Why necessary**: Type-safe value storage, metadata support, validation.
**Usage**: Configuration values, settings storage, type conversion.

### `config_schema.hpp`
```cpp
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
        
        Field& add_validator(std::function<void(const ConfigValue&)> validator) {
            validators.push_back(validator);
            return *this;
        }
        
        Field& min(double min_val) {
            return add_validator([min_val](const ConfigValue& v) {
                if (v.as<double>() < min_val) {
                    throw ValidationError("Value below minimum");
                }
            });
        }
        
        Field& max(double max_val);
        Field& regex(const std::string& pattern);
        Field& one_of(const std::vector<std::string>& values);
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
    void validate(const Config& config) const;
    std::vector<std::string> validate_with_errors(const Config& config) const;
    
    // Schema operations
    void merge(const ConfigSchema& other);
    Config apply_defaults(const Config& config) const;
    
    // Schema definition from JSON/YAML
    static ConfigSchema from_json(const std::string& json);
    static ConfigSchema from_yaml(const std::string& yaml);
    
    // Documentation generation
    std::string generate_documentation() const;
    std::string generate_example() const;
};

// Common validators
namespace validators {
    std::function<void(const ConfigValue&)> min_value(double min);
    std::function<void(const ConfigValue&)> max_value(double max);
    std::function<void(const ConfigValue&)> range(double min, double max);
    std::function<void(const ConfigValue&)> regex(const std::string& pattern);
    std::function<void(const ConfigValue&)> email();
    std::function<void(const ConfigValue&)> url();
    std::function<void(const ConfigValue&)> ipv4();
    std::function<void(const ConfigValue&)> ipv6();
    std::function<void(const ConfigValue&)> port();
    std::function<void(const ConfigValue&)> path_exists();
}
```
**Why necessary**: Configuration validation, documentation, type safety.
**Usage**: API configuration, settings validation, schema enforcement.

### `config_builder.hpp`
```cpp
class ConfigBuilder {
    Config config_;
    std::vector<std::string> current_path_;
    
public:
    ConfigBuilder() = default;
    
    // Fluent interface
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
    
    // Array building
    template<typename T>
    ConfigBuilder& add_to_array(const std::string& key, T value) {
        std::string full_path = build_path(key);
        auto array = config_.get_array<Config::Value>(full_path);
        array.push_back(Config::Value(std::move(value)));
        config_.set(full_path, array);
        return *this;
    }
    
    // Environment variables
    ConfigBuilder& from_env(const std::string& key, const std::string& env_var) {
        if (auto value = std::getenv(env_var.c_str())) {
            set(key, std::string(value));
        }
        return *this;
    }
    
    // File loading
    ConfigBuilder& from_file(const std::filesystem::path& path) {
        config_.load_from_file(path);
        return *this;
    }
    
    // Default values
    template<typename T>
    ConfigBuilder& set_default(const std::string& key, T value) {
        if (!config_.has(build_path(key))) {
            set(key, std::move(value));
        }
        return *this;
    }
    
    // Build final config
    Config build() {
        return std::move(config_);
    }
    
private:
    std::string build_path(const std::string& key) const;
};

// Usage example
auto config = ConfigBuilder()
    .from_file("config/default.json")
    .from_env("database.host", "DB_HOST")
    .from_env("database.port", "DB_PORT")
    .section("server")
        .set("host", "localhost")
        .set("port", 8080)
        .set_default("timeout", 30)
    .end_section()
    .build();
```
**Why necessary**: Fluent configuration building, convenient API.
**Usage**: Test configuration, programmatic config creation.

### `env_source.hpp`
```cpp
class EnvironmentSource : public ConfigSource {
    std::string prefix_;
    std::string delimiter_;
    std::map<std::string, std::string> mappings_;
    
public:
    struct Options {
        std::string prefix = "";
        std::string delimiter = "_";
        bool lowercase = true;
        std::map<std::string, std::string> custom_mappings;
    };
    
    explicit EnvironmentSource(const Options& opts = {});
    
    Config load() override {
        Config config;
        
        for (char** env = environ; *env != nullptr; ++env) {
            std::string env_str(*env);
            auto pos = env_str.find('=');
            if (pos != std::string::npos) {
                std::string key = env_str.substr(0, pos);
                std::string value = env_str.substr(pos + 1);
                
                if (should_include(key)) {
                    std::string config_key = transform_key(key);
                    config.set(config_key, parse_value(value));
                }
            }
        }
        
        return config;
    }
    
    void watch(std::function<void()> callback) override {
        // Environment variables don't change at runtime
    }
    
private:
    bool should_include(const std::string& key) const;
    std::string transform_key(const std::string& env_key) const;
    Config::Value parse_value(const std::string& value) const;
};

// Command line source
class CommandLineSource : public ConfigSource {
    int argc_;
    char** argv_;
    std::map<std::string, std::string> short_options_;
    std::map<std::string, std::string> long_options_;
    
public:
    CommandLineSource(int argc, char** argv);
    
    void define_option(const std::string& name, 
                       const std::string& short_opt,
                       const std::string& long_opt);
    
    Config load() override;
};
```
**Why necessary**: Environment-based configuration, 12-factor app support.
**Usage**: Docker containers, cloud deployments, dev/prod separation.

### `config_watcher.hpp`
```cpp
class ConfigWatcher {
    Config& config_;
    std::vector<std::filesystem::path> watched_files_;
    std::thread watcher_thread_;
    std::atomic<bool> running_{true};
    std::chrono::milliseconds check_interval_{1000};
    std::map<std::filesystem::path, std::filesystem::file_time_type> last_modified_;
    
public:
    explicit ConfigWatcher(Config& config);
    ~ConfigWatcher();
    
    void watch_file(const std::filesystem::path& path);
    void watch_directory(const std::filesystem::path& dir, 
                        const std::string& pattern = "*");
    
    void set_check_interval(std::chrono::milliseconds interval) {
        check_interval_ = interval;
    }
    
    void start();
    void stop();
    
    // Callbacks
    void on_change(std::function<void(const std::filesystem::path&)> callback);
    void on_reload(std::function<void(const Config&)> callback);
    
private:
    void watcher_loop();
    void check_files();
    void reload_file(const std::filesystem::path& path);
};

// Auto-reloading config
class AutoReloadConfig : public Config {
    ConfigWatcher watcher_;
    
public:
    AutoReloadConfig() : watcher_(*this) {
        watcher_.on_reload([this](const Config& new_config) {
            merge(new_config);
        });
    }
    
    void enable_auto_reload(const std::filesystem::path& config_file) {
        watcher_.watch_file(config_file);
        watcher_.start();
    }
};
```
**Why necessary**: Hot configuration reload, dynamic reconfiguration.
**Usage**: Server applications, development mode, A/B testing.

## Configuration Patterns

### Hierarchical Configuration
```cpp
Config config;
config.set("database.primary.host", "db1.example.com");
config.set("database.primary.port", 5432);
config.set("database.replica.host", "db2.example.com");
config.set("database.replica.port", 5432);

auto primary_config = config.get_section("database.primary");
auto host = primary_config.get<std::string>("host");
```

### Schema Validation
```cpp
ConfigSchema schema;
schema.field("server.port")
    .set_required()
    .set_default(8080)
    .min(1).max(65535);

schema.field("server.host")
    .set_default("localhost")
    .regex("^[a-zA-Z0-9.-]+$");

schema.field("database.connections")
    .set_required()
    .min(1).max(100);

try {
    schema.validate(config);
} catch (const ValidationError& e) {
    std::cerr << "Configuration error: " << e.what() << std::endl;
}
```

### Environment Override
```cpp
// config.json: {"port": 8080}
// Environment: APP_PORT=9090

Config config;
config.add_source(std::make_unique<FileSource>("config.json"));
config.add_source(std::make_unique<EnvironmentSource>(
    EnvironmentSource::Options{.prefix = "APP_"}
));

int port = config.get<int>("port");  // Returns 9090
```

### Feature Flags
```cpp
class FeatureFlags {
    Config& config_;
    
public:
    bool is_enabled(const std::string& feature) {
        return config_.get_or<bool>("features." + feature, false);
    }
    
    template<typename T>
    T get_param(const std::string& feature, const std::string& param) {
        return config_.get<T>("features." + feature + "." + param);
    }
};
```

## Performance Considerations

- **Lazy loading**: Load configuration on demand
- **Caching**: Cache parsed values
- **Minimal locking**: Use read-write locks
- **Batch updates**: Group configuration changes
- **Efficient formats**: Binary formats for large configs

## Testing Strategy

- **Schema validation**: Test all validators
- **Format parsers**: Test each format thoroughly
- **Error handling**: Invalid configs, missing files
- **Hot reload**: File modification detection
- **Thread safety**: Concurrent access tests

## Usage Guidelines

1. **Use schemas**: Always validate configuration
2. **Layer sources**: Files < Environment < Command line
3. **Provide defaults**: Never assume values exist
4. **Document config**: Include descriptions in schema
5. **Secure secrets**: Use secret management for passwords

## Anti-patterns to Avoid

- Hardcoding configuration values
- Not validating configuration
- Storing secrets in plain text
- Deep nesting (>3 levels)
- Runtime type confusion

## Dependencies
- `base/` - For Object patterns
- `filesystem/` - For file operations
- `error/` - For error handling
- `logging/` - For configuration logging
- Standard library (C++20)

## Future Enhancements
- Distributed configuration (etcd, Consul)
- Encrypted configuration
- Configuration versioning
- A/B testing support
- Remote configuration updates
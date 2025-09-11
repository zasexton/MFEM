# Configuration Module

## Overview

The `config/` folder contains two distinct types of configuration:

### 1. **Compile-Time Configuration** (Required by core library)
- `config.h` - Basic types, constants, platform detection
- `debug.h` - Assertion macros and debug utilities

These files are included by core library components like `object.h` and provide fundamental compile-time configuration.

### 2. **Runtime Configuration System** (Optional application feature)
- `config_manager.hpp` - Runtime configuration management
- `config_value.hpp` - Configuration value wrapper
- `config_source.hpp` - Configuration source interface
- Various parsers and utilities

This is a sophisticated runtime configuration system for applications that need to load settings from files, environment variables, etc.

## Compile-Time Configuration Files

### config.h
Provides:
- Type definitions (`index_t`, `id_type`, `real_t`)
- Platform detection macros
- Compiler detection
- Build configuration flags
- Memory alignment constants
- Numeric tolerances

Used throughout the core library for basic type definitions and compile-time configuration.

### debug.h
Provides:
- Assertion macros (`FEM_ASSERT`, `FEM_NUMERIC_ASSERT_MSG`)
- Debug logging macros
- Precondition/postcondition checks
- Debug-only code blocks

Essential for debugging and error checking in debug builds.

## Runtime Configuration System

### Purpose
The runtime configuration system allows applications to:
- Load settings from multiple sources (files, environment, command line)
- Validate configuration against schemas
- Support hot-reloading of configuration
- Merge configurations from multiple sources
- Type-safe access to configuration values

### Basic Usage

```cpp
#include "core/config/config_manager.hpp"

using namespace fem::core::config;

// Load configuration
ConfigManager config;
config.load_from_file("settings.json");

// Access values
int port = config.get<int>("server.port");
std::string host = config.get_or<std::string>("server.host", "localhost");

// Set values
config.set("database.connections", 10);

// Listen for changes
config.on_change([](const std::string& path) {
    std::cout << "Config changed: " << path << std::endl;
});
```

### Configuration Sources

```cpp
// Load from multiple sources (later sources override earlier)
config.add_source(std::make_unique<FileSource>("default.json"));
config.add_source(std::make_unique<FileSource>("custom.json"));
config.add_source(std::make_unique<EnvironmentSource>("APP_"));
config.add_source(std::make_unique<CommandLineSource>(argc, argv));
```

### Schema Validation

```cpp
ConfigSchema schema;
schema.field("server.port")
    .set_required()
    .set_type<int>()
    .set_range(1, 65535);

schema.field("database.host")
    .set_required()
    .set_type<std::string>();

config.validate(schema);  // Throws if invalid
```

## File Structure

```
config/
├── README.md                # This file

# Compile-time configuration (required)
├── config.h                # Basic types and constants
├── debug.h                  # Debug and assertion macros

# Runtime configuration system (optional)
├── config_manager.hpp       # Main configuration manager
├── config_value.hpp        # Value wrapper
├── config_schema.hpp       # Schema definition
├── config_source.hpp       # Source interface

# Parsers
├── parsers/
│   ├── json_parser.hpp
│   ├── yaml_parser.hpp
│   ├── xml_parser.hpp
│   ├── toml_parser.hpp
│   └── ini_parser.hpp

# Sources
├── sources/
│   ├── file_source.hpp
│   ├── env_source.hpp
│   ├── cmd_source.hpp
│   └── memory_source.hpp

# Utilities
└── utils/
    ├── config_builder.hpp
    ├── config_validator.hpp
    ├── config_watcher.hpp
    └── config_merger.hpp
```

## Dependencies

### Compile-Time Configuration
- No external dependencies
- C++20 standard library only

### Runtime Configuration System
- Optional: JSON parser (nlohmann/json or similar)
- Optional: YAML parser (yaml-cpp or similar)
- Optional: TOML parser (toml++ or similar)

## Testing

```cpp
// Test compile-time configuration
#include "core/config/config.h"
#include "core/config/debug.h"

TEST(ConfigTest, BasicTypes) {
    using namespace fem::config;
    
    EXPECT_EQ(sizeof(index_t), 8);
    EXPECT_EQ(sizeof(id_type), 8);
    EXPECT_GE(CACHE_LINE_SIZE, 64);
}

TEST(DebugTest, Assertions) {
    int x = 5;
    FEM_ASSERT(x > 0);
    FEM_ASSERT_MSG(x == 5, "x should be 5");
    
    #ifdef CORE_DEBUG_BUILD
    EXPECT_DEATH(FEM_ASSERT(false), "Assertion failed");
    #endif
}

// Test runtime configuration
TEST(ConfigManagerTest, BasicUsage) {
    ConfigManager config;
    config.set("test.value", 42);
    EXPECT_EQ(config.get<int>("test.value"), 42);
}
```

## Design Decisions

1. **Separation of Concerns**: Compile-time and runtime configuration are separate
2. **Header-Only**: Compile-time config is header-only for efficiency
3. **Type Safety**: Runtime config uses std::variant for type safety
4. **Thread Safety**: Runtime config uses read-write locks
5. **Extensibility**: New parsers and sources can be easily added

## Future Enhancements

- Distributed configuration (etcd, Consul)
- Encrypted configuration values
- Configuration versioning and migration
- GraphQL-style queries for complex configs
- Web UI for configuration management
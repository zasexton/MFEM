# FEM Core Logging Library

A comprehensive, high-performance logging infrastructure for the FEM solver that provides structured logging, async capabilities, performance tracing, and seamless integration with the base library architecture.

## 🎯 Quick Start

```cpp
#include "logging/logger.h"
#include "logging/loggermanager.h"

// Basic logging
auto logger = get_logger("fem.solver");
logger->info("Starting computation with {} elements", element_count);
logger->error("Matrix assembly failed: {}", error_msg);

// Async logging for performance-critical code
auto async_logger = get_async_logger("fem.performance");
async_logger->trace("Processing element {}", elem_id);  // Non-blocking

// Scoped tracing with automatic timing
void solve_system() {
    FEM_LOG_SCOPE(logger);  // Auto-logs entry/exit with timing

    // Your computation here
    logger->info("Factorizing matrix...");
    // Computation happens

    // Scope automatically logs exit with duration
}
```

## 📋 Table of Contents

- [Core Components](#core-components)
- [Usage Patterns](#usage-patterns)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Performance Considerations](#performance-considerations)
- [Integration Examples](#integration-examples)

## 🏗️ Core Components

### 1. Logger (`logger.h`) - The Heart of the System

**What it is:** The main logging interface that inherits from `Object`, `Entity`, and `Subject` from the base library.

**Why it's important:**
- Provides thread-safe, level-controlled logging with compile-time optimizations
- Supports multiple output destinations (sinks) simultaneously
- Extensible through components (async, filtering, timing)
- Integrates with the observer pattern for event notifications

**Key Features:**
```cpp
// Level-controlled logging
logger->trace("Detailed debugging info");
logger->debug("Development information");
logger->info("General information");
logger->warn("Warning conditions");
logger->error("Error conditions");
logger->fatal("Critical failures");

// Conditional logging
logger->log_if(condition, LogLevel::WARN, "Warning: {}", message);

// Exception logging
logger->log_exception(LogLevel::ERROR, "Operation failed", std::current_exception());

// Structured logging
logger->log_structured(LogLevel::INFO)
    .message("User action completed")
    .with("user_id", user_id)
    .with("action", action_type)
    .with("duration_ms", elapsed)
    .build();
```

### 2. LoggerManager (`loggermanager.h`) - Central Orchestration

**What it is:** Singleton manager that creates, configures, and manages all loggers in the system.

**Why it's important:**
- Provides centralized configuration and logger lifecycle management
- Supports pattern-based configuration for hierarchical loggers
- Factory-based sink creation for different output types
- Global sink management for system-wide logging policies

**Usage:**
```cpp
// Get or create loggers
auto solver_logger = LoggerManager::instance().get_logger("fem.solver");
auto mesh_logger = LoggerManager::instance().get_logger("fem.mesh");

// Configure logging levels
LoggerManager::instance().set_level("fem.*", LogLevel::DEBUG);
LoggerManager::instance().set_level("fem.solver", LogLevel::TRACE);

// Add global sinks
auto console_sink = LoggerManager::instance().create_sink("console", "main_console");
LoggerManager::instance().add_global_sink(console_sink);

// Pattern-based sink assignment
auto perf_sink = LoggerManager::instance().create_sink("file", "perf", {{"filename", "performance.log"}});
LoggerManager::instance().add_pattern_sink("fem.perf.*", perf_sink);

// Environment-based configuration
LoggerManager::instance().configure_from_env("FEM_LOG_LEVELS");
```

### 3. LogLevel (`loglevel.h`) - Severity Control

**What it is:** Enumeration and utilities for controlling log message importance.

**Why it's important:**
- Provides compile-time optimization by eliminating debug logs in release builds
- Supports runtime level filtering to control verbosity
- Includes color coding for terminal output

**Features:**
```cpp
// Compile-time optimization
#ifdef NDEBUG
    // TRACE/DEBUG logs compiled away in release builds
    static constexpr LogLevel COMPILE_TIME_MIN_LEVEL = LogLevel::INFO;
#endif

// Runtime level checking
if (logger->is_debug_enabled()) {
    // Expensive debug computation only when needed
    logger->debug("Complex data: {}", expensive_computation());
}

// Level threshold management
LogLevelThreshold threshold(LogLevel::WARN);
if (threshold.should_log(LogLevel::ERROR)) {
    // Log the error
}
```

### 4. Sinks (`logsink.h`) - Output Destinations

**What it is:** Output destinations that write formatted log messages to various targets.

**Why it's important:**
- Separates logging logic from output formatting and destination
- Supports multiple simultaneous outputs with different formats
- Thread-safe and configurable per-sink filtering

**Available Sinks:**

#### ConsoleSink - Terminal Output
```cpp
auto console = std::make_shared<ConsoleSink>(
    ConsoleSink::OutputMode::SPLIT_BY_LEVEL,  // WARN/ERROR to stderr, others to stdout
    true  // Enable colors
);
console->set_formatter(std::make_unique<BasicLogFormatter>());
logger->add_sink(console);
```

#### FileSink - File Output
```cpp
auto file_sink = std::make_shared<FileSink>("application.log", false, true);
file_sink->set_formatter(std::make_unique<JsonLogFormatter>());
file_sink->set_level(LogLevel::DEBUG);  // File gets more detail than console
logger->add_sink(file_sink);
```

#### MemorySink - In-Memory Buffer
```cpp
auto memory_sink = std::make_shared<MemorySink>(1000);  // Keep last 1000 messages
logger->add_sink(memory_sink);

// Later, analyze captured logs
auto messages = memory_sink->get_messages();
auto formatted_content = memory_sink->get_formatted_content();
```

#### MultiSink - Composite Output
```cpp
auto multi_sink = std::make_shared<MultiSink>();
multi_sink->add_sink(console_sink);
multi_sink->add_sink(file_sink);
multi_sink->add_sink(network_sink);
logger->add_sink(multi_sink);
```

### 5. Formatters (`logformatter.h`) - Message Formatting

**What it is:** Classes that convert log messages into formatted strings for output.

**Why it's important:**
- Provides consistent, configurable message formatting
- Supports different output formats for different use cases
- Handles complex data like timestamps, exceptions, and structured data

**Available Formatters:**

#### BasicLogFormatter - Human-Readable
```cpp
BasicLogFormatter::Options opts;
opts.include_timestamp = true;
opts.include_location = true;
opts.timestamp_format = "%Y-%m-%d %H:%M:%S";

auto formatter = std::make_unique<BasicLogFormatter>(opts);
// Output: [2024-09-15 14:30:25.123] [INFO ] [fem.solver] Starting mesh generation
```

#### JsonLogFormatter - Structured Data
```cpp
JsonLogFormatter::Options opts;
opts.pretty_print = true;
opts.include_context = true;

auto formatter = std::make_unique<JsonLogFormatter>(opts);
// Output: {"timestamp":"2024-09-15T14:30:25.123Z","level":"INFO","logger":"fem.solver","message":"Starting mesh generation"}
```

#### CsvLogFormatter - Data Analysis
```cpp
auto formatter = std::make_unique<CsvLogFormatter>();
// Header: timestamp,level,logger,thread_id,file,line,function,message
// Output: 2024-09-15 14:30:25,INFO,fem.solver,140234,solver.cpp,45,solve,Starting mesh generation
```

### 6. Async Logging (`asynclogger.h`) - High-Performance Logging

**What it is:** Component that adds asynchronous, non-blocking logging capabilities to any logger.

**Why it's important:**
- Eliminates I/O blocking in performance-critical code paths
- Provides configurable batching and queue management
- Includes backpressure handling and drop statistics

**Usage:**
```cpp
// Add async capability to any logger
auto logger = get_logger("fem.performance");
logger->add_component<AsyncLoggingComponent>();

// Configure async behavior
AsyncLoggingComponent::Config config;
config.queue_size = 16384;
config.batch_size = 200;
config.flush_interval_ms = 50;
config.block_when_full = false;  // Drop messages instead of blocking
config.worker_thread_count = 2;

logger->add_component<AsyncLoggingComponent>(config);

// Monitor performance
auto* async_comp = logger->get_component<AsyncLoggingComponent>();
auto stats = async_comp->get_statistics();
std::cout << "Messages queued: " << stats.messages_queued << std::endl;
std::cout << "Drop rate: " << stats.drop_rate() * 100 << "%" << std::endl;
```

### 7. Scoped Logging (`scopedlogger.h`) - Function Tracing & Timing

**What it is:** RAII-based loggers that automatically trace function entry/exit and measure performance.

**Why it's important:**
- Provides automatic performance profiling with zero overhead when disabled
- Handles exception detection and nested scope tracking
- Integrates with statistics collection for performance analysis

**Usage:**
```cpp
void expensive_computation() {
    // Automatic function tracing
    FEM_LOG_SCOPE(logger);

    // Manual scope with custom name
    FEM_LOG_SCOPE_NAMED(logger, "matrix_assembly", LogLevel::INFO);

    // Performance-focused scope (more sensitive timing)
    FEM_LOG_PERF_SCOPE(logger, "linear_solve");

    // Add checkpoints
    scope->checkpoint("Matrix assembled");
    // ... computation ...
    scope->checkpoint("Preconditioner applied");
    // ... more computation ...
    scope->checkpoint("System solved");

    // Mark success/failure
    if (converged) {
        scope->mark_success("Converged in {} iterations", iter_count);
    } else {
        scope->mark_failed("Failed to converge");
    }
}

// Simple timing
void matrix_multiply() {
    FEM_LOG_TIME(logger, "Matrix multiplication");
    // ... operation ...
    // Automatically logs: "⏱ Matrix multiplication took 1.23ms"
}
```

### 8. Filters (`logfilter.h`) - Message Filtering

**What it is:** Components that filter log messages based on various criteria.

**Why it's important:**
- Reduces log volume by filtering out unwanted messages
- Provides rate limiting to prevent log spam
- Supports content-based filtering for security/compliance

**Usage:**
```cpp
// Add filtering component to logger
logger->add_component<FilterComponent>();
auto* filter_comp = logger->get_component<FilterComponent>();

// Rate limiting
filter_comp->add_filter(std::make_unique<RateLimitFilter>(100, 1000)); // 100 msg/sec max

// Content filtering
filter_comp->add_filter(std::make_unique<ContentFilter>("password", ContentFilter::Action::BLOCK));

// Duplicate detection
filter_comp->add_filter(std::make_unique<DuplicateFilter>(std::chrono::seconds(60)));

// Level-based filtering (in addition to logger's level)
filter_comp->add_filter(std::make_unique<LevelFilter>(LogLevel::WARN, LevelFilter::Mode::ABOVE_ONLY));
```

### 9. Context Management (`logcontext.h`) - Thread-Local Context

**What it is:** Thread-local storage for automatically attaching metadata to log messages.

**Why it's important:**
- Provides automatic context injection without manual parameter passing
- Supports request tracing and correlation IDs
- Thread-safe and RAII-managed

**Usage:**
```cpp
void process_request(int request_id, const std::string& user_id) {
    // Set thread-local context
    LogContext::set("request_id", request_id);
    LogContext::set("user_id", user_id);

    // All subsequent logs in this thread automatically include context
    logger->info("Processing user request");
    // Output: [INFO] [request_id=123] [user_id=alice] Processing user request

    {
        // Scoped context (RAII)
        FEM_LOG_CONTEXT("operation", "data_validation");
        logger->debug("Validating input data");
        // Output: [DEBUG] [request_id=123] [user_id=alice] [operation=data_validation] Validating input data
    }
    // operation context automatically removed

    call_other_functions();  // Context propagates to called functions
}
```

### 10. Debug Utilities (`debugstream.h`, `assert.h`)

**What they are:** Development and debugging utilities that integrate with the logging system.

**Why they're important:**
- Provide debug output that compiles away in release builds
- Enhanced assertions with logging integration
- Development-time diagnostics

**Usage:**
```cpp
// Debug streams (compiled away in release)
dbg() << "Matrix state: " << matrix << std::endl;
dbg() << "Vector values: " << vector << std::endl;

// Enhanced assertions
FEM_ASSERT(matrix.is_square(), "Matrix must be square for operation {}", operation_name);
FEM_ASSERT(vector.size() == matrix.rows(), "Vector size {} doesn't match matrix rows {}",
           vector.size(), matrix.rows());

// Runtime checks with logging
FEM_CHECK(file.is_open(), "Failed to open configuration file: {}", filename);
```

## 🚀 Usage Patterns

### Basic Application Setup

```cpp
#include "logging/loggermanager.h"

int main() {
    // Initialize logging
    auto& manager = LoggerManager::instance();

    // Configure from environment
    manager.configure_from_env("FEM_LOG_LEVELS");

    // Set up console output
    auto console = manager.create_sink("console", "main");
    manager.add_global_sink(console);

    // Set up file logging
    auto file_sink = manager.create_sink("file", "app_log", {
        {"filename", "fem_solver.log"},
        {"auto_flush", "true"}
    });
    file_sink->set_formatter(std::make_unique<JsonLogFormatter>());
    manager.add_global_sink(file_sink);

    // Get application logger
    auto logger = get_logger("fem.main");
    logger->info("FEM Solver starting up");

    // Your application code
    run_solver();

    // Cleanup
    manager.flush_all();
    logger->info("FEM Solver shutting down");

    return 0;
}
```

### Performance-Critical Code

```cpp
class SolverEngine {
private:
    object_ptr<Logger> perf_logger_;

public:
    SolverEngine() {
        // Set up async logging for performance code
        perf_logger_ = get_async_logger("fem.solver.performance");

        // Configure for high throughput
        AsyncLoggingComponent::Config config;
        config.queue_size = 32768;
        config.batch_size = 500;
        config.flush_interval_ms = 100;
        config.block_when_full = false;  // Never block

        auto* async_comp = perf_logger_->get_component<AsyncLoggingComponent>();
        async_comp->configure(config);
    }

    void solve_timestep(double dt) {
        FEM_LOG_PERF_SCOPE(perf_logger_, "timestep_solve");

        // High-frequency logging with no I/O blocking
        for (int elem = 0; elem < element_count; ++elem) {
            perf_logger_->trace("Processing element {}", elem);
            process_element(elem);
        }
    }
};
```

### Hierarchical Logger Organization

```cpp
// Set up hierarchical loggers
auto main_logger = get_logger("fem");
auto solver_logger = get_logger("fem.solver");
auto mesh_logger = get_logger("fem.mesh");
auto io_logger = get_logger("fem.io");

// Configure levels hierarchically
LoggerManager::instance().set_level("fem", LogLevel::INFO);       // Base level
LoggerManager::instance().set_level("fem.solver", LogLevel::DEBUG); // More detail for solver
LoggerManager::instance().set_level("fem.mesh", LogLevel::WARN);   // Less noise from mesh

// Pattern-based sink assignment
auto perf_sink = LoggerManager::instance().create_sink("file", "performance", {
    {"filename", "performance.json"}
});
perf_sink->set_formatter(std::make_unique<JsonLogFormatter>());
LoggerManager::instance().add_pattern_sink("fem.solver.*", perf_sink);
```

### Error Handling and Diagnostics

```cpp
class ErrorHandler {
private:
    object_ptr<Logger> error_logger_;
    std::shared_ptr<MemorySink> error_sink_;

public:
    ErrorHandler() {
        error_logger_ = get_logger("fem.error");

        // Capture errors in memory for analysis
        error_sink_ = std::make_shared<MemorySink>(500);
        error_sink_->set_level(LogLevel::WARN);  // Only warnings and above
        error_logger_->add_sink(error_sink_);
    }

    void handle_solver_error(const std::exception& e) {
        // Log with context
        LogContext::set("error_type", "solver_failure");
        error_logger_->log_exception(LogLevel::ERROR, "Solver failed", std::make_exception_ptr(e));

        // Generate diagnostic report
        generate_error_report();
    }

    void generate_error_report() {
        auto error_messages = error_sink_->get_messages();
        std::ofstream report("error_report.json");

        report << "{\n";
        report << "  \"error_count\": " << error_messages.size() << ",\n";
        report << "  \"errors\": [\n";

        for (size_t i = 0; i < error_messages.size(); ++i) {
            auto formatter = JsonLogFormatter();
            report << "    " << formatter.format(error_messages[i]);
            if (i < error_messages.size() - 1) report << ",";
            report << "\n";
        }

        report << "  ]\n";
        report << "}\n";
    }
};
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set logging levels for different components
export FEM_LOG_LEVELS="fem.solver=DEBUG;fem.mesh=INFO;fem.io=WARN"

# Your application automatically picks up the configuration
./fem_solver  # Logging levels applied automatically
```

### Programmatic Configuration

```cpp
// Configuration file loading
void load_logging_config(const std::string& config_file) {
    std::ifstream file(config_file);
    std::string config_string;
    std::getline(file, config_string);

    LoggerManager::instance().configure_from_string(config_string);
}

// Runtime level adjustment
void increase_solver_verbosity() {
    auto solver_logger = LoggerManager::instance().find_logger("fem.solver");
    if (solver_logger) {
        LogLevel current = solver_logger->get_level();
        if (current > LogLevel::TRACE) {
            LogLevel new_level = static_cast<LogLevel>(static_cast<int>(current) - 1);
            solver_logger->set_level(new_level);
        }
    }
}
```

## 🔬 Advanced Features

### Custom Sinks

```cpp
class NetworkSink : public LogSink {
public:
    NetworkSink(const std::string& endpoint) : LogSink("NetworkSink"), endpoint_(endpoint) {}

    void write(const LogMessage& message) override {
        if (!should_log(message)) return;

        std::string formatted = get_formatter()->format(message);
        send_to_network(formatted);
    }

    void flush() override {
        flush_network_buffer();
    }

private:
    std::string endpoint_;
    void send_to_network(const std::string& message) { /* Implementation */ }
    void flush_network_buffer() { /* Implementation */ }
};

// Register and use custom sink
LoggerManager::instance().register_sink_type<NetworkSink>("network");
auto network_sink = LoggerManager::instance().create_sink("network", "central_log", {
    {"endpoint", "https://logs.company.com/api/v1/logs"}
});
```

### Performance Monitoring

```cpp
class PerformanceMonitor {
private:
    object_ptr<Logger> monitor_logger_;

public:
    PerformanceMonitor() {
        monitor_logger_ = get_logger("fem.performance");

        // Add scope timing component
        monitor_logger_->add_component<ScopeTimingComponent>();

        // Subscribe to scope events
        FEM_SUBSCRIBE_SCOPE_EVENTS([this](const ScopeEvent& event) {
            if (event.get_event_type() == ScopeEvent::SCOPE_SLOW_DETECTED) {
                handle_slow_operation(event);
            }
        });
    }

    void handle_slow_operation(const ScopeEvent& event) {
        monitor_logger_->warn("Slow operation detected: {} took {}ms",
                            event.get_scope_name(),
                            event.get_duration_us() / 1000);
    }

    void generate_performance_report() {
        auto stats = ScopeStatistics::instance().get_all_stats();

        std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
            return a.average_duration_us() > b.average_duration_us();
        });

        monitor_logger_->info("=== Performance Report ===");
        for (const auto& stat : stats) {
            monitor_logger_->info("Scope: {} | Calls: {} | Avg: {:.2f}μs | Max: {}μs | Failures: {}",
                                stat.name, stat.call_count, stat.average_duration_us(),
                                stat.max_duration_us, stat.failure_count);
        }
    }
};
```

## 📊 Performance Considerations

### Compile-Time Optimizations

The logging system provides compile-time optimizations that eliminate debug logging overhead in release builds:

```cpp
// In debug builds: All levels compiled in
#ifndef NDEBUG
    static constexpr LogLevel COMPILE_TIME_MIN_LEVEL = LogLevel::TRACE;
#else
    // In release builds: Only INFO and above compiled in
    static constexpr LogLevel COMPILE_TIME_MIN_LEVEL = LogLevel::INFO;
#endif
```

### Runtime Performance Tips

1. **Use Async Logging for Hot Paths**
   ```cpp
   auto perf_logger = get_async_logger("hot_path");
   // No I/O blocking in critical sections
   ```

2. **Check Log Levels Before Expensive Operations**
   ```cpp
   if (logger->is_debug_enabled()) {
       // Only compute expensive debug info when needed
       logger->debug("Expensive computation: {}", expensive_function());
   }
   ```

3. **Use Structured Logging for Analysis**
   ```cpp
   // More efficient than string formatting for large datasets
   logger->log_structured(LogLevel::INFO)
       .with("operation", "solve")
       .with("elements", element_count)
       .with("duration_ms", elapsed);
   ```

4. **Configure Appropriate Buffer Sizes**
   ```cpp
   AsyncLoggingComponent::Config config;
   config.queue_size = 65536;        // Large queue for burst logging
   config.batch_size = 1000;         // Large batches for efficiency
   config.flush_interval_ms = 200;   // Less frequent flushes
   ```

## 🔗 Integration Examples

### Integration with Base Library Components

```cpp
// Observer pattern integration
class LoggingObserver : public Observer<LoggerEvent> {
public:
    void on_event(const LoggerEvent& event) override {
        switch (event.get_type()) {
            case LoggerEvent::MESSAGE_LOGGED:
                // React to log messages
                break;
            case LoggerEvent::LEVEL_CHANGED:
                // React to level changes
                break;
        }
    }
};

// Component integration
class CustomLoggerComponent : public TypedComponent<CustomLoggerComponent> {
public:
    void on_attach(Entity* entity) override {
        logger_ = dynamic_cast<Logger*>(entity);
        // Custom initialization
    }

    void update(double dt) override {
        // Custom update logic
    }

private:
    Logger* logger_{nullptr};
};

// Factory integration for custom sinks
LoggerManager::instance().register_sink_type<DatabaseSink>("database");
auto db_sink = LoggerManager::instance().create_sink("database", "audit_log", {
    {"connection_string", "postgresql://..."},
    {"table_name", "audit_logs"}
});
```

### Error Handling Integration

```cpp
#include "error/error_handler.h"
#include "logging/logger.h"

class IntegratedErrorHandler : public ErrorHandler {
private:
    object_ptr<Logger> error_logger_;

public:
    IntegratedErrorHandler() {
        error_logger_ = get_logger("fem.error");

        // Integrate with error handling system
        set_error_callback([this](const Error& error) {
            LogContext::set("error_code", error.code());
            LogContext::set("error_domain", error.domain());

            error_logger_->log_exception(LogLevel::ERROR,
                                       "System error occurred: {}",
                                       error.message());
        });
    }
};
```

---

## 📚 Summary

The FEM Core Logging Library provides a comprehensive, high-performance logging solution that:

- **Scales from simple to complex requirements** with zero-overhead abstractions
- **Integrates seamlessly with the base library** architecture patterns
- **Provides production-ready features** like async logging, performance tracing, and structured output
- **Supports extensive customization** through components, sinks, formatters, and filters
- **Maintains thread safety** throughout all operations
- **Optimizes for performance** with compile-time and runtime optimizations

Whether you need basic application logging or sophisticated performance monitoring and tracing, this library provides the tools and patterns to implement robust logging solutions efficiently.
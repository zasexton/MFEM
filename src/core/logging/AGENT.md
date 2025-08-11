# AGENT.md - FEM Core Logging Library

## 🎯 Purpose
Comprehensive logging infrastructure for the FEM solver, providing structured logging, performance tracing, debugging utilities, and integration with the base infrastructure. Built on top of base/ patterns for consistency.

## 🏗️ Architecture Philosophy
- **Zero-cost abstractions**: Compile away in release builds where appropriate
- **Structured logging**: Rich metadata and context support
- **Async-first**: Non-blocking logging with async components
- **Composable**: Mix and match sinks, filters, formatters
- **Integration**: Leverages base/ patterns (Component, Observer, Factory)

## 📁 Files Overview

### Core Components
- **loglevel.h**: Severity levels and configuration
- **logmessage.h**: Core message structure with metadata
- **logger.h**: Main logger class (Entity-based with components)
- **loggermanager.h**: Centralized logger management (Singleton)

### Output & Formatting
- **logsink.h**: Output destinations (console, file, memory)
- **logformatter.h**: Message formatting (basic, JSON, CSV)
- **logfilter.h**: Message filtering (level, content, rate limiting)
- **logbuffer.h**: Buffering strategies (FIFO, circular, priority)

### Advanced Features
- **asynclogger.h**: Async logging component
- **scopedlogger.h**: RAII scope tracing and timing
- **logcontext.h**: Thread-local context (MDC/NDC)
- **debugstream.h**: Stream-based debug output
- **assert.h**: Assertion macros with logging

## 🔧 Key Classes & Their Powers

### Logger (Core)
```cpp
class Logger : public Object, public Entity, public Subject<LoggerEvent>
```
**Powers:**
- Component-based (add async, filters, etc.)
- Observer pattern for events
- Multiple sinks support
- Compile-time level optimization
- Source location tracking

**Key Methods:**
- `trace/debug/info/warn/error/fatal()`: Type-safe formatting
- `add_component<T>()`: Add capabilities
- `add_sink()`: Multiple outputs

### LoggerManager (Orchestrator)
```cpp
class LoggerManager : public Singleton<LoggerManager>
```
**Powers:**
- Centralized logger creation/management
- Pattern-based configuration
- Global sink management
- Statistics collection
- Factory-based sink creation

**Key Methods:**
- `get_logger(name)`: Get/create logger
- `set_level(pattern, level)`: Configure multiple loggers
- `create_sink(type, name)`: Factory-based sink creation

### Async Logging
```cpp
class AsyncLoggingComponent : public TypedComponent<AsyncLoggingComponent>
```
**Powers:**
- Lock-free message queuing
- Batch processing
- Worker thread pool
- Backpressure handling
- Drop statistics

**Usage:**
```cpp
logger->add_component<AsyncLoggingComponent>();
```

### Scoped Logger (RAII Tracing)
```cpp
class ScopedLogger : public Object, public IScoped
```
**Powers:**
- Automatic entry/exit logging
- Performance timing
- Exception detection
- Nested scope tracking
- Checkpoint support
- Statistics collection

**Usage:**
```cpp
FEM_LOG_SCOPE(logger);  // Auto function tracing
```

### Log Context (MDC/NDC)
```cpp
class LogContext, MDC, NDC
```
**Powers:**
- Thread-local context
- Automatic metadata attachment
- Scoped context (RAII)
- Structured data support

**Usage:**
```cpp
LogContext::set("user_id", 12345);
FEM_LOG_CONTEXT("element_id", elem_id);
```

### Sinks (Output Destinations)
```cpp
ConsoleSink, FileSink, MemorySink, MultiSink, NullSink
```
**Powers:**
- Configurable formatting
- Level filtering
- Thread-safe writing
- Color support (console)
- Rotation support (file)

### Formatters
```cpp
BasicLogFormatter, JsonLogFormatter, CsvLogFormatter, CompactLogFormatter
```
**Powers:**
- Customizable output format
- Structured data support
- Timestamp formatting
- Exception formatting

### Filters
```cpp
LevelFilter, ContentFilter, RateLimitFilter, DuplicateFilter
```
**Powers:**
- Message filtering
- Rate limiting
- Duplicate detection
- Regex matching
- Composite filters

## 💡 Common Patterns & Recipes

### Basic Logging
```cpp
auto logger = get_logger("fem.solver");
logger->info("Iteration {} converged", iter);
logger->error("Matrix singular at row {}", row);
```

### Async Logging
```cpp
auto logger = get_async_logger("performance");
logger->trace("Processing element {}", elem_id);  // Non-blocking
```

### Scoped Tracing
```cpp
void solve() {
    FEM_LOG_SCOPE(logger);  // Auto logs entry/exit
    
    // Mark checkpoints
    scope->checkpoint("Matrix assembled");
    scope->checkpoint("System solved");
}
```

### Structured Logging
```cpp
logger->log_structured(LogLevel::INFO)
    .message("Convergence achieved")
    .with("iterations", iter)
    .with("residual", res)
    .with("time_ms", elapsed)
    .build();
```

### Context Injection
```cpp
// Set thread-local context
LogContext::set("simulation_id", sim_id);
LogContext::set("timestep", t);

// All subsequent logs include context
logger->info("Processing");  // Includes sim_id, timestep
```

### Custom Sinks
```cpp
auto json_sink = make_shared<FileSink>("app.json");
json_sink->set_formatter(make_unique<JsonLogFormatter>());
logger->add_sink(json_sink);
```

### Performance Monitoring
```cpp
FEM_LOG_TIME(logger, "Matrix factorization");
// ... expensive operation ...
// Automatically logs duration
```

### Debug Streams
```cpp
dbg() << "Matrix:\n" << matrix << "\n";  // Compiles away in release
```

### Assertions with Logging
```cpp
FEM_ASSERT(matrix.is_positive_definite(), 
           "Stiffness matrix lost positive definiteness at step {}", step);
```

## 🚀 Performance Characteristics
- Async queue: Lock-free MPSC queue
- Message formatting: ~100-500ns
- File write: Buffered, batch writes
- Console write: Line buffered
- Context lookup: O(1) thread-local
- Filter check: <10ns for level filter

## 🔗 Integration with Base

| Base Pattern | Logging Usage |
|--------------|---------------|
| Component | AsyncLoggingComponent, FilterComponent |
| Observer | LoggerEvent notifications |
| Factory | Sink creation by type |
| Singleton | LoggerManager global instance |
| Registry | Logger collection management |
| Object | Logger inherits for lifecycle |
| Interface | ILoggable for loggable objects |

## ⚠️ Important Notes
- Compile-time optimization: Set `COMPILE_TIME_MIN_LEVEL` to remove trace/debug in release
- Thread-safety: All operations are thread-safe
- Async default: Use `get_async_logger()` for performance-critical paths
- Context is thread-local: Set per thread
- Scoped loggers handle exceptions: Automatic detection and logging

## 🎮 Quick Decision Guide

| Need | Use |
|------|-----|
| Basic logging | `get_logger(name)` |
| High-performance | `get_async_logger(name)` |
| Function tracing | `FEM_LOG_SCOPE(logger)` |
| Performance timing | `FEM_LOG_TIME(logger, op)` |
| Structured output | `JsonLogFormatter` |
| Debug output | `dbg() << value` |
| Assertions | `FEM_ASSERT(cond, msg)` |
| Thread context | `LogContext::set()` |
| Multiple outputs | `MultiSink` |
| Filter messages | Add `FilterComponent` |

## 📊 Configuration Examples

### Environment Variable
```bash
FEM_LOG_LEVELS="fem.solver=DEBUG;fem.mesh=TRACE;fem.io=WARN"
```

### Programmatic
```cpp
LoggerManager::instance()
    .set_level("fem.solver", LogLevel::DEBUG)
    .add_global_sink(console_sink)
    .add_pattern_sink("fem.perf.*", perf_sink);
```

### Component-Based Enhancement
```cpp
logger->add_component<AsyncLoggingComponent>();
logger->add_component<FilterComponent>()
    ->add_filter(make_unique<RateLimitFilter>(100, 1000));
logger->add_component<ScopeTimingComponent>();
```

## 🧪 Debug Helpers
- `LoggerManager::get_statistics()`: System-wide stats
- `AsyncLoggingComponent::get_statistics()`: Queue metrics
- `ScopeStatistics::get_all_stats()`: Performance data
- `MemorySink::get_messages()`: Captured log history

## 🔮 Future Extensions
- Remote logging sinks
- Log rotation and compression
- Correlation ID propagation
- OpenTelemetry integration
- Performance profiling integration
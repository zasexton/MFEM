# Core Layer Architecture – AGENT.md

## Purpose
The `core/` layer provides **foundational, domain-agnostic** building blocks that form the infrastructure for any complex C++ application. These components are completely independent of FEM/numerical concepts and focus on software engineering patterns, resource management, and system-level abstractions.

---

## Folder Layout

```
core/
  base/            # Absolute essentials (no external deps)
  io/              # Input/output abstractions and streams
  logging/         # Structured logging framework
  concurrency/     # Threading, async, and parallelism
  memory/          # Memory management and containers
  reflection/      # Runtime type info & introspection
  plugins/         # Dynamic loading and modularity
  config/          # Configuration management
  metrics/         # Performance monitoring
  tracing/         # Profiling and debugging support
  filesystem/      # File and resource management
  workflow/        # Task orchestration and state management
  error/           # Error handling and reporting
  serialization/   # Data persistence and transport
  events/          # Event system and messaging
  utilities/       # General-purpose utilities
```

**Key Changes:**
- Renamed `fs/` to `filesystem/` for clarity
- Renamed `result/` to `error/` and expanded scope
- Added `io/` for stream abstractions
- Added `serialization/` for data persistence
- Added `events/` as dedicated event infrastructure
- Added `utilities/` for misc helpers
- Removed FEM-specific references throughout

---

## `base/` – Minimal, universal primitives
**What it adds:**
- `Object` – Base class with unique ID and lifecycle management
- `NonCopyable`, `NonMovable` – Copy/move control policies (CRTP)
- `Interface` – Pure virtual interface base
- `Factory<T>` – Generic object factory pattern
- `Registry<T, Key>` – Type-safe object registry
- `Singleton<T>` – Thread-safe singleton pattern
- `Visitor` – Visitor pattern for double dispatch
- `Observer` – Observer/subject pattern
- `Component` / `Entity` – Entity-component system
- `StrongId<Tag>` – Type-safe identifiers
- `SharedPtr<T>`, `UniquePtr<T>` – Smart pointer wrappers

**Why it's necessary:**
These patterns form the irreducible foundation for building modular, extensible software with proper lifecycle management and loose coupling.

**General usage:**
Any complex domain (gaming, simulation, CAD, web services) benefits from these patterns for object management, extensibility, and cross-module communication.

---

## `io/` – Input/Output abstractions
**Adds:** `Stream`, `Reader`, `Writer`, `BufferedIO`, `AsyncIO`  
**Purpose:** Abstract I/O operations from concrete implementations  
**Usage:** File operations, network communication, inter-process communication

---

## `logging/` – Structured logging framework
**Adds:** `Logger`, `LogSink`, `LogLevel`, `LogScope`, `LogFormatter`  
**Purpose:** Comprehensive application logging with multiple outputs and formats  
**Usage:** Debug output, audit trails, performance logging, error tracking

---

## `concurrency/` – Parallel execution framework
**Adds:** `ThreadPool`, `Task`, `Future`, `Promise`, `parallel_for`, `AsyncExecutor`  
**Purpose:** Abstract threading and async operations  
**Usage:** Background processing, parallel algorithms, async I/O

---

## `memory/` – Memory management
**Adds:** `Arena`, `ObjectPool<T>`, `SmallVector<T,N>`, `StableVector<T>`, `MemoryMappedFile`  
**Purpose:** Efficient memory allocation strategies and specialized containers  
**Usage:** Performance-critical allocations, cache-friendly data structures

---

## `reflection/` – Runtime type information
**Adds:** `TypeInfo`, `TypeRegistry`, `PropertyInfo`, `MethodInfo`  
**Purpose:** Runtime introspection and type discovery  
**Usage:** Serialization, scripting bindings, plugin systems, debugging

---

## `plugins/` – Dynamic loading framework
**Adds:** `DynamicLibrary`, `PluginManager`, `PluginInterface`, `ModuleLoader`  
**Purpose:** Runtime extensibility through dynamic libraries  
**Usage:** Third-party extensions, optional features, modular architectures

---

## `config/` – Configuration management
**Adds:** `Config`, `PropertyTree`, `ConfigParser`, `EnvironmentVariables`  
**Purpose:** Centralized application configuration  
**Usage:** Settings management, feature flags, runtime parameters

---

## `metrics/` – Performance monitoring
**Adds:** `Counter`, `Gauge`, `Histogram`, `Timer`, `MetricsRegistry`  
**Purpose:** Application observability and performance tracking  
**Usage:** Performance analysis, monitoring, alerting

---

## `tracing/` – Profiling support
**Adds:** `Tracer`, `Span`, `TraceContext`, `ProfileScope`  
**Purpose:** Detailed execution tracing and profiling  
**Usage:** Performance optimization, debugging, distributed tracing

---

## `filesystem/` – File system abstractions
**Adds:** `Path`, `FileSystem`, `ResourceManager`, `Directory`, `FileWatcher`  
**Purpose:** Platform-independent file operations  
**Usage:** Asset loading, configuration files, data persistence

---

## `workflow/` – Task orchestration
**Adds:** `Command`, `UndoStack`, `StateMachine`, `Pipeline`, `Scheduler`  
**Purpose:** Complex operation management and workflow control  
**Usage:** Undo/redo systems, state management, task scheduling

---

## `error/` – Error handling framework
**Adds:** `Result<T,E>`, `Expected<T>`, `ErrorCode`, `Exception`, `ErrorHandler`  
**Purpose:** Robust error handling and propagation  
**Usage:** API boundaries, validation, error recovery

---

## `serialization/` – Data persistence
**Adds:** `Serializer`, `Deserializer`, `Archive`, `BinaryFormat`, `JsonFormat`  
**Purpose:** Object serialization and deserialization  
**Usage:** Save/load, network protocols, data exchange

---

## `events/` – Event system
**Adds:** `EventBus`, `Event`, `EventHandler`, `EventQueue`, `Signal/Slot`  
**Purpose:** Decoupled communication between components  
**Usage:** UI updates, system notifications, inter-module messaging

---

## `utilities/` – General utilities
**Adds:** `StringUtils`, `TimeUtils`, `UUID`, `Hash`, `Random`, `CommandLine`  
**Purpose:** Common utility functions used throughout  
**Usage:** String manipulation, time handling, unique identifiers

---

## Dependency Guidelines

### Strict Rules:
1. **`base/` depends on nothing** – It is the absolute foundation
2. **Downward only** – Higher-level folders may depend on lower ones
3. **No circular dependencies** – Ever
4. **Interfaces over implementations** – Cross-module communication via contracts
5. **No global state** – Except managed singletons in `base/`

### Dependency Hierarchy:
```
Level 0: base/
Level 1: error/, utilities/, io/
Level 2: memory/, filesystem/, serialization/
Level 3: logging/, config/, events/, concurrency/
Level 4: metrics/, tracing/, reflection/
Level 5: plugins/, workflow/
```

Each level can depend on levels below it, but not on peers or higher levels.

---

## Design Principles

### Domain Agnosticism
- **No domain-specific types** – No vectors, matrices, meshes, elements
- **No physics/math** – Pure software infrastructure
- **Generic patterns** – Applicable to any C++ application
- **Clean abstractions** – Hide platform/implementation details

### Performance Focus
- **Zero-cost abstractions** – Templates and compile-time optimization
- **Cache-friendly** – Data structure layout considerations
- **Lock-free where possible** – For high-concurrency scenarios
- **RAII everywhere** – Automatic resource management

### Modularity
- **Small, focused components** – Single responsibility
- **Loose coupling** – Communicate through interfaces
- **Pluggable implementations** – Easy to swap components
- **Optional features** – Can exclude unused components

---

## 🚀 Usage Examples (Domain-Agnostic)

### Web Service
```cpp
// Use factories for request handlers
using HandlerFactory = Factory<RequestHandler>;
HandlerFactory::instance().register_type<ApiHandler>("api");

// Thread pool for request processing
ThreadPool pool(std::thread::hardware_concurrency());

// Metrics for monitoring
Counter requests("http.requests");
Histogram latency("http.latency");

// Config for settings
auto config = Config::load("server.yml");
```

### Game Engine
```cpp
// Entity-component for game objects
Entity player("Player");
player.add_component<Transform>();
player.add_component<Renderer>();
player.add_component<Physics>();

// Event bus for game events
EventBus::publish(PlayerMoved{position});

// Resource manager for assets
auto texture = ResourceManager::load<Texture>("player.png");
```

### CAD Application
```cpp
// Command pattern for undo/redo
UndoStack history;
history.push(make_command<MoveCommand>(object, delta));

// Plugin system for tools
PluginManager::instance().load("mesh_tools.dll");

// Observer for document changes
document.attach_observer([](const DocEvent& e) {
    update_ui(e);
});
```

### Scientific Computing (Generic)
```cpp
// Pipeline for data processing
Pipeline pipeline;
pipeline.add_stage<LoadData>();
pipeline.add_stage<Validate>();
pipeline.add_stage<Process>();
pipeline.add_stage<Export>();

// Parallel execution
parallel_for(0, n, [](size_t i) {
    process_item(i);
});

// Tracing for performance analysis
TRACE_SCOPE("computation");
```

---

## Implementation Priority

### Phase 1: Foundation (Critical)
1. `base/` – Core patterns
2. `error/` – Error handling
3. `utilities/` – Common helpers
4. `memory/` – Memory management

### Phase 2: Infrastructure (Essential)
5. `filesystem/` – File operations
6. `logging/` – Debugging support
7. `config/` – Configuration
8. `concurrency/` – Parallelism

### Phase 3: Advanced (Important)
9. `events/` – Communication
10. `serialization/` – Persistence
11. `metrics/` – Monitoring
12. `io/` – I/O abstractions

### Phase 4: Optional (Nice-to-have)
13. `reflection/` – Runtime introspection
14. `plugins/` – Dynamic loading
15. `workflow/` – Complex orchestration
16. `tracing/` – Profiling

---

## Notes for FEM Integration

While the core layer is domain-agnostic, it will seamlessly support FEM-specific layers:

- **FEM objects** will inherit from `Object` for identity
- **Solvers** will use `Factory` for configuration-driven creation
- **Assembly** can leverage `parallel_for` and `ThreadPool`
- **Material models** as `Component` on element `Entity`
- **Mesh events** through `EventBus` for change notification
- **Solver progress** via `metrics/` and `logging/`

The key is that the core layer doesn't know or care about these domain concepts – it just provides the infrastructure patterns they need.

---

This architecture ensures the core layer remains a **pure, reusable foundation** that could be extracted and used in any C++ project, while still providing all the infrastructure needed for a sophisticated FEM solver or any other complex application.
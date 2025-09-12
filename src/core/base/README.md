# FEM Core Base Library

The **base** library provides the foundational design patterns and infrastructure classes for the entire FEM solver. These classes implement essential software engineering patterns that enable modularity, extensibility, and maintainability throughout the codebase.

## Overview

This library contains the core architectural patterns that all other FEM libraries depend on. It provides object lifecycle management, design pattern implementations, and communication infrastructure without any domain-specific FEM knowledge.

## Architecture Design Principles

- **Composition over Inheritance**: Use components and visitors instead of deep inheritance hierarchies
- **RAII (Resource Acquisition Is Initialization)**: Automatic resource management through smart pointers and object lifecycle
- **Type Safety**: Compile-time type checking with C++20 concepts and CRTP patterns
- **Thread Safety**: All infrastructure supports multi-threaded usage
- **Zero Overhead**: Templates and compile-time techniques minimize runtime cost
- **Extensibility**: Plugin architectures and runtime component composition

## Files and Components

### Core Object Infrastructure

#### `object.h/.cpp`
**Root base class for all objects in the FEM solver**
- Unique object identification with atomic ID generation
- Intrusive reference counting for automatic memory management
- Runtime type information (RTTI) support
- Object lifecycle tracking and debugging
- Thread-safe operations

```cpp
class MyElement : public fem::core::Object {
    // Automatically gets ID, ref counting, type info
};

auto element = fem::core::make_object<MyElement>();
auto id = element->id();  // Unique identifier
```

#### `policy.h/.cpp`
**CRTP classes for controlling copy and move semantics**
- `NonCopyable<T>`: Prevents copying, allows moving
- `NonMovable<T>`: Prevents moving, allows copying
- `NonCopyableNonMovable<T>`: Prevents both operations
- Zero overhead policy classes

```cpp
class ResourceHandle : public fem::core::NonCopyable<ResourceHandle> {
    // Can be moved but not copied
};
```

### Design Pattern Implementations

#### `factory.h/.cpp`
**Generic factory pattern for object creation**
- Type-safe object creation by name or type
- Registration of creator functions
- Support for parameterized construction
- Thread-safe registration and creation

```cpp
ElementFactory::instance().register_type<TriangleElement>("triangle");
auto element = ElementFactory::instance().create("triangle");
```

#### `singleton.h/.cpp`
**Thread-safe singleton template using CRTP**
- Lazy initialization with `std::call_once`
- Exception-safe construction
- Non-copyable and non-movable
- Automatic cleanup on program termination

```cpp
class ConfigManager : public fem::core::Singleton<ConfigManager> {
    friend class fem::core::Singleton<ConfigManager>;
private:
    ConfigManager() = default;
};

auto& config = ConfigManager::instance();
```

#### `registry.h/.cpp`
**Object collection management with fast lookup**
- Fast O(1) lookup by ID or name
- Automatic cleanup of destroyed objects
- Thread-safe operations
- Event notifications for registration/unregistration

```cpp
fem::core::Registry<Element> elements("Elements");
elements.register_object("main_beam", beam_element);
auto found = elements.find_by_key("main_beam");
```

#### `visitor.h/.cpp`
**Visitor pattern for operations on object hierarchies**
- Type-safe visitor dispatch
- Hierarchical traversal support
- Composite and conditional visitors
- Visitor registry for plugin architectures

```cpp
class AssemblyVisitor : public fem::core::Visitor<Element> {
public:
    void visit_impl(Element& elem) override {
        elem.compute_stiffness_matrix();
    }
};

for (auto& elem : elements) {
    elem->accept(assembly_visitor);
}
```

### Communication and Events

#### `observer.h/.cpp`
**Observer/Event pattern for loose coupling**
- Type-safe event handling
- Global event bus for application-wide communication
- RAII subscription management
- Thread-safe event dispatch

```cpp
// Subscribe to events
auto subscription = fem::core::subscribe_to_events<MeshEvent>(
    [](const MeshEvent& event) {
        std::cout << "Mesh changed: " << event.to_string() << "\n";
    });

// Emit events
fem::core::emit_event<MeshEvent>(MeshEvent::Type::ELEMENT_ADDED, "mesh_1");
```

#### `interface.h/.cpp`
**Pure virtual interfaces and contracts**
- Lightweight interface base classes
- Type-safe interface checking
- Cross-module communication contracts
- Plugin integration points

```cpp
class ILinearSolver : public fem::core::TypedInterface<ILinearSolver> {
public:
    virtual bool solve(const Matrix& A, const Vector& b, Vector& x) = 0;
};
```

### Advanced Architecture

#### `component.h/.cpp`
**Entity-Component-System architecture**
- Composition over inheritance
- Runtime behavior modification
- Component dependencies and validation
- Multi-physics support through component mixing

```cpp
fem::core::Entity element("ThermoMechanical");
element.add_component<MechanicalComponent>();
element.add_component<ThermalComponent>();
element.update(dt);  // Updates all components
```

## Dependencies

The base library has minimal external dependencies:
- **C++20 Standard Library**: Uses concepts, ranges, format, etc.
- **Threading Support**: `std::mutex`, `std::atomic` for thread safety
- **No External Libraries**: Pure standard C++ implementation

### Internal Dependencies
```
Object ←── Factory, Registry, Singleton
CopyMovePolicy ←── (independent)
Interface ←── (independent) 
Observer ←── (independent)
Visitor ←── (independent)
Component ←── Object (for ObjectEntity only)
```

## Usage Examples

### Basic Object Management
```cpp
// Create objects with automatic lifecycle management
auto solver = fem::core::make_object<LinearSolver>();
auto mesh = fem::core::make_object<Mesh>();

// Objects automatically cleaned up when references go out of scope
// Thread-safe reference counting handles shared ownership
```

### Factory-Based Creation
```cpp
// Register types once
SolverFactory::instance().register_type<DirectSolver>("direct");
SolverFactory::instance().register_type<IterativeSolver>("iterative");

// Create by name (useful for configuration files)
std::string solver_type = config.get("solver_type");
auto solver = SolverFactory::instance().create(solver_type);
```

### Event-Driven Communication
```cpp
// Components can communicate without tight coupling
class MeshManager {
    void refine_mesh() {
        // ... refine mesh ...
        fem::core::emit_event<MeshEvent>(MeshEvent::Type::MESH_REFINED, mesh_name_);
    }
};

class SolverManager {
    SolverManager() {
        // Listen for mesh changes
        mesh_subscription_ = fem::core::subscribe_to_events<MeshEvent>(
            [this](const MeshEvent& event) {
                if (event.get_mesh_type() == MeshEvent::Type::MESH_REFINED) {
                    this->invalidate_matrices();
                }
            });
    }
};
```

### Component-Based Multi-Physics
```cpp
// Runtime composition of physics
fem::core::Entity element("CoupledElement");

if (analysis_includes_mechanics) {
    element.add_component<MechanicalComponent>();
}

if (analysis_includes_thermal) {
    element.add_component<ThermalComponent>();
}

if (analysis_includes_damage) {
    element.add_component<DamageComponent>();
}

// All components work together automatically
element.update(time_step);
```

### Visitor-Based Operations
```cpp
// Multiple operations in single traversal
auto composite = fem::core::make_composite_visitor<Element>();
composite->add_visitor(std::make_unique<AssemblyVisitor>());
composite->add_visitor(std::make_unique<QualityCheckVisitor>());
composite->add_visitor(std::make_unique<ResultsExportVisitor>());

// Apply all operations efficiently
fem::core::VisitorCoordinator<Element>::apply_visitor(*composite, elements);
```

## Design Patterns Summary

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| **Object** | Identity & lifecycle | All persistent objects needing IDs |
| **Factory** | Object creation | Configuration-driven object creation |
| **Singleton** | Global services | Managers, caches, global state |
| **Registry** | Object collections | Fast lookup, lifecycle management |
| **Observer** | Loose coupling | Cross-module communication |
| **Visitor** | Operations on hierarchies | Multi-pass algorithms, extensible operations |
| **Interface** | Contracts | Algorithm abstraction, plugin points |
| **Component** | Composition | Multi-physics, runtime behavior |

## Thread Safety

All base library components are designed for multi-threaded usage:
- **Atomic operations** for ID generation and reference counting
- **Shared mutexes** for reader-writer scenarios
- **Lock-free algorithms** where possible
- **RAII patterns** for exception safety

## Performance Characteristics

- **Object creation**: ~10-50ns overhead for ID generation and registration
- **Factory lookup**: O(1) hash table lookup
- **Event dispatch**: O(n) where n is number of subscribers
- **Visitor dispatch**: Single virtual call per object
- **Component lookup**: O(1) hash table lookup
- **Memory overhead**: ~24-32 bytes per Object (ID + vtable + ref count)

## Testing

Each component includes comprehensive examples demonstrating:
- Basic usage patterns
- Thread safety verification
- Performance characteristics
- Integration scenarios
- Error handling

## Future Extensions

The base library is designed to support future enhancements:
- **Command pattern** for undo/redo operations
- **State machine** for complex object lifecycles
- **Reflection system** for runtime introspection
- **Serialization framework** for persistence
- **Plugin loader** for dynamic libraries

## Integration with FEM Libraries

The base library provides the foundation for all other FEM libraries:

```
┌─────────────────────────────────────────────────────────────┐
│                    FEM Applications                        │
├─────────────────────────────────────────────────────────────┤
│  Analysis │ Mechanics │ Thermal │ Fluid │ Electromagnetics │
├─────────────────────────────────────────────────────────────┤
│    Mesh   │  Geometry │ Material│ Solver│ Boundary Conds   │
├─────────────────────────────────────────────────────────────┤
│                     Core Base Library                      │
│  Object │ Factory │ Singleton │ Registry │ Observer │ etc. │
└─────────────────────────────────────────────────────────────┘
```

Each higher-level library leverages these patterns to implement domain-specific functionality while maintaining consistent architecture and behavior throughout the solver.
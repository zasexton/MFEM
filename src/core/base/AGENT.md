# AGENT.md - FEM Core Base Library

## 🎯 Purpose
This folder contains foundational design patterns and infrastructure classes that enable modularity, extensibility, and maintainability throughout the FEM solver. No domain-specific FEM knowledge here - just pure software engineering patterns.

## 🏗️ Architecture Philosophy
- **Composition over Inheritance**: Use components instead of deep hierarchies
- **RAII**: Automatic resource management through smart pointers
- **Type Safety**: C++20 concepts and CRTP patterns
- **Thread Safety**: All infrastructure supports multi-threading
- **Zero Overhead**: Templates minimize runtime cost

## 📁 Files Overview

### Core Infrastructure
- **object.h/cpp**: Root base class with ID tracking, ref counting, lifecycle management
- **policy.h/cpp**: CRTP classes for copy/move semantics control

### Design Patterns
- **factory.h/cpp**: Generic factory pattern for object creation
- **singleton.h/cpp**: Thread-safe singleton template
- **registry.h/cpp**: Object collection management with fast lookup
- **visitor.h/cpp**: Visitor pattern for object hierarchies
- **observer.h/cpp**: Event system for loose coupling
- **interface.h/cpp**: Pure virtual interfaces and contracts
- **component.h/cpp**: Entity-Component-System architecture

## 🔧 Key Classes & Their Powers

### Object System
```cpp
class Object
```
**Powers:**
- Unique ID generation (atomic, thread-safe)
- Intrusive reference counting
- Runtime type information
- Lifecycle tracking
- Debug info generation

**When to use:** Base for any persistent object needing identity

### Component System
```cpp
class Component, Entity, TypedComponent<T>
```
**Powers:**
- Runtime behavior composition
- Dependency checking
- Type-safe component access
- Multi-physics support through mixing

**When to use:** Building complex objects from reusable parts

### Factory Pattern
```cpp
template<typename T> class Factory
```
**Powers:**
- Create objects by string name
- Parameter-based construction
- Type registration
- Thread-safe creation

**When to use:** Configuration-driven object creation

### Registry Pattern
```cpp
template<typename T, typename Key> class Registry
```
**Powers:**
- O(1) lookup by ID or name
- Automatic cleanup of destroyed objects
- Event notifications
- Filtering and querying

**When to use:** Managing collections of objects

### Observer Pattern
```cpp
class Event, EventDispatcher, Observer<T>, Subject<T>
```
**Powers:**
- Type-safe event handling
- Global event bus
- RAII subscription management
- Thread-safe dispatch

**When to use:** Cross-module communication without coupling

### Visitor Pattern
```cpp
template<typename T> class Visitor, Visitable
```
**Powers:**
- Type-safe double dispatch
- Hierarchical traversal
- Composite visitors
- Conditional visiting

**When to use:** Operations on object hierarchies

### Interface System
```cpp
class Interface, TypedInterface<T>
```
**Powers:**
- Pure virtual contracts
- Zero overhead interfaces
- Type checking at compile time
- Cross-module contracts

**When to use:** Plugin architectures, algorithm abstraction

### Policy Classes
```cpp
NonCopyable<T>, NonMovable<T>, NonCopyableNonMovable<T>
```
**Powers:**
- Control copy/move semantics
- Zero overhead (empty base optimization)
- Clear intent in APIs

**When to use:** Resource management, singleton-like objects

## 💡 Common Patterns & Recipes

### Creating a Managed Object
```cpp
// Object with automatic lifecycle
auto element = make_object<Element>();
auto id = element->id();  // Unique ID
```

### Building with Components
```cpp
Entity element("MultiPhysics");
element.add_component<MechanicalComponent>();
element.add_component<ThermalComponent>();
element.update(dt);  // Updates all components
```

### Factory-Based Creation
```cpp
Factory<Solver>::instance().register_type<DirectSolver>("direct");
auto solver = Factory<Solver>::instance().create("direct");
```

### Event-Driven Communication
```cpp
// Publisher
emit_event<MeshEvent>(MeshEvent::REFINED, mesh_id);

// Subscriber
auto sub = subscribe_to_events<MeshEvent>([](const MeshEvent& e) {
    // Handle event
});
```

### Managing Collections
```cpp
Registry<Element> elements("Elements");
elements.register_object("beam_1", beam);
auto found = elements.find_by_key("beam_1");
```

## 🚀 Performance Characteristics
- Object creation: ~10-50ns overhead
- Factory lookup: O(1) hash table
- Event dispatch: O(n) subscribers
- Component lookup: O(1) hash table
- Memory per Object: ~24-32 bytes

## 🔗 Dependencies
- **External**: None (pure C++20)
- **Internal**: Minimal cross-dependencies within base/

## ⚠️ Important Notes
- All classes are thread-safe by default
- Use smart pointers (object_ptr<T>) for automatic memory management
- Components can have dependencies - they're checked at add time
- Events can carry arbitrary data through std::any
- Visitors can be chained and composed

## 🎮 Quick Decision Guide

| Need | Use |
|------|-----|
| Object with identity | Inherit from `Object` |
| Runtime composition | Use `Entity` + `Component` |
| Create by configuration | Use `Factory<T>` |
| Manage many objects | Use `Registry<T>` |
| Loose coupling | Use `Observer`/`Event` |
| Operate on hierarchies | Use `Visitor` |
| Define contracts | Use `Interface` |
| Control copying | Use `Policy` classes |
| Global service | Use `Singleton<T>` |

## 🧪 Testing Helpers
Each class includes debug methods:
- `to_string()`: Human-readable representation
- `debug_info()`: Detailed state information
- `get_statistics()`: Performance metrics

## 🔮 Future Extensions
- Command pattern for undo/redo
- State machines for complex lifecycles
- Reflection system
- Serialization framework
- Plugin loader for dynamic libraries
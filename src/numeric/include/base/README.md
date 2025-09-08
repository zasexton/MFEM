# FEM Numeric Base Library

## Overview

The FEM Numeric Base Library provides a foundational framework for IEEE 754-compliant numerical computing with support for multi-dimensional arrays, automatic differentiation, expression templates, and advanced indexing. This library forms the core infrastructure for finite element method (FEM) computations while maintaining strict numerical standards.

## Architecture Overview

### Core Components

#### 1. Foundation Layer (`numeric_base.h`)
The foundation defines core concepts, types, and error handling:

- **Concepts**: `NumberLike`, `IEEECompliant` - compile-time constraints for numeric types
- **Shape System**: Multi-dimensional shape representation with broadcasting support
- **Layout & Device**: Memory layout (row/column major) and device location abstraction
- **Error Hierarchy**: Specialized exceptions for dimension mismatches and computation errors
- **IEEE Compliance**: Runtime checking for NaN, Inf, and numerical validity

#### 2. Type System (`traits_base.h`)
Compile-time type introspection and optimization hints:

- **Type Detection**: Identifies complex numbers, dual numbers, and composite types
- **Storage Optimization**: Provides hints for SIMD, alignment, and relocation
- **Type Promotion**: NumPy-like type promotion rules for mixed-type operations
- **Container Traits**: Compile-time properties of containers and storage

#### 3. Storage Layer (`storage_base.h`)
Memory management with multiple strategies:

- **StorageBase**: Abstract interface for all storage implementations
- **DynamicStorage**: Heap-allocated storage using std::vector
- **StaticStorage**: Stack-allocated storage with compile-time size
- **AlignedStorage**: SIMD-aligned memory for vectorized operations

#### 4. Container System (`container_base.h`)
High-level container abstractions:

- **ContainerBase**: CRTP base for owned data containers
- **ViewContainer**: Non-owning references to data
- **Mixins**: SliceableContainer, BroadcastableContainer, AxisReducible

#### 5. View System (`view_base.h`)
Non-owning data access patterns:

- **ViewBase**: Simple contiguous views
- **StridedView**: Non-contiguous data with custom strides
- **MultiDimView**: N-dimensional views with arbitrary strides

#### 6. Iteration (`iterator_base.h`)
Iterator adapters for various access patterns:

- **ContainerIterator**: Standard random-access iterator
- **StridedIterator**: Iterator with custom stride
- **MultiDimIterator**: N-dimensional iteration
- **CheckedIterator**: IEEE compliance checking during iteration

#### 7. Slicing & Indexing (`slice_base.h`)
NumPy-like advanced indexing:

- **Slice**: Python-style slice objects with start:stop:step
- **MultiIndex**: Complex indexing with slices, arrays, masks
- **Special Sentinels**: `all`, `newaxis`, `ellipsis`
- **String Parsing**: Parse slice strings like "1:5:2" or "::2,3"

#### 8. Operations (`ops_base.h`)
Functor-based operations with IEEE compliance:

- **Arithmetic**: Addition, multiplication, division with NaN/Inf handling
- **Transcendental**: sin, cos, exp, log with domain checking
- **Comparisons**: IEEE-compliant comparison (NaN != NaN)
- **Reductions**: sum, mean, variance with numerical stability

#### 9. Expression Templates (`expression_base.h`)
Lazy evaluation system:

- **Expression Trees**: Build computation graphs without temporaries
- **Broadcasting**: Automatic shape broadcasting in expressions
- **Optimization**: Parallel and vectorized evaluation strategies
- **Type Safety**: Compile-time expression validation

#### 10. Broadcasting (`broadcast_base.h`)
NumPy-style broadcasting rules:

- **Shape Compatibility**: Determine broadcastable shapes
- **Index Mapping**: Map broadcasted indices to original data
- **Broadcast Iterators**: Iterate over broadcasted data without copying

#### 11. Automatic Differentiation (`dual_base.h`, `dual_math.h`, `dual_comparison.h`)
Forward-mode automatic differentiation:

- **DualBase**: Dual number implementation with value and derivatives
- **Chain Rule**: Automatic derivative propagation
- **Math Functions**: Extended math library with derivatives
- **Comparison**: Careful handling of non-differentiable operations

#### 12. Memory Management (`allocator_base.h`)
Custom allocators for specialized needs:

- **AlignedAllocator**: SIMD-aligned allocation
- **PoolAllocator**: Fast allocation from memory pools
- **TrackingAllocator**: Memory usage profiling
- **StackAllocator**: Stack-based temporary allocations

## Usage Examples

### Basic Container Creation
```cpp
using namespace fem::numeric;

// Create a shape
Shape shape({3, 4, 5});  // 3x4x5 tensor

// Dynamic storage
DynamicStorage<double> storage(shape.size());
ContainerBase<MyContainer, double> container(shape);

// Fill with value
container.fill(3.14);

// Check for numerical issues
if (container.has_nan()) {
    throw ComputationError("NaN detected");
}
```

### Advanced Slicing
```cpp
// NumPy-style slicing
MultiIndex idx = "2:5, ::2, :"_idx;  // [2:5, ::2, :]
auto view = container[idx];

// Using slice objects
Slice s(2, 10, 2);  // start=2, stop=10, step=2
auto indices = s.indices(container.size());
```

### Expression Templates
```cpp
// Lazy evaluation - no temporaries created
auto expr = (a + b) * c - d / 2.0;

// Force evaluation
auto result = expr.eval<DynamicStorage, double>();

// Parallel evaluation for large arrays
if (expr.is_parallelizable() && expr.size() > 1000) {
    expr.parallel_eval_to(result);
}
```

### Broadcasting
```cpp
Shape a_shape({3, 1, 5});
Shape b_shape({1, 4, 5});

// Check compatibility
if (BroadcastHelper::are_broadcastable(a_shape, b_shape)) {
    Shape result_shape = BroadcastHelper::broadcast_shape(a_shape, b_shape);
    // result_shape = (3, 4, 5)
}

// Broadcast iteration without copying
auto [begin, end] = BroadcastHelper::make_broadcast_range(
    data.data(), data.shape(), broadcast_shape);
```

### Automatic Differentiation
```cpp
using Dual = DualBase<double, 3>;  // 3 derivative directions

// Create independent variables
Dual x = make_independent(2.0, 0);  // df/dx
Dual y = make_independent(3.0, 1);  // df/dy

// Compute function and derivatives
Dual f = sin(x) * exp(y) + pow(x, 2);

double value = f.value();           // f(2, 3)
double dfdx = f.derivative(0);      // ∂f/∂x
double dfdy = f.derivative(1);      // ∂f/∂y
```

### Custom Storage
```cpp
// Stack allocation for small arrays
StaticStorage<float, 100> small_storage(50);

// Aligned storage for SIMD
AlignedStorage<double, 32> simd_storage(1024);

// Pool allocator for many small allocations
PoolAllocator<int> pool(256);
```

## Design Principles

1. IEEE 754 Compliance

All operations respect IEEE floating-point standards
NaN propagation and Inf handling follow IEEE rules
Optional runtime checking for numerical validity

2. Zero-Cost Abstractions

Heavy use of templates and CRTP for compile-time polymorphism
Expression templates eliminate temporaries
Inline functions and constexpr for optimization

3. Memory Safety

RAII for automatic resource management
Strong exception safety guarantees
Bounds checking in debug mode

4. Flexibility

Multiple storage strategies (heap, stack, aligned)
Pluggable allocators for custom memory management
View system for non-owning data access

5. Interoperability

NumPy-like broadcasting and indexing
STL-compatible iterators
Conversion to/from std::span

## Dependencies

### Internal Dependencies
```
numeric_base.h (foundation)
├── traits_base.h (uses concepts from numeric_base)
├── storage_base.h (uses traits and numeric_base)
├── iterator_base.h (uses numeric_base)
├── slice_base.h (uses Shape from numeric_base)
├── view_base.h (uses numeric_base, storage_base)
├── ops_base.h (uses numeric_base for compliance checking)
├── broadcast_base.h (uses Shape from numeric_base)
├── allocator_base.h (uses numeric_base)
├── container_base.h (uses all above)
├── expression_base.h (uses container, ops, broadcast)
└── dual_base.h, dual_math.h, dual_comparison.h (uses numeric_base)
```

### External Dependencies

- C++20 or later
- Standard library (memory, vector, array, span, concepts)
- Optional: OpenMP for parallel execution

## Performance Considerations


# AI Agent Guide: FEM Numeric Base Library

## Overview for AI Systems

This document provides architectural context and implementation guidance for AI systems working with the FEM Numeric Base Library. The library implements a sophisticated numerical computing framework with strict IEEE 754 compliance, designed for finite element methods but applicable to general scientific computing.

## Files Overview

- `numeric_base.h` - Fundamental numeric concepts and shape utilities.
- `traits_base.h` - Compile-time traits for numeric, complex, and dual types.
- `storage_base.h` - Abstract interface for memory storage strategies.
- `container_base.h` - CRTP base for owned containers providing core operations.
- `view_base.h` - Non-owning view interfaces including strided access.
- `iterator_base.h` - Random-access iterators for contiguous and strided traversal.
- `slice_base.h` - NumPy-style slicing and multi-index helpers.
- `ops_base.h` - Functors for arithmetic and reduction operations with IEEE checks.
- `broadcast_base.h` - Helpers implementing NumPy-style broadcasting rules.
- `expression_base.h` - Expression template framework for lazy evaluation.
- `allocator_base.h` - Allocator interfaces for aligned and pooled memory.
- `dual_base.h` - Forward-mode dual number implementation tracking derivatives.
- `dual_math.h` - Mathematical functions specialized for dual numbers.
- `dual_comparison.h` - Comparison operators and utilities for dual numbers.

## Architectural Philosophy

### Core Design Principles

1. **Compile-Time Polymorphism**: The library extensively uses CRTP (Curiously Recurring Template Pattern) and templates to achieve polymorphism without virtual function overhead. This is critical for performance in numerical computing.

2. **Expression Templates**: Lazy evaluation through expression templates eliminates temporaries and enables loop fusion. Expressions build ASTs at compile-time that are optimized before evaluation.

3. **IEEE 754 Compliance**: Every operation respects IEEE floating-point semantics. NaN propagates correctly, comparisons handle special values properly, and domain errors are caught.

4. **Zero-Cost Abstractions**: High-level interfaces compile down to efficient machine code. The abstraction penalty is paid at compile-time, not runtime.

5. **Separation of Concerns**: Storage, shape, operations, and algorithms are orthogonal. Any storage can work with any algorithm through well-defined interfaces.

## Key Architectural Patterns

### CRTP Usage Pattern
```cpp
template<typename Derived>
class Base {
    Derived& derived() { return static_cast<Derived&>(*this); }
    // Common interface using derived()
};

class Concrete : public Base<Concrete> {
    // Implement required methods
};
```

The library uses this pattern in:
- `NumericBase` for all numeric objects
- `ContainerBase` for owned containers
- `StorageBase` for memory strategies
- `ExpressionBase` for expression templates

### Expression Template Pattern
```cpp
// Build expression tree without evaluation
auto expr = a + b * c;  // Creates BinaryExpression<...> tree

// Evaluate when needed
container.eval_to(result);  // Single loop, no temporaries
```

Key components:
- `TerminalExpression`: Leaf nodes (actual data)
- `BinaryExpression`: Binary operations
- `UnaryExpression`: Unary operations
- `ScalarExpression`: Broadcasted scalars

### Storage Abstraction Pattern
```cpp
class StorageBase {
    virtual T* data() = 0;
    virtual size_t size() = 0;
    // Minimal virtual interface
};

class DynamicStorage : public StorageBase {
    std::vector<T> data_;  // Concrete implementation
};
```

This allows switching storage strategies without changing algorithms.

## Implementation Guidelines

### When Adding New Features

#### New Storage Types
1. Inherit from `StorageBase<T>`
2. Implement pure virtual methods
3. Consider adding to `storage_traits` for compile-time properties
4. Test with `CheckedIterator` for IEEE compliance

#### New Operations
1. Add functor to `ops_base.h` following existing patterns
2. Include IEEE compliance checking in `check_inputs()`
3. Handle special cases (NaN, Inf, domain errors)
4. Add to `OperationDispatcher` if runtime selection needed

#### New Container Types
1. Use CRTP with `ContainerBase` or `NumericBase`
2. Implement shape(), data(), size() at minimum
3. Support both owned and view semantics
4. Integrate with expression templates

#### New Index Types
1. Add variant to `IndexVariant` in `slice_base.h`
2. Implement normalization in `MultiIndex::normalize()`
3. Update `result_shape()` calculation
4. Extend `IndexParser` for string parsing

### Critical Invariants to Maintain

1. **Shape Consistency**: Operations must preserve or correctly transform shapes
2. **Memory Safety**: RAII everywhere, no raw new/delete
3. **IEEE Compliance**: NaN/Inf handling must follow IEEE 754
4. **Iterator Validity**: Iterators must remain valid per STL rules
5. **Exception Safety**: At least basic guarantee, prefer strong

### Performance Optimization Patterns

#### SIMD Optimization
```cpp
if (storage.supports_simd() && storage.is_contiguous()) {
    // Use vectorized implementation
    #pragma omp simd
    for (size_t i = 0; i < size; i += simd_traits<T>::vector_size) {
        // Process multiple elements
    }
}
```

#### Memory Access Patterns
- Prefer contiguous access (stride == 1)
- Use `StridedView` for non-contiguous without copying
- Check `is_contiguous()` before optimizations
- Align data for SIMD when possible

#### Expression Evaluation
- Small expressions: Direct evaluation
- Large expressions (>1000 elements): Consider parallelization
- Check `is_parallelizable()` and `is_vectorizable()`
- Use `complexity()` to decide evaluation strategy

## Common Pitfalls and Solutions

### Pitfall 1: Template Instantiation Explosion
**Problem**: Too many template instantiations increase compile time and binary size.
**Solution**: Use type erasure for non-performance-critical paths. Explicitly instantiate common types.

### Pitfall 2: Broadcasting Overhead
**Problem**: Broadcasting can hide expensive operations.
**Solution**: Check `is_broadcastable_with()` explicitly. Use `BroadcastIterator` for efficient access.

### Pitfall 3: View Lifetime Issues
**Problem**: Views can outlive their data source.
**Solution**: Views don't own data by design. Document lifetime requirements. Consider `shared_ptr` for shared ownership.

### Pitfall 4: IEEE Compliance Overhead
**Problem**: Checking every operation for NaN/Inf is expensive.
**Solution**: Use `NumericOptions` to control checking. Enable in debug, disable in release.

## Integration Points

### With Linear Algebra Libraries
- Storage is compatible with BLAS/LAPACK through `data()` pointer
- Use `Layout` to specify row/column major
- `AlignedStorage` ensures SIMD compatibility

### With Automatic Differentiation
- `DualBase` integrates as a numeric type
- Works with containers and expressions
- Special handling in `storage_optimization_traits`

### With Parallel Frameworks
- OpenMP support through pragmas
- `Device` abstraction for future GPU support
- Thread-safe read operations, synchronized writes

## Testing Strategies

### Unit Testing Focus Areas
1. **Shape Operations**: Broadcasting, reshaping, slicing
2. **IEEE Compliance**: NaN/Inf propagation, domain errors
3. **Memory Safety**: Valgrind/AddressSanitizer clean
4. **Performance**: Benchmark against raw loops
5. **Edge Cases**: Empty containers, singleton dimensions

### Property-Based Testing
```cpp
// Properties to verify
for_all_shapes([](const Shape& a, const Shape& b) {
    if (a.is_broadcastable_with(b)) {
        auto result = broadcast_shape(a, b);
        assert(result.size() >= max(a.size(), b.size()));
    }
});
```

## Code Generation Patterns

When generating code that uses this library:

### Container Creation
```cpp
// Prefer factory functions
auto storage = make_aligned_storage<double>(size);
auto container = make_container<double>(shape);

// Over direct construction
AlignedStorage<double> storage(size);  // Less flexible
```

### Expression Building
```cpp
// Build complex expressions incrementally
auto expr1 = make_expression(a) + make_expression(b);
auto expr2 = expr1 * make_scalar_expression(2.0, a.shape());

// Rather than one large expression
auto expr = (a + b) * 2.0 - c / d;  // Harder to debug
```

### Error Handling
```cpp
try {
    // Numeric operations
} catch (const DimensionError& e) {
    // Handle shape mismatches
} catch (const ComputationError& e) {
    // Handle numerical failures
} catch (const NumericError& e) {
    // Generic numeric error
}
```

## Debugging Support

### Key Inspection Points
1. `shape.to_string()`: Human-readable shape
2. `storage.is_contiguous()`: Memory layout
3. `container.has_nan()`: Numerical validity
4. `expr.complexity()`: Expression cost
5. `view.overlaps()`: Aliasing detection

### Compile-Time Debugging
```cpp
// Use static_assert for compile-time checks
static_assert(StorableType<MyType>, "Type not storable");
static_assert(are_compatible_v<T1, T2>, "Incompatible types");

// Use concepts for better error messages
template<NumberLike T>
void process(T value) { /* ... */ }
```

## Future Evolution Considerations

### Planned Extensions
1. **Sparse Storage**: Integrate with existing storage hierarchy
2. **GPU Support**: Extend Device enum and storage types
3. **Distributed Arrays**: Add communication layer
4. **Mixed Precision**: Extend type promotion rules
5. **JIT Compilation**: Expression template optimization

### Extension Points
- `StorageBase`: New storage strategies
- `Device`: New compute devices
- `Layout`: New memory layouts
- `IndexVariant`: New indexing modes
- `OperationDispatcher`: New operations

### Backward Compatibility
- Virtual interfaces are minimal and stable
- Template interfaces can extend without breaking
- New features add specializations, not modifications
- Follow semantic versioning for API changes

## Performance Profiling Guidance

### Hot Paths to Monitor
1. Expression evaluation loops
2. Broadcasting index calculations
3. Storage allocation/deallocation
4. Iterator advancement
5. Shape manipulation

### Optimization Opportunities
- Loop fusion in expressions
- SIMD vectorization in operations
- Memory pooling for small allocations
- Lazy evaluation for views
- Compile-time shape checking

## Conclusion

This library provides a sophisticated foundation for numerical computing with careful attention to correctness, performance, and extensibility. When working with this code:

1. Respect the IEEE 754 invariants
2. Leverage compile-time polymorphism
3. Use expression templates for optimization
4. Maintain separation of concerns
5. Profile before optimizing

The architecture is designed to be both powerful and predictable, enabling high-performance numerical computing while maintaining mathematical correctness.
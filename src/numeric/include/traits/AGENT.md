# AGENT.md - Numeric Traits System

## 🎯 Purpose
Compile-time type introspection and trait system enabling template metaprogramming, SFINAE-based dispatch, and C++20 concepts for the FEM numeric library. Zero runtime overhead type safety and optimization.

## 🏗️ Architecture Philosophy
- **Compile-Time Everything**: All decisions made at compile time
- **Zero Overhead**: Traits compile away to nothing
- **Composable**: Traits build on each other hierarchically
- **Detection-Based**: Safe detection over hard requirements
- **Concept-Driven**: C++20 concepts for clear constraints

## 📁 Files Overview

### Core Type System
- **type_traits.hpp**: Basic type properties (is_complex, is_pod, real_type)
- **numeric_traits.hpp**: Math properties (IEEE compliance, precision, limits)
- **sfinae_helpers.hpp**: Detection idiom and SFINAE utilities

### Container & Storage
- **container_traits.hpp**: Container properties (dense/sparse, layout, dimensions)
- **storage_traits.hpp**: Memory characteristics (alignment, contiguous, growth)
- **iterator_traits.hpp**: Iterator patterns (strided, checked, parallel-safe)

### Operations & Expressions
- **operation_traits.hpp**: Operation compatibility (addable, multiplicable, broadcastable)
- **expression_traits.hpp**: Expression template properties (lazy, broadcasting, SIMD)
- **concepts.hpp**: C++20 concepts combining all traits

## 🔧 Key Patterns & Usage

### Detection Idiom Pattern
```cpp
// Define what to detect
template<typename T>
using has_eval_t = decltype(std::declval<T>().eval());

// Check if detected
if constexpr (is_detected_v<has_eval_t, MyType>) {
    // MyType has eval() method
}
```

### Type Promotion Pattern
```cpp
// Automatic type promotion for mixed operations
template<typename T1, typename T2>
using result_t = promote_t<T1, T2>;  // float + double → double
```

### Concept-Based Dispatch
```cpp
template<NumericContainer C>  // Only numeric containers
void process(C& container) {
    if constexpr (IEEECompliant<typename C::value_type>) {
        // IEEE-specific optimizations
    }
}
```

### SFINAE Selection
```cpp
// Select implementation based on capabilities
template<typename T>
auto compute(T value) -> enable_if_t<has_simd_v<T>, T> {
    // SIMD implementation
}

template<typename T>
auto compute(T value) -> enable_if_t<!has_simd_v<T>, T> {
    // Scalar implementation
}
```

## 🎨 Design Patterns

### 1. Trait Hierarchy
```
numeric_traits<T>
    ├── is_complex
    ├── is_signed
    ├── has_nan
    └── promote_traits<T,U>
            └── type
```

### 2. Container Classification
```
Container → Dense → Contiguous → SIMD-Compatible
         → Sparse → Compressed → CSR/CSC/COO
         → View → Strided → Non-owning
         → Expression → Lazy → Broadcastable
```

### 3. Operation Validation
```cpp
Binary Op: can_operate<Op, T1, T2> → result_type<Op, T1, T2>
Unary Op:  can_operate<Op, T> → result_type<Op, T>
```

## 💡 Quick Reference

### Type Checks
```cpp
is_complex_v<T>           // Complex number type
is_pod_v<T>              // Plain old data
is_arithmetic_v<T>       // Supports arithmetic ops
is_numeric_v<T>          // Any numeric type
NumberLike<T>            // Concept: number-like behavior
IEEECompliant<T>         // Concept: IEEE 754 compliant
```

### Container Checks
```cpp
is_container_base_v<T>   // Derives from ContainerBase
has_shape_v<T>           // Has shape() method
is_resizable_v<T>        // Can change size
is_view_base_v<T>        // Is a view type
supports_simd_v<T>       // SIMD operations possible
```

### Operation Checks
```cpp
is_addable_v<T1, T2>     // Can add types
is_multiplicable_v<T1, T2> // Can multiply
is_broadcastable_v<S1, S2> // Shapes can broadcast
has_eval_v<T>            // Has eval() method
```

## 🚀 Performance Tips

1. **Use `if constexpr`** for compile-time branching
2. **Prefer concepts** over SFINAE for cleaner code
3. **Cache trait results** in using declarations
4. **Minimize template depth** to reduce compile time
5. **Use detection idiom** for optional features

## ⚠️ Common Pitfalls

1. **Forgetting constexpr**: Traits must be constexpr
2. **Circular dependencies**: Traits depending on each other
3. **Over-constraining**: Too many requirements limiting usage
4. **SFINAE failures**: Malformed detection expressions
5. **Template bloat**: Too many instantiations

## 🔗 Integration Points

### With Base Infrastructure
- Extends `numeric_traits` from `traits_base.hpp`
- Uses `NumberLike` and `IEEECompliant` concepts
- Leverages `ContainerBase`, `StorageBase`, `ExpressionBase`

### With Other Systems
- Storage layer uses storage traits for allocation
- Containers use container traits for optimization
- Expression templates use expression traits for evaluation
- Algorithms use operation traits for dispatch

## 📊 Trait Categories

### Level 1: Basic Types
- Fundamental properties (size, alignment)
- Numeric properties (signed, precision)
- Type classification (integral, floating, complex)

### Level 2: Containers
- Structure (dense, sparse, view)
- Capabilities (resizable, iterable)
- Memory layout (row/column major, strided)

### Level 3: Operations
- Binary operations (arithmetic, comparison)
- Unary operations (negation, functions)
- Reductions (sum, product, min/max)

### Level 4: Optimizations
- SIMD compatibility
- Parallelization safety
- Memory access patterns
- Broadcasting rules

## 🧪 Testing Traits

```cpp
// Compile-time tests
static_assert(is_complex_v<std::complex<double>>);
static_assert(promote_t<float, double> == double);
static_assert(IEEECompliant<float>);

// Runtime validation
if constexpr (has_nan_v<T>) {
    assert(!std::isnan(value));
}
```

## 🔄 Extension Points

### Adding New Traits
1. Define in appropriate header
2. Provide base template
3. Add specializations
4. Create convenience aliases
5. Document behavior

### Adding New Concepts
1. Combine existing traits
2. Add semantic requirements
3. Provide subsumption hierarchy
4. Test with various types

## 📈 Compilation Impact

- **Fast**: Type aliases, basic traits
- **Medium**: Detection idiom, SFINAE helpers
- **Slow**: Complex concepts, recursive traits
- **Very Slow**: Deep template instantiations

## 🎯 Usage Strategy

1. **Start with concepts** for clear interfaces
2. **Fall back to traits** for fine-grained control
3. **Use SFINAE helpers** for optional features
4. **Apply detection** for safe feature queries
5. **Optimize with traits** for performance paths
# Numeric Traits Library

## Overview

The `numeric/traits` folder provides a comprehensive compile-time type introspection and trait system for the FEM numeric library. It builds upon the foundational infrastructure in `numeric/base` and enables sophisticated template metaprogramming, SFINAE-based dispatch, and C++20 concepts for type safety and optimization.

## Architecture

The traits system is organized into several layers:

1. **Type Traits** - Basic type properties and classifications
2. **Numeric Traits** - Mathematical and IEEE compliance properties
3. **Container Traits** - Properties of data structures
4. **Expression Traits** - Expression template characteristics
5. **Operation Traits** - Operation compatibility and results
6. **Storage Traits** - Memory layout and storage properties
7. **Iterator Traits** - Iterator capabilities and patterns
8. **Concepts** - C++20 concepts for type constraints
9. **SFINAE Helpers** - Utilities for compile-time detection

## File Structure

```
numeric/traits/
├── type_traits.hpp         # Basic type properties
├── numeric_traits.hpp      # Numeric type characteristics
├── container_traits.hpp    # Container properties
├── expression_traits.hpp   # Expression template traits
├── operation_traits.hpp    # Operation compatibility
├── storage_traits.hpp      # Storage characteristics
├── iterator_traits.hpp     # Iterator properties
├── concepts.hpp            # C++20 concepts
└── sfinae_helpers.hpp      # SFINAE utilities
```

## Core Components

### 1. Type Traits (`type_traits.hpp`)

Provides fundamental type properties and classifications:

```cpp
// Check if type is complex
fem::numeric::traits::is_complex_v<std::complex<double>>  // true

// Check if type is POD
fem::numeric::traits::is_pod_v<float>  // true

// Get real type from complex
using real_t = fem::numeric::traits::real_type_t<std::complex<float>>;  // float

// Check numeric properties
fem::numeric::traits::is_arithmetic_v<double>  // true
fem::numeric::traits::is_numeric_v<std::complex<float>>  // true
```

### 2. Numeric Traits (`numeric_traits.hpp`)

Extends `std::numeric_limits` with additional numeric properties:

```cpp
// Get numeric limits
using limits = fem::numeric::traits::numeric_limits<double>;
constexpr auto eps = limits::epsilon();

// Check precision category
auto precision = fem::numeric::traits::precision_category_v<float>;  // Single

// Get numeric properties
using props = fem::numeric::traits::numeric_properties<double>;
static_assert(props::is_ieee);  // IEEE compliant
static_assert(props::has_nan);  // Has NaN support

// Type promotion
using promoted = fem::numeric::traits::promote_t<float, double>;  // double

// Storage requirements
auto bytes = fem::numeric::traits::storage_requirements<float>::bytes_needed(100);
```

### 3. Container Traits (`container_traits.hpp`)

Properties and capabilities of container types:

```cpp
// Check container properties
using traits = fem::numeric::traits::extended_container_traits<Matrix<double>>;
static_assert(traits::is_dense);
static_assert(traits::has_random_access);

// Get container category
auto category = traits::category;  // ContainerCategory::Dense

// Check capabilities
static_assert(traits::supports_simd);
static_assert(traits::is_resizable);

// Memory layout
auto layout = traits::layout;  // MemoryLayout::RowMajor
```

### 4. Expression Traits (`expression_traits.hpp`)

Properties of expression templates:

```cpp
// Check if type is an expression
fem::numeric::traits::is_expression_v<BinaryExpression<...>>  // true

// Get expression properties
using traits = fem::numeric::traits::expression_traits<Expr>;
static_assert(traits::is_lazy);  // Lazy evaluation
static_assert(traits::supports_broadcasting);

// Get result type
using result = fem::numeric::traits::expression_result_t<Add, Vector<float>, Vector<double>>;
// result is Vector<double> due to promotion
```

### 5. Operation Traits (`operation_traits.hpp`)

Operation compatibility and result types:

```cpp
// Check if operation is valid
constexpr bool can_add = fem::numeric::traits::is_addable_v<Matrix<float>, Matrix<float>>;

// Get operation result type
using result = fem::numeric::traits::binary_op_result_t<
    Vector<float>, Vector<double>, std::plus<>
>;  // Vector<double>

// Check broadcasting compatibility
constexpr bool can_broadcast = fem::numeric::traits::is_broadcastable_v<
    Shape{3, 1}, Shape{1, 5}
>;  // true - results in Shape{3, 5}
```

### 6. Storage Traits (`storage_traits.hpp`)

Memory and storage characteristics:

```cpp
// Get storage properties
using traits = fem::numeric::traits::extended_storage_traits<AlignedStorage<float>>;
static_assert(traits::is_contiguous);
static_assert(traits::alignment == 32);  // 32-byte aligned

// Check storage category
auto category = traits::category;  // StorageCategory::Aligned

// Memory requirements
auto min_size = traits::min_allocation_size;
auto grow_factor = traits::growth_factor;  // 1.5x growth
```

### 7. Iterator Traits (`iterator_traits.hpp`)

Iterator capabilities and patterns:

```cpp
// Check iterator category
using traits = fem::numeric::traits::numeric_iterator_traits<StridedIterator<double>>;
auto category = traits::numeric_category;  // NumericIteratorCategory::Strided

// Access pattern
auto pattern = traits::access_pattern;  // IteratorAccessPattern::Strided

// Check capabilities
static_assert(traits::is_mutable);
static_assert(traits::supports_parallel);
```

### 8. Concepts (`concepts.hpp`)

C++20 concepts for type constraints:

```cpp
// Basic concepts
template<fem::numeric::concepts::Arithmetic T>
void process(T value);  // Only accepts arithmetic types

// Container concepts
template<fem::numeric::concepts::DenseContainer C>
void optimize(C& container);  // Only dense containers

// Field concepts
template<fem::numeric::concepts::Field T>
Matrix<T> inverse(const Matrix<T>& m);  // Requires field operations

// Advanced concepts
template<fem::numeric::concepts::SIMDCompatible T>
void vectorized_compute(T* data, size_t n);
```

### 9. SFINAE Helpers (`sfinae_helpers.hpp`)

Utilities for compile-time detection and dispatch:

```cpp
// Detection idiom
template<typename T>
using has_eval_t = decltype(std::declval<T>().eval());

constexpr bool has_eval = fem::numeric::traits::is_detected_v<has_eval_t, MyType>;

// Type selection
using storage_t = fem::numeric::traits::detected_or_t<
    DynamicStorage<T>,  // Default
    has_storage_type_t,  // Try to detect
    Container          // From this type
>;

// Enable if numeric
template<typename T>
fem::numeric::traits::enable_if_t<fem::numeric::NumberLike<T>, T>
compute(T value);
```

## Usage Examples

### Example 1: Type-Safe Matrix Operations

```cpp
#include <numeric/traits/concepts.hpp>
#include <numeric/traits/operation_traits.hpp>

template<fem::numeric::concepts::NumericMatrix M1,
         fem::numeric::concepts::NumericMatrix M2>
auto multiply(const M1& a, const M2& b) 
    -> Matrix<fem::numeric::traits::promote_t<
        typename M1::value_type, 
        typename M2::value_type
    >>
{
    using result_type = fem::numeric::traits::promote_t<
        typename M1::value_type, 
        typename M2::value_type
    >;
    
    static_assert(
        fem::numeric::traits::is_multiplicable_v<M1, M2>,
        "Matrices must be multiplicable"
    );
    
    // Implementation...
}
```

### Example 2: Optimized Storage Selection

```cpp
#include <numeric/traits/storage_traits.hpp>

template<typename T, size_t N>
class SmallVector {
    using storage_type = std::conditional_t<
        N * sizeof(T) <= 256,  // Small buffer optimization
        fem::numeric::StaticStorage<T, N>,
        fem::numeric::DynamicStorage<T>
    >;
    
    static constexpr bool use_simd = 
        fem::numeric::traits::extended_storage_traits<storage_type>::supports_simd;
        
    storage_type storage_;
    
public:
    void process() {
        if constexpr (use_simd) {
            // SIMD implementation
        } else {
            // Scalar implementation
        }
    }
};
```

### Example 3: Expression Template Optimization

```cpp
#include <numeric/traits/expression_traits.hpp>

template<typename Expr>
auto evaluate(const Expr& expr) {
    using traits = fem::numeric::traits::expression_traits<Expr>;
    
    if constexpr (traits::is_lazy) {
        // Force evaluation
        return expr.eval();
    } else if constexpr (traits::is_terminal) {
        // Direct access
        return expr;
    } else {
        // Complex expression
        if constexpr (traits::supports_parallel) {
            return parallel_eval(expr);
        } else {
            return sequential_eval(expr);
        }
    }
}
```

### Example 4: Generic Algorithm with Trait Detection

```cpp
#include <numeric/traits/sfinae_helpers.hpp>
#include <numeric/traits/container_traits.hpp>

template<typename Container>
void optimize_layout(Container& c) {
    using traits = fem::numeric::traits::extended_container_traits<Container>;
    
    // Detect capabilities at compile time
    if constexpr (traits::is_resizable) {
        // Can change size
        c.reserve(calculate_optimal_size(c));
    }
    
    if constexpr (traits::has_static_shape) {
        // Compile-time shape known
        constexpr auto dims = traits::static_dimensions;
        // Optimize for known dimensions
    }
    
    if constexpr (fem::numeric::traits::has_is_contiguous_v<Container>) {
        if (c.is_contiguous()) {
            // Apply contiguous optimizations
        }
    }
}
```

## Integration with Base Infrastructure

The traits system seamlessly integrates with the base infrastructure:

```cpp
// Using base concepts
static_assert(fem::numeric::NumberLike<double>);
static_assert(fem::numeric::IEEECompliant<float>);

// Extending base traits
template<typename T>
struct my_traits : fem::numeric::numeric_traits<T> {
    // Add custom properties
    static constexpr bool supports_my_feature = /* ... */;
};

// Working with base containers
template<fem::numeric::concepts::NumericContainer C>
void process(C& container) {
    using base = fem::numeric::ContainerBase<
        typename C::value_type, C
    >;
    // Use base functionality
}
```

## Best Practices

### 1. Use Concepts for Constraints
```cpp
// Good: Clear requirements
template<fem::numeric::concepts::Field T>
T inverse(T value);

// Avoid: Manual SFINAE
template<typename T, 
         typename = std::enable_if_t</* complex conditions */>>
T inverse(T value);
```

### 2. Leverage Type Traits for Optimization
```cpp
template<typename T>
void compute(T* data, size_t n) {
    if constexpr (fem::numeric::traits::simd_traits<T>::supported) {
        // SIMD path
    } else {
        // Scalar path
    }
}
```

### 3. Use Detection Idiom for Optional Features
```cpp
template<typename T>
void process(T& obj) {
    if constexpr (fem::numeric::traits::has_eval_v<T>) {
        auto result = obj.eval();
    } else {
        auto result = obj;
    }
}
```

### 4. Prefer Trait Aliases
```cpp
// Good: Clear and concise
using result_t = fem::numeric::traits::promote_t<T1, T2>;

// Avoid: Verbose
using result_t = typename fem::numeric::traits::promote_traits<T1, T2>::type;
```

## Performance Considerations

1. **Compile-Time Evaluation**: All traits are evaluated at compile time with zero runtime overhead
2. **Constexpr Functions**: Use `constexpr` for compile-time computations
3. **Template Instantiation**: Be mindful of template bloat with complex trait hierarchies
4. **SFINAE Overhead**: Modern concepts are faster to compile than SFINAE

## Extending the System

### Adding New Traits

```cpp
namespace fem::numeric::traits {
    
    // Define new trait
    template<typename T>
    struct my_custom_trait {
        static constexpr bool value = /* detection logic */;
        using type = /* associated type */;
    };
    
    // Convenience alias
    template<typename T>
    inline constexpr bool my_custom_trait_v = my_custom_trait<T>::value;
    
    // Specializations for specific types
    template<>
    struct my_custom_trait<SpecialType> {
        static constexpr bool value = true;
        using type = SpecialResult;
    };
}
```

### Adding New Concepts

```cpp
namespace fem::numeric::concepts {
    
    template<typename T>
    concept MyCustomConcept = 
        NumberLike<T> &&
        requires(T a, T b) {
            { a.custom_operation(b) } -> std::same_as<T>;
        };
}
```

## Troubleshooting

### Common Issues

1. **Ambiguous Template Resolution**
   - Use explicit template parameters
   - Add more specific concepts/constraints

2. **Missing Trait Specialization**
   - Check if type needs custom specialization
   - Verify base traits are properly inherited

3. **Concept Not Satisfied**
   - Use `static_assert` to debug requirements
   - Check concept definition and type properties

4. **SFINAE Failures**
   - Ensure detection expressions are well-formed
   - Use `is_detected_v` for safe detection

## Dependencies

- C++20 compiler with concepts support
- Standard Library with `<concepts>`, `<type_traits>`
- `numeric/base/` infrastructure
- No external dependencies

## See Also

- [Numeric Base Infrastructure](../base/README.md)
- [C++20 Concepts Reference](https://en.cppreference.com/w/cpp/concepts)
- [Type Traits Reference](https://en.cppreference.com/w/cpp/header/type_traits)
- [Expression Templates Guide](../expressions/README.md)
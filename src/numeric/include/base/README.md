# Numeric Base Infrastructure

This folder contains the foundational infrastructure for the FEM numeric library. All components are designed to support IEEE-compliant numeric operations for any number-like type.

## ✅ Complete File List

### Core Infrastructure
1. **numeric_base.hpp** ✓
    - `NumberLike` and `IEEECompliant` concepts
    - `Shape` class for multi-dimensional arrays
    - `NumericBase` CRTP base class
    - `IEEEComplianceChecker` for IEEE 754 validation
    - `NumericMetadata` for type information
    - `NumericOptions` for runtime configuration
    - Error hierarchy (`NumericError`, `DimensionError`, etc.)

2. **container_base.hpp** ✓
    - `ContainerBase` template for Vector/Matrix/Tensor
    - Full STL-compatible iterator support
    - Element-wise operations with IEEE checking
    - Reduction operations (sum, product, min, max, mean)
    - Automatic NaN/Inf detection

3. **storage_base.hpp** ✓
    - `StorageBase` abstract interface
    - `DynamicStorage` (heap-allocated, growable)
    - `StaticStorage` (stack-allocated, fixed-size)
    - `AlignedStorage` (SIMD-aligned memory)
    - Support for all IEEE-compliant types

4. **expression_base.hpp** ✓
    - Expression templates for lazy evaluation
    - `BinaryExpression` and `UnaryExpression`
    - `TerminalExpression` and `ScalarExpression`
    - Broadcasting support
    - Parallel evaluation capability

5. **iterator_base.hpp** ✓
    - `ContainerIterator` for contiguous data
    - `StridedIterator` for non-contiguous views
    - `MultiDimIterator` for tensor traversal
    - `CheckedIterator` for runtime validation
    - Full random-access iterator support

6. **traits_base.hpp** ✓
    - `numeric_traits` for type properties
    - `promote_traits` for type promotion rules
    - `container_traits` for container properties
    - `storage_traits` for storage characteristics
    - `simd_traits` for vectorization info
    - Complex number specializations

7. **allocator_base.hpp** ✓
    - `AllocatorBase` interface
    - `AlignedAllocator` for SIMD operations
    - `PoolAllocator` for frequent allocations
    - `TrackingAllocator` for memory profiling
    - `StackAllocator` for temporary allocations

8. **view_base.hpp** ✓
    - `ViewBase` for non-owning references
    - `StridedView` for custom strides
    - `MultiDimView` for N-dimensional views
    - `ViewFactory` for view creation
    - Zero-copy operations

9. **slice_base.hpp** ✓
    - `Slice` object for NumPy-like slicing
    - `MultiIndex` for advanced indexing
    - Support for `All`, `NewAxis`, `Ellipsis`
    - `SliceParser` for string-based slicing
    - User-defined literals (`"1:5:2"_s`)

10. **ops_base.hpp** ✓
    - IEEE-compliant operation functors
    - Arithmetic operations (plus, minus, multiplies, divides)
    - Transcendental functions (sin, cos, exp, log)
    - Comparison operations with NaN handling
    - `OperationDispatcher` for runtime selection

11. **broadcast_base.hpp** ✓
    - `BroadcastHelper` for NumPy-style broadcasting
    - Shape compatibility checking
    - Index mapping for broadcast operations
    - `BroadcastIterator` for efficient traversal
    - `AxisReducer` for reduction operations

## 🏗️ Architecture Overview

### Design Principles
1. **IEEE Compliance**: All operations respect IEEE 754 standards
2. **Type Safety**: Compile-time concepts ensure type correctness
3. **Zero Overhead**: Template metaprogramming for compile-time optimization
4. **Flexibility**: Support for any number-like type (int, float, complex, custom)
5. **Independence**: No dependencies on FEM core classes

### Key Features

#### Type System
- **Concepts**: `NumberLike<T>` and `IEEECompliant<T>` ensure type safety
- **Promotion**: Automatic type promotion following NumPy rules
- **Complex Support**: Full support for `std::complex<T>`

#### Memory Management
- **Multiple Storage Backends**: Dynamic, static, aligned, strided
- **Custom Allocators**: Pool, stack, tracking allocators
- **Zero-Copy Views**: Efficient slicing without data duplication
- **SIMD Alignment**: Automatic alignment for vectorization

#### Expression Templates
- **Lazy Evaluation**: Operations build expression trees
- **Broadcasting**: NumPy-style shape broadcasting
- **Parallel Execution**: Automatic parallelization for large arrays
- **Memory Efficiency**: No temporary arrays created

#### Slicing & Indexing
- **NumPy-like Syntax**: Support for `[start:stop:step]` slicing
- **Advanced Indexing**: Integer arrays, boolean masks, ellipsis
- **Multi-dimensional**: Full support for N-dimensional indexing
- **String Literals**: `"1:5:2"_s` for convenient slice creation

#### Operations
- **IEEE 754 Compliant**: Proper NaN/Inf handling
- **Runtime Checking**: Optional validation of finite values
- **Vectorized**: SIMD-ready operation functors
- **Extensible**: Easy to add custom operations

## 📊 Component Dependencies

```
numeric_base.hpp (foundation - no dependencies)
    ├── traits_base.hpp
    ├── allocator_base.hpp
    ├── storage_base.hpp
    │   └── container_base.hpp
    ├── iterator_base.hpp
    ├── view_base.hpp
    ├── slice_base.hpp
    ├── ops_base.hpp
    ├── expression_base.hpp
    └── broadcast_base.hpp
```

## 🔧 Usage Examples

### Basic Container Creation
```cpp
using namespace fem::numeric;

// Will work with Vector, Matrix, Tensor once implemented
template<typename T>
class MyContainer : public ContainerBase<MyContainer<T>, T> {
    // Implementation details
};

// Supports any number-like type
MyContainer<float> f_cont;        // IEEE 754 single precision
MyContainer<double> d_cont;       // IEEE 754 double precision
MyContainer<std::complex<double>> c_cont;  // Complex numbers
MyContainer<int32_t> i_cont;      // Integers
```

### Expression Templates
```cpp
// Lazy evaluation - no temporaries created
auto expr = make_expression(A) + make_expression(B) * 2.0;

// Force evaluation when needed
Matrix<double> result;
expr.eval_to(result);
```

### Slicing
```cpp
// Create slices
auto s1 = Slice(1, 10, 2);     // 1:10:2
auto s2 = "5::-1"_s;           // 5::-1 using literal
auto s3 = Slice::all();        // Select all

// Multi-dimensional indexing
MultiIndex idx{s1, all, newaxis, Slice(0, 5)};
```

### Views
```cpp
// Create non-owning views
double data[100];
auto view = ViewBase<double>(data, 100);
auto subview = view.subview(10, 20);

// Strided views
auto strided = StridedView<double>(data, 50, 2);  // Every other element

// Multi-dimensional views
auto mdview = MultiDimView<double, 2>::from_contiguous(data, {10, 10});
```

### Broadcasting
```cpp
// Check compatibility
Shape s1{3, 1, 5};
Shape s2{1, 4, 5};
bool compatible = BroadcastHelper::are_broadcastable(s1, s2);

// Get broadcasted shape
Shape result = BroadcastHelper::broadcast_shape(s1, s2);  // {3, 4, 5}
```

### Custom Allocators
```cpp
// Aligned allocation for SIMD
using AlignedVec = MyContainer<float, AlignedStorage<float, 32>>;

// Memory pool for frequent allocations  
PoolAllocator<double> pool(1024);

// Track memory usage
using TrackedContainer = MyContainer<double, DynamicStorage<double, 
                                     TrackingAllocator<double>>>;
```

## ⚠️ Important Notes

### IEEE Compliance
- All operations handle NaN and Inf according to IEEE 754
- Optional runtime checking via `NumericOptions::check_finite`
- Complex numbers follow C++ standard (based on IEEE 754)

### Performance Considerations
- Expression templates eliminate temporaries
- SIMD alignment available via `AlignedStorage`
- Parallel execution for large arrays (OpenMP support)
- Zero-copy views for efficient slicing

### Type Safety
- Compile-time concepts ensure type compatibility
- Automatic type promotion follows NumPy rules
- Runtime bounds checking in debug mode

## 🚀 Next Steps

With this base infrastructure complete, the next phase is implementing:

1. **Vector** class (using `ContainerBase`)
2. **Matrix** class (using `ContainerBase`)
3. **Tensor** class (using `ContainerBase`)
4. **Sparse** variants of each
5. **Decomposition** algorithms
6. **Solver** implementations

All of these will build upon this solid foundation while maintaining:
- IEEE 754 compliance
- Support for any number-like type
- Independence from FEM core classes
- NumPy-like syntax and semantics

## 🧪 Testing Requirements

Each base component should be tested for:
- IEEE compliance (NaN, Inf handling)
- Type promotion correctness
- Memory safety (no leaks, proper alignment)
- Iterator correctness
- Broadcasting rules
- Performance benchmarks

## 📝 Additional Infrastructure Considerations

The current base infrastructure is comprehensive, but future additions might include:

1. **Parallel Execution Policy**
    - More sophisticated parallel execution strategies
    - GPU offloading support

2. **Serialization Base**
    - Binary/text serialization interfaces
    - Compatibility with standard formats

3. **Random Number Generation Base**
    - Interfaces for random number generators
    - Distribution support

4. **Optimization Base**
    - Interfaces for optimization algorithms
    - Gradient computation support

However, the current base infrastructure provides everything needed to build a fully functional, IEEE-compliant numeric 
library with intuitive syntax.
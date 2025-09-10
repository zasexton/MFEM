#pragma once

#ifndef NUMERIC_CONCEPTS_H
#define NUMERIC_CONCEPTS_H

#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../base/numeric_base.h"
#include "../base/container_base.h"
#include "../base/storage_base.h"
#include "../base/expression_base.h"
#include "../base/iterator_base.h"
#include "../base/broadcast_base.h"

#include "type_traits.h"
#include "numeric_traits.h"
#include "container_traits.h"
#include "expression_traits.h"
#include "operation_traits.h"
#include "storage_traits.h"
#include "iterator_traits.h"

namespace fem::numeric::concepts {

    // Import base concepts for convenience
    using fem::numeric::NumberLike;
    using fem::numeric::IEEECompliant;
    using fem::numeric::Container;
    using fem::numeric::NumericContainer;

    /**
     * @brief Concept for types that can be used in arithmetic operations
     * More restrictive than NumberLike - requires specific operations
     */
    template<typename T>
    concept Arithmetic = NumberLike<T> && requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
        { -a } -> std::convertible_to<T>;
        { +a } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for types that support comparison operations
     */
    template<typename T>
    concept Comparable = requires(T a, T b) {
        { a == b } -> std::convertible_to<bool>;
        { a != b } -> std::convertible_to<bool>;
        { a < b } -> std::convertible_to<bool>;
        { a <= b } -> std::convertible_to<bool>;
        { a > b } -> std::convertible_to<bool>;
        { a >= b } -> std::convertible_to<bool>;
    };

    /**
     * @brief Concept for types that can be ordered
     */
    template<typename T>
    concept Ordered = Comparable<T> && std::totally_ordered<T>;

    /**
     * @brief Concept for scalar numeric types (not containers)
     */
    template<typename T>
    concept Scalar = Arithmetic<T> && !Container<T> &&
                     (std::integral<T> || std::floating_point<T> ||
                      traits::is_complex_v<T>);

    /**
     * @brief Concept for field types (support all field operations)
     */
    template<typename T>
    concept Field = Arithmetic<T> && requires(T a, T b) {
        { T{0} };  // Additive identity
        { T{1} };  // Multiplicative identity
        { a + b } -> std::same_as<T>;
        { a * b } -> std::same_as<T>;
        { a - b } -> std::same_as<T>;
        { a / b } -> std::same_as<T>;  // Division (except by zero)
    };

    /**
     * @brief Stricter field concept for types that form a true mathematical field
     * Excludes integer types which don't have multiplicative inverses
     */
    template<typename T>
    concept RealField = Field<T> && std::floating_point<T>;

    template<typename T>
    concept ComplexField = Field<T> && traits::is_complex_v<T> &&
                           std::floating_point<typename T::value_type>;

    // ============================================================================
    // AUTOMATIC DIFFERENTIATION CONCEPTS
    // ============================================================================

    /**
     * @brief Basic differentiable scalar type (dual numbers, tape-based AD, etc.)
     */
    template<typename T>
    concept DifferentiableScalar = Arithmetic<T> && requires(T x) {
        typename T::value_type;     // Underlying value type
        typename T::derivative_type; // Derivative storage type

        { x.value() } -> std::convertible_to<typename T::value_type>;
        { x.derivative() } -> std::convertible_to<typename T::derivative_type>;
    };

    /**
     * @brief Forward-mode automatic differentiation
     */
    template<typename T>
    concept ForwardDifferentiable = DifferentiableScalar<T> && requires(T x, typename T::value_type v) {
        { T::variable(v) } -> std::same_as<T>;    // Create independent variable
        { T::constant(v) } -> std::same_as<T>;    // Create constant (zero derivative)
        { x.seed(v) };                             // Set derivative seed
        { x.forward() } -> std::same_as<typename T::derivative_type>; // Forward pass
    };

    /**
     * @brief Reverse-mode automatic differentiation (backpropagation)
     */
    template<typename T>
    concept ReverseDifferentiable = DifferentiableScalar<T> && requires(T x, typename T::value_type v) {
        typename T::tape_type;  // Computation tape/graph type

        { x.backward() };                          // Backward pass
        { x.grad() } -> std::convertible_to<typename T::derivative_type>;
        { x.zero_grad() };                         // Clear gradients
        { T::tape() } -> std::same_as<typename T::tape_type&>; // Access computation tape
    };

    /**
     * @brief Dual number type for forward-mode AD
     */
    template<typename T>
    concept DualNumber = ForwardDifferentiable<T> && requires(T x) {
        { x.real() } -> std::convertible_to<typename T::value_type>;      // Value part
        { x.dual() } -> std::convertible_to<typename T::derivative_type>; // Derivative part
        { T(typename T::value_type{}, typename T::derivative_type{}) };   // Constructor
    };

    /**
     * @brief Higher-order differentiation support
     */
    template<typename T>
    concept HigherOrderDifferentiable = DifferentiableScalar<T> && requires(T x, size_t n) {
        { x.derivative(n) };     // n-th derivative
        { x.hessian() };         // Second derivatives (for optimization)
        { x.taylor_coefficient(n) }; // Taylor series coefficients
    };

    /**
     * @brief Differentiable container (Vector/Matrix with AD support)
     */
    template<typename T>
    concept DifferentiableContainer = NumericContainer<T> &&
                                      DifferentiableScalar<typename T::value_type> && requires(T x) {
        { x.jacobian() };        // Jacobian matrix
        { x.gradient() };        // Gradient vector
        { x.backward() };        // Backpropagation through container
        { x.requires_grad(true) }; // Enable/disable gradient computation
    };

    // ============================================================================
    // OPERATION AND FUNCTOR CONCEPTS
    // ============================================================================

    /**
     * @brief Concept for unary operation functors
     */
    template<typename Op, typename T = void>
    concept UnaryOperation = requires(Op op, T value) {
        { op(value) } -> std::convertible_to<T>;
    } || (std::is_same_v<T, void> && requires(Op op) {
        { op(std::declval<double>()) };  // Works with at least doubles
    });

    /**
     * @brief Concept for binary operation functors
     */
    template<typename Op, typename T = void>
    concept BinaryOperation = requires(Op op, T lhs, T rhs) {
        { op(lhs, rhs) } -> std::convertible_to<T>;
    } || (std::is_same_v<T, void> && requires(Op op) {
        { op(std::declval<double>(), std::declval<double>()) };
    });

    /**
     * @brief Concept for elementwise operations
     */
    template<typename Op>
    concept ElementwiseOperation = requires {
            requires traits::is_elementwise_operation_v<Op>;
    };

    /**
     * @brief Concept for in-place operations
     */
    template<typename Op, typename T>
    concept InplaceOperation = requires(Op op, T& target, T value) {
        { op(target, value) } -> std::same_as<T&>;
    };

    /**
     * @brief Concept for types that support linear algebra operations
     */
    template<typename T>
    concept LinearAlgebraType = NumericContainer<T> && requires(T a, T b, typename T::value_type scalar) {
        // Vector space operations
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * scalar } -> std::convertible_to<T>;
        { scalar * a } -> std::convertible_to<T>;
        { a / scalar } -> std::convertible_to<T>;

        // Norm and inner product
        typename T::value_type;
        requires Arithmetic<typename T::value_type>;
    };

    /**
     * @brief Concept for matrix types
     */
    template<typename T>
    concept Matrix = LinearAlgebraType<T> && requires(T m) {
        { m.rows() } -> std::convertible_to<size_t>;
        { m.cols() } -> std::convertible_to<size_t>;
        { m.transpose() } -> std::convertible_to<T>;
        { m(0, 0) } -> std::convertible_to<typename T::value_type>;  // Element access
    };

    /**
     * @brief Concept for vector types
     */
    template<typename T>
    concept Vector = LinearAlgebraType<T> && requires(T v) {
        { v.size() } -> std::convertible_to<size_t>;
        { v[0] } -> std::convertible_to<typename T::value_type>;
        { v.norm() } -> std::convertible_to<typename T::value_type>;
    };

    /**
     * @brief Concept for norm computation
     *
     * For FEM applications, we typically need L1, L2, and Frobenius norms.
     * This concept can be satisfied by both vectors and matrices.
     */
    template<typename T>
    concept NormComputable = NumericContainer<T> && requires(T container) {
        { container.norm() } -> std::convertible_to<typename T::value_type>;      // Default (L2) norm
        { container.l1_norm() } -> std::convertible_to<typename T::value_type>;   // L1 norm
        { container.l2_norm() } -> std::convertible_to<typename T::value_type>;   // L2 norm
        { container.inf_norm() } -> std::convertible_to<typename T::value_type>;  // L-infinity norm
    };

    /**
     * @brief Extended norm concept for matrices
     */
    template<typename T>
    concept MatrixNormComputable = Matrix<T> && NormComputable<T> && requires(T matrix) {
        { matrix.frobenius_norm() } -> std::convertible_to<typename T::value_type>;  // Frobenius norm
        { matrix.spectral_norm() } -> std::convertible_to<typename T::value_type>;   // Spectral norm (optional)
    };

    /**
     * @brief Concept for tensor types
     */
    template<typename T>
    concept Tensor = NumericContainer<T> && requires(T t) {
        { t.shape() } -> std::convertible_to<Shape>;
        { t.ndim() } -> std::convertible_to<size_t>;
        { t.reshape(std::declval<Shape>()) } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for expression templates
     */
    template<typename T>
    concept Expression = traits::is_expression_v<T> && requires(T expr) {
        typename T::value_type;
        { expr.shape() } -> std::convertible_to<Shape>;
        { expr.eval() };  // Can be evaluated
    };

    /**
     * @brief Concept for lazy-evaluable expressions
     */
    template<typename T>
    concept LazyExpression = Expression<T> && requires(T expr) {
        { expr.is_lazy() } -> std::convertible_to<bool>;
        { expr.eval_to(std::declval<typename T::result_type&>()) };
    };

    /**
     * @brief Concept for storage types
     */
    template<typename T>
    concept Storage = traits::is_storage_v<T> && requires(T storage) {
        typename T::value_type;
        { storage.data() } -> std::convertible_to<typename T::value_type*>;
        { storage.size() } -> std::convertible_to<size_t>;
    };

    /**
     * @brief Concept for contiguous storage
     */
    template<typename T>
    concept ContiguousStorage = Storage<T> &&
                                traits::storage_properties<T>::is_contiguous;

    /**
     * @brief Concept for resizable storage
     */
    template<typename T>
    concept ResizableStorage = Storage<T> && requires(T storage, size_t n) {
        { storage.resize(n) };
        { storage.reserve(n) };
        { storage.capacity() } -> std::convertible_to<size_t>;
    };

    /**
     * @brief Concept for aligned storage (SIMD-ready)
     */
    template<typename T>
    concept AlignedStorage = Storage<T> &&
                             traits::alignment_traits<T>::is_simd_aligned;

    /**
     * @brief Concept for iterators that support numeric operations
     */
    template<typename T>
    concept NumericIterator = std::input_iterator<T> &&
                              NumberLike<typename std::iterator_traits<T>::value_type>;

    /**
     * @brief Concept for contiguous iterators
     */
    template<typename T>
    concept ContiguousIterator = NumericIterator<T> &&
                                 (std::contiguous_iterator<T> || traits::is_contiguous_iterator_v<T>);

    /**
     * @brief Concept for iterators that support parallel operations
     */
    template<typename T>
    concept ParallelIterator = std::random_access_iterator<T> &&
                               traits::is_parallel_safe_iterator_v<T>;

    /**
     * @brief Concept for reduction operations
     */
    template<typename Op, typename T>
    concept ReductionOperation = traits::is_reduction_operation_v<Op> &&
                                 requires(Op op, T init, T value) {
        { op(init, value) } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for operations that preserve type
     */
    template<typename Op, typename T>
    concept TypePreservingOperation =
            traits::preserves_type_v<Op, T> &&
            std::invocable<Op, T> &&
            std::same_as<std::invoke_result_t<Op, T>, T>;

    /**
     * @brief Concept for IEEE-safe operations
     */
    template<typename Op, typename T>
    concept IEEESafeOperation =
            IEEECompliant<T> &&
            traits::is_ieee_safe_v<Op, T>;

    /**
     * @brief Concept for broadcastable containers
     */
    template<typename T, typename U>
    concept Broadcastable = NumericContainer<T> && NumericContainer<U> &&
                            requires(T t, U u) {
        { BroadcastHelper::are_broadcastable(t.shape(), u.shape()) } -> std::convertible_to<bool>;
    };

    /**
     * @brief Concept for sliceable containers
     */
    template<typename T>
    concept Sliceable = Container<T> && requires(T container, Slice s) {
        { container[s] } -> std::convertible_to<T>;
        { container.slice(s) } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for containers that support views
     */
    template<typename T>
    concept ViewableContainer = Container<T> && requires(T container) {
        { container.view() };
        { container.as_strided(std::declval<Shape>(), std::declval<Shape>()) };
    };

    /**
     * @brief Concept for view types from base
     */
    template<typename T>
    concept View = requires(T view) {
        { view.data() } -> std::convertible_to<typename T::value_type*>;
        { view.size() } -> std::convertible_to<size_t>;
        { view.is_contiguous() } -> std::convertible_to<bool>;
        typename T::value_type;
    };

    /**
     * @brief Concept for allocator types from base
     */
    template<typename A>
    concept Allocator = requires(A a, typename A::value_type* p, size_t n) {
        typename A::value_type;
        typename A::size_type;
        typename A::difference_type;
        { a.allocate(n) } -> std::same_as<typename A::value_type*>;
        { a.deallocate(p, n) };
    };

    /**
     * @brief Concept for aligned allocators from base
     */
    template<typename A>
    concept AlignedAllocator = Allocator<A> && requires {
            { A::alignment } -> std::convertible_to<size_t>;
            requires A::alignment > 0;
    };

    /**
     * @brief Refined concept for decomposable matrices
     *
     * Note: Not all matrix types need to support all decompositions as member functions.
     * Consider using free functions or separate decomposition classes to reduce coupling.
     * This concept is best used for checking if a matrix provides built-in decompositions,
     * not as a constraint for algorithms that could work with external decomposition functions.
     */
    template<typename T>
    concept Decomposable = Matrix<T> && requires(T matrix) {
        { matrix.lu() };     // LU decomposition
        { matrix.qr() };     // QR decomposition
        { matrix.svd() };    // Singular value decomposition
    };

    /**
    * @brief Alternative: Check for specific decomposition support
    */
    template<typename T>
    concept LUDecomposable = Matrix<T> && requires(T matrix) {
        { matrix.lu() };
    };

    template<typename T>
    concept QRDecomposable = Matrix<T> && requires(T matrix) {
        { matrix.qr() };
    };

    template<typename T>
    concept SVDDecomposable = Matrix<T> && requires(T matrix) {
        { matrix.svd() };
    };

    /**
     * @brief Concept for types that support solver operations
     * Fixed: Using template parameter instead of auto for Vector type
     */
    template<typename T>
    concept Solvable = Matrix<T> && requires(T A) {
        typename T::value_type;
        requires requires(T A, fem::numeric::Vector<typename T::value_type> b) {
            { A.solve(b) };      // Solve Ax = b
        };
        { A.inverse() };     // Matrix inverse
        { A.determinant() }; // Determinant
    };

    /**
     * @brief Separate concepts for a matrix that has a solve method
     *        for linear algebra systems
     * Fixed: Using template parameter instead of auto
     */
    template<typename T>
    concept LinearSystemSolvable = Matrix<T> && requires(T A) {
        typename T::value_type;
        requires requires(T A, fem::numeric::Vector<typename T::value_type> b) {
            { A.solve(b) };
        };
    };

    /**
     * @brief Separate concept for a matrix object that is
     *        invertible
     */
    template<typename T>
    concept Invertible = Matrix<T> && requires(T A) {
        { A.inverse() };
        { A.determinant() };
    };

    /**
     * @brief Concept for sparse matrix types
     */
    template<typename T>
    concept SparseMatrix = Matrix<T> && requires(T sparse) {
        { sparse.nnz() } -> std::convertible_to<size_t>;  // Non-zero count
        { sparse.sparsity() } -> std::convertible_to<double>;  // Sparsity ratio
    };

    /**
     * @brief Concept for types that support in-place operations
     */
    template<typename T, typename Op>
    concept SupportsInPlace = requires(T t, typename T::value_type v) {
        { t += t } -> std::same_as<T&>;
        { t -= t } -> std::same_as<T&>;
        { t *= v } -> std::same_as<T&>;
        { t /= v } -> std::same_as<T&>;
    };

    /**
     * @brief Concept for types that can be serialized
     */
    template<typename T>
    concept Serializable = requires(T t, std::ostream& os, std::istream& is) {
        { os << t } -> std::same_as<std::ostream&>;
        { is >> t } -> std::same_as<std::istream&>;
        { t.to_bytes() };
        { T::from_bytes(std::declval<std::span<const std::byte>>()) };
    };

    /**
    * @brief Concept for block matrix operations
    */
    template<typename T>
    concept BlockMatrix = Matrix<T> && requires(T matrix) {
        { matrix.block(0, 0, 2, 2) } -> std::convertible_to<T>;
        { matrix.set_block(0, 0, std::declval<T>()) };
        { matrix.diagonal_blocks() };
    };

    /**
     * @brief Concept for banded matrices
     */
    template<typename T>
    concept BandedMatrix = SparseMatrix<T> && requires(T matrix) {
        { matrix.bandwidth() } -> std::convertible_to<std::pair<size_t, size_t>>;
        { matrix.diagonal(0) };  // Main diagonal
        { matrix.diagonal(1) };  // Super-diagonal
        { matrix.diagonal(-1) }; // Sub-diagonal
    };

    /**
     * @brief Concept for symmetric/Hermitian matrices
     */
    template<typename T>
    concept SymmetricMatrix = Matrix<T> && requires(T matrix) {
        { matrix.is_symmetric() } -> std::convertible_to<bool>;
        { matrix.symmetrize() } -> std::same_as<T>;
    };

    /**
     * @brief Concept for triangular matrices
     */
    template<typename T>
    concept TriangularMatrix = Matrix<T> && requires(T matrix) {
        { matrix.is_upper_triangular() } -> std::convertible_to<bool>;
        { matrix.is_lower_triangular() } -> std::convertible_to<bool>;
        { matrix.triu() } -> std::same_as<T>;  // Upper triangular part
        { matrix.tril() } -> std::same_as<T>;  // Lower triangular part
    };

    /**
     * @brief Concept for sorting and searching
     */
    template<typename T>
    concept Sortable = Container<T> && requires(T container) {
        { container.sort() };
        { container.argsort() } -> std::convertible_to<std::vector<size_t>>;
    };

    /**
    * @brief Concept for einsum operations on tensors
    */
    template<typename T>
    concept EinsumCapable = Tensor<T> && requires(T tensor) {
        { T::einsum("ij,jk->ik", tensor, tensor) } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for tensor contraction
     */
    template<typename T>
    concept TensorContractable = Tensor<T> && requires(T t1, T t2) {
        { t1.contract(t2, std::vector<size_t>{0}, std::vector<size_t>{1}) } -> std::convertible_to<T>;
    };

    /**
     * @brief Concept for range types that work with numeric algorithms
     */
    template<typename R>
    concept NumericRange = std::ranges::range<R> &&
                           NumberLike<std::ranges::range_value_t<R>>;

    /**
     * @brief Concept for ranges that can be processed in parallel
     */
    template<typename R>
    concept ParallelRange = NumericRange<R> &&
                            std::ranges::random_access_range<R> &&
                            std::ranges::sized_range<R>;


    /**
     * @brief Concept for GPU-compatible types
     */
    template<typename T>
    concept DeviceCompatible = std::is_trivially_copyable_v<T> &&
                               std::is_standard_layout_v<T> &&
                               requires(T t) {
        { t.to_device() };
        { t.to_host() };
        { T::device_type } -> std::convertible_to<Device>;
    };

    /**
     * @brief Concept combining multiple requirements for high-performance computing
     */
    template<typename T>
    concept HPCCompatible =
            NumericContainer<T> &&
            ContiguousStorage<typename T::storage_type> &&
            ParallelRange<T> &&
            DeviceCompatible<T> &&
            Serializable<T>;


    /**
     * @brief Validate that basic types work with our concepts
     * Note: Some static assertions are commented out because they depend on
     * NumberLike definition which may not support complex comparison operators
     */
    static_assert(NumberLike<double>);
    static_assert(NumberLike<float>);
    static_assert(NumberLike<int>);
    // static_assert(NumberLike<std::complex<double>>);  // May fail due to comparison operators
    static_assert(IEEECompliant<double>);
    static_assert(IEEECompliant<float>);
    static_assert(Field<double>);
    // static_assert(Field<std::complex<double>>);  // May fail due to NumberLike requirements

} // namespace fem::numeric::concepts

#endif //NUMERIC_CONCEPTS_H
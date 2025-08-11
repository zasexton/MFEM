#pragma once

#ifndef NUMERIC_CONTAINER_TRAITS_H
#define NUMERIC_CONTAINER_TRAITS_H

#include <iterator>
#include <type_traits>

#include "../base/container_base.h"
#include "../base/storage_base.h"
#include "../base/view_base.h"
#include "../base/expression_base.h"

#include "type_traits.h"
#include "numeric_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Extended container traits building on base infrastructure
     *
     * This complements the container_traits in traits_base.hpp with
     * additional compile-time information and detection capabilities
     */

    /**
     * @brief Container categories for dispatch (extends base categories)
     */
    enum class ContainerCategory {
        Dense,          // Contiguous memory (vector, array)
        Sparse,         // Sparse storage (CSR, COO, etc.)
        View,           // Non-owning view of data
        Expression,     // Lazy-evaluated expression
        Strided,        // Strided/sliced view
        Hybrid,         // Small-buffer optimized
        Unknown
    };

    /**
     * @brief Memory layout patterns
     */
    enum class MemoryLayout {
        RowMajor,       // C-style [row][col]
        ColumnMajor,    // Fortran-style [col][row]
        Strided,        // Custom strides
        Sparse,         // Non-contiguous sparse
        Unknown
    };

    /**
     * @brief Detect if type is derived from ContainerBase
     */
    template<typename T>
    struct is_container_base {
    private:
        template<typename D, typename V, typename S>
        static std::true_type test(const ContainerBase<D, V, S>*);
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    };

    template<typename T>
    inline constexpr bool is_container_base_v = is_container_base<T>::value;

    /**
     * @brief Detect if type is derived from ExpressionBase
     */
    template<typename T>
    struct is_expression_base {
    private:
        template<typename E>
        static std::true_type test(const ExpressionBase<E>*);
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    };

    template<typename T>
    inline constexpr bool is_expression_base_v = is_expression_base<T>::value;

    /**
     * @brief Detect if type is derived from ViewBase
     */
    template<typename T>
    struct is_view_base {
    private:
        template<typename V>
        static std::true_type test(const ViewBase<V>*);
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    };

    template<typename T>
    inline constexpr bool is_view_base_v = is_view_base<T>::value;

    /**
     * @brief Extended container properties
     * Builds upon the basic container_traits in traits_base.hpp
     */
    template<typename T>
    struct extended_container_traits {
        // Import base traits
        using base_traits = container_traits<T>;
        using value_type = typename base_traits::value_type;
        using size_type = typename base_traits::size_type;

        // Category classification
        static constexpr ContainerCategory category = [] {
            if constexpr (is_expression_base_v<T>) {
            return ContainerCategory::Expression;
        } else if constexpr (is_view_base_v<T>) {
            return ContainerCategory::View;
        } else if constexpr (is_container_base_v<T>) {
            return ContainerCategory::Dense;
        } else {
            return ContainerCategory::Unknown;
        }
        }();

        // Check if satisfies our Container concept
        static constexpr bool satisfies_container_concept = Container<T>;
        static constexpr bool satisfies_numeric_container = NumericContainer<T>;

        // Storage properties (if applicable)
        using storage_type = std::conditional_t<
                is_container_base_v<T>,
                typename T::storage_type,
                void
        >;

        // Iterator properties
        using iterator_type = std::conditional_t<
                requires { typename T::iterator; },
                typename T::iterator,
                void*
        >;

        using iterator_category = std::conditional_t<
                !std::is_same_v<iterator_type, void*>,
        typename std::iterator_traits<iterator_type>::iterator_category,
        void
        >;

        static constexpr bool has_random_access = std::is_same_v<
                iterator_category,
                std::random_access_iterator_tag
        >;

        // Dimension information
        static constexpr size_t static_dimensions = [] {
            if constexpr (requires { T::static_dimensions; }) {
            return T::static_dimensions;
        } else {
            return 0;  // Dynamic dimensions
        }
        }();

        static constexpr bool has_static_shape = static_dimensions > 0;

        // Memory layout
        static constexpr MemoryLayout layout = [] {
            if constexpr (requires { T::memory_layout; }) {
            return T::memory_layout;
        } else if constexpr (is_view_base_v<T>) {
            return MemoryLayout::Strided;
        } else if constexpr (is_container_base_v<T>) {
            return MemoryLayout::RowMajor;  // Default assumption
        } else {
            return MemoryLayout::Unknown;
        }
        }();

        // Performance hints
        static constexpr bool is_contiguous =
                (layout == MemoryLayout::RowMajor ||
                 layout == MemoryLayout::ColumnMajor) &&
                !is_view_base_v<T>;

        static constexpr bool supports_simd =
                is_contiguous &&
                is_numeric_type_v<value_type> &&
                alignof(T) >= 16;  // At least SSE alignment

        static constexpr bool is_lazy = is_expression_base_v<T>;

        static constexpr bool is_owning =
                is_container_base_v<T> && !is_view_base_v<T>;
    };

    /**
     * @brief Check if container supports arithmetic operations
     * Uses the operations defined in ops_base.hpp
     */
    template<typename T>
    struct supports_arithmetic_ops {
        static constexpr bool value = requires(T t) {
            { t += t } -> std::same_as<T&>;
            { t -= t } -> std::same_as<T&>;
            { t *= typename T::value_type{} } -> std::same_as<T&>;
            { t /= typename T::value_type{} } -> std::same_as<T&>;
        };
    };

    template<typename T>
    inline constexpr bool supports_arithmetic_ops_v = supports_arithmetic_ops<T>::value;

    /**
     * @brief Check if two containers are compatible for operations
     * Uses Shape class from numeric_base.hpp
     */
    template<typename C1, typename C2>
    struct are_containers_compatible {
        static constexpr bool value =
                Container<C1> && Container<C2> &&
                std::is_convertible_v<
                        typename C1::value_type,
                        typename promote_type<
                                typename C1::value_type,
                                typename C2::value_type
                        >::type
                >;
    };

    template<typename C1, typename C2>
    inline constexpr bool are_containers_compatible_v =
            are_containers_compatible<C1, C2>::value;

    /**
     * @brief Determine optimal evaluation strategy
     */
    template<typename Expr>
    struct evaluation_strategy {
        enum Strategy {
            Immediate,      // Evaluate immediately
            Lazy,          // Keep as expression
            Parallel,      // Use parallel evaluation
            Vectorized     // Use SIMD
        };

        static constexpr Strategy value = [] {
            if constexpr (is_expression_base_v<Expr>) {
            // Already an expression, keep lazy
            return Lazy;
        } else if constexpr (extended_container_traits<Expr>::supports_simd) {
            return Vectorized;
        } else if constexpr (requires { Expr::use_parallel; }) {
            return Parallel;
        } else {
            return Immediate;
        }
        }();
    };

    /**
     * @brief Helper to extract shape type
     */
    template<typename Container>
    using shape_type = std::conditional_t<
            requires { typename Container::shape_type; },
            typename Container::shape_type,
            Shape  // Use Shape from numeric_base.hpp
    >;

    /**
     * @brief Get container dimensionality at compile time if possible
     */
    template<typename Container>
    struct container_rank {
        static constexpr size_t value = [] {
            if constexpr (requires { Container::rank; }) {
            return Container::rank;
        } else if constexpr (requires { Container::ndim; }) {
            return Container::ndim;
        } else if constexpr (extended_container_traits<Container>::static_dimensions > 0) {
            return extended_container_traits<Container>::static_dimensions;
        } else {
            return 0;  // Dynamic rank
        }
        }();

        static constexpr bool is_static = value > 0;
    };

    template<typename Container>
    inline constexpr size_t container_rank_v = container_rank<Container>::value;

    /**
     * @brief Storage requirements calculator
     * Builds on storage_traits from traits_base.hpp
     */
    template<typename Container>
    struct container_storage_info {
        using storage_type = typename extended_container_traits<Container>::storage_type;
        using value_type = typename Container::value_type;

        static constexpr size_t element_size = sizeof(value_type);
        static constexpr size_t alignment = [] {
            if constexpr (!std::is_void_v<storage_type>) {
            return storage_traits<storage_type>::alignment;
        } else {
            return alignof(value_type);
        }
        }();

        static constexpr bool is_dynamic = [] {
            if constexpr (!std::is_void_v<storage_type>) {
            return storage_traits<storage_type>::is_dynamic;
        } else {
            return true;
        }
        }();

        static constexpr bool is_aligned = alignment > alignof(value_type);

        // Calculate bytes needed for n elements
        static constexpr size_t bytes_for(size_t n) noexcept {
            size_t bytes = n * element_size;
            if constexpr (is_aligned) {
                // Round up to alignment boundary
                return ((bytes + alignment - 1) / alignment) * alignment;
            }
            return bytes;
        }
    };

    /**
     * @brief Helper for selecting optimal container type based on requirements
     */
    template<typename T, size_t Size = 0, bool NeedsSIMD = false>
    struct optimal_container_selector {
        using type = std::conditional_t<
        Size == 0,
        // Dynamic size - use ContainerBase with DynamicStorage
        ContainerBase<void, T, DynamicStorage<T>>,
        // Static size
        std::conditional_t<
        NeedsSIMD,
        // Need SIMD alignment
        ContainerBase<void, T, AlignedStorage<T, 32>>,
        // Small size optimization
        std::conditional_t<
                Size * sizeof(T) <= 256,
                ContainerBase<void, T, StaticStorage<T, Size>>,
        ContainerBase<void, T, DynamicStorage<T>>
        >
        >
        >;
    };

    template<typename T, size_t Size = 0, bool NeedsSIMD = false>
    using optimal_container_t = typename optimal_container_selector<T, Size, NeedsSIMD>::type;

} // namespace fem::numeric::traits

#endif //NUMERIC_CONTAINER_TRAITS_H

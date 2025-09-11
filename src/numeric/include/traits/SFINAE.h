#pragma once

#ifndef NUMERIC_SFINAE_H
#define NUMERIC_SFINAE_H

#include <type_traits>
#include <utility>
#include <complex>
#include <concepts>

#include "../base/numeric_base.h"
#include "../base/container_base.h"
#include "../base/storage_base.h"
#include "../base/traits_base.h"
#include "type_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief SFINAE helpers for compile-time detection and selection
     *
     * These utilities enable safe compile-time dispatch based on type properties
     * and available operations. They integrate with the base infrastructure
     * and provide building blocks for more complex traits.
     */

    // ============================================================================
    // Basic SFINAE utilities
    // ============================================================================

    /**
     * @brief Always false for static_assert in constexpr contexts
     */
    template<typename...>
    inline constexpr bool always_false_v = false;

    // void_t and type_identity are provided by type_traits.h

    // ============================================================================
    // Detection idiom implementation
    // ============================================================================

    namespace detail {
        template<typename Default, typename AlwaysVoid,
                template<typename...> class Op, typename... Args>
        struct detector {
            using value_t = std::false_type;
            using type = Default;
        };

        template<typename Default, template<typename...> class Op, typename... Args>
        struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };
    }

    /**
     * @brief Detection trait - checks if Op<Args...> is valid
     */
    template<template<typename...> class Op, typename... Args>
    using is_detected = typename detail::detector<void, void, Op, Args...>::value_t;

    template<template<typename...> class Op, typename... Args>
    inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

    /**
     * @brief Detected type or default
     */
    template<typename Default, template<typename...> class Op, typename... Args>
    using detected_or = detail::detector<Default, void, Op, Args...>;

    template<typename Default, template<typename...> class Op, typename... Args>
    using detected_or_t = typename detected_or<Default, Op, Args...>::type;

    /**
     * @brief Detected type
     */
    template<template<typename...> class Op, typename... Args>
    using detected_t = typename detail::detector<void, void, Op, Args...>::type;

    /**
     * @brief Check if detected type is exactly Expected
     */
    template<typename Expected, template<typename...> class Op, typename... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

    template<typename Expected, template<typename...> class Op, typename... Args>
    inline constexpr bool is_detected_exact_v = is_detected_exact<Expected, Op, Args...>::value;

    /**
     * @brief Check if detected type is convertible to To
     */
    template<typename To, template<typename...> class Op, typename... Args>
    using is_detected_convertible = std::is_convertible<detected_t<Op, Args...>, To>;

    template<typename To, template<typename...> class Op, typename... Args>
    inline constexpr bool is_detected_convertible_v = is_detected_convertible<To, Op, Args...>::value;

    // ============================================================================
    // Operation detection helpers
    // ============================================================================

    /**
     * @brief Helper to detect binary operations
     */
    template<typename T, typename U, typename Op>
    using binary_op_result_t = decltype(std::declval<Op>()(std::declval<T>(), std::declval<U>()));

    template<typename T, typename U, typename Op>
    inline constexpr bool has_binary_op_v = is_detected_v<binary_op_result_t, T, U, Op>;

    /**
     * @brief Helper to detect unary operations
     */
    template<typename T, typename Op>
    using unary_op_result_t = decltype(std::declval<Op>()(std::declval<T>()));

    template<typename T, typename Op>
    inline constexpr bool has_unary_op_v = is_detected_v<unary_op_result_t, T, Op>;

    /**
     * @brief Detect if type has member function
     */
    template<typename T>
    using has_size_t = decltype(std::declval<T>().size());

    template<typename T>
    using has_data_t = decltype(std::declval<T>().data());

    template<typename T>
    using has_begin_t = decltype(std::declval<T>().begin());

    template<typename T>
    using has_end_t = decltype(std::declval<T>().end());

    template<typename T>
    inline constexpr bool has_size_v = is_detected_v<has_size_t, T>;

    template<typename T>
    inline constexpr bool has_data_v = is_detected_v<has_data_t, T>;

    template<typename T>
    inline constexpr bool has_begin_v = is_detected_v<has_begin_t, T>;

    template<typename T>
    inline constexpr bool has_end_v = is_detected_v<has_end_t, T>;

    // ============================================================================
    // Container and storage detection
    // ============================================================================

    /**
     * @brief Detect if type is derived from ContainerBase
     */
    template<typename T>
    using detect_container_base_helper = std::enable_if_t<
            std::is_base_of_v<ContainerBase<typename T::value_type, T>, T>
    >;

    template<typename T>
    inline constexpr bool detect_container_base_v =
            is_detected_v<detect_container_base_helper, T>;

    /**
     * @brief Detect if type has storage_type member
     */
    template<typename T>
    using has_storage_type_t = typename T::storage_type;

    template<typename T>
    inline constexpr bool has_storage_type_v = is_detected_v<has_storage_type_t, T>;

    /**
     * @brief Detect if type is a storage type
     */
    template<typename T>
    using detect_storage_helper = std::enable_if_t<
            std::is_base_of_v<StorageBase<typename T::value_type>, T>
    >;

    template<typename T>
    inline constexpr bool detect_storage_v =
            is_detected_v<detect_storage_helper, T>;

    // ============================================================================
    // Expression template detection
    // ============================================================================

    /**
     * @brief Detect if type is an expression
     */
    template<typename T>
    using detect_expression_base_helper = std::enable_if_t<
            std::is_base_of_v<ExpressionBase<T>, T>
    >;

    template<typename T>
    inline constexpr bool detect_expression_base_v =
            is_detected_v<detect_expression_base_helper, T>;

    /**
     * @brief Detect if type has eval() method
     */
    template<typename T>
    using has_eval_t = decltype(std::declval<T>().eval());

    template<typename T>
    inline constexpr bool has_eval_v = is_detected_v<has_eval_t, T>;

    // ============================================================================
    // View detection
    // ============================================================================

    /**
     * @brief Detect if type is a view
     */
    template<typename T>
    using detect_view_base_helper = std::enable_if_t<
            std::is_base_of_v<ViewBase<typename T::value_type>, T>
    >;

    template<typename T>
    inline constexpr bool detect_view_base_v =
            is_detected_v<detect_view_base_helper, T>;

    /**
     * @brief Detect if type has is_view() method
     */
    template<typename T>
    using has_is_view_t = decltype(std::declval<T>().is_view());

    template<typename T>
    inline constexpr bool has_is_view_v = is_detected_v<has_is_view_t, T>;

    // ============================================================================
    // Shape and dimension detection
    // ============================================================================

    /**
     * @brief Detect if type has shape() method
     */
    template<typename T>
    using has_shape_t = decltype(std::declval<T>().shape());

    template<typename T>
    inline constexpr bool has_shape_v = is_detected_v<has_shape_t, T>;

    /**
     * @brief Detect if type has dimensions() method
     */
    template<typename T>
    using has_dimensions_t = decltype(std::declval<T>().dimensions());

    template<typename T>
    inline constexpr bool has_dimensions_v = is_detected_v<has_dimensions_t, T>;

    // ============================================================================
    // Iterator detection
    // ============================================================================

    /**
     * @brief Detect iterator category
     */
    template<typename T>
    using iterator_category_t = typename std::iterator_traits<T>::iterator_category;

    template<typename T>
    inline constexpr bool detect_iterator_v =
            is_detected_v<iterator_category_t, T>;

    template<typename T>
    inline constexpr bool detect_random_access_iterator_v =
            detect_iterator_v<T> &&
            std::is_base_of_v<std::random_access_iterator_tag,
                    detected_t<iterator_category_t, T>>;

    // ============================================================================
    // Type selection helpers
    // ============================================================================

    /**
     * @brief Conditional type selection with SFINAE
     */
    template<bool Condition, typename T = void>
    using enable_if_t = std::enable_if_t<Condition, T>;

    /**
     * @brief Select first valid type from a list
     */
    template<typename Default, typename... Types>
    struct first_valid_type {
        using type = Default;
    };

    template<typename Default, typename First, typename... Rest>
    struct first_valid_type<Default, First, Rest...> {
        using type = std::conditional_t<
                !std::is_same_v<First, void>,
                First,
                typename first_valid_type<Default, Rest...>::type
        >;
    };

    template<typename Default, typename... Types>
    using first_valid_type_t = typename first_valid_type<Default, Types...>::type;

    // ============================================================================
    // Concept emulation for pre-C++20
    // ============================================================================

    /**
     * @brief Helper for requires-expression emulation
     */
    template<typename... Args>
    struct requires_impl {
        template<typename... Ts>
        static constexpr bool check() {
            return (... && Args::template value<Ts...>);
        }
    };

    /**
     * @brief Check if all conditions are satisfied
     */
    template<bool... Bs>
    inline constexpr bool conjunction_v = (... && Bs);

    template<bool... Bs>
    inline constexpr bool disjunction_v = (... || Bs);

    // ============================================================================
    // Common type computation with numeric types
    // ============================================================================

    /**
     * @brief Common type considering numeric promotion
     */
    template<typename... Types>
    struct numeric_common_type {
        using type = std::common_type_t<Types...>;
    };

    // Specialization for complex types
    template<typename T, typename U>
    struct numeric_common_type<std::complex<T>, U> {
        using type = std::complex<typename promote_traits<T, U>::type>;
    };

    template<typename T, typename U>
    struct numeric_common_type<T, std::complex<U>> {
        using type = std::complex<typename promote_traits<T, U>::type>;
    };

    template<typename T, typename U>
    struct numeric_common_type<std::complex<T>, std::complex<U>> {
        using type = std::complex<typename promote_traits<T, U>::type>;
    };

    template<typename... Types>
    using numeric_common_type_t = typename numeric_common_type<Types...>::type;

    // ============================================================================
    // Member type detection
    // ============================================================================

    /**
     * @brief Macro to generate member type detection
     */
#define DEFINE_HAS_MEMBER_TYPE(member_name) \
        template<typename T> \
        using has_##member_name##_t = typename T::member_name; \
        template<typename T> \
        inline constexpr bool has_##member_name##_v = is_detected_v<has_##member_name##_t, T>;

    // Common member types
    DEFINE_HAS_MEMBER_TYPE(value_type)
    DEFINE_HAS_MEMBER_TYPE(element_type)
    DEFINE_HAS_MEMBER_TYPE(allocator_type)
    DEFINE_HAS_MEMBER_TYPE(iterator)
    DEFINE_HAS_MEMBER_TYPE(const_iterator)
    DEFINE_HAS_MEMBER_TYPE(size_type)
    DEFINE_HAS_MEMBER_TYPE(difference_type)

#undef DEFINE_HAS_MEMBER_TYPE

    // ============================================================================
    // Specialization helpers
    // ============================================================================

    /**
     * @brief Helper to detect if type is a specialization of a template
     */
    template<typename T, template<typename...> class Template>
    struct is_specialization : std::false_type {};

    template<template<typename...> class Template, typename... Args>
    struct is_specialization<Template<Args...>, Template> : std::true_type {};

    template<typename T, template<typename...> class Template>
    inline constexpr bool is_specialization_v = is_specialization<T, Template>::value;

    // ============================================================================
    // Validation helpers
    // ============================================================================

    /**
     * @brief Validate that type satisfies numeric requirements
     */
    template<typename T>
    struct validate_numeric_type {
        static constexpr bool is_valid = NumberLike<T>;
        static constexpr bool is_ieee = IEEECompliant<T>;

        static_assert(is_valid || always_false_v<T>,
        "Type must satisfy NumberLike concept");
    };

    /**
     * @brief Validate container type
     */
    template<typename T>
    struct validate_container_type {
        static constexpr bool has_required_members =
                has_value_type_v<T> &&
                has_size_v<T> &&
                has_data_v<T>;

        static constexpr bool has_iterators =
                has_begin_v<T> &&
                has_end_v<T>;

        static constexpr bool is_valid =
                has_required_members && has_iterators;
    };

} // namespace fem::numeric::traits

#endif //NUMERIC_SFINAE_H

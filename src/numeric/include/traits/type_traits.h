#pragma once

#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include <type_traits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "../base/numeric_base.h"
#include "../base/traits_base.h"

namespace fem::numeric::traits {

    /**
     * @brief Type categories for numeric types
     */
    enum class TypeCategory {
        Integral,
        FloatingPoint,
        Complex,
        Unknown
    };

    /**
     * @brief Primary template for type categorization
     */
    template<typename T>
    struct type_category {
        static constexpr TypeCategory value = TypeCategory::Unknown;
    };

    // Specializations for integral types
    template<> struct type_category<bool> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<char> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<signed char> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<unsigned char> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<short> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<unsigned short> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<int> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<unsigned int> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<long> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<unsigned long> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<long long> { static constexpr TypeCategory value = TypeCategory::Integral; };
    template<> struct type_category<unsigned long long> { static constexpr TypeCategory value = TypeCategory::Integral; };

    // Specializations for floating-point types
    template<> struct type_category<float> { static constexpr TypeCategory value = TypeCategory::FloatingPoint; };
    template<> struct type_category<double> { static constexpr TypeCategory value = TypeCategory::FloatingPoint; };
    template<> struct type_category<long double> { static constexpr TypeCategory value = TypeCategory::FloatingPoint; };

    // Specializations for complex types
    template<typename T>
    struct type_category<std::complex<T>> {
        static constexpr TypeCategory value = TypeCategory::Complex;
    };

    /**
     * @brief Helper variable template for type category
     */
    template<typename T>
    inline constexpr TypeCategory type_category_v = type_category<std::remove_cv_t<T>>::value;

    /**
     * @brief Check if type is numeric using NumberLike concept from base
     */
    template<typename T>
    struct is_numeric : std::bool_constant<NumberLike<T>> {};

    template<typename T>
    inline constexpr bool is_numeric_v = is_numeric<T>::value;

    /**
     * @brief Check if type is IEEE compliant using concept from base
     */
    template<typename T>
    struct is_ieee_compliant : std::bool_constant<IEEECompliant<T>> {};

    template<typename T>
    inline constexpr bool is_ieee_compliant_v = is_ieee_compliant<T>::value;

    /**
     * @brief Check if type is complex
     * Alias to base is_complex_number
     */
    template<typename T>
    using is_complex = fem::numeric::is_complex_number<T>;

    template<typename T>
    inline constexpr bool is_complex_v = fem::numeric::is_complex_number_v<T>;

    /**
     * @brief Get the underlying real type from a potentially complex type
     * Alias to base scalar_type
     */
    template<typename T>
    using real_type = fem::numeric::scalar_type<T>;

    template<typename T>
    using real_type_t = fem::numeric::scalar_type_t<T>;

    /**
     * @brief Get the complex type for a given real type
     */
    template<typename T>
    struct complex_type {
        using type = std::complex<T>;
    };

    template<typename T>
    struct complex_type<std::complex<T>> {
    using type = std::complex<T>;
    };

    template<typename T>
    using complex_type_t = typename complex_type<T>::type;

    /**
     * @brief Check if type is a signed type
     */
    template<typename T>
    struct is_signed : std::is_signed<T> {};

    template<typename T>
    struct is_signed<std::complex<T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_signed_v = is_signed<T>::value;

    /**
     * @brief Check if type is an unsigned type
     */
    template<typename T>
    struct is_unsigned : std::is_unsigned<T> {};

    template<typename T>
    struct is_unsigned<std::complex<T>> : std::false_type {};

    template<typename T>
    inline constexpr bool is_unsigned_v = is_unsigned<T>::value;

    /**
     * @brief Remove const, volatile, and reference qualifiers
     */
    template<typename T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    /**
     * @brief Check if two types are the same after removing qualifiers
     */
    template<typename T, typename U>
    inline constexpr bool is_same_decay_v = std::is_same_v<
                                            std::decay_t<T>, std::decay_t<U>
    >;

    /**
     * @brief Get the size in bytes of a type
     */
    template<typename T>
    struct byte_size {
        static constexpr size_t value = sizeof(T);
    };

    template<typename T>
    inline constexpr size_t byte_size_v = byte_size<T>::value;

    /**
     * @brief Get the alignment requirement of a type
     */
    template<typename T>
    struct alignment_of {
        static constexpr size_t value = alignof(T);
    };

    template<typename T>
    inline constexpr size_t alignment_of_v = alignment_of<T>::value;

    /**
     * @brief Check if type has standard layout
     */
    template<typename T>
    inline constexpr bool has_standard_layout_v = std::is_standard_layout_v<T>;

    /**
     * @brief Check if type is trivially copyable
     */
    template<typename T>
    inline constexpr bool is_trivially_copyable_v = std::is_trivially_copyable_v<T>;

    /**
     * @brief Check if type is POD (Plain Old Data)
     */
    template<typename T>
    struct is_pod {
        static constexpr bool value = std::is_standard_layout_v<T> &&
                                      std::is_trivial_v<T>;
    };

    template<typename T>
    inline constexpr bool is_pod_v = is_pod<T>::value;

    /**
     * @brief Type identity - useful for preventing type deduction
     */
    template<typename T>
    struct type_identity {
        using type = T;
    };

    template<typename T>
    using type_identity_t = typename type_identity<T>::type;

    /**
     * @brief Conditional type selection
     */
    template<bool Condition, typename TrueType, typename FalseType>
    using conditional_t = std::conditional_t<Condition, TrueType, FalseType>;

    /**
     * @brief Enable if type is numeric (uses NumberLike from base)
     */
    template<typename T, typename Enable = void>
    struct enable_if_numeric {};

    template<typename T>
    struct enable_if_numeric<T, std::enable_if_t<NumberLike<T>>> {
    using type = T;
    };

    template<typename T>
    using enable_if_numeric_t = typename enable_if_numeric<T>::type;

    /**
     * @brief Enable if type is integral
     */
    template<typename T>
    using enable_if_integral_t = std::enable_if_t<std::is_integral_v<T>, T>;

    /**
     * @brief Enable if type is floating point
     */
    template<typename T>
    using enable_if_floating_t = std::enable_if_t<std::is_floating_point_v<T>, T>;

    /**
     * @brief Enable if type is complex
     */
    template<typename T>
    using enable_if_complex_t = std::enable_if_t<is_complex_v<T>, T>;

    /**
     * @brief Enable if type is IEEE compliant (uses concept from base)
     */
    template<typename T>
    using enable_if_ieee_t = std::enable_if_t<IEEECompliant<T>, T>;

    /**
     * @brief Common type among multiple types
     */
    template<typename... Types>
    using common_type_t = std::common_type_t<Types...>;

    /**
     * @brief Check if type is any of the given types
     */
    template<typename T, typename... Types>
    struct is_any_of : std::false_type {};

    template<typename T, typename First, typename... Rest>
    struct is_any_of<T, First, Rest...> : std::conditional_t<
            std::is_same_v<T, First>,
                                          std::true_type,
                                          is_any_of<T, Rest...>
    > {};

    template<typename T, typename... Types>
    inline constexpr bool is_any_of_v = is_any_of<T, Types...>::value;

    /**
     * @brief Check if all types are the same
     */
    template<typename T, typename... Types>
    struct are_same : std::true_type {};

    template<typename T, typename U, typename... Rest>
    struct are_same<T, U, Rest...> : std::conditional_t<
            std::is_same_v<T, U>,
                                     are_same<U, Rest...>,
                                     std::false_type
    > {};

    template<typename... Types>
    inline constexpr bool are_same_v = are_same<Types...>::value;

    /**
     * @brief Check if type is a valid index type
     */
    template<typename T>
    struct is_index_type : std::false_type {};

    template<typename T>
    requires (std::is_same_v<T, size_t> ||
              std::is_same_v<T, int> ||
              std::is_same_v<T, long> ||
              std::is_same_v<T, long long> ||
              std::is_same_v<T, ptrdiff_t>)
    struct is_index_type<T> : std::true_type {};

    template<typename T>
    inline constexpr bool is_index_type_v = is_index_type<T>::value;

    /**
     * @brief Get the scalar type from a potentially complex type
     * Alias to base scalar_type
     */
    template<typename T>
    using scalar_type = fem::numeric::scalar_type<T>;

    template<typename T>
    using scalar_type_t = fem::numeric::scalar_type_t<T>;

    /**
     * @brief Check if type supports arithmetic operations
     */
    template<typename T, typename = void>
    struct has_arithmetic_ops : std::false_type {};

    template<typename T>
    struct has_arithmetic_ops<T, std::void_t<
                                 decltype(std::declval<T>() + std::declval<T>()),
            decltype(std::declval<T>() - std::declval<T>()),
            decltype(std::declval<T>() * std::declval<T>()),
            decltype(std::declval<T>() / std::declval<T>())
    >> : std::true_type {};

    template<typename T>
    inline constexpr bool has_arithmetic_ops_v = has_arithmetic_ops<T>::value;

    /**
     * @brief Check if type supports comparison operations
     */
    template<typename T, typename = void>
    struct has_comparison_ops : std::false_type {};

    template<typename T>
    struct has_comparison_ops<T, std::void_t<
                                 decltype(std::declval<T>() == std::declval<T>()),
            decltype(std::declval<T>() != std::declval<T>()),
            decltype(std::declval<T>() < std::declval<T>()),
            decltype(std::declval<T>() <= std::declval<T>()),
            decltype(std::declval<T>() > std::declval<T>()),
            decltype(std::declval<T>() >= std::declval<T>())
    >> : std::true_type {};

    template<typename T>
    inline constexpr bool has_comparison_ops_v = has_comparison_ops<T>::value;

    /**
     * @brief Check if type is default constructible
     */
    template<typename T>
    inline constexpr bool is_default_constructible_v = std::is_default_constructible_v<T>;

    /**
     * @brief Check if type is copy constructible
     */
    template<typename T>
    inline constexpr bool is_copy_constructible_v = std::is_copy_constructible_v<T>;

    /**
     * @brief Check if type is move constructible
     */
    template<typename T>
    inline constexpr bool is_move_constructible_v = std::is_move_constructible_v<T>;

    /**
     * @brief Check if type is assignable
     */
    template<typename T>
    inline constexpr bool is_copy_assignable_v = std::is_copy_assignable_v<T>;

    template<typename T>
    inline constexpr bool is_move_assignable_v = std::is_move_assignable_v<T>;

    /**
     * @brief Get the rank (number of dimensions) of an array type
     */
    template<typename T>
    struct rank : std::rank<T> {};

    template<typename T>
    inline constexpr size_t rank_v = rank<T>::value;

    /**
     * @brief Get the extent (size) of an array dimension
     */
    template<typename T, size_t N = 0>
    struct extent : std::extent<T, N> {};

    template<typename T, size_t N = 0>
    inline constexpr size_t extent_v = extent<T, N>::value;

    /**
     * @brief Remove all array extents from a type
     */
    template<typename T>
    using remove_all_extents_t = std::remove_all_extents_t<T>;

    /**
     * @brief Check if type is an array
     */
    template<typename T>
    inline constexpr bool is_array_v = std::is_array_v<T>;

    /**
     * @brief Check if type is a pointer
     */
    template<typename T>
    inline constexpr bool is_pointer_v = std::is_pointer_v<T>;

    /**
     * @brief Remove pointer from type
     */
    template<typename T>
    using remove_pointer_t = std::remove_pointer_t<T>;

    /**
     * @brief Add pointer to type
     */
    template<typename T>
    using add_pointer_t = std::add_pointer_t<T>;

    /**
     * @brief Check if type is a reference
     */
    template<typename T>
    inline constexpr bool is_reference_v = std::is_reference_v<T>;

    /**
     * @brief Check if type is an lvalue reference
     */
    template<typename T>
    inline constexpr bool is_lvalue_reference_v = std::is_lvalue_reference_v<T>;

    /**
     * @brief Check if type is an rvalue reference
     */
    template<typename T>
    inline constexpr bool is_rvalue_reference_v = std::is_rvalue_reference_v<T>;

    /**
     * @brief Remove reference from type
     */
    template<typename T>
    using remove_reference_t = std::remove_reference_t<T>;

    /**
     * @brief Add lvalue reference to type
     */
    template<typename T>
    using add_lvalue_reference_t = std::add_lvalue_reference_t<T>;

    /**
     * @brief Add rvalue reference to type
     */
    template<typename T>
    using add_rvalue_reference_t = std::add_rvalue_reference_t<T>;

    /**
     * @brief Check if type is const-qualified
     */
    template<typename T>
    inline constexpr bool is_const_v = std::is_const_v<T>;

    /**
     * @brief Check if type is volatile-qualified
     */
    template<typename T>
    inline constexpr bool is_volatile_v = std::is_volatile_v<T>;

    /**
     * @brief Add const to type
     */
    template<typename T>
    using add_const_t = std::add_const_t<T>;

    /**
     * @brief Remove const from type
     */
    template<typename T>
    using remove_const_t = std::remove_const_t<T>;

    /**
     * @brief Add volatile to type
     */
    template<typename T>
    using add_volatile_t = std::add_volatile_t<T>;

    /**
     * @brief Remove volatile from type
     */
    template<typename T>
    using remove_volatile_t = std::remove_volatile_t<T>;

    /**
     * @brief Add const and volatile to type
     */
    template<typename T>
    using add_cv_t = std::add_cv_t<T>;

    /**
     * @brief Remove const and volatile from type
     */
    template<typename T>
    using remove_cv_t = std::remove_cv_t<T>;

    /**
     * @brief Make signed version of integral type
     */
    template<typename T>
    using make_signed_t = std::make_signed_t<T>;

    /**
     * @brief Make unsigned version of integral type
     */
    template<typename T>
    using make_unsigned_t = std::make_unsigned_t<T>;

    /**
     * @brief Decay type (remove cv-qualifiers and references, decay arrays and functions)
     */
    template<typename T>
    using decay_t = std::decay_t<T>;

    /**
     * @brief Check if types are convertible
     */
    template<typename From, typename To>
    inline constexpr bool is_convertible_v = std::is_convertible_v<From, To>;

    /**
     * @brief Check if type is a class
     */
    template<typename T>
    inline constexpr bool is_class_v = std::is_class_v<T>;

    /**
     * @brief Check if type is an enum
     */
    template<typename T>
    inline constexpr bool is_enum_v = std::is_enum_v<T>;

    /**
     * @brief Check if type is a union
     */
    template<typename T>
    inline constexpr bool is_union_v = std::is_union_v<T>;

    /**
     * @brief Check if type is a function
     */
    template<typename T>
    inline constexpr bool is_function_v = std::is_function_v<T>;

    /**
     * @brief Check if type is void
     */
    template<typename T>
    inline constexpr bool is_void_v = std::is_void_v<T>;

    /**
     * @brief Check if type is null pointer type
     */
    template<typename T>
    inline constexpr bool is_null_pointer_v = std::is_null_pointer_v<T>;

    /**
     * @brief Helper for SFINAE - always void
     */
    template<typename... Types>
    using void_t = std::void_t<Types...>;

    /**
     * @brief Integer sequence utilities
     */
    template<typename T, T... Ints>
    using integer_sequence = std::integer_sequence<T, Ints...>;

    template<size_t... Ints>
    using index_sequence = std::index_sequence<Ints...>;

    template<size_t N>
    using make_index_sequence = std::make_index_sequence<N>;

    template<typename... Types>
    using index_sequence_for = std::index_sequence_for<Types...>;

    /**
     * @brief Invoke result type
     */
    template<typename F, typename... Args>
    using invoke_result_t = std::invoke_result_t<F, Args...>;

    /**
     * @brief Check if callable
     */
    template<typename F, typename... Args>
    inline constexpr bool is_invocable_v = std::is_invocable_v<F, Args...>;

    /**
     * @brief Check if nothrow callable
     */
    template<typename F, typename... Args>
    inline constexpr bool is_nothrow_invocable_v = std::is_nothrow_invocable_v<F, Args...>;

} // namespace fem::numeric::traits

#endif //TYPE_TRAITS_H

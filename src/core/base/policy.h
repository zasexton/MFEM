#pragma once

#ifndef BASE_COPYMOVEPOLICY_H
#define BASE_COPYMOVEPOLICY_H

#include <type_traits>

namespace fem::core::base {

/**
 * @brief CRTP base class to make derived classes non-copyable
 *
 * Deletes copy constructor and copy assignment operator.
 * Move operations are still allowed unless explicitly deleted.
 *
 * Usage:
 *   class MyClass : public NonCopyable<MyClass> {
 *       // Class is now non-copyable but still movable
 *   };
 *
 * Use cases:
 * - Resource management classes (file handles, network connections)
 * - Singleton-like objects
 * - Objects with unique ownership semantics
 * - Large objects where copying would be expensive
 */
    template<typename Derived>
    class NonCopyable {
    protected:
        /**
         * @brief Protected constructor - only derived classes can construct
         */
        constexpr NonCopyable() = default;

        /**
         * @brief Protected destructor - prevents deletion through base pointer
         */
        ~NonCopyable() = default;

        /**
         * @brief Move constructor allowed
         */
        constexpr NonCopyable(NonCopyable&&) = default;

        /**
         * @brief Move assignment allowed
         */
        NonCopyable& operator=(NonCopyable&&) = default;

    public:
        /**
         * @brief Deleted copy constructor
         */
        NonCopyable(const NonCopyable&) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        NonCopyable& operator=(const NonCopyable&) = delete;
    };

/**
 * @brief CRTP base class to make derived classes non-movable
 *
 * Deletes move constructor and move assignment operator.
 * Copy operations are still allowed unless explicitly deleted.
 *
 * Usage:
 *   class MyClass : public NonMovable<MyClass> {
 *       // Class is now non-movable but still copyable
 *   };
 *
 * Use cases:
 * - Objects with location-dependent state
 * - Objects registered with external systems by address
 * - Objects containing self-references or internal pointers
 */
    template<typename Derived>
    class NonMovable {
    protected:
        /**
         * @brief Protected constructor - only derived classes can construct
         */
        constexpr NonMovable() = default;

        /**
         * @brief Protected destructor - prevents deletion through base pointer
         */
        ~NonMovable() = default;

        /**
         * @brief Copy constructor allowed
         */
        constexpr NonMovable(const NonMovable&) = default;

        /**
         * @brief Copy assignment allowed
         */
        NonMovable& operator=(const NonMovable&) = default;

    public:
        /**
         * @brief Deleted move constructor
         */
        NonMovable(NonMovable&&) = delete;

        /**
         * @brief Deleted move assignment operator
         */
        NonMovable& operator=(NonMovable&&) = delete;
    };

/**
 * @brief CRTP base class to make derived classes neither copyable nor movable
 *
 * Deletes all copy and move operations.
 * Objects can only be constructed in-place.
 *
 * Usage:
 *   class MyClass : public NonCopyableNonMovable<MyClass> {
 *       // Class cannot be copied or moved
 *   };
 *
 * Use cases:
 * - Singleton classes
 * - Resource wrappers with strict ownership
 * - Objects with complex internal state that cannot be safely transferred
 * - Hardware abstraction objects (representing physical devices)
 */
    template<typename Derived>
    class NonCopyableNonMovable {
    protected:
        /**
         * @brief Protected constructor - only derived classes can construct
         */
        constexpr NonCopyableNonMovable() = default;

        /**
         * @brief Protected destructor - prevents deletion through base pointer
         */
        ~NonCopyableNonMovable() = default;

    public:
        /**
         * @brief Deleted copy constructor
         */
        NonCopyableNonMovable(const NonCopyableNonMovable&) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        NonCopyableNonMovable& operator=(const NonCopyableNonMovable&) = delete;

        /**
         * @brief Deleted move constructor
         */
        NonCopyableNonMovable(NonCopyableNonMovable&&) = delete;

        /**
         * @brief Deleted move assignment operator
         */
        NonCopyableNonMovable& operator=(NonCopyableNonMovable&&) = delete;
    };

// === Type Traits for Detection ===

/**
 * @brief Check if a type is derived from NonCopyable
 */
    template<typename T>
    struct is_non_copyable : std::bool_constant<
            std::is_base_of_v<NonCopyable<T>, T> ||
            std::is_base_of_v<NonCopyableNonMovable<T>, T> ||
            !std::is_copy_constructible_v<T>
    > {};

template<typename T>
inline constexpr bool is_non_copyable_v = is_non_copyable<T>::value;

/**
 * @brief Check if a type is derived from NonMovable
 */
template<typename T>
struct is_non_movable : std::bool_constant<
        std::is_base_of_v<NonMovable<T>, T> ||
        std::is_base_of_v<NonCopyableNonMovable<T>, T> ||
        !std::is_move_constructible_v<T>
> {};

template<typename T>
inline constexpr bool is_non_movable_v = is_non_movable<T>::value;

// === Convenience Macros ===

/**
 * @brief Macro to make a class non-copyable (place in public section)
 */
#define FEM_NON_COPYABLE(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete;

/**
 * @brief Macro to make a class non-movable (place in public section)
 */
#define FEM_NON_MOVABLE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete;

/**
 * @brief Macro to make a class neither copyable nor movable
 */
#define FEM_NON_COPYABLE_NON_MOVABLE(ClassName) \
    FEM_NON_COPYABLE(ClassName) \
    FEM_NON_MOVABLE(ClassName)

// === Alternative Implementation (Classical Approach) ===

/**
 * @brief Classical non-copyable base class (non-CRTP)
 *
 * This is the traditional approach used by libraries like Boost.
 * Use this if you prefer a simpler inheritance model.
 */
class noncopyable {
protected:
    constexpr noncopyable() = default;
    ~noncopyable() = default;

public:
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
};

/**
 * @brief Classical non-movable base class (non-CRTP)
 */
class nonmovable {
protected:
    constexpr nonmovable() = default;
    ~nonmovable() = default;

public:
    nonmovable(nonmovable&&) = delete;
    nonmovable& operator=(nonmovable&&) = delete;
};

} // namespace fem::core

#endif //BASE_COPYMOVEPOLICY_H

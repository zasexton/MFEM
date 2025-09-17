#pragma once

#ifndef LOGGING_ASSERT_H
#define LOGGING_ASSERT_H

#include <cassert>
#include <stdexcept>
#include <format>
#include <functional>
#include <source_location>
#include <cstdio>

namespace fem::core::logging {

/**
 * @brief Assertion handler with logging integration
 *
 * This module is designed to be standalone to allow unit testing without
 * requiring the full logging infrastructure. When the logger is fully
 * integrated, the fprintf calls can be replaced with logger->fatal() and
 * logger->error() calls.
 *
 * Provides assertion macros that integrate with the logging system.
 * Failed assertions are logged before terminating or throwing.
 *
 * Usage context:
 * - Debug builds: Full assertions with detailed logging
 * - Release builds: Critical assertions only
 * - Precondition/postcondition checks
 * - Invariant verification
 * - Numerical validation (e.g., positive definiteness)
 */
    class AssertHandler {
    public:
        enum class AssertAction {
            ABORT,      // Call std::abort() (default in debug)
            THROW,      // Throw exception (default in release)
            LOG_ONLY,   // Only log the error
            CUSTOM      // Use custom handler
        };

        using assert_callback = std::function<void(
                const std::string& condition,
                const std::string& message,
                const std::source_location& location
        )>;

        /**
         * @brief Get singleton instance
         */
        static AssertHandler& instance() {
            static AssertHandler handler;
            return handler;
        }

        /**
         * @brief Handle assertion failure
         */
        [[noreturn]] void handle_failure(
                const std::string& condition,
                const std::string& message,
                const std::source_location& location) {

            // Output assertion failure
            // Note: When logger infrastructure is fully integrated, this will use the logger
            // For now, use stderr for standalone testing
            std::fprintf(stderr, "[FATAL] Assertion failed: %s\n"
                          "  Message: %s\n"
                          "  Location: %s:%u:%u\n"
                          "  Function: %s\n",
                          condition.c_str(), message.c_str(),
                          location.file_name(), location.line(), location.column(),
                          location.function_name());

            // Take action based on configuration
            switch (action_) {
                case AssertAction::ABORT:
                    std::abort();
                    break;

                case AssertAction::THROW:
                    throw assertion_error(std::format(
                            "Assertion '{}' failed: {}", condition, message));
                    break;

                case AssertAction::LOG_ONLY:
                    // Already logged, just return
                    // Note: This violates [[noreturn]] but is for special cases
                    std::terminate();
                    break;

                case AssertAction::CUSTOM:
                    if (custom_handler_) {
                        custom_handler_(condition, message, location);
                    }
                    std::terminate();
                    break;
            }

            // Shouldn't reach here
            std::terminate();
        }

        /**
         * @brief Handle verification failure (non-fatal)
         */
        bool handle_verify_failure(
                const std::string& condition,
                const std::string& message,
                const std::source_location& location) {

            // Output verification failure
            // Note: When logger infrastructure is fully integrated, this will use the logger
            std::fprintf(stderr, "[ERROR] Verification failed: %s\n"
                          "  Message: %s\n"
                          "  Location: %s:%u\n",
                          condition.c_str(), message.c_str(),
                          location.file_name(), location.line());

            return false;
        }

        /**
         * @brief Set assertion action
         */
        void set_action(AssertAction action) { action_ = action; }

        /**
         * @brief Get current assertion action
         */
        [[nodiscard]] AssertAction get_action() const { return action_; }

        /**
         * @brief Set custom assertion handler
         */
        void set_custom_handler(assert_callback handler) {
            custom_handler_ = std::move(handler);
            action_ = AssertAction::CUSTOM;
        }

        /**
         * @brief Exception thrown by assertions
         */
        class assertion_error : public std::logic_error {
        public:
            explicit assertion_error(const std::string& what)
                    : std::logic_error(what) {}
        };

    private:
        AssertHandler() {
#ifdef NDEBUG
            action_ = AssertAction::THROW;
#else
            action_ = AssertAction::ABORT;
#endif
        }

        AssertAction action_;
        assert_callback custom_handler_;
    };

} // namespace fem::core::logging

// Assertion macros

// Undefine any conflicting macros from debug.h if they exist
#ifdef FEM_ASSERT
#undef FEM_ASSERT
#endif
#ifdef FEM_ASSERT_ALWAYS
#undef FEM_ASSERT_ALWAYS
#endif
#ifdef FEM_VERIFY
#undef FEM_VERIFY
#endif
#ifdef FEM_VERIFY_DEBUG
#undef FEM_VERIFY_DEBUG
#endif
#ifdef FEM_UNREACHABLE
#undef FEM_UNREACHABLE
#endif
#ifdef FEM_NOT_IMPLEMENTED
#undef FEM_NOT_IMPLEMENTED
#endif
#ifdef FEM_STATIC_ASSERT
#undef FEM_STATIC_ASSERT
#endif
#ifdef FEM_PRECONDITION
#undef FEM_PRECONDITION
#endif
#ifdef FEM_POSTCONDITION
#undef FEM_POSTCONDITION
#endif
#ifdef FEM_INVARIANT
#undef FEM_INVARIANT
#endif

/**
 * @brief Debug assertion - only active in debug builds
 */
#ifdef NDEBUG
#define FEM_ASSERT(condition, ...) ((void)0)
#else
#define FEM_ASSERT(condition, ...) \
        do { \
            if (!(condition)) { \
                fem::core::logging::AssertHandler::instance().handle_failure( \
                    #condition, \
                    std::format(__VA_ARGS__), \
                    std::source_location::current() \
                ); \
            } \
        } while(0)
#endif

/**
 * @brief Release assertion - always active
 */
#define FEM_ASSERT_ALWAYS(condition, ...) \
    do { \
        if (!(condition)) { \
            fem::core::logging::AssertHandler::instance().handle_failure( \
                #condition, \
                std::format(__VA_ARGS__), \
                std::source_location::current() \
            ); \
        } \
    } while(0)

/**
 * @brief Verification - logs error but continues execution
 */
#define FEM_VERIFY(condition, ...) \
    ((condition) || fem::core::logging::AssertHandler::instance().handle_verify_failure( \
        #condition, \
        std::format(__VA_ARGS__), \
        std::source_location::current() \
    ))

/**
 * @brief Debug-only verification
 */
#ifdef NDEBUG
#define FEM_VERIFY_DEBUG(condition, ...) ((void)0)
#else
#define FEM_VERIFY_DEBUG(condition, ...) FEM_VERIFY(condition, __VA_ARGS__)
#endif

// Specialized assertions for FEM

/**
 * @brief Assert that a value is finite (not NaN or infinite)
 */
#define FEM_ASSERT_FINITE(value, name) \
    FEM_ASSERT(std::isfinite(value), \
        "{} must be finite, got: {}", name, value)

/**
 * @brief Assert that a value is positive
 */
#define FEM_ASSERT_POSITIVE(value, name) \
    FEM_ASSERT((value) > 0, \
        "{} must be positive, got: {}", name, value)

/**
 * @brief Assert that a value is non-negative
 */
#define FEM_ASSERT_NON_NEGATIVE(value, name) \
    FEM_ASSERT((value) >= 0, \
        "{} must be non-negative, got: {}", name, value)

/**
 * @brief Assert that a value is in range [min, max]
 */
#define FEM_ASSERT_IN_RANGE(value, min, max, name) \
    FEM_ASSERT((value) >= (min) && (value) <= (max), \
        "{} must be in range [{}, {}], got: {}", name, min, max, value)

/**
 * @brief Assert that a pointer is not null
 */
#define FEM_ASSERT_NOT_NULL(ptr, name) \
    FEM_ASSERT((ptr) != nullptr, \
        "{} must not be null", name)

/**
 * @brief Assert that a container is not empty
 */
#define FEM_ASSERT_NOT_EMPTY(container, name) \
    FEM_ASSERT(!(container).empty(), \
        "{} must not be empty", name)

/**
 * @brief Assert valid array index
 */
#define FEM_ASSERT_INDEX(index, size, name) \
    FEM_ASSERT((index) >= 0 && (index) < (size), \
        "{} index {} out of bounds [0, {})", name, index, size)

/**
 * @brief Assert matrix is square
 */
#define FEM_ASSERT_SQUARE_MATRIX(rows, cols, name) \
    FEM_ASSERT((rows) == (cols), \
        "{} must be square, got {}x{}", name, rows, cols)

/**
 * @brief Assert dimensions match
 */
#define FEM_ASSERT_DIMENSIONS_MATCH(dim1, dim2, name1, name2) \
    FEM_ASSERT((dim1) == (dim2), \
        "{} dimension {} does not match {} dimension {}", \
        name1, dim1, name2, dim2)

// Precondition/postcondition macros

/**
 * @brief Mark a precondition
 */
#define FEM_PRECONDITION(condition, ...) \
    FEM_ASSERT(condition, "Precondition failed: " __VA_ARGS__)

/**
 * @brief Mark a postcondition
 */
#define FEM_POSTCONDITION(condition, ...) \
    FEM_ASSERT(condition, "Postcondition failed: " __VA_ARGS__)

/**
 * @brief Mark an invariant
 */
#define FEM_INVARIANT(condition, ...) \
    FEM_ASSERT(condition, "Invariant violated: " __VA_ARGS__)

/**
 * @brief Mark unreachable code
 */
#define FEM_UNREACHABLE(...) \
    do { \
        fem::core::logging::AssertHandler::instance().handle_failure( \
            "false", \
            std::format("Unreachable code reached: " __VA_ARGS__), \
            std::source_location::current() \
        ); \
    } while(0)

/**
 * @brief Not yet implemented
 */
#define FEM_NOT_IMPLEMENTED(...) \
    do { \
        fem::core::logging::AssertHandler::instance().handle_failure( \
            "false", \
            std::format("Not implemented: " __VA_ARGS__), \
            std::source_location::current() \
        ); \
    } while(0)

// Static assertions with better messages

/**
 * @brief Static assertion with custom message
 */
#define FEM_STATIC_ASSERT(condition, message) \
    static_assert(condition, "FEM static assertion failed: " message)

/**
 * @brief Assert type traits
 */
#define FEM_ASSERT_TYPE_IS(Type, Trait, message) \
    FEM_STATIC_ASSERT(Trait<Type>::value, message)

#endif //LOGGING_ASSERT_H

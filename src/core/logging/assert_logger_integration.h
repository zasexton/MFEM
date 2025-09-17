#pragma once

/**
 * @file assert_logger_integration.h
 * @brief Integration layer for assert.h with logger infrastructure
 *
 * This file provides the integration between assert.h and the logger
 * infrastructure. When the logger is fully ready, include this file
 * after assert.h to enable logger integration.
 */

#ifndef ASSERT_LOGGER_INTEGRATION_H
#define ASSERT_LOGGER_INTEGRATION_H

#include "logger.h"
#include "loggermanager.h"

namespace fem::core::logging {

/**
 * @brief Enhanced assertion handler with full logger integration
 *
 * This class extends the basic AssertHandler to use the logger
 * infrastructure instead of fprintf.
 */
class LoggerAssertHandler {
public:
    /**
     * @brief Handle assertion failure with logger
     */
    [[noreturn]] static void handle_failure_with_logger(
            const std::string& condition,
            const std::string& message,
            const std::source_location& location) {

        // Get assert logger
        auto logger = get_logger("fem.assert");

        // Log the failure
        logger->fatal("Assertion failed: {}\n"
                      "  Message: {}\n"
                      "  Location: {}:{}:{}\n"
                      "  Function: {}",
                      condition, message,
                      location.file_name(), location.line(), location.column(),
                      location.function_name());

        // Take action based on configuration
        auto& handler = AssertHandler::instance();
        switch (handler.get_action()) {
            case AssertHandler::AssertAction::ABORT:
                std::abort();
                break;

            case AssertHandler::AssertAction::THROW:
                throw AssertHandler::assertion_error(
                    std::format("Assertion '{}' failed: {}", condition, message));
                break;

            case AssertHandler::AssertAction::LOG_ONLY:
                // Already logged, terminate
                std::terminate();
                break;

            case AssertHandler::AssertAction::CUSTOM:
                // Custom handler should have been called
                std::terminate();
                break;
        }

        // Shouldn't reach here
        std::terminate();
    }

    /**
     * @brief Handle verification failure with logger
     */
    static bool handle_verify_failure_with_logger(
            const std::string& condition,
            const std::string& message,
            const std::source_location& location) {

        auto logger = get_logger("fem.assert");

        logger->error("Verification failed: {}\n"
                      "  Message: {}\n"
                      "  Location: {}:{}",
                      condition, message,
                      location.file_name(), location.line());

        return false;
    }
};

// Optional: Macro to switch between logger and standalone versions
#ifdef FEM_USE_LOGGER_ASSERT

// Redefine macros to use logger versions
#undef FEM_ASSERT_ALWAYS
#define FEM_ASSERT_ALWAYS(condition, ...) \
    do { \
        if (!(condition)) { \
            fem::core::logging::LoggerAssertHandler::handle_failure_with_logger( \
                #condition, \
                std::format(__VA_ARGS__), \
                std::source_location::current() \
            ); \
        } \
    } while(0)

#undef FEM_VERIFY
#define FEM_VERIFY(condition, ...) \
    ((condition) || fem::core::logging::LoggerAssertHandler::handle_verify_failure_with_logger( \
        #condition, \
        std::format(__VA_ARGS__), \
        std::source_location::current() \
    ))

#endif // FEM_USE_LOGGER_ASSERT

} // namespace fem::core::logging

#endif // ASSERT_LOGGER_INTEGRATION_H
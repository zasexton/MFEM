#pragma once

#ifndef CORE_ERROR_ERROR_HANDLER_H
#define CORE_ERROR_ERROR_HANDLER_H

#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <csignal>
#include "exception_base.h"
#include "../base/singleton.h"

namespace fem::core::error {

/**
 * @brief Global error handler
 *
 * Provides centralized error handling, crash reporting, and graceful shutdowns
 */
class ErrorHandler : public base::Singleton<ErrorHandler> {
    friend class base::Singleton<ErrorHandler>;

public:
    using Handler = std::function<void(const Exception&)>;
    using PanicHandler = std::function<void(const char*)>;
    using SignalHandler = std::function<void(int)>;

    /**
     * @brief Set the global error handler
     */
    void set_handler(Handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_handler_ = std::move(handler);
    }

    /**
     * @brief Set the panic handler (for unrecoverable errors)
     */
    void set_panic_handler(PanicHandler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        panic_handler_ = std::move(handler);
    }

    /**
     * @brief Set handler for specific signal
     */
    void set_signal_handler(int signal, SignalHandler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        signal_handlers_[signal] = std::move(handler);
    }

    /**
     * @brief Handle an error
     */
    void handle_error(const Exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (error_handler_) {
            try {
                error_handler_(e);
            } catch (...) {
                // Error handler itself threw, fall back to default
                default_error_handler(e);
            }
        } else {
            default_error_handler(e);
        }

        // Update statistics
        error_count_++;
        last_error_code_ = e.code();
    }

    /**
     * @brief Panic - unrecoverable error
     */
    [[noreturn]] void panic(const char* message) {
        // Try custom panic handler first
        if (panic_handler_) {
            panic_handler_(message);
        }

        // Default panic behavior
        std::fprintf(stderr, "PANIC: %s\n", message);
        std::fflush(stderr);
        std::abort();
    }

    /**
     * @brief Scoped error handler override
     */
    class ScopedHandler {
    public:
        explicit ScopedHandler(Handler handler)
            : previous_(ErrorHandler::instance().error_handler_) {
            ErrorHandler::instance().set_handler(std::move(handler));
        }

        ~ScopedHandler() {
            ErrorHandler::instance().set_handler(std::move(previous_));
        }

        ScopedHandler(const ScopedHandler&) = delete;
        ScopedHandler& operator=(const ScopedHandler&) = delete;

    private:
        Handler previous_;
    };

    /**
     * @brief Install signal handlers for common signals
     */
    void install_signal_handlers() {
        // SIGSEGV - Segmentation fault
        std::signal(SIGSEGV, &ErrorHandler::signal_handler_wrapper);

        // SIGABRT - Abort
        std::signal(SIGABRT, &ErrorHandler::signal_handler_wrapper);

        // SIGFPE - Floating point exception
        std::signal(SIGFPE, &ErrorHandler::signal_handler_wrapper);

        // SIGILL - Illegal instruction
        std::signal(SIGILL, &ErrorHandler::signal_handler_wrapper);

#ifndef _WIN32
        // SIGBUS - Bus error (not on Windows)
        std::signal(SIGBUS, &ErrorHandler::signal_handler_wrapper);
#endif
    }

    /**
     * @brief Set up terminate handler
     */
    void set_terminate_handler() {
        std::set_terminate([]() {
            ErrorHandler::instance().handle_terminate();
        });
    }

    /**
     * @brief Get error statistics
     */
    size_t error_count() const noexcept {
        return error_count_.load();
    }

    ErrorCode last_error_code() const noexcept {
        return last_error_code_.load();
    }

    /**
     * @brief Reset error statistics
     */
    void reset_statistics() {
        error_count_ = 0;
        last_error_code_ = ErrorCode::Success;
    }

    /**
     * @brief Enable/disable error handler
     */
    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }

    bool is_enabled() const noexcept {
        return enabled_.load();
    }

private:
    ErrorHandler() : enabled_(true) {}

    void default_error_handler(const Exception& e) {
        if (!enabled_) return;

        // Output to stderr
        std::fprintf(stderr, "Error: %s\n", e.what());
        if (e.code() != ErrorCode::Unknown) {
            std::fprintf(stderr, "Code: %s\n",
                        core_error_category().message(static_cast<int>(e.code())).c_str());
        }
        std::fprintf(stderr, "Location: %s:%d\n",
                    e.where().file_name(), static_cast<int>(e.where().line()));

        // Output full diagnostic in debug mode
#ifndef NDEBUG
        e.print_diagnostic(std::cerr);
#endif
    }

    static void signal_handler_wrapper(int signal) {
        auto& handler = ErrorHandler::instance();

        // Check for custom signal handler
        auto it = handler.signal_handlers_.find(signal);
        if (it != handler.signal_handlers_.end() && it->second) {
            it->second(signal);
            return;
        }

        // Default signal handling
        const char* signal_name = "Unknown signal";
        switch (signal) {
            case SIGSEGV: signal_name = "Segmentation fault"; break;
            case SIGABRT: signal_name = "Abort"; break;
            case SIGFPE:  signal_name = "Floating point exception"; break;
            case SIGILL:  signal_name = "Illegal instruction"; break;
#ifndef _WIN32
            case SIGBUS:  signal_name = "Bus error"; break;
#endif
        }

        handler.panic(signal_name);
    }

    void handle_terminate() {
        try {
            // Try to get current exception
            if (auto e = std::current_exception()) {
                std::rethrow_exception(e);
            }
        } catch (const Exception& e) {
            handle_error(e);
        } catch (const std::exception& e) {
            Exception wrapped(e.what(), ErrorCode::Unknown);
            handle_error(wrapped);
        } catch (...) {
            Exception unknown("Unknown exception in terminate handler",
                            ErrorCode::Unknown);
            handle_error(unknown);
        }

        panic("Unhandled exception - terminating");
    }

private:
    mutable std::mutex mutex_;
    Handler error_handler_;
    PanicHandler panic_handler_;
    std::map<int, SignalHandler> signal_handlers_;
    std::atomic<size_t> error_count_{0};
    std::atomic<ErrorCode> last_error_code_{ErrorCode::Success};
    std::atomic<bool> enabled_{true};
};

/**
 * @brief RAII guard for error handling
 */
class ErrorGuard {
public:
    explicit ErrorGuard(std::function<void()> cleanup)
        : cleanup_(std::move(cleanup))
        , dismissed_(false) {
    }

    ~ErrorGuard() {
        if (!dismissed_ && cleanup_) {
            try {
                cleanup_();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }
    }

    void dismiss() noexcept {
        dismissed_ = true;
    }

    ErrorGuard(const ErrorGuard&) = delete;
    ErrorGuard& operator=(const ErrorGuard&) = delete;

private:
    std::function<void()> cleanup_;
    bool dismissed_;
};

// Global helper functions

/**
 * @brief Handle an error using the global handler
 */
inline void handle_error(const Exception& e) {
    ErrorHandler::instance().handle_error(e);
}

/**
 * @brief Panic with a message
 */
[[noreturn]] inline void panic(const char* message) {
    ErrorHandler::instance().panic(message);
}

/**
 * @brief Install default error handling
 */
inline void install_error_handlers() {
    ErrorHandler::instance().install_signal_handlers();
    ErrorHandler::instance().set_terminate_handler();
}

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_HANDLER_H
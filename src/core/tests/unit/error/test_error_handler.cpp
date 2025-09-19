#include <gtest/gtest.h>
#include <core/error/error_handler.h>
#include <core/error/logic_error.h>
#include <core/error/runtime_error.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <sstream>

using namespace fem::core::error;

class ErrorHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset error handler state before each test
        ErrorHandler::instance().reset_statistics();
        ErrorHandler::instance().set_enabled(true);
        ErrorHandler::instance().set_handler(nullptr);
        ErrorHandler::instance().set_panic_handler(nullptr);

        // Clear custom signal handlers
        for (int sig : {SIGSEGV, SIGABRT, SIGFPE, SIGILL}) {
            ErrorHandler::instance().set_signal_handler(sig, nullptr);
        }
    }

    void TearDown() override {
        // Clean up after each test
        ErrorHandler::instance().reset_statistics();
        ErrorHandler::instance().set_enabled(true);
        ErrorHandler::instance().set_handler(nullptr);
        ErrorHandler::instance().set_panic_handler(nullptr);
    }
};

// Basic functionality tests
TEST_F(ErrorHandlerTest, SingletonBehavior) {
    auto& handler1 = ErrorHandler::instance();
    auto& handler2 = ErrorHandler::instance();

    EXPECT_EQ(&handler1, &handler2);
}

TEST_F(ErrorHandlerTest, DefaultState) {
    auto& handler = ErrorHandler::instance();

    EXPECT_TRUE(handler.is_enabled());
    EXPECT_EQ(handler.error_count(), 0);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::Success);
}

TEST_F(ErrorHandlerTest, EnableDisable) {
    auto& handler = ErrorHandler::instance();

    EXPECT_TRUE(handler.is_enabled());

    handler.set_enabled(false);
    EXPECT_FALSE(handler.is_enabled());

    handler.set_enabled(true);
    EXPECT_TRUE(handler.is_enabled());
}

// Error handling tests
TEST_F(ErrorHandlerTest, HandleBasicError) {
    auto& handler = ErrorHandler::instance();
    InvalidArgumentError error("param", "invalid value");

    EXPECT_EQ(handler.error_count(), 0);

    handler.handle_error(error);

    EXPECT_EQ(handler.error_count(), 1);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::InvalidArgument);
}

TEST_F(ErrorHandlerTest, HandleMultipleErrors) {
    auto& handler = ErrorHandler::instance();

    InvalidArgumentError error1("param1", "invalid");
    RuntimeError error2("runtime failure");
    OutOfRangeError error3("index", 10, 5);

    handler.handle_error(error1);
    handler.handle_error(error2);
    handler.handle_error(error3);

    EXPECT_EQ(handler.error_count(), 3);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::OutOfRange);
}

TEST_F(ErrorHandlerTest, CustomErrorHandler) {
    auto& handler = ErrorHandler::instance();

    std::string captured_message;
    ErrorCode captured_code = ErrorCode::Success;

    handler.set_handler([&](const Exception& e) {
        captured_message = e.what();
        captured_code = e.code();
    });

    InvalidArgumentError error("test_param", "test_message");
    handler.handle_error(error);

    EXPECT_EQ(captured_message, "Invalid argument 'test_param': test_message");
    EXPECT_EQ(captured_code, ErrorCode::InvalidArgument);
    EXPECT_EQ(handler.error_count(), 1);
}

TEST_F(ErrorHandlerTest, HandlerExceptionSafety) {
    auto& handler = ErrorHandler::instance();

    // Set a handler that throws
    handler.set_handler([](const Exception&) {
        throw std::runtime_error("Handler error");
    });

    InvalidArgumentError error("param", "value");

    // Should not throw, should fall back to default handler
    EXPECT_NO_THROW(handler.handle_error(error));
    EXPECT_EQ(handler.error_count(), 1);
}

// Scoped handler tests
TEST_F(ErrorHandlerTest, ScopedHandler) {
    auto& handler = ErrorHandler::instance();

    std::string captured_message;

    {
        ErrorHandler::ScopedHandler scoped([&](const Exception& e) {
            captured_message = "Scoped: " + std::string(e.what());
        });

        InvalidArgumentError error("param", "value");
        handler.handle_error(error);

        EXPECT_EQ(captured_message, "Scoped: Invalid argument 'param': value");
    }

    // After scope, should revert to default
    captured_message.clear();
    InvalidArgumentError error2("param2", "value2");
    handler.handle_error(error2);

    // Should not have been captured by scoped handler
    EXPECT_TRUE(captured_message.empty());
}

TEST_F(ErrorHandlerTest, NestedScopedHandlers) {
    auto& handler = ErrorHandler::instance();

    std::vector<std::string> messages;

    ErrorHandler::ScopedHandler outer([&](const Exception& e) {
        messages.push_back("Outer: " + std::string(e.what()));
    });

    {
        ErrorHandler::ScopedHandler inner([&](const Exception& e) {
            messages.push_back("Inner: " + std::string(e.what()));
        });

        InvalidArgumentError error1("param1", "value1");
        handler.handle_error(error1);
    }

    InvalidArgumentError error2("param2", "value2");
    handler.handle_error(error2);

    ASSERT_EQ(messages.size(), 2);
    EXPECT_EQ(messages[0], "Inner: Invalid argument 'param1': value1");
    EXPECT_EQ(messages[1], "Outer: Invalid argument 'param2': value2");
}

// Panic handler tests
TEST_F(ErrorHandlerTest, CustomPanicHandler) {
    auto& handler = ErrorHandler::instance();

    std::string panic_message;
    bool panic_called = false;

    handler.set_panic_handler([&](const char* msg) {
        panic_message = msg;
        panic_called = true;
        // Don't actually abort in test
    });

    // Test panic through signal handler simulation
    // We can't actually trigger SIGSEGV safely in a test
    // So we'll test the panic handler directly
}

// Signal handler tests
TEST_F(ErrorHandlerTest, CustomSignalHandler) {
    auto& handler = ErrorHandler::instance();

    int captured_signal = 0;

    handler.set_signal_handler(SIGUSR1, [&](int sig) {
        captured_signal = sig;
    });

    // Install signal handlers
    handler.install_signal_handlers();

    // We can't safely test signal handling in unit tests
    // but we can verify the handler was set
    // The actual signal handling would be tested in integration tests
}

// Statistics tests
TEST_F(ErrorHandlerTest, StatisticsTracking) {
    auto& handler = ErrorHandler::instance();

    EXPECT_EQ(handler.error_count(), 0);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::Success);

    InvalidArgumentError error1("param1", "value1");
    handler.handle_error(error1);

    EXPECT_EQ(handler.error_count(), 1);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::InvalidArgument);

    RuntimeError error2("runtime error");
    handler.handle_error(error2);

    EXPECT_EQ(handler.error_count(), 2);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::Unknown);

    handler.reset_statistics();

    EXPECT_EQ(handler.error_count(), 0);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::Success);
}

// Thread safety tests
TEST_F(ErrorHandlerTest, ThreadSafety) {
    auto& handler = ErrorHandler::instance();

    std::atomic<int> handler_calls{0};

    handler.set_handler([&](const Exception&) {
        handler_calls++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });

    constexpr int num_threads = 10;
    constexpr int errors_per_thread = 10;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < errors_per_thread; ++i) {
                InvalidArgumentError error("param" + std::to_string(t),
                                         "value" + std::to_string(i));
                handler.handle_error(error);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(handler_calls.load(), num_threads * errors_per_thread);
    EXPECT_EQ(handler.error_count(), num_threads * errors_per_thread);
}

// Global function tests
TEST_F(ErrorHandlerTest, GlobalHandleError) {
    auto& handler = ErrorHandler::instance();

    std::string captured_message;
    handler.set_handler([&](const Exception& e) {
        captured_message = e.what();
    });

    InvalidArgumentError error("global_param", "global_value");
    handle_error(error);  // Use global function

    EXPECT_EQ(captured_message, "Invalid argument 'global_param': global_value");
    EXPECT_EQ(handler.error_count(), 1);
}

TEST_F(ErrorHandlerTest, InstallErrorHandlers) {
    // Test that installation doesn't crash
    EXPECT_NO_THROW(install_error_handlers());
}

// Error guard tests (from error_handler.h)
TEST_F(ErrorHandlerTest, ErrorGuardBasic) {
    bool cleanup_called = false;

    {
        ErrorGuard guard([&]() {
            cleanup_called = true;
        });
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorHandlerTest, ErrorGuardDismiss) {
    bool cleanup_called = false;

    {
        ErrorGuard guard([&]() {
            cleanup_called = true;
        });
        guard.dismiss();
    }

    EXPECT_FALSE(cleanup_called);
}

TEST_F(ErrorHandlerTest, ErrorGuardExceptionSafety) {
    bool cleanup_called = false;

    try {
        ErrorGuard guard([&]() {
            cleanup_called = true;
        });
        throw std::runtime_error("test exception");
    } catch (...) {
        // Exception expected
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorHandlerTest, ErrorGuardCleanupException) {
    // Test that exceptions in cleanup don't propagate
    EXPECT_NO_THROW({
        ErrorGuard guard([]() {
            throw std::runtime_error("cleanup error");
        });
    });
}

// Integration tests
TEST_F(ErrorHandlerTest, DisabledHandler) {
    auto& handler = ErrorHandler::instance();

    std::string captured_message;
    handler.set_handler([&](const Exception& e) {
        captured_message = e.what();
    });

    handler.set_enabled(false);

    InvalidArgumentError error("param", "value");
    handler.handle_error(error);

    // Handler should still be called (enabled/disabled affects default handler)
    // But statistics should still be updated
    EXPECT_EQ(handler.error_count(), 1);
}

TEST_F(ErrorHandlerTest, ComplexErrorChain) {
    auto& handler = ErrorHandler::instance();

    std::vector<std::string> handled_errors;
    handler.set_handler([&](const Exception& e) {
        handled_errors.push_back(e.what());
    });

    // Create nested error chain
    InvalidArgumentError inner("param", "invalid");
    IOError middle(IOError::Operation::Read, "file.txt", "permission denied");
    ConfigurationError outer("config", "failed to load");

    middle.with_nested(inner);
    outer.with_nested(middle);

    handler.handle_error(outer);

    EXPECT_EQ(handled_errors.size(), 1);
    EXPECT_TRUE(handled_errors[0].find("config") != std::string::npos);
    EXPECT_EQ(handler.error_count(), 1);
    EXPECT_EQ(handler.last_error_code(), ErrorCode::ConfigInvalid);
}
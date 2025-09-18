#include <gtest/gtest.h>
#include <core/error/nested_exception.h>
#include <core/error/exception_base.h>
#include <core/error/logic_error.h>
#include <core/error/runtime_error.h>
#include <core/error/error_code.h>
#include <stdexcept>
#include <vector>
#include <functional>
#include <memory>
#include <thread>
#include <future>
#include <chrono>

using namespace fem::core::error;

class NestedExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for tests if needed
    }
};

// ========== Basic exception chaining tests ==========
TEST_F(NestedExceptionTest, ExceptionWithNested) {
    Exception nested("Nested exception", ErrorCode::FileNotFound);
    Exception ex("Main exception", ErrorCode::Unknown);
    ex.with_nested(nested);

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Main exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Nested exception") != std::string::npos);
}

TEST_F(NestedExceptionTest, ExceptionChaining) {
    Exception nested("Inner exception", ErrorCode::FileNotFound);
    Exception ex("Outer exception", ErrorCode::Unknown);
    ex.with_nested(nested);

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Outer exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Inner exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Caused by") != std::string::npos);
}

TEST_F(NestedExceptionTest, MultipleNestedLevels) {
    Exception level3("Level 3 error", ErrorCode::SystemError);
    Exception level2("Level 2 error", ErrorCode::FileNotFound);
    Exception level1("Level 1 error", ErrorCode::Unknown);

    level2.with_nested(level3);
    level1.with_nested(level2);

    std::string full_msg = level1.full_message();
    EXPECT_TRUE(full_msg.find("Level 1 error") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Level 2 error") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Level 3 error") != std::string::npos);
}

// ========== Mixed exception type chaining ==========
TEST_F(NestedExceptionTest, LogicAndRuntimeMixed) {
    InvalidArgumentError logic_error("param", "invalid value");
    IOError runtime_error(IOError::Operation::Read, "file.txt", "not found");

    runtime_error.with_nested(logic_error);

    std::string full_msg = runtime_error.full_message();
    EXPECT_TRUE(full_msg.find("file.txt") != std::string::npos);
    EXPECT_TRUE(full_msg.find("param") != std::string::npos);
    EXPECT_TRUE(full_msg.find("invalid value") != std::string::npos);
}

TEST_F(NestedExceptionTest, ComplexChaining) {
    // Create a chain: Config -> Parse -> IO -> Logic
    InvalidArgumentError logic_ex("buffer_size", "must be positive");
    IOError io_ex(IOError::Operation::Read, "config.txt", "permission denied");
    ParseError parse_ex("malformed JSON", 25, "closing bracket");
    ConfigurationError config_ex("database", "failed to load configuration");

    io_ex.with_nested(logic_ex);
    parse_ex.with_nested(io_ex);
    config_ex.with_nested(parse_ex);

    std::string full_msg = config_ex.full_message();
    EXPECT_TRUE(full_msg.find("database") != std::string::npos);
    EXPECT_TRUE(full_msg.find("malformed JSON") != std::string::npos);
    EXPECT_TRUE(full_msg.find("config.txt") != std::string::npos);
    EXPECT_TRUE(full_msg.find("buffer_size") != std::string::npos);
}

// ========== Exception context with nesting ==========
TEST_F(NestedExceptionTest, ContextWithNesting) {
    Exception nested("Database connection failed", ErrorCode::IoError);
    nested.with_context("timeout after 30 seconds")
          .with_context("using connection pool");

    Exception main("Application startup failed", ErrorCode::Unknown);
    main.with_context("during initialization phase")
        .with_nested(nested);

    std::string full_msg = main.full_message();
    EXPECT_TRUE(full_msg.find("Application startup failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("initialization phase") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Database connection failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("timeout after 30 seconds") != std::string::npos);
    EXPECT_TRUE(full_msg.find("connection pool") != std::string::npos);
}

// ========== Exception throwing and catching with nesting ==========
TEST_F(NestedExceptionTest, ThrowAndCatchNested) {
    bool caught = false;
    try {
        try {
            throw InvalidArgumentError("param", "invalid");
        } catch (const Exception& inner) {
            IOError outer(IOError::Operation::Write, "output.txt", "failed");
            outer.with_nested(inner);
            throw outer;
        }
    } catch (const IOError& e) {
        caught = true;
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("output.txt") != std::string::npos);
        EXPECT_TRUE(full_msg.find("param") != std::string::npos);
    }
    EXPECT_TRUE(caught);
}

// ========== Real-world scenario tests ==========
TEST_F(NestedExceptionTest, FileProcessingScenario) {
    auto process_config_file = [](const std::string& filename, bool simulate_error) {
        if (simulate_error) {
            try {
                // Simulate file read error
                throw IOError(IOError::Operation::Read, filename, "file not found");
            } catch (const Exception& io_error) {
                // Wrap in configuration error
                ConfigurationError config_error("application_config", "failed to load configuration file");
                config_error.with_nested(io_error);
                throw config_error;
            }
        }
        return "success";
    };

    EXPECT_NO_THROW(process_config_file("config.json", false));

    try {
        process_config_file("missing.json", true);
        FAIL() << "Expected ConfigurationError to be thrown";
    } catch (const ConfigurationError& e) {
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("application_config") != std::string::npos);
        EXPECT_TRUE(full_msg.find("missing.json") != std::string::npos);
        EXPECT_TRUE(full_msg.find("file not found") != std::string::npos);
    }
}

TEST_F(NestedExceptionTest, NetworkServiceScenario) {
    auto call_external_service = [](bool network_fails, bool parse_fails) {
        try {
            if (network_fails) {
                throw NetworkError(NetworkError::Type::Timeout, "api.service.com", "request timeout");
            }
            if (parse_fails) {
                throw ParseError("{ invalid json", 2, "closing brace");
            }
            return "success";
        } catch (const Exception& e) {
            RuntimeError service_error("External service call failed");
            service_error.with_nested(e);
            throw service_error;
        }
    };

    EXPECT_NO_THROW(call_external_service(false, false));

    // Test network error propagation
    try {
        call_external_service(true, false);
        FAIL() << "Expected RuntimeError to be thrown";
    } catch (const RuntimeError& e) {
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("External service call failed") != std::string::npos);
        EXPECT_TRUE(full_msg.find("api.service.com") != std::string::npos);
        EXPECT_TRUE(full_msg.find("timeout") != std::string::npos);
    }

    // Test parse error propagation
    try {
        call_external_service(false, true);
        FAIL() << "Expected RuntimeError to be thrown";
    } catch (const RuntimeError& e) {
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("External service call failed") != std::string::npos);
        EXPECT_TRUE(full_msg.find("invalid json") != std::string::npos);
        EXPECT_TRUE(full_msg.find("closing brace") != std::string::npos);
    }
}

// ========== Memory and performance tests ==========
TEST_F(NestedExceptionTest, DeepNestingMemory) {
    const int depth = 100;
    Exception base("Base exception", ErrorCode::Unknown);
    Exception* current = &base;

    // Create a deep chain
    std::vector<std::unique_ptr<Exception>> exceptions;
    for (int i = 1; i < depth; ++i) {
        auto next = std::make_unique<Exception>("Level " + std::to_string(i), ErrorCode::Unknown);
        current->with_nested(*next);
        current = next.get();
        exceptions.push_back(std::move(next));
    }

    std::string full_msg = base.full_message();
    EXPECT_TRUE(full_msg.find("Base exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Level 1") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Level " + std::to_string(depth - 1)) != std::string::npos);

    // Verify the message is reasonable in size (not exponentially growing)
    EXPECT_LT(full_msg.size(), 50000u); // Should be manageable size
}

TEST_F(NestedExceptionTest, ConcurrentNestingOperations) {
    const int num_threads = 10;
    const int operations_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> successful_operations{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&successful_operations, operations_per_thread, i]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                try {
                    Exception inner("Inner " + std::to_string(i) + ":" + std::to_string(j), ErrorCode::FileNotFound);
                    Exception outer("Outer " + std::to_string(i) + ":" + std::to_string(j), ErrorCode::Unknown);
                    outer.with_nested(inner);

                    std::string full_msg = outer.full_message();
                    if (full_msg.find("Inner") != std::string::npos &&
                        full_msg.find("Outer") != std::string::npos) {
                        successful_operations++;
                    }
                } catch (...) {
                    // Should not happen in this test
                    FAIL() << "Unexpected exception in concurrent test";
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(successful_operations, num_threads * operations_per_thread);
}

// ========== Edge cases ==========
TEST_F(NestedExceptionTest, EmptyExceptionChaining) {
    Exception empty1("", ErrorCode::Unknown);
    Exception empty2("", ErrorCode::FileNotFound);
    empty1.with_nested(empty2);

    EXPECT_NO_THROW(empty1.full_message());
    std::string full_msg = empty1.full_message();
    EXPECT_FALSE(full_msg.empty());
}

TEST_F(NestedExceptionTest, SelfNesting) {
    Exception ex("Self-referencing exception", ErrorCode::Unknown);

    // This should not cause infinite recursion or crash
    ex.with_nested(ex);

    EXPECT_NO_THROW(ex.full_message());
    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Self-referencing exception") != std::string::npos);
}

TEST_F(NestedExceptionTest, LargeMessageChaining) {
    std::string large_message1(1000, 'A');
    std::string large_message2(1000, 'B');

    Exception inner(large_message1, ErrorCode::FileNotFound);
    Exception outer(large_message2, ErrorCode::Unknown);
    outer.with_nested(inner);

    std::string full_msg = outer.full_message();
    EXPECT_TRUE(full_msg.find(large_message1) != std::string::npos);
    EXPECT_TRUE(full_msg.find(large_message2) != std::string::npos);
}

// ========== Exception copying and moving ==========
TEST_F(NestedExceptionTest, CopyExceptionWithNested) {
    Exception inner("Inner exception", ErrorCode::FileNotFound);
    Exception original("Original exception", ErrorCode::Unknown);
    original.with_nested(inner);

    Exception copied = original;
    std::string original_msg = original.full_message();
    std::string copied_msg = copied.full_message();

    EXPECT_EQ(original_msg, copied_msg);
    EXPECT_TRUE(copied_msg.find("Inner exception") != std::string::npos);
    EXPECT_TRUE(copied_msg.find("Original exception") != std::string::npos);
}

TEST_F(NestedExceptionTest, MoveExceptionWithNested) {
    Exception inner("Inner exception", ErrorCode::FileNotFound);
    Exception original("Original exception", ErrorCode::Unknown);
    original.with_nested(inner);

    std::string expected_msg = original.full_message();
    Exception moved = std::move(original);
    std::string moved_msg = moved.full_message();

    EXPECT_EQ(expected_msg, moved_msg);
    EXPECT_TRUE(moved_msg.find("Inner exception") != std::string::npos);
    EXPECT_TRUE(moved_msg.find("Original exception") != std::string::npos);
}

// ========== Integration with different error types ==========
TEST_F(NestedExceptionTest, SystemErrorNesting) {
    SystemError system_error("File operation failed", 2); // errno 2 = ENOENT
    Exception wrapper("System operation wrapper", ErrorCode::Unknown);
    wrapper.with_nested(system_error);

    std::string full_msg = wrapper.full_message();
    EXPECT_TRUE(full_msg.find("System operation wrapper") != std::string::npos);
    EXPECT_TRUE(full_msg.find("File operation failed") != std::string::npos);
}

TEST_F(NestedExceptionTest, AllErrorTypesChaining) {
    // Create a chain with different error types
    InvalidArgumentError logic_error("param", "invalid");
    IOError io_error(IOError::Operation::Read, "file.txt", "not found");
    TimeoutError timeout_error("operation", std::chrono::milliseconds(5000));
    SystemError system_error("System call failed", 1);

    io_error.with_nested(logic_error);
    timeout_error.with_nested(io_error);
    system_error.with_nested(timeout_error);

    std::string full_msg = system_error.full_message();
    EXPECT_TRUE(full_msg.find("System call failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("operation") != std::string::npos);
    EXPECT_TRUE(full_msg.find("5000") != std::string::npos);
    EXPECT_TRUE(full_msg.find("file.txt") != std::string::npos);
    EXPECT_TRUE(full_msg.find("param") != std::string::npos);
}

// ========== Formatting and display tests ==========
TEST_F(NestedExceptionTest, FormattedExceptionDisplay) {
    Exception inner("Database connection timeout", ErrorCode::IoError);
    inner.with_context("host: db.example.com")
         .with_context("port: 5432");

    Exception outer("Service initialization failed", ErrorCode::Unknown);
    outer.with_context("during application startup")
         .with_nested(inner);

    std::string full_msg = outer.full_message();

    // Verify proper indentation and formatting
    EXPECT_TRUE(full_msg.find("Service initialization failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("during application startup") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Caused by") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Database connection timeout") != std::string::npos);
    EXPECT_TRUE(full_msg.find("host: db.example.com") != std::string::npos);
    EXPECT_TRUE(full_msg.find("port: 5432") != std::string::npos);
}

TEST_F(NestedExceptionTest, ExceptionHierarchyPreservation) {
    // Test that exception type information is preserved through nesting
    auto create_nested_chain = []() {
        try {
            throw InvalidArgumentError("value", "must be positive");
        } catch (const LogicError& logic_ex) {
            try {
                IOError io_ex(IOError::Operation::Write, "output.txt", "disk full");
                io_ex.with_nested(logic_ex);
                throw io_ex;
            } catch (const RuntimeError& runtime_ex) {
                Exception wrapper("Operation failed", ErrorCode::Unknown);
                wrapper.with_nested(runtime_ex);
                throw wrapper;
            }
        }
    };

    try {
        create_nested_chain();
        FAIL() << "Expected exception to be thrown";
    } catch (const Exception& e) {
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("Operation failed") != std::string::npos);
        EXPECT_TRUE(full_msg.find("output.txt") != std::string::npos);
        EXPECT_TRUE(full_msg.find("value") != std::string::npos);
        EXPECT_TRUE(full_msg.find("must be positive") != std::string::npos);
    }
}
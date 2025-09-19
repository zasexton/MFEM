#include <gtest/gtest.h>
#include <core/error/try_catch.h>
#include <core/error/logic_error.h>
#include <core/error/runtime_error.h>
#include <core/error/error_message.h>
#include <chrono>
#include <thread>
#include <atomic>

using namespace fem::core::error;

class TryCatchTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// try_catch function tests
TEST_F(TryCatchTest, TryCatchSuccess) {
    auto result = try_catch([]() {
        return 42;
    });

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(TryCatchTest, TryCatchVoidFunction) {
    bool executed = false;
    auto result = try_catch([&]() {
        executed = true;
    });

    EXPECT_TRUE(result.is_ok());
    EXPECT_TRUE(executed);
}

TEST_F(TryCatchTest, TryCatchWithArguments) {
    auto add = [](int a, int b) { return a + b; };
    auto result = try_catch(add, 10, 20);

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 30);
}

TEST_F(TryCatchTest, TryCatchFemException) {
    auto result = try_catch([]() -> int {
        throw InvalidArgumentError("param", "invalid value");
    });

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error().error_code(), ErrorCode::InvalidArgument);
    EXPECT_TRUE(result.error().message().find("invalid value") != std::string::npos);
}

TEST_F(TryCatchTest, TryCatchStdException) {
    auto result = try_catch([]() -> int {
        throw std::runtime_error("standard error");
    });

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error().error_code(), ErrorCode::Unknown);
    EXPECT_TRUE(result.error().message().find("standard error") != std::string::npos);
}

TEST_F(TryCatchTest, TryCatchUnknownException) {
    auto result = try_catch([]() -> int {
        throw 42;  // Non-exception type
    });

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error().error_code(), ErrorCode::Unknown);
    EXPECT_TRUE(result.error().message().find("Unknown exception") != std::string::npos);
}

// try_finally tests
TEST_F(TryCatchTest, TryFinallySuccess) {
    bool cleanup_called = false;
    int result = try_finally(
        []() { return 42; },
        [&]() { cleanup_called = true; }
    );

    EXPECT_EQ(result, 42);
    EXPECT_TRUE(cleanup_called);
}

TEST_F(TryCatchTest, TryFinallyException) {
    bool cleanup_called = false;

    EXPECT_THROW({
        try_finally(
            []() -> int { throw std::runtime_error("test"); },
            [&]() { cleanup_called = true; }
        );
    }, std::runtime_error);

    EXPECT_TRUE(cleanup_called);
}

TEST_F(TryCatchTest, TryFinallyVoidFunction) {
    bool executed = false;
    bool cleanup_called = false;

    try_finally(
        [&]() { executed = true; },
        [&]() { cleanup_called = true; }
    );

    EXPECT_TRUE(executed);
    EXPECT_TRUE(cleanup_called);
}

TEST_F(TryCatchTest, TryFinallyCleanupException) {
    bool main_executed = false;

    // Cleanup throwing shouldn't affect main function result
    int result = try_finally(
        [&]() { main_executed = true; return 42; },
        []() { throw std::runtime_error("cleanup error"); }
    );

    EXPECT_EQ(result, 42);
    EXPECT_TRUE(main_executed);
}

// try_on_error tests
TEST_F(TryCatchTest, TryOnErrorSuccess) {
    bool cleanup_called = false;
    int result = try_on_error(
        []() { return 42; },
        [&]() { cleanup_called = true; }
    );

    EXPECT_EQ(result, 42);
    EXPECT_FALSE(cleanup_called);  // Should not call cleanup on success
}

TEST_F(TryCatchTest, TryOnErrorException) {
    bool cleanup_called = false;

    EXPECT_THROW({
        try_on_error(
            []() -> int { throw std::runtime_error("test"); },
            [&]() { cleanup_called = true; }
        );
    }, std::runtime_error);

    EXPECT_TRUE(cleanup_called);
}

TEST_F(TryCatchTest, TryOnErrorCleanupException) {
    bool cleanup_called = false;

    // Original exception should be preserved even if cleanup throws
    EXPECT_THROW({
        try_on_error(
            []() -> int { throw std::runtime_error("original"); },
            [&]() {
                cleanup_called = true;
                throw std::logic_error("cleanup error");
            }
        );
    }, std::runtime_error);

    EXPECT_TRUE(cleanup_called);
}

// try_with_retry tests
TEST_F(TryCatchTest, TryWithRetrySuccess) {
    int attempt_count = 0;
    auto result = try_with_retry([&]() {
        attempt_count++;
        return 42;
    });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(attempt_count, 1);
}

TEST_F(TryCatchTest, TryWithRetryEventualSuccess) {
    int attempt_count = 0;
    auto result = try_with_retry([&]() {
        attempt_count++;
        if (attempt_count < 3) {
            throw std::runtime_error("temporary failure");
        }
        return 42;
    }, 5, std::chrono::milliseconds(1));

    EXPECT_EQ(result, 42);
    EXPECT_EQ(attempt_count, 3);
}

TEST_F(TryCatchTest, TryWithRetryAllFail) {
    int attempt_count = 0;

    EXPECT_THROW({
        try_with_retry([&]() {
            attempt_count++;
            throw std::runtime_error("persistent failure");
        }, 3, std::chrono::milliseconds(1));
    }, std::runtime_error);

    EXPECT_EQ(attempt_count, 3);
}

TEST_F(TryCatchTest, TryWithRetryVoidFunction) {
    int attempt_count = 0;

    try_with_retry([&]() {
        attempt_count++;
        if (attempt_count < 2) {
            throw std::runtime_error("temporary failure");
        }
    }, 3, std::chrono::milliseconds(1));

    EXPECT_EQ(attempt_count, 2);
}

// unwrap_or_throw tests
TEST_F(TryCatchTest, UnwrapOrThrowSuccess) {
    auto ok_result = Ok<int, ErrorInfo>(42);
    auto value = unwrap_or_throw(std::move(ok_result));

    EXPECT_EQ(value, 42);
}

TEST_F(TryCatchTest, UnwrapOrThrowErrorInfo) {
    auto error_result = Err<ErrorInfo, int>(
        make_error(ErrorCode::InvalidArgument, "test error")
    );

    EXPECT_THROW({
        unwrap_or_throw(std::move(error_result));
    }, Exception);
}

TEST_F(TryCatchTest, UnwrapOrThrowGenericError) {
    auto error_result = Err<ErrorCode, int>(ErrorCode::Unknown);

    EXPECT_THROW({
        unwrap_or_throw(std::move(error_result));
    }, RuntimeError);
}

// try_chain tests
TEST_F(TryCatchTest, TryChainSuccess) {
    auto add_one = [](int x) -> Result<int, ErrorCode> {
        return Ok<int, ErrorCode>(x + 1);
    };

    auto multiply_two = [](int x) -> Result<int, ErrorCode> {
        return Ok<int, ErrorCode>(x * 2);
    };

    auto chain = try_chain(add_one, multiply_two);
    auto result = chain(Ok<int, ErrorCode>(5));

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 12);  // (5 + 1) * 2
}

TEST_F(TryCatchTest, TryChainFailure) {
    auto add_one = [](int x) -> Result<int, ErrorCode> {
        return Ok<int, ErrorCode>(x + 1);
    };

    auto fail_op = [](int) -> Result<int, ErrorCode> {
        return Err<ErrorCode, int>(ErrorCode::InvalidArgument);
    };

    auto multiply_two = [](int x) -> Result<int, ErrorCode> {
        return Ok<int, ErrorCode>(x * 2);
    };

    auto chain = try_chain(add_one, fail_op, multiply_two);
    auto result = chain(Ok<int, ErrorCode>(5));

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

// try_collect tests
TEST_F(TryCatchTest, TryCollectMultipleOperations) {
    auto op1 = []() { return 42; };
    auto op2 = []() { return std::string("hello"); };
    auto op3 = []() { return 3.14; };

    auto results = try_collect(op1, op2, op3);

    EXPECT_EQ(std::get<0>(results), 42);
    EXPECT_EQ(std::get<1>(results), "hello");
    EXPECT_EQ(std::get<2>(results), 3.14);
}

// TryScope tests
TEST_F(TryCatchTest, TryScopeBasic) {
    bool error_handler_called = false;

    {
        TryScope scope([&]() {
            error_handler_called = true;
        });

        EXPECT_FALSE(scope.has_error());
        scope.set_error();
        EXPECT_TRUE(scope.has_error());
    }

    EXPECT_TRUE(error_handler_called);
}

TEST_F(TryCatchTest, TryScopeNoError) {
    bool error_handler_called = false;

    {
        TryScope scope([&]() {
            error_handler_called = true;
        });
        // Don't set error
    }

    EXPECT_FALSE(error_handler_called);
}

TEST_F(TryCatchTest, TryScopeClearError) {
    bool error_handler_called = false;

    {
        TryScope scope([&]() {
            error_handler_called = true;
        });

        scope.set_error();
        EXPECT_TRUE(scope.has_error());

        scope.clear_error();
        EXPECT_FALSE(scope.has_error());
    }

    EXPECT_FALSE(error_handler_called);
}

TEST_F(TryCatchTest, TryScopeHandlerException) {
    // Test that exceptions in error handler don't propagate
    EXPECT_NO_THROW({
        TryScope scope([]() {
            throw std::runtime_error("handler error");
        });
        scope.set_error();
    });
}

// Complex integration tests
TEST_F(TryCatchTest, NestedTryOperations) {
    auto result = try_catch([&]() {
        return try_finally(
            []() {
                return try_with_retry([]() {
                    static int attempts = 0;
                    attempts++;
                    if (attempts < 2) {
                        throw std::runtime_error("temporary");
                    }
                    return 42;
                }, 3, std::chrono::milliseconds(1));
            },
            []() {
                // Cleanup
            }
        );
    });

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(TryCatchTest, TryCatchWithCustomException) {
    class CustomException : public std::exception {
    public:
        const char* what() const noexcept override {
            return "custom exception";
        }
    };

    auto result = try_catch([]() -> int {
        throw CustomException();
    });

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error().error_code(), ErrorCode::Unknown);
    EXPECT_TRUE(result.error().message().find("custom exception") != std::string::npos);
}

TEST_F(TryCatchTest, ChainedErrorPropagation) {
    auto operation = [](int input) -> Result<int, ErrorInfo> {
        return try_catch([input]() {
            if (input < 0) {
                throw InvalidArgumentError("input", "must be non-negative");
            }
            return input * 2;
        });
    };

    auto result1 = operation(5);
    EXPECT_TRUE(result1.is_ok());
    EXPECT_EQ(result1.value(), 10);

    auto result2 = operation(-1);
    EXPECT_TRUE(result2.is_error());
    EXPECT_EQ(result2.error().error_code(), ErrorCode::InvalidArgument);
}

TEST_F(TryCatchTest, ResourceManagementPattern) {
    struct Resource {
        bool& destroyed;
        explicit Resource(bool& d) : destroyed(d) {}
        ~Resource() { destroyed = true; }
    };

    bool resource_destroyed = false;

    auto result = try_catch([&]() {
        auto resource = std::make_unique<Resource>(resource_destroyed);

        return try_finally(
            [&]() {
                // Simulate work that might throw
                return resource.get() ? 42 : 0;
            },
            [&]() {
                // Cleanup - resource will be destroyed automatically
                // but we can do additional cleanup here
            }
        );
    });

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 42);
    EXPECT_TRUE(resource_destroyed);
}

TEST_F(TryCatchTest, RetryWithExponentialBackoff) {
    auto start_time = std::chrono::steady_clock::now();
    int attempt_count = 0;

    try {
        try_with_retry([&]() {
            attempt_count++;
            throw std::runtime_error("always fails");
        }, 3, std::chrono::milliseconds(10));
    } catch (...) {
        // Expected to fail
    }

    auto duration = std::chrono::steady_clock::now() - start_time;

    EXPECT_EQ(attempt_count, 3);
    // Should have waited at least: 10ms + 20ms = 30ms
    EXPECT_GE(duration, std::chrono::milliseconds(30));
}

// Error message propagation tests
TEST_F(TryCatchTest, ErrorMessagePreservation) {
    const std::string original_message = "detailed error information";

    auto result = try_catch([&]() -> int {
        throw RuntimeError(original_message);
    });

    EXPECT_TRUE(result.is_error());
    EXPECT_TRUE(result.error().message().find(original_message) != std::string::npos);
}

TEST_F(TryCatchTest, SourceLocationPreservation) {
    auto result = try_catch([]() -> int {
        throw InvalidArgumentError("param", "invalid");
    });

    EXPECT_TRUE(result.is_error());
    // Source location should be preserved in ErrorInfo
    EXPECT_TRUE(result.error().location().line() > 0);
}

// Performance and stress tests
TEST_F(TryCatchTest, HighFrequencyRetry) {
    std::atomic<int> success_count{0};

    for (int i = 0; i < 100; ++i) {
        try {
            try_with_retry([&]() {
                if (i % 10 == 0) {
                    throw std::runtime_error("occasional failure");
                }
                success_count++;
            }, 2, std::chrono::milliseconds(1));
        } catch (...) {
            // Some operations expected to fail
        }
    }

    EXPECT_GT(success_count.load(), 80);  // Most should succeed
}
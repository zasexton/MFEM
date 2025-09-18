#include <gtest/gtest.h>
#include <core/error/expected.h>
#include <core/error/error_code.h>
#include <core/error/exception_base.h>
#include <string>
#include <memory>
#include <vector>

using namespace fem::core::error;

class ExpectedTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test exception for testing
class TestException : public std::runtime_error {
public:
    explicit TestException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// Basic Expected construction tests
TEST_F(ExpectedTest, ConstructWithValue) {
    Expected<int> exp(42);

    EXPECT_TRUE(exp.has_value());
    EXPECT_FALSE(exp.has_error());
    EXPECT_EQ(exp.value(), 42);
}

TEST_F(ExpectedTest, ConstructWithError) {
    auto error = std::make_exception_ptr(TestException("test error"));
    Expected<int> exp(error);

    EXPECT_FALSE(exp.has_value());
    EXPECT_TRUE(exp.has_error());
    EXPECT_THROW(exp.value(), TestException);
}

TEST_F(ExpectedTest, FactoryMethods) {
    auto success = Expected<int>::success(100);
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(success.value(), 100);

    auto failure = Expected<int>::failure(
        std::make_exception_ptr(TestException("failed"))
    );
    EXPECT_TRUE(failure.has_error());
    EXPECT_THROW(failure.value(), TestException);
}

// try_invoke tests
TEST_F(ExpectedTest, TryInvokeSuccess) {
    auto result = Expected<int>::try_invoke([]() {
        return 42;
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(ExpectedTest, TryInvokeThrows) {
    auto result = Expected<int>::try_invoke([]() -> int {
        throw TestException("invoke failed");
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_THROW(result.rethrow(), TestException);
}

TEST_F(ExpectedTest, TryInvokeWithArguments) {
    auto add = [](int a, int b) { return a + b; };
    auto result = Expected<int>::try_invoke(add, 10, 20);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 30);
}

TEST_F(ExpectedTest, TryInvokeVoidFunction) {
    int counter = 0;
    auto result = Expected<std::monostate>::try_invoke([&counter]() {
        counter = 42;
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(counter, 42);
}

// Value access tests
TEST_F(ExpectedTest, ValueAccess) {
    Expected<std::string> exp("test");

    EXPECT_EQ(exp.value(), "test");
    EXPECT_EQ(exp.get(), "test");

    const Expected<std::string> const_exp("const test");
    EXPECT_EQ(const_exp.value(), "const test");
}

TEST_F(ExpectedTest, ValueAccessOnError) {
    auto error = std::make_exception_ptr(TestException("no value"));
    Expected<int> exp(error);

    EXPECT_THROW(exp.value(), TestException);
    EXPECT_THROW(exp.get(), TestException);
}

TEST_F(ExpectedTest, ValueOr) {
    Expected<int> success(42);
    EXPECT_EQ(success.value_or(100), 42);

    auto error = std::make_exception_ptr(TestException("error"));
    Expected<int> failure(error);
    EXPECT_EQ(failure.value_or(100), 100);
}

TEST_F(ExpectedTest, ValueOrElse) {
    Expected<int> success(42);
    auto default_fn = []() { return 100; };
    EXPECT_EQ(success.value_or_else(default_fn), 42);

    auto error = std::make_exception_ptr(TestException("error"));
    Expected<int> failure(error);
    EXPECT_EQ(failure.value_or_else(default_fn), 100);
}

// Error access tests
TEST_F(ExpectedTest, ErrorAccess) {
    auto original_error = std::make_exception_ptr(TestException("test"));
    Expected<int> exp(original_error);

    auto retrieved_error = exp.error();
    EXPECT_EQ(retrieved_error, original_error);
}

TEST_F(ExpectedTest, ErrorAccessOnSuccess) {
    Expected<int> exp(42);
    EXPECT_THROW(exp.error(), LogicError);
}

TEST_F(ExpectedTest, Rethrow) {
    auto error = std::make_exception_ptr(TestException("rethrow test"));
    Expected<int> exp(error);

    EXPECT_THROW({
        try {
            exp.rethrow();
        } catch (const TestException& e) {
            EXPECT_STREQ(e.what(), "rethrow test");
            throw;
        }
    }, TestException);
}

TEST_F(ExpectedTest, RethrowOnSuccess) {
    Expected<int> exp(42);
    EXPECT_THROW(exp.rethrow(), LogicError);
}

// Boolean conversion tests
TEST_F(ExpectedTest, BooleanConversion) {
    Expected<int> success(42);
    EXPECT_TRUE(success);
    EXPECT_TRUE(static_cast<bool>(success));

    auto error = std::make_exception_ptr(TestException("error"));
    Expected<int> failure(error);
    EXPECT_FALSE(failure);
    EXPECT_FALSE(static_cast<bool>(failure));
}

// Monadic operations tests
TEST_F(ExpectedTest, MapSuccess) {
    Expected<int> exp(42);

    auto mapped = exp.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 84);
}

TEST_F(ExpectedTest, MapError) {
    auto error = std::make_exception_ptr(TestException("error"));
    Expected<int> exp(error);

    auto mapped = exp.map([](int x) { return x * 2; });

    EXPECT_FALSE(mapped.has_value());
    EXPECT_THROW(mapped.rethrow(), TestException);
}

TEST_F(ExpectedTest, MapThrows) {
    Expected<int> exp(42);

    auto mapped = exp.map([](int) -> int {
        throw TestException("map failed");
    });

    EXPECT_FALSE(mapped.has_value());
    EXPECT_THROW(mapped.rethrow(), TestException);
}

TEST_F(ExpectedTest, MapTypeChange) {
    Expected<int> exp(42);

    auto mapped = exp.map([](int x) {
        return std::to_string(x);
    });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), "42");
}

TEST_F(ExpectedTest, AndThenSuccess) {
    Expected<int> exp(10);

    auto result = exp.and_then([](int x) {
        return Expected<std::string>(std::to_string(x * 2));
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "20");
}

TEST_F(ExpectedTest, AndThenError) {
    auto error = std::make_exception_ptr(TestException("initial error"));
    Expected<int> exp(error);

    auto result = exp.and_then([](int x) {
        return Expected<std::string>(std::to_string(x));
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_THROW(result.rethrow(), TestException);
}

TEST_F(ExpectedTest, AndThenReturnsError) {
    Expected<int> exp(42);

    auto result = exp.and_then([](int) {
        auto err = std::make_exception_ptr(TestException("and_then error"));
        return Expected<std::string>(err);
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_THROW(result.rethrow(), TestException);
}

TEST_F(ExpectedTest, OrElseRecovers) {
    auto error = std::make_exception_ptr(TestException("original"));
    Expected<int> exp(error);

    auto recovered = exp.or_else([](std::exception_ptr) {
        return 42;  // Recover with a value
    });

    EXPECT_TRUE(recovered.has_value());
    EXPECT_EQ(recovered.value(), 42);
}

TEST_F(ExpectedTest, OrElseOnSuccess) {
    Expected<int> exp(42);

    auto result = exp.or_else([](std::exception_ptr) {
        return 100;
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);  // Original value unchanged
}

TEST_F(ExpectedTest, OrElseThrows) {
    auto error = std::make_exception_ptr(TestException("original"));
    Expected<int> exp(error);

    auto result = exp.or_else([](std::exception_ptr) -> int {
        throw TestException("or_else failed");
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_THROW(result.rethrow(), TestException);
}

// Conversion tests
TEST_F(ExpectedTest, ToResultSuccess) {
    Expected<int> exp(42);

    auto result = exp.to_result<ErrorCode>();

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(ExpectedTest, ToResultWithFemException) {
    auto error = std::make_exception_ptr(
        InvalidArgumentError("arg_name", "invalid")
    );
    Expected<int> exp(error);

    auto result = exp.to_result<ErrorCode>();

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST_F(ExpectedTest, ToResultWithStdException) {
    auto error = std::make_exception_ptr(
        std::runtime_error("runtime error")
    );
    Expected<int> exp(error);

    auto result = exp.to_result<ErrorCode>();

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::Unknown);
}

TEST_F(ExpectedTest, ToResultWithUnknownException) {
    auto error = std::make_exception_ptr(42);  // Non-exception type
    Expected<int> exp(error);

    auto result = exp.to_result<ErrorCode>();

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::Unknown);
}

// Move semantics tests
TEST_F(ExpectedTest, MoveConstructor) {
    Expected<std::unique_ptr<int>> exp1(std::make_unique<int>(42));
    Expected<std::unique_ptr<int>> exp2(std::move(exp1));

    EXPECT_TRUE(exp2.has_value());
    EXPECT_NE(exp2.value(), nullptr);
    EXPECT_EQ(*exp2.value(), 42);
}

TEST_F(ExpectedTest, MoveValue) {
    Expected<std::unique_ptr<int>> exp(std::make_unique<int>(42));

    auto ptr = std::move(exp).value();

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, 42);
}

// Complex type tests
TEST_F(ExpectedTest, VectorExpected) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    Expected<std::vector<int>> exp(vec);

    EXPECT_TRUE(exp.has_value());
    EXPECT_EQ(exp.value().size(), 5);
    EXPECT_EQ(exp.value()[0], 1);
}

TEST_F(ExpectedTest, StringExpected) {
    Expected<std::string> exp("hello world");

    EXPECT_TRUE(exp.has_value());
    EXPECT_EQ(exp.value(), "hello world");

    auto mapped = exp.map([](const std::string& s) {
        return s.length();
    });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 11);
}

// Helper function tests
TEST_F(ExpectedTest, MakeExpected) {
    auto exp = make_expected(42);

    EXPECT_TRUE(exp.has_value());
    EXPECT_EQ(exp.value(), 42);
}

TEST_F(ExpectedTest, MakeExpectedError) {
    auto error = std::make_exception_ptr(TestException("error"));
    auto exp = make_expected_error<int>(error);

    EXPECT_FALSE(exp.has_value());
    EXPECT_THROW(exp.rethrow(), TestException);
}

// Real-world usage examples
TEST_F(ExpectedTest, FileReadExample) {
    auto read_file = [](const std::string& path) -> Expected<std::string> {
        if (path.empty()) {
            return Expected<std::string>(
                std::make_exception_ptr(InvalidArgumentError("path", "Empty path"))
            );
        }
        if (path == "/not/found") {
            return Expected<std::string>(
                std::make_exception_ptr(RuntimeError("File not found"))
            );
        }
        return Expected<std::string>::success("file contents");
    };

    auto result1 = read_file("/valid/path");
    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "file contents");

    auto result2 = read_file("");
    EXPECT_FALSE(result2.has_value());
    EXPECT_THROW(result2.rethrow(), InvalidArgumentError);

    auto result3 = read_file("/not/found");
    EXPECT_FALSE(result3.has_value());
    EXPECT_THROW(result3.rethrow(), RuntimeError);
}

TEST_F(ExpectedTest, ChainedOperations) {
    auto parse_int = [](const std::string& s) -> Expected<int> {
        return Expected<int>::try_invoke([&s]() {
            return std::stoi(s);
        });
    };

    auto validate_positive = [](int x) -> Expected<int> {
        if (x > 0) {
            return Expected<int>::success(x);
        }
        return Expected<int>::failure(
            std::make_exception_ptr(LogicError("Must be positive"))
        );
    };

    auto process = [&](const std::string& input) -> Expected<int> {
        return parse_int(input)
            .and_then(validate_positive)
            .map([](int x) { return x * 2; });
    };

    auto result1 = process("42");
    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), 84);

    auto result2 = process("-5");
    EXPECT_FALSE(result2.has_value());
    EXPECT_THROW(result2.rethrow(), LogicError);

    auto result3 = process("abc");
    EXPECT_FALSE(result3.has_value());
    EXPECT_THROW(result3.rethrow(), std::invalid_argument);
}

TEST_F(ExpectedTest, ExceptionRecovery) {
    auto risky_operation = []() -> Expected<int> {
        return Expected<int>::try_invoke([]() -> int {
            throw TestException("operation failed");
        });
    };

    auto with_fallback = risky_operation()
        .or_else([](std::exception_ptr) {
            return 42;  // Fallback value
        });

    EXPECT_TRUE(with_fallback.has_value());
    EXPECT_EQ(with_fallback.value(), 42);
}
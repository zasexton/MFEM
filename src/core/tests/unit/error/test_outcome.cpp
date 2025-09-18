#include <gtest/gtest.h>
#include <core/error/outcome.h>
#include <core/error/error_code.h>
#include <core/error/exception_base.h>
#include <string>
#include <memory>
#include <vector>

using namespace fem::core::error;

class OutcomeTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test exception for testing
class TestException : public std::runtime_error {
public:
    explicit TestException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// Simple error enum for testing
enum class TestErrorCode : int {
    Success = 0,
    Error1 = 1,
    Error2 = 2
};

// Basic Outcome construction tests
TEST_F(OutcomeTest, ConstructWithValue) {
    Outcome<int> outcome(42);

    EXPECT_TRUE(outcome.has_value());
    EXPECT_FALSE(outcome.has_error());
    EXPECT_FALSE(outcome.has_exception());
    EXPECT_FALSE(outcome.has_failure());
    EXPECT_EQ(outcome.value(), 42);
}

TEST_F(OutcomeTest, ConstructWithError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));

    EXPECT_FALSE(outcome.has_value());
    EXPECT_TRUE(outcome.has_error());
    EXPECT_FALSE(outcome.has_exception());
    EXPECT_TRUE(outcome.has_failure());
    EXPECT_EQ(outcome.error(), ErrorCode::InvalidArgument);
}

TEST_F(OutcomeTest, ConstructWithException) {
    auto ex = std::make_exception_ptr(TestException("test error"));
    Outcome<int> outcome(ex);

    EXPECT_FALSE(outcome.has_value());
    EXPECT_FALSE(outcome.has_error());
    EXPECT_TRUE(outcome.has_exception());
    EXPECT_TRUE(outcome.has_failure());
    EXPECT_EQ(outcome.exception(), ex);
}

TEST_F(OutcomeTest, FactoryMethods) {
    auto success = Outcome<int, ErrorCode>::success(100);
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(success.value(), 100);

    auto failure = Outcome<int, ErrorCode>::failure(ErrorCode::ResourceNotFound);
    EXPECT_TRUE(failure.has_error());
    EXPECT_EQ(failure.error(), ErrorCode::ResourceNotFound);

    auto exception = Outcome<int, ErrorCode>::exception(
        std::make_exception_ptr(TestException("error"))
    );
    EXPECT_TRUE(exception.has_exception());
}

// from_call tests
TEST_F(OutcomeTest, FromCallSuccess) {
    auto outcome = Outcome<int>::from_call([]() {
        return 42;
    });

    EXPECT_TRUE(outcome.has_value());
    EXPECT_EQ(outcome.value(), 42);
}

TEST_F(OutcomeTest, FromCallThrowsFemException) {
    auto outcome = Outcome<int, ErrorCode>::from_call([]() -> int {
        throw InvalidArgumentError("arg_name", "invalid");
    });

    EXPECT_FALSE(outcome.has_value());
    EXPECT_TRUE(outcome.has_error());
    EXPECT_EQ(outcome.error(), ErrorCode::InvalidArgument);
}

TEST_F(OutcomeTest, FromCallThrowsOtherException) {
    auto outcome = Outcome<int>::from_call([]() -> int {
        throw TestException("test error");
    });

    EXPECT_FALSE(outcome.has_value());
    EXPECT_FALSE(outcome.has_error());
    EXPECT_TRUE(outcome.has_exception());

    EXPECT_THROW({
        try {
            std::rethrow_exception(outcome.exception());
        } catch (const TestException& e) {
            EXPECT_STREQ(e.what(), "test error");
            throw;
        }
    }, TestException);
}

TEST_F(OutcomeTest, FromCallVoidFunction) {
    int counter = 0;
    auto outcome = Outcome<std::monostate>::from_call([&counter]() {
        counter = 42;
    });

    EXPECT_TRUE(outcome.has_value());
    EXPECT_EQ(counter, 42);
}

// Boolean conversion tests
TEST_F(OutcomeTest, BooleanConversion) {
    Outcome<int> success(42);
    EXPECT_TRUE(success);
    EXPECT_TRUE(static_cast<bool>(success));

    Outcome<int, ErrorCode> error_outcome(Error<ErrorCode>(ErrorCode::ResourceNotFound));
    EXPECT_FALSE(error_outcome);
    EXPECT_FALSE(static_cast<bool>(error_outcome));

    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> exception_outcome(ex);
    EXPECT_FALSE(exception_outcome);
    EXPECT_FALSE(static_cast<bool>(exception_outcome));
}

// Value access tests
TEST_F(OutcomeTest, ValueAccess) {
    Outcome<std::string> outcome("test");

    EXPECT_EQ(outcome.value(), "test");

    const Outcome<std::string> const_outcome("const test");
    EXPECT_EQ(const_outcome.value(), "const test");
}

TEST_F(OutcomeTest, ValueAccessOnError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::ResourceNotFound));

    EXPECT_THROW(outcome.value(), RuntimeError);
}

TEST_F(OutcomeTest, ValueAccessOnException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> outcome(ex);

    EXPECT_THROW(outcome.value(), TestException);
}

TEST_F(OutcomeTest, ValueOr) {
    Outcome<int> success(42);
    EXPECT_EQ(success.value_or(100), 42);

    Outcome<int, ErrorCode> failure(Error<ErrorCode>(ErrorCode::ResourceNotFound));
    EXPECT_EQ(failure.value_or(100), 100);

    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> exception(ex);
    EXPECT_EQ(exception.value_or(100), 100);
}

TEST_F(OutcomeTest, ValueOrElse) {
    Outcome<int> success(42);
    auto default_fn = []() { return 100; };
    EXPECT_EQ(success.value_or_else(default_fn), 42);

    Outcome<int, ErrorCode> failure(Error<ErrorCode>(ErrorCode::ResourceNotFound));
    EXPECT_EQ(failure.value_or_else(default_fn), 100);

    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> exception(ex);
    EXPECT_EQ(exception.value_or_else(default_fn), 100);
}

// Error access tests
TEST_F(OutcomeTest, ErrorAccess) {
    Outcome<int> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));

    EXPECT_EQ(outcome.error(), ErrorCode::InvalidArgument);

    const Outcome<int> const_outcome(Error<ErrorCode>(ErrorCode::OutOfRange));
    EXPECT_EQ(const_outcome.error(), ErrorCode::OutOfRange);
}

TEST_F(OutcomeTest, ErrorAccessOnValue) {
    Outcome<int> outcome(42);
    EXPECT_THROW(outcome.error(), LogicError);
}

TEST_F(OutcomeTest, ErrorAccessOnException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int, ErrorCode> outcome(ex);
    EXPECT_THROW(outcome.error(), LogicError);
}

// Exception access tests
TEST_F(OutcomeTest, ExceptionAccess) {
    auto original_ex = std::make_exception_ptr(TestException("test"));
    Outcome<int> outcome(original_ex);

    auto retrieved_ex = outcome.exception();
    EXPECT_EQ(retrieved_ex, original_ex);
}

TEST_F(OutcomeTest, ExceptionAccessOnValue) {
    Outcome<int> outcome(42);
    EXPECT_THROW(outcome.exception(), LogicError);
}

TEST_F(OutcomeTest, ExceptionAccessOnError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::ResourceNotFound));
    EXPECT_THROW(outcome.exception(), LogicError);
}

// Conversion tests
TEST_F(OutcomeTest, ToResultFromValue) {
    Outcome<int> outcome(42);
    auto result = outcome.to_result();

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(OutcomeTest, ToResultFromError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));
    auto result = outcome.to_result();

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST_F(OutcomeTest, ToResultFromFemException) {
    // Note: There's no NotFoundError in the actual API, using RuntimeError instead
    auto ex = std::make_exception_ptr(RuntimeError("not found"));
    Outcome<int, ErrorCode> outcome(ex);
    auto result = outcome.to_result();

    EXPECT_TRUE(result.is_error());
    // RuntimeError defaults to Unknown error code
    EXPECT_EQ(result.error(), ErrorCode::Unknown);
}

TEST_F(OutcomeTest, ToResultFromOtherException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int, ErrorCode> outcome(ex);
    auto result = outcome.to_result();

    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::Unknown);
}

TEST_F(OutcomeTest, ToExpectedFromValue) {
    Outcome<int> outcome(42);
    auto expected = outcome.to_expected();

    EXPECT_TRUE(expected.has_value());
    EXPECT_EQ(expected.value(), 42);
}

TEST_F(OutcomeTest, ToExpectedFromException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> outcome(ex);
    auto expected = outcome.to_expected();

    EXPECT_FALSE(expected.has_value());
    EXPECT_TRUE(expected.has_error());
    EXPECT_EQ(expected.error(), ex);
}

TEST_F(OutcomeTest, ToExpectedFromError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));
    auto expected = outcome.to_expected();

    EXPECT_FALSE(expected.has_value());
    EXPECT_TRUE(expected.has_error());
    EXPECT_THROW(expected.rethrow(), RuntimeError);
}

// Monadic operations tests
TEST_F(OutcomeTest, MapSuccess) {
    Outcome<int> outcome(42);

    auto mapped = outcome.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 84);
}

TEST_F(OutcomeTest, MapWithError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));

    auto mapped = outcome.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error(), ErrorCode::InvalidArgument);
}

TEST_F(OutcomeTest, MapException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int> outcome(ex);

    auto mapped = outcome.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.has_exception());
    EXPECT_EQ(mapped.exception(), ex);
}

TEST_F(OutcomeTest, MapThrows) {
    Outcome<int> outcome(42);

    auto mapped = outcome.map([](int) -> int {
        throw TestException("map failed");
    });

    EXPECT_TRUE(mapped.has_exception());
    EXPECT_THROW(std::rethrow_exception(mapped.exception()), TestException);
}

TEST_F(OutcomeTest, MapTypeChange) {
    Outcome<int> outcome(42);

    auto mapped = outcome.map([](int x) {
        return std::to_string(x);
    });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), "42");
}

TEST_F(OutcomeTest, MapErrorFunction) {
    Outcome<int, TestErrorCode> outcome(Error<TestErrorCode>(TestErrorCode::Error1));

    auto mapped = outcome.map_error([](TestErrorCode e) {
        return e == TestErrorCode::Error1 ? TestErrorCode::Error2 : e;
    });

    EXPECT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error(), TestErrorCode::Error2);
}

TEST_F(OutcomeTest, MapErrorOnValue) {
    Outcome<int, TestErrorCode> outcome(42);

    auto mapped = outcome.map_error([](TestErrorCode e) {
        return e == TestErrorCode::Error1 ? TestErrorCode::Error2 : e;
    });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 42);
}

TEST_F(OutcomeTest, AndThenSuccess) {
    Outcome<int> outcome(10);

    auto result = outcome.and_then([](int x) {
        return Outcome<std::string>(std::to_string(x * 2));
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "20");
}

TEST_F(OutcomeTest, AndThenError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::InvalidArgument));

    auto result = outcome.and_then([](int x) {
        return Outcome<std::string, ErrorCode>(std::to_string(x));
    });

    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST_F(OutcomeTest, AndThenException) {
    auto ex = std::make_exception_ptr(TestException("error"));
    Outcome<int, ErrorCode> outcome(ex);

    auto result = outcome.and_then([](int x) {
        return Outcome<std::string, ErrorCode>(std::to_string(x));
    });

    EXPECT_TRUE(result.has_exception());
    EXPECT_EQ(result.exception(), ex);
}

// Visit tests
TEST_F(OutcomeTest, VisitValue) {
    Outcome<int> outcome(42);

    auto result = outcome.visit(
        [](int v) { return std::to_string(v); },
        [](ErrorCode) { return std::string("error"); },
        [](std::exception_ptr) { return std::string("exception"); }
    );

    EXPECT_EQ(result, "42");
}

TEST_F(OutcomeTest, VisitError) {
    Outcome<int, ErrorCode> outcome(Error<ErrorCode>(ErrorCode::ResourceNotFound));

    auto result = outcome.visit(
        [](int) { return std::string("value"); },
        [](ErrorCode e) {
            CoreErrorCategory category;
            return category.message(static_cast<int>(e));
        },
        [](std::exception_ptr) { return std::string("exception"); }
    );

    EXPECT_EQ(result, "Resource not found");
}

TEST_F(OutcomeTest, VisitException) {
    auto ex = std::make_exception_ptr(TestException("test"));
    Outcome<int> outcome(ex);

    auto result = outcome.visit(
        [](int) { return std::string("value"); },
        [](ErrorCode) { return std::string("error"); },
        [](std::exception_ptr) { return std::string("exception"); }
    );

    EXPECT_EQ(result, "exception");
}

// Move semantics tests
TEST_F(OutcomeTest, MoveConstructor) {
    Outcome<std::unique_ptr<int>> outcome1(std::make_unique<int>(42));
    Outcome<std::unique_ptr<int>> outcome2(std::move(outcome1));

    EXPECT_TRUE(outcome2.has_value());
    EXPECT_NE(outcome2.value(), nullptr);
    EXPECT_EQ(*outcome2.value(), 42);
}

TEST_F(OutcomeTest, MoveValue) {
    Outcome<std::unique_ptr<int>> outcome(std::make_unique<int>(42));

    auto ptr = std::move(outcome).value();

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, 42);
}

// Complex type tests
TEST_F(OutcomeTest, VectorOutcome) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    Outcome<std::vector<int>> outcome(vec);

    EXPECT_TRUE(outcome.has_value());
    EXPECT_EQ(outcome.value().size(), 5);
    EXPECT_EQ(outcome.value()[0], 1);
}

TEST_F(OutcomeTest, StringOutcome) {
    Outcome<std::string> outcome("hello world");

    EXPECT_TRUE(outcome.has_value());
    EXPECT_EQ(outcome.value(), "hello world");

    auto mapped = outcome.map([](const std::string& s) {
        return s.length();
    });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 11);
}

// Helper function tests
TEST_F(OutcomeTest, MakeOutcome) {
    auto outcome = make_outcome(42);

    EXPECT_TRUE(outcome.has_value());
    EXPECT_EQ(outcome.value(), 42);
}

TEST_F(OutcomeTest, MakeErrorOutcome) {
    auto outcome = make_error_outcome<ErrorCode, int>(ErrorCode::ResourceNotFound);

    EXPECT_TRUE(outcome.has_error());
    EXPECT_EQ(outcome.error(), ErrorCode::ResourceNotFound);
}

TEST_F(OutcomeTest, MakeExceptionOutcome) {
    auto ex = std::make_exception_ptr(TestException("error"));
    auto outcome = make_exception_outcome<int>(ex);

    EXPECT_TRUE(outcome.has_exception());
    EXPECT_EQ(outcome.exception(), ex);
}

// Real-world usage examples
TEST_F(OutcomeTest, FileOperationExample) {
    auto read_file = [](const std::string& path) -> Outcome<std::string, ErrorCode> {
        if (path.empty()) {
            return Outcome<std::string, ErrorCode>::failure(ErrorCode::InvalidArgument);
        }
        if (path == "/not/found") {
            return Outcome<std::string, ErrorCode>::failure(ErrorCode::FileNotFound);
        }
        if (path == "/corrupt") {
            return Outcome<std::string, ErrorCode>::exception(
                std::make_exception_ptr(TestException("File corrupted"))
            );
        }
        return Outcome<std::string, ErrorCode>::success("file contents");
    };

    auto result1 = read_file("/valid/path");
    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "file contents");

    auto result2 = read_file("");
    EXPECT_TRUE(result2.has_error());
    EXPECT_EQ(result2.error(), ErrorCode::InvalidArgument);

    auto result3 = read_file("/not/found");
    EXPECT_TRUE(result3.has_error());
    EXPECT_EQ(result3.error(), ErrorCode::FileNotFound);

    auto result4 = read_file("/corrupt");
    EXPECT_TRUE(result4.has_exception());
    EXPECT_THROW(std::rethrow_exception(result4.exception()), TestException);
}

TEST_F(OutcomeTest, ChainedOperations) {
    auto parse_int = [](const std::string& s) -> Outcome<int, ErrorCode> {
        return Outcome<int, ErrorCode>::from_call([&s]() {
            return std::stoi(s);
        });
    };

    auto validate_positive = [](int x) -> Outcome<int, ErrorCode> {
        if (x > 0) {
            return Outcome<int, ErrorCode>::success(x);
        }
        return Outcome<int, ErrorCode>::failure(ErrorCode::InvalidArgument); // No ValidationError
    };

    auto process = [&](const std::string& input) -> Outcome<int, ErrorCode> {
        return parse_int(input)
            .and_then(validate_positive)
            .map([](int x) { return x * 2; });
    };

    auto result1 = process("42");
    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), 84);

    auto result2 = process("-5");
    EXPECT_TRUE(result2.has_error());
    EXPECT_EQ(result2.error(), ErrorCode::InvalidArgument);

    auto result3 = process("abc");
    EXPECT_TRUE(result3.has_exception());
    // std::stoi throws std::invalid_argument
}

// Custom error type tests
struct CustomError {
    int code_val;
    std::string msg;

    // Required by ErrorType concept in Result
    int code() const { return code_val; }
    std::string_view message() const { return msg; }

    bool operator==(const CustomError& other) const {
        return code_val == other.code_val;
    }
};

TEST_F(OutcomeTest, CustomErrorType) {
    Outcome<int, CustomError> success(42);
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(success.value(), 42);

    CustomError err{100, "custom error"};
    Outcome<int, CustomError> failure{Error<CustomError>(err)}; // Use braces to avoid vexing parse
    EXPECT_TRUE(failure.has_error());
    EXPECT_EQ(failure.error().code(), 100);
    EXPECT_EQ(failure.error().message(), "custom error");

    auto ex = std::make_exception_ptr(TestException("exception"));
    Outcome<int, CustomError> exception(ex);
    EXPECT_TRUE(exception.has_exception());
}
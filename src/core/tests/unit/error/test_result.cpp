#include <gtest/gtest.h>
#include <core/error/result.h>
#include <core/error/error_code.h>
#include <string>
#include <memory>
#include <vector>

using namespace fem::core::error;

class ResultTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Basic Result construction tests
TEST_F(ResultTest, ConstructOkWithValue) {
    Result<int, ErrorCode> result = Ok<int, ErrorCode>(42);

    EXPECT_TRUE(result.is_ok());
    EXPECT_FALSE(result.is_error());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(ResultTest, ConstructErrWithErrorCode) {
    Result<int, ErrorCode> result = Error<ErrorCode>(ErrorCode::InvalidArgument);

    EXPECT_FALSE(result.is_ok());
    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST_F(ResultTest, ConstructOkWithString) {
    Result<std::string, ErrorCode> result = Ok<std::string, ErrorCode>("success");

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), "success");
}

TEST_F(ResultTest, ConstructOkWithMoveOnly) {
    Result<std::unique_ptr<int>, ErrorCode> result = Ok<std::unique_ptr<int>, ErrorCode>(std::make_unique<int>(42));

    EXPECT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);
    EXPECT_EQ(*result.value(), 42);
}

// Value access tests
TEST_F(ResultTest, ValueAccess) {
    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(100);
    EXPECT_EQ(ok_result.value(), 100);

    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);
    EXPECT_THROW({
        [[maybe_unused]] auto val = err_result.value();
    }, std::runtime_error);
}

TEST_F(ResultTest, ValueOrDefault) {
    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(42);
    EXPECT_EQ(ok_result.value_or(100), 42);

    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);
    EXPECT_EQ(err_result.value_or(100), 100);
}

// Value_or doesn't support value_or_else in the actual API
// We'll test value_or only

// Error access tests
TEST_F(ResultTest, ErrorAccess) {
    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::InvalidArgument);
    EXPECT_EQ(err_result.error(), ErrorCode::InvalidArgument);

    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(42);
    // Accessing error on success terminates, so we won't test that
}

// Boolean conversion tests
TEST_F(ResultTest, BooleanConversion) {
    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(42);
    EXPECT_TRUE(ok_result);
    EXPECT_TRUE(static_cast<bool>(ok_result));

    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);
    EXPECT_FALSE(err_result);
    EXPECT_FALSE(static_cast<bool>(err_result));
}

// Monadic operations tests
TEST_F(ResultTest, MapOkValue) {
    Result<int, ErrorCode> result = Ok<int, ErrorCode>(42);

    auto mapped = result.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.is_ok());
    EXPECT_EQ(mapped.value(), 84);
}

TEST_F(ResultTest, MapErrValue) {
    Result<int, ErrorCode> result = Err<ErrorCode, int>(ErrorCode::InvalidArgument);

    auto mapped = result.map([](int x) { return x * 2; });

    EXPECT_TRUE(mapped.is_error());
    EXPECT_EQ(mapped.error(), ErrorCode::InvalidArgument);
}

TEST_F(ResultTest, MapWithTypeChange) {
    Result<int, ErrorCode> result = Ok<int, ErrorCode>(42);

    auto mapped = result.map([](int x) { return std::to_string(x); });

    EXPECT_TRUE(mapped.is_ok());
    EXPECT_EQ(mapped.value(), "42");
}

// Test or_else with ErrorCode
TEST_F(ResultTest, OrElseError) {
    // Note: or_else transforms the error type, not the value
    Result<int, ErrorCode> result = Error<ErrorCode>(ErrorCode::InvalidArgument);

    // Transform to a different error type (still ErrorCode in this case)
    auto transformed = result.or_else([](ErrorCode) {
        return ErrorCode::TypeMismatch;  // Change the error
    });

    EXPECT_TRUE(transformed.is_error());
    EXPECT_EQ(transformed.error(), ErrorCode::TypeMismatch);
}

TEST_F(ResultTest, AndThenChaining) {
    auto divide = [](int x, int y) -> Result<int, ErrorCode> {
        if (y == 0) {
            return Err<ErrorCode, int>(ErrorCode::InvalidArgument);
        }
        return Ok<int, ErrorCode>(x / y);
    };

    Result<int, ErrorCode> result = Ok<int, ErrorCode>(100);
    auto chained = result.and_then([&](int x) { return divide(x, 2); });

    EXPECT_TRUE(chained.is_ok());
    EXPECT_EQ(chained.value(), 50);

    auto chained_err = result.and_then([&](int x) { return divide(x, 0); });
    EXPECT_TRUE(chained_err.is_error());
    EXPECT_EQ(chained_err.error(), ErrorCode::InvalidArgument);
}

TEST_F(ResultTest, OrElseChaining) {
    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);

    auto recovered = err_result.or_else([](ErrorCode) {
        return ErrorCode::InvalidArgument;  // or_else returns a new error
    });

    EXPECT_TRUE(recovered.is_error());
    EXPECT_EQ(recovered.error(), ErrorCode::InvalidArgument);
}

// transform doesn't exist in the actual API, skipping this test

// Helper function tests
TEST_F(ResultTest, OkAndErrHelpers) {
    auto ok_result = Ok<std::string, ErrorCode>("test");
    EXPECT_TRUE(ok_result.is_ok());
    EXPECT_EQ(ok_result.value(), "test");

    auto err_result = Err<ErrorCode, int>(ErrorCode::InvalidArgument);
    EXPECT_TRUE(err_result.is_error());
    EXPECT_EQ(err_result.error(), ErrorCode::InvalidArgument);
}

// TRY macro simulation tests
TEST_F(ResultTest, TryPatternSuccess) {
    auto operation = []() -> Result<int, ErrorCode> {
        auto step1 = []() -> Result<int, ErrorCode> { return Ok<int, ErrorCode>(10); };
        auto step2 = [](int x) -> Result<int, ErrorCode> { return Ok<int, ErrorCode>(x * 2); };

        // Simulate TRY macro behavior
        auto res1 = step1();
        if (!res1) return Error<ErrorCode>(ErrorCode(res1.error()));

        auto res2 = step2(res1.value());
        if (!res2) return Error<ErrorCode>(ErrorCode(res2.error()));

        return Ok<int, ErrorCode>(int(res2.value()));
    };

    auto result = operation();
    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value(), 20);
}

TEST_F(ResultTest, TryPatternFailure) {
    auto operation = []() -> Result<int, ErrorCode> {
        auto step1 = []() -> Result<int, ErrorCode> { return Ok<int, ErrorCode>(10); };
        auto step2 = [](int) -> Result<int, ErrorCode> { return Err<ErrorCode, int>(ErrorCode::InvalidArgument); };

        // Simulate TRY macro behavior
        auto res1 = step1();
        if (!res1) return Error<ErrorCode>(ErrorCode(res1.error()));

        auto res2 = step2(res1.value());
        if (!res2) return Error<ErrorCode>(ErrorCode(res2.error()));

        return Ok<int, ErrorCode>(int(res2.value()));
    };

    auto result = operation();
    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

// Note: Result<void, E> isn't directly supported in the same way
// We'd need to use a different approach for void results

// Complex type tests
TEST_F(ResultTest, VectorResult) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    Result<std::vector<int>, ErrorCode> result = Ok<std::vector<int>, ErrorCode>(std::move(vec));

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().size(), 5);
    EXPECT_EQ(result.value()[0], 1);
}

TEST_F(ResultTest, NestedResult) {
    using InnerResult = Result<int, ErrorCode>;
    using NestedResult = Result<InnerResult, ErrorCode>;

    NestedResult nested = Ok<InnerResult, ErrorCode>(Ok<int, ErrorCode>(42));
    EXPECT_TRUE(nested.is_ok());
    EXPECT_TRUE(nested.value().is_ok());
    EXPECT_EQ(nested.value().value(), 42);

    NestedResult nested_err = Ok<InnerResult, ErrorCode>(Err<ErrorCode, int>(ErrorCode::ResourceNotFound));
    EXPECT_TRUE(nested_err.is_ok());
    EXPECT_TRUE(nested_err.value().is_error());
    EXPECT_EQ(nested_err.value().error(), ErrorCode::ResourceNotFound);
}

// Custom error type tests
struct CustomError {
    int code_val;
    std::string msg;

    // Required by ErrorType concept
    int code() const { return code_val; }
    std::string_view message() const { return msg; }

    bool operator==(const CustomError& other) const {
        return code_val == other.code_val && msg == other.msg;
    }
};

TEST_F(ResultTest, CustomErrorType) {
    Result<int, CustomError> ok_result(42);
    EXPECT_TRUE(ok_result.is_ok());
    EXPECT_EQ(ok_result.value(), 42);

    CustomError err{100, "custom error"};
    Result<int, CustomError> err_result{Error<CustomError>(err)}; // Use braces
    EXPECT_TRUE(err_result.is_error());
    EXPECT_EQ(err_result.error().code(), 100);
    EXPECT_EQ(err_result.error().message(), "custom error");
}

// Move semantics tests
TEST_F(ResultTest, MoveOnlyValue) {
    auto ptr = std::make_unique<int>(42);
    Result<std::unique_ptr<int>, ErrorCode> result = Ok<std::unique_ptr<int>, ErrorCode>(std::move(ptr));

    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(ptr, nullptr);  // Original pointer should be moved

    auto extracted = std::move(result).value();
    EXPECT_NE(extracted, nullptr);
    EXPECT_EQ(*extracted, 42);
}

TEST_F(ResultTest, MoveConstruction) {
    Result<std::string, ErrorCode> result1 = Ok<std::string, ErrorCode>("test");
    Result<std::string, ErrorCode> result2 = std::move(result1);

    EXPECT_TRUE(result2.is_ok());
    EXPECT_EQ(result2.value(), "test");
}

TEST_F(ResultTest, MoveAssignment) {
    Result<std::string, ErrorCode> result1 = Ok<std::string, ErrorCode>("first");
    Result<std::string, ErrorCode> result2 = Ok<std::string, ErrorCode>("second");

    result2 = std::move(result1);

    EXPECT_TRUE(result2.is_ok());
    EXPECT_EQ(result2.value(), "first");
}

// Access tests - unwrap/expect not in API, using value() and value_or()
TEST_F(ResultTest, ValueOrAccess) {
    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(42);
    EXPECT_EQ(ok_result.value(), 42);
    EXPECT_EQ(ok_result.value_or(100), 42);

    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);
    EXPECT_EQ(err_result.value_or(100), 100);
}

// Pointer-like access tests
TEST_F(ResultTest, PointerAccess) {
    Result<std::string, ErrorCode> ok_result = Ok<std::string, ErrorCode>("test");
    EXPECT_NE(ok_result.operator->(), nullptr);
    EXPECT_EQ(ok_result->size(), 4);
    EXPECT_EQ(*ok_result, "test");

    Result<std::string, ErrorCode> err_result = Err<ErrorCode, std::string>(ErrorCode::InvalidArgument);
    EXPECT_EQ(err_result.operator->(), nullptr);
}

// Pattern matching simulation
TEST_F(ResultTest, PatternMatching) {
    auto process_result = [](const Result<int, ErrorCode>& result) -> std::string {
        if (result.is_ok()) {
            return "Success: " + std::to_string(result.value());
        } else {
            CoreErrorCategory category;
            return "Error: " + category.message(static_cast<int>(result.error()));
        }
    };

    Result<int, ErrorCode> ok_result = Ok<int, ErrorCode>(42);
    EXPECT_EQ(process_result(ok_result), "Success: 42");

    Result<int, ErrorCode> err_result = Err<ErrorCode, int>(ErrorCode::ResourceNotFound);
    EXPECT_TRUE(process_result(err_result).find("Error:") == 0);
}

// Real-world usage examples
TEST_F(ResultTest, FileOperationExample) {
    auto read_file = [](const std::string& path) -> Result<std::string, ErrorCode> {
        if (path.empty()) {
            return Err<ErrorCode, std::string>(ErrorCode::InvalidArgument);
        }
        if (path == "/not/found") {
            return Err<ErrorCode, std::string>(ErrorCode::FileNotFound);
        }
        return Ok<std::string, ErrorCode>("file contents");
    };

    auto result1 = read_file("/valid/path");
    EXPECT_TRUE(result1.is_ok());
    EXPECT_EQ(result1.value(), "file contents");

    auto result2 = read_file("");
    EXPECT_TRUE(result2.is_error());
    EXPECT_EQ(result2.error(), ErrorCode::InvalidArgument);

    auto result3 = read_file("/not/found");
    EXPECT_TRUE(result3.is_error());
    EXPECT_EQ(result3.error(), ErrorCode::FileNotFound);
}

TEST_F(ResultTest, ChainedOperations) {
    auto parse_int = [](const std::string& s) -> Result<int, ErrorCode> {
        try {
            return Ok<int, ErrorCode>(std::stoi(s));
        } catch (...) {
            return Err<ErrorCode, int>(ErrorCode::TypeMismatch); // No ParseError in actual API
        }
    };

    auto validate_positive = [](int x) -> Result<int, ErrorCode> {
        if (x > 0) {
            return Ok<int, ErrorCode>(int(x));  // Copy x to create rvalue
        }
        return Err<ErrorCode, int>(ErrorCode::InvalidArgument); // No ValidationError
    };

    auto process = [&](const std::string& input) -> Result<int, ErrorCode> {
        return parse_int(input)
            .and_then(validate_positive)
            .map([](int x) { return x * 2; });
    };

    EXPECT_EQ(process("42").value(), 84);
    EXPECT_EQ(process("-5").error(), ErrorCode::InvalidArgument);
    EXPECT_EQ(process("abc").error(), ErrorCode::TypeMismatch);
}
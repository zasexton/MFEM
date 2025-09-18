#include <gtest/gtest.h>
#include <core/error/status.h>
#include <core/error/error_code.h>
#include <string>
#include <vector>

using namespace fem::core::error;

class StatusTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Basic Status construction tests
TEST_F(StatusTest, DefaultConstructorIsOK) {
    Status status;

    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(status.is_error());
    EXPECT_EQ(status.code(), ErrorCode::Success);
}

TEST_F(StatusTest, ConstructWithErrorCode) {
    Status status(ErrorCode::InvalidArgument);

    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.is_error());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
}

TEST_F(StatusTest, ConstructWithErrorAndMessage) {
    Status status(ErrorCode::FileNotFound, "File not found");

    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.is_error());
    EXPECT_EQ(status.code(), ErrorCode::FileNotFound);
    EXPECT_EQ(status.message(), "File not found");
}

TEST_F(StatusTest, ConstructWithErrorMessageAndContext) {
    Status status(ErrorCode::IoError, "Read failed", "/path/to/file");

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::IoError);
    EXPECT_EQ(status.message(), "Read failed");
    EXPECT_EQ(status.context(), "/path/to/file");
}

// Factory method tests
TEST_F(StatusTest, FactoryOK) {
    auto status = Status::OK();

    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(status.is_error());
    EXPECT_EQ(status.code(), ErrorCode::Success);
}

TEST_F(StatusTest, FactoryError) {
    auto status = Status::Error(ErrorCode::InvalidArgument);

    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.is_error());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
}

TEST_F(StatusTest, FactoryErrorWithMessage) {
    auto status = Status::Error(ErrorCode::OutOfRange, "Index out of bounds");

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::OutOfRange);
    EXPECT_EQ(status.message(), "Index out of bounds");
}

// Boolean conversion tests
TEST_F(StatusTest, BooleanConversion) {
    Status ok_status = Status::OK();
    EXPECT_TRUE(ok_status);
    EXPECT_TRUE(static_cast<bool>(ok_status));

    Status error_status = Status::Error(ErrorCode::ResourceNotFound);
    EXPECT_FALSE(error_status);
    EXPECT_FALSE(static_cast<bool>(error_status));
}

// Message tests
TEST_F(StatusTest, DefaultMessageFromErrorCode) {
    Status status(ErrorCode::InvalidArgument);

    // Should get default message from error category
    auto msg = status.message();
    EXPECT_FALSE(msg.empty());
    // CoreErrorCategory returns "Invalid argument" for this code
    EXPECT_TRUE(msg.find("Invalid argument") != std::string::npos);
}

TEST_F(StatusTest, CustomMessageOverridesDefault) {
    Status status(ErrorCode::InvalidArgument, "Custom error message");

    EXPECT_EQ(status.message(), "Custom error message");
}

TEST_F(StatusTest, EmptyContext) {
    Status status(ErrorCode::ResourceNotFound, "Not found");

    EXPECT_TRUE(status.context().empty());
}

// Update tests
TEST_F(StatusTest, UpdateErrorCode) {
    Status status = Status::OK();
    EXPECT_TRUE(status.ok());

    status.update(ErrorCode::InvalidArgument);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
}

TEST_F(StatusTest, UpdateErrorCodeAndMessage) {
    Status status = Status::OK();

    status.update(ErrorCode::ResourceNotFound, "Resource missing");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::ResourceNotFound);
    EXPECT_EQ(status.message(), "Resource missing");
}

TEST_F(StatusTest, AddContext) {
    Status status(ErrorCode::IoError, "Write failed");

    status.add_context("file.txt");
    EXPECT_EQ(status.context(), "file.txt");

    status.add_context("line 42");
    EXPECT_TRUE(status.context().find("file.txt") != std::string::npos);
    EXPECT_TRUE(status.context().find("line 42") != std::string::npos);
}

TEST_F(StatusTest, ChainedUpdates) {
    Status status = Status::OK();

    status.update(ErrorCode::InvalidArgument, "Bad input")
          .add_context("parameter: x");

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(status.message(), "Bad input");
    EXPECT_EQ(status.context(), "parameter: x");
}

// Comparison tests
TEST_F(StatusTest, Equality) {
    Status status1 = Status::OK();
    Status status2 = Status::OK();
    Status status3 = Status::Error(ErrorCode::ResourceNotFound);
    Status status4 = Status::Error(ErrorCode::ResourceNotFound);
    Status status5 = Status::Error(ErrorCode::InvalidArgument);

    EXPECT_EQ(status1, status2);
    EXPECT_EQ(status3, status4);
    EXPECT_NE(status1, status3);
    EXPECT_NE(status3, status5);
}

TEST_F(StatusTest, EqualityIgnoresMessage) {
    Status status1(ErrorCode::ResourceNotFound, "Message 1");
    Status status2(ErrorCode::ResourceNotFound, "Message 2");

    // Status equality is based only on error code
    EXPECT_EQ(status1, status2);
}

// String representation tests
TEST_F(StatusTest, ToStringOK) {
    Status status = Status::OK();

    EXPECT_EQ(status.to_string(), "OK");
}

TEST_F(StatusTest, ToStringError) {
    Status status(ErrorCode::InvalidArgument, "Invalid input");

    auto str = status.to_string();
    EXPECT_TRUE(str.find("Invalid input") != std::string::npos);
}

TEST_F(StatusTest, ToStringWithContext) {
    Status status(ErrorCode::FileNotFound, "File not found", "config.json");

    auto str = status.to_string();
    EXPECT_TRUE(str.find("File not found") != std::string::npos);
    EXPECT_TRUE(str.find("config.json") != std::string::npos);
}

// StatusBuilder tests
TEST_F(StatusTest, StatusBuilderEmptyIsOK) {
    StatusBuilder builder;

    EXPECT_TRUE(builder.ok());
    EXPECT_EQ(builder.error_count(), 0);
    EXPECT_TRUE(builder.status().ok());
}

TEST_F(StatusTest, StatusBuilderAddSingleError) {
    StatusBuilder builder;

    builder.add_error(Status::Error(ErrorCode::InvalidArgument, "Bad arg"));

    EXPECT_FALSE(builder.ok());
    EXPECT_EQ(builder.error_count(), 1);

    auto status = builder.status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
    EXPECT_TRUE(status.message().find("Bad arg") != std::string::npos);
}

TEST_F(StatusTest, StatusBuilderAddMultipleErrors) {
    StatusBuilder builder;

    builder.add_error(Status::Error(ErrorCode::InvalidArgument, "First error"))
           .add_error(Status::Error(ErrorCode::ResourceNotFound, "Second error"))
           .add_error(Status::Error(ErrorCode::OutOfRange, "Third error"));

    EXPECT_FALSE(builder.ok());
    EXPECT_EQ(builder.error_count(), 3);

    auto status = builder.status();
    EXPECT_FALSE(status.ok());
    // Should keep first error code
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
    // Message should indicate multiple errors
    EXPECT_TRUE(status.message().find("2 more errors") != std::string::npos);
}

TEST_F(StatusTest, StatusBuilderAddErrorWithCodeAndMessage) {
    StatusBuilder builder;

    builder.add_error(ErrorCode::IoError, "Disk full");

    EXPECT_FALSE(builder.ok());
    EXPECT_EQ(builder.error_count(), 1);

    auto status = builder.status();
    EXPECT_EQ(status.code(), ErrorCode::IoError);
    EXPECT_TRUE(status.message().find("Disk full") != std::string::npos);
}

TEST_F(StatusTest, StatusBuilderIgnoreOKStatus) {
    StatusBuilder builder;

    builder.add_error(Status::OK())
           .add_error(Status::Error(ErrorCode::ResourceNotFound))
           .add_error(Status::OK());

    EXPECT_FALSE(builder.ok());
    EXPECT_EQ(builder.error_count(), 1);  // Only one error added

    auto status = builder.status();
    EXPECT_EQ(status.code(), ErrorCode::ResourceNotFound);
}

TEST_F(StatusTest, StatusBuilderChainedAddErrors) {
    StatusBuilder builder;

    auto final_status = builder
        .add_error(ErrorCode::InvalidArgument, "Bad input")
        .add_error(ErrorCode::OutOfRange, "Index too large")
        .status();

    EXPECT_FALSE(final_status.ok());
    EXPECT_EQ(final_status.code(), ErrorCode::InvalidArgument);
}

// Real-world usage examples
TEST_F(StatusTest, FileOperationExample) {
    auto open_file = [](const std::string& path) -> Status {
        if (path.empty()) {
            return Status::Error(ErrorCode::InvalidArgument, "Path cannot be empty");
        }
        if (path == "/not/found") {
            return Status::Error(ErrorCode::FileNotFound, "File does not exist")
                         .add_context(path);
        }
        if (path == "/no/permission") {
            return Status::Error(ErrorCode::FileAccessDenied, "Access denied")
                         .add_context(path);
        }
        return Status::OK();
    };

    EXPECT_TRUE(open_file("/valid/path").ok());
    EXPECT_EQ(open_file("").code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(open_file("/not/found").code(), ErrorCode::FileNotFound);
    EXPECT_EQ(open_file("/no/permission").code(), ErrorCode::FileAccessDenied);
}

TEST_F(StatusTest, ValidationExample) {
    auto validate_config = [](const std::string& config) -> Status {
        StatusBuilder builder;

        if (config.empty()) {
            builder.add_error(ErrorCode::InvalidArgument, "Config is empty");
        }
        if (config.find("required_field") == std::string::npos) {
            builder.add_error(ErrorCode::ConfigInvalid, "Missing required_field");
        }
        if (config.find("version") == std::string::npos) {
            builder.add_error(ErrorCode::RequiredConfigMissing, "Missing version");
        }

        return builder.status();
    };

    auto status1 = validate_config("required_field: value\nversion: 1.0");
    EXPECT_TRUE(status1.ok());

    auto status2 = validate_config("");
    EXPECT_FALSE(status2.ok());
    // Should have multiple errors

    auto status3 = validate_config("version: 1.0");
    EXPECT_FALSE(status3.ok());
    EXPECT_EQ(status3.code(), ErrorCode::ConfigInvalid);
}

TEST_F(StatusTest, ChainedOperations) {
    auto step1 = []() -> Status {
        return Status::OK();
    };

    auto step2 = []() -> Status {
        return Status::Error(ErrorCode::InvalidState, "Not initialized");
    };

    auto step3 = []() -> Status {
        return Status::OK();
    };

    auto run_pipeline = [&]() -> Status {
        // Using RETURN_IF_ERROR macro pattern
        Status s = step1();
        if (!s.ok()) return s;

        s = step2();
        if (!s.ok()) return s;

        s = step3();
        if (!s.ok()) return s;

        return Status::OK();
    };

    auto result = run_pipeline();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code(), ErrorCode::InvalidState);
    EXPECT_TRUE(result.message().find("Not initialized") != std::string::npos);
}

// Copy semantics tests
TEST_F(StatusTest, CopyConstructor) {
    Status status1(ErrorCode::ResourceNotFound, "Original");
    Status status2(status1);

    EXPECT_EQ(status2.code(), ErrorCode::ResourceNotFound);
    EXPECT_EQ(status2.message(), "Original");
}

TEST_F(StatusTest, CopyAssignment) {
    Status status1(ErrorCode::InvalidArgument, "First");
    Status status2 = Status::OK();

    status2 = status1;

    EXPECT_EQ(status2.code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(status2.message(), "First");
}

// Status as return type
TEST_F(StatusTest, FunctionReturnStatus) {
    auto process = [](int value) -> Status {
        if (value < 0) {
            return Status::Error(ErrorCode::InvalidArgument, "Value must be positive");
        }
        if (value > 100) {
            return Status::Error(ErrorCode::OutOfRange, "Value too large");
        }
        return Status::OK();
    };

    EXPECT_TRUE(process(50).ok());
    EXPECT_EQ(process(-1).code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(process(200).code(), ErrorCode::OutOfRange);
}

// Status aggregation pattern
TEST_F(StatusTest, AggregateMultipleStatuses) {
    std::vector<Status> statuses = {
        Status::OK(),
        Status::Error(ErrorCode::InvalidArgument),
        Status::OK(),
        Status::Error(ErrorCode::ResourceNotFound),
        Status::Error(ErrorCode::IoError)
    };

    StatusBuilder builder;
    for (const auto& status : statuses) {
        builder.add_error(status);
    }

    EXPECT_FALSE(builder.ok());
    EXPECT_EQ(builder.error_count(), 3);  // Three errors

    auto final_status = builder.status();
    EXPECT_EQ(final_status.code(), ErrorCode::InvalidArgument);  // First error
}

// Status with detailed context
TEST_F(StatusTest, DetailedContext) {
    Status status = Status::Error(ErrorCode::TypeMismatch, "JSON parsing failed"); // No ParseError, using TypeMismatch

    status.add_context("file: config.json");
    status.add_context("line: 42");
    status.add_context("column: 15");

    auto context = status.context();
    EXPECT_TRUE(context.find("config.json") != std::string::npos);
    EXPECT_TRUE(context.find("42") != std::string::npos);
    EXPECT_TRUE(context.find("15") != std::string::npos);

    auto str = status.to_string();
    EXPECT_TRUE(str.find("JSON parsing failed") != std::string::npos);
    EXPECT_TRUE(str.find(context) != std::string::npos);
}
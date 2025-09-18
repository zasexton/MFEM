#include <gtest/gtest.h>
#include <core/error/error_code.h>
#include <sstream>
#include <system_error>

using namespace fem::core::error;

class ErrorCodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state if needed
    }
};

// Basic ErrorCode tests
TEST_F(ErrorCodeTest, SuccessCodeIsZero) {
    ErrorCode code = ErrorCode::Success;
    EXPECT_EQ(static_cast<int>(code), 0);
}

TEST_F(ErrorCodeTest, ErrorCodeComparison) {
    ErrorCode code1 = ErrorCode::InvalidArgument;
    ErrorCode code2 = ErrorCode::InvalidArgument;
    ErrorCode code3 = ErrorCode::OutOfRange;

    EXPECT_EQ(code1, code2);
    EXPECT_NE(code1, code3);
}

TEST_F(ErrorCodeTest, ErrorCodeToInt) {
    ErrorCode code = ErrorCode::FileNotFound;
    int value = static_cast<int>(code);
    EXPECT_EQ(value, static_cast<int>(ErrorCode::FileNotFound));
}

// ErrorInfo tests
TEST_F(ErrorCodeTest, ErrorInfoConstruction) {
    ErrorInfo info(ErrorCode::InvalidArgument, "bad value");
    EXPECT_EQ(info.error_code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(info.code(), static_cast<int>(ErrorCode::InvalidArgument));
    EXPECT_FALSE(info.context().empty());
}

TEST_F(ErrorCodeTest, ErrorInfoMessage) {
    ErrorInfo info(ErrorCode::InvalidArgument, "test context");
    std::string_view msg = info.message();
    EXPECT_FALSE(msg.empty());
    EXPECT_TRUE(msg.find("Invalid argument") != std::string::npos);
    EXPECT_TRUE(msg.find("test context") != std::string::npos);
}

TEST_F(ErrorCodeTest, ErrorCategoryMessage) {
    CoreErrorCategory category;
    EXPECT_STREQ(category.name(), "fem::core");
    EXPECT_EQ(category.message(0), "Success");
    EXPECT_EQ(category.message(static_cast<int>(ErrorCode::InvalidArgument)), "Invalid argument");
    EXPECT_EQ(category.message(static_cast<int>(ErrorCode::SystemError)), "System error");
}

TEST_F(ErrorCodeTest, MakeErrorHelper) {
    auto error = make_error(ErrorCode::InvalidArgument, "bad input");
    EXPECT_EQ(error.error_code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(error.context(), "bad input");
}

TEST_F(ErrorCodeTest, ErrorInfoFullMessage) {
    ErrorInfo info(ErrorCode::InvalidArgument, "bad input");
    std::string formatted = info.full_message();
    EXPECT_TRUE(formatted.find("Invalid argument") != std::string::npos);
    EXPECT_TRUE(formatted.find("bad input") != std::string::npos);
}

TEST_F(ErrorCodeTest, ErrorInfoWithContext) {
    ErrorInfo info(ErrorCode::FileNotFound, "test_file.txt");
    std::string formatted = info.full_message();
    EXPECT_TRUE(formatted.find("File not found") != std::string::npos);
    EXPECT_TRUE(formatted.find("test_file.txt") != std::string::npos);
}

TEST_F(ErrorCodeTest, ErrorCodeSuccess) {
    ErrorCode success = ErrorCode::Success;
    ErrorCode error = ErrorCode::InvalidArgument;
    EXPECT_EQ(static_cast<int>(success), 0);
    EXPECT_NE(static_cast<int>(error), 0);
}

TEST_F(ErrorCodeTest, ErrorCodeValues) {
    EXPECT_EQ(static_cast<int>(ErrorCode::Success), 0);
    EXPECT_EQ(static_cast<int>(ErrorCode::Unknown), 1);
    EXPECT_EQ(static_cast<int>(ErrorCode::InvalidArgument), 3);
}

// Test unknown error code handling
TEST_F(ErrorCodeTest, UnknownErrorCode) {
    CoreErrorCategory category;
    std::string msg = category.message(99999);
    EXPECT_TRUE(msg.find("Unknown error code") != std::string::npos);
}

// Test error code categories
TEST_F(ErrorCodeTest, ErrorCodeCategories) {
    // General errors (1-99)
    EXPECT_GT(static_cast<int>(ErrorCode::InvalidArgument), 0);
    EXPECT_LT(static_cast<int>(ErrorCode::InvalidArgument), 100);

    // Resource errors (100-199)
    EXPECT_GE(static_cast<int>(ErrorCode::OutOfMemory), 100);
    EXPECT_LT(static_cast<int>(ErrorCode::OutOfMemory), 200);

    // I/O errors (200-299)
    EXPECT_GE(static_cast<int>(ErrorCode::FileNotFound), 200);
    EXPECT_LT(static_cast<int>(ErrorCode::FileNotFound), 300);

    // System errors (500-599)
    EXPECT_GE(static_cast<int>(ErrorCode::SystemError), 500);
    EXPECT_LT(static_cast<int>(ErrorCode::SystemError), 600);
}

// Test std::error_code integration
TEST_F(ErrorCodeTest, StdErrorCodeCreation) {
    ErrorCode fem_code = ErrorCode::InvalidArgument;
    std::error_code std_code = make_error_code(fem_code);

    EXPECT_EQ(std_code.value(), static_cast<int>(fem_code));
    EXPECT_STREQ(std_code.category().name(), "fem::core");
}

TEST_F(ErrorCodeTest, StdErrorCodeMessage) {
    std::error_code code = make_error_code(ErrorCode::FileNotFound);
    std::string message = code.message();

    EXPECT_FALSE(message.empty());
    EXPECT_EQ(message, "File not found");
}

TEST_F(ErrorCodeTest, StdErrorCodeComparison) {
    std::error_code code1 = make_error_code(ErrorCode::InvalidArgument);
    std::error_code code2 = make_error_code(ErrorCode::InvalidArgument);
    std::error_code code3 = make_error_code(ErrorCode::OutOfRange);

    EXPECT_EQ(code1, code2);
    EXPECT_NE(code1, code3);
}

TEST_F(ErrorCodeTest, ErrorConditionMapping) {
    // Test that error codes map to appropriate conditions
    std::error_code code = make_error_code(ErrorCode::InvalidArgument);
    std::error_condition condition = code.default_error_condition();

    EXPECT_EQ(condition.value(), static_cast<int>(ErrorCode::InvalidArgument));
}

// Test error code streaming
TEST_F(ErrorCodeTest, StreamOutput) {
    ErrorCode code = ErrorCode::FileNotFound;
    std::ostringstream oss;
    oss << "Error code: " << static_cast<int>(code);

    EXPECT_EQ(oss.str(), "Error code: 200");
}

// Test all error codes have proper metadata
TEST_F(ErrorCodeTest, AllErrorCodesHaveMessages) {
    std::vector<ErrorCode> all_codes = {
        ErrorCode::Success,
        ErrorCode::Unknown,
        ErrorCode::NotImplemented,
        ErrorCode::InvalidArgument,
        ErrorCode::InvalidState,
        ErrorCode::OutOfRange,
        ErrorCode::TypeMismatch,
        ErrorCode::OutOfMemory,
        ErrorCode::ResourceNotFound,
        ErrorCode::ResourceBusy,
        ErrorCode::ResourceExhausted,
        ErrorCode::AllocationFailed,
        ErrorCode::FileNotFound,
        ErrorCode::FileAccessDenied,
        ErrorCode::FileAlreadyExists,
        ErrorCode::InvalidPath,
        ErrorCode::IoError,
        ErrorCode::EndOfFile,
        ErrorCode::DeadlockDetected,
        ErrorCode::ThreadCreationFailed,
        ErrorCode::SynchronizationError,
        ErrorCode::TimeoutExpired,
        ErrorCode::ConfigNotFound,
        ErrorCode::ConfigInvalid,
        ErrorCode::ConfigTypeMismatch,
        ErrorCode::RequiredConfigMissing,
        ErrorCode::SystemError,
        ErrorCode::PlatformNotSupported,
        ErrorCode::FeatureDisabled
    };

    CoreErrorCategory category;
    for (ErrorCode code : all_codes) {
        // Each code should have a message
        std::string message = category.message(static_cast<int>(code));
        EXPECT_FALSE(message.empty());
        EXPECT_TRUE(message.find("Unknown error code") == std::string::npos ||
                    code == static_cast<ErrorCode>(99999));
    }
}

// Test error code in switch statement
TEST_F(ErrorCodeTest, ErrorCodeSwitch) {
    auto process_error = [](ErrorCode code) -> std::string {
        switch (code) {
            case ErrorCode::Success:
                return "Success";
            case ErrorCode::InvalidArgument:
                return "Bad argument";
            case ErrorCode::FileNotFound:
                return "Missing file";
            default:
                return "Other error";
        }
    };

    EXPECT_EQ(process_error(ErrorCode::Success), "Success");
    EXPECT_EQ(process_error(ErrorCode::InvalidArgument), "Bad argument");
    EXPECT_EQ(process_error(ErrorCode::FileNotFound), "Missing file");
    EXPECT_EQ(process_error(ErrorCode::Unknown), "Other error");
}

// Test error code value ranges
TEST_F(ErrorCodeTest, ErrorCodeValueRanges) {
    // Success is 0
    EXPECT_EQ(static_cast<int>(ErrorCode::Success), 0);

    // General errors are 1-99
    EXPECT_GT(static_cast<int>(ErrorCode::Unknown), 0);
    EXPECT_LT(static_cast<int>(ErrorCode::TypeMismatch), 100);

    // Resource errors are 100-199
    EXPECT_GE(static_cast<int>(ErrorCode::OutOfMemory), 100);
    EXPECT_LT(static_cast<int>(ErrorCode::AllocationFailed), 200);
}

// Test ErrorInfo with source location (if supported)
TEST_F(ErrorCodeTest, ErrorInfoWithLocation) {
    // Note: std::source_location may not be available on all compilers
    // The ErrorInfo class includes location info but we can't easily test it
    // without compiler support. Just test that it compiles.
    ErrorInfo info(ErrorCode::InvalidArgument, "test");
    std::string full_msg = info.full_message();
    EXPECT_FALSE(full_msg.empty());
}
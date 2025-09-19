#include <gtest/gtest.h>
#include <core/error/exception_base.h>
#include <core/error/logic_error.h>
#include <core/error/error_code.h>
#include <sstream>
#include <typeinfo>

using namespace fem::core::error;

class ExceptionBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state if needed
    }
};

// StackTrace tests - NOTE: StackTrace methods are not yet implemented
// These tests are commented out until the implementation is available
/*
TEST_F(ExceptionBaseTest, StackTraceConstruction) {
    StackTrace trace;
    EXPECT_TRUE(trace.frames().empty());
}

TEST_F(ExceptionBaseTest, StackTraceCapture) {
    StackTrace trace;
    trace.capture(0);

    // Note: Stack trace capture is platform-dependent
    // We mainly test that it doesn't crash
    std::string formatted = trace.format();
    // Don't assert anything specific about the content
    // as it's implementation-dependent
}

TEST_F(ExceptionBaseTest, StackTraceFormat) {
    StackTrace trace;
    std::string formatted = trace.format();
    EXPECT_TRUE(formatted.empty()); // Empty trace should format to empty string
}
*/

// Basic Exception tests
TEST_F(ExceptionBaseTest, ExceptionConstruction) {
    Exception ex("Test message");
    EXPECT_STREQ(ex.what(), "Test message");
    EXPECT_EQ(ex.code(), ErrorCode::Unknown);
}

TEST_F(ExceptionBaseTest, ExceptionWithErrorCode) {
    Exception ex("Test message", ErrorCode::InvalidArgument);
    EXPECT_STREQ(ex.what(), "Test message");
    EXPECT_EQ(ex.code(), ErrorCode::InvalidArgument);
}

TEST_F(ExceptionBaseTest, ExceptionCopyConstructor) {
    Exception original("Original message", ErrorCode::FileNotFound);
    original.with_context("test context");

    Exception copy(original);
    EXPECT_STREQ(copy.what(), "Original message");
    EXPECT_EQ(copy.code(), ErrorCode::FileNotFound);
    EXPECT_EQ(copy.context().size(), 1);
    EXPECT_EQ(copy.context()[0], "test context");
}

TEST_F(ExceptionBaseTest, ExceptionMoveConstructor) {
    Exception original("Original message", ErrorCode::FileNotFound);
    original.with_context("test context");

    Exception moved(std::move(original));
    EXPECT_STREQ(moved.what(), "Original message");
    EXPECT_EQ(moved.code(), ErrorCode::FileNotFound);
    EXPECT_EQ(moved.context().size(), 1);
    EXPECT_EQ(moved.context()[0], "test context");
}

TEST_F(ExceptionBaseTest, ExceptionCopyAssignment) {
    Exception original("Original message", ErrorCode::FileNotFound);
    original.with_context("test context");

    Exception copy("Different", ErrorCode::Unknown);
    copy = original;

    EXPECT_STREQ(copy.what(), "Original message");
    EXPECT_EQ(copy.code(), ErrorCode::FileNotFound);
    EXPECT_EQ(copy.context().size(), 1);
    EXPECT_EQ(copy.context()[0], "test context");
}

TEST_F(ExceptionBaseTest, ExceptionMoveAssignment) {
    Exception original("Original message", ErrorCode::FileNotFound);
    original.with_context("test context");

    Exception moved("Different", ErrorCode::Unknown);
    moved = std::move(original);

    EXPECT_STREQ(moved.what(), "Original message");
    EXPECT_EQ(moved.code(), ErrorCode::FileNotFound);
    EXPECT_EQ(moved.context().size(), 1);
    EXPECT_EQ(moved.context()[0], "test context");
}

TEST_F(ExceptionBaseTest, ExceptionWithContext) {
    Exception ex("Test message");
    ex.with_context("first context")
      .with_context("second context");

    const auto& contexts = ex.context();
    EXPECT_EQ(contexts.size(), 2);
    EXPECT_EQ(contexts[0], "first context");
    EXPECT_EQ(contexts[1], "second context");
}

TEST_F(ExceptionBaseTest, ExceptionWithFormattedContext) {
    Exception ex("Test message");
    ex.with_context("Value: {}", 42);

    const auto& contexts = ex.context();
    EXPECT_EQ(contexts.size(), 1);
    EXPECT_EQ(contexts[0], "Value: 42");
}

// NOTE: with_stack_trace test commented out due to unimplemented StackTrace methods
/*
TEST_F(ExceptionBaseTest, ExceptionWithStackTrace) {
    Exception ex("Test message");
    ex.with_stack_trace();

    // Stack trace capture doesn't crash
    std::string full_msg = ex.full_message();
    EXPECT_FALSE(full_msg.empty());
}
*/

TEST_F(ExceptionBaseTest, ExceptionWithNested) {
    Exception nested("Nested exception", ErrorCode::FileNotFound);
    Exception ex("Main exception", ErrorCode::Unknown);
    ex.with_nested(nested);

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Main exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Nested exception") != std::string::npos);
}

TEST_F(ExceptionBaseTest, ExceptionFullMessage) {
    Exception ex("Test exception", ErrorCode::InvalidArgument);
    ex.with_context("test context");

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Test exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("test context") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Invalid argument") != std::string::npos);
}

TEST_F(ExceptionBaseTest, ExceptionDescribe) {
    Exception ex("Test exception", ErrorCode::InvalidArgument);
    std::string description = ex.describe();

    EXPECT_TRUE(description.find("Test exception") != std::string::npos);
    EXPECT_TRUE(description.find("Invalid argument") != std::string::npos);
}

TEST_F(ExceptionBaseTest, ExceptionPrintDiagnostic) {
    Exception ex("Test exception", ErrorCode::InvalidArgument);
    std::ostringstream oss;
    ex.print_diagnostic(oss);

    std::string output = oss.str();
    EXPECT_TRUE(output.find("Test exception") != std::string::npos);
}

TEST_F(ExceptionBaseTest, ExceptionFormattedConstructor) {
    Exception ex(ErrorCode::InvalidArgument,
                std::source_location::current(),
                "Value {} is invalid", 42);

    EXPECT_STREQ(ex.what(), "Value 42 is invalid");
    EXPECT_EQ(ex.code(), ErrorCode::InvalidArgument);
}

// LogicError tests
TEST_F(ExceptionBaseTest, LogicErrorConstruction) {
    LogicError ex("Logic error message");
    EXPECT_STREQ(ex.what(), "Logic error message");
    EXPECT_EQ(ex.code(), ErrorCode::InvalidArgument);
}

TEST_F(ExceptionBaseTest, LogicErrorInheritance) {
    LogicError ex("Logic error");
    Exception* base = &ex;
    EXPECT_STREQ(base->what(), "Logic error");
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
}

// RuntimeError tests
TEST_F(ExceptionBaseTest, RuntimeErrorConstruction) {
    RuntimeError ex("Runtime error message");
    EXPECT_STREQ(ex.what(), "Runtime error message");
    EXPECT_EQ(ex.code(), ErrorCode::Unknown);
}

TEST_F(ExceptionBaseTest, RuntimeErrorInheritance) {
    RuntimeError ex("Runtime error");
    Exception* base = &ex;
    EXPECT_STREQ(base->what(), "Runtime error");
    EXPECT_EQ(base->code(), ErrorCode::Unknown);
}

// SystemError tests
TEST_F(ExceptionBaseTest, SystemErrorConstruction) {
    SystemError ex("System operation failed", ENOENT);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
    EXPECT_EQ(ex.system_error_code(), ENOENT);
}

TEST_F(ExceptionBaseTest, SystemErrorMessage) {
    SystemError ex("System operation failed", ENOENT);
    std::string message = ex.what();
    EXPECT_TRUE(message.find("System operation failed") != std::string::npos);
}

// InvalidArgumentError tests
TEST_F(ExceptionBaseTest, InvalidArgumentErrorConstruction) {
    InvalidArgumentError ex("param_name", "must be positive");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("param_name") != std::string::npos);
    EXPECT_TRUE(message.find("must be positive") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "param_name");
}

TEST_F(ExceptionBaseTest, InvalidArgumentErrorInheritance) {
    InvalidArgumentError ex("param", "invalid");
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
}

// OutOfRangeError tests
TEST_F(ExceptionBaseTest, OutOfRangeErrorConstruction) {
    OutOfRangeError ex("array access", 10, 5);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("array access") != std::string::npos);
    EXPECT_TRUE(message.find("10") != std::string::npos);
    EXPECT_TRUE(message.find("5") != std::string::npos);
    EXPECT_EQ(ex.index(), 10);
    EXPECT_EQ(ex.size(), 5);
}

TEST_F(ExceptionBaseTest, OutOfRangeErrorInheritance) {
    OutOfRangeError ex("test", 1, 1);
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::OutOfRange);
    EXPECT_EQ(base->code(), ErrorCode::OutOfRange);
}

// NotImplementedError tests
TEST_F(ExceptionBaseTest, NotImplementedErrorConstruction) {
    NotImplementedError ex("feature_xyz");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Not implemented") != std::string::npos);
    EXPECT_TRUE(message.find("feature_xyz") != std::string::npos);
}

TEST_F(ExceptionBaseTest, NotImplementedErrorInheritance) {
    NotImplementedError ex("feature");
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::NotImplemented);
    EXPECT_EQ(base->code(), ErrorCode::NotImplemented);
}

// Exception chaining tests
TEST_F(ExceptionBaseTest, ExceptionChaining) {
    Exception nested("Inner exception", ErrorCode::FileNotFound);
    Exception outer("Outer exception", ErrorCode::Unknown);
    outer.with_nested(nested);

    std::string full_msg = outer.full_message();
    EXPECT_TRUE(full_msg.find("Outer exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Inner exception") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Caused by") != std::string::npos);
}

// Exception context tests
TEST_F(ExceptionBaseTest, MultipleContexts) {
    Exception ex("Base message");
    ex.with_context("Context 1")
      .with_context("Context 2")
      .with_context("Context 3");

    const auto& contexts = ex.context();
    EXPECT_EQ(contexts.size(), 3);
    EXPECT_EQ(contexts[0], "Context 1");
    EXPECT_EQ(contexts[1], "Context 2");
    EXPECT_EQ(contexts[2], "Context 3");

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("Context 1") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Context 2") != std::string::npos);
    EXPECT_TRUE(full_msg.find("Context 3") != std::string::npos);
}

// Exception throwing and catching tests
TEST_F(ExceptionBaseTest, ThrowAndCatchException) {
    bool caught = false;
    try {
        throw Exception("Test exception", ErrorCode::InvalidArgument);
    } catch (const Exception& e) {
        caught = true;
        EXPECT_STREQ(e.what(), "Test exception");
        EXPECT_EQ(e.code(), ErrorCode::InvalidArgument);
    }
    EXPECT_TRUE(caught);
}

TEST_F(ExceptionBaseTest, ThrowAndCatchStdException) {
    bool caught = false;
    try {
        throw Exception("Test exception", ErrorCode::InvalidArgument);
    } catch (const std::exception& e) {
        caught = true;
        EXPECT_STREQ(e.what(), "Test exception");
    }
    EXPECT_TRUE(caught);
}

TEST_F(ExceptionBaseTest, ThrowAndCatchSpecificException) {
    bool caught = false;
    try {
        throw InvalidArgumentError("param", "invalid value");
    } catch (const InvalidArgumentError& e) {
        caught = true;
        EXPECT_EQ(e.argument_name(), "param");
    } catch (const Exception& e) {
        FAIL() << "Should have caught InvalidArgumentError specifically";
    }
    EXPECT_TRUE(caught);
}

// Source location tests (basic - actual functionality depends on compiler support)
TEST_F(ExceptionBaseTest, SourceLocationCapture) {
    Exception ex("Test with location");
    const auto& loc = ex.where();

    // Basic test that location was captured
    // Actual values depend on compiler support for std::source_location
    EXPECT_TRUE(loc.file_name() != nullptr);
    EXPECT_TRUE(loc.function_name() != nullptr);
}

// Memory safety tests
TEST_F(ExceptionBaseTest, ExceptionCopyWithNested) {
    Exception nested("Nested", ErrorCode::FileNotFound);
    Exception original("Original", ErrorCode::Unknown);
    original.with_nested(nested);

    // Test that copying preserves nested exception
    Exception copy = original;
    std::string orig_msg = original.full_message();
    std::string copy_msg = copy.full_message();

    EXPECT_EQ(orig_msg, copy_msg);
}

TEST_F(ExceptionBaseTest, ExceptionMoveWithNested) {
    Exception nested("Nested", ErrorCode::FileNotFound);
    Exception original("Original", ErrorCode::Unknown);
    original.with_nested(nested);

    std::string orig_msg = original.full_message();

    // Test that moving preserves nested exception
    Exception moved = std::move(original);
    std::string moved_msg = moved.full_message();

    EXPECT_EQ(orig_msg, moved_msg);
}
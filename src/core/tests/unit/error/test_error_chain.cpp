#include <gtest/gtest.h>
#include <core/error/error_chain.h>
#include <core/error/logic_error.h>
#include <core/error/runtime_error.h>
#include <core/error/error_message.h>
#include <core/error/status.h>

using namespace fem::core::error;

class ErrorChainTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Basic ErrorChain tests
TEST_F(ErrorChainTest, EmptyChain) {
    ErrorChain chain;

    EXPECT_FALSE(chain.has_errors());
    EXPECT_TRUE(chain.empty());
    EXPECT_EQ(chain.error_count(), 0);
    EXPECT_EQ(chain.first_error(), nullptr);
    EXPECT_EQ(chain.last_error(), nullptr);
}

TEST_F(ErrorChainTest, AddErrorCode) {
    ErrorChain chain;

    chain.add_error(ErrorCode::InvalidArgument, "test context");

    EXPECT_TRUE(chain.has_errors());
    EXPECT_FALSE(chain.empty());
    EXPECT_EQ(chain.error_count(), 1);
    EXPECT_NE(chain.first_error(), nullptr);
    EXPECT_EQ(chain.first_error(), chain.last_error());
}

TEST_F(ErrorChainTest, AddErrorInfo) {
    ErrorChain chain;

    auto error_info = make_error(ErrorCode::SystemError, "runtime failure");
    chain.add_error(error_info);

    EXPECT_TRUE(chain.has_errors());
    EXPECT_EQ(chain.error_count(), 1);

    const auto& errors = chain.errors();
    EXPECT_EQ(errors.size(), 1);

    auto* error_variant = std::get_if<ErrorInfo>(&errors[0]);
    EXPECT_NE(error_variant, nullptr);
    EXPECT_EQ(error_variant->error_code(), ErrorCode::SystemError);
}

TEST_F(ErrorChainTest, AddException) {
    ErrorChain chain;

    InvalidArgumentError exception("param", "invalid value");
    chain.add_error(exception);

    EXPECT_TRUE(chain.has_errors());
    EXPECT_EQ(chain.error_count(), 1);

    const auto& errors = chain.errors();
    EXPECT_EQ(errors.size(), 1);

    auto* exception_variant = std::get_if<Exception>(&errors[0]);
    EXPECT_NE(exception_variant, nullptr);
    EXPECT_EQ(exception_variant->code(), ErrorCode::InvalidArgument);
}

TEST_F(ErrorChainTest, MultipleErrors) {
    ErrorChain chain;

    chain.add_error(ErrorCode::InvalidArgument, "first error");
    chain.add_error(RuntimeError("second error"));
    chain.add_error(make_error(ErrorCode::OutOfRange, "third error"));

    EXPECT_EQ(chain.error_count(), 3);

    const auto& errors = chain.errors();
    EXPECT_EQ(errors.size(), 3);

    // Check first error
    auto* first = std::get_if<ErrorInfo>(&errors[0]);
    EXPECT_NE(first, nullptr);
    EXPECT_EQ(first->error_code(), ErrorCode::InvalidArgument);

    // Check second error
    auto* second = std::get_if<Exception>(&errors[1]);
    EXPECT_NE(second, nullptr);
    EXPECT_EQ(second->code(), ErrorCode::SystemError);

    // Check third error
    auto* third = std::get_if<ErrorInfo>(&errors[2]);
    EXPECT_NE(third, nullptr);
    EXPECT_EQ(third->error_code(), ErrorCode::OutOfRange);
}

TEST_F(ErrorChainTest, FirstAndLastError) {
    ErrorChain chain;

    chain.add_error(ErrorCode::InvalidArgument, "first");
    chain.add_error(ErrorCode::SystemError, "middle");
    chain.add_error(ErrorCode::OutOfRange, "last");

    auto* first = chain.first_error();
    auto* last = chain.last_error();

    EXPECT_NE(first, nullptr);
    EXPECT_NE(last, nullptr);
    EXPECT_NE(first, last);

    auto* first_info = std::get_if<ErrorInfo>(first);
    auto* last_info = std::get_if<ErrorInfo>(last);

    EXPECT_EQ(first_info->error_code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(last_info->error_code(), ErrorCode::OutOfRange);
}

TEST_F(ErrorChainTest, ClearChain) {
    ErrorChain chain;

    chain.add_error(ErrorCode::InvalidArgument, "error1");
    chain.add_error(ErrorCode::SystemError, "error2");

    EXPECT_TRUE(chain.has_errors());
    EXPECT_EQ(chain.error_count(), 2);

    chain.clear();

    EXPECT_FALSE(chain.has_errors());
    EXPECT_TRUE(chain.empty());
    EXPECT_EQ(chain.error_count(), 0);
}

TEST_F(ErrorChainTest, MergeChains) {
    ErrorChain chain1;
    ErrorChain chain2;

    chain1.add_error(ErrorCode::InvalidArgument, "error1");
    chain1.add_error(ErrorCode::SystemError, "error2");

    chain2.add_error(ErrorCode::OutOfRange, "error3");
    chain2.add_error(ErrorCode::ConfigInvalid, "error4");

    chain1.merge(chain2);

    EXPECT_EQ(chain1.error_count(), 4);
    EXPECT_EQ(chain2.error_count(), 2);  // Original chain unchanged

    const auto& errors = chain1.errors();
    auto* first = std::get_if<ErrorInfo>(&errors[0]);
    auto* last = std::get_if<ErrorInfo>(&errors[3]);

    EXPECT_EQ(first->error_code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(last->error_code(), ErrorCode::ConfigInvalid);
}

TEST_F(ErrorChainTest, FormatChain) {
    ErrorChain chain;

    chain.add_error(ErrorCode::InvalidArgument, "first error");
    chain.add_error(RuntimeError("second error"));

    std::string formatted = chain.format();

    EXPECT_TRUE(formatted.find("Error chain with 2 error(s)") != std::string::npos);
    EXPECT_TRUE(formatted.find("1.") != std::string::npos);
    EXPECT_TRUE(formatted.find("2.") != std::string::npos);
    EXPECT_TRUE(formatted.find("first error") != std::string::npos);
    EXPECT_TRUE(formatted.find("second error") != std::string::npos);
}

TEST_F(ErrorChainTest, FormatEmptyChain) {
    ErrorChain chain;

    std::string formatted = chain.format();
    EXPECT_EQ(formatted, "No errors");
}

// TEST_F(ErrorChainTest, ToResult) {
//     ErrorChain chain;
//
//     // Empty chain should produce success
//     auto success_result = chain.to_result<int>();
//     EXPECT_TRUE(success_result.is_ok());
//
//     // Chain with errors should produce error
//     chain.add_error(ErrorCode::InvalidArgument, "error");
//     auto error_result = chain.to_result<int>();
//     EXPECT_TRUE(error_result.is_error());
// }

TEST_F(ErrorChainTest, ThrowIfErrors) {
    ErrorChain empty_chain;
    EXPECT_NO_THROW(empty_chain.throw_if_errors());

    ErrorChain error_chain;
    error_chain.add_error(ErrorCode::SystemError, "error");
    EXPECT_THROW(error_chain.throw_if_errors(), RuntimeError);
}

// ErrorCollector tests - disabled due to Result<T, ErrorChain> constraint issues
// TEST_F(ErrorChainTest, ErrorCollectorAllSuccess) {
//     auto op1 = []() -> Result<int, ErrorInfo> {
//         return Ok<int, ErrorInfo>(42);
//     };
//
//     auto op2 = []() -> Result<std::string, ErrorInfo> {
//         return Ok<std::string, ErrorInfo>("hello");
//     };
//
//     ErrorCollector collector(op1, op2);
//     auto result = collector.collect();
//
//     EXPECT_TRUE(result.is_ok());
// }
//
// TEST_F(ErrorChainTest, ErrorCollectorWithErrors) {
//     auto success_op = []() -> Result<int, ErrorInfo> {
//         return Ok<int, ErrorInfo>(42);
//     };
//
//     auto error_op = []() -> Result<int, ErrorInfo> {
//         return Err<ErrorInfo, int>(make_error(ErrorCode::SystemError, "failed"));
//     };
//
//     ErrorCollector collector(success_op, error_op);
//     auto result = collector.collect();
//
//     EXPECT_TRUE(result.is_error());
//     EXPECT_EQ(result.error().error_count(), 1);
// }

// ValidationChain tests
TEST_F(ErrorChainTest, EmptyValidation) {
    ValidationChain validator;

    auto status = validator.run();
    EXPECT_TRUE(status.ok());
}

TEST_F(ErrorChainTest, AllValidationsPass) {
    ValidationChain validator;

    validator.validate([]() {
        return Status::OK();
    });

    validator.validate("named_validation", []() {
        return Status::OK();
    });

    auto status = validator.run();
    EXPECT_TRUE(status.ok());
}

TEST_F(ErrorChainTest, ValidationFailure) {
    ValidationChain validator;

    validator.validate([]() {
        return Status(ErrorCode::InvalidArgument, "first validation failed");
    });

    validator.validate([]() {
        return Status::OK();
    });

    validator.validate("named_validation", []() {
        return Status(ErrorCode::OutOfRange, "named validation failed");
    });

    auto status = validator.run();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);

    std::string message = status.message();
    EXPECT_TRUE(message.find("Error chain with") != std::string::npos);
    EXPECT_TRUE(message.find("first validation failed") != std::string::npos);
    EXPECT_TRUE(message.find("named_validation") != std::string::npos);
}

TEST_F(ErrorChainTest, FastFailValidation) {
    ValidationChain validator;

    int validation_count = 0;

    validator.validate([&]() {
        validation_count++;
        return Status(ErrorCode::InvalidArgument, "first failure");
    });

    validator.validate([&]() {
        validation_count++;
        return Status::OK();
    });

    auto status = validator.run_fast_fail();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(validation_count, 1);  // Should stop after first failure
}

TEST_F(ErrorChainTest, NamedValidationFormatting) {
    ValidationChain validator;

    validator.validate("parameter_check", []() {
        return Status(ErrorCode::InvalidArgument, "parameter is invalid");
    });

    auto status = validator.run();
    EXPECT_FALSE(status.ok());

    std::string message = status.message();
    EXPECT_TRUE(message.find("parameter_check:") != std::string::npos);
    EXPECT_TRUE(message.find("parameter is invalid") != std::string::npos);
}

// Complex scenarios
TEST_F(ErrorChainTest, ComplexErrorAggregation) {
    ErrorChain chain;

    // Simulate a batch operation with mixed success/failure
    std::vector<std::string> items = {"item1", "item2", "item3", "item4"};

    for (size_t i = 0; i < items.size(); ++i) {
        if (i % 2 == 1) {  // Fail on odd indices
            chain.add_error(InvalidArgumentError("item", items[i] + " is invalid"));
        }
    }

    EXPECT_EQ(chain.error_count(), 2);  // item2 and item4 failed

    std::string formatted = chain.format();
    EXPECT_TRUE(formatted.find("item2 is invalid") != std::string::npos);
    EXPECT_TRUE(formatted.find("item4 is invalid") != std::string::npos);
}

TEST_F(ErrorChainTest, NestedValidationChains) {
    ValidationChain outer_validator;

    outer_validator.validate("inner_validation", []() {
        ValidationChain inner_validator;

        inner_validator.validate([]() {
            return Status(ErrorCode::InvalidArgument, "inner failure");
        });

        return inner_validator.run();
    });

    auto status = outer_validator.run();
    EXPECT_FALSE(status.ok());

    std::string message = status.message();
    EXPECT_TRUE(message.find("inner_validation:") != std::string::npos);
    EXPECT_TRUE(message.find("inner failure") != std::string::npos);
}

TEST_F(ErrorChainTest, ErrorChainWithMixedTypes) {
    ErrorChain chain;

    // Add different types of errors
    chain.add_error(ErrorCode::InvalidArgument);  // Just error code
    chain.add_error(ErrorCode::RuntimeError, "with context");  // With context
    chain.add_error(make_error(ErrorCode::OutOfRange, "detailed message"));  // ErrorInfo
    chain.add_error(LogicError("logic error message"));  // Exception

    EXPECT_EQ(chain.error_count(), 4);

    std::string formatted = chain.format();
    EXPECT_TRUE(formatted.find("Error chain with 4 error(s)") != std::string::npos);
}

TEST_F(ErrorChainTest, ChainMergePreservesOrder) {
    ErrorChain chain1;
    ErrorChain chain2;

    chain1.add_error(ErrorCode::InvalidArgument, "error1");
    chain1.add_error(ErrorCode::RuntimeError, "error2");

    chain2.add_error(ErrorCode::OutOfRange, "error3");

    chain1.merge(chain2);

    const auto& errors = chain1.errors();
    EXPECT_EQ(errors.size(), 3);

    auto* first = std::get_if<ErrorInfo>(&errors[0]);
    auto* second = std::get_if<ErrorInfo>(&errors[1]);
    auto* third = std::get_if<ErrorInfo>(&errors[2]);

    EXPECT_EQ(first->error_code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(second->error_code(), ErrorCode::RuntimeError);
    EXPECT_EQ(third->error_code(), ErrorCode::OutOfRange);
}

TEST_F(ErrorChainTest, ValidationChainMethodChaining) {
    ValidationChain validator;

    auto status = validator
        .validate([]() { return Status::OK(); })
        .validate("test1", []() { return Status::OK(); })
        .validate("test2", []() { return Status(ErrorCode::InvalidArgument, "failed"); })
        .run();

    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.message().find("test2") != std::string::npos);
}

// Performance tests
TEST_F(ErrorChainTest, LargeErrorChain) {
    ErrorChain chain;

    const size_t num_errors = 1000;
    for (size_t i = 0; i < num_errors; ++i) {
        chain.add_error(ErrorCode::RuntimeError, "error " + std::to_string(i));
    }

    EXPECT_EQ(chain.error_count(), num_errors);

    // Test that formatting doesn't crash with large chains
    std::string formatted = chain.format();
    EXPECT_TRUE(formatted.find("Error chain with 1000 error(s)") != std::string::npos);
}

TEST_F(ErrorChainTest, ValidationPerformance) {
    ValidationChain validator;

    const size_t num_validations = 100;
    for (size_t i = 0; i < num_validations; ++i) {
        validator.validate([i]() {
            if (i % 10 == 0) {
                return Status(ErrorCode::InvalidArgument, "validation " + std::to_string(i));
            }
            return Status::OK();
        });
    }

    auto status = validator.run();
    EXPECT_FALSE(status.ok());

    // Should have collected all failures, not just the first one
    std::string message = status.message();
    EXPECT_TRUE(message.find("Error chain with") != std::string::npos);
}
#include <gtest/gtest.h>
#include <core/error/validation.h>
#include <vector>
#include <string>
#include <regex>
#include <cmath>

using namespace fem::core::error;

class ValidationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Validator tests for numeric types
TEST_F(ValidationTest, Validator_NotNull_Valid) {
    int value = 42;
    int* ptr = &value;
    auto validator = validate(ptr, "pointer");
    validator.not_null();

    EXPECT_TRUE(validator.is_valid());
    EXPECT_TRUE(validator.status().ok());
}

TEST_F(ValidationTest, Validator_NotNull_Invalid) {
    int* ptr = nullptr;
    auto validator = validate(ptr, "pointer");
    validator.not_null();

    EXPECT_FALSE(validator.is_valid());
    auto status = validator.status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status.code());
    EXPECT_TRUE(status.message().find("must not be null") != std::string::npos);
}

TEST_F(ValidationTest, Validator_InRange_Valid) {
    auto validator = validate(5, "value");
    validator.in_range(0, 10);

    EXPECT_TRUE(validator.is_valid());
    EXPECT_TRUE(validator.status().ok());
}

TEST_F(ValidationTest, Validator_InRange_Invalid) {
    auto validator = validate(15, "value");
    validator.in_range(0, 10);

    EXPECT_FALSE(validator.is_valid());
    auto status = validator.status();
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.message().find("must be in range") != std::string::npos);
}

TEST_F(ValidationTest, Validator_Positive_Valid) {
    auto validator = validate(42, "value");
    validator.positive();

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, Validator_Positive_Invalid) {
    auto validator = validate(0, "value");
    validator.positive();

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be positive") != std::string::npos);
}

TEST_F(ValidationTest, Validator_NonNegative_Valid) {
    auto validator1 = validate(0, "value");
    validator1.non_negative();
    EXPECT_TRUE(validator1.is_valid());

    auto validator2 = validate(42, "value");
    validator2.non_negative();
    EXPECT_TRUE(validator2.is_valid());
}

TEST_F(ValidationTest, Validator_NonNegative_Invalid) {
    auto validator = validate(-5, "value");
    validator.non_negative();

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be non-negative") != std::string::npos);
}

TEST_F(ValidationTest, Validator_Finite_Valid) {
    auto validator = validate(42.5, "value");
    validator.finite();

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, Validator_Finite_Invalid) {
    double inf = std::numeric_limits<double>::infinity();
    auto validator1 = validate(inf, "value");
    validator1.finite();

    EXPECT_FALSE(validator1.is_valid());

    double nan = std::numeric_limits<double>::quiet_NaN();
    auto validator2 = validate(nan, "value");
    validator2.finite();

    EXPECT_FALSE(validator2.is_valid());
}

TEST_F(ValidationTest, Validator_Satisfies_Valid) {
    auto is_even = [](const int& x) { return x % 2 == 0; };

    auto validator = validate(4, "value");
    validator.satisfies(is_even, "must be even");

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, Validator_Satisfies_Invalid) {
    auto is_even = [](const int& x) { return x % 2 == 0; };

    auto validator = validate(5, "value");
    validator.satisfies(is_even, "must be even");

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be even") != std::string::npos);
}

TEST_F(ValidationTest, Validator_ChainedValidations) {
    auto validator = validate(5, "value");
    validator.positive()
             .in_range(0, 10)
             .satisfies([](const int& x) { return x != 6; }, "must not be 6");

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, Validator_MultipleErrors) {
    auto validator = validate(-5, "value");
    validator.positive()
             .in_range(0, 10);

    EXPECT_FALSE(validator.is_valid());
    auto status = validator.status();
    EXPECT_FALSE(status.ok());
    // Both errors should be in the message
    EXPECT_TRUE(status.message().find("must be positive") != std::string::npos);
    EXPECT_TRUE(status.message().find("must be in range") != std::string::npos);
}

TEST_F(ValidationTest, Validator_ThrowIfInvalid) {
    auto validator = validate(-5, "value");
    validator.positive();

    EXPECT_THROW(validator.throw_if_invalid(), InvalidArgumentError);
}

// StringValidator tests
TEST_F(ValidationTest, StringValidator_NotEmpty_Valid) {
    auto validator = validate_string("hello", "name");
    validator.not_empty();

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, StringValidator_NotEmpty_Invalid) {
    auto validator = validate_string("", "name");
    validator.not_empty();

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must not be empty") != std::string::npos);
}

TEST_F(ValidationTest, StringValidator_MinLength_Valid) {
    auto validator = validate_string("hello", "name");
    validator.min_length(3);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, StringValidator_MinLength_Invalid) {
    auto validator = validate_string("hi", "name");
    validator.min_length(3);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be at least 3 characters") != std::string::npos);
}

TEST_F(ValidationTest, StringValidator_MaxLength_Valid) {
    auto validator = validate_string("hello", "name");
    validator.max_length(10);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, StringValidator_MaxLength_Invalid) {
    auto validator = validate_string("this is a very long string", "name");
    validator.max_length(10);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be at most 10 characters") != std::string::npos);
}

TEST_F(ValidationTest, StringValidator_Matches_Valid) {
    std::regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");

    auto validator = validate_string("test@example.com", "email");
    validator.matches(email_pattern, "email pattern");

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, StringValidator_Matches_Invalid) {
    std::regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");

    auto validator = validate_string("not_an_email", "email");
    validator.matches(email_pattern, "email pattern");

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must match email pattern") != std::string::npos);
}

TEST_F(ValidationTest, StringValidator_OneOf_Valid) {
    std::vector<std::string> allowed = {"red", "green", "blue"};

    auto validator = validate_string("red", "color");
    validator.one_of(allowed);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, StringValidator_OneOf_Invalid) {
    std::vector<std::string> allowed = {"red", "green", "blue"};

    auto validator = validate_string("yellow", "color");
    validator.one_of(allowed);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must be one of the allowed values") != std::string::npos);
}

TEST_F(ValidationTest, StringValidator_ChainedValidations) {
    auto validator = validate_string("hello", "username");
    validator.not_empty()
             .min_length(3)
             .max_length(20)
             .matches(std::regex("[a-z]+"), "lowercase letters");

    EXPECT_TRUE(validator.is_valid());
}

// CollectionValidator tests
TEST_F(ValidationTest, CollectionValidator_NotEmpty_Valid) {
    std::vector<int> vec{1, 2, 3};
    auto validator = validate_collection(vec, "vector");
    validator.not_empty();

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, CollectionValidator_NotEmpty_Invalid) {
    std::vector<int> vec;
    auto validator = validate_collection(vec, "vector");
    validator.not_empty();

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must not be empty") != std::string::npos);
}

TEST_F(ValidationTest, CollectionValidator_Size_Valid) {
    std::vector<int> vec{1, 2, 3};
    auto validator = validate_collection(vec, "vector");
    validator.size(3);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, CollectionValidator_Size_Invalid) {
    std::vector<int> vec{1, 2, 3};
    auto validator = validate_collection(vec, "vector");
    validator.size(5);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must have size 5") != std::string::npos);
}

TEST_F(ValidationTest, CollectionValidator_MinSize_Valid) {
    std::vector<int> vec{1, 2, 3};
    auto validator = validate_collection(vec, "vector");
    validator.min_size(2);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, CollectionValidator_MinSize_Invalid) {
    std::vector<int> vec{1};
    auto validator = validate_collection(vec, "vector");
    validator.min_size(2);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must have at least 2 elements") != std::string::npos);
}

TEST_F(ValidationTest, CollectionValidator_MaxSize_Valid) {
    std::vector<int> vec{1, 2, 3};
    auto validator = validate_collection(vec, "vector");
    validator.max_size(5);

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, CollectionValidator_MaxSize_Invalid) {
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    auto validator = validate_collection(vec, "vector");
    validator.max_size(5);

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("must have at most 5 elements") != std::string::npos);
}

TEST_F(ValidationTest, CollectionValidator_AllSatisfy_Valid) {
    std::vector<int> vec{2, 4, 6, 8};
    auto validator = validate_collection(vec, "vector");
    validator.all_satisfy([](const int& x) { return x % 2 == 0; }, "be even");

    EXPECT_TRUE(validator.is_valid());
}

TEST_F(ValidationTest, CollectionValidator_AllSatisfy_Invalid) {
    std::vector<int> vec{2, 3, 6, 8};
    auto validator = validate_collection(vec, "vector");
    validator.all_satisfy([](const int& x) { return x % 2 == 0; }, "be even");

    EXPECT_FALSE(validator.is_valid());
    EXPECT_TRUE(validator.status().message().find("all elements must be even") != std::string::npos);
}

TEST_F(ValidationTest, CollectionValidator_ChainedValidations) {
    std::vector<int> vec{2, 4, 6};
    auto validator = validate_collection(vec, "vector");
    validator.not_empty()
             .min_size(2)
             .max_size(10)
             .all_satisfy([](const int& x) { return x > 0; }, "be positive");

    EXPECT_TRUE(validator.is_valid());
}

// BatchValidator tests
TEST_F(ValidationTest, BatchValidator_AllValid) {
    BatchValidator batch;

    int value = 5;
    auto val1 = validate(value, "value");
    val1.positive().in_range(0, 10);

    std::string name = "test";
    auto val2 = validate_string(name, "name");
    val2.not_empty().min_length(3);

    std::vector<int> vec{1, 2, 3};
    auto val3 = validate_collection(vec, "vector");
    val3.not_empty().size(3);

    batch.add(val1).add(val2).add(val3);

    EXPECT_TRUE(batch.is_valid());
    EXPECT_TRUE(batch.status().ok());
}

TEST_F(ValidationTest, BatchValidator_SomeInvalid) {
    BatchValidator batch;

    int value = -5;
    auto val1 = validate(value, "value");
    val1.positive();

    std::string name = "";
    auto val2 = validate_string(name, "name");
    val2.not_empty();

    std::vector<int> vec{1, 2, 3};
    auto val3 = validate_collection(vec, "vector");
    val3.size(5);  // Wrong size

    batch.add(val1).add(val2).add(val3);

    EXPECT_FALSE(batch.is_valid());
    auto status = batch.status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status.code());
}

TEST_F(ValidationTest, BatchValidator_ThrowIfInvalid) {
    BatchValidator batch;

    int value = -5;
    auto val = validate(value, "value");
    val.positive();

    batch.add(val);

    EXPECT_THROW(batch.throw_if_invalid(), ErrorChainException);
}

// Integration tests
TEST_F(ValidationTest, IntegrationExample_FunctionWithValidation) {
    auto safe_divide = [](double numerator, double denominator) -> std::pair<Status, double> {
        // Validate inputs
        auto num_val = validate(numerator, "numerator");
        num_val.finite();

        auto denom_val = validate(denominator, "denominator");
        denom_val.finite()
                  .satisfies([](const double& x) { return x != 0; }, "must be non-zero");

        BatchValidator batch;
        batch.add(num_val).add(denom_val);

        if (!batch.is_valid()) {
            return {batch.status(), 0.0};
        }

        return {Status::OK(), numerator / denominator};
    };

    auto [status1, result1] = safe_divide(10.0, 2.0);
    EXPECT_TRUE(status1.ok());
    EXPECT_DOUBLE_EQ(5.0, result1);

    auto [status2, result2] = safe_divide(10.0, 0.0);
    EXPECT_FALSE(status2.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status2.code());

    double inf = std::numeric_limits<double>::infinity();
    auto [status3, result3] = safe_divide(inf, 2.0);
    EXPECT_FALSE(status3.ok());
}

TEST_F(ValidationTest, IntegrationExample_ConfigValidation) {
    struct ServerConfig {
        std::string host;
        int port;
        std::string protocol;
        std::vector<std::string> allowed_origins;
    };

    auto validate_config = [](const ServerConfig& config) -> Status {
        BatchValidator batch;

        // Validate host
        auto host_val = validate_string(config.host, "host");
        host_val.not_empty()
                .matches(std::regex(R"(^[a-zA-Z0-9.-]+$)"), "valid hostname");
        batch.add(host_val);

        // Validate port
        auto port_val = validate(config.port, "port");
        port_val.in_range(1, 65535);
        batch.add(port_val);

        // Validate protocol
        auto proto_val = validate_string(config.protocol, "protocol");
        proto_val.one_of({"http", "https", "ws", "wss"});
        batch.add(proto_val);

        // Validate allowed origins
        auto origins_val = validate_collection(config.allowed_origins, "allowed_origins");
        origins_val.min_size(1)
                   .all_satisfy([](const std::string& origin) {
                       return !origin.empty();
                   }, "be non-empty");
        batch.add(origins_val);

        return batch.status();
    };

    ServerConfig valid_config{
        "example.com",
        443,
        "https",
        {"https://app.example.com", "https://admin.example.com"}
    };

    EXPECT_TRUE(validate_config(valid_config).ok());

    ServerConfig invalid_config{
        "",          // Invalid: empty host
        70000,       // Invalid: port out of range
        "ftp",       // Invalid: not allowed protocol
        {}           // Invalid: empty origins
    };

    auto status = validate_config(invalid_config);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status.code());
}

TEST_F(ValidationTest, IntegrationExample_ChainedValidationWithThrow) {
    auto process_user_input = [](const std::string& username, int age) {
        // This function uses exceptions for error handling
        auto username_val = validate_string(username, "username");
        username_val.not_empty()
                    .min_length(3)
                    .max_length(20)
                    .matches(std::regex("^[a-zA-Z0-9_]+$"), "alphanumeric or underscore");

        username_val.throw_if_invalid();

        auto age_val = validate(age, "age");
        age_val.in_range(1, 150);

        age_val.throw_if_invalid();

        // Process valid input
        return std::format("User {} (age {}) registered", username, age);
    };

    EXPECT_NO_THROW({
        auto result = process_user_input("john_doe", 25);
        EXPECT_TRUE(result.find("john_doe") != std::string::npos);
    });

    EXPECT_THROW(process_user_input("ab", 25), InvalidArgumentError);  // Username too short
    EXPECT_THROW(process_user_input("user@123", 25), InvalidArgumentError);  // Invalid character
    EXPECT_THROW(process_user_input("valid_user", 200), InvalidArgumentError);  // Age out of range
}
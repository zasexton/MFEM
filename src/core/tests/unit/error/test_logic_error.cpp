#include <gtest/gtest.h>
#include <core/error/logic_error.h>
#include <core/error/error_code.h>
#include <typeinfo>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>
#include <limits>
#include <chrono>
#include <list>
#include <functional>

using namespace fem::core::error;

class LogicErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for tests if needed
    }
};

// ========== Basic LogicError tests ==========
TEST_F(LogicErrorTest, LogicErrorConstruction) {
    LogicError ex("Logic error message");
    EXPECT_STREQ(ex.what(), "Logic error message");
    EXPECT_EQ(ex.code(), ErrorCode::InvalidArgument);
}

TEST_F(LogicErrorTest, LogicErrorWithSourceLocation) {
    LogicError ex("Test error");
    const auto& loc = ex.where();
    EXPECT_TRUE(loc.file_name() != nullptr);
    EXPECT_TRUE(loc.function_name() != nullptr);
    EXPECT_GT(loc.line(), 0u);
}

TEST_F(LogicErrorTest, LogicErrorInheritance) {
    LogicError ex("Logic error");
    Exception* base = &ex;
    std::exception* std_base = &ex;

    EXPECT_STREQ(base->what(), "Logic error");
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
    EXPECT_STREQ(std_base->what(), "Logic error");
}

// ========== InvalidArgumentError comprehensive tests ==========
TEST_F(LogicErrorTest, InvalidArgumentErrorBasicConstruction) {
    InvalidArgumentError ex("param_name", "must be positive");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("param_name") != std::string::npos);
    EXPECT_TRUE(message.find("must be positive") != std::string::npos);
    EXPECT_TRUE(message.find("Invalid argument") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "param_name");
}

TEST_F(LogicErrorTest, InvalidArgumentErrorWithIntegerValue) {
    InvalidArgumentError ex("count", -5, "must be non-negative");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("count") != std::string::npos);
    EXPECT_TRUE(message.find("-5") != std::string::npos);
    EXPECT_TRUE(message.find("must be non-negative") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "count");
}

TEST_F(LogicErrorTest, InvalidArgumentErrorWithFloatingPoint) {
    InvalidArgumentError ex("ratio", 3.14159, "must be between 0 and 1");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("ratio") != std::string::npos);
    EXPECT_TRUE(message.find("3.14159") != std::string::npos);
    EXPECT_TRUE(message.find("must be between 0 and 1") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "ratio");
}

TEST_F(LogicErrorTest, InvalidArgumentErrorWithStringValue) {
    InvalidArgumentError ex("filename", std::string("invalid_file.txt"), "file does not exist");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("filename") != std::string::npos);
    EXPECT_TRUE(message.find("invalid_file.txt") != std::string::npos);
    EXPECT_TRUE(message.find("file does not exist") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "filename");
}

TEST_F(LogicErrorTest, InvalidArgumentErrorWithBooleanValue) {
    InvalidArgumentError ex("enable_logging", false, "must be true for debug builds");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("enable_logging") != std::string::npos);
    EXPECT_TRUE(message.find("false") != std::string::npos || message.find("0") != std::string::npos);
    EXPECT_TRUE(message.find("must be true for debug builds") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "enable_logging");
}

// Global operator<< for CustomType to fix the friend operator issue
struct CustomType {
    int value;
};

std::ostream& operator<<(std::ostream& os, const CustomType& ct) {
    return os << "CustomType{" << ct.value << "}";
}

// Add std::formatter specialization for CustomType
template<>
struct std::formatter<CustomType> : std::formatter<std::string> {
    auto format(const CustomType& ct, std::format_context& ctx) const {
        return std::formatter<std::string>::format(std::format("CustomType{{{}}}", ct.value), ctx);
    }
};

TEST_F(LogicErrorTest, InvalidArgumentErrorWithCustomType) {
    CustomType custom{42};
    InvalidArgumentError ex("config", custom, "invalid configuration");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("config") != std::string::npos);
    EXPECT_TRUE(message.find("42") != std::string::npos);
    EXPECT_TRUE(message.find("invalid configuration") != std::string::npos);
    EXPECT_EQ(ex.argument_name(), "config");
}

TEST_F(LogicErrorTest, InvalidArgumentErrorInheritance) {
    InvalidArgumentError ex("param", "invalid");
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
}

// ========== DomainError tests ==========
TEST_F(LogicErrorTest, DomainErrorConstruction) {
    DomainError ex("sqrt", "negative input");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("sqrt") != std::string::npos);
    EXPECT_TRUE(message.find("negative input") != std::string::npos);
    EXPECT_TRUE(message.find("Domain error") != std::string::npos);
    EXPECT_EQ(ex.function_name(), "sqrt");
}

TEST_F(LogicErrorTest, DomainErrorMathematicalContext) {
    DomainError ex("log", "argument must be positive");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("log") != std::string::npos);
    EXPECT_TRUE(message.find("argument must be positive") != std::string::npos);
}

// ========== LengthError tests ==========
TEST_F(LogicErrorTest, LengthErrorConstruction) {
    LengthError ex("vector resize", 1000, 500);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("vector resize") != std::string::npos);
    EXPECT_TRUE(message.find("1000") != std::string::npos);
    EXPECT_TRUE(message.find("500") != std::string::npos);
    EXPECT_EQ(ex.requested(), 1000);
    EXPECT_EQ(ex.maximum(), 500);
}

TEST_F(LogicErrorTest, LengthErrorStringAllocation) {
    LengthError ex("string allocation",
                   std::string::npos,
                   std::string::npos - 1);

    EXPECT_EQ(ex.requested(), std::string::npos);
    EXPECT_EQ(ex.maximum(), std::string::npos - 1);
}

// ========== OutOfRangeError tests ==========
TEST_F(LogicErrorTest, OutOfRangeErrorIndexBased) {
    OutOfRangeError ex("array access", 10, 5);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("array access") != std::string::npos);
    EXPECT_TRUE(message.find("10") != std::string::npos);
    EXPECT_TRUE(message.find("5") != std::string::npos);
    EXPECT_TRUE(message.find("out of range") != std::string::npos);
    EXPECT_EQ(ex.index(), 10);
    EXPECT_EQ(ex.size(), 5);
}

TEST_F(LogicErrorTest, OutOfRangeErrorValueBasedInteger) {
    OutOfRangeError ex("temperature", 150, -50, 100);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("temperature") != std::string::npos);
    EXPECT_TRUE(message.find("150") != std::string::npos);
    EXPECT_TRUE(message.find("-50") != std::string::npos);
    EXPECT_TRUE(message.find("100") != std::string::npos);
}

TEST_F(LogicErrorTest, OutOfRangeErrorValueBasedFloat) {
    OutOfRangeError ex("probability", 1.5, 0.0, 1.0);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("probability") != std::string::npos);
    EXPECT_TRUE(message.find("1.5") != std::string::npos);
    EXPECT_TRUE(message.find("0") != std::string::npos);
    EXPECT_TRUE(message.find("1") != std::string::npos);
}

TEST_F(LogicErrorTest, OutOfRangeErrorInheritance) {
    OutOfRangeError ex("test", 1, 1);
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
}

// ========== InvalidStateError tests ==========
TEST_F(LogicErrorTest, InvalidStateErrorConstruction) {
    InvalidStateError ex("Connection", "CONNECTED", "DISCONNECTED");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Connection") != std::string::npos);
    EXPECT_TRUE(message.find("CONNECTED") != std::string::npos);
    EXPECT_TRUE(message.find("DISCONNECTED") != std::string::npos);
    EXPECT_EQ(ex.object_name(), "Connection");
    EXPECT_EQ(ex.expected_state(), "CONNECTED");
    EXPECT_EQ(ex.actual_state(), "DISCONNECTED");
}

TEST_F(LogicErrorTest, InvalidStateErrorStateMachine) {
    InvalidStateError ex("FileStream", "OPEN", "CLOSED");

    EXPECT_EQ(ex.object_name(), "FileStream");
    EXPECT_EQ(ex.expected_state(), "OPEN");
    EXPECT_EQ(ex.actual_state(), "CLOSED");
}

// ========== TypeMismatchError tests ==========
TEST_F(LogicErrorTest, TypeMismatchErrorConstruction) {
    TypeMismatchError ex("cast operation", typeid(int), typeid(double));
    std::string message = ex.what();

    EXPECT_TRUE(message.find("cast operation") != std::string::npos);
    EXPECT_TRUE(message.find("type mismatch") != std::string::npos);
    EXPECT_EQ(&ex.expected_type(), &typeid(int));
    EXPECT_EQ(&ex.actual_type(), &typeid(double));
}

TEST_F(LogicErrorTest, TypeMismatchErrorVariousTypes) {
    struct TestCases {
        std::string context;
        const std::type_info& expected;
        const std::type_info& actual;
    };

    std::vector<TestCases> cases = {
        {"template instantiation", typeid(std::string), typeid(const char*)},
        {"function call", typeid(std::vector<int>), typeid(std::list<int>)},
        {"cast operation", typeid(float), typeid(int)},
        {"container element", typeid(std::unique_ptr<int>), typeid(std::shared_ptr<int>)},
        {"callback parameter", typeid(std::function<void()>), typeid(void(*)())},
        {"enum conversion", typeid(std::error_code), typeid(int)}
    };

    for (const auto& test_case : cases) {
        TypeMismatchError ex(test_case.context, test_case.expected, test_case.actual);
        std::string message = ex.what();

        EXPECT_TRUE(message.find(test_case.context) != std::string::npos);
        EXPECT_TRUE(message.find("type mismatch") != std::string::npos);
        EXPECT_EQ(&ex.expected_type(), &test_case.expected);
        EXPECT_EQ(&ex.actual_type(), &test_case.actual);
    }
}

// ========== NullPointerError tests ==========
TEST_F(LogicErrorTest, NullPointerErrorConstruction) {
    NullPointerError ex("data_ptr");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("data_ptr") != std::string::npos);
    EXPECT_TRUE(message.find("Null pointer") != std::string::npos);
    EXPECT_EQ(ex.pointer_name(), "data_ptr");
}

TEST_F(LogicErrorTest, NullPointerErrorVariousPointers) {
    std::vector<std::string> pointer_names = {
        "buffer", "callback", "context", "user_data", "resource"
    };

    for (const auto& name : pointer_names) {
        NullPointerError ex(name);
        EXPECT_EQ(ex.pointer_name(), name);
        EXPECT_TRUE(std::string(ex.what()).find(name) != std::string::npos);
    }
}

// ========== NotImplementedError tests ==========
TEST_F(LogicErrorTest, NotImplementedErrorConstruction) {
    NotImplementedError ex("advanced_feature");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Not implemented") != std::string::npos);
    EXPECT_TRUE(message.find("advanced_feature") != std::string::npos);
    EXPECT_EQ(ex.feature(), "advanced_feature");
}

TEST_F(LogicErrorTest, NotImplementedErrorVariousFeatures) {
    std::vector<std::string> features = {
        "GPU acceleration", "network protocol", "file format",
        "compression algorithm", "encryption method"
    };

    for (const auto& feature : features) {
        NotImplementedError ex(feature);
        EXPECT_EQ(ex.feature(), feature);
        EXPECT_TRUE(std::string(ex.what()).find(feature) != std::string::npos);
    }
}

TEST_F(LogicErrorTest, NotImplementedErrorInheritance) {
    NotImplementedError ex("feature");
    LogicError* logic = &ex;
    Exception* base = &ex;

    EXPECT_EQ(logic->code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(base->code(), ErrorCode::InvalidArgument);
}

// ========== PreconditionError tests ==========
TEST_F(LogicErrorTest, PreconditionErrorBasic) {
    PreconditionError ex("divide", "divisor != 0");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("divide") != std::string::npos);
    EXPECT_TRUE(message.find("divisor != 0") != std::string::npos);
    EXPECT_TRUE(message.find("Precondition violation") != std::string::npos);
    EXPECT_EQ(ex.function_name(), "divide");
    EXPECT_EQ(ex.condition(), "divisor != 0");
}

TEST_F(LogicErrorTest, PreconditionErrorWithMessage) {
    PreconditionError ex("calculate_sqrt", "x >= 0", "Square root of negative number");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("calculate_sqrt") != std::string::npos);
    EXPECT_TRUE(message.find("x >= 0") != std::string::npos);
    EXPECT_TRUE(message.find("Square root of negative number") != std::string::npos);
}

// ========== PostconditionError tests ==========
TEST_F(LogicErrorTest, PostconditionErrorBasic) {
    PostconditionError ex("sort_array", "array is sorted");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("sort_array") != std::string::npos);
    EXPECT_TRUE(message.find("array is sorted") != std::string::npos);
    EXPECT_TRUE(message.find("Postcondition violation") != std::string::npos);
    EXPECT_EQ(ex.function_name(), "sort_array");
    EXPECT_EQ(ex.condition(), "array is sorted");
}

TEST_F(LogicErrorTest, PostconditionErrorWithMessage) {
    PostconditionError ex("allocate_memory", "result != nullptr", "Memory allocation failed");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("allocate_memory") != std::string::npos);
    EXPECT_TRUE(message.find("result != nullptr") != std::string::npos);
    EXPECT_TRUE(message.find("Memory allocation failed") != std::string::npos);
}

// ========== InvariantError tests ==========
TEST_F(LogicErrorTest, InvariantErrorBasic) {
    InvariantError ex("BinaryTree", "left_height - right_height <= 1");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("BinaryTree") != std::string::npos);
    EXPECT_TRUE(message.find("left_height - right_height <= 1") != std::string::npos);
    EXPECT_TRUE(message.find("Invariant violation") != std::string::npos);
    EXPECT_EQ(ex.class_name(), "BinaryTree");
    EXPECT_EQ(ex.invariant(), "left_height - right_height <= 1");
}

TEST_F(LogicErrorTest, InvariantErrorWithMessage) {
    InvariantError ex("Stack", "size <= capacity", "Stack overflow detected");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Stack") != std::string::npos);
    EXPECT_TRUE(message.find("size <= capacity") != std::string::npos);
    EXPECT_TRUE(message.find("Stack overflow detected") != std::string::npos);
}

// ========== Exception throwing and catching tests ==========
TEST_F(LogicErrorTest, ThrowAndCatchLogicError) {
    bool caught = false;
    try {
        throw LogicError("Test logic error");
    } catch (const LogicError& e) {
        caught = true;
        EXPECT_STREQ(e.what(), "Test logic error");
        EXPECT_EQ(e.code(), ErrorCode::InvalidArgument);
    }
    EXPECT_TRUE(caught);
}

TEST_F(LogicErrorTest, ThrowAndCatchSpecificError) {
    bool caught_invalid_arg = false;
    bool caught_logic = false;

    try {
        throw InvalidArgumentError("param", "invalid value");
    } catch (const InvalidArgumentError& e) {
        caught_invalid_arg = true;
        EXPECT_EQ(e.argument_name(), "param");
    } catch (const LogicError& e) {
        caught_logic = true;
        FAIL() << "Should have caught InvalidArgumentError specifically";
    }

    EXPECT_TRUE(caught_invalid_arg);
    EXPECT_FALSE(caught_logic);
}

TEST_F(LogicErrorTest, ThrowAndCatchAsException) {
    bool caught = false;
    try {
        throw OutOfRangeError("array", 5, 3);
    } catch (const Exception& e) {
        caught = true;
        EXPECT_EQ(e.code(), ErrorCode::InvalidArgument);
    }
    EXPECT_TRUE(caught);
}

// ========== Polymorphic behavior tests ==========
TEST_F(LogicErrorTest, PolymorphicBehavior) {
    std::vector<std::unique_ptr<LogicError>> errors;

    errors.push_back(std::make_unique<InvalidArgumentError>("param", "bad"));
    errors.push_back(std::make_unique<OutOfRangeError>("index", 10, 5));
    errors.push_back(std::make_unique<NotImplementedError>("feature"));

    for (const auto& error : errors) {
        EXPECT_EQ(error->code(), ErrorCode::InvalidArgument);
        EXPECT_FALSE(std::string(error->what()).empty());
    }
}

// ========== Real-world scenario tests ==========
TEST_F(LogicErrorTest, ArrayBoundsChecking) {
    auto safe_array_access = [](const std::vector<int>& arr, size_t index) -> int {
        if (index >= arr.size()) {
            throw OutOfRangeError("array access", index, arr.size());
        }
        return arr[index];
    };

    std::vector<int> data = {1, 2, 3};

    EXPECT_EQ(safe_array_access(data, 0), 1);
    EXPECT_EQ(safe_array_access(data, 2), 3);

    EXPECT_THROW(safe_array_access(data, 3), OutOfRangeError);
    EXPECT_THROW(safe_array_access(data, 100), OutOfRangeError);
}

TEST_F(LogicErrorTest, ParameterValidation) {
    auto validate_name = [](const std::string& name) {
        if (name.empty()) {
            throw InvalidArgumentError("name", "cannot be empty");
        }
    };

    EXPECT_NO_THROW(validate_name("valid"));
    EXPECT_THROW(validate_name(""), InvalidArgumentError);
}

// ========== Context and formatting tests ==========
TEST_F(LogicErrorTest, ErrorWithContext) {
    InvalidArgumentError ex("count", "must be positive");
    ex.with_context("in function calculate_factorial")
      .with_context("called from main");

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("count") != std::string::npos);
    EXPECT_TRUE(full_msg.find("calculate_factorial") != std::string::npos);
    EXPECT_TRUE(full_msg.find("called from main") != std::string::npos);
}

// ========== Template error tests ==========
TEST_F(LogicErrorTest, TemplateParameterValidation) {
    auto validate_positive = [](const auto& value, const std::string& name) {
        if (value <= 0) {
            throw InvalidArgumentError(name, value, "must be positive");
        }
    };

    const int negative_int = -5;
    const double negative_double = -3.14;
    const std::string log_level = "invalid";

    EXPECT_THROW(validate_positive(negative_int, "buffer_size"), InvalidArgumentError);
    EXPECT_THROW(validate_positive(negative_double, "learning_rate"), InvalidArgumentError);

    // Test with string that requires conversion to proper error message
    EXPECT_THROW(InvalidArgumentError("log_level", log_level, "must be one of: debug, info, warning, error"),
                 InvalidArgumentError);
}

// ========== Numeric limits tests ==========
TEST_F(LogicErrorTest, NumericLimitsInOutOfRange) {
    OutOfRangeError ex1("integer_value", std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::min(), std::numeric_limits<int>::max() - 1);
    OutOfRangeError ex2("float_value", std::numeric_limits<float>::infinity(),
                        0.0f, std::numeric_limits<float>::max());
    OutOfRangeError ex3("double_precision", 1.23456789012345, 0.0, 1.0);

    // Just verify they can be constructed without throwing
    EXPECT_NO_THROW(ex1.what());
    EXPECT_NO_THROW(ex2.what());
    EXPECT_NO_THROW(ex3.what());
}

// ========== Edge cases and stress tests ==========
TEST_F(LogicErrorTest, VeryLongErrorMessages) {
    std::string long_arg_name(1000, 'a');
    std::string long_reason(1000, 'b');

    InvalidArgumentError ex(long_arg_name, long_reason);
    std::string message = ex.what();

    EXPECT_TRUE(message.find(long_arg_name) != std::string::npos);
    EXPECT_TRUE(message.find(long_reason) != std::string::npos);
}

TEST_F(LogicErrorTest, UnicodeInErrorMessages) {
    InvalidArgumentError ex("файл", "файл не найден");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("файл") != std::string::npos);
    EXPECT_TRUE(message.find("файл не найден") != std::string::npos);
}

TEST_F(LogicErrorTest, EmptyStringsHandling) {
    InvalidArgumentError ex("", "");
    EXPECT_NO_THROW(ex.what());
    EXPECT_EQ(ex.argument_name(), "");
}
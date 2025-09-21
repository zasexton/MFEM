#include <gtest/gtest.h>
#include <core/error/contract.h>
#include <vector>
#include <string>
#include <cmath>

using namespace fem::core::error;

class ContractTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset handler to default behavior
        ContractChecker::set_handler(nullptr);
    }
    void TearDown() override {
        // Reset handler to default behavior
        ContractChecker::set_handler(nullptr);
    }
};

// ContractViolation tests
TEST_F(ContractTest, ContractViolation_Precondition) {
    ContractViolation violation(
        ContractViolation::Type::Precondition,
        "x > 0",
        "x must be positive"
    );

    EXPECT_EQ(ContractViolation::Type::Precondition, violation.type());
    EXPECT_EQ("x > 0", violation.condition());
    std::string msg = violation.what();
    EXPECT_TRUE(msg.find("Precondition violated") != std::string::npos);
}

TEST_F(ContractTest, ContractViolation_Postcondition) {
    ContractViolation violation(
        ContractViolation::Type::Postcondition,
        "result != nullptr",
        "result must not be null"
    );

    EXPECT_EQ(ContractViolation::Type::Postcondition, violation.type());
    EXPECT_EQ("result != nullptr", violation.condition());
    std::string msg = violation.what();
    EXPECT_TRUE(msg.find("Postcondition violated") != std::string::npos);
}

TEST_F(ContractTest, ContractViolation_Invariant) {
    ContractViolation violation(
        ContractViolation::Type::Invariant,
        "size <= capacity",
        "size exceeds capacity"
    );

    EXPECT_EQ(ContractViolation::Type::Invariant, violation.type());
    EXPECT_EQ("size <= capacity", violation.condition());
    std::string msg = violation.what();
    EXPECT_TRUE(msg.find("Invariant violated") != std::string::npos);
}

// ContractChecker tests
TEST_F(ContractTest, ContractChecker_Require_Pass) {
    EXPECT_NO_THROW(
        ContractChecker::require(true, "true", "always true")
    );
}

TEST_F(ContractTest, ContractChecker_Require_Fail) {
    EXPECT_THROW(
        ContractChecker::require(false, "false", "always false"),
        ContractViolation
    );

    try {
        ContractChecker::require(false, "x > 0", "x must be positive");
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Precondition, e.type());
        EXPECT_EQ("x > 0", e.condition());
    }
}

TEST_F(ContractTest, ContractChecker_Ensure_Pass) {
    EXPECT_NO_THROW(
        ContractChecker::ensure(true, "true", "always true")
    );
}

TEST_F(ContractTest, ContractChecker_Ensure_Fail) {
    EXPECT_THROW(
        ContractChecker::ensure(false, "false", "always false"),
        ContractViolation
    );

    try {
        ContractChecker::ensure(false, "result > 0", "result must be positive");
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Postcondition, e.type());
        EXPECT_EQ("result > 0", e.condition());
    }
}

TEST_F(ContractTest, ContractChecker_Invariant_Pass) {
    EXPECT_NO_THROW(
        ContractChecker::invariant(true, "true", "always true")
    );
}

TEST_F(ContractTest, ContractChecker_Invariant_Fail) {
    EXPECT_THROW(
        ContractChecker::invariant(false, "false", "always false"),
        ContractViolation
    );

    try {
        ContractChecker::invariant(false, "balanced", "tree must be balanced");
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Invariant, e.type());
        EXPECT_EQ("balanced", e.condition());
    }
}

TEST_F(ContractTest, ContractChecker_CustomHandler) {
    bool handler_called = false;
    ContractViolation::Type captured_type;
    std::string captured_condition;

    ContractChecker::set_handler([&](const ContractViolation& violation) {
        handler_called = true;
        captured_type = violation.type();
        captured_condition = violation.condition();
    });

    // Custom handler should prevent exception from being thrown
    EXPECT_NO_THROW(
        ContractChecker::require(false, "test", "test message")
    );

    EXPECT_TRUE(handler_called);
    EXPECT_EQ(ContractViolation::Type::Precondition, captured_type);
    EXPECT_EQ("test", captured_condition);
}

// InvariantGuard tests
TEST_F(ContractTest, InvariantGuard_ValidInvariant) {
    bool invariant_valid = true;

    {
        InvariantGuard guard(
            [&]() { return invariant_valid; },
            "test_invariant"
        );
        // Invariant holds throughout
    }

    SUCCEED();  // No exception thrown
}

TEST_F(ContractTest, InvariantGuard_InvalidAtEntry) {
    bool invariant_valid = false;

    EXPECT_THROW({
        InvariantGuard guard(
            [&]() { return invariant_valid; },
            "test_invariant"
        );
    }, ContractViolation);
}

TEST_F(ContractTest, InvariantGuard_InvalidAtExit) {
    bool invariant_valid = true;

    {
        InvariantGuard guard(
            [&]() { return invariant_valid; },
            "test_invariant"
        );

        invariant_valid = false;  // Invalidate during scope
    }
    // Destructor suppresses exception

    SUCCEED();
}

TEST_F(ContractTest, InvariantGuard_ExceptionMessage) {
    bool invariant_valid = false;

    try {
        InvariantGuard guard(
            [&]() { return invariant_valid; },
            "size_check"
        );
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Invariant, e.type());
        EXPECT_EQ("size_check", e.condition());
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Entry") != std::string::npos);
    }
}

// FunctionContract tests
TEST_F(ContractTest, FunctionContract_NoPreconditions) {
    auto func = make_contracted(std::function<int(int)>([](int x) { return x * 2; }));

    EXPECT_EQ(10, func(5));
    EXPECT_EQ(-10, func(-5));
}

TEST_F(ContractTest, FunctionContract_PreconditionPass) {
    auto func = make_contracted(std::function<int(int)>([](int x) { return x * 2; }));
    func.require([](int x) { return x > 0; }, "x must be positive");

    EXPECT_EQ(10, func(5));
    EXPECT_EQ(20, func(10));
}

TEST_F(ContractTest, FunctionContract_PreconditionFail) {
    auto func = make_contracted(std::function<int(int)>([](int x) { return x * 2; }));
    func.require([](int x) { return x > 0; }, "x must be positive");

    EXPECT_THROW(func(-5), ContractViolation);

    try {
        func(-5);
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Precondition, e.type());
        EXPECT_EQ("x must be positive", e.condition());
    }
}

TEST_F(ContractTest, FunctionContract_PostconditionPass) {
    auto func = make_contracted(std::function<int(int)>([](int x) { return x * 2; }));
    func.ensure([](const int& result, int x) { return result == x * 2; }, "result is double");

    EXPECT_EQ(10, func(5));
    EXPECT_EQ(-10, func(-5));
}

TEST_F(ContractTest, FunctionContract_PostconditionFail) {
    // Simulate a buggy function that doesn't meet its postcondition
    auto func = make_contracted(std::function<int(int)>([](int x) { return x + 1; }));
    func.ensure([](const int& result, int x) { return result == x * 2; }, "result is double");

    EXPECT_THROW(func(5), ContractViolation);

    try {
        func(5);
        FAIL() << "Expected ContractViolation to be thrown";
    } catch (const ContractViolation& e) {
        EXPECT_EQ(ContractViolation::Type::Postcondition, e.type());
        EXPECT_EQ("result is double", e.condition());
    }
}

TEST_F(ContractTest, FunctionContract_MultiplePreconditions) {
    auto func = make_contracted(std::function<int(int, int)>([](int x, int y) {
        return x / y;
    }));

    func.require([](int, int y) { return y != 0; }, "divisor non-zero")
        .require([](int x, int) { return x >= 0; }, "dividend non-negative");

    EXPECT_EQ(2, func(10, 5));

    EXPECT_THROW(func(10, 0), ContractViolation);  // Violates first precondition
    EXPECT_THROW(func(-10, 5), ContractViolation); // Violates second precondition
}

TEST_F(ContractTest, FunctionContract_MultiplePostconditions) {
    auto func = make_contracted(std::function<int(int)>([](int x) {
        return x * x;
    }));

    func.ensure([](const int& result, int) { return result >= 0; }, "result non-negative")
        .ensure([](const int& result, int x) { return result == x * x; }, "result is square");

    EXPECT_EQ(25, func(5));
    EXPECT_EQ(25, func(-5));
}

TEST_F(ContractTest, FunctionContract_ChainedContracts) {
    auto safe_divide = make_contracted(std::function<double(double, double)>(
        [](double x, double y) { return x / y; }
    ));

    safe_divide
        .require([](double, double y) { return std::abs(y) > 1e-10; }, "divisor must be non-zero")
        .require([](double x, double y) { return std::isfinite(x) && std::isfinite(y); }, "inputs must be finite")
        .ensure([](const double& result, double, double) { return std::isfinite(result); }, "result must be finite");

    EXPECT_DOUBLE_EQ(2.5, safe_divide(5.0, 2.0));

    EXPECT_THROW(safe_divide(5.0, 0.0), ContractViolation);

    double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(safe_divide(inf, 2.0), ContractViolation);
}

// SoftContract tests
TEST_F(ContractTest, SoftContract_CheckPrecondition_Pass) {
    auto status = SoftContract::check_precondition(true, "always true");
    EXPECT_TRUE(status.ok());
}

TEST_F(ContractTest, SoftContract_CheckPrecondition_Fail) {
    auto status = SoftContract::check_precondition(false, "x must be positive");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status.code());
    EXPECT_TRUE(status.message().find("Precondition failed") != std::string::npos);
    EXPECT_TRUE(status.message().find("x must be positive") != std::string::npos);
}

TEST_F(ContractTest, SoftContract_CheckPostcondition_Pass) {
    auto status = SoftContract::check_postcondition(true, "always true");
    EXPECT_TRUE(status.ok());
}

TEST_F(ContractTest, SoftContract_CheckPostcondition_Fail) {
    auto status = SoftContract::check_postcondition(false, "result must be valid");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidState, status.code());
    EXPECT_TRUE(status.message().find("Postcondition failed") != std::string::npos);
    EXPECT_TRUE(status.message().find("result must be valid") != std::string::npos);
}

TEST_F(ContractTest, SoftContract_CheckInvariant_Pass) {
    auto status = SoftContract::check_invariant(true, "always true");
    EXPECT_TRUE(status.ok());
}

TEST_F(ContractTest, SoftContract_CheckInvariant_Fail) {
    auto status = SoftContract::check_invariant(false, "tree must be balanced");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ErrorCode::InvalidState, status.code());
    EXPECT_TRUE(status.message().find("Invariant violated") != std::string::npos);
    EXPECT_TRUE(status.message().find("tree must be balanced") != std::string::npos);
}

// Integration tests
TEST_F(ContractTest, IntegrationExample_SafeVector) {
    // A vector wrapper with contracts
    class SafeVector {
    public:
        SafeVector(size_t capacity) : capacity_(capacity) {
            data_.reserve(capacity);
        }

        void push(int value) {
            ContractChecker::require(size() < capacity_, "size < capacity", "vector not full");

            size_t old_size = size();
            data_.push_back(value);

            ContractChecker::ensure(size() == old_size + 1, "size increased", "size must increase by 1");
            ContractChecker::invariant(size() <= capacity_, "size <= capacity", "invariant maintained");
        }

        int pop() {
            ContractChecker::require(!empty(), "!empty", "vector not empty");

            size_t old_size = size();
            int value = data_.back();
            data_.pop_back();

            ContractChecker::ensure(size() == old_size - 1, "size decreased", "size must decrease by 1");
            ContractChecker::invariant(size() <= capacity_, "size <= capacity", "invariant maintained");

            return value;
        }

        size_t size() const { return data_.size(); }
        bool empty() const { return data_.empty(); }

    private:
        std::vector<int> data_;
        size_t capacity_;
    };

    SafeVector vec(3);

    EXPECT_NO_THROW(vec.push(1));
    EXPECT_NO_THROW(vec.push(2));
    EXPECT_NO_THROW(vec.push(3));

    // Vector is full
    EXPECT_THROW(vec.push(4), ContractViolation);

    EXPECT_EQ(3, vec.pop());
    EXPECT_EQ(2, vec.pop());
    EXPECT_EQ(1, vec.pop());

    // Vector is empty
    EXPECT_THROW(vec.pop(), ContractViolation);
}

TEST_F(ContractTest, IntegrationExample_FunctionWithSoftContracts) {
    // Function that uses soft contracts for error handling without exceptions
    auto safe_sqrt = [](double x) -> std::pair<Status, double> {
        // Check precondition
        if (auto status = SoftContract::check_precondition(x >= 0, "x must be non-negative");
            !status.ok()) {
            return {status, 0.0};
        }

        double result = std::sqrt(x);

        // Check postcondition
        if (auto status = SoftContract::check_postcondition(
                std::abs(result * result - x) < 1e-9,
                "result squared equals x");
            !status.ok()) {
            return {status, 0.0};
        }

        return {Status::OK(), result};
    };

    auto [status1, result1] = safe_sqrt(4.0);
    EXPECT_TRUE(status1.ok());
    EXPECT_DOUBLE_EQ(2.0, result1);

    auto [status2, result2] = safe_sqrt(-1.0);
    EXPECT_FALSE(status2.ok());
    EXPECT_EQ(ErrorCode::InvalidArgument, status2.code());
}
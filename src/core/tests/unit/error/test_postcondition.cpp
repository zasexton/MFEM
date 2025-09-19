#include <gtest/gtest.h>
#include <core/error/postcondition.h>
#include <cmath>
#include <vector>
#include <string>

using namespace fem::core::error;

class PostconditionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// PostconditionChecker tests
TEST_F(PostconditionTest, PostconditionChecker_EnsureChanged_Valid) {
    PostconditionChecker<int> checker(10);
    EXPECT_NO_THROW(checker.ensure_changed(20, "value"));
    EXPECT_NO_THROW(checker.ensure_changed(5, "value"));
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureChanged_Invalid) {
    PostconditionChecker<int> checker(10);
    EXPECT_THROW(checker.ensure_changed(10, "value"), PostconditionError);
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureUnchanged_Valid) {
    PostconditionChecker<int> checker(10);
    EXPECT_NO_THROW(checker.ensure_unchanged(10, "value"));

    PostconditionChecker<std::string> str_checker("hello");
    EXPECT_NO_THROW(str_checker.ensure_unchanged("hello", "string"));
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureUnchanged_Invalid) {
    PostconditionChecker<int> checker(10);
    EXPECT_THROW(checker.ensure_unchanged(20, "value"), PostconditionError);
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureIncreased_Valid) {
    PostconditionChecker<int> checker(10);
    EXPECT_NO_THROW(checker.ensure_increased(11, "value"));
    EXPECT_NO_THROW(checker.ensure_increased(100, "value"));
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureIncreased_Invalid) {
    PostconditionChecker<int> checker(10);
    EXPECT_THROW(checker.ensure_increased(10, "value"), PostconditionError);
    EXPECT_THROW(checker.ensure_increased(9, "value"), PostconditionError);
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureDecreased_Valid) {
    PostconditionChecker<int> checker(10);
    EXPECT_NO_THROW(checker.ensure_decreased(9, "value"));
    EXPECT_NO_THROW(checker.ensure_decreased(0, "value"));
}

TEST_F(PostconditionTest, PostconditionChecker_EnsureDecreased_Invalid) {
    PostconditionChecker<int> checker(10);
    EXPECT_THROW(checker.ensure_decreased(10, "value"), PostconditionError);
    EXPECT_THROW(checker.ensure_decreased(11, "value"), PostconditionError);
}

TEST_F(PostconditionTest, PostconditionChecker_OldValue) {
    PostconditionChecker<int> checker(42);
    EXPECT_EQ(42, checker.old_value());
}

// ReturnValueValidator tests
TEST_F(PostconditionTest, ReturnValueValidator_NotNull_Valid) {
    int value = 42;
    int* ptr = &value;
    auto validator = ReturnValueValidator<int*>(ptr, "test_func");
    validator.not_null();
    EXPECT_NO_THROW(validator.validate());
}

TEST_F(PostconditionTest, ReturnValueValidator_NotNull_Invalid) {
    int* ptr = nullptr;
    auto validator = ReturnValueValidator<int*>(ptr, "test_func");
    validator.not_null();
    EXPECT_THROW(validator.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_InRange_Valid) {
    auto validator = ReturnValueValidator<int>(5, "test_func");
    validator.in_range(0, 10);
    EXPECT_NO_THROW(validator.validate());
}

TEST_F(PostconditionTest, ReturnValueValidator_InRange_Invalid) {
    auto validator = ReturnValueValidator<int>(15, "test_func");
    validator.in_range(0, 10);
    EXPECT_THROW(validator.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_Positive_Valid) {
    auto validator = ReturnValueValidator<double>(42.5, "test_func");
    validator.positive();
    EXPECT_NO_THROW(validator.validate());
}

TEST_F(PostconditionTest, ReturnValueValidator_Positive_Invalid) {
    auto validator = ReturnValueValidator<double>(-1.0, "test_func");
    validator.positive();
    EXPECT_THROW(validator.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_Finite_Valid) {
    auto validator = ReturnValueValidator<double>(42.5, "test_func");
    validator.finite();
    EXPECT_NO_THROW(validator.validate());
}

TEST_F(PostconditionTest, ReturnValueValidator_Finite_Invalid) {
    double inf = std::numeric_limits<double>::infinity();
    auto validator = ReturnValueValidator<double>(inf, "test_func");
    validator.finite();
    EXPECT_THROW(validator.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_Satisfies_CustomPredicate) {
    auto is_even = [](const int& x) { return x % 2 == 0; };

    auto validator1 = ReturnValueValidator<int>(4, "test_func");
    validator1.satisfies(is_even, "is not even");
    EXPECT_NO_THROW(validator1.validate());

    auto validator2 = ReturnValueValidator<int>(5, "test_func");
    validator2.satisfies(is_even, "is not even");
    EXPECT_THROW(validator2.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_ChainedValidations) {
    auto validator1 = ReturnValueValidator<int>(5, "test_func");
    validator1.positive().in_range(0, 10);
    EXPECT_NO_THROW(validator1.validate());

    auto validator2 = ReturnValueValidator<int>(-5, "test_func");
    validator2.positive().in_range(0, 10);
    EXPECT_THROW(validator2.validate(), PostconditionError);
}

TEST_F(PostconditionTest, ReturnValueValidator_Get) {
    auto validator = ReturnValueValidator<int>(42, "test_func");
    EXPECT_EQ(42, validator.get());
}

// StateTransition tests
TEST_F(PostconditionTest, StateTransition_SimpleTransition_Valid) {
    StateTransition sm("StateMachine");
    sm.allow("idle", "running")
      .allow("running", "stopped")
      .allow("stopped", "idle");

    EXPECT_NO_THROW(sm.validate("idle", "running"));
    EXPECT_NO_THROW(sm.validate("running", "stopped"));
    EXPECT_NO_THROW(sm.validate("stopped", "idle"));
}

TEST_F(PostconditionTest, StateTransition_SimpleTransition_Invalid) {
    StateTransition sm("StateMachine");
    sm.allow("idle", "running")
      .allow("running", "stopped");

    EXPECT_THROW(sm.validate("idle", "stopped"), PostconditionError);
    EXPECT_THROW(sm.validate("stopped", "running"), PostconditionError);
}

TEST_F(PostconditionTest, StateTransition_WithGuard_Valid) {
    bool can_transition = true;
    StateTransition sm("StateMachine");
    sm.allow("idle", "running", [&]() { return can_transition; });

    EXPECT_NO_THROW(sm.validate("idle", "running"));
}

TEST_F(PostconditionTest, StateTransition_WithGuard_Invalid) {
    bool can_transition = false;
    StateTransition sm("StateMachine");
    sm.allow("idle", "running", [&]() { return can_transition; });

    EXPECT_THROW(sm.validate("idle", "running"), PostconditionError);
}

// SideEffectVerifier tests
TEST_F(PostconditionTest, SideEffectVerifier_AllEffectsOccur) {
    bool effect1_occurred = true;
    bool effect2_occurred = true;

    SideEffectVerifier verifier("test_func");
    verifier.expect("effect1", [&]() { return effect1_occurred; })
            .expect("effect2", [&]() { return effect2_occurred; });

    EXPECT_NO_THROW(verifier.verify());
}

TEST_F(PostconditionTest, SideEffectVerifier_MissingEffect) {
    bool effect1_occurred = true;
    bool effect2_occurred = false;

    SideEffectVerifier verifier("test_func");
    verifier.expect("effect1", [&]() { return effect1_occurred; })
            .expect("effect2", [&]() { return effect2_occurred; });

    EXPECT_THROW(verifier.verify(), PostconditionError);
}

TEST_F(PostconditionTest, SideEffectVerifier_NoExpectedEffects) {
    SideEffectVerifier verifier("test_func");
    EXPECT_NO_THROW(verifier.verify());
}

// ResourceGuarantee tests
TEST_F(PostconditionTest, ResourceGuarantee_NoThrow_Success) {
    ResourceGuarantee guarantee(ResourceGuarantee::Guarantee::NoThrow);

    auto operation = []() { return 42; };
    EXPECT_EQ(42, guarantee.enforce(operation));
}

TEST_F(PostconditionTest, ResourceGuarantee_NoThrow_Failure) {
    ResourceGuarantee guarantee(ResourceGuarantee::Guarantee::NoThrow);

    auto operation = []() -> int { throw std::runtime_error("test"); };
    EXPECT_THROW(guarantee.enforce(operation), PostconditionError);
}

TEST_F(PostconditionTest, ResourceGuarantee_Basic) {
    ResourceGuarantee guarantee(ResourceGuarantee::Guarantee::Basic);

    auto operation = []() { return 42; };
    EXPECT_EQ(42, guarantee.enforce(operation));
}

TEST_F(PostconditionTest, ResourceGuarantee_Strong) {
    ResourceGuarantee guarantee(ResourceGuarantee::Guarantee::Strong);

    auto operation = []() { return 42; };
    EXPECT_EQ(42, guarantee.enforce(operation));
}

TEST_F(PostconditionTest, ResourceGuarantee_NoChange) {
    ResourceGuarantee guarantee(ResourceGuarantee::Guarantee::NoChange);

    auto operation = []() { return 42; };
    EXPECT_EQ(42, guarantee.enforce(operation));
}

// NumericPostcondition tests
TEST_F(PostconditionTest, NumericPostcondition_EnsureNormalized_Valid) {
    EXPECT_NO_THROW(NumericPostcondition::ensure_normalized(1.0, "value"));
    EXPECT_NO_THROW(NumericPostcondition::ensure_normalized(1.000001, "value", 0.00001));
    EXPECT_NO_THROW(NumericPostcondition::ensure_normalized(0.999999, "value", 0.00001));
}

TEST_F(PostconditionTest, NumericPostcondition_EnsureNormalized_Invalid) {
    EXPECT_THROW(NumericPostcondition::ensure_normalized(0.5, "value"), PostconditionError);
    EXPECT_THROW(NumericPostcondition::ensure_normalized(2.0, "value"), PostconditionError);
    EXPECT_THROW(NumericPostcondition::ensure_normalized(1.1, "value", 0.01), PostconditionError);
}

class MockMatrix {
public:
    using value_type = double;
    MockMatrix(size_t r, size_t c, bool symmetric = true)
        : rows_(r), cols_(c), symmetric_(symmetric) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double operator()(size_t i, size_t j) const {
        if (symmetric_ || i == j) return 1.0;
        if (!symmetric_ && i == 0 && j == 1) return 2.0;  // Asymmetric element
        return 1.0;
    }

private:
    size_t rows_, cols_;
    bool symmetric_;
};

TEST_F(PostconditionTest, NumericPostcondition_EnsureSymmetric_Valid) {
    MockMatrix mat(3, 3, true);
    EXPECT_NO_THROW(NumericPostcondition::ensure_symmetric(mat, "matrix", 0.01));
}

TEST_F(PostconditionTest, NumericPostcondition_EnsureSymmetric_Invalid) {
    MockMatrix mat(3, 3, false);
    EXPECT_THROW(NumericPostcondition::ensure_symmetric(mat, "matrix", 0.01), PostconditionError);
}

TEST_F(PostconditionTest, NumericPostcondition_EnsureConserved_Valid) {
    double before = 100.0;
    double after = 100.0001;
    EXPECT_NO_THROW(NumericPostcondition::ensure_conserved(before, after, "energy", 0.001));
}

TEST_F(PostconditionTest, NumericPostcondition_EnsureConserved_Invalid) {
    double before = 100.0;
    double after = 105.0;
    EXPECT_THROW(NumericPostcondition::ensure_conserved(before, after, "energy", 0.001), PostconditionError);
}

// PostconditionScope tests
TEST_F(PostconditionTest, PostconditionScope_SuccessfulVerification) {
    bool was_checked = false;
    {
        PostconditionScope scope([&]() {
            was_checked = true;
        });
        // Normal execution
    }
    EXPECT_TRUE(was_checked);
}

TEST_F(PostconditionTest, PostconditionScope_ExceptionSkipsVerification) {
    bool was_checked = false;
    try {
        PostconditionScope scope([&]() {
            was_checked = true;
        });
        throw std::runtime_error("test exception");
    } catch (...) {
        // Exception caught
    }
    EXPECT_FALSE(was_checked);
}

TEST_F(PostconditionTest, PostconditionScope_VerificationFailureDoesNotThrow) {
    // Postcondition scope should not throw from destructor
    {
        PostconditionScope scope([]() {
            throw PostconditionError("<test>", "condition", "test failure");
        });
        // Should not throw even though verification fails
    }
    // If we get here, the test passes
    SUCCEED();
}

// Helper function tests
TEST_F(PostconditionTest, CaptureOld_Helper) {
    int old_value = 10;
    auto checker = capture_old(old_value);
    EXPECT_EQ(10, checker.old_value());
    EXPECT_NO_THROW(checker.ensure_changed(20, "value"));
}

TEST_F(PostconditionTest, ValidateReturn_Helper) {
    int return_value = 42;
    auto validator = validate_return(return_value, "test_func");
    validator.positive();
    EXPECT_NO_THROW(validator.validate());
}

TEST_F(PostconditionTest, PostconditionScope_Helper) {
    bool verified = false;
    {
        auto scope = postcondition_scope([&]() { verified = true; });
    }
    EXPECT_TRUE(verified);
}

// Integration tests
TEST_F(PostconditionTest, IntegrationExample_FunctionWithPostconditions) {
    // Simulate a function that increments a counter
    auto increment_counter = [](int& counter) -> int {
        auto old_checker = capture_old(counter);
        int old_value = counter;

        counter++;

        // Validate postconditions
        old_checker.ensure_increased(counter, "counter");

        auto return_validator = validate_return(counter, "increment_counter");
        return_validator.positive()
                        .satisfies([&](const int& v) { return v == old_value + 1; },
                                 "should be old_value + 1");
        return_validator.validate();

        return counter;
    };

    int counter = 0;
    EXPECT_EQ(1, increment_counter(counter));
    EXPECT_EQ(1, counter);
}

TEST_F(PostconditionTest, IntegrationExample_StateTransition) {
    // Simulate a simple state machine
    class SimpleMachine {
    public:
        SimpleMachine() : state_("idle") {
            transitions_.allow("idle", "running")
                       .allow("running", "paused")
                       .allow("paused", "running")
                       .allow("running", "stopped")
                       .allow("stopped", "idle");
        }

        void start() {
            std::string old_state = state_;
            state_ = "running";
            transitions_.validate(old_state, state_);
        }

        void pause() {
            std::string old_state = state_;
            state_ = "paused";
            transitions_.validate(old_state, state_);
        }

        void stop() {
            std::string old_state = state_;
            state_ = "stopped";
            transitions_.validate(old_state, state_);
        }

        const std::string& state() const { return state_; }

    private:
        std::string state_;
        StateTransition transitions_{"SimpleMachine"};
    };

    SimpleMachine machine;
    EXPECT_EQ("idle", machine.state());

    EXPECT_NO_THROW(machine.start());
    EXPECT_EQ("running", machine.state());

    EXPECT_NO_THROW(machine.pause());
    EXPECT_EQ("paused", machine.state());

    // Invalid transition - paused to stopped directly
    EXPECT_THROW(machine.stop(), PostconditionError);
}
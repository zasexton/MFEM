/**
 * @file test_assert.cpp
 * @brief Unit tests for assert.h functionality
 *
 * Tests assertion macros, handlers, and FEM-specific assertions
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <regex>
#include <cmath>
#include <vector>
#include <sstream>
#include <atomic>
#include <chrono>
#include <limits>

// Undefine conflicting macros from debug.h if they exist
#ifdef FEM_ASSERT
#undef FEM_ASSERT
#endif
#ifdef FEM_ASSERT_ALWAYS
#undef FEM_ASSERT_ALWAYS
#endif
#ifdef FEM_VERIFY
#undef FEM_VERIFY
#endif
#ifdef FEM_VERIFY_DEBUG
#undef FEM_VERIFY_DEBUG
#endif
#ifdef FEM_UNREACHABLE
#undef FEM_UNREACHABLE
#endif
#ifdef FEM_NOT_IMPLEMENTED
#undef FEM_NOT_IMPLEMENTED
#endif
#ifdef FEM_STATIC_ASSERT
#undef FEM_STATIC_ASSERT
#endif
#ifdef FEM_PRECONDITION
#undef FEM_PRECONDITION
#endif
#ifdef FEM_POSTCONDITION
#undef FEM_POSTCONDITION
#endif
#ifdef FEM_INVARIANT
#undef FEM_INVARIANT
#endif
#ifdef FEM_ASSERT_FINITE
#undef FEM_ASSERT_FINITE
#endif
#ifdef FEM_ASSERT_POSITIVE
#undef FEM_ASSERT_POSITIVE
#endif
#ifdef FEM_ASSERT_NON_NEGATIVE
#undef FEM_ASSERT_NON_NEGATIVE
#endif
#ifdef FEM_ASSERT_IN_RANGE
#undef FEM_ASSERT_IN_RANGE
#endif
#ifdef FEM_ASSERT_NOT_NULL
#undef FEM_ASSERT_NOT_NULL
#endif
#ifdef FEM_ASSERT_NOT_EMPTY
#undef FEM_ASSERT_NOT_EMPTY
#endif
#ifdef FEM_ASSERT_INDEX
#undef FEM_ASSERT_INDEX
#endif
#ifdef FEM_ASSERT_SQUARE_MATRIX
#undef FEM_ASSERT_SQUARE_MATRIX
#endif
#ifdef FEM_ASSERT_DIMENSIONS_MATCH
#undef FEM_ASSERT_DIMENSIONS_MATCH
#endif
#ifdef FEM_ASSERT_TYPE_IS
#undef FEM_ASSERT_TYPE_IS
#endif

// Now include assert.h with a clean slate
#include "logging/assert.h"

using namespace fem::core::logging;
using namespace testing;

/**
 * Test fixture for assertion tests
 */
class AssertTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset assert handler to default state for testing
        // In tests, we want to throw exceptions rather than abort
        AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);
    }

    void TearDown() override {
        // Reset to default
        AssertHandler::instance().set_action(
#ifdef NDEBUG
            AssertHandler::AssertAction::THROW
#else
            AssertHandler::AssertAction::ABORT
#endif
        );
    }
};

/**
 * Test AssertHandler basic functionality
 */
TEST_F(AssertTest, HandlerSingleton) {
    auto& handler1 = AssertHandler::instance();
    auto& handler2 = AssertHandler::instance();
    EXPECT_EQ(&handler1, &handler2);
}

TEST_F(AssertTest, SetGetAction) {
    auto& handler = AssertHandler::instance();

    handler.set_action(AssertHandler::AssertAction::THROW);
    EXPECT_EQ(handler.get_action(), AssertHandler::AssertAction::THROW);

    handler.set_action(AssertHandler::AssertAction::ABORT);
    EXPECT_EQ(handler.get_action(), AssertHandler::AssertAction::ABORT);

    handler.set_action(AssertHandler::AssertAction::LOG_ONLY);
    EXPECT_EQ(handler.get_action(), AssertHandler::AssertAction::LOG_ONLY);
}

/**
 * Test FEM_ASSERT_ALWAYS macro with throw action
 */
TEST_F(AssertTest, AssertAlwaysThrows) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    // Should not throw when condition is true
    EXPECT_NO_THROW(FEM_ASSERT_ALWAYS(true, "Should not throw"));

    // Should throw when condition is false
    EXPECT_THROW(
        FEM_ASSERT_ALWAYS(false, "Expected failure with value {}", 42),
        AssertHandler::assertion_error
    );

    // Verify exception contains expected message
    try {
        FEM_ASSERT_ALWAYS(false, "Expected failure with value {}", 42);
        FAIL() << "Should have thrown";
    } catch (const AssertHandler::assertion_error& e) {
        std::string msg = e.what();
        EXPECT_THAT(msg, HasSubstr("false"));
        EXPECT_THAT(msg, HasSubstr("Expected failure with value 42"));
    }
}

/**
 * Test custom assertion handler
 */
TEST_F(AssertTest, CustomHandler) {
    bool handler_called = false;
    std::string captured_condition;
    std::string captured_message;

    AssertHandler::instance().set_custom_handler(
        [&](const std::string& condition, const std::string& message,
            const std::source_location&) {
            handler_called = true;
            captured_condition = condition;
            captured_message = message;
        }
    );

    // Custom handler should be called but still terminate
    // We can't test termination directly, so we test the action is set
    EXPECT_EQ(AssertHandler::instance().get_action(),
              AssertHandler::AssertAction::CUSTOM);
}

/**
 * Test FEM_VERIFY macro (non-fatal verification)
 */
TEST_F(AssertTest, VerifyMacro) {
    // Note: FEM_VERIFY returns false on failure, doesn't throw
    // This is different from FEM_ASSERT which throws/aborts

    // Should return true when condition is true
    bool result1 = FEM_VERIFY(2 + 2 == 4, "Math works");
    EXPECT_TRUE(result1);

    // Should return false when condition is false
    bool result2 = FEM_VERIFY(2 + 2 == 5, "Math broken");
    EXPECT_FALSE(result2);

    // Verify can be used in if statements
    if (!FEM_VERIFY(1 > 2, "Should fail")) {
        EXPECT_TRUE(true); // We expect to get here
    } else {
        FAIL() << "Verify should have returned false";
    }
}

/**
 * Test debug-only assertions
 */
TEST_F(AssertTest, DebugAssertions) {
#ifdef NDEBUG
    // In release mode, FEM_ASSERT should do nothing
    FEM_ASSERT(false, "Should not execute in release");
    FEM_VERIFY_DEBUG(false, "Should not execute in release");
    EXPECT_TRUE(true); // Test passed if we got here
#else
    // In debug mode, FEM_ASSERT should be active
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    EXPECT_NO_THROW(FEM_ASSERT(true, "OK in debug"));
    EXPECT_THROW(
        FEM_ASSERT(false, "Fails in debug"),
        AssertHandler::assertion_error
    );

    // FEM_VERIFY_DEBUG should work like FEM_VERIFY
    EXPECT_TRUE(FEM_VERIFY_DEBUG(true, "OK"));
    EXPECT_FALSE(FEM_VERIFY_DEBUG(false, "Fail"));
#endif
}

/**
 * Test numerical assertions
 */
TEST_F(AssertTest, NumericalAssertions) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    // Test FEM_ASSERT_FINITE
    double finite_val = 3.14;
    double inf_val = std::numeric_limits<double>::infinity();
    double nan_val = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(FEM_ASSERT_FINITE(finite_val, "finite_val"));
    EXPECT_THROW(
        FEM_ASSERT_FINITE(inf_val, "inf_val"),
        AssertHandler::assertion_error
    );
    EXPECT_THROW(
        FEM_ASSERT_FINITE(nan_val, "nan_val"),
        AssertHandler::assertion_error
    );

    // Test FEM_ASSERT_POSITIVE
    EXPECT_NO_THROW(FEM_ASSERT_POSITIVE(5, "positive"));
    EXPECT_THROW(
        FEM_ASSERT_POSITIVE(0, "zero"),
        AssertHandler::assertion_error
    );
    EXPECT_THROW(
        FEM_ASSERT_POSITIVE(-5, "negative"),
        AssertHandler::assertion_error
    );

    // Test FEM_ASSERT_NON_NEGATIVE
    EXPECT_NO_THROW(FEM_ASSERT_NON_NEGATIVE(5, "positive"));
    EXPECT_NO_THROW(FEM_ASSERT_NON_NEGATIVE(0, "zero"));
    EXPECT_THROW(
        FEM_ASSERT_NON_NEGATIVE(-5, "negative"),
        AssertHandler::assertion_error
    );

    // Test FEM_ASSERT_IN_RANGE
    EXPECT_NO_THROW(FEM_ASSERT_IN_RANGE(5, 0, 10, "value"));
    EXPECT_NO_THROW(FEM_ASSERT_IN_RANGE(0, 0, 10, "lower_bound"));
    EXPECT_NO_THROW(FEM_ASSERT_IN_RANGE(10, 0, 10, "upper_bound"));
    EXPECT_THROW(
        FEM_ASSERT_IN_RANGE(-1, 0, 10, "too_low"),
        AssertHandler::assertion_error
    );
    EXPECT_THROW(
        FEM_ASSERT_IN_RANGE(11, 0, 10, "too_high"),
        AssertHandler::assertion_error
    );
}

/**
 * Test pointer and container assertions
 */
TEST_F(AssertTest, PointerContainerAssertions) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    // Test FEM_ASSERT_NOT_NULL
    int* valid_ptr = new int(42);
    int* null_ptr = nullptr;

    EXPECT_NO_THROW(FEM_ASSERT_NOT_NULL(valid_ptr, "valid_ptr"));
    EXPECT_THROW(
        FEM_ASSERT_NOT_NULL(null_ptr, "null_ptr"),
        AssertHandler::assertion_error
    );

    delete valid_ptr;

    // Test FEM_ASSERT_NOT_EMPTY
    std::vector<int> non_empty = {1, 2, 3};
    std::vector<int> empty;

    EXPECT_NO_THROW(FEM_ASSERT_NOT_EMPTY(non_empty, "non_empty"));
    EXPECT_THROW(
        FEM_ASSERT_NOT_EMPTY(empty, "empty"),
        AssertHandler::assertion_error
    );

    // Test FEM_ASSERT_INDEX
    std::vector<int> vec = {1, 2, 3};
    EXPECT_NO_THROW(FEM_ASSERT_INDEX(0, vec.size(), "index"));
    EXPECT_NO_THROW(FEM_ASSERT_INDEX(2, vec.size(), "index"));
    EXPECT_THROW(
        FEM_ASSERT_INDEX(-1, vec.size(), "negative_index"),
        AssertHandler::assertion_error
    );
    EXPECT_THROW(
        FEM_ASSERT_INDEX(3, vec.size(), "out_of_bounds"),
        AssertHandler::assertion_error
    );
}

/**
 * Test matrix-related assertions
 */
TEST_F(AssertTest, MatrixAssertions) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    // Test FEM_ASSERT_SQUARE_MATRIX
    EXPECT_NO_THROW(FEM_ASSERT_SQUARE_MATRIX(3, 3, "square"));
    EXPECT_THROW(
        FEM_ASSERT_SQUARE_MATRIX(3, 4, "non_square"),
        AssertHandler::assertion_error
    );

    // Test FEM_ASSERT_DIMENSIONS_MATCH
    EXPECT_NO_THROW(FEM_ASSERT_DIMENSIONS_MATCH(5, 5, "vec1", "vec2"));
    EXPECT_THROW(
        FEM_ASSERT_DIMENSIONS_MATCH(3, 4, "vec1", "vec2"),
        AssertHandler::assertion_error
    );
}

/**
 * Test precondition, postcondition, and invariant macros
 */
TEST_F(AssertTest, ContractAssertions) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    // Test FEM_PRECONDITION
    auto divide = [](double a, double b) -> double {
        FEM_PRECONDITION(std::abs(b) > 1e-10, "Divisor must be non-zero");
        return a / b;
    };

    EXPECT_NO_THROW(divide(10, 2));
    EXPECT_THROW(divide(10, 0), AssertHandler::assertion_error);

    // Test FEM_POSTCONDITION
    auto compute_positive = [](int x) -> int {
        int result = std::abs(x);
        FEM_POSTCONDITION(result >= 0, "Result must be non-negative");
        return result;
    };

    EXPECT_NO_THROW(compute_positive(-5));
    EXPECT_NO_THROW(compute_positive(5));

    // Test FEM_INVARIANT
    class Counter {
    public:
        void increment() {
            count_++;
            check_invariant();
        }

        void set_count(int c) {
            count_ = c;
            check_invariant();
        }

        int get_count() const { return count_; }

    private:
        void check_invariant() {
            FEM_INVARIANT(count_ >= 0, "Count must be non-negative");
        }

        int count_ = 0;
    };

    Counter counter;
    EXPECT_NO_THROW(counter.increment());
    EXPECT_NO_THROW(counter.set_count(10));
    EXPECT_THROW(counter.set_count(-1), AssertHandler::assertion_error);
}

/**
 * Test FEM_UNREACHABLE macro
 */
TEST_F(AssertTest, UnreachableMacro) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    auto process_enum = [](int value) {
        switch (value) {
            case 1: return "one";
            case 2: return "two";
            default:
                FEM_UNREACHABLE("Invalid enum value: {}", value);
        }
    };

    EXPECT_EQ(process_enum(1), "one");
    EXPECT_EQ(process_enum(2), "two");
    EXPECT_THROW(process_enum(3), AssertHandler::assertion_error);
}

/**
 * Test FEM_NOT_IMPLEMENTED macro
 */
TEST_F(AssertTest, NotImplementedMacro) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    auto future_feature = []() {
        FEM_NOT_IMPLEMENTED("Advanced feature coming in v2.0");
    };

    EXPECT_THROW(future_feature(), AssertHandler::assertion_error);

    // Verify exception message
    try {
        future_feature();
    } catch (const AssertHandler::assertion_error& e) {
        std::string msg = e.what();
        EXPECT_THAT(msg, HasSubstr("Not implemented: Advanced feature"));
    }
}

/**
 * Test assertion with complex formatting
 */
TEST_F(AssertTest, ComplexFormatting) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    struct Point {
        double x, y;
    };

    Point p = {3.14, 2.71};

    try {
        FEM_ASSERT_ALWAYS(false, "Point ({:.2f}, {:.2f}) is invalid at iteration {}",
                          p.x, p.y, 42);
        FAIL() << "Should have thrown";
    } catch (const AssertHandler::assertion_error& e) {
        std::string msg = e.what();
        EXPECT_THAT(msg, HasSubstr("Point (3.14, 2.71) is invalid at iteration 42"));
    }
}

/**
 * Test thread safety of assertions
 */
TEST_F(AssertTest, ThreadSafety) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    const int num_threads = 10;
    const int iterations = 100;
    std::atomic<int> failure_count{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < iterations; ++i) {
                try {
                    // Mix of passing and failing assertions
                    if (i % 2 == 0) {
                        FEM_ASSERT_ALWAYS(true, "Thread {} iteration {}", t, i);
                    } else {
                        FEM_ASSERT_ALWAYS(false, "Thread {} iteration {}", t, i);
                    }
                } catch (const AssertHandler::assertion_error&) {
                    failure_count++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should have exactly num_threads * iterations/2 failures
    EXPECT_EQ(failure_count.load(), num_threads * iterations / 2);
}

/**
 * Test assertion message in exception
 */
TEST_F(AssertTest, ExceptionMessage) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    try {
        FEM_ASSERT_ALWAYS(1 + 1 == 3, "Math is broken!");
    } catch (const AssertHandler::assertion_error& e) {
        std::string msg = e.what();
        EXPECT_THAT(msg, HasSubstr("1 + 1 == 3"));
        EXPECT_THAT(msg, HasSubstr("Math is broken"));
    }
}

/**
 * Test LOG_ONLY action (doesn't terminate)
 */
TEST_F(AssertTest, LogOnlyAction) {
    // Note: LOG_ONLY violates [[noreturn]] but is for special cases
    // It will call std::terminate(), so we can't fully test it
    // We can only verify the action is set correctly

    AssertHandler::instance().set_action(AssertHandler::AssertAction::LOG_ONLY);
    EXPECT_EQ(AssertHandler::instance().get_action(),
              AssertHandler::AssertAction::LOG_ONLY);
}

/**
 * Test static assertions (compile-time)
 */
TEST_F(AssertTest, StaticAssertions) {
    // These are compile-time checks, so we just verify they compile
    FEM_STATIC_ASSERT(sizeof(int) >= 4, "int must be at least 32 bits");
    FEM_ASSERT_TYPE_IS(int, std::is_integral, "int must be integral");

    // If these compile, the test passes
    EXPECT_TRUE(true);
}

/**
 * Test multiple assertions in sequence
 */
TEST_F(AssertTest, MultipleAssertions) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    auto validate_data = [](const std::vector<double>& data) {
        FEM_ASSERT_NOT_EMPTY(data, "data");

        for (size_t i = 0; i < data.size(); ++i) {
            FEM_ASSERT_FINITE(data[i], "data[" + std::to_string(i) + "]");
            FEM_ASSERT_IN_RANGE(data[i], -100, 100, "data[" + std::to_string(i) + "]");
        }

        return true;
    };

    // Valid data should pass
    std::vector<double> good_data = {1.0, 2.0, 3.0};
    EXPECT_TRUE(validate_data(good_data));

    // Empty data should fail
    std::vector<double> empty_data;
    EXPECT_THROW(validate_data(empty_data), AssertHandler::assertion_error);

    // Data with NaN should fail
    std::vector<double> nan_data = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(validate_data(nan_data), AssertHandler::assertion_error);

    // Data out of range should fail
    std::vector<double> oor_data = {1.0, 200.0, 3.0};
    EXPECT_THROW(validate_data(oor_data), AssertHandler::assertion_error);
}

/**
 * Test assertion with lambda in condition
 */
TEST_F(AssertTest, LambdaCondition) {
    AssertHandler::instance().set_action(AssertHandler::AssertAction::THROW);

    auto is_valid = [](int x) { return x > 0 && x < 100; };

    int value1 = 50;
    int value2 = 150;

    EXPECT_NO_THROW(FEM_ASSERT_ALWAYS(is_valid(value1), "Value {} is invalid", value1));
    EXPECT_THROW(
        FEM_ASSERT_ALWAYS(is_valid(value2), "Value {} is invalid", value2),
        AssertHandler::assertion_error
    );
}

/**
 * Test performance overhead (disabled assertions)
 */
TEST_F(AssertTest, PerformanceOverhead) {
    // Test that disabled assertions have minimal overhead
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        // These should be no-ops in release mode
#ifdef NDEBUG
        FEM_ASSERT(i >= 0, "Should not execute");
        FEM_VERIFY_DEBUG(i < 1000001, "Should not execute");
#endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // In release mode, should be very fast (< 10ms for 1M iterations)
#ifdef NDEBUG
    EXPECT_LT(duration.count(), 10);
#endif

    // Test passes if we complete
    EXPECT_TRUE(true);
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
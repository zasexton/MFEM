#include <gtest/gtest.h>
#include <core/error/error_guard.h>
#include <core/error/logic_error.h>
#include <core/error/runtime_error.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>

using namespace fem::core::error;

class ErrorGuardTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ScopeGuard tests
TEST_F(ErrorGuardTest, ScopeGuardBasic) {
    bool cleanup_called = false;

    {
        ScopeGuard guard([&]() {
            cleanup_called = true;
        });
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorGuardTest, ScopeGuardDismiss) {
    bool cleanup_called = false;

    {
        ScopeGuard guard([&]() {
            cleanup_called = true;
        });
        guard.dismiss();
    }

    EXPECT_FALSE(cleanup_called);
}

TEST_F(ErrorGuardTest, ScopeGuardExecute) {
    bool cleanup_called = false;

    {
        ScopeGuard guard([&]() {
            cleanup_called = true;
        });
        guard.execute();
        EXPECT_TRUE(cleanup_called);

        cleanup_called = false;
    }

    // Should not be called again in destructor
    EXPECT_FALSE(cleanup_called);
}

TEST_F(ErrorGuardTest, ScopeGuardMove) {
    bool cleanup_called = false;

    {
        ScopeGuard guard1([&]() {
            cleanup_called = true;
        });

        ScopeGuard guard2 = std::move(guard1);
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorGuardTest, ScopeGuardExceptionSafety) {
    bool cleanup_called = false;

    try {
        ScopeGuard guard([&]() {
            cleanup_called = true;
        });
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }

    EXPECT_TRUE(cleanup_called);
}

// ExceptionGuard tests
TEST_F(ErrorGuardTest, ExceptionGuardNormalExit) {
    bool rollback_called = false;

    {
        ExceptionGuard guard([&]() {
            rollback_called = true;
        });
        // Normal execution, no exception
    }

    EXPECT_FALSE(rollback_called);
}

TEST_F(ErrorGuardTest, ExceptionGuardOnException) {
    bool rollback_called = false;

    try {
        ExceptionGuard guard([&]() {
            rollback_called = true;
        });
        throw std::runtime_error("test exception");
    } catch (...) {
        // Expected
    }

    EXPECT_TRUE(rollback_called);
}

TEST_F(ErrorGuardTest, ExceptionGuardDismiss) {
    bool rollback_called = false;

    try {
        ExceptionGuard guard([&]() {
            rollback_called = true;
        });
        guard.dismiss();
        throw std::runtime_error("test exception");
    } catch (...) {
        // Expected
    }

    EXPECT_FALSE(rollback_called);
}

// SuccessGuard tests
TEST_F(ErrorGuardTest, SuccessGuardOnSuccess) {
    bool commit_called = false;

    {
        SuccessGuard guard([&]() {
            commit_called = true;
        });
        // Normal execution, no exception
    }

    EXPECT_TRUE(commit_called);
}

TEST_F(ErrorGuardTest, SuccessGuardOnException) {
    bool commit_called = false;

    try {
        SuccessGuard guard([&]() {
            commit_called = true;
        });
        throw std::runtime_error("test exception");
    } catch (...) {
        // Expected
    }

    EXPECT_FALSE(commit_called);
}

TEST_F(ErrorGuardTest, SuccessGuardExecute) {
    bool commit_called = false;

    {
        SuccessGuard guard([&]() {
            commit_called = true;
        });
        guard.execute();
        EXPECT_TRUE(commit_called);

        commit_called = false;
    }

    // Should not be called again in destructor
    EXPECT_FALSE(commit_called);
}

TEST_F(ErrorGuardTest, SuccessGuardDismiss) {
    bool commit_called = false;

    {
        SuccessGuard guard([&]() {
            commit_called = true;
        });
        guard.dismiss();
    }

    EXPECT_FALSE(commit_called);
}

// TransactionGuard tests
TEST_F(ErrorGuardTest, TransactionGuardAutoRollback) {
    bool commit_called = false;
    bool rollback_called = false;

    {
        TransactionGuard guard(
            [&]() { commit_called = true; },
            [&]() { rollback_called = true; }
        );
        // Don't commit explicitly
    }

    EXPECT_FALSE(commit_called);
    EXPECT_TRUE(rollback_called);
}

TEST_F(ErrorGuardTest, TransactionGuardExplicitCommit) {
    bool commit_called = false;
    bool rollback_called = false;

    {
        TransactionGuard guard(
            [&]() { commit_called = true; },
            [&]() { rollback_called = true; }
        );
        guard.commit();
        EXPECT_TRUE(guard.is_committed());
    }

    EXPECT_TRUE(commit_called);
    EXPECT_FALSE(rollback_called);
}

TEST_F(ErrorGuardTest, TransactionGuardExplicitRollback) {
    bool commit_called = false;
    bool rollback_called = false;

    {
        TransactionGuard guard(
            [&]() { commit_called = true; },
            [&]() { rollback_called = true; }
        );
        guard.rollback();
        EXPECT_TRUE(guard.is_committed());
    }

    EXPECT_FALSE(commit_called);
    EXPECT_TRUE(rollback_called);
}

TEST_F(ErrorGuardTest, TransactionGuardDoubleCommit) {
    int commit_count = 0;
    int rollback_count = 0;

    {
        TransactionGuard guard(
            [&]() { commit_count++; },
            [&]() { rollback_count++; }
        );
        guard.commit();
        guard.commit();  // Should be ignored
    }

    EXPECT_EQ(commit_count, 1);
    EXPECT_EQ(rollback_count, 0);
}

// ValueGuard tests
TEST_F(ErrorGuardTest, ValueGuardBasic) {
    int value = 42;
    int original = value;

    {
        ValueGuard<int> guard(value);
        value = 100;
    }

    EXPECT_EQ(value, original);
}

TEST_F(ErrorGuardTest, ValueGuardWithInitialValue) {
    int value = 42;
    int original = value;

    {
        ValueGuard<int> guard(value, 100);
        EXPECT_EQ(value, 100);
    }

    EXPECT_EQ(value, original);
}

TEST_F(ErrorGuardTest, ValueGuardDismiss) {
    int value = 42;

    {
        ValueGuard<int> guard(value, 100);
        EXPECT_EQ(value, 100);
        guard.dismiss();
    }

    EXPECT_EQ(value, 100);  // Should not be restored
}

TEST_F(ErrorGuardTest, ValueGuardRestore) {
    int value = 42;
    int original = value;

    {
        ValueGuard<int> guard(value, 100);
        EXPECT_EQ(value, 100);
        guard.restore();
        EXPECT_EQ(value, original);
    }

    EXPECT_EQ(value, original);
}

TEST_F(ErrorGuardTest, ValueGuardOldValue) {
    int value = 42;

    ValueGuard<int> guard(value, 100);
    EXPECT_EQ(guard.old_value(), 42);
}

// ErrorAccumulator tests
TEST_F(ErrorGuardTest, ErrorAccumulatorBasic) {
    ErrorAccumulator accumulator;

    EXPECT_FALSE(accumulator.has_errors());
    EXPECT_EQ(accumulator.error_count(), 0);

    accumulator.try_operation("operation1", []() {
        throw std::runtime_error("error1");
    });

    EXPECT_TRUE(accumulator.has_errors());
    EXPECT_EQ(accumulator.error_count(), 1);
}

TEST_F(ErrorGuardTest, ErrorAccumulatorMultipleErrors) {
    ErrorAccumulator accumulator;

    accumulator.try_operation("op1", []() {
        throw std::runtime_error("error1");
    });

    accumulator.try_operation("op2", []() {
        throw std::logic_error("error2");
    });

    accumulator.try_operation("op3", []() {
        // Success, no exception
    });

    EXPECT_TRUE(accumulator.has_errors());
    EXPECT_EQ(accumulator.error_count(), 2);

    auto& errors = accumulator.errors();
    EXPECT_EQ(errors.size(), 2);
    EXPECT_EQ(errors[0].first, "op1");
    EXPECT_EQ(errors[1].first, "op2");
}

TEST_F(ErrorGuardTest, ErrorAccumulatorWithResult) {
    ErrorAccumulator accumulator;

    auto result1 = accumulator.try_with_result("successful_op", []() {
        return 42;
    });

    auto result2 = accumulator.try_with_result("failing_op", []() -> int {
        throw std::runtime_error("failed");
    });

    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), 42);

    EXPECT_FALSE(result2.has_value());

    EXPECT_TRUE(accumulator.has_errors());
    EXPECT_EQ(accumulator.error_count(), 1);
}

TEST_F(ErrorGuardTest, ErrorAccumulatorThrowIfErrors) {
    ErrorAccumulator accumulator;

    accumulator.try_operation("op1", []() {
        throw std::runtime_error("error1");
    });

    EXPECT_THROW(accumulator.throw_if_errors(), AggregateException);

    // Test no throw when no errors
    ErrorAccumulator clean_accumulator;
    EXPECT_NO_THROW(clean_accumulator.throw_if_errors());
}

TEST_F(ErrorGuardTest, ErrorAccumulatorSummary) {
    ErrorAccumulator accumulator;

    accumulator.try_operation("op1", []() {
        throw std::runtime_error("error1");
    });

    accumulator.try_operation("op2", []() {
        throw std::logic_error("error2");
    });

    std::string summary = accumulator.error_summary();
    EXPECT_TRUE(summary.find("Errors (2)") != std::string::npos);
    EXPECT_TRUE(summary.find("op1") != std::string::npos);
    EXPECT_TRUE(summary.find("op2") != std::string::npos);
    EXPECT_TRUE(summary.find("error1") != std::string::npos);
    EXPECT_TRUE(summary.find("error2") != std::string::npos);
}

TEST_F(ErrorGuardTest, ErrorAccumulatorClear) {
    ErrorAccumulator accumulator;

    accumulator.try_operation("op1", []() {
        throw std::runtime_error("error1");
    });

    EXPECT_TRUE(accumulator.has_errors());

    accumulator.clear();

    EXPECT_FALSE(accumulator.has_errors());
    EXPECT_EQ(accumulator.error_count(), 0);
}

// RetryGuard tests
TEST_F(ErrorGuardTest, RetryGuardSuccess) {
    RetryGuard retry;

    int attempt_count = 0;
    auto result = retry.execute([&]() {
        attempt_count++;
        return 42;
    });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(attempt_count, 1);
}

TEST_F(ErrorGuardTest, RetryGuardEventualSuccess) {
    RetryGuard::Config config;
    config.max_attempts = 3;
    config.initial_delay = std::chrono::milliseconds(1);

    RetryGuard retry(config);

    int attempt_count = 0;
    auto result = retry.execute([&]() {
        attempt_count++;
        if (attempt_count < 3) {
            throw std::runtime_error("temporary failure");
        }
        return 42;
    });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(attempt_count, 3);
}

TEST_F(ErrorGuardTest, RetryGuardAllAttemptsFail) {
    RetryGuard::Config config;
    config.max_attempts = 3;
    config.initial_delay = std::chrono::milliseconds(1);

    RetryGuard retry(config);

    int attempt_count = 0;
    EXPECT_THROW({
        retry.execute([&]() {
            attempt_count++;
            throw std::runtime_error("persistent failure");
        });
    }, std::runtime_error);

    EXPECT_EQ(attempt_count, 3);
}

TEST_F(ErrorGuardTest, RetryGuardConditionalRetry) {
    RetryGuard::Config config;
    config.max_attempts = 5;
    config.initial_delay = std::chrono::milliseconds(1);

    RetryGuard retry(config);

    int attempt_count = 0;
    EXPECT_THROW({
        retry.execute_if(
            [&]() {
                attempt_count++;
                if (attempt_count <= 2) {
                    throw std::runtime_error("retryable");
                } else {
                    throw std::logic_error("non-retryable");
                }
            },
            [](const std::exception* e, size_t /* attempt */) {
                if (!e) return false;
                // Only retry runtime_error, not logic_error
                return dynamic_cast<const std::runtime_error*>(e) != nullptr;
            }
        );
    }, std::logic_error);

    EXPECT_EQ(attempt_count, 3);  // 2 retryable + 1 non-retryable
}

// StateSnapshot tests
TEST_F(ErrorGuardTest, StateSnapshotBasic) {
    struct State {
        int value;
        std::string name;
    };

    State original{42, "original"};
    State current = original;

    StateSnapshot<State> snapshot(current);

    current.value = 100;
    current.name = "modified";

    snapshot.restore(current);

    EXPECT_EQ(current.value, original.value);
    EXPECT_EQ(current.name, original.name);
}

TEST_F(ErrorGuardTest, StateSnapshotCapture) {
    struct State {
        int value;
    };

    State state{42};
    StateSnapshot<State> snapshot(state);

    state.value = 100;
    snapshot.capture(state);

    state.value = 200;
    snapshot.restore(state);

    EXPECT_EQ(state.value, 100);
}

TEST_F(ErrorGuardTest, StateSnapshotGet) {
    struct State {
        int value;
        std::string name;
    };

    State original{42, "test"};
    StateSnapshot<State> snapshot(original);

    const auto& retrieved = snapshot.get();
    EXPECT_EQ(retrieved.value, 42);
    EXPECT_EQ(retrieved.name, "test");
}

TEST_F(ErrorGuardTest, StateSnapshotHasSnapshot) {
    StateSnapshot<int> empty_snapshot(42);
    EXPECT_TRUE(empty_snapshot.has_snapshot());
}

// Helper function tests
TEST_F(ErrorGuardTest, HelperFunctions) {
    bool cleanup_called = false;

    {
        auto guard = make_scope_guard([&]() {
            cleanup_called = true;
        });
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorGuardTest, ValueGuardHelper) {
    int value = 42;
    int original = value;

    {
        auto guard = make_value_guard(value);
        value = 100;
    }

    EXPECT_EQ(value, original);
}

TEST_F(ErrorGuardTest, TransactionHelper) {
    bool commit_called = false;
    bool rollback_called = false;

    {
        auto transaction = make_transaction(
            [&]() { commit_called = true; },
            [&]() { rollback_called = true; }
        );
        transaction.commit();
    }

    EXPECT_TRUE(commit_called);
    EXPECT_FALSE(rollback_called);
}

// Macro tests
TEST_F(ErrorGuardTest, MacroScopeExit) {
    bool cleanup_called = false;

    {
        FEM_SCOPE_EXIT(cleanup_called = true);
    }

    EXPECT_TRUE(cleanup_called);
}

TEST_F(ErrorGuardTest, MacroOnException) {
    bool rollback_called = false;

    try {
        FEM_ON_EXCEPTION(rollback_called = true);
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }

    EXPECT_TRUE(rollback_called);
}

TEST_F(ErrorGuardTest, MacroOnSuccess) {
    bool commit_called = false;

    {
        FEM_ON_SUCCESS(commit_called = true);
    }

    EXPECT_TRUE(commit_called);
}

TEST_F(ErrorGuardTest, MacroValueGuard) {
    int value = 42;
    int original = value;

    {
        FEM_VALUE_GUARD(value);
        value = 100;
    }

    EXPECT_EQ(value, original);
}

// Complex scenarios
TEST_F(ErrorGuardTest, NestedGuards) {
    std::vector<std::string> execution_order;

    try {
        ScopeGuard outer([&]() {
            execution_order.push_back("outer_scope");
        });

        ExceptionGuard exception([&]() {
            execution_order.push_back("exception_rollback");
        });

        SuccessGuard success([&]() {
            execution_order.push_back("success_commit");
        });

        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }

    EXPECT_EQ(execution_order.size(), 2);
    EXPECT_EQ(execution_order[0], "exception_rollback");
    EXPECT_EQ(execution_order[1], "outer_scope");
}

TEST_F(ErrorGuardTest, GuardExceptionInCleanup) {
    // Test that exceptions in guard cleanup don't propagate
    // This should not throw because ScopeGuard suppresses exceptions in destructor
    EXPECT_NO_THROW({
        ScopeGuard guard([]() {
            throw std::runtime_error("cleanup error");
        });
    });

    // Test that ExceptionGuard's cleanup exception doesn't cause termination
    // when another exception is already active
    bool original_exception_caught = false;
    try {
        ExceptionGuard guard([]() {
            throw std::runtime_error("rollback error");  // This will be suppressed
        });
        throw std::logic_error("original error");  // This will propagate
    } catch (const std::logic_error& e) {
        original_exception_caught = true;
        EXPECT_STREQ(e.what(), "original error");
    } catch (...) {
        FAIL() << "Unexpected exception type caught";
    }
    EXPECT_TRUE(original_exception_caught);
}
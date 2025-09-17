#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <future>
#include <barrier>

#include "logging/logcontext.h"
#include "logging/logmessage.h"
#include "logging/loglevel.h"

using namespace fem::core::logging;
using namespace std::chrono_literals;

namespace {

// Helper function to create a test log message (unused in current tests)
// LogMessage create_test_message(LogLevel level = LogLevel::INFO,
//                               const std::string& logger = "test",
//                               const std::string& message = "test message") {
//     return LogMessage(level, logger, message);
// }

// Helper to get fresh LogContext instance for isolated tests
class LogContextFixture {
public:
    LogContextFixture() {
        // Clear all context before each test
        auto& ctx = LogContext::get_instance();
        ctx.clear_global_context();
        ctx.clear_thread_context();
    }

    ~LogContextFixture() {
        // Cleanup after each test
        auto& ctx = LogContext::get_instance();
        ctx.clear_global_context();
        ctx.clear_thread_context();
    }
};

} // anonymous namespace

// =============================================================================
// LogContext Basic Functionality Tests
// =============================================================================

class LogContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(LogContextTest, SingletonInstance) {
    auto& ctx1 = LogContext::get_instance();
    auto& ctx2 = LogContext::get_instance();

    EXPECT_EQ(&ctx1, &ctx2);
}

TEST_F(LogContextTest, GlobalContextOperations) {
    auto& ctx = LogContext::get_instance();

    // Test setting and getting string values
    ctx.set_global_context("app.name", "test_app");
    ctx.set_global_context("app.version", "1.0.0");

    auto name = ctx.get_context<std::string>("app.name");
    auto version = ctx.get_context<std::string>("app.version");

    ASSERT_TRUE(name.has_value());
    ASSERT_TRUE(version.has_value());
    EXPECT_EQ(name.value(), "test_app");
    EXPECT_EQ(version.value(), "1.0.0");

    // Test setting numeric values
    ctx.set_global_context("port", 8080);
    ctx.set_global_context("timeout", 30.5);

    auto port = ctx.get_context<int>("port");
    auto timeout = ctx.get_context<double>("timeout");

    ASSERT_TRUE(port.has_value());
    ASSERT_TRUE(timeout.has_value());
    EXPECT_EQ(port.value(), 8080);
    EXPECT_DOUBLE_EQ(timeout.value(), 30.5);
}

TEST_F(LogContextTest, ThreadLocalContextOperations) {
    auto& ctx = LogContext::get_instance();

    ctx.set_thread_context("request.id", "req_123");
    ctx.set_thread_context("user.id", 42);

    auto request_id = ctx.get_context<std::string>("request.id");
    auto user_id = ctx.get_context<int>("user.id");

    ASSERT_TRUE(request_id.has_value());
    ASSERT_TRUE(user_id.has_value());
    EXPECT_EQ(request_id.value(), "req_123");
    EXPECT_EQ(user_id.value(), 42);
}

TEST_F(LogContextTest, ContextHierarchyResolution) {
    auto& ctx = LogContext::get_instance();

    // Set up hierarchy: Global < Thread < Scope
    ctx.set_global_context("priority", "global");
    ctx.set_thread_context("priority", "thread");

    // Global should be overridden by thread
    auto priority = ctx.get_context<std::string>("priority");
    ASSERT_TRUE(priority.has_value());
    EXPECT_EQ(priority.value(), "thread");

    // Test scope override
    {
        auto scope = ctx.create_scope("test_scope");
        scope.set_context("priority", "scope");

        auto scope_priority = ctx.get_context<std::string>("priority");
        ASSERT_TRUE(scope_priority.has_value());
        EXPECT_EQ(scope_priority.value(), "scope");
    }

    // Should revert to thread level after scope ends
    auto after_scope = ctx.get_context<std::string>("priority");
    ASSERT_TRUE(after_scope.has_value());
    EXPECT_EQ(after_scope.value(), "thread");
}

TEST_F(LogContextTest, NonExistentContextReturnsNullopt) {
    auto& ctx = LogContext::get_instance();

    auto missing = ctx.get_context<std::string>("non.existent");
    EXPECT_FALSE(missing.has_value());

    auto missing_int = ctx.get_context<int>("also.missing");
    EXPECT_FALSE(missing_int.has_value());
}

TEST_F(LogContextTest, TypeConversionHandling) {
    auto& ctx = LogContext::get_instance();

    // Set as string, retrieve as different types
    ctx.set_global_context("number", "42");
    ctx.set_global_context("decimal", "3.14159");
    ctx.set_global_context("boolean_true", "true");
    ctx.set_global_context("boolean_false", "false");
    ctx.set_global_context("boolean_numeric", "1");

    auto number = ctx.get_context<int>("number");
    auto decimal = ctx.get_context<double>("decimal");
    auto bool_true = ctx.get_context<bool>("boolean_true");
    auto bool_false = ctx.get_context<bool>("boolean_false");
    auto bool_numeric = ctx.get_context<bool>("boolean_numeric");

    ASSERT_TRUE(number.has_value());
    EXPECT_EQ(number.value(), 42);

    ASSERT_TRUE(decimal.has_value());
    EXPECT_DOUBLE_EQ(decimal.value(), 3.14159);

    ASSERT_TRUE(bool_true.has_value());
    EXPECT_TRUE(bool_true.value());

    ASSERT_TRUE(bool_false.has_value());
    EXPECT_FALSE(bool_false.value());

    ASSERT_TRUE(bool_numeric.has_value());
    EXPECT_TRUE(bool_numeric.value());
}

TEST_F(LogContextTest, InvalidTypeConversionReturnsNullopt) {
    auto& ctx = LogContext::get_instance();

    ctx.set_global_context("invalid_number", "not_a_number");

    auto invalid = ctx.get_context<int>("invalid_number");
    EXPECT_FALSE(invalid.has_value());

    auto invalid_double = ctx.get_context<double>("invalid_number");
    EXPECT_FALSE(invalid_double.has_value());
}

// =============================================================================
// ContextScope RAII Tests
// =============================================================================

class ContextScopeTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(ContextScopeTest, BasicScopeLifecycle) {
    auto& ctx = LogContext::get_instance();

    ctx.set_thread_context("base", "thread_level");

    EXPECT_EQ(ctx.get_scope_depth(), 0);

    {
        auto scope = ctx.create_scope("test_scope");
        EXPECT_EQ(ctx.get_scope_depth(), 1);
        EXPECT_EQ(scope.get_scope_name(), "test_scope");
        EXPECT_TRUE(scope.is_valid());

        scope.set_context("scoped", "value");
        auto scoped_value = ctx.get_context<std::string>("scoped");
        ASSERT_TRUE(scoped_value.has_value());
        EXPECT_EQ(scoped_value.value(), "value");

        // Should still have access to thread-level context
        auto base_value = ctx.get_context<std::string>("base");
        ASSERT_TRUE(base_value.has_value());
        EXPECT_EQ(base_value.value(), "thread_level");
    }

    // Scope should be cleaned up
    EXPECT_EQ(ctx.get_scope_depth(), 0);

    // Scoped context should be gone
    auto scoped_after = ctx.get_context<std::string>("scoped");
    EXPECT_FALSE(scoped_after.has_value());

    // Thread context should remain
    auto base_after = ctx.get_context<std::string>("base");
    ASSERT_TRUE(base_after.has_value());
    EXPECT_EQ(base_after.value(), "thread_level");
}

TEST_F(ContextScopeTest, NestedScopes) {
    auto& ctx = LogContext::get_instance();

    ctx.set_thread_context("level", "thread");

    {
        auto scope1 = ctx.create_scope("scope1");
        scope1.set_context("level", "scope1");
        scope1.set_context("scope1_only", "value1");

        EXPECT_EQ(ctx.get_scope_depth(), 1);

        auto level1 = ctx.get_context<std::string>("level");
        ASSERT_TRUE(level1.has_value());
        EXPECT_EQ(level1.value(), "scope1");

        {
            auto scope2 = ctx.create_scope("scope2");
            scope2.set_context("level", "scope2");
            scope2.set_context("scope2_only", "value2");

            EXPECT_EQ(ctx.get_scope_depth(), 2);

            auto level2 = ctx.get_context<std::string>("level");
            auto scope1_val = ctx.get_context<std::string>("scope1_only");
            auto scope2_val = ctx.get_context<std::string>("scope2_only");

            ASSERT_TRUE(level2.has_value());
            EXPECT_EQ(level2.value(), "scope2");

            ASSERT_TRUE(scope1_val.has_value());
            EXPECT_EQ(scope1_val.value(), "value1");

            ASSERT_TRUE(scope2_val.has_value());
            EXPECT_EQ(scope2_val.value(), "value2");
        }

        // Back to scope1 level
        EXPECT_EQ(ctx.get_scope_depth(), 1);

        auto level_back = ctx.get_context<std::string>("level");
        ASSERT_TRUE(level_back.has_value());
        EXPECT_EQ(level_back.value(), "scope1");

        // scope2_only should be gone
        auto scope2_gone = ctx.get_context<std::string>("scope2_only");
        EXPECT_FALSE(scope2_gone.has_value());
    }

    // Back to thread level
    EXPECT_EQ(ctx.get_scope_depth(), 0);

    auto level_final = ctx.get_context<std::string>("level");
    ASSERT_TRUE(level_final.has_value());
    EXPECT_EQ(level_final.value(), "thread");
}

TEST_F(ContextScopeTest, IsolatedScope) {
    auto& ctx = LogContext::get_instance();

    ctx.set_global_context("global", "global_value");
    ctx.set_thread_context("thread", "thread_value");

    {
        auto isolated = ctx.create_scope("isolated", LogContext::ScopeType::ISOLATED);

        // Should not inherit from parent contexts
        auto global_val = ctx.get_context<std::string>("global");
        auto thread_val = ctx.get_context<std::string>("thread");

        // Global should still be accessible (lowest priority)
        ASSERT_TRUE(global_val.has_value());
        EXPECT_EQ(global_val.value(), "global_value");

        // Thread should still be accessible
        ASSERT_TRUE(thread_val.has_value());
        EXPECT_EQ(thread_val.value(), "thread_value");

        // But scope starts empty
        isolated.set_context("isolated_only", "isolated_value");
        auto isolated_val = ctx.get_context<std::string>("isolated_only");
        ASSERT_TRUE(isolated_val.has_value());
        EXPECT_EQ(isolated_val.value(), "isolated_value");
    }

    // Isolated context should be gone
    auto isolated_after = ctx.get_context<std::string>("isolated_only");
    EXPECT_FALSE(isolated_after.has_value());
}

TEST_F(ContextScopeTest, ReadOnlyScope) {
    auto& ctx = LogContext::get_instance();

    ctx.set_thread_context("readonly_test", "original");

    {
        auto readonly = ctx.create_scope("readonly", LogContext::ScopeType::READONLY);

        // Should be able to read existing values
        auto original = ctx.get_context<std::string>("readonly_test");
        ASSERT_TRUE(original.has_value());
        EXPECT_EQ(original.value(), "original");

        // Attempting to set should be ignored (not throw)
        readonly.set_context("readonly_test", "modified");
        readonly.set_context("new_key", "new_value");

        // Values should be unchanged
        auto after_attempt = ctx.get_context<std::string>("readonly_test");
        ASSERT_TRUE(after_attempt.has_value());
        EXPECT_EQ(after_attempt.value(), "original");

        auto new_key = ctx.get_context<std::string>("new_key");
        EXPECT_FALSE(new_key.has_value());
    }
}

TEST_F(ContextScopeTest, ScopeWithInitialContext) {
    auto& ctx = LogContext::get_instance();

    LogContext::ContextMap initial = {
        {"init1", "value1"},
        {"init2", "value2"}
    };

    {
        auto scope = ctx.create_scope("initialized", LogContext::ScopeType::INHERIT, initial);

        auto val1 = ctx.get_context<std::string>("init1");
        auto val2 = ctx.get_context<std::string>("init2");

        ASSERT_TRUE(val1.has_value());
        EXPECT_EQ(val1.value(), "value1");

        ASSERT_TRUE(val2.has_value());
        EXPECT_EQ(val2.value(), "value2");
    }

    // Should be cleaned up
    auto val1_after = ctx.get_context<std::string>("init1");
    auto val2_after = ctx.get_context<std::string>("init2");
    EXPECT_FALSE(val1_after.has_value());
    EXPECT_FALSE(val2_after.has_value());
}

TEST_F(ContextScopeTest, ScopeMoveSemantics) {
    auto& ctx = LogContext::get_instance();

    auto create_scope = [&ctx]() {
        auto scope = ctx.create_scope("movable");
        scope.set_context("move_test", "moved");
        return scope;
    };

    {
        auto moved_scope = create_scope();
        EXPECT_TRUE(moved_scope.is_valid());
        EXPECT_EQ(moved_scope.get_scope_name(), "movable");

        auto val = ctx.get_context<std::string>("move_test");
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(val.value(), "moved");
    }

    // Should clean up properly
    EXPECT_EQ(ctx.get_scope_depth(), 0);
    auto val_after = ctx.get_context<std::string>("move_test");
    EXPECT_FALSE(val_after.has_value());
}

// =============================================================================
// Context Validation Tests
// =============================================================================

class ContextValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(ContextValidationTest, ValidatorRegistrationAndUsage) {
    auto& ctx = LogContext::get_instance();

    // Register validator for user ID format
    ctx.register_validator("user.id", [](const std::string& /*key*/, const std::string& value) {
        return !value.empty() && value.length() <= 10 && value.find_first_not_of("0123456789") == std::string::npos;
    });

    // Valid user ID
    EXPECT_TRUE(ctx.validate_context("user.id", "12345"));
    EXPECT_TRUE(ctx.validate_context("user.id", "0"));

    // Invalid user IDs
    EXPECT_FALSE(ctx.validate_context("user.id", ""));  // Empty
    EXPECT_FALSE(ctx.validate_context("user.id", "12345678901"));  // Too long
    EXPECT_FALSE(ctx.validate_context("user.id", "abc123"));  // Non-numeric

    // Non-matching keys should pass (no validator)
    EXPECT_TRUE(ctx.validate_context("other.key", "any_value"));
}

TEST_F(ContextValidationTest, MultipleValidators) {
    auto& ctx = LogContext::get_instance();

    // Email validator
    ctx.register_validator("email", [](const std::string& /*key*/, const std::string& value) {
        return value.find('@') != std::string::npos;
    });

    // Length validator for any key containing "name"
    ctx.register_validator("name", [](const std::string& /*key*/, const std::string& value) {
        return value.length() >= 2 && value.length() <= 50;
    });

    EXPECT_TRUE(ctx.validate_context("user.email", "test@example.com"));
    EXPECT_FALSE(ctx.validate_context("user.email", "invalid_email"));

    EXPECT_TRUE(ctx.validate_context("user.name", "John Doe"));
    EXPECT_FALSE(ctx.validate_context("user.name", "A"));  // Too short
    EXPECT_FALSE(ctx.validate_context("full.name", std::string(51, 'x')));  // Too long
}

// =============================================================================
// Thread Safety and Concurrency Tests
// =============================================================================

class LogContextConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(LogContextConcurrencyTest, GlobalContextThreadSafety) {
    auto& ctx = LogContext::get_instance();

    const int num_threads = 4;
    const int operations_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> error_count{0};

    // Multiple threads setting and getting global context
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&ctx, &error_count, t, operations_per_thread]() {
            try {
                for (int i = 0; i < operations_per_thread; ++i) {
                    std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
                    std::string value = "value_" + std::to_string(i);

                    ctx.set_global_context(key, value);

                    auto retrieved = ctx.get_context<std::string>(key);
                    if (!retrieved.has_value() || retrieved.value() != value) {
                        error_count++;
                    }
                }
            } catch (const std::exception& e) {
                error_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(error_count.load(), 0);

    // Verify all values are still accessible
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < operations_per_thread; ++i) {
            std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
            std::string expected = "value_" + std::to_string(i);

            auto value = ctx.get_context<std::string>(key);
            ASSERT_TRUE(value.has_value()) << "Missing key: " << key;
            EXPECT_EQ(value.value(), expected);
        }
    }
}

TEST_F(LogContextConcurrencyTest, ThreadLocalContextIsolation) {
    auto& ctx = LogContext::get_instance();

    const int num_threads = 4;
    std::vector<std::future<bool>> futures;

    for (int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&ctx, t]() {
            try {
                // Each thread sets its own thread-local context
                ctx.set_thread_context("thread.id", t);
                ctx.set_thread_context("thread.name", "worker_" + std::to_string(t));

                // Brief delay to allow other threads to run
                std::this_thread::sleep_for(10ms);

                // Verify thread-local context is isolated
                auto thread_id = ctx.get_context<int>("thread.id");
                auto thread_name = ctx.get_context<std::string>("thread.name");

                if (!thread_id.has_value() || thread_id.value() != t) {
                    return false;
                }

                if (!thread_name.has_value() || thread_name.value() != ("worker_" + std::to_string(t))) {
                    return false;
                }

                return true;
            } catch (const std::exception& e) {
                return false;
            }
        }));
    }

    for (auto& future : futures) {
        EXPECT_TRUE(future.get());
    }
}

TEST_F(LogContextConcurrencyTest, ConcurrentScopeOperations) {
    auto& ctx = LogContext::get_instance();

    const int num_threads = 4;
    std::vector<std::future<bool>> futures;

    for (int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&ctx, t]() {
            try {
                ctx.set_thread_context("base", "thread_" + std::to_string(t));

                // Create nested scopes
                {
                    auto scope1 = ctx.create_scope("scope1_t" + std::to_string(t));
                    scope1.set_context("level", "1");

                    {
                        auto scope2 = ctx.create_scope("scope2_t" + std::to_string(t));
                        scope2.set_context("level", "2");

                        auto level = ctx.get_context<std::string>("level");
                        if (!level.has_value() || level.value() != "2") {
                            return false;
                        }

                        auto base = ctx.get_context<std::string>("base");
                        if (!base.has_value() || base.value() != ("thread_" + std::to_string(t))) {
                            return false;
                        }
                    }

                    // Back to scope1
                    auto level = ctx.get_context<std::string>("level");
                    if (!level.has_value() || level.value() != "1") {
                        return false;
                    }
                }

                // Back to thread level
                if (ctx.get_scope_depth() != 0) {
                    return false;
                }

                return true;
            } catch (const std::exception& e) {
                return false;
            }
        }));
    }

    for (auto& future : futures) {
        EXPECT_TRUE(future.get());
    }
}

// =============================================================================
// Effective Context and Resolution Tests
// =============================================================================

class ContextResolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(ContextResolutionTest, EffectiveContextMerging) {
    auto& ctx = LogContext::get_instance();

    // Set up hierarchy
    ctx.set_global_context("global_only", "global_value");
    ctx.set_global_context("overridden", "global_override");

    ctx.set_thread_context("thread_only", "thread_value");
    ctx.set_thread_context("overridden", "thread_override");

    {
        auto scope = ctx.create_scope("test_scope");
        scope.set_context("scope_only", "scope_value");
        scope.set_context("overridden", "scope_override");

        // Test message-specific context
        LogContext::ContextMap message_ctx = {
            {"message_only", "message_value"},
            {"overridden", "message_override"}
        };

        auto effective = ctx.get_effective_context(&message_ctx);

        // Verify all levels are present with correct priority
        EXPECT_EQ(effective["global_only"], "global_value");
        EXPECT_EQ(effective["thread_only"], "thread_value");
        EXPECT_EQ(effective["scope_only"], "scope_value");
        EXPECT_EQ(effective["message_only"], "message_value");

        // Verify override priority (message wins)
        EXPECT_EQ(effective["overridden"], "message_override");

        // Test without message context
        auto scope_effective = ctx.get_effective_context();
        EXPECT_EQ(scope_effective["overridden"], "scope_override");
        EXPECT_EQ(scope_effective.find("message_only"), scope_effective.end());
    }

    // After scope ends
    auto thread_effective = ctx.get_effective_context();
    EXPECT_EQ(thread_effective["overridden"], "thread_override");
    EXPECT_EQ(thread_effective.find("scope_only"), thread_effective.end());
}

TEST_F(ContextResolutionTest, MessageContextPriority) {
    auto& ctx = LogContext::get_instance();

    ctx.set_global_context("key", "global");
    ctx.set_thread_context("key", "thread");

    {
        auto scope = ctx.create_scope("test");
        scope.set_context("key", "scope");

        // Without message context
        auto without_msg = ctx.get_context<std::string>("key");
        ASSERT_TRUE(without_msg.has_value());
        EXPECT_EQ(without_msg.value(), "scope");

        // With message context (highest priority)
        LogContext::ContextMap message_ctx = {{"key", "message"}};
        auto with_msg = ctx.get_context<std::string>("key", &message_ctx);
        ASSERT_TRUE(with_msg.has_value());
        EXPECT_EQ(with_msg.value(), "message");
    }
}

// =============================================================================
// Context Statistics and Utility Tests
// =============================================================================

class ContextUtilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(ContextUtilityTest, ContextStatistics) {
    auto& ctx = LogContext::get_instance();

    // Set up some context
    ctx.set_global_context("global1", "value1");
    ctx.set_global_context("global2", "value2");

    ctx.set_thread_context("thread1", "value1");
    ctx.set_thread_context("thread2", "value2");
    ctx.set_thread_context("thread3", "value3");

    auto stats = ctx.get_context_stats();

    EXPECT_EQ(stats.global_keys, 2);
    EXPECT_EQ(stats.thread_keys, 3);
    EXPECT_EQ(stats.scope_depth, 0);
    EXPECT_EQ(stats.scope_keys, 0);
    EXPECT_EQ(stats.thread_id, std::this_thread::get_id());

    {
        // Use ISOLATED scope to avoid inheriting thread context
        auto scope = ctx.create_scope("test", LogContext::ScopeType::ISOLATED);
        scope.set_context("scope1", "value1");
        scope.set_context("scope2", "value2");

        auto scope_stats = ctx.get_context_stats();
        EXPECT_EQ(scope_stats.scope_depth, 1);
        EXPECT_EQ(scope_stats.scope_keys, 2);
    }

    // After scope
    auto final_stats = ctx.get_context_stats();
    EXPECT_EQ(final_stats.scope_depth, 0);
    EXPECT_EQ(final_stats.scope_keys, 0);
}

TEST_F(ContextUtilityTest, ContextClearOperations) {
    auto& ctx = LogContext::get_instance();

    // Set up context at all levels
    ctx.set_global_context("global", "value");
    ctx.set_thread_context("thread", "value");

    {
        // Use ISOLATED scope to test clearing thread context
        auto scope = ctx.create_scope("test", LogContext::ScopeType::ISOLATED);
        scope.set_context("scope", "value");

        // Clear thread context (shouldn't affect scope or global)
        ctx.clear_thread_context();

        auto global_val = ctx.get_context<std::string>("global");
        auto thread_val = ctx.get_context<std::string>("thread");
        auto scope_val = ctx.get_context<std::string>("scope");

        ASSERT_TRUE(global_val.has_value());
        EXPECT_FALSE(thread_val.has_value()); // Thread context should be cleared
        ASSERT_TRUE(scope_val.has_value());   // Scope context should remain
    }

    // Clear global context
    ctx.clear_global_context();

    auto global_after = ctx.get_context<std::string>("global");
    EXPECT_FALSE(global_after.has_value());
}

// =============================================================================
// Macro Usage Tests
// =============================================================================

TEST_F(ContextUtilityTest, ConvenienceMacrosUsage) {
    auto& ctx = LogContext::get_instance();

    // Test global macro
    LOG_CONTEXT_GLOBAL("macro.global", "global_macro_value");
    auto global_val = ctx.get_context<std::string>("macro.global");
    ASSERT_TRUE(global_val.has_value());
    EXPECT_EQ(global_val.value(), "global_macro_value");

    // Test thread macro
    LOG_CONTEXT_THREAD("macro.thread", "thread_macro_value");
    auto thread_val = ctx.get_context<std::string>("macro.thread");
    ASSERT_TRUE(thread_val.has_value());
    EXPECT_EQ(thread_val.value(), "thread_macro_value");

    // Test scope macros
    {
        LOG_CONTEXT_SCOPE("macro_scope");
        EXPECT_EQ(ctx.get_scope_depth(), 1);

        LOG_CONTEXT_SET("macro.scope", "scope_macro_value");
        auto scope_val = ctx.get_context<std::string>("macro.scope");
        ASSERT_TRUE(scope_val.has_value());
        EXPECT_EQ(scope_val.value(), "scope_macro_value");
    }

    EXPECT_EQ(ctx.get_scope_depth(), 0);
    auto scope_after = ctx.get_context<std::string>("macro.scope");
    EXPECT_FALSE(scope_after.has_value());

    // Test isolated scope macro
    {
        LOG_CONTEXT_ISOLATED("isolated_scope");
        EXPECT_EQ(ctx.get_scope_depth(), 1);

        LOG_CONTEXT_SET("isolated.key", "isolated_value");
        auto isolated_val = ctx.get_context<std::string>("isolated.key");
        ASSERT_TRUE(isolated_val.has_value());
        EXPECT_EQ(isolated_val.value(), "isolated_value");
    }

    auto isolated_after = ctx.get_context<std::string>("isolated.key");
    EXPECT_FALSE(isolated_after.has_value());
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

class ContextEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = std::make_unique<LogContextFixture>();
    }

    void TearDown() override {
        fixture_.reset();
    }

    std::unique_ptr<LogContextFixture> fixture_;
};

TEST_F(ContextEdgeCasesTest, EmptyKeyAndValueHandling) {
    auto& ctx = LogContext::get_instance();

    // Empty key should work
    ctx.set_global_context("", "empty_key_value");
    auto empty_key = ctx.get_context<std::string>("");
    ASSERT_TRUE(empty_key.has_value());
    EXPECT_EQ(empty_key.value(), "empty_key_value");

    // Empty value should work
    ctx.set_global_context("empty_value", "");
    auto empty_value = ctx.get_context<std::string>("empty_value");
    ASSERT_TRUE(empty_value.has_value());
    EXPECT_EQ(empty_value.value(), "");
}

TEST_F(ContextEdgeCasesTest, LargeContextValues) {
    auto& ctx = LogContext::get_instance();

    std::string large_value(10000, 'x');
    ctx.set_global_context("large_value", large_value);

    auto retrieved = ctx.get_context<std::string>("large_value");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved.value(), large_value);
}

TEST_F(ContextEdgeCasesTest, SpecialCharacterHandling) {
    auto& ctx = LogContext::get_instance();

    std::string special_key = "key.with-special_chars@#$%";
    std::string special_value = "value with spaces\nand\ttabs and émojis 🚀";

    ctx.set_global_context(special_key, special_value);

    auto retrieved = ctx.get_context<std::string>(special_key);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved.value(), special_value);
}

TEST_F(ContextEdgeCasesTest, ContextOverwriteBehavior) {
    auto& ctx = LogContext::get_instance();

    ctx.set_global_context("overwrite_test", "original");
    auto original = ctx.get_context<std::string>("overwrite_test");
    ASSERT_TRUE(original.has_value());
    EXPECT_EQ(original.value(), "original");

    // Overwrite with different type
    ctx.set_global_context("overwrite_test", 42);
    auto overwritten = ctx.get_context<std::string>("overwrite_test");
    ASSERT_TRUE(overwritten.has_value());
    EXPECT_EQ(overwritten.value(), "42");

    // Should also work as int
    auto as_int = ctx.get_context<int>("overwrite_test");
    ASSERT_TRUE(as_int.has_value());
    EXPECT_EQ(as_int.value(), 42);
}

// =============================================================================
// Performance and Stress Tests (Disabled by Default)
// =============================================================================

TEST(LogContextPerformanceTest, DISABLED_HighVolumeOperations) {
    auto fixture = std::make_unique<LogContextFixture>();
    auto& ctx = LogContext::get_instance();

    const int num_operations = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_operations; ++i) {
        ctx.set_global_context("perf_key_" + std::to_string(i % 100), i);
        auto val = ctx.get_context<int>("perf_key_" + std::to_string(i % 100));
        (void)val; // Suppress unused variable warning
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Performed " << num_operations << " context operations in "
              << duration.count() << " microseconds" << std::endl;

    // Should complete in reasonable time (adjust threshold as needed)
    EXPECT_LT(duration.count(), 1000000); // Less than 1 second
}

TEST(LogContextPerformanceTest, DISABLED_HighVolumeScopeOperations) {
    auto fixture = std::make_unique<LogContextFixture>();
    auto& ctx = LogContext::get_instance();

    const int num_scopes = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_scopes; ++i) {
        auto scope = ctx.create_scope("perf_scope_" + std::to_string(i));
        scope.set_context("iteration", i);
        auto val = scope.get_context<int>("iteration");
        (void)val; // Suppress unused variable warning
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Created/destroyed " << num_scopes << " scopes in "
              << duration.count() << " microseconds" << std::endl;

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 1000000); // Less than 1 second

    // All scopes should be cleaned up
    EXPECT_EQ(ctx.get_scope_depth(), 0);
}
#include <gtest/gtest.h>
#include <core/error/error_context.h>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

using namespace fem::core::error;

class ErrorContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any existing context
        ErrorContext::clear();
        ErrorContext::clear_global();
    }

    void TearDown() override {
        // Clean up after tests
        ErrorContext::clear();
        ErrorContext::clear_global();
    }
};

// Basic context operations
TEST_F(ErrorContextTest, SetAndGetValue) {
    ErrorContext::set("key1", std::string("value1"));
    ErrorContext::set("key2", 42);
    ErrorContext::set("key3", 3.14);
    ErrorContext::set("key4", true);

    auto str_val = ErrorContext::get<std::string>("key1");
    ASSERT_TRUE(str_val.has_value());
    EXPECT_EQ(str_val.value(), "value1");

    auto int_val = ErrorContext::get<int>("key2");
    ASSERT_TRUE(int_val.has_value());
    EXPECT_EQ(int_val.value(), 42);

    auto double_val = ErrorContext::get<double>("key3");
    ASSERT_TRUE(double_val.has_value());
    EXPECT_DOUBLE_EQ(double_val.value(), 3.14);

    auto bool_val = ErrorContext::get<bool>("key4");
    ASSERT_TRUE(bool_val.has_value());
    EXPECT_TRUE(bool_val.value());
}

TEST_F(ErrorContextTest, GetNonExistentValue) {
    auto result = ErrorContext::get<std::string>("nonexistent");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ErrorContextTest, GetWrongType) {
    ErrorContext::set("key", std::string("value"));

    auto result = ErrorContext::get<int>("key");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ErrorContextTest, ClearContext) {
    ErrorContext::set("key1", "value1");
    ErrorContext::set("key2", 42);

    EXPECT_TRUE(ErrorContext::get<std::string>("key1").has_value());
    EXPECT_TRUE(ErrorContext::get<int>("key2").has_value());

    ErrorContext::clear();

    EXPECT_FALSE(ErrorContext::get<std::string>("key1").has_value());
    EXPECT_FALSE(ErrorContext::get<int>("key2").has_value());
}

// Global context tests
TEST_F(ErrorContextTest, GlobalContext) {
    ErrorContext::set_global("global_key", std::string("global_value"));
    ErrorContext::set_global("global_int", 100);

    auto str_val = ErrorContext::get_global<std::string>("global_key");
    ASSERT_TRUE(str_val.has_value());
    EXPECT_EQ(str_val.value(), "global_value");

    auto int_val = ErrorContext::get_global<int>("global_int");
    ASSERT_TRUE(int_val.has_value());
    EXPECT_EQ(int_val.value(), 100);
}

TEST_F(ErrorContextTest, GlobalAndLocalContextSeparation) {
    ErrorContext::set_global("shared_key", std::string("global"));
    ErrorContext::set("shared_key", std::string("local"));

    auto global = ErrorContext::get_global<std::string>("shared_key");
    auto local = ErrorContext::get<std::string>("shared_key");

    ASSERT_TRUE(global.has_value());
    ASSERT_TRUE(local.has_value());
    EXPECT_EQ(global.value(), "global");
    EXPECT_EQ(local.value(), "local");
}

TEST_F(ErrorContextTest, ClearGlobalContext) {
    ErrorContext::set_global("key", "value");
    EXPECT_TRUE(ErrorContext::get_global<std::string>("key").has_value());

    ErrorContext::clear_global();
    EXPECT_FALSE(ErrorContext::get_global<std::string>("key").has_value());
}

// Scope tests
TEST_F(ErrorContextTest, ScopeCreation) {
    {
        ErrorContext::Scope scope("test_scope");
        EXPECT_EQ(ErrorContext::scope_path(), "test_scope");
    }

    // After scope exits, path should be empty
    EXPECT_EQ(ErrorContext::scope_path(), "");
}

TEST_F(ErrorContextTest, NestedScopes) {
    ErrorContext::Scope scope1("scope1");
    EXPECT_EQ(ErrorContext::scope_path(), "scope1");

    {
        ErrorContext::Scope scope2("scope2");
        EXPECT_EQ(ErrorContext::scope_path(), "scope1.scope2");

        {
            ErrorContext::Scope scope3("scope3");
            EXPECT_EQ(ErrorContext::scope_path(), "scope1.scope2.scope3");
        }

        EXPECT_EQ(ErrorContext::scope_path(), "scope1.scope2");
    }

    EXPECT_EQ(ErrorContext::scope_path(), "scope1");
}

TEST_F(ErrorContextTest, ScopeWithValues) {
    ErrorContext::Scope scope("test");
    scope.add("key1", "value1")
         .add("key2", 42);

    auto str_val = ErrorContext::get<std::string>("key1");
    ASSERT_TRUE(str_val.has_value());
    EXPECT_EQ(str_val.value(), "value1");

    auto int_val = ErrorContext::get<int>("key2");
    ASSERT_TRUE(int_val.has_value());
    EXPECT_EQ(int_val.value(), 42);
}

TEST_F(ErrorContextTest, ScopeElapsedTime) {
    ErrorContext::Scope scope("timer_test");

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto elapsed = scope.elapsed();
    EXPECT_GE(elapsed.count(), 50);
    EXPECT_LT(elapsed.count(), 100);  // Should be less than 100ms
}

// Capture tests
TEST_F(ErrorContextTest, CaptureContext) {
    ErrorContext::set("string_key", std::string("string_value"));
    ErrorContext::set("int_key", 42);
    ErrorContext::set("double_key", 3.14);
    ErrorContext::set("bool_key", true);

    auto captured = ErrorContext::capture();

    EXPECT_EQ(captured["string_key"], "string_value");
    EXPECT_EQ(captured["int_key"], "42");
    EXPECT_EQ(captured["double_key"], "3.140000");
    EXPECT_EQ(captured["bool_key"], "true");
    EXPECT_TRUE(captured.find("thread_id") != captured.end());
}

TEST_F(ErrorContextTest, CaptureWithScope) {
    ErrorContext::Scope scope("test_scope");
    ErrorContext::set("key", "value");

    auto captured = ErrorContext::capture();

    EXPECT_EQ(captured["scope"], "test_scope");
    EXPECT_TRUE(captured.find("test_scope.key") != captured.end());
}

TEST_F(ErrorContextTest, CaptureGlobalContext) {
    ErrorContext::set_global("global1", std::string("value1"));
    ErrorContext::set_global("global2", 100);

    auto captured = ErrorContext::capture_global();

    EXPECT_EQ(captured["global1"], "value1");
    EXPECT_EQ(captured["global2"], "100");
}

// Format tests
TEST_F(ErrorContextTest, FormatEmptyContext) {
    ErrorContext::StringMap empty;
    auto formatted = ErrorContext::format(empty);
    EXPECT_EQ(formatted, "");
}

TEST_F(ErrorContextTest, FormatContext) {
    ErrorContext::StringMap context = {
        {"key1", "value1"},
        {"key2", "value2"},
        {"key3", "value3"}
    };

    auto formatted = ErrorContext::format(context);

    EXPECT_TRUE(formatted.find("key1: value1") != std::string::npos);
    EXPECT_TRUE(formatted.find("key2: value2") != std::string::npos);
    EXPECT_TRUE(formatted.find("key3: value3") != std::string::npos);
}

TEST_F(ErrorContextTest, FormatWithCustomIndent) {
    ErrorContext::StringMap context = {{"key", "value"}};

    auto formatted = ErrorContext::format(context, "    ");
    EXPECT_TRUE(formatted.find("    key: value") != std::string::npos);
}

// Thread safety tests
TEST_F(ErrorContextTest, ThreadLocalContext) {
    std::atomic<bool> thread1_done{false};
    std::atomic<bool> thread2_done{false};

    std::thread thread1([&]() {
        ErrorContext::set("thread_key", std::string("thread1_value"));
        auto value = ErrorContext::get<std::string>("thread_key");
        EXPECT_TRUE(value.has_value());
        EXPECT_EQ(value.value(), "thread1_value");
        thread1_done = true;
    });

    std::thread thread2([&]() {
        ErrorContext::set("thread_key", std::string("thread2_value"));
        auto value = ErrorContext::get<std::string>("thread_key");
        EXPECT_TRUE(value.has_value());
        EXPECT_EQ(value.value(), "thread2_value");
        thread2_done = true;
    });

    thread1.join();
    thread2.join();

    EXPECT_TRUE(thread1_done);
    EXPECT_TRUE(thread2_done);

    // Main thread should not see thread-local values
    auto main_value = ErrorContext::get<std::string>("thread_key");
    EXPECT_FALSE(main_value.has_value());
}

TEST_F(ErrorContextTest, GlobalContextThreadSafety) {
    const int num_threads = 10;
    const int operations_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string key = "thread_" + std::to_string(t) + "_" + std::to_string(i);
                ErrorContext::set_global(key, t * 1000 + i);

                // Try to read back
                auto value = ErrorContext::get_global<int>(key);
                EXPECT_TRUE(value.has_value());
                if (value.has_value()) {
                    EXPECT_EQ(value.value(), t * 1000 + i);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all values are present
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < operations_per_thread; ++i) {
            std::string key = "thread_" + std::to_string(t) + "_" + std::to_string(i);
            auto value = ErrorContext::get_global<int>(key);
            EXPECT_TRUE(value.has_value());
            if (value.has_value()) {
                EXPECT_EQ(value.value(), t * 1000 + i);
            }
        }
    }
}

// ExtendedContext tests
TEST_F(ErrorContextTest, ExtendedContextCapture) {
    ErrorContext::set("local_key", std::string("local_value"));
    ErrorContext::set_global("global_key", std::string("global_value"));

    ErrorContext::Scope scope("test_scope");

    auto info = ExtendedContext::capture();

    EXPECT_TRUE(info.location.is_valid());
    EXPECT_FALSE(info.stack_trace.empty());
    EXPECT_FALSE(info.context.empty());
    EXPECT_FALSE(info.global_context.empty());
    EXPECT_EQ(info.scope_path, "test_scope");
}

TEST_F(ErrorContextTest, ExtendedContextFormat) {
    ErrorContext::set("key", std::string("value"));
    ErrorContext::Scope scope("format_test");

    auto info = ExtendedContext::capture();
    auto formatted = ExtendedContext::format(info);

    // Check that formatted output contains expected elements
    EXPECT_TRUE(formatted.find("Timestamp:") != std::string::npos);
    EXPECT_TRUE(formatted.find("Location:") != std::string::npos);
    EXPECT_TRUE(formatted.find("Thread:") != std::string::npos);
    EXPECT_TRUE(formatted.find("Scope: format_test") != std::string::npos);
    EXPECT_TRUE(formatted.find("Context:") != std::string::npos);
}

// ContextGuard tests
TEST_F(ErrorContextTest, ContextGuardBasic) {
    {
        ContextGuard guard;
        guard.add("temp_key", std::string("temp_value"));

        auto value = ErrorContext::get<std::string>("temp_key");
        EXPECT_TRUE(value.has_value());
        EXPECT_EQ(value.value(), "temp_value");
    }

    // After guard exits, values should still be there
    // (as noted in the implementation, removal isn't implemented)
    auto value = ErrorContext::get<std::string>("temp_key");
    EXPECT_TRUE(value.has_value());
}

TEST_F(ErrorContextTest, ContextGuardWithVariadicConstructor) {
    {
        ContextGuard guard("key1", "value1", "key2", 42);

        auto str_val = ErrorContext::get<std::string>("key1");
        EXPECT_TRUE(str_val.has_value());
        EXPECT_EQ(str_val.value(), "value1");

        auto int_val = ErrorContext::get<int>("key2");
        EXPECT_TRUE(int_val.has_value());
        EXPECT_EQ(int_val.value(), 42);
    }
}

TEST_F(ErrorContextTest, ContextGuardChaining) {
    ContextGuard guard;
    guard.add("key1", "value1")
         .add("key2", 42)
         .add("key3", true);

    EXPECT_TRUE(ErrorContext::get<std::string>("key1").has_value());
    EXPECT_TRUE(ErrorContext::get<int>("key2").has_value());
    EXPECT_TRUE(ErrorContext::get<bool>("key3").has_value());
}

// Edge cases and special types
TEST_F(ErrorContextTest, SetConstCharPointer) {
    const char* cstr = "c-string value";
    ErrorContext::set("cstr_key", cstr);

    auto captured = ErrorContext::capture();
    EXPECT_EQ(captured["cstr_key"], "c-string value");
}

TEST_F(ErrorContextTest, SetLongValue) {
    long long_val = 1234567890L;
    ErrorContext::set("long_key", long_val);

    auto value = ErrorContext::get<long>("long_key");
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), long_val);

    auto captured = ErrorContext::capture();
    EXPECT_EQ(captured["long_key"], "1234567890");
}

TEST_F(ErrorContextTest, SetUnknownType) {
    struct CustomType {
        int value = 42;
    };

    CustomType custom;
    ErrorContext::set("custom_key", custom);

    auto captured = ErrorContext::capture();
    // Should show type name for unknown types
    EXPECT_TRUE(captured["custom_key"].find("<") != std::string::npos);
    EXPECT_TRUE(captured["custom_key"].find(">") != std::string::npos);
}

// Macro tests
TEST_F(ErrorContextTest, ContextScopeMacro) {
    {
        FEM_CONTEXT_SCOPE("macro_scope");
        EXPECT_EQ(ErrorContext::scope_path(), "macro_scope");
    }
    EXPECT_EQ(ErrorContext::scope_path(), "");
}

TEST_F(ErrorContextTest, ContextValueMacro) {
    FEM_CONTEXT_VALUE("macro_key", "macro_value");

    auto value = ErrorContext::get<std::string>("macro_key");
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "macro_value");
}

TEST_F(ErrorContextTest, ContextGuardMacro) {
    {
        FEM_CONTEXT_GUARD("guard_key1", "guard_value1", "guard_key2", 42);

        auto str_val = ErrorContext::get<std::string>("guard_key1");
        EXPECT_TRUE(str_val.has_value());
        EXPECT_EQ(str_val.value(), "guard_value1");

        auto int_val = ErrorContext::get<int>("guard_key2");
        EXPECT_TRUE(int_val.has_value());
        EXPECT_EQ(int_val.value(), 42);
    }
}

TEST_F(ErrorContextTest, CaptureContextMacro) {
    ErrorContext::set("test_key", "test_value");
    ErrorContext::Scope scope("capture_test");

    auto info = FEM_CAPTURE_CONTEXT();

    EXPECT_TRUE(info.location.is_valid());
    EXPECT_EQ(info.scope_path, "capture_test");
    EXPECT_FALSE(info.context.empty());
}
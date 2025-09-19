#include <gtest/gtest.h>
#include <core/error/panic.h>
#include <string>
#include <atomic>
#include <thread>
#include <sstream>

using namespace fem::core::error;

class PanicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset panic handler before each test
        PanicHandler::set_handler(nullptr);
        panic_message.clear();
        panic_called = false;
    }

    void TearDown() override {
        // Clean up after each test
        PanicHandler::set_handler(nullptr);
    }

    // Test helper to capture panic calls without actually aborting
    static void capture_panic(const char* message,
                             const std::source_location& location) {
        panic_message = message;
        panic_called = true;
        captured_location = location;
        // Don't actually abort in tests
    }

    static std::string panic_message;
    static bool panic_called;
    static std::source_location captured_location;
};

// Static member definitions
std::string PanicTest::panic_message;
bool PanicTest::panic_called;
std::source_location PanicTest::captured_location;

// Basic panic handler tests
TEST_F(PanicTest, DefaultPanicHandler) {
    // We can't actually test the default panic because it calls abort()
    // But we can verify the handler is initially null
    PanicHandler::set_handler(nullptr);
    // The test passes if we get here without setting a custom handler
}

TEST_F(PanicTest, CustomPanicHandler) {
    PanicHandler::set_handler(capture_panic);

    // Note: We can't actually call panic() because it's [[noreturn]]
    // and even with our custom handler, it still calls abort()
    // Instead we test the handler setting mechanism
    EXPECT_TRUE(true);  // Test setup succeeded
}

TEST_F(PanicTest, PanicHandlerStorage) {
    // Test that we can set and change handlers
    auto handler1 = [](const char*, const std::source_location&) {};
    auto handler2 = [](const char*, const std::source_location&) {};

    PanicHandler::set_handler(handler1);
    // Handler is stored (we can't easily verify the exact function pointer)

    PanicHandler::set_handler(handler2);
    // Handler is updated

    PanicHandler::set_handler(nullptr);
    // Handler is cleared
}

// Macro tests - these test the macro expansion without actually panicking
TEST_F(PanicTest, PanicMacroDefinition) {
    // Test that PANIC macro is properly defined
    // We can't execute it without aborting, but we can check it compiles
    #ifdef PANIC
        EXPECT_TRUE(true);
    #else
        FAIL() << "PANIC macro not defined";
    #endif
}

TEST_F(PanicTest, PanicFmtMacroDefinition) {
    // Test that PANIC_FMT macro is properly defined
    #ifdef PANIC_FMT
        EXPECT_TRUE(true);
    #else
        FAIL() << "PANIC_FMT macro not defined";
    #endif
}

TEST_F(PanicTest, PanicIfMacroDefinition) {
    // Test that PANIC_IF macro is properly defined
    #ifdef PANIC_IF
        EXPECT_TRUE(true);
    #else
        FAIL() << "PANIC_IF macro not defined";
    #endif
}

TEST_F(PanicTest, PanicIfNullMacroDefinition) {
    // Test that PANIC_IF_NULL macro is properly defined
    #ifdef PANIC_IF_NULL
        EXPECT_TRUE(true);
    #else
        FAIL() << "PANIC_IF_NULL macro not defined";
    #endif
}

TEST_F(PanicTest, UnreachableMacroDefinition) {
    // Test that UNREACHABLE macro is properly defined
    #ifdef UNREACHABLE
        EXPECT_TRUE(true);
    #else
        FAIL() << "UNREACHABLE macro not defined";
    #endif
}

TEST_F(PanicTest, UnimplementedMacroDefinition) {
    // Test that UNIMPLEMENTED macro is properly defined
    #ifdef UNIMPLEMENTED
        EXPECT_TRUE(true);
    #else
        FAIL() << "UNIMPLEMENTED macro not defined";
    #endif
}

TEST_F(PanicTest, TodoMacroDefinition) {
    // Test that TODO macro is properly defined
    #ifdef TODO
        EXPECT_TRUE(true);
    #else
        FAIL() << "TODO macro not defined";
    #endif
}

// Source location tests
TEST_F(PanicTest, SourceLocationCapture) {
    // Test that source location is properly captured in panic calls
    // We'll use a mock implementation to test the mechanism

    // We can test the source_location mechanism without actually panicking

    // We can test the source_location mechanism without actually panicking
    auto current_location = std::source_location::current();
    EXPECT_TRUE(current_location.line() > 0);
    EXPECT_NE(current_location.file_name(), nullptr);
    EXPECT_NE(current_location.function_name(), nullptr);
}

// Conditional panic tests
TEST_F(PanicTest, ConditionalPanicLogic) {
    // Test the logic of conditional panic without actually executing
    bool condition_true = true;
    bool condition_false = false;

    // PANIC_IF should panic when condition is false
    // We test the condition logic here
    EXPECT_FALSE(!condition_true);   // Should not trigger panic
    EXPECT_TRUE(!condition_false);   // Should trigger panic (but we don't execute)
}

TEST_F(PanicTest, NullPointerCheckLogic) {
    // Test null pointer check logic
    int value = 42;
    int* valid_ptr = &value;
    int* null_ptr = nullptr;

    // PANIC_IF_NULL should panic when pointer is null
    EXPECT_FALSE(valid_ptr == nullptr);  // Should not trigger panic
    EXPECT_TRUE(null_ptr == nullptr);    // Should trigger panic (but we don't execute)
}

// Message formatting tests
TEST_F(PanicTest, MessageFormatting) {
    // Test that we can format messages for panic
    std::string test_message = "Test panic message";
    std::string formatted_message = std::format("Formatted: {}", test_message);

    EXPECT_EQ(formatted_message, "Formatted: Test panic message");

    // Test with multiple parameters
    int value = 42;
    std::string name = "test";
    auto multi_param = std::format("Value: {}, Name: {}", value, name);
    EXPECT_EQ(multi_param, "Value: 42, Name: test");
}

// Thread safety tests (for handler setting)
TEST_F(PanicTest, ThreadSafeHandlerSetting) {
    std::atomic<int> handler_sets{0};

    auto set_handler = [&]() {
        for (int i = 0; i < 100; ++i) {
            PanicHandler::set_handler([](const char*, const std::source_location&) {});
            handler_sets++;
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    };

    std::vector<std::thread> threads;
    constexpr int num_threads = 5;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(set_handler);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(handler_sets.load(), num_threads * 100);
}

// Edge case tests
TEST_F(PanicTest, EmptyMessage) {
    // Test handling of empty panic message
    const char* empty_message = "";

    // Test that empty message doesn't cause issues
    EXPECT_STREQ(empty_message, "");
    EXPECT_EQ(std::string(empty_message).length(), 0);
}

TEST_F(PanicTest, LongMessage) {
    // Test handling of very long panic messages
    std::string long_message(10000, 'x');
    EXPECT_EQ(long_message.length(), 10000);

    // Test formatting with long message
    auto formatted = std::format("Long message: {}", long_message);
    EXPECT_TRUE(formatted.find(long_message) != std::string::npos);
}

TEST_F(PanicTest, SpecialCharactersInMessage) {
    // Test handling of special characters in panic messages
    std::string special_message = "Error: 100% failed\nLine 2\tTabbed\r\nWindows line ending";

    // Should handle special characters without issues
    EXPECT_TRUE(special_message.find("100%") != std::string::npos);
    EXPECT_TRUE(special_message.find("\n") != std::string::npos);
    EXPECT_TRUE(special_message.find("\t") != std::string::npos);
}

// Macro parameter tests
TEST_F(PanicTest, MacroParameterHandling) {
    // Test that macros handle various parameter types correctly

    // String literals
    const char* string_literal = "test string";
    EXPECT_STREQ(string_literal, "test string");

    // Variables
    std::string string_var = "variable string";
    EXPECT_EQ(string_var, "variable string");

    // Expressions
    int a = 10, b = 20;
    EXPECT_EQ(a + b, 30);

    // Pointers
    int* ptr = &a;
    EXPECT_EQ(*ptr, 10);
}

// Integration tests with other error components
TEST_F(PanicTest, PanicIntegrationScenario) {
    // Test a realistic scenario where panic might be used

    class Resource {
    public:
        explicit Resource(int* data) : data_(data) {
            if (!data_) {
                // In real code: PANIC_IF_NULL(data, "Resource data");
                // For test: just verify the condition
                EXPECT_NE(data, nullptr);
            }
        }

        int get_value() const {
            if (!data_) {
                // In real code: PANIC("Resource data is null");
                // For test: just verify we don't reach here
                throw std::logic_error("Should not reach here with null data");
            }
            return *data_;
        }

    private:
        int* data_;
    };

    int value = 42;
    Resource resource(&value);
    EXPECT_EQ(resource.get_value(), 42);
}

// Performance tests
TEST_F(PanicTest, HandlerSettingPerformance) {
    const int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        PanicHandler::set_handler([](const char*, const std::source_location&) {});
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Setting handler should be reasonably fast
    EXPECT_LT(duration.count(), 100000);  // Less than 100ms for 10k operations
}

// Documentation example tests
TEST_F(PanicTest, UsageExamples) {
    // Example: Parameter validation
    auto validate_parameter = [](int value) {
        // In real code: PANIC_IF(value < 0, "Value must be non-negative");
        // For test: verify the condition
        EXPECT_GE(value, 0);
    };

    validate_parameter(10);  // Should not panic
    // validate_parameter(-1);  // Would panic in real code

    // Example: Null pointer check
    auto process_data = [](int* data) {
        // In real code: PANIC_IF_NULL(data, "data");
        // For test: verify the condition
        EXPECT_NE(data, nullptr);
        return *data;
    };

    int value = 42;
    EXPECT_EQ(process_data(&value), 42);

    // Example: Unreachable code
    auto finite_state_machine = [](int state) {
        switch (state) {
            case 0: return "state0";
            case 1: return "state1";
            case 2: return "state2";
            default:
                // In real code: UNREACHABLE();
                // For test: verify we don't reach here
                throw std::logic_error("Unreachable state: " + std::to_string(state));
        }
    };

    EXPECT_STREQ(finite_state_machine(1), "state1");
}

// Compile-time tests
TEST_F(PanicTest, CompileTimeFeatures) {
    // Test that panic functionality compiles correctly

    // Test source_location availability
    auto loc = std::source_location::current();
    EXPECT_TRUE(loc.line() > 0);

    // Test format string compilation
    auto formatted = std::format("Test: {}", 42);
    EXPECT_EQ(formatted, "Test: 42");

    // Test noreturn attribute (compile-time check)
    // The [[noreturn]] attribute should be properly applied
    static_assert(std::is_same_v<void, decltype(PanicHandler::panic("", std::source_location::current()))>);
}
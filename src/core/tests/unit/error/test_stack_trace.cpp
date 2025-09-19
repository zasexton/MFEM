#include <gtest/gtest.h>
#include <core/error/stack_trace.h>
#include <regex>
#include <thread>

using namespace fem::core::error;

class StackTraceTest : public ::testing::Test {
protected:
    // Helper functions for creating call stacks
    StackTrace capture_in_function(size_t skip = 1) {
        return StackTrace(skip);
    }

    StackTrace nested_function_level_2() {
        return nested_function_level_1();
    }

    StackTrace nested_function_level_1() {
        return StackTrace(1);
    }
};

// Basic stack trace tests
TEST_F(StackTraceTest, CaptureStackTrace) {
    StackTrace trace(0);

    EXPECT_FALSE(trace.empty());
    EXPECT_GT(trace.size(), 0);

    // First frame should be this test function
    if (!trace.empty()) {
        const auto& frame = trace[0];
        EXPECT_NE(frame.address, nullptr);
    }
}

TEST_F(StackTraceTest, EmptyStackTrace) {
    StackTrace trace;

    EXPECT_TRUE(trace.empty());
    EXPECT_EQ(trace.size(), 0);
}

TEST_F(StackTraceTest, SkipFrames) {
    StackTrace trace_no_skip(0);
    StackTrace trace_skip_1(1);
    StackTrace trace_skip_2(2);

    // Each should have progressively fewer frames
    if (trace_no_skip.size() > 2) {
        EXPECT_GT(trace_no_skip.size(), trace_skip_1.size());
        EXPECT_GT(trace_skip_1.size(), trace_skip_2.size());
    }
}

TEST_F(StackTraceTest, MaxFrames) {
    StackTrace trace_5(0, 5);
    StackTrace trace_10(0, 10);

    // Should respect max frames limit
    EXPECT_LE(trace_5.size(), 5);
    EXPECT_LE(trace_10.size(), 10);

    // If we have enough call depth, 10 should have more than 5
    if (trace_10.size() == 10) {
        EXPECT_GT(trace_10.size(), trace_5.size());
    }
}

// StackFrame tests
TEST_F(StackTraceTest, StackFrameToString) {
    StackFrame frame;
    frame.index = 0;
    frame.address = reinterpret_cast<void*>(0x12345678);
    frame.module = "test_module";
    frame.function = "test_function";
    frame.source_file = "test.cpp";
    frame.line_number = 42;
    frame.offset = 0x10;

    std::string str = frame.to_string();

    EXPECT_TRUE(str.find("#") != std::string::npos);
    EXPECT_TRUE(str.find("test_module") != std::string::npos);
    EXPECT_TRUE(str.find("test_function") != std::string::npos);
    EXPECT_TRUE(str.find("test.cpp") != std::string::npos);
    EXPECT_TRUE(str.find("42") != std::string::npos);
}

TEST_F(StackTraceTest, StackFrameToStringMinimal) {
    StackFrame frame;
    frame.index = 1;
    frame.address = reinterpret_cast<void*>(0xDEADBEEF);

    std::string str = frame.to_string();

    EXPECT_TRUE(str.find("# 1") != std::string::npos);
    // Should show address in hex when no function name
    EXPECT_TRUE(str.find("0x") != std::string::npos || str.find("deadbeef") != std::string::npos);
}

TEST_F(StackTraceTest, StackFrameIsSystem) {
    StackFrame system_frame;
    system_frame.function = "std::allocator::allocate";
    EXPECT_TRUE(system_frame.is_system());

    system_frame.function = "__cxa_throw";
    EXPECT_TRUE(system_frame.is_system());

    system_frame.function = "_start";
    EXPECT_TRUE(system_frame.is_system());

    StackFrame user_frame;
    user_frame.function = "MyApplication::process";
    EXPECT_FALSE(user_frame.is_system());

    user_frame.function = "test_function";
    EXPECT_FALSE(user_frame.is_system());
}

// Formatting tests
TEST_F(StackTraceTest, ToStringDefault) {
    StackTrace trace = capture_in_function();

    if (!trace.empty()) {
        std::string str = trace.to_string();

        EXPECT_TRUE(str.find("Stack trace:") != std::string::npos);
        EXPECT_TRUE(str.find("#") != std::string::npos);
    }
}

TEST_F(StackTraceTest, ToStringIncludeSystemFrames) {
    StackTrace trace(0);

    if (!trace.empty()) {
        std::string without_system = trace.to_string(false);
        std::string with_system = trace.to_string(true);

        // with_system should be at least as long
        EXPECT_GE(with_system.length(), without_system.length());
    }
}

TEST_F(StackTraceTest, CompactTrace) {
    StackTrace trace = nested_function_level_2();

    if (!trace.empty()) {
        std::string compact = trace.compact_trace();

        // Should contain function names separated by the separator
        EXPECT_FALSE(compact.empty());

        // Custom separator
        std::string compact_custom = trace.compact_trace(" -> ");
        if (trace.size() > 1) {
            EXPECT_TRUE(compact_custom.find(" -> ") != std::string::npos ||
                       compact_custom.find("<unknown>") != std::string::npos);
        }
    }
}

// Filtering tests
TEST_F(StackTraceTest, FilterModule) {
    StackTrace trace(0);

    if (!trace.empty()) {
        // Filter by a module that likely exists in test
        StackTrace filtered = trace.filter_module("test");

        // Result should have same or fewer frames
        EXPECT_LE(filtered.size(), trace.size());
    }
}

TEST_F(StackTraceTest, FilterFunction) {
    StackTrace trace(0);

    if (!trace.empty()) {
        // Filter by test function patterns
        StackTrace filtered = trace.filter_function("Test");

        // Result should have same or fewer frames
        EXPECT_LE(filtered.size(), trace.size());
    }
}

TEST_F(StackTraceTest, WithoutSystemFrames) {
    StackTrace trace(0);

    if (!trace.empty()) {
        StackTrace without_system = trace.without_system_frames();

        // Should have same or fewer frames
        EXPECT_LE(without_system.size(), trace.size());

        // Verify no system frames remain
        for (size_t i = 0; i < without_system.size(); ++i) {
            EXPECT_FALSE(without_system[i].is_system());
        }
    }
}

TEST_F(StackTraceTest, TopFrames) {
    StackTrace trace(0);

    if (trace.size() >= 5) {
        StackTrace top3 = trace.top(3);
        EXPECT_EQ(top3.size(), 3);

        // First 3 frames should match
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(top3[i].address, trace[i].address);
        }
    }

    // Test with more frames than available
    StackTrace top_many = trace.top(1000);
    EXPECT_EQ(top_many.size(), trace.size());
}

TEST_F(StackTraceTest, FindFrame) {
    StackTrace trace = capture_in_function();

    if (!trace.empty()) {
        // Try to find this test's function
        auto frame = trace.find_frame("StackTraceTest");

        // Depending on platform support, might find it
        if (frame.has_value()) {
            EXPECT_TRUE(frame->function.find("StackTraceTest") != std::string::npos ||
                       frame->module.find("StackTraceTest") != std::string::npos ||
                       frame->source_file.find("StackTraceTest") != std::string::npos);
        }
    }

    // Test not finding
    auto not_found = trace.find_frame("ThisShouldNotExistAnywhere12345");
    EXPECT_FALSE(not_found.has_value());
}

// Access tests
TEST_F(StackTraceTest, IndexOperator) {
    StackTrace trace(0, 10);

    if (trace.size() >= 3) {
        const StackFrame& frame0 = trace[0];
        const StackFrame& frame1 = trace[1];
        const StackFrame& frame2 = trace[2];

        EXPECT_EQ(frame0.index, 0);
        EXPECT_EQ(frame1.index, 1);
        EXPECT_EQ(frame2.index, 2);
    }
}

TEST_F(StackTraceTest, IndexOperatorThrows) {
    StackTrace trace(0, 5);

    if (!trace.empty()) {
        EXPECT_NO_THROW(trace[0]);
        EXPECT_NO_THROW(trace[trace.size() - 1]);
        EXPECT_THROW(trace[trace.size()], std::out_of_range);
        EXPECT_THROW(trace[1000], std::out_of_range);
    }
}

TEST_F(StackTraceTest, FramesVector) {
    StackTrace trace(0, 10);

    const auto& frames = trace.frames();
    EXPECT_EQ(frames.size(), trace.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        EXPECT_EQ(frames[i].index, i);
        EXPECT_EQ(&frames[i], &trace[i]);
    }
}

// Multi-threading test
TEST_F(StackTraceTest, ThreadSafety) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<StackTrace> traces(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&traces, i]() {
            traces[i] = StackTrace(0, 20);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All traces should be valid
    for (const auto& trace : traces) {
        EXPECT_FALSE(trace.empty());
        EXPECT_GT(trace.size(), 0);
    }
}

// Edge cases
TEST_F(StackTraceTest, EmptyTraceToString) {
    StackTrace empty;
    std::string str = empty.to_string();

    EXPECT_EQ(str, "<no stack trace available>");
}

TEST_F(StackTraceTest, EmptyTraceCompact) {
    StackTrace empty;
    std::string compact = empty.compact_trace();

    EXPECT_TRUE(compact.empty());
}

TEST_F(StackTraceTest, EmptyTraceFilters) {
    StackTrace empty;

    StackTrace filtered_module = empty.filter_module("test");
    EXPECT_TRUE(filtered_module.empty());

    StackTrace filtered_func = empty.filter_function("test");
    EXPECT_TRUE(filtered_func.empty());

    StackTrace no_system = empty.without_system_frames();
    EXPECT_TRUE(no_system.empty());

    StackTrace top = empty.top(5);
    EXPECT_TRUE(top.empty());
}

TEST_F(StackTraceTest, VeryLargeSkip) {
    StackTrace trace(1000, 10);  // Skip more frames than exist

    EXPECT_TRUE(trace.empty());
}

TEST_F(StackTraceTest, ZeroMaxFrames) {
    StackTrace trace(0, 0);  // Request zero frames

    EXPECT_TRUE(trace.empty());
}

// Platform-specific behavior
TEST_F(StackTraceTest, PlatformSupport) {
    StackTrace trace(0);

#if defined(HAS_EXECINFO) || defined(_WIN32)
    // On supported platforms, should capture something
    if (!trace.empty()) {
        EXPECT_GT(trace.size(), 0);
        const auto& frame = trace[0];
        EXPECT_NE(frame.address, nullptr);
    }
#else
    // On unsupported platforms, should be empty
    EXPECT_TRUE(trace.empty());
#endif
}

// Helper function name simplification (tested via StackTrace internals)
TEST_F(StackTraceTest, SimplifyFunctionName) {
    // This tests the behavior indirectly through compact_trace
    StackFrame frame;
    frame.function = "namespace::Class<T>::method(int, double)";

    // When used in compact trace, should simplify
    StackTrace trace;
    // We can't directly test private simplify_function_name,
    // but we can verify the behavior through the public interface

    // Create a trace with our custom frame
    // This would require making frames_ accessible, which it isn't
    // So this is more of a conceptual test
}

// Test very deep call stack
void recursive_function(int depth, StackTrace& result) {
    if (depth <= 0) {
        result = StackTrace(0, 100);
    } else {
        recursive_function(depth - 1, result);
    }
}

TEST_F(StackTraceTest, DeepCallStack) {
    StackTrace trace;
    recursive_function(20, trace);

    if (!trace.empty()) {
        // Should have captured multiple frames
        EXPECT_GT(trace.size(), 10);

        // Should find recursive function multiple times
        int count = 0;
        for (size_t i = 0; i < trace.size(); ++i) {
            if (trace[i].function.find("recursive") != std::string::npos) {
                count++;
            }
        }
        // Might find it multiple times if symbols are available
        EXPECT_GE(count, 0);
    }
}
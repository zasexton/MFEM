#include <gtest/gtest.h>
#include <core/memory/circular_buffer.h>
#include <core/memory/ring_buffer.h>
#include <core/memory/memory_resource.h>
#include <type_traits>
#include <vector>
#include <string>

namespace fcm = fem::core::memory;

class CircularBufferTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test that circular_buffer is an alias for ring_buffer
TEST_F(CircularBufferTest, IsAliasForRingBuffer) {
    // Verify that circular_buffer is actually the same type as ring_buffer
    using CircularBufferType = fcm::circular_buffer<int>;
    using RingBufferType = fcm::ring_buffer<int>;

    bool same_type = std::is_same_v<CircularBufferType, RingBufferType>;
    EXPECT_TRUE(same_type);
}

// Test that circular_buffer works with different allocators
TEST_F(CircularBufferTest, AllocatorTemplateParameter) {
    using CircularBufferDefaultAlloc = fcm::circular_buffer<int>;
    using CircularBufferCustomAlloc = fcm::circular_buffer<int, fcm::polymorphic_allocator<int>>;
    using RingBufferDefaultAlloc = fcm::ring_buffer<int>;
    using RingBufferCustomAlloc = fcm::ring_buffer<int, fcm::polymorphic_allocator<int>>;

    // Verify the types match for both default and custom allocators
    EXPECT_TRUE((std::is_same_v<CircularBufferDefaultAlloc, RingBufferDefaultAlloc>));
    EXPECT_TRUE((std::is_same_v<CircularBufferCustomAlloc, RingBufferCustomAlloc>));
}

// Basic functionality test to ensure circular_buffer works as expected
TEST_F(CircularBufferTest, BasicFunctionality) {
    fcm::circular_buffer<int> buffer(5);

    // Test basic operations
    EXPECT_EQ(buffer.capacity(), 5);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());

    // Push some elements
    EXPECT_TRUE(buffer.push(1));
    EXPECT_TRUE(buffer.push(2));
    EXPECT_TRUE(buffer.push(3));

    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.front(), 1);

    // Pop an element
    EXPECT_TRUE(buffer.pop());
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_EQ(buffer.front(), 2);

    // Test circular behavior
    EXPECT_TRUE(buffer.push(4));
    EXPECT_TRUE(buffer.push(5));
    EXPECT_TRUE(buffer.push(6));
    EXPECT_TRUE(buffer.full());

    // Should fail when full
    EXPECT_FALSE(buffer.push(7));
}

// Test with complex types
TEST_F(CircularBufferTest, StringType) {
    fcm::circular_buffer<std::string> buffer(3);

    buffer.emplace("hello");
    buffer.emplace("world");
    buffer.emplace("test");

    EXPECT_EQ(buffer.front(), "hello");
    EXPECT_EQ(buffer[0], "hello");
    EXPECT_EQ(buffer[1], "world");
    EXPECT_EQ(buffer[2], "test");

    buffer.pop();
    buffer.emplace("new");

    EXPECT_EQ(buffer[0], "world");
    EXPECT_EQ(buffer[1], "test");
    EXPECT_EQ(buffer[2], "new");
}

// Test with custom allocator
TEST_F(CircularBufferTest, CustomAllocator) {
    auto mr = fcm::default_resource();
    fcm::circular_buffer<int> buffer(10, mr);

    EXPECT_EQ(buffer.capacity(), 10);

    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(buffer.push(i * 10));
    }

    EXPECT_TRUE(buffer.full());

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(buffer[i], i * 10);
    }
}

// Test copy and move semantics
TEST_F(CircularBufferTest, CopySemantics) {
    fcm::circular_buffer<int> original(5);
    original.push(1);
    original.push(2);
    original.push(3);

    // Copy constructor
    fcm::circular_buffer<int> copy(original);
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.capacity(), original.capacity());
    for (size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }

    // Copy assignment
    fcm::circular_buffer<int> assigned(3);
    assigned = original;
    EXPECT_EQ(assigned.size(), original.size());
    EXPECT_EQ(assigned.capacity(), original.capacity());
}

TEST_F(CircularBufferTest, MoveSemantics) {
    fcm::circular_buffer<int> original(5);
    original.push(10);
    original.push(20);
    original.push(30);

    size_t orig_size = original.size();
    size_t orig_cap = original.capacity();

    // Move constructor
    fcm::circular_buffer<int> moved(std::move(original));
    EXPECT_EQ(moved.size(), orig_size);
    EXPECT_EQ(moved.capacity(), orig_cap);
    EXPECT_EQ(moved[0], 10);
    EXPECT_EQ(moved[1], 20);
    EXPECT_EQ(moved[2], 30);

    // Original should be empty after move
    EXPECT_EQ(original.size(), 0);
    EXPECT_EQ(original.capacity(), 0);
}

// Test iteration
TEST_F(CircularBufferTest, ForEach) {
    fcm::circular_buffer<int> buffer(5);
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    buffer.push(4);

    std::vector<int> collected;
    buffer.for_each([&](int value) {
        collected.push_back(value);
    });

    EXPECT_EQ(collected, std::vector<int>({1, 2, 3, 4}));
}

// Test clear operation
TEST_F(CircularBufferTest, Clear) {
    fcm::circular_buffer<int> buffer(5);
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);

    buffer.clear();

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 5); // Capacity unchanged
}

// Test pop with value extraction
TEST_F(CircularBufferTest, PopWithValue) {
    fcm::circular_buffer<int> buffer(3);
    buffer.push(42);
    buffer.push(84);

    int value;
    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 42);

    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 84);

    EXPECT_FALSE(buffer.pop(value)); // Empty
}

// Test wrap-around behavior
TEST_F(CircularBufferTest, WrapAround) {
    fcm::circular_buffer<int> buffer(3);

    // Fill buffer
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    EXPECT_TRUE(buffer.full());

    // Pop and push to cause wrap
    buffer.pop();
    buffer.push(4);

    // Check order
    EXPECT_EQ(buffer[0], 2);
    EXPECT_EQ(buffer[1], 3);
    EXPECT_EQ(buffer[2], 4);

    // More wrap-around
    buffer.pop();
    buffer.pop();
    buffer.push(5);
    buffer.push(6);

    EXPECT_EQ(buffer[0], 4);
    EXPECT_EQ(buffer[1], 5);
    EXPECT_EQ(buffer[2], 6);
}

// Stress test with many operations
TEST_F(CircularBufferTest, StressTest) {
    fcm::circular_buffer<int> buffer(100);

    // Fill and empty multiple times
    for (int cycle = 0; cycle < 10; ++cycle) {
        // Fill buffer
        for (int i = 0; i < 100; ++i) {
            EXPECT_TRUE(buffer.push(cycle * 1000 + i));
        }
        EXPECT_TRUE(buffer.full());

        // Partial empty and refill
        for (int i = 0; i < 50; ++i) {
            EXPECT_TRUE(buffer.pop());
        }

        for (int i = 0; i < 50; ++i) {
            EXPECT_TRUE(buffer.push(cycle * 1000 + 100 + i));
        }

        // Verify contents
        for (int i = 0; i < 100; ++i) {
            int expected = (i < 50) ? (cycle * 1000 + 50 + i) : (cycle * 1000 + 100 + i - 50);
            EXPECT_EQ(buffer[i], expected);
        }

        // Empty completely
        while (!buffer.empty()) {
            buffer.pop();
        }
    }
}

#if CORE_MEMORY_ENABLE_TELEMETRY
// Test telemetry functionality
TEST_F(CircularBufferTest, Telemetry) {
    fcm::circular_buffer<int> buffer(3);

    const auto& telemetry = buffer.telemetry();
    EXPECT_EQ(telemetry.pushes, 0);
    EXPECT_EQ(telemetry.pops, 0);
    EXPECT_EQ(telemetry.drops, 0);

    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    EXPECT_EQ(telemetry.pushes, 3);

    // Try to push when full
    EXPECT_FALSE(buffer.push(4));
    EXPECT_EQ(telemetry.drops, 1);

    buffer.pop();
    EXPECT_EQ(telemetry.pops, 1);

    EXPECT_EQ(telemetry.peak_size, 3);
}
#endif
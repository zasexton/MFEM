#include <gtest/gtest.h>
#include <core/memory/ring_buffer.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <string>
#include <memory>

namespace fcm = fem::core::memory;

class RingBufferTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test helper class for tracking construction/destruction
class TestObject {
public:
    static int construct_count;
    static int destruct_count;
    static int copy_count;
    static int move_count;

    int value;
    bool should_throw_on_copy;
    bool should_throw_on_move;

    explicit TestObject(int v = 0, bool throw_copy = false, bool throw_move = false)
        : value(v), should_throw_on_copy(throw_copy), should_throw_on_move(throw_move) {
        ++construct_count;
    }

    TestObject(const TestObject& other)
        : value(other.value), should_throw_on_copy(other.should_throw_on_copy), should_throw_on_move(other.should_throw_on_move) {
        if (should_throw_on_copy) throw std::runtime_error("Copy constructor exception");
        ++copy_count;
    }

    TestObject(TestObject&& other)
        : value(other.value), should_throw_on_copy(other.should_throw_on_copy), should_throw_on_move(other.should_throw_on_move) {
        if (should_throw_on_move) throw std::runtime_error("Move constructor exception");
        other.value = -1;
        ++move_count;
    }

    TestObject& operator=(const TestObject& other) {
        if (this == &other) return *this;
        if (should_throw_on_copy) throw std::runtime_error("Copy assignment exception");
        value = other.value;
        should_throw_on_copy = other.should_throw_on_copy;
        should_throw_on_move = other.should_throw_on_move;
        return *this;
    }

    TestObject& operator=(TestObject&& other) {
        if (this == &other) return *this;
        if (should_throw_on_move) throw std::runtime_error("Move assignment exception");
        value = other.value;
        should_throw_on_copy = other.should_throw_on_copy;
        should_throw_on_move = other.should_throw_on_move;
        other.value = -1;
        return *this;
    }

    ~TestObject() { ++destruct_count; }

    static void reset_counters() {
        construct_count = destruct_count = copy_count = move_count = 0;
    }
};

int TestObject::construct_count = 0;
int TestObject::destruct_count = 0;
int TestObject::copy_count = 0;
int TestObject::move_count = 0;

// Move-only type for testing
class MoveOnlyType {
public:
    int value;

    explicit MoveOnlyType(int v) : value(v) {}
    MoveOnlyType(const MoveOnlyType&) = delete;
    MoveOnlyType& operator=(const MoveOnlyType&) = delete;

    MoveOnlyType(MoveOnlyType&& other) noexcept : value(other.value) {
        other.value = -1;
    }

    MoveOnlyType& operator=(MoveOnlyType&& other) noexcept {
        if (this == &other) return *this;
        value = other.value;
        other.value = -1;
        return *this;
    }
};

// === Basic Construction Tests ===

TEST_F(RingBufferTest, ConstructionWithCapacity) {
    fcm::ring_buffer<int> buffer(5);

    EXPECT_EQ(buffer.capacity(), 5);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
}

TEST_F(RingBufferTest, ConstructionWithMemoryResource) {
    auto mr = fcm::default_resource();
    fcm::ring_buffer<int> buffer(10, mr);

    EXPECT_EQ(buffer.capacity(), 10);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(RingBufferTest, ConstructionWithAllocator) {
    fcm::polymorphic_allocator<int> alloc(fcm::default_resource());
    fcm::ring_buffer<int> buffer(8, alloc);

    EXPECT_EQ(buffer.capacity(), 8);
    EXPECT_EQ(buffer.size(), 0);
}

// Note: Zero capacity construction triggers assertion in debug builds
// This is implementation-defined behavior so we skip testing it

// === Basic Operations Tests ===

TEST_F(RingBufferTest, PushAndPopBasic) {
    fcm::ring_buffer<int> buffer(3);

    EXPECT_TRUE(buffer.push(1));
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.front(), 1);

    EXPECT_TRUE(buffer.push(2));
    EXPECT_TRUE(buffer.push(3));
    EXPECT_TRUE(buffer.full());
    EXPECT_EQ(buffer.size(), 3);

    EXPECT_FALSE(buffer.push(4)); // Should fail when full
    EXPECT_EQ(buffer.size(), 3);
}

TEST_F(RingBufferTest, PopFromBuffer) {
    fcm::ring_buffer<int> buffer(3);
    buffer.push(10);
    buffer.push(20);
    buffer.push(30);

    EXPECT_EQ(buffer.front(), 10);
    EXPECT_TRUE(buffer.pop());
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_EQ(buffer.front(), 20);

    EXPECT_TRUE(buffer.pop());
    EXPECT_EQ(buffer.front(), 30);

    EXPECT_TRUE(buffer.pop());
    EXPECT_TRUE(buffer.empty());

    EXPECT_FALSE(buffer.pop()); // Should fail when empty
}

TEST_F(RingBufferTest, PopWithValueExtraction) {
    fcm::ring_buffer<int> buffer(3);
    buffer.push(42);
    buffer.push(84);

    int value;
    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 42);
    EXPECT_EQ(buffer.size(), 1);

    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 84);
    EXPECT_TRUE(buffer.empty());

    EXPECT_FALSE(buffer.pop(value)); // Should fail when empty
}

TEST_F(RingBufferTest, EmplaceBack) {
    fcm::ring_buffer<std::string> buffer(3);

    EXPECT_TRUE(buffer.emplace("hello"));
    EXPECT_TRUE(buffer.emplace("world"));
    EXPECT_TRUE(buffer.emplace("test"));

    EXPECT_EQ(buffer.front(), "hello");
    buffer.pop();
    EXPECT_EQ(buffer.front(), "world");
    buffer.pop();
    EXPECT_EQ(buffer.front(), "test");
}

// === Circular Buffer Mechanics Tests ===

TEST_F(RingBufferTest, CircularBufferWrapAround) {
    fcm::ring_buffer<int> buffer(3);

    // Fill buffer
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);

    // Pop one element
    EXPECT_TRUE(buffer.pop());
    EXPECT_EQ(buffer.size(), 2);

    // Push another element (should wrap around)
    EXPECT_TRUE(buffer.push(4));
    EXPECT_EQ(buffer.size(), 3);

    // Verify order: 2, 3, 4
    EXPECT_EQ(buffer.front(), 2);
    buffer.pop();
    EXPECT_EQ(buffer.front(), 3);
    buffer.pop();
    EXPECT_EQ(buffer.front(), 4);
}

TEST_F(RingBufferTest, MultipleWrapArounds) {
    fcm::ring_buffer<int> buffer(3);

    // Multiple cycles of fill, partial empty, refill
    for (int cycle = 0; cycle < 5; ++cycle) {
        buffer.push(cycle * 10 + 1);
        buffer.push(cycle * 10 + 2);
        buffer.push(cycle * 10 + 3);

        EXPECT_EQ(buffer.front(), cycle * 10 + 1);
        buffer.pop();
        EXPECT_EQ(buffer.front(), cycle * 10 + 2);
        buffer.pop();
        EXPECT_EQ(buffer.front(), cycle * 10 + 3);
        buffer.pop();

        EXPECT_TRUE(buffer.empty());
    }
}

TEST_F(RingBufferTest, IndexingAccess) {
    fcm::ring_buffer<int> buffer(5);

    buffer.push(10);
    buffer.push(20);
    buffer.push(30);

    EXPECT_EQ(buffer[0], 10); // front
    EXPECT_EQ(buffer[1], 20);
    EXPECT_EQ(buffer[2], 30); // back

    // Pop front and add new element
    buffer.pop();
    buffer.push(40);

    EXPECT_EQ(buffer[0], 20); // new front
    EXPECT_EQ(buffer[1], 30);
    EXPECT_EQ(buffer[2], 40); // new back
}

TEST_F(RingBufferTest, ForEachIteration) {
    fcm::ring_buffer<int> buffer(5);
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);

    std::vector<int> collected;
    buffer.for_each([&](int value) { collected.push_back(value); });

    EXPECT_EQ(collected, std::vector<int>({1, 2, 3}));

    // Test after wrap-around
    buffer.pop(); // Remove 1
    buffer.push(4);

    collected.clear();
    buffer.for_each([&](int value) { collected.push_back(value); });
    EXPECT_EQ(collected, std::vector<int>({2, 3, 4}));
}

// === Copy and Move Semantics Tests ===

TEST_F(RingBufferTest, CopyConstructor) {
    fcm::ring_buffer<int> original(5);
    original.push(10);
    original.push(20);
    original.push(30);

    fcm::ring_buffer<int> copy(original);

    EXPECT_EQ(copy.capacity(), original.capacity());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy[0], 10);
    EXPECT_EQ(copy[1], 20);
    EXPECT_EQ(copy[2], 30);

    // Modifications to copy shouldn't affect original
    copy.push(40);
    EXPECT_EQ(copy.size(), 4);
    EXPECT_EQ(original.size(), 3);
}

TEST_F(RingBufferTest, MoveConstructor) {
    fcm::ring_buffer<int> original(5);
    original.push(10);
    original.push(20);
    original.push(30);

    fcm::ring_buffer<int> moved(std::move(original));

    EXPECT_EQ(moved.capacity(), 5);
    EXPECT_EQ(moved.size(), 3);
    EXPECT_EQ(moved[0], 10);
    EXPECT_EQ(moved[1], 20);
    EXPECT_EQ(moved[2], 30);

    // Original should be in valid but unspecified state
    EXPECT_EQ(original.capacity(), 0);
    EXPECT_EQ(original.size(), 0);
}

TEST_F(RingBufferTest, CopyAssignment) {
    fcm::ring_buffer<int> original(5);
    original.push(10);
    original.push(20);

    fcm::ring_buffer<int> copy(3);
    copy.push(99);

    copy = original;

    EXPECT_EQ(copy.capacity(), 5);
    EXPECT_EQ(copy.size(), 2);
    EXPECT_EQ(copy[0], 10);
    EXPECT_EQ(copy[1], 20);
}

TEST_F(RingBufferTest, MoveAssignment) {
    fcm::ring_buffer<int> original(5);
    original.push(10);
    original.push(20);

    fcm::ring_buffer<int> moved(3);
    moved.push(99);

    moved = std::move(original);

    EXPECT_EQ(moved.capacity(), 5);
    EXPECT_EQ(moved.size(), 2);
    EXPECT_EQ(moved[0], 10);
    EXPECT_EQ(moved[1], 20);
}

TEST_F(RingBufferTest, SelfAssignment) {
    fcm::ring_buffer<int> buffer(3);
    buffer.push(1);
    buffer.push(2);

    buffer = buffer;

    EXPECT_EQ(buffer.capacity(), 3);
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
}

// === Clear and Reset Tests ===

TEST_F(RingBufferTest, ClearBuffer) {
    TestObject::reset_counters();

    fcm::ring_buffer<TestObject> buffer(3);
    buffer.emplace(1);
    buffer.emplace(2);
    buffer.emplace(3);

    int objects_created = TestObject::construct_count;
    buffer.clear();

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 3); // Capacity should remain unchanged
    EXPECT_EQ(TestObject::destruct_count, objects_created); // All objects should be destroyed
}

// === Exception Safety Tests ===

TEST_F(RingBufferTest, ExceptionSafetyInPush) {
    TestObject::reset_counters();

    fcm::ring_buffer<TestObject> buffer(3);
    buffer.emplace(1);
    buffer.emplace(2);

    // Try to add an object that throws on copy
    try {
        TestObject throwing_obj(3, true); // throws on copy
        buffer.push(throwing_obj);
        FAIL() << "Expected exception";
    } catch (const std::runtime_error&) {
        // Exception expected
    }

    // Buffer should remain in valid state
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_EQ(buffer[0].value, 1);
    EXPECT_EQ(buffer[1].value, 2);
}

TEST_F(RingBufferTest, ExceptionSafetyInEmplace) {
    fcm::ring_buffer<TestObject> buffer(3);
    buffer.emplace(1);
    buffer.emplace(2);

    // Try to emplace an object that throws during construction
    try {
        buffer.emplace(3, true); // throws on copy (but this is direct construction)
        // This should succeed since emplace constructs in-place
        EXPECT_EQ(buffer.size(), 3);
    } catch (...) {
        // If it throws, buffer should still be in valid state
        EXPECT_EQ(buffer.size(), 2);
    }
}

// === Move-Only Type Tests ===

TEST_F(RingBufferTest, MoveOnlyType) {
    fcm::ring_buffer<MoveOnlyType> buffer(3);

    EXPECT_TRUE(buffer.emplace(10));
    EXPECT_TRUE(buffer.emplace(20));
    EXPECT_TRUE(buffer.emplace(30));

    EXPECT_EQ(buffer.front().value, 10);
    buffer.pop();
    EXPECT_EQ(buffer.front().value, 20);

    // Test move semantics
    MoveOnlyType obj(99);
    EXPECT_TRUE(buffer.push(std::move(obj)));
    EXPECT_EQ(obj.value, -1); // Should be moved from
}

// === Telemetry Tests ===

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(RingBufferTest, TelemetryTracking) {
    fcm::ring_buffer<int> buffer(3);

    auto& telemetry = buffer.telemetry();
    EXPECT_EQ(telemetry.pushes, 0);
    EXPECT_EQ(telemetry.pops, 0);
    EXPECT_EQ(telemetry.drops, 0);
    EXPECT_EQ(telemetry.heap_allocs, 1); // Constructor allocates storage
    EXPECT_EQ(telemetry.capacity, 3);

    buffer.push(1);
    buffer.push(2);
    EXPECT_EQ(telemetry.pushes, 2);
    EXPECT_EQ(telemetry.peak_size, 2);

    buffer.pop();
    EXPECT_EQ(telemetry.pops, 1);

    // Fill buffer and try to push one more (should drop)
    buffer.push(3);
    buffer.push(4);
    EXPECT_FALSE(buffer.push(5)); // Should fail
    EXPECT_EQ(telemetry.drops, 1);
}

TEST_F(RingBufferTest, TelemetryCallback) {
    fcm::ring_buffer<int> buffer(2);

    std::vector<std::string> events;
    buffer.set_telemetry_callback([&](const char* event, const auto& /*telemetry*/) {
        events.push_back(std::string(event));
    });

    buffer.push(1);
    buffer.push(2);
    EXPECT_FALSE(buffer.push(3)); // Should trigger drop event

    // Should have received "drop" event
    EXPECT_TRUE(std::find(events.begin(), events.end(), "drop") != events.end());
}
#endif

// === Edge Cases and Stress Tests ===

TEST_F(RingBufferTest, SingleElementBuffer) {
    fcm::ring_buffer<int> buffer(1);

    EXPECT_TRUE(buffer.push(42));
    EXPECT_TRUE(buffer.full());
    EXPECT_FALSE(buffer.push(99)); // Should fail

    EXPECT_EQ(buffer.front(), 42);
    EXPECT_TRUE(buffer.pop());
    EXPECT_TRUE(buffer.empty());
}

TEST_F(RingBufferTest, LargeCapacityBuffer) {
    fcm::ring_buffer<int> buffer(1000);

    // Fill buffer completely
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(buffer.push(i));
    }
    EXPECT_TRUE(buffer.full());
    EXPECT_FALSE(buffer.push(1000)); // Should fail

    // Verify all elements in correct order
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(buffer[i], i);
    }

    // Empty buffer
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(buffer.front(), i);
        EXPECT_TRUE(buffer.pop());
    }
    EXPECT_TRUE(buffer.empty());
}

TEST_F(RingBufferTest, InterleavedOperations) {
    fcm::ring_buffer<int> buffer(5);

    // Complex pattern of push/pop operations
    buffer.push(1);
    buffer.push(2);
    buffer.pop(); // Remove 1
    buffer.push(3);
    buffer.push(4);
    buffer.pop(); // Remove 2
    buffer.pop(); // Remove 3
    buffer.push(5);
    buffer.push(6);
    buffer.push(7);

    // Should have: 4, 5, 6, 7
    EXPECT_EQ(buffer.size(), 4);
    EXPECT_EQ(buffer[0], 4);
    EXPECT_EQ(buffer[1], 5);
    EXPECT_EQ(buffer[2], 6);
    EXPECT_EQ(buffer[3], 7);
}

TEST_F(RingBufferTest, DestructorCallsElementDestructors) {
    TestObject::reset_counters();

    {
        fcm::ring_buffer<TestObject> buffer(3);
        buffer.emplace(1);
        buffer.emplace(2);
        buffer.emplace(3);

        EXPECT_EQ(TestObject::construct_count, 3);
        // Destructor should be called automatically
    }

    EXPECT_EQ(TestObject::destruct_count, 3);
}

// === Complex Type Tests ===

TEST_F(RingBufferTest, StringOperations) {
    fcm::ring_buffer<std::string> buffer(3);

    buffer.push("hello");
    buffer.push("world");
    buffer.push("test");

    EXPECT_EQ(buffer.front(), "hello");
    buffer.pop();

    buffer.push("replacement");

    EXPECT_EQ(buffer[0], "world");
    EXPECT_EQ(buffer[1], "test");
    EXPECT_EQ(buffer[2], "replacement");
}

TEST_F(RingBufferTest, UniquePointerOperations) {
    fcm::ring_buffer<std::unique_ptr<int>> buffer(3);

    buffer.push(std::make_unique<int>(10));
    buffer.push(std::make_unique<int>(20));
    buffer.push(std::make_unique<int>(30));

    EXPECT_EQ(*buffer.front(), 10);

    auto ptr = std::move(buffer.front());
    buffer.pop();

    EXPECT_EQ(*ptr, 10);
    EXPECT_EQ(*buffer.front(), 20);
}

// === Performance-Related Tests ===

TEST_F(RingBufferTest, NoUnnecessaryMoves) {
    TestObject::reset_counters();

    fcm::ring_buffer<TestObject> buffer(3);

    TestObject obj1(1);
    TestObject obj2(2);

    int moves_before = TestObject::move_count;
    buffer.push(std::move(obj1));
    buffer.push(std::move(obj2));
    int moves_after = TestObject::move_count;

    // Should have exactly 2 moves (one for each push)
    EXPECT_EQ(moves_after - moves_before, 2);
}

TEST_F(RingBufferTest, EmplaceVsPushPerformance) {
    TestObject::reset_counters();

    fcm::ring_buffer<TestObject> buffer(10);

    // Test emplace (should construct in-place)
    int constructs_before = TestObject::construct_count;
    buffer.emplace(42);
    int constructs_after = TestObject::construct_count;

    EXPECT_EQ(constructs_after - constructs_before, 1); // Only direct construction
    EXPECT_EQ(TestObject::copy_count, 0); // No copies
    EXPECT_EQ(TestObject::move_count, 0); // No moves
}
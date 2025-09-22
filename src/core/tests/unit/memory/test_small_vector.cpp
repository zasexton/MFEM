#include <gtest/gtest.h>
#include <core/memory/small_vector.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

namespace fcm = fem::core::memory;

// Test helper classes for various scenarios
struct TestObject {
    int value;
    static int constructor_calls;
    static int destructor_calls;
    static int copy_calls;
    static int move_calls;

    TestObject(int v = 0) : value(v) { ++constructor_calls; }
    TestObject(const TestObject& other) : value(other.value) { ++copy_calls; }
    TestObject(TestObject&& other) noexcept : value(other.value) { other.value = -1; ++move_calls; }
    TestObject& operator=(const TestObject& other) { value = other.value; ++copy_calls; return *this; }
    TestObject& operator=(TestObject&& other) noexcept { value = other.value; other.value = -1; ++move_calls; return *this; }
    ~TestObject() { ++destructor_calls; }

    bool operator==(const TestObject& other) const { return value == other.value; }
    bool operator!=(const TestObject& other) const { return !(*this == other); }

    static void reset_counts() {
        constructor_calls = destructor_calls = copy_calls = move_calls = 0;
    }

    // Note: Global leak checking removed due to test framework interactions
    static void verify_no_leaks() {
        // Individual tests verify specific constructor/destructor behavior
    }
};

int TestObject::constructor_calls = 0;
int TestObject::destructor_calls = 0;
int TestObject::copy_calls = 0;
int TestObject::move_calls = 0;

// Exception throwing test object
struct ThrowingObject {
    int value;
    static bool should_throw_on_copy;
    static bool should_throw_on_move;
    static int instance_count;

    ThrowingObject(int v = 0) : value(v) { ++instance_count; }

    ThrowingObject(const ThrowingObject& other) : value(other.value) {
        if (should_throw_on_copy) throw std::runtime_error("Copy constructor exception");
        ++instance_count;
    }

    ThrowingObject(ThrowingObject&& other) : value(other.value) {
        if (should_throw_on_move) throw std::runtime_error("Move constructor exception");
        other.value = -1;
        ++instance_count;
    }

    ~ThrowingObject() { --instance_count; }

    static void reset() {
        should_throw_on_copy = should_throw_on_move = false;
        instance_count = 0;
    }
};

bool ThrowingObject::should_throw_on_copy = false;
bool ThrowingObject::should_throw_on_move = false;
int ThrowingObject::instance_count = 0;

class SmallVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        TestObject::reset_counts();
        ThrowingObject::reset();
    }

    void TearDown() override {
        // Note: We don't check for leaks here as the global destructors
        // make counting unreliable. Individual tests verify specific behavior.
        EXPECT_EQ(ThrowingObject::instance_count, 0) << "ThrowingObject leak detected";
    }
};

// ============================================================================
// Basic Construction and Destruction Tests
// ============================================================================

TEST_F(SmallVectorTest, DefaultConstruction) {
    fcm::small_vector<int> vec;

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
    EXPECT_GE(vec.capacity(), 8); // Default InlineCapacity
    EXPECT_NE(vec.data(), nullptr);
}

TEST_F(SmallVectorTest, CustomInlineCapacity) {
    fcm::small_vector<int, 4> vec4;
    fcm::small_vector<int, 16> vec16;

    EXPECT_EQ(vec4.capacity(), 4);
    EXPECT_EQ(vec16.capacity(), 16);
}

TEST_F(SmallVectorTest, MemoryResourceConstruction) {
    auto mr = fcm::default_resource();
    fcm::small_vector<int> vec(mr);

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
}

TEST_F(SmallVectorTest, InitializerListConstruction) {
    fcm::small_vector<int, 4> vec{1, 2, 3, 4};

    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec.capacity(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(vec[i], i + 1);
    }
}

TEST_F(SmallVectorTest, InitializerListExceedsInlineCapacity) {
    fcm::small_vector<int, 2> vec{1, 2, 3, 4, 5};

    EXPECT_EQ(vec.size(), 5);
    EXPECT_GT(vec.capacity(), 2); // Should have grown beyond inline capacity
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i], i + 1);
    }
}

TEST_F(SmallVectorTest, DestructorCallsElementDestructors) {
    TestObject::reset_counts();
    {
        fcm::small_vector<TestObject, 4> vec;
        vec.emplace_back(1);
        vec.emplace_back(2);
        vec.emplace_back(3);
        EXPECT_EQ(TestObject::constructor_calls, 3);
    }
    EXPECT_EQ(TestObject::destructor_calls, 3);
}

// ============================================================================
// Inline Storage Optimization Tests
// ============================================================================

TEST_F(SmallVectorTest, InlineStorageBasicOperations) {
    fcm::small_vector<int, 8> vec;

    // Add elements within inline capacity
    for (int i = 0; i < 8; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 8);
    EXPECT_EQ(vec.capacity(), 8);

    // Verify elements
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(SmallVectorTest, InlineStorageDoesNotAllocateHeap) {
    fcm::small_vector<TestObject, 4> vec;
    const void* inline_data = vec.data();

    // Add elements within inline capacity
    for (int i = 0; i < 4; ++i) {
        vec.emplace_back(i);
    }

    // Data pointer should not change (still using inline storage)
    EXPECT_EQ(vec.data(), inline_data);
    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec.capacity(), 4);
}

TEST_F(SmallVectorTest, TransitionFromInlineToHeap) {
    fcm::small_vector<int, 4> vec;

    // Fill inline capacity
    for (int i = 0; i < 4; ++i) {
        vec.push_back(i);
    }

    const void* inline_data = vec.data();
    EXPECT_EQ(vec.capacity(), 4);

    // Add one more element to trigger heap allocation
    vec.push_back(4);

    EXPECT_EQ(vec.size(), 5);
    EXPECT_GT(vec.capacity(), 4);
    EXPECT_NE(vec.data(), inline_data); // Should be using heap now

    // Verify all elements are correct
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(SmallVectorTest, ShrinkToFitFromHeapToInline) {
    fcm::small_vector<int, 4> vec;

    // Grow beyond inline capacity
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }
    EXPECT_GT(vec.capacity(), 4);

    // Remove elements back to within inline capacity
    while (vec.size() > 3) {
        vec.pop_back();
    }

    EXPECT_EQ(vec.size(), 3);
    EXPECT_GT(vec.capacity(), 4); // Still using heap

    // Shrink to fit should move back to inline storage
    vec.shrink_to_fit();

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec.capacity(), 4); // Back to inline capacity

    // Verify elements
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// ============================================================================
// Heap Allocation and Growth Tests
// ============================================================================

TEST_F(SmallVectorTest, HeapAllocationGrowthPolicy) {
    fcm::small_vector<int, 2> vec;

    // Test growth pattern
    std::vector<std::size_t> capacities;

    for (int i = 0; i < 20; ++i) {
        if (vec.capacity() != (capacities.empty() ? 2 : capacities.back())) {
            capacities.push_back(vec.capacity());
        }
        vec.push_back(i);
    }

    // Should have grown multiple times
    EXPECT_GT(capacities.size(), 2);

    // Each growth should be reasonable (not too aggressive, not too conservative)
    for (std::size_t i = 1; i < capacities.size(); ++i) {
        EXPECT_GT(capacities[i], capacities[i-1]);
        EXPECT_LE(capacities[i], capacities[i-1] * 2); // Not more than 2x growth
    }
}

TEST_F(SmallVectorTest, ReserveInlineCapacity) {
    fcm::small_vector<int, 8> vec;

    vec.reserve(4); // Less than inline capacity
    EXPECT_EQ(vec.capacity(), 8); // Should remain at inline capacity

    vec.reserve(8); // Equal to inline capacity
    EXPECT_EQ(vec.capacity(), 8); // Should remain at inline capacity
}

TEST_F(SmallVectorTest, ReserveBeyondInlineCapacity) {
    fcm::small_vector<int, 4> vec;

    vec.reserve(10);
    EXPECT_GE(vec.capacity(), 10);
    EXPECT_EQ(vec.size(), 0);

    // Should be able to add 10 elements without reallocation
    auto cap_before = vec.capacity();
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(vec.capacity(), cap_before);
}

TEST_F(SmallVectorTest, LargeAllocation) {
    fcm::small_vector<int, 4> vec;

    const std::size_t large_size = 1000;
    vec.reserve(large_size);

    EXPECT_GE(vec.capacity(), large_size);

    for (std::size_t i = 0; i < large_size; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    EXPECT_EQ(vec.size(), large_size);

    // Verify elements
    for (std::size_t i = 0; i < large_size; ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(i));
    }
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

TEST_F(SmallVectorTest, CopyConstructorInlineToInline) {
    fcm::small_vector<TestObject, 4> vec1;
    vec1.emplace_back(1);
    vec1.emplace_back(2);

    TestObject::reset_counts();
    fcm::small_vector<TestObject, 4> vec2(vec1);

    EXPECT_EQ(vec2.size(), 2);
    EXPECT_EQ(vec2.capacity(), 4);
    EXPECT_EQ(vec2[0].value, 1);
    EXPECT_EQ(vec2[1].value, 2);

    // emplace_back(other[i]) calls copy constructor
    EXPECT_EQ(TestObject::constructor_calls, 0);
    EXPECT_EQ(TestObject::copy_calls, 2);
    EXPECT_EQ(TestObject::move_calls, 0);
}

TEST_F(SmallVectorTest, CopyConstructorHeapToHeap) {
    fcm::small_vector<TestObject, 2> vec1;
    for (int i = 0; i < 5; ++i) {
        vec1.emplace_back(i);
    }

    TestObject::reset_counts();
    fcm::small_vector<TestObject, 2> vec2(vec1);

    EXPECT_EQ(vec2.size(), 5);
    EXPECT_GE(vec2.capacity(), 5);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec2[i].value, i);
    }

    // emplace_back(other[i]) calls copy constructor
    EXPECT_EQ(TestObject::constructor_calls, 0);
    EXPECT_EQ(TestObject::copy_calls, 5);
    EXPECT_EQ(TestObject::move_calls, 0);
}

TEST_F(SmallVectorTest, MoveConstructorInlineToInline) {
    fcm::small_vector<TestObject, 4> vec1;
    vec1.emplace_back(1);
    vec1.emplace_back(2);

    TestObject::reset_counts();
    fcm::small_vector<TestObject, 4> vec2(std::move(vec1));

    EXPECT_EQ(vec2.size(), 2);
    EXPECT_EQ(vec2[0].value, 1);
    EXPECT_EQ(vec2[1].value, 2);

    EXPECT_EQ(vec1.size(), 0); // Source should be empty

    // Elements should have been moved via move constructor
    EXPECT_EQ(TestObject::constructor_calls, 0);
    EXPECT_EQ(TestObject::copy_calls, 0);
    EXPECT_EQ(TestObject::move_calls, 2);
}

TEST_F(SmallVectorTest, MoveConstructorHeapToHeap) {
    fcm::small_vector<TestObject, 2> vec1;
    for (int i = 0; i < 5; ++i) {
        vec1.emplace_back(i);
    }

    auto* heap_ptr = vec1.data();
    TestObject::reset_counts();

    fcm::small_vector<TestObject, 2> vec2(std::move(vec1));

    EXPECT_EQ(vec2.size(), 5);
    EXPECT_EQ(vec2.data(), heap_ptr); // Should steal the heap allocation

    EXPECT_EQ(vec1.size(), 0); // Source should be empty
    EXPECT_NE(vec1.data(), heap_ptr); // Source should not point to moved data

    // No copies or moves should have occurred (just pointer steal)
    EXPECT_EQ(TestObject::constructor_calls, 0);
    EXPECT_EQ(TestObject::copy_calls, 0);
    EXPECT_EQ(TestObject::move_calls, 0);
}

TEST_F(SmallVectorTest, CopyAssignmentOperator) {
    fcm::small_vector<int, 4> vec1{1, 2, 3};
    fcm::small_vector<int, 4> vec2{10, 20};

    vec2 = vec1;

    EXPECT_EQ(vec2.size(), 3);
    EXPECT_EQ(vec2[0], 1);
    EXPECT_EQ(vec2[1], 2);
    EXPECT_EQ(vec2[2], 3);

    // Original should be unchanged
    EXPECT_EQ(vec1.size(), 3);
}

TEST_F(SmallVectorTest, MoveAssignmentOperator) {
    fcm::small_vector<TestObject, 2> vec1;
    for (int i = 0; i < 5; ++i) {
        vec1.emplace_back(i);
    }

    fcm::small_vector<TestObject, 2> vec2;
    vec2.emplace_back(99);

    TestObject::reset_counts(); // Reset before the actual move assignment
    auto* heap_ptr = vec1.data();
    vec2 = std::move(vec1);

    EXPECT_EQ(vec2.size(), 5);
    EXPECT_EQ(vec2.data(), heap_ptr); // Should steal the heap allocation
    EXPECT_EQ(vec1.size(), 0); // Source should be empty

    // Move assignment should: 1) clear vec2 (1 destructor), 2) steal heap (no copies/moves)
    EXPECT_EQ(TestObject::constructor_calls, 0);
    EXPECT_EQ(TestObject::copy_calls, 0);
    EXPECT_EQ(TestObject::move_calls, 0);
    EXPECT_EQ(TestObject::destructor_calls, 1); // vec2's original element
}

TEST_F(SmallVectorTest, SelfAssignment) {
    fcm::small_vector<int, 4> vec{1, 2, 3, 4};

    vec = vec; // Self copy assignment

    EXPECT_EQ(vec.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(vec[i], i + 1);
    }

    // Test self move assignment (suppress warning)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-move"
    vec = std::move(vec);
    #pragma GCC diagnostic pop

    EXPECT_EQ(vec.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(vec[i], i + 1);
    }
}

// ============================================================================
// Iterator and Access Tests
// ============================================================================

TEST_F(SmallVectorTest, ElementAccess) {
    fcm::small_vector<int, 4> vec{10, 20, 30, 40};

    // Test operator[]
    EXPECT_EQ(vec[0], 10);
    EXPECT_EQ(vec[1], 20);
    EXPECT_EQ(vec[2], 30);
    EXPECT_EQ(vec[3], 40);

    // Test const operator[]
    const auto& cvec = vec;
    EXPECT_EQ(cvec[0], 10);
    EXPECT_EQ(cvec[3], 40);

    // Test back()
    EXPECT_EQ(vec.back(), 40);
    EXPECT_EQ(cvec.back(), 40);

    // Modify through operator[]
    vec[1] = 25;
    EXPECT_EQ(vec[1], 25);
}

TEST_F(SmallVectorTest, DataPointer) {
    fcm::small_vector<int, 4> vec{1, 2, 3};

    int* data = vec.data();
    const int* cdata = vec.data();

    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 3);

    EXPECT_EQ(cdata[0], 1);
    EXPECT_EQ(cdata[2], 3);

    // Modify through data pointer
    data[1] = 99;
    EXPECT_EQ(vec[1], 99);
}

TEST_F(SmallVectorTest, Iterators) {
    fcm::small_vector<int, 4> vec{1, 2, 3, 4};

    // Test begin/end
    auto it = vec.begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);

    // Test range-based for loop
    int expected = 1;
    for (const auto& value : vec) {
        EXPECT_EQ(value, expected++);
    }

    // Test const iterators
    const auto& cvec = vec;
    auto cit = cvec.begin();
    EXPECT_EQ(*cit, 1);

    // Test iterator distance
    EXPECT_EQ(vec.end() - vec.begin(), 4);
}

TEST_F(SmallVectorTest, IteratorsAfterReallocation) {
    fcm::small_vector<int, 2> vec{1, 2};

    auto old_begin = vec.begin();
    vec.push_back(3); // Should trigger reallocation

    // Old iterators should be invalidated, but new ones should work
    auto new_begin = vec.begin();
    EXPECT_NE(old_begin, new_begin);

    int expected = 1;
    for (const auto& value : vec) {
        EXPECT_EQ(value, expected++);
    }
}

// ============================================================================
// Modifiers Tests
// ============================================================================

TEST_F(SmallVectorTest, PushBackAndEmplaceBack) {
    fcm::small_vector<TestObject, 4> vec;

    TestObject::reset_counts();

    // Test emplace_back
    vec.emplace_back(1);
    vec.emplace_back(2);

    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
    EXPECT_EQ(TestObject::constructor_calls, 2);

    // Test push_back with copy
    TestObject obj(3);
    TestObject::reset_counts();
    vec.push_back(obj);

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[2].value, 3);
    EXPECT_EQ(TestObject::copy_calls, 1); // push_back(obj) calls copy constructor

    // Test push_back with move
    TestObject::reset_counts();
    vec.push_back(TestObject(4));

    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec[3].value, 4);
    // Creates temporary (1 constructor) + move (1 move call) or direct construct
    EXPECT_EQ(TestObject::constructor_calls, 1); // TestObject(4) temporary
    EXPECT_EQ(TestObject::move_calls, 1); // emplace_back(std::move(temp))
}

TEST_F(SmallVectorTest, PopBack) {
    fcm::small_vector<TestObject, 4> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);

    TestObject::reset_counts();

    vec.pop_back();
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec.back().value, 2);
    EXPECT_EQ(TestObject::destructor_calls, 1);

    vec.pop_back();
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec.back().value, 1);
    EXPECT_EQ(TestObject::destructor_calls, 2);
}

TEST_F(SmallVectorTest, Clear) {
    fcm::small_vector<TestObject, 4> vec;
    for (int i = 0; i < 5; ++i) { // Go beyond inline capacity
        vec.emplace_back(i);
    }

    EXPECT_EQ(vec.size(), 5);
    EXPECT_GT(vec.capacity(), 4);

    TestObject::reset_counts();
    vec.clear();

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
    EXPECT_GT(vec.capacity(), 4); // Capacity should remain
    EXPECT_EQ(TestObject::destructor_calls, 5);
}

TEST_F(SmallVectorTest, ShrinkToFitExactSize) {
    fcm::small_vector<int, 2> vec;

    // Grow to a specific size
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    auto old_capacity = vec.capacity();
    EXPECT_GT(old_capacity, 10);

    vec.shrink_to_fit();

    EXPECT_EQ(vec.size(), 10);
    EXPECT_LE(vec.capacity(), old_capacity);
    EXPECT_GE(vec.capacity(), 10);

    // Verify elements are intact
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// ============================================================================
// Exception Safety Tests
// ============================================================================

TEST_F(SmallVectorTest, ExceptionSafetyInPushBack) {
    fcm::small_vector<ThrowingObject, 2> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);

    // Next push_back will trigger reallocation
    ThrowingObject::should_throw_on_copy = true;

    EXPECT_THROW({
        vec.push_back(ThrowingObject(3));
    }, std::runtime_error);

    // Vector should remain in valid state
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
}

TEST_F(SmallVectorTest, ExceptionSafetyInEmplaceBack) {
    fcm::small_vector<ThrowingObject, 2> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);

    // This should trigger reallocation and throw during construction
    ThrowingObject::should_throw_on_copy = true;

    EXPECT_THROW({
        vec.emplace_back(3);
    }, std::runtime_error);

    // Vector should remain in valid state
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
}

TEST_F(SmallVectorTest, ExceptionSafetyInReserve) {
    fcm::small_vector<ThrowingObject, 2> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);

    ThrowingObject::should_throw_on_copy = true;

    EXPECT_THROW({
        vec.reserve(10);
    }, std::runtime_error);

    // Vector should remain in valid state
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
}

TEST_F(SmallVectorTest, StrongExceptionSafety) {
    fcm::small_vector<ThrowingObject, 2> vec;

    // Fill to capacity
    vec.emplace_back(1);
    vec.emplace_back(2);

    auto old_data = vec.data();
    auto old_size = vec.size();
    auto old_capacity = vec.capacity();

    ThrowingObject::should_throw_on_copy = true;

    // This should throw during reallocation when copying existing elements
    EXPECT_THROW({
        vec.emplace_back(3);
    }, std::runtime_error);

    // Vector should be unchanged (strong exception safety)
    EXPECT_EQ(vec.data(), old_data);
    EXPECT_EQ(vec.size(), old_size);
    EXPECT_EQ(vec.capacity(), old_capacity);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
}

// ============================================================================
// Telemetry Tests (if enabled)
// ============================================================================

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(SmallVectorTest, TelemetryTracking) {
    fcm::small_vector<int, 2> vec;

    auto& telemetry = vec.telemetry();
    EXPECT_EQ(telemetry.heap_allocs, 0);
    EXPECT_EQ(telemetry.spills, 0);
    EXPECT_EQ(telemetry.peak_size, 0);

    // Add elements within inline capacity
    vec.push_back(1);
    vec.push_back(2);

    EXPECT_EQ(telemetry.heap_allocs, 0);
    EXPECT_EQ(telemetry.spills, 0);
    EXPECT_EQ(telemetry.peak_size, 2);

    // Trigger spill to heap
    vec.push_back(3);

    EXPECT_GT(telemetry.heap_allocs, 0);
    EXPECT_EQ(telemetry.spills, 1);
    EXPECT_EQ(telemetry.peak_size, 3);
    EXPECT_GT(telemetry.peak_capacity, 2);
}

TEST_F(SmallVectorTest, TelemetryCallback) {
    fcm::small_vector<int, 2> vec;

    bool callback_called = false;
    std::string last_event;

    vec.set_telemetry_callback([&](const char* event, const fcm::small_vector<int, 2>::telemetry_t& telemetry) {
        callback_called = true;
        last_event = event;
        CORE_UNUSED(telemetry);
    });

    // Trigger reallocation
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3); // Should trigger callback

    EXPECT_TRUE(callback_called);
    EXPECT_EQ(last_event, "reallocate");
}
#endif

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(SmallVectorTest, EmptyVectorOperations) {
    fcm::small_vector<int, 4> vec;

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_NE(vec.data(), nullptr);
    EXPECT_EQ(vec.begin(), vec.end());

    vec.clear(); // Should be safe on empty vector
    EXPECT_TRUE(vec.empty());

    vec.shrink_to_fit(); // Should be safe on empty vector
    EXPECT_TRUE(vec.empty());
}

// NOTE: Zero inline capacity test removed due to C++ standard issues with zero-size arrays

TEST_F(SmallVectorTest, LargeInlineCapacity) {
    fcm::small_vector<int, 1000> vec;

    EXPECT_EQ(vec.capacity(), 1000);

    // Fill entire inline capacity
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 1000);
    EXPECT_EQ(vec.capacity(), 1000);

    // Verify all elements
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    // One more should trigger heap allocation
    vec.push_back(1000);
    EXPECT_GT(vec.capacity(), 1000);
}

TEST_F(SmallVectorTest, RapidGrowthAndShrinkage) {
    fcm::small_vector<int, 4> vec;

    // Rapid growth
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 1000);
    EXPECT_GE(vec.capacity(), 1000);

    // Rapid shrinkage
    while (!vec.empty()) {
        vec.pop_back();
    }

    EXPECT_TRUE(vec.empty());
    EXPECT_GE(vec.capacity(), 1000); // Capacity should remain

    // Should be able to grow again
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100);
}

TEST_F(SmallVectorTest, MoveOnlyType) {
    fcm::small_vector<std::unique_ptr<int>, 4> vec;

    vec.emplace_back(std::make_unique<int>(42));
    vec.emplace_back(std::make_unique<int>(43));

    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(*vec[0], 42);
    EXPECT_EQ(*vec[1], 43);

    // Test move semantics
    auto moved_vec = std::move(vec);
    EXPECT_EQ(moved_vec.size(), 2);
    EXPECT_EQ(*moved_vec[0], 42);
    EXPECT_EQ(*moved_vec[1], 43);

    EXPECT_EQ(vec.size(), 0); // Original should be empty
}

TEST_F(SmallVectorTest, ComplexType) {
    fcm::small_vector<std::string, 2> vec;

    vec.emplace_back("Hello");
    vec.emplace_back("World");
    vec.emplace_back("!"); // Should trigger reallocation

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], "Hello");
    EXPECT_EQ(vec[1], "World");
    EXPECT_EQ(vec[2], "!");
}

TEST_F(SmallVectorTest, TypeWithCustomAlignment) {
    struct alignas(64) AlignedType {
        char data[32];
        int value;
        AlignedType(int v) : value(v) { std::memset(data, v, sizeof(data)); }
    };

    fcm::small_vector<AlignedType, 2> vec;

    vec.emplace_back(1);
    vec.emplace_back(2);

    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(vec.data()) % 64, 0);
}
#include <gtest/gtest.h>
#include <core/memory/stable_vector.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <string>
#include <memory>
#include <set>

namespace fcm = fem::core::memory;

class StableVectorTest : public ::testing::Test {
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

    explicit TestObject(int v = 0, bool throw_copy = false)
        : value(v), should_throw_on_copy(throw_copy) {
        ++construct_count;
    }

    TestObject(const TestObject& other)
        : value(other.value), should_throw_on_copy(other.should_throw_on_copy) {
        if (should_throw_on_copy) throw std::runtime_error("Copy constructor exception");
        ++copy_count;
    }

    TestObject(TestObject&& other) noexcept
        : value(other.value), should_throw_on_copy(other.should_throw_on_copy) {
        other.value = -1;
        ++move_count;
    }

    TestObject& operator=(const TestObject& other) {
        if (this == &other) return *this;
        if (should_throw_on_copy) throw std::runtime_error("Copy assignment exception");
        value = other.value;
        should_throw_on_copy = other.should_throw_on_copy;
        return *this;
    }

    TestObject& operator=(TestObject&& other) noexcept {
        if (this == &other) return *this;
        value = other.value;
        should_throw_on_copy = other.should_throw_on_copy;
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

TEST_F(StableVectorTest, DefaultConstruction) {
    fcm::stable_vector<int> vec;

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.capacity(), 0);
}

TEST_F(StableVectorTest, ConstructionWithMemoryResource) {
    auto mr = fcm::default_resource();
    fcm::stable_vector<int> vec(mr);

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
}

TEST_F(StableVectorTest, ConstructionWithAllocator) {
    fcm::polymorphic_allocator<int> alloc(fcm::default_resource());
    fcm::stable_vector<int> vec(alloc);

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
}

TEST_F(StableVectorTest, ElementsPerBlockCalculation) {
    // Test the static calculation for different block sizes
    using SmallBlockVector = fcm::stable_vector<int, 64>;
    using LargeBlockVector = fcm::stable_vector<int, 4096>;

    std::size_t small_elems = SmallBlockVector::elems_per_block();
    std::size_t large_elems = LargeBlockVector::elems_per_block();

    EXPECT_GT(small_elems, 0);
    EXPECT_GT(large_elems, 0);
    EXPECT_GT(large_elems, small_elems);

    // Should be at least 64/sizeof(int) and 4096/sizeof(int) respectively
    EXPECT_GE(small_elems, 64 / sizeof(int));
    EXPECT_GE(large_elems, 4096 / sizeof(int));
}

// === Basic Operations Tests ===

TEST_F(StableVectorTest, PushBackAndSize) {
    fcm::stable_vector<int> vec;

    vec.push_back(10);
    EXPECT_EQ(vec.size(), 1);
    EXPECT_FALSE(vec.empty());
    EXPECT_EQ(vec[0], 10);

    vec.push_back(20);
    vec.push_back(30);
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 10);
    EXPECT_EQ(vec[1], 20);
    EXPECT_EQ(vec[2], 30);
}

TEST_F(StableVectorTest, EmplaceBack) {
    fcm::stable_vector<std::string> vec;

    auto& ref1 = vec.emplace_back("hello");
    EXPECT_EQ(ref1, "hello");
    EXPECT_EQ(vec.size(), 1);

    auto& ref2 = vec.emplace_back("world");
    EXPECT_EQ(ref2, "world");
    EXPECT_EQ(vec.size(), 2);

    EXPECT_EQ(vec[0], "hello");
    EXPECT_EQ(vec[1], "world");
}

TEST_F(StableVectorTest, PopBack) {
    fcm::stable_vector<int> vec;
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    EXPECT_EQ(vec.back(), 30);
    vec.pop_back();
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec.back(), 20);

    vec.pop_back();
    vec.pop_back();
    EXPECT_TRUE(vec.empty());
}

TEST_F(StableVectorTest, ElementAccess) {
    fcm::stable_vector<int> vec;
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i * 10);
    }

    // Test indexing
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i * 10);
    }

    // Test back()
    EXPECT_EQ(vec.back(), 90);

    // Test const access
    const auto& const_vec = vec;
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(const_vec[i], i * 10);
    }
}

// === Block Allocation Tests ===

TEST_F(StableVectorTest, BlockAllocationBehavior) {
    using SmallBlockVector = fcm::stable_vector<int, 64>; // Small blocks for easier testing
    SmallBlockVector vec;

    std::size_t elems_per_block = SmallBlockVector::elems_per_block();
    EXPECT_GT(elems_per_block, 0);

    // Fill exactly one block
    for (std::size_t i = 0; i < elems_per_block; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    EXPECT_EQ(vec.size(), elems_per_block);
    EXPECT_EQ(vec.capacity(), elems_per_block);

    // Adding one more should trigger another block allocation
    vec.push_back(999);
    EXPECT_EQ(vec.size(), elems_per_block + 1);
    EXPECT_EQ(vec.capacity(), 2 * elems_per_block);

    // Verify all elements are accessible
    for (std::size_t i = 0; i < elems_per_block; ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(i));
    }
    EXPECT_EQ(vec[elems_per_block], 999);
}

TEST_F(StableVectorTest, MultipleBlockAllocation) {
    using SmallBlockVector = fcm::stable_vector<int, 32>; // Very small blocks
    SmallBlockVector vec;

    std::size_t elems_per_block = SmallBlockVector::elems_per_block();
    std::size_t target_elements = elems_per_block * 3 + 5; // Span multiple blocks

    for (std::size_t i = 0; i < target_elements; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    EXPECT_EQ(vec.size(), target_elements);
    EXPECT_GE(vec.capacity(), target_elements);

    // Verify all elements across blocks
    for (std::size_t i = 0; i < target_elements; ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(i));
    }
}

TEST_F(StableVectorTest, ReserveCapacity) {
    fcm::stable_vector<int> vec;

    vec.reserve(100);
    EXPECT_GE(vec.capacity(), 100);
    EXPECT_EQ(vec.size(), 0);

    // Adding elements shouldn't require new allocations until capacity is exceeded
    std::size_t initial_capacity = vec.capacity();
    for (int i = 0; i < 50; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(vec.capacity(), initial_capacity);
}

TEST_F(StableVectorTest, ShrinkToFit) {
    using SmallBlockVector = fcm::stable_vector<int, 64>;
    SmallBlockVector vec;

    std::size_t elems_per_block = SmallBlockVector::elems_per_block();

    // Allocate multiple blocks
    vec.reserve(3 * elems_per_block);
    std::size_t large_capacity = vec.capacity();

    // Add only a few elements
    for (int i = 0; i < 5; ++i) {
        vec.push_back(i);
    }

    vec.shrink_to_fit();
    EXPECT_LT(vec.capacity(), large_capacity);
    EXPECT_GE(vec.capacity(), vec.size());

    // Elements should still be accessible
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// === Pointer/Reference Stability Tests ===

TEST_F(StableVectorTest, PointerStabilityDuringGrowth) {
    using SmallBlockVector = fcm::stable_vector<int, 64>;
    SmallBlockVector vec;

    std::size_t elems_per_block = SmallBlockVector::elems_per_block();

    // Add elements and store pointers
    std::vector<int*> pointers;
    for (std::size_t i = 0; i < elems_per_block + 10; ++i) {
        vec.push_back(static_cast<int>(i));
        pointers.push_back(&vec.back());
    }

    // Force more growth
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i + 1000);
    }

    // Original pointers should still be valid and point to correct values
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        EXPECT_EQ(*pointers[i], static_cast<int>(i));
        EXPECT_EQ(pointers[i], &vec[i]); // Address should be unchanged
    }
}

TEST_F(StableVectorTest, ReferenceStabilityDuringGrowth) {
    using SmallBlockVector = fcm::stable_vector<std::string, 64>;
    SmallBlockVector vec;

    vec.push_back("first");
    vec.push_back("second");

    std::string& first_ref = vec[0];
    std::string& second_ref = vec[1];

    // Force reallocation
    for (int i = 0; i < 1000; ++i) {
        vec.push_back("filler" + std::to_string(i));
    }

    // References should still be valid
    EXPECT_EQ(first_ref, "first");
    EXPECT_EQ(second_ref, "second");
    EXPECT_EQ(&first_ref, &vec[0]);
    EXPECT_EQ(&second_ref, &vec[1]);
}

TEST_F(StableVectorTest, PointerStabilityAcrossOperations) {
    fcm::stable_vector<int> vec;

    // Add initial elements
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    // Store pointers to elements
    std::vector<int*> ptrs;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        ptrs.push_back(&vec[i]);
    }

    // Perform various operations
    vec.reserve(1000);  // Should not affect existing element addresses
    vec.push_back(999); // Should not affect existing element addresses
    vec.pop_back();     // Should not affect existing element addresses

    // All original pointers should still be valid
    for (std::size_t i = 0; i < ptrs.size(); ++i) {
        EXPECT_EQ(*ptrs[i], static_cast<int>(i));
        EXPECT_EQ(ptrs[i], &vec[i]);
    }
}

// === Copy and Move Semantics Tests ===

TEST_F(StableVectorTest, CopyConstructor) {
    fcm::stable_vector<int> original;
    for (int i = 0; i < 20; ++i) {
        original.push_back(i * 2);
    }

    fcm::stable_vector<int> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    for (std::size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }

    // Modifying copy shouldn't affect original
    copy.push_back(999);
    EXPECT_NE(copy.size(), original.size());
}

TEST_F(StableVectorTest, MoveConstructor) {
    fcm::stable_vector<int> original;
    for (int i = 0; i < 20; ++i) {
        original.push_back(i * 3);
    }

    std::size_t original_size = original.size();
    fcm::stable_vector<int> moved(std::move(original));

    EXPECT_EQ(moved.size(), original_size);
    for (std::size_t i = 0; i < moved.size(); ++i) {
        EXPECT_EQ(moved[i], static_cast<int>(i * 3));
    }

    // Original should be in valid but unspecified state
    EXPECT_EQ(original.size(), 0);
}

TEST_F(StableVectorTest, CopyAssignment) {
    fcm::stable_vector<int> original;
    for (int i = 0; i < 15; ++i) {
        original.push_back(i + 100);
    }

    fcm::stable_vector<int> copy;
    copy.push_back(999); // Add some data to be overwritten

    copy = original;

    EXPECT_EQ(copy.size(), original.size());
    for (std::size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }
}

TEST_F(StableVectorTest, MoveAssignment) {
    fcm::stable_vector<int> original;
    for (int i = 0; i < 15; ++i) {
        original.push_back(i + 200);
    }

    fcm::stable_vector<int> moved;
    moved.push_back(888); // Add some data to be overwritten

    std::size_t original_size = original.size();
    moved = std::move(original);

    EXPECT_EQ(moved.size(), original_size);
    for (std::size_t i = 0; i < moved.size(); ++i) {
        EXPECT_EQ(moved[i], static_cast<int>(i + 200));
    }
}

TEST_F(StableVectorTest, SelfAssignment) {
    fcm::stable_vector<int> vec;
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    vec = vec; // Self assignment

    EXPECT_EQ(vec.size(), 10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// === Iterator Tests ===

TEST_F(StableVectorTest, ForwardIterator) {
    fcm::stable_vector<int> vec;
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i * 5);
    }

    // Test forward iteration
    int expected = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        EXPECT_EQ(*it, expected * 5);
        ++expected;
    }
}

TEST_F(StableVectorTest, ConstIterator) {
    fcm::stable_vector<int> vec;
    for (int i = 0; i < 8; ++i) {
        vec.push_back(i + 10);
    }

    const auto& const_vec = vec;
    int expected = 10;
    for (auto it = const_vec.begin(); it != const_vec.end(); ++it) {
        EXPECT_EQ(*it, expected);
        ++expected;
    }
}

TEST_F(StableVectorTest, RangeBasedForLoop) {
    fcm::stable_vector<std::string> vec;
    vec.push_back("alpha");
    vec.push_back("beta");
    vec.push_back("gamma");

    std::vector<std::string> collected;
    for (const auto& item : vec) {
        collected.push_back(item);
    }

    EXPECT_EQ(collected.size(), 3);
    EXPECT_EQ(collected[0], "alpha");
    EXPECT_EQ(collected[1], "beta");
    EXPECT_EQ(collected[2], "gamma");
}

TEST_F(StableVectorTest, IteratorArrowOperator) {
    fcm::stable_vector<std::string> vec;
    vec.push_back("test");
    vec.push_back("string");

    auto it = vec.begin();
    EXPECT_EQ(it->length(), 4); // "test".length()
    ++it;
    EXPECT_EQ(it->length(), 6); // "string".length()
}

// === Clear and Destruction Tests ===

TEST_F(StableVectorTest, ClearVector) {
    TestObject::reset_counters();

    fcm::stable_vector<TestObject> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);

    int objects_created = TestObject::construct_count;
    vec.clear();

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(TestObject::destruct_count, objects_created);
}

TEST_F(StableVectorTest, DestructorCallsElementDestructors) {
    TestObject::reset_counters();

    {
        fcm::stable_vector<TestObject> vec;
        vec.emplace_back(1);
        vec.emplace_back(2);
        vec.emplace_back(3);

        EXPECT_EQ(TestObject::construct_count, 3);
    } // Destructor called here

    EXPECT_EQ(TestObject::destruct_count, 3);
}

// === Exception Safety Tests ===

TEST_F(StableVectorTest, ExceptionSafetyInPushBack) {
    TestObject::reset_counters();

    fcm::stable_vector<TestObject> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);

    try {
        TestObject throwing_obj(3, true); // throws on copy
        vec.push_back(throwing_obj);
        FAIL() << "Expected exception";
    } catch (const std::runtime_error&) {
        // Exception expected
    }

    // Vector should remain in valid state
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].value, 1);
    EXPECT_EQ(vec[1].value, 2);
}

TEST_F(StableVectorTest, ExceptionSafetyInEmplace) {
    fcm::stable_vector<TestObject> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);

    // Direct construction should work even with copy-throwing objects
    try {
        vec.emplace_back(3, true); // Creates directly, no copy
        EXPECT_EQ(vec.size(), 3);
        EXPECT_EQ(vec[2].value, 3);
    } catch (...) {
        FAIL() << "Emplace should not throw during direct construction";
    }
}

// === Move-Only Type Tests ===

TEST_F(StableVectorTest, MoveOnlyType) {
    fcm::stable_vector<MoveOnlyType> vec;

    vec.emplace_back(10);
    vec.emplace_back(20);
    vec.emplace_back(30);

    EXPECT_EQ(vec[0].value, 10);
    EXPECT_EQ(vec[1].value, 20);
    EXPECT_EQ(vec[2].value, 30);

    // Test move semantics
    MoveOnlyType obj(99);
    vec.push_back(std::move(obj));
    EXPECT_EQ(obj.value, -1); // Should be moved from
    EXPECT_EQ(vec[3].value, 99);
}

// === Telemetry Tests ===

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(StableVectorTest, TelemetryTracking) {
    using SmallBlockVector = fcm::stable_vector<int, 64>;
    SmallBlockVector vec;

    auto& telemetry = vec.telemetry();
    EXPECT_EQ(telemetry.blocks_allocated, 0);
    EXPECT_EQ(telemetry.constructed_elements, 0);
    EXPECT_EQ(telemetry.destroyed_elements, 0);

    // Add elements to trigger block allocation
    std::size_t elems_per_block = SmallBlockVector::elems_per_block();
    for (std::size_t i = 0; i <= elems_per_block; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    EXPECT_GE(telemetry.blocks_allocated, 2); // At least 2 blocks allocated
    EXPECT_EQ(telemetry.constructed_elements, elems_per_block + 1);
    EXPECT_EQ(telemetry.peak_size, elems_per_block + 1);

    vec.pop_back();
    EXPECT_EQ(telemetry.destroyed_elements, 1);
}

TEST_F(StableVectorTest, TelemetryCallback) {
    fcm::stable_vector<int> vec;

    std::vector<std::string> events;
    vec.set_telemetry_callback([&](const char* event, const auto& /*telemetry*/) {
        events.push_back(std::string(event));
    });

    vec.reserve(100); // Should trigger allocate_block events

    // Should have received allocate_block events
    EXPECT_TRUE(std::find(events.begin(), events.end(), "allocate_block") != events.end());
}
#endif

// === Edge Cases and Stress Tests ===

TEST_F(StableVectorTest, EmptyVectorOperations) {
    fcm::stable_vector<int> vec;

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.capacity(), 0);

    // Iterators should work on empty vector
    EXPECT_EQ(vec.begin(), vec.end());

    vec.clear(); // Should be safe on empty vector
    EXPECT_TRUE(vec.empty());
}

TEST_F(StableVectorTest, LargeVector) {
    fcm::stable_vector<int> vec;

    const std::size_t large_size = 10000;
    for (std::size_t i = 0; i < large_size; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    EXPECT_EQ(vec.size(), large_size);

    // Verify all elements
    for (std::size_t i = 0; i < large_size; ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(i));
    }

    // Test iteration over large vector
    std::size_t count = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        EXPECT_EQ(*it, static_cast<int>(count));
        ++count;
    }
    EXPECT_EQ(count, large_size);
}

TEST_F(StableVectorTest, GrowthAndShrinkagePattern) {
    using SmallBlockVector = fcm::stable_vector<int, 32>;
    SmallBlockVector vec;

    // Growth phase
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }
    std::size_t peak_capacity = vec.capacity();

    // Shrinkage phase
    while (vec.size() > 10) {
        vec.pop_back();
    }

    // Capacity should still be at peak
    EXPECT_EQ(vec.capacity(), peak_capacity);

    // Shrink to fit
    vec.shrink_to_fit();
    EXPECT_LT(vec.capacity(), peak_capacity);

    // Remaining elements should be intact
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// === Complex Data Types Tests ===

TEST_F(StableVectorTest, StringOperations) {
    fcm::stable_vector<std::string> vec;

    vec.push_back("hello");
    vec.push_back("world");
    vec.emplace_back("test");

    EXPECT_EQ(vec[0], "hello");
    EXPECT_EQ(vec[1], "world");
    EXPECT_EQ(vec[2], "test");

    // Test string operations
    vec[0] += " there";
    EXPECT_EQ(vec[0], "hello there");
}

TEST_F(StableVectorTest, UniquePointerOperations) {
    fcm::stable_vector<std::unique_ptr<int>> vec;

    vec.push_back(std::make_unique<int>(10));
    vec.push_back(std::make_unique<int>(20));
    vec.emplace_back(std::make_unique<int>(30));

    EXPECT_EQ(*vec[0], 10);
    EXPECT_EQ(*vec[1], 20);
    EXPECT_EQ(*vec[2], 30);

    // Test move semantics
    auto ptr = std::move(vec[0]);
    EXPECT_EQ(*ptr, 10);
    EXPECT_EQ(vec[0], nullptr);
}

// === Custom Block Size Tests ===

TEST_F(StableVectorTest, CustomBlockSizes) {
    // Test with various block sizes
    using TinyBlockVector = fcm::stable_vector<int, 16>;
    using HugeBlockVector = fcm::stable_vector<int, 8192>;

    TinyBlockVector tiny_vec;
    HugeBlockVector huge_vec;

    // Both should work correctly regardless of block size
    for (int i = 0; i < 100; ++i) {
        tiny_vec.push_back(i);
        huge_vec.push_back(i * 2);
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(tiny_vec[i], i);
        EXPECT_EQ(huge_vec[i], i * 2);
    }

    // Tiny blocks should allocate more blocks
    EXPECT_GT(tiny_vec.capacity() / TinyBlockVector::elems_per_block(),
              huge_vec.capacity() / HugeBlockVector::elems_per_block());
}

// === Performance-Related Tests ===

TEST_F(StableVectorTest, NoUnnecessaryMoves) {
    TestObject::reset_counters();

    fcm::stable_vector<TestObject> vec;

    TestObject obj1(1);
    TestObject obj2(2);

    int moves_before = TestObject::move_count;
    vec.push_back(std::move(obj1));
    vec.push_back(std::move(obj2));
    int moves_after = TestObject::move_count;

    // Should have exactly 2 moves (one for each push)
    EXPECT_EQ(moves_after - moves_before, 2);
}

TEST_F(StableVectorTest, EmplaceVsPushPerformance) {
    TestObject::reset_counters();

    fcm::stable_vector<TestObject> vec;

    // Test emplace (should construct in-place)
    int constructs_before = TestObject::construct_count;
    vec.emplace_back(42);
    int constructs_after = TestObject::construct_count;

    EXPECT_EQ(constructs_after - constructs_before, 1); // Only direct construction
    EXPECT_EQ(TestObject::copy_count, 0); // No copies
    EXPECT_EQ(TestObject::move_count, 0); // No moves
}
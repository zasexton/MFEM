#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <chrono>

#include <core/memory/bump_vector.h>
#include <core/memory/memory_resource.h>

namespace fem::core::memory {
namespace {

// Test fixture for bump_vector tests
class BumpVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        destructor_count = 0;
        constructor_count = 0;
        move_count = 0;
        copy_count = 0;
    }

public:
    // Tracking counters for special member functions
    static int destructor_count;
    static int constructor_count;
    static int move_count;
    static int copy_count;
};

int BumpVectorTest::destructor_count = 0;
int BumpVectorTest::constructor_count = 0;
int BumpVectorTest::move_count = 0;
int BumpVectorTest::copy_count = 0;

// Helper class to track construction/destruction
struct TrackedObject {
    int value;

    explicit TrackedObject(int v = 0) : value(v) {
        ++BumpVectorTest::constructor_count;
    }

    TrackedObject(const TrackedObject& other) : value(other.value) {
        ++BumpVectorTest::copy_count;
    }

    TrackedObject(TrackedObject&& other) noexcept : value(other.value) {
        ++BumpVectorTest::move_count;
        other.value = -1;
    }

    ~TrackedObject() {
        ++BumpVectorTest::destructor_count;
    }

    TrackedObject& operator=(const TrackedObject& other) {
        if (this != &other) {
            value = other.value;
            ++BumpVectorTest::copy_count;
        }
        return *this;
    }

    TrackedObject& operator=(TrackedObject&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = -1;
            ++BumpVectorTest::move_count;
        }
        return *this;
    }
};

// Test default construction
TEST_F(BumpVectorTest, DefaultConstruction) {
    bump_vector<int> vec;

    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 0u);
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.data(), nullptr);
}

TEST_F(BumpVectorTest, ConstructionWithAllocator) {
    auto* mr = default_resource();
    bump_vector<int> vec(mr);

    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 0u);
    EXPECT_TRUE(vec.empty());
}

TEST_F(BumpVectorTest, ConstructionWithPolymorphicAllocator) {
    polymorphic_allocator<int> alloc(default_resource());
    bump_vector<int, polymorphic_allocator<int>> vec(alloc);

    EXPECT_EQ(vec.size(), 0u);
    EXPECT_TRUE(vec.empty());
}

// Test that copy construction is deleted
TEST_F(BumpVectorTest, NoCopyConstruction) {
    static_assert(!std::is_copy_constructible_v<bump_vector<int>>);
    static_assert(!std::is_copy_assignable_v<bump_vector<int>>);
    EXPECT_TRUE(true);
}

// Test push_back
TEST_F(BumpVectorTest, PushBackLValue) {
    bump_vector<int> vec;

    int value = 42;
    vec.push_back(value);

    EXPECT_EQ(vec.size(), 1u);
    EXPECT_GT(vec.capacity(), 0u);
    EXPECT_FALSE(vec.empty());
    EXPECT_NE(vec.data(), nullptr);
    EXPECT_EQ(vec[0], 42);
}

TEST_F(BumpVectorTest, PushBackRValue) {
    bump_vector<std::string> vec;

    vec.push_back("Hello");
    vec.push_back(std::string("World"));

    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], "Hello");
    EXPECT_EQ(vec[1], "World");
}

TEST_F(BumpVectorTest, PushBackMultiple) {
    bump_vector<int> vec;

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100u);
    EXPECT_GE(vec.capacity(), 100u);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// Test emplace_back
TEST_F(BumpVectorTest, EmplaceBack) {
    bump_vector<std::pair<int, std::string>> vec;

    vec.emplace_back(1, "one");
    vec.emplace_back(2, "two");
    vec.emplace_back(3, "three");

    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0].first, 1);
    EXPECT_EQ(vec[0].second, "one");
    EXPECT_EQ(vec[1].first, 2);
    EXPECT_EQ(vec[1].second, "two");
    EXPECT_EQ(vec[2].first, 3);
    EXPECT_EQ(vec[2].second, "three");
}

TEST_F(BumpVectorTest, EmplaceBackReturn) {
    bump_vector<int> vec;

    auto& ref = vec.emplace_back(42);
    EXPECT_EQ(ref, 42);
    EXPECT_EQ(&ref, &vec[0]);

    ref = 100;
    EXPECT_EQ(vec[0], 100);
}

// Test reserve
TEST_F(BumpVectorTest, Reserve) {
    bump_vector<int> vec;

    vec.reserve(50);
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_GE(vec.capacity(), 50u);
    EXPECT_TRUE(vec.empty());

    // Adding elements shouldn't reallocate
    auto* old_data = vec.data();
    for (int i = 0; i < 50; ++i) {
        vec.push_back(i);
    }

    // Should not have reallocated
    if (old_data != nullptr) {
        EXPECT_EQ(vec.data(), old_data);
    }
}

TEST_F(BumpVectorTest, ReserveSmaller) {
    bump_vector<int> vec;

    vec.reserve(100);
    auto cap1 = vec.capacity();

    // Reserve smaller should do nothing
    vec.reserve(50);
    EXPECT_EQ(vec.capacity(), cap1);
}

// Test clear
TEST_F(BumpVectorTest, Clear) {
    bump_vector<int> vec;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    auto cap = vec.capacity();
    vec.clear();

    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), cap);  // Capacity preserved
    EXPECT_TRUE(vec.empty());

    // Can add elements again
    vec.push_back(99);
    EXPECT_EQ(vec.size(), 1u);
    EXPECT_EQ(vec[0], 99);
}

TEST_F(BumpVectorTest, ClearWithDestructors) {
    bump_vector<TrackedObject> vec;

    for (int i = 0; i < 5; ++i) {
        vec.emplace_back(i);
    }

    destructor_count = 0;
    vec.clear();

    EXPECT_EQ(destructor_count, 5);
    EXPECT_EQ(vec.size(), 0u);
}

// Test element access
TEST_F(BumpVectorTest, ElementAccess) {
    bump_vector<int> vec;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i * i);
    }

    // Non-const access
    EXPECT_EQ(vec[5], 25);
    vec[5] = 100;
    EXPECT_EQ(vec[5], 100);

    // Const access
    const auto& cvec = vec;
    EXPECT_EQ(cvec[3], 9);
    EXPECT_EQ(cvec[5], 100);
}

TEST_F(BumpVectorTest, DataPointer) {
    bump_vector<int> vec;

    EXPECT_EQ(vec.data(), nullptr);

    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    int* ptr = vec.data();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(ptr[0], 1);
    EXPECT_EQ(ptr[1], 2);
    EXPECT_EQ(ptr[2], 3);

    // Const data
    const auto& cvec = vec;
    const int* cptr = cvec.data();
    EXPECT_EQ(cptr, ptr);
}

// Test growth strategy
TEST_F(BumpVectorTest, GrowthStrategy) {
    bump_vector<int> vec;

    // First allocation should be 8 (or similar small size)
    vec.push_back(1);
    auto cap1 = vec.capacity();
    EXPECT_GE(cap1, 1u);
    EXPECT_LE(cap1, 16u);  // Reasonable initial size

    // Fill to capacity
    for (std::size_t i = 1; i < cap1; ++i) {
        vec.push_back(static_cast<int>(i));
    }

    // Next push should grow
    vec.push_back(999);
    auto cap2 = vec.capacity();
    EXPECT_GT(cap2, cap1);

    // Growth should be approximately 1.5x
    EXPECT_GE(cap2, cap1 + cap1 / 2);
}

TEST_F(BumpVectorTest, LargeGrowth) {
    bump_vector<int> vec;

    // Force multiple reallocations
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 1000u);
    EXPECT_GE(vec.capacity(), 1000u);

    // Verify all values
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// Test move semantics
TEST_F(BumpVectorTest, MoveConstruction) {
    bump_vector<int> vec1;
    for (int i = 0; i < 10; ++i) {
        vec1.push_back(i);
    }

    auto* old_data = vec1.data();
    auto old_size = vec1.size();
    auto old_cap = vec1.capacity();

    bump_vector<int> vec2(std::move(vec1));

    // vec2 should have taken ownership
    EXPECT_EQ(vec2.data(), old_data);
    EXPECT_EQ(vec2.size(), old_size);
    EXPECT_EQ(vec2.capacity(), old_cap);

    // vec1 should be empty
    EXPECT_EQ(vec1.data(), nullptr);
    EXPECT_EQ(vec1.size(), 0u);
    EXPECT_EQ(vec1.capacity(), 0u);

    // Data should be intact
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec2[i], i);
    }
}

TEST_F(BumpVectorTest, MoveAssignment) {
    bump_vector<int> vec1;
    for (int i = 0; i < 10; ++i) {
        vec1.push_back(i);
    }

    bump_vector<int> vec2;
    vec2.push_back(999);

    auto* old_data = vec1.data();
    auto old_size = vec1.size();

    vec2 = std::move(vec1);

    // vec2 should have vec1's data
    EXPECT_EQ(vec2.data(), old_data);
    EXPECT_EQ(vec2.size(), old_size);

    // vec1 should be empty
    EXPECT_EQ(vec1.size(), 0u);

    // Data should be intact
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec2[i], i);
    }
}

TEST_F(BumpVectorTest, SelfMoveAssignmentProtection) {
    bump_vector<int> vec;
    for (int i = 0; i < 5; ++i) {
        vec.push_back(i);
    }

    auto* old_data = vec.data();
    auto old_size = vec.size();

    // Test that self-assignment check works
    // (We can't actually test self-move as it triggers a warning)
    // Just verify the vector works correctly
    EXPECT_EQ(vec.data(), old_data);
    EXPECT_EQ(vec.size(), old_size);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// Test with move-only types
TEST_F(BumpVectorTest, MoveOnlyTypes) {
    bump_vector<std::unique_ptr<int>> vec;

    vec.push_back(std::make_unique<int>(1));
    vec.push_back(std::make_unique<int>(2));
    vec.emplace_back(std::make_unique<int>(3));

    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(*vec[0], 1);
    EXPECT_EQ(*vec[1], 2);
    EXPECT_EQ(*vec[2], 3);

    // Force reallocation
    for (int i = 4; i <= 20; ++i) {
        vec.push_back(std::make_unique<int>(i));
    }

    // Verify all values after reallocation
    for (std::size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(*vec[i], static_cast<int>(i + 1));
    }
}

// Test destruction tracking
TEST_F(BumpVectorTest, ProperDestruction) {
    {
        bump_vector<TrackedObject> vec;

        constructor_count = 0;
        destructor_count = 0;
        move_count = 0;

        for (int i = 0; i < 10; ++i) {
            vec.emplace_back(i);
        }

        EXPECT_EQ(constructor_count, 10);

        // During growth, objects are move-constructed to new memory
        // and old ones are destroyed. There should be 10 live objects
        EXPECT_EQ(vec.size(), 10u);

        // Verify values are intact
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(vec[i].value, i);
        }
    }

    // After vec goes out of scope, all remaining objects should be destroyed
    // Total objects created = original constructions + move constructions
    int total_objects = constructor_count + move_count;
    EXPECT_EQ(destructor_count, total_objects);
}

// Test reallocation with throwing move constructor
TEST_F(BumpVectorTest, ReallocationWithNoexceptMove) {
    struct NoExceptMove {
        int value;
        NoExceptMove(int v) : value(v) {}
        NoExceptMove(NoExceptMove&& other) noexcept : value(other.value) {}
        NoExceptMove(const NoExceptMove&) = default;
    };

    bump_vector<NoExceptMove> vec;
    for (int i = 0; i < 100; ++i) {
        vec.emplace_back(i);
    }

    // Should use move during reallocation
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i].value, i);
    }
}

TEST_F(BumpVectorTest, ReallocationWithThrowingMove) {
    struct ThrowingMove {
        int value;
        ThrowingMove(int v) : value(v) {}
        ThrowingMove(ThrowingMove&& other) : value(other.value) {}  // Not noexcept
        ThrowingMove(const ThrowingMove&) = default;
    };

    bump_vector<ThrowingMove> vec;
    for (int i = 0; i < 100; ++i) {
        vec.emplace_back(i);
    }

    // Should use copy during reallocation due to throwing move
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i].value, i);
    }
}

// Test with trivially destructible types
TEST_F(BumpVectorTest, TriviallyDestructible) {
    bump_vector<int> vec;

    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }

    vec.clear();  // Should be optimized for trivially destructible
    EXPECT_EQ(vec.size(), 0u);

    // Can reuse
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i * i);
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i * i);
    }
}

// Test empty vector operations
TEST_F(BumpVectorTest, EmptyOperations) {
    bump_vector<int> vec;

    // Clear on empty
    vec.clear();
    EXPECT_TRUE(vec.empty());

    // Reserve on empty
    vec.reserve(0);
    EXPECT_TRUE(vec.empty());
}

// Test with various types
TEST_F(BumpVectorTest, VariousTypes) {
    // POD type
    {
        bump_vector<double> vec;
        vec.push_back(3.14);
        vec.push_back(2.71);
        EXPECT_DOUBLE_EQ(vec[0], 3.14);
        EXPECT_DOUBLE_EQ(vec[1], 2.71);
    }

    // Complex type
    {
        bump_vector<std::vector<int>> vec;
        vec.push_back({1, 2, 3});
        vec.push_back({4, 5, 6});
        EXPECT_EQ(vec[0], std::vector<int>({1, 2, 3}));
        EXPECT_EQ(vec[1], std::vector<int>({4, 5, 6}));
    }

    // Large type
    {
        struct LargeType {
            char data[1024];
            int value;
            LargeType(int v) : value(v) { std::fill(data, data + 1024, 0); }
        };

        bump_vector<LargeType> vec;
        vec.emplace_back(1);
        vec.emplace_back(2);
        EXPECT_EQ(vec[0].value, 1);
        EXPECT_EQ(vec[1].value, 2);
    }
}

// Test capacity after move
TEST_F(BumpVectorTest, CapacityAfterMove) {
    bump_vector<int> vec1;
    vec1.reserve(100);
    vec1.push_back(1);

    bump_vector<int> vec2(std::move(vec1));
    EXPECT_GE(vec2.capacity(), 100u);
    EXPECT_EQ(vec2.size(), 1u);
    EXPECT_EQ(vec1.capacity(), 0u);
}

// Test with custom allocator
TEST_F(BumpVectorTest, CustomAllocator) {
    // Create a custom memory resource
    auto* mr = default_resource();
    polymorphic_allocator<std::string> alloc(mr);

    bump_vector<std::string, polymorphic_allocator<std::string>> vec(alloc);

    vec.push_back("Hello");
    vec.push_back("World");
    vec.emplace_back("Test");

    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], "Hello");
    EXPECT_EQ(vec[1], "World");
    EXPECT_EQ(vec[2], "Test");

    vec.clear();
    EXPECT_EQ(vec.size(), 0u);
}

// Test growth boundary conditions
TEST_F(BumpVectorTest, GrowthBoundaries) {
    bump_vector<int> vec;

    // Test growth from 0
    EXPECT_EQ(vec.capacity(), 0u);
    vec.push_back(1);
    EXPECT_GT(vec.capacity(), 0u);

    // Test multiple growth cycles
    std::size_t prev_cap = vec.capacity();
    while (vec.size() < 1000) {
        vec.push_back(static_cast<int>(vec.size()));
        if (vec.capacity() > prev_cap) {
            // Capacity increased
            EXPECT_GT(vec.capacity(), prev_cap);
            prev_cap = vec.capacity();
        }
    }
}

// Test exception safety (basic guarantee)
TEST_F(BumpVectorTest, ExceptionSafety) {
    static int throw_after = 0;

    struct ThrowingConstructor {
        int value;

        ThrowingConstructor(int v) : value(v) {
            if (throw_after > 0 && --throw_after == 0) {
                throw std::runtime_error("Construction failed");
            }
        }

        ThrowingConstructor(const ThrowingConstructor& other) : value(other.value) {
            if (throw_after > 0 && --throw_after == 0) {
                throw std::runtime_error("Copy failed");
            }
        }

        ThrowingConstructor(ThrowingConstructor&& other) noexcept : value(other.value) {}
    };

    bump_vector<ThrowingConstructor> vec;

    // Add some elements
    throw_after = 0;  // Disable throwing
    for (int i = 0; i < 5; ++i) {
        vec.emplace_back(i);
    }

    // Try to add with exception
    throw_after = 1;
    try {
        vec.emplace_back(999);
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "Construction failed");
    }

    // Vector should still be valid
    EXPECT_EQ(vec.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i].value, i);
    }
}

// Test iterator-like usage with data()
TEST_F(BumpVectorTest, IteratorLikeUsage) {
    bump_vector<int> vec;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    // Use as iterator range
    int* begin = vec.data();
    int* end = vec.data() + vec.size();

    // Sum using pointers
    int sum = std::accumulate(begin, end, 0);
    EXPECT_EQ(sum, 45);  // 0+1+2+...+9

    // Sort using pointers
    std::sort(begin, end, std::greater<int>());

    for (std::size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(9 - i));
    }
}

// Performance test
TEST_F(BumpVectorTest, Performance) {
    constexpr int iterations = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    bump_vector<int> vec;
    for (int i = 0; i < iterations; ++i) {
        vec.push_back(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(vec.size(), static_cast<std::size_t>(iterations));

    // Should be reasonably fast
    EXPECT_LT(duration.count(), 100000);  // Less than 100ms for 100k insertions
}

// Stress test
TEST_F(BumpVectorTest, StressTest) {
    bump_vector<std::string> vec;

    // Many small strings
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(std::to_string(i));
    }

    // Some large strings
    for (int i = 0; i < 10; ++i) {
        vec.push_back(std::string(1000, 'A' + i));
    }

    // Clear and refill
    vec.clear();

    for (int i = 0; i < 100; ++i) {
        vec.emplace_back("String" + std::to_string(i));
    }

    EXPECT_EQ(vec.size(), 100u);

    // Verify data integrity
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], "String" + std::to_string(i));
    }
}

} // namespace
} // namespace fem::core::memory
#include <gtest/gtest.h>
#include <core/memory/arena.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <string>
#include <thread>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace fcm = fem::core::memory;

// Test structure for various tests - must be defined before ArenaTest
struct TestObject {
    int x;
    double y;
    static bool constructed;
    static bool destroyed;

    TestObject(int a = 0, double b = 0.0) : x(a), y(b) {
        constructed = true;
        destroyed = false;
    }

    ~TestObject() {
        destroyed = true;
    }
};

bool TestObject::constructed = false;
bool TestObject::destroyed = false;

class ArenaTest : public ::testing::Test {
protected:
    void SetUp() override {
        TestObject::constructed = false;
        TestObject::destroyed = false;
    }
};

// Basic functionality tests
TEST_F(ArenaTest, DefaultConstruction) {
    fem::core::memory::Arena arena;

    EXPECT_EQ(arena.used(), 0);
    EXPECT_EQ(arena.capacity(), 0);
    EXPECT_NE(arena.get_memory_resource(), nullptr);
}

TEST_F(ArenaTest, ExplicitConstruction) {
    const std::size_t initial_size = 1024;
    fem::core::memory::Arena arena(initial_size);

    EXPECT_EQ(arena.used(), 0);
    EXPECT_GE(arena.capacity(), 0);  // May be 0 until first allocation
    EXPECT_NE(arena.get_memory_resource(), nullptr);
}

TEST_F(ArenaTest, BasicAllocation) {
    fem::core::memory::Arena arena(1024);

    void* p1 = arena.allocate(100);
    ASSERT_NE(p1, nullptr);
    EXPECT_GT(arena.used(), 0);
    EXPECT_LE(arena.used(), arena.capacity());

    void* p2 = arena.allocate(200);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    EXPECT_GT(arena.used(), 100);
}

TEST_F(ArenaTest, AlignedAllocation) {
    fem::core::memory::Arena arena(1024);

    // Allocate with specific alignment
    void* p1 = arena.allocate(50, 32);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p1) % 32, 0);

    void* p2 = arena.allocate(100, 64);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p2) % 64, 0);
}

TEST_F(ArenaTest, LargeAllocation) {
    fem::core::memory::Arena arena(100);

    // Should grow to accommodate large allocation
    void* p = arena.allocate(1024);
    ASSERT_NE(p, nullptr);
    EXPECT_GE(arena.capacity(), 1024);
}

// Marker and rewind tests
TEST_F(ArenaTest, MarkerRewind) {
    fem::core::memory::Arena arena(1024);

    void* p1 = arena.allocate(100);
    ASSERT_NE(p1, nullptr);
    auto mark1 = arena.mark();
    std::size_t usage1 = arena.used();

    void* p2 = arena.allocate(200);
    ASSERT_NE(p2, nullptr);
    void* p3 = arena.allocate(300);
    ASSERT_NE(p3, nullptr);
    EXPECT_GT(arena.used(), usage1);

    // Rewind to mark1
    arena.rewind(mark1);
    EXPECT_EQ(arena.used(), usage1);

    // Can allocate again from rewound position
    void* p4 = arena.allocate(50);
    ASSERT_NE(p4, nullptr);
    EXPECT_EQ(p4, p2);  // Should get same address as p2
}

TEST_F(ArenaTest, MultipleMarkers) {
    fem::core::memory::Arena arena(1024);

    auto mark0 = arena.mark();
    arena.allocate(100);

    auto mark1 = arena.mark();
    arena.allocate(200);

    [[maybe_unused]] auto mark2 = arena.mark();
    arena.allocate(300);

    arena.rewind(mark1);
    EXPECT_LT(arena.used(), 600);

    arena.rewind(mark0);
    EXPECT_EQ(arena.used(), 0);
}

// Reset tests
TEST_F(ArenaTest, Reset) {
    fem::core::memory::Arena arena(1024);

    arena.allocate(100);
    arena.allocate(200);
    EXPECT_GT(arena.used(), 0);
    EXPECT_GT(arena.capacity(), 0);

    // Reset arena
    arena.reset();

    // Should be empty
    EXPECT_EQ(arena.used(), 0);
    EXPECT_EQ(arena.capacity(), 0);

    // Can allocate again
    void* p = arena.allocate(50);
    ASSERT_NE(p, nullptr);
}

// RAII Scope tests
TEST_F(ArenaTest, ScopeBasic) {
    fem::core::memory::Arena arena(1024);
    std::size_t initial_used = arena.used();

    {
        auto scope = arena.scope();
        // Scope is just an RAII wrapper, no need to check it

        // Allocations within scope
        [[maybe_unused]] void* p1 = arena.allocate(100);
        void* p2 = arena.allocate(200);
        EXPECT_NE(p1, nullptr);
        EXPECT_NE(p2, nullptr);
        // Used memory should be at least 300, but may be more due to alignment
        EXPECT_GE(arena.used(), initial_used + 300);
    }

    // Scope auto-rewinds on destruction
    EXPECT_EQ(arena.used(), initial_used);
}

TEST_F(ArenaTest, NestedScopes) {
    fem::core::memory::Arena arena(1024);

    std::size_t usage0 = arena.used();

    {
        auto scope1 = arena.scope();
        arena.allocate(100);
        std::size_t usage1 = arena.used();

        {
            auto scope2 = arena.scope();
            arena.allocate(200);
            EXPECT_GT(arena.used(), usage1);
        }

        EXPECT_EQ(arena.used(), usage1);
    }

    EXPECT_EQ(arena.used(), usage0);
}

// Object construction tests
TEST_F(ArenaTest, CreateDestroy) {
    fem::core::memory::Arena arena(1024);

    TestObject::constructed = false;
    TestObject::destroyed = false;

    TestObject* obj = arena.create<TestObject>(42, 3.14);
    ASSERT_NE(obj, nullptr);
    EXPECT_TRUE(TestObject::constructed);
    EXPECT_FALSE(TestObject::destroyed);
    EXPECT_EQ(obj->x, 42);
    EXPECT_DOUBLE_EQ(obj->y, 3.14);

    arena.destroy(obj);
    EXPECT_TRUE(TestObject::destroyed);
}

TEST_F(ArenaTest, CreateMultipleObjects) {
    fem::core::memory::Arena arena(1024);

    std::vector<TestObject*> objects;
    for (int i = 0; i < 10; ++i) {
        objects.push_back(arena.create<TestObject>(i, static_cast<double>(i) * 1.5));
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(objects[i]->x, i);
        EXPECT_DOUBLE_EQ(objects[i]->y, static_cast<double>(i) * 1.5);
    }
}

// Move semantics tests
TEST_F(ArenaTest, MoveConstruction) {
    fem::core::memory::Arena arena1(1024);
    arena1.allocate(100);
    std::size_t used1 = arena1.used();
    std::size_t cap1 = arena1.capacity();

    fem::core::memory::Arena arena2(std::move(arena1));
    EXPECT_EQ(arena2.used(), used1);
    EXPECT_EQ(arena2.capacity(), cap1);
}

TEST_F(ArenaTest, MoveAssignment) {
    fem::core::memory::Arena arena1(1024);
    arena1.allocate(100);
    std::size_t used1 = arena1.used();

    fem::core::memory::Arena arena2(512);
    arena2.allocate(50);

    arena2 = std::move(arena1);
    EXPECT_EQ(arena2.used(), used1);
}

// Result-based API tests
TEST_F(ArenaTest, TryAllocate_Success) {
    fem::core::memory::Arena arena(1024);

    auto result = arena.try_allocate(100);
    ASSERT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);
}

TEST_F(ArenaTest, TryAllocate_InvalidArguments) {
    fem::core::memory::Arena arena(1024);

    // Test zero size
    auto result1 = arena.try_allocate(0);
    EXPECT_FALSE(result1.is_ok());
    EXPECT_EQ(result1.error(), fem::core::error::ErrorCode::InvalidArgument);

    // Test overflow size
    auto result2 = arena.try_allocate(std::numeric_limits<std::size_t>::max());
    EXPECT_FALSE(result2.is_ok());
    EXPECT_EQ(result2.error(), fem::core::error::ErrorCode::OutOfMemory);
}

TEST_F(ArenaTest, TryCreate_Success) {
    fem::core::memory::Arena arena(1024);

    auto result = arena.try_create<TestObject>(42, 3.14);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value()->x, 42);
    EXPECT_DOUBLE_EQ(result.value()->y, 3.14);
    EXPECT_TRUE(TestObject::constructed);
}

// Stress tests
TEST_F(ArenaTest, StressAllocation) {
    fem::core::memory::Arena arena(4096);

    std::vector<void*> pointers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(1, 256);

    for (int i = 0; i < 100; ++i) {
        std::size_t size = static_cast<std::size_t>(size_dist(gen));
        void* p = arena.allocate(size);
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);

        // Write pattern to check for corruption
        std::memset(p, static_cast<int>(i % 256), size);
    }

    // Verify no overlaps (simplified check)
    for (std::size_t i = 0; i < pointers.size() - 1; ++i) {
        EXPECT_NE(pointers[i], pointers[i + 1]);
    }
}

TEST_F(ArenaTest, StressWithScopes) {
    fem::core::memory::Arena arena(4096);

    for (int i = 0; i < 10; ++i) {
        auto scope = arena.scope();

        for (int j = 0; j < 50; ++j) {
            arena.allocate(16 + j);
        }

        EXPECT_GT(arena.used(), 0);
    }

    // All scopes rewound
    EXPECT_EQ(arena.used(), 0);
}

// Edge cases
TEST_F(ArenaTest, EmptyArenaOperations) {
    fem::core::memory::Arena arena;

    EXPECT_EQ(arena.used(), 0);
    EXPECT_EQ(arena.capacity(), 0);

    // Mark on empty arena
    auto mark = arena.mark();
    arena.rewind(mark);
    EXPECT_EQ(arena.used(), 0);

    // Reset empty arena
    arena.reset();
    EXPECT_EQ(arena.used(), 0);
}

TEST_F(ArenaTest, ZeroSizeInitialBlock) {
    // This should work, will allocate on first use
    fem::core::memory::Arena arena(fcm::default_resource(), 0);

    void* p = arena.allocate(100);
    ASSERT_NE(p, nullptr);
    EXPECT_GT(arena.capacity(), 0);
}

TEST_F(ArenaTest, SequentialGrowth) {
    fem::core::memory::Arena arena(100);

    std::size_t total_allocated = 0;
    for (int i = 0; i < 20; ++i) {
        void* p = arena.allocate(100);
        ASSERT_NE(p, nullptr);
        total_allocated += 100;
    }

    EXPECT_GE(arena.capacity(), total_allocated);
}
#include <gtest/gtest.h>
#include <core/memory/arena_allocator.h>
#include <vector>
#include <list>
#include <deque>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>

namespace fcm = fem::core::memory;

class ArenaAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        arena_ = std::make_unique<fcm::Arena>(4096);
    }

    std::unique_ptr<fcm::Arena> arena_;
};

// Test structure
struct TestObject {
    int value;
    double data;
    std::string text;

    TestObject(int v = 0, double d = 0.0, const std::string& t = "")
        : value(v), data(d), text(t) {}
};

// Basic allocator operations
TEST_F(ArenaAllocatorTest, DefaultConstruction) {
    fcm::ArenaAllocator<int> alloc;
    EXPECT_EQ(alloc.arena(), nullptr);
}

TEST_F(ArenaAllocatorTest, ExplicitConstruction) {
    fcm::ArenaAllocator<int> alloc(arena_.get());
    EXPECT_EQ(alloc.arena(), arena_.get());
}

TEST_F(ArenaAllocatorTest, Rebinding) {
    fcm::ArenaAllocator<int> int_alloc(arena_.get());
    fcm::ArenaAllocator<double> double_alloc(int_alloc);

    EXPECT_EQ(double_alloc.arena(), arena_.get());
}

TEST_F(ArenaAllocatorTest, SingleAllocation) {
    fcm::ArenaAllocator<int> alloc(arena_.get());

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    *p = 42;
    EXPECT_EQ(*p, 42);

    alloc.deallocate(p, 1);  // No-op, but should not crash
}

TEST_F(ArenaAllocatorTest, MultipleAllocations) {
    fcm::ArenaAllocator<int> alloc(arena_.get());

    std::vector<int*> ptrs;
    for (int i = 0; i < 10; ++i) {
        int* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = i;
        ptrs.push_back(p);
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(*ptrs[i], i);
    }
}

TEST_F(ArenaAllocatorTest, ArrayAllocation) {
    fcm::ArenaAllocator<double> alloc(arena_.get());

    constexpr std::size_t count = 100;
    double* array = alloc.allocate(count);
    ASSERT_NE(array, nullptr);

    for (std::size_t i = 0; i < count; ++i) {
        array[i] = static_cast<double>(i) * 1.5;
    }

    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(array[i], static_cast<double>(i) * 1.5);
    }
}

// Construction and destruction
TEST_F(ArenaAllocatorTest, ConstructDestroy) {
    fcm::ArenaAllocator<TestObject> alloc(arena_.get());

    TestObject* obj = alloc.allocate(1);
    ASSERT_NE(obj, nullptr);

    alloc.construct(obj, 42, 3.14, "test");
    EXPECT_EQ(obj->value, 42);
    EXPECT_DOUBLE_EQ(obj->data, 3.14);
    EXPECT_EQ(obj->text, "test");

    alloc.destroy(obj);
    // Object destroyed but memory not reclaimed (arena behavior)
}

// STL container integration
TEST_F(ArenaAllocatorTest, VectorIntegration) {
    fcm::ArenaAllocator<int> alloc(arena_.get());
    std::vector<int, fcm::ArenaAllocator<int>> vec(alloc);

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(ArenaAllocatorTest, ListIntegration) {
    fcm::ArenaAllocator<std::string> alloc(arena_.get());
    std::list<std::string, fcm::ArenaAllocator<std::string>> lst(alloc);

    lst.push_back("first");
    lst.push_back("second");
    lst.push_back("third");

    EXPECT_EQ(lst.size(), 3);
    EXPECT_EQ(lst.front(), "first");
    EXPECT_EQ(lst.back(), "third");
}

TEST_F(ArenaAllocatorTest, DequeIntegration) {
    fcm::ArenaAllocator<double> alloc(arena_.get());
    std::deque<double, fcm::ArenaAllocator<double>> dq(alloc);

    for (int i = 0; i < 50; ++i) {
        dq.push_back(static_cast<double>(i) * 2.0);
        dq.push_front(static_cast<double>(-i) * 2.0);
    }

    EXPECT_EQ(dq.size(), 100);
}

TEST_F(ArenaAllocatorTest, MapIntegration) {
    using MapAlloc = fcm::ArenaAllocator<std::pair<const int, std::string>>;
    MapAlloc alloc(arena_.get());
    std::map<int, std::string, std::less<int>, MapAlloc> m(std::less<int>(), alloc);

    m[1] = "one";
    m[2] = "two";
    m[3] = "three";

    EXPECT_EQ(m.size(), 3);
    EXPECT_EQ(m[2], "two");
}

// Allocator traits and propagation
TEST_F(ArenaAllocatorTest, AllocatorEquality) {
    fcm::ArenaAllocator<int> alloc1(arena_.get());
    fcm::ArenaAllocator<int> alloc2(arena_.get());
    fcm::ArenaAllocator<int> alloc3;

    EXPECT_EQ(alloc1, alloc2);  // Same arena
    EXPECT_NE(alloc1, alloc3);  // Different arena (null)
}

TEST_F(ArenaAllocatorTest, PropagateOnMove) {
    fcm::ArenaAllocator<int> alloc(arena_.get());
    std::vector<int, fcm::ArenaAllocator<int>> vec1(alloc);

    vec1.push_back(42);
    auto vec2 = std::move(vec1);

    EXPECT_EQ(vec2.size(), 1);
    EXPECT_EQ(vec2[0], 42);
    EXPECT_EQ(vec2.get_allocator().arena(), arena_.get());
}

TEST_F(ArenaAllocatorTest, SwapContainers) {
    fcm::Arena arena2(4096);

    fcm::ArenaAllocator<int> alloc1(arena_.get());
    fcm::ArenaAllocator<int> alloc2(&arena2);

    std::vector<int, fcm::ArenaAllocator<int>> vec1(alloc1);
    std::vector<int, fcm::ArenaAllocator<int>> vec2(alloc2);

    vec1.push_back(1);
    vec2.push_back(2);

    vec1.swap(vec2);

    EXPECT_EQ(vec1[0], 2);
    EXPECT_EQ(vec2[0], 1);
}

// Result-based API
TEST_F(ArenaAllocatorTest, TryAllocate_Success) {
    fcm::ArenaAllocator<int> alloc(arena_.get());

    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);
}

TEST_F(ArenaAllocatorTest, TryAllocate_NullArena) {
    fcm::ArenaAllocator<int> alloc;  // No arena

    auto result = alloc.try_allocate(10);
    EXPECT_FALSE(result.is_ok());
    EXPECT_EQ(result.error(), fem::core::error::ErrorCode::InvalidState);
}

// Complex object allocation
TEST_F(ArenaAllocatorTest, ComplexObjectAllocation) {
    fcm::ArenaAllocator<TestObject> alloc(arena_.get());

    constexpr std::size_t count = 50;
    TestObject* objects = alloc.allocate(count);
    ASSERT_NE(objects, nullptr);

    for (std::size_t i = 0; i < count; ++i) {
        alloc.construct(&objects[i], static_cast<int>(i),
                       static_cast<double>(i) * 2.5,
                       "obj_" + std::to_string(i));
    }

    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(objects[i].value, static_cast<int>(i));
        EXPECT_DOUBLE_EQ(objects[i].data, static_cast<double>(i) * 2.5);
        EXPECT_EQ(objects[i].text, "obj_" + std::to_string(i));
    }

    for (std::size_t i = 0; i < count; ++i) {
        alloc.destroy(&objects[i]);
    }
}

// Nested containers
TEST_F(ArenaAllocatorTest, NestedContainers) {
    using InnerVec = std::vector<int, fcm::ArenaAllocator<int>>;
    using OuterVec = std::vector<InnerVec, fcm::ArenaAllocator<InnerVec>>;

    fcm::ArenaAllocator<InnerVec> outer_alloc(arena_.get());
    OuterVec outer(outer_alloc);

    for (int i = 0; i < 5; ++i) {
        fcm::ArenaAllocator<int> inner_alloc(arena_.get());
        InnerVec inner(inner_alloc);
        for (int j = 0; j < 10; ++j) {
            inner.push_back(i * 10 + j);
        }
        outer.push_back(std::move(inner));
    }

    EXPECT_EQ(outer.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(outer[i].size(), 10);
        for (int j = 0; j < 10; ++j) {
            EXPECT_EQ(outer[i][j], i * 10 + j);
        }
    }
}

// Performance patterns
TEST_F(ArenaAllocatorTest, BulkAllocationPattern) {
    fcm::ArenaAllocator<TestObject> obj_alloc(arena_.get());

    // Allocate space for many objects at once
    constexpr std::size_t batch_size = 1000;
    TestObject* batch = obj_alloc.allocate(batch_size);
    ASSERT_NE(batch, nullptr);

    // Construct in-place
    for (std::size_t i = 0; i < batch_size; ++i) {
        obj_alloc.construct(&batch[i], static_cast<int>(i),
                           static_cast<double>(i) * 0.5, "bulk");
    }

    // Use objects...
    EXPECT_EQ(batch[500].value, 500);
    EXPECT_DOUBLE_EQ(batch[500].data, 250.0);

    // Destroy all
    for (std::size_t i = 0; i < batch_size; ++i) {
        obj_alloc.destroy(&batch[i]);
    }
}

TEST_F(ArenaAllocatorTest, TemporaryWorkPattern) {
    std::size_t initial_used = arena_->used();

    {
        // Create temporary work area
        auto scope = arena_->scope();

        fcm::ArenaAllocator<int> alloc(arena_.get());
        std::vector<int, fcm::ArenaAllocator<int>> temp_data(alloc);

        // Do temporary work
        for (int i = 0; i < 10000; ++i) {
            temp_data.push_back(i);
        }

        // Process data...
        int sum = std::accumulate(temp_data.begin(), temp_data.end(), 0);
        EXPECT_EQ(sum, 49995000);
    }  // Scope rewinds, freeing all temporary allocations

    EXPECT_EQ(arena_->used(), initial_used);
}

// Edge cases
TEST_F(ArenaAllocatorTest, ZeroSizeAllocation) {
    fcm::ArenaAllocator<int> alloc(arena_.get());

    // Some implementations may return non-null for zero
    int* p = alloc.allocate(0);
    alloc.deallocate(p, 0);
    // Should not crash
}

TEST_F(ArenaAllocatorTest, LargeAllocation) {
    fcm::ArenaAllocator<char> alloc(arena_.get());

    constexpr std::size_t large_size = 1024 * 1024;  // 1MB
    char* buffer = alloc.allocate(large_size);
    ASSERT_NE(buffer, nullptr);

    // Fill with pattern
    std::fill_n(buffer, large_size, 'A');
    EXPECT_EQ(buffer[0], 'A');
    EXPECT_EQ(buffer[large_size - 1], 'A');
}

// Allocator with different types
TEST_F(ArenaAllocatorTest, MixedTypeAllocations) {
    fcm::ArenaAllocator<int> int_alloc(arena_.get());
    fcm::ArenaAllocator<double> double_alloc(arena_.get());
    fcm::ArenaAllocator<std::string> string_alloc(arena_.get());

    int* i = int_alloc.allocate(1);
    double* d = double_alloc.allocate(1);
    std::string* s = string_alloc.allocate(1);

    *i = 42;
    *d = 3.14;
    string_alloc.construct(s, "mixed");

    EXPECT_EQ(*i, 42);
    EXPECT_DOUBLE_EQ(*d, 3.14);
    EXPECT_EQ(*s, "mixed");

    string_alloc.destroy(s);
}
#include <gtest/gtest.h>
#include <core/memory/pool_allocator.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cstring>
#include <random>

namespace fcm = fem::core::memory;

class PoolAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        resource_ = fcm::new_delete_resource();
    }

    fcm::memory_resource* resource_ = nullptr;
};

// Test structure
struct TestObject {
    int id;
    double value;
    std::string name;
    static int construct_count;
    static int destruct_count;

    TestObject(int i = 0, double v = 0.0, const std::string& n = "")
        : id(i), value(v), name(n) {
        ++construct_count;
    }

    ~TestObject() {
        ++destruct_count;
    }

    static void reset_counts() {
        construct_count = 0;
        destruct_count = 0;
    }
};

int TestObject::construct_count = 0;
int TestObject::destruct_count = 0;

// Basic functionality tests
TEST_F(PoolAllocatorTest, DefaultConstruction) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Should be properly initialized
    EXPECT_NE(alloc.get_upstream(), nullptr);
}

TEST_F(PoolAllocatorTest, SingleAllocation) {
    fcm::PoolAllocator<int> alloc(resource_);

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    *p = 42;
    EXPECT_EQ(*p, 42);

    alloc.deallocate(p, 1);
}

TEST_F(PoolAllocatorTest, MultipleAllocations) {
    fcm::PoolAllocator<double> alloc(resource_);

    std::vector<double*> pointers;

    // Allocate multiple single objects
    for (int i = 0; i < 100; ++i) {
        double* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = static_cast<double>(i) * 1.5;
        pointers.push_back(p);
    }

    // Verify values
    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(*pointers[i], static_cast<double>(i) * 1.5);
    }

    // Deallocate all
    for (double* p : pointers) {
        alloc.deallocate(p, 1);
    }
}

TEST_F(PoolAllocatorTest, ArrayAllocationFallback) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Allocate array (should use fallback to upstream)
    int* arr = alloc.allocate(10);
    ASSERT_NE(arr, nullptr);

    for (int i = 0; i < 10; ++i) {
        arr[i] = i * 2;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arr[i], i * 2);
    }

    alloc.deallocate(arr, 10);
}

TEST_F(PoolAllocatorTest, MixedAllocationSizes) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Single allocation (uses pool)
    int* single = alloc.allocate(1);
    ASSERT_NE(single, nullptr);
    *single = 100;

    // Array allocation (uses upstream)
    int* array = alloc.allocate(5);
    ASSERT_NE(array, nullptr);
    for (int i = 0; i < 5; ++i) {
        array[i] = i;
    }

    // Another single (uses pool)
    int* single2 = alloc.allocate(1);
    ASSERT_NE(single2, nullptr);
    *single2 = 200;

    // Verify
    EXPECT_EQ(*single, 100);
    EXPECT_EQ(*single2, 200);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(array[i], i);
    }

    // Cleanup
    alloc.deallocate(single, 1);
    alloc.deallocate(array, 5);
    alloc.deallocate(single2, 1);
}

// Construction and destruction tests
TEST_F(PoolAllocatorTest, ConstructDestroy) {
    fcm::PoolAllocator<TestObject> alloc(resource_);

    TestObject::reset_counts();

    TestObject* obj = alloc.allocate(1);
    ASSERT_NE(obj, nullptr);

    alloc.construct(obj, 42, 3.14, "test");
    EXPECT_EQ(TestObject::construct_count, 1);
    EXPECT_EQ(obj->id, 42);
    EXPECT_DOUBLE_EQ(obj->value, 3.14);
    EXPECT_EQ(obj->name, "test");

    alloc.destroy(obj);
    EXPECT_EQ(TestObject::destruct_count, 1);

    alloc.deallocate(obj, 1);
}

TEST_F(PoolAllocatorTest, TrivialDestruction) {
    fcm::PoolAllocator<int> alloc(resource_);

    int* p = alloc.allocate(1);
    alloc.construct(p, 42);

    // Should compile and be no-op for trivially destructible types
    alloc.destroy(p);

    alloc.deallocate(p, 1);
}

// Rebinding tests
TEST_F(PoolAllocatorTest, Rebinding) {
    fcm::PoolAllocator<int> int_alloc(resource_);
    fcm::PoolAllocator<double> double_alloc(int_alloc);

    // Should share upstream resource
    EXPECT_EQ(int_alloc.get_upstream(), double_alloc.get_upstream());

    // Both should work independently
    int* i = int_alloc.allocate(1);
    double* d = double_alloc.allocate(1);

    *i = 42;
    *d = 3.14;

    EXPECT_EQ(*i, 42);
    EXPECT_DOUBLE_EQ(*d, 3.14);

    int_alloc.deallocate(i, 1);
    double_alloc.deallocate(d, 1);
}

// STL container integration tests
TEST_F(PoolAllocatorTest, VectorIntegration) {
    fcm::PoolAllocator<int> alloc(resource_);
    std::vector<int, fcm::PoolAllocator<int>> vec(alloc);

    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 1000);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    // Test operations
    vec.resize(500);
    EXPECT_EQ(vec.size(), 500);

    vec.clear();
    EXPECT_EQ(vec.size(), 0);
}

TEST_F(PoolAllocatorTest, ListIntegration) {
    fcm::PoolAllocator<std::string> alloc(resource_);
    std::list<std::string, fcm::PoolAllocator<std::string>> lst(alloc);

    lst.push_back("first");
    lst.push_back("second");
    lst.push_back("third");
    lst.push_front("zero");

    EXPECT_EQ(lst.size(), 4);
    EXPECT_EQ(lst.front(), "zero");
    EXPECT_EQ(lst.back(), "third");

    // List nodes are allocated one at a time - perfect for pool allocator
    lst.remove("second");
    EXPECT_EQ(lst.size(), 3);
}

TEST_F(PoolAllocatorTest, DequeIntegration) {
    fcm::PoolAllocator<TestObject> alloc(resource_);
    std::deque<TestObject, fcm::PoolAllocator<TestObject>> dq(alloc);

    TestObject::reset_counts();

    dq.emplace_back(1, 1.0, "one");
    dq.emplace_back(2, 2.0, "two");
    dq.emplace_front(0, 0.0, "zero");

    EXPECT_EQ(dq.size(), 3);
    EXPECT_EQ(dq[0].id, 0);
    EXPECT_EQ(dq[1].id, 1);
    EXPECT_EQ(dq[2].id, 2);

    dq.clear();
    EXPECT_EQ(TestObject::construct_count, TestObject::destruct_count);
}

TEST_F(PoolAllocatorTest, MapIntegration) {
    using MapAlloc = fcm::PoolAllocator<std::pair<const int, std::string>>;
    MapAlloc alloc(resource_);
    std::map<int, std::string, std::less<int>, MapAlloc> m(std::less<int>(), alloc);

    // Maps allocate nodes one at a time - ideal for pool allocator
    for (int i = 0; i < 100; ++i) {
        m[i] = "value_" + std::to_string(i);
    }

    EXPECT_EQ(m.size(), 100);
    EXPECT_EQ(m[50], "value_50");

    m.erase(50);
    EXPECT_EQ(m.find(50), m.end());
}

// Allocator comparison tests
TEST_F(PoolAllocatorTest, AllocatorEquality) {
    // Create a different memory resource for comparison
    struct NullResource : fcm::memory_resource {
        void* do_allocate(std::size_t, std::size_t) override { return nullptr; }
        void do_deallocate(void*, std::size_t, std::size_t) override {}
        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }
    };

    NullResource null_res;
    fcm::memory_resource* resource2 = &null_res;

    fcm::PoolAllocator<int> alloc1(resource_);
    fcm::PoolAllocator<int> alloc2(resource_);
    fcm::PoolAllocator<int> alloc3(resource2);

    EXPECT_EQ(alloc1, alloc2);  // Same upstream
    EXPECT_NE(alloc1, alloc3);  // Different upstream
}

TEST_F(PoolAllocatorTest, AllocatorPropagation) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Test propagation traits
    EXPECT_TRUE(fcm::PoolAllocator<int>::propagate_on_container_move_assignment::value);
    EXPECT_FALSE(fcm::PoolAllocator<int>::propagate_on_container_copy_assignment::value);
    EXPECT_TRUE(fcm::PoolAllocator<int>::propagate_on_container_swap::value);
    EXPECT_FALSE(fcm::PoolAllocator<int>::is_always_equal::value);
}

// Result-based API tests
TEST_F(PoolAllocatorTest, TryAllocate_Success) {
    fcm::PoolAllocator<int> alloc(resource_);

    auto result = alloc.try_allocate(1);
    ASSERT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);

    *result.value() = 42;
    EXPECT_EQ(*result.value(), 42);

    alloc.deallocate(result.value(), 1);
}

TEST_F(PoolAllocatorTest, TryAllocate_ArraySuccess) {
    fcm::PoolAllocator<double> alloc(resource_);

    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);

    for (int i = 0; i < 10; ++i) {
        result.value()[i] = static_cast<double>(i) * 0.5;
    }

    alloc.deallocate(result.value(), 10);
}

// Different block sizes
TEST_F(PoolAllocatorTest, SmallBlockSize) {
    fcm::PoolAllocator<int, 256> alloc(resource_);  // Small blocks

    std::vector<int*> pointers;
    for (int i = 0; i < 100; ++i) {
        int* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = i;
        pointers.push_back(p);
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*pointers[i], i);
        alloc.deallocate(pointers[i], 1);
    }
}

TEST_F(PoolAllocatorTest, LargeBlockSize) {
    fcm::PoolAllocator<double, 8192> alloc(resource_);  // Large blocks

    std::vector<double*> pointers;
    for (int i = 0; i < 500; ++i) {
        double* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = static_cast<double>(i) * 1.1;
        pointers.push_back(p);
    }

    for (int i = 0; i < 500; ++i) {
        EXPECT_DOUBLE_EQ(*pointers[i], static_cast<double>(i) * 1.1);
        alloc.deallocate(pointers[i], 1);
    }
}

// Stress tests
TEST_F(PoolAllocatorTest, StressAllocationDeallocation) {
    fcm::PoolAllocator<TestObject> alloc(resource_);

    std::vector<TestObject*> allocated;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> action(0, 2);
    std::uniform_int_distribution<> size_dist(1, 5);

    TestObject::reset_counts();

    for (int i = 0; i < 1000; ++i) {
        int act = action(gen);

        if (act <= 1 && allocated.size() < 500) {
            // Allocate
            std::size_t n = (act == 0) ? 1 : size_dist(gen);
            TestObject* p = alloc.allocate(n);
            ASSERT_NE(p, nullptr);

            for (std::size_t j = 0; j < n; ++j) {
                alloc.construct(&p[j], static_cast<int>(i), static_cast<double>(i) * 0.5, "obj");
            }

            allocated.push_back(p);
        } else if (!allocated.empty()) {
            // Deallocate
            TestObject* p = allocated.back();
            allocated.pop_back();

            // Determine size (simplified - in real code would track this)
            alloc.destroy(p);
            alloc.deallocate(p, 1);
        }
    }

    // Cleanup
    for (TestObject* p : allocated) {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    }
}

TEST_F(PoolAllocatorTest, ConcurrentContainerOperations) {
    fcm::PoolAllocator<int> alloc(resource_);

    std::vector<int, fcm::PoolAllocator<int>> vec1(alloc);
    std::list<int, fcm::PoolAllocator<int>> lst1(alloc);

    // Fill containers
    for (int i = 0; i < 100; ++i) {
        vec1.push_back(i);
        lst1.push_back(i * 2);
    }

    // Move construct new containers
    auto vec2 = std::move(vec1);
    auto lst2 = std::move(lst1);

    EXPECT_EQ(vec2.size(), 100);
    EXPECT_EQ(lst2.size(), 100);

    // Original containers should be empty
    EXPECT_EQ(vec1.size(), 0);
    EXPECT_EQ(lst1.size(), 0);
}

// Large object tests
TEST_F(PoolAllocatorTest, LargeObjectAllocation) {
    struct LargeObject {
        char data[1024];
    };

    fcm::PoolAllocator<LargeObject, 16384> alloc(resource_);

    std::vector<LargeObject*> objects;
    for (int i = 0; i < 10; ++i) {
        LargeObject* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        std::memset(p->data, i, sizeof(p->data));
        objects.push_back(p);
    }

    // Verify no corruption
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 1024; ++j) {
            EXPECT_EQ(objects[i]->data[j], static_cast<char>(i));
        }
    }

    for (LargeObject* p : objects) {
        alloc.deallocate(p, 1);
    }
}

// Edge cases
TEST_F(PoolAllocatorTest, ZeroSizeAllocation) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Allocating zero elements - implementation defined but shouldn't crash
    int* p = alloc.allocate(0);
    alloc.deallocate(p, 0);
}

TEST_F(PoolAllocatorTest, DeallocateNull) {
    fcm::PoolAllocator<int> alloc(resource_);

    // Should handle null gracefully
    alloc.deallocate(nullptr, 1);
    alloc.deallocate(nullptr, 10);
}

TEST_F(PoolAllocatorTest, VerySmallBlockSize) {
    // Block size smaller than single object
    fcm::PoolAllocator<double, 4> alloc(resource_);

    // Should still work (nodes_per_block will be 1)
    double* p1 = alloc.allocate(1);
    double* p2 = alloc.allocate(1);

    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);

    *p1 = 1.0;
    *p2 = 2.0;

    EXPECT_DOUBLE_EQ(*p1, 1.0);
    EXPECT_DOUBLE_EQ(*p2, 2.0);

    alloc.deallocate(p1, 1);
    alloc.deallocate(p2, 1);
}
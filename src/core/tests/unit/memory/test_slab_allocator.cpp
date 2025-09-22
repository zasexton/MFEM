#include <gtest/gtest.h>
#include <core/memory/slab_allocator.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <cstring>

namespace fcm = fem::core::memory;

class SlabAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        resource_ = fcm::new_delete_resource();
    }

    void TearDown() override {
    }

    fcm::memory_resource* resource_ = nullptr;
};

// Test object with instrumentation
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
TEST_F(SlabAllocatorTest, BasicConstruction) {
    fcm::SlabAllocator<int> alloc(resource_);

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 42;
    EXPECT_EQ(*p, 42);

    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, ConstructionWithSlabSize) {
    fcm::SlabAllocator<int> alloc(resource_, 512);  // 512 nodes per slab

    std::vector<int*> ptrs;
    for (int i = 0; i < 100; ++i) {
        int* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = i;
        ptrs.push_back(p);
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*ptrs[i], i);
    }

    for (int* p : ptrs) {
        alloc.deallocate(p, 1);
    }
}

TEST_F(SlabAllocatorTest, SharedPoolConstruction) {
    auto pool = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 128}, resource_);

    fcm::SlabAllocator<int> alloc(pool);

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 100;
    EXPECT_EQ(*p, 100);

    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, SingleAllocation) {
    fcm::SlabAllocator<double> alloc(resource_);

    double* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 3.14159;
    EXPECT_DOUBLE_EQ(*p, 3.14159);

    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, MultipleAllocations) {
    fcm::SlabAllocator<TestObject> alloc(resource_);
    TestObject::reset_counts();

    std::vector<TestObject*> objects;
    for (int i = 0; i < 50; ++i) {
        TestObject* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        alloc.construct(p, i, i * 1.5, "obj_" + std::to_string(i));
        objects.push_back(p);
    }

    EXPECT_EQ(TestObject::construct_count, 50);

    // Verify objects
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(objects[i]->id, i);
        EXPECT_DOUBLE_EQ(objects[i]->value, i * 1.5);
        EXPECT_EQ(objects[i]->name, "obj_" + std::to_string(i));
    }

    // Destroy and deallocate
    for (TestObject* p : objects) {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    }

    EXPECT_EQ(TestObject::destruct_count, 50);
}

TEST_F(SlabAllocatorTest, ArrayAllocationFallback) {
    fcm::SlabAllocator<int> alloc(resource_);

    // Array allocation should fall back to default resource
    int* arr = alloc.allocate(100);
    ASSERT_NE(arr, nullptr);

    for (int i = 0; i < 100; ++i) {
        arr[i] = i * 2;
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(arr[i], i * 2);
    }

    alloc.deallocate(arr, 100);
}

TEST_F(SlabAllocatorTest, MixedAllocationSizes) {
    fcm::SlabAllocator<int> alloc(resource_);

    // Single allocation (uses pool)
    int* single = alloc.allocate(1);
    ASSERT_NE(single, nullptr);
    *single = 42;

    // Array allocation (uses fallback)
    int* array = alloc.allocate(50);
    ASSERT_NE(array, nullptr);
    for (int i = 0; i < 50; ++i) {
        array[i] = i;
    }

    EXPECT_EQ(*single, 42);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(array[i], i);
    }

    alloc.deallocate(single, 1);
    alloc.deallocate(array, 50);
}

TEST_F(SlabAllocatorTest, ConstructDestroy) {
    struct DestructorTest {
        bool* flag;
        DestructorTest(bool* f) : flag(f) { *flag = false; }
        ~DestructorTest() { *flag = true; }
    };

    fcm::SlabAllocator<DestructorTest> alloc(resource_);

    DestructorTest* p = alloc.allocate(1);
    bool destroyed = false;
    alloc.construct(p, &destroyed);

    EXPECT_FALSE(destroyed);

    alloc.destroy(p);
    EXPECT_TRUE(destroyed);

    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, TryAllocate_Success) {
    fcm::SlabAllocator<int> alloc(resource_);

    auto result = alloc.try_allocate(1);
    ASSERT_TRUE(result.is_ok());
    int* p = result.value();
    ASSERT_NE(p, nullptr);
    *p = 999;
    EXPECT_EQ(*p, 999);

    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, TryAllocate_ArraySuccess) {
    fcm::SlabAllocator<double> alloc(resource_);

    auto result = alloc.try_allocate(100);
    ASSERT_TRUE(result.is_ok());
    double* arr = result.value();
    ASSERT_NE(arr, nullptr);

    for (int i = 0; i < 100; ++i) {
        arr[i] = i * 0.5;
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], i * 0.5);
    }

    alloc.deallocate(arr, 100);
}

// Rebinding tests
TEST_F(SlabAllocatorTest, Rebinding) {
    auto pool = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 256}, resource_);

    fcm::SlabAllocator<int> int_alloc(pool);
    fcm::SlabAllocator<double> double_alloc(int_alloc);

    // They should NOT share the same pool since sizes differ
    EXPECT_NE(int_alloc, double_alloc);

    double* p = double_alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 2.718;
    EXPECT_DOUBLE_EQ(*p, 2.718);

    double_alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, RebindingSameSize) {
    // Test rebinding between types of the same size/alignment
    auto pool = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 256}, resource_);

    fcm::SlabAllocator<int> int_alloc(pool);
    fcm::SlabAllocator<float> float_alloc(int_alloc);  // Same size as int on most platforms

    if (sizeof(int) == sizeof(float) && alignof(int) == alignof(float)) {
        // They should share the same pool since size/alignment match
        EXPECT_EQ(int_alloc, float_alloc);
    } else {
        // Different size/alignment - different pools
        EXPECT_NE(int_alloc, float_alloc);
    }

    float* p = float_alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 3.14f;
    EXPECT_FLOAT_EQ(*p, 3.14f);

    float_alloc.deallocate(p, 1);
}

// Allocator traits tests
TEST_F(SlabAllocatorTest, AllocatorEquality) {
    auto pool1 = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 128}, resource_);
    auto pool2 = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 128}, resource_);

    fcm::SlabAllocator<int> alloc1(pool1);
    fcm::SlabAllocator<int> alloc2(pool1);  // Same pool
    fcm::SlabAllocator<int> alloc3(pool2);  // Different pool

    EXPECT_EQ(alloc1, alloc2);
    EXPECT_NE(alloc1, alloc3);
}

TEST_F(SlabAllocatorTest, AllocatorPropagation) {
    fcm::SlabAllocator<int> alloc(resource_);

    using traits = std::allocator_traits<fcm::SlabAllocator<int>>;

    EXPECT_FALSE(traits::propagate_on_container_copy_assignment::value);
    EXPECT_TRUE(traits::propagate_on_container_move_assignment::value);
    EXPECT_TRUE(traits::propagate_on_container_swap::value);
    EXPECT_FALSE(traits::is_always_equal::value);
}

// Container integration tests
TEST_F(SlabAllocatorTest, VectorIntegration) {
    fcm::SlabAllocator<int> alloc(resource_);
    std::vector<int, fcm::SlabAllocator<int>> vec(alloc);

    for (int i = 0; i < 200; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 200);

    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(SlabAllocatorTest, ListIntegration) {
    fcm::SlabAllocator<std::string> alloc(resource_);
    std::list<std::string, fcm::SlabAllocator<std::string>> lst(alloc);

    for (int i = 0; i < 100; ++i) {
        lst.push_back("item_" + std::to_string(i));
    }

    EXPECT_EQ(lst.size(), 100);

    int i = 0;
    for (const auto& item : lst) {
        EXPECT_EQ(item, "item_" + std::to_string(i++));
    }
}

TEST_F(SlabAllocatorTest, DequeIntegration) {
    fcm::SlabAllocator<double> alloc(resource_);
    std::deque<double, fcm::SlabAllocator<double>> dq(alloc);

    for (int i = 0; i < 150; ++i) {
        if (i % 2 == 0) {
            dq.push_back(i * 1.1);
        } else {
            dq.push_front(i * 1.1);
        }
    }

    EXPECT_EQ(dq.size(), 150);
}

TEST_F(SlabAllocatorTest, MapIntegration) {
    using PairAlloc = fcm::SlabAllocator<std::pair<const int, std::string>>;
    PairAlloc alloc(resource_);

    std::map<int, std::string, std::less<int>, PairAlloc> m(alloc);

    for (int i = 0; i < 100; ++i) {
        m[i] = "value_" + std::to_string(i);
    }

    EXPECT_EQ(m.size(), 100);

    for (int i = 0; i < 100; ++i) {
        auto it = m.find(i);
        ASSERT_NE(it, m.end());
        EXPECT_EQ(it->second, "value_" + std::to_string(i));
    }
}

// Stress tests
TEST_F(SlabAllocatorTest, StressAllocationDeallocation) {
    fcm::SlabAllocator<TestObject> alloc(resource_, 64);  // Small slab size
    TestObject::reset_counts();

    std::vector<TestObject*> objects;

    // Allocate many objects (will require multiple slabs)
    for (int i = 0; i < 500; ++i) {
        TestObject* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        alloc.construct(p, i, i * 0.5, "stress_" + std::to_string(i));
        objects.push_back(p);
    }

    EXPECT_EQ(TestObject::construct_count, 500);

    // Random deallocation
    std::random_shuffle(objects.begin(), objects.end());

    for (TestObject* p : objects) {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    }

    EXPECT_EQ(TestObject::destruct_count, 500);
}

TEST_F(SlabAllocatorTest, RapidReusePattern) {
    fcm::SlabAllocator<int> alloc(resource_, 32);

    for (int cycle = 0; cycle < 100; ++cycle) {
        std::vector<int*> batch;

        // Allocate batch
        for (int i = 0; i < 32; ++i) {
            int* p = alloc.allocate(1);
            ASSERT_NE(p, nullptr);
            *p = cycle * 100 + i;
            batch.push_back(p);
        }

        // Verify
        for (int i = 0; i < 32; ++i) {
            EXPECT_EQ(*batch[i], cycle * 100 + i);
        }

        // Deallocate batch
        for (int* p : batch) {
            alloc.deallocate(p, 1);
        }
    }
}

TEST_F(SlabAllocatorTest, MixedSingleAndBulkOperations) {
    fcm::SlabAllocator<char> alloc(resource_);

    std::vector<std::pair<char*, std::size_t>> allocations;

    // Mix single and bulk allocations
    for (int i = 0; i < 50; ++i) {
        std::size_t size = (i % 5 == 0) ? 100 : 1;  // Every 5th is bulk
        char* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);
        std::memset(p, static_cast<int>('A' + (i % 26)), size);
        allocations.push_back({p, size});
    }

    // Verify data
    for (std::size_t i = 0; i < allocations.size(); ++i) {
        auto [ptr, size] = allocations[i];
        char expected = static_cast<char>('A' + (i % 26));
        for (std::size_t j = 0; j < size; ++j) {
            EXPECT_EQ(ptr[j], expected);
        }
    }

    // Deallocate
    for (auto [ptr, size] : allocations) {
        alloc.deallocate(ptr, size);
    }
}

// Large object tests
TEST_F(SlabAllocatorTest, LargeObjectAllocation) {
    struct LargeObject {
        char data[4096];
        int id;

        LargeObject(int i) : id(i) {
            std::memset(data, static_cast<int>(i % 256), sizeof(data));
        }
    };

    fcm::SlabAllocator<LargeObject> alloc(resource_, 16);  // Few large objects per slab

    std::vector<LargeObject*> objects;
    for (int i = 0; i < 20; ++i) {
        LargeObject* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        alloc.construct(p, i);
        objects.push_back(p);
    }

    // Verify data integrity
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(objects[i]->id, i);
        for (int j = 0; j < 4096; ++j) {
            EXPECT_EQ(objects[i]->data[j], static_cast<char>(i % 256));
        }
    }

    for (LargeObject* p : objects) {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    }
}

// Edge cases
TEST_F(SlabAllocatorTest, ZeroSizeAllocation) {
    fcm::SlabAllocator<int> alloc(resource_);

    // Allocating zero elements
    int* p = alloc.allocate(0);
    alloc.deallocate(p, 0);
}

TEST_F(SlabAllocatorTest, DeallocateNull) {
    fcm::SlabAllocator<int> alloc(resource_);

    // Should handle null gracefully
    alloc.deallocate(nullptr, 1);
    alloc.deallocate(nullptr, 100);
}

TEST_F(SlabAllocatorTest, TrivialTypeOptimization) {
    struct TrivialType {
        int x;
        double y;
    };

    fcm::SlabAllocator<TrivialType> alloc(resource_);

    TrivialType* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    // Should work with trivial types
    p->x = 42;
    p->y = 3.14;

    alloc.destroy(p);  // Should be no-op for trivial destructor
    alloc.deallocate(p, 1);
}

TEST_F(SlabAllocatorTest, SharedPoolMultipleAllocators) {
    auto pool = std::make_shared<fcm::MemoryPool>(
        fcm::MemoryPool::Config{sizeof(int), alignof(int), 64}, resource_);

    fcm::SlabAllocator<int> alloc1(pool);
    fcm::SlabAllocator<int> alloc2(pool);

    // Both allocators use the same pool
    std::vector<int*> ptrs1, ptrs2;

    for (int i = 0; i < 30; ++i) {
        int* p1 = alloc1.allocate(1);
        *p1 = i;
        ptrs1.push_back(p1);

        int* p2 = alloc2.allocate(1);
        *p2 = i + 100;
        ptrs2.push_back(p2);
    }

    // Verify all allocations
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(*ptrs1[i], i);
        EXPECT_EQ(*ptrs2[i], i + 100);
    }

    // Deallocate
    for (int* p : ptrs1) {
        alloc1.deallocate(p, 1);
    }
    for (int* p : ptrs2) {
        alloc2.deallocate(p, 1);
    }
}
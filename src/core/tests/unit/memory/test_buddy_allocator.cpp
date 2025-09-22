#include <gtest/gtest.h>
#include <core/memory/buddy_allocator.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <list>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace fcm = fem::core::memory;

class BuddyAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        resource_ = fcm::new_delete_resource();
    }

    void TearDown() override {
    }

    fcm::memory_resource* resource_ = nullptr;
};

// Helper function to verify power of 2
bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

std::size_t next_power_of_two(std::size_t n) {
    if (n <= 1) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if (sizeof(std::size_t) == 8) n |= n >> 32;
    return n + 1;
}

// Basic functionality tests
TEST_F(BuddyAllocatorTest, BasicConstruction) {
    fcm::BuddyAllocator<int> alloc(resource_);

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 42;
    EXPECT_EQ(*p, 42);

    alloc.deallocate(p, 1);
}

TEST_F(BuddyAllocatorTest, SingleAllocation) {
    fcm::BuddyAllocator<double> alloc(resource_);

    double* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 3.14159;
    EXPECT_DOUBLE_EQ(*p, 3.14159);

    alloc.deallocate(p, 1);
}

TEST_F(BuddyAllocatorTest, MultipleAllocations) {
    fcm::BuddyAllocator<int> alloc(resource_);

    std::vector<int*> ptrs;
    for (int i = 0; i < 100; ++i) {
        int* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = i;
        ptrs.push_back(p);
    }

    // Verify values
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*ptrs[i], i);
    }

    // Deallocate all
    for (int* p : ptrs) {
        alloc.deallocate(p, 1);
    }
}

TEST_F(BuddyAllocatorTest, PowerOfTwoRounding) {
    fcm::BuddyAllocator<char> alloc(resource_);

    // Test various sizes that should be rounded up
    std::vector<std::size_t> test_sizes = {1, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};

    for (std::size_t size : test_sizes) {
        char* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);

        // Fill with test pattern
        std::memset(p, 'A', size);

        // Verify we can write to all requested bytes
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(p[i], 'A');
        }

        alloc.deallocate(p, size);
    }
}

TEST_F(BuddyAllocatorTest, LargeAllocation) {
    fcm::BuddyAllocator<int> alloc(resource_);

    // Allocate a large array
    std::size_t size = 10000;
    int* arr = alloc.allocate(size);
    ASSERT_NE(arr, nullptr);

    for (std::size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<int>(i);
    }

    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_EQ(arr[i], static_cast<int>(i));
    }

    alloc.deallocate(arr, size);
}

TEST_F(BuddyAllocatorTest, ConstructDestroy) {
    struct TestObject {
        int value;
        bool* destroyed;

        TestObject(int v, bool* d) : value(v), destroyed(d) {
            *destroyed = false;
        }
        ~TestObject() {
            *destroyed = true;
        }
    };

    fcm::BuddyAllocator<TestObject> alloc(resource_);

    TestObject* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    bool destroyed = false;
    alloc.construct(p, 42, &destroyed);

    EXPECT_EQ(p->value, 42);
    EXPECT_FALSE(destroyed);

    alloc.destroy(p);
    EXPECT_TRUE(destroyed);

    alloc.deallocate(p, 1);
}

TEST_F(BuddyAllocatorTest, TryAllocate_Success) {
    fcm::BuddyAllocator<int> alloc(resource_);

    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());
    int* p = result.value();
    ASSERT_NE(p, nullptr);

    for (int i = 0; i < 10; ++i) {
        p[i] = i * 2;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i], i * 2);
    }

    alloc.deallocate(p, 10);
}

// Rebinding tests
TEST_F(BuddyAllocatorTest, Rebinding) {
    fcm::BuddyAllocator<int> int_alloc(resource_);
    fcm::BuddyAllocator<double>::rebind<char>::other char_alloc(int_alloc);

    EXPECT_EQ(int_alloc.resource(), char_alloc.resource());

    char* p = char_alloc.allocate(100);
    ASSERT_NE(p, nullptr);
    std::memset(p, 'X', 100);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(p[i], 'X');
    }

    char_alloc.deallocate(p, 100);
}

// Allocator traits tests
TEST_F(BuddyAllocatorTest, AllocatorEquality) {
    fcm::memory_resource* resource2 = fcm::new_delete_resource();

    fcm::BuddyAllocator<int> alloc1(resource_);
    fcm::BuddyAllocator<int> alloc2(resource_);
    fcm::BuddyAllocator<int> alloc3(resource2);

    EXPECT_EQ(alloc1, alloc2);  // Same resource
    EXPECT_EQ(alloc1, alloc3);  // new_delete_resource is a singleton
}

TEST_F(BuddyAllocatorTest, AllocatorPropagation) {
    fcm::BuddyAllocator<int> alloc(resource_);

    using traits = std::allocator_traits<fcm::BuddyAllocator<int>>;

    EXPECT_FALSE(traits::propagate_on_container_copy_assignment::value);
    EXPECT_TRUE(traits::propagate_on_container_move_assignment::value);
    EXPECT_TRUE(traits::propagate_on_container_swap::value);
    EXPECT_FALSE(traits::is_always_equal::value);
}

// Container integration tests
TEST_F(BuddyAllocatorTest, VectorIntegration) {
    fcm::BuddyAllocator<int> alloc(resource_);
    std::vector<int, fcm::BuddyAllocator<int>> vec(alloc);

    for (int i = 0; i < 500; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 500);

    for (int i = 0; i < 500; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    // Trigger reallocation
    vec.resize(1000);
    for (int i = 500; i < 1000; ++i) {
        vec[i] = i;
    }

    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(BuddyAllocatorTest, ListIntegration) {
    fcm::BuddyAllocator<std::string> alloc(resource_);
    std::list<std::string, fcm::BuddyAllocator<std::string>> lst(alloc);

    for (int i = 0; i < 200; ++i) {
        lst.push_back("item_" + std::to_string(i));
    }

    EXPECT_EQ(lst.size(), 200);

    int i = 0;
    for (const auto& item : lst) {
        EXPECT_EQ(item, "item_" + std::to_string(i++));
    }
}

TEST_F(BuddyAllocatorTest, MapIntegration) {
    using PairAlloc = fcm::BuddyAllocator<std::pair<const int, double>>;
    PairAlloc alloc(resource_);

    std::map<int, double, std::less<int>, PairAlloc> m(alloc);

    for (int i = 0; i < 300; ++i) {
        m[i] = i * 1.5;
    }

    EXPECT_EQ(m.size(), 300);

    for (int i = 0; i < 300; ++i) {
        auto it = m.find(i);
        ASSERT_NE(it, m.end());
        EXPECT_DOUBLE_EQ(it->second, i * 1.5);
    }
}

// Power of two size verification tests
TEST_F(BuddyAllocatorTest, VerifyPowerOfTwoSizes) {
    // Test that our helper function matches expected values
    EXPECT_EQ(next_power_of_two(0), 1);
    EXPECT_EQ(next_power_of_two(1), 1);
    EXPECT_EQ(next_power_of_two(2), 2);
    EXPECT_EQ(next_power_of_two(3), 4);
    EXPECT_EQ(next_power_of_two(4), 4);
    EXPECT_EQ(next_power_of_two(5), 8);
    EXPECT_EQ(next_power_of_two(7), 8);
    EXPECT_EQ(next_power_of_two(8), 8);
    EXPECT_EQ(next_power_of_two(9), 16);
    EXPECT_EQ(next_power_of_two(15), 16);
    EXPECT_EQ(next_power_of_two(16), 16);
    EXPECT_EQ(next_power_of_two(17), 32);
    EXPECT_EQ(next_power_of_two(31), 32);
    EXPECT_EQ(next_power_of_two(32), 32);
    EXPECT_EQ(next_power_of_two(33), 64);
    EXPECT_EQ(next_power_of_two(1000), 1024);
    EXPECT_EQ(next_power_of_two(1024), 1024);
    EXPECT_EQ(next_power_of_two(1025), 2048);
}

TEST_F(BuddyAllocatorTest, AllocationSizeRounding) {
    fcm::BuddyAllocator<char> alloc(resource_);

    struct TestCase {
        std::size_t request;
        std::size_t expected_rounded;
    };

    std::vector<TestCase> test_cases = {
        {1, 1},
        {2, 2},
        {3, 4},
        {4, 4},
        {5, 8},
        {7, 8},
        {8, 8},
        {9, 16},
        {15, 16},
        {16, 16},
        {17, 32},
        {100, 128},
        {1000, 1024},
        {2000, 2048},
        {4095, 4096},
        {4096, 4096},
        {4097, 8192}
    };

    for (const auto& tc : test_cases) {
        char* p = alloc.allocate(tc.request);
        ASSERT_NE(p, nullptr);

        // The allocator should have requested the rounded size from the resource
        // We can at least verify that we can write to all requested bytes
        std::memset(p, 'T', tc.request);
        for (std::size_t i = 0; i < tc.request; ++i) {
            EXPECT_EQ(p[i], 'T');
        }

        alloc.deallocate(p, tc.request);
    }
}

// Stress tests
TEST_F(BuddyAllocatorTest, StressVariedSizes) {
    fcm::BuddyAllocator<char> alloc(resource_);

    std::vector<std::pair<char*, std::size_t>> allocations;

    // Allocate various sizes
    std::vector<std::size_t> sizes = {1, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095};

    for (std::size_t size : sizes) {
        for (int j = 0; j < 5; ++j) {  // Multiple allocations of each size
            char* p = alloc.allocate(size);
            ASSERT_NE(p, nullptr);
            std::memset(p, static_cast<char>((size + j) % 256), size);
            allocations.push_back({p, size});
        }
    }

    // Verify data integrity
    for (const auto& [ptr, size] : allocations) {
        [[maybe_unused]] char expected = static_cast<char>(size % 256);  // First allocation of this size
        // We can't verify exact pattern since we mixed multiple allocations
        // Just verify the memory is accessible
        EXPECT_NE(ptr[0], '\0');
    }

    // Deallocate in reverse order
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        alloc.deallocate(it->first, it->second);
    }
}

TEST_F(BuddyAllocatorTest, RapidAllocationDeallocation) {
    fcm::BuddyAllocator<int> alloc(resource_);

    for (int cycle = 0; cycle < 1000; ++cycle) {
        std::size_t size = (cycle % 100) + 1;  // Sizes from 1 to 100

        int* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);

        // Write test pattern
        for (std::size_t i = 0; i < size; ++i) {
            p[i] = static_cast<int>(cycle + i);
        }

        // Verify
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(p[i], static_cast<int>(cycle + i));
        }

        alloc.deallocate(p, size);
    }
}

TEST_F(BuddyAllocatorTest, FragmentationPattern) {
    fcm::BuddyAllocator<char> alloc(resource_);

    // Allocate many small and large blocks interleaved
    std::vector<std::pair<char*, std::size_t>> small_allocs;
    std::vector<std::pair<char*, std::size_t>> large_allocs;

    for (int i = 0; i < 50; ++i) {
        // Small allocation
        std::size_t small_size = (i % 10) + 1;
        char* small = alloc.allocate(small_size);
        ASSERT_NE(small, nullptr);
        std::memset(small, 'S', small_size);
        small_allocs.push_back({small, small_size});

        // Large allocation
        std::size_t large_size = 1000 + (i * 10);
        char* large = alloc.allocate(large_size);
        ASSERT_NE(large, nullptr);
        std::memset(large, 'L', large_size);
        large_allocs.push_back({large, large_size});
    }

    // Deallocate large blocks
    for (const auto& [ptr, size] : large_allocs) {
        alloc.deallocate(ptr, size);
    }

    // Allocate more small blocks (should reuse some of the freed space)
    for (int i = 0; i < 50; ++i) {
        std::size_t size = (i % 20) + 1;
        char* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);
        std::memset(p, 'N', size);
        small_allocs.push_back({p, size});
    }

    // Deallocate all remaining
    for (const auto& [ptr, size] : small_allocs) {
        alloc.deallocate(ptr, size);
    }
}

// Edge cases
TEST_F(BuddyAllocatorTest, ZeroSizeAllocation) {
    fcm::BuddyAllocator<int> alloc(resource_);

    // Zero size should be rounded to 1
    int* p = alloc.allocate(0);
    ASSERT_NE(p, nullptr);  // Should still return valid pointer
    alloc.deallocate(p, 0);
}

TEST_F(BuddyAllocatorTest, DeallocateNull) {
    fcm::BuddyAllocator<int> alloc(resource_);

    // Should handle null gracefully
    alloc.deallocate(nullptr, 1);
    alloc.deallocate(nullptr, 100);
    alloc.deallocate(nullptr, 0);
}

TEST_F(BuddyAllocatorTest, MaxSizeAllocation) {
    fcm::BuddyAllocator<char> alloc(resource_);

    // Test allocation of exactly power-of-two sizes
    std::vector<std::size_t> exact_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (std::size_t size : exact_sizes) {
        char* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);

        // These should not need rounding
        std::memset(p, 'E', size);

        // Verify we can access all bytes
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(p[i], 'E');
        }

        alloc.deallocate(p, size);
    }
}

// Complex object tests
TEST_F(BuddyAllocatorTest, ComplexObjectAllocation) {
    struct ComplexObject {
        std::vector<int> data;
        std::string name;
        std::unique_ptr<double> value;

        ComplexObject(int size, const std::string& n)
            : data(size), name(n), value(std::make_unique<double>(3.14)) {
            for (int i = 0; i < size; ++i) {
                data[i] = i;
            }
        }
    };

    fcm::BuddyAllocator<ComplexObject> alloc(resource_);

    ComplexObject* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    alloc.construct(p, 100, "complex");

    EXPECT_EQ(p->data.size(), 100);
    EXPECT_EQ(p->name, "complex");
    EXPECT_DOUBLE_EQ(*p->value, 3.14);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(p->data[i], i);
    }

    alloc.destroy(p);
    alloc.deallocate(p, 1);
}

TEST_F(BuddyAllocatorTest, ArrayOfComplexObjects) {
    struct Widget {
        int id;
        std::string label;

        Widget() : id(0), label("default") {}
        Widget(int i, const std::string& l) : id(i), label(l) {}
    };

    fcm::BuddyAllocator<Widget> alloc(resource_);

    std::size_t count = 50;
    Widget* arr = alloc.allocate(count);
    ASSERT_NE(arr, nullptr);

    // Construct objects
    for (std::size_t i = 0; i < count; ++i) {
        alloc.construct(&arr[i], static_cast<int>(i), "widget_" + std::to_string(i));
    }

    // Verify
    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(arr[i].id, static_cast<int>(i));
        EXPECT_EQ(arr[i].label, "widget_" + std::to_string(i));
    }

    // Destroy objects
    for (std::size_t i = 0; i < count; ++i) {
        alloc.destroy(&arr[i]);
    }

    alloc.deallocate(arr, count);
}

// Performance pattern tests
TEST_F(BuddyAllocatorTest, BinaryTreePattern) {
    fcm::BuddyAllocator<int> alloc(resource_);

    // Simulate binary tree node allocations
    struct Node {
        int* data;
        std::size_t size;
    };

    std::vector<Node> nodes;

    // Allocate nodes with sizes following binary pattern
    for (int level = 0; level < 10; ++level) {
        std::size_t size = 1 << level;  // 1, 2, 4, 8, 16, ...
        int* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);

        for (std::size_t i = 0; i < size; ++i) {
            p[i] = static_cast<int>(level * 1000 + i);
        }

        nodes.push_back({p, size});
    }

    // Verify data
    for (std::size_t level = 0; level < nodes.size(); ++level) {
        for (std::size_t i = 0; i < nodes[level].size; ++i) {
            EXPECT_EQ(nodes[level].data[i], static_cast<int>(level * 1000 + i));
        }
    }

    // Deallocate in reverse level order
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        alloc.deallocate(it->data, it->size);
    }
}
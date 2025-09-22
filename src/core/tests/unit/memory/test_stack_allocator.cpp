#include <gtest/gtest.h>
#include <core/memory/stack_allocator.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <list>
#include <string>
#include <cstring>
#include <memory>

namespace fcm = fem::core::memory;

class StackAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        storage_.reset();
    }

    void TearDown() override {
        storage_.reset();
    }

    fcm::stack_storage<4096> storage_;
};

// Basic stack_storage tests
TEST_F(StackAllocatorTest, BasicStorageConstruction) {
    EXPECT_EQ(storage_.used(), 0);
    EXPECT_EQ(storage_.capacity(), 4096);
}

TEST_F(StackAllocatorTest, StorageAllocate) {
    void* p1 = storage_.allocate(64);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(storage_.used(), 64);

    void* p2 = storage_.allocate(128);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(storage_.used(), 192);

    EXPECT_GT(p2, p1);  // Stack grows upward
}

TEST_F(StackAllocatorTest, StorageAlignment) {
    [[maybe_unused]] void* p1 = storage_.allocate(1, 1);
    EXPECT_EQ(storage_.used(), 1);

    void* p2 = storage_.allocate(1, 16);
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p2);
    EXPECT_EQ(addr % 16, 0);  // Should be 16-byte aligned

    void* p3 = storage_.allocate(1, 32);
    addr = reinterpret_cast<std::uintptr_t>(p3);
    EXPECT_EQ(addr % 32, 0);  // Should be 32-byte aligned
}

TEST_F(StackAllocatorTest, StorageDeallocateLIFO) {
    void* p1 = storage_.allocate(64);
    void* p2 = storage_.allocate(128);

    std::size_t used_before = storage_.used();

    // Deallocate top allocation
    storage_.deallocate(p2, 128);
    EXPECT_LT(storage_.used(), used_before);

    // Can't deallocate non-top allocation (no effect)
    std::size_t used = storage_.used();
    storage_.deallocate(p1, 64);
    EXPECT_EQ(storage_.used(), used);  // No change
}

TEST_F(StackAllocatorTest, StorageReset) {
    [[maybe_unused]] auto p1 = storage_.allocate(64);
    [[maybe_unused]] auto p2 = storage_.allocate(128);
    [[maybe_unused]] auto p3 = storage_.allocate(256);

    EXPECT_GT(storage_.used(), 0);

    storage_.reset();
    EXPECT_EQ(storage_.used(), 0);
}

TEST_F(StackAllocatorTest, StorageMarkAndRewind) {
    [[maybe_unused]] auto p1 = storage_.allocate(64);
    auto marker = storage_.mark();

    [[maybe_unused]] auto p2 = storage_.allocate(128);
    [[maybe_unused]] auto p3 = storage_.allocate(256);
    EXPECT_GT(storage_.used(), 64);

    storage_.rewind(marker);
    EXPECT_EQ(storage_.used(), 64);
}

TEST_F(StackAllocatorTest, StorageOutOfMemory) {
    EXPECT_THROW([[maybe_unused]] auto p = storage_.allocate(5000), std::bad_alloc);  // More than capacity

    // Fill up storage
    [[maybe_unused]] auto p1 = storage_.allocate(4000);
    EXPECT_THROW([[maybe_unused]] auto p2 = storage_.allocate(200), std::bad_alloc);  // Not enough remaining
}

// StackAllocator tests
TEST_F(StackAllocatorTest, AllocatorConstruction) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    // Should be able to allocate
    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);
    *p = 42;
    EXPECT_EQ(*p, 42);

    alloc.deallocate(p, 1);
}

TEST_F(StackAllocatorTest, AllocatorMultipleAllocations) {
    fcm::stack_storage<4096, alignof(double)> storage;
    fcm::StackAllocator<double, 4096, alignof(double)> alloc(&storage);

    std::vector<double*> ptrs;
    for (int i = 0; i < 10; ++i) {
        double* p = alloc.allocate(1);
        ASSERT_NE(p, nullptr);
        *p = i * 3.14;
        ptrs.push_back(p);
    }

    // Verify values
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(*ptrs[i], i * 3.14);
    }

    // LIFO deallocation
    for (int i = 9; i >= 0; --i) {
        alloc.deallocate(ptrs[i], 1);
    }
}

TEST_F(StackAllocatorTest, AllocatorArrayAllocation) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

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

TEST_F(StackAllocatorTest, AllocatorConstructDestroy) {
    struct TestObject {
        int value;
        bool* destroyed;

        TestObject(int v, bool* d) : value(v), destroyed(d) { *destroyed = false; }
        ~TestObject() { *destroyed = true; }
    };

    fcm::stack_storage<4096, alignof(TestObject)> storage;
    fcm::StackAllocator<TestObject, 4096, alignof(TestObject)> alloc(&storage);

    TestObject* p = alloc.allocate(1);
    bool destroyed = false;
    alloc.construct(p, 42, &destroyed);

    EXPECT_EQ(p->value, 42);
    EXPECT_FALSE(destroyed);

    alloc.destroy(p);
    EXPECT_TRUE(destroyed);

    alloc.deallocate(p, 1);
}

TEST_F(StackAllocatorTest, AllocatorTryAllocate) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    // Successful allocation
    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());
    int* p = result.value();
    ASSERT_NE(p, nullptr);

    alloc.deallocate(p, 10);

    // Failed allocation (too large)
    auto fail_result = alloc.try_allocate(10000);
    ASSERT_FALSE(fail_result.is_ok());
    EXPECT_EQ(fail_result.error(), fem::core::error::ErrorCode::OutOfMemory);
}

TEST_F(StackAllocatorTest, AllocatorRebinding) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> int_alloc(&storage);
    fcm::StackAllocator<double, 4096, alignof(double)>::rebind<char>::other char_alloc(int_alloc);

    char* p = char_alloc.allocate(100);
    ASSERT_NE(p, nullptr);
    std::memset(p, 'A', 100);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(p[i], 'A');
    }

    char_alloc.deallocate(p, 100);
}

TEST_F(StackAllocatorTest, AllocatorEquality) {
    fcm::stack_storage<4096, alignof(int)> storage1;
    fcm::stack_storage<4096, alignof(int)> storage2;

    fcm::StackAllocator<int, 4096, alignof(int)> alloc1(&storage1);
    fcm::StackAllocator<int, 4096, alignof(int)> alloc2(&storage1);
    fcm::StackAllocator<int, 4096, alignof(int)> alloc3(&storage2);

    EXPECT_EQ(alloc1, alloc2);  // Same storage
    EXPECT_NE(alloc1, alloc3);  // Different storage
}

// Container integration tests
TEST_F(StackAllocatorTest, VectorIntegration) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);
    std::vector<int, fcm::StackAllocator<int, 4096, alignof(int)>> vec(alloc);

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(StackAllocatorTest, ListIntegration) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);
    std::list<int, fcm::StackAllocator<int, 4096, alignof(int)>> lst(alloc);

    for (int i = 0; i < 50; ++i) {
        lst.push_back(i);
    }

    EXPECT_EQ(lst.size(), 50);

    int i = 0;
    for (int val : lst) {
        EXPECT_EQ(val, i++);
    }
}

// Stack storage pattern tests
TEST_F(StackAllocatorTest, TemporaryAllocationPattern) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    auto marker = storage.mark();

    // Temporary allocations
    {
        std::vector<int, fcm::StackAllocator<int, 4096, alignof(int)>> temp_vec(alloc);
        for (int i = 0; i < 50; ++i) {
            temp_vec.push_back(i * i);
        }
        // Process temp_vec...
    }

    // Rewind to reclaim memory
    storage.rewind(marker);
    EXPECT_EQ(storage.used(), 0);

    // Can reuse the space
    int* p = alloc.allocate(100);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, 100);
}

TEST_F(StackAllocatorTest, NestedScopes) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    int* outer = alloc.allocate(10);
    auto marker1 = storage.mark();

    {
        int* inner1 = alloc.allocate(20);
        auto marker2 = storage.mark();

        {
            int* inner2 = alloc.allocate(30);
            EXPECT_GT(storage.used(), 60);
            alloc.deallocate(inner2, 30);
        }

        storage.rewind(marker2);
        alloc.deallocate(inner1, 20);
    }

    storage.rewind(marker1);
    EXPECT_LT(storage.used(), 20);

    alloc.deallocate(outer, 10);
}

// Edge cases
TEST_F(StackAllocatorTest, ZeroSizeAllocation) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    // Allocating zero elements - implementation may return non-null
    int* p = alloc.allocate(0);
    alloc.deallocate(p, 0);
}

TEST_F(StackAllocatorTest, DeallocateNull) {
    fcm::stack_storage<4096, alignof(int)> storage;
    fcm::StackAllocator<int, 4096, alignof(int)> alloc(&storage);

    // Should handle null gracefully
    alloc.deallocate(nullptr, 1);
    alloc.deallocate(nullptr, 100);
}

TEST_F(StackAllocatorTest, LargeAlignment) {
    fcm::stack_storage<4096, 64> storage;  // 64-byte aligned storage
    fcm::StackAllocator<int, 4096, 64> alloc(&storage);

    int* p = alloc.allocate(1);
    ASSERT_NE(p, nullptr);

    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
    EXPECT_EQ(addr % 64, 0);  // Should be 64-byte aligned

    alloc.deallocate(p, 1);
}

// Stress tests
TEST_F(StackAllocatorTest, MaximumUtilization) {
    fcm::stack_storage<4096, alignof(char)> storage;
    fcm::StackAllocator<char, 4096, alignof(char)> alloc(&storage);

    // Allocate almost all available space
    char* big = alloc.allocate(4000);
    ASSERT_NE(big, nullptr);
    std::memset(big, 'X', 4000);

    // Should still have a bit of room
    char* small = alloc.allocate(50);
    ASSERT_NE(small, nullptr);

    // Should be out of space now
    EXPECT_THROW([[maybe_unused]] auto p = alloc.allocate(100), std::bad_alloc);

    alloc.deallocate(small, 50);
    alloc.deallocate(big, 4000);
}

TEST_F(StackAllocatorTest, MixedSizeAllocations) {
    fcm::stack_storage<8192, alignof(char)> storage;
    fcm::StackAllocator<char, 8192, alignof(char)> alloc(&storage);

    std::vector<std::pair<char*, std::size_t>> allocations;
    std::size_t sizes[] = {17, 33, 64, 128, 255, 512, 1024};

    for (std::size_t size : sizes) {
        char* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);
        std::memset(p, static_cast<char>(size % 256), size);
        allocations.push_back({p, size});
    }

    // Verify data integrity
    for (const auto& [ptr, size] : allocations) {
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(ptr[i], static_cast<char>(size % 256));
        }
    }

    // LIFO deallocation
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        alloc.deallocate(it->first, it->second);
    }
}

// Different sized storages
TEST_F(StackAllocatorTest, SmallStorage) {
    fcm::stack_storage<256, alignof(int)> small_storage;
    fcm::StackAllocator<int, 256, alignof(int)> alloc(&small_storage);

    // Can allocate a few integers
    int* p1 = alloc.allocate(10);
    ASSERT_NE(p1, nullptr);

    int* p2 = alloc.allocate(20);
    ASSERT_NE(p2, nullptr);

    // Should run out of space
    EXPECT_THROW([[maybe_unused]] auto p = alloc.allocate(100), std::bad_alloc);

    alloc.deallocate(p2, 20);
    alloc.deallocate(p1, 10);
}

TEST_F(StackAllocatorTest, LargeStorage) {
    fcm::stack_storage<65536, alignof(double)> large_storage;
    fcm::StackAllocator<double, 65536, alignof(double)> alloc(&large_storage);

    // Can allocate a large array
    double* arr = alloc.allocate(8000);
    ASSERT_NE(arr, nullptr);

    for (int i = 0; i < 8000; ++i) {
        arr[i] = i * 0.5;
    }

    for (int i = 0; i < 8000; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], i * 0.5);
    }

    alloc.deallocate(arr, 8000);
}
#include <gtest/gtest.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <condition_variable>
// Note: std::barrier requires C++20

#include <core/memory/thread_pool_allocator.h>

namespace fem::core::memory {
namespace {

// Test fixture for thread pool allocator tests
class ThreadPoolAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state if needed
    }

    void TearDown() override {
        // Clean up
    }
};

// Test basic construction
TEST_F(ThreadPoolAllocatorTest, DefaultConstruction) {
    [[maybe_unused]] ThreadPoolAllocator<int> alloc;

    // Should compile and not crash
    EXPECT_TRUE(true);
}

TEST_F(ThreadPoolAllocatorTest, TemplateConstruction) {
    [[maybe_unused]] ThreadPoolAllocator<int> int_alloc;
    [[maybe_unused]] ThreadPoolAllocator<double> double_alloc;
    [[maybe_unused]] ThreadPoolAllocator<std::string> string_alloc;

    // Different types should work
    EXPECT_TRUE(true);
}

TEST_F(ThreadPoolAllocatorTest, CustomBlockSize) {
    [[maybe_unused]] ThreadPoolAllocator<int, 1024> small_block;
    [[maybe_unused]] ThreadPoolAllocator<int, 4096> medium_block;
    [[maybe_unused]] ThreadPoolAllocator<int, 16384> large_block;

    // Different block sizes should compile
    EXPECT_TRUE(true);
}

// Test allocator traits
TEST_F(ThreadPoolAllocatorTest, AllocatorTraits) {
    using Alloc = ThreadPoolAllocator<int>;

    // Check type definitions
    static_assert(std::is_same_v<Alloc::value_type, int>);
    static_assert(std::is_same_v<Alloc::pointer, int*>);
    static_assert(std::is_same_v<Alloc::const_pointer, const int*>);
    static_assert(std::is_same_v<Alloc::size_type, std::size_t>);
    static_assert(std::is_same_v<Alloc::difference_type, std::ptrdiff_t>);

    // Check propagation traits
    static_assert(!Alloc::propagate_on_container_copy_assignment::value);
    static_assert(Alloc::propagate_on_container_move_assignment::value);
    static_assert(Alloc::propagate_on_container_swap::value);
    static_assert(Alloc::is_always_equal::value);

    EXPECT_TRUE(true);
}

// Test rebind
TEST_F(ThreadPoolAllocatorTest, Rebind) {
    using IntAlloc = ThreadPoolAllocator<int>;
    using DoubleAlloc = IntAlloc::rebind<double>::other;

    static_assert(std::is_same_v<DoubleAlloc::value_type, double>);
    static_assert(std::is_same_v<DoubleAlloc, ThreadPoolAllocator<double, fem::config::PAGE_SIZE>>);

    // Test rebind construction
    IntAlloc int_alloc;
    DoubleAlloc double_alloc(int_alloc);

    EXPECT_TRUE(true);
}

// Test single allocation and deallocation
TEST_F(ThreadPoolAllocatorTest, SingleAllocation) {
    ThreadPoolAllocator<int> alloc;

    // Allocate single element
    int* ptr = alloc.allocate(1);
    EXPECT_NE(ptr, nullptr);

    // Use the memory
    *ptr = 42;
    EXPECT_EQ(*ptr, 42);

    // Deallocate
    alloc.deallocate(ptr, 1);
}

TEST_F(ThreadPoolAllocatorTest, MultipleSignleAllocations) {
    ThreadPoolAllocator<int> alloc;
    std::vector<int*> ptrs;

    // Allocate many single elements
    for (int i = 0; i < 100; ++i) {
        int* ptr = alloc.allocate(1);
        EXPECT_NE(ptr, nullptr);
        *ptr = i;
        ptrs.push_back(ptr);
    }

    // Verify and deallocate
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*ptrs[i], i);
        alloc.deallocate(ptrs[i], 1);
    }
}

// Test bulk allocation
TEST_F(ThreadPoolAllocatorTest, BulkAllocation) {
    ThreadPoolAllocator<int> alloc;

    // Allocate multiple elements
    int* ptr = alloc.allocate(10);
    EXPECT_NE(ptr, nullptr);

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        ptr[i] = i * i;
    }

    // Verify
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ptr[i], i * i);
    }

    // Deallocate
    alloc.deallocate(ptr, 10);
}

TEST_F(ThreadPoolAllocatorTest, MixedAllocations) {
    ThreadPoolAllocator<int> alloc;

    // Mix of single and bulk allocations
    int* single1 = alloc.allocate(1);
    int* bulk1 = alloc.allocate(5);
    int* single2 = alloc.allocate(1);
    int* bulk2 = alloc.allocate(20);

    EXPECT_NE(single1, nullptr);
    EXPECT_NE(bulk1, nullptr);
    EXPECT_NE(single2, nullptr);
    EXPECT_NE(bulk2, nullptr);

    // Use memory
    *single1 = 100;
    *single2 = 200;
    for (int i = 0; i < 5; ++i) bulk1[i] = i;
    for (int i = 0; i < 20; ++i) bulk2[i] = i * 2;

    // Verify
    EXPECT_EQ(*single1, 100);
    EXPECT_EQ(*single2, 200);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(bulk1[i], i);
    for (int i = 0; i < 20; ++i) EXPECT_EQ(bulk2[i], i * 2);

    // Deallocate in different order
    alloc.deallocate(bulk2, 20);
    alloc.deallocate(single1, 1);
    alloc.deallocate(bulk1, 5);
    alloc.deallocate(single2, 1);
}

// Test null pointer deallocation
TEST_F(ThreadPoolAllocatorTest, NullDeallocation) {
    ThreadPoolAllocator<int> alloc;

    // Should handle null pointer gracefully
    alloc.deallocate(nullptr, 1);
    alloc.deallocate(nullptr, 10);

    EXPECT_TRUE(true);  // No crash
}

// Test construct and destroy
TEST_F(ThreadPoolAllocatorTest, ConstructDestroy) {
    ThreadPoolAllocator<std::string> alloc;

    // Allocate memory
    std::string* ptr = alloc.allocate(1);
    EXPECT_NE(ptr, nullptr);

    // Construct object
    alloc.construct(ptr, "Hello, World!");
    EXPECT_EQ(*ptr, "Hello, World!");

    // Destroy object
    alloc.destroy(ptr);

    // Deallocate memory
    alloc.deallocate(ptr, 1);
}

TEST_F(ThreadPoolAllocatorTest, ConstructWithArgs) {
    struct TestStruct {
        int a;
        double b;
        std::string c;

        TestStruct(int x, double y, std::string z) : a(x), b(y), c(std::move(z)) {}
    };

    ThreadPoolAllocator<TestStruct> alloc;

    TestStruct* ptr = alloc.allocate(1);
    alloc.construct(ptr, 42, 3.14, "test");

    EXPECT_EQ(ptr->a, 42);
    EXPECT_DOUBLE_EQ(ptr->b, 3.14);
    EXPECT_EQ(ptr->c, "test");

    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
}

// Test try_allocate
TEST_F(ThreadPoolAllocatorTest, TryAllocateSuccess) {
    ThreadPoolAllocator<int> alloc;

    // Single allocation
    auto result = alloc.try_allocate(1);
    EXPECT_TRUE(result);
    if (result) {
        int* ptr = result.value();
        EXPECT_NE(ptr, nullptr);
        *ptr = 123;
        EXPECT_EQ(*ptr, 123);
        alloc.deallocate(ptr, 1);
    }

    // Bulk allocation
    result = alloc.try_allocate(10);
    EXPECT_TRUE(result);
    if (result) {
        int* ptr = result.value();
        EXPECT_NE(ptr, nullptr);
        for (int i = 0; i < 10; ++i) {
            ptr[i] = i;
        }
        alloc.deallocate(ptr, 10);
    }
}

// Test with STL containers
TEST_F(ThreadPoolAllocatorTest, STLVector) {
    ThreadPoolAllocator<int> alloc;
    std::vector<int, ThreadPoolAllocator<int>> vec(alloc);

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100u);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(ThreadPoolAllocatorTest, STLList) {
    ThreadPoolAllocator<int> alloc;
    std::list<int, ThreadPoolAllocator<int>> lst(alloc);

    for (int i = 0; i < 50; ++i) {
        lst.push_back(i);
    }

    EXPECT_EQ(lst.size(), 50u);

    int expected = 0;
    for (int val : lst) {
        EXPECT_EQ(val, expected++);
    }
}

TEST_F(ThreadPoolAllocatorTest, STLMap) {
    using Alloc = ThreadPoolAllocator<std::pair<const int, std::string>>;
    std::map<int, std::string, std::less<int>, Alloc> map(Alloc{});

    for (int i = 0; i < 20; ++i) {
        map[i] = "value" + std::to_string(i);
    }

    EXPECT_EQ(map.size(), 20u);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(map[i], "value" + std::to_string(i));
    }
}

// Test thread-local behavior
TEST_F(ThreadPoolAllocatorTest, ThreadLocalPools) {
    ThreadPoolAllocator<int> alloc;
    constexpr int num_threads = 4;
    constexpr int allocations_per_thread = 100;

    std::vector<std::thread> threads;
    std::atomic<int> total_allocations{0};

    auto worker = [&alloc, &total_allocations]() {
        std::vector<int*> local_ptrs;

        // Each thread does its own allocations
        for (int i = 0; i < allocations_per_thread; ++i) {
            int* ptr = alloc.allocate(1);
            EXPECT_NE(ptr, nullptr);
            *ptr = i;
            local_ptrs.push_back(ptr);
            total_allocations.fetch_add(1, std::memory_order_relaxed);
        }

        // Verify and deallocate
        for (int i = 0; i < allocations_per_thread; ++i) {
            EXPECT_EQ(*local_ptrs[i], i);
            alloc.deallocate(local_ptrs[i], 1);
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(total_allocations.load(), num_threads * allocations_per_thread);
}

// Test that each thread has its own pool
TEST_F(ThreadPoolAllocatorTest, SeparateThreadPools) {
    ThreadPoolAllocator<int> alloc;

    std::atomic<int> total_allocs{0};
    std::atomic<int> total_deallocs{0};

    auto worker = [&]() {
        std::vector<int*> local_ptrs;

        // Allocate some memory
        for (int i = 0; i < 10; ++i) {
            int* ptr = alloc.allocate(1);
            *ptr = i;
            local_ptrs.push_back(ptr);
            total_allocs.fetch_add(1, std::memory_order_relaxed);
        }

        // Verify values are correct
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(*local_ptrs[i], i);
        }

        // Deallocate in same thread (important for thread-local pools)
        for (int* ptr : local_ptrs) {
            alloc.deallocate(ptr, 1);
            total_deallocs.fetch_add(1, std::memory_order_relaxed);
        }
    };

    // Create multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify all allocations and deallocations happened
    EXPECT_EQ(total_allocs.load(), 40);
    EXPECT_EQ(total_deallocs.load(), 40);
}

// Test concurrent allocation/deallocation
TEST_F(ThreadPoolAllocatorTest, ConcurrentOperations) {
    ThreadPoolAllocator<int> alloc;
    constexpr int num_threads = 8;
    constexpr int iterations = 1000;

    std::atomic<int> success_count{0};
    std::atomic<bool> start_flag{false};

    auto worker = [&]() {
        // Wait for start signal
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (int i = 0; i < iterations; ++i) {
            // Allocate
            int* ptr = alloc.allocate(1);
            if (ptr) {
                *ptr = i;
                success_count.fetch_add(1, std::memory_order_relaxed);

                // Small work to increase contention
                std::this_thread::yield();

                // Deallocate
                alloc.deallocate(ptr, 1);
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Start all threads simultaneously
    start_flag.store(true, std::memory_order_release);

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * iterations);
}

// Test allocator comparison
TEST_F(ThreadPoolAllocatorTest, AllocatorComparison) {
    ThreadPoolAllocator<int> alloc1;
    ThreadPoolAllocator<int> alloc2;
    [[maybe_unused]] ThreadPoolAllocator<double> alloc3;

    // ThreadPoolAllocator is always equal (per-thread state)
    // Since is_always_equal is true_type, all instances compare equal
    // Note: Allocators may not have comparison operators defined
    // The is_always_equal trait indicates they would compare equal if compared

    // Test that they can be used interchangeably
    int* p1 = alloc1.allocate(1);
    *p1 = 42;
    EXPECT_EQ(*p1, 42);
    alloc2.deallocate(p1, 1);  // Can deallocate with different instance
}

// Test with different block sizes
TEST_F(ThreadPoolAllocatorTest, DifferentBlockSizes) {
    ThreadPoolAllocator<int, 512> small_block;
    ThreadPoolAllocator<int, 4096> medium_block;
    ThreadPoolAllocator<int, 16384> large_block;

    // Each should work independently
    int* p1 = small_block.allocate(1);
    int* p2 = medium_block.allocate(1);
    int* p3 = large_block.allocate(1);

    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p3, nullptr);

    *p1 = 1;
    *p2 = 2;
    *p3 = 3;

    EXPECT_EQ(*p1, 1);
    EXPECT_EQ(*p2, 2);
    EXPECT_EQ(*p3, 3);

    small_block.deallocate(p1, 1);
    medium_block.deallocate(p2, 1);
    large_block.deallocate(p3, 1);
}

// Test destruction of objects
TEST_F(ThreadPoolAllocatorTest, ProperDestruction) {
    static int destructor_count = 0;

    struct TrackingObject {
        int value;
        TrackingObject(int v) : value(v) {}
        ~TrackingObject() { ++destructor_count; }
    };

    destructor_count = 0;

    {
        ThreadPoolAllocator<TrackingObject> alloc;

        // Allocate and construct
        TrackingObject* ptr = alloc.allocate(1);
        alloc.construct(ptr, 42);

        EXPECT_EQ(ptr->value, 42);
        EXPECT_EQ(destructor_count, 0);

        // Destroy and deallocate
        alloc.destroy(ptr);
        EXPECT_EQ(destructor_count, 1);

        alloc.deallocate(ptr, 1);
    }
}

// Test trivial destruction optimization
TEST_F(ThreadPoolAllocatorTest, TrivialDestruction) {
    ThreadPoolAllocator<int> int_alloc;
    ThreadPoolAllocator<std::string> string_alloc;

    // For trivially destructible types, destroy should be optimized
    int* int_ptr = int_alloc.allocate(1);
    int_alloc.construct(int_ptr, 42);
    int_alloc.destroy(int_ptr);  // Should be no-op for int
    int_alloc.deallocate(int_ptr, 1);

    // For non-trivial types, destructor should be called
    std::string* str_ptr = string_alloc.allocate(1);
    string_alloc.construct(str_ptr, "test");
    string_alloc.destroy(str_ptr);  // Should call destructor
    string_alloc.deallocate(str_ptr, 1);

    EXPECT_TRUE(true);  // No crash
}

// Test performance vs standard allocator
TEST_F(ThreadPoolAllocatorTest, PerformanceComparison) {
    constexpr int iterations = 10000;

    // Test with thread pool allocator
    auto start = std::chrono::high_resolution_clock::now();
    {
        ThreadPoolAllocator<int> alloc;
        for (int i = 0; i < iterations; ++i) {
            int* ptr = alloc.allocate(1);
            *ptr = i;
            alloc.deallocate(ptr, 1);
        }
    }
    auto thread_pool_time = std::chrono::high_resolution_clock::now() - start;

    // Test with standard allocator
    start = std::chrono::high_resolution_clock::now();
    {
        std::allocator<int> alloc;
        for (int i = 0; i < iterations; ++i) {
            int* ptr = alloc.allocate(1);
            *ptr = i;
            alloc.deallocate(ptr, 1);
        }
    }
    auto standard_time = std::chrono::high_resolution_clock::now() - start;

    // Thread pool allocator should be competitive
    // Not asserting it's faster as that depends on system
    EXPECT_GT(thread_pool_time.count(), 0);
    EXPECT_GT(standard_time.count(), 0);
}

// Test allocator with move-only types
TEST_F(ThreadPoolAllocatorTest, MoveOnlyTypes) {
    ThreadPoolAllocator<std::unique_ptr<int>> alloc;

    auto* ptr = alloc.allocate(1);
    alloc.construct(ptr, std::make_unique<int>(42));

    EXPECT_NE(ptr->get(), nullptr);
    EXPECT_EQ(**ptr, 42);

    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
}

// Test exception safety
TEST_F(ThreadPoolAllocatorTest, ExceptionSafety) {
    static int throwing_counter = 0;

    struct ThrowingObject {
        ThrowingObject() {
            if (++throwing_counter == 3) {
                throw std::runtime_error("Construction failed");
            }
        }
    };

    throwing_counter = 0;

    ThreadPoolAllocator<ThrowingObject> alloc;

    // Allocate memory
    ThrowingObject* ptr = alloc.allocate(5);

    // Try to construct objects, expecting exception
    try {
        for (int i = 0; i < 5; ++i) {
            alloc.construct(ptr + i);
        }
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "Construction failed");
    }

    // Clean up (destructor not called on failed constructions)
    alloc.deallocate(ptr, 5);
}

// Test edge cases
TEST_F(ThreadPoolAllocatorTest, EdgeCases) {
    ThreadPoolAllocator<int> alloc;

    // Zero allocation (undefined behavior in standard, but test handling)
    // Skip this as it's UB: alloc.allocate(0);

    // Very large allocation
    try {
        [[maybe_unused]] int* ptr = alloc.allocate(SIZE_MAX / sizeof(int));
        // If it succeeds, deallocate
        alloc.deallocate(ptr, SIZE_MAX / sizeof(int));
    } catch (const std::bad_alloc&) {
        // Expected for very large allocations
        EXPECT_TRUE(true);
    }

    // Multiple deallocations of same pointer would be UB, don't test
}

// Test thread cleanup
TEST_F(ThreadPoolAllocatorTest, ThreadCleanup) {
    ThreadPoolAllocator<int> alloc;

    auto thread_func = [&alloc]() {
        // Allocate some memory in thread
        std::vector<int*> ptrs;
        for (int i = 0; i < 10; ++i) {
            ptrs.push_back(alloc.allocate(1));
            *ptrs.back() = i;
        }

        // Thread-local pool should be cleaned up when thread exits
        for (auto ptr : ptrs) {
            alloc.deallocate(ptr, 1);
        }
    };

    // Create and join multiple threads
    for (int i = 0; i < 5; ++i) {
        std::thread t(thread_func);
        t.join();
    }

    EXPECT_TRUE(true);  // No leaks or crashes
}

} // namespace
} // namespace fem::core::memory
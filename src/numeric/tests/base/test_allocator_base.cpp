#include <gtest/gtest.h>
#include <base/allocator_base.h>

#include <cstdint>
#include <vector>
#include <thread>
#include <numeric>   // for std::gcd
#include <limits>

using namespace fem::numeric;

// Utility: compute an n such that n*sizeof(T) is a multiple of Alignment
template <typename T>
static size_t n_multiple_for_alignment(size_t Alignment, size_t cap = 128) {
    size_t n = 1;
    while (n <= cap) {
        if ((n * sizeof(T)) % Alignment == 0) return n;
        ++n;
    }
    // Fallback (shouldn't happen in practice with cap=128)
    return ((Alignment + sizeof(T) - 1) / sizeof(T)) * 2;
}

// A type to test construct/destroy
struct Tracked {
    static inline int ctor_count = 0;
    static inline int dtor_count = 0;
    int v;
    explicit Tracked(int x = 0) : v(x) { ++ctor_count; }
    ~Tracked() { ++dtor_count; }
};

// ============================================================================
// AlignedAllocator tests
// ============================================================================

TEST(AlignedAllocator, AllocateAlignmentAndNullOnZero) {
    using A = AlignedAllocator<double, 32>;
    A alloc;

    // n such that bytes are a multiple of 32
    size_t n = n_multiple_for_alignment<double>(A::alignment);
    auto p = alloc.allocate(n);
    ASSERT_NE(p, nullptr);

    // alignment check
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    EXPECT_EQ(addr % A::alignment, 0u);

    alloc.deallocate(p, n);

    // n == 0 -> nullptr
    EXPECT_EQ(alloc.allocate(0), nullptr);
    alloc.deallocate(nullptr, 0); // no crash
    SUCCEED();
}

TEST(AlignedAllocator, ConstructAndDestroyObjects) {
    using A = AlignedAllocator<Tracked, 32>;
    A alloc;

    Tracked::ctor_count = 0;
    Tracked::dtor_count = 0;

    const size_t n = n_multiple_for_alignment<Tracked>(A::alignment);
    Tracked* p = alloc.allocate(n);
    ASSERT_NE(p, nullptr);

    for (size_t i = 0; i < n; ++i) {
        alloc.construct(&p[i], static_cast<int>(i));
    }
    EXPECT_EQ(Tracked::ctor_count, static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(p[i].v, static_cast<int>(i));
    }
    for (size_t i = 0; i < n; ++i) {
        alloc.destroy(&p[i]);
    }
    EXPECT_EQ(Tracked::dtor_count, static_cast<int>(n));
    alloc.deallocate(p, n);

    // equality/inequality operators
    A a1, a2;
    EXPECT_TRUE(a1 == a2);
    EXPECT_FALSE(a1 != a2);
    SUCCEED();
}

TEST(AlignedAllocator, DifferentAlignmentValues) {
    // Test various alignment values (must be powers of 2)
    {
        AlignedAllocator<double, 16> alloc16;
        auto p = alloc16.allocate(10);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0u);
        alloc16.deallocate(p, 10);
    }

    {
        AlignedAllocator<double, 64> alloc64;
        auto p = alloc64.allocate(10);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64, 0u);
        alloc64.deallocate(p, 10);
    }

    {
        AlignedAllocator<double, 128> alloc128;
        auto p = alloc128.allocate(10);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 128, 0u);
        alloc128.deallocate(p, 10);
    }
    SUCCEED();
}

TEST(AlignedAllocator, RebindMechanism) {
    // Test the rebind mechanism required for STL compatibility
    using IntAlloc = AlignedAllocator<int, 32>;
    using DoubleAlloc = IntAlloc::rebind<double>::other;

    static_assert(std::is_same_v<DoubleAlloc, AlignedAllocator<double, 32>>);

    DoubleAlloc alloc;
    auto p = alloc.allocate(5);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, 5);
    SUCCEED();
}

TEST(AlignedAllocator, MaxSizeAndLargeAllocations) {
    AlignedAllocator<double, 32> alloc;

    // max_size should return a reasonable value
    auto max = alloc.max_size();
    EXPECT_GT(max, 0u);
    EXPECT_LE(max, std::numeric_limits<size_t>::max() / sizeof(double));

    // Test allocation at a large but reasonable size
    // Use 1GB worth of doubles as a "large" allocation
    size_t large_but_reasonable = (1ULL << 30) / sizeof(double);  // 1GB / 8 bytes

    try {
        auto p = alloc.allocate(large_but_reasonable);
        if (p != nullptr) {
            // If we actually got the memory, make sure we can deallocate it
            alloc.deallocate(p, large_but_reasonable);
            SUCCEED() << "Successfully allocated and deallocated 1GB";
        } else {
            SUCCEED() << "Allocator returned nullptr for large allocation";
        }
    } catch (const std::bad_alloc&) {
            // This is also acceptable behavior
            SUCCEED() << "Allocator threw bad_alloc for large allocation";
    }

    // Test truly huge allocation that should definitely fail
    // Use max_size() + 1 to ensure failure
    if (max < std::numeric_limits<size_t>::max()) {
        // Use a try-catch block instead of EXPECT_THROW to avoid nodiscard warning
        bool threw_exception = false;
        try {
            [[maybe_unused]] auto p = alloc.allocate(max + 1);  // Capture but mark as unused
            // If we get here, allocation succeeded (unlikely)
            alloc.deallocate(p, max + 1);
        } catch (const std::bad_alloc&) {
            threw_exception = true;
        }
        EXPECT_TRUE(threw_exception) << "Expected bad_alloc for allocation exceeding max_size";
    }
}

// ============================================================================
// PoolAllocator tests
// ============================================================================

TEST(PoolAllocator, ContiguousWithinBlockAndRollover) {
    // pool_size must yield bytes multiple of 32 (aligned_alloc requirement).
    // For int(4B), choose pool_size multiple of 8.
    PoolAllocator<int> pool(32);

    // First few allocations should be contiguous within the same block.
    int* p0 = pool.allocate(5);   // offset 0
    ASSERT_NE(p0, nullptr);
    int* p1 = pool.allocate(3);   // offset 5
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(p1, p0 + 5);

    // Fill the remaining space to exactly hit the end of block.
    // Current offset is 8; pool_size=32 -> remaining 24
    int* p2 = pool.allocate(24);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2, p0 + 8);

    // Next allocation must roll over to a NEW block (offset would exceed).
    int* p3 = pool.allocate(1);
    ASSERT_NE(p3, nullptr);
    // Can't use relational comparisons across different arrays,
    // but pointer equality should differ from any address in the first block start.
    EXPECT_NE(p3, p0);

    // Reset should bring us back to the beginning of the first block.
    pool.reset();
    int* p4 = pool.allocate(5);
    ASSERT_NE(p4, nullptr);
    EXPECT_EQ(p4, p0);  // after reset, first address in block 0 repeats
    SUCCEED();
}

TEST(PoolAllocator, ThreadedAllocationsAreSafe) {
    PoolAllocator<int> pool(1024); // ample room to avoid frequent new blocks

    const int threads = 4;
    const int per_thread_allocs = 64;
    const int each_n = 2;

    std::vector<int*> addrs;
    addrs.reserve(threads * per_thread_allocs);

    std::mutex v_mtx;

    auto worker = [&]() {
        for (int i = 0; i < per_thread_allocs; ++i) {
            int* p = pool.allocate(each_n);
            ASSERT_NE(p, nullptr);
            std::lock_guard<std::mutex> lk(v_mtx);
            addrs.push_back(p);
        }
    };

    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t) th.emplace_back(worker);
    for (auto& tt : th) tt.join();

    // All pointers should be non-null and at least not all the same.
    // (We can't check overlap robustly, but we can ensure a decent spread.)
    std::sort(addrs.begin(), addrs.end());
    auto unique = std::unique(addrs.begin(), addrs.end()) - addrs.begin();
    EXPECT_GT(unique, threads); // expect more than one unique pointer
    SUCCEED();
}

TEST(PoolAllocator, ZeroSizeAllocation) {
    PoolAllocator<int> pool(64);

    // Zero-size allocation should return nullptr or valid pointer
    auto p = pool.allocate(0);
    if (p != nullptr) {
        pool.deallocate(p, 0);  // Should not crash
    }
    SUCCEED();
}

TEST(PoolAllocator, FragmentationAfterReset) {
    PoolAllocator<int> pool(100);

    // Fragment the pool
    std::vector<int*> ptrs;
    for (int i = 0; i < 10; ++i) {
    ptrs.push_back(pool.allocate(5));
    }

    // Reset should defragment
    pool.reset();

    // Should be able to allocate a large contiguous block
    int* large = pool.allocate(50);
    ASSERT_NE(large, nullptr);

    // And continue with more allocations
    int* next = pool.allocate(30);
    ASSERT_NE(next, nullptr);
    SUCCEED();
}

TEST(PoolAllocator, DifferentTypes) {
    // Test with different types to ensure proper alignment handling
    {
        PoolAllocator<char> charPool(64);
        char* c = charPool.allocate(10);
        ASSERT_NE(c, nullptr);
    }

    {
        PoolAllocator<double> doublePool(64);
        double* d = doublePool.allocate(8);
        ASSERT_NE(d, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(d) % alignof(double), 0u);
    }

    struct Aligned16 {
        alignas(16) double data[2];
    };

    {
        PoolAllocator<Aligned16> alignedPool(64);
        Aligned16* a = alignedPool.allocate(2);
        ASSERT_NE(a, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 16, 0u);
    }
    SUCCEED();
}

// ============================================================================
// TrackingAllocator tests
// ============================================================================

TEST(TrackingAllocator, CountersIncreaseAndDecrease) {
    using TA = TrackingAllocator<int>;
    TA::reset_stats();
    TA alloc;

    EXPECT_EQ(TA::get_bytes_allocated(), 0u);
    EXPECT_EQ(TA::get_peak_bytes(), 0u);
    EXPECT_EQ(TA::get_allocation_count(), 0u);

    int* p = alloc.allocate(10);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(TA::get_allocation_count(), 1u);
    EXPECT_EQ(TA::get_bytes_allocated(), 10u * sizeof(int));
    EXPECT_EQ(TA::get_peak_bytes(), 10u * sizeof(int));

    int* q = alloc.allocate(20);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(TA::get_allocation_count(), 2u);
    EXPECT_EQ(TA::get_bytes_allocated(), (10u + 20u) * sizeof(int));
    EXPECT_EQ(TA::get_peak_bytes(), (10u + 20u) * sizeof(int));

    alloc.deallocate(p, 10);
    EXPECT_EQ(TA::get_bytes_allocated(), 20u * sizeof(int));
    EXPECT_EQ(TA::get_peak_bytes(), 30u * sizeof(int));

    alloc.deallocate(q, 20);
    EXPECT_EQ(TA::get_bytes_allocated(), 0u);

    // reset_stats should zero everything
    TA::reset_stats();
    EXPECT_EQ(TA::get_bytes_allocated(), 0u);
    EXPECT_EQ(TA::get_peak_bytes(), 0u);
    EXPECT_EQ(TA::get_allocation_count(), 0u);
    SUCCEED();
}

TEST(TrackingAllocator, WorksWithManualConstructDestroy) {
    using TA = TrackingAllocator<Tracked>;
    TA::reset_stats();
    TA alloc;

    Tracked* p = alloc.allocate(3);
    for (int i = 0; i < 3; ++i) new (&p[i]) Tracked(i * 10);

    EXPECT_EQ(TA::get_allocation_count(), 1u);
    EXPECT_EQ(TA::get_bytes_allocated(), 3u * sizeof(Tracked));

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(p[i].v, i * 10);
        p[i].~Tracked();
    }
    alloc.deallocate(p, 3);
    EXPECT_EQ(TA::get_bytes_allocated(), 0u);
    SUCCEED();
}

TEST(TrackingAllocator, ThreadSafetyOfStats) {
    using TA = TrackingAllocator<int>;
    TA::reset_stats();

    const int num_threads = 4;
    const int allocs_per_thread = 100;

    auto worker = [&]() {
        TA alloc;
        std::vector<int*> ptrs;
        for (int i = 0; i < allocs_per_thread; ++i) {
            ptrs.push_back(alloc.allocate(10));
        }
        // Deallocate half
        for (size_t i = 0; static_cast<size_t>(i < allocs_per_thread / 2); ++i) {
            alloc.deallocate(ptrs[i], 10);
        }
        // Let the rest be deallocated when thread ends
        for (size_t i = static_cast<size_t>(allocs_per_thread / 2); i < allocs_per_thread; ++i) {
            alloc.deallocate(ptrs[i], 10);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    // All memory should be freed
    EXPECT_EQ(TA::get_bytes_allocated(), 0u);
    // Total allocations should match
    EXPECT_EQ(TA::get_allocation_count(),
    static_cast<size_t>(num_threads * allocs_per_thread));
}

TEST(TrackingAllocator, MemoryLeakDetection) {
    using TA = TrackingAllocator<int>;
    TA::reset_stats();

    {
        TA alloc;
        int* leak = alloc.allocate(100);
        // Intentionally don't deallocate to simulate a leak
        EXPECT_GT(TA::get_bytes_allocated(), 0u);

        // In a real scenario, you'd check this at program end
        if (TA::get_bytes_allocated() > 0) {
            // Memory leak detected!
            // Clean up for the test
            alloc.deallocate(leak, 100);
        }
    }

    EXPECT_EQ(TA::get_bytes_allocated(), 0u);
}
// ============================================================================
// StackAllocator tests
// ============================================================================

TEST(StackAllocator, UsesStackThenFallsBackToHeap) {
    // StackSize is in "elements", not bytes.
    constexpr size_t StackElems = 16;
    StackAllocator<int, StackElems> st;

    // First allocation: within stack buffer.
    int* p1 = st.allocate(8);
    ASSERT_NE(p1, nullptr);
    // Alignment should be at least alignof(int) and buffer is alignas(32)
    auto addr1 = reinterpret_cast<std::uintptr_t>(p1);
    EXPECT_EQ(addr1 % alignof(int), 0u);
    EXPECT_EQ(addr1 % 32, 0u); // due to alignas(32) buffer

    // Second allocation spills over the stack buffer -> heap
    int* p2 = st.allocate(10); // 8 + 10 > 16
    ASSERT_NE(p2, nullptr);
    // We can't portably check origin, but deallocate should free heap pointers safely.
    st.deallocate(p2, 10);

    // Reset and re-allocate same size: should reuse the exact same stack address.
    st.reset();
    int* p1_again = st.allocate(8);
    ASSERT_NE(p1_again, nullptr);
    EXPECT_EQ(p1_again, p1);
    SUCCEED();
}

TEST(StackAllocator, ResetAllowsMultipleReuses) {
    StackAllocator<double, 32> st; // 32 doubles fit in stack buffer
    double* a = st.allocate(16);
    double* b = st.allocate(16); // fills stack buffer exactly
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    // After reset, we should get back the beginning again
    st.reset();
    double* a2 = st.allocate(16);
    ASSERT_EQ(a2, a);
    SUCCEED();
}

TEST(StackAllocator, ExactFit) {
    constexpr size_t StackSize = 64;
    StackAllocator<int, StackSize> stack;

    // Allocate exactly the stack size
    int* p = stack.allocate(StackSize);
    ASSERT_NE(p, nullptr);

    // Next allocation should go to heap
    int* heap = stack.allocate(1);
    ASSERT_NE(heap, nullptr);

    // These should be different memory regions
    EXPECT_NE(p, heap);

    stack.deallocate(heap, 1);
    SUCCEED();
}

TEST(StackAllocator, AlignmentPreserved) {
    // Test that alignment is preserved for different types
    struct Aligned32 {
        alignas(32) double data[4];
    };

    StackAllocator<Aligned32, 4> stack;
    Aligned32* p = stack.allocate(2);
    ASSERT_NE(p, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 32, 0u);
    SUCCEED();
}

TEST(StackAllocator, ConstructDestroy) {
    StackAllocator<Tracked, 10> stack;

    Tracked::ctor_count = 0;
    Tracked::dtor_count = 0;

    Tracked* p = stack.allocate(5);
    ASSERT_NE(p, nullptr);

    // Construct objects
    for (int i = 0; i < 5; ++i) {
        new (&p[i]) Tracked(i*100);
        //stack.construct(&p[i], i * 100);
    }
    EXPECT_EQ(Tracked::ctor_count, 5);

    // Verify values
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(p[i].v, i * 100);
    }

    // Destroy objects
    for (int i = 0; i < 5; ++i) {
        p[i].~Tracked();
        //stack.destroy(&p[i]);
    }
    EXPECT_EQ(Tracked::dtor_count, 5);

    // Note: StackAllocator doesn't actually free stack memory on deallocate
    // It only resets on reset()
    SUCCEED();
}

TEST(AllocatorEdgeCases, NullptrHandling) {
    AlignedAllocator<int, 32> aligned;
    PoolAllocator<int> pool(64);
    TrackingAllocator<int> tracking;
    StackAllocator<int, 32> stack;

    // Deallocating nullptr should be safe
    EXPECT_NO_THROW(aligned.deallocate(nullptr, 0));
    EXPECT_NO_THROW(pool.deallocate(nullptr, 0));
    EXPECT_NO_THROW(tracking.deallocate(nullptr, 0));
    EXPECT_NO_THROW(stack.deallocate(nullptr, 0));

    // Destroying nullptr should be safe
    int* null_int = nullptr;
    EXPECT_NO_THROW(aligned.destroy(null_int));

    // For AlignedAllocator construct method
    EXPECT_NO_THROW(aligned.construct(null_int, 42));  // Should handle gracefully
}
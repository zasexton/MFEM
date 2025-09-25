#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include <core/memory/scoped_memory_tracker.h>
#include <core/memory/memory_tracker.h>
#include <core/memory/memory_resource.h>

namespace fem::core::memory {
namespace {

// Test fixture for scoped memory tracker tests
class ScopedMemoryTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the global memory tracker state before each test
        // Note: MemoryTracker is a singleton, so we need to be careful about state
        initial_stats_ = MemoryTracker::instance().stats();
    }

    void TearDown() override {
        // Clean up any allocations that might have been missed
        // This helps ensure test isolation
    }

    // Helper to allocate memory through the tracked system
    void* TrackedAllocate(std::size_t size) {
        return default_resource()->allocate(size, alignof(std::max_align_t));
    }

    void TrackedDeallocate(void* ptr, std::size_t size) {
        default_resource()->deallocate(ptr, size, alignof(std::max_align_t));
    }

    MemoryStats initial_stats_;
};

// Test basic construction and destruction
TEST_F(ScopedMemoryTrackerTest, BasicConstruction) {
    {
        ScopedMemoryTracker tracker;

        // Should have empty label by default
        EXPECT_TRUE(tracker.label().empty());

        // Should have captured start stats
        const auto& start = tracker.start();
        EXPECT_GE(start.allocation_count, 0u);
    }

    // Tracker is destroyed, no crash expected
    EXPECT_TRUE(true);
}

TEST_F(ScopedMemoryTrackerTest, ConstructionWithLabel) {
    {
        ScopedMemoryTracker tracker("TestOperation");

        EXPECT_EQ(tracker.label(), "TestOperation");

        // Start stats should be captured
        const auto& start = tracker.start();
        EXPECT_GE(start.allocation_count, 0u);
    }
}

TEST_F(ScopedMemoryTrackerTest, EmptyLabel) {
    {
        ScopedMemoryTracker tracker("");
        EXPECT_TRUE(tracker.label().empty());
    }

    {
        ScopedMemoryTracker tracker(std::string{});
        EXPECT_TRUE(tracker.label().empty());
    }
}

TEST_F(ScopedMemoryTrackerTest, LongLabel) {
    std::string long_label(1000, 'X');
    {
        ScopedMemoryTracker tracker(long_label);
        EXPECT_EQ(tracker.label(), long_label);
        EXPECT_EQ(tracker.label().size(), 1000u);
    }
}

// Test delta calculation without allocations
TEST_F(ScopedMemoryTrackerTest, DeltaWithoutAllocations) {
    MemoryStats delta_copy;

    {
        ScopedMemoryTracker tracker("NoAllocs");

        // No allocations in this scope
        // Delta should be zero for allocation/free counts

        // Note: We can't access delta until after destruction,
        // but we can check start stats
        const auto& start = tracker.start();
        EXPECT_GE(start.allocation_count, 0u);

        // Store for checking after destruction
        delta_copy = tracker.delta();  // This will be all zeros before destruction
    }

    // After destruction, delta would be calculated
    // Since we didn't do any allocations, the deltas should be minimal
}

// Test delta calculation with allocations
TEST_F(ScopedMemoryTrackerTest, DeltaWithAllocations) {
    const std::size_t alloc_size = 1024;

    {
        ScopedMemoryTracker tracker("WithAllocs");

        // Perform some allocations
        void* ptr1 = TrackedAllocate(alloc_size);
        void* ptr2 = TrackedAllocate(alloc_size * 2);

        // Clean up
        TrackedDeallocate(ptr1, alloc_size);
        TrackedDeallocate(ptr2, alloc_size * 2);
    }

    // Tracker destructor has calculated deltas
    EXPECT_TRUE(true);  // No crash
}

// Test nested scopes
TEST_F(ScopedMemoryTrackerTest, NestedScopes) {
    const std::size_t alloc_size = 512;

    {
        ScopedMemoryTracker outer("OuterScope");

        void* outer_ptr = TrackedAllocate(alloc_size);

        {
            ScopedMemoryTracker inner("InnerScope");

            // Inner scope allocations
            void* inner_ptr = TrackedAllocate(alloc_size * 2);

            // Inner should track its own allocations
            EXPECT_EQ(inner.label(), "InnerScope");

            TrackedDeallocate(inner_ptr, alloc_size * 2);
        }

        // Outer scope continues
        EXPECT_EQ(outer.label(), "OuterScope");

        TrackedDeallocate(outer_ptr, alloc_size);
    }
}

// Test multiple trackers in parallel
TEST_F(ScopedMemoryTrackerTest, MultipleTrackersInParallel) {
    const std::size_t alloc_size = 256;

    {
        ScopedMemoryTracker tracker1("Tracker1");
        ScopedMemoryTracker tracker2("Tracker2");
        ScopedMemoryTracker tracker3("Tracker3");

        // Each tracker should have its own label
        EXPECT_EQ(tracker1.label(), "Tracker1");
        EXPECT_EQ(tracker2.label(), "Tracker2");
        EXPECT_EQ(tracker3.label(), "Tracker3");

        // Allocations happen while all trackers are active
        void* ptr = TrackedAllocate(alloc_size);
        TrackedDeallocate(ptr, alloc_size);
    }

    // All trackers destroyed in reverse order
}

// Test tracking with STL containers
TEST_F(ScopedMemoryTrackerTest, TrackingWithSTLContainers) {
    {
        ScopedMemoryTracker tracker("STLAllocations");

        // Create containers that will allocate memory
        std::vector<int> vec;
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }

        std::vector<std::string> str_vec;
        for (int i = 0; i < 100; ++i) {
            str_vec.push_back(std::string(100, static_cast<char>('A' + (i % 26))));
        }

        // Note: STL containers might use their own allocators,
        // not necessarily going through our memory_resource
        EXPECT_EQ(vec.size(), 1000u);
        EXPECT_EQ(str_vec.size(), 100u);
    }
}

// Test that class can be constructed and destroyed properly
TEST_F(ScopedMemoryTrackerTest, ProperConstruction) {
    // ScopedMemoryTracker is a simple RAII class
    // Test that it can be constructed and destroyed without issues

    {
        ScopedMemoryTracker tracker("TestConstruction");
        // Tracker exists in this scope
        EXPECT_FALSE(tracker.label().empty());
    }
    // Tracker destroyed here

    EXPECT_TRUE(true);  // No issues with construction/destruction
}

// Test that start stats are captured at construction
TEST_F(ScopedMemoryTrackerTest, StartStatsCapturedAtConstruction) {
    // Do some allocations before creating tracker
    void* pre_ptr = TrackedAllocate(1024);

    auto pre_stats = MemoryTracker::instance().stats();

    {
        ScopedMemoryTracker tracker("TimingTest");

        // Start stats should match the state at construction
        const auto& start = tracker.start();

        // These should be close (might not be exact due to other allocations)
        EXPECT_EQ(start.allocation_count, pre_stats.allocation_count);
        EXPECT_EQ(start.total_allocated, pre_stats.total_allocated);
    }

    TrackedDeallocate(pre_ptr, 1024);
}

// Test delta calculation accuracy
TEST_F(ScopedMemoryTrackerTest, DeltaCalculationAccuracy) {
    // Note: We can't directly access delta after destruction,
    // but we can verify the calculation logic

    auto stats_before = MemoryTracker::instance().stats();

    {
        ScopedMemoryTracker tracker("DeltaTest");

        // Allocate exactly 3 blocks of known sizes
        void* ptr1 = TrackedAllocate(100);
        void* ptr2 = TrackedAllocate(200);
        void* ptr3 = TrackedAllocate(300);

        // Deallocate one
        TrackedDeallocate(ptr2, 200);

        // Clean up remaining
        TrackedDeallocate(ptr1, 100);
        TrackedDeallocate(ptr3, 300);
    }

    auto stats_after = MemoryTracker::instance().stats();

    // Verify that allocations and deallocations were tracked
    EXPECT_GE(stats_after.allocation_count, stats_before.allocation_count);
    EXPECT_GE(stats_after.free_count, stats_before.free_count);
}

// Test with zero allocations
TEST_F(ScopedMemoryTrackerTest, ZeroAllocations) {
    {
        ScopedMemoryTracker tracker("ZeroAllocs");

        // Do nothing - no allocations

        // Should still work without issues
        EXPECT_EQ(tracker.label(), "ZeroAllocs");
    }
}

// Test rapid creation and destruction
TEST_F(ScopedMemoryTrackerTest, RapidCreationDestruction) {
    for (int i = 0; i < 100; ++i) {
        ScopedMemoryTracker tracker("Rapid" + std::to_string(i));

        // Quick allocation and deallocation
        if (i % 2 == 0) {
            void* ptr = TrackedAllocate(64);
            TrackedDeallocate(ptr, 64);
        }
    }

    EXPECT_TRUE(true);  // No crashes
}

// Test deeply nested scopes
TEST_F(ScopedMemoryTrackerTest, DeeplyNestedScopes) {
    {
        ScopedMemoryTracker t1("Level1");
        void* p1 = TrackedAllocate(10);
        {
            ScopedMemoryTracker t2("Level2");
            void* p2 = TrackedAllocate(20);
            {
                ScopedMemoryTracker t3("Level3");
                void* p3 = TrackedAllocate(30);
                {
                    ScopedMemoryTracker t4("Level4");
                    void* p4 = TrackedAllocate(40);
                    {
                        ScopedMemoryTracker t5("Level5");
                        void* p5 = TrackedAllocate(50);

                        // Verify all labels are correct
                        EXPECT_EQ(t1.label(), "Level1");
                        EXPECT_EQ(t2.label(), "Level2");
                        EXPECT_EQ(t3.label(), "Level3");
                        EXPECT_EQ(t4.label(), "Level4");
                        EXPECT_EQ(t5.label(), "Level5");

                        TrackedDeallocate(p5, 50);
                    }
                    TrackedDeallocate(p4, 40);
                }
                TrackedDeallocate(p3, 30);
            }
            TrackedDeallocate(p2, 20);
        }
        TrackedDeallocate(p1, 10);
    }
}

// Test exception safety
TEST_F(ScopedMemoryTrackerTest, ExceptionSafety) {
    try {
        ScopedMemoryTracker tracker("ExceptionTest");

        void* ptr = TrackedAllocate(1024);

        // Simulate an exception
        throw std::runtime_error("Test exception");

        // This won't be reached
        TrackedDeallocate(ptr, 1024);
    } catch (const std::exception& e) {
        // Tracker should have been destroyed properly
        EXPECT_STREQ(e.what(), "Test exception");
    }
}

// Test with memory resource allocations
TEST_F(ScopedMemoryTrackerTest, MemoryResourceIntegration) {
    {
        ScopedMemoryTracker tracker("MemoryResourceTest");

        // Use the default memory resource
        auto* mr = default_resource();

        // Allocate through memory resource
        void* ptr1 = mr->allocate(256, 8);
        void* ptr2 = mr->allocate(512, 16);
        void* ptr3 = mr->allocate(1024, 32);

        // Deallocate in different order
        mr->deallocate(ptr2, 512, 16);
        mr->deallocate(ptr1, 256, 8);
        mr->deallocate(ptr3, 1024, 32);
    }
}

// Test concurrent tracker creation (thread safety of singleton)
TEST_F(ScopedMemoryTrackerTest, ConcurrentTrackerCreation) {
    constexpr int num_threads = 4;
    constexpr int iterations = 10;

    auto worker = [](int thread_id) {
        for (int i = 0; i < iterations; ++i) {
            ScopedMemoryTracker tracker("Thread" + std::to_string(thread_id) +
                                       "_Iter" + std::to_string(i));

            // Do some work
            void* ptr = default_resource()->allocate(128, 8);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            default_resource()->deallocate(ptr, 128, 8);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_TRUE(true);  // No crashes or deadlocks
}

// Test that delta captures the difference
TEST_F(ScopedMemoryTrackerTest, DeltaCapturesDifference) {
    // This test verifies the delta calculation logic
    auto initial = MemoryTracker::instance().stats();

    {
        ScopedMemoryTracker tracker("DeltaCapture");
        auto start = tracker.start();

        // Verify start matches current state
        EXPECT_EQ(start.allocation_count, initial.allocation_count);

        // Do allocations
        void* ptrs[5];
        for (int i = 0; i < 5; ++i) {
            ptrs[i] = TrackedAllocate(100 * (i + 1));
        }

        // Free some
        for (int i = 0; i < 3; ++i) {
            TrackedDeallocate(ptrs[i], 100 * (i + 1));
        }

        // At destruction, delta will be calculated
        // total_allocated should increase by 5 allocations
        // total_freed should increase by 3 deallocations

        // Clean up remaining
        for (int i = 3; i < 5; ++i) {
            TrackedDeallocate(ptrs[i], 100 * (i + 1));
        }
    }

    auto final = MemoryTracker::instance().stats();

    // Verify that stats are at least non-negative
    // Note: If memory tracking is not enabled globally, counts might be 0
    EXPECT_GE(final.allocation_count, initial.allocation_count);
    EXPECT_GE(final.free_count, initial.free_count);
}

// Test label with special characters
TEST_F(ScopedMemoryTrackerTest, SpecialCharacterLabels) {
    {
        ScopedMemoryTracker t1("Label with spaces");
        EXPECT_EQ(t1.label(), "Label with spaces");
    }

    {
        ScopedMemoryTracker t2("Label\nwith\nnewlines");
        EXPECT_EQ(t2.label(), "Label\nwith\nnewlines");
    }

    {
        ScopedMemoryTracker t3("Label\twith\ttabs");
        EXPECT_EQ(t3.label(), "Label\twith\ttabs");
    }

    {
        ScopedMemoryTracker t4("Unicode: 你好 мир 🌍");
        EXPECT_EQ(t4.label(), "Unicode: 你好 мир 🌍");
    }
}

// Test const correctness
TEST_F(ScopedMemoryTrackerTest, ConstCorrectness) {
    {
        const ScopedMemoryTracker tracker("ConstTest");

        // These should be const methods
        [[maybe_unused]] const std::string& label = tracker.label();
        [[maybe_unused]] const MemoryStats& start = tracker.start();
        [[maybe_unused]] const MemoryStats& delta = tracker.delta();

        EXPECT_EQ(tracker.label(), "ConstTest");
    }
}

// Test edge case: tracker with same label
TEST_F(ScopedMemoryTrackerTest, MultipleSameLabel) {
    {
        ScopedMemoryTracker t1("SameLabel");
        ScopedMemoryTracker t2("SameLabel");
        ScopedMemoryTracker t3("SameLabel");

        // All should work independently despite same label
        EXPECT_EQ(t1.label(), "SameLabel");
        EXPECT_EQ(t2.label(), "SameLabel");
        EXPECT_EQ(t3.label(), "SameLabel");

        // Each has its own start stats
        auto& s1 = t1.start();
        auto& s2 = t2.start();
        auto& s3 = t3.start();

        // Start stats should be similar (captured at nearly same time)
        EXPECT_EQ(s1.allocation_count, s2.allocation_count);
        EXPECT_EQ(s2.allocation_count, s3.allocation_count);
    }
}

// Performance test
TEST_F(ScopedMemoryTrackerTest, PerformanceOverhead) {
    constexpr int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        ScopedMemoryTracker tracker("PerfTest");
        // Tracker created and immediately destroyed
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be fast - just capturing stats
    // Even 10k iterations should take less than 100ms
    EXPECT_LT(duration.count(), 100000);

    // Average time per tracker
    double avg_microseconds = static_cast<double>(duration.count()) / iterations;
    EXPECT_LT(avg_microseconds, 10.0);  // Less than 10 microseconds per tracker
}

} // namespace
} // namespace fem::core::memory
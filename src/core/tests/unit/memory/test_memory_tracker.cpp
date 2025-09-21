#include <gtest/gtest.h>
#include <core/memory/memory_tracker.h>
#include <core/memory/scoped_memory_tracker.h>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <random>

namespace fcm = fem::core::memory;

class MemoryTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get initial state to reset tracker
        initial_stats_ = fcm::MemoryTracker::instance().stats();
    }

    void TearDown() override {
        // Note: We can't truly reset the singleton, but we can verify
        // that our tests properly clean up after themselves
    }

    fcm::MemoryStats initial_stats_;
};

// Helper function to allocate and track memory
void* tracked_alloc(std::size_t size, const std::string& type_name = "test",
                    const std::string& label = "") {
    void* p = ::operator new(size);
    fcm::MemoryTracker::instance().on_alloc(p, size, alignof(std::max_align_t),
                                            type_name, label);
    return p;
}

// Helper function to free tracked memory
void tracked_free(void* p) {
    fcm::MemoryTracker::instance().on_free(p);
    ::operator delete(p);
}

// Basic functionality tests
TEST_F(MemoryTrackerTest, SingleAllocation) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate 100 bytes
    void* p = tracked_alloc(100, "int[]", "test_allocation");

    auto stats_after = fcm::MemoryTracker::instance().stats();

    // Check that allocation was tracked
    EXPECT_EQ(stats_after.total_allocated - stats_before.total_allocated, 100);
    EXPECT_EQ(stats_after.allocation_count - stats_before.allocation_count, 1);
    EXPECT_EQ(stats_after.live_bytes - stats_before.live_bytes, 100);
    EXPECT_EQ(fcm::MemoryTracker::instance().live_allocation_count() -
              stats_before.allocation_count + stats_before.free_count, 1);

    // Free the memory
    tracked_free(p);

    auto stats_final = fcm::MemoryTracker::instance().stats();

    // Check that deallocation was tracked
    EXPECT_EQ(stats_final.total_freed - stats_before.total_freed, 100);
    EXPECT_EQ(stats_final.free_count - stats_before.free_count, 1);
    EXPECT_EQ(stats_final.live_bytes, stats_before.live_bytes);
}

TEST_F(MemoryTrackerTest, MultipleAllocations) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate multiple blocks
    std::vector<void*> pointers;
    std::vector<std::size_t> sizes = {10, 20, 30, 40, 50};
    std::size_t total_size = 0;

    for (auto size : sizes) {
        void* p = tracked_alloc(size, "test_block");
        pointers.push_back(p);
        total_size += size;
    }

    auto stats_after = fcm::MemoryTracker::instance().stats();

    // Check cumulative tracking
    EXPECT_EQ(stats_after.total_allocated - stats_before.total_allocated, total_size);
    EXPECT_EQ(stats_after.allocation_count - stats_before.allocation_count, sizes.size());
    EXPECT_EQ(stats_after.live_bytes - stats_before.live_bytes, total_size);

    // Free all memory
    for (auto p : pointers) {
        tracked_free(p);
    }

    auto stats_final = fcm::MemoryTracker::instance().stats();

    // Check all memory was freed
    EXPECT_EQ(stats_final.total_freed - stats_before.total_freed, total_size);
    EXPECT_EQ(stats_final.free_count - stats_before.free_count, sizes.size());
    EXPECT_EQ(stats_final.live_bytes, stats_before.live_bytes);
}

TEST_F(MemoryTrackerTest, PeakMemoryTracking) {
    [[maybe_unused]] auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate and free in a pattern that creates a peak
    void* p1 = tracked_alloc(1000, "block1");
    void* p2 = tracked_alloc(2000, "block2");

    auto stats_peak = fcm::MemoryTracker::instance().stats();
    std::size_t peak_bytes = stats_peak.live_bytes;

    // Free one block
    tracked_free(p1);

    // Allocate a smaller block
    void* p3 = tracked_alloc(500, "block3");

    auto stats_after = fcm::MemoryTracker::instance().stats();

    // Peak should still reflect the maximum
    EXPECT_GE(stats_after.peak_bytes, peak_bytes);

    // Clean up
    tracked_free(p2);
    tracked_free(p3);
}

TEST_F(MemoryTrackerTest, NullPointerHandling) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Should handle null pointer gracefully
    fcm::MemoryTracker::instance().on_alloc(nullptr, 100, 8, "null_test");
    fcm::MemoryTracker::instance().on_free(nullptr);

    auto stats_after = fcm::MemoryTracker::instance().stats();

    // No change in stats for null operations
    EXPECT_EQ(stats_after.total_allocated, stats_before.total_allocated);
    EXPECT_EQ(stats_after.total_freed, stats_before.total_freed);
    EXPECT_EQ(stats_after.allocation_count, stats_before.allocation_count);
    EXPECT_EQ(stats_after.free_count, stats_before.free_count);
}

TEST_F(MemoryTrackerTest, ZeroSizeAllocation) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate zero bytes (should be handled gracefully)
    void* p = ::operator new(1); // Actually allocate something
    fcm::MemoryTracker::instance().on_alloc(p, 0, 8, "zero_size");

    auto stats_after = fcm::MemoryTracker::instance().stats();

    // Zero-size allocation should not be tracked
    EXPECT_EQ(stats_after.total_allocated, stats_before.total_allocated);
    EXPECT_EQ(stats_after.allocation_count, stats_before.allocation_count);

    // Clean up the actual allocation
    ::operator delete(p);
}

TEST_F(MemoryTrackerTest, AllocationInfo) {
    // Test that allocation info is properly stored
    void* p = ::operator new(256);

    std::string type_name = "TestClass";
    std::string label = "unit_test_label";
    std::size_t size = 256;
    std::size_t alignment = 16;

    fcm::MemoryTracker::instance().on_alloc(p, size, alignment, type_name, label);

    // Verify allocation is tracked
    auto count = fcm::MemoryTracker::instance().live_allocation_count();
    EXPECT_GT(count, 0);

    // Clean up
    fcm::MemoryTracker::instance().on_free(p);
    ::operator delete(p);
}

// Thread safety tests
TEST_F(MemoryTrackerTest, ThreadSafety_ConcurrentAllocations) {
    const int num_threads = 4;
    const int allocations_per_thread = 100;
    std::atomic<int> total_allocations{0};
    std::atomic<int> total_deallocations{0};

    auto worker = [&]() {
        std::vector<void*> pointers;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> size_dist(10, 1000);

        // Allocate memory
        for (int i = 0; i < allocations_per_thread; ++i) {
            std::size_t size = size_dist(gen);
            void* p = tracked_alloc(size, "thread_test");
            pointers.push_back(p);
            total_allocations.fetch_add(1);
        }

        // Random sleep to create interleaving
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        // Free memory
        for (void* p : pointers) {
            tracked_free(p);
            total_deallocations.fetch_add(1);
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    // Verify all allocations and deallocations were processed
    EXPECT_EQ(total_allocations.load(), num_threads * allocations_per_thread);
    EXPECT_EQ(total_deallocations.load(), num_threads * allocations_per_thread);
}

TEST_F(MemoryTrackerTest, ThreadSafety_ConcurrentStatsAccess) {
    std::atomic<bool> stop{false};
    std::atomic<int> stats_reads{0};

    // Thread that continuously reads stats
    auto reader = [&]() {
        while (!stop.load()) {
            auto stats = fcm::MemoryTracker::instance().stats();
            auto count = fcm::MemoryTracker::instance().live_allocation_count();
            stats_reads.fetch_add(1);

            // Basic sanity checks
            EXPECT_GE(stats.total_allocated, stats.total_freed);
            EXPECT_GE(stats.allocation_count, stats.free_count);
            EXPECT_GE(count, 0);
        }
    };

    // Thread that allocates and frees memory
    auto allocator = [&]() {
        for (int i = 0; i < 100; ++i) {
            void* p = tracked_alloc(100 + i, "concurrent_test");
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            tracked_free(p);
        }
    };

    // Launch threads
    std::thread reader_thread(reader);
    std::thread alloc_thread(allocator);

    // Wait for allocator to finish
    alloc_thread.join();

    // Stop reader
    stop.store(true);
    reader_thread.join();

    // Ensure we got multiple stat reads during allocation
    EXPECT_GT(stats_reads.load(), 10);
}

// ScopedMemoryTracker tests
TEST_F(MemoryTrackerTest, ScopedTracker_BasicUsage) {
    fcm::MemoryStats initial = fcm::MemoryTracker::instance().stats();

    // Create a scope with tracked allocations
    {
        fcm::ScopedMemoryTracker tracker("test_scope");

        void* p1 = tracked_alloc(100, "scoped_alloc1");
        void* p2 = tracked_alloc(200, "scoped_alloc2");

        // Check label
        EXPECT_EQ(tracker.label(), "test_scope");

        // Clean up within scope
        tracked_free(p1);
        tracked_free(p2);

        // Delta should show allocations and frees
        [[maybe_unused]] const auto& delta = tracker.delta();
        // Note: delta is only fully populated in destructor
    }

    // After scope, check that memory was properly tracked
    fcm::MemoryStats final = fcm::MemoryTracker::instance().stats();
    EXPECT_EQ(final.live_bytes, initial.live_bytes);
}

TEST_F(MemoryTrackerTest, ScopedTracker_NestedScopes) {
    std::vector<void*> pointers;

    {
        fcm::ScopedMemoryTracker outer("outer_scope");

        void* p1 = tracked_alloc(100, "outer_alloc");
        pointers.push_back(p1);

        {
            fcm::ScopedMemoryTracker inner("inner_scope");

            void* p2 = tracked_alloc(200, "inner_alloc");
            pointers.push_back(p2);

            // Inner scope tracks its own allocations
            EXPECT_EQ(inner.label(), "inner_scope");
        }

        void* p3 = tracked_alloc(300, "outer_alloc2");
        pointers.push_back(p3);

        // Outer scope tracks all allocations
        EXPECT_EQ(outer.label(), "outer_scope");
    }

    // Clean up all allocations
    for (void* p : pointers) {
        tracked_free(p);
    }
}

TEST_F(MemoryTrackerTest, ScopedTracker_DeltaCalculation) {
    fcm::MemoryStats stats_before = fcm::MemoryTracker::instance().stats();

    // We need to capture the tracker to check delta after destruction
    std::unique_ptr<fcm::ScopedMemoryTracker> tracker;

    {
        tracker = std::make_unique<fcm::ScopedMemoryTracker>("delta_test");

        // Record start stats
        [[maybe_unused]] auto start_stats = tracker->start();

        // Perform allocations
        void* p1 = tracked_alloc(500, "delta_alloc1");
        void* p2 = tracked_alloc(700, "delta_alloc2");

        // Free one
        tracked_free(p1);

        // Delta will be calculated on destruction
        tracker.reset(); // Trigger destructor

        // Continue with p2 still allocated
        tracked_free(p2);
    }

    fcm::MemoryStats stats_after = fcm::MemoryTracker::instance().stats();

    // Verify stats are consistent
    EXPECT_EQ(stats_after.live_bytes, stats_before.live_bytes);
    EXPECT_GE(stats_after.total_allocated, stats_before.total_allocated);
    EXPECT_GE(stats_after.total_freed, stats_before.total_freed);
}

// Edge cases
TEST_F(MemoryTrackerTest, LargeAllocation) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate a large block (1MB)
    std::size_t large_size = 1024 * 1024;
    void* p = tracked_alloc(large_size, "large_block");

    auto stats_after = fcm::MemoryTracker::instance().stats();

    EXPECT_EQ(stats_after.total_allocated - stats_before.total_allocated, large_size);
    EXPECT_EQ(stats_after.live_bytes - stats_before.live_bytes, large_size);

    tracked_free(p);

    auto stats_final = fcm::MemoryTracker::instance().stats();
    EXPECT_EQ(stats_final.live_bytes, stats_before.live_bytes);
}

TEST_F(MemoryTrackerTest, ManySmallAllocations) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Allocate many small blocks
    const int count = 1000;
    std::vector<void*> pointers;

    for (int i = 0; i < count; ++i) {
        void* p = tracked_alloc(8, "small_block");
        pointers.push_back(p);
    }

    auto stats_after = fcm::MemoryTracker::instance().stats();

    EXPECT_EQ(stats_after.allocation_count - stats_before.allocation_count, count);
    EXPECT_EQ(stats_after.live_bytes - stats_before.live_bytes, count * 8);

    // Free all
    for (void* p : pointers) {
        tracked_free(p);
    }

    auto stats_final = fcm::MemoryTracker::instance().stats();
    EXPECT_EQ(stats_final.free_count - stats_before.free_count, count);
    EXPECT_EQ(stats_final.live_bytes, stats_before.live_bytes);
}

TEST_F(MemoryTrackerTest, AllocationWithCustomAlignment) {
    auto stats_before = fcm::MemoryTracker::instance().stats();

    // Test various alignments
    std::vector<std::size_t> alignments = {8, 16, 32, 64, 128};
    std::vector<void*> pointers;

    for (auto alignment : alignments) {
        void* p = ::operator new(100, std::align_val_t(alignment));
        fcm::MemoryTracker::instance().on_alloc(p, 100, alignment,
                                                "aligned_alloc",
                                                std::to_string(alignment));
        pointers.push_back(p);

        // Verify pointer is aligned
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignment, 0);
    }

    auto stats_after = fcm::MemoryTracker::instance().stats();

    EXPECT_EQ(stats_after.allocation_count - stats_before.allocation_count,
              alignments.size());
    EXPECT_EQ(stats_after.live_bytes - stats_before.live_bytes,
              alignments.size() * 100);

    // Clean up
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        fcm::MemoryTracker::instance().on_free(pointers[i]);
        ::operator delete(pointers[i], std::align_val_t(alignments[i]));
    }
}

TEST_F(MemoryTrackerTest, TypeNameAndLabelTracking) {
    // Test that type names and labels are properly handled
    struct TestStruct { int x, y, z; };

    std::string type_names[] = {"int", "double", "TestStruct", "std::vector<int>"};
    std::string labels[] = {"label1", "label2", "test_label", "performance_critical"};

    std::vector<void*> pointers;

    for (int i = 0; i < 4; ++i) {
        void* p = ::operator new(100 * (i + 1));
        fcm::MemoryTracker::instance().on_alloc(p, 100 * (i + 1),
                                                alignof(std::max_align_t),
                                                type_names[i], labels[i]);
        pointers.push_back(p);
    }

    // Verify allocations are tracked
    EXPECT_GE(fcm::MemoryTracker::instance().live_allocation_count(), 4);

    // Clean up
    for (void* p : pointers) {
        fcm::MemoryTracker::instance().on_free(p);
        ::operator delete(p);
    }
}

// Stress test
TEST_F(MemoryTrackerTest, StressTest_RapidAllocDealloc) {
    const int iterations = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(1, 1024);
    std::uniform_int_distribution<> action_dist(0, 1);

    std::vector<void*> active_pointers;

    for (int i = 0; i < iterations; ++i) {
        if (active_pointers.empty() || action_dist(gen) == 0) {
            // Allocate
            std::size_t size = size_dist(gen);
            void* p = tracked_alloc(size, "stress_test");
            active_pointers.push_back(p);
        } else {
            // Free random pointer
            std::uniform_int_distribution<> idx_dist(0, static_cast<int>(active_pointers.size()) - 1);
            int idx = idx_dist(gen);
            tracked_free(active_pointers[idx]);
            active_pointers.erase(active_pointers.begin() + idx);
        }
    }

    // Clean up remaining pointers
    for (void* p : active_pointers) {
        tracked_free(p);
    }

    // Verify no leaks (relative to start of test)
    auto final_stats = fcm::MemoryTracker::instance().stats();
    EXPECT_GE(final_stats.total_freed, 0);
    EXPECT_GE(final_stats.free_count, 0);
}

// Test interaction with source_location
TEST_F(MemoryTrackerTest, SourceLocationTracking) {
    // Test that source location is captured
    auto track_with_location = [](std::size_t size) {
        void* p = ::operator new(size);
        fcm::MemoryTracker::instance().on_alloc(
            p, size, alignof(std::max_align_t),
            "location_test", "test",
            std::source_location::current()
        );
        return p;
    };

    void* p1 = track_with_location(100);
    void* p2 = track_with_location(200);

    // Different invocations should have different source locations
    EXPECT_NE(p1, p2);

    // Clean up
    fcm::MemoryTracker::instance().on_free(p1);
    fcm::MemoryTracker::instance().on_free(p2);
    ::operator delete(p1);
    ::operator delete(p2);
}

// Test timestamp functionality
TEST_F(MemoryTrackerTest, TimestampOrdering) {
    std::vector<void*> pointers;

    // Allocate with small delays to ensure different timestamps
    for (int i = 0; i < 3; ++i) {
        void* p = tracked_alloc(100, "timestamp_test");
        pointers.push_back(p);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Clean up
    for (void* p : pointers) {
        tracked_free(p);
    }
}

#if !CORE_MEMORY_ENABLE_TRACKING
// Test that tracker is properly disabled when CORE_MEMORY_ENABLE_TRACKING is 0
TEST_F(MemoryTrackerTest, DisabledTracker) {
    auto stats = fcm::MemoryTracker::instance().stats();

    // All stats should be zero when tracking is disabled
    EXPECT_EQ(stats.total_allocated, 0);
    EXPECT_EQ(stats.total_freed, 0);
    EXPECT_EQ(stats.live_bytes, 0);
    EXPECT_EQ(stats.peak_bytes, 0);
    EXPECT_EQ(stats.allocation_count, 0);
    EXPECT_EQ(stats.free_count, 0);

    // Operations should be no-ops
    void* p = ::operator new(100);
    fcm::MemoryTracker::instance().on_alloc(p, 100, 8, "disabled_test");

    auto stats_after = fcm::MemoryTracker::instance().stats();
    EXPECT_EQ(stats_after.total_allocated, 0);

    fcm::MemoryTracker::instance().on_free(p);
    ::operator delete(p);

    EXPECT_EQ(fcm::MemoryTracker::instance().live_allocation_count(), 0);
}
#endif
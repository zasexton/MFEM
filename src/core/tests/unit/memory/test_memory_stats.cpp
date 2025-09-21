#include <gtest/gtest.h>
#include <core/memory/memory_stats.h>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>  // For std::memcpy

using namespace fem::core::memory;

class MemoryStatsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Basic MemoryStats tests
TEST_F(MemoryStatsTest, DefaultConstruction) {
    MemoryStats stats;

    EXPECT_EQ(0u, stats.total_allocated);
    EXPECT_EQ(0u, stats.total_freed);
    EXPECT_EQ(0u, stats.live_bytes);
    EXPECT_EQ(0u, stats.peak_bytes);
    EXPECT_EQ(0u, stats.allocation_count);
    EXPECT_EQ(0u, stats.free_count);
}

TEST_F(MemoryStatsTest, ManualUpdate) {
    MemoryStats stats;

    // Simulate allocation
    stats.total_allocated = 1024;
    stats.live_bytes = 1024;
    stats.peak_bytes = 1024;
    stats.allocation_count = 1;

    EXPECT_EQ(1024u, stats.total_allocated);
    EXPECT_EQ(0u, stats.total_freed);
    EXPECT_EQ(1024u, stats.live_bytes);
    EXPECT_EQ(1024u, stats.peak_bytes);
    EXPECT_EQ(1u, stats.allocation_count);
    EXPECT_EQ(0u, stats.free_count);

    // Simulate more allocations
    stats.total_allocated += 2048;
    stats.live_bytes += 2048;
    stats.peak_bytes = stats.live_bytes;  // Update peak
    stats.allocation_count++;

    EXPECT_EQ(3072u, stats.total_allocated);
    EXPECT_EQ(3072u, stats.live_bytes);
    EXPECT_EQ(3072u, stats.peak_bytes);
    EXPECT_EQ(2u, stats.allocation_count);

    // Simulate deallocation
    stats.total_freed = 1024;
    stats.live_bytes = stats.total_allocated - stats.total_freed;
    stats.free_count = 1;

    EXPECT_EQ(3072u, stats.total_allocated);
    EXPECT_EQ(1024u, stats.total_freed);
    EXPECT_EQ(2048u, stats.live_bytes);
    EXPECT_EQ(3072u, stats.peak_bytes);  // Peak should remain unchanged
    EXPECT_EQ(1u, stats.free_count);
}

// accumulate function tests
TEST_F(MemoryStatsTest, Accumulate_EmptyIntoEmpty) {
    MemoryStats into;
    MemoryStats from;

    accumulate(into, from);

    EXPECT_EQ(0u, into.total_allocated);
    EXPECT_EQ(0u, into.total_freed);
    EXPECT_EQ(0u, into.live_bytes);
    EXPECT_EQ(0u, into.peak_bytes);
    EXPECT_EQ(0u, into.allocation_count);
    EXPECT_EQ(0u, into.free_count);
}

TEST_F(MemoryStatsTest, Accumulate_IntoEmpty) {
    MemoryStats into;
    MemoryStats from;

    from.total_allocated = 1000;
    from.total_freed = 200;
    from.live_bytes = 800;
    from.peak_bytes = 900;
    from.allocation_count = 10;
    from.free_count = 2;

    accumulate(into, from);

    EXPECT_EQ(1000u, into.total_allocated);
    EXPECT_EQ(200u, into.total_freed);
    EXPECT_EQ(800u, into.live_bytes);
    EXPECT_EQ(900u, into.peak_bytes);
    EXPECT_EQ(10u, into.allocation_count);
    EXPECT_EQ(2u, into.free_count);
}

TEST_F(MemoryStatsTest, Accumulate_IntoExisting) {
    MemoryStats into;
    into.total_allocated = 500;
    into.total_freed = 100;
    into.live_bytes = 400;
    into.peak_bytes = 450;
    into.allocation_count = 5;
    into.free_count = 1;

    MemoryStats from;
    from.total_allocated = 1000;
    from.total_freed = 200;
    from.live_bytes = 800;
    from.peak_bytes = 900;
    from.allocation_count = 10;
    from.free_count = 2;

    accumulate(into, from);

    EXPECT_EQ(1500u, into.total_allocated);
    EXPECT_EQ(300u, into.total_freed);
    EXPECT_EQ(1200u, into.live_bytes);
    EXPECT_EQ(900u, into.peak_bytes);  // Should take the larger peak
    EXPECT_EQ(15u, into.allocation_count);
    EXPECT_EQ(3u, into.free_count);
}

TEST_F(MemoryStatsTest, Accumulate_PeakComparison) {
    MemoryStats into;
    into.peak_bytes = 1000;

    MemoryStats from1;
    from1.peak_bytes = 500;

    accumulate(into, from1);
    EXPECT_EQ(1000u, into.peak_bytes);  // Should keep the larger peak

    MemoryStats from2;
    from2.peak_bytes = 2000;

    accumulate(into, from2);
    EXPECT_EQ(2000u, into.peak_bytes);  // Should update to larger peak
}

TEST_F(MemoryStatsTest, Accumulate_Multiple) {
    MemoryStats total;

    // Accumulate from multiple sources
    std::vector<MemoryStats> sources(5);
    for (size_t i = 0; i < sources.size(); ++i) {
        sources[i].total_allocated = (i + 1) * 100;
        sources[i].total_freed = (i + 1) * 10;
        sources[i].live_bytes = sources[i].total_allocated - sources[i].total_freed;
        sources[i].peak_bytes = (i + 1) * 150;
        sources[i].allocation_count = (i + 1);
        sources[i].free_count = i;
    }

    for (const auto& source : sources) {
        accumulate(total, source);
    }

    // Sum: 100 + 200 + 300 + 400 + 500 = 1500
    EXPECT_EQ(1500u, total.total_allocated);
    // Sum: 10 + 20 + 30 + 40 + 50 = 150
    EXPECT_EQ(150u, total.total_freed);
    // Sum of live bytes
    EXPECT_EQ(1350u, total.live_bytes);
    // Max peak: 5 * 150 = 750
    EXPECT_EQ(750u, total.peak_bytes);
    // Sum: 1 + 2 + 3 + 4 + 5 = 15
    EXPECT_EQ(15u, total.allocation_count);
    // Sum: 0 + 1 + 2 + 3 + 4 = 10
    EXPECT_EQ(10u, total.free_count);
}

// Realistic usage scenarios
TEST_F(MemoryStatsTest, SimulateAllocationPattern) {
    MemoryStats stats;

    // Simulate allocation lifecycle
    auto allocate = [&stats](size_t bytes) {
        stats.total_allocated += bytes;
        stats.live_bytes += bytes;
        if (stats.live_bytes > stats.peak_bytes) {
            stats.peak_bytes = stats.live_bytes;
        }
        stats.allocation_count++;
    };

    auto deallocate = [&stats](size_t bytes) {
        stats.total_freed += bytes;
        stats.live_bytes -= bytes;
        stats.free_count++;
    };

    // Allocation pattern
    allocate(1024);  // 1KB
    EXPECT_EQ(1024u, stats.live_bytes);
    EXPECT_EQ(1024u, stats.peak_bytes);

    allocate(2048);  // 2KB
    EXPECT_EQ(3072u, stats.live_bytes);
    EXPECT_EQ(3072u, stats.peak_bytes);

    deallocate(1024);  // Free 1KB
    EXPECT_EQ(2048u, stats.live_bytes);
    EXPECT_EQ(3072u, stats.peak_bytes);  // Peak unchanged

    allocate(4096);  // 4KB
    EXPECT_EQ(6144u, stats.live_bytes);
    EXPECT_EQ(6144u, stats.peak_bytes);  // New peak

    deallocate(2048);  // Free 2KB
    deallocate(4096);  // Free 4KB
    EXPECT_EQ(0u, stats.live_bytes);
    EXPECT_EQ(6144u, stats.peak_bytes);  // Peak remains

    // Verify final stats
    EXPECT_EQ(7168u, stats.total_allocated);  // 1024 + 2048 + 4096
    EXPECT_EQ(7168u, stats.total_freed);      // 1024 + 2048 + 4096
    EXPECT_EQ(3u, stats.allocation_count);
    EXPECT_EQ(3u, stats.free_count);
}

TEST_F(MemoryStatsTest, MultiplePools) {
    // Simulate stats from multiple memory pools
    MemoryStats pool1;
    pool1.total_allocated = 10000;
    pool1.total_freed = 3000;
    pool1.live_bytes = 7000;
    pool1.peak_bytes = 8000;
    pool1.allocation_count = 100;
    pool1.free_count = 30;

    MemoryStats pool2;
    pool2.total_allocated = 5000;
    pool2.total_freed = 4000;
    pool2.live_bytes = 1000;
    pool2.peak_bytes = 4500;
    pool2.allocation_count = 50;
    pool2.free_count = 40;

    MemoryStats pool3;
    pool3.total_allocated = 20000;
    pool3.total_freed = 20000;
    pool3.live_bytes = 0;
    pool3.peak_bytes = 15000;
    pool3.allocation_count = 200;
    pool3.free_count = 200;

    // Aggregate stats
    MemoryStats global;
    accumulate(global, pool1);
    accumulate(global, pool2);
    accumulate(global, pool3);

    EXPECT_EQ(35000u, global.total_allocated);
    EXPECT_EQ(27000u, global.total_freed);
    EXPECT_EQ(8000u, global.live_bytes);
    EXPECT_EQ(15000u, global.peak_bytes);  // Max of all peaks
    EXPECT_EQ(350u, global.allocation_count);
    EXPECT_EQ(270u, global.free_count);
}

TEST_F(MemoryStatsTest, MemoryLeak_Detection) {
    MemoryStats stats;

    // Simulate memory operations
    stats.total_allocated = 10000;
    stats.total_freed = 8000;
    stats.live_bytes = stats.total_allocated - stats.total_freed;

    // Check for potential leak
    bool has_leak = stats.live_bytes > 0;
    EXPECT_TRUE(has_leak);
    EXPECT_EQ(2000u, stats.live_bytes);  // 2KB leaked

    // After cleanup
    stats.total_freed = stats.total_allocated;
    stats.live_bytes = 0;

    has_leak = stats.live_bytes > 0;
    EXPECT_FALSE(has_leak);
}

TEST_F(MemoryStatsTest, ZeroCopy) {
    // MemoryStats should be trivially copyable for efficient passing
    EXPECT_TRUE(std::is_trivially_copyable_v<MemoryStats>);
    EXPECT_TRUE(std::is_standard_layout_v<MemoryStats>);

    // Can be used in arrays efficiently
    MemoryStats stats_array[10] = {};
    for (int i = 0; i < 10; ++i) {
        stats_array[i].total_allocated = i * 100;
    }

    // Can be memcpy'd
    MemoryStats copy;
    std::memcpy(&copy, &stats_array[5], sizeof(MemoryStats));
    EXPECT_EQ(500u, copy.total_allocated);
}

TEST_F(MemoryStatsTest, Ratios_Calculations) {
    MemoryStats stats;
    stats.total_allocated = 10000;
    stats.total_freed = 7500;
    stats.live_bytes = 2500;
    stats.peak_bytes = 8000;
    stats.allocation_count = 100;
    stats.free_count = 75;

    // Calculate various ratios that might be useful
    double fragmentation_ratio = 0.0;
    if (stats.peak_bytes > 0) {
        fragmentation_ratio = static_cast<double>(stats.live_bytes) / static_cast<double>(stats.peak_bytes);
    }
    EXPECT_DOUBLE_EQ(0.3125, fragmentation_ratio);  // 2500/8000

    double deallocation_ratio = 0.0;
    if (stats.allocation_count > 0) {
        deallocation_ratio = static_cast<double>(stats.free_count) / static_cast<double>(stats.allocation_count);
    }
    EXPECT_DOUBLE_EQ(0.75, deallocation_ratio);  // 75/100

    double average_allocation_size = 0.0;
    if (stats.allocation_count > 0) {
        average_allocation_size = static_cast<double>(stats.total_allocated) / static_cast<double>(stats.allocation_count);
    }
    EXPECT_DOUBLE_EQ(100.0, average_allocation_size);  // 10000/100
}

TEST_F(MemoryStatsTest, EdgeCases_Overflow) {
    MemoryStats stats;

    // Test with large values (but not overflow)
    size_t large_value = std::numeric_limits<size_t>::max() / 2;
    stats.total_allocated = large_value;
    stats.total_freed = large_value - 1000;
    stats.live_bytes = 1000;
    stats.peak_bytes = large_value;

    EXPECT_EQ(large_value, stats.total_allocated);
    EXPECT_EQ(1000u, stats.live_bytes);

    // Accumulate should handle large values
    MemoryStats other;
    other.total_allocated = 1000;
    other.peak_bytes = 2000;

    accumulate(stats, other);
    EXPECT_EQ(large_value + 1000, stats.total_allocated);
    EXPECT_EQ(large_value, stats.peak_bytes);  // Should keep larger
}

TEST_F(MemoryStatsTest, Consistency_Checks) {
    MemoryStats stats;

    // Helper to check consistency
    auto is_consistent = [](const MemoryStats& s) {
        // live_bytes should equal total_allocated - total_freed
        if (s.total_allocated >= s.total_freed) {
            return s.live_bytes == (s.total_allocated - s.total_freed);
        }
        return false;  // freed more than allocated is inconsistent
    };

    // Valid state
    stats.total_allocated = 5000;
    stats.total_freed = 3000;
    stats.live_bytes = 2000;
    EXPECT_TRUE(is_consistent(stats));

    // Invalid state (would indicate a bug in tracking)
    stats.live_bytes = 1000;  // Should be 2000
    EXPECT_FALSE(is_consistent(stats));

    // Fix it
    stats.live_bytes = stats.total_allocated - stats.total_freed;
    EXPECT_TRUE(is_consistent(stats));
}
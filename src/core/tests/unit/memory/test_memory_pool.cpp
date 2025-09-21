#include <gtest/gtest.h>
#include <core/memory/memory_pool.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <set>
#include <cstring>

namespace fcm = fem::core::memory;

class MemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        resource_ = fcm::new_delete_resource();
    }

    fcm::memory_resource* resource_ = nullptr;
};

// Basic functionality tests
TEST_F(MemoryPoolTest, BasicConstruction) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 64
    };

    fcm::MemoryPool pool(cfg, resource_);

    EXPECT_EQ(pool.object_size(), 32);
    EXPECT_EQ(pool.alignment(), 8);
    EXPECT_EQ(pool.nodes_per_block(), 64);
    EXPECT_EQ(pool.block_bytes(), 32 * 64);
    EXPECT_EQ(pool.block_count(), 0);  // No blocks allocated yet
}

TEST_F(MemoryPoolTest, MinimumObjectSize) {
    // Object size smaller than pointer size should be adjusted
    fcm::MemoryPool::Config cfg{
        .object_size = 2,  // Smaller than sizeof(void*)
        .alignment = 8,
        .nodes_per_block = 100
    };

    fcm::MemoryPool pool(cfg, resource_);

    EXPECT_GE(pool.object_size(), sizeof(void*));
}

// Allocation and deallocation tests
TEST_F(MemoryPoolTest, SingleAllocation) {
    fcm::MemoryPool::Config cfg{
        .object_size = 64,
        .alignment = 16,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);

    // Should have allocated one block
    EXPECT_EQ(pool.block_count(), 1);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0);

    pool.deallocate(p);
}

TEST_F(MemoryPoolTest, MultipleAllocations) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 5
    };

    fcm::MemoryPool pool(cfg, resource_);

    std::vector<void*> pointers;

    // Allocate 10 objects (should need 2 blocks)
    for (int i = 0; i < 10; ++i) {
        void* p = pool.allocate();
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);
    }

    EXPECT_EQ(pool.block_count(), 2);

    // Check all pointers are unique
    std::set<void*> unique_ptrs(pointers.begin(), pointers.end());
    EXPECT_EQ(unique_ptrs.size(), 10);

    // Deallocate all
    for (void* p : pointers) {
        pool.deallocate(p);
    }

    // Blocks are still allocated but nodes are free
    EXPECT_EQ(pool.block_count(), 2);
    EXPECT_EQ(pool.free_count(), 10);
}

TEST_F(MemoryPoolTest, AllocationReuse) {
    fcm::MemoryPool::Config cfg{
        .object_size = 48,
        .alignment = 8,
        .nodes_per_block = 4
    };

    fcm::MemoryPool pool(cfg, resource_);

    // Allocate and deallocate
    void* p1 = pool.allocate();
    ASSERT_NE(p1, nullptr);
    pool.deallocate(p1);

    // Should reuse the same memory
    void* p2 = pool.allocate();
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p1, p2);
}

TEST_F(MemoryPoolTest, DeallocateNull) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    // Should handle null gracefully
    pool.deallocate(nullptr);
    EXPECT_EQ(pool.block_count(), 0);
}

// Reserve tests
TEST_F(MemoryPoolTest, ReserveNodes) {
    fcm::MemoryPool::Config cfg{
        .object_size = 64,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    // Reserve 25 nodes (should allocate 3 blocks)
    pool.reserve_nodes(25);

    EXPECT_EQ(pool.block_count(), 3);
    EXPECT_GE(pool.free_count(), 25);

    // Allocate should not need new blocks
    std::vector<void*> pointers;
    for (int i = 0; i < 25; ++i) {
        void* p = pool.allocate();
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);
    }

    EXPECT_EQ(pool.block_count(), 3);  // No new blocks needed

    // Cleanup
    for (void* p : pointers) {
        pool.deallocate(p);
    }
}

TEST_F(MemoryPoolTest, ReserveIdempotent) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    pool.reserve_nodes(15);
    std::size_t blocks1 = pool.block_count();

    // Reserve again with smaller number should not allocate more
    pool.reserve_nodes(10);
    EXPECT_EQ(pool.block_count(), blocks1);
}

// Result-based API tests
TEST_F(MemoryPoolTest, TryAllocate_Success) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    auto result = pool.try_allocate();
    ASSERT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);

    pool.deallocate(result.value());
}

TEST_F(MemoryPoolTest, TryReserveNodes_Success) {
    fcm::MemoryPool::Config cfg{
        .object_size = 64,
        .alignment = 8,
        .nodes_per_block = 8
    };

    fcm::MemoryPool pool(cfg, resource_);

    auto result = pool.try_reserve_nodes(20);
    ASSERT_TRUE(result.is_ok());

    EXPECT_GE(pool.free_count(), 20);
}

// Move semantics tests
TEST_F(MemoryPoolTest, MoveConstruction) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool1(cfg, resource_);
    void* p1 = pool1.allocate();
    void* p2 = pool1.allocate();

    fcm::MemoryPool pool2(std::move(pool1));

    EXPECT_EQ(pool2.object_size(), 32);
    EXPECT_EQ(pool2.block_count(), 1);

    // Should be able to deallocate to moved pool
    pool2.deallocate(p1);
    pool2.deallocate(p2);

    // And allocate again
    void* p3 = pool2.allocate();
    ASSERT_NE(p3, nullptr);
    pool2.deallocate(p3);
}

TEST_F(MemoryPoolTest, MoveAssignment) {
    fcm::MemoryPool::Config cfg1{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool::Config cfg2{
        .object_size = 64,
        .alignment = 16,
        .nodes_per_block = 5
    };

    fcm::MemoryPool pool1(cfg1, resource_);
    fcm::MemoryPool pool2(cfg2, resource_);

    void* p1 = pool1.allocate();

    pool2 = std::move(pool1);

    EXPECT_EQ(pool2.object_size(), 32);
    EXPECT_EQ(pool2.nodes_per_block(), 10);

    pool2.deallocate(p1);
}

// Release tests
TEST_F(MemoryPoolTest, ReleaseAll) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    // Allocate some nodes
    std::vector<void*> pointers;
    for (int i = 0; i < 15; ++i) {
        pointers.push_back(pool.allocate());
    }

    EXPECT_GE(pool.block_count(), 2);

    pool.release_all();

    EXPECT_EQ(pool.block_count(), 0);
    EXPECT_EQ(pool.free_count(), 0);

    // Can still allocate after release
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    pool.deallocate(p);
}

TEST_F(MemoryPoolTest, ShrinkToFit) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    pool.reserve_nodes(30);
    std::size_t blocks_before = pool.block_count();

    // shrink_to_fit is intentionally no-op for stability
    pool.shrink_to_fit();
    EXPECT_EQ(pool.block_count(), blocks_before);
}

// Stress tests
TEST_F(MemoryPoolTest, StressAllocationPattern) {
    fcm::MemoryPool::Config cfg{
        .object_size = 48,
        .alignment = 8,
        .nodes_per_block = 16
    };

    fcm::MemoryPool pool(cfg, resource_);

    std::vector<void*> allocated;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> action(0, 1);

    for (int i = 0; i < 1000; ++i) {
        if (action(gen) == 0 || allocated.empty()) {
            // Allocate
            void* p = pool.allocate();
            ASSERT_NE(p, nullptr);
            allocated.push_back(p);
        } else {
            // Deallocate random
            std::uniform_int_distribution<> idx(0, static_cast<int>(allocated.size()) - 1);
            int index = idx(gen);
            pool.deallocate(allocated[index]);
            allocated.erase(allocated.begin() + index);
        }
    }

    // Cleanup
    for (void* p : allocated) {
        pool.deallocate(p);
    }
}

TEST_F(MemoryPoolTest, LargeObjectSize) {
    fcm::MemoryPool::Config cfg{
        .object_size = 4096,  // Large objects
        .alignment = 64,
        .nodes_per_block = 4
    };

    fcm::MemoryPool pool(cfg, resource_);

    std::vector<void*> pointers;
    for (int i = 0; i < 10; ++i) {
        void* p = pool.allocate();
        ASSERT_NE(p, nullptr);

        // Verify alignment
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64, 0);

        // Write pattern to verify no overlap
        std::memset(p, i, 4096);
        pointers.push_back(p);
    }

    // Verify patterns
    for (int i = 0; i < 10; ++i) {
        auto* bytes = static_cast<unsigned char*>(pointers[i]);
        for (int j = 0; j < 4096; ++j) {
            EXPECT_EQ(bytes[j], i) << "Corruption at index " << j;
        }
    }

    // Cleanup
    for (void* p : pointers) {
        pool.deallocate(p);
    }
}

// Edge cases
TEST_F(MemoryPoolTest, SingleNodePerBlock) {
    fcm::MemoryPool::Config cfg{
        .object_size = 1024,
        .alignment = 8,
        .nodes_per_block = 1  // Only one node per block
    };

    fcm::MemoryPool pool(cfg, resource_);

    void* p1 = pool.allocate();
    void* p2 = pool.allocate();
    void* p3 = pool.allocate();

    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    ASSERT_NE(p3, nullptr);

    EXPECT_EQ(pool.block_count(), 3);  // Each allocation needs a new block

    pool.deallocate(p1);
    pool.deallocate(p2);
    pool.deallocate(p3);
}

TEST_F(MemoryPoolTest, AllocateDeallocatePattern) {
    fcm::MemoryPool::Config cfg{
        .object_size = 64,
        .alignment = 8,
        .nodes_per_block = 8
    };

    fcm::MemoryPool pool(cfg, resource_);

    // LIFO deallocation pattern
    std::vector<void*> stack;
    for (int i = 0; i < 20; ++i) {
        stack.push_back(pool.allocate());
    }

    while (!stack.empty()) {
        pool.deallocate(stack.back());
        stack.pop_back();
    }

    EXPECT_EQ(pool.free_count(), 24);  // 3 blocks * 8 nodes

    // FIFO deallocation pattern
    std::vector<void*> queue;
    for (int i = 0; i < 20; ++i) {
        queue.push_back(pool.allocate());
    }

    for (void* p : queue) {
        pool.deallocate(p);
    }

    EXPECT_EQ(pool.free_count(), 24);
}

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(MemoryPoolTest, TelemetryTracking) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 10
    };

    fcm::MemoryPool pool(cfg, resource_);

    std::vector<std::string> events;
    pool.set_telemetry_callback(
        [&events](const char* event, [[maybe_unused]] const fcm::MemoryPool::telemetry_t& t) {
            events.push_back(event);
        });

    void* p1 = pool.allocate();
    void* p2 = pool.allocate();

    auto& telemetry = pool.telemetry();
    EXPECT_EQ(telemetry.blocks_allocated, 1);
    EXPECT_EQ(telemetry.nodes_total, 10);
    EXPECT_EQ(telemetry.alloc_calls, 2);

    pool.deallocate(p1);
    EXPECT_EQ(telemetry.dealloc_calls, 1);

    pool.deallocate(p2);
    EXPECT_EQ(telemetry.dealloc_calls, 2);
    EXPECT_EQ(telemetry.nodes_free, 10);

    EXPECT_GE(events.size(), 1);
    EXPECT_EQ(events[0], "allocate_block");
}

TEST_F(MemoryPoolTest, PeakUsageTracking) {
    fcm::MemoryPool::Config cfg{
        .object_size = 32,
        .alignment = 8,
        .nodes_per_block = 5
    };

    fcm::MemoryPool pool(cfg, resource_);

    std::vector<void*> pointers;

    // Allocate 8 nodes
    for (int i = 0; i < 8; ++i) {
        pointers.push_back(pool.allocate());
    }

    auto& telemetry = pool.telemetry();
    EXPECT_EQ(telemetry.peak_in_use, 8);

    // Deallocate some
    for (int i = 0; i < 4; ++i) {
        pool.deallocate(pointers[i]);
    }

    // Peak should remain 8
    EXPECT_EQ(telemetry.peak_in_use, 8);

    // Allocate more
    for (int i = 0; i < 6; ++i) {
        pointers.push_back(pool.allocate());
    }

    // New peak
    EXPECT_EQ(telemetry.peak_in_use, 10);
}
#endif
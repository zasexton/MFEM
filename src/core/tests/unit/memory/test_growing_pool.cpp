#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include <memory/growing_pool.h>

namespace fcm = fem::core::memory;

class GrowingPoolSimpleTest : public ::testing::Test {
protected:
    fcm::GrowingPool::Config create_basic_config(std::size_t object_size = 32) {
        return fcm::GrowingPool::Config{
            .object_size = object_size,
            .alignment = alignof(std::max_align_t),
            .initial_nodes_per_block = 4,  // Small for easier testing
            .growth_factor = 2.0,
            .max_nodes_per_block = 64
        };
    }
};

// === Basic Construction Tests ===

TEST_F(GrowingPoolSimpleTest, BasicConstruction) {
    auto config = create_basic_config();
    fcm::GrowingPool pool(config);

    EXPECT_EQ(pool.current_nodes_per_block(), config.initial_nodes_per_block);
    EXPECT_EQ(pool.block_count(), 0);
}

TEST_F(GrowingPoolSimpleTest, BasicAllocation) {
    auto config = create_basic_config();
    fcm::GrowingPool pool(config);

    void* ptr = pool.allocate();
    EXPECT_NE(ptr, nullptr);

    pool.deallocate(ptr);
}

TEST_F(GrowingPoolSimpleTest, MultipleAllocations) {
    auto config = create_basic_config();
    fcm::GrowingPool pool(config);

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate());
        EXPECT_NE(ptrs.back(), nullptr);
    }

    // Clean up
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
}

TEST_F(GrowingPoolSimpleTest, GrowthTriggering) {
    fcm::GrowingPool::Config config{
        .object_size = 16,
        .alignment = alignof(std::max_align_t),
        .initial_nodes_per_block = 2,
        .growth_factor = 2.0,
        .max_nodes_per_block = 32
    };

    fcm::GrowingPool pool(config);

    std::size_t initial_nodes_per_block = pool.current_nodes_per_block();

    // Allocate enough to trigger growth
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate());
    }

    // Should have grown beyond initial size
    EXPECT_GT(pool.current_nodes_per_block(), initial_nodes_per_block);

    // Clean up
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
}

TEST_F(GrowingPoolSimpleTest, MaxLimit) {
    fcm::GrowingPool::Config config{
        .object_size = 16,
        .alignment = alignof(std::max_align_t),
        .initial_nodes_per_block = 4,
        .growth_factor = 2.0,
        .max_nodes_per_block = 8  // Low limit for testing
    };

    fcm::GrowingPool pool(config);

    // Allocate enough to trigger growth beyond the limit
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
        ptrs.push_back(pool.allocate());
    }

    // Should not exceed max limit
    EXPECT_LE(pool.current_nodes_per_block(), 8);

    // Clean up
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
}

TEST_F(GrowingPoolSimpleTest, TryAllocate) {
    auto config = create_basic_config();
    fcm::GrowingPool pool(config);

    auto result = pool.try_allocate();
    EXPECT_TRUE(result.is_ok());
    EXPECT_NE(result.value(), nullptr);

    pool.deallocate(result.value());
}

TEST_F(GrowingPoolSimpleTest, MoveSemantics) {
    auto config = create_basic_config();
    fcm::GrowingPool original(config);

    // Allocate some memory
    void* ptr = original.allocate();
    EXPECT_NE(ptr, nullptr);
    std::size_t nodes_after_alloc = original.current_nodes_per_block();

    // Move construct
    fcm::GrowingPool moved(std::move(original));
    EXPECT_EQ(moved.current_nodes_per_block(), nodes_after_alloc);

    // Clean up using moved pool
    moved.deallocate(ptr);
}

TEST_F(GrowingPoolSimpleTest, CustomAlignment) {
    fcm::GrowingPool::Config config{
        .object_size = 64,
        .alignment = 64,  // Custom alignment
        .initial_nodes_per_block = 4,
        .growth_factor = 2.0,
        .max_nodes_per_block = 64
    };

    fcm::GrowingPool pool(config);

    void* ptr = pool.allocate();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0);

    pool.deallocate(ptr);
}

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(GrowingPoolSimpleTest, TelemetryCallback) {
    auto config = create_basic_config();
    fcm::GrowingPool pool(config);

    bool callback_called = false;
    pool.set_telemetry_callback([&](const char* event, const auto& telemetry) {
        callback_called = true;
        EXPECT_STREQ(event, "refill");
        EXPECT_GT(telemetry.refills, 0);
    });

    // Trigger refill by allocating
    void* ptr = pool.allocate();
    EXPECT_TRUE(callback_called);

    pool.deallocate(ptr);
}
#endif
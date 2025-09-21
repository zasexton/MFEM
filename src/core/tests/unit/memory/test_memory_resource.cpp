#include <gtest/gtest.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <set>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>

namespace fcm = fem::core::memory;

class MemoryResourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset default resource to new_delete for each test
        fcm::set_default_resource(fcm::new_delete_resource());
    }
};

// Custom test memory resource for tracking allocations
class TestMemoryResource : public fcm::memory_resource {
public:
    struct AllocationInfo {
        void* ptr;
        std::size_t bytes;
        std::size_t alignment;
    };

    TestMemoryResource() = default;

    std::size_t allocation_count() const { return allocations_.size(); }
    std::size_t total_allocated_bytes() const { return total_bytes_; }
    bool has_active_allocations() const { return !allocations_.empty(); }

    const std::vector<AllocationInfo>& allocations() const { return allocations_; }

    void reset_stats() {
        allocations_.clear();
        deallocations_.clear();
        total_bytes_ = 0;
    }

    // Track if specific pointer was deallocated
    bool was_deallocated(void* p) const {
        return deallocations_.find(p) != deallocations_.end();
    }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        void* p = ::operator new(bytes, std::align_val_t(alignment));
        allocations_.push_back({p, bytes, alignment});
        total_bytes_ += bytes;
        return p;
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        deallocations_.insert(p);

        // Remove from allocations list
        auto it = std::find_if(allocations_.begin(), allocations_.end(),
            [p](const AllocationInfo& info) { return info.ptr == p; });

        if (it != allocations_.end()) {
            allocations_.erase(it);
        }

        ::operator delete(p, std::align_val_t(alignment));
    }

    bool do_is_equal(const fcm::memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    std::vector<AllocationInfo> allocations_;
    std::set<void*> deallocations_;
    std::size_t total_bytes_ = 0;
};

// Test new_delete_resource
TEST_F(MemoryResourceTest, NewDeleteResource_BasicAllocation) {
    auto* resource = fcm::new_delete_resource();
    ASSERT_NE(resource, nullptr);

    // Basic allocation
    void* p = resource->allocate(100);
    ASSERT_NE(p, nullptr);

    // Write to allocated memory to verify it's valid
    std::memset(p, 42, 100);

    resource->deallocate(p, 100);
}

TEST_F(MemoryResourceTest, NewDeleteResource_AlignedAllocation) {
    auto* resource = fcm::new_delete_resource();

    // Test various alignments
    std::vector<std::size_t> alignments = {8, 16, 32, 64, 128};

    for (auto alignment : alignments) {
        void* p = resource->allocate(100, alignment);
        ASSERT_NE(p, nullptr);

        // Check alignment
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignment, 0)
            << "Pointer not aligned to " << alignment << " bytes";

        resource->deallocate(p, 100, alignment);
    }
}

TEST_F(MemoryResourceTest, NewDeleteResource_MultipleAllocations) {
    auto* resource = fcm::new_delete_resource();

    std::vector<void*> pointers;
    std::vector<std::size_t> sizes = {10, 100, 1000, 10000};

    // Allocate multiple blocks
    for (auto size : sizes) {
        void* p = resource->allocate(size);
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);
    }

    // Deallocate in reverse order
    for (int i = static_cast<int>(pointers.size()) - 1; i >= 0; --i) {
        resource->deallocate(pointers[i], sizes[i]);
    }
}

TEST_F(MemoryResourceTest, NewDeleteResource_IsEqual) {
    auto* resource1 = fcm::new_delete_resource();
    auto* resource2 = fcm::new_delete_resource();

    // Should be the same instance (singleton)
    EXPECT_TRUE(resource1->is_equal(*resource2));
    EXPECT_EQ(resource1, resource2);
}

// Test null_memory_resource
TEST_F(MemoryResourceTest, NullMemoryResource_ThrowsOnAllocate) {
    auto* resource = fcm::null_memory_resource();
    ASSERT_NE(resource, nullptr);

    // Should throw bad_alloc
    EXPECT_THROW((void)resource->allocate(100), std::bad_alloc);
    EXPECT_THROW((void)resource->allocate(1), std::bad_alloc);
    EXPECT_THROW((void)resource->allocate(0), std::bad_alloc);
}

TEST_F(MemoryResourceTest, NullMemoryResource_DeallocateNoOp) {
    auto* resource = fcm::null_memory_resource();

    // Deallocate should do nothing (not crash)
    // Note: nullptr test would violate non-null requirement in std::pmr
    // resource->deallocate(nullptr, 100);
    // Just test with a fake pointer (implementation does nothing anyway)
    resource->deallocate(reinterpret_cast<void*>(0x1234), 100);
}

TEST_F(MemoryResourceTest, NullMemoryResource_IsEqual) {
    auto* resource1 = fcm::null_memory_resource();
    auto* resource2 = fcm::null_memory_resource();

    // Should be the same instance (singleton)
    EXPECT_TRUE(resource1->is_equal(*resource2));
    EXPECT_EQ(resource1, resource2);
}

// Test default_resource
TEST_F(MemoryResourceTest, DefaultResource_InitialValue) {
    auto* resource = fcm::default_resource();
    ASSERT_NE(resource, nullptr);

    // Default should be new_delete_resource
    EXPECT_TRUE(resource->is_equal(*fcm::new_delete_resource()));
}

TEST_F(MemoryResourceTest, DefaultResource_SetAndGet) {
    TestMemoryResource test_resource;

    auto* original = fcm::set_default_resource(&test_resource);
    EXPECT_TRUE(original->is_equal(*fcm::new_delete_resource()));

    auto* current = fcm::default_resource();
    EXPECT_EQ(current, &test_resource);

    // Restore original
    fcm::set_default_resource(original);
}

TEST_F(MemoryResourceTest, DefaultResource_NullHandling) {
    auto* original = fcm::default_resource();

    // Setting nullptr should be ignored
    auto* prev = fcm::set_default_resource(nullptr);
    EXPECT_EQ(prev, original);
    EXPECT_EQ(fcm::default_resource(), original);
}

// Test polymorphic_allocator
TEST_F(MemoryResourceTest, PolymorphicAllocator_DefaultResource) {
    fcm::polymorphic_allocator<int> allocator;
    EXPECT_EQ(allocator.resource(), fcm::default_resource());
}

TEST_F(MemoryResourceTest, PolymorphicAllocator_ExplicitResource) {
    TestMemoryResource test_resource;
    fcm::polymorphic_allocator<int> allocator(&test_resource);

    EXPECT_EQ(allocator.resource(), &test_resource);
}

TEST_F(MemoryResourceTest, PolymorphicAllocator_AllocateAndDeallocate) {
    TestMemoryResource test_resource;
    fcm::polymorphic_allocator<int> allocator(&test_resource);

    // Allocate array of 10 ints
    int* p = allocator.allocate(10);
    ASSERT_NE(p, nullptr);

    // Check tracking
    EXPECT_EQ(test_resource.allocation_count(), 1);
    EXPECT_EQ(test_resource.total_allocated_bytes(), 10 * sizeof(int));

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        p[i] = i * 10;
    }

    allocator.deallocate(p, 10);

    // Check deallocation
    EXPECT_FALSE(test_resource.has_active_allocations());
    EXPECT_TRUE(test_resource.was_deallocated(p));
}

TEST_F(MemoryResourceTest, PolymorphicAllocator_CopyConstruction) {
    TestMemoryResource test_resource;
    fcm::polymorphic_allocator<int> allocator1(&test_resource);
    fcm::polymorphic_allocator<double> allocator2(allocator1);

    EXPECT_EQ(allocator2.resource(), &test_resource);
}

TEST_F(MemoryResourceTest, PolymorphicAllocator_Comparison) {
    TestMemoryResource resource1;
    TestMemoryResource resource2;

    fcm::polymorphic_allocator<int> alloc1(&resource1);
    fcm::polymorphic_allocator<int> alloc2(&resource1);
    fcm::polymorphic_allocator<int> alloc3(&resource2);

    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 == alloc3);
    EXPECT_FALSE(alloc1 != alloc2);
    EXPECT_TRUE(alloc1 != alloc3);
}

TEST_F(MemoryResourceTest, PolymorphicAllocator_WithSTLContainer) {
    TestMemoryResource test_resource;

    // Use with std::vector
    std::vector<int, fcm::polymorphic_allocator<int>> vec(&test_resource);

    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    EXPECT_GT(test_resource.allocation_count(), 0);
    EXPECT_GT(test_resource.total_allocated_bytes(), 0);

    vec.clear();
    vec.shrink_to_fit();
}

// Test monotonic_buffer_resource (minimal fallback version)
TEST_F(MemoryResourceTest, MonotonicBufferResource_BasicUsage) {
    TestMemoryResource upstream;
    fcm::monotonic_buffer_resource buffer(&upstream);

    void* p = buffer.allocate(100);
    ASSERT_NE(p, nullptr);

    // In fallback mode, it just delegates to upstream
    EXPECT_EQ(upstream.allocation_count(), 1);

    buffer.deallocate(p, 100);

    // Release is a no-op in fallback
    buffer.release();
}

TEST_F(MemoryResourceTest, MonotonicBufferResource_WithInitialSize) {
    TestMemoryResource upstream;
    fcm::monotonic_buffer_resource buffer(1024, &upstream);

    void* p = buffer.allocate(100);
    ASSERT_NE(p, nullptr);

    buffer.deallocate(p, 100);
}

// Custom memory resource implementation test
class CountingMemoryResource : public fcm::memory_resource {
public:
    CountingMemoryResource() = default;

    std::size_t allocate_calls() const { return allocate_count_; }
    std::size_t deallocate_calls() const { return deallocate_count_; }

    void reset_counts() {
        allocate_count_ = 0;
        deallocate_count_ = 0;
    }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocate_count_;
        return ::operator new(bytes, std::align_val_t(alignment));
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        ++deallocate_count_;
        ::operator delete(p, std::align_val_t(alignment));
    }

    bool do_is_equal(const fcm::memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    mutable std::atomic<std::size_t> allocate_count_{0};
    mutable std::atomic<std::size_t> deallocate_count_{0};
};

TEST_F(MemoryResourceTest, CustomMemoryResource_CallCounting) {
    CountingMemoryResource resource;

    void* p1 = resource.allocate(100);
    void* p2 = resource.allocate(200);

    EXPECT_EQ(resource.allocate_calls(), 2);
    EXPECT_EQ(resource.deallocate_calls(), 0);

    resource.deallocate(p1, 100);
    EXPECT_EQ(resource.deallocate_calls(), 1);

    resource.deallocate(p2, 200);
    EXPECT_EQ(resource.deallocate_calls(), 2);
}

// Thread safety test for default resource
TEST_F(MemoryResourceTest, DefaultResource_ThreadSafety) {
    std::atomic<int> errors{0};
    const int num_threads = 4;
    const int iterations = 1000;

    auto worker = [&errors, iterations]() {
        for (int i = 0; i < iterations; ++i) {
            auto* resource = fcm::default_resource();
            void* p = resource->allocate(100 + i % 100);

            if (!p) {
                errors.fetch_add(1);
                continue;
            }

            // Use the memory
            std::memset(p, i % 256, 100 + i % 100);

            resource->deallocate(p, 100 + i % 100);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(errors.load(), 0);
}

// Edge cases
TEST_F(MemoryResourceTest, EdgeCases_ZeroSizeAllocation) {
    auto* resource = fcm::new_delete_resource();

    // Zero size allocation should still return valid pointer
    void* p = resource->allocate(0);
    EXPECT_NE(p, nullptr);

    resource->deallocate(p, 0);
}

TEST_F(MemoryResourceTest, EdgeCases_LargeAlignment) {
    auto* resource = fcm::new_delete_resource();

    // Test large alignment (page size)
    std::size_t page_size = 4096;
    void* p = resource->allocate(100, page_size);
    ASSERT_NE(p, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % page_size, 0);

    resource->deallocate(p, 100, page_size);
}

TEST_F(MemoryResourceTest, EdgeCases_VeryLargeAllocation) {
    auto* resource = fcm::new_delete_resource();

    // Allocate 1MB
    std::size_t large_size = 1024 * 1024;
    void* p = resource->allocate(large_size);
    ASSERT_NE(p, nullptr);

    // Use some of the memory
    std::memset(p, 0, large_size);

    resource->deallocate(p, large_size);
}

// Integration test with multiple resources
TEST_F(MemoryResourceTest, Integration_MultipleResourcesAndAllocators) {
    TestMemoryResource resource1;
    TestMemoryResource resource2;

    fcm::polymorphic_allocator<int> alloc1(&resource1);
    fcm::polymorphic_allocator<double> alloc2(&resource2);

    // Allocate from different resources
    int* ints = alloc1.allocate(10);
    double* doubles = alloc2.allocate(20);

    EXPECT_EQ(resource1.allocation_count(), 1);
    EXPECT_EQ(resource2.allocation_count(), 1);

    // Use the allocations
    for (int i = 0; i < 10; ++i) {
        ints[i] = i;
    }
    for (int i = 0; i < 20; ++i) {
        doubles[i] = i * 1.5;
    }

    alloc1.deallocate(ints, 10);
    alloc2.deallocate(doubles, 20);

    EXPECT_FALSE(resource1.has_active_allocations());
    EXPECT_FALSE(resource2.has_active_allocations());
}
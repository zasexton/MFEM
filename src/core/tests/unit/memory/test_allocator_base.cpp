#include <gtest/gtest.h>
#include <core/memory/allocator_base.h>
#include <vector>
#include <list>
#include <memory>
#include <type_traits>
#include <cstring>

namespace fcm = fem::core::memory;

// Statistics structure shared between TestAllocator instances
struct TestAllocatorStats {
    std::size_t allocation_count = 0;
    std::size_t deallocation_count = 0;
    std::size_t total_allocated = 0;
    std::size_t total_deallocated = 0;
};

class AllocatorBaseTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test allocator derived from AllocatorBase using CRTP
template<class T>
class TestAllocator : public fcm::AllocatorBase<TestAllocator<T>, T> {
public:
    using base_type = fcm::AllocatorBase<TestAllocator<T>, T>;
    using value_type = typename base_type::value_type;
    using pointer = typename base_type::pointer;
    using size_type = typename base_type::size_type;

    // Proper rebind template that produces TestAllocator<U> not AllocatorBase<TestAllocator, U>
    template<class U>
    struct rebind {
        using other = TestAllocator<U>;
    };

    TestAllocator() : stats_(std::make_shared<TestAllocatorStats>()) {}

    // Copy constructor - share statistics
    TestAllocator(const TestAllocator& other) noexcept : stats_(other.stats_) {}

    // Rebind constructor - share statistics
    template<class U>
    TestAllocator(const TestAllocator<U>& other) noexcept : stats_(other.stats_) {}

    pointer do_allocate(size_type n) {
        ++stats_->allocation_count;
        stats_->total_allocated += n * sizeof(T);
        return static_cast<pointer>(::operator new(n * sizeof(T)));
    }

    void do_deallocate(pointer p, size_type n) {
        ++stats_->deallocation_count;
        stats_->total_deallocated += n * sizeof(T);
        ::operator delete(p);
    }

    // Test-specific tracking
    std::size_t allocation_count() const { return stats_->allocation_count; }
    std::size_t deallocation_count() const { return stats_->deallocation_count; }
    std::size_t total_allocated() const { return stats_->total_allocated; }
    std::size_t total_deallocated() const { return stats_->total_deallocated; }

    void reset_stats() {
        stats_->allocation_count = 0;
        stats_->deallocation_count = 0;
        stats_->total_allocated = 0;
        stats_->total_deallocated = 0;
    }

    // Comparison operators required for STL containers
    template<class U>
    bool operator==(const TestAllocator<U>& other) const noexcept {
        // Compare based on shared stats pointer
        return stats_ == other.stats_;
    }

    template<class U>
    bool operator!=(const TestAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

    // Friend declaration to allow access to stats_ in rebind constructor
    template<class> friend class TestAllocator;

private:
    std::shared_ptr<TestAllocatorStats> stats_;
};

// Test AllocatorBase CRTP functionality
TEST_F(AllocatorBaseTest, AllocatorBase_BasicTypes) {
    TestAllocator<int> alloc;

    // Check type aliases
    static_assert(std::is_same_v<TestAllocator<int>::value_type, int>);
    static_assert(std::is_same_v<TestAllocator<int>::pointer, int*>);
    static_assert(std::is_same_v<TestAllocator<int>::const_pointer, const int*>);
    static_assert(std::is_same_v<TestAllocator<int>::size_type, std::size_t>);
    static_assert(std::is_same_v<TestAllocator<int>::difference_type, std::ptrdiff_t>);
}

TEST_F(AllocatorBaseTest, AllocatorBase_Rebind) {
    using IntAlloc = TestAllocator<int>;
    // DoubleAlloc typedef is intentionally used in static_assert below
    using DoubleAlloc = TestAllocator<double>;

    // Test rebind
    using ReboundAlloc = IntAlloc::rebind<double>::other;
    static_assert(std::is_same_v<ReboundAlloc::value_type, double>);
    // Verify DoubleAlloc and ReboundAlloc are equivalent
    static_assert(std::is_same_v<ReboundAlloc, DoubleAlloc>);
}

TEST_F(AllocatorBaseTest, AllocatorBase_PropagationTraits) {
    using Alloc = TestAllocator<int>;

    // Check propagation traits
    static_assert(!Alloc::propagate_on_container_copy_assignment::value);
    static_assert(Alloc::propagate_on_container_move_assignment::value);
    static_assert(Alloc::propagate_on_container_swap::value);
    static_assert(!Alloc::is_always_equal::value);
}

TEST_F(AllocatorBaseTest, AllocatorBase_AllocateAndDeallocate) {
    TestAllocator<int> alloc;

    // Allocate array of 10 ints
    int* p = alloc.allocate(10);
    ASSERT_NE(p, nullptr);

    // Check tracking
    EXPECT_EQ(alloc.allocation_count(), 1);
    EXPECT_EQ(alloc.total_allocated(), 10 * sizeof(int));

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        p[i] = i * 10;
    }

    // Verify values
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i], i * 10);
    }

    alloc.deallocate(p, 10);

    // Check deallocation tracking
    EXPECT_EQ(alloc.deallocation_count(), 1);
    EXPECT_EQ(alloc.total_deallocated(), 10 * sizeof(int));
}

TEST_F(AllocatorBaseTest, AllocatorBase_MultipleAllocations) {
    TestAllocator<double> alloc;

    std::vector<double*> pointers;
    std::vector<std::size_t> sizes = {1, 5, 10, 20, 100};

    // Multiple allocations
    for (auto size : sizes) {
        double* p = alloc.allocate(size);
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);

        // Initialize
        for (std::size_t i = 0; i < size; ++i) {
            p[i] = static_cast<double>(i) * 1.5;
        }
    }

    EXPECT_EQ(alloc.allocation_count(), sizes.size());

    // Deallocate all
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        alloc.deallocate(pointers[i], sizes[i]);
    }

    EXPECT_EQ(alloc.deallocation_count(), sizes.size());
}

// Test that AllocatorBase is tagged correctly
TEST_F(AllocatorBaseTest, AllocatorBase_TagInheritance) {
    TestAllocator<int> alloc;

    // Should inherit from allocator_base_tag
    static_assert(std::is_base_of_v<fcm::allocator_base_tag, TestAllocator<int>>);
}

// Simple allocator not derived from AllocatorBase
template<class T>
class SimpleAllocator {
public:
    using value_type = T;
    using is_always_equal = std::false_type;  // Explicitly set to false

    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) {
        ::operator delete(p);
    }

    // Add comparison operators for completeness
    template<class U>
    bool operator==(const SimpleAllocator<U>&) const noexcept { return true; }

    template<class U>
    bool operator!=(const SimpleAllocator<U>&) const noexcept { return false; }
};

// Test allocator concepts
TEST_F(AllocatorBaseTest, Concepts_StdAllocatorLike) {
    // Standard allocator should satisfy StdAllocatorLike
    static_assert(fcm::StdAllocatorLike<std::allocator<int>>);

    // Our test allocator should also satisfy it
    static_assert(fcm::StdAllocatorLike<TestAllocator<int>>);
}

TEST_F(AllocatorBaseTest, Concepts_SimpleAllocatorLike) {
    // Simple allocator should satisfy SimpleAllocatorLike
    static_assert(fcm::SimpleAllocatorLike<SimpleAllocator<int>>);

    // Test allocator should also satisfy it
    static_assert(fcm::SimpleAllocatorLike<TestAllocator<int>>);
}

// Test allocator_properties
TEST_F(AllocatorBaseTest, AllocatorProperties_StdAllocator) {
    using Props = fcm::allocator_properties<std::allocator<int>>;

    EXPECT_TRUE(Props::is_std_allocator);
    EXPECT_TRUE(Props::is_simple_allocator);
    EXPECT_TRUE(Props::is_always_equal);
}

TEST_F(AllocatorBaseTest, AllocatorProperties_TestAllocator) {
    using Props = fcm::allocator_properties<TestAllocator<int>>;

    EXPECT_TRUE(Props::is_std_allocator);
    EXPECT_TRUE(Props::is_simple_allocator);
    EXPECT_FALSE(Props::is_always_equal);
}

TEST_F(AllocatorBaseTest, AllocatorProperties_SimpleAllocator) {
    using Props = fcm::allocator_properties<SimpleAllocator<int>>;

    EXPECT_TRUE(Props::is_std_allocator);
    EXPECT_TRUE(Props::is_simple_allocator);
    EXPECT_FALSE(Props::is_always_equal);
}

// Test StdAllocatorAdapter
class RawAllocator {
public:
    using value_type = void;

    void* allocate(std::size_t n) {
        ++alloc_count_;
        return ::operator new(n);
    }

    void deallocate(void* p, std::size_t) {
        ++dealloc_count_;
        ::operator delete(p);
    }

    std::size_t alloc_count() const { return alloc_count_; }
    std::size_t dealloc_count() const { return dealloc_count_; }

private:
    mutable std::size_t alloc_count_ = 0;
    mutable std::size_t dealloc_count_ = 0;
};

TEST_F(AllocatorBaseTest, StdAllocatorAdapter_BasicUsage) {
    RawAllocator raw;
    fcm::StdAllocatorAdapter<RawAllocator, int> adapter(&raw);

    // Check type aliases
    static_assert(std::is_same_v<decltype(adapter)::value_type, int>);
    static_assert(std::is_same_v<decltype(adapter)::pointer, int*>);

    // Allocate and deallocate
    int* p = adapter.allocate(10);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(raw.alloc_count(), 1);

    adapter.deallocate(p, 10);
    EXPECT_EQ(raw.dealloc_count(), 1);
}

TEST_F(AllocatorBaseTest, StdAllocatorAdapter_Rebind) {
    RawAllocator raw;
    fcm::StdAllocatorAdapter<RawAllocator, int> int_adapter(&raw);

    // Rebind to double
    using DoubleAdapter = fcm::StdAllocatorAdapter<RawAllocator, int>::rebind<double>::other;
    DoubleAdapter double_adapter(int_adapter);

    EXPECT_EQ(double_adapter.raw(), &raw);

    double* p = double_adapter.allocate(5);
    ASSERT_NE(p, nullptr);

    double_adapter.deallocate(p, 5);
}

TEST_F(AllocatorBaseTest, StdAllocatorAdapter_Comparison) {
    RawAllocator raw1, raw2;
    fcm::StdAllocatorAdapter<RawAllocator, int> adapter1(&raw1);
    fcm::StdAllocatorAdapter<RawAllocator, int> adapter2(&raw1);
    fcm::StdAllocatorAdapter<RawAllocator, int> adapter3(&raw2);

    EXPECT_TRUE(adapter1 == adapter2);
    EXPECT_FALSE(adapter1 == adapter3);
    EXPECT_FALSE(adapter1 != adapter2);
    EXPECT_TRUE(adapter1 != adapter3);
}

TEST_F(AllocatorBaseTest, StdAllocatorAdapter_WithSTLContainer) {
    RawAllocator raw;
    fcm::StdAllocatorAdapter<RawAllocator, int> adapter(&raw);

    // Use with std::vector
    std::vector<int, fcm::StdAllocatorAdapter<RawAllocator, int>> vec(adapter);

    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    EXPECT_GT(raw.alloc_count(), 0);

    vec.clear();
}

// Test with STL containers
TEST_F(AllocatorBaseTest, Integration_WithVector) {
    TestAllocator<int> alloc;

    std::vector<int, TestAllocator<int>> vec(alloc);

    // Add elements
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_GT(alloc.allocation_count(), 0);
    EXPECT_EQ(vec.size(), 100);

    // Verify values
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    vec.clear();
    vec.shrink_to_fit();

    // After shrink_to_fit, should have deallocations
    EXPECT_GT(alloc.deallocation_count(), 0);
}

TEST_F(AllocatorBaseTest, Integration_WithList) {
    TestAllocator<std::string> alloc;

    std::list<std::string, TestAllocator<std::string>> list(alloc);

    // Add elements
    list.push_back("first");
    list.push_back("second");
    list.push_back("third");

    EXPECT_GT(alloc.allocation_count(), 0);
    EXPECT_EQ(list.size(), 3);

    // Verify elements
    auto it = list.begin();
    EXPECT_EQ(*it++, "first");
    EXPECT_EQ(*it++, "second");
    EXPECT_EQ(*it++, "third");

    list.clear();
    EXPECT_GT(alloc.deallocation_count(), 0);
}

// Test derived allocator with alignment support
template<class T>
class AlignedTestAllocator : public fcm::AllocatorBase<AlignedTestAllocator<T>, T> {
public:
    using base_type = fcm::AllocatorBase<AlignedTestAllocator<T>, T>;
    using pointer = typename base_type::pointer;
    using size_type = typename base_type::size_type;

    // Proper rebind template
    template<class U>
    struct rebind {
        using other = AlignedTestAllocator<U>;
    };

    explicit AlignedTestAllocator(std::size_t alignment = alignof(T))
        : alignment_(alignment) {}

    template<class U>
    AlignedTestAllocator(const AlignedTestAllocator<U>& other) noexcept
        : alignment_(other.alignment()) {}

    pointer do_allocate(size_type n) {
        return static_cast<pointer>(::operator new(n * sizeof(T), std::align_val_t(alignment_)));
    }

    void do_deallocate(pointer p, size_type n) {
        ::operator delete(p, n * sizeof(T), std::align_val_t(alignment_));
    }

    std::size_t alignment() const { return alignment_; }

    // Comparison operators required for STL containers
    template<class U>
    bool operator==(const AlignedTestAllocator<U>& other) const noexcept {
        return alignment_ == other.alignment();
    }

    template<class U>
    bool operator!=(const AlignedTestAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    std::size_t alignment_;
};

TEST_F(AllocatorBaseTest, AlignedAllocator_Alignment) {
    const std::size_t alignment = 64;
    AlignedTestAllocator<int> alloc(alignment);

    int* p = alloc.allocate(10);
    ASSERT_NE(p, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignment, 0);

    alloc.deallocate(p, 10);
}

// Edge cases
TEST_F(AllocatorBaseTest, EdgeCases_ZeroAllocation) {
    TestAllocator<int> alloc;

    // Zero allocation should still work
    int* p = alloc.allocate(0);
    EXPECT_NE(p, nullptr);

    alloc.deallocate(p, 0);
}

TEST_F(AllocatorBaseTest, EdgeCases_LargeAllocation) {
    TestAllocator<char> alloc;

    // Large allocation (1MB)
    std::size_t size = 1024 * 1024;
    char* p = alloc.allocate(size);
    ASSERT_NE(p, nullptr);

    // Use the memory
    std::memset(p, 0, size);

    alloc.deallocate(p, size);

    EXPECT_EQ(alloc.total_allocated(), size);
    EXPECT_EQ(alloc.total_deallocated(), size);
}

// Test allocator with different types
TEST_F(AllocatorBaseTest, DifferentTypes) {
    // Test with various types
    {
        TestAllocator<char> alloc;
        char* p = alloc.allocate(100);
        ASSERT_NE(p, nullptr);
        alloc.deallocate(p, 100);
    }

    {
        TestAllocator<double> alloc;
        double* p = alloc.allocate(50);
        ASSERT_NE(p, nullptr);
        alloc.deallocate(p, 50);
    }

    {
        struct LargeStruct {
            char data[1024];
        };
        TestAllocator<LargeStruct> alloc;
        LargeStruct* p = alloc.allocate(10);
        ASSERT_NE(p, nullptr);
        alloc.deallocate(p, 10);
    }
}
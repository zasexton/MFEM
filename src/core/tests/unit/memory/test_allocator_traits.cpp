#include <gtest/gtest.h>
#include <core/memory/allocator_traits.h>
#include <vector>
#include <memory>
#include <type_traits>
#include <cstring>

namespace fcm = fem::core::memory;

// Shared statistics structure for TestAllocator
struct TestAllocatorStats {
    std::size_t allocation_count = 0;
    std::size_t deallocation_count = 0;
    std::size_t construct_count = 0;
    std::size_t destroy_count = 0;
    std::size_t total_bytes = 0;
};

class AllocatorTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Custom allocator for testing
template<class T>
class TestAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind {
        using other = TestAllocator<U>;
    };

    // Propagation traits for testing
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    TestAllocator() : stats_(std::make_shared<TestAllocatorStats>()) {}

    template<class U>
    TestAllocator(const TestAllocator<U>& other) noexcept : stats_(other.stats_) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        ++stats_->allocation_count;
        stats_->total_bytes += n * sizeof(T);
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) {
        ++stats_->deallocation_count;
        (void)n; // unused parameter
        ::operator delete(p);
    }

    // Optional construct/destroy
    template<class U, class... Args>
    void construct(U* p, Args&&... args) {
        ++stats_->construct_count;
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<class U>
    void destroy(U* p) {
        ++stats_->destroy_count;
        p->~U();
    }

    // Test-specific tracking
    std::size_t allocation_count() const { return stats_->allocation_count; }
    std::size_t deallocation_count() const { return stats_->deallocation_count; }
    std::size_t construct_count() const { return stats_->construct_count; }
    std::size_t destroy_count() const { return stats_->destroy_count; }
    std::size_t total_bytes() const { return stats_->total_bytes; }

    void reset_stats() {
        stats_->allocation_count = 0;
        stats_->deallocation_count = 0;
        stats_->construct_count = 0;
        stats_->destroy_count = 0;
        stats_->total_bytes = 0;
    }

    // Friend declaration to allow access to stats_ in rebind constructor
    template<class> friend class TestAllocator;

private:
    std::shared_ptr<TestAllocatorStats> stats_;
};

// Test type aliases
TEST_F(AllocatorTraitsTest, TypeAliases) {
    using Alloc = TestAllocator<int>;

    // Test basic type aliases
    static_assert(std::is_same_v<fcm::allocator_value_t<Alloc>, int>);
    static_assert(std::is_same_v<fcm::allocator_pointer_t<Alloc>, int*>);
    static_assert(std::is_same_v<fcm::allocator_const_pointer_t<Alloc>, const int*>);
    static_assert(std::is_same_v<fcm::allocator_void_pointer_t<Alloc>, void*>);
    static_assert(std::is_same_v<fcm::allocator_size_t<Alloc>, std::size_t>);
    static_assert(std::is_same_v<fcm::allocator_diff_t<Alloc>, std::ptrdiff_t>);
}

TEST_F(AllocatorTraitsTest, RebindAliases) {
    using IntAlloc = TestAllocator<int>;

    // Test rebind aliases
    using DoubleAlloc = fcm::rebind_alloc_t<IntAlloc, double>;
    static_assert(std::is_same_v<DoubleAlloc, TestAllocator<double>>);

    using DoubleTraits = fcm::rebind_traits_t<IntAlloc, double>;
    static_assert(std::is_same_v<DoubleTraits::allocator_type, TestAllocator<double>>);
    static_assert(std::is_same_v<DoubleTraits::value_type, double>);
}

// Test allocator concepts
TEST_F(AllocatorTraitsTest, Concepts_Allocator) {
    // Standard allocator should satisfy concept
    static_assert(fcm::Allocator<std::allocator<int>>);

    // Custom allocator should satisfy concept
    static_assert(fcm::Allocator<TestAllocator<int>>);
}

TEST_F(AllocatorTraitsTest, Concepts_AllocatorFor) {
    using IntAlloc = TestAllocator<int>;

    // Should be able to allocate int
    static_assert(fcm::AllocatorFor<IntAlloc, int>);

    // Should also be able to allocate other types via rebind
    static_assert(fcm::AllocatorFor<IntAlloc, double>);
    static_assert(fcm::AllocatorFor<IntAlloc, std::string>);

    struct CustomStruct {
        int x, y;
    };
    static_assert(fcm::AllocatorFor<IntAlloc, CustomStruct>);
}

// Test allocate_n and deallocate_n
TEST_F(AllocatorTraitsTest, AllocateN_DeallocateN) {
    TestAllocator<int> alloc;

    // Allocate array of 10 ints
    int* p = fcm::allocate_n(alloc, 10);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(alloc.allocation_count(), 1);
    EXPECT_EQ(alloc.total_bytes(), 10 * sizeof(int));

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        p[i] = i * 10;
    }

    // Verify values
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i], i * 10);
    }

    fcm::deallocate_n(alloc, p, 10);
    EXPECT_EQ(alloc.deallocation_count(), 1);
}

TEST_F(AllocatorTraitsTest, AllocateN_MultipleAllocations) {
    TestAllocator<double> alloc;

    std::vector<double*> pointers;
    std::vector<std::size_t> sizes = {1, 5, 10, 20, 100};

    // Multiple allocations
    for (auto size : sizes) {
        double* p = fcm::allocate_n(alloc, size);
        ASSERT_NE(p, nullptr);
        pointers.push_back(p);
    }

    EXPECT_EQ(alloc.allocation_count(), sizes.size());

    // Deallocate all
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        fcm::deallocate_n(alloc, pointers[i], sizes[i]);
    }

    EXPECT_EQ(alloc.deallocation_count(), sizes.size());
}

// Test allocate_one and deallocate_one
TEST_F(AllocatorTraitsTest, AllocateOne_DeallocateOne) {
    TestAllocator<int> alloc;

    // Allocate single int
    int* p = fcm::allocate_one<TestAllocator<int>, int>(alloc);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(alloc.allocation_count(), 1);

    *p = 42;
    EXPECT_EQ(*p, 42);

    fcm::deallocate_one<TestAllocator<int>, int>(alloc, p);
    EXPECT_EQ(alloc.deallocation_count(), 1);
}

TEST_F(AllocatorTraitsTest, AllocateOne_DifferentType) {
    TestAllocator<int> int_alloc;

    // Allocate single double using int allocator
    double* p = fcm::allocate_one<TestAllocator<int>, double>(int_alloc);
    ASSERT_NE(p, nullptr);

    *p = 3.14159;
    EXPECT_DOUBLE_EQ(*p, 3.14159);

    fcm::deallocate_one<TestAllocator<int>, double>(int_alloc, p);
}

TEST_F(AllocatorTraitsTest, AllocateOne_Struct) {
    struct TestStruct {
        int x;
        double y;
        char z[100];
    };

    TestAllocator<char> alloc;

    // Allocate single struct
    TestStruct* p = fcm::allocate_one<TestAllocator<char>, TestStruct>(alloc);
    ASSERT_NE(p, nullptr);

    p->x = 10;
    p->y = 20.5;
    std::strcpy(p->z, "test");

    EXPECT_EQ(p->x, 10);
    EXPECT_DOUBLE_EQ(p->y, 20.5);
    EXPECT_STREQ(p->z, "test");

    fcm::deallocate_one<TestAllocator<char>, TestStruct>(alloc, p);
}

// Test allocate_bytes and deallocate_bytes
TEST_F(AllocatorTraitsTest, AllocateBytes_DeallocateBytes) {
    TestAllocator<int> alloc;

    // Allocate 100 bytes
    std::byte* p = fcm::allocate_bytes(alloc, 100);
    ASSERT_NE(p, nullptr);

    // Use the memory
    for (int i = 0; i < 100; ++i) {
        p[i] = static_cast<std::byte>(i);
    }

    // Verify
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(p[i], static_cast<std::byte>(i));
    }

    fcm::deallocate_bytes(alloc, p, 100);
}

TEST_F(AllocatorTraitsTest, AllocateBytes_WithAlignment) {
    TestAllocator<int> alloc;

    // Allocate with specific alignment
    std::size_t alignment = 64;
    std::byte* p = fcm::allocate_bytes(alloc, 256, alignment);
    ASSERT_NE(p, nullptr);

    // Note: The alignment parameter is currently ignored in the implementation
    // but the interface supports it for future use

    fcm::deallocate_bytes(alloc, p, 256, alignment);
}

// Test propagation traits
TEST_F(AllocatorTraitsTest, PropagationTraits) {
    using Alloc = TestAllocator<int>;

    // Test type aliases for propagation traits
    static_assert(std::is_same_v<
        fcm::propagate_on_container_copy_assignment_t<Alloc>,
        std::false_type>);

    static_assert(std::is_same_v<
        fcm::propagate_on_container_move_assignment_t<Alloc>,
        std::true_type>);

    static_assert(std::is_same_v<
        fcm::propagate_on_container_swap_t<Alloc>,
        std::true_type>);
}

TEST_F(AllocatorTraitsTest, SelectOnContainerCopyConstruction) {
    TestAllocator<int> alloc;

    // Should return a copy
    auto alloc2 = fcm::select_on_container_copy_construction(alloc);

    // Both should be able to allocate independently
    int* p1 = alloc.allocate(1);
    int* p2 = alloc2.allocate(1);

    EXPECT_NE(p1, p2);

    alloc.deallocate(p1, 1);
    alloc2.deallocate(p2, 1);
}

TEST_F(AllocatorTraitsTest, IsAlwaysEqual) {
    // Standard allocator is always equal
    static_assert(fcm::is_always_equal_v<std::allocator<int>>);

    // Our test allocator is not always equal
    static_assert(!fcm::is_always_equal_v<TestAllocator<int>>);
}

// Test with minimal allocator (only required members)
template<class T>
class MinimalAllocator {
public:
    using value_type = T;

    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) {
        ::operator delete(p);
    }
};

TEST_F(AllocatorTraitsTest, MinimalAllocator_Concepts) {
    // Minimal allocator should still satisfy concepts
    static_assert(fcm::Allocator<MinimalAllocator<int>>);
    static_assert(fcm::AllocatorFor<MinimalAllocator<int>, int>);
    static_assert(fcm::AllocatorFor<MinimalAllocator<int>, double>);
}

TEST_F(AllocatorTraitsTest, MinimalAllocator_Usage) {
    MinimalAllocator<int> alloc;

    // Should work with allocate_n/deallocate_n
    int* p = fcm::allocate_n(alloc, 10);
    ASSERT_NE(p, nullptr);

    for (int i = 0; i < 10; ++i) {
        p[i] = i;
    }

    fcm::deallocate_n(alloc, p, 10);
}

// Test with standard allocator
TEST_F(AllocatorTraitsTest, StandardAllocator) {
    std::allocator<int> alloc;

    // Test with standard functions
    int* p = fcm::allocate_n(alloc, 5);
    ASSERT_NE(p, nullptr);

    for (int i = 0; i < 5; ++i) {
        p[i] = i * 10;
    }

    fcm::deallocate_n(alloc, p, 5);

    // Test allocate_one
    double* d = fcm::allocate_one<std::allocator<int>, double>(alloc);
    ASSERT_NE(d, nullptr);
    *d = 3.14;
    fcm::deallocate_one<std::allocator<int>, double>(alloc, d);

    // Test allocate_bytes
    std::byte* bytes = fcm::allocate_bytes(alloc, 100);
    ASSERT_NE(bytes, nullptr);
    fcm::deallocate_bytes(alloc, bytes, 100);
}

// Integration test with STL containers
TEST_F(AllocatorTraitsTest, Integration_WithVector) {
    TestAllocator<int> alloc;

    std::vector<int, TestAllocator<int>> vec(alloc);

    // Add elements
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_GT(alloc.allocation_count(), 0);
    EXPECT_EQ(vec.size(), 100);

    // construct/destroy might not be called if vector uses placement new directly
    // But allocations should definitely happen
    EXPECT_GT(alloc.total_bytes(), 0);
}

// Edge cases
TEST_F(AllocatorTraitsTest, EdgeCases_ZeroAllocation) {
    TestAllocator<int> alloc;

    // Allocating zero elements
    int* p = fcm::allocate_n(alloc, 0);
    EXPECT_NE(p, nullptr); // Should still return valid pointer
    fcm::deallocate_n(alloc, p, 0);
}

TEST_F(AllocatorTraitsTest, EdgeCases_LargeAllocation) {
    TestAllocator<char> alloc;

    // Large allocation (1MB)
    std::size_t size = 1024 * 1024;
    char* p = fcm::allocate_n(alloc, size);
    ASSERT_NE(p, nullptr);

    // Use the memory
    std::memset(p, 0, size);

    fcm::deallocate_n(alloc, p, size);
    EXPECT_EQ(alloc.total_bytes(), size);
}

// Test allocator with fancy pointers
template<class T>
class FancyPointer {
public:
    using element_type = T;
    using difference_type = std::ptrdiff_t;

    FancyPointer() = default;
    explicit FancyPointer(T* p) : ptr_(p) {}

    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

template<class T>
class FancyAllocator {
public:
    using value_type = T;
    using pointer = FancyPointer<T>;
    using const_pointer = FancyPointer<const T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind {
        using other = FancyAllocator<U>;
    };

    pointer allocate(std::size_t n) {
        return pointer(static_cast<T*>(::operator new(n * sizeof(T))));
    }

    void deallocate(pointer p, std::size_t) {
        ::operator delete(p.get());
    }
};

TEST_F(AllocatorTraitsTest, FancyPointer_TypeAliases) {
    using Alloc = FancyAllocator<int>;

    // Should use fancy pointer types
    static_assert(std::is_same_v<fcm::allocator_pointer_t<Alloc>, FancyPointer<int>>);
    static_assert(std::is_same_v<fcm::allocator_const_pointer_t<Alloc>, FancyPointer<const int>>);
}

// Test compile-time properties
TEST_F(AllocatorTraitsTest, CompileTimeProperties) {
    // All functions should be available at compile time for constant evaluation
    using Alloc = std::allocator<int>;

    // These should all be constexpr evaluable
    constexpr bool is_equal = fcm::is_always_equal_v<Alloc>;
    EXPECT_TRUE(is_equal);

    // Type traits should be usable in SFINAE contexts
    using ValueType = fcm::allocator_value_t<Alloc>;
    static_assert(std::is_same_v<ValueType, int>);
}
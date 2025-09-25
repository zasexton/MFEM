#include <gtest/gtest.h>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <limits>

#include <core/memory/malloc_allocator.h>
#include <core/error/error_code.h>

namespace fem::core::memory {
namespace {

// Test structure with destructor tracking
struct DestructorTracker {
    static int destruction_count;
    static int construction_count;
    int value;

    DestructorTracker(int v = 0) : value(v) { ++construction_count; }
    DestructorTracker(const DestructorTracker& other) : value(other.value) { ++construction_count; }
    DestructorTracker(DestructorTracker&& other) noexcept : value(other.value) { ++construction_count; }
    ~DestructorTracker() { ++destruction_count; }

    DestructorTracker& operator=(const DestructorTracker& other) {
        value = other.value;
        return *this;
    }

    static void reset() {
        destruction_count = 0;
        construction_count = 0;
    }
};

int DestructorTracker::destruction_count = 0;
int DestructorTracker::construction_count = 0;

// Test structure with specific alignment requirements
struct alignas(32) AlignedStruct {
    std::uint64_t data[4];
    int id;

    AlignedStruct() : data{}, id(0) {}
    explicit AlignedStruct(int i) : data{}, id(i) {}
};

// Large structure for stress testing
struct LargeStruct {
    char data[1024];

    LargeStruct() : data{} {}
    explicit LargeStruct(char c) { std::fill(std::begin(data), std::end(data), c); }
};

// POD type for testing trivial operations
struct PodType {
    int x;
    double y;
    char z[16];
};

class MallocAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        DestructorTracker::reset();
    }

    void TearDown() override {
        // Verify no memory leaks in DestructorTracker
        EXPECT_EQ(DestructorTracker::construction_count, DestructorTracker::destruction_count)
            << "Memory leak detected: constructed=" << DestructorTracker::construction_count
            << ", destroyed=" << DestructorTracker::destruction_count;
    }
};

// Basic construction tests
TEST_F(MallocAllocatorTest, DefaultConstruction) {
    [[maybe_unused]] MallocAllocator<int> alloc;
    // MallocAllocator is stateless, so nothing specific to check
    // Just ensure it compiles and doesn't crash
    SUCCEED();
}

TEST_F(MallocAllocatorTest, CopyConstruction) {
    MallocAllocator<int> alloc1;
    MallocAllocator<int> alloc2(alloc1);

    // Both allocators should be equal (stateless)
    EXPECT_EQ(alloc1, alloc2);
}

TEST_F(MallocAllocatorTest, CrossTypeCopyConstruction) {
    MallocAllocator<int> int_alloc;
    MallocAllocator<double> double_alloc(int_alloc);

    // Should compile and work correctly
    SUCCEED();
}

// Basic allocation and deallocation tests
TEST_F(MallocAllocatorTest, BasicAllocationDeallocation) {
    MallocAllocator<int> alloc;

    const std::size_t n = 10;
    int* ptr = alloc.allocate(n);
    ASSERT_NE(ptr, nullptr);

    // Use the memory
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<int>(i * 2);
    }

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i], static_cast<int>(i * 2));
    }

    // Should not crash
    alloc.deallocate(ptr, n);
}

TEST_F(MallocAllocatorTest, SingleElementAllocation) {
    MallocAllocator<double> alloc;

    double* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    *ptr = 3.14159;
    EXPECT_DOUBLE_EQ(*ptr, 3.14159);

    alloc.deallocate(ptr, 1);
}

TEST_F(MallocAllocatorTest, LargeAllocation) {
    MallocAllocator<LargeStruct> alloc;

    const std::size_t n = 100;
    LargeStruct* ptr = alloc.allocate(n);
    ASSERT_NE(ptr, nullptr);

    // Initialize and verify
    for (std::size_t i = 0; i < n; ++i) {
        new (&ptr[i]) LargeStruct(static_cast<char>('A' + (i % 26)));
    }

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i].data[0], static_cast<char>('A' + (i % 26)));
    }

    // Cleanup
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i].~LargeStruct();
    }

    alloc.deallocate(ptr, n);
}

TEST_F(MallocAllocatorTest, ZeroAllocation) {
    MallocAllocator<int> alloc;

    // Zero allocation behavior is implementation-defined
    // but should not crash
    int* ptr = alloc.allocate(0);
    alloc.deallocate(ptr, 0);
}

// Alignment tests
TEST_F(MallocAllocatorTest, AlignmentForBasicTypes) {
    MallocAllocator<double> double_alloc;
    MallocAllocator<std::int64_t> int64_alloc;

    double* d_ptr = double_alloc.allocate(1);
    ASSERT_NE(d_ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(d_ptr) % alignof(double), 0u);
    double_alloc.deallocate(d_ptr, 1);

    std::int64_t* i_ptr = int64_alloc.allocate(1);
    ASSERT_NE(i_ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(i_ptr) % alignof(std::int64_t), 0u);
    int64_alloc.deallocate(i_ptr, 1);
}

TEST_F(MallocAllocatorTest, AlignmentForAlignedStruct) {
    MallocAllocator<AlignedStruct> alloc;

    AlignedStruct* ptr = alloc.allocate(5);
    ASSERT_NE(ptr, nullptr);

    // Check alignment based on whether aligned new is available
#if defined(__cpp_aligned_new)
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignof(AlignedStruct), 0u);
#else
    // Without aligned new, we can't guarantee alignment > alignof(std::max_align_t)
    // But it should at least be aligned to max_align_t
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignof(std::max_align_t), 0u);
#endif

    alloc.deallocate(ptr, 5);
}

// try_allocate tests
TEST_F(MallocAllocatorTest, TryAllocateSuccess) {
    MallocAllocator<int> alloc;

    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());

    int* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        ptr[i] = i * i;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ptr[i], i * i);
    }

    alloc.deallocate(ptr, 10);
}

TEST_F(MallocAllocatorTest, TryAllocateFailureSimulation) {
    MallocAllocator<char> alloc;

    // We can't easily force std::bad_alloc without exhausting memory,
    // but we can test that try_allocate returns proper Result type
    auto result = alloc.try_allocate(100);
    ASSERT_TRUE(result.is_ok());

    char* ptr = result.value();
    ASSERT_NE(ptr, nullptr);
    alloc.deallocate(ptr, 100);

    // The error path would return ErrorCode::OutOfMemory
    // but we can't test it without mocking operator new
}

// Construct and destroy tests
TEST_F(MallocAllocatorTest, ConstructDestroy) {
    MallocAllocator<DestructorTracker> alloc;

    DestructorTracker* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    // Construct object
    alloc.construct(ptr, 42);
    EXPECT_EQ(ptr->value, 42);
    EXPECT_EQ(DestructorTracker::construction_count, 1);
    EXPECT_EQ(DestructorTracker::destruction_count, 0);

    // Destroy object
    alloc.destroy(ptr);
    EXPECT_EQ(DestructorTracker::destruction_count, 1);

    alloc.deallocate(ptr, 1);
}

TEST_F(MallocAllocatorTest, ConstructDestroyArray) {
    MallocAllocator<DestructorTracker> alloc;

    const std::size_t n = 5;
    DestructorTracker* ptr = alloc.allocate(n);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for (std::size_t i = 0; i < n; ++i) {
        alloc.construct(ptr + i, static_cast<int>(i * 10));
    }

    EXPECT_EQ(DestructorTracker::construction_count, static_cast<int>(n));

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i].value, static_cast<int>(i * 10));
    }

    // Destroy objects
    for (std::size_t i = 0; i < n; ++i) {
        alloc.destroy(ptr + i);
    }

    EXPECT_EQ(DestructorTracker::destruction_count, static_cast<int>(n));

    alloc.deallocate(ptr, n);
}

TEST_F(MallocAllocatorTest, ConstructWithMultipleArgs) {
    MallocAllocator<std::pair<int, std::string>> alloc;

    auto* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    alloc.construct(ptr, 123, "test string");
    EXPECT_EQ(ptr->first, 123);
    EXPECT_EQ(ptr->second, "test string");

    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
}

TEST_F(MallocAllocatorTest, TrivialDestructorOptimization) {
    MallocAllocator<PodType> alloc;

    PodType* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    alloc.construct(ptr);
    ptr->x = 100;
    ptr->y = 3.14;

    // For trivially destructible types, destroy should be a no-op
    alloc.destroy(ptr); // Should compile to nothing

    alloc.deallocate(ptr, 1);
}

// Rebind tests
TEST_F(MallocAllocatorTest, RebindBasic) {
    using IntAlloc = MallocAllocator<int>;
    using DoubleAlloc = IntAlloc::rebind<double>::other;

    static_assert(std::is_same_v<DoubleAlloc::value_type, double>);
    static_assert(std::is_same_v<DoubleAlloc, MallocAllocator<double>>);
}

TEST_F(MallocAllocatorTest, RebindToVoid) {
    using IntAlloc = MallocAllocator<int>;
    using VoidAlloc = IntAlloc::rebind<void>::other;

    static_assert(std::is_same_v<VoidAlloc::value_type, void>);
    static_assert(std::is_same_v<VoidAlloc, MallocAllocator<void>>);
}

TEST_F(MallocAllocatorTest, RebindChain) {
    using IntAlloc = MallocAllocator<int>;
    using CharAlloc = IntAlloc::rebind<char>::other;
    using DoubleAlloc = CharAlloc::rebind<double>::other;
    using FinalAlloc = DoubleAlloc::rebind<int>::other;

    static_assert(std::is_same_v<FinalAlloc, IntAlloc>);
}

// Comparison tests
TEST_F(MallocAllocatorTest, EqualityComparison) {
    MallocAllocator<int> alloc1;
    MallocAllocator<int> alloc2;

    // All MallocAllocators are equal (stateless)
    EXPECT_EQ(alloc1, alloc2);
    EXPECT_FALSE(alloc1 != alloc2);
}

TEST_F(MallocAllocatorTest, EqualityComparisonDifferentTypes) {
    MallocAllocator<int> int_alloc;
    MallocAllocator<double> double_alloc;
    MallocAllocator<std::string> string_alloc;

    // All MallocAllocators are equal regardless of type
    EXPECT_EQ(int_alloc, double_alloc);
    EXPECT_EQ(double_alloc, string_alloc);
    EXPECT_EQ(int_alloc, string_alloc);
}

// Allocator traits tests
TEST_F(MallocAllocatorTest, AllocatorTraits) {
    using Alloc = MallocAllocator<int>;

    static_assert(std::is_same_v<Alloc::value_type, int>);
    static_assert(std::is_same_v<Alloc::pointer, int*>);
    static_assert(std::is_same_v<Alloc::const_pointer, const int*>);
    static_assert(std::is_same_v<Alloc::size_type, std::size_t>);
    static_assert(std::is_same_v<Alloc::difference_type, std::ptrdiff_t>);

    static_assert(!Alloc::propagate_on_container_copy_assignment::value);
    static_assert(Alloc::propagate_on_container_move_assignment::value);
    static_assert(Alloc::propagate_on_container_swap::value);
    static_assert(Alloc::is_always_equal::value);
}

TEST_F(MallocAllocatorTest, StatelessProperty) {
    // MallocAllocator should be empty (stateless)
    static_assert(std::is_empty_v<MallocAllocator<int>>);

    // Size should be 1 (empty class optimization)
    EXPECT_EQ(sizeof(MallocAllocator<int>), 1u);
}

// STL container integration tests
TEST_F(MallocAllocatorTest, VectorIntegration) {
    std::vector<int, MallocAllocator<int>> vec;
    vec.reserve(100);

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100u);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    // Test operations
    vec.erase(vec.begin() + 50);
    EXPECT_EQ(vec.size(), 99u);

    vec.clear();
    EXPECT_EQ(vec.size(), 0u);
}

TEST_F(MallocAllocatorTest, ListIntegration) {
    std::list<std::string, MallocAllocator<std::string>> list;

    list.push_back("first");
    list.push_back("second");
    list.push_front("zero");

    EXPECT_EQ(list.size(), 3u);
    EXPECT_EQ(list.front(), "zero");
    EXPECT_EQ(list.back(), "second");

    list.sort();
    auto it = list.begin();
    EXPECT_EQ(*it++, "first");
    EXPECT_EQ(*it++, "second");
    EXPECT_EQ(*it++, "zero");
}

TEST_F(MallocAllocatorTest, DequeIntegration) {
    std::deque<double, MallocAllocator<double>> deque;

    for (double d = 0.0; d < 10.0; d += 0.5) {
        deque.push_back(d);
    }

    EXPECT_EQ(deque.size(), 20u);
    EXPECT_DOUBLE_EQ(deque.front(), 0.0);
    EXPECT_DOUBLE_EQ(deque.back(), 9.5);

    deque.pop_front();
    deque.pop_back();
    EXPECT_EQ(deque.size(), 18u);
}

TEST_F(MallocAllocatorTest, SetIntegration) {
    using AllocType = MallocAllocator<int>;
    std::set<int, std::less<int>, AllocType> set;

    for (int i = 0; i < 50; ++i) {
        set.insert(i * 2);
    }

    EXPECT_EQ(set.size(), 50u);
    EXPECT_EQ(*set.begin(), 0);
    EXPECT_EQ(*set.rbegin(), 98);

    auto it = set.find(20);
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, 20);
}

TEST_F(MallocAllocatorTest, MapIntegration) {
    using PairAlloc = MallocAllocator<std::pair<const std::string, int>>;
    std::map<std::string, int, std::less<std::string>, PairAlloc> map;

    map["one"] = 1;
    map["two"] = 2;
    map["three"] = 3;

    EXPECT_EQ(map.size(), 3u);
    EXPECT_EQ(map["two"], 2);

    map.erase("one");
    EXPECT_EQ(map.size(), 2u);
    EXPECT_EQ(map.find("one"), map.end());
}

TEST_F(MallocAllocatorTest, UnorderedMapIntegration) {
    using PairAlloc = MallocAllocator<std::pair<const int, std::string>>;
    std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>, PairAlloc> umap;

    for (int i = 0; i < 100; ++i) {
        umap[i] = std::to_string(i);
    }

    EXPECT_EQ(umap.size(), 100u);
    EXPECT_EQ(umap[42], "42");

    umap.clear();
    EXPECT_EQ(umap.size(), 0u);
}

// Edge cases and stress tests
TEST_F(MallocAllocatorTest, MultipleAllocations) {
    MallocAllocator<int> alloc;
    std::vector<int*> ptrs;
    const std::size_t num_allocs = 100;

    // Allocate multiple blocks
    for (std::size_t i = 0; i < num_allocs; ++i) {
        int* ptr = alloc.allocate(i + 1);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);

        // Initialize
        for (std::size_t j = 0; j <= i; ++j) {
            ptr[j] = static_cast<int>(i * 1000 + j);
        }
    }

    // Verify all allocations
    for (std::size_t i = 0; i < num_allocs; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            EXPECT_EQ(ptrs[i][j], static_cast<int>(i * 1000 + j));
        }
    }

    // Deallocate in reverse order
    for (std::size_t i = num_allocs; i > 0; --i) {
        alloc.deallocate(ptrs[i - 1], i);
    }
}

TEST_F(MallocAllocatorTest, MoveSemantics) {
    std::vector<std::string, MallocAllocator<std::string>> vec1;
    vec1.emplace_back("Hello");
    vec1.emplace_back("World");

    std::vector<std::string, MallocAllocator<std::string>> vec2 = std::move(vec1);
    EXPECT_EQ(vec2.size(), 2u);
    EXPECT_EQ(vec2[0], "Hello");
    EXPECT_EQ(vec2[1], "World");
}

TEST_F(MallocAllocatorTest, SwapContainers) {
    std::vector<int, MallocAllocator<int>> vec1{1, 2, 3};
    std::vector<int, MallocAllocator<int>> vec2{4, 5, 6, 7, 8};

    std::swap(vec1, vec2);

    EXPECT_EQ(vec1.size(), 5u);
    EXPECT_EQ(vec2.size(), 3u);
    EXPECT_EQ(vec1[0], 4);
    EXPECT_EQ(vec2[0], 1);
}

TEST_F(MallocAllocatorTest, NestedContainers) {
    using InnerVec = std::vector<int, MallocAllocator<int>>;
    using OuterVec = std::vector<InnerVec, MallocAllocator<InnerVec>>;

    OuterVec matrix;
    for (int i = 0; i < 3; ++i) {
        InnerVec row;
        for (int j = 0; j < 3; ++j) {
            row.push_back(i * 3 + j);
        }
        matrix.push_back(row);
    }

    EXPECT_EQ(matrix.size(), 3u);
    EXPECT_EQ(matrix[1][1], 4);
    EXPECT_EQ(matrix[2][2], 8);
}

TEST_F(MallocAllocatorTest, AllocationPattern) {
    MallocAllocator<char> alloc;

    // Allocate various sizes
    char* small = alloc.allocate(16);
    char* medium = alloc.allocate(256);
    char* large = alloc.allocate(4096);

    ASSERT_NE(small, nullptr);
    ASSERT_NE(medium, nullptr);
    ASSERT_NE(large, nullptr);

    // Use the allocations
    std::fill(small, small + 16, 'S');
    std::fill(medium, medium + 256, 'M');
    std::fill(large, large + 4096, 'L');

    // Verify
    EXPECT_EQ(small[0], 'S');
    EXPECT_EQ(medium[255], 'M');
    EXPECT_EQ(large[4095], 'L');

    // Cleanup
    alloc.deallocate(small, 16);
    alloc.deallocate(medium, 256);
    alloc.deallocate(large, 4096);
}

TEST_F(MallocAllocatorTest, CustomObjectLifecycle) {
    struct CustomObject {
        std::string name;
        std::vector<int> data;

        CustomObject(const std::string& n, std::initializer_list<int> d)
            : name(n), data(d) {}
    };

    MallocAllocator<CustomObject> alloc;

    CustomObject* obj = alloc.allocate(1);
    ASSERT_NE(obj, nullptr);

    alloc.construct(obj, "test", std::initializer_list<int>{1, 2, 3, 4, 5});
    EXPECT_EQ(obj->name, "test");
    EXPECT_EQ(obj->data.size(), 5u);
    EXPECT_EQ(obj->data[2], 3);

    alloc.destroy(obj);
    alloc.deallocate(obj, 1);
}

// Performance characteristics test
TEST_F(MallocAllocatorTest, ConsistentAllocation) {
    MallocAllocator<int> alloc;

    // Multiple allocations of the same size should work consistently
    const std::size_t size = 100;
    const std::size_t iterations = 50;

    for (std::size_t i = 0; i < iterations; ++i) {
        int* ptr = alloc.allocate(size);
        ASSERT_NE(ptr, nullptr);

        // Use memory to ensure it's valid
        ptr[0] = static_cast<int>(i);
        ptr[size - 1] = static_cast<int>(i * 2);

        EXPECT_EQ(ptr[0], static_cast<int>(i));
        EXPECT_EQ(ptr[size - 1], static_cast<int>(i * 2));

        alloc.deallocate(ptr, size);
    }
}

TEST_F(MallocAllocatorTest, AllocatorCopySemantics) {
    MallocAllocator<int> alloc1;
    MallocAllocator<int> alloc2 = alloc1;

    // Both allocators should work independently
    int* ptr1 = alloc1.allocate(10);
    int* ptr2 = alloc2.allocate(20);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);

    alloc1.deallocate(ptr1, 10);
    alloc2.deallocate(ptr2, 20);
}

} // namespace
} // namespace fem::core::memory
#include <gtest/gtest.h>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <core/memory/aligned_allocator.h>
#include <core/memory/memory_resource.h>
#include <core/error/error_code.h>

namespace fem::core::memory {
namespace {

// Test structure with specific alignment requirements
struct alignas(32) AlignedStruct {
    std::uint64_t data[4];
    int id;

    AlignedStruct() : data{}, id(0) {}
    explicit AlignedStruct(int i) : data{}, id(i) {}
};

// Test structure with destructor tracking
struct DestructorTracker {
    static int destruction_count;
    int value;

    DestructorTracker(int v = 0) : value(v) {}
    ~DestructorTracker() { ++destruction_count; }
};

int DestructorTracker::destruction_count = 0;

// Custom memory resource for testing
class TestMemoryResource : public memory_resource {
public:
    struct AllocationRecord {
        std::size_t bytes;
        std::size_t alignment;
    };

    mutable std::vector<AllocationRecord> allocations;
    mutable std::size_t total_allocated = 0;
    mutable std::size_t total_deallocated = 0;
    bool fail_allocation = false;

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (fail_allocation) {
            throw std::bad_alloc();
        }

        allocations.push_back({bytes, alignment});
        total_allocated += bytes;

        // Allocate with proper alignment
        void* ptr = nullptr;
        if (alignment > alignof(std::max_align_t)) {
            #ifdef _WIN32
                ptr = _aligned_malloc(bytes, alignment);
            #else
                if (posix_memalign(&ptr, alignment, bytes) != 0) {
                    throw std::bad_alloc();
                }
            #endif
        } else {
            ptr = std::malloc(bytes);
        }

        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        if (!p) return;
        total_deallocated += bytes;

        #ifdef _WIN32
            if (alignment > alignof(std::max_align_t)) {
                _aligned_free(p);
            } else {
                std::free(p);
            }
        #else
            (void)alignment; // Suppress unused parameter warning
            std::free(p);
        #endif
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }
};

class AlignedAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        DestructorTracker::destruction_count = 0;
    }
};

// Basic allocation and deallocation tests
TEST_F(AlignedAllocatorTest, DefaultConstruction) {
    AlignedAllocator<int> alloc;
    EXPECT_NE(alloc.resource(), nullptr);
    EXPECT_EQ(alloc.resource(), default_resource());
}

TEST_F(AlignedAllocatorTest, ConstructionWithResource) {
    TestMemoryResource mr;
    AlignedAllocator<int> alloc(&mr);
    EXPECT_EQ(alloc.resource(), &mr);
}

TEST_F(AlignedAllocatorTest, BasicAllocationDeallocation) {
    TestMemoryResource mr;
    AlignedAllocator<int> alloc(&mr);

    const std::size_t n = 10;
    int* ptr = alloc.allocate(n);
    ASSERT_NE(ptr, nullptr);

    // Verify allocation was tracked
    EXPECT_EQ(mr.allocations.size(), 1u);
    EXPECT_EQ(mr.allocations[0].bytes, n * sizeof(int));
    EXPECT_EQ(mr.allocations[0].alignment, alignof(int));

    // Use the memory
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<int>(i);
    }

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i], static_cast<int>(i));
    }

    alloc.deallocate(ptr, n);
    EXPECT_EQ(mr.total_deallocated, n * sizeof(int));
}

TEST_F(AlignedAllocatorTest, NullptrDeallocation) {
    AlignedAllocator<int> alloc;
    EXPECT_NO_THROW(alloc.deallocate(nullptr, 0));
    EXPECT_NO_THROW(alloc.deallocate(nullptr, 100));
}

// Alignment tests
TEST_F(AlignedAllocatorTest, DefaultAlignment) {
    TestMemoryResource mr;
    AlignedAllocator<double> alloc(&mr);

    double* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(mr.allocations[0].alignment, alignof(double));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignof(double), 0u);

    alloc.deallocate(ptr, 1);
}

TEST_F(AlignedAllocatorTest, CustomAlignment16) {
    TestMemoryResource mr;
    AlignedAllocator<int, 16> alloc(&mr);

    int* ptr = alloc.allocate(5);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(mr.allocations[0].alignment, 16u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 16, 0u);

    alloc.deallocate(ptr, 5);
}

TEST_F(AlignedAllocatorTest, CustomAlignment32) {
    TestMemoryResource mr;
    AlignedAllocator<char, 32> alloc(&mr);

    char* ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(mr.allocations[0].alignment, 32u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 32, 0u);

    alloc.deallocate(ptr, 100);
}

TEST_F(AlignedAllocatorTest, CustomAlignment64) {
    TestMemoryResource mr;
    AlignedAllocator<std::uint8_t, 64> alloc(&mr);

    std::uint8_t* ptr = alloc.allocate(256);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(mr.allocations[0].alignment, 64u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);

    alloc.deallocate(ptr, 256);
}

TEST_F(AlignedAllocatorTest, AlignedStructAllocation) {
    TestMemoryResource mr;
    AlignedAllocator<AlignedStruct> alloc(&mr);

    AlignedStruct* ptr = alloc.allocate(3);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(mr.allocations[0].alignment, alignof(AlignedStruct));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignof(AlignedStruct), 0u);

    alloc.deallocate(ptr, 3);
}

// try_allocate tests
TEST_F(AlignedAllocatorTest, TryAllocateSuccess) {
    TestMemoryResource mr;
    AlignedAllocator<int> alloc(&mr);

    auto result = alloc.try_allocate(10);
    ASSERT_TRUE(result.is_ok());

    int* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    // Use the memory
    for (int i = 0; i < 10; ++i) {
        ptr[i] = i;
    }

    alloc.deallocate(ptr, 10);
}

TEST_F(AlignedAllocatorTest, TryAllocateFailure) {
    TestMemoryResource mr;
    mr.fail_allocation = true;
    AlignedAllocator<int> alloc(&mr);

    auto result = alloc.try_allocate(10);
    ASSERT_FALSE(result.is_ok());
    EXPECT_EQ(result.error(), fem::core::error::ErrorCode::OutOfMemory);
}

// Construct and destroy tests
TEST_F(AlignedAllocatorTest, ConstructDestroy) {
    TestMemoryResource mr;
    AlignedAllocator<DestructorTracker> alloc(&mr);

    DestructorTracker* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    // Construct object
    alloc.construct(ptr, 42);
    EXPECT_EQ(ptr->value, 42);
    EXPECT_EQ(DestructorTracker::destruction_count, 0);

    // Destroy object
    alloc.destroy(ptr);
    EXPECT_EQ(DestructorTracker::destruction_count, 1);

    alloc.deallocate(ptr, 1);
}

TEST_F(AlignedAllocatorTest, ConstructDestroyArray) {
    TestMemoryResource mr;
    AlignedAllocator<DestructorTracker> alloc(&mr);

    const std::size_t n = 5;
    DestructorTracker* ptr = alloc.allocate(n);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for (std::size_t i = 0; i < n; ++i) {
        alloc.construct(ptr + i, static_cast<int>(i));
    }

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i].value, static_cast<int>(i));
    }
    EXPECT_EQ(DestructorTracker::destruction_count, 0);

    // Destroy objects
    for (std::size_t i = 0; i < n; ++i) {
        alloc.destroy(ptr + i);
    }
    EXPECT_EQ(DestructorTracker::destruction_count, static_cast<int>(n));

    alloc.deallocate(ptr, n);
}

TEST_F(AlignedAllocatorTest, ConstructWithArgs) {
    TestMemoryResource mr;
    AlignedAllocator<std::pair<int, double>> alloc(&mr);

    auto* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    alloc.construct(ptr, 42, 3.14);
    EXPECT_EQ(ptr->first, 42);
    EXPECT_DOUBLE_EQ(ptr->second, 3.14);

    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
}

// Rebind tests
TEST_F(AlignedAllocatorTest, RebindBasic) {
    using IntAlloc = AlignedAllocator<int>;
    using DoubleAlloc = IntAlloc::rebind<double>::other;

    static_assert(std::is_same_v<DoubleAlloc::value_type, double>);
    static_assert(DoubleAlloc::alignment >= alignof(double));
}

TEST_F(AlignedAllocatorTest, RebindWithAlignment) {
    using IntAlloc = AlignedAllocator<int, 32>;
    using CharAlloc = IntAlloc::rebind<char>::other;

    static_assert(std::is_same_v<CharAlloc::value_type, char>);
    static_assert(CharAlloc::alignment == 32);
}

TEST_F(AlignedAllocatorTest, RebindCopyConstructor) {
    TestMemoryResource mr;
    AlignedAllocator<int, 16> int_alloc(&mr);

    AlignedAllocator<double, 16> double_alloc(int_alloc);
    EXPECT_EQ(double_alloc.resource(), &mr);
}

// Comparison tests
TEST_F(AlignedAllocatorTest, EqualityComparison) {
    TestMemoryResource mr1;
    TestMemoryResource mr2;

    AlignedAllocator<int> alloc1(&mr1);
    AlignedAllocator<int> alloc2(&mr1);
    AlignedAllocator<int> alloc3(&mr2);

    EXPECT_EQ(alloc1, alloc2);
    EXPECT_NE(alloc1, alloc3);
}

TEST_F(AlignedAllocatorTest, EqualityComparisonDifferentTypes) {
    TestMemoryResource mr;

    // Allocators with different types but same alignment should compare equal
    // if they use the same memory resource
    AlignedAllocator<int, 16> int_alloc(&mr);
    AlignedAllocator<char, 16> char_alloc(&mr);

    EXPECT_EQ(int_alloc, char_alloc);

    // Allocators with different alignments should compare not equal
    AlignedAllocator<int> int_default(&mr);
    AlignedAllocator<double> double_default(&mr);

    // These have different alignments (4 vs 8), so should not be equal
    EXPECT_NE(int_default, double_default);
}

TEST_F(AlignedAllocatorTest, EqualityComparisonDifferentAlignment) {
    TestMemoryResource mr;

    AlignedAllocator<int, 16> alloc1(&mr);
    AlignedAllocator<int, 32> alloc2(&mr);

    EXPECT_NE(alloc1, alloc2);
}

// Allocator traits tests
TEST_F(AlignedAllocatorTest, AllocatorTraits) {
    using Alloc = AlignedAllocator<int>;

    static_assert(std::is_same_v<Alloc::value_type, int>);
    static_assert(std::is_same_v<Alloc::pointer, int*>);
    static_assert(std::is_same_v<Alloc::const_pointer, const int*>);
    static_assert(std::is_same_v<Alloc::size_type, std::size_t>);
    static_assert(std::is_same_v<Alloc::difference_type, std::ptrdiff_t>);

    static_assert(!Alloc::propagate_on_container_copy_assignment::value);
    static_assert(Alloc::propagate_on_container_move_assignment::value);
    static_assert(Alloc::propagate_on_container_swap::value);
    static_assert(!Alloc::is_always_equal::value);
}

// STL container integration tests
TEST_F(AlignedAllocatorTest, VectorIntegration) {
    TestMemoryResource mr;
    AlignedAllocator<int, 16> alloc(&mr);

    std::vector<int, AlignedAllocator<int, 16>> vec(alloc);
    vec.reserve(10);

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    // Check that memory was allocated with correct alignment
    EXPECT_GT(mr.allocations.size(), 0u);
    for (const auto& record : mr.allocations) {
        EXPECT_EQ(record.alignment, 16u);
    }
}

TEST_F(AlignedAllocatorTest, ListIntegration) {
    TestMemoryResource mr;
    AlignedAllocator<double, 32> alloc(&mr);

    std::list<double, AlignedAllocator<double, 32>> list(alloc);

    for (double i = 0; i < 5; ++i) {
        list.push_back(i * 1.5);
    }

    EXPECT_EQ(list.size(), 5u);

    double expected = 0.0;
    for (double val : list) {
        EXPECT_DOUBLE_EQ(val, expected * 1.5);
        expected += 1.0;
    }
}

TEST_F(AlignedAllocatorTest, DequeIntegration) {
    TestMemoryResource mr;
    AlignedAllocator<std::string> alloc(&mr);

    std::deque<std::string, AlignedAllocator<std::string>> deque(alloc);

    deque.push_back("hello");
    deque.push_back("world");
    deque.push_front("start");

    EXPECT_EQ(deque.size(), 3u);
    EXPECT_EQ(deque[0], "start");
    EXPECT_EQ(deque[1], "hello");
    EXPECT_EQ(deque[2], "world");
}

TEST_F(AlignedAllocatorTest, MapIntegration) {
    TestMemoryResource mr;
    using PairAlloc = AlignedAllocator<std::pair<const int, std::string>, 16>;

    PairAlloc alloc(&mr);
    std::map<int, std::string, std::less<int>, PairAlloc> test_map(alloc);

    test_map[1] = "one";
    test_map[2] = "two";
    test_map[3] = "three";

    EXPECT_EQ(test_map.size(), 3u);
    EXPECT_EQ(test_map[2], "two");
}

// Edge cases and stress tests
TEST_F(AlignedAllocatorTest, LargeAllocation) {
    TestMemoryResource mr;
    AlignedAllocator<double, 64> alloc(&mr);

    const std::size_t large_size = 10000;
    double* ptr = alloc.allocate(large_size);
    ASSERT_NE(ptr, nullptr);

    // Verify alignment for large allocation
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);

    // Use the memory
    for (std::size_t i = 0; i < large_size; ++i) {
        ptr[i] = static_cast<double>(i);
    }

    alloc.deallocate(ptr, large_size);
}

TEST_F(AlignedAllocatorTest, ZeroSizeAllocation) {
    TestMemoryResource mr;
    AlignedAllocator<int> alloc(&mr);

    // Zero-size allocation behavior is implementation-defined
    // but should not crash
    int* ptr = alloc.allocate(0);
    alloc.deallocate(ptr, 0);
}

TEST_F(AlignedAllocatorTest, MultipleAllocations) {
    TestMemoryResource mr;
    AlignedAllocator<int, 32> alloc(&mr);

    std::vector<int*> ptrs;
    const std::size_t num_allocs = 50;

    // Allocate multiple blocks
    for (std::size_t i = 0; i < num_allocs; ++i) {
        int* ptr = alloc.allocate(i + 1);
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 32, 0u);
        ptrs.push_back(ptr);
    }

    // Deallocate in reverse order
    for (std::size_t i = num_allocs; i > 0; --i) {
        alloc.deallocate(ptrs[i - 1], i);
    }

    EXPECT_EQ(mr.allocations.size(), num_allocs);
    EXPECT_EQ(mr.total_allocated, mr.total_deallocated);
}

TEST_F(AlignedAllocatorTest, MoveSemantics) {
    TestMemoryResource mr;
    AlignedAllocator<int, 16> alloc(&mr);

    std::vector<int, AlignedAllocator<int, 16>> vec1(alloc);
    vec1.resize(10);

    std::vector<int, AlignedAllocator<int, 16>> vec2 = std::move(vec1);
    EXPECT_EQ(vec2.size(), 10u);

    // Due to propagate_on_container_move_assignment = true,
    // vec2 should use the same allocator
    EXPECT_EQ(vec2.get_allocator().resource(), &mr);
}

TEST_F(AlignedAllocatorTest, SwapSemantics) {
    TestMemoryResource mr1;
    TestMemoryResource mr2;

    AlignedAllocator<int, 16> alloc1(&mr1);
    AlignedAllocator<int, 16> alloc2(&mr2);

    std::vector<int, AlignedAllocator<int, 16>> vec1(alloc1);
    std::vector<int, AlignedAllocator<int, 16>> vec2(alloc2);

    vec1.resize(5);
    vec2.resize(10);

    std::swap(vec1, vec2);

    // Due to propagate_on_container_swap = true,
    // allocators should be swapped
    EXPECT_EQ(vec1.size(), 10u);
    EXPECT_EQ(vec2.size(), 5u);
}

// Performance characteristics test
TEST_F(AlignedAllocatorTest, AllocationPattern) {
    TestMemoryResource mr;
    AlignedAllocator<char, 128> alloc(&mr);

    // Allocate with very high alignment
    char* ptr = alloc.allocate(256);
    ASSERT_NE(ptr, nullptr);

    // Verify the high alignment is respected
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 128, 0u);

    alloc.deallocate(ptr, 256);

    // Verify the allocation record
    EXPECT_EQ(mr.allocations.size(), 1u);
    EXPECT_EQ(mr.allocations[0].bytes, 256u);
    EXPECT_EQ(mr.allocations[0].alignment, 128u);
}

} // namespace
} // namespace fem::core::memory
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <array>

#include <traits/storage_traits.h>

// Additional mock storage types for testing
namespace test_storage {

// Non-storage type
struct NotAStorage {
    int value;
};

// Storage with view semantics
template<typename T>
class ViewStorage : public fem::numeric::StorageBase<T> {
public:
    using value_type = T;
    T* data() { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// Sparse storage mock
template<typename T>
class SparseStorage : public fem::numeric::StorageBase<T> {
public:
    using value_type = T;

    size_t nnz() const { return nnz_; }
    const char* format() const { return "CSR"; }
    size_t size() const { return rows_ * cols_; }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t nnz_ = 0;
};

// Pool-allocated storage mock
template<typename T>
class PoolStorage : public fem::numeric::StorageBase<T> {
public:
    using value_type = T;

    void* pool() { return pool_; }
    size_t chunk_size() const { return chunk_size_; }
    size_t size() const { return size_; }

private:
    void* pool_ = nullptr;
    size_t chunk_size_ = 4096;
    size_t size_ = 0;
};

// Device storage mock
template<typename T>
class DeviceStorage : public fem::numeric::StorageBase<T> {
public:
    using value_type = T;
    using device_type = fem::numeric::Device;
    static constexpr fem::numeric::Device device_type_value = fem::numeric::Device::GPU;
    static constexpr bool is_unified = false;

    size_t size() const { return size_; }

private:
    size_t size_ = 0;
};

// Unified memory storage mock
template<typename T>
class UnifiedStorage : public fem::numeric::StorageBase<T> {
public:
    using value_type = T;
    using device_type = fem::numeric::Device;
    static constexpr bool is_unified = true;

    size_t size() const { return size_; }

private:
    size_t size_ = 0;
};

// Operation mocks for suitability testing
struct SIMDOperation {
    static constexpr bool requires_simd = true;
};

struct ParallelOperation {
    static constexpr bool requires_parallel = true;
};

struct BasicOperation {};

} // namespace test_storage

// Bring namespaces into scope for tests
using namespace fem::numeric;
using namespace fem::numeric::traits;

// Test fixture
class StorageTraitsTest : public ::testing::Test {
protected:
    // No additional setup needed
};

// Tests for is_storage trait
TEST_F(StorageTraitsTest, IsStorageDetection) {
    // Positive cases
    EXPECT_TRUE((is_storage_v<DynamicStorage<float>>));
    EXPECT_TRUE((is_storage_v<StaticStorage<int, 10>>));
    EXPECT_TRUE((is_storage_v<AlignedStorage<double, 32>>));
    EXPECT_TRUE((is_storage_v<test_storage::ViewStorage<int>>));
    EXPECT_TRUE((is_storage_v<test_storage::SparseStorage<float>>));

    // Negative cases
    EXPECT_FALSE((is_storage_v<test_storage::NotAStorage>));
    EXPECT_FALSE((is_storage_v<int>));
    EXPECT_FALSE((is_storage_v<std::vector<int>>));
}

// Tests for specific storage type detection
TEST_F(StorageTraitsTest, SpecificStorageTypeDetection) {
    EXPECT_TRUE((is_dynamic_storage_v<DynamicStorage<float>>));
    EXPECT_FALSE((is_dynamic_storage_v<StaticStorage<int, 10>>));

    EXPECT_TRUE((is_aligned_storage_v<AlignedStorage<double, 32>>));
    EXPECT_FALSE((is_aligned_storage_v<DynamicStorage<float>>));
}

// Tests for storage_value_type extraction
TEST_F(StorageTraitsTest, StorageValueTypeExtraction) {
    static_assert(std::is_same_v<storage_value_type_t<DynamicStorage<float>>, float>);
    static_assert(std::is_same_v<storage_value_type_t<StaticStorage<int, 10>>, int>);
    static_assert(std::is_same_v<storage_value_type_t<AlignedStorage<double, 32>>, double>);
    static_assert(std::is_same_v<storage_value_type_t<test_storage::NotAStorage>, void>);
}

// Tests for storage_capacity
TEST_F(StorageTraitsTest, StorageCapacity) {
    // Static storage capacity
    using StaticStore = StaticStorage<int, 100>;
    EXPECT_EQ((storage_capacity<StaticStore>::static_capacity), 100u);
    EXPECT_TRUE((storage_capacity<StaticStore>::is_static));

    StaticStore static_storage;
    EXPECT_EQ((storage_capacity<StaticStore>::capacity(static_storage)), 100u);

    // Dynamic storage capacity
    using DynamicStore = DynamicStorage<float>;
    EXPECT_EQ((storage_capacity<DynamicStore>::static_capacity), 0u);
    EXPECT_FALSE((storage_capacity<DynamicStore>::is_static));

    DynamicStore dynamic_storage;
    EXPECT_EQ((storage_capacity<DynamicStore>::capacity(dynamic_storage)), 0u);
}

// Tests for alignment_traits
TEST_F(StorageTraitsTest, AlignmentTraits) {
    using AlignedStore32 = AlignedStorage<float, 32>;
    using AlignedStore64 = AlignedStorage<double, 64>;
    using AlignedStore4096 = AlignedStorage<int, 4096>;

    // Required alignment
    EXPECT_EQ((alignment_traits<AlignedStore32>::required_alignment), 32u);
    EXPECT_EQ((alignment_traits<AlignedStore64>::required_alignment), 64u);

    // SIMD alignment check
    EXPECT_TRUE((alignment_traits<AlignedStore32>::is_simd_aligned));
    EXPECT_TRUE((alignment_traits<AlignedStore64>::is_simd_aligned));
    EXPECT_FALSE((alignment_traits<DynamicStorage<char>>::is_simd_aligned));

    // Cache alignment check
    EXPECT_FALSE((alignment_traits<AlignedStore32>::is_cache_aligned));
    EXPECT_TRUE((alignment_traits<AlignedStore64>::is_cache_aligned));

    // Page alignment check
    EXPECT_TRUE((alignment_traits<AlignedStore4096>::is_page_aligned));
    EXPECT_FALSE((alignment_traits<AlignedStore32>::is_page_aligned));
}

// Tests for growth_strategy
TEST_F(StorageTraitsTest, GrowthStrategy) {
    using StaticStore = StaticStorage<int, 50>;
    using DynamicStore = DynamicStorage<float>;

    // Static storage has no growth
    EXPECT_EQ((growth_strategy<StaticStore>::strategy),
              (growth_strategy<StaticStore>::None));
    EXPECT_EQ((growth_strategy<StaticStore>::next_capacity(100)), 100u);

    // Dynamic storage has geometric growth
    EXPECT_EQ((growth_strategy<DynamicStore>::strategy),
              (growth_strategy<DynamicStore>::Geometric));
    EXPECT_EQ((growth_strategy<DynamicStore>::growth_factor), 2.0);
    EXPECT_EQ((growth_strategy<DynamicStore>::next_capacity(100)), 200u);
}

// Tests for storage_properties
TEST_F(StorageTraitsTest, StorageProperties) {
    using DynamicProps = storage_properties<DynamicStorage<float>>;
    using StaticProps = storage_properties<StaticStorage<int, 100>>;
    using AlignedProps = storage_properties<AlignedStorage<double, 64>>;

    // Category classification
    EXPECT_EQ(DynamicProps::category, StorageCategory::Dynamic);
    EXPECT_EQ(StaticProps::category, StorageCategory::Static);
    EXPECT_EQ(AlignedProps::category, StorageCategory::Aligned);

    // Contiguity
    EXPECT_TRUE(DynamicProps::is_contiguous);
    EXPECT_TRUE(StaticProps::is_contiguous);
    EXPECT_TRUE(AlignedProps::is_contiguous);

    // Resizability
    EXPECT_TRUE(DynamicProps::is_resizable);
    EXPECT_FALSE(StaticProps::is_resizable);

    // SIMD readiness
    EXPECT_FALSE(DynamicProps::simd_ready);
    EXPECT_FALSE(StaticProps::simd_ready);
    EXPECT_TRUE(AlignedProps::simd_ready);

    // Ownership model
    EXPECT_EQ(DynamicProps::ownership, OwnershipModel::Owning);
    EXPECT_EQ(StaticProps::ownership, OwnershipModel::Owning);
}

// Tests for operation suitability
TEST_F(StorageTraitsTest, OperationSuitability) {
    using AlignedStore = AlignedStorage<float, 32>;
    using RegularStore = DynamicStorage<float>;

    // SIMD operations need aligned storage
    EXPECT_TRUE((is_storage_suitable_for_operation_v<AlignedStore, test_storage::SIMDOperation>));
    EXPECT_FALSE((is_storage_suitable_for_operation_v<RegularStore, test_storage::SIMDOperation>));

    // Parallel operations need contiguous storage
    EXPECT_TRUE((is_storage_suitable_for_operation_v<AlignedStore, test_storage::ParallelOperation>));
    EXPECT_TRUE((is_storage_suitable_for_operation_v<RegularStore, test_storage::ParallelOperation>));

    // Basic operations work with any storage
    EXPECT_TRUE((is_storage_suitable_for_operation_v<AlignedStore, test_storage::BasicOperation>));
    EXPECT_TRUE((is_storage_suitable_for_operation_v<RegularStore, test_storage::BasicOperation>));
}

// Tests for memory_footprint
TEST_F(StorageTraitsTest, MemoryFootprint) {
    using StaticStore = StaticStorage<int, 100>;
    using DynamicStore = DynamicStorage<double>;

    // Static storage has no overhead
    EXPECT_EQ((memory_footprint<StaticStore>::overhead), 0u);
    EXPECT_EQ((memory_footprint<StaticStore>::element_size), sizeof(int));
    EXPECT_EQ((memory_footprint<StaticStore>::bytes_required(100)), 100u * sizeof(int));

    // Dynamic storage has pointer overhead
    EXPECT_GT((memory_footprint<DynamicStore>::overhead), 0u);
    EXPECT_EQ((memory_footprint<DynamicStore>::element_size), sizeof(double));

    // Efficiency calculation
    double static_eff = memory_footprint<StaticStore>::efficiency(100);
    EXPECT_DOUBLE_EQ(static_eff, 1.0);  // No overhead

    double dynamic_eff = memory_footprint<DynamicStore>::efficiency(100);
    EXPECT_LT(dynamic_eff, 1.0);  // Has overhead
    EXPECT_GT(dynamic_eff, 0.9);  // But still efficient for large n
}

// Tests for storage conversion compatibility
TEST_F(StorageTraitsTest, StorageConversion) {
    using Static10 = StaticStorage<int, 10>;
    using Static20 = StaticStorage<int, 20>;
    using DynamicInt = DynamicStorage<int>;
    using DynamicFloat = DynamicStorage<float>;

    // Static to static with compatible sizes
    EXPECT_TRUE((can_convert_storage_v<Static10, Static20>));
    EXPECT_FALSE((can_convert_storage_v<Static20, Static10>));

    // Dynamic to static not allowed
    EXPECT_FALSE((can_convert_storage_v<DynamicInt, Static10>));

    // Dynamic to dynamic with compatible types
    EXPECT_TRUE((can_convert_storage_v<DynamicInt, DynamicFloat>));

    // Incompatible value types
    struct CustomType {};
    using DynamicCustom = DynamicStorage<CustomType>;
    EXPECT_FALSE((can_convert_storage_v<DynamicInt, DynamicCustom>));
}

// Tests for optimal_storage_selector
TEST_F(StorageTraitsTest, OptimalStorageSelector) {
    // Small static size -> StaticStorage
    using SmallFixed = optimal_storage_t<int, 10, false, false>;
    static_assert(std::is_same_v<SmallFixed, StaticStorage<int, 10>>);

    // Large static size -> DynamicStorage
    using LargeFixed = optimal_storage_t<double, 1000, false, false>;
    static_assert(std::is_same_v<LargeFixed, DynamicStorage<double>>);

    // SIMD requirement -> AlignedStorage
    using SIMDStorage = optimal_storage_t<float, 0, true, false>;
    static_assert(std::is_same_v<SIMDStorage, AlignedStorage<float, 32>>);

    // Growth requirement -> DynamicStorage or AlignedStorage
    using GrowableStorage = optimal_storage_t<int, 10, false, true>;
    static_assert(std::is_same_v<GrowableStorage, DynamicStorage<int>>);

    using GrowableSIMD = optimal_storage_t<float, 10, true, true>;
    static_assert(std::is_same_v<GrowableSIMD, AlignedStorage<float, 32>>);
}

// Tests for storage_iterator_traits
TEST_F(StorageTraitsTest, StorageIteratorTraits) {
    using DynamicIter = storage_iterator_traits<DynamicStorage<int>>;
    using StaticIter = storage_iterator_traits<StaticStorage<float, 10>>;

    // Contiguous storage uses raw pointers
    static_assert(std::is_same_v<typename DynamicIter::iterator, int*>);
    static_assert(std::is_same_v<typename DynamicIter::const_iterator, const int*>);
    static_assert(std::is_same_v<typename StaticIter::iterator, float*>);

    // Random access availability
    EXPECT_TRUE(DynamicIter::has_random_access);
    EXPECT_TRUE(StaticIter::has_random_access);
}

// Tests for cache_hints
TEST_F(StorageTraitsTest, CacheHints) {
    using FloatStore = DynamicStorage<float>;
    using LargeStore = DynamicStorage<std::array<double, 8>>;

    // Elements per cache line
    constexpr size_t float_per_line = cache_hints<FloatStore>::elements_per_cache_line;
    EXPECT_EQ(float_per_line, 64u / sizeof(float));

    // Prefetch distance for contiguous storage
    EXPECT_GT((cache_hints<FloatStore>::prefetch_distance), 0u);

    // Streaming stores for large elements
    EXPECT_FALSE((cache_hints<FloatStore>::use_streaming_stores));
    EXPECT_TRUE((cache_hints<LargeStore>::use_streaming_stores));
}

// Tests for sparse storage detection
TEST_F(StorageTraitsTest, SparseStorageDetection) {
    EXPECT_TRUE((is_sparse_storage_v<test_storage::SparseStorage<float>>));
    EXPECT_FALSE((is_sparse_storage_v<DynamicStorage<float>>));
    EXPECT_FALSE((is_sparse_storage_v<StaticStorage<int, 10>>));
}

// Tests for pool allocation detection
TEST_F(StorageTraitsTest, PoolAllocationDetection) {
    EXPECT_TRUE((is_pool_allocated_v<test_storage::PoolStorage<int>>));
    EXPECT_FALSE((is_pool_allocated_v<DynamicStorage<int>>));
    EXPECT_FALSE((is_pool_allocated_v<StaticStorage<int, 10>>));
}

// Tests for device memory traits
TEST_F(StorageTraitsTest, DeviceMemoryTraits) {
    using CPUTraits = device_memory_traits<DynamicStorage<float>>;
    using DeviceTraits = device_memory_traits<test_storage::DeviceStorage<float>>;
    using UnifiedTraits = device_memory_traits<test_storage::UnifiedStorage<float>>;

    // CPU storage
    EXPECT_FALSE(CPUTraits::is_device_memory);
    EXPECT_FALSE(CPUTraits::is_unified_memory);
    EXPECT_FALSE(CPUTraits::requires_explicit_transfer);
    EXPECT_EQ(CPUTraits::device_type, Device::CPU);

    // Device storage
    EXPECT_TRUE(DeviceTraits::is_device_memory);
    EXPECT_FALSE(DeviceTraits::is_unified_memory);
    EXPECT_TRUE(DeviceTraits::requires_explicit_transfer);

    // Unified memory storage
    EXPECT_TRUE(UnifiedTraits::is_device_memory);
    EXPECT_TRUE(UnifiedTraits::is_unified_memory);
    EXPECT_FALSE(UnifiedTraits::requires_explicit_transfer);
}

// Compile-time tests using static_assert
namespace compile_time_tests {

    using namespace fem::numeric;
    using namespace fem::numeric::traits;

    // Test that trait results are constexpr
    constexpr bool test_is_storage = is_storage_v<DynamicStorage<int>>;
    static_assert(test_is_storage);

    constexpr bool test_alignment = alignment_traits<AlignedStorage<float, 32>>::is_simd_aligned;
    static_assert(test_alignment);

    constexpr size_t test_capacity = storage_capacity<StaticStorage<int, 42>>::static_capacity;
    static_assert(test_capacity == 42);

    constexpr StorageCategory test_category = storage_properties<DynamicStorage<float>>::category;
    static_assert(test_category == StorageCategory::Dynamic);

    // Test optimal storage selection at compile time
    using OptimalSmall = optimal_storage_t<char, 32, false, false>;
    static_assert(std::is_same_v<OptimalSmall, StaticStorage<char, 32>>);

    using OptimalSIMD = optimal_storage_t<float, 0, true, false>;
    static_assert(std::is_same_v<OptimalSIMD, AlignedStorage<float, 32>>);
}
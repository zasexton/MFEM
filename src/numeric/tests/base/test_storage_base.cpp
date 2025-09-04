// test_storage_base.cpp - Updated for refactored storage classes
#include <gtest/gtest.h>
#include <base/dual_base.h>
#include <base/storage_base.h>
#include <base/numeric_base.h>

#include <vector>
#include <memory>
#include <limits>
#include <span>

using namespace fem::numeric;

// ============================================================================
// DynamicStorage - Test core functionality
// ============================================================================

TEST(DynamicStorage, CompleteCoverage) {
    // Default constructor
    DynamicStorage<int> ds;
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_EQ(ds.capacity(), 0u);
    EXPECT_TRUE(ds.empty());  // Now using base class implementation

    // Size constructor
    DynamicStorage<int> ds1(3);
    EXPECT_EQ(ds1.size(), 3u);

    // Size+value constructor
    DynamicStorage<int> ds2(3, 42);
    EXPECT_EQ(ds2.size(), 3u);
    EXPECT_EQ(ds2[0], 42);

    // Iterator constructor
    std::vector<int> v{1, 2, 3};
    DynamicStorage<int> ds3(v.begin(), v.end());
    EXPECT_EQ(ds3.size(), 3u);
    EXPECT_EQ(ds3[1], 2);

    // Initializer list
    DynamicStorage<int> ds4{1, 2, 3};
    EXPECT_EQ(ds4.size(), 3u);

    // Copy constructor
    DynamicStorage<int> ds5(ds4);
    EXPECT_EQ(ds5.size(), ds4.size());

    // Move constructor
    DynamicStorage<int> ds6(std::move(ds5));
    EXPECT_EQ(ds6.size(), 3u);

    // Copy assignment
    ds = ds4;
    EXPECT_EQ(ds.size(), ds4.size());

    // Move assignment
    ds1 = std::move(ds6);
    EXPECT_EQ(ds1.size(), 3u);

    // Test data access
    int* data = ds2.data();
    ASSERT_NE(data, nullptr);

    // Test operator[] (now in base class)
    int& ref = ds2[0];
    ref = 10;
    EXPECT_EQ(ds2[0], 10);

    // Test at() method (new in base class)
    EXPECT_EQ(ds2.at(0), 10);
    EXPECT_THROW(ds2.at(100), std::out_of_range);

    // Test resize
    ds2.resize(2);
    EXPECT_EQ(ds2.size(), 2u);
    ds2.resize(5, 99);
    EXPECT_EQ(ds2.size(), 5u);
    EXPECT_EQ(ds2[4], 99);

    // Test reserve
    ds2.reserve(10);
    EXPECT_GE(ds2.capacity(), 10u);

    // Test layout/device/contiguous (now with default implementations)
    EXPECT_EQ(ds2.layout(), Layout::RowMajor);
    EXPECT_EQ(ds2.device(), Device::CPU);
    EXPECT_TRUE(ds2.is_contiguous());

    // Test clone
    auto clone = ds2.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), ds2.size());

    // Test fill (now in base class)
    ds2.fill(77);
    EXPECT_EQ(ds2[0], 77);
    EXPECT_EQ(ds2[4], 77);

    // Test swap with same type
    DynamicStorage<int> other(2, 88);
    StorageBase<int>& b1 = ds2;
    StorageBase<int>& b2 = other;
    size_t orig_size = ds2.size();
    b1.swap(b2);
    EXPECT_EQ(ds2.size(), 2u);
    EXPECT_EQ(other.size(), orig_size);

    // Test clear
    ds2.clear();
    EXPECT_TRUE(ds2.empty());
    EXPECT_EQ(ds2.size(), 0u);

    // Test span() method (now in base class)
    DynamicStorage<int> ds_span(5, 100);
    auto span = ds_span.span();
    EXPECT_EQ(span.size(), 5u);
    EXPECT_EQ(span[0], 100);

    // Test as_bytes() method (now in base class)
    auto bytes = ds_span.as_bytes();
    EXPECT_EQ(bytes.size(), 5 * sizeof(int));

    // Test as_writable_bytes()
    auto writable_bytes = ds_span.as_writable_bytes();
    EXPECT_EQ(writable_bytes.size(), 5 * sizeof(int));
}

TEST(DynamicStorage, ConstMethods) {
    const DynamicStorage<int> cds(2, 50);

    // Test const methods
    EXPECT_EQ(cds.size(), 2u);
    EXPECT_GE(cds.capacity(), 2u);
    EXPECT_FALSE(cds.empty());

    const int* cdata = cds.data();
    ASSERT_NE(cdata, nullptr);

    // Test const operator[]
    const int& cref = cds[0];
    EXPECT_EQ(cref, 50);

    // Test const at()
    EXPECT_EQ(cds.at(1), 50);
    EXPECT_THROW(cds.at(10), std::out_of_range);

    // Test const span
    auto cspan = cds.span();
    EXPECT_EQ(cspan.size(), 2u);

    // Test const as_bytes
    auto cbytes = cds.as_bytes();
    EXPECT_EQ(cbytes.size(), 2 * sizeof(int));

    EXPECT_EQ(cds.layout(), Layout::RowMajor);
    EXPECT_EQ(cds.device(), Device::CPU);
    EXPECT_TRUE(cds.is_contiguous());

    auto cclone = cds.clone();
    ASSERT_NE(cclone, nullptr);
}

// ============================================================================
// StaticStorage - Test core functionality
// ============================================================================

TEST(StaticStorage, CompleteCoverage) {
    // Default constructor
    StaticStorage<int, 10> ss;
    EXPECT_TRUE(ss.empty());
    EXPECT_EQ(ss.size(), 0u);
    EXPECT_EQ(ss.capacity(), 10u);

    // Size constructor
    StaticStorage<int, 10> ss1(5);
    EXPECT_EQ(ss1.size(), 5u);

    // Size+value constructor
    StaticStorage<int, 10> ss2(5, 42);
    EXPECT_EQ(ss2.size(), 5u);
    EXPECT_EQ(ss2[0], 42);

    // Test data access
    int* data = ss2.data();
    ASSERT_NE(data, nullptr);

    // Test operator[] (now in base class)
    int& ref = ss2[0];
    ref = 10;
    EXPECT_EQ(ss2[0], 10);

    // Test at() method
    EXPECT_EQ(ss2.at(0), 10);
    EXPECT_THROW(ss2.at(10), std::out_of_range);

    // Test resize
    ss2.resize(3);
    EXPECT_EQ(ss2.size(), 3u);
    ss2.resize(7, 99);
    EXPECT_EQ(ss2.size(), 7u);
    EXPECT_EQ(ss2[6], 99);

    // Test reserve (no-op for static storage)
    ss2.reserve(8);
    EXPECT_EQ(ss2.capacity(), 10u);

    // Test fill (now in base class)
    ss2.fill(77);
    for (size_t i = 0; i < ss2.size(); ++i) {
        EXPECT_EQ(ss2[i], 77);
    }

    // Test swap with same type
    StaticStorage<int, 10> other(2, 88);
    StorageBase<int>& b1 = ss2;
    StorageBase<int>& b2 = other;
    size_t orig_size = ss2.size();
    b1.swap(b2);
    EXPECT_EQ(ss2.size(), 2u);
    EXPECT_EQ(other.size(), orig_size);

    // Test clear
    ss2.clear();
    EXPECT_TRUE(ss2.empty());
    EXPECT_EQ(ss2.size(), 0u);
    EXPECT_EQ(ss2.capacity(), 10u);  // Capacity unchanged

    // Test span() method
    StaticStorage<int, 10> ss_span(5, 100);
    auto span = ss_span.span();
    EXPECT_EQ(span.size(), 5u);
    EXPECT_EQ(span[0], 100);

    // Test layout/device/contiguous
    EXPECT_EQ(ss2.layout(), Layout::RowMajor);
    EXPECT_EQ(ss2.device(), Device::CPU);
    EXPECT_TRUE(ss2.is_contiguous());

    // Test clone
    auto clone = ss_span.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), ss_span.size());
}

TEST(StaticStorage, ErrorConditions) {
    // Test construction beyond capacity
    EXPECT_THROW((StaticStorage<int, 5>(6)), std::length_error);
    EXPECT_THROW((StaticStorage<int, 5>(6, 10)), std::length_error);

    // Test resize beyond capacity
    StaticStorage<int, 5> err(3);
    EXPECT_THROW(err.resize(6), std::length_error);
    EXPECT_THROW(err.resize(6, 10), std::length_error);

    // Test reserve beyond capacity
    EXPECT_THROW(err.reserve(6), std::length_error);
}

TEST(StaticStorage, ConstMethods) {
    const StaticStorage<int, 10> css(3, 50);

    EXPECT_EQ(css.size(), 3u);
    EXPECT_EQ(css.capacity(), 10u);
    EXPECT_FALSE(css.empty());

    const int* cdata = css.data();
    ASSERT_NE(cdata, nullptr);

    const int& cref = css[0];
    EXPECT_EQ(cref, 50);

    EXPECT_EQ(css.at(1), 50);
    EXPECT_THROW(css.at(10), std::out_of_range);

    auto cspan = css.span();
    EXPECT_EQ(cspan.size(), 3u);

    auto cbytes = css.as_bytes();
    EXPECT_EQ(cbytes.size(), 3 * sizeof(int));

    EXPECT_EQ(css.layout(), Layout::RowMajor);
    EXPECT_EQ(css.device(), Device::CPU);
    EXPECT_TRUE(css.is_contiguous());

    auto cclone = css.clone();
    ASSERT_NE(cclone, nullptr);
}

// ============================================================================
// AlignedStorage - Test core functionality
// ============================================================================

TEST(AlignedStorage, CompleteCoverage) {
    // Default constructor
    AlignedStorage<int, 32> as;
    EXPECT_TRUE(as.empty());
    EXPECT_EQ(as.size(), 0u);

    // Test alignment constant (using typedef to avoid macro issues with comma)
    using AlignedInt32 = AlignedStorage<int, 32>;
    EXPECT_EQ(AlignedInt32::alignment, 32u);

    // Size constructor
    AlignedStorage<int, 32> as1(3);
    EXPECT_EQ(as1.size(), 3u);

    // Size+value constructor
    AlignedStorage<int, 32> as2(5, 42);
    EXPECT_EQ(as2.size(), 5u);
    EXPECT_EQ(as2[0], 42);

    // Copy constructor
    AlignedStorage<int, 32> as3(as2);
    EXPECT_EQ(as3.size(), as2.size());
    EXPECT_EQ(as3[0], as2[0]);

    // Move constructor
    AlignedStorage<int, 32> as4(std::move(as3));
    EXPECT_EQ(as4.size(), 5u);

    // Copy assignment with data
    as1 = as2;
    EXPECT_EQ(as1.size(), as2.size());

    // Move assignment
    as = std::move(as4);
    EXPECT_EQ(as.size(), 5u);

    // Self-assignment (should be no-op)
    as2 = as2;
    EXPECT_EQ(as2.size(), 5u);

    // Test data access and alignment
    int* data = as2.data();
    ASSERT_NE(data, nullptr);
    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 32, 0u);

    // Test operator[]
    int& ref = as2[0];
    ref = 10;
    EXPECT_EQ(as2[0], 10);

    // Test at()
    EXPECT_EQ(as2.at(0), 10);
    EXPECT_THROW(as2.at(100), std::out_of_range);

    // Test all resize paths
    as2.resize(3);      // shrink
    EXPECT_EQ(as2.size(), 3u);

    as2.resize(5);      // grow with default construct
    EXPECT_EQ(as2.size(), 5u);

    as2.resize(8, 99);  // grow with value
    EXPECT_EQ(as2.size(), 8u);
    EXPECT_EQ(as2[7], 99);

    as2.resize(4, 88);  // shrink (value ignored)
    EXPECT_EQ(as2.size(), 4u);

    // Test reserve
    as2.reserve(20);
    EXPECT_GE(as2.capacity(), 20u);
    EXPECT_EQ(as2.size(), 4u);  // Size unchanged

    // Test fill
    as2.fill(77);
    for (size_t i = 0; i < as2.size(); ++i) {
        EXPECT_EQ(as2[i], 77);
    }

    // Test swap with same type
    AlignedStorage<int, 32> other(2, 88);
    StorageBase<int>& b1 = as2;
    StorageBase<int>& b2 = other;
    size_t orig_size = as2.size();
    b1.swap(b2);
    EXPECT_EQ(as2.size(), 2u);
    EXPECT_EQ(other.size(), orig_size);

    // Test clear
    as2.clear();
    EXPECT_TRUE(as2.empty());
    EXPECT_EQ(as2.size(), 0u);

    // Test span
    AlignedStorage<int, 32> as_span(5, 100);
    auto span = as_span.span();
    EXPECT_EQ(span.size(), 5u);
    EXPECT_EQ(span[0], 100);

    // Test layout/device/contiguous
    EXPECT_EQ(as2.layout(), Layout::RowMajor);
    EXPECT_EQ(as2.device(), Device::CPU);
    EXPECT_TRUE(as2.is_contiguous());

    // Test clone
    auto clone = as_span.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), as_span.size());
}

TEST(AlignedStorage, EmptyToEmpty) {
    // Test copy assignment from empty to non-empty
    AlignedStorage<int, 32> empty;
    AlignedStorage<int, 32> nonempty(2, 10);
    nonempty = empty;
    EXPECT_TRUE(nonempty.empty());
    EXPECT_EQ(nonempty.data(), nullptr);

    // Test copy assignment from non-empty to empty
    AlignedStorage<int, 32> empty2;
    AlignedStorage<int, 32> nonempty2(3, 20);
    empty2 = nonempty2;
    EXPECT_FALSE(empty2.empty());
    EXPECT_EQ(empty2.size(), 3u);
    EXPECT_EQ(empty2[0], 20);
}

TEST(AlignedStorage, ConstMethods) {
    const AlignedStorage<int, 32> cas(3, 50);

    EXPECT_EQ(cas.size(), 3u);
    EXPECT_GE(cas.capacity(), 3u);
    EXPECT_FALSE(cas.empty());

    const int* cdata = cas.data();
    ASSERT_NE(cdata, nullptr);
    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(cdata) % 32, 0u);

    const int& cref = cas[0];
    EXPECT_EQ(cref, 50);

    EXPECT_EQ(cas.at(1), 50);
    EXPECT_THROW(cas.at(10), std::out_of_range);

    auto cspan = cas.span();
    EXPECT_EQ(cspan.size(), 3u);

    auto cbytes = cas.as_bytes();
    EXPECT_EQ(cbytes.size(), 3 * sizeof(int));

    EXPECT_EQ(cas.layout(), Layout::RowMajor);
    EXPECT_EQ(cas.device(), Device::CPU);
    EXPECT_TRUE(cas.is_contiguous());

    auto cclone = cas.clone();
    ASSERT_NE(cclone, nullptr);
}

// ============================================================================
// Cross-type swap errors - Updated error type
// ============================================================================

TEST(StorageSwap, CrossTypeErrors) {
    DynamicStorage<int> ds(2, 1);
    StaticStorage<int, 10> ss(2, 2);
    AlignedStorage<int, 32> as(2, 3);

    StorageBase<int>& bds = ds;
    StorageBase<int>& bss = ss;
    StorageBase<int>& bas = as;

    // Now throws std::logic_error instead of std::runtime_error
    EXPECT_THROW(bds.swap(bss), std::logic_error);
    EXPECT_THROW(bds.swap(bas), std::logic_error);
    EXPECT_THROW(bss.swap(bas), std::logic_error);
}

// ============================================================================
// Base class shared method tests
// ============================================================================

TEST(StorageBase, SharedMethods) {
    // Test that all storage types use the same base implementations
    DynamicStorage<double> ds(5, 1.5);
    StaticStorage<double, 10> ss(5, 2.5);
    AlignedStorage<double, 64> as(5, 3.5);

    // Test empty() - should be false for all
    EXPECT_FALSE(ds.empty());
    EXPECT_FALSE(ss.empty());
    EXPECT_FALSE(as.empty());

    // Clear and test empty() - should be true for all
    ds.clear();
    ss.clear();
    as.clear();
    EXPECT_TRUE(ds.empty());
    EXPECT_TRUE(ss.empty());
    EXPECT_TRUE(as.empty());

    // Test fill() on all types
    ds.resize(3);
    ss.resize(3);
    as.resize(3);

    ds.fill(10.0);
    ss.fill(10.0);
    as.fill(10.0);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(ds[i], 10.0);
        EXPECT_EQ(ss[i], 10.0);
        EXPECT_EQ(as[i], 10.0);
    }

    // Test at() bounds checking on all types
    EXPECT_THROW(ds.at(10), std::out_of_range);
    EXPECT_THROW(ss.at(10), std::out_of_range);
    EXPECT_THROW(as.at(10), std::out_of_range);

    // Test span() on all types
    auto ds_span = ds.span();
    auto ss_span = ss.span();
    auto as_span = as.span();

    EXPECT_EQ(ds_span.size(), 3u);
    EXPECT_EQ(ss_span.size(), 3u);
    EXPECT_EQ(as_span.size(), 3u);

    // Test as_bytes() on all types
    auto ds_bytes = ds.as_bytes();
    auto ss_bytes = ss.as_bytes();
    auto as_bytes = as.as_bytes();

    EXPECT_EQ(ds_bytes.size(), 3 * sizeof(double));
    EXPECT_EQ(ss_bytes.size(), 3 * sizeof(double));
    EXPECT_EQ(as_bytes.size(), 3 * sizeof(double));
}

// ============================================================================
// Allocation failure test
// ============================================================================
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif

TEST(AlignedStorage, AllocationFailure) {
    // Try to allocate an unreasonably large amount
    try {
        size_t huge = std::numeric_limits<size_t>::max() / sizeof(int) / 2;
        AlignedStorage<int, 32> as(huge);
        // If we get here, the system has a lot of memory
        SUCCEED();
    } catch (const std::bad_alloc&) {
        // Expected behavior
        SUCCEED();
    }
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// ============================================================================
// Different alignment sizes
// ============================================================================

TEST(AlignedStorage, DifferentAlignments) {
    // Test different alignment values
    AlignedStorage<float, 16> as16(10);
    AlignedStorage<float, 32> as32(10);
    AlignedStorage<float, 64> as64(10);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(as16.data()) % 16, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(as32.data()) % 32, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(as64.data()) % 64, 0u);

    // Verify they all work the same
    as16.fill(1.0f);
    as32.fill(2.0f);
    as64.fill(3.0f);

    EXPECT_EQ(as16[5], 1.0f);
    EXPECT_EQ(as32[5], 2.0f);
    EXPECT_EQ(as64[5], 3.0f);
}

// Add these tests to test_storage_base.cpp after the existing tests

// ============================================================================
// Storage with Dual Numbers - Test composite type support
// ============================================================================


using namespace fem::numeric::autodiff;

// Test DynamicStorage with dual numbers
TEST(DynamicStorage, DualNumbers) {
    using Dual = DualBase<double, 3>;  // Dual numbers with 3 derivatives

    // Default constructor
    DynamicStorage<Dual> ds;
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_TRUE(ds.empty());

    // Size constructor - should default construct dual numbers
    DynamicStorage<Dual> ds1(5);
    EXPECT_EQ(ds1.size(), 5u);
    EXPECT_EQ(ds1[0].value(), 0.0);  // Default constructed
    EXPECT_EQ(ds1[0].derivative(0), 0.0);

    // Size+value constructor with a dual number
    Dual dual_val(2.5);
    dual_val.seed(0, 1.0);  // Set first derivative to 1
    DynamicStorage<Dual> ds2(3, dual_val);
    EXPECT_EQ(ds2.size(), 3u);
    EXPECT_EQ(ds2[0].value(), 2.5);
    EXPECT_EQ(ds2[0].derivative(0), 1.0);
    EXPECT_EQ(ds2[1].value(), 2.5);
    EXPECT_EQ(ds2[1].derivative(0), 1.0);

    // Initializer list
    Dual d1(1.0), d2(2.0), d3(3.0);
    d1.seed(0, 1.0);
    d2.seed(1, 1.0);
    d3.seed(2, 1.0);
    DynamicStorage<Dual> ds3{d1, d2, d3};
    EXPECT_EQ(ds3.size(), 3u);
    EXPECT_EQ(ds3[0].value(), 1.0);
    EXPECT_EQ(ds3[0].derivative(0), 1.0);
    EXPECT_EQ(ds3[1].value(), 2.0);
    EXPECT_EQ(ds3[1].derivative(1), 1.0);

    // Copy constructor
    DynamicStorage<Dual> ds4(ds3);
    EXPECT_EQ(ds4.size(), ds3.size());
    EXPECT_EQ(ds4[0].value(), ds3[0].value());
    EXPECT_EQ(ds4[0].derivative(0), ds3[0].derivative(0));

    // Move constructor
    DynamicStorage<Dual> ds5(std::move(ds4));
    EXPECT_EQ(ds5.size(), 3u);
    EXPECT_EQ(ds5[0].value(), 1.0);

    // Test operator[] access
    ds2[1] = Dual(5.0);
    EXPECT_EQ(ds2[1].value(), 5.0);

    // Test at() with bounds checking
    EXPECT_EQ(ds2.at(0).value(), 2.5);
    EXPECT_THROW(ds2.at(10), std::out_of_range);

    // Test resize with dual numbers
    ds2.resize(5);
    EXPECT_EQ(ds2.size(), 5u);
    EXPECT_EQ(ds2[3].value(), 0.0);  // New elements default constructed

    ds2.resize(7, Dual(10.0));
    EXPECT_EQ(ds2.size(), 7u);
    EXPECT_EQ(ds2[6].value(), 10.0);

    // Test fill with dual number
    Dual fill_val(7.7);
    fill_val.seed(1, 2.0);
    ds2.fill(fill_val);
    for (size_t i = 0; i < ds2.size(); ++i) {
        EXPECT_EQ(ds2[i].value(), 7.7);
        EXPECT_EQ(ds2[i].derivative(1), 2.0);
    }

    // Test swap
    DynamicStorage<Dual> other(2);
    other[0] = Dual(100.0);
    StorageBase<Dual>& b1 = ds2;
    StorageBase<Dual>& b2 = other;
    size_t orig_size = ds2.size();
    b1.swap(b2);
    EXPECT_EQ(ds2.size(), 2u);
    EXPECT_EQ(other.size(), orig_size);
    EXPECT_EQ(ds2[0].value(), 100.0);

    // Test clone
    auto clone = ds2.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), ds2.size());
    EXPECT_EQ((*clone)[0].value(), ds2[0].value());

    // Test clear
    ds2.clear();
    EXPECT_TRUE(ds2.empty());

    // Note: as_bytes() won't work for Dual (not trivially copyable)
    // This is expected and correct behavior
}

// Test StaticStorage with dual numbers
TEST(StaticStorage, DualNumbers) {
    using Dual = DualBase<double, 2>;  // Dual numbers with 2 derivatives

    // Default constructor
    StaticStorage<Dual, 10> ss;
    EXPECT_TRUE(ss.empty());
    EXPECT_EQ(ss.capacity(), 10u);

    // Size constructor
    StaticStorage<Dual, 10> ss1(5);
    EXPECT_EQ(ss1.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ss1[i].value(), 0.0);
        EXPECT_EQ(ss1[i].derivative(0), 0.0);
        EXPECT_EQ(ss1[i].derivative(1), 0.0);
    }

    // Size+value constructor
    Dual val(3.14);
    val.seed(0, 1.0);
    StaticStorage<Dual, 10> ss2(3, val);
    EXPECT_EQ(ss2.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(ss2[i].value(), 3.14);
        EXPECT_EQ(ss2[i].derivative(0), 1.0);
    }

    // Test resize
    ss2.resize(5);
    EXPECT_EQ(ss2.size(), 5u);
    EXPECT_EQ(ss2[3].value(), 0.0);  // New elements default constructed

    Dual resize_val(2.71);
    resize_val.seed(1, 1.0);
    ss2.resize(7, resize_val);
    EXPECT_EQ(ss2.size(), 7u);
    EXPECT_EQ(ss2[6].value(), 2.71);
    EXPECT_EQ(ss2[6].derivative(1), 1.0);

    // Test fill
    Dual fill_val(5.5);
    fill_val.seed(0, 2.0);
    fill_val.derivative(1) = 3.0;
    ss2.fill(fill_val);
    for (size_t i = 0; i < ss2.size(); ++i) {
        EXPECT_EQ(ss2[i].value(), 5.5);
        EXPECT_EQ(ss2[i].derivative(0), 2.0);
        EXPECT_EQ(ss2[i].derivative(1), 3.0);
    }

    // Test capacity constraints
    EXPECT_THROW((StaticStorage<Dual, 5>(6)), std::length_error);
    EXPECT_THROW((StaticStorage<Dual, 5>(6, val)), std::length_error);

    // Test clone
    auto clone = ss2.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), ss2.size());
    EXPECT_EQ((*clone)[0].value(), ss2[0].value());
}

// Test AlignedStorage with dual numbers
TEST(AlignedStorage, DualNumbers) {
    using Dual = DualBase<double, 4>;  // Dual numbers with 4 derivatives

    // Default constructor
    AlignedStorage<Dual, 64> as;
    EXPECT_TRUE(as.empty());

    // Size constructor
    AlignedStorage<Dual, 64> as1(5);
    EXPECT_EQ(as1.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(as1[i].value(), 0.0);
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(as1[i].derivative(j), 0.0);
        }
    }

    // Size+value constructor
    Dual val(1.414);
    val.seed(2, 1.0);
    AlignedStorage<Dual, 64> as2(4, val);
    EXPECT_EQ(as2.size(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(as2[i].value(), 1.414);
        EXPECT_EQ(as2[i].derivative(2), 1.0);
    }

    // Test data alignment (should still be aligned even with complex type)
    Dual* data = as2.data();
    ASSERT_NE(data, nullptr);
    // The alignment should be at least as requested
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 64, 0u);

    // Copy constructor
    AlignedStorage<Dual, 64> as3(as2);
    EXPECT_EQ(as3.size(), as2.size());
    for (size_t i = 0; i < as3.size(); ++i) {
        EXPECT_EQ(as3[i].value(), as2[i].value());
        EXPECT_EQ(as3[i].derivative(2), as2[i].derivative(2));
    }

    // Move constructor
    AlignedStorage<Dual, 64> as4(std::move(as3));
    EXPECT_EQ(as4.size(), 4u);
    EXPECT_EQ(as4[0].value(), 1.414);

    // Test resize operations
    as2.resize(2);  // Shrink
    EXPECT_EQ(as2.size(), 2u);

    as2.resize(6);  // Grow with default construction
    EXPECT_EQ(as2.size(), 6u);
    EXPECT_EQ(as2[5].value(), 0.0);

    Dual grow_val(9.9);
    grow_val.seed(3, 5.0);
    as2.resize(8, grow_val);  // Grow with value
    EXPECT_EQ(as2.size(), 8u);
    EXPECT_EQ(as2[7].value(), 9.9);
    EXPECT_EQ(as2[7].derivative(3), 5.0);

    // Test fill
    Dual fill_val(3.33);
    for (size_t i = 0; i < 4; ++i) {
        fill_val.derivative(i) = static_cast<double>(i + 1);
    }
    as2.fill(fill_val);
    for (size_t i = 0; i < as2.size(); ++i) {
        EXPECT_EQ(as2[i].value(), 3.33);
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(as2[i].derivative(j), static_cast<double>(j + 1));
        }
    }

    // Test reserve
    as2.reserve(20);
    EXPECT_GE(as2.capacity(), 20u);
    EXPECT_EQ(as2.size(), 8u);  // Size should not change

    // Test SIMD support (should be false for dual numbers)
    EXPECT_FALSE(as2.supports_simd());

    // Test swap
    AlignedStorage<Dual, 64> other(2);
    other[0] = Dual(111.0);
    other[0].seed(0, 1.0);

    StorageBase<Dual>& b1 = as2;
    StorageBase<Dual>& b2 = other;
    size_t orig_size = as2.size();
    b1.swap(b2);
    EXPECT_EQ(as2.size(), 2u);
    EXPECT_EQ(other.size(), orig_size);
    EXPECT_EQ(as2[0].value(), 111.0);
    EXPECT_EQ(as2[0].derivative(0), 1.0);

    // Test clone
    auto clone = as2.clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->size(), as2.size());
    EXPECT_EQ((*clone)[0].value(), as2[0].value());

    // Test clear
    as2.clear();
    EXPECT_TRUE(as2.empty());
    EXPECT_EQ(as2.size(), 0u);
}

// Test that different storage types with dual numbers can't be swapped
TEST(StorageSwap, DualNumberCrossTypeErrors) {
    using Dual = DualBase<double, 2>;

    DynamicStorage<Dual> ds(2);
    StaticStorage<Dual, 10> ss(2);
    AlignedStorage<Dual, 32> as(2);

    StorageBase<Dual>& bds = ds;
    StorageBase<Dual>& bss = ss;
    StorageBase<Dual>& bas = as;

    EXPECT_THROW(bds.swap(bss), std::logic_error);
    EXPECT_THROW(bds.swap(bas), std::logic_error);
    EXPECT_THROW(bss.swap(bas), std::logic_error);
}

// Test operations with dual numbers containing different derivative counts
TEST(DynamicStorage, DualNumbersVaryingDerivatives) {
    // Test with 1 derivative (common for single-variable differentiation)
    using Dual1 = DualBase<double, 1>;
    DynamicStorage<Dual1> ds1(3);
    ds1[0] = Dual1(1.0, 1.0);  // value=1, derivative=1
    ds1[1] = Dual1(2.0, 0.0);
    ds1[2] = Dual1(3.0, 0.0);

    EXPECT_EQ(ds1[0].value(), 1.0);
    EXPECT_EQ(ds1[0].derivative(0), 1.0);

    // Test with 10 derivatives (for problems with many variables)
    using Dual10 = DualBase<double, 10>;
    DynamicStorage<Dual10> ds10(2);
    Dual10 val(5.0);
    for (size_t i = 0; i < 10; ++i) {
        val.derivative(i) = static_cast<double>(i);
    }
    ds10.fill(val);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(ds10[0].derivative(i), static_cast<double>(i));
        EXPECT_EQ(ds10[1].derivative(i), static_cast<double>(i));
    }
}

// Test that storage optimization traits work correctly for dual numbers
TEST(StorageTraits, DualNumberOptimizations) {
    using Dual = DualBase<double, 3>;

    // Dual numbers should not support SIMD
    EXPECT_FALSE(storage_optimization_traits<Dual>::supports_simd);

    // Dual numbers should prefer alignment
    EXPECT_TRUE(storage_optimization_traits<Dual>::prefers_alignment);

    // Dual numbers are not trivially relocatable
    EXPECT_FALSE(storage_optimization_traits<Dual>::is_trivially_relocatable);

    // Dual numbers don't support fast fill
    EXPECT_FALSE(storage_optimization_traits<Dual>::supports_fast_fill);

    // Create storage and verify it respects these traits
    AlignedStorage<Dual, 32> as(5);
    EXPECT_FALSE(as.supports_simd());
}

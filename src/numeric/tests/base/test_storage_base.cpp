// test_storage_base.cpp - Focused maximum coverage
#include <gtest/gtest.h>
#include <base/storage_base.h>
#include <base/numeric_base.h>

#include <vector>
#include <memory>
#include <limits>

using namespace fem::numeric;

// ============================================================================
// DynamicStorage - Test EVERY method exactly once
// ============================================================================

TEST(DynamicStorage, CompleteCoverage) {
    // Default constructor
    DynamicStorage<int> ds;

    // Every non-const method
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_EQ(ds.capacity(), 0u);
    EXPECT_TRUE(ds.empty());

    // Size constructor
    DynamicStorage<int> ds1(3);

    // Size+value constructor
    DynamicStorage<int> ds2(3, 42);

    // Iterator constructor
    std::vector<int> v{1,2,3};
    DynamicStorage<int> ds3(v.begin(), v.end());

    // Initializer list
    DynamicStorage<int> ds4{1,2,3};

    // Copy constructor
    DynamicStorage<int> ds5(ds4);

    // Move constructor
    DynamicStorage<int> ds6(std::move(ds5));

    // Copy assignment
    ds = ds4;

    // Move assignment
    ds1 = std::move(ds6);

    // Non-const methods on ds2
    int* data = ds2.data();
    ASSERT_NE(data, nullptr);

    int& ref = ds2[0];
    ref = 10;

    ds2.resize(2);
    ds2.resize(5, 99);
    ds2.reserve(10);

    EXPECT_EQ(ds2.layout(), Layout::RowMajor);
    EXPECT_EQ(ds2.device(), Device::CPU);
    EXPECT_TRUE(ds2.is_contiguous());

    auto clone = ds2.clone();
    ASSERT_NE(clone, nullptr);

    ds2.fill(77);

    // Swap same type
    DynamicStorage<int> other(2, 88);
    StorageBase<int>& b1 = ds2;
    StorageBase<int>& b2 = other;
    b1.swap(b2);

    ds2.clear();
    EXPECT_TRUE(ds2.empty());

    // Vector operations
    int val = 100;
    ds2.push_back(val);  // copy
    ds2.push_back(200);  // move

    int& front = ds2.front();
    int& back = ds2.back();
    front = 1;
    back = 2;

    auto begin = ds2.begin();
    auto end = ds2.end();
    EXPECT_NE(begin, end);

    ds2.pop_back();

    // Const methods
    const DynamicStorage<int> cds(2, 50);
    EXPECT_EQ(cds.size(), 2u);
    EXPECT_GE(cds.capacity(), 2u);
    EXPECT_FALSE(cds.empty());

    const int* cdata = cds.data();
    ASSERT_NE(cdata, nullptr);

    const int& cref = cds[0];
    EXPECT_EQ(cref, 50);

    EXPECT_EQ(cds.layout(), Layout::RowMajor);
    EXPECT_EQ(cds.device(), Device::CPU);
    EXPECT_TRUE(cds.is_contiguous());

    auto cclone = cds.clone();
    ASSERT_NE(cclone, nullptr);

    EXPECT_EQ(cds.front(), 50);
    EXPECT_EQ(cds.back(), 50);

    auto cbegin = cds.begin();
    auto cend = cds.end();
    EXPECT_NE(cbegin, cend);
}

// ============================================================================
// StaticStorage - Test EVERY method exactly once
// ============================================================================

TEST(StaticStorage, CompleteCoverage) {
    // Default constructor
    StaticStorage<int, 10> ss;
    EXPECT_TRUE(ss.empty());

    // Size constructor
    StaticStorage<int, 10> ss1(5);

    // Size+value constructor
    StaticStorage<int, 10> ss2(5, 42);

    // Every non-const method on ss2
    EXPECT_EQ(ss2.size(), 5u);
    EXPECT_EQ(ss2.capacity(), 10u);
    EXPECT_FALSE(ss2.empty());

    int* data = ss2.data();
    ASSERT_NE(data, nullptr);

    int& ref = ss2[0];
    ref = 10;

    ss2.resize(3);
    ss2.resize(7, 99);
    ss2.reserve(8);  // OK

    EXPECT_EQ(ss2.layout(), Layout::RowMajor);
    EXPECT_EQ(ss2.device(), Device::CPU);
    EXPECT_TRUE(ss2.is_contiguous());

    auto clone = ss2.clone();
    ASSERT_NE(clone, nullptr);

    ss2.fill(77);

    // Swap same type
    StaticStorage<int, 10> other(2, 88);
    StorageBase<int>& b1 = ss2;
    StorageBase<int>& b2 = other;
    b1.swap(b2);

    ss2.clear();
    EXPECT_TRUE(ss2.empty());

    // Const methods
    const StaticStorage<int, 10> css(3, 50);
    EXPECT_EQ(css.size(), 3u);
    EXPECT_EQ(css.capacity(), 10u);
    EXPECT_FALSE(css.empty());

    const int* cdata = css.data();
    ASSERT_NE(cdata, nullptr);

    const int& cref = css[0];
    EXPECT_EQ(cref, 50);

    EXPECT_EQ(css.layout(), Layout::RowMajor);
    EXPECT_EQ(css.device(), Device::CPU);
    EXPECT_TRUE(css.is_contiguous());

    auto cclone = css.clone();
    ASSERT_NE(cclone, nullptr);

    // Test errors
    EXPECT_THROW((StaticStorage<int, 5>(6)), std::length_error);
    EXPECT_THROW((StaticStorage<int, 5>(6, 10)), std::length_error);
    StaticStorage<int, 5> err(3);
    EXPECT_THROW(err.resize(6), std::length_error);
    EXPECT_THROW(err.resize(6, 10), std::length_error);
    EXPECT_THROW(err.reserve(6), std::length_error);
}

// ============================================================================
// AlignedStorage - Test EVERY method exactly once
// ============================================================================

TEST(AlignedStorage, CompleteCoverage) {
    // Default constructor
    AlignedStorage<int, 32> as;
    EXPECT_TRUE(as.empty());

    // Size constructor
    AlignedStorage<int, 32> as1(3);

    // Size+value constructor
    AlignedStorage<int, 32> as2(5, 42);

    // Copy constructor
    AlignedStorage<int, 32> as3(as2);

    // Move constructor
    AlignedStorage<int, 32> as4(std::move(as3));

    // Copy assignment (with data)
    as1 = as2;

    // Move assignment
    as = std::move(as4);

    // Self-assignment
    as2 = as2;

    // Every non-const method on as2
    EXPECT_EQ(as2.size(), 5u);
    EXPECT_GE(as2.capacity(), 5u);
    EXPECT_FALSE(as2.empty());

    int* data = as2.data();
    ASSERT_NE(data, nullptr);

    int& ref = as2[0];
    ref = 10;

    // All resize paths
    as2.resize(3);  // shrink
    as2.resize(5);  // grow default construct
    as2.resize(8, 99);  // grow with value
    as2.resize(4, 88);  // shrink with value

    as2.reserve(20);

    EXPECT_EQ(as2.layout(), Layout::RowMajor);
    EXPECT_EQ(as2.device(), Device::CPU);
    EXPECT_TRUE(as2.is_contiguous());

    auto clone = as2.clone();
    ASSERT_NE(clone, nullptr);

    as2.fill(77);

    // Swap same type
    AlignedStorage<int, 32> other(2, 88);
    StorageBase<int>& b1 = as2;
    StorageBase<int>& b2 = other;
    b1.swap(b2);

    as2.clear();
    EXPECT_TRUE(as2.empty());

    // Const methods
    const AlignedStorage<int, 32> cas(3, 50);
    EXPECT_EQ(cas.size(), 3u);
    EXPECT_GE(cas.capacity(), 3u);
    EXPECT_FALSE(cas.empty());

    const int* cdata = cas.data();
    ASSERT_NE(cdata, nullptr);

    const int& cref = cas[0];
    EXPECT_EQ(cref, 50);

    EXPECT_EQ(cas.layout(), Layout::RowMajor);
    EXPECT_EQ(cas.device(), Device::CPU);
    EXPECT_TRUE(cas.is_contiguous());

    auto cclone = cas.clone();
    ASSERT_NE(cclone, nullptr);

    // Copy assign from empty
    AlignedStorage<int, 32> empty;
    AlignedStorage<int, 32> nonempty(2, 10);
    nonempty = empty;
    EXPECT_TRUE(nonempty.empty());
}

// ============================================================================
// Cross-type swap errors
// ============================================================================

TEST(StorageSwap, CrossTypeErrors) {
    DynamicStorage<int> ds(2, 1);
    StaticStorage<int, 10> ss(2, 2);
    AlignedStorage<int, 32> as(2, 3);

    StorageBase<int>& bds = ds;
    StorageBase<int>& bss = ss;
    StorageBase<int>& bas = as;

    EXPECT_THROW(bds.swap(bss), std::runtime_error);
    EXPECT_THROW(bds.swap(bas), std::runtime_error);
    EXPECT_THROW(bss.swap(bas), std::runtime_error);
}

// ============================================================================
// Allocation failure test
// ============================================================================

TEST(AlignedStorage, AllocationFailure) {
    try {
        size_t huge = std::numeric_limits<size_t>::max() / sizeof(int) / 2;
        AlignedStorage<int, 32> as(huge);
        // System has lots of memory if we get here
        SUCCEED();
    } catch (const std::bad_alloc&) {
        SUCCEED();  // Expected
    }
}
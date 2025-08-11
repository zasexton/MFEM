#include <gtest/gtest.h>
#include <base/iterator_base.h>
#include <base/numeric_base.h>

#include <vector>
#include <array>
#include <complex>
#include <algorithm>
#include <limits>

using namespace fem::numeric;

// ============================================================================
// Iterator category & trait sanity
// ============================================================================

TEST(IteratorTraits, CategoriesAndNumericFlags) {
    using CI_d  = ContainerIterator<double>;
    using SI_d  = StridedIterator<double>;
    using CI_i  = ContainerIterator<int>;
    using CI_c  = ContainerIterator<std::complex<double>>;
    using CHK_d = CheckedIterator<CI_d>;

    // Categories
    static_assert(std::is_same_v<
            std::iterator_traits<CI_d>::iterator_category,
            std::random_access_iterator_tag>);
    static_assert(std::is_same_v<
            std::iterator_traits<SI_d>::iterator_category,
            std::random_access_iterator_tag>);

    // CheckedIterator should inherit iterator_category of underlying
    static_assert(std::is_same_v<
            std::iterator_traits<CHK_d>::iterator_category,
            std::random_access_iterator_tag>);

    // numeric_iterator_traits
    static_assert(numeric_iterator_traits<CI_d>::is_numeric);
    static_assert(!numeric_iterator_traits<CI_c>::is_numeric);          // complex not NumberLike
    static_assert(numeric_iterator_traits<CI_d>::is_ieee_compliant);     // double
    static_assert(!numeric_iterator_traits<CI_i>::is_ieee_compliant);    // int

    static_assert(numeric_iterator_traits<CI_d>::is_contiguous);
    static_assert(!numeric_iterator_traits<CI_d>::is_strided);

    static_assert(!numeric_iterator_traits<SI_d>::is_contiguous);
    static_assert(numeric_iterator_traits<SI_d>::is_strided);

    // CheckedIterator is neither contiguous nor strided by this simple trait rule
    static_assert(!numeric_iterator_traits<CHK_d>::is_contiguous);
    static_assert(!numeric_iterator_traits<CHK_d>::is_strided);

    SUCCEED();
}

// ============================================================================
// ContainerIterator tests
// ============================================================================

TEST(ContainerIterator, BasicOperations) {
    int a[5] = {10, 20, 30, 40, 50};

    ContainerIterator<int> it(a);
    EXPECT_EQ(*it, 10);
    EXPECT_EQ(it[2], 30);

    // ++ / --
    EXPECT_EQ(*(++it), 20);
    it++;
    EXPECT_EQ(*it, 30);
    EXPECT_EQ(*(it--), 30);
    EXPECT_EQ(*it, 20);
    EXPECT_EQ(*(--it), 10);

    // + / - / += / -=
    auto it2 = it + 3;
    EXPECT_EQ(*it2, 40);
    it2 = 4 + it;             // commutative overload
    EXPECT_EQ(*it2, 50);
    it2 -= 2;                 // now at 30
    EXPECT_EQ(*it2, 30);
    it2 += 1;                 // now at 40
    EXPECT_EQ(*it2, 40);

    // difference & comparisons
    EXPECT_EQ((it2 - it), 3);
    EXPECT_TRUE(it < it2);
    EXPECT_TRUE(it2 > it);
    EXPECT_TRUE(it <= it2);
    EXPECT_TRUE(it2 >= it);

    // base pointer
    EXPECT_EQ(it.base(), &a[0]);
    EXPECT_EQ((it + 4).base(), &a[4]);
}

TEST(ContainerIterator, ArrowOperatorWithStruct) {
    struct Foo { int x; double y; };
    Foo arr[2] = { {1, 2.5}, {3, 4.5} };

    ContainerIterator<Foo> it(arr);
    EXPECT_EQ(it->x, 1);
    EXPECT_DOUBLE_EQ(it->y, 2.5);
    ++it;
    EXPECT_EQ(it->x, 3);
    EXPECT_DOUBLE_EQ(it->y, 4.5);
}

// ============================================================================
// StridedIterator tests
// ============================================================================

TEST(StridedIterator, BasicOperations) {
    int a[10];
    for (int i = 0; i < 10; ++i) a[i] = i;

    StridedIterator<int> it(a, 2);  // even elements: 0,2,4,6,8
    EXPECT_EQ(*it, 0);
    ++it;
    EXPECT_EQ(*it, 2);
    it++;
    EXPECT_EQ(*it, 4);

    // indexing in element units (not raw pointer units)
    EXPECT_EQ(it[1], 6);     // 4 + 1*stride -> a[6]
    EXPECT_EQ(it[-1], 2);    // 4 - 1*stride -> a[2]

    // arithmetic
    auto it2 = it + 2;       // -> a[8]
    EXPECT_EQ(*it2, 8);
    auto it3 = 1 + it;       // -> a[6]
    EXPECT_EQ(*it3, 6);
    it2 -= 1;                // -> a[6]
    EXPECT_EQ(*it2, 6);

    // difference counts elements, not bytes
    EXPECT_EQ((it2 - it), 1);

    // comparisons
    EXPECT_TRUE(it < it2);
    EXPECT_TRUE(it2 > it);
    EXPECT_TRUE(it <= it2);
    EXPECT_TRUE(it2 >= it);

    // accessors
    EXPECT_EQ(it2.base(), &a[6]);
    EXPECT_EQ(it2.stride(), 2);
}

// ============================================================================
// MultiDimIterator tests
// ============================================================================

TEST(MultiDimIterator, RowMajor2x3Traversal) {
    // Buffer laid out row-major [2 x 3]:
    // row 0: 0,1,2
    // row 1: 3,4,5
    std::vector<int> buf = {0,1,2,3,4,5};
    using It = MultiDimIterator<int, 2>;
    std::array<size_t,2> shape   {2,3};
    std::array<size_t,2> strides {3,1};  // linear = i*3 + j

    It it(buf.data(), strides, shape);

    std::vector<int> seen;
    seen.reserve(buf.size());

    // indices progression sanity:
    // start: {0,0} -> 0
    // ++ -> {0,1} -> 1
    // ++ -> {0,2} -> 2
    // ++ -> {1,0} -> 3  (carry)
    EXPECT_EQ(it.indices()[0], 0u);
    EXPECT_EQ(it.indices()[1], 0u);
    EXPECT_EQ(*it, 0);
    ++it; // {0,1}
    EXPECT_EQ(it.indices()[0], 0u);
    EXPECT_EQ(it.indices()[1], 1u);
    ++it; // {0,2}
    EXPECT_EQ(it.indices()[1], 2u);
    ++it; // {1,0}
    EXPECT_EQ(it.indices()[0], 1u);
    EXPECT_EQ(it.indices()[1], 0u);

    // Restart and traverse entire tensor
    It it2(buf.data(), strides, shape);
    size_t count = 0;
    for (; !it2.is_end(); ++it2) {
    seen.push_back(*it2);
    ++count;
    }
    EXPECT_EQ(count, buf.size());
    EXPECT_EQ(seen, buf);
}

TEST(MultiDimIterator, ArrowOperatorWithStruct3D) {
    struct P { int v; };
    // 2 x 2 x 2 tensor, row-major-like strides {4,2,1}
    std::vector<P> buf(8);
    for (size_t i = 0; i < 8; ++i) buf[i].v = static_cast<int>(i);

    using It = MultiDimIterator<P, 3>;
    std::array<size_t,3> shape   {2,2,2};
    std::array<size_t,3> strides {4,2,1};

    It it(buf.data(), strides, shape);
    EXPECT_EQ(it->v, 0);
    ++it; // {0,0,1}
    EXPECT_EQ(it->v, 1);
    ++it; // {0,1,0}
    EXPECT_EQ(it->v, 2);
    ++it; // {0,1,1}
    EXPECT_EQ(it->v, 3);
}

// ============================================================================
// CheckedIterator tests
// ============================================================================

TEST(CheckedIterator, PassThroughWhenDisabled) {
    // With check_finite=false (default), NaN/Inf should not throw.
    std::vector<double> v = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    ContainerIterator<double> raw(v.data());
    CheckedIterator<ContainerIterator<double>> chk(raw);

    // Iterate and read values; no throw expected
    EXPECT_NO_THROW({
        auto it = chk;
        double a = *it;       (void)a;
        ++it;
        double b = *it;       (void)b; // NaN ok when disabled
        ++it;
        double c = *it;       (void)c;
    });

    // operator[] should also pass through
    EXPECT_NO_THROW({
        auto x = chk[1];
        (void)x;
    });

    // operator-> exists; call it (returns double*) and dereference
    EXPECT_NO_THROW({
        auto p = chk.operator->();
        (void)*p;
    });
}

TEST(CheckedIterator, ThrowsWhenEnabled) {
    // Enable finite checking
    auto& opts = NumericOptions::defaults();
    const bool old_flag = opts.check_finite;
    opts.check_finite = true;

    std::vector<double> v = { 1.0, 2.0,
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity() };
    ContainerIterator<double> raw(v.data());
    CheckedIterator<ContainerIterator<double>> chk(raw);

    // Advance to NaN and dereference
    auto it = chk;
    ++it;          // at 2.0
    ++it;          // at NaN
    EXPECT_THROW(*it, ComputationError);

    // operator[] hitting NaN
    EXPECT_THROW( (void)chk[2], ComputationError );

    // Advance to Inf and use operator-> (which checks before returning)
    it = chk;
    it++; it++; it++; // at Inf
    EXPECT_THROW( (void)it.operator->(), ComputationError );

    // restore
    opts.check_finite = old_flag;
}

TEST(CheckedIterator, BaseAndEquality) {
    std::vector<int> v = {1,2,3};
    ContainerIterator<int> raw(v.data());
    CheckedIterator<ContainerIterator<int>> a(raw), b(raw);
    EXPECT_TRUE(a == b);
    ++b;
    EXPECT_TRUE(a != b);
    EXPECT_EQ(b.base(), raw + 1);
}

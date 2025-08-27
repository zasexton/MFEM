// test_storage_base.cpp
#include <gtest/gtest.h>

#include <base/storage_base.h>
#include <base/numeric_base.h>

#include <cstdint>
#include <cstddef>
#include <climits>
#include <new>
#include <stdexcept>
#include <utility>
#include <memory>  // For unique_ptr

// Helper functions to avoid including <algorithm>
namespace test_utils {
    template<typename T>
    constexpr T max(T a, T b) { return (a > b) ? a : b; }
}

// Helper: choose n so n*sizeof(T) is a multiple of Alignment
template <typename T>
static size_t n_multiple_for_alignment(size_t Alignment, size_t cap = 256) {
    for (size_t n = 1; n <= cap; ++n) {
        if ((n * sizeof(T)) % Alignment == 0) return n;
    }
    size_t bytes = ((Alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
    return test_utils::max<size_t>(1, bytes / sizeof(T));
}

struct Tracked {
    static inline int ctor = 0;
    static inline int dtor = 0;
    int v{};

    Tracked() : v(0) { ++ctor; }
    explicit Tracked(int x) : v(x) { ++ctor; }
    Tracked(const Tracked& o) : v(o.v) { ++ctor; }
    Tracked(Tracked&& o) noexcept : v(o.v) { ++ctor; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) = default;
    ~Tracked() { ++dtor; }

    // Arithmetic operations to satisfy NumberLike concept
    Tracked operator+(const Tracked& other) const { return Tracked(v + other.v); }
    Tracked operator-(const Tracked& other) const { return Tracked(v - other.v); }
    Tracked operator*(const Tracked& other) const { return Tracked(v * other.v); }
    Tracked operator/(const Tracked& other) const { return Tracked(v / other.v); }

    // Comparison operations
    bool operator==(const Tracked& other) const { return v == other.v; }
    bool operator!=(const Tracked& other) const { return v != other.v; }
    bool operator<(const Tracked& other) const { return v < other.v; }
    bool operator>(const Tracked& other) const { return v > other.v; }

    // Required for some algorithms
    operator int() const { return v; }
};

// ============================================================================
// DynamicStorage Tests
// ============================================================================

TEST(DynamicStorage, BasicsAndVectorLikeOps) {
    fem::numeric::DynamicStorage<int> ds;
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.capacity(), 0u);
    EXPECT_EQ(ds.layout(), fem::numeric::Layout::RowMajor);
    EXPECT_EQ(ds.device(), fem::numeric::Device::CPU);
    EXPECT_TRUE(ds.is_contiguous());

    ds.push_back(1);
    ds.push_back(2);
    ds.push_back(3);
    EXPECT_EQ(ds.size(), 3u);
    EXPECT_EQ(ds.front(), 1);
    EXPECT_EQ(ds.back(), 3);

    int sum = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
        sum += ds[i];
    }
    EXPECT_EQ(sum, 6);

    EXPECT_EQ(ds[0], 1);
    ds[1] = 22;
    EXPECT_EQ(ds[1], 22);

    auto cap_before = ds.capacity();
    ds.reserve(test_utils::max<size_t>(cap_before + 10, 32));
    EXPECT_GE(ds.capacity(), cap_before + 10);

    ds.resize(5, -7);
    EXPECT_EQ(ds.size(), 5u);
    EXPECT_EQ(ds[3], -7);
    EXPECT_EQ(ds[4], -7);

    ds.clear();
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.size(), 0u);

    ds.resize(4);
    ds.fill(9);
    for (size_t i = 0; i < ds.size(); ++i) {
        EXPECT_EQ(ds[i], 9);
    }
}

TEST(DynamicStorage, CloneAndSwap) {
    fem::numeric::DynamicStorage<double> a(3, 2.5);
    auto up = a.clone();
    ASSERT_TRUE(up);

    auto* b = dynamic_cast<fem::numeric::DynamicStorage<double>*>(up.get());
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ((*b)[i], 2.5);
    }

    (*b)[1] = -1.0;
    EXPECT_DOUBLE_EQ(a[1], 2.5);

    fem::numeric::DynamicStorage<double> c(2, 7.0);
    b->swap(c);
    EXPECT_EQ(b->size(), 2u);
    EXPECT_EQ((*b)[0], 7.0);
    EXPECT_EQ(c.size(), 3u);
    EXPECT_EQ(c[1], -1.0);
}

TEST(DynamicStorage, SwapThrowsWithDifferentStorageType) {
    fem::numeric::DynamicStorage<double> dyn(2, 1.0);
    fem::numeric::StaticStorage<double, 4> st(2, 5.0);
    fem::numeric::StorageBase<double>& base_dyn = dyn;
    fem::numeric::StorageBase<double>& base_st = st;

    EXPECT_THROW(base_dyn.swap(base_st), std::runtime_error);
}

// ============================================================================
// StaticStorage Tests
// ============================================================================

TEST(StaticStorage, ConstructorsAndBounds) {
    fem::numeric::StaticStorage<int, 8> s0;
    EXPECT_TRUE(s0.empty());
    EXPECT_EQ(s0.capacity(), 8u);
    EXPECT_EQ(s0.size(), 0u);
    EXPECT_TRUE(s0.is_contiguous());
    EXPECT_EQ(s0.layout(), fem::numeric::Layout::RowMajor);
    EXPECT_EQ(s0.device(), fem::numeric::Device::CPU);

    fem::numeric::StaticStorage<int, 8> s1(5);
    EXPECT_EQ(s1.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(s1[i], 0);
    }

    fem::numeric::StaticStorage<int, 8> s2(6, 3);
    EXPECT_EQ(s2.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(s2[i], 3);
    }

    EXPECT_THROW((fem::numeric::StaticStorage<int, 4>(5)), std::length_error);
}

TEST(StaticStorage, ResizeReserveClearFillAndClone) {
    fem::numeric::StaticStorage<double, 6> s(3, 1.5);
    s.resize(5, -2.0);
    EXPECT_EQ(s.size(), 5u);
    EXPECT_DOUBLE_EQ(s[3], -2.0);
    EXPECT_DOUBLE_EQ(s[4], -2.0);

    s.resize(2);
    EXPECT_EQ(s.size(), 2u);

    EXPECT_THROW(s.reserve(7), std::length_error);
    EXPECT_THROW(s.resize(7), std::length_error);

    s.fill(9.0);
    for (size_t i = 0; i < s.size(); ++i) {
        EXPECT_DOUBLE_EQ(s[i], 9.0);
    }

    auto up = s.clone();
    auto* sc = dynamic_cast<fem::numeric::StaticStorage<double, 6>*>(up.get());
    ASSERT_NE(sc, nullptr);
    ASSERT_EQ(sc->size(), s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        EXPECT_DOUBLE_EQ((*sc)[i], 9.0);
    }

    s.clear();
    EXPECT_EQ(s.size(), 0u);
    EXPECT_TRUE(s.empty());
}

TEST(StaticStorage, SwapSameTypeAndThrowsOnDifferentType) {
    fem::numeric::StaticStorage<int, 5> a(3, 1);
    fem::numeric::StaticStorage<int, 5> b(2, 8);

    a.swap(b);
    EXPECT_EQ(a.size(), 2u);
    EXPECT_EQ(a[0], 8);
    EXPECT_EQ(b.size(), 3u);
    EXPECT_EQ(b[1], 1);

    fem::numeric::DynamicStorage<int> d(3, 7);
    fem::numeric::StorageBase<int>& base_a = a;
    fem::numeric::StorageBase<int>& base_d = d;
    EXPECT_THROW(base_a.swap(base_d), std::runtime_error);
}

// ============================================================================
// AlignedStorage Tests
// ============================================================================

TEST(AlignedStorage, AlignmentBasicsAndCtor) {
    constexpr size_t Align = 32;
    using AS = fem::numeric::AlignedStorage<double, Align>;

    size_t n = n_multiple_for_alignment<double>(Align);
    AS a(n);
    EXPECT_EQ(a.size(), n);
    EXPECT_EQ(a.capacity(), n);
    EXPECT_TRUE(a.is_contiguous());
    EXPECT_EQ(a.layout(), fem::numeric::Layout::RowMajor);
    EXPECT_EQ(a.device(), fem::numeric::Device::CPU);

    auto addr = reinterpret_cast<std::uintptr_t>(a.data());
    ASSERT_NE(a.data(), nullptr);
    EXPECT_EQ(addr % Align, 0u);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(a[i], 0.0);
    }
}

TEST(AlignedStorage, ResizeReserveFillAndClear) {
    constexpr size_t Align = 32;
    using AS = fem::numeric::AlignedStorage<int, Align>;
    const size_t n0 = 8;
    const size_t n1 = 16;

    AS a(n0, 3);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], 3);
    }

    a.resize(n1, -5);
    EXPECT_EQ(a.size(), n1);
    EXPECT_GE(a.capacity(), n1);
    for (size_t i = 0; i < n0; ++i) {
        EXPECT_EQ(a[i], 3);
    }
    for (size_t i = n0; i < n1; ++i) {
        EXPECT_EQ(a[i], -5);
    }

    a.reserve(24);
    EXPECT_GE(a.capacity(), 24u);
    EXPECT_EQ(a.size(), n1);

    a.fill(42);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], 42);
    }

    a.clear();
    EXPECT_EQ(a.size(), 0u);
    EXPECT_TRUE(a.empty());
}

TEST(AlignedStorage, CopyAndMoveSemantics) {
    constexpr size_t Align = 32;
    using AS = fem::numeric::AlignedStorage<double, Align>;
    const size_t n = n_multiple_for_alignment<double>(Align);

    AS a(n, 1.25);
    AS b = a;
    ASSERT_EQ(b.size(), a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(b[i], a[i]);
    }

    AS c(std::move(a));
    EXPECT_EQ(c.size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(c[i], 1.25);
    }
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(a.data(), nullptr);

    AS d(n, 0.0);
    d = c;
    EXPECT_EQ(d.size(), c.size());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(d[i], 1.25);
    }

    AS e(n, -9.0);
    e = std::move(c);
    EXPECT_EQ(e.size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(e[i], 1.25);
    }
    EXPECT_EQ(c.size(), 0u);
    EXPECT_EQ(c.data(), nullptr);
}

TEST(AlignedStorage, CloneAndSwapAndTypeMismatch) {
    constexpr size_t Align = 32;
    using AS = fem::numeric::AlignedStorage<double, Align>;
    const size_t n = n_multiple_for_alignment<double>(Align);

    AS a(n, 2.0);
    auto up = a.clone();
    auto* ac = dynamic_cast<AS*>(up.get());
    ASSERT_NE(ac, nullptr);
    ASSERT_EQ(ac->size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ((*ac)[i], 2.0);
    }

    AS b(n, -1.0);
    a.swap(b);
    EXPECT_DOUBLE_EQ(a[0], -1.0);
    EXPECT_DOUBLE_EQ(b[0], 2.0);

    fem::numeric::DynamicStorage<double> dyn(2, 3.14);
    fem::numeric::StorageBase<double>& base_al = a;
    fem::numeric::StorageBase<double>& base_ds = dyn;
    EXPECT_THROW(base_al.swap(base_ds), std::runtime_error);
}

TEST(AlignedStorage, TrackedObjectLifetimes) {
    Tracked::ctor = Tracked::dtor = 0;

    constexpr size_t Align = 32;
    using AS = fem::numeric::AlignedStorage<Tracked, Align>;
    const size_t n = n_multiple_for_alignment<Tracked>(Align);
    const size_t m = 2 * n;

    {
        AS a(n);
        EXPECT_EQ(a.size(), n);
        EXPECT_EQ(Tracked::ctor, static_cast<int>(n));

        a.resize(m, Tracked{7});
        EXPECT_EQ(a.size(), m);
        EXPECT_GE(Tracked::ctor, static_cast<int>(m));

        a.resize(n);
        EXPECT_EQ(a.size(), n);
    }
    EXPECT_EQ(Tracked::ctor, Tracked::dtor);
}

// ============================================================================
// Polymorphic behavior via StorageBase<T>
// ============================================================================

TEST(StorageBasePolymorphism, VirtualDispatchWorks) {
    // Dynamic
    std::unique_ptr<fem::numeric::StorageBase<int>> p =
        std::make_unique<fem::numeric::DynamicStorage<int>>(3, 5);
    EXPECT_EQ(p->size(), 3u);
    EXPECT_EQ((*p)[1], 5);
    p->fill(9);
    EXPECT_EQ((*p)[2], 9);
    auto c = p->clone();
    EXPECT_EQ(c->size(), 3u);
    EXPECT_EQ((*c)[0], 9);

    // Static
    std::unique_ptr<fem::numeric::StorageBase<int>> ps =
        std::make_unique<fem::numeric::StaticStorage<int, 4>>(2, 1);
    EXPECT_EQ(ps->capacity(), 4u);
    EXPECT_EQ(ps->size(), 2u);
    ps->resize(3, 7);
    EXPECT_EQ(ps->size(), 3u);
    EXPECT_EQ((*ps)[2], 7);

    // Aligned
    constexpr size_t Align = 32;
    const size_t n = n_multiple_for_alignment<int>(Align);
    std::unique_ptr<fem::numeric::StorageBase<int>> pa =
        std::make_unique<fem::numeric::AlignedStorage<int, Align>>(n, 4);
    EXPECT_EQ(pa->size(), n);
    auto addr = reinterpret_cast<std::uintptr_t>(pa->data());
    EXPECT_EQ(addr % Align, 0u);
}
// ─────────────────────────────────────────────────────────────────────────────
// test_vector.cpp   •  unit tests for numeric/vector/vector.h
// Compile with:  g++ -std=c++20 -I<catch2-install-dir>/include  test_vector.cpp
// ─────────────────────────────────────────────────────────────────────────────
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <limits>

#include "numeric/vector/vector.h"


// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
template<class V>
static bool all_zero(
        const V& v,
        typename V::value_type tol =
        std::is_floating_point_v<typename V::value_type>
        ? std::numeric_limits<typename V::value_type>::epsilon()*4
        : typename V::value_type{} )
{
    using T = typename V::value_type;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>)
            if (std::fabs(v[i]) > tol) {
                return false;
            }
        if ((v[i] != Catch::Approx(T{}))) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Static-extent (N = 3) tests
// -----------------------------------------------------------------------------
TEST_CASE("Static Vector – construction & element access", "[vector][static]")
{
    using numeric::vector::Vector;

    Vector<double,3,numeric::vector::Dense> a;                          // default
    REQUIRE(all_zero(a));

    Vector<double,3,numeric::vector::Dense> b{1.0, 2.0, 3.0};           // init-list
    REQUIRE(b[0] == Catch::Approx(1.0));
    REQUIRE(b.at(2) == Catch::Approx(3.0));

    Vector<double,3,numeric::vector::Dense> c(5.0);                     // fill-value
    for (double x : c)  REQUIRE(x == Catch::Approx(5.0));

    // size/iter
    REQUIRE(c.size()   == 3);
    REQUIRE(std::distance(c.begin(), c.end()) == 3);

    // equality (std::array fast-path)
    REQUIRE(b != c);
    Vector<double,3,numeric::vector::Dense> d{1.0, 2.0, 3.0};
    REQUIRE(b == d);
}

// -----------------------------------------------------------------------------
// Dynamic-extent tests
// -----------------------------------------------------------------------------
TEST_CASE("Dynamic Vector – constructors, resize, equality", "[vector][dynamic]")
{
    using numeric::vector::Vector;

    Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> v0;        // empty
    REQUIRE(v0.empty());

    Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> v1(4);     // length ctor
    REQUIRE(v1.size() == 4);
    REQUIRE(all_zero(v1));

    Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> v2(4, 7);  // fill-value ctor
    for (int x : v2) REQUIRE(x == 7);

    Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> v3{1,2,3,4};   // init-list ctor
    REQUIRE(v3[2] == 3);

    SECTION("copy & move")
    {
        Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> cpy(v3);   // copy ctor
        REQUIRE(cpy == v3);

        Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> mv(std::move(cpy)); // move ctor
        REQUIRE(mv == v3);
        REQUIRE( (cpy.size() == 0 || cpy.data() == nullptr) );        // moved-from
    }

    SECTION("resize keeps prefix")
    {
        v3.resize(6);            // grow
        REQUIRE(v3.size() == 6);
        REQUIRE(v3[0] == 1);
        REQUIRE(v3[3] == 4);

        v3.resize(2);            // shrink
        REQUIRE(v3.size() == 2);
        REQUIRE(v3[1] == 2);
    }

    SECTION("memcmp fast-path for trivially-copyable T")
    {
        Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> w1{9,8,7,6};
        Vector<int, numeric::vector::Dynamic, numeric::vector::Dense> w2{9,8,7,6};
        REQUIRE(w1 == w2);
        w2[3] = 42;
        REQUIRE(w1 != w2);
    }
}

// -----------------------------------------------------------------------------
// Bounds checking (debug builds)
// -----------------------------------------------------------------------------
TEST_CASE("at() throws on bad index", "[vector][bounds]")
{
    using numeric::vector::Vector;
    Vector<double,3, numeric::vector::Dense> s{0.0, 0.0, 0.0};

    REQUIRE_THROWS_AS( s.at(5) , std::out_of_range );

    using DVec = Vector<float, numeric::vector::Dynamic, numeric::vector::Dense>;
    DVec d(2);
    REQUIRE_THROWS_AS( d.at(9) , std::out_of_range );
}
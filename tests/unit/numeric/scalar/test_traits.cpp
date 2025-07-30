// tests/unit/numeric/scalar/test_traits.cpp
//
// Unit tests for numeric::scalar::traits
//
// Add this file to your test target; it needs only Catch2 and the header
// under test.  Example CMake snippet:
//
//     add_executable(test_traits test_traits.cpp)
//     target_link_libraries(test_traits PRIVATE Catch2::Catch2WithMain)
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <complex>
#include <type_traits>
#include "numeric/scalar/traits.h"

using namespace numeric::scalar;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  is_complex_v
// ─────────────────────────────────────────────────────────────────────────────
static_assert( is_complex_v<std::complex<float>> );
static_assert( is_complex_v<std::complex<long double>> );
static_assert( !is_complex_v<double> );
static_assert( !is_complex_v<int> );

TEST_CASE("is_complex_v behaves", "[traits][is_complex]") {
    REQUIRE( is_complex_v<std::complex<double>> );
    REQUIRE_FALSE( is_complex_v<float> );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  real_t   (underlying real type)
// ─────────────────────────────────────────────────────────────────────────────
static_assert( std::is_same_v< real_t<double> , double > );
static_assert( std::is_same_v< real_t<std::complex<float>> , float > );

TEST_CASE("real_t extracts base type", "[traits][real]") {
    REQUIRE( std::is_same_v< real_t<std::complex<long double>>,
                     long double > );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  imag::type  (void for non-complex, R for std::complex<R>)
// ─────────────────────────────────────────────────────────────────────────────
static_assert( std::is_void_v< imag<double>::type > );
static_assert( std::is_same_v< imag<std::complex<float>>::type , float > );

TEST_CASE("imag trait works", "[traits][imag]") {
    REQUIRE( std::is_void_v< imag<int>::type > );
    REQUIRE( std::is_same_v< imag<std::complex<double>>::type , double > );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  promote<A,B>::type  – common scalar type
// ─────────────────────────────────────────────────────────────────────────────
//template<class A, class B>
//using promote_t = typename promote<A,B>::type;

// ─ real / real
static_assert( std::is_same_v< promote_t<float, double>, double > );

// ─ real / complex   (both orders)
static_assert( std::is_same_v<
        promote_t<double, std::complex<float>>,
        std::complex<double> > );

static_assert( std::is_same_v<
        promote_t<std::complex<float>, double>,
        std::complex<double> > );

// ─ complex / complex
static_assert( std::is_same_v<
        promote_t<std::complex<float>, std::complex<double>>,
        std::complex<double> > );

TEST_CASE("promote produces expected types", "[traits][promote]") {
    using C1 = promote_t<float, double>;
    using C2 = promote_t<double, std::complex<float>>;
    using C3 = promote_t<std::complex<float>, std::complex<double>>;

    // Run-time confirmation gives nicer failure output than static_assert alone
    REQUIRE( std::is_same_v<C1, double> );
    REQUIRE( std::is_same_v<C2, std::complex<double>> );
    REQUIRE( std::is_same_v<C3, std::complex<double>> );
}

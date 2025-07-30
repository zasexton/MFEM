// tests/unit/numeric/scalar/test_concepts.cpp
//
// Unit tests for numeric::scalar::concepts
// Requires Catch2 v3 (add via FetchContent or CPM as you do for other tests).

#include <complex>
#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>


#include "numeric/scalar/concepts.h"

using namespace numeric::scalar;

// -----------------------------------------------------------------------------
// 1. Positive & negative checks for each concept (compile-time)
// -----------------------------------------------------------------------------
static_assert( RealLike<float> );
static_assert( RealLike<double> );
static_assert( RealLike<long double> );
static_assert( !RealLike<int> );
static_assert( !RealLike<std::complex<double>> );

static_assert( ComplexLike<std::complex<float>> );
static_assert( ComplexLike<std::complex<long double>> );
static_assert( !ComplexLike<double> );
static_assert( !ComplexLike<std::complex<int>> );        // underlying type not floating

static_assert( ScalarLike<float> );
static_assert( ScalarLike<std::complex<double>> );
static_assert( !ScalarLike<int> );                       // by design, integrals are not ScalarLike

static_assert( SignedIntegral<signed char> );
static_assert( !SignedIntegral<unsigned int> );

static_assert( UnsignedIntegral<unsigned long long> );
static_assert( !UnsignedIntegral< long > );

// -----------------------------------------------------------------------------
// 2. real_t deduces the correct type
// -----------------------------------------------------------------------------
TEMPLATE_TEST_CASE("real_t resolves correctly","[concepts][trait]",
float, double, long double)
{
using R  = TestType;
using C  = std::complex<R>;
// complex → cast to real
STATIC_REQUIRE( std::is_same_v< real_t<C>, R > );
// real → itself
STATIC_REQUIRE( std::is_same_v< real_t<R>,  R > );
}

// -----------------------------------------------------------------------------
// 3. Constraint substitution failure: helper metafunction
// -----------------------------------------------------------------------------
template<ScalarLike T>
constexpr bool accepts_scalar() { return true; }

TEST_CASE("Function templates constrained by ScalarLike",
          "[concepts][requires]")
{
    REQUIRE( accepts_scalar<double>() );
    REQUIRE( accepts_scalar<std::complex<float>>() );
    // The following line *must not compile*; uncomment to verify manually:
    // accepts_scalar(42);
}

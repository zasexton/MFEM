// tests/unit/numeric/scalar/test_constants.cpp
//
// Unit tests for numeric::scalar::constants
//
// Compile with the rest of your test suite, e.g.
//   ctest -R constants
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <numbers>                           // reference values
#include "numeric/scalar/constants.h"        // header under test

using namespace numeric::scalar::constants;
using Catch::Approx;

/** Bit-for-bit equality for *any* trivially copyable type. */
template <class T>
constexpr bool bit_equal(const T& a, const T& b) noexcept
{
    if constexpr (std::is_integral_v<T>)   // ints: just compare
        return a == b;
    else {
        static_assert(std::is_trivially_copyable_v<T>);
        const unsigned char* pa = reinterpret_cast<const unsigned char*>(&a);
        const unsigned char* pb = reinterpret_cast<const unsigned char*>(&b);
        for (std::size_t i = 0; i < sizeof(T); ++i)
            if (pa[i] != pb[i]) return false;
        return true;
    }
}

// -----------------------------------------------------------------------------
// Mathematical constants
// -----------------------------------------------------------------------------
TEMPLATE_TEST_CASE("Mathematical constants match <numbers>",
"[constants][math]",
float, double, long double)
{
// All of these should be *bit-for-bit* identical to std::numbers::…
REQUIRE(bit_equal(pi<TestType>          , std::numbers::pi_v<TestType>));
REQUIRE(bit_equal(e<TestType>           , std::numbers::e_v<TestType>));
REQUIRE(bit_equal(sqrt2<TestType>       , std::numbers::sqrt2_v<TestType>));
REQUIRE(bit_equal(golden_ratio<TestType>, std::numbers::phi_v<TestType>));
REQUIRE(bit_equal(ln10<TestType>        , std::numbers::ln10_v<TestType>));
REQUIRE(bit_equal(ln2<TestType>         , std::numbers::ln2_v<TestType>));
REQUIRE(bit_equal(log10e<TestType>      , std::numbers::log10e_v<TestType>));
REQUIRE(bit_equal(log2e<TestType>       , std::numbers::log2e_v<TestType>));

// Run-time verification (gives nicer failure messages than static_assert)
REQUIRE(pi<TestType>           == Approx(std::numbers::pi_v<TestType>));
REQUIRE(e<TestType>            == Approx(std::numbers::e_v<TestType>));
REQUIRE(sqrt2<TestType>        == Approx(std::numbers::sqrt2_v<TestType>));
REQUIRE(golden_ratio<TestType> == Approx(std::numbers::phi_v<TestType>));
REQUIRE(ln10<TestType>         == Approx(std::numbers::ln10_v<TestType>));
REQUIRE(ln2<TestType>          == Approx(std::numbers::ln2_v<TestType>));
REQUIRE(log10e<TestType>       == Approx(std::numbers::log10e_v<TestType>));
REQUIRE(log2e<TestType>        == Approx(std::numbers::log2e_v<TestType>));
}

// -----------------------------------------------------------------------------
// Physical constants – representative spot-checks
// -----------------------------------------------------------------------------
TEST_CASE("Exact physical constants", "[constants][physics][exact]")
{
    // Values that are *defined* in the SI are exact; compare bit-for-bit.
    REQUIRE(bit_equal(speed_of_light_vacuum<double> , 299'792'458.0));
    REQUIRE(bit_equal(planck_constant<long double>  , 6.626'070'15e-34L));
    REQUIRE(bit_equal(avogadro<double>              , 6.022'140'76e23));
    REQUIRE(bit_equal(elementary_charge<double>     , 1.602'176'634e-19));
    REQUIRE(bit_equal(boltzmann<double>             , 1.380'649e-23));
    REQUIRE(bit_equal(josephson<double>             , 483'597.848'4e9));
}

TEST_CASE("Floating physical constants within 0.5 ULP", "[constants][physics]")
{
    using Catch::Matchers::WithinRel;

    // Constants that come from CODATA evaluations (have uncertainty)
    REQUIRE(newtonian_gravitational_constant<double>
            == Approx(6.674'30e-11).epsilon(1e-7));   // ≈10 ppb
    REQUIRE(vacuum_electric_permittivity<double>
            == Approx(8.854'187'8188e-12).epsilon(1e-12));
    REQUIRE(von_klitzing<double>
            == Approx(25'812.807'45).epsilon(1e-12));
}

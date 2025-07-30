// tests/unit/numeric/test_epsilon.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <numeric/scalar/epsilon.h>

using namespace numeric::scalar::epsilon;
using Catch::Approx;

TEST_CASE("almost_equal basic", "[epsilon]")
{
    constexpr double eps = std::numeric_limits<double>::epsilon();

    REQUIRE(almost_equal(1.0, 1.0));
    REQUIRE(almost_equal(1.0, 1.0 + 5 * eps));
    REQUIRE_FALSE(almost_equal(1.0, 1.0 + 1e-6));
}

TEST_CASE("is_zero", "[epsilon]")
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    REQUIRE(is_zero(0.0));
    REQUIRE(is_zero(5 * eps));
    REQUIRE_FALSE(is_zero(1e-3));
}

TEST_CASE("relative_error", "[epsilon]")
{
    REQUIRE(relative_error(2.0, 2.0) == Approx(0.0));
    REQUIRE(relative_error(2.0, 4.0) == Approx(0.5));
}

TEST_CASE("ULP comparison", "[epsilon]")
{
    double a = 1.0;
    double b = std::nextafter(a, 2.0);      // exactly 1-ULP apart
    REQUIRE(ulp_distance(a, a) == 0);
    REQUIRE(ulp_distance(a, b) == 1);
    REQUIRE(almost_equal_ulps(a, b, 1));
    REQUIRE_FALSE(almost_equal_ulps(a, b, 0));
}

TEST_CASE("tolerant_less / greater", "[epsilon]")
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    double a = 1.0;
    double b = 1.0 + 0.5 * eps;
    REQUIRE_FALSE(tolerant_less(a, b, eps));
    REQUIRE_FALSE(tolerant_greater(b, a, eps));

    b = 1.0 + 100 * eps;
    REQUIRE(tolerant_less(a, b, eps));
    REQUIRE(tolerant_greater(b, a, eps));
}

TEST_CASE("in_range inclusive", "[epsilon]")
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    REQUIRE(in_range(0.0, 0.0, 1.0));
    REQUIRE(in_range(-eps * 5, 0.0, 1.0));
    REQUIRE_FALSE(in_range(-1e-3, 0.0, 1.0));
}

TEST_CASE("almost_integer & same_sign", "[epsilon]")
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    REQUIRE(almost_integer(3.0));
    REQUIRE(almost_integer(3.0 + 4 * eps));
    REQUIRE_FALSE(almost_integer(3.1));

    REQUIRE(same_sign( 2.0, 3.0));
    REQUIRE(same_sign(-2.0,-3.0));
    REQUIRE_FALSE(same_sign(2.0,-3.0));
    REQUIRE_FALSE(same_sign(eps*2, 5.0));   // first essentially zero
}

TEST_CASE("is_finite", "[epsilon]")
{
    REQUIRE(is_finite(1.0));
    REQUIRE_FALSE(is_finite(std::numeric_limits<double>::infinity()));
    REQUIRE_FALSE(is_finite(std::numeric_limits<double>::quiet_NaN()));
}

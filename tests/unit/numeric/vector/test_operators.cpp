// ─────────────────────────────────────────────────────────────────────────────
//  test_operators.cpp   –  unit tests for numeric/vector/operators.h
// ─────────────────────────────────────────────────────────────────────────────
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "numeric/vector/vector.h"
#include "numeric/vector/operators.h"

namespace nv = numeric::vector;
using Sparse = nv::Sparse;
using Dense  = nv::Dense;
using SVec   = nv::Vector<double, 3, Dense>;
using DVec   = nv::Vector<double, nv::Dynamic, Dense>;

using Catch::Approx;

TEST_CASE("Construction and type promotion", "[vector][ops]")
{
    nv::Vector<int,3,Dense>    ai{1,2,3};
    nv::Vector<float,3,Dense>  bf{4,5,6};
    nv::Vector<float,3,Dense>  cf{0,0,0};
    nv::Vector<int,3,Dense>    df{0,0,0};
    nv::Vector<int,3,Dense>    ef{1,1,1};

    auto sum = ai + bf;                      // int + float → float
    STATIC_REQUIRE( std::is_same_v<decltype(sum), nv::Vector<float,3,Dense>> );
    REQUIRE( sum == nv::Vector<float,3,Dense>{5,7,9});

    auto mixed = 2.5 * ai;                   // scalar * int → double
    STATIC_REQUIRE( std::is_same_v<decltype(mixed), nv::Vector<double,3,Dense>> );
    REQUIRE( mixed == nv::Vector<double,3,Dense>{2.5,5.0,7.5});

    cf += ai;
    STATIC_REQUIRE(std::is_same_v<decltype(cf), nv::Vector<float, 3, Dense>> );
    REQUIRE( cf == nv::Vector<float, 3, Dense>{1.0, 2.0, 3.0});

    cf -= ai;
    STATIC_REQUIRE(std::is_same_v<decltype(cf), nv::Vector<float, 3, Dense>> );
    REQUIRE( cf == nv::Vector<float, 3, Dense>{0.0, 0.0, 0.0});

    df += bf;
    STATIC_REQUIRE(std::is_same_v<decltype(df), nv::Vector<int, 3, Dense>> );
    REQUIRE( df == nv::Vector<int, 3, Dense>{4, 5, 6});

    df -= bf;
    STATIC_REQUIRE(std::is_same_v<decltype(df), nv::Vector<int, 3, Dense>> );
    REQUIRE( df == nv::Vector<int, 3, Dense>{0, 0, 0});

    ef *= ai;
    STATIC_REQUIRE(std::is_same_v<decltype(ef), nv::Vector<int, 3, Dense>> );
    REQUIRE( ef == nv::Vector<int, 3, Dense>{1, 2, 3});
}


TEST_CASE("Unary ± and scalar ops", "[vector][ops]") {
    SVec a{1, -2, 3};

    REQUIRE(+a == a);
    REQUIRE((-a)[0] == Approx(-1));

    auto b = a + 2.0;          // vector + scalar
    REQUIRE(b == SVec{3, 0, 5});
    REQUIRE((2.0 + a) == b);

    REQUIRE((a - 2.0) == SVec{-1, -4, 1});
    REQUIRE((2.0 - a) == SVec{ 1,  4,-1});

    REQUIRE((a * 2.0) == SVec{2,-4,6});
    REQUIRE((2.0 * a) == a*2.0);
    REQUIRE((a / 2.0) == SVec{0.5,-1,1.5});
}

TEST_CASE("Vector ± vector & dot / cross", "[vector][ops]") {
    SVec u{1,2,3}, v{4,5,6};

    REQUIRE(u+v == SVec{5,7,9});
    REQUIRE(v-u == SVec{3,3,3});
    REQUIRE(u*v == SVec{4,10,18});

    REQUIRE(dot(u,v) == Approx(32.0));

    REQUIRE(cross(u,v) == SVec{-3,6,-3});
    REQUIRE(triple_product(u,v,SVec{7,8,9}) == Approx(  0.0  ));
}
/*
TEST_CASE("Compound assignment and BLAS-1 kernels", "[vector][ops]") {
    SVec y{1,2,3}, x{1,1,1};

    y += x;
    REQUIRE(y == SVec{2,3,4});

    y *= 0.5;
    REQUIRE(y == SVec{1,1.5,2});

    nv::axpy(2.0, x, y);                 // y += 2·x  → {3,3.5,4}
    REQUIRE(y == SVec{3,3.5,4});

    nv::axpby(1.0, x, 2.0, y);           // y = x + 2y → {7,8,9}
    REQUIRE(y == SVec{7,8,9});

    nv::xpay(x, -1.0, y);                // y = x − y  → {-6,-7,-8}
    REQUIRE(y == SVec{-6,-7,-8});
}
*/
TEST_CASE("Element-wise multiply / divide & reductions", "[vector][ops]") {
    SVec a{1,2,3}, b{2,4,6};

    REQUIRE( (a*b) == SVec{2,8,18} );
    REQUIRE( (b/a) == SVec{2,2,2}   );

    REQUIRE(sum(a)  == Approx(6.0));
    REQUIRE(min(b)  == Approx(2.0));
    REQUIRE(max(b)  == Approx(6.0));
    REQUIRE(norm2(a)== Approx(14.0));
    REQUIRE(norm(a) == Approx(std::sqrt(14.0)));

    REQUIRE(normalize(a)*norm(a) == normalize(a)*std::sqrt(14.0)); // scaled back
}
/*
TEST_CASE("Unary wrappers abs/sqrt/exp/log/sign", "[vector][ops]") {
    SVec v{-1.0, 4.0, 9.0};

    REQUIRE( numeric::vector::abs(v)  == SVec{1,4,9});
    auto w = numeric::vector::sqrt(v);
    REQUIRE( std::isnan(w[0]));
    REQUIRE( w[1] == Approx(2.0));
    REQUIRE( w[2] == Approx(3.0));
    REQUIRE( numeric::vector::sign(v) == SVec{-1,1,1});

    // round-trip exp+log
    auto ww = log(exp(v));
    REQUIRE( ww[1] == Approx(std::log(std::exp(4.0))) );
}

TEST_CASE("pow / clamp / weighted dot", "[vector][ops]") {
    SVec v{1,2,3};
    REQUIRE( pow(v,2.0) == SVec{1,4,9} );

    auto c = clamp(v,1.5,2.5);           // → {1.5,2,2.5}
    REQUIRE( c == SVec{1.5,2,2.5} );

    REQUIRE( dot_w(v,v,SVec{0.5,1.0,2.0}) == Approx(1*1*0.5 + 2*2*1 + 3*3*2));
}

TEST_CASE("Prefix-sum (inclusive scan)", "[vector][ops]") {
    DVec in(5), out(5);
    in[0]=1; in[1]=2; in[2]=3; in[3]=4; in[4]=5;

    nv::prefix_sum(in,out);
    REQUIRE( (out[0]== Approx(1) && out[4]== Approx(15)) );
}

TEST_CASE("Projection onto plane", "[vector][ops]") {
    SVec v{1,2,3};
    SVec n = normalize(SVec{0,0,1});     // z-axis
    auto v_perp = project_orthogonal(v,n);
    REQUIRE( v_perp == SVec{1,2,0} );
}
*/

TEST_CASE("Dense and sparse arithmetic", "[vector][ops][sparse]") {
    nv::Vector<int,3,Dense> d{1,2,3};
    nv::Vector<int,3,Sparse> s{{0,1},{2,2}};
    auto sum = d + s;
    REQUIRE(sum == nv::Vector<int,3,Dense>{2,2,5});
}
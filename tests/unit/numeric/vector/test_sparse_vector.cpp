#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "numeric/vector/vector.h"

namespace nv = numeric::vector;
using Sparse = nv::Sparse;
using Dense = nv::Dense;

TEST_CASE("Static sparse vector basic", "[vector][sparse][static]")
{
    nv::Vector<int,5,Sparse> v;
    REQUIRE(v.nnz() == 0);
    v.push_back(0,1);
    v.push_back(4,3);
    v.set(2,2);
    REQUIRE(v.nnz() == 3);
    REQUIRE(v.at(0) == 1);
    REQUIRE(v.at(2) == 2);
    REQUIRE(v.at(4) == 3);
    // iteration
    int sum=0;
    for(const auto& [i,x] : v) {
        sum += x;
    }
    REQUIRE(sum==6);
}

TEST_CASE("Dynamic sparse vector resize and equality", "[vector][sparse][dynamic]")
{
    nv::Vector<int, nv::Dynamic, Sparse> a(5);
    a.push_back(1,2);
    a.push_back(3,4);
    nv::Vector<int, nv::Dynamic, Sparse> b(a);
    REQUIRE(a==b);
    a.resize(3);
    REQUIRE(a.size()==3);
    REQUIRE(a.nnz()==1);
    nv::Vector<int, nv::Dynamic, Sparse> c(3, {{1, 2}});
    REQUIRE(a==c);
    swap(a, b);
    REQUIRE(b.size()==3);
    REQUIRE(a.size()==5);
}
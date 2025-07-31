// ─────────────────────────────────────────────────────────────────────────────
// test_vector_traits.cpp
//
//   • VectorLike concept / is_vector_v
//   • scalar_t, static_size_v, extent()
//   • rank_v
//   • promote<> specialisations
//
// Requires: Catch2 v3 (already fetched in root CMakeLists.txt)
// ─────────────────────────────────────────────────────────────────────────────
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <complex>

#include "numeric/vector/traits.h"
#include "numeric/vector/vector.h"

namespace nv  = numeric::vector;
namespace nsc = numeric::scalar;



// -----------------------------------------------------------------------------
// 1.  Concept & trait sanity (compile-time)
// -----------------------------------------------------------------------------
static_assert( nv::VectorLike< nv::Vector<double,3> >   );
static_assert( nv::VectorLike< nv::Vector<int ,nv::Dynamic> > );
static_assert( !nv::VectorLike< int > );

static_assert( nv::is_vector_v< nv::Vector<float,4> > );
static_assert( !nv::is_vector_v< double > );

// scalar_t
static_assert( std::is_same_v<
        nv::scalar_t_t< nv::Vector<std::complex<float>,5>>,
        std::complex<float> > );

// static_size
static_assert( nv::static_size_v< nv::Vector<double,3> > == 3 );
static_assert( nv::static_size_v<
        nv::Vector<int,nv::Dynamic> > == nv::Dynamic );

// rank
static_assert( nv::rank_v< nv::Vector<float,7> > == 1 );

static_assert( nv::rank_v< nv::Vector<float,7> > == 1 );
TEST_CASE("extent helper", "[vector][traits]")
{
    nv::Vector<int,3> s{1,2,3};
    nv::Vector<int,nv::Dynamic> d{1,2,3,4};
    REQUIRE(nv::extent(s) == 3);
    REQUIRE(nv::extent(d) == 4);
}
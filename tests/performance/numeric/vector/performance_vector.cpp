// ─────────────────────────────────────────────────────────────────────────────
// performance_vector.cpp
//
// Micro–benchmarks for numeric/vector/vector.h
//   • allocation + fill bandwidth
//   • dot product throughput
//   • SAXPY (y = a + β·b)
//
// Build notes:
//   • Requires Catch2 v3 with BENCHMARK support.
//   • Compile with -O3 and define CATCH_CONFIG_ENABLE_BENCHMARKING.
//     Example CMake snippet:
//
//     target_compile_definitions(performance_vector PRIVATE
//         CATCH_CONFIG_ENABLE_BENCHMARKING
//     )
//     target_compile_options(performance_vector PRIVATE -O3 -DNDEBUG)
//
// ─────────────────────────────────────────────────────────────────────────────
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "numeric/vector/vector.h"   // container under test

#include <numeric>   // std::inner_product
#include <random>

namespace nv = numeric::vector;

// Helper: fill a vector with reproducible random data
template<class Vec>
static void fill_random(Vec& v, std::mt19937_64& rng)
{
    using T = typename Vec::value_type;
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    for (auto& x : v) x = dist(rng);
}

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Vector micro-benchmarks", "[vector][performance]")
{
    using DVec = nv::Vector<double, nv::Dynamic>;

    //constexpr std::size_t nSmall = 4'096;       // ~4 k elements  (≈ 32 KiB)
    constexpr std::size_t nLarge = 1'000'000;   // 1 M elements  (≈ 8 MiB)

    std::mt19937_64 rng(123456);

    BENCHMARK("default ctor          dyn")      { DVec v; return v.size(); };

    BENCHMARK("size ctor             dyn")      { DVec v(nLarge); return v.size(); };

    BENCHMARK("fill-value ctor       dyn")      { DVec v(nLarge, 1.23); return v[0]; };

    BENCHMARK("init-list ctor        dyn")      { DVec v{1.0,2.0,3.0,4.0}; return v.size(); };

    BENCHMARK("construct + fill   1 M doubles")
    {
        DVec v(nLarge);
        fill_random(v, rng);
        return v.size();
    };

    // Pre-allocate two big vectors for math-only kernels
    DVec a(nLarge), b(nLarge);
    fill_random(a, rng);
    fill_random(b, rng);

    BENCHMARK("dot product        1 M doubles")
    {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    };

    BENCHMARK("SAXPY  y = a + β·b  (1 M)")
    {
        DVec y(nLarge);
        const double beta = 2.5;
        for (std::size_t i = 0; i < nLarge; ++i) {
            y[i] = a[i] + beta * b[i];
        }
        return y[0];    // prevent optimisation-out
    };

    BENCHMARK("copy ctor          1 M doubles")
                {
                    DVec c(a);           // a is the pre-filled 1-M vector from earlier
                    return c[0];
                };

    BENCHMARK("copy assign        1 M doubles")
                {
                    DVec c(nLarge);
                    c = a;
                    return c[0];
                };

    BENCHMARK("move ctor          1 M doubles")
                {
                    DVec tmp(nLarge);
                    fill_random(tmp, rng);
                    DVec c(std::move(tmp));
                    return c[0];
                };

    BENCHMARK("move assign        1 M doubles")
                {
                    DVec src(nLarge);
                    fill_random(src, rng);
                    DVec dst(nLarge);
                    dst = std::move(src);
                    return dst[0];
                };
    BENCHMARK("resize shrink  →½     1 M")
                {
                    DVec v(nLarge);
                    v.resize(nLarge/2);
                    return v.size();
                };

    BENCHMARK("resize grow    ×2     1 M")
                {
                    DVec v(nLarge);
                    v.resize(nLarge*2);
                    return v.size();
                };

    BENCHMARK("fill-zero          1 M doubles")
                {
                    DVec v(nLarge);
                    std::fill(v.begin(), v.end(), 0.0);
                    return v[0];
                };

    // -------------------- static (N = 3) -------------------
    using SVec = nv::Vector<double,3>;
    SVec p{1.0,2.0,3.0};

    BENCHMARK("default ctor          N=3")      { SVec v; return v.size(); };

    BENCHMARK("fill-value ctor       N=3")      { SVec v(4.56); return v[0]; };

    BENCHMARK("copy ctor             N=3")      { SVec q(p); return q[1]; };

    BENCHMARK("copy assign           N=3")      { SVec q; q = p; return q[2]; };

    BENCHMARK("move ctor             N=3")      { SVec tmp; SVec q(std::move(tmp)); return q.size(); };

    BENCHMARK("move assign           N=3")      { SVec src; SVec dst; dst = std::move(src); return dst.size(); };
}

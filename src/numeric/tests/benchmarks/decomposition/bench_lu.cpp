#include <benchmark/benchmark.h>

#include <decompositions/lu.h>
#include <core/matrix.h>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

static Matrix<double> make_general(std::size_t m, std::size_t n, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double> A(m, n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = dist(gen);
    return A;
}

static void BM_LU_Unblocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_general(n, n, 71u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        std::vector<int> piv;
        int info = lu_factor(A, piv);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(piv.data());
    }
    state.SetItemsProcessed(state.iterations() * n * n * n / 3);
}

static void BM_LU_Blocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_general(n, n, 73u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        std::vector<int> piv;
        int info = lu_factor_blocked(A, piv, 128);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(piv.data());
    }
    state.SetItemsProcessed(state.iterations() * n * n * n / 3);
}

BENCHMARK(BM_LU_Unblocked)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_LU_Blocked)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();


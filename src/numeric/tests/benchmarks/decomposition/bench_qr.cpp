#include <benchmark/benchmark.h>

#include <decompositions/qr.h>
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

static void BM_QR_Unblocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t m = n; // square for simplicity
    Matrix<double> A0 = make_general(m, n, 91u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        std::vector<double> tau;
        int info = qr_factor(A, tau);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(tau.data());
    }
    state.SetItemsProcessed(state.iterations() * m * n * n);
}

static void BM_QR_Blocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t m = n; // square for simplicity
    Matrix<double> A0 = make_general(m, n, 93u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        std::vector<double> tau;
        int info = qr_factor_blocked(A, tau, 64);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(tau.data());
    }
    state.SetItemsProcessed(state.iterations() * m * n * n);
}

BENCHMARK(BM_QR_Unblocked)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_QR_Blocked)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();


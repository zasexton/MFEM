#include <benchmark/benchmark.h>

#include <decompositions/cholesky.h>
#include <core/matrix.h>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

static Matrix<double> make_spd(std::size_t n, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double> G(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            G(i, j) = dist(gen);
    // A = G^T G + n*I
    Matrix<double> A(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += G(k, i) * G(k, j);
            A(i, j) = s + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    return A;
}

static void BM_Cholesky_Unblocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_spd(n, 42u);
    for (auto _ : state) {
        benchmark::DoNotOptimize(A0);
        Matrix<double> A = A0;
        int info = cholesky_factor(A);
        benchmark::DoNotOptimize(info);
    }
    state.SetItemsProcessed(state.iterations() * n * n * n / 3);
}

static void BM_Cholesky_Blocked(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_spd(n, 43u);
    for (auto _ : state) {
        benchmark::DoNotOptimize(A0);
        Matrix<double> A = A0;
        int info = cholesky_factor_blocked(A, fem::numeric::linear_algebra::Uplo::Lower, 128);
        benchmark::DoNotOptimize(info);
    }
    state.SetItemsProcessed(state.iterations() * n * n * n / 3);
}

BENCHMARK(BM_Cholesky_Unblocked)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_Cholesky_Blocked)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();


#include <benchmark/benchmark.h>

#include <decompositions/eigen.h>
#include <core/matrix.h>
#include <core/vector.h>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

namespace {

Matrix<double> make_symmetric(std::size_t n, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double> A(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double val = dist(gen);
            A(i, j) = val;
            A(j, i) = val;
        }
    }
    return A;
}

} // namespace

static void BM_Eigen_Symmetric_Full(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_symmetric(n, 77u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        Vector<double> evals;
        Matrix<double> evecs;
        int info = eigen_symmetric(A, evals, evecs);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(evals.data());
        benchmark::DoNotOptimize(evecs.data());
    }
    const double work = static_cast<double>(state.iterations()) * n * n * n;
    state.SetItemsProcessed(static_cast<int64_t>(work));
}

static void BM_Eigen_Symmetric_ValuesOnly(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix<double> A0 = make_symmetric(n, 131u);
    for (auto _ : state) {
        Matrix<double> A = A0;
        Vector<double> evals;
        int info = eigen_symmetric_values(A, evals);
        benchmark::DoNotOptimize(info);
        benchmark::DoNotOptimize(evals.data());
    }
    const double work = static_cast<double>(state.iterations()) * n * n * n;
    state.SetItemsProcessed(static_cast<int64_t>(work));
}

BENCHMARK(BM_Eigen_Symmetric_Full)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BM_Eigen_Symmetric_ValuesOnly)->Arg(64)->Arg(128)->Arg(256);

BENCHMARK_MAIN();


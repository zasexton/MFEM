#include <benchmark/benchmark.h>

#include <decompositions/eigen.h>
#include <core/matrix.h>
#include <vector>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

namespace {

template <typename T>
Matrix<T> make_hermitian(std::size_t n, uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<typename numeric_traits<T>::scalar_type> dist(-1.0, 1.0);
  Matrix<T> A(n, n, T{});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      if constexpr (is_complex_number_v<T>) {
        T v(dist(gen), dist(gen));
        A(i, j) = v;
        A(j, i) = fem::numeric::decompositions::detail::conj_if_complex(v);
      } else {
        T v = static_cast<T>(dist(gen));
        A(i, j) = v;
        A(j, i) = v;
      }
    }
  }
  return A;
}

} // namespace

static void BM_Tridiag_SequentialPerColumn(benchmark::State& state)
{
  using T = double;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  Matrix<T> A0 = make_hermitian<T>(n, 123u);
  for (auto _ : state) {
    Matrix<T> A = A0;
    std::vector<typename numeric_traits<T>::scalar_type> d, e;
    detail::hermitian_to_tridiagonal(A, d, e, nullptr, /*block_size=*/32, /*use_panel_update=*/false);
    benchmark::DoNotOptimize(d.data());
    benchmark::DoNotOptimize(e.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
}

static void BM_Tridiag_PanelAggregated(benchmark::State& state)
{
  using T = double;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  Matrix<T> A0 = make_hermitian<T>(n, 456u);
  for (auto _ : state) {
    Matrix<T> A = A0;
    std::vector<typename numeric_traits<T>::scalar_type> d, e;
    detail::hermitian_to_tridiagonal(A, d, e, nullptr, /*block_size=*/32, /*use_panel_update=*/true);
    benchmark::DoNotOptimize(d.data());
    benchmark::DoNotOptimize(e.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
}

BENCHMARK(BM_Tridiag_SequentialPerColumn)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BM_Tridiag_PanelAggregated)->Arg(64)->Arg(128)->Arg(256);

BENCHMARK_MAIN();


#include <benchmark/benchmark.h>

#include <linear_algebra/householder_wy.h>
#include <core/matrix.h>
#include <vector>
#include <random>

using namespace fem::numeric;
using namespace fem::numeric::linear_algebra;

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
        A(j, i) = conj_if_complex(v);
      } else {
        T v = static_cast<T>(dist(gen));
        A(i, j) = v;
        A(j, i) = v;
      }
    }
  }
  return A;
}

template <typename T>
void form_VT(std::size_t n, std::size_t kb, uint32_t seed, Matrix<T>& V, Matrix<T>& Tm)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<typename numeric_traits<T>::scalar_type> dist(-0.5, 0.5);
  V = Matrix<T>(n, kb, T{});
  for (std::size_t j = 0; j < kb; ++j) {
    for (std::size_t i = 0; i < j; ++i) V(i, j) = T{};
    V(j, j) = T{1};
    for (std::size_t i = j + 1; i < n; ++i) {
      if constexpr (is_complex_number_v<T>) V(i, j) = T(dist(gen), dist(gen));
      else V(i, j) = static_cast<T>(dist(gen));
    }
  }
  // Random positive taus for a plausible block
  std::vector<T> tau(kb, T{});
  for (std::size_t j = 0; j < kb; ++j) {
    if constexpr (is_complex_number_v<T>) tau[j] = T(dist(gen), dist(gen));
    else tau[j] = static_cast<T>(std::abs(dist(gen)) + 0.5);
  }
  form_block_T_forward_columnwise(V, tau, Tm);
}

// Heavy baseline: materialize large temporaries
template <typename T>
void heavy_two_sided_update(Matrix<T>& A, const Matrix<T>& V, const Matrix<T>& Tm)
{
  const std::size_t n = A.rows();
  const std::size_t kb = V.cols();
  Matrix<T> W(n, kb, T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, A, V, T{0}, W);
  Matrix<T> X(kb, kb, T{});
  gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, V, W, T{0}, X);
  Matrix<T> X2(kb, kb, T{});
  gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, Tm, X, T{0}, X2);
  Matrix<T> VX(n, kb, T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, V, X2, T{0}, VX);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < kb; ++j)
      W(i, j) = static_cast<T>(W(i, j) - static_cast<T>(0.5) * VX(i, j));
  // D = V W^H + W V^H
  Matrix<T> D1(n, n, T{});
  gemm(Trans::NoTrans, Trans::ConjTranspose, T{1}, V, W, T{0}, D1);
  Matrix<T> D2(n, n, T{});
  gemm(Trans::NoTrans, Trans::ConjTranspose, T{1}, W, V, T{0}, D2);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j) {
      auto val = static_cast<T>(A(i, j) - (D1(i, j) + D2(i, j)));
      A(i, j) = val;
      if (j != i) A(j, i) = conj_if_complex(val);
    }
}

} // namespace

static void BM_WY_TwoSided_Heavy(benchmark::State& state)
{
  using T = double;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  const std::size_t kb = static_cast<std::size_t>(state.range(1));
  Matrix<T> A0 = make_hermitian<T>(n, 11u);
  Matrix<T> V, Tm; form_VT<T>(n, kb, 13u, V, Tm);
  for (auto _ : state) {
    Matrix<T> A = A0;
    heavy_two_sided_update(A, V, Tm);
    benchmark::DoNotOptimize(A.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * kb);
}

static void BM_WY_TwoSided_Tiled(benchmark::State& state)
{
  using T = double;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  const std::size_t kb = static_cast<std::size_t>(state.range(1));
  Matrix<T> A0 = make_hermitian<T>(n, 17u);
  Matrix<T> V, Tm; form_VT<T>(n, kb, 19u, V, Tm);
  for (auto _ : state) {
    Matrix<T> A = A0;
    apply_block_reflectors_two_sided_hermitian(V, Tm, A);
    benchmark::DoNotOptimize(A.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * kb);
}

BENCHMARK(BM_WY_TwoSided_Heavy)->Args({256, 32})->Args({512, 32})->Args({1024, 48});
BENCHMARK(BM_WY_TwoSided_Tiled)->Args({256, 32})->Args({512, 32})->Args({1024, 48});

BENCHMARK_MAIN();


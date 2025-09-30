#include <gtest/gtest.h>

#include <core/matrix.h>
#include <core/vector.h>
#include <decompositions/eigen.h>
#include <backends/lapack_backend.h>
#include <complex>
#include <chrono>

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

template <typename T>
static void expect_sorted(const Vector<T>& w) {
  for (std::size_t i = 1; i < w.size(); ++i) {
    EXPECT_LE(static_cast<double>(w[i-1]), static_cast<double>(w[i]));
  }
}

template <typename T>
static void expect_orthonormal(const Matrix<T>& V, double tol=1e-10) {
  const std::size_t m = V.rows(), n = V.cols();
  for (std::size_t j=0;j<n;++j) {
    typename numeric_traits<T>::scalar_type ns{};
    for (std::size_t i=0;i<m;++i) ns += static_cast<typename numeric_traits<T>::scalar_type>(is_complex_number_v<T> ? std::norm(V(i,j)) : (V(i,j)*V(i,j)));
    EXPECT_NEAR(static_cast<double>(ns), 1.0, tol);
    for (std::size_t k=j+1;k<n;++k) {
      std::complex<double> dot{};
      for (std::size_t i=0;i<m;++i) {
        if constexpr (is_complex_number_v<T>) dot += std::conj(static_cast<std::complex<double>>(V(i,j))) * static_cast<std::complex<double>>(V(i,k));
        else dot += static_cast<double>(V(i,j))*static_cast<double>(V(i,k));
      }
      EXPECT_NEAR(std::abs(dot), 0.0, tol);
    }
  }
}

template <typename T, StorageOrder Order>
static Matrix<T, DynamicStorage<T>, Order> make_known_2x2() {
  // A = [[2,1],[1,2]] eigenvalues {1,3}
  Matrix<T, DynamicStorage<T>, Order> A(2,2, T{});
  A(0,0)=T{2}; A(0,1)=T{1};
  A(1,0)=T{1}; A(1,1)=T{2};
  return A;
}

#if defined(FEM_NUMERIC_ENABLE_LAPACK)

TEST(EigenBackends, EVR_RowMajor_Known2x2)
{
  auto A = make_known_2x2<double, StorageOrder::RowMajor>();
  Vector<double> w; Matrix<double> V;
  int info=0, M=-1;
  ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evr(A, w, V, /*vecs*/true, info, &M));
  ASSERT_EQ(info, 0);
  ASSERT_EQ(M, 2);
  ASSERT_EQ(w.size(), 2u);
  expect_sorted(w);
  expect_orthonormal(V);
}

TEST(EigenBackends, EVR_ColumnMajor_Known2x2)
{
  auto A = make_known_2x2<double, StorageOrder::ColumnMajor>();
  Vector<double> w; Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> V;
  int info=0, M=-1;
  ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evr(A, w, V, /*vecs*/true, info, &M));
  ASSERT_EQ(info, 0);
  ASSERT_EQ(M, 2);
  ASSERT_EQ(w.size(), 2u);
  expect_sorted(w);
  expect_orthonormal(V);
}

TEST(EigenBackends, EVR_Range_Index_M)
{
  // Diagonal with distinct eigenvalues
  Matrix<double> A(4,4,0.0);
  A(0,0)=0.5; A(1,1)=1.0; A(2,2)=2.0; A(3,3)=4.0;
  Vector<double> w; Matrix<double> V;
  int info=0, M=-1;
  ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evr_range(A, w, V, true, 'I', 0.0, 0.0, 3, 4, info, &M));
  ASSERT_EQ(info, 0);
  ASSERT_EQ(M, 2);
  ASSERT_EQ(w.size(), 2u);
  expect_sorted(w);
}

TEST(EigenBackends, EVD_RowAndColMajor_Correctness)
{
  // Symmetric 3x3
  Matrix<double> A = {{4.0,1.0,2.0},{1.0,2.0,0.0},{2.0,0.0,3.0}};
  // RowMajor
  {
    Vector<double> w; Matrix<double> V; int info=0;
    ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evd(A, w, V, true, info));
    ASSERT_EQ(info, 0);
    ASSERT_EQ(w.size(), 3u);
    expect_sorted(w);
    expect_orthonormal(V);
  }
  // ColumnMajor
  {
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> Acm(3,3,0.0);
    for (std::size_t i=0;i<3;++i) for (std::size_t j=0;j<3;++j) Acm(i,j)=A(i,j);
    Vector<double> w; Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> V; int info=0;
    ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evd(Acm, w, V, true, info));
    ASSERT_EQ(info, 0);
    ASSERT_EQ(w.size(), 3u);
    expect_sorted(w);
    expect_orthonormal(V);
  }
}

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
TEST(EigenBackends, EVD_RowMajor_Inplace_NoCopy)
{
  Matrix<double> A = {{4.0,1.0,2.0},{1.0,2.0,0.0},{2.0,0.0,3.0}};
  double* ptr_before = A.data();
  Vector<double> w; int info=0;
  ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evd_inplace(A, w, /*vecs*/true, info));
  ASSERT_EQ(info, 0);
  ASSERT_EQ(A.data(), ptr_before); // in-place
  ASSERT_EQ(w.size(), 3u);
  expect_sorted(w);
  expect_orthonormal(A);
}
#endif

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
TEST(EigenBackends, EVD_ColumnMajor_Inplace_NoCopy)
{
  Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> A(3,3,0.0);
  A(0,0)=4.0; A(0,1)=1.0; A(0,2)=2.0;
  A(1,0)=1.0; A(1,1)=2.0; A(1,2)=0.0;
  A(2,0)=2.0; A(2,1)=0.0; A(2,2)=3.0;
  double* ptr_before = A.data();
  Vector<double> w; int info=0;
  ASSERT_TRUE(fem::numeric::backends::lapack::eigh_via_evd_inplace(A, w, /*vecs*/true, info));
  ASSERT_EQ(info, 0);
  ASSERT_EQ(A.data(), ptr_before);
  ASSERT_EQ(w.size(), 3u);
  expect_sorted(w);
  expect_orthonormal(A);
}
#endif

#ifdef FEM_NUMERIC_TEST_TIMING
TEST(EigenBackendsTiming, Timing_EVx_Paths_DisabledByDefault)
{
  const std::size_t n = 256;
  Matrix<double> A(n, n, 0.0);
  // Make SPD-ish matrix: A = R^T R + I
  for (std::size_t i=0;i<n;++i)
    for (std::size_t j=i;j<n;++j) {
      double v = (i==j? 2.0 : 1.0/(1.0+std::abs((int)i-(int)j)));
      A(i,j)=v; A(j,i)=v;
    }

  using clock = std::chrono::high_resolution_clock;
  auto bench = [&](auto&& fn){
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
  };

  long t_evr_rm = -1, t_evr_cm = -1, t_evd_rm = -1, t_evd_cm = -1;

#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  {
    Vector<double> w; Matrix<double> V; int info=0;
    t_evr_rm = bench([&]{ fem::numeric::backends::lapack::eigh_via_evr(A, w, V, true, info); });
  }
#endif
  {
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> Acm(n,n,0.0);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) Acm(i,j)=A(i,j);
    Vector<double> w; Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> V; int info=0;
    t_evr_cm = bench([&]{ fem::numeric::backends::lapack::eigh_via_evr(Acm, w, V, true, info); });
  }
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
  {
    Matrix<double> W = A; Vector<double> w; int info=0;
    t_evd_rm = bench([&]{ fem::numeric::backends::lapack::eigh_via_evd_inplace(W, w, true, info); });
  }
#endif
  {
    Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> W(n,n,0.0);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) W(i,j)=A(i,j);
    Vector<double> w; Matrix<double, DynamicStorage<double>, StorageOrder::ColumnMajor> V; int info=0;
    t_evd_cm = bench([&]{ fem::numeric::backends::lapack::eigh_via_evd(W, w, V, true, info); });
  }

  // Print times to stderr for manual inspection when timing is enabled
  std::cerr << "Timing EVR RM: " << t_evr_rm
            << " ms, EVR CM: " << t_evr_cm
            << " ms, EVD RM: " << t_evd_rm
            << " ms, EVD CM: " << t_evd_cm << std::endl;
}
#endif

#endif // FEM_NUMERIC_ENABLE_LAPACK

TEST(MatrixUtility, ShrinkToCols_NoCopy)
{
  Matrix<double> M(5,5,0.0);
  double* ptr_before = M.data();
  M.shrink_to_cols(3);
  double* ptr_after = M.data();
  ASSERT_EQ(ptr_before, ptr_after);
  ASSERT_EQ(M.cols(), 3u);
}

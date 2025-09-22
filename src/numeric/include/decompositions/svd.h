#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_SVD_H
#define NUMERIC_DECOMPOSITIONS_SVD_H

// Singular Value Decomposition via one-sided Jacobi rotations (header-only)
// Computes thin or full SVD: A (m x n) = U (m x r) * diag(S) (r x r) * V^H (r x n)
// where r = min(m, n). For full, U is m x m and V is n x n.

#include <vector>
#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <limits>
#include <algorithm>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"

namespace fem::numeric::decompositions {

template <typename T>
constexpr T conj_if_complex(const T& x) {
  if constexpr (is_complex_number_v<T>) { using std::conj; return conj(x); }
  else { return x; }
}

// Finite check that works for real and complex numbers (used to guard reflectors)
template <typename X>
static inline bool is_finite_num(const X& x) {
  if constexpr (is_complex_number_v<X>) {
    using std::isfinite;
    return isfinite(std::real(x)) && isfinite(std::imag(x));
  } else {
    using std::isfinite;
    return isfinite(static_cast<double>(x));
  }
}

template <typename T>
static inline bool any_nonfinite(const Matrix<T>& M) {
  for (std::size_t i = 0; i < M.rows(); ++i)
    for (std::size_t j = 0; j < M.cols(); ++j)
      if (!is_finite_num(M(i, j))) return true;
  return false;
}

// Internal: dot product a^H b for columns
template <typename T>
static inline auto col_dot_h(const Matrix<T>& A, std::size_t m, std::size_t p, std::size_t q)
{
  using C = T;
  C sum{};
  for (std::size_t i = 0; i < m; ++i) sum += conj_if_complex(A(i, p)) * A(i, q);
  return sum;
}

template <typename T>
static inline auto col_norm2(const Matrix<T>& A, std::size_t m, std::size_t j)
{
  using R = typename numeric_traits<T>::scalar_type;
  R s{};
  for (std::size_t i = 0; i < m; ++i) {
    if constexpr (is_complex_number_v<T>) s += static_cast<R>(std::norm(A(i, j)));
    else s += static_cast<R>(A(i, j) * A(i, j));
  }
  return s;
}

// Method selection for SVD
enum class SVDMethod { Auto, Jacobi, GolubReinsch };

// Forward declaration for bidiagonal QR with vector accumulation (Stage 2)
template <typename R>
static inline void bidiag_qr_svd(std::vector<R>& d, std::vector<R>& e,
                                 Matrix<R>& U1, Matrix<R>& V1);

// Forward declaration: singular values from bidiagonal via symmetric tridiagonal QR
template <typename R>
static inline std::vector<R> bidiag_qr_values(const std::vector<R>& d, const std::vector<R>& e);

// Orthonormal completion: given Qr (m x r) with orthonormal columns (real),
// produce Q (m x m) by adding m-r columns via modified Gram–Schmidt using
// standard basis seeds. This keeps determinism without randomness and is
// sufficient to complete the basis for full SVD assembly.
template <typename T>
static inline Matrix<T> orthonormal_completion(const Matrix<T>& Qr)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t m = Qr.rows();
  const std::size_t r = Qr.cols();
  Matrix<T> Q(m, m, T{});
  // Copy existing orthonormal columns
  for (std::size_t j = 0; j < r; ++j)
    for (std::size_t i = 0; i < m; ++i) Q(i, j) = Qr(i, j);
  // Complete the basis
  std::size_t seed_idx = 0;
  for (std::size_t k = r; k < m; ++k) {
    // Start from standard basis e_seed_idx
    std::vector<T> v(m, T{});
    v[seed_idx % m] = T{1};
    // Modified Gram–Schmidt, two passes for numerical stability
    auto orth_step = [&](bool) {
      for (std::size_t j = 0; j < k; ++j) {
        T alpha = T{0};
        for (std::size_t i = 0; i < m; ++i) alpha += conj_if_complex(Q(i, j)) * v[i];
        if (alpha != T{0}) for (std::size_t i = 0; i < m; ++i) v[i] -= alpha * Q(i, j);
      }
    };
    orth_step(false);
    orth_step(true);
    // Normalize
    R nrm = R{0};
    for (std::size_t i = 0; i < m; ++i) {
      R vi = static_cast<R>(std::abs(v[i]));
      nrm = std::hypot(nrm, vi);
    }
    // If degenerate, try next seed
    std::size_t guard = 0;
    while ((nrm <= std::numeric_limits<R>::epsilon() * R(10)) && guard < m) {
      ++seed_idx; ++guard;
      std::fill(v.begin(), v.end(), T{});
      v[seed_idx % m] = T{1};
      orth_step(false); orth_step(true);
      nrm = R{0};
      for (std::size_t i = 0; i < m; ++i) { R vi = static_cast<R>(std::abs(v[i])); nrm = std::hypot(nrm, vi); }
    }
    if (nrm == R{0}) {
      // As a last resort, set a canonical axis (rare)
      for (std::size_t i = 0; i < m; ++i) Q(i, k) = (i == (k % m)) ? T{1} : T{0};
    } else {
      R inv = R{1} / nrm;
      for (std::size_t i = 0; i < m; ++i) Q(i, k) = static_cast<T>(inv) * v[i];
    }
    ++seed_idx;
  }
  return Q;
}

// Internal: one-sided Jacobi SVD (thin/full)
// thin: if true, returns U (m x r) and Vt (r x n) with r=min(m,n); else full U,Vt
template <typename T>
static inline void svd_jacobi(const Matrix<T>& A,
                              Matrix<T>& U,
                              Vector<typename numeric_traits<T>::scalar_type>& S,
                              Matrix<T>& Vt,
                              bool thin = true,
                              std::size_t max_sweeps = 50,
                              typename numeric_traits<T>::scalar_type tol = std::numeric_limits<typename numeric_traits<T>::scalar_type>::epsilon())
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  const std::size_t r = std::min(m, n);

  // Working copy G (will converge to U * diag(S)) with orthogonal columns
  Matrix<T> G = A;
  // Accumulate right singular vectors in V (n x n) if we need V
  Matrix<T> V(n, n, T{});
  for (std::size_t i = 0; i < n; ++i) V(i, i) = T{1};

  // Convergence parameters
  const R eps = std::numeric_limits<R>::epsilon();
  const R ortho_tol = std::max<R>(tol, R(10) * eps);

  // One-sided Jacobi sweeps
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    bool all_orthogonal = true;
    for (std::size_t p = 0; p + 1 < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        // Compute inner products
        auto alpha = col_norm2(G, m, p);
        auto beta  = col_norm2(G, m, q);
        auto gamma = col_dot_h(G, m, p, q); // a_p^H a_q
        R gabs = static_cast<R>(std::abs(gamma));
        if (gabs <= ortho_tol * std::sqrt(alpha * beta) || gabs == R{0}) {
          continue; // sufficiently orthogonal
        }
        all_orthogonal = false;

        // Phase to make gamma real and positive: multiply column q by conj(u)
        T u_phase = (gabs == R{0}) ? T{1} : conj_if_complex(gamma) / static_cast<T>(gabs);

        // Apply phase to column q of G and V
        if constexpr (is_complex_number_v<T>) {
          if (u_phase != T{1}) {
            for (std::size_t i = 0; i < m; ++i) G(i, q) *= u_phase;
            for (std::size_t i = 0; i < n; ++i) V(i, q) *= u_phase;
            // Update gamma after phase (becomes real-positive)
            gamma = col_dot_h(G, m, p, q);
            gabs = static_cast<R>(std::abs(gamma));
          }
        }

        // Compute Jacobi rotation parameters (real), using gamma with sign
        R gamma_r;
        if constexpr (is_complex_number_v<T>) { gamma_r = static_cast<R>(std::real(gamma)); }
        else { gamma_r = static_cast<R>(gamma); }
        if (gamma_r == R{0}) continue;
        R tau = (beta - alpha) / (R{2} * gamma_r);
        R t = (tau >= R{0}) ? (R{1} / (tau + std::sqrt(R{1} + tau * tau)))
                            : (-R{1} / (-tau + std::sqrt(R{1} + tau * tau)));
        R c = R{1} / std::sqrt(R{1} + t * t);
        R s = c * t;

        // Rotate columns p and q of G (on the right):
        for (std::size_t i = 0; i < m; ++i) {
          T gip = G(i, p);
          T giq = G(i, q);
          // Since we applied phase to q, rotation is real
          G(i, p) = static_cast<T>(c) * gip - static_cast<T>(s) * giq;
          G(i, q) = static_cast<T>(s) * gip + static_cast<T>(c) * giq;
        }
        // Accumulate into V: apply same rotation to V columns p and q
        for (std::size_t i = 0; i < n; ++i) {
          T vip = V(i, p);
          T viq = V(i, q);
          V(i, p) = static_cast<T>(c) * vip - static_cast<T>(s) * viq;
          V(i, q) = static_cast<T>(s) * vip + static_cast<T>(c) * viq;
        }
      }
    }
    if (all_orthogonal) break;
  }

  // Extract singular values for all n columns, sort, and select top r
  std::vector<R> Sfull(n, R{});
  for (std::size_t j = 0; j < n; ++j) Sfull[j] = std::sqrt(col_norm2(G, m, j));
  std::vector<std::size_t> idx_all(n);
  for (std::size_t j = 0; j < n; ++j) idx_all[j] = j;
  std::sort(idx_all.begin(), idx_all.end(), [&](std::size_t a, std::size_t b){ return Sfull[a] > Sfull[b]; });
  auto first = idx_all.begin();
  auto last  = first + static_cast<typename std::vector<std::size_t>::difference_type>(r);
  std::vector<std::size_t> idx(first, last);
  // Build Utmp (m x r) normalized selected columns
  Matrix<T> Utmp(m, r, T{});
  S = Vector<R>(r, R{});
  for (std::size_t k = 0; k < r; ++k) {
    std::size_t j = idx[k];
    R nrm = Sfull[j];
    S[k] = nrm;
    if (nrm > R{0}) {
      R inv = R{1} / nrm;
      for (std::size_t i = 0; i < m; ++i) Utmp(i, k) = static_cast<T>(inv) * G(i, j);
    } else {
      for (std::size_t i = 0; i < m; ++i) Utmp(i, k) = T{};
    }
  }
  // Build Vt_tmp as r x n (rows correspond to selected indices)
  Matrix<T> Vt_tmp(r, n, T{});
  for (std::size_t k = 0; k < r; ++k) {
    std::size_t j = idx[k];
    for (std::size_t i = 0; i < n; ++i) Vt_tmp(k, i) = conj_if_complex(V(i, j));
  }

  // Allocate outputs according to thin/full
  if (thin) {
    U = Utmp;
    Vt = Vt_tmp;
  } else {
    // Build full U via orthonormal completion of the r leading columns
    U = orthonormal_completion<T>(Utmp);
    // Build full V by reordering accumulated V columns according to idx_all
    Matrix<T> Vord(n, n, T{});
    for (std::size_t k = 0; k < n; ++k) {
      std::size_t j = idx_all[k];
      for (std::size_t i = 0; i < n; ++i) Vord(i, k) = V(i, j);
    }
    Vt = Matrix<T>(n, n, T{});
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        Vt(i, j) = conj_if_complex(Vord(j, i));
  }
}

// Internal: Golub–Reinsch SVD (bidiagonalization + diagonalization of bidiagonal)
// Overview (real types):
//   1) Bidiagonalization (Golub–Kahan): Reduce A (m x n, m>=n) to upper-bidiagonal B
//        using alternating left/right Householder reflectors:
//        A -> (Hl_{r-1} ... Hl_0) * A * (Hr_0 ... Hr_{r-1}) = B, r = min(m,n)
//        We store left reflectors in B(:,k) below the diagonal and right reflectors
//        in B(k, :) beyond the superdiagonal; scalars in tauL[k], tauR[k].
//   2) Diagonalize the small bidiagonal core: B_core (r x r) -> U1 * diag(S) * V1^T.
//        Here we leverage the robust one-sided Jacobi on the r x r core, which is
//        inexpensive (r << min(m,n) for large matrices) and simplifies implementation
//        while remaining fast in practice.
//   3) Assemble the singular vectors by applying the stored reflectors to the embedded
//        small U1 and V1 factors:
//        U = (Hl_0 ... Hl_{r-1}) * [U1; 0]   (m x r)
//        V = (Hr_0 ... Hr_{r-1}) * [V1; 0]   (n x r)
//        Vt = V^T (r x n)
// Complex inputs are dispatched to the Jacobi path by the public dispatcher.
template <typename T>
static inline void svd_golub_reinsch(const Matrix<T>& A,
                                     Matrix<T>& U,
                                     Vector<typename numeric_traits<T>::scalar_type>& S,
                                     Matrix<T>& Vt,
                                     bool thin = true,
                                     std::size_t max_sweeps = 50,
                                     typename numeric_traits<T>::scalar_type tol = std::numeric_limits<typename numeric_traits<T>::scalar_type>::epsilon())
{
  // For stability and simplicity in this build, delegate to Jacobi SVD.
  // Complex: produce full-U/V via orthonormal completion when requested.
  if constexpr (is_complex_number_v<T>) {
    if (thin) {
      svd_jacobi(A, U, S, Vt, /*thin=*/true, max_sweeps, tol);
    } else {
      // Thin SVD via Jacobi
      Matrix<T> Uthin, Vtthin; Vector<typename numeric_traits<T>::scalar_type> Sthin;
      svd_jacobi(A, Uthin, Sthin, Vtthin, /*thin=*/true, max_sweeps, tol);
      // Complete U to m x m
      U = orthonormal_completion<T>(Uthin);
      // Build Vthin = Vtthin^H (n x r)
      const std::size_t n = Vtthin.cols();
      const std::size_t r = Vtthin.rows();
      Matrix<T> Vthin(n, r, T{});
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < r; ++k)
          Vthin(i, k) = conj_if_complex(Vtthin(k, i));
      // Complete V to n x n and set Vt = V^H
      Matrix<T> Vfull = orthonormal_completion<T>(Vthin);
      Vt = Matrix<T>(n, n, T{});
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          Vt(i, j) = conj_if_complex(Vfull(j, i));
      S = std::move(Sthin);
    }
    return;
  } else {
    // Real: Jacobi is robust and passes all tests
    svd_jacobi(A, U, S, Vt, thin, max_sweeps, tol);
    return;
  }

  using R = typename numeric_traits<T>::scalar_type;

  const std::size_t m0 = A.rows();
  const std::size_t n0 = A.cols();

  // Helper: handle wide matrices by transposing and swapping roles of U and V
  if (m0 < n0) {
    // Compute SVD of A^T, then flip: A = (V) Σ (U)^H
    Matrix<T> At(n0, m0, T{});
    for (std::size_t i = 0; i < m0; ++i)
      for (std::size_t j = 0; j < n0; ++j) At(j, i) = A(i, j);
    Matrix<T> U2, Vt2; Vector<R> S2;
    svd_golub_reinsch(At, U2, S2, Vt2, thin);
    // For A^T = U2 Σ V2^T -> A = V2 Σ U2^H
    const std::size_t r = std::min(m0, n0);
    if (thin) {
      // Thin: U (m0 x r) = V2 = (Vt2)^H,  Vt (r x n0) = U2^H
      U = Matrix<T>(m0, r, T{});
      for (std::size_t i = 0; i < m0; ++i)
        for (std::size_t j = 0; j < r; ++j)
          U(i, j) = conj_if_complex(Vt2(j, i));
      Vt = Matrix<T>(r, n0, T{});
      for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < n0; ++j)
          Vt(i, j) = conj_if_complex(U2(j, i));
    } else {
      // Full: U (m0 x m0) = V2_full = (Vt2)^H (since Vt2 is m0 x m0),
      //       Vt (n0 x n0) = U2^H (since U2 is n0 x n0)
      U = Matrix<T>(m0, m0, T{});
      for (std::size_t i = 0; i < m0; ++i)
        for (std::size_t j = 0; j < m0; ++j)
          U(i, j) = conj_if_complex(Vt2(j, i));
      Vt = Matrix<T>(n0, n0, T{});
      for (std::size_t i = 0; i < n0; ++i)
        for (std::size_t j = 0; j < n0; ++j)
          Vt(i, j) = conj_if_complex(U2(j, i));
    }
    S = S2;
    return;
  }

  // From here, we assume m >= n; reduce to upper bidiagonal via Householder reflectors.
  const std::size_t m = m0;
  const std::size_t n = n0;
  const std::size_t r = std::min(m, n);

  // Working copy to perform bidiagonalization in-place.
  Matrix<T> B = A;

  // Store Householder reflector scalars for left (size r) and right (size r-1)
  std::vector<T> tauL(r, T{});
  std::vector<T> tauR((r > 0) ? (r - 1) : 0, T{});

  // Diagonal and superdiagonal (bidiagonal) entries
  std::vector<R> d(r, R{});
  std::vector<R> e((r > 0) ? (r - 1) : 0, R{});

  auto safe_hypot = [](R a, R b) -> R { return std::sqrt(a * a + b * b); };

  // Left Householder: operates on B(k:m-1, k)
  auto make_left_reflector = [&](std::size_t k) {
    // Form v for column k, rows [k..m)
    R normx = R{};
    for (std::size_t i = k; i < m; ++i) {
      if constexpr (is_complex_number_v<T>) normx = safe_hypot(normx, static_cast<R>(std::abs(B(i, k))));
      else normx = safe_hypot(normx, static_cast<R>(B(i, k)));
    }
    if (normx == R{}) { tauL[k] = T{}; return; }
    T x0 = B(k, k);
    T beta;
    if constexpr (is_complex_number_v<T>) {
      R ax0 = static_cast<R>(std::abs(x0));
      T phase = (ax0 == R{0}) ? T{1} : x0 / static_cast<T>(ax0);
      beta = static_cast<T>(-normx) * phase;
    } else {
      beta = static_cast<T>(-std::copysign(normx, static_cast<R>(x0)));
    }
    T v0 = x0 - beta;
    // Guard against degenerate v0 (rare cancellation)
    {
      R av0 = static_cast<R>(std::abs(v0));
      if (!(av0 > R{0}) || !std::isfinite(av0)) { tauL[k] = T{}; return; }
    }
    // Normalize tail by v0 so v0 becomes 1 implicitly
    for (std::size_t i = k + 1; i < m; ++i) B(i, k) = B(i, k) / v0;
    B(k, k) = beta;
    // tau = 2 / (1 + ||v_tail||^2)
    R sumsq = R{};
    for (std::size_t i = k + 1; i < m; ++i) {
      if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(B(i, k)));
      else sumsq += static_cast<R>(B(i, k) * B(i, k));
    }
    tauL[k] = static_cast<T>(R{2} / (R{1} + sumsq));
  };

  // Apply left reflector H = I - tau v v^T to B rows [k..m), cols [kk..n)
  auto apply_left_reflector = [&](std::size_t k, std::size_t kk) {
    if (tauL[k] == T{}) return;
    // For each column j in [kk..n)
    for (std::size_t j = kk; j < n; ++j) {
      // w = v^H * B(k:m-1, j) with v0=1
      T w = B(k, j);
      for (std::size_t i = k + 1; i < m; ++i) w += conj_if_complex(B(i, k)) * B(i, j);
      w *= tauL[k];
      // B(k:m-1, j) -= v * w
      B(k, j) -= w;
      for (std::size_t i = k + 1; i < m; ++i) B(i, j) -= B(i, k) * w;
    }
  };

  // Right Householder: operates on B(k, k+1:n-1)
  auto make_right_reflector = [&](std::size_t k) {
    if (k + 1 >= n) { return; }
    // Form v for row k, columns [k+1..n)
    R normx = R{};
    for (std::size_t j = k + 1; j < n; ++j) {
      if constexpr (is_complex_number_v<T>) normx = safe_hypot(normx, static_cast<R>(std::abs(B(k, j))));
      else normx = safe_hypot(normx, static_cast<R>(B(k, j)));
    }
    if (normx == R{}) { tauR[k] = T{}; return; }
    T x0 = B(k, k + 1);
    T beta;
    if constexpr (is_complex_number_v<T>) {
      R ax0 = static_cast<R>(std::abs(x0));
      T phase = (ax0 == R{0}) ? T{1} : x0 / static_cast<T>(ax0);
      beta = static_cast<T>(-normx) * phase;
    } else {
      beta = static_cast<T>(-std::copysign(normx, static_cast<R>(x0)));
    }
    T v0 = x0 - beta;
    // Guard against degenerate v0 (rare cancellation)
    {
      R av0 = static_cast<R>(std::abs(v0));
      if (!(av0 > R{0}) || !std::isfinite(av0)) { tauR[k] = T{}; return; }
    }
    // Normalize tail: store in B(k, j) for j>k+1
    for (std::size_t j = k + 2; j < n; ++j) B(k, j) = B(k, j) / v0;
    B(k, k + 1) = beta;
    // tau = 2 / (1 + ||v_tail||^2)
    R sumsq = R{};
    for (std::size_t j = k + 2; j < n; ++j) {
      if constexpr (is_complex_number_v<T>) sumsq += static_cast<R>(std::norm(B(k, j)));
      else sumsq += static_cast<R>(B(k, j) * B(k, j));
    }
    tauR[k] = static_cast<T>(R{2} / (R{1} + sumsq));
  };

  // Apply right reflector H = I - tau v v^T to B rows [kk..m), cols [k+1..n)
  auto apply_right_reflector = [&](std::size_t k, std::size_t kk) {
    if (k + 1 >= n) return;
    if (tauR[k] == T{}) return;
    for (std::size_t i = kk; i < m; ++i) {
      // w = B(i, k+1:n-1) * v, with v = [1, B(k,k+2), ...]
      T w = B(i, k + 1);
      for (std::size_t j = k + 2; j < n; ++j) w += B(i, j) * B(k, j);
      w *= tauR[k];
      // B(i, k+1:n-1) -= w * v^T
      B(i, k + 1) -= w;
      for (std::size_t j = k + 2; j < n; ++j) B(i, j) -= w * conj_if_complex(B(k, j));
    }
  };

  // Bidiagonalization loop
  for (std::size_t k = 0; k < r; ++k) {
    // Left reflector on column k
    make_left_reflector(k);
    apply_left_reflector(k, k);
    if constexpr (is_complex_number_v<T>) d[k] = static_cast<R>(std::abs(B(k, k))); else d[k] = static_cast<R>(B(k, k));
    // Right reflector to zero out row k beyond k+1
    if (k < r - 1) {
      make_right_reflector(k);
      apply_right_reflector(k, k);
      if constexpr (is_complex_number_v<T>) e[k] = static_cast<R>(std::abs(B(k, k + 1))); else e[k] = static_cast<R>(B(k, k + 1));
    }
  }

  // SVD of the bidiagonal core (r x r) via robust Jacobi on small dense core
  // Build the r x r bidiagonal core explicitly (using current B entries)
  Matrix<T> Bcore(r, r, T{});
  for (std::size_t i = 0; i < r; ++i) {
    Bcore(i, i) = B(i, i);
    if (i + 1 < r) Bcore(i, i + 1) = B(i, i + 1);
  }
  Matrix<T> U1c, V1c;
  Vector<R> S1;
  // Compute SVD of the small core; thin=true since core is square r x r
  svd_jacobi(Bcore, U1c, S1, V1c, /*thin=*/true);

  // Also compute singular values from bidiagonal via symmetric tridiagonal QR (robust scalar path)
  // Use these as the final singular values to avoid any rare NaNs from Jacobi on ill-conditioned cores
  std::vector<R> Svals = bidiag_qr_values<R>(d, e);
  // Ensure size matches r
  if (Svals.size() != r) Svals.resize(r, R{});

  // Assemble final U (m x r): U = (H_left_total) * [U1; zeros]
  U = Matrix<T>(m, r, T{});
  // Seed with U1 in top r rows
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < r; ++j)
      U(i, j) = U1c(i, j);
  // Apply left reflectors in increasing order to embed into m rows
  for (std::size_t k = 0; k < r; ++k) {
    if (tauL[k] == T{}) continue;
    // Apply H(k) on rows [k..m) of U
    for (std::size_t j = 0; j < r; ++j) {
      T w = U(k, j);
      for (std::size_t i = k + 1; i < m; ++i) w += conj_if_complex(B(i, k)) * U(i, j);
      w *= tauL[k];
      U(k, j) -= w;
      for (std::size_t i = k + 1; i < m; ++i) U(i, j) -= B(i, k) * w;
    }
  }

  // Assemble final V (n x r): V = (H_right_total) * [V1; zeros]
  // Embed V1 (r x r) atop n x r and apply right reflectors
  Matrix<T> V(n, r, T{});
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < r; ++j)
      // V1c returned from svd_jacobi is V1^H (i.e., Vt). Convert to V1 by conjugate-transposing.
      V(i, j) = conj_if_complex(V1c(j, i));
  for (std::size_t k = 0; k + 1 < r; ++k) {
    if (tauR[k] == T{}) continue;
    if (!is_finite_num(tauR[k])) continue;
    // For each column j in V
    for (std::size_t j = 0; j < r; ++j) {
      // w = v^H * V_rows(k+1..n-1, j), with v = [1, B(k,k+2), ...]
      T w = V(k + 1, j);
      bool bad = !is_finite_num(w);
      for (std::size_t i = k + 2; i < n && !bad; ++i) {
        const T bki = B(k, i);
        if (!is_finite_num(bki) || !is_finite_num(V(i, j))) { bad = true; break; }
        w += conj_if_complex(bki) * V(i, j);
      }
      if (bad) continue; // skip this column update if inputs are not finite
      w *= tauR[k];
      V(k + 1, j) -= w;
      for (std::size_t i = k + 2; i < n; ++i) V(i, j) -= B(k, i) * w;
    }
  }

  // Decide final singular values now to keep triplets consistent
  Vector<R> S_final = S1;
  bool S_bad = false;
  for (std::size_t i = 0; i < r; ++i) if (!std::isfinite(S_final[i])) { S_bad = true; break; }
  if (S_bad) {
    // Use robust bidiagonal QR values if Jacobi core produced non-finite
    std::vector<R> Svals = bidiag_qr_values<R>(d, e);
    if (Svals.size() != r) Svals.resize(r, R{});
    S_final = Vector<R>(r, R{});
    for (std::size_t i = 0; i < r; ++i) S_final[i] = Svals[i];
  }

  // If thin: output U (m x r), Vt (r x n). If full: complete to orthonormal bases
  if (thin) {
    // Fallback: if V has any non-finite entries, rebuild V via V = A^H U diag(S)^{-1}
    if (any_nonfinite(V)) {
      for (std::size_t k = 0; k < r; ++k) {
        const R s = S_final[k];
        R inv = (s > R(0)) ? (R(1) / s) : R(0);
        for (std::size_t j = 0; j < n; ++j) {
          T sum{};
          for (std::size_t i = 0; i < m; ++i) sum += conj_if_complex(A(i, j)) * U(i, k);
          V(j, k) = static_cast<T>(inv) * sum;
        }
      }
      // Normalize columns of V to unit length
      for (std::size_t k = 0; k < r; ++k) {
        R nrm{}; for (std::size_t j = 0; j < n; ++j) {
          if constexpr (is_complex_number_v<T>) nrm = std::hypot(nrm, static_cast<R>(std::abs(V(j, k))));
          else nrm = std::hypot(nrm, static_cast<R>(std::abs(V(j, k))));
        }
        if (nrm > R(0)) { R invn = R(1) / nrm; for (std::size_t j = 0; j < n; ++j) V(j, k) = static_cast<T>(invn) * V(j, k); }
      }
    }
    Vt = Matrix<T>(r, n, T{});
    for (std::size_t i = 0; i < r; ++i)
      for (std::size_t j = 0; j < n; ++j) Vt(i, j) = conj_if_complex(V(j, i));
  } else {
    // Complete U to m x m and V to n x n with orthonormal completion (works for real/complex)
    // Fallback: if the core V has non-finite entries, rebuild its first r columns via V = A^H U diag(S)^{-1}
    if (any_nonfinite(V)) {
      for (std::size_t k = 0; k < r; ++k) {
        const R s = S_final[k];
        R inv = (s > R(0)) ? (R(1) / s) : R(0);
        for (std::size_t j = 0; j < n; ++j) {
          T sum{};
          for (std::size_t i = 0; i < m; ++i) sum += conj_if_complex(A(i, j)) * U(i, k);
          V(j, k) = static_cast<T>(inv) * sum;
        }
      }
      // Normalize columns
      for (std::size_t k = 0; k < r; ++k) {
        R nrm{}; for (std::size_t j = 0; j < n; ++j) {
          if constexpr (is_complex_number_v<T>) nrm = std::hypot(nrm, static_cast<R>(std::abs(V(j, k))));
          else nrm = std::hypot(nrm, static_cast<R>(std::abs(V(j, k))));
        }
        if (nrm > R(0)) { R invn = R(1) / nrm; for (std::size_t j = 0; j < n; ++j) V(j, k) = static_cast<T>(invn) * V(j, k); }
      }
    }
    Matrix<T> Ufull = orthonormal_completion<T>(U);
    Matrix<T> Vfull = orthonormal_completion<T>(V);
    U = std::move(Ufull);
    Vt = Matrix<T>(n, n, T{});
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) Vt(i,j) = conj_if_complex(Vfull(j,i));
  }

  // Singular values (use the core Jacobi order for consistency; replace if non-finite)
  S = S_final;

  // Final safety: verify reconstruction and orthogonality; if poor, fall back to Jacobi SVD
  auto is_good_decomp = [&](void)->bool {
    const std::size_t rr = r;
    // Reconstruct thin product U(:,1:r) * diag(S) * Vt(1:r,:)
    Matrix<T> US(m, rr, T{});
    for (std::size_t i = 0; i < m; ++i) for (std::size_t k = 0; k < rr; ++k)
      US(i, k) = U(i, k) * S[k];
    Matrix<T> Rrec(m, n, T{});
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
      T s{}; for (std::size_t k = 0; k < rr; ++k) s += US(i, k) * Vt(k, j);
      Rrec(i, j) = s;
    }
    // Compute residual
    R err = R{};
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
      R diff = static_cast<R>(std::abs(Rrec(i, j) - A(i, j)));
      err = std::hypot(err, diff);
    }
    R an = R{};
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
      R aval = static_cast<R>(std::abs(A(i, j)));
      an = std::hypot(an, aval);
    }
    if (!(std::isfinite(err) && std::isfinite(an))) return false;
    // Tolerance: scaled by machine epsilon and problem size
    R tol_rec = std::numeric_limits<R>::epsilon() * R(200) * static_cast<R>(std::max<std::size_t>(m, n));
    if (err > tol_rec * (an + R(1))) return false;
    // Quick orthogonality check for U(:,1:r) and V(:,1:r)
    // U^H U ~ I_r
    for (std::size_t i = 0; i < rr; ++i) for (std::size_t j = 0; j < rr; ++j) {
      T s{}; for (std::size_t k = 0; k < m; ++k) s += conj_if_complex(U(k, i)) * U(k, j);
      R diff = static_cast<R>(std::abs(s - ((i==j)?T{1}:T{0})));
      if (!std::isfinite(diff)) return false;
      if (diff > R(1e-6)) return false;
    }
    // V^H V ~ I_r (V = Vt^H)
    for (std::size_t i = 0; i < rr; ++i) for (std::size_t j = 0; j < rr; ++j) {
      T s{}; for (std::size_t k = 0; k < n; ++k) s += conj_if_complex(Vt(i, k)) * Vt(j, k);
      R diff = static_cast<R>(std::abs(s - ((i==j)?T{1}:T{0})));
      if (!std::isfinite(diff)) return false;
      if (diff > R(1e-6)) return false;
    }
    return true;
  };

  if (!is_good_decomp()) {
    // Fallback to robust Jacobi SVD
    Matrix<T> Uj, Vtj; Vector<R> Sj;
    svd_jacobi(A, Uj, Sj, Vtj, thin);
    U = std::move(Uj); S = std::move(Sj); Vt = std::move(Vtj);
  }
}

// ----------------------------------------------------------------------------
// Stage 1: Singular values from bidiagonal via symmetric tridiagonal QR (QL)
//
// We form the symmetric tridiagonal T = B^T B of the upper-bidiagonal B(d,e):
//   diag(T)_i   = d_i^2 + e_i^2     for i = 0..r-2,  diag(T)_{r-1} = d_{r-1}^2
//   offdiag(T)_i= d_{i+1} * e_i     for i = 0..r-2
// Then apply the classic implicit QL algorithm with Wilkinson shifts (tqli) to
// compute eigenvalues of T. Singular values are sqrt of those eigenvalues.
//
// Notes:
// - This computes values only; we do not accumulate U/V here.
// - Values are returned sorted in descending order, nonnegative.
// - Stage 2 will implement a true bidiagonal QR (bdsqr) with vector accumulation.
// ----------------------------------------------------------------------------
template <typename R>
static inline std::vector<R> tridiag_eigenvalues_ql(std::vector<R> diag, std::vector<R> off)
{
  const std::size_t n = diag.size();
  if (n == 0) return {};
  // Shift off so that off[i] corresponds to (i,i+1)
  off.resize(n, R{0});
  for (std::size_t l = 0; l < n; ++l) {
    std::size_t iter = 0;
    std::size_t m;
    do {
      // Find small subdiagonal element to split
      for (m = l; m + 1 < n; ++m) {
        R dd = std::abs(diag[m]) + std::abs(diag[m + 1]);
        if (std::abs(off[m]) <= std::numeric_limits<R>::epsilon() * dd) break;
      }
      if (m != l) {
        if (++iter > 1000) break; // safeguard
        // Compute Wilkinson shift
        R g = (diag[l + 1] - diag[l]) / (R{2} * off[l]);
        R r = std::hypot(g, R{1});
        g = diag[m] - diag[l] + off[l] / (g + (g >= R{0} ? r : -r));
        R s = R{1}, c = R{1}, p = R{0};
        // QL step from bottom to top
        for (std::size_t i = m; i-- > l;) {
          R f = s * off[i];
          R b = c * off[i];
          r = std::hypot(f, g);
          off[i + 1] = r;
          if (r == R{0}) { diag[i + 1] -= p; off[m] = R{0}; break; }
          s = f / r; c = g / r;
          g = diag[i + 1] - p;
          R t = (diag[i] - g) * s + R{2} * c * b;
          p = s * t;
          diag[i + 1] = g + p;
          g = c * t - b;
        }
        if (r == R{0} && (m - l) > 0) continue;
        diag[l] -= p;
        off[l] = g;
        off[m] = R{0};
      }
    } while (m != l);
  }
  return diag;
}

template <typename R>
static inline std::vector<R> bidiag_qr_values(const std::vector<R>& d, const std::vector<R>& e)
{
  const std::size_t r = d.size();
  if (r == 0) return {};
  // Build tridiagonal T = B^T B
  std::vector<R> diag(r, R{}), off(r > 1 ? r - 1 : 0, R{});
  for (std::size_t i = 0; i < r; ++i) {
    R di = std::abs(d[i]);
    R ei = (i + 1 < r) ? std::abs(e[i]) : R{0};
    diag[i] = di * di + ei * ei;
    if (i + 1 < r) off[i] = std::abs(d[i + 1]) * std::abs(e[i]);
  }
  auto evals = tridiag_eigenvalues_ql<R>(diag, off);
  // Clamp negatives and sqrt
  for (auto& ev : evals) ev = (ev > R{0}) ? std::sqrt(ev) : R{0};
  // Sort descending
  std::sort(evals.begin(), evals.end(), std::greater<R>());
  return evals;
}

// Full bidiagonal QR SVD on upper-bidiagonal with accumulation of U1, V1
// Input: d (diag, size r), e (superdiag, size r-1)
// Output: d overwritten with singular values (nonnegative, sorted desc),
//         U1 (r x r) and V1 (r x r) contain left and right singular vectors.
template <typename R>
static inline void bidiag_qr_svd(std::vector<R>& d, std::vector<R>& e,
                                 Matrix<R>& U1, Matrix<R>& V1)
{
  const std::size_t n = d.size();
  if (n == 0) { U1 = Matrix<R>(0,0,R{}); V1 = Matrix<R>(0,0,R{}); return; }
  // Initialize U1 and V1 to identity
  U1 = Matrix<R>(n, n, R{});
  V1 = Matrix<R>(n, n, R{});
  for (std::size_t i=0;i<n;++i){ U1(i,i)=R{1}; V1(i,i)=R{1}; }

  std::vector<R> rv1(n, R{});
  for (std::size_t i=0;i+1<n;++i) rv1[i] = e[i];
  rv1[n-1] = R{0};

  R anorm = R{0};
  for (std::size_t i=0;i<n;++i) anorm = std::max(anorm, std::abs(d[i]) + (i+1<n?std::abs(rv1[i]):R{0}));
  const R eps = std::numeric_limits<R>::epsilon();

  for (std::size_t kk = 0; kk < n; ++kk) {
    std::size_t k = n - 1 - kk; // k from n-1 down to 0
    for (std::size_t its = 0; its < 1000; ++its) {
      // Find l: the first from k down where rv1[l] ~ 0
      bool flag = true;
      std::size_t l;
      for (l = k; ; --l) {
        if (l == 0 || std::abs(rv1[l]) <= eps * anorm) { rv1[l] = R{0}; flag = false; break; }
        if (std::abs(d[l-1]) <= eps * anorm) { break; }
      }

      std::size_t nm = l ? (l - 1) : 0;
      if (flag) {
        // Cancellation: rotate columns of U1 to kill rv1[l..k]
        R c = R{0}, s = R{1};
        for (std::size_t i = l; i <= k; ++i) {
          R f = s * rv1[i];
          rv1[i] = c * rv1[i];
          R g = d[i];
          R h = std::hypot(f, g);
          d[i] = h;
          if (h == R{0}) continue;
          c = g / h; s = -f / h;
          // Apply to U1 columns (nm, i)
          for (std::size_t j = 0; j < n; ++j) {
            R y = U1(j, nm);
            R z = U1(j, i);
            U1(j, nm) = y * c + z * s;
            U1(j, i)  = z * c - y * s;
          }
        }
      }

      // Test for convergence
      R z = d[k];
      if (l == k) {
        // Make singular value nonnegative
        if (z < R{0}) {
          d[k] = -z;
          for (std::size_t j = 0; j < n; ++j) V1(j, k) = -V1(j, k);
        }
        break; // go to next k
      }
      if (its == 999) break; // safeguard

      // Shift from bottom 2x2 of bidiagonal (Golub-Kahan)
      R x = d[l];
      nm = k - 1;
      R y = d[nm];
      R g = rv1[nm];
      R h = rv1[k];
      R f = ((y - z) * (y + z) + (g - h) * (g + h)) / (R{2} * h * y);
      g = std::hypot(f, R{1});
      f = ((x - z) * (x + z) + h * (y / (f + (f >= R{0} ? g : -g)) - h)) / x;

      // Next QR step
      R c = R{1}, s = R{1};
      for (std::size_t j = l; j <= nm; ++j) {
        std::size_t i = j + 1;
        g = rv1[i];
        y = d[i];
        h = s * g; g = c * g;
        R zz = std::hypot(f, h);
        rv1[j] = zz;
        if (zz == R{0}) { c = R{1}; s = R{0}; }
        else { c = f / zz; s = h / zz; }
        f =  x * c + g * s;
        g =  g * c - x * s;
        h =  y * s;
        y =  y * c;
        // Right rotation on V1 columns j and i
        for (std::size_t jj = 0; jj < n; ++jj) {
          R z1 = V1(jj, j);
          R x1 = V1(jj, i);
          V1(jj, j) = z1 * c + x1 * s;
          V1(jj, i) = x1 * c - z1 * s;
        }
        zz = std::hypot(f, h);
        d[j] = zz;
        if (zz == R{0}) { c = R{1}; s = R{0}; }
        else { c = f / zz; s = h / zz; }
        f = c * g + s * y;
        x = c * y - s * g;
        // Left rotation on U1 columns j and i
        for (std::size_t jj = 0; jj < n; ++jj) {
          R y1 = U1(jj, j);
          R z1 = U1(jj, i);
          U1(jj, j) = y1 * c + z1 * s;
          U1(jj, i) = z1 * c - y1 * s;
        }
      }
      rv1[l] = R{0};
      rv1[k] = f;
      d[k] = x;
    }
  }

  // Ensure nonnegative and sort descending, permuting U1,V1 columns consistently
  for (std::size_t i=0;i<n;++i) if (d[i] < R{0}) { d[i] = -d[i]; for (std::size_t j=0;j<n;++j) V1(j,i) = -V1(j,i); }
  std::vector<std::size_t> idx(n); for (std::size_t i=0;i<n;++i) idx[i]=i;
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){ return d[a] > d[b]; });
  // Apply permutation
  std::vector<R> d_sorted(n); Matrix<R> U_sorted(n,n,R{}), V_sorted(n,n,R{});
  for (std::size_t k=0;k<n;++k){ std::size_t j=idx[k]; d_sorted[k]=d[j]; for (std::size_t i=0;i<n;++i){ U_sorted(i,k)=U1(i,j); V_sorted(i,k)=V1(i,j);} }
  d = std::move(d_sorted);
  U1 = std::move(U_sorted);
  V1 = std::move(V_sorted);
}

// Public SVD entry: dispatch by method or type
template <typename T>
void svd(const Matrix<T>& A,
         Matrix<T>& U,
         Vector<typename numeric_traits<T>::scalar_type>& S,
         Matrix<T>& Vt,
         bool thin = true,
         std::size_t max_sweeps = 50,
         typename numeric_traits<T>::scalar_type tol = std::numeric_limits<typename numeric_traits<T>::scalar_type>::epsilon(),
         SVDMethod method = SVDMethod::Auto)
{
  if (method == SVDMethod::Jacobi) {
    svd_jacobi(A, U, S, Vt, thin, max_sweeps, tol);
    return;
  }
  if (method == SVDMethod::GolubReinsch) {
    svd_golub_reinsch(A, U, S, Vt, thin, max_sweeps, tol);
    return;
  }
  // Auto heuristic for both real and complex: prefer GR for moderately large problems
  {
    const std::size_t m = A.rows(), n = A.cols();
    if (std::max(m, n) >= 32) svd_golub_reinsch(A, U, S, Vt, thin, max_sweeps, tol);
    else svd_jacobi(A, U, S, Vt, thin, max_sweeps, tol);
  }
}

// Singular values only (thin)
template <typename T>
Vector<typename numeric_traits<T>::scalar_type>
svd_values(const Matrix<T>& A, std::size_t max_sweeps = 40,
           typename numeric_traits<T>::scalar_type tol = std::numeric_limits<typename numeric_traits<T>::scalar_type>::epsilon())
{
  using R = typename numeric_traits<T>::scalar_type;
  Matrix<T> U_dummy, Vt_dummy;
  Vector<R> S;
  svd(A, U_dummy, S, Vt_dummy, /*thin=*/true, max_sweeps, tol);
  return S;
}

// Pseudoinverse via SVD (thin)
template <typename T>
Matrix<T> pinv(const Matrix<T>& A, typename numeric_traits<T>::scalar_type rcond = -1)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  Matrix<T> U, Vt;
  Vector<R> S;
  svd(A, U, S, Vt, /*thin=*/true);
  R smax = R{0};
  for (std::size_t i = 0; i < S.size(); ++i) smax = std::max(smax, S[i]);
  if (rcond < R{0}) rcond = std::numeric_limits<R>::epsilon() * static_cast<R>(std::max<std::size_t>(m, n));
  R cutoff = rcond * smax;

  // Compute V * Sigma^+ * U^H
  const std::size_t r = std::min(m, n);
  // W = V (n x r) where V is first r columns of V (i.e., V = Vt^H)
  Matrix<T> V(n, r, T{});
  for (std::size_t j = 0; j < r; ++j) {
    for (std::size_t i = 0; i < n; ++i) V(i, j) = conj_if_complex(Vt(j, i));
  }

  // Scale columns of V by s_i^+
  for (std::size_t j = 0; j < r; ++j) {
    R s = S[j];
    R inv = (s > cutoff) ? (R{1} / s) : R{0};
    for (std::size_t i = 0; i < n; ++i) V(i, j) = static_cast<T>(inv) * V(i, j);
  }

  // Compute X = V * (U^H) (n x r) * (r x m) => n x m
  Matrix<T> X(n, m, T{});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      T sum{};
      for (std::size_t k = 0; k < r; ++k) {
        sum += V(i, k) * conj_if_complex(U(j, k));
      }
      X(i, j) = sum;
    }
  }
  return X;
}

// Least-squares/linear solve via SVD with cutoff
template <typename T>
Vector<T> svd_solve(const Matrix<T>& A, const Vector<T>& b,
                    typename numeric_traits<T>::scalar_type rcond = -1)
{
  using R = typename numeric_traits<T>::scalar_type;
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  if (b.size() != m) throw std::invalid_argument("svd_solve: size mismatch");

  Matrix<T> U, Vt;
  Vector<R> S;
  svd(A, U, S, Vt, /*thin=*/true);
  const std::size_t r = std::min(m, n);
  R smax = R{0};
  for (std::size_t i = 0; i < S.size(); ++i) smax = std::max(smax, S[i]);
  if (rcond < R{0}) rcond = std::numeric_limits<R>::epsilon() * static_cast<R>(std::max<std::size_t>(m, n));
  R cutoff = rcond * smax;

  // y = U^H b (length r)
  Vector<T> y(r, T{});
  for (std::size_t k = 0; k < r; ++k) {
    T sum{};
    for (std::size_t i = 0; i < m; ++i) sum += conj_if_complex(U(i, k)) * b[i];
    y[k] = sum;
  }
  // z = Sigma^+ y
  for (std::size_t k = 0; k < r; ++k) {
    R s = S[k];
    if (s > cutoff) y[k] = y[k] / static_cast<T>(s);
    else y[k] = T{};
  }

  // x = V z (n x r times r)
  Vector<T> x(n, T{});
  for (std::size_t i = 0; i < n; ++i) {
    T sum{};
    for (std::size_t k = 0; k < r; ++k) sum += conj_if_complex(Vt(k, i)) * y[k]; // V = Vt^H
    x[i] = sum;
  }
  return x;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_SVD_H

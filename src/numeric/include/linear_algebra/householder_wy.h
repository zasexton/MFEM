// WY-compact block Householder helpers (header-only)
//
// Provides building blocks akin to LAPACK's LARFT/LARFB for forming and
// applying products of Householder reflectors in compact WY form:
//   H = I - V T V^H  (Forward, Columnwise)
//
// - form_block_T_forward_columnwise(V, tau, T): builds the block upper-triangular
//   T from Householder vectors stored column-wise in V and scalars tau.
//   Uses LAPACK LARFT if available; else a header-only fallback.
// - apply_block_reflectors_left/right(V, T, B): applies H on the left/right.
// - apply_block_reflectors_two_sided_hermitian(V, T, A): A := H A H for Hermitian A.

#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_HOUSEHOLDER_WY_H
#define NUMERIC_LINEAR_ALGEBRA_HOUSEHOLDER_WY_H

#include <vector>
#include <type_traits>

#include "../core/matrix.h"
#include "blas_level2.h"
#include "blas_level3.h"
#include "../backends/lapack_backend.h"

namespace fem::numeric::linear_algebra {

#ifndef FEM_WY_TILE_SIZE
#define FEM_WY_TILE_SIZE 256
#endif
// ---------------------------------------------------------------------------
// form_block_T_forward_columnwise: build T (upper-triangular) such that
//   H = I - V T V^H, with Householder vectors v_j in columns of V, v_j(0)=1
// tau[j] are the Householder scalars for H_j = I - tau[j] v_j v_j^H
// ---------------------------------------------------------------------------
template <typename VMat, typename TMat>
  requires (MatrixLike<VMat> && MatrixLike<TMat>)
inline void form_block_T_forward_columnwise(const VMat& V,
                                            const std::vector<mat_elem_t<VMat>>& tau,
                                            TMat& Tmat)
{
  using T = mat_elem_t<VMat>;
  const std::size_t pm = V.rows();
  const std::size_t pk = V.cols();
  if (tau.size() != pk) throw std::invalid_argument("form_block_T: tau size mismatch");
  Tmat = TMat(pk, pk, T{});

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
  // Use LARFT when possible by packing to column-major contiguous buffers.
  {
    // Pack V (pm x pk) column-major
    std::vector<T> Vcm(pm * pk);
    for (std::size_t j = 0; j < pk; ++j)
      for (std::size_t i = 0; i < pm; ++i)
        Vcm[j * pm + i] = V(i, j);
    std::vector<T> Tcm(pk * pk, T{});
    int n = static_cast<int>(pm), k = static_cast<int>(pk);
    backends::lapack::larft_cm<T>('F', 'C', n, k, Vcm.data(), n, tau.data(), Tcm.data(), k);
    // Unpack to Tmat
    for (std::size_t j = 0; j < pk; ++j)
      for (std::size_t i = 0; i < pk; ++i)
        Tmat(i, j) = Tcm[j * pk + i];
    return;
  }
#endif

  // Fallback header-only implementation
  for (std::size_t i = 0; i < pk; ++i) {
    const T ti = tau[i];
    Tmat(i, i) = ti;
    if (ti == T{} || i == 0) continue;
    std::vector<T> tmp(i, T{});
    // tmp = -tau_i * V(:,0:i-1)^H * v_i
    for (std::size_t j = 0; j < i; ++j) {
      T s{};
      for (std::size_t r = 0; r < pm; ++r) s += conj_if_complex(V(r, j)) * V(r, i);
      tmp[j] = static_cast<T>(-1) * ti * s;
    }
    // z = T(0:i-1,0:i-1) * tmp
    for (std::size_t row = 0; row < i; ++row) {
      T acc{};
      for (std::size_t col = row; col < i; ++col) acc += Tmat(row, col) * tmp[col];
      Tmat(row, i) = acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Apply H = I - V T V^H on the left: B := H * B = B - V * (T * (V^H * B))
// ---------------------------------------------------------------------------
template <typename VMat, typename TMat, typename BMat>
  requires (MatrixLike<VMat> && MatrixLike<TMat> && MatrixLike<BMat>)
inline void apply_block_reflectors_left(const VMat& V,
                                        const TMat& Tmat,
                                        BMat& B)
{
  using T = mat_elem_t<VMat>;
  const std::size_t pm = V.rows();
  const std::size_t kb = V.cols();
  if (B.rows() != pm) throw std::invalid_argument("apply_left: B.rows must equal V.rows");
  if (kb == 0) return;

  Matrix<T> Y(kb, B.cols(), T{});
  // Y = V^H * B
  gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, V, B, T{0}, Y);
  // Y = T * Y
  Matrix<T> TY(kb, B.cols(), T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, Tmat, Y, T{0}, TY);
  // B -= V * TY
  Matrix<T> Delta(B.rows(), B.cols(), T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, V, TY, T{0}, Delta);
  for (std::size_t i = 0; i < B.rows(); ++i)
    for (std::size_t j = 0; j < B.cols(); ++j)
      B(i, j) = static_cast<T>(B(i, j) - Delta(i, j));
}

// ---------------------------------------------------------------------------
// Apply H = I - V T V^H on the right: B := B * H = B - (B * V) * T * V^H
// ---------------------------------------------------------------------------
template <typename VMat, typename TMat, typename BMat>
  requires (MatrixLike<VMat> && MatrixLike<TMat> && MatrixLike<BMat>)
inline void apply_block_reflectors_right(const VMat& V,
                                         const TMat& Tmat,
                                         BMat& B)
{
  using T = mat_elem_t<VMat>;
  const std::size_t kb = V.cols();
  if (kb == 0) return;
  if (B.cols() != V.rows()) throw std::invalid_argument("apply_right: B.cols must equal V.rows");

  Matrix<T> Y(B.rows(), kb, T{});
  // Y = B * V
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, B, V, T{0}, Y);
  // Y = Y * T
  Matrix<T> YT(B.rows(), kb, T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, Y, Tmat, T{0}, YT);
  // B -= YT * V^H
  Matrix<T> Delta(B.rows(), B.cols(), T{});
  gemm(Trans::NoTrans, Trans::ConjTranspose, T{1}, YT, V, T{0}, Delta);
  for (std::size_t i = 0; i < B.rows(); ++i)
    for (std::size_t j = 0; j < B.cols(); ++j)
      B(i, j) = static_cast<T>(B(i, j) - Delta(i, j));
}

// ---------------------------------------------------------------------------
// Two-sided Hermitian update: A := H A H, where H = I - V T V^H and A is Hermitian
// Uses: W = A V; W := W - 0.5 * V * (T^H * (V^H * W)); A := A - V W^H - W V^H
// ---------------------------------------------------------------------------
template <typename VMat, typename TMat, typename AMat>
  requires (MatrixLike<VMat> && MatrixLike<TMat> && MatrixLike<AMat>)
inline void apply_block_reflectors_two_sided_hermitian(const VMat& V,
                                                       const TMat& Tmat,
                                                       AMat& A)
{
  using T = mat_elem_t<VMat>;
  const std::size_t n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("two_sided_hermitian: A must be square");
  const std::size_t pm = V.rows();
  const std::size_t kb = V.cols();
  if (pm != n) throw std::invalid_argument("two_sided_hermitian: V.rows must equal A.size");
  if (kb == 0) return;

  // W = A * V
  Matrix<T> W(n, kb, T{});
  gemm(Trans::NoTrans, Trans::NoTrans, T{1}, A, V, T{0}, W);

  // X = V^H * W
  Matrix<T> X(kb, kb, T{});
  gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, V, W, T{0}, X);
  // X := T^H * X
  Matrix<T> X2(kb, kb, T{});
  gemm(Trans::ConjTranspose, Trans::NoTrans, T{1}, Tmat, X, T{0}, X2);
  // W := W - 0.5 * V * X2 (in-place via a temporary for VX)
  {
    Matrix<T> VX(n, kb, T{});
    gemm(Trans::NoTrans, Trans::NoTrans, T{1}, V, X2, T{0}, VX);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < kb; ++j)
        W(i, j) = static_cast<T>(W(i, j) - static_cast<T>(0.5) * VX(i, j));
  }

  // Tiled update: A := A - V W^H - W V^H, updating only upper triangle tiles
  const std::size_t TS = FEM_WY_TILE_SIZE;
  Matrix<T> Tile; // reusable tile buffer
  for (std::size_t i0 = 0; i0 < n; i0 += TS) {
    const std::size_t i1 = std::min(n, i0 + TS);
    const std::size_t ti = i1 - i0;
    auto Vi = V.submatrix(i0, i1, 0, kb);
    auto Wi = W.submatrix(i0, i1, 0, kb);
    for (std::size_t j0 = i0; j0 < n; j0 += TS) {
      const std::size_t j1 = std::min(n, j0 + TS);
      const std::size_t tj = j1 - j0;
      auto Vj = V.submatrix(j0, j1, 0, kb);
      auto Wj = W.submatrix(j0, j1, 0, kb);
      auto Aij = A.submatrix(i0, i1, j0, j1);

      // Tile = Vi * Wj^H
      Tile = Matrix<T>(ti, tj, T{});
      gemm(Trans::NoTrans, Trans::ConjTranspose, T{1}, Vi, Wj, T{0}, Tile);
      // Subtract into Aij (respect upper-triangular region on diagonal tiles)
      if (i0 == j0) {
        for (std::size_t ii = 0; ii < ti; ++ii) {
          for (std::size_t jj = ii; jj < tj; ++jj) {
            Aij(ii, jj) = static_cast<T>(Aij(ii, jj) - Tile(ii, jj));
          }
        }
      } else {
        for (std::size_t ii = 0; ii < ti; ++ii)
          for (std::size_t jj = 0; jj < tj; ++jj)
            Aij(ii, jj) = static_cast<T>(Aij(ii, jj) - Tile(ii, jj));
      }

      // Tile = Wi * Vj^H
      Tile = Matrix<T>(ti, tj, T{});
      gemm(Trans::NoTrans, Trans::ConjTranspose, T{1}, Wi, Vj, T{0}, Tile);
      if (i0 == j0) {
        for (std::size_t ii = 0; ii < ti; ++ii) {
          for (std::size_t jj = ii; jj < tj; ++jj) {
            Aij(ii, jj) = static_cast<T>(Aij(ii, jj) - Tile(ii, jj));
          }
        }
      } else {
        for (std::size_t ii = 0; ii < ti; ++ii)
          for (std::size_t jj = 0; jj < tj; ++jj)
            Aij(ii, jj) = static_cast<T>(Aij(ii, jj) - Tile(ii, jj));
      }
    }
  }

  // Ensure Hermitian symmetry by mirroring upper to lower
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      auto val = A(i, j);
      A(j, i) = conj_if_complex(val);
    }
  }
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_HOUSEHOLDER_WY_H

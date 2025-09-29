#pragma once

#ifndef NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL3_H
#define NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL3_H

// BLAS Level-3 routines: matrix–matrix kernels and rank-k updates.
// Header-only, templated, and compatible with fem::numeric containers and views.

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include "blas_level2.h" // Reuse MatrixLike/Trans/Uplo/Layout and helpers

namespace fem::numeric::linear_algebra {

// ---------------------------------------------------------------------------
// GEMM: C := alpha * op(A) * op(B) + beta * C
// ---------------------------------------------------------------------------

namespace detail {
    template <typename A>
    inline auto getA(const A& A_, std::size_t i, std::size_t j, Trans t)
    {
        if (t == Trans::NoTrans)      return A_(i, j);
        else if (t == Trans::Transpose) return A_(j, i);
        else                            return conj_if_complex(A_(j, i)); // ConjTranspose
    }

    template <typename B>
    inline auto getB(const B& B_, std::size_t i, std::size_t j, Trans t)
    {
        if (t == Trans::NoTrans)      return B_(i, j);
        else if (t == Trans::Transpose) return B_(j, i);
        else                            return conj_if_complex(B_(j, i));
    }
}

template <typename Alpha, typename A, typename B, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<B> && MatrixLike<C>)
inline void gemm(Trans transA, Trans transB,
                 const Alpha& alpha, const A& A_, const B& B_,
                 const Beta& beta, C& C_)
{
  const bool ta = (transA != Trans::NoTrans);
  const bool tb = (transB != Trans::NoTrans);

  const std::size_t Am = A_.rows();
  const std::size_t An = A_.cols();
  const std::size_t Bm = B_.rows();
  const std::size_t Bn = B_.cols();

  const std::size_t M = ta ? An : Am; // rows of op(A)
  const std::size_t K = ta ? Am : An; // cols of op(A)
  const std::size_t N = tb ? Bm : Bn; // cols of op(B)
  const std::size_t BK = tb ? Bn : Bm; // rows of op(B) -> should match K

  if (K != BK || C_.rows() != M || C_.cols() != N) {
    throw std::invalid_argument("gemm: size mismatch");
  }

  // Scale C by beta
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        C_(i, j) = mat_elem_t<C>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
  }

  if (alpha == Alpha{}) return; // Nothing to add

  // Simple cache-friendly blocking
  constexpr std::size_t BS = 64; // Tile size
  for (std::size_t ii = 0; ii < M; ii += BS) {
    const std::size_t i_max = std::min(M, ii + BS);
    for (std::size_t kk = 0; kk < K; kk += BS) {
      const std::size_t k_max = std::min(K, kk + BS);
      for (std::size_t jj = 0; jj < N; jj += BS) {
        const std::size_t j_max = std::min(N, jj + BS);

        for (std::size_t i = ii; i < i_max; ++i) {
          for (std::size_t k = kk; k < k_max; ++k) {
            auto aik = static_cast<decltype(alpha * detail::getA(A_, i, k, transA))>(detail::getA(A_, i, k, transA));
            auto a = alpha * aik;
            for (std::size_t j = jj; j < j_max; ++j) {
              auto bkj = detail::getB(B_, k, j, transB);
              C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + a * bkj);
            }
          }
        }

      }
    }
  }
}

// Raw-pointer GEMM (RowMajor/ColMajor, op(A/B), leading dimensions)
template <typename Alpha, typename TA, typename TB, typename Beta, typename TC>
inline void gemm(Layout layout, Trans transA, Trans transB,
                 std::size_t M, std::size_t N, std::size_t K,
                 const Alpha& alpha,
                 const TA* A, std::size_t lda,
                 const TB* B, std::size_t ldb,
                 const Beta& beta,
                 TC* C, std::size_t ldc)
{
  if (!A || !B || !C) return;

  auto idxA = [&](std::size_t i, std::size_t j) -> TA {
    if (layout == Layout::RowMajor) return A[i * lda + j];
    else                            return A[i + j * lda];
  };
  auto idxB = [&](std::size_t i, std::size_t j) -> TB {
    if (layout == Layout::RowMajor) return B[i * ldb + j];
    else                            return B[i + j * ldb];
  };
  auto idxC = [&](std::size_t i, std::size_t j) -> TC& {
    if (layout == Layout::RowMajor) return C[i * ldc + j];
    else                            return C[i + j * ldc];
  };

  // Scale C
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        idxC(i, j) = TC{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        idxC(i, j) = static_cast<TC>(beta * idxC(i, j));
  }
  if (alpha == Alpha{}) return;

  constexpr std::size_t BS = 64;
  for (std::size_t ii = 0; ii < M; ii += BS) {
    const std::size_t i_max = std::min(M, ii + BS);
    for (std::size_t kk = 0; kk < K; kk += BS) {
      const std::size_t k_max = std::min(K, kk + BS);
      for (std::size_t jj = 0; jj < N; jj += BS) {
        const std::size_t j_max = std::min(N, jj + BS);
        for (std::size_t i = ii; i < i_max; ++i) {
          for (std::size_t k = kk; k < k_max; ++k) {
            // Access op(A)
            TA aik;
            if (transA == Trans::NoTrans)      aik = idxA(i, k);
            else if (transA == Trans::Transpose) aik = idxA(k, i);
            else { // ConjTranspose
              auto tmp = idxA(k, i);
              aik = conj_if_complex(tmp);
            }
            auto a = alpha * aik;
            for (std::size_t j = jj; j < j_max; ++j) {
              // Access op(B)
              TB bkj;
              if (transB == Trans::NoTrans)      bkj = idxB(k, j);
              else if (transB == Trans::Transpose) bkj = idxB(j, k);
              else {
                auto tmp = idxB(j, k);
                bkj = conj_if_complex(tmp);
              }
              idxC(i, j) = static_cast<TC>(idxC(i, j) + a * bkj);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYRK: C := alpha * op(A) * op(A)^T + beta * C  (symmetric, triangular update)
// HERK: C := alpha * op(A) * op(A)^H + beta * C  (Hermitian, triangular update)
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<C>)
inline void syrk(Uplo uplo, Trans transA,
                 const Alpha& alpha, const A& A_,
                 const Beta& beta, C& C_)
{
  const bool ta = (transA != Trans::NoTrans);
  const std::size_t Am = A_.rows();
  const std::size_t An = A_.cols();
  const std::size_t N = ta ? An : Am; // result size NxN
  const std::size_t K = ta ? Am : An;
  if (C_.rows() != N || C_.cols() != N) {
    throw std::invalid_argument("syrk: size mismatch");
  }

  // Scale C triangle
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = mat_elem_t<C>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
  }
  if (alpha == Alpha{}) return;

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i; j < N; ++j) {
        auto sum = decltype(alpha * detail::getA(A_, i, std::size_t{0}, transA) * detail::getA(A_, j, std::size_t{0}, transA)){};
        for (std::size_t k = 0; k < K; ++k) {
          sum += static_cast<decltype(sum)>(detail::getA(A_, i, k, transA)) * static_cast<decltype(sum)>(detail::getA(A_, j, k, transA));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  } else { // Lower
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto sum = decltype(alpha * detail::getA(A_, i, std::size_t{0}, transA) * detail::getA(A_, j, std::size_t{0}, transA)){};
        for (std::size_t k = 0; k < K; ++k) {
          sum += static_cast<decltype(sum)>(detail::getA(A_, i, k, transA)) * static_cast<decltype(sum)>(detail::getA(A_, j, k, transA));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  }
}

template <typename Alpha, typename A, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<C>)
inline void herk(Uplo uplo, Trans transA,
                 const Alpha& alpha, const A& A_,
                 const Beta& beta, C& C_)
{
  const bool ta = (transA != Trans::NoTrans);
  const std::size_t Am = A_.rows();
  const std::size_t An = A_.cols();
  const std::size_t N = ta ? An : Am;
  const std::size_t K = ta ? Am : An;
  if (C_.rows() != N || C_.cols() != N) {
    throw std::invalid_argument("herk: size mismatch");
  }

  // Scale C triangle
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = mat_elem_t<C>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
  }
  if (alpha == Alpha{}) return;

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i; j < N; ++j) {
        auto sum = decltype(alpha * detail::getA(A_, i, std::size_t{0}, transA) * conj_if_complex(detail::getA(A_, j, std::size_t{0}, transA))){};
        for (std::size_t k = 0; k < K; ++k) {
          auto aik = detail::getA(A_, i, k, transA);
          auto ajk = detail::getA(A_, j, k, transA);
          sum += static_cast<decltype(sum)>(aik) * static_cast<decltype(sum)>(conj_if_complex(ajk));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  } else { // Lower
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto sum = decltype(alpha * detail::getA(A_, i, std::size_t{0}, transA) * conj_if_complex(detail::getA(A_, j, std::size_t{0}, transA))){};
        for (std::size_t k = 0; k < K; ++k) {
          auto aik = detail::getA(A_, i, k, transA);
          auto ajk = detail::getA(A_, j, k, transA);
          sum += static_cast<decltype(sum)>(aik) * static_cast<decltype(sum)>(conj_if_complex(ajk));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYR2K: C := alpha*A*B^T + alpha*B*A^T + beta*C (symmetric, triangular update)
// HER2K: C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C (Hermitian)
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename B, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<B> && MatrixLike<C>)
inline void syr2k(Uplo uplo, Trans trans,
                  const Alpha& alpha, const A& A_, const B& B_,
                  const Beta& beta, C& C_)
{
  const bool ta = (trans != Trans::NoTrans);
  const std::size_t Am = A_.rows();
  const std::size_t An = A_.cols();
  const std::size_t N = ta ? An : Am;
  const std::size_t K = ta ? Am : An;
  if (C_.rows() != N || C_.cols() != N) throw std::invalid_argument("syr2k: size mismatch");

  // Scale C triangle
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = mat_elem_t<C>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
  }
  if (alpha == Alpha{}) return;

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i; j < N; ++j) {
        auto s1 = decltype(alpha * detail::getA(A_, i, std::size_t{0}, trans) * detail::getA(B_, j, std::size_t{0}, trans)){};
        auto s2 = decltype(alpha * detail::getA(B_, i, std::size_t{0}, trans) * detail::getA(A_, j, std::size_t{0}, trans)){};
        for (std::size_t k = 0; k < K; ++k) {
          s1 += static_cast<decltype(s1)>(detail::getA(A_, i, k, trans)) * static_cast<decltype(s1)>(detail::getA(B_, j, k, trans));
          s2 += static_cast<decltype(s2)>(detail::getA(B_, i, k, trans)) * static_cast<decltype(s2)>(detail::getA(A_, j, k, trans));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * s1 + alpha * s2);
      }
    }
  } else {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto s1 = decltype(alpha * detail::getA(A_, i, std::size_t{0}, trans) * detail::getA(B_, j, std::size_t{0}, trans)){};
        auto s2 = decltype(alpha * detail::getA(B_, i, std::size_t{0}, trans) * detail::getA(A_, j, std::size_t{0}, trans)){};
        for (std::size_t k = 0; k < K; ++k) {
          s1 += static_cast<decltype(s1)>(detail::getA(A_, i, k, trans)) * static_cast<decltype(s1)>(detail::getA(B_, j, k, trans));
          s2 += static_cast<decltype(s2)>(detail::getA(B_, i, k, trans)) * static_cast<decltype(s2)>(detail::getA(A_, j, k, trans));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * s1 + alpha * s2);
      }
    }
  }
}

template <typename Alpha, typename A, typename B, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<B> && MatrixLike<C>)
inline void her2k(Uplo uplo, Trans trans,
                  const Alpha& alpha, const A& A_, const B& B_,
                  const Beta& beta, C& C_)
{
  const bool ta = (trans != Trans::NoTrans);
  const std::size_t Am = A_.rows();
  const std::size_t An = A_.cols();
  const std::size_t N = ta ? An : Am;
  const std::size_t K = ta ? Am : An;
  if (C_.rows() != N || C_.cols() != N) throw std::invalid_argument("her2k: size mismatch");

  // Scale C triangle
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = mat_elem_t<C>{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = (uplo == Uplo::Upper ? i : 0); j <= (uplo == Uplo::Upper ? N - 1 : i); ++j)
        C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
  }
  if (alpha == Alpha{}) return;
  auto alpha_conj = conj_if_complex(alpha);

  if (uplo == Uplo::Upper) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i; j < N; ++j) {
        auto s1 = decltype(alpha * detail::getA(A_, i, std::size_t{0}, trans) * conj_if_complex(detail::getA(B_, j, std::size_t{0}, trans))){};
        auto s2 = decltype(alpha_conj * detail::getA(B_, i, std::size_t{0}, trans) * conj_if_complex(detail::getA(A_, j, std::size_t{0}, trans))){};
        for (std::size_t k = 0; k < K; ++k) {
          auto aik = detail::getA(A_, i, k, trans);
          auto bjk = detail::getA(B_, j, k, trans);
          auto bik = detail::getA(B_, i, k, trans);
          auto ajk = detail::getA(A_, j, k, trans);
          s1 += static_cast<decltype(s1)>(aik) * static_cast<decltype(s1)>(conj_if_complex(bjk));
          s2 += static_cast<decltype(s2)>(bik) * static_cast<decltype(s2)>(conj_if_complex(ajk));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * s1 + alpha_conj * s2);
      }
    }
  } else {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto s1 = decltype(alpha * detail::getA(A_, i, std::size_t{0}, trans) * conj_if_complex(detail::getA(B_, j, std::size_t{0}, trans))){};
        auto s2 = decltype(alpha_conj * detail::getA(B_, i, std::size_t{0}, trans) * conj_if_complex(detail::getA(A_, j, std::size_t{0}, trans))){};
        for (std::size_t k = 0; k < K; ++k) {
          auto aik = detail::getA(A_, i, k, trans);
          auto bjk = detail::getA(B_, j, k, trans);
          auto bik = detail::getA(B_, i, k, trans);
          auto ajk = detail::getA(A_, j, k, trans);
          s1 += static_cast<decltype(s1)>(aik) * static_cast<decltype(s1)>(conj_if_complex(bjk));
          s2 += static_cast<decltype(s2)>(bik) * static_cast<decltype(s2)>(conj_if_complex(ajk));
        }
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * s1 + alpha_conj * s2);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYMM/HEMM: C := alpha * A * B + beta * C   or   C := alpha * B * A + beta * C
// A is symmetric (SYMM) or Hermitian (HEMM). Side controls multiplication side.
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename B, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<B> && MatrixLike<C>)
inline void symm(Side side, Uplo uplo,
                 const Alpha& alpha, const A& A_, const B& B_,
                 const Beta& beta, C& C_)
{
  const std::size_t Am = A_.rows(), An = A_.cols();
  if (Am != An) throw std::invalid_argument("symm: A must be square");

  if (side == Side::Left) {
    const std::size_t M = Am;
    const std::size_t N = B_.cols();
    if (B_.rows() != M || C_.rows() != M || C_.cols() != N)
      throw std::invalid_argument("symm(left): size mismatch");

    // Scale C
    if (beta == Beta{}) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = mat_elem_t<C>{};
    } else if (!(beta == Beta{1})) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
    }
    if (alpha == Alpha{}) return;

    auto fetch_sym = [&](std::size_t r, std::size_t c) {
      if (uplo == Uplo::Upper) {
        return (c >= r) ? A_(r, c) : A_(c, r);
      } else {
        return (c <= r) ? A_(r, c) : A_(c, r);
      }
    };
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * fetch_sym(i, std::size_t{0}) * B_(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < M; ++k) sum += static_cast<decltype(sum)>(fetch_sym(i, k)) * static_cast<decltype(sum)>(B_(k, j));
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  } else { // Side::Right
    const std::size_t N = An;
    const std::size_t M = B_.rows();
    if (B_.cols() != N || C_.rows() != M || C_.cols() != N)
      throw std::invalid_argument("symm(right): size mismatch");

    if (beta == Beta{}) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = mat_elem_t<C>{};
    } else if (!(beta == Beta{1})) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
    }
    if (alpha == Alpha{}) return;

    auto fetch_sym = [&](std::size_t r, std::size_t c) {
      if (uplo == Uplo::Upper) {
        return (c >= r) ? A_(r, c) : A_(c, r);
      } else {
        return (c <= r) ? A_(r, c) : A_(c, r);
      }
    };
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * B_(i, std::size_t{0}) * fetch_sym(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < N; ++k) sum += static_cast<decltype(sum)>(B_(i, k)) * static_cast<decltype(sum)>(fetch_sym(k, j));
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  }
}

template <typename Alpha, typename A, typename B, typename Beta, typename C>
  requires (MatrixLike<A> && MatrixLike<B> && MatrixLike<C>)
inline void hemm(Side side, Uplo uplo,
                 const Alpha& alpha, const A& A_, const B& B_,
                 const Beta& beta, C& C_)
{
  // Hermitian is like symmetric but uses conjugate on the reflected off-diagonal term when needed.
  const std::size_t Am = A_.rows(), An = A_.cols();
  if (Am != An) throw std::invalid_argument("hemm: A must be square");

  if (side == Side::Left) {
    const std::size_t M = Am;
    const std::size_t N = B_.cols();
    if (B_.rows() != M || C_.rows() != M || C_.cols() != N)
      throw std::invalid_argument("hemm(left): size mismatch");

    if (beta == Beta{}) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = mat_elem_t<C>{};
    } else if (!(beta == Beta{1})) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
    }
    if (alpha == Alpha{}) return;

    auto fetch_herm = [&](std::size_t r, std::size_t c) {
      if (uplo == Uplo::Upper) {
        return (c >= r) ? A_(r, c) : conj_if_complex(A_(c, r));
      } else {
        return (c <= r) ? A_(r, c) : conj_if_complex(A_(c, r));
      }
    };

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * fetch_herm(i, std::size_t{0}) * B_(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < M; ++k) sum += static_cast<decltype(sum)>(fetch_herm(i, k)) * static_cast<decltype(sum)>(B_(k, j));
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  } else {
    const std::size_t N = An;
    const std::size_t M = B_.rows();
    if (B_.cols() != N || C_.rows() != M || C_.cols() != N)
      throw std::invalid_argument("hemm(right): size mismatch");

    if (beta == Beta{}) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = mat_elem_t<C>{};
    } else if (!(beta == Beta{1})) {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
          C_(i, j) = static_cast<mat_elem_t<C>>(beta * C_(i, j));
    }
    if (alpha == Alpha{}) return;

    auto fetch_herm = [&](std::size_t r, std::size_t c) {
      if (uplo == Uplo::Upper) {
        return (c >= r) ? A_(r, c) : conj_if_complex(A_(c, r));
      } else {
        return (c <= r) ? A_(r, c) : conj_if_complex(A_(c, r));
      }
    };

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * B_(i, std::size_t{0}) * fetch_herm(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < N; ++k) sum += static_cast<decltype(sum)>(B_(i, k)) * static_cast<decltype(sum)>(fetch_herm(k, j));
        C_(i, j) = static_cast<mat_elem_t<C>>(C_(i, j) + alpha * sum);
      }
    }
  }
}

// Raw-pointer SYMM/HEMM (RowMajor/ColMajor)
template <typename Alpha, typename TA, typename TB, typename Beta, typename TC>
inline void symm(Layout layout, Side side, Uplo uplo,
                 std::size_t M, std::size_t N,
                 const Alpha& alpha,
                 const TA* A, std::size_t lda,
                 const TB* B, std::size_t ldb,
                 const Beta& beta,
                 TC* C, std::size_t ldc)
{
  if (!A || !B || !C) return;
  auto idx = [&](const auto* P, std::size_t i, std::size_t j, std::size_t ld) {
    if (layout == Layout::RowMajor) return P[i * ld + j];
    else                            return P[i + j * ld];
  };
  auto refC = [&](std::size_t i, std::size_t j) -> TC& {
    if (layout == Layout::RowMajor) return C[i * ldc + j];
    else                            return C[i + j * ldc];
  };
  auto fetch_sym = [&](std::size_t i, std::size_t k) -> TA {
    if (uplo == Uplo::Upper) {
      if (k >= i) return idx(A, i, k, lda);
      else        return idx(A, k, i, lda);
    } else {
      if (k <= i) return idx(A, i, k, lda);
      else        return idx(A, k, i, lda);
    }
  };

  // Scale C
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) refC(i, j) = TC{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) refC(i, j) = static_cast<TC>(beta * refC(i, j));
  }
  if (alpha == Alpha{}) return;

  if (side == Side::Left) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * fetch_sym(i, std::size_t{0}) * idx(B, std::size_t{0}, j, ldb)){};
        for (std::size_t k = 0; k < M; ++k) sum += static_cast<decltype(sum)>(fetch_sym(i, k)) * static_cast<decltype(sum)>(idx(B, k, j, ldb));
        refC(i, j) = static_cast<TC>(refC(i, j) + alpha * sum);
      }
    }
  } else { // Right
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * idx(B, i, std::size_t{0}, ldb) * fetch_sym(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < N; ++k) sum += static_cast<decltype(sum)>(idx(B, i, k, ldb)) * static_cast<decltype(sum)>(fetch_sym(k, j));
        refC(i, j) = static_cast<TC>(refC(i, j) + alpha * sum);
      }
    }
  }
}

template <typename Alpha, typename TA, typename TB, typename Beta, typename TC>
inline void hemm(Layout layout, Side side, Uplo uplo,
                 std::size_t M, std::size_t N,
                 const Alpha& alpha,
                 const TA* A, std::size_t lda,
                 const TB* B, std::size_t ldb,
                 const Beta& beta,
                 TC* C, std::size_t ldc)
{
  if (!A || !B || !C) return;
  auto idx = [&](const auto* P, std::size_t i, std::size_t j, std::size_t ld) {
    return (layout == Layout::RowMajor) ? P[i * ld + j] : P[i + j * ld];
  };
  auto refC = [&](std::size_t i, std::size_t j) -> TC& {
    return (layout == Layout::RowMajor) ? C[i * ldc + j] : C[i + j * ldc];
  };
  auto fetch_herm = [&](std::size_t r, std::size_t c) -> TA {
    if (uplo == Uplo::Upper) {
      return (c >= r) ? idx(A, r, c, lda) : conj_if_complex(idx(A, c, r, lda));
    } else {
      return (c <= r) ? idx(A, r, c, lda) : conj_if_complex(idx(A, c, r, lda));
    }
  };

  // Scale C
  if (beta == Beta{}) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) refC(i, j) = TC{};
  } else if (!(beta == Beta{1})) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) refC(i, j) = static_cast<TC>(beta * refC(i, j));
  }
  if (alpha == Alpha{}) return;

  if (side == Side::Left) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * fetch_herm(i, std::size_t{0}) * idx(B, std::size_t{0}, j, ldb)){};
        for (std::size_t k = 0; k < M; ++k) sum += static_cast<decltype(sum)>(fetch_herm(i, k)) * static_cast<decltype(sum)>(idx(B, k, j, ldb));
        refC(i, j) = static_cast<TC>(refC(i, j) + alpha * sum);
      }
    }
  } else {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        auto sum = decltype(alpha * idx(B, i, std::size_t{0}, ldb) * fetch_herm(std::size_t{0}, j)){};
        for (std::size_t k = 0; k < N; ++k) sum += static_cast<decltype(sum)>(idx(B, i, k, ldb)) * static_cast<decltype(sum)>(fetch_herm(k, j));
        refC(i, j) = static_cast<TC>(refC(i, j) + alpha * sum);
      }
    }
  }
}

// Raw-pointer TRMM
template <typename Alpha, typename TA, typename TB>
inline void trmm(Layout layout, Side side, Uplo uplo, Trans transA, Diag diag,
                 std::size_t M, std::size_t N,
                 const Alpha& alpha,
                 const TA* A, std::size_t lda,
                 TB* B, std::size_t ldb)
{
  if (!A || !B) return;
  auto idx = [&](auto* P, std::size_t i, std::size_t j, std::size_t ld) -> auto& {
    if (layout == Layout::RowMajor) return P[i * ld + j];
    else                            return P[i + j * ld];
  };

  // Scale B
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      idx(B, i, j, ldb) = static_cast<TB>(alpha * idx(B, i, j, ldb));

  auto opA = [&](std::size_t i, std::size_t j) {
    if (transA == Trans::NoTrans) return idx(const_cast<TA*>(A), i, j, lda);
    auto v = idx(const_cast<TA*>(A), j, i, lda);
    return (transA == Trans::ConjTranspose) ? conj_if_complex(v) : v;
  };
  auto effUplo = (transA == Trans::NoTrans) ? uplo : (uplo == Uplo::Upper ? Uplo::Lower : Uplo::Upper);

  if (side == Side::Left) {
    std::vector<TB> temp(M * N, TB{});
    auto T = [&](std::size_t i, std::size_t j) -> TB& { return temp[i * N + j]; };
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t k = 0; k < M; ++k) {
        bool active = (effUplo == Uplo::Upper) ? (k >= i) : (k <= i);
        if (!active) continue;
        auto a = (diag == Diag::Unit && i == k) ? TA{1} : opA(i, k);
        for (std::size_t j = 0; j < N; ++j) T(i, j) = static_cast<TB>(T(i, j) + a * idx(B, k, j, ldb));
      }
    }
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        idx(B, i, j, ldb) = T(i, j);
  } else {
    std::vector<TB> temp(M * N, TB{});
    auto T = [&](std::size_t i, std::size_t j) -> TB& { return temp[i * N + j]; };
    for (std::size_t k = 0; k < N; ++k) {
      for (std::size_t j = 0; j < N; ++j) {
        bool active = (effUplo == Uplo::Upper) ? (j >= k) : (j <= k);
        if (!active) continue;
        auto a = (diag == Diag::Unit && j == k) ? TA{1} : opA(k, j);
        for (std::size_t i = 0; i < M; ++i) T(i, j) = static_cast<TB>(T(i, j) + idx(B, i, k, ldb) * a);
      }
    }
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        idx(B, i, j, ldb) = T(i, j);
  }
}

// Raw-pointer TRSM
template <typename Alpha, typename TA, typename TB>
inline void trsm(Layout layout, Side side, Uplo uplo, Trans transA, Diag diag,
                 std::size_t M, std::size_t N,
                 const Alpha& alpha,
                 const TA* A, std::size_t lda,
                 TB* B, std::size_t ldb)
{
  if (!A || !B) return;
  auto idx = [&](auto* P, std::size_t i, std::size_t j, std::size_t ld) -> auto& {
    if (layout == Layout::RowMajor) return P[i * ld + j];
    else                            return P[i + j * ld];
  };
  // Scale B by alpha
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      idx(B, i, j, ldb) = static_cast<TB>(alpha * idx(B, i, j, ldb));

  auto opA = [&](std::size_t i, std::size_t j) {
    if (transA == Trans::NoTrans) return idx(const_cast<TA*>(A), i, j, lda);
    auto v = idx(const_cast<TA*>(A), j, i, lda);
    return (transA == Trans::ConjTranspose) ? conj_if_complex(v) : v;
  };
  auto effUplo = (transA == Trans::NoTrans) ? uplo : (uplo == Uplo::Upper ? Uplo::Lower : Uplo::Upper);

  if (side == Side::Left) {
    if (effUplo == Uplo::Upper) {
      for (std::size_t i_ = 0; i_ < M; ++i_) {
        std::size_t i = M - 1 - i_;
        if (diag == Diag::NonUnit) {
          auto aii = opA(i, i);
          for (std::size_t j = 0; j < N; ++j) idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) / aii);
        }
        for (std::size_t k = 0; k < i; ++k) {
          auto aki = opA(k, i);
          if (aki == decltype(aki){}) continue;
          for (std::size_t j = 0; j < N; ++j) idx(B, k, j, ldb) = static_cast<TB>(idx(B, k, j, ldb) - aki * idx(B, i, j, ldb));
        }
      }
    } else {
      for (std::size_t i = 0; i < M; ++i) {
        if (diag == Diag::NonUnit) {
          auto aii = opA(i, i);
          for (std::size_t j = 0; j < N; ++j) idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) / aii);
        }
        for (std::size_t k = i + 1; k < M; ++k) {
          auto aki = opA(k, i);
          if (aki == decltype(aki){}) continue;
          for (std::size_t j = 0; j < N; ++j) idx(B, k, j, ldb) = static_cast<TB>(idx(B, k, j, ldb) - aki * idx(B, i, j, ldb));
        }
      }
    }
  } else { // Right
    if (effUplo == Uplo::Upper) {
      // Forward substitution on columns (upper): use known k<j
      for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < j; ++k) {
          auto akj = opA(k, j);
          if (akj == decltype(akj){}) continue;
          for (std::size_t i = 0; i < M; ++i) {
            idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) - idx(B, i, k, ldb) * akj);
          }
        }
        if (diag == Diag::NonUnit) {
          auto ajj = opA(j, j);
          for (std::size_t i = 0; i < M; ++i) idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) / ajj);
        }
      }
    } else {
      // Backward substitution on columns (lower): use known k>j
      for (std::size_t jj = 0; jj < N; ++jj) {
        std::size_t j = N - 1 - jj;
        for (std::size_t k = j + 1; k < N; ++k) {
          auto akj = opA(k, j);
          if (akj == decltype(akj){}) continue;
          for (std::size_t i = 0; i < M; ++i) {
            idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) - idx(B, i, k, ldb) * akj);
          }
        }
        if (diag == Diag::NonUnit) {
          auto ajj = opA(j, j);
          for (std::size_t i = 0; i < M; ++i) idx(B, i, j, ldb) = static_cast<TB>(idx(B, i, j, ldb) / ajj);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// TRMM: B := alpha * op(A) * B   or   B := alpha * B * op(A)   (triangular A)
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename B>
  requires (MatrixLike<A> && MatrixLike<B>)
inline void trmm(Side side, Uplo uplo, Trans transA, Diag diag,
                 const Alpha& alpha, const A& A_, B& B_)
{
  const std::size_t Am = A_.rows(), An = A_.cols();
  if (Am != An) throw std::invalid_argument("trmm: A must be square");

  // Scale B by alpha first
  for (std::size_t i = 0; i < B_.rows(); ++i)
    for (std::size_t j = 0; j < B_.cols(); ++j)
      B_(i, j) = static_cast<mat_elem_t<B>>(alpha * B_(i, j));

  auto opA = [&](std::size_t i, std::size_t j) {
    if (transA == Trans::NoTrans) return A_(i, j);
    auto v = A_(j, i);
    return (transA == Trans::ConjTranspose) ? conj_if_complex(v) : v;
  };

  auto effUplo = (transA == Trans::NoTrans) ? uplo : (uplo == Uplo::Upper ? Uplo::Lower : Uplo::Upper);

  if (side == Side::Left) {
    const std::size_t M = B_.rows();
    const std::size_t N = B_.cols();
    if (M != Am) throw std::invalid_argument("trmm(left): size mismatch");

    // Compute C = op(A) * B into a temp to avoid in-place hazards
    std::vector<mat_elem_t<B>> temp(M * N, mat_elem_t<B>{});
    auto T = [&](std::size_t i, std::size_t j) -> mat_elem_t<B>& { return temp[i * N + j]; };

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t k = 0; k < M; ++k) {
        // Check if (i,k) lies in effective triangle
        bool active = (effUplo == Uplo::Upper) ? (k >= i) : (k <= i);
        if (!active) continue;
        auto a = (diag == Diag::Unit && i == k) ? mat_elem_t<A>{1} : opA(i, k);
        for (std::size_t j = 0; j < N; ++j) {
          T(i, j) = static_cast<mat_elem_t<B>>(T(i, j) + a * B_(k, j));
        }
      }
    }
    // Copy back
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        B_(i, j) = T(i, j);
  } else { // Right
    const std::size_t M = B_.rows();
    const std::size_t N = B_.cols();
    if (N != An) throw std::invalid_argument("trmm(right): size mismatch");

    std::vector<mat_elem_t<B>> temp(M * N, mat_elem_t<B>{});
    auto T = [&](std::size_t i, std::size_t j) -> mat_elem_t<B>& { return temp[i * N + j]; };

    for (std::size_t k = 0; k < N; ++k) {
      for (std::size_t j = 0; j < N; ++j) {
        bool active = (effUplo == Uplo::Upper) ? (j >= k) : (j <= k);
        if (!active) continue;
        auto a = (diag == Diag::Unit && j == k) ? mat_elem_t<A>{1} : opA(k, j);
        for (std::size_t i = 0; i < M; ++i) {
          T(i, j) = static_cast<mat_elem_t<B>>(T(i, j) + B_(i, k) * a);
        }
      }
    }
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j)
        B_(i, j) = T(i, j);
  }
}

// ---------------------------------------------------------------------------
// TRSM: Solve op(A) * X = alpha * B   or   X * op(A) = alpha * B
// In-place on B (B becomes X)
// ---------------------------------------------------------------------------

template <typename Alpha, typename A, typename B>
  requires (MatrixLike<A> && MatrixLike<B>)
inline void trsm(Side side, Uplo uplo, Trans transA, Diag diag,
                 const Alpha& alpha, const A& A_, B& B_)
{
  const std::size_t Am = A_.rows(), An = A_.cols();
  if (Am != An) throw std::invalid_argument("trsm: A must be square");

  // Scale right-hand side by alpha
  for (std::size_t i = 0; i < B_.rows(); ++i)
    for (std::size_t j = 0; j < B_.cols(); ++j)
      B_(i, j) = static_cast<mat_elem_t<B>>(alpha * B_(i, j));

  auto opA = [&](std::size_t i, std::size_t j) {
    if (transA == Trans::NoTrans) return A_(i, j);
    auto v = A_(j, i);
    return (transA == Trans::ConjTranspose) ? conj_if_complex(v) : v;
  };
  auto effUplo = (transA == Trans::NoTrans) ? uplo : (uplo == Uplo::Upper ? Uplo::Lower : Uplo::Upper);

  if (side == Side::Left) {
    const std::size_t M = B_.rows();
    const std::size_t N = B_.cols();
    if (M != Am) throw std::invalid_argument("trsm(left): size mismatch");

    if (effUplo == Uplo::Upper) {
      // Backward substitution
      for (std::size_t i_ = 0; i_ < M; ++i_) {
        std::size_t i = M - 1 - i_;
        // Divide by diagonal if non-unit
        if (diag == Diag::NonUnit) {
          auto aii = opA(i, i);
          for (std::size_t j = 0; j < N; ++j) B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) / aii);
        }
        // Update rows above
        for (std::size_t k = 0; k < i; ++k) {
          auto aki = opA(k, i);
          if (aki == decltype(aki){}) continue;
          for (std::size_t j = 0; j < N; ++j) {
            B_(k, j) = static_cast<mat_elem_t<B>>(B_(k, j) - aki * B_(i, j));
          }
        }
      }
    } else {
      // Forward substitution (lower)
      for (std::size_t i = 0; i < M; ++i) {
        if (diag == Diag::NonUnit) {
          auto aii = opA(i, i);
          for (std::size_t j = 0; j < N; ++j) B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) / aii);
        }
        for (std::size_t k = i + 1; k < M; ++k) {
          auto aki = opA(k, i);
          if (aki == decltype(aki){}) continue;
          for (std::size_t j = 0; j < N; ++j) {
            B_(k, j) = static_cast<mat_elem_t<B>>(B_(k, j) - aki * B_(i, j));
          }
        }
      }
    }
  } else { // Right side: X * op(A) = alpha * B
    const std::size_t M = B_.rows();
    const std::size_t N = B_.cols();
    if (N != An) throw std::invalid_argument("trsm(right): size mismatch");

    if (effUplo == Uplo::Upper) {
      // Forward substitution over columns (upper): use k<j
      for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < j; ++k) {
          auto akj = opA(k, j);
          if (akj == decltype(akj){}) continue;
          for (std::size_t i = 0; i < M; ++i) {
            B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) - B_(i, k) * akj);
          }
        }
        if (diag == Diag::NonUnit) {
          auto ajj = opA(j, j);
          for (std::size_t i = 0; i < M; ++i) B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) / ajj);
        }
      }
    } else {
      // Backward substitution over columns (lower): use k>j
      for (std::size_t jj = 0; jj < N; ++jj) {
        std::size_t j = N - 1 - jj;
        for (std::size_t k = j + 1; k < N; ++k) {
          auto akj = opA(k, j);
          if (akj == decltype(akj){}) continue;
          for (std::size_t i = 0; i < M; ++i) {
            B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) - B_(i, k) * akj);
          }
        }
        if (diag == Diag::NonUnit) {
          auto ajj = opA(j, j);
          for (std::size_t i = 0; i < M; ++i) B_(i, j) = static_cast<mat_elem_t<B>>(B_(i, j) / ajj);
        }
      }
    }
  }
}

} // namespace fem::numeric::linear_algebra

#endif // NUMERIC_LINEAR_ALGEBRA_BLAS_LEVEL3_H

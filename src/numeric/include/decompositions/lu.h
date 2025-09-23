#pragma once

#ifndef NUMERIC_DECOMPOSITIONS_LU_H
#define NUMERIC_DECOMPOSITIONS_LU_H

#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level3.h" // gemm
#include "../backends/lapack_backend.h"

namespace fem::numeric::decompositions {

// Forward declaration (blocked LU)
template <typename T, typename Storage, StorageOrder Order>
int lu_factor_blocked(Matrix<T, Storage, Order>& A, std::vector<int>& piv, std::size_t block = 64);

// LU factorization (best path)
// - ColumnMajor + LAPACK: full GETRF on A (zero-copy), 0-based pivots returned
// - Otherwise: blocked LU (panel uses LAPACK when available; else unblocked)
template <typename T, typename Storage, StorageOrder Order>
int lu_factor(Matrix<T, Storage, Order>& A, std::vector<int>& piv)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
#if defined(FEM_NUMERIC_ENABLE_LAPACK)
    if constexpr (Order == StorageOrder::ColumnMajor) {
        const std::size_t kmax = std::min(m, n);
        piv.resize(kmax);
        std::vector<int> ipiv(kmax, 0);
        int info = 0;
        int M = static_cast<int>(m), N = static_cast<int>(n), lda = static_cast<int>(A.rows());
        backends::lapack::getrf_cm<T>(M, N, A.data(), lda, ipiv.data(), info);
        for (std::size_t i = 0; i < kmax; ++i) piv[i] = ipiv[i] - 1; // 0-based
        return info;
    }
#endif
    return lu_factor_blocked(A, piv);
}

// Apply recorded row permutations to a dense vector (in-place), consistent with LU pivots.
template <typename T>
void lu_apply_pivots(const std::vector<int>& piv, Vector<T>& b)
{
    const std::size_t kmax = piv.size();
    for (std::size_t k = 0; k < kmax; ++k) {
        int p = piv[k];
        if (static_cast<std::size_t>(p) != k) {
            auto tmp = b[k];
            b[k] = b[static_cast<std::size_t>(p)];
            b[static_cast<std::size_t>(p)] = tmp;
        }
    }
}

// Forward and backward substitution using packed LU (L unit lower, U upper)
template <typename T>
void lu_solve_inplace(const Matrix<T>& LU, Vector<T>& b)
{
    const std::size_t n = LU.rows();
    if (LU.cols() != n || b.size() != n) {
        throw std::invalid_argument("lu_solve_inplace: dimension mismatch");
    }

    // Forward solve Ly = Pb (but b already permuted)
    for (std::size_t i = 0; i < n; ++i) {
        T sum = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            sum -= LU(i, j) * b[j];
        }
        b[i] = sum; // L has 1 on diagonal
    }

    // Backward solve Ux = y
    for (std::size_t i = n; i-- > 0;) {
        T sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= LU(i, j) * b[j];
        }
        auto uii = LU(i, i);
        if (uii == T{0}) {
            throw std::runtime_error("lu_solve_inplace: singular matrix (zero on U diagonal)");
        }
        b[i] = sum / uii;
    }
}

// Solve a single RHS vector using LU and pivots
template <typename T>
void lu_solve(const Matrix<T>& LU, const std::vector<int>& piv, Vector<T>& b)
{
    if (LU.rows() != LU.cols() || b.size() != LU.rows()) {
        throw std::invalid_argument("lu_solve: dimension mismatch");
    }
    Vector<T> bp = b; // make a working copy
    lu_apply_pivots(piv, bp);
    lu_solve_inplace(LU, bp);
    b = std::move(bp);
}

// Solve multiple RHS stored in columns of B (in-place)
template <typename T>
void lu_solve_inplace(const Matrix<T>& LU, const std::vector<int>& piv, Matrix<T>& B)
{
    const std::size_t n = LU.rows();
    if (LU.cols() != n || B.rows() != n) {
        throw std::invalid_argument("lu_solve_inplace (matrix): dimension mismatch");
    }
    // Apply pivots to each column of B
    const std::size_t nrhs = B.cols();
    for (std::size_t k = 0; k < piv.size(); ++k) {
        std::size_t p = static_cast<std::size_t>(piv[k]);
        if (p != k) {
            for (std::size_t j = 0; j < nrhs; ++j) {
                auto tmp = B(k, j);
                B(k, j) = B(p, j);
                B(p, j) = tmp;
            }
        }
    }
    // Forward substitution for each RHS
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            T sum = B(i, j);
            for (std::size_t k = 0; k < i; ++k) sum -= LU(i, k) * B(k, j);
            B(i, j) = sum;
        }
    }
    // Backward substitution for each RHS
    for (std::size_t i = n; i-- > 0;) {
        auto uii = LU(i, i);
        if (uii == T{0}) {
            throw std::runtime_error("lu_solve_inplace (matrix): singular matrix");
        }
        for (std::size_t j = 0; j < B.cols(); ++j) {
            T sum = B(i, j);
            for (std::size_t k = i + 1; k < n; ++k) sum -= LU(i, k) * B(k, j);
            B(i, j) = sum / uii;
        }
    }
}

// Convenience wrapper: returns a copy of B with the solution (does not modify input B)
template <typename T>
Matrix<T> lu_solve(const Matrix<T>& LU, const std::vector<int>& piv, const Matrix<T>& B)
{
    Matrix<T> X = B;
    lu_solve_inplace(LU, piv, X);
    return X;
}

// Optional: determinant from LU (may overflow for large n)
template <typename T>
T lu_determinant(const Matrix<T>& LU, const std::vector<int>& piv)
{
    const std::size_t n = LU.rows();
    if (LU.cols() != n) throw std::invalid_argument("lu_determinant: LU must be square");
    // Compute sign from pivots: count parity of row swaps
    int swaps = 0;
    for (std::size_t k = 0; k < piv.size(); ++k) {
        if (static_cast<std::size_t>(piv[k]) != k) swaps++;
    }
    T det = (swaps % 2 == 0) ? T{1} : T{-1};
    for (std::size_t i = 0; i < n; ++i) det *= LU(i, i);
    return det;
}

} // namespace fem::numeric::decompositions

// ---------------------------------------------------------------------------
// Blocked LU factorization with partial pivoting (GETRF-like), right-looking
// Performs panel factorization with row swaps applied to the full matrix, then
// updates the trailing submatrix with GEMM. Falls back to LAPACK if enabled.
// Returns 0 on success or k+1 if zero pivot at step k.
// ---------------------------------------------------------------------------
namespace fem::numeric::decompositions {

template <typename T, typename Storage, StorageOrder Order>
int lu_factor_blocked(Matrix<T, Storage, Order>& A, std::vector<int>& piv, std::size_t block = 64)
{
    using std::abs;
    using namespace fem::numeric::linear_algebra;
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t kmax = std::min(m, n);
    piv.resize(kmax);

    // Backend try (no-op unless linked)
    {
        int info_backend = 0;
        if (backends::lapack::getrf_inplace(A, piv, info_backend)) return info_backend;
    }

    if (kmax == 0) return 0;
    const std::size_t bs = std::max<std::size_t>(1, block);
    int info = 0;

    for (std::size_t k = 0; k < kmax; k += bs) {
        const std::size_t kb = std::min(bs, kmax - k);

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
        // Panel factorization using LAPACK (GETRF) on (m-k) x kb block
        {
            const std::size_t pm = m - k; const std::size_t pn = kb;
            std::vector<int> ipiv(kb, 0);
            int info_panel = 0;
            if constexpr (Order == StorageOrder::ColumnMajor) {
                auto P = A.submatrix(k, m, k, k + kb);
                int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(A.rows());
                backends::lapack::getrf_cm<T>(M, N, P.data(), lda, ipiv.data(), info_panel);
            }
#if defined(FEM_NUMERIC_ENABLE_LAPACKE)
            else if constexpr (Order == StorageOrder::RowMajor) {
                auto P = A.submatrix(k, m, k, k + kb);
                int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(A.cols());
                info_panel = backends::lapack::getrf_rm<T>(M, N, P.data(), lda, ipiv.data());
            } else {
#else
            else {
#endif
                // Copy panel to column-major buffer
                std::vector<T> panel(pm * pn);
                for (std::size_t j = 0; j < pn; ++j)
                    for (std::size_t i = 0; i < pm; ++i)
                        panel[j * pm + i] = A(k + i, k + j);
                int M = static_cast<int>(pm), N = static_cast<int>(pn), lda = static_cast<int>(pm);
                backends::lapack::getrf_cm<T>(M, N, panel.data(), lda, ipiv.data(), info_panel);
                // Copy panel back
                for (std::size_t j = 0; j < pn; ++j)
                    for (std::size_t i = 0; i < pm; ++i)
                        A(k + i, k + j) = panel[j * pm + i];
            }
            if (info_panel > 0 && info == 0) info = static_cast<int>(k) + info_panel;

            // Apply row swaps to entire matrix A(k+i, :) <-> A(p, :)
            for (std::size_t j = 0; j < kb; ++j) {
                std::size_t p = static_cast<std::size_t>(ipiv[j] - 1) + k; // absolute row
                std::size_t r = k + j;
                if (p != r) A.swap_rows(r, p);
            }

            // Compute U12: solve U11 * X = A12
            if (k + kb < n) {
                auto U11 = A.submatrix(k, k + kb, k, k + kb);
                auto A12 = A.submatrix(k, k + kb, k + kb, n);
                trsm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, T{1}, U11, A12);
            }
        }
#else
        // Fallback panel factorization (unblocked) when LAPACK disabled
        for (std::size_t j = 0; j < kb; ++j) {
            std::size_t col = k + j;
            std::size_t p = col;
            auto maxval = abs(A(col, col));
            for (std::size_t i = col + 1; i < m; ++i) {
                auto v = abs(A(i, col));
                if (v > maxval) { maxval = v; p = i; }
            }
            piv[col] = static_cast<int>(p);
            if (maxval == decltype(maxval){0}) { if (info == 0) info = static_cast<int>(col) + 1; continue; }
            if (p != col) A.swap_rows(col, p);
            auto pivot = A(col, col);
            for (std::size_t i = col + 1; i < m; ++i) {
                A(i, col) = A(i, col) / pivot;
                auto lik = A(i, col);
                for (std::size_t jj = col + 1; jj < k + kb; ++jj) A(i, jj) -= lik * A(col, jj);
            }
        }
        if (k + kb < n) {
            auto U11 = A.submatrix(k, k + kb, k, k + kb);
            auto A12 = A.submatrix(k, k + kb, k + kb, n);
            trsm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, T{1}, U11, A12);
        }
#endif

        // Trailing update with GEMM: A22 -= L21 * U12
        const std::size_t row2 = k + kb;
        const std::size_t col2 = k + kb;
        if (row2 < m && col2 < n) {
            auto L21 = A.submatrix(row2, m, k, k + kb);
            auto U12 = A.submatrix(k, k + kb, col2, n);
            auto A22 = A.submatrix(row2, m, col2, n);
            gemm(Trans::NoTrans, Trans::NoTrans, T{-1}, L21, U12, T{1}, A22);
        }
    }
    return info;
}

} // namespace fem::numeric::decompositions


#endif // NUMERIC_DECOMPOSITIONS_LU_H

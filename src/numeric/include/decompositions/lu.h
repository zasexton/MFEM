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

namespace fem::numeric::decompositions {

// LU factorization with partial pivoting (Gaussian elimination)
// Packs L (unit diagonal) in the strict lower part of A and U in the upper part.
// Returns 0 on success, or k+1 if a zero pivot was detected at step k.
// piv will have size min(m, n) and record the pivot row chosen at each step (0-based).
template <typename T>
int lu_factor(Matrix<T>& A, std::vector<int>& piv)
{
    using std::abs;
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t kmax = std::min(m, n);

    piv.resize(kmax);
    int info = 0;

    for (std::size_t k = 0; k < kmax; ++k) {
        // Find pivot row p in [k, m)
        std::size_t p = k;
        auto maxval = abs(A(k, k));
        for (std::size_t i = k + 1; i < m; ++i) {
            auto v = abs(A(i, k));
            if (v > maxval) { maxval = v; p = i; }
        }
        piv[k] = static_cast<int>(p);

        // If pivot is zero, mark singular but continue to generate a partial factorization
        if (maxval == decltype(maxval){0}) {
            if (info == 0) info = static_cast<int>(k) + 1;
            continue;
        }

        // Swap rows k and p if needed
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) {
                auto tmp = A(k, j);
                A(k, j) = A(p, j);
                A(p, j) = tmp;
            }
        }

        // Compute multipliers and update trailing submatrix
        auto pivot = A(k, k);
        for (std::size_t i = k + 1; i < m; ++i) {
            A(i, k) = A(i, k) / pivot;              // L(i,k)
            auto lik = A(i, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                A(i, j) -= lik * A(k, j);           // A(i,j) = A(i,j) - L(i,k)*U(k,j)
            }
        }
    }
    return info;
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

#endif // NUMERIC_DECOMPOSITIONS_LU_H


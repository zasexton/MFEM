#ifndef NUMERIC_DECOMPOSITIONS_LDLT_BLOCKED_BK_H
#define NUMERIC_DECOMPOSITIONS_LDLT_BLOCKED_BK_H

// Blocked Bunch-Kaufman algorithm for symmetric indefinite factorization
// This implements a panel-based factorization with 2x2 pivot blocks
// and BLAS-3 rank-k updates for the trailing matrix

#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level2.h"
#include "../linear_algebra/blas_level3.h"
#include "workspace.h"

namespace fem::numeric::decompositions {

// Forward declaration for the unblocked version
template <typename T, typename Storage, StorageOrder Order>
int ldlt_factor_bk_rook(Matrix<T, Storage, Order>& A,
                        std::vector<int>& ipiv,
                        fem::numeric::linear_algebra::Uplo uplo);

// Helper function for conjugate
template <typename T>
constexpr T conj_bk(const T& x) {
    if constexpr (is_complex_number_v<T>) {
        using std::conj;
        return conj(x);
    } else {
        return x;
    }
}

// Blocked Bunch-Kaufman factorization with symmetric indefinite pivoting
// Returns 0 on success, >0 if singular at step i
template <typename T, typename Storage, StorageOrder Order>
int ldlt_factor_blocked_bk(Matrix<T, Storage, Order>& A,
                           std::vector<int>& ipiv,
                           fem::numeric::linear_algebra::Uplo uplo = fem::numeric::linear_algebra::Uplo::Lower,
                           std::size_t block_size = 0)
{
    using namespace fem::numeric::linear_algebra;
    using R = typename numeric_traits<T>::scalar_type;

    const std::size_t n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("ldlt_factor_blocked_bk: matrix must be square");
    }

    ipiv.resize(n);
    std::fill(ipiv.begin(), ipiv.end(), 0);

    if (n == 0) return 0;

    // Adaptive block size
    const std::size_t auto_block = std::min<std::size_t>(256, std::max<std::size_t>(128, n/4));
    const std::size_t nb = (block_size > 0) ? block_size : auto_block;

    // Get workspace for panel operations
    auto& workspace = get_thread_local_workspace<T>();

    // Alpha threshold for pivot selection (from LAPACK)
    const R alpha = (static_cast<R>(1) + std::sqrt(static_cast<R>(17))) / static_cast<R>(8);

    int info = 0;

    if (uplo == Uplo::Lower) {
        // Lower triangular storage
        std::size_t k = 0;
        while (k < n) {
            // Determine block size for this panel
            const std::size_t kb = std::min(nb, n - k);

            // Factor the panel A(k:n, k:k+kb)
            std::size_t panel_pivots = 0;

            // Panel factorization loop
            for (std::size_t j = 0; j < kb && k + j < n; ) {
                std::size_t jb = 1; // Size of current pivot block (1x1 or 2x2)

                // Find pivot for column k+j
                R absakk = std::abs(A(k + j, k + j));
                std::size_t imax = k + j;
                R colmax = 0;

                // Search for maximum in column k+j
                if (k + j + 1 < n) {
                    for (std::size_t i = k + j + 1; i < n; ++i) {
                        R absval = std::abs(A(i, k + j));
                        if (absval > colmax) {
                            colmax = absval;
                            imax = i;
                        }
                    }
                }

                // Pivot selection using alpha threshold
                if (absakk >= alpha * colmax) {
                    // Use 1x1 pivot at diagonal
                    ipiv[k + j] = static_cast<int>(k + j + 1); // 1-based for LAPACK compatibility
                    jb = 1;
                } else {
                    // Need to check for 2x2 pivot
                    R rowmax = colmax;
                    std::size_t jmax = imax;

                    // Find max in row imax
                    for (std::size_t jj = k + j + 1; jj < imax; ++jj) {
                        R absval = std::abs(A(imax, jj));
                        if (absval > rowmax) {
                            rowmax = absval;
                            jmax = jj;
                        }
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax)) {
                        // Use 1x1 pivot at diagonal
                        ipiv[k + j] = static_cast<int>(k + j + 1);
                        jb = 1;
                    } else if (std::abs(A(imax, imax)) >= alpha * rowmax) {
                        // Use 1x1 pivot at A(imax,imax), need to swap rows/cols
                        ipiv[k + j] = static_cast<int>(imax + 1);

                        // Swap rows and columns k+j and imax
                        if (imax != k + j) {
                            // Swap columns in A(0:k+j-1, k+j) and A(0:k+j-1, imax)
                            for (std::size_t i = 0; i < k + j; ++i) {
                                std::swap(A(k + j, i), A(imax, i));
                            }
                            // Swap elements in diagonal
                            std::swap(A(k + j, k + j), A(imax, imax));
                            // Swap rows in A(k+j:imax-1, k+j) and A(imax, k+j+1:imax-1)
                            for (std::size_t i = k + j + 1; i < imax; ++i) {
                                std::swap(A(i, k + j), A(imax, i));
                            }
                            // Swap columns in A(imax+1:n, k+j) and A(imax+1:n, imax)
                            for (std::size_t i = imax + 1; i < n; ++i) {
                                std::swap(A(i, k + j), A(i, imax));
                            }
                        }
                        jb = 1;
                    } else {
                        // Use 2x2 pivot
                        ipiv[k + j] = static_cast<int>(-(imax + 1)); // Negative for 2x2
                        ipiv[k + j + 1] = static_cast<int>(-(imax + 1));

                        // Swap rows/columns to bring 2x2 pivot to diagonal
                        if (imax != k + j + 1) {
                            // Swap row/col k+j+1 and imax
                            for (std::size_t i = 0; i <= k + j; ++i) {
                                std::swap(A(k + j + 1, i), A(imax, i));
                            }
                            std::swap(A(k + j + 1, k + j + 1), A(imax, imax));
                            for (std::size_t i = k + j + 2; i < imax; ++i) {
                                std::swap(A(i, k + j + 1), A(imax, i));
                            }
                            for (std::size_t i = imax + 1; i < n; ++i) {
                                std::swap(A(i, k + j + 1), A(i, imax));
                            }
                        }
                        jb = 2;
                    }
                }

                // Factor the pivot block
                if (jb == 1) {
                    // 1x1 pivot
                    T d11 = A(k + j, k + j);
                    if (std::abs(d11) < std::numeric_limits<R>::epsilon()) {
                        if (info == 0) info = static_cast<int>(k + j + 1);
                        d11 = std::numeric_limits<R>::epsilon();
                    }

                    // Scale column by 1/d11
                    if (k + j + 1 < n) {
                        T d11_inv = static_cast<T>(1) / d11;
                        for (std::size_t i = k + j + 1; i < n; ++i) {
                            A(i, k + j) *= d11_inv;
                        }
                    }
                } else {
                    // 2x2 pivot
                    T d11 = A(k + j, k + j);
                    T d21 = A(k + j + 1, k + j);
                    T d22 = A(k + j + 1, k + j + 1);

                    // Compute determinant
                    T det = d11 * d22 - d21 * conj_bk(d21);
                    if (std::abs(det) < std::numeric_limits<R>::epsilon()) {
                        if (info == 0) info = static_cast<int>(k + j + 1);
                        det = std::numeric_limits<R>::epsilon();
                    }

                    // Invert 2x2 block
                    T det_inv = static_cast<T>(1) / det;
                    T dinv11 = d22 * det_inv;
                    T dinv12 = -d21 * det_inv;
                    T dinv22 = d11 * det_inv;

                    // Scale columns by inv(D)
                    if (k + j + 2 < n) {
                        for (std::size_t i = k + j + 2; i < n; ++i) {
                            T l1 = A(i, k + j);
                            T l2 = A(i, k + j + 1);
                            A(i, k + j) = dinv11 * l1 + dinv12 * l2;
                            A(i, k + j + 1) = conj_bk(dinv12) * l1 + dinv22 * l2;
                        }
                    }
                }

                // Update trailing submatrix within panel
                if (k + j + jb < n) {
                    std::size_t m = n - (k + j + jb);

                    if (jb == 1) {
                        // Rank-1 update: A22 = A22 - L21 * d11 * L21^T
                        T d11 = A(k + j, k + j);
                        for (std::size_t i = k + j + 1; i < n; ++i) {
                            for (std::size_t ii = i; ii < n; ++ii) {
                                A(ii, i) -= A(ii, k + j) * d11 * conj_bk(A(i, k + j));
                            }
                        }
                    } else {
                        // Rank-2 update: A22 = A22 - L21 * D * L21^T
                        T d11 = A(k + j, k + j);
                        T d21 = A(k + j + 1, k + j);
                        T d22 = A(k + j + 1, k + j + 1);

                        for (std::size_t i = k + j + 2; i < n; ++i) {
                            T l1 = A(i, k + j);
                            T l2 = A(i, k + j + 1);
                            for (std::size_t ii = i; ii < n; ++ii) {
                                T ll1 = A(ii, k + j);
                                T ll2 = A(ii, k + j + 1);
                                A(ii, i) -= ll1 * d11 * conj_bk(l1) + ll1 * d21 * conj_bk(l2) +
                                           ll2 * conj_bk(d21) * conj_bk(l1) + ll2 * d22 * conj_bk(l2);
                            }
                        }
                    }
                }

                j += jb;
                panel_pivots += jb;
            }

            // Update trailing matrix with BLAS-3 if we have processed a full panel
            if (k + panel_pivots < n && panel_pivots > 0) {
                // Extract L21 and D from factored panel
                std::size_t m = n - (k + panel_pivots);

                // Prepare workspace for L21*D
                T* LD = workspace.get_buffer(m * panel_pivots);

                // Copy L21 and scale by D
                for (std::size_t j = 0; j < panel_pivots; ++j) {
                    if (ipiv[k + j] > 0) {
                        // 1x1 block
                        T d = A(k + j, k + j);
                        for (std::size_t i = 0; i < m; ++i) {
                            LD[i + j * m] = A(k + panel_pivots + i, k + j) * d;
                        }
                    } else if (j + 1 < panel_pivots && ipiv[k + j] < 0 && ipiv[k + j + 1] == ipiv[k + j]) {
                        // 2x2 block
                        T d11 = A(k + j, k + j);
                        T d21 = A(k + j + 1, k + j);
                        T d22 = A(k + j + 1, k + j + 1);

                        for (std::size_t i = 0; i < m; ++i) {
                            T l1 = A(k + panel_pivots + i, k + j);
                            T l2 = A(k + panel_pivots + i, k + j + 1);
                            LD[i + j * m] = l1 * d11 + l2 * conj_bk(d21);
                            LD[i + (j + 1) * m] = l1 * d21 + l2 * d22;
                        }
                        ++j; // Skip next column as it's part of 2x2 block
                    }
                }

                // Update trailing matrix: A22 = A22 - L21 * D * L21^T
                // This is a symmetric rank-k update (SYRK-like)
                auto A22 = A.submatrix(k + panel_pivots, n, k + panel_pivots, n);
                auto L21 = A.submatrix(k + panel_pivots, n, k, k + panel_pivots);

                // Use GEMM for the update (not optimal but works)
                // TODO: Could optimize with SYRK when available
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = i; j < m; ++j) {
                        T sum = 0;
                        for (std::size_t kk = 0; kk < panel_pivots; ++kk) {
                            sum += L21(j, kk) * conj_bk(LD[i + kk * m]);
                        }
                        A22(j, i) -= sum;
                    }
                }
            }

            k += panel_pivots;
        }
    } else {
        // Upper triangular storage - similar but with column-wise processing
        // For brevity, using unblocked fallback for upper case
        return ldlt_factor_bk_rook(A, ipiv, uplo);
    }

    return info;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_LDLT_BLOCKED_BK_H
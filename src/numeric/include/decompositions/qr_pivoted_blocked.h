#ifndef NUMERIC_DECOMPOSITIONS_QR_PIVOTED_BLOCKED_H
#define NUMERIC_DECOMPOSITIONS_QR_PIVOTED_BLOCKED_H

// Blocked Column-Pivoted QR factorization
// Implements panel pivoting with norm downdating and BLAS-3 trailing matrix updates

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "../core/matrix.h"
#include "../core/vector.h"
#include "../base/traits_base.h"
#include "../linear_algebra/blas_level2.h"
#include "../linear_algebra/blas_level3.h"
#include "../linear_algebra/householder_wy.h"
#include "workspace.h"

namespace fem::numeric::decompositions {

// Helper for conjugate
template <typename T>
constexpr T conj_cpqr(const T& x) {
    if constexpr (is_complex_number_v<T>) {
        using std::conj;
        return conj(x);
    } else {
        return x;
    }
}

// Blocked column-pivoted QR factorization
// Returns 0 on success, rank is returned in rank_out
template <typename T, typename Storage, StorageOrder Order>
int qr_factor_pivoted_blocked(Matrix<T, Storage, Order>& A,
                              std::vector<T>& tau,
                              std::vector<int>& jpiv,
                              std::size_t& rank_out,
                              typename numeric_traits<T>::scalar_type tol,
                              std::size_t block_size = 0)
{
    using namespace fem::numeric::linear_algebra;
    using R = typename numeric_traits<T>::scalar_type;

    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t k = std::min(m, n);

    tau.assign(k, T{});
    jpiv.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
        jpiv[j] = static_cast<int>(j);
    }

    if (k == 0) {
        rank_out = 0;
        return 0;
    }

    // Adaptive block size
    const std::size_t auto_block = std::min<std::size_t>(256, std::max<std::size_t>(128, n/4));
    const std::size_t nb = (block_size > 0) ? block_size : auto_block;

    // Get workspace
    auto& workspace = get_thread_local_workspace<T>();

    // Column norms (current and original)
    std::vector<R> cnorm(n, R{});
    std::vector<R> cnorm0(n, R{});

    // Compute initial column norms
    for (std::size_t j = 0; j < n; ++j) {
        R sum = 0;
        for (std::size_t i = 0; i < m; ++i) {
            if constexpr (is_complex_number_v<T>) {
                sum += std::norm(A(i, j));
            } else {
                sum += A(i, j) * A(i, j);
            }
        }
        cnorm[j] = std::sqrt(sum);
        cnorm0[j] = cnorm[j];
    }

    // Determine initial tolerance
    R norm_max = *std::max_element(cnorm.begin(), cnorm.end());
    R tol_abs = (tol > 0) ? tol : (std::numeric_limits<R>::epsilon() * std::max(m, n) * norm_max);

    rank_out = 0;
    std::size_t j = 0;

    // Main blocked loop
    while (j < k) {
        // Determine block size for this iteration
        const std::size_t jb = std::min(nb, k - j);

        // Panel factorization with column pivoting
        std::size_t panel_rank = 0;

        for (std::size_t jj = 0; jj < jb && j + jj < k; ++jj) {
            // Find pivot column in remaining columns
            std::size_t pvt = j + jj;
            R max_norm = cnorm[j + jj];

            for (std::size_t jp = j + jj + 1; jp < n; ++jp) {
                if (cnorm[jp] > max_norm) {
                    max_norm = cnorm[jp];
                    pvt = jp;
                }
            }

            // Check for rank deficiency
            if (max_norm < tol_abs) {
                break; // Remaining matrix is negligible
            }

            // Swap columns if needed
            if (pvt != j + jj) {
                // Swap columns in A
                for (std::size_t i = 0; i < m; ++i) {
                    std::swap(A(i, j + jj), A(i, pvt));
                }
                // Swap pivot indices
                std::swap(jpiv[j + jj], jpiv[pvt]);
                // Swap column norms
                std::swap(cnorm[j + jj], cnorm[pvt]);
                std::swap(cnorm0[j + jj], cnorm0[pvt]);
            }

            // Compute Householder reflector for column j+jj
            R norm_x = 0;
            for (std::size_t i = j + jj; i < m; ++i) {
                if constexpr (is_complex_number_v<T>) {
                    norm_x += std::norm(A(i, j + jj));
                } else {
                    norm_x += A(i, j + jj) * A(i, j + jj);
                }
            }
            norm_x = std::sqrt(norm_x);

            if (norm_x > std::numeric_limits<R>::epsilon()) {
                // Compute reflector
                T alpha = A(j + jj, j + jj);
                T beta;
                R norm_v;

                if constexpr (is_complex_number_v<T>) {
                    R sign = (std::real(alpha) >= 0) ? R(1) : R(-1);
                    beta = -sign * std::abs(alpha) * norm_x / std::abs(alpha);
                    if (std::abs(alpha) > 0) {
                        beta = -sign * norm_x * alpha / std::abs(alpha);
                    } else {
                        beta = -norm_x;
                    }
                } else {
                    beta = (alpha >= 0) ? -norm_x : norm_x;
                }

                // Store reflector
                tau[j + jj] = (beta - alpha) / beta;
                T scale = T(1) / (alpha - beta);

                // Apply reflector to column j+jj
                A(j + jj, j + jj) = beta;
                for (std::size_t i = j + jj + 1; i < m; ++i) {
                    A(i, j + jj) *= scale;
                }

                // Apply reflector to remaining panel columns
                if (jj + 1 < jb && j + jj + 1 < n) {
                    // Compute w = A^H * v
                    T* w = workspace.get_buffer(n - (j + jj + 1));
                    for (std::size_t jjj = j + jj + 1; jjj < j + jb && jjj < n; ++jjj) {
                        T sum = conj_cpqr(A(j + jj, jjj));
                        for (std::size_t i = j + jj + 1; i < m; ++i) {
                            sum += conj_cpqr(A(i, j + jj)) * A(i, jjj);
                        }
                        w[jjj - (j + jj + 1)] = sum * tau[j + jj];
                    }

                    // Apply update: A = A - v * w^H
                    for (std::size_t jjj = j + jj + 1; jjj < j + jb && jjj < n; ++jjj) {
                        A(j + jj, jjj) -= w[jjj - (j + jj + 1)];
                        for (std::size_t i = j + jj + 1; i < m; ++i) {
                            A(i, jjj) -= A(i, j + jj) * w[jjj - (j + jj + 1)];
                        }
                    }
                }

                panel_rank++;
                rank_out++;
            } else {
                tau[j + jj] = 0;
            }

            // Update column norms for downdating
            for (std::size_t jp = j + jj + 1; jp < n; ++jp) {
                if (cnorm[jp] > 0) {
                    R temp = std::abs(A(j + jj, jp)) / cnorm[jp];
                    temp = 1 - temp * temp;
                    if (temp > 0.05) {
                        // Safe downdating
                        cnorm[jp] *= std::sqrt(temp);
                    } else {
                        // Recompute norm to avoid accumulation of errors
                        R sum = 0;
                        for (std::size_t i = j + jj + 1; i < m; ++i) {
                            if constexpr (is_complex_number_v<T>) {
                                sum += std::norm(A(i, jp));
                            } else {
                                sum += A(i, jp) * A(i, jp);
                            }
                        }
                        cnorm[jp] = std::sqrt(sum);
                    }
                }
            }
        }

        // Apply block reflector to trailing matrix if we have a panel
        if (panel_rank > 0 && j + panel_rank < n) {
            // Build WY representation for the panel
            auto panel = A.submatrix(j, m, j, j + panel_rank);
            Matrix<T> V(m - j, panel_rank);

            // Copy Householder vectors to V
            for (std::size_t jj = 0; jj < panel_rank; ++jj) {
                V(jj, jj) = T(1);
                for (std::size_t i = jj + 1; i < m - j; ++i) {
                    V(i, jj) = panel(i, jj);
                }
            }

            // Build T matrix
            Matrix<T> Tmat(panel_rank, panel_rank);
            std::vector<T> tau_panel(panel_rank);
            for (std::size_t jj = 0; jj < panel_rank; ++jj) {
                tau_panel[jj] = tau[j + jj];
            }
            form_block_T_forward_columnwise(V, tau_panel, Tmat);

            // Apply to trailing matrix: C = (I - V*T*V^H) * C
            if (j + panel_rank < n) {
                auto C = A.submatrix(j, m, j + panel_rank, n);

                // W = V^H * C
                Matrix<T> W(panel_rank, n - j - panel_rank);
                for (std::size_t i = 0; i < panel_rank; ++i) {
                    for (std::size_t jj = 0; jj < n - j - panel_rank; ++jj) {
                        T sum = 0;
                        for (std::size_t ii = 0; ii < m - j; ++ii) {
                            sum += conj_cpqr(V(ii, i)) * C(ii, jj);
                        }
                        W(i, jj) = sum;
                    }
                }

                // Y = T * W
                Matrix<T> Y(panel_rank, n - j - panel_rank);
                for (std::size_t i = 0; i < panel_rank; ++i) {
                    for (std::size_t jj = 0; jj < n - j - panel_rank; ++jj) {
                        T sum = 0;
                        for (std::size_t ii = 0; ii < panel_rank; ++ii) {
                            sum += Tmat(i, ii) * W(ii, jj);
                        }
                        Y(i, jj) = sum;
                    }
                }

                // C = C - V * Y
                for (std::size_t i = 0; i < m - j; ++i) {
                    for (std::size_t jj = 0; jj < n - j - panel_rank; ++jj) {
                        T sum = 0;
                        for (std::size_t ii = 0; ii < panel_rank; ++ii) {
                            sum += V(i, ii) * Y(ii, jj);
                        }
                        C(i, jj) -= sum;
                    }
                }

                // Update column norms after block update
                for (std::size_t jp = j + panel_rank; jp < n; ++jp) {
                    R sum = 0;
                    for (std::size_t i = j + panel_rank; i < m; ++i) {
                        if constexpr (is_complex_number_v<T>) {
                            sum += std::norm(A(i, jp));
                        } else {
                            sum += A(i, jp) * A(i, jp);
                        }
                    }
                    cnorm[jp] = std::sqrt(sum);
                }
            }
        }

        j += panel_rank;

        if (panel_rank == 0) {
            break; // Matrix is rank deficient
        }
    }

    return 0;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_QR_PIVOTED_BLOCKED_H
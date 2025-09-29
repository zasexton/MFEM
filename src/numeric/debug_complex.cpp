#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include "include/decompositions/eigen.h"
#include "include/core/matrix.h"
#include "include/core/vector.h"

using namespace fem::numeric;
using namespace fem::numeric::decompositions;
using Z = std::complex<double>;

template<typename T>
void print_matrix(const Matrix<T>& M, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            if constexpr (std::is_same_v<T, Z>) {
                std::cout << "(" << std::setw(7) << std::setprecision(4) << M(i,j).real()
                          << "," << std::setw(7) << std::setprecision(4) << M(i,j).imag() << ") ";
            } else {
                std::cout << std::setw(10) << std::setprecision(6) << M(i, j) << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Create a simple 3x3 Hermitian matrix
    Matrix<Z> A(3, 3, Z(0,0));

    // Set up a Hermitian matrix
    A(0,0) = Z(2.0, 0.0);  // Diagonal must be real
    A(1,1) = Z(3.0, 0.0);
    A(2,2) = Z(1.0, 0.0);

    A(0,1) = Z(1.0, 1.0);   // Off-diagonal
    A(1,0) = Z(1.0, -1.0);  // Conjugate

    A(0,2) = Z(0.5, -0.5);
    A(2,0) = Z(0.5, 0.5);   // Conjugate

    A(1,2) = Z(0.0, 1.0);
    A(2,1) = Z(0.0, -1.0);  // Conjugate

    print_matrix(A, "Original Hermitian A");

    // Check that it's actually Hermitian
    std::cout << "Verifying Hermitian property:\n";
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z diff = A(i,j) - std::conj(A(j,i));
            if (std::abs(diff) > 1e-10) {
                std::cout << "  A[" << i << "," << j << "] != conj(A[" << j << "," << i << "])\n";
            }
        }
    }
    std::cout << "Hermitian check complete.\n\n";

    // Apply tridiagonalization
    Matrix<Z> A_work = A;
    std::vector<double> diag;
    std::vector<double> sub;
    Matrix<Z> Q;

    fem::numeric::decompositions::detail::hermitian_to_tridiagonal(A_work, diag, sub, &Q);

    print_matrix(Q, "Q from tridiagonalization");

    // Check Q is unitary (Q^H * Q = I)
    std::cout << "Checking Q^H * Q = I:\n";
    Matrix<Z> QHQ(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z sum(0,0);
            for (size_t k = 0; k < 3; ++k) {
                sum += std::conj(Q(k,i)) * Q(k,j);
            }
            QHQ(i,j) = sum;
        }
    }
    print_matrix(QHQ, "Q^H * Q");

    // Build T from diag and sub
    std::cout << "Tridiagonal T:\n";
    std::cout << "diag = [";
    for (size_t i = 0; i < diag.size(); ++i) {
        std::cout << diag[i];
        if (i < diag.size()-1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "sub = [";
    for (size_t i = 0; i < sub.size(); ++i) {
        std::cout << sub[i];
        if (i < sub.size()-1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // Compute Q^H * A * Q
    Matrix<Z> QH(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            QH(i,j) = std::conj(Q(j,i));
        }
    }

    Matrix<Z> QHA(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z sum(0,0);
            for (size_t k = 0; k < 3; ++k) {
                sum += QH(i,k) * A(k,j);
            }
            QHA(i,j) = sum;
        }
    }

    Matrix<Z> QHAQ(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z sum(0,0);
            for (size_t k = 0; k < 3; ++k) {
                sum += QHA(i,k) * Q(k,j);
            }
            QHAQ(i,j) = sum;
        }
    }

    print_matrix(QHAQ, "Q^H * A * Q (should be tridiagonal)");

    // Try to reconstruct A = Q * T * Q^H
    Matrix<double> T(3, 3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        T(i,i) = diag[i];
        if (i < 2) {
            T(i,i+1) = sub[i];
            T(i+1,i) = sub[i];
        }
    }

    Matrix<Z> QT(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z sum(0,0);
            for (size_t k = 0; k < 3; ++k) {
                sum += Q(i,k) * T(k,j);
            }
            QT(i,j) = sum;
        }
    }

    Matrix<Z> recon(3, 3, Z(0,0));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z sum(0,0);
            for (size_t k = 0; k < 3; ++k) {
                sum += QT(i,k) * std::conj(Q(j,k));
            }
            recon(i,j) = sum;
        }
    }

    print_matrix(recon, "Reconstructed A = Q * T * Q^H");

    // Show error
    Matrix<Z> error(3, 3, Z(0,0));
    double max_error = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            error(i,j) = recon(i,j) - A(i,j);
            max_error = std::max(max_error, std::abs(error(i,j)));
        }
    }
    print_matrix(error, "Error (Recon - Original)");
    std::cout << "Max error: " << max_error << "\n";

    return 0;
}
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>
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
    std::cout << std::fixed << std::setprecision(6);

    // Create a simple 4x4 Hermitian matrix
    Matrix<Z> A(4, 4, Z(0,0));

    // Fill with random Hermitian values
    std::srand(321);
    for (size_t i = 0; i < 4; ++i) {
        A(i, i) = Z(rand() % 5 + 1, 0);  // Real diagonal
        for (size_t j = i + 1; j < 4; ++j) {
            double real = (rand() % 10 - 5) / 5.0;
            double imag = (rand() % 10 - 5) / 5.0;
            A(i, j) = Z(real, imag);
            A(j, i) = Z(real, -imag);  // Hermitian
        }
    }

    print_matrix(A, "Original Hermitian A");

    // Compute eigendecomposition
    Vector<double> evals;
    Matrix<Z> evecs;

    int info = eigen_symmetric(A, evals, evecs);

    std::cout << "Eigendecomposition result: " << info << "\n\n";

    // Print eigenvalues
    std::cout << "Eigenvalues:\n";
    for (size_t i = 0; i < evals.size(); ++i) {
        std::cout << "λ[" << i << "] = " << evals[i] << "\n";
    }
    std::cout << "\n";

    // Check eigenvector properties
    std::cout << "Checking A*v = λ*v for each eigenvector:\n";
    double max_error = 0;
    for (size_t j = 0; j < evecs.cols(); ++j) {
        // Extract eigenvector j
        Vector<Z> v(evecs.rows(), Z(0,0));
        for (size_t i = 0; i < evecs.rows(); ++i) {
            v[i] = evecs(i, j);
        }

        // Compute A*v
        Vector<Z> Av(A.rows(), Z(0,0));
        for (size_t i = 0; i < A.rows(); ++i) {
            Z sum = Z(0,0);
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * v[k];
            }
            Av[i] = sum;
        }

        // Check against λ*v
        double error = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            Z diff = Av[i] - evals[j] * v[i];
            error = std::max(error, std::abs(diff));
        }
        max_error = std::max(max_error, error);
        std::cout << "Vector " << j << ": max|A*v - λ*v| = " << error << "\n";
    }
    std::cout << "Overall max error: " << max_error << "\n\n";

    // Check reconstruction
    std::cout << "Checking reconstruction A = V*Λ*V^H:\n";
    Matrix<Z> Lambda(evecs.cols(), evecs.cols(), Z(0,0));
    for (size_t i = 0; i < evecs.cols(); ++i) {
        Lambda(i, i) = Z(evals[i], 0);
    }

    // Compute V*Λ
    Matrix<Z> VL(evecs.rows(), evecs.cols(), Z(0,0));
    for (size_t i = 0; i < evecs.rows(); ++i) {
        for (size_t j = 0; j < evecs.cols(); ++j) {
            VL(i, j) = evecs(i, j) * evals[j];
        }
    }

    // Compute V*Λ*V^H = A_reconstructed
    Matrix<Z> A_recon(evecs.rows(), evecs.rows(), Z(0,0));
    for (size_t i = 0; i < evecs.rows(); ++i) {
        for (size_t j = 0; j < evecs.rows(); ++j) {
            Z sum = Z(0,0);
            for (size_t k = 0; k < evecs.cols(); ++k) {
                sum += VL(i, k) * std::conj(evecs(j, k));
            }
            A_recon(i, j) = sum;
        }
    }

    print_matrix(A_recon, "Reconstructed A");

    // Check reconstruction error
    double recon_error = 0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            recon_error = std::max(recon_error, std::abs(A_recon(i, j) - A(i, j)));
        }
    }
    std::cout << "Max reconstruction error: " << recon_error << "\n";

    return 0;
}
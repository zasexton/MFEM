#include <iostream>
#include <iomanip>
#include <vector>
#include "include/decompositions/eigen.h"
#include "include/core/matrix.h"

using namespace fem::numeric;
using namespace fem::numeric::decompositions;

template<typename T>
void print_matrix(const Matrix<T>& M, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            std::cout << std::setw(12) << std::setprecision(6) << M(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template<typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < B.cols(); ++j) {
            T s{};
            for (std::size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

template<typename T>
Matrix<T> transpose(const Matrix<T>& A) {
    Matrix<T> AT(A.cols(), A.rows(), T{});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            AT(j, i) = A(i, j);
    return AT;
}

int main() {
    // Test simple 3x3 symmetric matrix
    Matrix<double> A = {{4.0, 1.0, 2.0},
                        {1.0, 2.0, 0.0},
                        {2.0, 0.0, 3.0}};

    print_matrix(A, "Original A");

    Matrix<double> A_work = A;
    std::vector<double> diag;
    std::vector<double> sub;
    Matrix<double> Q;

    fem::numeric::decompositions::detail::hermitian_to_tridiagonal(A_work, diag, sub, &Q);

    print_matrix(Q, "Q from tridiagonalization");

    // Build tridiagonal matrix
    Matrix<double> T(3, 3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        T(i, i) = diag[i];
        if (i < 2) {
            T(i, i+1) = sub[i];
            T(i+1, i) = sub[i];
        }
    }
    print_matrix(T, "Tridiagonal T");

    // Try both reconstructions
    std::cout << "Testing Q * T * Q^T:\n";
    auto temp1 = matmul(Q, T);
    auto recon1 = matmul(temp1, transpose(Q));
    print_matrix(recon1, "Q * T * Q^T");

    std::cout << "Testing Q^T * A * Q:\n";
    auto temp2 = matmul(transpose(Q), A);
    auto recon2 = matmul(temp2, Q);
    print_matrix(recon2, "Q^T * A * Q");

    return 0;
}
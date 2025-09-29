#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

void print_matrix(const vector<vector<Z>>& A, const string& name) {
    cout << name << ":\n";
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            cout << "(" << setw(7) << setprecision(4) << A[i][j].real()
                 << "," << setw(7) << setprecision(4) << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }
}

// Apply similarity transformation: A = D^H * A * D
// where D is diagonal with D[idx][idx] = phase
void apply_diagonal_similarity(vector<vector<Z>>& A, size_t idx, Z phase) {
    size_t n = A.size();
    Z phase_conj = conj(phase);

    // Multiply column idx from right by phase
    for (size_t i = 0; i < n; ++i) {
        if (i != idx) {  // Diagonal stays real
            A[i][idx] *= phase;
        }
    }

    // Multiply row idx from left by phase_conj
    for (size_t j = 0; j < n; ++j) {
        if (j != idx) {  // Diagonal stays real
            A[idx][j] *= phase_conj;
        }
    }
    // Note: A[idx][idx] gets phase_conj * A[idx][idx] * phase = A[idx][idx] (real)
}

int main() {
    cout << fixed << setprecision(6);

    // After Householder transformation on column 0, we have this matrix:
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(-1.118034, -1.118034), Z(0.0, 0.0)},
        {Z(-1.118034, 1.118034), Z(1.8, 0.0), Z(0.0, -1.4)},
        {Z(0.0, 0.0), Z(0.0, 1.4), Z(2.2, 0.0)}
    };

    print_matrix(A, "After Householder (complex subdiagonal)");

    // The problem: A[1][0] = (-1.118034, 1.118034) is complex
    Z subdiag = A[1][0];
    cout << "\nSubdiagonal A[1][0] = (" << subdiag.real() << ", " << subdiag.imag() << ")\n";

    if (abs(subdiag.imag()) > 1e-10) {
        cout << "Subdiagonal is complex, applying phase correction...\n";

        // Compute phase to rotate subdiag to real
        // We want phase such that phase * subdiag is real
        // This means phase * subdiag = |subdiag|
        // So phase = |subdiag| / subdiag = conj(subdiag) / |subdiag|
        // Wait, that's not right. We want phase_conj * subdiag to be real.
        // So we need phase_conj * subdiag = |subdiag|
        // Thus phase_conj = |subdiag| / subdiag = conj(subdiag/|subdiag|)
        // Therefore phase = subdiag / |subdiag|
        Z phase = subdiag / abs(subdiag);
        cout << "Phase factor to apply: (" << phase.real() << ", " << phase.imag() << ")\n";

        // Apply diagonal similarity to row/column 1
        apply_diagonal_similarity(A, 1, phase);

        print_matrix(A, "\nAfter phase correction");

        cout << "\nNew A[1][0] = (" << A[1][0].real() << ", " << A[1][0].imag() << ")\n";
        cout << "New A[0][1] = (" << A[0][1].real() << ", " << A[0][1].imag() << ")\n";

        // Check Hermitian property
        bool is_hermitian = true;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                Z diff = A[i][j] - conj(A[j][i]);
                if (abs(diff) > 1e-10) {
                    cout << "Not Hermitian at [" << i << "," << j << "]: diff = " << diff << "\n";
                    is_hermitian = false;
                }
            }
        }
        if (is_hermitian) {
            cout << "Matrix is still Hermitian!\n";
        }

        // Check if subdiagonal is real
        cout << "\nSubdiagonal elements:\n";
        cout << "A[1][0] = " << A[1][0] << " (should be real)\n";
        cout << "A[2][1] = " << A[2][1] << " (should be real for full tridiagonal)\n";
    }

    return 0;
}
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
    cout << "\n";
}

int main() {
    // After Householder, we have this matrix
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(-1.1180, -1.1180), Z(0.0, 0.0)},
        {Z(-1.1180, 1.1180), Z(1.8, 0.0), Z(0.0, -1.4)},
        {Z(0.0, 0.0), Z(0.0, 1.4), Z(2.2, 0.0)}
    };

    print_matrix(A, "After Householder");

    // The problem: A[1][0] = (-1.118, 1.118) is complex
    // We need to apply a diagonal similarity to make it real

    Z alpha = A[1][0];
    cout << "Subdiagonal element: (" << alpha.real() << ", " << alpha.imag() << ")\n";
    cout << "|alpha| = " << abs(alpha) << "\n";

    // Compute phase
    Z phase = alpha / abs(alpha);
    Z phase_conj = conj(phase);
    cout << "Phase: (" << phase.real() << ", " << phase.imag() << ")\n";
    cout << "Conj(phase): (" << phase_conj.real() << ", " << phase_conj.imag() << ")\n\n";

    // Apply D^H * A * D where D = diag(1, phase_conj, 1)
    // This means:
    // - Multiply row 1 by phase (from left)
    // - Multiply column 1 by phase_conj (from right)

    // Create a copy to modify
    auto A_corrected = A;

    // Multiply row 1 by phase (from the left)
    for (size_t j = 0; j < 3; ++j) {
        A_corrected[1][j] *= phase;
    }

    // Multiply column 1 by phase_conj (from the right)
    for (size_t i = 0; i < 3; ++i) {
        A_corrected[i][1] *= phase_conj;
    }

    // Note: A[1][1] gets both: phase * A[1][1] * phase_conj
    // Since A[1][1] is real (Hermitian diagonal), this stays real

    print_matrix(A_corrected, "After phase correction");

    cout << "New A[1][0] = (" << A_corrected[1][0].real() << ", " << A_corrected[1][0].imag() << ")\n";
    cout << "New A[0][1] = (" << A_corrected[0][1].real() << ", " << A_corrected[0][1].imag() << ")\n";

    // Verify it's still Hermitian
    cout << "\nVerifying Hermitian property:\n";
    bool is_hermitian = true;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Z diff = A_corrected[i][j] - conj(A_corrected[j][i]);
            if (abs(diff) > 1e-10) {
                cout << "  A[" << i << "," << j << "] != conj(A[" << j << "," << i << "])\n";
                cout << "  Difference: " << diff << "\n";
                is_hermitian = false;
            }
        }
    }
    if (is_hermitian) {
        cout << "Matrix is Hermitian!\n";
    }

    // Check if subdiagonal is now real
    cout << "\nSubdiagonal elements:\n";
    cout << "A[1][0] = " << A_corrected[1][0] << " (should be real)\n";
    cout << "A[2][1] = " << A_corrected[2][1] << " (should be real)\n";

    return 0;
}
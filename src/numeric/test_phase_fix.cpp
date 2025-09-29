#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

// Print complex matrix
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

// Check if matrix is Hermitian
bool is_hermitian(const vector<vector<Z>>& A) {
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Z diff = A[i][j] - conj(A[j][i]);
            if (abs(diff) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

// Check if tridiagonal with real subdiagonals
bool is_real_tridiagonal(const vector<vector<Z>>& A) {
    size_t n = A.size();

    // Check off-tridiagonal elements are zero
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (abs(static_cast<int>(i) - static_cast<int>(j)) > 1) {
                if (abs(A[i][j]) > 1e-10) {
                    cout << "Non-zero at [" << i << "," << j << "] = " << A[i][j] << "\n";
                    return false;
                }
            }
        }
    }

    // Check diagonal is real
    for (size_t i = 0; i < n; ++i) {
        if (abs(A[i][i].imag()) > 1e-10) {
            cout << "Complex diagonal at [" << i << "," << i << "] = " << A[i][i] << "\n";
            return false;
        }
    }

    // Check subdiagonals are real
    for (size_t i = 0; i < n-1; ++i) {
        if (abs(A[i+1][i].imag()) > 1e-10) {
            cout << "Complex subdiag at [" << i+1 << "," << i << "] = " << A[i+1][i] << "\n";
            return false;
        }
        if (abs(A[i][i+1].imag()) > 1e-10) {
            cout << "Complex superdiag at [" << i << "," << i+1 << "] = " << A[i][i+1] << "\n";
            return false;
        }
    }

    return true;
}

int main() {
    cout << fixed << setprecision(6);

    // Test Hermitian matrix
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(1.0, 1.0), Z(0.5, -0.5)},
        {Z(1.0, -1.0), Z(3.0, 0.0), Z(0.0, 1.0)},
        {Z(0.5, 0.5), Z(0.0, -1.0), Z(1.0, 0.0)}
    };

    print_matrix(A, "Original Hermitian A");
    cout << "Is Hermitian? " << (is_hermitian(A) ? "Yes" : "No") << "\n\n";

    // The problem: after Householder reduction, we get complex subdiagonals
    // Example after first step:
    vector<vector<Z>> A_step1 = {
        {Z(2.0, 0.0), Z(-1.118034, -1.118034), Z(0.0, 0.0)},
        {Z(-1.118034, 1.118034), Z(1.8, 0.0), Z(0.0, -1.4)},
        {Z(0.0, 0.0), Z(0.0, 1.4), Z(2.2, 0.0)}
    };

    print_matrix(A_step1, "After Householder step 1 (complex subdiag!)");
    cout << "Is real tridiagonal? " << (is_real_tridiagonal(A_step1) ? "Yes" : "No") << "\n\n";

    // The fix: extract phase and apply diagonal similarity
    Z subdiag = A_step1[1][0];
    cout << "Subdiagonal element: " << subdiag << "\n";

    if (abs(subdiag.imag()) > 1e-10) {
        // Compute phase
        Z phase = subdiag / abs(subdiag);
        cout << "Phase to remove: " << phase << "\n";

        // Create diagonal matrix D with phase at position 1
        vector<Z> D = {Z(1, 0), phase, Z(1, 0)};

        // Apply similarity: A' = D^H * A * D
        vector<vector<Z>> A_fixed = A_step1;

        // Apply D from right (multiply columns)
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                A_fixed[i][j] *= D[j];
            }
        }

        // Apply D^H from left (multiply rows)
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                A_fixed[i][j] *= conj(D[i]);
            }
        }

        print_matrix(A_fixed, "\nAfter phase correction D^H * A * D");
        cout << "Is Hermitian? " << (is_hermitian(A_fixed) ? "Yes" : "No") << "\n";
        cout << "Is real tridiagonal? " << (is_real_tridiagonal(A_fixed) ? "Yes" : "No") << "\n\n";

        // Key insight: The phase must be tracked cumulatively!
        cout << "Phase tracking:\n";
        cout << "Row 0: phase = 1\n";
        cout << "Row 1: phase = " << phase << "\n";
        cout << "Row 2: phase = " << phase << " (inherited from row 1)\n";
    }

    return 0;
}
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

int main() {
    cout << fixed << setprecision(6);

    // Simulate the phase extraction from hermitian_to_tridiagonal result
    size_t n = 3;
    vector<Z> phase(n, Z(1, 0));

    // After first Householder step, subdiag[0] = (-1.118, 1.118)
    Z a1 = Z(-1.118034, 1.118034);
    double aa1 = abs(a1);
    Z s1 = a1 / aa1;  // Unit phase

    phase[0] = Z(1, 0);
    phase[1] = phase[0] * s1;

    cout << "After step 1:\n";
    cout << "a1 = " << a1 << ", |a1| = " << aa1 << "\n";
    cout << "s1 = " << s1 << "\n";
    cout << "phase[0] = " << phase[0] << "\n";
    cout << "phase[1] = " << phase[1] << "\n\n";

    // After second step, assume we get some complex subdiag
    // Let's say it's (-0.99, -0.99) as we saw in the test
    Z a2 = Z(-0.99, -0.99);
    double aa2 = abs(a2);
    Z s2 = a2 / aa2;

    phase[2] = phase[1] * s2;

    cout << "After step 2:\n";
    cout << "a2 = " << a2 << ", |a2| = " << aa2 << "\n";
    cout << "s2 = " << s2 << "\n";
    cout << "phase[2] = " << phase[2] << "\n\n";

    // Check phase properties
    cout << "Phase magnitudes:\n";
    for (size_t i = 0; i < n; ++i) {
        cout << "|phase[" << i << "]| = " << abs(phase[i]) << "\n";
    }

    // The key question: if we apply these phases to a real tridiagonal,
    // do we recover the complex Hermitian?

    cout << "\nPhase matrix D:\n";
    for (size_t i = 0; i < n; ++i) {
        cout << "D[" << i << "," << i << "] = " << phase[i] << "\n";
    }

    // If T is real tridiagonal and A = Q * D^H * T * D * Q^H
    // Then D transforms the real T to complex with correct phases

    return 0;
}
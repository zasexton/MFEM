#include <iostream>
#include <complex>
#include <cmath>

using namespace std;
using Z = complex<double>;

int main() {
    // Simple 2x2 Hermitian matrix
    Z A[2][2] = {{Z(3,0), Z(1,1)},
                 {Z(1,-1), Z(2,0)}};

    cout << "Original 2x2 Hermitian:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << "(" << A[i][j].real() << "," << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }

    // We want to zero A[1][0] using a Householder
    // x = [A[1][0]] = [1-i]
    Z x = A[1][0];
    double xnorm = abs(x);
    cout << "\nElement to zero: x = (" << x.real() << "," << x.imag() << ")\n";
    cout << "|x| = " << xnorm << "\n";

    // For a 1x1 case (zeroing a single element), the Householder is simpler
    // We need a unitary transformation U such that U*A*U^H has real subdiagonal

    // The key is that we want U*[1-i] to be real
    // This means U = phase^* where phase = x/|x|
    Z phase = x / xnorm;
    cout << "phase = (" << phase.real() << "," << phase.imag() << ")\n";

    // Apply: A' = U^H * A * U where U = conj(phase)
    Z U = conj(phase);
    cout << "U = conj(phase) = (" << U.real() << "," << U.imag() << ")\n";

    // New A[1][0] = U^H * A[1][0] = phase * (1-i) = |x|
    Z new_10 = phase * x;
    cout << "New A[1][0] = phase * x = (" << new_10.real() << "," << new_10.imag() << ")\n";
    cout << "Should be real and equal to " << xnorm << "\n";

    return 0;
}
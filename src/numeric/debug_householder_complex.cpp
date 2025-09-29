#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <cmath>

using namespace std;
using Z = complex<double>;

int main() {
    // Test Householder for complex vector
    // We want to zero out all elements except the first
    vector<Z> x = {Z(1.0, -1.0), Z(0.5, 0.5)};

    cout << "Original x: [";
    for (const auto& xi : x) {
        cout << "(" << xi.real() << "," << xi.imag() << ") ";
    }
    cout << "]\n";

    // Compute ||x||
    double xnorm = 0;
    for (const auto& xi : x) {
        xnorm += norm(xi);  // |xi|^2
    }
    xnorm = sqrt(xnorm);
    cout << "||x|| = " << xnorm << "\n";

    // Choose alpha to make the result real
    // Standard choice: alpha = -sign(x[0]) * ||x|| * exp(i*arg(x[0]))
    Z alpha;
    if (abs(x[0]) < 1e-10) {
        alpha = Z(xnorm, 0);
    } else {
        Z phase = x[0] / abs(x[0]);
        alpha = -phase * xnorm;
    }
    cout << "alpha = (" << alpha.real() << "," << alpha.imag() << ")\n";

    // Compute v = x - alpha*e1
    vector<Z> v = x;
    v[0] -= alpha;
    cout << "v = [";
    for (const auto& vi : v) {
        cout << "(" << vi.real() << "," << vi.imag() << ") ";
    }
    cout << "]\n";

    // Compute ||v||^2
    double vnorm2 = 0;
    for (const auto& vi : v) {
        vnorm2 += norm(vi);
    }
    cout << "||v||^2 = " << vnorm2 << "\n";

    // beta = 2/||v||^2
    double beta = 2.0 / vnorm2;
    cout << "beta = " << beta << "\n";

    // Apply Householder: y = H*x = x - beta*(v^H*x)*v
    Z vHx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vHx += conj(v[i]) * x[i];
    }
    cout << "v^H * x = (" << vHx.real() << "," << vHx.imag() << ")\n";

    vector<Z> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] - beta * vHx * v[i];
    }

    cout << "H*x = [";
    for (const auto& yi : y) {
        cout << "(" << yi.real() << "," << yi.imag() << ") ";
    }
    cout << "]\n";

    // The first element should be real and equal to ±||x||
    cout << "\nExpected: first element = " << xnorm << " (real)\n";
    cout << "Got: first element = (" << y[0].real() << "," << y[0].imag() << ")\n";

    // Now let's test with a 2x2 Hermitian submatrix
    cout << "\n=== Testing 2x2 Hermitian Reduction ===\n";
    Z A[2][2] = {{Z(2.0, 0), Z(0.0, 1.0)},
                 {Z(0.0, -1.0), Z(3.0, 0)}};

    cout << "Original A:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << "(" << setw(4) << A[i][j].real() << "," << setw(4) << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }

    // Extract column below diagonal
    Z x_col = A[1][0];  // Element to zero out
    cout << "\nx to zero = (" << x_col.real() << "," << x_col.imag() << ")\n";
    double x_mag = abs(x_col);
    cout << "|x| = " << x_mag << "\n";

    // For a scalar, the Householder that zeros it is trivial
    // But we need the resulting (1,0) element to be real
    // After transformation: A'[1,0] should be real

    // The transformation should make A[1,0] = |A[1,0]| (real)

    return 0;
}
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

// Apply Householder: y = (I - beta*v*v^H)*x
vector<Z> apply_householder(const vector<Z>& x, const vector<Z>& v, double beta) {
    Z vHx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vHx += conj(v[i]) * x[i];
    }

    vector<Z> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] - beta * vHx * v[i];
    }
    return y;
}

int main() {
    // Test vector from a Hermitian matrix column
    vector<Z> x = {Z(1.0, -1.0), Z(0.5, 0.5)};

    cout << fixed << setprecision(6);
    cout << "Original x: ";
    for (const auto& xi : x) {
        cout << "(" << xi.real() << "," << xi.imag() << ") ";
    }
    cout << "\n";

    // Compute ||x||
    double xnorm = 0;
    for (const auto& xi : x) {
        xnorm += norm(xi);  // |xi|^2
    }
    xnorm = sqrt(xnorm);
    cout << "||x|| = " << xnorm << "\n";

    // Choose alpha following LAPACK's approach
    Z alpha;
    if (abs(x[0]) < 1e-14) {
        alpha = xnorm;
    } else {
        // Key: alpha = -phase(x[0]) * ||x||
        Z x0_phase = x[0] / abs(x[0]);
        alpha = -x0_phase * xnorm;
    }

    cout << "alpha = (" << alpha.real() << "," << alpha.imag() << ")\n";
    cout << "|alpha| = " << abs(alpha) << "\n";

    // Compute v = x - alpha*e1
    vector<Z> v = x;
    v[0] -= alpha;

    cout << "v = ";
    for (const auto& vi : v) {
        cout << "(" << vi.real() << "," << vi.imag() << ") ";
    }
    cout << "\n";

    // Compute ||v||^2
    double vnorm2 = 0;
    for (const auto& vi : v) {
        vnorm2 += norm(vi);
    }
    cout << "||v||^2 = " << vnorm2 << "\n";

    if (vnorm2 < 1e-14) {
        cout << "v is zero, no reflection needed\n";
        return 0;
    }

    double beta = 2.0 / vnorm2;
    cout << "beta = " << beta << "\n";

    // Apply Householder to x
    auto y = apply_householder(x, v, beta);

    cout << "\nH*x = ";
    for (const auto& yi : y) {
        cout << "(" << yi.real() << "," << yi.imag() << ") ";
    }
    cout << "\n";

    cout << "Expected: (alpha, 0) = (" << alpha.real() << "," << alpha.imag() << ") (0,0)\n";

    // Check if y[0] equals alpha
    Z diff = y[0] - alpha;
    cout << "Difference: (" << diff.real() << "," << diff.imag() << ")\n";
    cout << "|difference| = " << abs(diff) << "\n";

    // Now let's see what happens when we apply this to a Hermitian matrix
    cout << "\n=== Hermitian Matrix Application ===\n";

    // Consider the 3x3 Hermitian matrix from before
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(1.0, 1.0), Z(0.5, -0.5)},
        {Z(1.0, -1.0), Z(3.0, 0.0), Z(0.0, 1.0)},
        {Z(0.5, 0.5), Z(0.0, -1.0), Z(1.0, 0.0)}
    };

    cout << "After Householder, A[1][0] should become alpha = ("
         << alpha.real() << "," << alpha.imag() << ")\n";

    // The key question: what do we store as the subdiagonal?
    cout << "\nWhat to store as subdiagonal:\n";
    cout << "Option 1: alpha itself = (" << alpha.real() << "," << alpha.imag() << ")\n";
    cout << "Option 2: |alpha| = " << abs(alpha) << " (always real)\n";
    cout << "Option 3: Since H*x = alpha*e1, and we want real, we need additional phase correction\n";

    return 0;
}
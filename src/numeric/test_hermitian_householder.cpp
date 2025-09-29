#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

using namespace std;
using Z = complex<double>;

// Compute Householder that makes H*x = ||x||*e1 (real)
void householder_complex(vector<Z>& x, Z& alpha, vector<Z>& v, double& beta) {
    // Compute ||x||
    double xnorm = 0;
    for (const auto& xi : x) {
        xnorm += norm(xi);
    }
    xnorm = sqrt(xnorm);

    if (xnorm < 1e-14) {
        alpha = 0;
        beta = 0;
        return;
    }

    // For complex, we want H*x to be real
    // Standard approach: alpha = -sign(x[0]) * ||x|| * exp(i*arg(x[0]))
    if (abs(x[0]) < 1e-14) {
        alpha = xnorm;
    } else {
        Z phase = x[0] / abs(x[0]);
        // Choose alpha to maximize |v[0]| = |x[0] - alpha|
        alpha = -phase * xnorm;
    }

    // Compute v = x - alpha*e1
    v = x;
    v[0] -= alpha;

    // Compute ||v||^2
    double vnorm2 = 0;
    for (const auto& vi : v) {
        vnorm2 += norm(vi);
    }

    if (vnorm2 < 1e-14) {
        beta = 0;
        return;
    }

    beta = 2.0 / vnorm2;
}

int main() {
    // Test case: 3x3 Hermitian matrix
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(1.0, 1.0), Z(0.5, -0.5)},
        {Z(1.0, -1.0), Z(3.0, 0.0), Z(0.0, 1.0)},
        {Z(0.5, 0.5), Z(0.0, -1.0), Z(1.0, 0.0)}
    };

    cout << "Original Hermitian A:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << "(" << A[i][j].real() << "," << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }

    // First Householder: zero out A[2][0] and A[3][0]
    vector<Z> x = {A[1][0], A[2][0]};
    cout << "\nFirst column below diagonal: ";
    for (const auto& xi : x) {
        cout << "(" << xi.real() << "," << xi.imag() << ") ";
    }
    cout << "\n";

    Z alpha;
    vector<Z> v;
    double beta;
    householder_complex(x, alpha, v, beta);

    cout << "alpha = (" << alpha.real() << "," << alpha.imag() << ")\n";
    cout << "|alpha| = " << abs(alpha) << "\n";
    cout << "beta = " << beta << "\n";

    // Apply H*x
    Z vHx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vHx += conj(v[i]) * x[i];
    }

    vector<Z> Hx(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        Hx[i] = x[i] - beta * vHx * v[i];
    }

    cout << "\nH*x = ";
    for (const auto& h : Hx) {
        cout << "(" << h.real() << "," << h.imag() << ") ";
    }
    cout << "\n";
    cout << "Expected: (" << abs(alpha) << ", 0) (0, 0)\n";

    // The key insight: after Householder, we get alpha*e1
    // But alpha might be complex!
    // We need an additional phase rotation to make it real

    cout << "\n=== Phase correction ===\n";
    if (abs(alpha.imag()) > 1e-10) {
        Z phase = alpha / abs(alpha);
        cout << "alpha has phase: (" << phase.real() << "," << phase.imag() << ")\n";
        cout << "Need to rotate by conj(phase) to make it real\n";

        // After rotation, the subdiagonal becomes |alpha|
        cout << "After phase correction, subdiagonal = " << abs(alpha) << "\n";
    } else {
        cout << "alpha is already real (or close enough)\n";
    }

    return 0;
}
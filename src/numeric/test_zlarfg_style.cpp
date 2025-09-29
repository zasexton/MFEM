#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

// LAPACK-style ZLARFG
void zlarfg(int n, Z& alpha, vector<Z>& x, Z& tau) {
    if (n <= 1) {
        tau = 0;
        return;
    }

    // Compute xnorm = ||x[1:]||
    double xnorm = 0;
    for (int i = 1; i < n; ++i) {
        xnorm += norm(x[i]);
    }
    xnorm = sqrt(xnorm);

    if (xnorm < 1e-14 && abs(alpha.imag()) < 1e-14) {
        // H = I
        tau = 0;
    } else {
        // Compute beta = -sign(Re(alpha)) * ||[alpha, x[1:]]||
        double alphr = alpha.real();
        double alphi = alpha.imag();
        double norm_all = sqrt(alphr*alphr + alphi*alphi + xnorm*xnorm);

        double beta;
        if (alphr >= 0) {
            beta = -norm_all;
        } else {
            beta = norm_all;
        }

        // Compute tau
        tau = Z((beta - alphr) / beta, -alphi / beta);

        // Scale x
        Z scale_factor = 1.0 / (alpha - Z(beta, 0));
        for (int i = 1; i < n; ++i) {
            x[i] *= scale_factor;
        }

        // Set x[0] = 1
        x[0] = 1;

        // Update alpha to beta (real)
        alpha = beta;
    }
}

// Apply Householder: y = (I - tau*v*v^H)*x
vector<Z> apply_householder(const vector<Z>& x, const vector<Z>& v, Z tau) {
    Z vHx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vHx += conj(v[i]) * x[i];
    }

    vector<Z> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] - tau * vHx * v[i];
    }
    return y;
}

int main() {
    cout << fixed << setprecision(6);

    // Test vector
    vector<Z> x = {Z(1.0, -1.0), Z(0.5, 0.5)};
    vector<Z> x_orig = x;  // Save original

    cout << "Original x: ";
    for (const auto& xi : x) {
        cout << "(" << xi.real() << "," << xi.imag() << ") ";
    }
    cout << "\n";

    Z alpha = x[0];
    Z tau;

    // Apply ZLARFG
    zlarfg(x.size(), alpha, x, tau);

    cout << "\nAfter ZLARFG:\n";
    cout << "alpha (should be real) = (" << alpha.real() << "," << alpha.imag() << ")\n";
    cout << "tau = (" << tau.real() << "," << tau.imag() << ")\n";
    cout << "v = ";
    for (const auto& vi : x) {
        cout << "(" << vi.real() << "," << vi.imag() << ") ";
    }
    cout << "\n";

    // Test: apply to original vector
    auto result = apply_householder(x_orig, x, tau);

    cout << "\nH*x_orig = ";
    for (const auto& r : result) {
        cout << "(" << r.real() << "," << r.imag() << ") ";
    }
    cout << "\n";
    cout << "Expected: (" << alpha.real() << ",0) (0,0)\n";

    return 0;
}
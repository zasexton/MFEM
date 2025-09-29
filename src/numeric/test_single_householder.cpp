#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Z = complex<double>;

// Apply Householder from left and right: A' = H * A * H^H
void apply_householder_symmetric(vector<vector<Z>>& A, const vector<Z>& v, double beta, size_t k) {
    size_t n = A.size();
    size_t m = n - k - 1;

    if (m == 0 || beta == 0) return;

    // Compute p = A * v (only the trailing submatrix)
    vector<Z> p(m, 0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            p[i] += A[k+1+i][k+1+j] * v[j];
        }
    }

    // Compute K = (beta/2) * v^H * p
    Z K = 0;
    for (size_t i = 0; i < m; ++i) {
        K += conj(v[i]) * p[i];
    }
    K *= beta * 0.5;

    // Compute w = p - K*v
    vector<Z> w(m);
    for (size_t i = 0; i < m; ++i) {
        w[i] = p[i] - K * v[i];
    }

    // Update A = A - beta*(v*w^H + w*v^H)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = i; j < m; ++j) {
            Z update = beta * (v[i] * conj(w[j]) + w[i] * conj(v[j]));
            A[k+1+i][k+1+j] -= update;
            if (i != j) {
                A[k+1+j][k+1+i] = conj(A[k+1+i][k+1+j]);
            }
        }
    }
}

int main() {
    // Test with 3x3 Hermitian matrix
    vector<vector<Z>> A = {
        {Z(2.0, 0.0), Z(1.0, 1.0), Z(0.5, -0.5)},
        {Z(1.0, -1.0), Z(3.0, 0.0), Z(0.0, 1.0)},
        {Z(0.5, 0.5), Z(0.0, -1.0), Z(1.0, 0.0)}
    };

    cout << fixed << setprecision(4);
    cout << "Original A:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << "(" << setw(7) << A[i][j].real() << "," << setw(7) << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }

    // First step: zero out A[2][0]
    vector<Z> x = {A[1][0], A[2][0]};
    cout << "\nColumn 0 below diagonal: ";
    for (auto xi : x) cout << "(" << xi.real() << "," << xi.imag() << ") ";
    cout << "\n";

    // Compute norm
    double xnorm = 0;
    for (auto xi : x) xnorm += norm(xi);
    xnorm = sqrt(xnorm);
    cout << "||x|| = " << xnorm << "\n";

    // Choose alpha
    Z alpha;
    if (abs(x[0]) < 1e-14) {
        alpha = xnorm;
    } else {
        Z phase = x[0] / abs(x[0]);
        alpha = -phase * xnorm;
    }
    cout << "alpha = (" << alpha.real() << "," << alpha.imag() << ")\n";
    cout << "|alpha| = " << abs(alpha) << "\n";

    // Compute v = x - alpha*e1
    vector<Z> v = x;
    v[0] -= alpha;

    double vnorm2 = 0;
    for (auto vi : v) vnorm2 += norm(vi);
    double beta = 2.0 / vnorm2;

    cout << "v = ";
    for (auto vi : v) cout << "(" << vi.real() << "," << vi.imag() << ") ";
    cout << "\nbeta = " << beta << "\n";

    // Apply Householder to the column
    // This zeros out A[2][0]
    Z vHx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vHx += conj(v[i]) * x[i];
    }

    vector<Z> newcol(2);
    for (size_t i = 0; i < 2; ++i) {
        newcol[i] = x[i] - beta * vHx * v[i];
    }

    cout << "\nAfter H*x: ";
    for (auto nc : newcol) cout << "(" << nc.real() << "," << nc.imag() << ") ";
    cout << "\n";

    // Now apply from left and right to trailing 2x2 submatrix
    apply_householder_symmetric(A, v, beta, 0);

    // Update the column elements
    A[1][0] = alpha;
    A[0][1] = conj(alpha);
    A[2][0] = 0;
    A[0][2] = 0;

    cout << "\nA after first Householder:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << "(" << setw(7) << A[i][j].real() << "," << setw(7) << A[i][j].imag() << ") ";
        }
        cout << "\n";
    }

    // Check if A[1][0] is real
    cout << "\nA[1][0] = (" << A[1][0].real() << "," << A[1][0].imag() << ")\n";
    if (abs(A[1][0].imag()) > 1e-10) {
        cout << "WARNING: Subdiagonal is not real!\n";
        cout << "Need phase correction by factor exp(i*" << arg(A[1][0]) << ")\n";
    }

    return 0;
}
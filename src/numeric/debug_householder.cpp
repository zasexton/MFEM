#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

int main() {
    // Test simple 3x3 symmetric matrix
    double A[3][3] = {{4.0, 1.0, 2.0},
                      {1.0, 2.0, 0.0},
                      {2.0, 0.0, 3.0}};

    cout << "Original A:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << setw(10) << A[i][j] << " ";
        }
        cout << "\n";
    }

    // First Householder step to zero A(2,0)
    // x = [A(1,0), A(2,0)] = [1, 2]
    double x[2] = {A[1][0], A[2][0]};
    cout << "\nx = [" << x[0] << ", " << x[1] << "]\n";

    // ||x|| = sqrt(1^2 + 2^2) = sqrt(5)
    double xnorm = sqrt(x[0]*x[0] + x[1]*x[1]);
    cout << "||x|| = " << xnorm << "\n";

    // alpha = -sign(x[0]) * ||x|| = -sqrt(5)
    double alpha = (x[0] >= 0) ? -xnorm : xnorm;
    cout << "alpha = " << alpha << "\n";

    // v = x - alpha*e1
    double v[2] = {x[0] - alpha, x[1]};
    cout << "v = [" << v[0] << ", " << v[1] << "]\n";

    // ||v||^2
    double vnorm2 = v[0]*v[0] + v[1]*v[1];
    cout << "||v||^2 = " << vnorm2 << "\n";

    // beta = 2 / ||v||^2
    double beta = 2.0 / vnorm2;
    cout << "beta = " << beta << "\n";

    // Extract the 2x2 trailing submatrix
    double A22[2][2] = {{A[1][1], A[1][2]},
                        {A[2][1], A[2][2]}};

    cout << "\nTrailing 2x2 submatrix:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << setw(10) << A22[i][j] << " ";
        }
        cout << "\n";
    }

    // p = A22 * v
    double p[2] = {A22[0][0]*v[0] + A22[0][1]*v[1],
                   A22[1][0]*v[0] + A22[1][1]*v[1]};
    cout << "\np = A22*v = [" << p[0] << ", " << p[1] << "]\n";

    // w = beta*p
    double w[2] = {beta * p[0], beta * p[1]};
    cout << "w = beta*p = [" << w[0] << ", " << w[1] << "]\n";

    // K = beta * v^T * p / 2
    double K = beta * (v[0]*p[0] + v[1]*p[1]) * 0.5;
    cout << "K = beta * v^T * p / 2 = " << K << "\n";

    // w = w - K*v
    w[0] -= K * v[0];
    w[1] -= K * v[1];
    cout << "w (final) = [" << w[0] << ", " << w[1] << "]\n";

    // Update A22 = A22 - v*w^T - w*v^T
    cout << "\nUpdating A22 = A22 - v*w^T - w*v^T:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double update = v[i]*w[j] + w[i]*v[j];
            cout << "  A22[" << i << "][" << j << "] update: " << A22[i][j]
                 << " - " << update << " = " << (A22[i][j] - update) << "\n";
            A22[i][j] -= update;
        }
    }

    cout << "\nUpdated A22:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << setw(10) << A22[i][j] << " ";
        }
        cout << "\n";
    }

    // The full updated matrix should be:
    cout << "\nFull matrix after first Householder step:\n";
    A[0][1] = alpha;  // This becomes the subdiagonal
    A[1][0] = alpha;
    A[0][2] = 0;      // This is zeroed
    A[2][0] = 0;
    A[1][1] = A22[0][0];
    A[1][2] = A22[0][1];
    A[2][1] = A22[1][0];
    A[2][2] = A22[1][1];

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << setw(10) << A[i][j] << " ";
        }
        cout << "\n";
    }

    return 0;
}
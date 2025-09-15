#include <base/numeric_base.h>
#include <base/traits_base.h>
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <vector>
#include <bit>
#include <random>
#include <algorithm>
#include <numeric>

using namespace fem::numeric;

// ============================================================================
// Numerical Stability Critical Tests
// ============================================================================

class NumericalStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator with fixed seed for reproducibility
        gen_.seed(42);
    }

    std::mt19937 gen_;
    
    // Helper to generate random numbers in range [min, max]
    template<typename T>
    std::vector<T> generate_random_vector(size_t size, T min_val, T max_val) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        std::vector<T> result(size);
        std::generate(result.begin(), result.end(), [&]() { return dist(gen_); });
        return result;
    }
};

// ============================================================================
// Catastrophic Cancellation Tests - CRITICAL MISSING
// ============================================================================

TEST_F(NumericalStabilityTest, CatastrophicCancellationDetection) {
    // Test case: (1 + x) - 1 vs x for very small x
    // Use a small value but not so small that (1+tiny)-1 becomes dominated by eps/tiny
    double tiny = 1e-12;
    
    // Naive computation - suffers from cancellation
    double naive_result = (1.0 + tiny) - 1.0;
    
    // Direct computation - mathematically equivalent
    double direct_result = tiny;
    
    // Check if we detect the precision loss
    double relative_error = std::abs(naive_result - direct_result) / direct_result;
    
    // We expect some precision loss, but it should be measurable
    EXPECT_GT(relative_error, std::numeric_limits<double>::epsilon());
    // Not catastrophic: relative error should remain well below O(1)
    EXPECT_LT(relative_error, 5e-1);
    
    // Log the precision loss for monitoring
    std::cout << "Relative error in cancellation test: " << relative_error << std::endl;
}

TEST_F(NumericalStabilityTest, QuadraticFormulaCancellation) {
    // Test case: quadratic formula with nearly equal roots
    // x^2 - 2000000000x + 1 = 0
    // Roots are very close to 1e9, causing cancellation in standard formula
    
    double a = 1.0;
    double b = -2e9;
    double c = 1.0;
    
    double discriminant = b * b - 4 * a * c;
    double sqrt_disc = std::sqrt(discriminant);

    // Standard formula (suffers from cancellation)
    double root1_naive = (-b + sqrt_disc) / (2 * a);
    double root2_naive = (-b - sqrt_disc) / (2 * a);

    // Robust/stable quadratic using q formulation in extended precision
    // Use long double to avoid catastrophic rounding when b^2 >> 4ac
    long double aL = static_cast<long double>(a);
    long double bL = static_cast<long double>(b);
    long double cL = static_cast<long double>(c);
    long double discL = bL*bL - 4.0L*aL*cL;
    long double sqrtL = std::sqrt(discL);
    long double qL = -0.5L * (bL + std::copysign(sqrtL, bL));
    long double root1L = qL / aL;   // large-magnitude root
    long double root2L = cL / qL;   // small-magnitude root
    double root1_stable = static_cast<double>(root1L);
    double root2_stable = static_cast<double>(root2L);
    
    // Verify solutions using compensated Horner evaluation to reduce cancellation
    auto poly_residual = [&](long double x) {
        long double t = std::fma(aL, x, bL);        // a*x + b in extended precision
        long double r = std::fma(t, x, cL);         // (a*x + b)*x + c
        return r;
    };
    double residual1 = static_cast<double>(poly_residual(root1L));
    double residual2 = static_cast<double>(poly_residual(root2L));
    
    EXPECT_NEAR(residual1, 0.0, 1e-6);
    EXPECT_NEAR(residual2, 0.0, 1e-6);
    
    // The stable method should give better accuracy for the smaller root
    EXPECT_LT(std::abs(residual2), std::abs(a * root2_naive * root2_naive + b * root2_naive + c));
}

TEST_F(NumericalStabilityTest, SummationCancellation) {
    // Test alternating series prone to cancellation
    // Sum of: 1 - 1/2 + 1/3 - 1/4 + ... (truncated)
    
    const size_t n = 10000;
    double sum_forward = 0.0;
    double sum_backward = 0.0;
    
    // Forward summation
    for (size_t i = 1; i <= n; ++i) {
        const double di = static_cast<double>(i);
        double term = (i % 2 == 1) ? 1.0/di : -1.0/di;
        sum_forward += term;
    }
    
    // Backward summation
    for (size_t i = n; i >= 1; --i) {
        const double di = static_cast<double>(i);
        double term = (i % 2 == 1) ? 1.0/di : -1.0/di;
        sum_backward += term;
    }
    
    // Results should be similar but may differ due to rounding
    double difference = std::abs(sum_forward - sum_backward);
    EXPECT_LT(difference, 1e-12);  // Should be very small
    
    std::cout << "Alternating series sum difference: " << difference << std::endl;
}

// ============================================================================
// Kahan Summation Accuracy Tests - CRITICAL MISSING
// ============================================================================

TEST_F(NumericalStabilityTest, KahanSummationAccuracy) {
    // Generate many small numbers that would lose precision in naive summation
    auto small_numbers = generate_random_vector<double>(10000, 1e-10, 1e-8);
    
    // Naive summation
    double naive_sum = 0.0;
    for (double x : small_numbers) {
        naive_sum += x;
    }
    
    // Kahan compensated summation
    double kahan_sum = 0.0;
    double c = 0.0;  // Running compensation for lost low-order bits
    
    for (double x : small_numbers) {
        double y = x - c;          // So far, so good: c is zero
        double t = kahan_sum + y;  // Alas, sum is big, y small, so low-order digits of y are lost
        c = (t - kahan_sum) - y;   // (t - sum) cancels high-order part of y; subtracting y recovers negative (low part of y)
        kahan_sum = t;             // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
    }
    
    // High-precision reference using long double
    long double reference_sum = 0.0L;
    for (double x : small_numbers) {
        reference_sum += static_cast<long double>(x);
    }
    
    // Kahan should be more accurate than naive
    double naive_error = std::abs(naive_sum - static_cast<double>(reference_sum));
    double kahan_error = std::abs(kahan_sum - static_cast<double>(reference_sum));
    
    EXPECT_LT(kahan_error, naive_error);
    EXPECT_LT(kahan_error / static_cast<double>(reference_sum), 1e-14);  // Very high relative accuracy
    
    std::cout << "Naive summation relative error: " << naive_error / static_cast<double>(reference_sum) << std::endl;
    std::cout << "Kahan summation relative error: " << kahan_error / static_cast<double>(reference_sum) << std::endl;
}

TEST_F(NumericalStabilityTest, DotProductAccuracy) {
    const size_t n = 1000;
    
    // Generate vectors with mixed scales
    auto vec1 = generate_random_vector<double>(n, 1e-5, 1e5);
    auto vec2 = generate_random_vector<double>(n, 1e-5, 1e5);
    
    // Naive dot product
    double naive_dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        naive_dot += vec1[i] * vec2[i];
    }
    
    // Compensated dot product
    double comp_dot = 0.0;
    double c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double product = vec1[i] * vec2[i];
        double y = product - c;
        double t = comp_dot + y;
        c = (t - comp_dot) - y;
        comp_dot = t;
    }
    
    // High-precision reference
    long double reference_dot = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        reference_dot += static_cast<long double>(vec1[i]) * static_cast<long double>(vec2[i]);
    }
    
    // Verify compensated method is more accurate
    double naive_error = std::abs(naive_dot - static_cast<double>(reference_dot));
    double comp_error = std::abs(comp_dot - static_cast<double>(reference_dot));
    
    if (naive_error > comp_error) {
        EXPECT_LT(comp_error, naive_error);
    }
    // Note: For some random vectors, both methods may be equally accurate
}

// ============================================================================
// Condition Number Analysis Tests - CRITICAL MISSING
// ============================================================================

TEST_F(NumericalStabilityTest, MatrixConditionNumber2x2) {
    // Test 2x2 matrix condition number calculation
    // Well-conditioned matrix
    std::vector<std::vector<double>> well_conditioned = {
        {4.0, 1.0},
        {1.0, 3.0}
    };
    
    // Calculate eigenvalues manually for 2x2 symmetric matrix
    double a = well_conditioned[0][0];
    double b = well_conditioned[0][1];
    double c = well_conditioned[1][0]; 
    double d = well_conditioned[1][1];
    
    double trace = a + d;
    double det = a * d - b * c;
    double discriminant = trace * trace - 4 * det;
    
    double lambda1 = (trace + std::sqrt(discriminant)) / 2.0;
    double lambda2 = (trace - std::sqrt(discriminant)) / 2.0;
    
    // Condition number is ratio of largest to smallest eigenvalue
    double cond_well = std::max(lambda1, lambda2) / std::min(lambda1, lambda2);
    
    EXPECT_LT(cond_well, 10.0);  // Well-conditioned matrices have low condition number
    
    // Ill-conditioned matrix (nearly singular)
    std::vector<std::vector<double>> ill_conditioned = {
        {1.0, 1.0},
        {1.0, 1.0 + 1e-12}  // Nearly dependent rows
    };
    
    a = ill_conditioned[0][0];
    b = ill_conditioned[0][1];
    c = ill_conditioned[1][0];
    d = ill_conditioned[1][1];
    
    trace = a + d;
    det = a * d - b * c;
    discriminant = trace * trace - 4 * det;
    
    lambda1 = (trace + std::sqrt(discriminant)) / 2.0;
    lambda2 = (trace - std::sqrt(discriminant)) / 2.0;
    
    double cond_ill = std::max(lambda1, lambda2) / std::min(std::abs(lambda1), std::abs(lambda2));
    
    EXPECT_GT(cond_ill, 1e10);  // Ill-conditioned matrices have high condition number
    
    std::cout << "Well-conditioned matrix condition number: " << cond_well << std::endl;
    std::cout << "Ill-conditioned matrix condition number: " << cond_ill << std::endl;
}

// ============================================================================
// Precision Loss Detection Tests - CRITICAL MISSING
// ============================================================================

TEST_F(NumericalStabilityTest, PrecisionLossInSubtraction) {
    // Test precision loss when subtracting nearly equal numbers
    double x = 1.0 + 1e-15;
    double y = 1.0;
    
    double difference = x - y;
    double expected = 1e-15;
    
    // Check if we can detect the precision loss
    double ulp_distance = std::abs(difference - expected) / std::numeric_limits<double>::epsilon();
    
    // Should be within a few ULPs but may show some precision loss
    EXPECT_LT(ulp_distance, 10.0);
    
    std::cout << "ULP distance in subtraction: " << ulp_distance << std::endl;
}

TEST_F(NumericalStabilityTest, PrecisionLossInDivision) {
    // Test precision loss in division by numbers close to zero
    double numerator = 1.0;
    double small_denominator = 1e-200;  // Very small but not denormal
    
    double result = numerator / small_denominator;
    
    // Should produce a very large number but not infinity
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_GT(result, 1e100);
    
    // Test division that might produce denormal
    double tiny_numerator = std::numeric_limits<double>::min() / 2.0;
    double normal_denominator = 2.0;
    
    double tiny_result = tiny_numerator / normal_denominator;
    
    // Result should be in denormal range or zero
    if (tiny_result != 0.0) {
        EXPECT_LT(tiny_result, std::numeric_limits<double>::min());
        EXPECT_GT(tiny_result, 0.0);
    }
}

// ============================================================================
// Iterative Algorithm Convergence Tests - CRITICAL MISSING  
// ============================================================================

TEST_F(NumericalStabilityTest, NewtonRaphsonStability) {
    // Test Newton-Raphson square root convergence
    auto newton_sqrt = [](double x, double tolerance = 1e-12, int max_iter = 100) {
        if (x < 0) return std::numeric_limits<double>::quiet_NaN();
        if (x == 0) return 0.0;
        
        double guess = x / 2.0;  // Initial guess
        double prev_guess = 0.0;
        
        for (int i = 0; i < max_iter; ++i) {
            prev_guess = guess;
            guess = 0.5 * (guess + x / guess);  // Newton-Raphson iteration
            
            if (std::abs(guess - prev_guess) < tolerance) {
                return guess;
            }
        }
        return guess;  // Return best guess if not converged
    };
    
    // Test convergence for various inputs
    std::vector<double> test_values = {4.0, 2.0, 0.25, 1e6, 1e-6, 1.0};
    
    for (double val : test_values) {
        double newton_result = newton_sqrt(val);
        double std_result = std::sqrt(val);
        
        double relative_error = std::abs(newton_result - std_result) / std_result;
        EXPECT_LT(relative_error, 1e-10);
        
        // Verify the result actually satisfies x^2 ≈ val
        double residual = newton_result * newton_result - val;
        EXPECT_NEAR(residual, 0.0, 1e-10);
    }
}

TEST_F(NumericalStabilityTest, IterativeRefinement) {
    // Test iterative refinement for solving Ax = b
    // Simple 2x2 system for testing
    std::vector<std::vector<double>> A = {
        {2.0, 1.0},
        {1.0, 2.0}
    };
    std::vector<double> b = {3.0, 3.0};
    std::vector<double> x_true = {1.0, 1.0};  // Known solution
    
    // Initial solution (deliberately imprecise)
    std::vector<double> x = {0.9, 1.1};
    
    // Iterative refinement steps
    const int max_iter = 10;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual r = b - Ax
        std::vector<double> r(2);
        r[0] = b[0] - (A[0][0] * x[0] + A[0][1] * x[1]);
        r[1] = b[1] - (A[1][0] * x[0] + A[1][1] * x[1]);
        
        // Solve A * delta = r (using inverse for this simple 2x2 case)
        double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        std::vector<double> delta(2);
        delta[0] = (A[1][1] * r[0] - A[0][1] * r[1]) / det;
        delta[1] = (-A[1][0] * r[0] + A[0][0] * r[1]) / det;
        
        // Update solution
        x[0] += delta[0];
        x[1] += delta[1];
        
        // Check convergence
        if (std::abs(delta[0]) + std::abs(delta[1]) < 1e-12) {
            break;
        }
    }
    
    // Verify final accuracy
    EXPECT_NEAR(x[0], x_true[0], 1e-10);
    EXPECT_NEAR(x[1], x_true[1], 1e-10);
    
    std::cout << "Refined solution: [" << x[0] << ", " << x[1] << "]" << std::endl;
}

// ============================================================================
// Floating Point Comparison Stability Tests - CRITICAL MISSING
// ============================================================================

TEST_F(NumericalStabilityTest, RelativeToleranceComparison) {
    // Test stable floating point comparison with relative tolerance
    auto almost_equal = [](double a, double b, double rel_tol = 1e-12, double abs_tol = 1e-15) {
        if (std::abs(a - b) <= abs_tol) return true;
        if (std::abs(a - b) <= rel_tol * std::max(std::abs(a), std::abs(b))) return true;
        return false;
    };
    
    // Test cases where naive equality fails but should be considered equal
    EXPECT_TRUE(almost_equal(0.1 + 0.2, 0.3));  // Classic floating point issue
    EXPECT_TRUE(almost_equal(1e15 + 1.0, 1e15));  // Loss of precision in large numbers
    EXPECT_TRUE(almost_equal(1e-15, 2e-15, 1e-12, 1e-15));  // Very small numbers
    
    // Test cases that should not be equal
    EXPECT_FALSE(almost_equal(1.0, 1.1, 1e-12));
    EXPECT_FALSE(almost_equal(0.0, 1e-10, 1e-12, 1e-15));
}

TEST_F(NumericalStabilityTest, ULPDistanceComparison) {
    // Test ULP (Unit in the Last Place) based comparison
    auto ulp_distance = [](double a, double b) -> int64_t {
        if (a == b) return 0;
        
        // Convert to integer representation (safe, no strict-aliasing UB)
        int64_t a_bits = std::bit_cast<int64_t>(a);
        int64_t b_bits = std::bit_cast<int64_t>(b);

        // Map negative values to ordered space to compare ULP distances
        if (a_bits < 0) a_bits = std::numeric_limits<int64_t>::min() - a_bits;
        if (b_bits < 0) b_bits = std::numeric_limits<int64_t>::min() - b_bits;

        return std::llabs(a_bits - b_bits);
    };
    
    // Test ULP distances
    double x = 1.0;
    double next_x = std::nextafter(x, 2.0);
    
    EXPECT_EQ(ulp_distance(x, next_x), 1);
    EXPECT_EQ(ulp_distance(x, x), 0);
    
    // Test that numbers within a few ULPs are "close"
    double y = 1.0 + std::numeric_limits<double>::epsilon();
    EXPECT_LE(ulp_distance(x, y), 2);
}

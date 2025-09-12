#include <base/numeric_base.h>
#include <base/traits_base.h>
#include <base/storage_base.h>
#include <base/ops_base.h>
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <complex>

using namespace fem::numeric;

// ============================================================================
// Performance Regression Detection Tests
// ============================================================================

class PerformanceRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator
        gen_.seed(42);  // Fixed seed for reproducible benchmarks
        
        // Warm up CPU
        volatile double warmup = 0.0;
        for (int i = 0; i < 1000000; ++i) {
            warmup += std::sin(i * 0.001);
        }
    }
    
    std::mt19937 gen_;
    
    // Benchmark helper function
    template<typename Func>
    double benchmark_function(Func&& func, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;  // Average ns per iteration
    }
    
    // Generate test data
    template<typename T>
    std::vector<T> generate_test_data(size_t size, T min_val = T{-1000}, T max_val = T{1000}) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        std::vector<T> data(size);
        std::generate(data.begin(), data.end(), [&]() { return dist(gen_); });
        return data;
    }
};

// ============================================================================
// Arithmetic Operations Performance - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, BasicArithmeticFloat) {
    const size_t data_size = 10000;
    auto data1 = generate_test_data<float>(data_size);
    auto data2 = generate_test_data<float>(data_size);
    std::vector<float> result(data_size);
    
    // Addition benchmark
    double add_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] + data2[i];
        }
    });
    
    // Multiplication benchmark
    double mul_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] * data2[i];
        }
    });
    
    // Division benchmark
    double div_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] / (data2[i] + 1e-6f);  // Avoid division by zero
        }
    });
    
    // Performance expectations (adjust based on platform)
    EXPECT_LT(add_time, 50000.0) << "Float addition too slow: " << add_time << " ns per iteration";
    EXPECT_LT(mul_time, 50000.0) << "Float multiplication too slow: " << mul_time << " ns per iteration";
    EXPECT_LT(div_time, 200000.0) << "Float division too slow: " << div_time << " ns per iteration";
    
    std::cout << "Float arithmetic performance (ns per " << data_size << " operations):" << std::endl;
    std::cout << "  Addition: " << add_time << std::endl;
    std::cout << "  Multiplication: " << mul_time << std::endl;
    std::cout << "  Division: " << div_time << std::endl;
}

TEST_F(PerformanceRegressionTest, BasicArithmeticDouble) {
    const size_t data_size = 10000;
    auto data1 = generate_test_data<double>(data_size);
    auto data2 = generate_test_data<double>(data_size);
    std::vector<double> result(data_size);
    
    // Addition benchmark
    double add_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] + data2[i];
        }
    });
    
    // Multiplication benchmark
    double mul_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] * data2[i];
        }
    });
    
    // FMA (Fused Multiply-Add) benchmark if available
    double fma_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::fma(data1[i], data2[i], result[i]);
        }
    });
    
    // Performance expectations
    EXPECT_LT(add_time, 50000.0) << "Double addition too slow: " << add_time << " ns per iteration";
    EXPECT_LT(mul_time, 50000.0) << "Double multiplication too slow: " << mul_time << " ns per iteration";
    EXPECT_LT(fma_time, 100000.0) << "Double FMA too slow: " << fma_time << " ns per iteration";
    
    std::cout << "Double arithmetic performance (ns per " << data_size << " operations):" << std::endl;
    std::cout << "  Addition: " << add_time << std::endl;
    std::cout << "  Multiplication: " << mul_time << std::endl;
    std::cout << "  FMA: " << fma_time << std::endl;
}

// ============================================================================
// Complex Number Performance - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, ComplexArithmetic) {
    const size_t data_size = 5000;
    std::vector<std::complex<double>> data1(data_size);
    std::vector<std::complex<double>> data2(data_size);
    std::vector<std::complex<double>> result(data_size);
    
    // Generate complex test data
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (size_t i = 0; i < data_size; ++i) {
        data1[i] = std::complex<double>(dist(gen_), dist(gen_));
        data2[i] = std::complex<double>(dist(gen_), dist(gen_));
    }
    
    // Complex addition
    double add_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] + data2[i];
        }
    });
    
    // Complex multiplication
    double mul_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] * data2[i];
        }
    });
    
    // Complex division
    double div_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] / data2[i];
        }
    });
    
    // Complex absolute value
    std::vector<double> abs_result(data_size);
    double abs_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            abs_result[i] = std::abs(data1[i]);
        }
    });
    
    // Performance expectations for complex operations
    EXPECT_LT(add_time, 100000.0) << "Complex addition too slow: " << add_time << " ns per iteration";
    EXPECT_LT(mul_time, 300000.0) << "Complex multiplication too slow: " << mul_time << " ns per iteration";
    EXPECT_LT(div_time, 800000.0) << "Complex division too slow: " << div_time << " ns per iteration";
    EXPECT_LT(abs_time, 200000.0) << "Complex absolute value too slow: " << abs_time << " ns per iteration";
    
    std::cout << "Complex arithmetic performance (ns per " << data_size << " operations):" << std::endl;
    std::cout << "  Addition: " << add_time << std::endl;
    std::cout << "  Multiplication: " << mul_time << std::endl;
    std::cout << "  Division: " << div_time << std::endl;
    std::cout << "  Absolute value: " << abs_time << std::endl;
}

// ============================================================================
// Mathematical Functions Performance - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, TranscendentalFunctions) {
    const size_t data_size = 1000;
    auto input_data = generate_test_data<double>(data_size, -10.0, 10.0);
    std::vector<double> result(data_size);
    
    // Sine function
    double sin_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::sin(input_data[i]);
        }
    });
    
    // Cosine function
    double cos_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::cos(input_data[i]);
        }
    });
    
    // Exponential function
    double exp_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::exp(input_data[i]);
        }
    });
    
    // Logarithm function (ensure positive inputs)
    auto positive_data = generate_test_data<double>(data_size, 0.1, 100.0);
    double log_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::log(positive_data[i]);
        }
    });
    
    // Square root function
    double sqrt_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = std::sqrt(positive_data[i]);
        }
    });
    
    // Performance expectations for transcendental functions
    EXPECT_LT(sin_time, 1000000.0) << "Sine function too slow: " << sin_time << " ns per iteration";
    EXPECT_LT(cos_time, 1000000.0) << "Cosine function too slow: " << cos_time << " ns per iteration";
    EXPECT_LT(exp_time, 1500000.0) << "Exponential function too slow: " << exp_time << " ns per iteration";
    EXPECT_LT(log_time, 1500000.0) << "Logarithm function too slow: " << log_time << " ns per iteration";
    EXPECT_LT(sqrt_time, 300000.0) << "Square root function too slow: " << sqrt_time << " ns per iteration";
    
    std::cout << "Transcendental functions performance (ns per " << data_size << " operations):" << std::endl;
    std::cout << "  Sin: " << sin_time << std::endl;
    std::cout << "  Cos: " << cos_time << std::endl;
    std::cout << "  Exp: " << exp_time << std::endl;
    std::cout << "  Log: " << log_time << std::endl;
    std::cout << "  Sqrt: " << sqrt_time << std::endl;
}

// ============================================================================
// Memory Operations Performance - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, MemoryOperations) {
    const size_t data_size = 100000;
    std::vector<double> src(data_size);
    std::vector<double> dst(data_size);
    
    // Initialize source data
    std::iota(src.begin(), src.end(), 1.0);
    
    // Memory copy benchmark
    double copy_time = benchmark_function([&]() {
        std::copy(src.begin(), src.end(), dst.begin());
    }, 100);  // Fewer iterations due to large data
    
    // Memory fill benchmark
    double fill_time = benchmark_function([&]() {
        std::fill(dst.begin(), dst.end(), 42.0);
    }, 100);
    
    // Memory transform benchmark
    double transform_time = benchmark_function([&]() {
        std::transform(src.begin(), src.end(), dst.begin(), 
                      [](double x) { return x * 2.0 + 1.0; });
    }, 100);
    
    // Sequential access benchmark
    double sequential_time = benchmark_function([&]() {
        double sum = 0.0;
        for (size_t i = 0; i < data_size; ++i) {
            sum += src[i];
        }
        volatile double result = sum;  // Prevent optimization
        (void)result;
    }, 100);
    
    // Random access benchmark
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);
    
    double random_time = benchmark_function([&]() {
        double sum = 0.0;
        for (size_t idx : indices) {
            sum += src[idx];
        }
        volatile double result = sum;  // Prevent optimization
        (void)result;
    }, 10);  // Even fewer iterations for random access
    
    // Performance expectations (adjust based on platform and data size)
    double per_element_copy = copy_time / data_size;
    double per_element_fill = fill_time / data_size;
    double per_element_transform = transform_time / data_size;
    double per_element_seq = sequential_time / data_size;
    double per_element_rand = random_time / data_size;
    
    EXPECT_LT(per_element_copy, 10.0) << "Memory copy too slow: " << per_element_copy << " ns per element";
    EXPECT_LT(per_element_fill, 5.0) << "Memory fill too slow: " << per_element_fill << " ns per element";
    EXPECT_LT(per_element_transform, 15.0) << "Memory transform too slow: " << per_element_transform << " ns per element";
    EXPECT_LT(per_element_seq, 5.0) << "Sequential access too slow: " << per_element_seq << " ns per element";
    EXPECT_LT(per_element_rand, 50.0) << "Random access too slow: " << per_element_rand << " ns per element";
    
    std::cout << "Memory operations performance (ns per element, " << data_size << " elements):" << std::endl;
    std::cout << "  Copy: " << per_element_copy << std::endl;
    std::cout << "  Fill: " << per_element_fill << std::endl;
    std::cout << "  Transform: " << per_element_transform << std::endl;
    std::cout << "  Sequential access: " << per_element_seq << std::endl;
    std::cout << "  Random access: " << per_element_rand << std::endl;
}

// ============================================================================
// Cache Performance Tests - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, CacheFriendliness) {
    // Test cache-friendly vs cache-unfriendly access patterns
    const size_t matrix_size = 1000;
    std::vector<std::vector<double>> matrix(matrix_size, std::vector<double>(matrix_size));
    
    // Initialize matrix
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            matrix[i][j] = static_cast<double>(i * matrix_size + j);
        }
    }
    
    // Row-major access (cache-friendly)
    double row_major_time = benchmark_function([&]() {
        double sum = 0.0;
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                sum += matrix[i][j];
            }
        }
        volatile double result = sum;  // Prevent optimization
        (void)result;
    }, 10);
    
    // Column-major access (cache-unfriendly)
    double col_major_time = benchmark_function([&]() {
        double sum = 0.0;
        for (size_t j = 0; j < matrix_size; ++j) {
            for (size_t i = 0; i < matrix_size; ++i) {
                sum += matrix[i][j];
            }
        }
        volatile double result = sum;  // Prevent optimization
        (void)result;
    }, 10);
    
    // Cache-friendly should be significantly faster
    double cache_ratio = col_major_time / row_major_time;
    EXPECT_GT(cache_ratio, 2.0) << "Cache effect not observed. Ratio: " << cache_ratio;
    EXPECT_LT(cache_ratio, 20.0) << "Excessive cache penalty. Ratio: " << cache_ratio;
    
    std::cout << "Cache performance test:" << std::endl;
    std::cout << "  Row-major time: " << row_major_time << " ns" << std::endl;
    std::cout << "  Column-major time: " << col_major_time << " ns" << std::endl;
    std::cout << "  Cache penalty ratio: " << cache_ratio << std::endl;
}

// ============================================================================
// SIMD Performance Estimation - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, SIMDPotential) {
    const size_t data_size = 10000;
    auto data1 = generate_test_data<float>(data_size);
    auto data2 = generate_test_data<float>(data_size);
    std::vector<float> result(data_size);
    
    // Scalar implementation
    double scalar_time = benchmark_function([&]() {
        for (size_t i = 0; i < data_size; ++i) {
            result[i] = data1[i] * data2[i] + 1.0f;
        }
    });
    
    // Unrolled loop (potential for auto-vectorization)
    double unrolled_time = benchmark_function([&]() {
        size_t i;
        for (i = 0; i + 3 < data_size; i += 4) {
            result[i] = data1[i] * data2[i] + 1.0f;
            result[i+1] = data1[i+1] * data2[i+1] + 1.0f;
            result[i+2] = data1[i+2] * data2[i+2] + 1.0f;
            result[i+3] = data1[i+3] * data2[i+3] + 1.0f;
        }
        // Handle remaining elements
        for (; i < data_size; ++i) {
            result[i] = data1[i] * data2[i] + 1.0f;
        }
    });
    
    // The unrolled version might be faster due to auto-vectorization
    double speedup_ratio = scalar_time / unrolled_time;
    
    // We expect at least some improvement (but may not always occur)
    EXPECT_GT(speedup_ratio, 0.8) << "Significant performance regression in unrolled code";
    EXPECT_LT(speedup_ratio, 8.0) << "Unrealistic speedup detected";
    
    std::cout << "SIMD potential test:" << std::endl;
    std::cout << "  Scalar time: " << scalar_time << " ns" << std::endl;
    std::cout << "  Unrolled time: " << unrolled_time << " ns" << std::endl;
    std::cout << "  Speedup ratio: " << speedup_ratio << std::endl;
}

// ============================================================================
// Compiler Optimization Detection - CRITICAL BASELINE
// ============================================================================

TEST_F(PerformanceRegressionTest, OptimizationEffectiveness) {
    const size_t iterations = 1000000;
    
    // Test that should be optimized away
    volatile double dummy_result = 0.0;
    
    double optimizable_time = benchmark_function([&]() {
        double x = 1.0;
        for (size_t i = 0; i < 1000; ++i) {
            x = x * 1.0;  // Should be optimized away
        }
        dummy_result = x;  // Prevent complete optimization
    });
    
    // Test that cannot be optimized away
    double non_optimizable_time = benchmark_function([&]() {
        double x = 1.0;
        for (size_t i = 0; i < 1000; ++i) {
            x = x * (1.0 + 1e-15);  // Cannot be optimized away easily
        }
        dummy_result = x;
    });
    
    // The optimizable version should be much faster if optimizations work
    double optimization_ratio = non_optimizable_time / optimizable_time;
    
    // We expect significant optimization (but exact ratio depends on compiler)
    EXPECT_GT(optimization_ratio, 2.0) << "Compiler optimizations seem ineffective";
    
    std::cout << "Compiler optimization test:" << std::endl;
    std::cout << "  Optimizable time: " << optimizable_time << " ns" << std::endl;
    std::cout << "  Non-optimizable time: " << non_optimizable_time << " ns" << std::endl;
    std::cout << "  Optimization ratio: " << optimization_ratio << std::endl;
}
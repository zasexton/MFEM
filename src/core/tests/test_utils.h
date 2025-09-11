#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <chrono>
#include <thread>
#include <random>
#include <filesystem>
#include <fstream>
#include <functional>

namespace fem::core::test {

// ==============================================================================
// Timing Utilities
// ==============================================================================

/**
 * @brief Simple timer for performance measurements
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    double elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::nano>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ==============================================================================
// File System Utilities
// ==============================================================================

/**
 * @brief RAII temporary directory for tests
 */
class TempDirectory {
public:
    explicit TempDirectory(const std::string& prefix = "test_") {
        namespace fs = std::filesystem;

        // Create unique temp directory
        auto temp_path = fs::temp_directory_path();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(10000, 99999);

        path_ = temp_path / (prefix + std::to_string(dis(gen)));
        fs::create_directories(path_);
    }

    ~TempDirectory() {
        // Clean up temp directory
        std::error_code ec;  // Ignore errors during cleanup
        std::filesystem::remove_all(path_, ec);
    }

    std::filesystem::path path() const { return path_; }

    std::filesystem::path create_file(const std::string& name,
                                      const std::string& content = "") {
        auto file_path = path_ / name;
        std::ofstream file(file_path);
        file << content;
        return file_path;
    }

    std::filesystem::path create_directory(const std::string& name) {
        auto dir_path = path_ / name;
        std::filesystem::create_directories(dir_path);
        return dir_path;
    }

private:
    std::filesystem::path path_;
};

// ==============================================================================
// Thread Utilities
// ==============================================================================

/**
 * @brief Helper to run function on multiple threads
 */
template<typename Func>
void run_parallel(int num_threads, Func&& func) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(func, i);
    }

    for (auto& t : threads) {
        t.join();
    }
}

/**
 * @brief Barrier for synchronizing threads in tests
 */
class TestBarrier {
public:
    explicit TestBarrier(std::size_t count)
        : threshold_(count), count_(count), generation_(0) {}

    void wait() {
        auto gen = generation_.load();
        if (--count_ == 0) {
            generation_.fetch_add(1);
            count_.store(threshold_);
            cv_.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this, gen] { return gen != generation_.load(); });
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_.store(threshold_);
        generation_.store(0);
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t threshold_;
    std::atomic<std::size_t> count_;
    std::atomic<std::size_t> generation_;
};

// ==============================================================================
// Memory Utilities
// ==============================================================================

/**
 * @brief Track memory allocations for leak detection
 * Note: This is a simplified version. Real implementation would need
 * to hook into new/delete operators
 */
class MemoryTracker {
public:
    static void start_tracking() {
        tracking_ = true;
        allocations_ = 0;
        deallocations_ = 0;
        bytes_allocated_ = 0;
        bytes_deallocated_ = 0;
    }

    static void stop_tracking() {
        tracking_ = false;
    }

    static void record_allocation(std::size_t bytes) {
        if (tracking_) {
            allocations_.fetch_add(1);
            bytes_allocated_.fetch_add(bytes);
        }
    }

    static void record_deallocation(std::size_t bytes) {
        if (tracking_) {
            deallocations_.fetch_add(1);
            bytes_deallocated_.fetch_add(bytes);
        }
    }

    static bool has_leaks() {
        return allocations_ != deallocations_ ||
               bytes_allocated_ != bytes_deallocated_;
    }

    static std::size_t get_allocations() { return allocations_; }
    static std::size_t get_deallocations() { return deallocations_; }
    static std::size_t get_bytes_allocated() { return bytes_allocated_; }
    static std::size_t get_bytes_deallocated() { return bytes_deallocated_; }

    static void reset() {
        allocations_ = 0;
        deallocations_ = 0;
        bytes_allocated_ = 0;
        bytes_deallocated_ = 0;
    }

private:
    static inline bool tracking_ = false;
    static inline std::atomic<std::size_t> allocations_{0};
    static inline std::atomic<std::size_t> deallocations_{0};
    static inline std::atomic<std::size_t> bytes_allocated_{0};
    static inline std::atomic<std::size_t> bytes_deallocated_{0};
};

// ==============================================================================
// Custom Matchers for Google Test
// ==============================================================================

/**
 * @brief Matcher to check if a value is within tolerance
 */
MATCHER_P2(IsNear, expected, tolerance,
           "is within " + testing::PrintToString(tolerance) +
           " of " + testing::PrintToString(expected)) {
    return std::abs(arg - expected) <= tolerance;
}

/**
 * @brief Matcher to check if a container is sorted
 */
MATCHER(IsSorted, "is sorted") {
    return std::is_sorted(arg.begin(), arg.end());
}

/**
 * @brief Matcher to check if a container is reverse sorted
 */
MATCHER(IsReverseSorted, "is reverse sorted") {
    return std::is_sorted(arg.rbegin(), arg.rend());
}

/**
 * @brief Matcher to check if all elements satisfy a predicate
 */
MATCHER_P(AllSatisfy, predicate, "all elements satisfy predicate") {
    return std::all_of(arg.begin(), arg.end(), predicate);
}

/**
 * @brief Matcher to check if any element satisfies a predicate
 */
MATCHER_P(AnySatisfy, predicate, "any element satisfies predicate") {
    return std::any_of(arg.begin(), arg.end(), predicate);
}

// ==============================================================================
// Test Data Generators
// ==============================================================================

/**
 * @brief Generate random test data
 */
template<typename T>
class RandomGenerator {
public:
    RandomGenerator(T min = std::numeric_limits<T>::min(),
                   T max = std::numeric_limits<T>::max())
        : gen_(std::random_device{}()), dist_(min, max) {}

    T next() { return dist_(gen_); }

    std::vector<T> generate(std::size_t count) {
        std::vector<T> result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(next());
        }
        return result;
    }

    void seed(unsigned int seed) {
        gen_.seed(seed);
    }

private:
    std::mt19937 gen_;
    std::conditional_t<std::is_integral_v<T>,
                      std::uniform_int_distribution<T>,
                      std::uniform_real_distribution<T>> dist_;
};

// Specialization for bool
template<>
class RandomGenerator<bool> {
public:
    RandomGenerator(double true_probability = 0.5)
        : gen_(std::random_device{}()), dist_(true_probability) {}

    bool next() { return dist_(gen_); }

    std::vector<bool> generate(std::size_t count) {
        std::vector<bool> result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(next());
        }
        return result;
    }

private:
    std::mt19937 gen_;
    std::bernoulli_distribution dist_;
};

// ==============================================================================
// Test Fixtures
// ==============================================================================

/**
 * @brief Base fixture with common setup/teardown
 */
class CoreTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for all tests
        test_start_time_ = std::chrono::steady_clock::now();
    }

    void TearDown() override {
        // Common teardown
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - test_start_time_).count();

        if (::testing::Test::HasFailure()) {
            std::cerr << "Test failed after " << duration << " ms\n";
        }
    }

    // Helper methods available to all test fixtures
    template<typename Func>
    void assert_no_throw(Func&& func) {
        ASSERT_NO_THROW(func());
    }

    template<typename Exception, typename Func>
    void assert_throws_with_message(Func&& func, const std::string& expected_msg) {
        ASSERT_THROW({
            try {
                func();
            } catch (const Exception& e) {
                EXPECT_THAT(e.what(), testing::HasSubstr(expected_msg));
                throw;
            }
        }, Exception);
    }

    // Get elapsed time since test started
    double elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(
            now - test_start_time_).count();
    }

private:
    std::chrono::steady_clock::time_point test_start_time_;
};

/**
 * @brief Parameterized test fixture
 */
template<typename T>
class CoreParameterizedFixture : public CoreTestFixture,
                                 public ::testing::WithParamInterface<T> {
protected:
    T GetParam() const { return ::testing::WithParamInterface<T>::GetParam(); }
};

// ==============================================================================
// Performance Test Helpers
// ==============================================================================

/**
 * @brief Measure average execution time
 */
template<typename Func>
double measure_average_time_ms(Func&& func, int iterations = 1000) {
    // Warm up
    for (int i = 0; i < std::min(10, iterations/10); ++i) {
        func();
    }

    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    return timer.elapsed_ms() / iterations;
}

/**
 * @brief Measure operations per second
 */
template<typename Func>
double measure_ops_per_second(Func&& func, double duration_seconds = 1.0) {
    int operations = 0;
    Timer timer;

    while (timer.elapsed_ms() < duration_seconds * 1000) {
        func();
        ++operations;
    }

    return operations / (timer.elapsed_ms() / 1000.0);
}

/**
 * @brief Performance assertion macro
 */
#define ASSERT_PERFORMANCE(func, max_ms, iterations) \
    do { \
        auto avg_time = fem::core::test::measure_average_time_ms(func, iterations); \
        ASSERT_LT(avg_time, max_ms) \
            << "Performance requirement not met: " << avg_time << " ms > " << max_ms << " ms"; \
    } while(0)

#define EXPECT_PERFORMANCE(func, max_ms, iterations) \
    do { \
        auto avg_time = fem::core::test::measure_average_time_ms(func, iterations); \
        EXPECT_LT(avg_time, max_ms) \
            << "Performance requirement not met: " << avg_time << " ms > " << max_ms << " ms"; \
    } while(0)

// ==============================================================================
// Test Macros
// ==============================================================================

// Macro to test multiple values against same condition
#define TEST_ALL_EQ(container, expected) \
    do { \
        for (const auto& elem : container) { \
            EXPECT_EQ(elem, expected); \
        } \
    } while(0)

// Macro to test that code compiles (SFINAE tests)
#define ASSERT_COMPILES(...) \
    do { \
        auto lambda = []() { __VA_ARGS__; }; \
        (void)lambda; \
    } while(0)

// ==============================================================================
// Output Helpers
// ==============================================================================

/**
 * @brief Print container contents for debugging
 */
template<typename Container>
void print_container(const Container& c, const std::string& name = "Container") {
    std::cout << name << ": [";
    bool first = true;
    for (const auto& elem : c) {
        if (!first) std::cout << ", ";
        std::cout << elem;
        first = false;
    }
    std::cout << "]\n";
}

} // namespace fem::core::test

// ==============================================================================
// Google Test Printers for Custom Types
// ==============================================================================

// Example: Add printers for your custom types here
// namespace testing {
//     template<>
//     void PrintTo(const YourType& value, std::ostream* os) {
//         *os << value.to_string();
//     }
// }
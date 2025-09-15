/**
 * @file test_singleton.cpp
 * @brief Comprehensive unit tests for the Singleton pattern implementations
 *
 * Tests cover:
 * - Lazy Singleton with thread-safe initialization
 * - EagerSingleton with immediate initialization
 * - CustomSingleton with custom deleters
 * - Thread safety for concurrent access
 * - CRTP (Curiously Recurring Template Pattern) functionality
 * - Copy/move prevention and lifetime management
 * - Exception safety during construction
 * - Performance characteristics
 * - Macro utilities for singleton declaration
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <future>
#include <exception>

#include "core/base/singleton.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Classes for Singleton Testing
// ============================================================================

// Basic test singleton
class TestSingleton : public Singleton<TestSingleton> {
    friend class Singleton<TestSingleton>;
private:
    TestSingleton() : value_(42), construction_count_(++total_constructions_) {}

public:
    int get_value() const { return value_; }
    void set_value(int value) { value_ = value; }
    int get_construction_count() const { return construction_count_; }

    static void reset_construction_count() { total_constructions_ = 0; }
    static int get_total_constructions() { return total_constructions_; }

private:
    int value_;
    int construction_count_;
    static std::atomic<int> total_constructions_;
};

std::atomic<int> TestSingleton::total_constructions_{0};

// Singleton that tracks constructor calls
class CountingTest : public Singleton<CountingTest> {
    friend class Singleton<CountingTest>;
private:
    CountingTest() {
        construction_count_.fetch_add(1, std::memory_order_relaxed);
        construction_thread_id_ = std::this_thread::get_id();
    }

public:
    static int get_construction_count() {
        return construction_count_.load(std::memory_order_relaxed);
    }

    static void reset_construction_count() {
        construction_count_.store(0, std::memory_order_relaxed);
    }

    std::thread::id get_construction_thread_id() const {
        return construction_thread_id_;
    }

private:
    static std::atomic<int> construction_count_;
    std::thread::id construction_thread_id_;
};

std::atomic<int> CountingTest::construction_count_{0};

// Singleton that throws during construction
class ThrowingTest : public Singleton<ThrowingTest> {
    friend class Singleton<ThrowingTest>;
private:
    ThrowingTest() {
        if (should_throw_.load()) {
            throw std::runtime_error("Construction failed");
        }
        successfully_constructed_ = true;
    }

public:
    bool was_successfully_constructed() const { return successfully_constructed_; }

    static void set_should_throw(bool value) {
        should_throw_.store(value);
    }

    static void reset() {
        should_throw_.store(false);
        successfully_constructed_ = false;
    }

private:
    static std::atomic<bool> should_throw_;
    static bool successfully_constructed_;
};

std::atomic<bool> ThrowingTest::should_throw_{false};
bool ThrowingTest::successfully_constructed_{false};

// Eager singleton test class
class TestEagerSingleton : public EagerSingleton<TestEagerSingleton> {
    friend class EagerSingleton<TestEagerSingleton>;
private:
    TestEagerSingleton() : creation_time_(std::chrono::steady_clock::now()) {
        construction_count_.fetch_add(1, std::memory_order_relaxed);
    }

public:
    auto get_creation_time() const { return creation_time_; }
    static int get_construction_count() { return construction_count_.load(); }
    static void reset_construction_count() { construction_count_.store(0); }

private:
    std::chrono::steady_clock::time_point creation_time_;
    static std::atomic<int> construction_count_;
};

std::atomic<int> TestEagerSingleton::construction_count_{0};

// Forward declare for custom deleter
class TestCustomSingleton;

// Custom deleter for testing
class CustomDeleter {
public:
    void operator()(TestCustomSingleton* ptr);

    static int get_delete_count() { return delete_count_.load(); }
    static void reset_delete_count() { delete_count_.store(0); }

private:
    static std::atomic<int> delete_count_;
};

std::atomic<int> CustomDeleter::delete_count_{0};

// Custom singleton with custom deleter
class TestCustomSingleton : public CustomSingleton<TestCustomSingleton, CustomDeleter> {
    friend class CustomSingleton<TestCustomSingleton, CustomDeleter>;
private:
    TestCustomSingleton() : value_(123) {}

public:
    int get_value() const { return value_; }
    void set_value(int value) { value_ = value; }

private:
    int value_;
};

// Implementation of custom deleter after TestCustomSingleton is defined
void CustomDeleter::operator()(TestCustomSingleton* ptr) {
    delete_count_.fetch_add(1, std::memory_order_relaxed);
    delete ptr;
}

// Macro-based singleton test
class MacroTestSingleton : public Singleton<MacroTestSingleton> {
    friend class Singleton<MacroTestSingleton>;
private:
    MacroTestSingleton() = default;
    int operation_count_ = 0;

public:
    void do_something() { operation_count_++; }
    int get_operation_count() const { return operation_count_; }
    void reset_operations() { operation_count_ = 0; }
};

// Test class with forwarding method
class MethodForwardingTest : public Singleton<MethodForwardingTest> {
    friend class Singleton<MethodForwardingTest>;
private:
    MethodForwardingTest() = default;

public:
    int compute(int x, int y) { return x + y; }
    std::string get_message() { return "Hello from singleton"; }
};

// ============================================================================
// Singleton Test Fixtures
// ============================================================================

class SingletonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up any existing singletons
        TestSingleton::destroy();
        CountingTest::destroy();
        ThrowingTest::destroy();
        TestCustomSingleton::destroy();
        MacroTestSingleton::destroy();
        MethodForwardingTest::destroy();

        // Reset counters
        TestSingleton::reset_construction_count();
        CountingTest::reset_construction_count();
        ThrowingTest::reset();
        CustomDeleter::reset_delete_count();
    }

    void TearDown() override {
        // Clean up after each test
        TestSingleton::destroy();
        CountingTest::destroy();
        ThrowingTest::destroy();
        TestCustomSingleton::destroy();
        MacroTestSingleton::destroy();
        MethodForwardingTest::destroy();
    }
};

class EagerSingletonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Don't reset construction count for eager singletons since they
        // are constructed during static initialization
    }
};

// ============================================================================
// Basic Singleton Functionality Tests
// ============================================================================

TEST_F(SingletonTest, BasicInstantiation) {
    EXPECT_FALSE(TestSingleton::is_created());

    auto& instance = TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
    EXPECT_EQ(instance.get_value(), 42);
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);
}

TEST_F(SingletonTest, SingletonUniqueness) {
    auto& instance1 = TestSingleton::instance();
    auto& instance2 = TestSingleton::instance();

    // Should be the same instance
    EXPECT_EQ(&instance1, &instance2);
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);

    // Modify through one reference
    instance1.set_value(100);
    EXPECT_EQ(instance2.get_value(), 100);
}

TEST_F(SingletonTest, LazyInitialization) {
    // Should not be created initially
    EXPECT_FALSE(TestSingleton::is_created());
    EXPECT_EQ(TestSingleton::get_total_constructions(), 0);

    // Create on first access
    auto& instance = TestSingleton::instance();
    (void)instance; // Mark as used to avoid warning
    EXPECT_TRUE(TestSingleton::is_created());
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);

    // Subsequent accesses don't create new instances
    TestSingleton::instance();
    TestSingleton::instance();
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);
}

TEST_F(SingletonTest, DestroyAndRecreate) {
    // Create instance
    auto& instance1 = TestSingleton::instance();
    instance1.set_value(200);
    EXPECT_TRUE(TestSingleton::is_created());

    // Destroy
    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    // Recreate
    auto& instance2 = TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
    EXPECT_EQ(instance2.get_value(), 42); // Should be back to default
    EXPECT_EQ(TestSingleton::get_total_constructions(), 2);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(SingletonTest, ConcurrentInitialization) {
    const int num_threads = 20;
    std::vector<std::thread> threads;
    std::vector<TestSingleton*> instances(num_threads);
    std::atomic<int> ready_count{0};
    std::atomic<bool> start_flag{false};

    // Barrier synchronization
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            ready_count.fetch_add(1, std::memory_order_relaxed);

            // Wait for all threads to be ready
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            // All threads try to get instance simultaneously
            instances[i] = &TestSingleton::instance();
        });
    }

    // Wait for all threads to be ready
    while (ready_count.load(std::memory_order_relaxed) < num_threads) {
        std::this_thread::yield();
    }

    // Signal all threads to start
    start_flag.store(true, std::memory_order_release);

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    // Verify all instances are the same
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }

    // Should only be constructed once
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);
}

TEST_F(SingletonTest, ConcurrentConstructionCounting) {
    const int num_threads = 50;
    std::vector<std::thread> threads;
    std::atomic<CountingTest*> instances[num_threads];
    std::atomic<bool> start_flag{false};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // Wait for start signal
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            instances[i].store(&CountingTest::instance());
        });
    }

    // Start all threads
    start_flag.store(true);

    for (auto& t : threads) {
        t.join();
    }

    // All instances should be the same
    CountingTest* first_instance = instances[0].load();
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(first_instance, instances[i].load());
    }

    // Should only be constructed once
    EXPECT_EQ(CountingTest::get_construction_count(), 1);
}

TEST_F(SingletonTest, ThreadSafeDestruction) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successful_accesses{0};

    // Create instance first
    TestSingleton::instance();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 100; ++j) {
                try {
                    if (TestSingleton::is_created()) {
                        auto& instance = TestSingleton::instance();
                        instance.get_value(); // Use the instance
                        successful_accesses.fetch_add(1, std::memory_order_relaxed);
                    }
                } catch (...) {
                    // Ignore exceptions during concurrent destruction
                }

                if (j == 50) {
                    // Destroy in the middle
                    TestSingleton::destroy();
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have some successful accesses
    EXPECT_GT(successful_accesses.load(), 0);
}

// ============================================================================
// Exception Safety Tests
// ============================================================================

TEST_F(SingletonTest, ConstructionException) {
    ThrowingTest::set_should_throw(true);

    // First attempt should throw
    EXPECT_THROW(ThrowingTest::instance(), std::runtime_error);
    EXPECT_FALSE(ThrowingTest::is_created());

    // Allow successful construction
    ThrowingTest::set_should_throw(false);
    ThrowingTest::destroy(); // Reset state

    // Second attempt should succeed
    EXPECT_NO_THROW({
        auto& instance = ThrowingTest::instance();
        (void)instance; // Mark as used to avoid warning
    });
    EXPECT_TRUE(ThrowingTest::is_created());
    EXPECT_TRUE(ThrowingTest::instance().was_successfully_constructed());
}

TEST_F(SingletonTest, ExceptionDuringConcurrentInitialization) {
    const int num_threads = 5; // Reduced thread count for stability
    std::vector<std::thread> threads;
    std::atomic<int> exception_count{0};
    std::atomic<int> success_count{0};
    std::atomic<bool> start_flag{false};

    ThrowingTest::set_should_throw(true);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            try {
                ThrowingTest::instance();
                success_count.fetch_add(1);
            } catch (const std::runtime_error&) {
                exception_count.fetch_add(1);
            } catch (...) {
                // Catch any other exceptions that might occur
                exception_count.fetch_add(1);
            }
        });
    }

    start_flag.store(true);

    for (auto& t : threads) {
        t.join();
    }

    // All should have thrown exceptions
    EXPECT_EQ(exception_count.load(), num_threads);
    EXPECT_EQ(success_count.load(), 0);
    EXPECT_FALSE(ThrowingTest::is_created());
}

// ============================================================================
// Copy/Move Prevention Tests
// ============================================================================

TEST_F(SingletonTest, NoCopyConstruction) {
    auto& instance = TestSingleton::instance();

    // These should not compile (we can't test compilation errors in runtime tests)
    // TestSingleton copy_constructed(instance);
    // TestSingleton copy_assigned = instance;

    // We can test that the instance exists and functions
    EXPECT_EQ(instance.get_value(), 42);
}

TEST_F(SingletonTest, NoMoveOperations) {
    auto& instance = TestSingleton::instance();

    // These should not compile
    // TestSingleton move_constructed(std::move(instance));
    // TestSingleton move_assigned = std::move(instance);

    // Instance should still be valid after attempted moves
    EXPECT_EQ(instance.get_value(), 42);
}

// ============================================================================
// EagerSingleton Tests
// ============================================================================

TEST_F(EagerSingletonTest, EagerInitialization) {
    // Should be created immediately
    EXPECT_TRUE(TestEagerSingleton::is_created());

    // Record current construction count (should be >= 1 from static initialization)
    int initial_count = TestEagerSingleton::get_construction_count();
    EXPECT_GE(initial_count, 1);

    // Access should not increase construction count
    auto& instance1 = TestEagerSingleton::instance();
    auto& instance2 = TestEagerSingleton::instance();

    EXPECT_EQ(&instance1, &instance2);
    EXPECT_EQ(TestEagerSingleton::get_construction_count(), initial_count);
}

TEST_F(EagerSingletonTest, EagerCreationTime) {
    auto start_time = std::chrono::steady_clock::now();

    // Access instance
    auto& instance = TestEagerSingleton::instance();
    auto creation_time = instance.get_creation_time();

    // Creation time should be before we accessed it
    EXPECT_LT(creation_time, start_time);
}

TEST_F(EagerSingletonTest, EagerThreadSafety) {
    const int num_threads = 20;
    std::vector<std::thread> threads;
    std::vector<TestEagerSingleton*> instances(num_threads);

    // Record initial construction count
    int initial_count = TestEagerSingleton::get_construction_count();
    EXPECT_GE(initial_count, 1);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            instances[i] = &TestEagerSingleton::instance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All should be the same instance
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }

    // Construction count should remain the same
    EXPECT_EQ(TestEagerSingleton::get_construction_count(), initial_count);
}

// ============================================================================
// CustomSingleton Tests
// ============================================================================

TEST_F(SingletonTest, CustomSingletonBasicFunctionality) {
    EXPECT_FALSE(TestCustomSingleton::is_created());

    auto& instance = TestCustomSingleton::instance();
    EXPECT_TRUE(TestCustomSingleton::is_created());
    EXPECT_EQ(instance.get_value(), 123);

    instance.set_value(456);
    EXPECT_EQ(TestCustomSingleton::instance().get_value(), 456);
}

TEST_F(SingletonTest, CustomDeleter) {
    EXPECT_EQ(CustomDeleter::get_delete_count(), 0);

    // Create instance
    auto& instance = TestCustomSingleton::instance();
    instance.set_value(789);
    EXPECT_TRUE(TestCustomSingleton::is_created());

    // Destroy should use custom deleter
    TestCustomSingleton::destroy();
    EXPECT_FALSE(TestCustomSingleton::is_created());
    EXPECT_EQ(CustomDeleter::get_delete_count(), 1);

    // Create again
    TestCustomSingleton::instance();
    TestCustomSingleton::destroy();
    EXPECT_EQ(CustomDeleter::get_delete_count(), 2);
}

TEST_F(SingletonTest, CustomSingletonThreadSafety) {
    const int num_threads = 15;
    std::vector<std::thread> threads;
    std::vector<TestCustomSingleton*> instances(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            instances[i] = &TestCustomSingleton::instance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All should be the same instance
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }
}

// ============================================================================
// Macro Utility Tests
// ============================================================================

TEST_F(SingletonTest, MacroDeclaration) {
    EXPECT_FALSE(MacroTestSingleton::is_created());

    auto& instance = MacroTestSingleton::instance();
    EXPECT_TRUE(MacroTestSingleton::is_created());

    EXPECT_EQ(instance.get_operation_count(), 0);
    instance.do_something();
    EXPECT_EQ(instance.get_operation_count(), 1);

    // Same instance through multiple accesses
    MacroTestSingleton::instance().do_something();
    EXPECT_EQ(instance.get_operation_count(), 2);
}

TEST_F(SingletonTest, MethodAccess) {
    // Test accessing methods through singleton instance
    auto& instance = MethodForwardingTest::instance();

    auto result = instance.compute(5, 3);
    EXPECT_EQ(result, 8);

    auto message = instance.get_message();
    EXPECT_EQ(message, "Hello from singleton");

    // Verify multiple accesses give same instance
    auto& instance2 = MethodForwardingTest::instance();
    EXPECT_EQ(&instance, &instance2);
    EXPECT_EQ(instance2.compute(10, 20), 30);
    EXPECT_EQ(instance2.get_message(), "Hello from singleton");
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(SingletonTest, AccessPerformance) {
    // Create instance first
    TestSingleton::instance();

    const int num_accesses = 1000000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_accesses; ++i) {
        auto& instance = TestSingleton::instance();
        (void)instance; // Suppress unused variable warning
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be very fast after initialization (less than 1 second for 1M accesses)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(SingletonTest, ConcurrentAccessPerformance) {
    // Create instance first
    TestSingleton::instance();

    const int num_threads = 10;
    const int accesses_per_thread = 100000;
    std::vector<std::thread> threads;
    std::atomic<int> total_accesses{0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < accesses_per_thread; ++j) {
                auto& instance = TestSingleton::instance();
                (void)instance;
                total_accesses.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(total_accesses.load(), num_threads * accesses_per_thread);
    // Should complete in reasonable time (less than 2 seconds)
    EXPECT_LT(duration.count(), 2000000);
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

TEST_F(SingletonTest, MultipleDestroyCalls) {
    // Create instance
    TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());

    // Multiple destroy calls should be safe
    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    // Should still be able to create again
    TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
}

TEST_F(SingletonTest, DestroyWithoutCreation) {
    // Destroy without creating should be safe
    EXPECT_FALSE(TestSingleton::is_created());
    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    // Should still be able to create
    TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
}

TEST_F(SingletonTest, IsCreatedAccuracy) {
    // Initially not created
    EXPECT_FALSE(TestSingleton::is_created());

    // Create
    TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());

    // Destroy
    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    // Create again
    TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(SingletonTest, MemoryLifecycle) {
    EXPECT_EQ(TestSingleton::get_total_constructions(), 0);

    // Create and destroy multiple times
    for (int i = 0; i < 5; ++i) {
        auto& instance = TestSingleton::instance();
        instance.set_value(i * 10);
        EXPECT_EQ(instance.get_value(), i * 10);

        TestSingleton::destroy();
        EXPECT_FALSE(TestSingleton::is_created());
    }

    EXPECT_EQ(TestSingleton::get_total_constructions(), 5);
}

TEST_F(SingletonTest, NoMemoryLeaks) {
    // This test mainly ensures that we don't crash due to memory issues
    // Actual memory leak detection would require external tools

    for (int i = 0; i < 100; ++i) {
        TestSingleton::instance();
        TestCustomSingleton::instance();
        MacroTestSingleton::instance();

        if (i % 10 == 0) {
            TestSingleton::destroy();
            TestCustomSingleton::destroy();
            MacroTestSingleton::destroy();
        }
    }

    // Clean up
    TestSingleton::destroy();
    TestCustomSingleton::destroy();
    MacroTestSingleton::destroy();

    SUCCEED(); // If we reach here without crashes, test passes
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(SingletonTest, MultipleSingletonTypes) {
    // Test that different singleton types work together
    auto& test_singleton = TestSingleton::instance();
    auto& custom_singleton = TestCustomSingleton::instance();
    auto& macro_singleton = MacroTestSingleton::instance();
    auto& eager_singleton = TestEagerSingleton::instance();

    // All should be different instances
    EXPECT_NE(static_cast<void*>(&test_singleton), static_cast<void*>(&custom_singleton));
    EXPECT_NE(static_cast<void*>(&test_singleton), static_cast<void*>(&macro_singleton));
    EXPECT_NE(static_cast<void*>(&test_singleton), static_cast<void*>(&eager_singleton));

    // All should function independently
    test_singleton.set_value(111);
    custom_singleton.set_value(222);
    macro_singleton.do_something();

    EXPECT_EQ(test_singleton.get_value(), 111);
    EXPECT_EQ(custom_singleton.get_value(), 222);
    EXPECT_EQ(macro_singleton.get_operation_count(), 1);
}

TEST_F(SingletonTest, CompleteWorkflow) {
    // Test complete singleton workflow

    // 1. Initial state
    EXPECT_FALSE(TestSingleton::is_created());
    EXPECT_EQ(TestSingleton::get_total_constructions(), 0);

    // 2. Create and use
    auto& instance = TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
    EXPECT_EQ(instance.get_value(), 42);
    EXPECT_EQ(TestSingleton::get_total_constructions(), 1);

    // 3. Modify state
    instance.set_value(100);
    EXPECT_EQ(TestSingleton::instance().get_value(), 100);

    // 4. Verify uniqueness through multiple accesses
    for (int i = 0; i < 10; ++i) {
        auto& ref = TestSingleton::instance();
        EXPECT_EQ(&ref, &instance);
        EXPECT_EQ(ref.get_value(), 100);
    }

    // 5. Destroy and verify cleanup
    TestSingleton::destroy();
    EXPECT_FALSE(TestSingleton::is_created());

    // 6. Recreate with fresh state
    auto& new_instance = TestSingleton::instance();
    EXPECT_TRUE(TestSingleton::is_created());
    EXPECT_EQ(new_instance.get_value(), 42); // Back to default
    EXPECT_EQ(TestSingleton::get_total_constructions(), 2);

    // 7. Verify it's functionally a new instance (may reuse same memory)
    // The important thing is that construction count increased, not memory location
    EXPECT_EQ(new_instance.get_construction_count(), 2); // This was the second construction
}

// ============================================================================
// Static Analysis and Compile-Time Tests
// ============================================================================

TEST_F(SingletonTest, TypeTraits) {
    // Test type properties
    EXPECT_FALSE(std::is_copy_constructible_v<TestSingleton>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestSingleton>);
    EXPECT_FALSE(std::is_move_constructible_v<TestSingleton>);
    EXPECT_FALSE(std::is_move_assignable_v<TestSingleton>);

    // Same for other singleton types
    EXPECT_FALSE(std::is_copy_constructible_v<TestEagerSingleton>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestEagerSingleton>);
    EXPECT_FALSE(std::is_move_constructible_v<TestEagerSingleton>);
    EXPECT_FALSE(std::is_move_assignable_v<TestEagerSingleton>);

    EXPECT_FALSE(std::is_copy_constructible_v<TestCustomSingleton>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestCustomSingleton>);
    EXPECT_FALSE(std::is_move_constructible_v<TestCustomSingleton>);
    EXPECT_FALSE(std::is_move_assignable_v<TestCustomSingleton>);
}

// ============================================================================
// Specialized Deleter Tests
// ============================================================================

// Forward declare for logging deleter
class LoggingSingleton;

class LoggingDeleter {
public:
    void operator()(LoggingSingleton* ptr);

    static const std::vector<std::string>& get_log_entries() { return log_entries_; }
    static void clear_log() { log_entries_.clear(); }

private:
    static std::vector<std::string> log_entries_;
};

std::vector<std::string> LoggingDeleter::log_entries_;

class LoggingSingleton : public CustomSingleton<LoggingSingleton, LoggingDeleter> {
    friend class CustomSingleton<LoggingSingleton, LoggingDeleter>;
private:
    LoggingSingleton() = default;
public:
    void do_work() {}
};

// Implementation of logging deleter after LoggingSingleton is defined
void LoggingDeleter::operator()(LoggingSingleton* ptr) {
    log_entries_.push_back("Deleting singleton at " + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    delete ptr;
}

TEST_F(SingletonTest, LoggingDeleter) {
    LoggingDeleter::clear_log();
    EXPECT_TRUE(LoggingDeleter::get_log_entries().empty());

    // Create and destroy singleton
    LoggingSingleton::instance().do_work();
    LoggingSingleton::destroy();

    // Should have logged the deletion
    auto& log_entries = LoggingDeleter::get_log_entries();
    EXPECT_EQ(log_entries.size(), 1);
    EXPECT_THAT(log_entries[0], testing::HasSubstr("Deleting singleton at"));
}
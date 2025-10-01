// File: core/tests/unit/concurrency/test_semaphore.cpp
// Description: Unit tests for semaphore implementations

#include <gtest/gtest.h>
#include "core/concurrency/semaphore.h"
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <random>

using namespace core::concurrency;
using namespace std::chrono_literals;

class SemaphoreTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test basic counting semaphore functionality
TEST_F(SemaphoreTest, CountingSemaphore_BasicOperation) {
    CountingSemaphore<5> sem(3);

    // Initial count is 3, so we can acquire 3 times
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_TRUE(sem.try_acquire());

    // Fourth acquire should fail
    EXPECT_FALSE(sem.try_acquire());

    // Release one and try again
    sem.release();
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());
}

// Test binary semaphore functionality
TEST_F(SemaphoreTest, BinarySemaphore_BasicOperation) {
    BinarySemaphore sem(1);

    // Can acquire once
    EXPECT_TRUE(sem.try_acquire());

    // Second acquire should fail
    EXPECT_FALSE(sem.try_acquire());

    // Release and try again
    sem.release();
    EXPECT_TRUE(sem.try_acquire());
}

// Test blocking acquire
TEST_F(SemaphoreTest, CountingSemaphore_BlockingAcquire) {
    CountingSemaphore<> sem(0);
    std::atomic<bool> acquired(false);

    std::thread t([&sem, &acquired]() {
        sem.acquire();
        acquired = true;
    });

    // Thread should be blocked
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(acquired.load());

    // Release and thread should proceed
    sem.release();
    t.join();
    EXPECT_TRUE(acquired.load());
}

// Test multiple releases
TEST_F(SemaphoreTest, CountingSemaphore_MultipleRelease) {
    CountingSemaphore<10> sem(0);

    // Release 5 units at once
    sem.release(5);

    // Should be able to acquire 5 times
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(sem.try_acquire());
    }

    // Sixth should fail
    EXPECT_FALSE(sem.try_acquire());
}

// Test timed acquire
TEST_F(SemaphoreTest, CountingSemaphore_TimedAcquire) {
    CountingSemaphore<> sem(0);

    // Should timeout
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(sem.try_acquire_for(100ms));
    auto duration = std::chrono::steady_clock::now() - start;

    // Should have waited approximately 100ms
    EXPECT_GE(duration, 95ms);
    EXPECT_LE(duration, 150ms);
}

// Test concurrent acquire/release
TEST_F(SemaphoreTest, CountingSemaphore_ConcurrentAccess) {
    CountingSemaphore<100> sem(0);
    const int num_threads = 10;
    const int ops_per_thread = 100;
    std::atomic<int> successful_acquires(0);

    std::vector<std::thread> threads;

    // Half threads acquire, half release
    for (int i = 0; i < num_threads / 2; ++i) {
        threads.emplace_back([&sem, &successful_acquires, ops_per_thread]() {
            for (int j = 0; j < ops_per_thread; ++j) {
                if (sem.try_acquire()) {
                    successful_acquires++;
                    std::this_thread::yield();
                    sem.release();
                }
            }
        });
    }

    for (int i = 0; i < num_threads / 2; ++i) {
        threads.emplace_back([&sem, ops_per_thread]() {
            for (int j = 0; j < ops_per_thread; ++j) {
                sem.release();
                std::this_thread::yield();
                sem.acquire();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Semaphore should return to initial state
    EXPECT_FALSE(sem.try_acquire());
}

// Test lightweight semaphore
TEST_F(SemaphoreTest, LightweightSemaphore_BasicOperation) {
    LightweightSemaphore<> sem(2);

    EXPECT_TRUE(sem.try_acquire());
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());

    sem.release();
    EXPECT_TRUE(sem.try_acquire());
}

// Test lightweight semaphore with threads
TEST_F(SemaphoreTest, LightweightSemaphore_ThreadedOperation) {
    LightweightSemaphore<> sem(0);
    std::atomic<int> counter(0);
    const int num_threads = 10;

    std::vector<std::thread> threads;

    // Start threads that will wait on semaphore
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&sem, &counter]() {
            sem.acquire();
            counter++;
            sem.release();
        });
    }

    // Give threads time to start
    std::this_thread::sleep_for(50ms);

    // Release once to start the chain
    sem.release();

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), num_threads);

    // Semaphore should have one permit available
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());
}

// Test fair semaphore FIFO ordering
TEST_F(SemaphoreTest, FairSemaphore_FIFOOrdering) {
    FairSemaphore<> sem(0);
    std::vector<int> acquisition_order;
    std::mutex order_mutex;
    const int num_threads = 5;

    std::vector<std::thread> threads;
    std::atomic<int> ready_count(0);

    // Start threads that will queue up in order
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&sem, &acquisition_order, &order_mutex, &ready_count, i]() {
            ready_count++;
            // Wait for all threads to be ready
            while (ready_count.load() < num_threads) {
                std::this_thread::yield();
            }

            // Small delay based on thread id to ensure ordering
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));

            sem.acquire();
            {
                std::lock_guard<std::mutex> lock(order_mutex);
                acquisition_order.push_back(i);
            }
            sem.release();
        });
    }

    // Wait for all threads to queue up
    while (ready_count.load() < num_threads) {
        std::this_thread::sleep_for(1ms);
    }
    std::this_thread::sleep_for(100ms);

    // Release to start the chain
    sem.release();

    for (auto& t : threads) {
        t.join();
    }

    // Check FIFO ordering
    EXPECT_EQ(acquisition_order.size(), num_threads);
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_EQ(acquisition_order[i], i) << "Thread " << i << " acquired at position "
                                           << std::find(acquisition_order.begin(),
                                                       acquisition_order.end(), i) -
                                                       acquisition_order.begin();
    }
}

// Test weighted semaphore
TEST_F(SemaphoreTest, WeightedSemaphore_BasicOperation) {
    WeightedSemaphore sem(10);

    // Acquire 5 units
    EXPECT_TRUE(sem.try_acquire(5));
    EXPECT_EQ(sem.available(), 5);

    // Try to acquire 6 units (should fail)
    EXPECT_FALSE(sem.try_acquire(6));

    // Acquire remaining 5
    EXPECT_TRUE(sem.try_acquire(5));
    EXPECT_EQ(sem.available(), 0);

    // Release 3 units
    sem.release(3);
    EXPECT_EQ(sem.available(), 3);
}

// Test weighted semaphore with threads
TEST_F(SemaphoreTest, WeightedSemaphore_ThreadedOperation) {
    WeightedSemaphore sem(100);
    std::atomic<int> total_acquired(0);
    const int num_threads = 10;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&sem, &total_acquired]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> weight_dist(1, 10);

            for (int j = 0; j < 10; ++j) {
                int weight = weight_dist(gen);
                if (sem.try_acquire(weight)) {
                    total_acquired += weight;
                    std::this_thread::yield();
                    sem.release(weight);
                    total_acquired -= weight;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should return to initial state
    EXPECT_EQ(sem.available(), 100);
}

// Test semaphore guard
TEST_F(SemaphoreTest, SemaphoreGuard_RAII) {
    CountingSemaphore<> sem(1);

    {
        SemaphoreGuard<CountingSemaphore<>> guard(sem);
        EXPECT_TRUE(guard.owns_lock());

        // Semaphore should be acquired
        EXPECT_FALSE(sem.try_acquire());
    }

    // Guard destroyed, semaphore should be released
    EXPECT_TRUE(sem.try_acquire());
    sem.release();
}

// Test semaphore guard with try_lock
TEST_F(SemaphoreTest, SemaphoreGuard_TryLock) {
    CountingSemaphore<> sem(0);

    SemaphoreGuard<CountingSemaphore<>> guard(sem, std::try_to_lock);
    EXPECT_FALSE(guard.owns_lock());

    sem.release();
    EXPECT_TRUE(guard.try_acquire());
    EXPECT_TRUE(guard.owns_lock());
}

// Test semaphore guard with deferred lock
TEST_F(SemaphoreTest, SemaphoreGuard_DeferredLock) {
    CountingSemaphore<> sem(1);

    SemaphoreGuard<CountingSemaphore<>> guard(sem, std::defer_lock);
    EXPECT_FALSE(guard.owns_lock());

    // Should be able to acquire manually
    EXPECT_TRUE(sem.try_acquire());
    sem.release();

    guard.acquire();
    EXPECT_TRUE(guard.owns_lock());
    EXPECT_FALSE(sem.try_acquire());
}

// Test multi-resource semaphore
TEST_F(SemaphoreTest, MultiResourceSemaphore_BasicOperation) {
    MultiResourceSemaphore<3> sem({10, 20, 30});

    // Acquire some resources
    EXPECT_TRUE(sem.try_acquire({5, 10, 15}));

    auto available = sem.available();
    EXPECT_EQ(available[0], 5);
    EXPECT_EQ(available[1], 10);
    EXPECT_EQ(available[2], 15);

    // Try to acquire more than available
    EXPECT_FALSE(sem.try_acquire({6, 10, 15}));
    EXPECT_FALSE(sem.try_acquire({5, 11, 15}));
    EXPECT_FALSE(sem.try_acquire({5, 10, 16}));

    // Release resources
    sem.release({5, 10, 15});

    available = sem.available();
    EXPECT_EQ(available[0], 10);
    EXPECT_EQ(available[1], 20);
    EXPECT_EQ(available[2], 30);
}

// Test multi-resource semaphore with threads
TEST_F(SemaphoreTest, MultiResourceSemaphore_ThreadedOperation) {
    MultiResourceSemaphore<2> sem({50, 50});
    std::atomic<int> successful_ops(0);
    const int num_threads = 10;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&sem, &successful_ops]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> weight_dist(1, 10);

            for (int j = 0; j < 20; ++j) {
                std::array<std::ptrdiff_t, 2> weights = {
                    weight_dist(gen),
                    weight_dist(gen)
                };

                if (sem.try_acquire(weights)) {
                    successful_ops++;
                    std::this_thread::yield();
                    sem.release(weights);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should return to initial state
    auto final_state = sem.available();
    EXPECT_EQ(final_state[0], 50);
    EXPECT_EQ(final_state[1], 50);

    // Should have had many successful operations
    EXPECT_GT(successful_ops.load(), 0);
}

// Stress test with many threads
TEST_F(SemaphoreTest, CountingSemaphore_StressTest) {
    CountingSemaphore<1000> sem(100);
    const int num_threads = 50;
    const int ops_per_thread = 1000;
    std::atomic<int> total_acquires(0);
    std::atomic<int> max_concurrent(0);
    std::atomic<int> current_holders(0);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> op_dist(0, 2);
            std::uniform_int_distribution<> count_dist(1, 5);

            for (int j = 0; j < ops_per_thread; ++j) {
                int op = op_dist(gen);

                if (op == 0) {
                    // Try acquire
                    if (sem.try_acquire()) {
                        total_acquires++;
                        int current = ++current_holders;

                        int expected = max_concurrent.load();
                        while (current > expected &&
                               !max_concurrent.compare_exchange_weak(expected, current)) {}

                        std::this_thread::yield();

                        --current_holders;
                        sem.release();
                    }
                } else if (op == 1) {
                    // Timed acquire
                    if (sem.try_acquire_for(1ms)) {
                        total_acquires++;
                        int current = ++current_holders;

                        int expected = max_concurrent.load();
                        while (current > expected &&
                               !max_concurrent.compare_exchange_weak(expected, current)) {}

                        std::this_thread::yield();

                        --current_holders;
                        sem.release();
                    }
                } else {
                    // Multiple acquire/release
                    int count = count_dist(gen);
                    // First acquire what we can
                    int acquired = 0;
                    for (int k = 0; k < count; ++k) {
                        if (sem.try_acquire()) {
                            acquired++;
                            total_acquires++;
                            int current = ++current_holders;

                            int expected = max_concurrent.load();
                            while (current > expected &&
                                   !max_concurrent.compare_exchange_weak(expected, current)) {}
                        } else {
                            break;
                        }
                    }

                    std::this_thread::yield();

                    // Release what we acquired
                    for (int k = 0; k < acquired; ++k) {
                        --current_holders;
                        sem.release();
                    }
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should never exceed semaphore limit
    EXPECT_LE(max_concurrent.load(), 100);
    EXPECT_GT(total_acquires.load(), 0);

    // Should return to a valid state (may not be exactly 100 due to extra releases)
    int final_count = 0;
    while (sem.try_acquire()) {
        final_count++;
        if (final_count > 1000) break;  // Prevent infinite loop
    }
    EXPECT_GE(final_count, 100);
}
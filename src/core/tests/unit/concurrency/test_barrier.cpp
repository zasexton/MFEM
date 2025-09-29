#include <gtest/gtest.h>
#include <core/concurrency/barrier.h>

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <random>

using namespace fem::core::concurrency;

class BarrierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any static counters
    }

    void TearDown() override {
        // Clean up after each test
    }
};

// Basic barrier functionality tests
TEST_F(BarrierTest, BasicConstruction) {
    Barrier barrier(3);
    EXPECT_EQ(barrier.thread_count(), 3);
    EXPECT_EQ(barrier.waiting_count(), 3);
    EXPECT_EQ(barrier.generation(), 0);
}

TEST_F(BarrierTest, SingleThreadArrival) {
    Barrier barrier(1);

    // Single thread should pass through immediately
    barrier.arrive_and_wait();

    EXPECT_EQ(barrier.generation(), 1);
    EXPECT_EQ(barrier.waiting_count(), 1); // Reset for next cycle
}

TEST_F(BarrierTest, TwoThreadSynchronization) {
    Barrier barrier(2);
    std::atomic<int> phase{0};
    std::atomic<bool> thread1_phase1{false};
    std::atomic<bool> thread2_phase1{false};

    auto worker = [&](int thread_id) {
        // Phase 1
        phase.store(1);
        if (thread_id == 1) thread1_phase1 = true;
        if (thread_id == 2) thread2_phase1 = true;

        barrier.arrive_and_wait();

        // Phase 2 - both threads should reach here together
        phase.store(2);

        // Verify both threads completed phase 1
        EXPECT_TRUE(thread1_phase1.load());
        EXPECT_TRUE(thread2_phase1.load());
    };

    std::thread t1(worker, 1);
    std::thread t2(worker, 2);

    t1.join();
    t2.join();

    EXPECT_EQ(phase.load(), 2);
    EXPECT_EQ(barrier.generation(), 1);
}

TEST_F(BarrierTest, MultipleThreadSynchronization) {
    constexpr int num_threads = 8;
    Barrier barrier(num_threads);

    std::atomic<int> phase1_completed{0};
    std::atomic<int> phase2_completed{0};

    auto worker = [&]() {
        // Phase 1 work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        phase1_completed++;

        barrier.arrive_and_wait();

        // Phase 2 work - all threads should see phase1_completed == num_threads
        EXPECT_EQ(phase1_completed.load(), num_threads);
        phase2_completed++;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(phase1_completed.load(), num_threads);
    EXPECT_EQ(phase2_completed.load(), num_threads);
}

TEST_F(BarrierTest, MultipleBarrierCycles) {
    constexpr int num_threads = 4;
    constexpr int num_cycles = 5;
    Barrier barrier(num_threads);

    std::atomic<int> total_completions{0};

    auto worker = [&]() {
        for (int cycle = 0; cycle < num_cycles; ++cycle) {
            barrier.arrive_and_wait();
            total_completions++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(total_completions.load(), num_threads * num_cycles);
    EXPECT_EQ(barrier.generation(), num_cycles);
}

TEST_F(BarrierTest, ArriveAndDrop) {
    Barrier barrier(3);
    std::atomic<bool> thread2_passed{false};
    std::atomic<bool> thread3_passed{false};

    auto worker2 = [&]() {
        barrier.arrive_and_wait();
        thread2_passed = true;
    };

    auto worker3 = [&]() {
        barrier.arrive_and_wait();
        thread3_passed = true;
    };

    std::thread t2(worker2);
    std::thread t3(worker3);

    // Main thread drops out
    barrier.arrive_and_drop();

    t2.join();
    t3.join();

    EXPECT_TRUE(thread2_passed.load());
    EXPECT_TRUE(thread3_passed.load());
    EXPECT_EQ(barrier.thread_count(), 2); // Threshold reduced
}

TEST_F(BarrierTest, Reset) {
    Barrier barrier(2);

    // Use barrier once
    std::thread t1([&]() { barrier.arrive_and_wait(); });
    std::thread t2([&]() { barrier.arrive_and_wait(); });

    t1.join();
    t2.join();

    EXPECT_EQ(barrier.generation(), 1);

    // Reset for different thread count
    barrier.reset(3);
    EXPECT_EQ(barrier.thread_count(), 3);
    EXPECT_EQ(barrier.waiting_count(), 3);

    // Use with new thread count
    std::thread t3([&]() { barrier.arrive_and_wait(); });
    std::thread t4([&]() { barrier.arrive_and_wait(); });
    std::thread t5([&]() { barrier.arrive_and_wait(); });

    t3.join();
    t4.join();
    t5.join();

    EXPECT_EQ(barrier.generation(), 2); // Generation continues
}

// FlexBarrier tests
class FlexBarrierTest : public ::testing::Test {};

TEST_F(FlexBarrierTest, BasicFunctionality) {
    bool callback_executed = false;
    FlexBarrier barrier(2, [&]() { callback_executed = true; });

    std::thread t1([&]() { barrier.arrive_and_wait(); });
    std::thread t2([&]() { barrier.arrive_and_wait(); });

    t1.join();
    t2.join();

    EXPECT_TRUE(callback_executed);
    EXPECT_EQ(barrier.generation(), 1);
}

TEST_F(FlexBarrierTest, TimeoutSuccess) {
    FlexBarrier barrier(2);

    auto future = std::async(std::launch::async, [&]() {
        return barrier.arrive_and_wait_for(std::chrono::milliseconds(100));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    barrier.arrive_and_wait();

    EXPECT_TRUE(future.get());
}

TEST_F(FlexBarrierTest, TimeoutFailure) {
    FlexBarrier barrier(2);

    auto start = std::chrono::steady_clock::now();
    bool result = barrier.arrive_and_wait_for(std::chrono::milliseconds(50));
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    EXPECT_GE(duration, std::chrono::milliseconds(40)); // Allow some tolerance
}

TEST_F(FlexBarrierTest, CompletionCallback) {
    std::atomic<int> callback_count{0};

    FlexBarrier barrier(3, [&]() { callback_count++; });

    auto worker = [&]() { barrier.arrive_and_wait(); };

    std::thread t1(worker);
    std::thread t2(worker);
    std::thread t3(worker);

    t1.join();
    t2.join();
    t3.join();

    EXPECT_EQ(callback_count.load(), 1);
}

TEST_F(FlexBarrierTest, ChangeCallback) {
    std::atomic<int> callback1_count{0};
    std::atomic<int> callback2_count{0};

    FlexBarrier barrier(2, [&]() { callback1_count++; });

    // Use with first callback
    std::thread t1([&]() { barrier.arrive_and_wait(); });
    std::thread t2([&]() { barrier.arrive_and_wait(); });
    t1.join();
    t2.join();

    // Change callback
    barrier.set_completion_callback([&]() { callback2_count++; });

    // Use with second callback
    std::thread t3([&]() { barrier.arrive_and_wait(); });
    std::thread t4([&]() { barrier.arrive_and_wait(); });
    t3.join();
    t4.join();

    EXPECT_EQ(callback1_count.load(), 1);
    EXPECT_EQ(callback2_count.load(), 1);
}

// Semaphore tests
class SemaphoreTest : public ::testing::Test {};

TEST_F(SemaphoreTest, BasicAcquireRelease) {
    Semaphore sem(1);

    EXPECT_EQ(sem.available(), 1);

    sem.acquire();
    EXPECT_EQ(sem.available(), 0);

    sem.release();
    EXPECT_EQ(sem.available(), 1);
}

TEST_F(SemaphoreTest, TryAcquire) {
    Semaphore sem(1);

    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());

    sem.release();
    EXPECT_TRUE(sem.try_acquire());
}

TEST_F(SemaphoreTest, TryAcquireTimeout) {
    Semaphore sem(0);

    auto start = std::chrono::steady_clock::now();
    bool result = sem.try_acquire_for(std::chrono::milliseconds(50));
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    EXPECT_GE(duration, std::chrono::milliseconds(40));
}

TEST_F(SemaphoreTest, MultiplePermits) {
    Semaphore sem(3);

    EXPECT_EQ(sem.available(), 3);

    sem.acquire();
    sem.acquire();
    EXPECT_EQ(sem.available(), 1);

    sem.release(2);
    EXPECT_EQ(sem.available(), 3);
}

TEST_F(SemaphoreTest, ConcurrentAccess) {
    constexpr int num_threads = 8;
    constexpr int permits = 3;
    Semaphore sem(permits);

    std::atomic<int> concurrent_count{0};
    std::atomic<int> max_concurrent{0};

    auto worker = [&]() {
        sem.acquire();

        int current = concurrent_count.fetch_add(1) + 1;
        int expected_max = max_concurrent.load();
        while (current > expected_max &&
               !max_concurrent.compare_exchange_weak(expected_max, current)) {
            // Update max_concurrent
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        concurrent_count.fetch_sub(1);
        sem.release();
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_LE(max_concurrent.load(), permits);
    EXPECT_EQ(concurrent_count.load(), 0);
    EXPECT_EQ(sem.available(), permits);
}

// RAII Guard tests
class GuardTest : public ::testing::Test {};

TEST_F(GuardTest, BarrierGuardBasic) {
    Barrier barrier(2);
    bool worker_completed = false;

    std::thread worker([&]() {
        BarrierGuard guard(barrier);
        worker_completed = true;
        // Guard automatically triggers barrier on destruction
    });

    barrier.arrive_and_wait(); // Main thread waits

    worker.join();
    EXPECT_TRUE(worker_completed);
}

TEST_F(GuardTest, BarrierGuardManualTrigger) {
    Barrier barrier(2);
    bool manual_triggered = false;

    std::thread worker([&]() {
        BarrierGuard guard(barrier);
        guard.trigger(); // Manual trigger
        manual_triggered = true;
        // No automatic trigger on destruction
    });

    barrier.arrive_and_wait();

    worker.join();
    EXPECT_TRUE(manual_triggered);
}

TEST_F(GuardTest, BarrierGuardDrop) {
    Barrier barrier(3);

    std::thread worker1([&]() {
        BarrierGuard guard(barrier);
        guard.drop(); // Drop participation
    });

    std::thread worker2([&]() {
        barrier.arrive_and_wait();
    });

    // Main thread can now proceed with just worker2
    barrier.arrive_and_wait();

    worker1.join();
    worker2.join();

    EXPECT_EQ(barrier.thread_count(), 2);
}

TEST_F(GuardTest, SemaphoreGuardBasic) {
    Semaphore sem(1);

    {
        SemaphoreGuard guard(sem);
        EXPECT_EQ(sem.available(), 0);
        // Guard automatically releases on destruction
    }

    EXPECT_EQ(sem.available(), 1);
}

TEST_F(GuardTest, SemaphoreGuardManualRelease) {
    Semaphore sem(1);

    SemaphoreGuard guard(sem);
    EXPECT_EQ(sem.available(), 0);

    guard.release();
    EXPECT_EQ(sem.available(), 1);

    // Destruction should not release again
}

// Stress tests
TEST_F(BarrierTest, StressTest) {
    constexpr int num_threads = 16;
    constexpr int num_iterations = 100;
    Barrier barrier(num_threads);

    std::atomic<int> errors{0};

    auto worker = [&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10);

        for (int i = 0; i < num_iterations; ++i) {
            // Random work duration
            std::this_thread::sleep_for(std::chrono::microseconds(dis(gen)));

            try {
                barrier.arrive_and_wait();
            } catch (...) {
                errors++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(errors.load(), 0);
    EXPECT_EQ(barrier.generation(), num_iterations);
}

TEST_F(FlexBarrierTest, StressTestWithTimeout) {
    constexpr int num_threads = 8;
    constexpr int num_iterations = 50;
    FlexBarrier barrier(num_threads);

    std::atomic<int> successful_syncs{0};
    std::atomic<int> timeouts{0};

    auto worker = [&]() {
        for (int i = 0; i < num_iterations; ++i) {
            if (barrier.arrive_and_wait_for(std::chrono::milliseconds(100))) {
                successful_syncs++;
            } else {
                timeouts++;
                break; // Exit on timeout
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should complete successfully (no timeouts expected)
    EXPECT_EQ(timeouts.load(), 0);
    EXPECT_EQ(successful_syncs.load(), num_threads * num_iterations);
}
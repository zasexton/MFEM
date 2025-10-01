#include <gtest/gtest.h>
#include "core/concurrency/mutex.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

using namespace fem::core::concurrency;
using namespace std::chrono_literals;

class MutexTest : public ::testing::Test {
protected:
    static constexpr int NUM_THREADS = 8;
    static constexpr int ITERATIONS = 10000;
};

// -----------------------------------------------------------------------------
// TimedMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, TimedMutex_BasicLocking) {
    TimedMutex mutex;
    int counter = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < ITERATIONS; ++j) {
                mutex.lock();
                ++counter;
                mutex.unlock();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter, NUM_THREADS * ITERATIONS);
}

TEST_F(MutexTest, TimedMutex_TryLock) {
    TimedMutex mutex;

    EXPECT_TRUE(mutex.try_lock());

    std::thread t([&] {
        EXPECT_FALSE(mutex.try_lock());
    });
    t.join();

    mutex.unlock();

    EXPECT_TRUE(mutex.try_lock());
    mutex.unlock();
}

TEST_F(MutexTest, TimedMutex_TryLockFor) {
    TimedMutex mutex;
    mutex.lock();

    std::thread t([&] {
        auto start = std::chrono::steady_clock::now();
        EXPECT_FALSE(mutex.try_lock_for(100ms));
        auto duration = std::chrono::steady_clock::now() - start;
        EXPECT_GE(duration, 100ms);
        EXPECT_LT(duration, 200ms);
    });

    t.join();
    mutex.unlock();
}

// -----------------------------------------------------------------------------
// RecursiveMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, RecursiveMutex_Reentrancy) {
    RecursiveMutex mutex;

    mutex.lock();
    EXPECT_TRUE(mutex.is_locked_by_current_thread());
    EXPECT_EQ(mutex.recursion_level(), 1u);

    mutex.lock();  // Should not deadlock
    EXPECT_EQ(mutex.recursion_level(), 2u);

    mutex.unlock();
    EXPECT_EQ(mutex.recursion_level(), 1u);
    EXPECT_TRUE(mutex.is_locked_by_current_thread());

    mutex.unlock();
    EXPECT_EQ(mutex.recursion_level(), 0u);
    EXPECT_FALSE(mutex.is_locked_by_current_thread());
}

TEST_F(MutexTest, RecursiveMutex_ThreadOwnership) {
    RecursiveMutex mutex;
    std::atomic<bool> thread_started{false};

    mutex.lock();
    auto main_thread_id = std::this_thread::get_id();
    EXPECT_EQ(mutex.owner(), main_thread_id);

    std::thread t([&] {
        thread_started = true;
        EXPECT_FALSE(mutex.try_lock());
        EXPECT_NE(mutex.owner(), std::this_thread::get_id());
    });

    while (!thread_started) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(10ms);

    mutex.unlock();
    t.join();
}

// -----------------------------------------------------------------------------
// DebugMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, DebugMutex_LockTracking) {
    DebugMutex mutex;

    mutex.lock("test_location");
    EXPECT_EQ(mutex.owner(), std::this_thread::get_id());
    EXPECT_EQ(mutex.lock_location(), "test_location");

    std::this_thread::sleep_for(10ms);
    auto duration = mutex.held_duration();
    EXPECT_GE(duration.count(), 10);

    mutex.unlock();
    EXPECT_EQ(mutex.owner(), std::thread::id{});
}

TEST_F(MutexTest, DebugMutex_DeadlockDetection) {
    DebugMutex mutex;

    mutex.lock("first_lock");

    // Trying to lock the same mutex again should detect deadlock
    EXPECT_THROW(mutex.lock("second_lock"), std::runtime_error);

    mutex.unlock();
}

TEST_F(MutexTest, DebugMutex_ContentionTracking) {
    DebugMutex mutex;
    std::atomic<int> counter{0};

    auto initial_contention = mutex.contention_count();

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 100; ++j) {
                mutex.lock();
                ++counter;
                std::this_thread::sleep_for(100us);
                mutex.unlock();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter, 400);
    EXPECT_GT(mutex.contention_count(), initial_contention);
}

// -----------------------------------------------------------------------------
// SharedMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, SharedMutex_MultipleReaders) {
    SharedMutex mutex;
    std::atomic<int> reader_count{0};
    std::atomic<int> max_concurrent_readers{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&] {
            mutex.lock_shared();

            int current = ++reader_count;
            int max_val = max_concurrent_readers.load();
            while (current > max_val &&
                   !max_concurrent_readers.compare_exchange_weak(max_val, current)) {
                max_val = max_concurrent_readers.load();
            }

            std::this_thread::sleep_for(10ms);
            --reader_count;

            mutex.unlock_shared();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(max_concurrent_readers.load(), 1);
    EXPECT_LE(max_concurrent_readers.load(), NUM_THREADS);
}

TEST_F(MutexTest, SharedMutex_WriterExclusivity) {
    SharedMutex mutex;
    int shared_data = 0;
    std::atomic<bool> writer_active{false};

    // Start readers
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            for (int j = 0; j < 100; ++j) {
                mutex.lock_shared();
                // Verify no writer is active during read
                EXPECT_FALSE(writer_active);
                [[maybe_unused]] int temp = shared_data;
                mutex.unlock_shared();
                std::this_thread::yield();
            }
        });
    }

    // Start writers
    std::vector<std::thread> writers;
    for (int i = 0; i < 2; ++i) {
        writers.emplace_back([&] {
            for (int j = 0; j < 50; ++j) {
                mutex.lock();
                writer_active = true;
                // Verify exclusive access
                EXPECT_EQ(mutex.reader_count(), 0u);
                ++shared_data;
                writer_active = false;
                mutex.unlock();
                std::this_thread::yield();
            }
        });
    }

    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    EXPECT_EQ(shared_data, 100);
}

// -----------------------------------------------------------------------------
// PriorityMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, PriorityMutex_PriorityOrdering) {
    PriorityMutex mutex;
    std::vector<int> acquisition_order;
    std::mutex order_mutex;

    // Lock the mutex initially
    mutex.lock();

    // Start threads with different priorities
    std::vector<std::thread> threads;
    std::atomic<int> threads_ready{0};

    for (int priority : {1, 10, 5, 8, 3}) {
        threads.emplace_back([&, priority] {
            ++threads_ready;
            while (threads_ready < 5) {
                std::this_thread::yield();
            }

            mutex.lock(priority);
            {
                std::lock_guard<std::mutex> lock(order_mutex);
                acquisition_order.push_back(priority);
            }
            mutex.unlock();
        });
    }

    // Wait for all threads to be ready
    while (threads_ready < 5) {
        std::this_thread::sleep_for(1ms);
    }
    std::this_thread::sleep_for(10ms);  // Let them queue up

    // Release the mutex
    mutex.unlock();

    for (auto& t : threads) {
        t.join();
    }

    // Verify priority ordering (highest priority first)
    EXPECT_EQ(acquisition_order.size(), 5u);
    for (size_t i = 1; i < acquisition_order.size(); ++i) {
        EXPECT_GE(acquisition_order[i-1], acquisition_order[i]);
    }
}

// -----------------------------------------------------------------------------
// AdaptiveMutex Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, AdaptiveMutex_BasicLocking) {
    AdaptiveMutex mutex;
    int counter = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < ITERATIONS; ++j) {
                mutex.lock();
                ++counter;
                mutex.unlock();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter, NUM_THREADS * ITERATIONS);
}

TEST_F(MutexTest, AdaptiveMutex_SpinAdaptation) {
    AdaptiveMutex mutex;

    auto initial_spin = mutex.get_spin_count();

    // Create contention
    std::atomic<bool> stop{false};
    std::thread holder([&] {
        while (!stop) {
            mutex.lock();
            std::this_thread::sleep_for(1ms);
            mutex.unlock();
            std::this_thread::yield();
        }
    });

    std::this_thread::sleep_for(10ms);

    // Try to acquire lock multiple times under contention
    for (int i = 0; i < 10; ++i) {
        mutex.lock();
        mutex.unlock();
    }

    stop = true;
    holder.join();

    // Spin count should have adapted
    EXPECT_NE(mutex.get_spin_count(), initial_spin);
    EXPECT_GT(mutex.contention_count(), 0u);
}

// -----------------------------------------------------------------------------
// RAII Lock Guards Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, ScopedLock_RAII) {
    TimedMutex mutex;
    int counter = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < ITERATIONS; ++j) {
                ScopedLock<TimedMutex> lock(mutex);
                ++counter;
                // Lock automatically released at end of scope
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter, NUM_THREADS * ITERATIONS);
}

TEST_F(MutexTest, ScopedLock_DeferLock) {
    TimedMutex mutex;

    {
        ScopedLock<TimedMutex> lock(mutex, std::defer_lock);
        EXPECT_FALSE(lock.owns_lock());

        lock.lock();
        EXPECT_TRUE(lock.owns_lock());

        lock.unlock();
        EXPECT_FALSE(lock.owns_lock());
    }

    // Should be able to lock after scope exit
    EXPECT_TRUE(mutex.try_lock());
    mutex.unlock();
}

TEST_F(MutexTest, SharedLock_MultipleReaders) {
    SharedMutex mutex;
    std::atomic<int> active_readers{0};
    std::atomic<int> max_readers{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&] {
            SharedLock<SharedMutex> lock(mutex);

            int current = ++active_readers;
            int max_val = max_readers.load();
            while (current > max_val &&
                   !max_readers.compare_exchange_weak(max_val, current)) {
                max_val = max_readers.load();
            }

            std::this_thread::sleep_for(10ms);
            --active_readers;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(max_readers.load(), 1);
}

// -----------------------------------------------------------------------------
// Stress Tests
// -----------------------------------------------------------------------------
TEST_F(MutexTest, StressTest_HighContention) {
    TimedMutex mutex;
    std::atomic<int> counter{0};
    std::atomic<bool> stop{false};

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS * 2; ++i) {
        threads.emplace_back([&] {
            while (!stop) {
                ScopedLock<TimedMutex> lock(mutex);
                ++counter;
                if (counter >= 100000) {
                    stop = true;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GE(counter.load(), 100000);
}

TEST_F(MutexTest, StressTest_MixedOperations) {
    SharedMutex mutex;
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    int shared_value = 0;

    auto deadline = std::chrono::steady_clock::now() + 100ms;

    // Reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        readers.emplace_back([&] {
            while (std::chrono::steady_clock::now() < deadline) {
                ReadLock lock(mutex);
                [[maybe_unused]] int temp = shared_value;
                ++read_count;
            }
        });
    }

    // Writer threads
    std::vector<std::thread> writers;
    for (int i = 0; i < 2; ++i) {
        writers.emplace_back([&] {
            while (std::chrono::steady_clock::now() < deadline) {
                WriteLock lock(mutex);
                ++shared_value;
                ++write_count;
            }
        });
    }

    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    EXPECT_EQ(shared_value, write_count.load());
    EXPECT_GT(read_count.load(), write_count.load());  // Reads should be more frequent
}
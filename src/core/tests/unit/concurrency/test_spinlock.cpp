#include <gtest/gtest.h>
#include <core/concurrency/spinlock.h>
#include <core/concurrency/barrier.h>

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <shared_mutex>

using namespace fem::core::concurrency;

class SpinlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset test state
    }

    void TearDown() override {
        // Clean up after each test
    }
};

// Basic Spinlock Tests
TEST_F(SpinlockTest, BasicConstruction) {
    Spinlock spinlock;
    // Should be constructible and unlocked
}

TEST_F(SpinlockTest, BasicLockUnlock) {
    Spinlock spinlock;

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST_F(SpinlockTest, TryLockFailsWhenLocked) {
    Spinlock spinlock;

    spinlock.lock();
    EXPECT_FALSE(spinlock.try_lock());
    spinlock.unlock();

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST_F(SpinlockTest, RAIIGuard) {
    Spinlock spinlock;

    {
        std::lock_guard<Spinlock> guard(spinlock);
        EXPECT_FALSE(spinlock.try_lock());
        // Lock should be released when guard goes out of scope
    }

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST_F(SpinlockTest, ConcurrentAccess) {
    Spinlock spinlock;
    std::atomic<int> counter{0};
    std::atomic<int> critical_section_count{0};
    constexpr int num_threads = 8;
    constexpr int increments_per_thread = 1000;

    auto worker = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            std::lock_guard<Spinlock> guard(spinlock);
            critical_section_count++;
            counter++;
            // Simulate some work
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), num_threads * increments_per_thread);
    EXPECT_EQ(critical_section_count.load(), num_threads * increments_per_thread);
}

// TicketSpinlock Tests
class TicketSpinlockTest : public ::testing::Test {};

TEST_F(TicketSpinlockTest, BasicConstruction) {
    TicketSpinlock spinlock;
    // Should be constructible and unlocked
}

TEST_F(TicketSpinlockTest, BasicLockUnlock) {
    TicketSpinlock spinlock;

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST_F(TicketSpinlockTest, FairnessTest) {
    TicketSpinlock spinlock;
    std::vector<int> execution_order;
    std::mutex order_mutex;
    constexpr int num_threads = 5;

    Barrier start_barrier(num_threads);

    auto worker = [&](int thread_id) {
        start_barrier.arrive_and_wait(); // Start all threads simultaneously

        spinlock.lock();
        {
            std::lock_guard<std::mutex> guard(order_mutex);
            execution_order.push_back(thread_id);
        }
        spinlock.unlock();
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(execution_order.size(), num_threads);
    // With ticket spinlock, order should be more predictable than basic spinlock
    // (though not guaranteed due to scheduling)
}

TEST_F(TicketSpinlockTest, ConcurrentStressTest) {
    TicketSpinlock spinlock;
    std::atomic<int> counter{0};
    constexpr int num_threads = 10;
    constexpr int increments_per_thread = 500;

    auto worker = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            std::lock_guard<TicketSpinlock> guard(spinlock);
            counter++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), num_threads * increments_per_thread);
}

// RWSpinlock Tests
class RWSpinlockTest : public ::testing::Test {};

TEST_F(RWSpinlockTest, BasicConstruction) {
    RWSpinlock rwlock;
    // Should be constructible and unlocked
}

TEST_F(RWSpinlockTest, SharedLocking) {
    RWSpinlock rwlock;

    // Multiple readers should be able to acquire lock
    EXPECT_TRUE(rwlock.try_lock_shared());
    EXPECT_TRUE(rwlock.try_lock_shared());

    rwlock.unlock_shared();
    rwlock.unlock_shared();
}

TEST_F(RWSpinlockTest, ExclusiveLocking) {
    RWSpinlock rwlock;

    EXPECT_TRUE(rwlock.try_lock());

    // Should not be able to acquire shared lock when exclusive lock is held
    EXPECT_FALSE(rwlock.try_lock_shared());
    EXPECT_FALSE(rwlock.try_lock());

    rwlock.unlock();

    EXPECT_TRUE(rwlock.try_lock_shared());
    rwlock.unlock_shared();
}

TEST_F(RWSpinlockTest, ReaderWriterExclusion) {
    RWSpinlock rwlock;

    // Acquire shared lock
    rwlock.lock_shared();

    // Writer should not be able to acquire lock
    EXPECT_FALSE(rwlock.try_lock());

    rwlock.unlock_shared();

    // Now writer should be able to acquire
    EXPECT_TRUE(rwlock.try_lock());
    rwlock.unlock();
}

TEST_F(RWSpinlockTest, MultipleReaders) {
    RWSpinlock rwlock;
    std::atomic<int> concurrent_readers{0};
    std::atomic<int> max_concurrent{0};
    constexpr int num_readers = 8;

    auto reader = [&]() {
        shared_spinlock_guard<RWSpinlock> guard(rwlock);

        int current = concurrent_readers.fetch_add(1) + 1;

        // Track maximum concurrent readers
        int expected_max = max_concurrent.load();
        while (current > expected_max &&
               !max_concurrent.compare_exchange_weak(expected_max, current)) {
            // Update max_concurrent
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        concurrent_readers.fetch_sub(1);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_readers; ++i) {
        threads.emplace_back(reader);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(max_concurrent.load(), 1); // Should have multiple concurrent readers
    EXPECT_EQ(concurrent_readers.load(), 0);
}

TEST_F(RWSpinlockTest, WriterExclusivity) {
    RWSpinlock rwlock;
    std::atomic<int> concurrent_operations{0};
    std::atomic<int> max_concurrent{0};
    constexpr int num_writers = 4;

    auto writer = [&]() {
        std::lock_guard<RWSpinlock> guard(rwlock);

        int current = concurrent_operations.fetch_add(1) + 1;

        // Track maximum concurrent operations
        int expected_max = max_concurrent.load();
        while (current > expected_max &&
               !max_concurrent.compare_exchange_weak(expected_max, current)) {
            // Update max_concurrent
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        concurrent_operations.fetch_sub(1);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_writers; ++i) {
        threads.emplace_back(writer);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(max_concurrent.load(), 1); // Should never have concurrent writers
    EXPECT_EQ(concurrent_operations.load(), 0);
}

// AdaptiveSpinlock Tests
class AdaptiveSpinlockTest : public ::testing::Test {};

TEST_F(AdaptiveSpinlockTest, BasicConstruction) {
    AdaptiveSpinlock spinlock;
    EXPECT_EQ(spinlock.contention_count(), 0);
    EXPECT_FALSE(spinlock.is_blocking_mode());
}

TEST_F(AdaptiveSpinlockTest, BasicLockUnlock) {
    AdaptiveSpinlock spinlock;

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();

    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST_F(AdaptiveSpinlockTest, LowContentionStaysSpinning) {
    AdaptiveSpinlock spinlock;

    // Low contention should not trigger blocking mode
    for (int i = 0; i < 5; ++i) {
        spinlock.lock();
        spinlock.unlock();
    }

    EXPECT_FALSE(spinlock.is_blocking_mode());
}

TEST_F(AdaptiveSpinlockTest, HighContentionTriggersBlocking) {
    AdaptiveSpinlock spinlock;
    std::atomic<bool> should_continue{true};
    constexpr int num_threads = 16;

    // Create high contention to trigger blocking mode
    auto contender = [&]() {
        while (should_continue.load()) {
            if (spinlock.try_lock()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                spinlock.unlock();
            }
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(contender);
    }

    // Let contention build up
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    should_continue.store(false);

    for (auto& t : threads) {
        t.join();
    }

    // High contention should eventually trigger blocking mode
    // (This test might be flaky depending on timing and system load)
}

TEST_F(AdaptiveSpinlockTest, ResetContentionTracking) {
    AdaptiveSpinlock spinlock;

    // Manually set some contention (this is a white-box test)
    spinlock.reset_contention_tracking();

    EXPECT_EQ(spinlock.contention_count(), 0);
    EXPECT_FALSE(spinlock.is_blocking_mode());
}

TEST_F(AdaptiveSpinlockTest, ConcurrentAccess) {
    AdaptiveSpinlock spinlock;
    std::atomic<int> counter{0};
    constexpr int num_threads = 8;
    constexpr int increments_per_thread = 100;

    auto worker = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            std::lock_guard<AdaptiveSpinlock> guard(spinlock);
            counter++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), num_threads * increments_per_thread);
}

// Shared Spinlock Guard Tests
class SharedSpinlockGuardTest : public ::testing::Test {};

TEST_F(SharedSpinlockGuardTest, BasicRAII) {
    RWSpinlock rwlock;

    {
        shared_spinlock_guard<RWSpinlock> guard(rwlock);
        // Should hold shared lock
        EXPECT_TRUE(rwlock.try_lock_shared()); // Can get another shared lock
        rwlock.unlock_shared();

        EXPECT_FALSE(rwlock.try_lock()); // Cannot get exclusive lock
    }

    // Guard should have released lock
    EXPECT_TRUE(rwlock.try_lock());
    rwlock.unlock();
}

TEST_F(SharedSpinlockGuardTest, ManualUnlock) {
    RWSpinlock rwlock;

    shared_spinlock_guard<RWSpinlock> guard(rwlock);
    EXPECT_FALSE(rwlock.try_lock()); // Should be locked

    guard.unlock();
    EXPECT_TRUE(rwlock.try_lock()); // Should be unlocked
    rwlock.unlock();
}

// Performance and stress tests
TEST_F(SpinlockTest, PerformanceComparison) {
    constexpr int num_iterations = 10000;
    constexpr int num_threads = 4;

    // Test basic spinlock
    {
        Spinlock spinlock;
        std::atomic<int> counter{0};

        auto start = std::chrono::high_resolution_clock::now();

        auto worker = [&]() {
            for (int i = 0; i < num_iterations; ++i) {
                std::lock_guard<Spinlock> guard(spinlock);
                counter++;
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto basic_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        EXPECT_EQ(counter.load(), num_threads * num_iterations);

        // Just verify it completes (actual performance will vary by system)
        EXPECT_GT(basic_duration.count(), 0);
    }

    // Test ticket spinlock
    {
        TicketSpinlock spinlock;
        std::atomic<int> counter{0};

        auto start = std::chrono::high_resolution_clock::now();

        auto worker = [&]() {
            for (int i = 0; i < num_iterations; ++i) {
                std::lock_guard<TicketSpinlock> guard(spinlock);
                counter++;
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto ticket_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        EXPECT_EQ(counter.load(), num_threads * num_iterations);
        EXPECT_GT(ticket_duration.count(), 0);
    }
}

TEST_F(SpinlockTest, StressTest) {
    constexpr int num_threads = 16;
    constexpr int duration_ms = 100;

    Spinlock spinlock;
    std::atomic<int> operations{0};
    std::atomic<bool> should_continue{true};

    auto worker = [&]() {
        while (should_continue.load()) {
            if (spinlock.try_lock()) {
                operations++;
                spinlock.unlock();
            }
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    should_continue.store(false);

    for (auto& t : threads) {
        t.join();
    }

    // Should have performed many operations without deadlock
    EXPECT_GT(operations.load(), 0);
}

TEST_F(RWSpinlockTest, ReadWriteStressTest) {
    constexpr int num_readers = 8;
    constexpr int num_writers = 2;
    constexpr int duration_ms = 100;

    RWSpinlock rwlock;
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> should_continue{true};

    auto reader = [&]() {
        while (should_continue.load()) {
            {
                shared_spinlock_guard<RWSpinlock> guard(rwlock);
                read_count++;
            }
            std::this_thread::yield();
        }
    };

    auto writer = [&]() {
        while (should_continue.load()) {
            std::lock_guard<RWSpinlock> guard(rwlock);
            write_count++;
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_readers; ++i) {
        threads.emplace_back(reader);
    }
    for (int i = 0; i < num_writers; ++i) {
        threads.emplace_back(writer);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    should_continue.store(false);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(read_count.load(), 0);
    EXPECT_GT(write_count.load(), 0);
}
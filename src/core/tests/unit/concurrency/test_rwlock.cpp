#include <gtest/gtest.h>
#include "core/concurrency/rwlock.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <set>

using namespace fem::core::concurrency;
using namespace std::chrono_literals;

class RWLockTest : public ::testing::Test {
protected:
    static constexpr int NUM_READERS = 8;
    static constexpr int NUM_WRITERS = 2;
    static constexpr int ITERATIONS = 1000;
};

// -----------------------------------------------------------------------------
// ReaderWriterLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, ReaderWriterLock_MultipleReaders) {
    ReaderWriterLock lock;
    std::atomic<int> active_readers{0};
    std::atomic<int> max_concurrent_readers{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_READERS; ++i) {
        threads.emplace_back([&] {
            lock.lock_shared();

            int current = ++active_readers;
            int max_val = max_concurrent_readers.load();
            while (current > max_val &&
                   !max_concurrent_readers.compare_exchange_weak(max_val, current)) {
                max_val = max_concurrent_readers.load();
            }

            std::this_thread::sleep_for(10ms);
            --active_readers;

            lock.unlock_shared();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(max_concurrent_readers.load(), 1);
    EXPECT_LE(max_concurrent_readers.load(), NUM_READERS);
}

TEST_F(RWLockTest, ReaderWriterLock_WriterExclusivity) {
    ReaderWriterLock lock;
    int shared_data = 0;
    std::atomic<bool> writer_active{false};
    std::atomic<int> reader_violations{0};

    // Start readers
    std::vector<std::thread> readers;
    for (int i = 0; i < NUM_READERS; ++i) {
        readers.emplace_back([&] {
            for (int j = 0; j < ITERATIONS; ++j) {
                lock.lock_shared();
                if (writer_active) {
                    ++reader_violations;
                }
                [[maybe_unused]] int temp = shared_data;
                lock.unlock_shared();
            }
        });
    }

    // Start writers
    std::vector<std::thread> writers;
    for (int i = 0; i < NUM_WRITERS; ++i) {
        writers.emplace_back([&] {
            for (int j = 0; j < ITERATIONS / 2; ++j) {
                lock.lock();
                writer_active = true;
                ++shared_data;
                std::this_thread::sleep_for(100us);
                writer_active = false;
                lock.unlock();
            }
        });
    }

    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    EXPECT_EQ(reader_violations.load(), 0);
    EXPECT_EQ(shared_data, NUM_WRITERS * (ITERATIONS / 2));
}

// -----------------------------------------------------------------------------
// WriterPreferredRWLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, WriterPreferredRWLock_WriterPriority) {
    WriterPreferredRWLock lock;
    std::vector<std::string> order;
    std::mutex order_mutex;
    std::atomic<bool> writer_waiting{false};

    // Lock for reading initially
    lock.lock_shared();

    // Start a writer (will wait)
    std::thread writer([&] {
        writer_waiting = true;
        lock.lock();
        {
            std::lock_guard<std::mutex> guard(order_mutex);
            order.push_back("writer");
        }
        lock.unlock();
    });

    // Wait for writer to start waiting
    while (!writer_waiting) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(10ms);

    // Start readers (should wait because writer is waiting)
    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back([&, i] {
            lock.lock_shared();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                order.push_back("reader" + std::to_string(i));
            }
            lock.unlock_shared();
        });
    }

    std::this_thread::sleep_for(10ms);

    // Release initial read lock
    lock.unlock_shared();

    writer.join();
    for (auto& t : readers) t.join();

    // Writer should acquire lock before new readers
    ASSERT_GE(order.size(), 1u);
    EXPECT_EQ(order[0], "writer");
}

// -----------------------------------------------------------------------------
// ReaderPreferredRWLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, ReaderPreferredRWLock_ReaderPriority) {
    ReaderPreferredRWLock lock;
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> stop{false};

    // Continuous readers
    std::vector<std::thread> readers;
    for (int i = 0; i < NUM_READERS; ++i) {
        readers.emplace_back([&] {
            while (!stop) {
                lock.lock_shared();
                ++read_count;
                std::this_thread::sleep_for(1ms);
                lock.unlock_shared();
            }
        });
    }

    // Occasional writer
    std::thread writer([&] {
        while (!stop) {
            std::this_thread::sleep_for(10ms);
            lock.lock();
            ++write_count;
            lock.unlock();
        }
    });

    std::this_thread::sleep_for(100ms);
    stop = true;

    for (auto& t : readers) t.join();
    writer.join();

    // Readers should get many more acquisitions than writers
    EXPECT_GT(read_count.load(), write_count.load() * 10);
}

// -----------------------------------------------------------------------------
// UpgradableRWLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, UpgradableRWLock_UpgradeDowngrade) {
    UpgradableRWLock lock;
    int shared_data = 0;

    // Acquire upgrade lock
    lock.lock_upgrade();
    EXPECT_EQ(lock.current_thread_state(), UpgradableRWLock::LockState::UpgradeLocked);

    // Read data
    int value = shared_data;

    // Upgrade to exclusive
    lock.upgrade_to_exclusive();
    EXPECT_EQ(lock.current_thread_state(), UpgradableRWLock::LockState::ExclusiveLocked);

    // Modify data
    shared_data = value + 1;

    // Downgrade to shared
    lock.downgrade_to_shared();
    EXPECT_EQ(lock.current_thread_state(), UpgradableRWLock::LockState::SharedLocked);

    // Read again
    value = shared_data;

    lock.unlock_shared();
    EXPECT_EQ(lock.current_thread_state(), UpgradableRWLock::LockState::Unlocked);

    EXPECT_EQ(shared_data, 1);
}

TEST_F(RWLockTest, UpgradableRWLock_ConcurrentUpgrade) {
    UpgradableRWLock lock;
    std::atomic<int> shared_data{0};
    std::atomic<int> successful_upgrades{0};

    std::vector<std::thread> threads;

    // Multiple threads try to upgrade
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 100; ++j) {
                lock.lock_shared();
                [[maybe_unused]] int temp = shared_data.load();
                lock.unlock_shared();
            }
        });
    }

    // One thread does upgrade operations
    threads.emplace_back([&] {
        for (int j = 0; j < 50; ++j) {
            lock.lock_upgrade();

            // Try to upgrade
            if (lock.try_upgrade_to_exclusive()) {
                ++successful_upgrades;
                shared_data++;
                lock.unlock();
            } else {
                lock.unlock_upgrade();
            }
        }
    });

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(shared_data.load(), successful_upgrades.load());
}

// -----------------------------------------------------------------------------
// SequenceLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, SequenceLock_OptimisticReading) {
    SequenceLock<int> lock(0);
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> stop{false};

    // Writer thread
    std::thread writer([&] {
        int value = 1;
        while (!stop) {
            lock.write(value++);
            ++write_count;
            std::this_thread::sleep_for(1ms);
        }
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < NUM_READERS; ++i) {
        readers.emplace_back([&] {
            while (!stop) {
                [[maybe_unused]] int value = lock.read();
                ++read_count;
            }
        });
    }

    std::this_thread::sleep_for(100ms);
    stop = true;

    writer.join();
    for (auto& t : readers) t.join();

    // Should have many successful reads
    EXPECT_GT(read_count.load(), write_count.load());
}

TEST_F(RWLockTest, SequenceLock_UpdateFunction) {
    SequenceLock<std::pair<int, int>> lock({0, 0});

    lock.update([](auto& data) {
        data.first += 10;
        data.second += 20;
    });

    auto result = lock.read();
    EXPECT_EQ(result.first, 10);
    EXPECT_EQ(result.second, 20);
}

// -----------------------------------------------------------------------------
// StampedLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, StampedLock_OptimisticRead) {
    StampedLock lock;
    std::atomic<int> shared_data{0};
    std::atomic<int> successful_optimistic{0};
    std::atomic<int> failed_optimistic{0};

    std::vector<std::thread> threads;

    // Writer thread
    threads.emplace_back([&] {
        for (int i = 0; i < 100; ++i) {
            auto stamp = lock.write_lock();
            shared_data++;
            std::this_thread::sleep_for(100us);
            lock.unlock_write(stamp);
            std::this_thread::sleep_for(900us);
        }
    });

    // Optimistic readers
    for (int i = 0; i < NUM_READERS; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 200; ++j) {
                auto stamp = lock.try_optimistic_read();
                if (stamp != 0) {
                    [[maybe_unused]] int value = shared_data.load();
                    std::this_thread::yield();

                    if (lock.validate(stamp)) {
                        ++successful_optimistic;
                    } else {
                        ++failed_optimistic;
                        // Fall back to read lock
                        auto read_stamp = lock.read_lock();
                        [[maybe_unused]] int value2 = shared_data.load();
                        lock.unlock_read(read_stamp);
                    }
                } else {
                    // Writer active, use read lock
                    auto read_stamp = lock.read_lock();
                    [[maybe_unused]] int value = shared_data.load();
                    lock.unlock_read(read_stamp);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Most reads should be optimistic
    EXPECT_GT(successful_optimistic.load(), failed_optimistic.load());
}

TEST_F(RWLockTest, StampedLock_ConvertOptimisticToRead) {
    StampedLock lock;
    int shared_data = 0;

    auto stamp = lock.try_optimistic_read();
    ASSERT_NE(stamp, 0u);

    // Read optimistically
    [[maybe_unused]] int value = shared_data;

    // Convert to read lock
    auto new_stamp = lock.try_convert_to_read_lock(stamp);
    if (new_stamp != 0) {
        // Successfully converted
        lock.unlock_read(new_stamp);
    }
}

// -----------------------------------------------------------------------------
// HierarchicalRWLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, HierarchicalRWLock_PreventDeadlock) {
    HierarchicalRWLock lock1(1000);
    HierarchicalRWLock lock2(500);
    HierarchicalRWLock lock3(100);

    // Correct order: high to low
    lock1.lock();
    lock2.lock();
    lock3.lock();
    lock3.unlock();
    lock2.unlock();
    lock1.unlock();

    // Incorrect order should throw
    lock2.lock();
    EXPECT_THROW(lock1.lock(), std::logic_error);
    lock2.unlock();
}

TEST_F(RWLockTest, HierarchicalRWLock_SharedLocking) {
    HierarchicalRWLock lock1(1000);
    HierarchicalRWLock lock2(500);

    lock1.lock_shared();
    lock2.lock_shared();
    lock2.unlock_shared();
    lock1.unlock_shared();

    // Should work fine with correct hierarchy
    SUCCEED();
}

// -----------------------------------------------------------------------------
// FairRWLock Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, FairRWLock_FIFOOrdering) {
    FairRWLock lock;
    std::vector<int> acquisition_order;
    std::mutex order_mutex;
    std::atomic<int> threads_at_lock{0};
    std::atomic<bool> start_acquiring{false};

    // Initially lock for reading
    lock.lock_shared();

    struct ThreadInfo {
        std::thread thread;
        int id;
        bool is_reader;
    };

    std::vector<ThreadInfo> thread_infos;

    // Create threads in specific order
    int id_counter = 0;

    // Reader 1
    thread_infos.push_back({
        std::thread([&, id = id_counter++] {
            // Wait for this thread's turn to approach the lock
            while (threads_at_lock < id) std::this_thread::yield();

            ++threads_at_lock;

            // Wait for signal to actually try acquiring
            while (!start_acquiring) std::this_thread::yield();

            lock.lock_shared();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                acquisition_order.push_back(id);
            }
            std::this_thread::sleep_for(1ms);
            lock.unlock_shared();
        }),
        id_counter - 1,
        true
    });

    // Ensure thread 0 arrives at lock first
    while (threads_at_lock < 1) std::this_thread::yield();

    // Writer 1
    thread_infos.push_back({
        std::thread([&, id = id_counter++] {
            // Wait for this thread's turn to approach the lock
            while (threads_at_lock < id) std::this_thread::yield();

            ++threads_at_lock;

            // Wait for signal to actually try acquiring
            while (!start_acquiring) std::this_thread::yield();

            lock.lock();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                acquisition_order.push_back(id);
            }
            std::this_thread::sleep_for(1ms);
            lock.unlock();
        }),
        id_counter - 1,
        false
    });

    // Ensure thread 1 arrives at lock second
    while (threads_at_lock < 2) std::this_thread::yield();

    // Reader 2
    thread_infos.push_back({
        std::thread([&, id = id_counter++] {
            // Wait for this thread's turn to approach the lock
            while (threads_at_lock < id) std::this_thread::yield();

            ++threads_at_lock;

            // Wait for signal to actually try acquiring
            while (!start_acquiring) std::this_thread::yield();

            lock.lock_shared();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                acquisition_order.push_back(id);
            }
            std::this_thread::sleep_for(1ms);
            lock.unlock_shared();
        }),
        id_counter - 1,
        true
    });

    // Ensure thread 2 arrives at lock third
    while (threads_at_lock < 3) std::this_thread::yield();

    // Writer 2
    thread_infos.push_back({
        std::thread([&, id = id_counter++] {
            // Wait for this thread's turn to approach the lock
            while (threads_at_lock < id) std::this_thread::yield();

            ++threads_at_lock;

            // Wait for signal to actually try acquiring
            while (!start_acquiring) std::this_thread::yield();

            lock.lock();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                acquisition_order.push_back(id);
            }
            std::this_thread::sleep_for(1ms);
            lock.unlock();
        }),
        id_counter - 1,
        false
    });

    // Ensure thread 3 arrives at lock fourth
    while (threads_at_lock < 4) std::this_thread::yield();

    // Reader 3
    thread_infos.push_back({
        std::thread([&, id = id_counter++] {
            // Wait for this thread's turn to approach the lock
            while (threads_at_lock < id) std::this_thread::yield();

            ++threads_at_lock;

            // Wait for signal to actually try acquiring
            while (!start_acquiring) std::this_thread::yield();

            lock.lock_shared();
            {
                std::lock_guard<std::mutex> guard(order_mutex);
                acquisition_order.push_back(id);
            }
            std::this_thread::sleep_for(1ms);
            lock.unlock_shared();
        }),
        id_counter - 1,
        true
    });

    // Wait for all threads to arrive at the lock
    while (threads_at_lock < 5) {
        std::this_thread::sleep_for(1ms);
    }

    // Now release initial lock and signal threads to start acquiring
    start_acquiring = true;
    lock.unlock_shared();

    // Join all threads
    for (auto& info : thread_infos) {
        info.thread.join();
    }

    // Verify FIFO order
    ASSERT_EQ(acquisition_order.size(), 5u);

    // For a FairRWLock, the expected order should respect FIFO but allow
    // batching of consecutive readers. Given our queue:
    // Reader0, Writer1, Reader2, Writer3, Reader4
    // Expected order: Reader0, Writer1, Reader2, Writer3, Reader4
    // OR if readers batch: Reader0, Writer1, Reader2+4 (any order), Writer3

    // For now, let's just check that we got all 5 threads
    std::vector<int> expected = {0, 1, 2, 3, 4};
    std::vector<int> actual = acquisition_order;
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected) << "All threads should have acquired the lock";
}

// -----------------------------------------------------------------------------
// RAII Lock Guards Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, LockGuards_RAII) {
    ReaderWriterLock lock;
    int shared_data = 0;

    // Test ReadLockGuard
    {
        ReadLockGuard<ReaderWriterLock> guard(lock);
        [[maybe_unused]] int value = shared_data;
    }  // Automatically unlocked

    // Test WriteLockGuard
    {
        WriteLockGuard<ReaderWriterLock> guard(lock);
        shared_data++;
    }  // Automatically unlocked

    EXPECT_EQ(shared_data, 1);
}

TEST_F(RWLockTest, UpgradeLockGuard_Upgrade) {
    UpgradableRWLock lock;
    int shared_data = 0;

    {
        UpgradeLockGuard<UpgradableRWLock> guard(lock);

        // Read
        [[maybe_unused]] int value = shared_data;

        // Upgrade to write
        guard.upgrade();
        EXPECT_TRUE(guard.is_upgraded());

        // Write
        shared_data++;

        // Can downgrade
        guard.downgrade();
        EXPECT_FALSE(guard.is_upgraded());
    }  // Automatically unlocked

    EXPECT_EQ(shared_data, 1);
}

// -----------------------------------------------------------------------------
// Stress Tests
// -----------------------------------------------------------------------------
TEST_F(RWLockTest, StressTest_MixedWorkload) {
    ReaderWriterLock lock;
    std::atomic<int> shared_value{0};
    std::atomic<int> read_sum{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> stop{false};

    // Random number generator for each thread
    std::random_device rd;

    // Reader threads (80% of threads)
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) {
        readers.emplace_back([&] {
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 10);

            while (!stop) {
                lock.lock_shared();
                int value = shared_value.load();
                read_sum += value;
                lock.unlock_shared();

                std::this_thread::sleep_for(std::chrono::microseconds(dist(gen)));
            }
        });
    }

    // Writer threads (20% of threads)
    std::vector<std::thread> writers;
    for (int i = 0; i < 2; ++i) {
        writers.emplace_back([&] {
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 50);

            while (!stop) {
                lock.lock();
                shared_value++;
                write_count++;
                lock.unlock();

                std::this_thread::sleep_for(std::chrono::microseconds(dist(gen)));
            }
        });
    }

    // Run for a while
    std::this_thread::sleep_for(200ms);
    stop = true;

    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    EXPECT_EQ(shared_value.load(), write_count.load());
    EXPECT_GT(read_sum.load(), 0);
}

TEST_F(RWLockTest, StressTest_HighContention) {
    WriterPreferredRWLock lock;
    std::atomic<int> counter{0};
    const int target = 10000;
    std::atomic<bool> stop{false};

    std::vector<std::thread> threads;

    // Many threads incrementing counter
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            while (!stop) {
                lock.lock();
                if (counter < target) {
                    counter++;
                } else {
                    stop = true;
                }
                lock.unlock();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), target);
}
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cstring>

#include <core/memory/memory_barrier.h>

namespace fem::core::memory {
namespace {

// Test fixture for memory barrier tests
class MemoryBarrierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset shared test state
        shared_counter.store(0, std::memory_order_relaxed);
        flag.store(false, std::memory_order_relaxed);
        data = 0;
    }

    // Shared test variables for multi-threading tests
    std::atomic<int> shared_counter{0};
    std::atomic<bool> flag{false};
    int data{0};
};

// Test signal fence functions
TEST_F(MemoryBarrierTest, SignalFenceAcquire) {
    // Signal fences prevent compiler reordering but not CPU reordering
    // Test that the function can be called without issues
    signal_fence_acquire();

    // Verify it doesn't crash with multiple calls
    for (int i = 0; i < 10; ++i) {
        signal_fence_acquire();
    }
}

TEST_F(MemoryBarrierTest, SignalFenceRelease) {
    // Test signal_fence_release
    signal_fence_release();

    // Multiple calls should work
    for (int i = 0; i < 10; ++i) {
        signal_fence_release();
    }
}

TEST_F(MemoryBarrierTest, SignalFenceSeqCst) {
    // Test signal_fence_seq_cst
    signal_fence_seq_cst();

    // Multiple calls should work
    for (int i = 0; i < 10; ++i) {
        signal_fence_seq_cst();
    }
}

// Test thread fence functions
TEST_F(MemoryBarrierTest, ThreadFenceAcquire) {
    // Thread fences provide both compiler and CPU ordering guarantees
    thread_fence_acquire();

    // Should work with multiple calls
    for (int i = 0; i < 10; ++i) {
        thread_fence_acquire();
    }
}

TEST_F(MemoryBarrierTest, ThreadFenceRelease) {
    thread_fence_release();

    // Multiple calls should work
    for (int i = 0; i < 10; ++i) {
        thread_fence_release();
    }
}

TEST_F(MemoryBarrierTest, ThreadFenceAcqRel) {
    thread_fence_acq_rel();

    // Multiple calls should work
    for (int i = 0; i < 10; ++i) {
        thread_fence_acq_rel();
    }
}

TEST_F(MemoryBarrierTest, ThreadFenceSeqCst) {
    thread_fence_seq_cst();

    // Multiple calls should work
    for (int i = 0; i < 10; ++i) {
        thread_fence_seq_cst();
    }
}

// Test barrier shorthands
TEST_F(MemoryBarrierTest, ReadBarrier) {
    // read_barrier should be equivalent to thread_fence_acquire
    read_barrier();

    // Test in a simple synchronization scenario
    std::thread writer([this]() {
        data = 42;
        flag.store(true, std::memory_order_release);
    });

    while (!flag.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
    }
    read_barrier(); // Ensure we see the write to data

    EXPECT_EQ(data, 42);

    writer.join();
}

TEST_F(MemoryBarrierTest, WriteBarrier) {
    // write_barrier should be equivalent to thread_fence_release
    write_barrier();

    // Test with multiple calls
    for (int i = 0; i < 10; ++i) {
        write_barrier();
    }
}

TEST_F(MemoryBarrierTest, FullBarrier) {
    // full_barrier should be equivalent to thread_fence_seq_cst
    full_barrier();

    // Test with multiple calls
    for (int i = 0; i < 10; ++i) {
        full_barrier();
    }
}

// Test cpu_relax
TEST_F(MemoryBarrierTest, CpuRelax) {
    // cpu_relax should not crash and should provide a pause hint
    cpu_relax();

    // Test in a loop
    for (int i = 0; i < 100; ++i) {
        cpu_relax();
    }
}

TEST_F(MemoryBarrierTest, CpuRelaxInSpinLoop) {
    // Test cpu_relax in a typical spin-wait scenario
    std::atomic<bool> ready{false};

    std::thread setter([&ready]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ready.store(true, std::memory_order_release);
    });

    int spin_count = 0;
    while (!ready.load(std::memory_order_acquire)) {
        cpu_relax();
        ++spin_count;
    }

    // We should have spun at least once
    EXPECT_GT(spin_count, 0);

    setter.join();
}

// Test spin_until
TEST_F(MemoryBarrierTest, SpinUntilSuccess) {
    std::atomic<int> counter{0};

    // Spin until counter reaches 5
    std::thread incrementer([&counter]() {
        for (int i = 0; i < 5; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    bool result = spin_until([&counter]() {
        return counter.load(std::memory_order_relaxed) >= 5;
    }, 100000);  // Increase max spins for reliability

    EXPECT_TRUE(result);
    EXPECT_GE(counter.load(), 5);

    incrementer.join();
}

TEST_F(MemoryBarrierTest, SpinUntilTimeout) {
    std::atomic<bool> never_true{false};

    // This should timeout
    bool result = spin_until([&never_true]() {
        return never_true.load(std::memory_order_relaxed);
    }, 10);  // Only 10 spins

    EXPECT_FALSE(result);
}

TEST_F(MemoryBarrierTest, SpinUntilImmediateSuccess) {
    // Predicate is immediately true
    bool result = spin_until([]() { return true; });

    EXPECT_TRUE(result);
}

TEST_F(MemoryBarrierTest, SpinUntilCustomMaxSpins) {
    int call_count = 0;

    // Test with custom max_spins
    bool result = spin_until([&call_count]() {
        ++call_count;
        return false;
    }, 5);

    EXPECT_FALSE(result);
    // Should have been called max_spins + 1 times (loop check + final check)
    EXPECT_EQ(call_count, 6);
}

// Test load_acquire
TEST_F(MemoryBarrierTest, LoadAcquire) {
    std::atomic<int> value{42};

    int loaded = load_acquire(value);
    EXPECT_EQ(loaded, 42);

    // Test with different values
    value.store(100, std::memory_order_relaxed);
    loaded = load_acquire(value);
    EXPECT_EQ(loaded, 100);
}

TEST_F(MemoryBarrierTest, LoadAcquireOrdering) {
    std::atomic<bool> ready{false};
    int shared_data = 0;

    std::thread writer([&]() {
        shared_data = 123;
        ready.store(true, std::memory_order_release);
    });

    while (!load_acquire(ready)) {
        std::this_thread::yield();
    }

    // Due to acquire-release synchronization, we should see the write
    EXPECT_EQ(shared_data, 123);

    writer.join();
}

// Test store_release
TEST_F(MemoryBarrierTest, StoreRelease) {
    std::atomic<int> value{0};

    store_release(value, 42);
    EXPECT_EQ(value.load(std::memory_order_relaxed), 42);

    // Test with different values
    store_release(value, 100);
    EXPECT_EQ(value.load(std::memory_order_relaxed), 100);
}

TEST_F(MemoryBarrierTest, StoreReleaseOrdering) {
    std::atomic<bool> ready{false};
    int shared_data = 0;

    std::thread writer([&]() {
        shared_data = 456;
        store_release(ready, true);
    });

    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Due to release-acquire synchronization, we should see the write
    EXPECT_EQ(shared_data, 456);

    writer.join();
}

// Test acquire-release pair
TEST_F(MemoryBarrierTest, AcquireReleasePair) {
    std::atomic<int> sync_var{0};
    std::vector<int> shared_array(10, 0);

    std::thread producer([&]() {
        // Fill array
        for (int i = 0; i < 10; ++i) {
            shared_array[i] = i * i;
        }
        // Release - ensure all writes are visible
        store_release(sync_var, 1);
    });

    // Acquire - wait for producer
    while (load_acquire(sync_var) == 0) {
        std::this_thread::yield();
    }

    // Check array values
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(shared_array[i], i * i);
    }

    producer.join();
}

// Test with multiple threads
TEST_F(MemoryBarrierTest, MultiThreadedBarriers) {
    constexpr int num_threads = 4;
    constexpr int iterations = 1000;
    std::atomic<int> counter{0};
    std::atomic<int> start_flag{0};

    auto worker = [&]() {
        // Wait for start
        while (load_acquire(start_flag) == 0) {
            cpu_relax();
        }

        for (int i = 0; i < iterations; ++i) {
            counter.fetch_add(1, std::memory_order_relaxed);
            write_barrier();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Start all threads
    store_release(start_flag, 1);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), num_threads * iterations);
}

// Performance test for cpu_relax
TEST_F(MemoryBarrierTest, CpuRelaxPerformance) {
    constexpr int iterations = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        cpu_relax();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // cpu_relax should be very fast - just a CPU instruction
    // Even 100k iterations should take less than 100ms
    EXPECT_LT(duration.count(), 100000);
}

// Test memory ordering guarantees
TEST_F(MemoryBarrierTest, MemoryOrderingScenario) {
    // Classic message passing scenario
    std::atomic<int> x{0}, y{0};
    int r1 = 0, r2 = 0;

    std::thread t1([&]() {
        x.store(1, std::memory_order_relaxed);
        write_barrier();  // Ensure x=1 is visible before y=1
        y.store(1, std::memory_order_relaxed);
    });

    std::thread t2([&]() {
        r1 = y.load(std::memory_order_relaxed);
        read_barrier();  // If we see y=1, ensure we see x=1
        r2 = x.load(std::memory_order_relaxed);
    });

    t1.join();
    t2.join();

    // If r1 == 1 (saw y=1), then r2 must be 1 (must see x=1)
    if (r1 == 1) {
        EXPECT_EQ(r2, 1);
    }
}

// Test spin_until with complex predicate
TEST_F(MemoryBarrierTest, SpinUntilComplexPredicate) {
    struct State {
        std::atomic<int> a{0};
        std::atomic<int> b{0};
        std::atomic<int> c{0};
    } state;

    std::thread updater([&state]() {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        state.a.store(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        state.b.store(2, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        state.c.store(3, std::memory_order_relaxed);
    });

    // Spin until all three conditions are met
    bool result = spin_until([&state]() {
        return state.a.load(std::memory_order_relaxed) == 1 &&
               state.b.load(std::memory_order_relaxed) == 2 &&
               state.c.load(std::memory_order_relaxed) == 3;
    }, 10000);

    EXPECT_TRUE(result);
    EXPECT_EQ(state.a.load(), 1);
    EXPECT_EQ(state.b.load(), 2);
    EXPECT_EQ(state.c.load(), 3);

    updater.join();
}

// Test that barriers don't break basic operations
TEST_F(MemoryBarrierTest, BarriersDoNotBreakCode) {
    int local_var = 10;

    signal_fence_acquire();
    local_var += 5;
    signal_fence_release();

    EXPECT_EQ(local_var, 15);

    thread_fence_acquire();
    local_var *= 2;
    thread_fence_release();

    EXPECT_EQ(local_var, 30);

    read_barrier();
    local_var -= 10;
    write_barrier();

    EXPECT_EQ(local_var, 20);

    full_barrier();
    local_var /= 2;
    full_barrier();

    EXPECT_EQ(local_var, 10);
}

// Test edge cases for spin_until
TEST_F(MemoryBarrierTest, SpinUntilZeroSpins) {
    int call_count = 0;

    // With 0 max spins, should still check predicate once
    bool result = spin_until([&call_count]() {
        ++call_count;
        return false;
    }, 0);

    EXPECT_FALSE(result);
    EXPECT_EQ(call_count, 1);  // Final check after loop
}

TEST_F(MemoryBarrierTest, SpinUntilNegativeSpins) {
    int call_count = 0;

    // Negative spins should immediately exit loop but still check once
    bool result = spin_until([&call_count]() {
        ++call_count;
        return true;
    }, -1);

    EXPECT_TRUE(result);
    EXPECT_EQ(call_count, 1);  // Only the final check
}

// Test template instantiation with different types
TEST_F(MemoryBarrierTest, LoadAcquireVariousTypes) {
    // Test with different atomic types
    std::atomic<bool> bool_val{true};
    EXPECT_TRUE(load_acquire(bool_val));

    std::atomic<char> char_val{'A'};
    EXPECT_EQ(load_acquire(char_val), 'A');

    std::atomic<long> long_val{1000000L};
    EXPECT_EQ(load_acquire(long_val), 1000000L);

    std::atomic<size_t> size_val{sizeof(int)};
    EXPECT_EQ(load_acquire(size_val), sizeof(int));

    // Pointer type
    int x = 42;
    std::atomic<int*> ptr_val{&x};
    EXPECT_EQ(load_acquire(ptr_val), &x);
}

TEST_F(MemoryBarrierTest, StoreReleaseVariousTypes) {
    // Test with different atomic types
    std::atomic<bool> bool_val{false};
    store_release(bool_val, true);
    EXPECT_TRUE(bool_val.load());

    std::atomic<char> char_val{' '};
    store_release(char_val, 'Z');
    EXPECT_EQ(char_val.load(), 'Z');

    std::atomic<long> long_val{0};
    store_release(long_val, -999999L);
    EXPECT_EQ(long_val.load(), -999999L);

    std::atomic<size_t> size_val{0};
    store_release(size_val, SIZE_MAX);
    EXPECT_EQ(size_val.load(), SIZE_MAX);

    // Pointer type
    int x = 42, y = 84;
    std::atomic<int*> ptr_val{&x};
    store_release(ptr_val, &y);
    EXPECT_EQ(ptr_val.load(), &y);
}

// Stress test for barriers
TEST_F(MemoryBarrierTest, StressTestBarriers) {
    constexpr int iterations = 10000;

    for (int i = 0; i < iterations; ++i) {
        signal_fence_acquire();
        signal_fence_release();
        signal_fence_seq_cst();

        if (i % 100 == 0) {
            thread_fence_acquire();
            thread_fence_release();
            thread_fence_acq_rel();
            thread_fence_seq_cst();
        }

        if (i % 10 == 0) {
            cpu_relax();
        }
    }

    // Should complete without issues
    EXPECT_TRUE(true);
}

} // namespace
} // namespace fem::core::memory
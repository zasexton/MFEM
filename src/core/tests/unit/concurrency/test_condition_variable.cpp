// File: core/tests/unit/concurrency/test_condition_variable.cpp
// Description: Unit tests for enhanced condition variable implementations

#include <gtest/gtest.h>
#include "core/concurrency/condition_variable.h"
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <random>

using namespace core::concurrency;
using namespace std::chrono_literals;

class ConditionVariableTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test ConditionVariableAny basic functionality
TEST_F(ConditionVariableTest, ConditionVariableAny_BasicNotify) {
    ConditionVariableAny cv;
    std::mutex mutex;
    bool ready = false;
    bool processed = false;

    std::thread worker([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&ready] { return ready; });
        processed = true;
    });

    // Give the worker time to start waiting
    std::this_thread::sleep_for(50ms);

    {
        std::lock_guard<std::mutex> lock(mutex);
        ready = true;
    }
    cv.notify_one();

    worker.join();

    EXPECT_TRUE(processed);
    EXPECT_EQ(cv.get_notify_one_count(), 1);
    EXPECT_GT(cv.get_wake_count(), 0);
}

// Test ConditionVariableAny with timeout
TEST_F(ConditionVariableTest, ConditionVariableAny_Timeout) {
    ConditionVariableAny cv;
    std::mutex mutex;
    bool ready = false;

    std::thread worker([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, 100ms, [&ready] { return ready; });
    });

    worker.join();

    EXPECT_FALSE(ready);
    EXPECT_EQ(cv.get_timeout_count(), 1);
}

// Test ConditionVariableAny notify_all
TEST_F(ConditionVariableTest, ConditionVariableAny_NotifyAll) {
    ConditionVariableAny cv;
    std::mutex mutex;
    bool ready = false;
    std::atomic<int> processed_count(0);
    const int num_threads = 5;

    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&ready] { return ready; });
            processed_count++;
        });
    }

    // Give workers time to start waiting
    std::this_thread::sleep_for(50ms);
    EXPECT_EQ(cv.get_waiters_count(), num_threads);

    {
        std::lock_guard<std::mutex> lock(mutex);
        ready = true;
    }
    cv.notify_all();

    for (auto& t : workers) {
        t.join();
    }

    EXPECT_EQ(processed_count.load(), num_threads);
    EXPECT_EQ(cv.get_notify_all_count(), 1);
}

// Test EventConditionVariable
TEST_F(ConditionVariableTest, EventConditionVariable_Signal) {
    EventConditionVariable cv;
    std::atomic<bool> processed(false);

    std::thread worker([&]() {
        cv.wait();
        processed = true;
    });

    // Give worker time to start waiting
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(processed.load());

    cv.signal();
    worker.join();

    EXPECT_TRUE(processed.load());
    EXPECT_TRUE(cv.is_signaled());
}

// Test EventConditionVariable auto-reset
TEST_F(ConditionVariableTest, EventConditionVariable_AutoReset) {
    EventConditionVariable cv;
    cv.set_auto_reset(true);

    std::atomic<int> wait_count(0);

    std::thread worker1([&]() {
        cv.wait();
        wait_count++;
    });

    std::this_thread::sleep_for(50ms);
    cv.signal();
    worker1.join();

    EXPECT_EQ(wait_count.load(), 1);
    EXPECT_FALSE(cv.is_signaled());  // Should be reset after wait

    std::thread worker2([&]() {
        bool result = cv.wait_for(100ms);
        if (result) wait_count++;
    });

    std::this_thread::sleep_for(50ms);
    cv.signal();
    worker2.join();

    EXPECT_EQ(wait_count.load(), 2);
}

// Test EventConditionVariable timeout
TEST_F(ConditionVariableTest, EventConditionVariable_Timeout) {
    EventConditionVariable cv;

    auto start = std::chrono::steady_clock::now();
    bool result = cv.wait_for(100ms);
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    EXPECT_GE(duration, 95ms);
    EXPECT_LE(duration, 150ms);
}

// Test BroadcastConditionVariable
TEST_F(ConditionVariableTest, BroadcastConditionVariable_NotifyAll) {
    BroadcastConditionVariable cv;
    std::mutex mutex;
    bool ready = false;
    std::atomic<int> processed_count(0);
    const int num_threads = 10;

    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&ready] { return ready; });
            processed_count++;
        });
    }

    // Give workers time to start waiting
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(cv.get_waiters_count(), num_threads);

    {
        std::lock_guard<std::mutex> lock(mutex);
        ready = true;
    }
    cv.notify_all();

    for (auto& t : workers) {
        t.join();
    }

    EXPECT_EQ(processed_count.load(), num_threads);
}

// Test BroadcastConditionVariable timeout
TEST_F(ConditionVariableTest, BroadcastConditionVariable_Timeout) {
    BroadcastConditionVariable cv;
    std::mutex mutex;
    bool ready = false;

    std::unique_lock<std::mutex> lock(mutex);
    auto status = cv.wait_for(lock, 100ms);

    EXPECT_EQ(status, std::cv_status::timeout);
    EXPECT_FALSE(ready);
}

// Test PriorityConditionVariable
TEST_F(ConditionVariableTest, PriorityConditionVariable_Priority) {
    PriorityConditionVariable<int> cv;
    std::mutex mutex;
    std::vector<int> wake_order;
    std::mutex order_mutex;
    std::atomic<int> ready_count(0);

    auto create_waiter = [&](int id, int priority) {
        return std::thread([&, id, priority]() {
            std::unique_lock<std::mutex> lock(mutex);
            ready_count++;
            cv.wait(lock, priority);

            std::lock_guard<std::mutex> order_lock(order_mutex);
            wake_order.push_back(id);
        });
    };

    // Create threads with different priorities
    // Higher priority value = wake first
    std::vector<std::thread> threads;
    threads.push_back(create_waiter(0, 1));  // Low priority

    while (ready_count < 1) std::this_thread::sleep_for(10ms);

    threads.push_back(create_waiter(1, 3));  // High priority

    while (ready_count < 2) std::this_thread::sleep_for(10ms);

    threads.push_back(create_waiter(2, 2));  // Medium priority

    while (ready_count < 3) std::this_thread::sleep_for(10ms);

    // Wake threads one by one
    for (int i = 0; i < 3; ++i) {
        std::this_thread::sleep_for(50ms);
        cv.notify_one();
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(wake_order.size(), 3);
    // Should wake in priority order: 1 (priority 3), 2 (priority 2), 0 (priority 1)
    EXPECT_EQ(wake_order[0], 1);
    EXPECT_EQ(wake_order[1], 2);
    EXPECT_EQ(wake_order[2], 0);
}

// Test PriorityConditionVariable notify_all
TEST_F(ConditionVariableTest, PriorityConditionVariable_NotifyAll) {
    PriorityConditionVariable<int> cv;
    std::mutex mutex;
    std::atomic<int> woken_count(0);
    const int num_threads = 5;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, i);  // Different priorities
            woken_count++;
        });
    }

    // Give threads time to start waiting
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(cv.get_waiters_count(), num_threads);

    cv.notify_all();

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(woken_count.load(), num_threads);
}

// Test BarrierConditionVariable
TEST_F(ConditionVariableTest, BarrierConditionVariable_Basic) {
    const int num_threads = 5;
    BarrierConditionVariable cv(num_threads);
    std::mutex mutex;
    std::atomic<int> reached_count(0);
    std::atomic<int> trigger_count(0);

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            reached_count++;
            bool is_trigger = cv.wait(lock);
            if (is_trigger) {
                trigger_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(reached_count.load(), num_threads);
    EXPECT_EQ(trigger_count.load(), 1);  // Only one thread should be the trigger
}

// Test BarrierConditionVariable timeout
TEST_F(ConditionVariableTest, BarrierConditionVariable_Timeout) {
    const int num_threads = 3;
    BarrierConditionVariable cv(num_threads);
    std::mutex mutex;
    std::atomic<int> timeout_count(0);

    std::vector<std::thread> threads;

    // Only start 2 threads, so barrier won't complete
    for (int i = 0; i < num_threads - 1; ++i) {
        threads.emplace_back([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            auto result = cv.wait_for(lock, 100ms);
            if (!result.has_value()) {
                timeout_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(timeout_count.load(), num_threads - 1);
}

// Test BarrierConditionVariable reset
TEST_F(ConditionVariableTest, BarrierConditionVariable_Reset) {
    const int num_threads = 2;
    BarrierConditionVariable cv(num_threads);
    std::mutex mutex;
    std::atomic<int> cycle_count(0);

    auto barrier_cycle = [&]() {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock);
                cycle_count++;
            });
        }
        for (auto& t : threads) {
            t.join();
        }
    };

    // First cycle
    barrier_cycle();
    EXPECT_EQ(cycle_count.load(), num_threads);

    // Second cycle
    barrier_cycle();
    EXPECT_EQ(cycle_count.load(), num_threads * 2);
}

// Test FutureConditionVariable with value
TEST_F(ConditionVariableTest, FutureConditionVariable_Value) {
    FutureConditionVariable<int> cv;
    std::mutex mutex;
    int result = 0;

    std::thread waiter([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        result = cv.wait(lock);
    });

    // Give waiter time to start
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(cv.is_ready());

    cv.set_value(42);
    EXPECT_TRUE(cv.is_ready());

    waiter.join();
    EXPECT_EQ(result, 42);
}

// Test FutureConditionVariable with exception
TEST_F(ConditionVariableTest, FutureConditionVariable_Exception) {
    FutureConditionVariable<int> cv;
    std::mutex mutex;
    bool caught_exception = false;

    std::thread waiter([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        try {
            cv.wait(lock);
        } catch (const std::runtime_error& e) {
            caught_exception = true;
        }
    });

    // Give waiter time to start
    std::this_thread::sleep_for(50ms);

    try {
        throw std::runtime_error("test error");
    } catch (...) {
        cv.set_exception(std::current_exception());
    }

    waiter.join();
    EXPECT_TRUE(caught_exception);
}

// Test FutureConditionVariable timeout
TEST_F(ConditionVariableTest, FutureConditionVariable_Timeout) {
    FutureConditionVariable<int> cv;
    std::mutex mutex;

    std::unique_lock<std::mutex> lock(mutex);
    auto result = cv.wait_for(lock, 100ms);

    EXPECT_FALSE(result.has_value());
}

// Test FutureConditionVariable<void>
TEST_F(ConditionVariableTest, FutureConditionVariable_Void) {
    FutureConditionVariable<void> cv;
    std::mutex mutex;
    bool completed = false;

    std::thread waiter([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);
        completed = true;
    });

    // Give waiter time to start
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(cv.is_ready());

    cv.set_value();
    EXPECT_TRUE(cv.is_ready());

    waiter.join();
    EXPECT_TRUE(completed);
}

// Test CountdownConditionVariable
TEST_F(ConditionVariableTest, CountdownConditionVariable_Basic) {
    const size_t count = 3;
    CountdownConditionVariable cv(count);
    std::mutex mutex;
    std::atomic<bool> completed(false);

    std::thread waiter([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);
        completed = true;
    });

    // Give waiter time to start
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(completed.load());
    EXPECT_EQ(cv.get_count(), count);

    // Count down
    cv.count_down();
    EXPECT_EQ(cv.get_count(), 2);
    EXPECT_FALSE(completed.load());

    cv.count_down();
    EXPECT_EQ(cv.get_count(), 1);
    EXPECT_FALSE(completed.load());

    cv.count_down();
    EXPECT_EQ(cv.get_count(), 0);

    waiter.join();
    EXPECT_TRUE(completed.load());
}

// Test CountdownConditionVariable batch countdown
TEST_F(ConditionVariableTest, CountdownConditionVariable_BatchCountdown) {
    const size_t count = 10;
    CountdownConditionVariable cv(count);
    std::mutex mutex;
    std::atomic<bool> completed(false);

    std::thread waiter([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);
        completed = true;
    });

    // Give waiter time to start
    std::this_thread::sleep_for(50ms);

    // Count down multiple at once
    cv.count_down(5);
    EXPECT_EQ(cv.get_count(), 5);
    EXPECT_FALSE(completed.load());

    cv.count_down(5);
    EXPECT_EQ(cv.get_count(), 0);

    waiter.join();
    EXPECT_TRUE(completed.load());
}

// Test CountdownConditionVariable reset
TEST_F(ConditionVariableTest, CountdownConditionVariable_Reset) {
    const size_t count = 2;
    CountdownConditionVariable cv(count);

    cv.count_down();
    EXPECT_EQ(cv.get_count(), 1);

    cv.reset();
    EXPECT_EQ(cv.get_count(), count);

    cv.count_down();
    cv.count_down();
    EXPECT_EQ(cv.get_count(), 0);
}

// Test CountdownConditionVariable timeout
TEST_F(ConditionVariableTest, CountdownConditionVariable_Timeout) {
    CountdownConditionVariable cv(2);
    std::mutex mutex;

    std::unique_lock<std::mutex> lock(mutex);
    auto start = std::chrono::steady_clock::now();
    bool result = cv.wait_for(lock, 100ms);
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    EXPECT_GE(duration, 95ms);
    EXPECT_LE(duration, 150ms);
    EXPECT_EQ(cv.get_count(), 2);  // Count unchanged
}

// Stress test with multiple condition variables
TEST_F(ConditionVariableTest, StressTest_MultipleCVs) {
    const int num_iterations = 10;
    const int num_threads = 4;

    ConditionVariableAny cv_any;
    EventConditionVariable cv_event;
    BroadcastConditionVariable cv_broadcast;
    std::mutex mutex;
    std::atomic<int> counter(0);

    std::vector<std::thread> threads;

    // Test different CV types concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < num_iterations; ++j) {
                // Each thread uses a different CV type
                int cv_type = i % 3;

                if (cv_type == 0) {
                    // ConditionVariableAny - producer/consumer pattern
                    if (i % 2 == 0) {
                        std::unique_lock<std::mutex> lock(mutex);
                        counter++;
                        cv_any.notify_one();
                    } else {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv_any.wait_for(lock, 10ms, [&counter] {
                            return counter.load() > 0;
                        });
                        if (counter > 0) counter--;
                    }
                } else if (cv_type == 1) {
                    // EventConditionVariable - signal pattern
                    if (i % 2 == 0) {
                        cv_event.signal();
                        counter++;
                    } else {
                        if (cv_event.wait_for(10ms)) {
                            counter++;
                            cv_event.reset();
                        }
                    }
                } else {
                    // BroadcastConditionVariable - broadcast pattern
                    if (i % 2 == 0) {
                        cv_broadcast.notify_all();
                        counter++;
                    } else {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv_broadcast.wait_for(lock, 10ms);
                        counter++;
                    }
                }

                std::this_thread::yield();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(counter.load(), 0);
}
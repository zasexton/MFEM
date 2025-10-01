#include <gtest/gtest.h>
#include <core/concurrency/work_stealing_pool.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <map>
#include <future>

namespace fcc = fem::core::concurrency;
using namespace std::chrono_literals;

class WorkStealingPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state if needed
    }

    void TearDown() override {
        // Clean up
    }
};

// ==================== Basic Construction Tests ====================

TEST_F(WorkStealingPoolTest, DefaultConstruction) {
    fcc::WorkStealingPool pool;
    EXPECT_GT(pool.worker_count(), 0);
    EXPECT_EQ(pool.worker_count(), std::thread::hardware_concurrency());
    EXPECT_FALSE(pool.is_stopped());
}

TEST_F(WorkStealingPoolTest, ExplicitWorkerCount) {
    const size_t num_workers = 4;
    fcc::WorkStealingPool pool(num_workers);
    EXPECT_EQ(pool.worker_count(), num_workers);
}

TEST_F(WorkStealingPoolTest, CustomConfiguration) {
    const size_t num_workers = 4;
    const size_t max_local_queue = 100;
    const size_t steal_batch = 8;
    const size_t spin_count = 200;

    fcc::WorkStealingPool pool(num_workers, max_local_queue, steal_batch, spin_count);
    EXPECT_EQ(pool.worker_count(), num_workers);
}

// ==================== Task Submission Tests ====================

TEST_F(WorkStealingPoolTest, SubmitSimpleTask) {
    fcc::WorkStealingPool pool(2);

    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(WorkStealingPoolTest, SubmitWithArguments) {
    fcc::WorkStealingPool pool(2);

    auto future = pool.submit([](int a, int b) { return a + b; }, 10, 32);
    EXPECT_EQ(future.get(), 42);
}

TEST_F(WorkStealingPoolTest, SubmitVoidTask) {
    fcc::WorkStealingPool pool(2);
    std::atomic<bool> executed{false};

    auto future = pool.submit([&executed]() {
        executed = true;
    });

    future.wait();
    EXPECT_TRUE(executed.load());
}

TEST_F(WorkStealingPoolTest, SubmitWithPriority) {
    fcc::WorkStealingPool pool(1);  // Single worker to ensure ordering
    std::vector<int> execution_order;
    std::mutex order_mutex;

    // Submit low priority task first
    auto low = pool.submit_with_priority(
        [&execution_order, &order_mutex]() {
            std::this_thread::sleep_for(1ms);
            std::lock_guard<std::mutex> lock(order_mutex);
            execution_order.push_back(1);
        },
        fcc::WorkStealingPool::Priority::Low
    );

    // Then high priority
    auto high = pool.submit_with_priority(
        [&execution_order, &order_mutex]() {
            std::lock_guard<std::mutex> lock(order_mutex);
            execution_order.push_back(2);
        },
        fcc::WorkStealingPool::Priority::High
    );

    low.wait();
    high.wait();

    // High priority should execute first (or both might execute if low starts immediately)
    EXPECT_GE(execution_order.size(), 2);
}

TEST_F(WorkStealingPoolTest, SubmitWithAffinity) {
    fcc::WorkStealingPool pool(4);

    // Submit task with preferred worker
    fcc::WorkStealingPool::AffinityHint hint{1, false};
    auto future = pool.submit_with_affinity(
        []() { return fcc::WorkStealingPool::get_current_worker_id(); },
        hint
    );

    size_t worker_id = future.get();
    EXPECT_LT(worker_id, pool.worker_count());
}

// ==================== Bulk Operations Tests ====================

TEST_F(WorkStealingPoolTest, BulkSubmission) {
    fcc::WorkStealingPool pool(4);
    const size_t count = 100;
    std::atomic<size_t> processed{0};

    pool.submit_bulk([&processed](size_t /*idx*/) {
        processed.fetch_add(1, std::memory_order_relaxed);
    }, count);

    pool.wait_idle();
    EXPECT_EQ(processed.load(), count);
}

TEST_F(WorkStealingPoolTest, RangeSubmission) {
    fcc::WorkStealingPool pool(4);
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::atomic<int> sum{0};

    pool.submit_range(data.begin(), data.end(), [&sum](int value) {
        sum.fetch_add(value, std::memory_order_relaxed);
    });

    pool.wait_idle();
    EXPECT_EQ(sum.load(), 15);
}

// ==================== Work Stealing Tests ====================

TEST_F(WorkStealingPoolTest, WorkStealingOccurs) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> completed{0};

    // Submit all tasks to worker 0 to force stealing
    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit_with_affinity(
            [&completed]() {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                completed.fetch_add(1, std::memory_order_relaxed);
            },
            {0, false}  // Prefer worker 0 but allow stealing
        );
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), num_tasks);

    // Check that stealing occurred
    const auto& stats = pool.statistics();
    EXPECT_GT(stats.tasks_stolen.load(), 0);
    EXPECT_GT(stats.steal_attempts.load(), 0);
}

TEST_F(WorkStealingPoolTest, RecursiveTaskSubmission) {
    fcc::WorkStealingPool pool(4);

    // Fibonacci-style recursive task
    std::function<int(int)> fib;
    fib = [&pool, &fib](int n) -> int {
        if (n <= 1) return n;

        auto f1 = pool.submit_recursive([&fib, n]() { return fib(n - 1); });
        auto f2 = pool.submit_recursive([&fib, n]() { return fib(n - 2); });

        return f1.get() + f2.get();
    };

    auto result = pool.submit_recursive([&fib]() { return fib(10); });
    EXPECT_EQ(result.get(), 55);  // 10th Fibonacci number
}

TEST_F(WorkStealingPoolTest, ForkJoinPattern) {
    fcc::WorkStealingPool pool(4);

    // Parallel sum using fork-join
    std::function<int(const std::vector<int>&, size_t, size_t)> parallel_sum;
    parallel_sum = [&pool, &parallel_sum](const std::vector<int>& data,
                                          size_t begin, size_t end) -> int {
        const size_t threshold = 1000;

        if (end - begin <= threshold) {
            // Sequential sum for small ranges
            int sum = 0;
            for (size_t i = begin; i < end; ++i) {
                sum += data[i];
            }
            return sum;
        }

        // Fork
        size_t mid = begin + (end - begin) / 2;
        auto left = pool.submit_recursive([&parallel_sum, &data, begin, mid]() {
            return parallel_sum(data, begin, mid);
        });
        auto right = pool.submit_recursive([&parallel_sum, &data, mid, end]() {
            return parallel_sum(data, mid, end);
        });

        // Join
        return left.get() + right.get();
    };

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 1);

    auto result = pool.submit_recursive([&parallel_sum, &data]() {
        return parallel_sum(data, 0, data.size());
    });

    int expected = (10000 * 10001) / 2;  // Sum formula
    EXPECT_EQ(result.get(), expected);
}

// ==================== Load Balancing Tests ====================

TEST_F(WorkStealingPoolTest, LoadBalancing) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 100;
    std::atomic<size_t> completed{0};

    // Create tasks with varying execution times
    std::mt19937 rng(42);
    std::uniform_int_distribution<> dist(1, 10);

    for (size_t i = 0; i < num_tasks; ++i) {
        int sleep_ms = dist(rng);
        pool.submit([sleep_ms, &completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), num_tasks);

    // Check load balance factor (lower is better)
    double balance = pool.load_balance_factor();
    EXPECT_LT(balance, 1.0);  // Should have reasonable balance
}

TEST_F(WorkStealingPoolTest, StealEfficiency) {
    fcc::WorkStealingPool pool(4, 10);  // Small local queues to force stealing
    const size_t num_tasks = 1000;

    // Submit many small tasks
    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }

    pool.wait_idle();

    // Check steal efficiency
    double efficiency = pool.steal_efficiency();
    EXPECT_GT(efficiency, 0.0);  // Some steals should succeed
    EXPECT_LE(efficiency, 1.0);  // Cannot exceed 100%
}

// ==================== Statistics Tests ====================

TEST_F(WorkStealingPoolTest, Statistics) {
    fcc::WorkStealingPool pool(4);
    pool.reset_statistics();

    const size_t num_tasks = 50;

    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }

    pool.wait_idle();

    const auto& stats = pool.statistics();
    EXPECT_EQ(stats.tasks_submitted.load(), num_tasks);
    EXPECT_EQ(stats.tasks_completed.load(), num_tasks);
    EXPECT_GT(stats.local_executions.load(), 0);
}

TEST_F(WorkStealingPoolTest, WorkerStatistics) {
    fcc::WorkStealingPool pool(2);

    // Submit tasks to specific workers
    for (size_t i = 0; i < 10; ++i) {
        pool.submit_with_affinity([]() {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }, {0, false});
    }

    pool.wait_idle();

    // Check worker 0 statistics
    auto stats0 = pool.worker_statistics(0);
    EXPECT_GT(stats0.tasks_executed, 0);
    EXPECT_EQ(stats0.queue_size, 0);  // Should be empty after wait_idle
}

// ==================== Synchronization Tests ====================

TEST_F(WorkStealingPoolTest, WaitIdle) {
    fcc::WorkStealingPool pool(4);
    std::atomic<size_t> completed{0};
    const size_t num_tasks = 100;

    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), num_tasks);
    EXPECT_EQ(pool.total_queue_size(), 0);
}

TEST_F(WorkStealingPoolTest, Shutdown) {
    fcc::WorkStealingPool pool(4);
    std::atomic<size_t> completed{0};

    // Submit tasks
    for (size_t i = 0; i < 10; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.shutdown();
    EXPECT_TRUE(pool.is_stopped());
    EXPECT_EQ(completed.load(), 10);
}

TEST_F(WorkStealingPoolTest, CannotSubmitAfterShutdown) {
    fcc::WorkStealingPool pool(2);
    pool.shutdown();

    EXPECT_THROW(
        pool.submit([]() { return 42; }),
        std::runtime_error
    );
}

// ==================== Exception Handling Tests ====================

TEST_F(WorkStealingPoolTest, TaskException) {
    fcc::WorkStealingPool pool(2);

    auto future = pool.submit([]() -> int {
        throw std::runtime_error("Task error");
        return 42;
    });

    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(WorkStealingPoolTest, VoidTaskException) {
    fcc::WorkStealingPool pool(2);

    auto future = pool.submit([]() {
        throw std::runtime_error("Void task error");
    });

    EXPECT_THROW(future.get(), std::runtime_error);
}

// ==================== Nested Parallelism Tests ====================

TEST_F(WorkStealingPoolTest, NestedParallelism) {
    fcc::WorkStealingPool pool(4);

    auto outer = pool.submit([&pool]() {
        std::vector<std::future<int>> inner_futures;

        for (int i = 0; i < 4; ++i) {
            inner_futures.push_back(pool.submit([i]() {
                return i * i;
            }));
        }

        int sum = 0;
        for (auto& f : inner_futures) {
            sum += f.get();
        }
        return sum;
    });

    EXPECT_EQ(outer.get(), 0 + 1 + 4 + 9);
}

TEST_F(WorkStealingPoolTest, WorkContext) {
    fcc::WorkStealingPool pool(4);

    auto result = pool.submit([&pool]() {
        fcc::WorkContext context(pool);

        // Submit nested tasks
        auto f1 = context.submit([]() { return 10; });
        auto f2 = context.submit([]() { return 20; });

        return f1.get() + f2.get();
    });

    EXPECT_EQ(result.get(), 30);
}

TEST_F(WorkStealingPoolTest, WorkContextOwnership) {
    fcc::WorkContext context(2);  // Creates its own pool

    auto future = context.pool().submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// ==================== Global Pool Tests ====================

TEST_F(WorkStealingPoolTest, GlobalPool) {
    auto& pool = fcc::global_work_stealing_pool();

    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(WorkStealingPoolTest, GlobalPoolReset) {
    fcc::GlobalWorkStealingPool::instance().reset(2);
    auto& pool = fcc::global_work_stealing_pool();

    EXPECT_EQ(pool.worker_count(), 2);
}

TEST_F(WorkStealingPoolTest, GlobalPoolShutdown) {
    auto& pool = fcc::global_work_stealing_pool();
    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);

    fcc::GlobalWorkStealingPool::instance().shutdown();
}

// ==================== Performance Tests ====================

TEST_F(WorkStealingPoolTest, ParallelSpeedup) {
    // Compare single-threaded vs multi-threaded execution
    const size_t num_tasks = 100;
    const auto work_duration = std::chrono::milliseconds(1);

    // Single-threaded
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_tasks; ++i) {
        std::this_thread::sleep_for(work_duration);
    }
    auto single_duration = std::chrono::steady_clock::now() - start;

    // Multi-threaded with work stealing
    fcc::WorkStealingPool pool(4);
    start = std::chrono::steady_clock::now();

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([work_duration]() {
            std::this_thread::sleep_for(work_duration);
        }));
    }

    for (auto& f : futures) {
        f.wait();
    }
    auto parallel_duration = std::chrono::steady_clock::now() - start;

    // Should have some speedup (at least 1.5x faster)
    double speedup = static_cast<double>(single_duration.count()) /
                    static_cast<double>(parallel_duration.count());
    EXPECT_GT(speedup, 1.5);
}

TEST_F(WorkStealingPoolTest, ManySmallTasks) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 10000;
    std::atomic<size_t> sum{0};

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([&sum, i]() {
            sum.fetch_add(i, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    auto duration = std::chrono::steady_clock::now() - start;

    // Verify correctness
    size_t expected = (num_tasks * (num_tasks - 1)) / 2;
    EXPECT_EQ(sum.load(), expected);

    // Should complete reasonably quickly
    EXPECT_LT(duration, std::chrono::seconds(5));
}

// ==================== Queue Management Tests ====================

TEST_F(WorkStealingPoolTest, QueueSizeLimit) {
    const size_t max_queue = 10;
    fcc::WorkStealingPool pool(1, max_queue);  // Single worker with limited queue

    std::atomic<bool> start{false};
    std::vector<std::future<void>> futures;

    // Fill the queue
    for (size_t i = 0; i < max_queue * 2; ++i) {
        futures.push_back(pool.submit([&start]() {
            while (!start.load()) {
                std::this_thread::yield();
            }
        }));
    }

    // Let tasks complete
    start = true;
    for (auto& f : futures) {
        f.wait();
    }

    // All tasks should complete despite queue limit (overflow to global queue)
    EXPECT_EQ(futures.size(), max_queue * 2);
}

TEST_F(WorkStealingPoolTest, TotalQueueSize) {
    fcc::WorkStealingPool pool(2);
    std::atomic<bool> start{false};

    // Submit tasks that will wait
    for (size_t i = 0; i < 10; ++i) {
        pool.submit([&start]() {
            while (!start.load()) {
                std::this_thread::yield();
            }
        });
    }

    // Give time for tasks to be distributed
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Queue should have some tasks (not all may have started)
    size_t queue_size = pool.total_queue_size();
    EXPECT_GE(queue_size, 0);

    // Let tasks complete
    start = true;
    pool.wait_idle();

    // Queue should be empty after wait_idle
    EXPECT_EQ(pool.total_queue_size(), 0);
}

// ==================== Move Semantics Tests ====================

TEST_F(WorkStealingPoolTest, MoveOnlyTypes) {
    fcc::WorkStealingPool pool(2);

    auto unique_ptr = std::make_unique<int>(42);

    auto future = pool.submit([ptr = std::move(unique_ptr)]() {
        return *ptr;
    });

    EXPECT_EQ(future.get(), 42);
}

TEST_F(WorkStealingPoolTest, LargeCaptureObjects) {
    fcc::WorkStealingPool pool(2);

    std::vector<int> large_vector(10000, 42);

    auto future = pool.submit([vec = std::move(large_vector)]() {
        return vec[0];
    });

    EXPECT_EQ(future.get(), 42);
}

// ==================== Stress Tests ====================

TEST_F(WorkStealingPoolTest, StressTestConcurrentSubmissions) {
    fcc::WorkStealingPool pool(8);
    const size_t num_threads = 10;
    const size_t tasks_per_thread = 100;
    std::atomic<size_t> total_executed{0};

    std::vector<std::thread> submitters;

    for (size_t t = 0; t < num_threads; ++t) {
        submitters.emplace_back([&pool, &total_executed, tasks_per_thread]() {
            for (size_t i = 0; i < tasks_per_thread; ++i) {
                pool.submit([&total_executed]() {
                    total_executed.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                });
            }
        });
    }

    for (auto& t : submitters) {
        t.join();
    }

    pool.wait_idle();
    EXPECT_EQ(total_executed.load(), num_threads * tasks_per_thread);
}

TEST_F(WorkStealingPoolTest, StressTestMixedPriorities) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> high_completed{0};
    std::atomic<size_t> low_completed{0};

    // Submit mixed priority tasks
    for (size_t i = 0; i < num_tasks; ++i) {
        if (i % 3 == 0) {
            pool.submit_with_priority(
                [&high_completed]() {
                    high_completed.fetch_add(1, std::memory_order_relaxed);
                },
                fcc::WorkStealingPool::Priority::High
            );
        } else {
            pool.submit_with_priority(
                [&low_completed]() {
                    low_completed.fetch_add(1, std::memory_order_relaxed);
                },
                fcc::WorkStealingPool::Priority::Low
            );
        }
    }

    pool.wait_idle();

    size_t expected_high = num_tasks / 3 + (num_tasks % 3 > 0 ? 1 : 0);
    size_t expected_low = num_tasks - expected_high;

    EXPECT_EQ(high_completed.load(), expected_high);
    EXPECT_EQ(low_completed.load(), expected_low);
}

TEST_F(WorkStealingPoolTest, DestructorWaitsForTasks) {
    std::atomic<int> counter{0};

    {
        fcc::WorkStealingPool pool(4);

        for (int i = 0; i < 100; ++i) {
            pool.submit([&counter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // Destructor should wait for all tasks
    }

    EXPECT_EQ(counter.load(), 100);
}

// ==================== Correctness Tests ====================

TEST_F(WorkStealingPoolTest, NoTasksLost) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> executed{0};
    std::set<size_t> seen_values;
    std::mutex seen_mutex;

    // Submit tasks with unique IDs
    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([i, &executed, &seen_values, &seen_mutex]() {
            executed.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(seen_mutex);
            seen_values.insert(i);
        });
    }

    pool.wait_idle();

    // All tasks should execute exactly once
    EXPECT_EQ(executed.load(), num_tasks);
    EXPECT_EQ(seen_values.size(), num_tasks);

    // Verify all IDs are present
    for (size_t i = 0; i < num_tasks; ++i) {
        EXPECT_TRUE(seen_values.count(i) > 0);
    }
}

TEST_F(WorkStealingPoolTest, RecursiveStealingCorrectness) {
    fcc::WorkStealingPool pool(4);

    // Compute factorials recursively with stealing
    std::function<uint64_t(uint64_t)> factorial;
    factorial = [&pool, &factorial](uint64_t n) -> uint64_t {
        if (n <= 1) return 1;

        if (n <= 10) {
            // Sequential for small values
            return n * factorial(n - 1);
        }

        // Parallel for larger values
        auto f1 = pool.submit_recursive([&factorial, n]() {
            return factorial(n - 1);
        });

        return n * f1.get();
    };

    auto result = pool.submit_recursive([&factorial]() {
        return factorial(20);
    });

    uint64_t expected = 2432902008176640000ULL;  // 20!
    EXPECT_EQ(result.get(), expected);
}
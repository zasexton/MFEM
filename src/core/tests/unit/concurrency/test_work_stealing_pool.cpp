#include <gtest/gtest.h>
#include <core/concurrency/work_stealing_pool.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

namespace fcc = fem::core::concurrency;
using namespace std::chrono_literals;

class WorkStealingPoolTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== Basic Tests ====================

TEST_F(WorkStealingPoolTest, DefaultConstruction) {
    fcc::WorkStealingPool pool;
    EXPECT_GT(pool.worker_count(), 0);
    EXPECT_EQ(pool.worker_count(), std::thread::hardware_concurrency());
}

TEST_F(WorkStealingPoolTest, ExplicitWorkerCount) {
    const size_t num_workers = 4;
    fcc::WorkStealingPool pool(num_workers);
    EXPECT_EQ(pool.worker_count(), num_workers);
}

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

// ==================== Concurrency Tests ====================

TEST_F(WorkStealingPoolTest, MultipleTasks) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 100;
    std::atomic<size_t> completed{0};

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([&completed]() {
            completed.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) {
        f.wait();
    }

    EXPECT_EQ(completed.load(), num_tasks);
}

TEST_F(WorkStealingPoolTest, WorkStealingOccurs) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> completed{0};

    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), num_tasks);

    // Check that stealing occurred
    size_t total_stolen = 0;
    for (size_t i = 0; i < pool.worker_count(); ++i) {
        auto stats = pool.get_worker_stats(i);
        total_stolen += stats.tasks_stolen;
    }
    EXPECT_GT(total_stolen, 0);
}

TEST_F(WorkStealingPoolTest, NestedTaskSubmission) {
    fcc::WorkStealingPool pool(4);

    auto result = pool.submit([&pool]() {
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

    EXPECT_EQ(result.get(), 0 + 1 + 4 + 9);
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
}

TEST_F(WorkStealingPoolTest, Shutdown) {
    fcc::WorkStealingPool pool(4);
    std::atomic<size_t> completed{0};

    for (size_t i = 0; i < 10; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.shutdown();
    EXPECT_EQ(completed.load(), 10);
}

// ==================== Exception Handling ====================

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

// ==================== Stress Tests ====================

TEST_F(WorkStealingPoolTest, StressConcurrentSubmissions) {
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

TEST_F(WorkStealingPoolTest, StressRapidSubmitWaitCycles) {
    fcc::WorkStealingPool pool(4);
    const size_t cycles = 100;

    for (size_t cycle = 0; cycle < cycles; ++cycle) {
        std::atomic<bool> executed{false};

        pool.submit([&executed]() {
            executed = true;
        });

        pool.wait_idle();
        EXPECT_TRUE(executed.load())
            << "Task not executed in cycle " << cycle;
    }
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

// ==================== Chase-Lev Deque Tests ====================

TEST_F(WorkStealingPoolTest, ChaseLevDequeCorrectness) {
    // Test that Chase-Lev deque properly handles work stealing
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> sum{0};

    // Submit many tasks to ensure stealing occurs
    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([&sum, i]() {
            sum.fetch_add(i, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();

    size_t expected = (num_tasks * (num_tasks - 1)) / 2;
    EXPECT_EQ(sum.load(), expected);
}

TEST_F(WorkStealingPoolTest, LocalityAwareStealingStats) {
    fcc::WorkStealingPool pool(4);

    // Submit many tasks
    for (int i = 0; i < 1000; ++i) {
        pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        });
    }

    pool.wait_idle();

    // Verify stealing occurred and stats are reasonable
    for (size_t i = 0; i < pool.worker_count(); ++i) {
        auto stats = pool.get_worker_stats(i);
        EXPECT_GT(stats.tasks_executed, 0);

        if (stats.steal_attempts > 0) {
            EXPECT_GE(stats.steal_success_rate, 0.0);
            EXPECT_LE(stats.steal_success_rate, 1.0);
        }
    }
}

// ==================== Edge Cases ====================

TEST_F(WorkStealingPoolTest, EdgeCaseEmptyPoolWaitIdle) {
    fcc::WorkStealingPool pool(4);

    for (int i = 0; i < 100; ++i) {
        pool.wait_idle();  // Should return immediately
    }
}

TEST_F(WorkStealingPoolTest, EdgeCaseSingleWorker) {
    fcc::WorkStealingPool pool(1);
    std::atomic<int> count{0};

    for (int i = 0; i < 50; ++i) {
        pool.submit([&count]() {
            count.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    EXPECT_EQ(count.load(), 50);
}

TEST_F(WorkStealingPoolTest, EdgeCaseZeroWorkTasks) {
    fcc::WorkStealingPool pool(4);
    const size_t num_tasks = 1000;
    std::atomic<size_t> count{0};

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_tasks; ++i) {
        pool.submit([&count]() {
            count.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_EQ(count.load(), num_tasks);
    EXPECT_LT(duration, std::chrono::seconds(2));
}

// ==================== Performance Verification ====================

TEST_F(WorkStealingPoolTest, ParallelSpeedup) {
    const size_t num_tasks = 100;
    const auto work_duration = std::chrono::milliseconds(1);

    // Single-threaded (approximate)
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

    fcc::GlobalWorkStealingPool::instance().reset();
}

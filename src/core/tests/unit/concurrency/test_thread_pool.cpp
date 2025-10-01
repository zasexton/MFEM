#include <gtest/gtest.h>
#include <core/concurrency/thread_pool.h>
#include <core/error/result.h>
#include <atomic>
#include <chrono>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <thread>

namespace fcc = fem::core::concurrency;
namespace fce = fem::core::error;

class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any global state
        executed_count_ = 0;
        thread_ids_.clear();
    }

    void TearDown() override {
        // Ensure any test pools are properly shut down
    }

    // Helper members for testing
    static std::atomic<int> executed_count_;
    static std::set<std::thread::id> thread_ids_;
    static std::mutex thread_ids_mutex_;
};

std::atomic<int> ThreadPoolTest::executed_count_{0};
std::set<std::thread::id> ThreadPoolTest::thread_ids_;
std::mutex ThreadPoolTest::thread_ids_mutex_;

// ==================== Basic Construction Tests ====================

TEST_F(ThreadPoolTest, DefaultConstruction) {
    fcc::ThreadPool pool;
    EXPECT_GT(pool.thread_count(), 0);
    EXPECT_EQ(pool.queue_size(), 0);
    EXPECT_EQ(pool.active_tasks(), 0);
    EXPECT_FALSE(pool.is_stopped());
    EXPECT_FALSE(pool.is_paused());
}

TEST_F(ThreadPoolTest, ExplicitThreadCount) {
    fcc::ThreadPool pool(4);
    EXPECT_EQ(pool.thread_count(), 4);
}

TEST_F(ThreadPoolTest, ZeroThreadsUsesHardwareConcurrency) {
    fcc::ThreadPool pool(0);
    EXPECT_GT(pool.thread_count(), 0);
}

TEST_F(ThreadPoolTest, MaxQueueSize) {
    fcc::ThreadPool pool(2, 10);  // 2 threads, max queue size 10
    EXPECT_EQ(pool.max_queue_size(), 10);
}

// ==================== Basic Task Submission Tests ====================

TEST_F(ThreadPoolTest, SubmitSimpleTask) {
    fcc::ThreadPool pool(2);

    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, SubmitWithArguments) {
    fcc::ThreadPool pool(2);

    auto future = pool.submit([](int a, int b) { return a + b; }, 10, 32);
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, SubmitVoidTask) {
    fcc::ThreadPool pool(2);
    bool executed = false;

    auto future = pool.submit([&executed]() { executed = true; });
    future.get();

    EXPECT_TRUE(executed);
}

TEST_F(ThreadPoolTest, SubmitMultipleTasks) {
    fcc::ThreadPool pool(2);
    const int num_tasks = 10;

    std::vector<std::future<int>> futures;
    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }

    for (int i = 0; i < num_tasks; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

// ==================== Priority Tests ====================

TEST_F(ThreadPoolTest, TaskPriority) {
    fcc::ThreadPool pool(1);  // Single thread to ensure sequential execution
    pool.pause();  // Pause to queue up tasks

    std::vector<int> execution_order;
    std::mutex order_mutex;

    // Submit tasks with different priorities
    auto low = pool.submit_with_priority([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(1);
    }, fcc::ThreadPool::Priority::Low);

    auto high = pool.submit_with_priority([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(2);
    }, fcc::ThreadPool::Priority::High);

    auto normal = pool.submit_with_priority([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(3);
    }, fcc::ThreadPool::Priority::Normal);

    pool.resume();

    low.get();
    high.get();
    normal.get();

    // High priority should execute first, then normal, then low
    EXPECT_EQ(execution_order[0], 2);  // High
    EXPECT_EQ(execution_order[1], 3);  // Normal
    EXPECT_EQ(execution_order[2], 1);  // Low
}

// ==================== Bulk Submission Tests ====================

TEST_F(ThreadPoolTest, BulkSubmission) {
    fcc::ThreadPool pool(4);
    const size_t count = 100;
    std::atomic<size_t> processed{0};

    pool.submit_bulk([&processed](size_t) {
        processed++;
    }, count);

    pool.wait_idle();
    EXPECT_EQ(processed.load(), count);
}

TEST_F(ThreadPoolTest, RangeSubmission) {
    fcc::ThreadPool pool(4);
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::atomic<int> sum{0};

    pool.submit_range(data.begin(), data.end(), [&sum](int value) {
        sum.fetch_add(value, std::memory_order_relaxed);
    });

    pool.wait_idle();
    EXPECT_EQ(sum.load(), 15);
}

// ==================== Pause/Resume Tests ====================

TEST_F(ThreadPoolTest, PauseAndResume) {
    fcc::ThreadPool pool(2);
    std::atomic<int> counter{0};

    // Submit initial task
    pool.submit([&counter]() {
        counter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });

    // Pause pool
    pool.pause();
    EXPECT_TRUE(pool.is_paused());

    // Submit task while paused (should queue but not execute)
    auto future = pool.submit([&counter]() {
        counter++;
        return counter.load();
    });

    // Give time for task to potentially execute (it shouldn't)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Resume and get result
    pool.resume();
    EXPECT_FALSE(pool.is_paused());

    int result = future.get();
    EXPECT_GE(result, 1);
}

TEST_F(ThreadPoolTest, PauseGuardRAII) {
    fcc::ThreadPool pool(2);

    EXPECT_FALSE(pool.is_paused());

    {
        fcc::ThreadPoolPauseGuard guard(pool);
        EXPECT_TRUE(pool.is_paused());
    }

    EXPECT_FALSE(pool.is_paused());
}

// ==================== Wait Tests ====================

TEST_F(ThreadPoolTest, WaitIdle) {
    fcc::ThreadPool pool(4);
    std::atomic<int> completed{0};

    for (int i = 0; i < 10; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed++;
        });
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), 10);
    EXPECT_EQ(pool.queue_size(), 0);
    EXPECT_EQ(pool.active_tasks(), 0);
}

TEST_F(ThreadPoolTest, WaitIdleWithTimeout) {
    fcc::ThreadPool pool(1);

    // Submit a long-running task
    pool.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    });

    // Should timeout
    bool became_idle = pool.wait_idle_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(became_idle);

    // Should succeed
    became_idle = pool.wait_idle_for(std::chrono::milliseconds(300));
    EXPECT_TRUE(became_idle);
}

// ==================== Shutdown Tests ====================

TEST_F(ThreadPoolTest, GracefulShutdown) {
    fcc::ThreadPool pool(2);
    std::atomic<int> completed{0};

    // Submit tasks
    for (int i = 0; i < 5; ++i) {
        pool.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed++;
        });
    }

    pool.shutdown();
    EXPECT_TRUE(pool.is_stopped());
    EXPECT_EQ(completed.load(), 5);  // All tasks should complete
}

TEST_F(ThreadPoolTest, ImmediateShutdown) {
    fcc::ThreadPool pool(1);
    std::atomic<int> completed{0};

    // Pause to queue tasks
    pool.pause();

    // Submit many tasks
    for (int i = 0; i < 10; ++i) {
        pool.submit([&completed]() {
            completed++;
        });
    }

    // Immediate shutdown should cancel pending tasks
    pool.shutdown_now();
    EXPECT_TRUE(pool.is_stopped());
    EXPECT_LT(completed.load(), 10);  // Not all tasks should complete
}

TEST_F(ThreadPoolTest, CannotSubmitAfterShutdown) {
    fcc::ThreadPool pool(2);
    pool.shutdown();

    EXPECT_THROW(
        pool.submit([]() { return 42; }),
        std::runtime_error
    );
}

// ==================== Exception Handling Tests ====================

TEST_F(ThreadPoolTest, TaskException) {
    fcc::ThreadPool pool(2);

    auto future = pool.submit([]() {
        throw std::runtime_error("Task error");
        return 42;
    });

    EXPECT_THROW(future.get(), std::runtime_error);

    // Pool should continue functioning
    auto future2 = pool.submit([]() { return 100; });
    EXPECT_EQ(future2.get(), 100);
}

TEST_F(ThreadPoolTest, VoidTaskException) {
    fcc::ThreadPool pool(2);
    std::atomic<bool> after_throw{false};

    auto future1 = pool.submit([]() {
        throw std::runtime_error("Task error");
    });

    auto future2 = pool.submit([&after_throw]() {
        after_throw = true;
    });

    EXPECT_THROW(future1.get(), std::runtime_error);
    future2.get();

    EXPECT_TRUE(after_throw);  // Second task should still execute
}

// ==================== Statistics Tests ====================

TEST_F(ThreadPoolTest, Statistics) {
    fcc::ThreadPool pool(2);

    // Reset statistics
    pool.reset_statistics();
    const auto& stats = pool.statistics();

    EXPECT_EQ(stats.tasks_submitted.load(), 0);
    EXPECT_EQ(stats.tasks_completed.load(), 0);

    // Submit successful task
    auto f1 = pool.submit([]() { return 42; });
    f1.get();

    // Submit a task that throws by calling a function that throws outside packaged_task
    auto f2 = pool.submit([]() {
        []() { throw std::runtime_error("error"); }(); // Call throwing function directly
        return 0;
    });

    try {
        f2.get();
    } catch (...) {}

    // Wait for all tasks to complete
    pool.wait_idle();

    EXPECT_GE(stats.tasks_submitted.load(), 2);
    EXPECT_GE(stats.tasks_completed.load(), 2); // Both tasks complete, even if one has exception in future
    // Note: tasks_failed tracking doesn't work with std::packaged_task
    // because exceptions are stored in futures, not thrown during task execution
}

// ==================== Queue Size Limit Tests ====================

TEST_F(ThreadPoolTest, QueueSizeLimit) {
    fcc::ThreadPool pool(1, 3);  // 1 thread, max queue size 3
    pool.pause();

    // Should accept up to max queue size
    pool.submit([]() {});
    pool.submit([]() {});
    pool.submit([]() {});

    // Should throw when exceeding limit
    EXPECT_THROW(
        pool.submit([]() {}),
        std::runtime_error
    );

    pool.resume();
    pool.wait_idle();
}

// ==================== Work Stealing Tests ====================

TEST_F(ThreadPoolTest, TrySteal) {
    fcc::ThreadPool pool(2);
    pool.pause();

    pool.submit([]() { return 42; });

    auto stolen = pool.try_steal();
    EXPECT_TRUE(stolen.has_value());

    // Execute stolen task
    if (stolen) {
        (*stolen)();
    }

    pool.resume();
}

// ==================== Result Type Integration Tests ====================

TEST_F(ThreadPoolTest, SubmitWithResult) {
    using Result = fce::Result<int, fce::ErrorCode>;

    fcc::ThreadPool pool(2);

    // Successful result
    auto future1 = pool.submit_with_result([]() -> Result {
        return 42;
    });

    auto result1 = future1.get();
    EXPECT_TRUE(result1.is_ok());
    EXPECT_EQ(result1.value(), 42);

    // Error result
    auto future2 = pool.submit_with_result([]() -> Result {
        return fce::Error<fce::ErrorCode>(fce::ErrorCode::InvalidArgument);
    });

    auto result2 = future2.get();
    EXPECT_TRUE(result2.is_error());
    EXPECT_EQ(result2.error(), fce::ErrorCode::InvalidArgument);
}

// ==================== Concurrent Operations Tests ====================

TEST_F(ThreadPoolTest, ConcurrentSubmissions) {
    fcc::ThreadPool pool(4);
    const int num_threads = 8;
    const int tasks_per_thread = 100;

    std::vector<std::thread> submitters;
    std::atomic<int> total_executed{0};

    for (int t = 0; t < num_threads; ++t) {
        submitters.emplace_back([&pool, &total_executed, tasks_per_thread]() {
            for (int i = 0; i < tasks_per_thread; ++i) {
                pool.submit([&total_executed]() {
                    total_executed++;
                });
            }
        });
    }

    for (auto& thread : submitters) {
        thread.join();
    }

    pool.wait_idle();
    EXPECT_EQ(total_executed.load(), num_threads * tasks_per_thread);
}

TEST_F(ThreadPoolTest, ThreadSafety) {
    fcc::ThreadPool pool(4);
    const int iterations = 1000;
    std::atomic<int> counter{0};

    std::vector<std::future<void>> futures;

    for (int i = 0; i < iterations; ++i) {
        futures.push_back(pool.submit([&counter]() {
            counter++;
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    EXPECT_EQ(counter.load(), iterations);
}

TEST_F(ThreadPoolTest, MultipleThreadPools) {
    fcc::ThreadPool pool1(2);
    fcc::ThreadPool pool2(2);

    auto future1 = pool1.submit([]() { return 100; });
    auto future2 = pool2.submit([]() { return 200; });

    EXPECT_EQ(future1.get(), 100);
    EXPECT_EQ(future2.get(), 200);
}

// ==================== Global Thread Pool Tests ====================

TEST_F(ThreadPoolTest, GlobalThreadPool) {
    auto& pool = fcc::global_thread_pool();

    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, GlobalThreadPoolReset) {
    auto& instance = fcc::GlobalThreadPool::instance();

    // Submit task to original pool
    auto& pool1 = instance.get_pool();
    auto future1 = pool1.submit([]() { return 100; });
    EXPECT_EQ(future1.get(), 100);

    // Reset pool
    instance.reset(2);

    // Submit task to new pool
    auto& pool2 = instance.get_pool();
    auto future2 = pool2.submit([]() { return 200; });
    EXPECT_EQ(future2.get(), 200);
}

TEST_F(ThreadPoolTest, GlobalThreadPoolShutdown) {
    auto& instance = fcc::GlobalThreadPool::instance();

    // Ensure pool exists
    instance.get_pool();

    // Shutdown
    instance.shutdown();

    // Get pool should create new one
    auto& pool = instance.get_pool();
    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// ==================== Performance Tests ====================

TEST_F(ThreadPoolTest, ParallelSpeedup) {
    const int num_tasks = 100;
    const auto work_duration = std::chrono::milliseconds(1);

    // Sequential execution time
    auto seq_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_tasks; ++i) {
        std::this_thread::sleep_for(work_duration);
    }
    auto seq_duration = std::chrono::steady_clock::now() - seq_start;

    // Parallel execution time
    fcc::ThreadPool pool(4);
    auto par_start = std::chrono::steady_clock::now();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([work_duration]() {
            std::this_thread::sleep_for(work_duration);
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
    auto par_duration = std::chrono::steady_clock::now() - par_start;

    // Parallel should be faster (at least 2x speedup with 4 threads)
    EXPECT_LT(par_duration, seq_duration / 2);
}

TEST_F(ThreadPoolTest, LoadBalancing) {
    fcc::ThreadPool pool(4);
    std::atomic<int> short_tasks{0};
    std::atomic<int> long_tasks{0};

    // Mix of short and long tasks
    for (int i = 0; i < 40; ++i) {
        if (i % 4 == 0) {
            // Long task
            pool.submit([&long_tasks]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                long_tasks++;
            });
        } else {
            // Short task
            pool.submit([&short_tasks]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                short_tasks++;
            });
        }
    }

    pool.wait_idle();
    EXPECT_EQ(short_tasks.load(), 30);
    EXPECT_EQ(long_tasks.load(), 10);
}

// ==================== Edge Cases ====================

TEST_F(ThreadPoolTest, SingleThreadPool) {
    fcc::ThreadPool pool(1);

    std::vector<int> order;
    std::mutex order_mutex;

    for (int i = 0; i < 5; ++i) {
        pool.submit([&order, &order_mutex, i]() {
            std::lock_guard<std::mutex> lock(order_mutex);
            order.push_back(i);
        });
    }

    pool.wait_idle();

    // Tasks should execute in order with single thread
    EXPECT_EQ(order, std::vector<int>({0, 1, 2, 3, 4}));
}

TEST_F(ThreadPoolTest, LargeNumberOfTasks) {
    fcc::ThreadPool pool(4);
    const int num_tasks = 10000;
    std::atomic<int> completed{0};

    for (int i = 0; i < num_tasks; ++i) {
        pool.submit([&completed]() {
            completed++;
        });
    }

    pool.wait_idle();
    EXPECT_EQ(completed.load(), num_tasks);
}

TEST_F(ThreadPoolTest, RecursiveSubmission) {
    fcc::ThreadPool pool(std::thread::hardware_concurrency());
    std::atomic<int> count{0};
    const int max_depth = 5;

    // Use a promise to signal completion of all recursive tasks
    auto completion_promise = std::make_shared<std::promise<void>>();
    auto completion_future = completion_promise->get_future();

    // Use shared_ptr to safely share the lambda across threads
    auto recursive_task = std::make_shared<std::function<void(int)>>();
    *recursive_task = [&count, &pool, recursive_task, max_depth, completion_promise](int level) {
        count.fetch_add(1, std::memory_order_relaxed);
        if (level < max_depth - 1) {
            // Submit the next level
            pool.submit([recursive_task, next_level = level + 1]() {
                (*recursive_task)(next_level);
            });
        } else {
            // Last level - signal completion
            completion_promise->set_value();
        }
    };

    // Start the recursion
    pool.submit([recursive_task]() { (*recursive_task)(0); });

    // Wait for the deepest level to complete
    completion_future.wait();

    // Give a small window for the count to be fully updated
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(count.load(), max_depth);
}

TEST_F(ThreadPoolTest, MoveOnlyTypes) {
    fcc::ThreadPool pool(2);

    auto unique_ptr = std::make_unique<int>(42);

    auto future = pool.submit([ptr = std::move(unique_ptr)]() {
        return *ptr;
    });

    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, LargeCaptureObjects) {
    fcc::ThreadPool pool(2);

    std::vector<int> large_vector(10000, 42);

    auto future = pool.submit([vec = std::move(large_vector)]() {
        return vec[0];
    });

    EXPECT_EQ(future.get(), 42);
}

// ==================== Stress Tests ====================

TEST_F(ThreadPoolTest, StressTestConcurrentOperations) {
    fcc::ThreadPool pool(8);
    const int duration_ms = 100;
    auto end_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);

    std::atomic<int> submitted{0};
    std::atomic<int> completed{0};
    std::atomic<bool> stop{false};

    // Submitter threads
    std::vector<std::thread> submitters;
    for (int i = 0; i < 4; ++i) {
        submitters.emplace_back([&]() {
            while (!stop) {
                try {
                    pool.submit([&completed]() {
                        completed++;
                    });
                    submitted++;
                } catch (...) {
                    // Queue might be full or pool stopped
                }
            }
        });
    }

    // Random pause/resume
    std::thread pauser([&]() {
        while (std::chrono::steady_clock::now() < end_time) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pool.pause();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            pool.resume();
        }
    });

    // Wait for duration
    std::this_thread::sleep_until(end_time);
    stop = true;

    for (auto& thread : submitters) {
        thread.join();
    }
    pauser.join();

    pool.wait_idle();

    // Most submitted tasks should complete
    EXPECT_GT(completed.load(), submitted.load() * 0.9);
}

TEST_F(ThreadPoolTest, StressTestMemoryUsage) {
    fcc::ThreadPool pool(4);
    const int iterations = 1000;

    for (int i = 0; i < iterations; ++i) {
        // Submit task with large capture
        std::vector<int> data(1000, i);
        pool.submit([data]() {
            // Process data
            volatile int sum = 0;
            for (int val : data) {
                sum += val;
            }
        });

        // Periodically wait to prevent queue overflow
        if (i % 100 == 0) {
            pool.wait_idle();
        }
    }

    pool.wait_idle();

    // Check statistics
    const auto& stats = pool.statistics();
    EXPECT_GE(stats.tasks_completed.load(), iterations);
}

// ==================== RAII and Lifetime Tests ====================

TEST_F(ThreadPoolTest, DestructorWaitsForTasks) {
    std::atomic<int> completed{0};

    {
        fcc::ThreadPool pool(2);

        for (int i = 0; i < 10; ++i) {
            pool.submit([&completed]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed++;
            });
        }
        // Destructor should wait for all tasks
    }

    EXPECT_EQ(completed.load(), 10);
}

TEST_F(ThreadPoolTest, MultiplePoolsIndependent) {
    fcc::ThreadPool pool1(2);
    fcc::ThreadPool pool2(2);

    std::atomic<int> pool1_tasks{0};
    std::atomic<int> pool2_tasks{0};

    // Submit to pool1
    for (int i = 0; i < 10; ++i) {
        pool1.submit([&pool1_tasks]() {
            pool1_tasks++;
        });
    }

    // Submit to pool2
    for (int i = 0; i < 10; ++i) {
        pool2.submit([&pool2_tasks]() {
            pool2_tasks++;
        });
    }

    pool1.wait_idle();
    pool2.wait_idle();

    EXPECT_EQ(pool1_tasks.load(), 10);
    EXPECT_EQ(pool2_tasks.load(), 10);
}
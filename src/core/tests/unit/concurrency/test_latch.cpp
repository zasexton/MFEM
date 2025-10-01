#include <gtest/gtest.h>
#include "core/concurrency/latch.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>

using namespace fem::core::concurrency;
using namespace std::chrono_literals;

class LatchTest : public ::testing::Test {
protected:
    static constexpr int NUM_THREADS = 8;
};

// -----------------------------------------------------------------------------
// Latch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, Latch_BasicCountdown) {
    Latch latch(3);

    EXPECT_EQ(latch.count(), 3);
    EXPECT_FALSE(latch.is_ready());

    latch.count_down();
    EXPECT_EQ(latch.count(), 2);

    latch.count_down();
    EXPECT_EQ(latch.count(), 1);

    latch.count_down();
    EXPECT_EQ(latch.count(), 0);
    EXPECT_TRUE(latch.is_ready());
}

TEST_F(LatchTest, Latch_MultiThreadWait) {
    const int count = 5;
    Latch latch(count);
    std::atomic<int> threads_started{0};
    std::atomic<int> threads_finished{0};

    // Start waiting threads
    std::vector<std::thread> waiters;
    for (int i = 0; i < NUM_THREADS; ++i) {
        waiters.emplace_back([&] {
            ++threads_started;
            latch.wait();
            ++threads_finished;
        });
    }

    // Wait for threads to start
    while (threads_started < NUM_THREADS) {
        std::this_thread::yield();
    }

    // No thread should finish yet
    std::this_thread::sleep_for(10ms);
    EXPECT_EQ(threads_finished.load(), 0);

    // Count down
    for (int i = 0; i < count; ++i) {
        latch.count_down();
    }

    // All threads should finish
    for (auto& t : waiters) {
        t.join();
    }
    EXPECT_EQ(threads_finished.load(), NUM_THREADS);
}

TEST_F(LatchTest, Latch_CountDownAndWait) {
    Latch latch(3);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&] {
            ++counter;
            latch.count_down_and_wait();
            // All threads should see counter == 3
            EXPECT_EQ(counter.load(), 3);
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(LatchTest, Latch_Arrive) {
    Latch latch(2);

    latch.arrive();
    EXPECT_EQ(latch.count(), 1);

    latch.arrive();
    EXPECT_EQ(latch.count(), 0);
    EXPECT_TRUE(latch.is_ready());
}

// -----------------------------------------------------------------------------
// CountdownEvent Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, CountdownEvent_AddAndSignal) {
    CountdownEvent event(2);

    EXPECT_EQ(event.count(), 2);
    EXPECT_FALSE(event.is_set());

    event.add_count(3);
    EXPECT_EQ(event.count(), 5);

    EXPECT_FALSE(event.signal(2));
    EXPECT_EQ(event.count(), 3);

    EXPECT_FALSE(event.signal(2));
    EXPECT_EQ(event.count(), 1);

    EXPECT_TRUE(event.signal());
    EXPECT_EQ(event.count(), 0);
    EXPECT_TRUE(event.is_set());
}

TEST_F(LatchTest, CountdownEvent_Reset) {
    CountdownEvent event(2);

    event.signal(2);
    EXPECT_TRUE(event.is_set());

    event.reset();
    EXPECT_EQ(event.count(), 2);
    EXPECT_FALSE(event.is_set());

    event.reset(5);
    EXPECT_EQ(event.count(), 5);
}

TEST_F(LatchTest, CountdownEvent_WaitWithTimeout) {
    CountdownEvent event(1);

    auto future = std::async(std::launch::async, [&] {
        return event.wait_for(50ms);
    });

    std::this_thread::sleep_for(20ms);
    event.signal();

    EXPECT_TRUE(future.get());
}

// -----------------------------------------------------------------------------
// CompletionLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, CompletionLatch_WithCallbacks) {
    CompletionLatch latch(3);
    std::atomic<int> callback_count{0};

    latch.on_completion([&] {
        ++callback_count;
    });

    latch.on_completion([&] {
        ++callback_count;
    });

    EXPECT_FALSE(latch.is_completed());
    EXPECT_EQ(latch.completion_ratio(), 0.0);

    latch.complete_one();
    EXPECT_DOUBLE_EQ(latch.completion_ratio(), 1.0 / 3.0);

    latch.complete(2);
    EXPECT_TRUE(latch.is_completed());
    EXPECT_DOUBLE_EQ(latch.completion_ratio(), 1.0);

    // Callbacks should have been executed
    std::this_thread::sleep_for(10ms);
    EXPECT_EQ(callback_count.load(), 2);
}

TEST_F(LatchTest, CompletionLatch_Progress) {
    CompletionLatch latch(10);

    for (int i = 0; i < 5; ++i) {
        latch.complete_one();
    }

    auto [completed, total] = latch.progress();
    EXPECT_EQ(completed, 5);
    EXPECT_EQ(total, 10);
    EXPECT_DOUBLE_EQ(latch.completion_ratio(), 0.5);
}

// -----------------------------------------------------------------------------
// BarrierLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, BarrierLatch_Synchronization) {
    const int num_threads = 4;
    const int num_phases = 3;
    BarrierLatch barrier(num_threads);
    BarrierLatch phase_barrier(num_threads);  // Second barrier for phase synchronization

    std::atomic<int> phase_counter{0};
    std::vector<std::vector<int>> thread_phases(num_threads);

    std::vector<std::thread> threads;
    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid] {
            for (int phase = 0; phase < num_phases; ++phase) {
                thread_phases[tid].push_back(phase_counter.load());
                barrier.arrive_and_wait();
                if (tid == 0) {
                    // First thread increments phase after barrier
                    ++phase_counter;
                }
                phase_barrier.arrive_and_wait();  // Ensure all threads see the new phase
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should see the same phase values
    for (int phase = 0; phase < num_phases; ++phase) {
        int expected_phase = phase;
        for (int tid = 0; tid < num_threads; ++tid) {
            EXPECT_EQ(thread_phases[tid][phase], expected_phase);
        }
    }
}

TEST_F(LatchTest, BarrierLatch_CompletionCallback) {
    std::atomic<int> callback_count{0};

    BarrierLatch barrier(3, [&] {
        ++callback_count;
    });

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 2; ++j) {
                barrier.arrive_and_wait();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Callback should be called twice (once per phase)
    EXPECT_EQ(callback_count.load(), 2);
}

// -----------------------------------------------------------------------------
// TimedLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, TimedLatch_Timeout) {
    TimedLatch latch(2, 50ms);

    auto start = std::chrono::steady_clock::now();
    bool success = latch.wait();
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(success);  // Should timeout
    EXPECT_GE(duration, 50ms);
    EXPECT_LT(duration, 100ms);

    auto [ready, timed_out] = latch.status();
    EXPECT_FALSE(ready);
    EXPECT_TRUE(timed_out);
}

TEST_F(LatchTest, TimedLatch_SuccessBeforeTimeout) {
    TimedLatch latch(2, 100ms);

    std::thread t([&] {
        std::this_thread::sleep_for(20ms);
        latch.count_down(2);
    });

    bool success = latch.wait();
    t.join();

    EXPECT_TRUE(success);
    auto [ready, timed_out] = latch.status();
    EXPECT_TRUE(ready);
    EXPECT_FALSE(timed_out);
}

TEST_F(LatchTest, TimedLatch_ForceExpire) {
    TimedLatch latch(5);

    std::thread t([&] {
        std::this_thread::sleep_for(20ms);
        latch.expire();
    });

    bool success = latch.wait_for(100ms);
    t.join();

    EXPECT_FALSE(success);
    auto [ready, timed_out] = latch.status();
    EXPECT_FALSE(ready);
    EXPECT_TRUE(timed_out);
}

// -----------------------------------------------------------------------------
// MultiStageLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, MultiStageLatch_StageProgression) {
    std::vector<std::ptrdiff_t> stage_counts = {2, 3, 1};
    std::vector<std::size_t> completed_stages;

    MultiStageLatch latch(stage_counts, [&](std::size_t stage) {
        completed_stages.push_back(stage);
    });

    EXPECT_EQ(latch.current_stage(), 0u);
    EXPECT_EQ(latch.num_stages(), 3u);

    // Complete stage 0
    latch.arrive(0);
    EXPECT_FALSE(latch.is_stage_complete(0));
    latch.arrive(0);
    EXPECT_TRUE(latch.is_stage_complete(0));

    // Complete stage 1
    for (int i = 0; i < 3; ++i) {
        latch.arrive(1);
    }
    EXPECT_TRUE(latch.is_stage_complete(1));

    // Complete stage 2
    latch.arrive(2);
    EXPECT_TRUE(latch.is_stage_complete(2));

    // Check callback invocations
    EXPECT_EQ(completed_stages.size(), 3u);
    EXPECT_EQ(completed_stages[0], 0u);
    EXPECT_EQ(completed_stages[1], 1u);
    EXPECT_EQ(completed_stages[2], 2u);
}

TEST_F(LatchTest, MultiStageLatch_ConcurrentStages) {
    std::vector<std::ptrdiff_t> stage_counts = {4, 4, 4};
    MultiStageLatch latch(stage_counts);

    std::vector<std::thread> threads;

    // Workers for each stage
    for (std::size_t stage = 0; stage < 3; ++stage) {
        for (int worker = 0; worker < 4; ++worker) {
            threads.emplace_back([&, stage] {
                // Simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(10 * (stage + 1)));
                latch.arrive(stage);
            });
        }
    }

    // Wait for all stages
    latch.wait_all();

    for (auto& t : threads) {
        t.join();
    }

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(latch.is_stage_complete(i));
    }
}

// -----------------------------------------------------------------------------
// DependentLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, DependentLatch_SimpleDependency) {
    auto latch1 = std::make_shared<DependentLatch>(1);
    auto latch2 = std::make_shared<DependentLatch>(1);
    auto latch3 = std::make_shared<DependentLatch>(1);

    // latch3 depends on latch1 and latch2
    latch3->add_dependency(latch1);
    latch3->add_dependency(latch2);

    EXPECT_EQ(latch3->num_dependencies(), 2u);

    // Count down latch3 (but it won't complete yet)
    latch3->count_down();
    EXPECT_FALSE(latch3->is_complete());

    // Complete latch1
    latch1->count_down();
    EXPECT_TRUE(latch1->is_complete());
    EXPECT_FALSE(latch3->is_complete());

    // Complete latch2
    latch2->count_down();
    EXPECT_TRUE(latch2->is_complete());
    EXPECT_TRUE(latch3->is_complete());
}

TEST_F(LatchTest, DependentLatch_ChainedDependencies) {
    std::atomic<int> completion_order{0};
    std::vector<int> order(3);

    auto latch1 = std::make_shared<DependentLatch>(1, [&] { order[0] = ++completion_order; });
    auto latch2 = std::make_shared<DependentLatch>(1, [&] { order[1] = ++completion_order; });
    auto latch3 = std::make_shared<DependentLatch>(1, [&] { order[2] = ++completion_order; });

    latch2->add_dependency(latch1);
    latch3->add_dependency(latch2);

    // Complete in reverse order
    latch3->count_down();
    latch2->count_down();
    latch1->count_down();

    latch3->wait();

    // Should complete in dependency order
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
    EXPECT_EQ(order[2], 3);
}

// -----------------------------------------------------------------------------
// GroupLatch Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, GroupLatch_MultipleGroups) {
    GroupLatch latch;

    latch.create_group("group1", 2);
    latch.create_group("group2", 3);
    latch.create_group("group3", 1);

    auto groups = latch.groups();
    EXPECT_EQ(groups.size(), 3u);

    EXPECT_FALSE(latch.is_complete("group1"));
    EXPECT_FALSE(latch.is_complete("group2"));
    EXPECT_FALSE(latch.is_complete("group3"));

    latch.count_down("group1", 2);
    EXPECT_TRUE(latch.is_complete("group1"));

    latch.count_down("group3");
    EXPECT_TRUE(latch.is_complete("group3"));

    latch.count_down("group2", 3);
    EXPECT_TRUE(latch.is_complete("group2"));
}

TEST_F(LatchTest, GroupLatch_WaitAny) {
    GroupLatch latch;

    latch.create_group("fast", 1);
    latch.create_group("slow", 5);

    std::thread t([&] {
        std::this_thread::sleep_for(20ms);
        latch.count_down("fast");
    });

    auto completed = latch.wait_any();
    t.join();

    EXPECT_EQ(completed, "fast");
    EXPECT_TRUE(latch.is_complete("fast"));
    EXPECT_FALSE(latch.is_complete("slow"));
}

TEST_F(LatchTest, GroupLatch_Reset) {
    std::atomic<int> callback_count{0};

    GroupLatch latch([&]([[maybe_unused]] const std::string& name) {
        ++callback_count;
    });

    latch.create_group("test", 2);
    latch.count_down("test", 2);

    EXPECT_TRUE(latch.is_complete("test"));
    EXPECT_EQ(callback_count.load(), 1);

    latch.reset("test");
    EXPECT_FALSE(latch.is_complete("test"));
    EXPECT_EQ(latch.count("test"), 2);

    latch.count_down("test", 2);
    EXPECT_TRUE(latch.is_complete("test"));
    EXPECT_EQ(callback_count.load(), 2);
}

// -----------------------------------------------------------------------------
// Utility Functions Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, CompositeLatch) {
    auto latch1 = std::make_shared<Latch>(2);
    auto latch2 = std::make_shared<Latch>(1);

    CompositeLatch composite;
    composite.add_latch(latch1);
    composite.add_latch(latch2);

    EXPECT_FALSE(composite.try_wait());

    latch1->count_down(2);
    EXPECT_FALSE(composite.try_wait());

    latch2->count_down();
    EXPECT_TRUE(composite.try_wait());
}

TEST_F(LatchTest, AutoLatch) {
    auto latch = make_auto_latch(5, 50ms);

    auto start = std::chrono::steady_clock::now();
    bool success = latch->wait();
    auto duration = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(success);  // Should expire
    EXPECT_GE(duration, 50ms);
    EXPECT_LT(duration, 100ms);
}

// -----------------------------------------------------------------------------
// Stress Tests
// -----------------------------------------------------------------------------
TEST_F(LatchTest, StressTest_ManyThreads) {
    const int num_threads = 100;
    Latch latch(num_threads);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&] {
            // Do some work
            ++counter;
            // Signal completion
            latch.count_down();
        });
    }

    latch.wait();
    EXPECT_EQ(counter.load(), num_threads);

    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(LatchTest, StressTest_RapidCountdown) {
    const int count = 10000;
    CountdownEvent event(count);

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < count / 10; ++j) {
                event.signal();
            }
        });
    }

    event.wait();
    EXPECT_TRUE(event.is_set());

    for (auto& t : threads) {
        t.join();
    }
}